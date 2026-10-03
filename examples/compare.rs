// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
// EXPLICIT: a benchmark driven by `bench/compare.py` — a panic is the failure report.
#![allow(unsafe_code)]
// EXPLICIT: two driver calls, `cuProfilerStart` / `cuProfilerStop`, bracket the nsys capture.
#![allow(clippy::many_single_char_names)]
// EXPLICIT: `b, s, h, d` and `q, k, v, o` are the attention papers' notation, as in the library.

//! The candle half of `bench/compare.py` (which holds the method, the `PyTorch` half and the
//! report). Reads the shared seeded inputs, runs one of three modes, writes JSON / safetensors:
//!
//! - `accuracy`: each candidate's forward + backward ONCE, outputs saved for the fp64 scoring,
//!   then a second run compared BITWISE with the first (determinism);
//! - `time`: per candidate and phase, `--warmup` calls, then `--samples` samples of `--calls`
//!   back-to-back calls between two device syncs (ms per call);
//! - `nsys`: ONE candidate and phase, `--warmup` calls, then `--calls` calls inside a
//!   `cuProfilerStart`/`cuProfilerStop` window (run under `nsys --capture-range=cudaProfilerApi`).
//!
//! The unit every candidate computes: the fused projection `qkv` `[b, s, 3·h·64]` to the merged
//! output `[b, s, h·64]`; the backward from `dout` (via the loss head `sum(o ∘ dout)`) to `dqkv`.
//! Phases: `fwd` (no autograd), `train` (forward + head + backward), `head` (the head alone, on a
//! stand-in for `o`; one per side, subtracted by the report).
//!
//! Usage: `compare <accuracy|time|nsys> --inputs F --out D --heads H [--causal] [--round K]
//!         [--samples R] [--calls N] [--warmup W] [--only NAME] [--phase fwd|train|head]`

use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::{D, Device, Tensor, Var};
use candle_fused_attn::{fused_attention, fused_attention_qkv};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

/// The candle candidates, in the report's order.
const CANDIDATES: [&str; 2] = ["candle_fused_qkv", "candle_fused"];

/// The parsed command line.
struct Args {
    /// `accuracy`, `time` or `nsys`.
    mode: String,
    /// The safetensors file holding `qkv` and `dout`.
    inputs: PathBuf,
    /// Where the outputs go.
    out: PathBuf,
    /// Attention heads.
    heads: usize,
    /// Causal masking.
    causal: bool,
    /// Round number (rotates the candidate order; names the output file).
    round: usize,
    /// Timing samples per candidate and phase.
    samples: usize,
    /// Back-to-back calls per sample (or inside the nsys window).
    calls: usize,
    /// Untimed calls first.
    warmup: usize,
    /// One candidate only (`nsys`).
    only: Option<String>,
    /// One phase only (`nsys`).
    phase: Option<String>,
}

/// `compare <mode> --key value … [--causal]`, into [`Args`].
fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mode = it.next().expect("mode: accuracy | time | nsys");
    let mut kv: HashMap<String, String> = HashMap::new();
    let mut causal = false;
    while let Some(k) = it.next() {
        if k == "--causal" {
            causal = true;
        } else {
            let v = it.next().unwrap_or_else(|| panic!("{k} needs a value"));
            kv.insert(k.trim_start_matches("--").to_owned(), v);
        }
    }
    let num = |k: &str, default: usize| kv.get(k).map_or(default, |v| v.parse().unwrap());
    Args {
        mode,
        inputs: PathBuf::from(kv.get("inputs").expect("--inputs")),
        out: PathBuf::from(kv.get("out").expect("--out")),
        heads: num("heads", 6),
        causal,
        round: num("round", 0),
        samples: num("samples", 15),
        calls: num("calls", 10),
        warmup: num("warmup", 10),
        only: kv.get("only").cloned(),
        phase: kv.get("phase").cloned(),
    }
}

/// The shared state of one run: inputs on the device, the trainable projection, the scale.
struct Bench {
    /// The CUDA device, for the syncs that bound each timing sample.
    dev: Device,
    /// The fused projection `[b, s, 3·h·d]`, trainable.
    qkv: Var,
    /// The upstream gradient `[b, s, h·d]`, injected by the loss head.
    dout: Tensor,
    /// Attention heads.
    heads: usize,
    /// `1/√d`.
    scale: f32,
    /// Causal masking.
    causal: bool,
}

impl Bench {
    /// The merged attention output of `qkv` by candidate `name`.
    fn attend(&self, name: &str, qkv: &Tensor) -> Tensor {
        let (b, s, three_hd) = qkv.dims3().unwrap();
        let (h, hd) = (self.heads, three_hd / 3);
        let d = hd / h;
        match name {
            "candle_fused_qkv" => fused_attention_qkv(qkv, h, self.scale, self.causal).unwrap(),
            "candle_fused" => {
                let split = |i: usize| {
                    qkv.narrow(2, i * hd, hd)
                        .unwrap()
                        .reshape((b, s, h, d))
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                };
                let o = fused_attention(&split(0), &split(1), &split(2), self.scale, self.causal);
                o.unwrap()
                    .transpose(1, 2)
                    .unwrap()
                    .reshape((b, s, hd))
                    .unwrap()
            }
            other => panic!("unknown candidate {other}"),
        }
    }

    /// The loss head `sum(o ∘ dout)`, whose gradient with respect to `o` is `dout`. Two-step
    /// reduction: candle's one-shot `sum_all` over this many floats is slow on its own.
    fn head(&self, o: &Tensor) -> Tensor {
        (o * &self.dout)
            .unwrap()
            .sum_keepdim(D::Minus1)
            .unwrap()
            .sum_all()
            .unwrap()
    }

    /// Forward + head + backward; returns `(o, dqkv)`.
    fn train(&self, name: &str) -> (Tensor, Tensor) {
        let o = self.attend(name, self.qkv.as_tensor());
        let grads = self.head(&o).backward().unwrap();
        let g = grads
            .get(self.qkv.as_tensor())
            .expect("no gradient reached qkv")
            .clone();
        (o, g)
    }

    /// One call of `phase` for candidate `name` (`head` ignores `name`).
    fn call(&self, name: &str, phase: &str, standin: &Var) {
        match phase {
            "fwd" => drop(self.attend(name, &self.qkv.as_tensor().detach())),
            "train" => drop(self.train(name)),
            "head" => drop(self.head(standin.as_tensor()).backward().unwrap()),
            other => panic!("unknown phase {other}"),
        }
    }
}

/// `xs` as a JSON array of numbers.
fn json_floats(xs: &[f64]) -> String {
    let body: Vec<String> = xs.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", body.join(", "))
}

fn main() {
    let a = parse_args();
    let dev = Device::new_cuda(0).unwrap();
    let t = candle_core::safetensors::load(&a.inputs, &dev).unwrap();
    let qkv = t.get("qkv").expect("inputs: qkv").clone();
    let dout = t.get("dout").expect("inputs: dout").clone();
    let (_, _, three_hd) = qkv.dims3().unwrap();
    let d = three_hd / 3 / a.heads;
    let bench = Bench {
        qkv: Var::from_tensor(&qkv).unwrap(),
        dout,
        heads: a.heads,
        scale: 1.0 / (d as f32).sqrt(),
        causal: a.causal,
        dev,
    };
    let standin = Var::from_tensor(&bench.dout.zeros_like().unwrap()).unwrap();
    std::fs::create_dir_all(&a.out).unwrap();

    match a.mode.as_str() {
        "accuracy" => {
            let mut json = String::from("[");
            for (i, name) in CANDIDATES.iter().enumerate() {
                let (o1, g1) = bench.train(name);
                let (o2, g2) = bench.train(name);
                let same = |x: &Tensor, y: &Tensor| {
                    let (x, y) = (x.flatten_all().unwrap(), y.flatten_all().unwrap());
                    let (x, y) = (x.to_vec1::<f32>().unwrap(), y.to_vec1::<f32>().unwrap());
                    x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits())
                };
                let deterministic = same(&o1, &o2) && same(&g1, &g2);
                let file = a.out.join(format!("{name}.safetensors"));
                let map = HashMap::from([("o".to_owned(), o1), ("dqkv".to_owned(), g1)]);
                candle_core::safetensors::save(&map, &file).unwrap();
                let sep = if i == 0 { "" } else { ", " };
                write!(
                    json,
                    "{sep}{{\"name\": \"{name}\", \"deterministic\": {deterministic}}}"
                )
                .unwrap();
            }
            json.push(']');
            std::fs::write(a.out.join("candle_accuracy.json"), json).unwrap();
        }
        "time" => {
            // The order rotates with the round, so no candidate is always first (or last).
            let n = CANDIDATES.len();
            let order: Vec<&str> = (0..n).map(|i| CANDIDATES[(i + a.round) % n]).collect();
            let mut rows: Vec<String> = Vec::new();
            let mut time = |name: &str, phase: &str| {
                for _ in 0..a.warmup {
                    bench.call(name, phase, &standin);
                }
                let ms: Vec<f64> = (0..a.samples)
                    .map(|_| {
                        bench.dev.synchronize().unwrap();
                        let t0 = Instant::now();
                        for _ in 0..a.calls {
                            bench.call(name, phase, &standin);
                        }
                        bench.dev.synchronize().unwrap();
                        t0.elapsed().as_secs_f64() * 1e3 / a.calls as f64
                    })
                    .collect();
                rows.push(format!(
                    "{{\"name\": \"{name}\", \"phase\": \"{phase}\", \"ms\": {}}}",
                    json_floats(&ms)
                ));
            };
            for name in &order {
                time(name, "fwd");
                time(name, "train");
            }
            time("head", "head");
            let file = a.out.join(format!("candle_time_r{}.json", a.round));
            std::fs::write(file, format!("[{}]", rows.join(",\n "))).unwrap();
        }
        "nsys" => {
            let name = a.only.as_deref().expect("--only");
            let phase = a.phase.as_deref().expect("--phase");
            for _ in 0..a.warmup {
                bench.call(name, phase, &standin);
            }
            bench.dev.synchronize().unwrap();
            // SAFETY: plain driver calls on the current (candle's) context, which is live;
            // their result codes are checked.
            let ok = sys::CUresult::CUDA_SUCCESS;
            assert_eq!(unsafe { sys::cuProfilerStart() }, ok, "cuProfilerStart");
            for _ in 0..a.calls {
                bench.call(name, phase, &standin);
            }
            bench.dev.synchronize().unwrap();
            assert_eq!(unsafe { sys::cuProfilerStop() }, ok, "cuProfilerStop");
        }
        other => panic!("unknown mode {other}"),
    }
}
