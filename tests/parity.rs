// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
// EXPLICIT: tests — a panic is the failure report.

//! Both entry points against an INDEPENDENT composed attention (`softmax(q·kᵀ·s)·v` from candle
//! ops, differentiated by candle's autograd): forward values and the gradient of the fused qkv
//! projection, causal and not. `fused_attention` gets the head-split views of the projection
//! (the layout candle-mi feeds it), `fused_attention_qkv` the projection itself. CPU always;
//! CUDA (feature `cuda`, a device present) against the CPU at the measured band.

use candle_core::{D, DType, Device, Tensor, Var};
use candle_fused_attn::{fused_attention, fused_attention_qkv};

/// Which attention computes the merged `[b, s, h·d]` output from the `[b, s, 3·h·d]` projection.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// The reference: composed ops, autograd.
    Composed,
    /// `fused_attention` on the head-split views.
    Fused,
    /// `fused_attention_qkv` on the projection.
    FusedQkv,
}

/// `q, k, v` as the head-split views of one `[b, s, 3·h·d]` projection, like candle-mi's.
fn qkv_views(qkv: &Tensor, h: usize, d: usize) -> (Tensor, Tensor, Tensor) {
    let (b, s, _) = qkv.dims3().unwrap();
    let split = |i: usize| {
        qkv.narrow(2, i * h * d, h * d)
            .unwrap()
            .reshape((b, s, h, d))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
    };
    (split(0), split(1), split(2))
}

/// The reference: composed attention, autograd-differentiable, `[b, h, s, d]`.
fn composed(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> Tensor {
    let s = q.dim(2).unwrap();
    let mut scores = (q
        .contiguous()
        .unwrap()
        .matmul(&k.contiguous().unwrap().t().unwrap())
        .unwrap()
        * scale)
        .unwrap();
    if causal {
        let bias: Vec<f32> = (0..s)
            .flat_map(|i| (0..s).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        scores = scores
            .broadcast_add(&Tensor::from_vec(bias, (s, s), q.device()).unwrap())
            .unwrap();
    }
    let max = scores.max_keepdim(D::Minus1).unwrap();
    let e = scores.broadcast_sub(&max).unwrap().exp().unwrap();
    let p = e.broadcast_div(&e.sum_keepdim(D::Minus1).unwrap()).unwrap();
    p.matmul(&v.contiguous().unwrap()).unwrap()
}

/// The merged output of `path`.
fn attend(path: Path, qkv: &Tensor, h: usize, d: usize, causal: bool) -> Tensor {
    let (b, s, _) = qkv.dims3().unwrap();
    let scale = 1.0 / (d as f64).sqrt();
    let merge = |o: Tensor| o.transpose(1, 2).unwrap().reshape((b, s, h * d)).unwrap();
    let (q, k, v) = qkv_views(qkv, h, d);
    match path {
        Path::Composed => merge(composed(&q, &k, &v, scale, causal)),
        Path::Fused => merge(fused_attention(&q, &k, &v, scale as f32, causal).unwrap()),
        Path::FusedQkv => fused_attention_qkv(qkv, h, scale as f32, causal).unwrap(),
    }
}

/// Max |a − b| over all elements, both moved to the CPU.
fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    let (a, b) = (
        a.to_device(&Device::Cpu).unwrap(),
        b.to_device(&Device::Cpu).unwrap(),
    );
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// Output and the gradient of `sum(out ∘ w)` w.r.t. the projection.
fn run(path: Path, qkv: &Var, w: &Tensor, h: usize, d: usize, causal: bool) -> (Tensor, Tensor) {
    let out = attend(path, qkv.as_tensor(), h, d, causal);
    let grads = (&out * w).unwrap().sum_all().unwrap().backward().unwrap();
    (out, grads.get(qkv.as_tensor()).unwrap().clone())
}

/// Deterministic inputs on `device`: the projection and the loss weights.
fn inputs(b: usize, s: usize, h: usize, d: usize, device: &Device) -> (Var, Tensor) {
    // A fixed pseudo-random sequence (no RNG crate): sin / cos of a large-stride index.
    let x: Vec<f32> = (0..b * s * 3 * h * d)
        .map(|i| (i as f32 * 0.618_034).sin() * 1.5)
        .collect();
    let w: Vec<f32> = (0..b * s * h * d)
        .map(|i| (i as f32 * 0.414_213).cos())
        .collect();
    (
        Var::from_tensor(&Tensor::from_vec(x, (b, s, 3 * h * d), device).unwrap()).unwrap(),
        Tensor::from_vec(w, (b, s, h * d), device).unwrap(),
    )
}

/// Every fused path on `device` against the composed path on the CPU; `band` the bound.
fn check(device: &Device, (b, s, h, d): (usize, usize, usize, usize), causal: bool, band: f32) {
    let (qkv_c, w_c) = inputs(b, s, h, d, &Device::Cpu);
    let (qkv, w) = inputs(b, s, h, d, device);
    let (o_ref, g_ref) = run(Path::Composed, &qkv_c, &w_c, h, d, causal);
    // The null band: the composed path's own spread between devices (0 on the CPU).
    let (o_null, g_null) = run(Path::Composed, &qkv, &w, h, d, causal);
    let (no, ng) = (max_abs(&o_null, &o_ref), max_abs(&g_null, &g_ref));
    for path in [Path::Fused, Path::FusedQkv] {
        let (o, g) = run(path, &qkv, &w, h, d, causal);
        let (eo, eg) = (max_abs(&o, &o_ref), max_abs(&g, &g_ref));
        println!(
            "{device:?} {path:?} b{b} s{s} h{h} d{d} causal={causal}: out {eo:.2e} grad {eg:.2e} | null out {no:.2e} grad {ng:.2e}"
        );
        assert!(
            eo < band && eg < band,
            "{path:?} mismatch: out {eo}, grad {eg}"
        );
    }
}

#[test]
fn cpu_matches_composed_autograd() {
    for causal in [false, true] {
        check(&Device::Cpu, (2, 7, 3, 8), causal, 1e-5);
        check(&Device::Cpu, (1, 33, 2, 64), causal, 1e-5);
    }
}

#[test]
fn outputs_are_merge_heads_ready() {
    let (qkv, _) = inputs(2, 5, 3, 8, &Device::Cpu);
    let (q, k, v) = qkv_views(qkv.as_tensor(), 3, 8);
    let o = fused_attention(&q, &k, &v, 0.3, false).unwrap();
    assert_eq!(o.dims(), &[2, 3, 5, 8]);
    assert!(
        o.transpose(1, 2).unwrap().is_contiguous(),
        "O is physically [b, s, h, d]"
    );
    let o = fused_attention_qkv(qkv.as_tensor(), 3, 0.3, false).unwrap();
    assert_eq!(o.dims(), &[2, 5, 24]);
    assert!(o.is_contiguous());
    assert_eq!(o.dtype(), DType::F32);
}

#[test]
fn backward_twice_over_one_graph() {
    // The saved L is read, not consumed: two backward passes over the same graph agree.
    let (qkv, w) = inputs(1, 9, 2, 8, &Device::Cpu);
    let out = attend(Path::FusedQkv, qkv.as_tensor(), 2, 8, false);
    let loss = (&out * &w).unwrap().sum_all().unwrap();
    let g1 = loss
        .backward()
        .unwrap()
        .get(qkv.as_tensor())
        .unwrap()
        .clone();
    let g2 = loss
        .backward()
        .unwrap()
        .get(qkv.as_tensor())
        .unwrap()
        .clone();
    assert_eq!(max_abs(&g1, &g2), 0.0);
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    #[test]
    fn cuda_matches_cpu() {
        let Ok(dev) = Device::new_cuda(0) else {
            println!("no CUDA device: skipped");
            return;
        };
        for causal in [false, true] {
            for shape in [(2, 7, 3, 64), (2, 240, 6, 64), (1, 97, 2, 64)] {
                check(&dev, shape, causal, 5e-6);
            }
        }
    }

    #[test]
    fn cuda_backward_is_deterministic() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let (qkv, w) = inputs(4, 240, 6, 64, &dev);
        for path in [Path::Fused, Path::FusedQkv] {
            let (_, g1) = run(path, &qkv, &w, 6, 64, false);
            let (_, g2) = run(path, &qkv, &w, 6, 64, false);
            assert_eq!(max_abs(&g1, &g2), 0.0, "{path:?}: two backward runs differ");
        }
    }
}
