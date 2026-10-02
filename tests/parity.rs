// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
// EXPLICIT: tests — a panic is the failure report.

//! `fused_attention` against an INDEPENDENT composed attention (`softmax(q·kᵀ·s)·v` from candle
//! ops, differentiated by candle's autograd): forward values and all three gradients, causal and
//! not, on strided views of a fused qkv projection (the layout candle-mi feeds it). CPU always;
//! CUDA (feature `cuda`, a device present) against the CPU at the measured band.

use candle_core::{D, DType, Device, Tensor, Var};
use candle_fused_attn::fused_attention;

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

/// The reference: composed attention, autograd-differentiable.
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

/// Max |a − b| over all elements.
fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
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

/// Output and the gradient of `sum(out ∘ w)` w.r.t. the qkv projection, through `attn`.
fn run(
    qkv: &Var,
    w: &Tensor,
    h: usize,
    d: usize,
    attn: &dyn Fn(&Tensor, &Tensor, &Tensor) -> Tensor,
) -> (Tensor, Tensor) {
    let (q, k, v) = qkv_views(qkv.as_tensor(), h, d);
    let out = attn(&q, &k, &v);
    let loss = (out.contiguous().unwrap() * w).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    (out, grads.get(qkv.as_tensor()).unwrap().clone())
}

/// Deterministic inputs on `device`: the projection and the loss weights.
fn inputs(b: usize, s: usize, h: usize, d: usize, device: &Device) -> (Var, Tensor) {
    let n = b * s * 3 * h * d;
    // A fixed pseudo-random sequence (no RNG crate): sin of a large-stride index.
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.618_034).sin() * 1.5).collect();
    let w: Vec<f32> = (0..b * h * s * d)
        .map(|i| (i as f32 * 0.414_213).cos())
        .collect();
    (
        Var::from_tensor(&Tensor::from_vec(x, (b, s, 3 * h * d), device).unwrap()).unwrap(),
        Tensor::from_vec(w, (b, h, s, d), device).unwrap(),
    )
}

fn check_cpu(b: usize, s: usize, h: usize, d: usize, causal: bool) {
    let scale = 1.0 / (d as f64).sqrt();
    let (qkv, w) = inputs(b, s, h, d, &Device::Cpu);
    let (o_ref, g_ref) = run(&qkv, &w, h, d, &|q, k, v| composed(q, k, v, scale, causal));
    let (o, g) = run(&qkv, &w, h, d, &|q, k, v| {
        fused_attention(q, k, v, scale as f32, causal).unwrap()
    });
    let (eo, eg) = (max_abs(&o, &o_ref), max_abs(&g, &g_ref));
    println!("cpu b{b} s{s} h{h} d{d} causal={causal}: |dO| {eo:.2e} |dgrad| {eg:.2e}");
    assert!(eo < 1e-5 && eg < 1e-5, "cpu mismatch: out {eo}, grad {eg}");
}

#[test]
fn cpu_matches_composed_autograd() {
    for causal in [false, true] {
        check_cpu(2, 7, 3, 8, causal);
        check_cpu(1, 33, 2, 64, causal);
    }
}

#[test]
fn output_is_merge_heads_ready() {
    let (qkv, _) = inputs(2, 5, 3, 8, &Device::Cpu);
    let (q, k, v) = qkv_views(qkv.as_tensor(), 3, 8);
    let o = fused_attention(&q, &k, &v, 0.3, false).unwrap();
    assert_eq!(o.dims(), &[2, 3, 5, 8]);
    assert!(
        o.transpose(1, 2).unwrap().is_contiguous(),
        "O is physically [b, s, h, d]"
    );
    assert_eq!(o.dtype(), DType::F32);
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    fn check_cuda(b: usize, s: usize, h: usize, causal: bool) {
        let Ok(dev) = Device::new_cuda(0) else {
            println!("no CUDA device: skipped");
            return;
        };
        let d = 64;
        let scale = 1.0 / (d as f64).sqrt();
        let (qkv_c, w_c) = inputs(b, s, h, d, &Device::Cpu);
        let (qkv_g, w_g) = inputs(b, s, h, d, &dev);
        // The null band: the composed path's own CPU-vs-CUDA spread.
        let (o_rc, g_rc) = run(&qkv_c, &w_c, h, d, &|q, k, v| {
            composed(q, k, v, scale, causal)
        });
        let (o_rg, g_rg) = run(&qkv_g, &w_g, h, d, &|q, k, v| {
            composed(q, k, v, scale, causal)
        });
        let (null_o, null_g) = (
            max_abs(&o_rg.to_device(&Device::Cpu).unwrap(), &o_rc),
            max_abs(&g_rg.to_device(&Device::Cpu).unwrap(), &g_rc),
        );
        let (o, g) = run(&qkv_g, &w_g, h, d, &|q, k, v| {
            fused_attention(q, k, v, scale as f32, causal).unwrap()
        });
        let (eo, eg) = (
            max_abs(&o.to_device(&Device::Cpu).unwrap(), &o_rc),
            max_abs(&g.to_device(&Device::Cpu).unwrap(), &g_rc),
        );
        println!(
            "cuda b{b} s{s} h{h} causal={causal}: fused-vs-cpu out {eo:.2e} grad {eg:.2e} | null band out {null_o:.2e} grad {null_g:.2e}"
        );
        assert!(eo < 5e-6 && eg < 5e-6, "cuda mismatch: out {eo}, grad {eg}");
    }

    #[test]
    fn cuda_matches_cpu() {
        for causal in [false, true] {
            check_cuda(2, 7, 3, causal);
            check_cuda(2, 240, 6, causal);
            check_cuda(1, 97, 2, causal);
        }
    }

    #[test]
    fn cuda_backward_is_deterministic() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let (qkv, w) = inputs(4, 240, 6, 64, &dev);
        let f =
            |q: &Tensor, k: &Tensor, v: &Tensor| fused_attention(q, k, v, 0.125, false).unwrap();
        let (_, g1) = run(&qkv, &w, 6, 64, &f);
        let (_, g2) = run(&qkv, &w, 6, 64, &f);
        assert_eq!(max_abs(&g1, &g2), 0.0, "two backward runs differ");
    }
}
