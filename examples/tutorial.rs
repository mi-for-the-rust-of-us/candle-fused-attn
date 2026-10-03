// SPDX-License-Identifier: MIT OR Apache-2.0
//! The code of `docs/tutorial.md`, step by step. CI runs it on the CPU path
//! (`cargo run --example tutorial`), and `scripts/check-tutorial.py` checks that every code block
//! of the tutorial appears here verbatim, so the tutorial cannot drift from code that runs. On a
//! CUDA machine: `cargo run --release --features cuda --example tutorial`.

use candle_core::{D, Device, Result, Tensor, Var};
use candle_fused_attn::fused_attention_qkv;

/// The composed attention most candle models write: split the fused projection into heads,
/// `softmax(q·kᵀ·scale)·v`, merge the heads back.
fn composed_attention(qkv: &Tensor, heads: usize, scale: f64) -> Result<Tensor> {
    let (batch, seq, three_hidden) = qkv.dims3()?;
    let hidden = three_hidden / 3;
    let head_dim = hidden / heads;
    let split = |i: usize| -> Result<Tensor> {
        qkv.narrow(2, i * hidden, hidden)?
            .reshape((batch, seq, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()
    };
    let (q, k, v) = (split(0)?, split(1)?, split(2)?);
    let scores = (q.matmul(&k.t()?)? * scale)?;
    let max = scores.max_keepdim(D::Minus1)?;
    let exp = scores.broadcast_sub(&max)?.exp()?;
    let probs = exp.broadcast_div(&exp.sum_keepdim(D::Minus1)?)?;
    probs
        .matmul(&v)?
        .transpose(1, 2)?
        .reshape((batch, seq, hidden))
}

/// The largest absolute difference between two tensors of one shape.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

/// The gradient of `sum(attention(qkv) ∘ weights)` with respect to `qkv`, by `fused` or not.
fn qkv_gradient(qkv: &Var, weights: &Tensor, heads: usize, fused: bool) -> Result<Tensor> {
    let scale = 0.125_f32; // 1 / √64
    let o = if fused {
        fused_attention_qkv(qkv.as_tensor(), heads, scale, false)?
    } else {
        composed_attention(qkv.as_tensor(), heads, f64::from(scale))?
    };
    let grads = (o * weights)?.sum_all()?.backward()?;
    grads
        .get(qkv.as_tensor())
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("no gradient reached qkv".into()))
}

/// The tutorial, step by step.
fn main() -> Result<()> {
    // Step 1 — the device: CUDA when built with `--features cuda` and a GPU is present, else
    // the CPU, where the same call runs the composed reference.
    let device = Device::cuda_if_available(0)?;
    println!("device: {device:?}");

    // Step 2 — the drop-in: one call from the fused projection to the merged output.
    let (batch, seq, heads, head_dim) = (4, 48, 6, 64);
    let qkv = Tensor::randn(0f32, 1.0, (batch, seq, 3 * heads * head_dim), &device)?;
    let o = fused_attention_qkv(&qkv, heads, 0.125, false)?;
    println!("output: {:?}", o.dims()); // [batch, seq, heads · head_dim]

    // Step 3 — check it on YOUR shapes: forward and gradients against your composed attention.
    let reference = composed_attention(&qkv, heads, 0.125)?;
    let forward_gap = max_abs_diff(&o, &reference)?;
    let qkv_var = Var::from_tensor(&qkv)?;
    let weights = Tensor::randn(0f32, 1.0, (batch, seq, heads * head_dim), &device)?;
    let fused_grad = qkv_gradient(&qkv_var, &weights, heads, true)?;
    let composed_grad = qkv_gradient(&qkv_var, &weights, heads, false)?;
    let gradient_gap = max_abs_diff(&fused_grad, &composed_grad)?;
    println!("max |fused - composed|: forward {forward_gap:.2e}, gradient {gradient_gap:.2e}");
    // The bar: your composed attention's own CPU-vs-device spread, measured, not assumed.
    let band = if device.is_cuda() {
        let on_cpu = composed_attention(&qkv.to_device(&Device::Cpu)?, heads, 0.125)?;
        max_abs_diff(&reference.to_device(&Device::Cpu)?, &on_cpu)?.max(1e-6)
    } else {
        1e-6
    };
    if forward_gap > 4.0 * band {
        candle_core::bail!(
            "forward gap {forward_gap:.2e} exceeds 4 x the measured band {band:.2e}"
        );
    }

    // Step 4 — determinism: the backward twice, compared BIT for bit.
    let again = qkv_gradient(&qkv_var, &weights, heads, true)?;
    let (first, second) = (
        fused_grad.flatten_all()?.to_vec1::<f32>()?,
        again.flatten_all()?.to_vec1::<f32>()?,
    );
    let identical = first
        .iter()
        .zip(&second)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("two backward runs bit-identical: {identical}");
    if !identical {
        candle_core::bail!("the backward is not deterministic on this device");
    }

    // Step 5 — inside a training loop: the op is differentiable like any candle op.
    let hidden = heads * head_dim;
    let x = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device)?;
    let w_qkv =
        Var::from_tensor(&(Tensor::randn(0f32, 1.0, (hidden, 3 * hidden), &device)? * 0.02)?)?;
    let target = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device)?;
    for step in 0..3 {
        let qkv = x.broadcast_matmul(w_qkv.as_tensor())?;
        let out = fused_attention_qkv(&qkv, heads, 0.125, false)?;
        let loss = (out - &target)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        let grad = grads
            .get(w_qkv.as_tensor())
            .ok_or_else(|| candle_core::Error::Msg("no gradient reached w_qkv".into()))?;
        w_qkv.set(&(w_qkv.as_tensor() - (grad * 0.1)?)?)?;
        println!("step {step}: loss {:.6}", loss.to_scalar::<f32>()?);
    }
    Ok(())
}
