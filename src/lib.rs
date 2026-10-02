// SPDX-License-Identifier: MIT OR Apache-2.0
//! A fused fp32 scaled-dot-product attention for candle, with a real backward.
//!
//! [`fused_attention`] computes `softmax(q·kᵀ·scale [+ causal mask])·v` for `[batch, heads, seq,
//! head_dim]` tensors as ONE `CustomOp3`: on CUDA a FlashAttention-2-style forward (online
//! softmax, the `[seq, seq]` scores never written) and a deterministic three-kernel backward; on
//! CPU the composed reference, so every test runs anywhere.
//!
//! It works against STOCK candle: the PTX is loaded into candle's own context and launched on
//! candle's own stream.
//!
//! # The saved state
//!
//! candle's `CustomOp3` has no channel for intermediates kept from the forward for the
//! backward. The op therefore returns one packed 1-D buffer, `O` (physically `[b, s, h, d]`)
//! followed by the per-row log-sum-exp `L` (`[b, h, s]`), and [`fused_attention`] narrows `O`
//! out of it (offset 0: a free view). The backward receives `L` inside the op's output and a
//! zero gradient for it. (kaio-candle, the prior art, instead re-runs the forward in `bwd`.)
//!
//! # Layouts
//!
//! `q`, `k`, `v` may be any strided views whose LAST dimension is contiguous (e.g. the head
//! split of a fused qkv projection): the kernels read through the strides, no copy is made. The
//! output is `[b, h, s, d]` as the transpose of a contiguous `[b, s, h, d]` buffer, so the usual
//! merge-heads `transpose(1, 2).reshape((b, s, h·d))` costs nothing.
//!
//! # Limits (v0.1)
//!
//! f32 only; on CUDA `head_dim` must be 64 (the CPU path takes any). No dropout, no additive
//! mask beyond `causal`.

// EXPLICIT: `b, h, s, d` (batch, heads, seq, head_dim) and `q, k, v, o, l` are the notation of the
// attention papers this code follows; longer names would hide the math.
#![allow(clippy::many_single_char_names)]

use candle_core::{CpuStorage, CustomOp3, D, DType, Device, Error, Layout, Result, Shape, Tensor};

#[cfg(feature = "cuda")]
mod cuda;

/// The head dimension the CUDA kernels are specialised for.
pub const CUDA_HEAD_DIM: usize = 64;

/// Fused scaled-dot-product attention: `softmax(q·kᵀ·scale)·v`, rows `j > i` masked when
/// `causal`.
///
/// `q`, `k`, `v`: f32, `[batch, heads, seq, head_dim]`, one shape, one device; the last dimension
/// contiguous. Returns `[batch, heads, seq, head_dim]` (a transposed view of a contiguous
/// `[batch, seq, heads, head_dim]` buffer).
///
/// # Errors
///
/// On a dtype other than f32, mismatched shapes or devices, a non-contiguous last dimension, or
/// (CUDA) a head dimension other than [`CUDA_HEAD_DIM`].
pub fn fused_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let (b, h, s, d) = q.dims4()?;
    let n = b * s * h * d;
    let (q, k, v) = (float4_rows(q)?, float4_rows(k)?, float4_rows(v)?);
    let packed = q.apply_op3(&k, &v, FusedAttention { scale, causal })?;
    packed
        .narrow(0, 0, n)?
        .reshape((b, s, h, d))?
        .transpose(1, 2)
}

/// Whether a `[b, h, s, d]` layout's offset and (batch, head, seq) strides are multiples of 4
/// floats: the CUDA kernels read rows as float4.
pub(crate) fn float4_aligned(l: &Layout) -> bool {
    l.start_offset() % 4 == 0 && l.stride().iter().rev().skip(1).all(|s| s % 4 == 0)
}

/// `t` itself when [`float4_aligned`], else a contiguous copy (a `[b, h, s, d]` contiguous
/// tensor with `d` a multiple of 4 always is).
pub(crate) fn float4_rows(t: &Tensor) -> Result<Tensor> {
    if float4_aligned(t.layout()) {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// The op: `(q, k, v) → [O | L]`, see the crate docs for the packing.
#[derive(Debug, Clone, Copy)]
struct FusedAttention {
    /// The softmax scale, usually `1/√head_dim`.
    scale: f32,
    /// Mask keys after the query.
    causal: bool,
}

/// `(batch, heads, seq, head_dim)` of three layouts that must agree.
fn check_shapes(q: &Layout, k: &Layout, v: &Layout) -> Result<(usize, usize, usize, usize)> {
    let dims = q.shape().dims4()?;
    if k.shape().dims4()? != dims || v.shape().dims4()? != dims {
        return Err(Error::Msg(format!(
            "fused_attention: q, k, v must share one [b, h, s, d] shape, got {:?} {:?} {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    for (name, l) in [("q", q), ("k", k), ("v", v)] {
        if l.stride().last() != Some(&1) {
            return Err(Error::Msg(format!(
                "fused_attention: {name}'s head_dim must be contiguous, strides {:?}",
                l.stride()
            )));
        }
    }
    Ok(dims)
}

/// A strided f32 CPU storage gathered into a contiguous `[b, h, s, d]` tensor.
fn gather_cpu(storage: &CpuStorage, layout: &Layout) -> Result<Tensor> {
    let data = storage.as_slice::<f32>()?;
    let (b, h, s, d) = layout.shape().dims4()?;
    let (start, st) = (layout.start_offset(), layout.stride());
    let at = |i: usize| st.get(i).copied().unwrap_or(0);
    let index = (0..b * h * s * d).map(|e| {
        let (bi, hi, si, di) = (e / (h * s * d), e / (s * d) % h, e / d % s, e % d);
        start + bi * at(0) + hi * at(1) + si * at(2) + di * at(3)
    });
    let values = index
        .map(|i| data.get(i).copied())
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| Error::Msg("fused_attention: layout outside its storage".into()))?;
    Tensor::from_vec(values, (b, h, s, d), &Device::Cpu)
}

/// The composed reference forward: `(O [b, h, s, d], L [b, h, s, 1])`, on any device.
fn composed_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<(Tensor, Tensor)> {
    let scores = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * f64::from(scale))?;
    let scores = if causal {
        scores.broadcast_add(&causal_bias(q.dim(2)?, q.device())?)?
    } else {
        scores
    };
    let max = scores.max_keepdim(D::Minus1)?;
    let e = scores.broadcast_sub(&max)?.exp()?;
    let sum = e.sum_keepdim(D::Minus1)?;
    let o = e.broadcast_div(&sum)?.matmul(&v.contiguous()?)?;
    Ok((o, (max + sum.log()?)?))
}

/// `[s, s]`: 0 where `j ≤ i`, −∞ where `j > i`.
fn causal_bias(s: usize, device: &Device) -> Result<Tensor> {
    let bias: Vec<f32> = (0..s)
        .flat_map(|i| (0..s).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(bias, (s, s), device)
}

/// `[O | L]` → `(O [b, h, s, d], L [b, h, s, 1])`, views of the packed buffer.
fn unpack(packed: &Tensor, (b, h, s, d): (usize, usize, usize, usize)) -> Result<(Tensor, Tensor)> {
    let n = b * s * h * d;
    let o = packed
        .narrow(0, 0, n)?
        .reshape((b, s, h, d))?
        .transpose(1, 2)?;
    let l = packed.narrow(0, n, b * h * s)?.reshape((b, h, s, 1))?;
    Ok((o, l))
}

/// The composed backward on any device: `(dQ, dK, dV)` from the saved `L`, never the forward's
/// `P` (it is recomputed, as on CUDA).
fn composed_backward(
    op: FusedAttention,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    packed: &Tensor,
    grad: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = q.dims4()?;
    let (q, k, v) = (
        q.detach().contiguous()?,
        k.detach().contiguous()?,
        v.detach().contiguous()?,
    );
    let (o, l) = unpack(&packed.detach(), dims)?;
    let (d_o, _) = unpack(grad, dims)?;
    let d_o = d_o.contiguous()?;
    let scale = f64::from(op.scale);
    let mut p = (q.matmul(&k.t()?.contiguous()?)? * scale)?
        .broadcast_sub(&l)?
        .exp()?;
    if op.causal {
        let keep = causal_bias(dims.2, q.device())?.exp()?; // 1 where j <= i, 0 above
        p = p.broadcast_mul(&keep)?;
    }
    let dv = p.t()?.contiguous()?.matmul(&d_o)?;
    let dp = d_o.matmul(&v.t()?.contiguous()?)?;
    let dsum = (&d_o * &o)?.sum_keepdim(D::Minus1)?;
    let ds = (p * dp.broadcast_sub(&dsum)?)?;
    let dq = (ds.matmul(&k)? * scale)?;
    let dk = (ds.t()?.contiguous()?.matmul(&q)? * scale)?;
    Ok((dq, dk, dv))
}

impl CustomOp3 for FusedAttention {
    fn name(&self) -> &'static str {
        "fused-attention"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        check_shapes(l1, l2, l3)?;
        let (q, k, v) = (
            gather_cpu(s1, l1)?,
            gather_cpu(s2, l2)?,
            gather_cpu(s3, l3)?,
        );
        let (o, l) = composed_forward(&q, &k, &v, self.scale, self.causal)?;
        let packed = Tensor::cat(&[o.transpose(1, 2)?.flatten_all()?, l.flatten_all()?], 0)?;
        let len = packed.elem_count();
        Ok((CpuStorage::F32(packed.to_vec1::<f32>()?), Shape::from(len)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dims = check_shapes(l1, l2, l3)?;
        cuda::forward(self.scale, self.causal, dims, (s1, l1), (s2, l2), (s3, l3))
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        if q.dtype() != DType::F32 {
            return Err(Error::Msg("fused_attention: f32 only".into()));
        }
        let grad = grad_res.contiguous()?;
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            let (dq, dk, dv) = cuda::backward(*self, q, k, v, res, &grad)?;
            return Ok((Some(dq), Some(dk), Some(dv)));
        }
        let (dq, dk, dv) = composed_backward(*self, q, k, v, res, &grad)?;
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}
