// SPDX-License-Identifier: MIT OR Apache-2.0
//! A fused fp32 scaled-dot-product attention for candle, with a real backward.
//!
//! Two entry points compute `softmax(q·kᵀ·scale [+ causal mask])·v`:
//! - [`fused_attention_qkv`] takes the fused `[batch, seq, 3·heads·head_dim]` projection itself
//!   and returns the merged `[batch, seq, heads·head_dim]` output. Its backward writes ONE
//!   gradient, interleaved exactly as the projection is: no head split, no head merge, and none
//!   of the per-view zero padding and summing candle's `narrow` backward would otherwise do
//!   (measured: ~3.7 ms of glue per layer at b 64 · h 6 · s 240, more than half the kernels' time).
//! - [`fused_attention`] takes `[batch, heads, seq, head_dim]` `q`, `k`, `v` (any strides with
//!   `head_dim` contiguous).
//!
//! On CUDA: a FlashAttention-2-style forward (online softmax, the `[seq, seq]` scores never
//! written) and a deterministic two-kernel backward. On CPU: the composed reference, so every
//! test runs anywhere. It works against STOCK candle: the PTX is loaded into candle's own context
//! and launched on candle's own stream.
//!
//! # The saved state
//!
//! candle's custom ops have no channel for intermediates kept from the forward for the backward,
//! but candle hands `bwd` the SAME op instance the forward ran (`Op::CustomOp*` holds its `Arc`).
//! The op therefore keeps the per-row log-sum-exp `L` (`[b, h, s]`, 1/64 of `O`) in a field, and
//! its output is `O` alone. (kaio-candle, the prior art, instead re-runs the forward in `bwd`.)
//!
//! # Limits (v0.1)
//!
//! f32 only; on CUDA `head_dim` must be 64 (the CPU path takes any). No dropout, no additive
//! mask beyond `causal`.

// EXPLICIT: `b, h, s, d` (batch, heads, seq, head_dim) and `q, k, v, o, l` are the notation of the
// attention papers this code follows; longer names would hide the math.
#![allow(clippy::many_single_char_names)]

use std::sync::Mutex;

use candle_core::{
    CpuStorage, CustomOp1, CustomOp3, D, DType, Device, Error, Layout, Result, Shape, Tensor,
};

#[cfg(feature = "cuda")]
mod cuda;

/// The head dimension the CUDA kernels are specialised for.
pub const CUDA_HEAD_DIM: usize = 64;

/// `(batch, heads, seq, head_dim)`.
pub(crate) type Dims = (usize, usize, usize, usize);

/// Fused attention over a fused qkv projection.
///
/// `qkv`: f32 `[batch, seq, 3·heads·head_dim]`, laid out `[q | k | v]` along the last dimension,
/// each third split into `heads` heads of `head_dim` (the layout of a fused `c_attn`/`qkv` linear).
/// Returns `[batch, seq, heads·head_dim]`, heads merged.
///
/// # Errors
///
/// On a dtype other than f32, a last dimension not divisible by `3·heads`, or (CUDA) a head
/// dimension other than [`CUDA_HEAD_DIM`].
pub fn fused_attention_qkv(qkv: &Tensor, heads: usize, scale: f32, causal: bool) -> Result<Tensor> {
    let l = qkv.layout();
    let aligned = l.start_offset() % 4 == 0
        && matches!(l.stride(), &[sb, ss, 1] if sb % 4 == 0 && ss % 4 == 0);
    let qkv = if aligned {
        qkv.clone()
    } else {
        qkv.contiguous()?
    };
    qkv.apply_op1(FusedAttentionQkv {
        heads,
        scale,
        causal,
        lse: Mutex::default(),
    })
}

/// Fused scaled-dot-product attention: `softmax(q·kᵀ·scale)·v`, rows `j > i` masked when
/// `causal`.
///
/// `q`, `k`, `v`: f32, `[batch, heads, seq, head_dim]`, one shape, one device; the last dimension
/// contiguous. Returns `[batch, heads, seq, head_dim]` (a transposed view of a contiguous
/// `[batch, seq, heads, head_dim]` buffer, so a merge-heads `transpose(1, 2).reshape(..)` is free).
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
    let (q, k, v) = (float4_rows(q)?, float4_rows(k)?, float4_rows(v)?);
    q.apply_op3(
        &k,
        &v,
        FusedAttention {
            scale,
            causal,
            lse: Mutex::default(),
        },
    )?
    .transpose(1, 2)
}

/// Whether a `[b, h, s, d]` layout's offset and (batch, head, seq) strides are multiples of 4
/// floats, `d` contiguous: the CUDA kernels read rows as float4.
fn float4_aligned(l: &Layout) -> bool {
    l.start_offset() % 4 == 0
        && matches!(l.stride(), &[sb, sh, ss, 1] if sb % 4 == 0 && sh % 4 == 0 && ss % 4 == 0)
}

/// `t` itself when [`float4_aligned`], else a contiguous copy (a contiguous `[b, h, s, d]` tensor
/// with `d` a multiple of 4 always is).
fn float4_rows(t: &Tensor) -> Result<Tensor> {
    if float4_aligned(t.layout()) {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// The (batch, head, seq) strides of a rank-4 `[b, h, s, d]` layout.
fn bhs_strides(l: &Layout) -> Result<[usize; 3]> {
    match l.stride() {
        &[sb, sh, ss, 1] => Ok([sb, sh, ss]),
        other => Err(Error::Msg(format!(
            "fused_attention: [b, h, s, d] with d contiguous expected, strides {other:?}"
        ))),
    }
}

/// The `(dims, [b, h, s] strides of q / k / v, offsets of q / k / v)` of a qkv projection layout.
fn qkv_geometry(l: &Layout, heads: usize) -> Result<(Dims, [usize; 3], [usize; 3])> {
    let (b, s, three_hd) = l.shape().dims3()?;
    if heads == 0 || three_hd % (3 * heads) != 0 {
        return Err(Error::Msg(format!(
            "fused_attention_qkv: last dimension {three_hd} is not 3 x {heads} heads x head_dim"
        )));
    }
    let hd = three_hd / 3;
    let d = hd / heads;
    match l.stride() {
        &[sb, ss, 1] => Ok(((b, heads, s, d), [sb, d, ss], [0, hd, 2 * hd])),
        other => Err(Error::Msg(format!(
            "fused_attention_qkv: the last dimension must be contiguous, strides {other:?}"
        ))),
    }
}

/// `(batch, heads, seq, head_dim)` of three layouts that must agree.
fn check_shapes(q: &Layout, k: &Layout, v: &Layout) -> Result<Dims> {
    let dims = q.shape().dims4()?;
    if k.shape().dims4()? != dims || v.shape().dims4()? != dims {
        return Err(Error::Msg(format!(
            "fused_attention: q, k, v must share one [b, h, s, d] shape, got {:?} {:?} {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    Ok(dims)
}

/// A strided `[b, h, s, d]` operand of an f32 CPU storage, gathered into a contiguous tensor.
fn gather_cpu(
    storage: &CpuStorage,
    start: usize,
    (b, h, s, d): Dims,
    [sb, sh, ss]: [usize; 3],
) -> Result<Tensor> {
    let data = storage.as_slice::<f32>()?;
    let values = (0..b * h * s * d)
        .map(|e| {
            let (bi, hi, si, di) = (e / (h * s * d), e / (s * d) % h, e / d % s, e % d);
            data.get(start + bi * sb + hi * sh + si * ss + di).copied()
        })
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| Error::Msg("fused_attention: layout outside its storage".into()))?;
    Tensor::from_vec(values, (b, h, s, d), &Device::Cpu)
}

/// The composed reference forward: `(O [b, h, s, d], L [b, h, s])`, on any device.
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
    Ok((o, (max + sum.log()?)?.squeeze(D::Minus1)?))
}

/// `[s, s]`: 0 where `j ≤ i`, −∞ where `j > i`.
fn causal_bias(s: usize, device: &Device) -> Result<Tensor> {
    let bias: Vec<f32> = (0..s)
        .flat_map(|i| (0..s).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(bias, (s, s), device)
}

/// The composed backward on any device, all operands `[b, h, s, d]` and `L` `[b, h, s]`:
/// `(dQ, dK, dV)`, recomputing `P` from `L` as the CUDA kernels do.
fn composed_backward(
    (scale, causal): (f32, bool),
    (q, k, v): (&Tensor, &Tensor, &Tensor),
    o: &Tensor,
    l: &Tensor,
    d_o: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (q, k, v) = (
        q.detach().contiguous()?,
        k.detach().contiguous()?,
        v.detach().contiguous()?,
    );
    let (o, d_o) = (o.detach().contiguous()?, d_o.contiguous()?);
    let scale = f64::from(scale);
    let mut p = (q.matmul(&k.t()?.contiguous()?)? * scale)?
        .broadcast_sub(&l.unsqueeze(D::Minus1)?)?
        .exp()?;
    if causal {
        let keep = causal_bias(q.dim(2)?, q.device())?.exp()?; // 1 where j <= i, 0 above
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

/// Keep `L` for the backward.
fn save(slot: &Mutex<Option<Tensor>>, l: Tensor) -> Result<()> {
    *slot
        .lock()
        .map_err(|_| Error::Msg("fused_attention: saved state poisoned".into()))? = Some(l);
    Ok(())
}

/// The `L` the forward kept.
fn saved(slot: &Mutex<Option<Tensor>>) -> Result<Tensor> {
    slot.lock()
        .map_err(|_| Error::Msg("fused_attention: saved state poisoned".into()))?
        .clone()
        .ok_or_else(|| Error::Msg("fused_attention: backward before forward".into()))
}

/// `[b, h, s, d]` O → the op's `[b, s, ...]` CPU output.
fn cpu_output(o: &Tensor, shape: Shape) -> Result<(CpuStorage, Shape)> {
    Ok((
        CpuStorage::F32(o.transpose(1, 2)?.flatten_all()?.to_vec1::<f32>()?),
        shape,
    ))
}

/// The CUDA backward of either op: dqkv `[b, s, 3, h, d]` from three `(tensor, offset, strides)`
/// operands, the output `O` (`[b, s, h, d]` contiguous), its gradient read with `[b, h, s]`
/// strides `g`, and the saved `L`.
#[cfg(feature = "cuda")]
fn cuda_backward(
    (scale, causal): (f32, bool),
    dims: Dims,
    operands: [(&Tensor, usize, [usize; 3]); 3],
    o: &Tensor,
    (grad, g): (&Tensor, [usize; 3]),
    l: &Tensor,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    let [(tq, oq, sq), (tk, ok, sk), (tv, ov, sv)] = operands;
    let (gq, gk, gv) = (
        tq.storage_and_layout(),
        tk.storage_and_layout(),
        tv.storage_and_layout(),
    );
    let (go, gg, gl) = (
        o.storage_and_layout(),
        grad.storage_and_layout(),
        l.storage_and_layout(),
    );
    if !go.1.is_contiguous() || !gl.1.is_contiguous() {
        return Err(Error::Msg(
            "fused_attention: O and L must be contiguous".into(),
        ));
    }
    let q = cuda::view(cuda::cuda_of("q", &gq.0)?, gq.1, oq, sq)?;
    let k = cuda::view(cuda::cuda_of("k", &gk.0)?, gk.1, ok, sk)?;
    let v = cuda::view(cuda::cuda_of("v", &gv.0)?, gv.1, ov, sv)?;
    let d_o = cuda::view(cuda::cuda_of("grad", &gg.0)?, gg.1, 0, g)?;
    let so = cuda::cuda_of("O", &go.0)?;
    let ov_ = so.as_cuda_slice::<f32>()?.slice(go.1.start_offset()..);
    let sl = cuda::cuda_of("L", &gl.0)?;
    let lv = sl.as_cuda_slice::<f32>()?.slice(gl.1.start_offset()..);
    let dev = so.device().clone();
    let dqkv = cuda::backward(&dev, (scale, causal), dims, (&q, &k, &v), (&ov_, &lv), &d_o)?;
    let (b, h, s, d) = dims;
    Ok(cuda::tensor_of(dqkv, &dev, &[b, s, 3, h, d]))
}

/// `grad` itself when its rows are float4-aligned with the last dimension contiguous, else a
/// contiguous copy.
fn aligned_grad(grad: &Tensor) -> Result<Tensor> {
    let l = grad.layout();
    let ok = l.start_offset() % 4 == 0
        && l.stride().last() == Some(&1)
        && l.stride().iter().all(|s| *s == 1 || s % 4 == 0);
    if ok {
        Ok(grad.clone())
    } else {
        grad.contiguous()
    }
}

/// The `[b, h, s, d]`-in, `[b, s, h, d]`-out op behind [`fused_attention`].
#[derive(Debug)]
struct FusedAttention {
    /// The softmax scale, usually `1/√head_dim`.
    scale: f32,
    /// Mask keys after the query.
    causal: bool,
    /// The forward's `L`, for the backward.
    lse: Mutex<Option<Tensor>>,
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
        let dims = check_shapes(l1, l2, l3)?;
        let (q, k, v) = (
            gather_cpu(s1, l1.start_offset(), dims, bhs_strides(l1)?)?,
            gather_cpu(s2, l2.start_offset(), dims, bhs_strides(l2)?)?,
            gather_cpu(s3, l3.start_offset(), dims, bhs_strides(l3)?)?,
        );
        let (o, l) = composed_forward(&q, &k, &v, self.scale, self.causal)?;
        save(&self.lse, l)?;
        let (b, h, s, d) = dims;
        cpu_output(&o, Shape::from((b, s, h, d)))
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
        use candle_core::backend::BackendStorage;
        let dims = check_shapes(l1, l2, l3)?;
        if ![l1, l2, l3].iter().all(|l| float4_aligned(l)) {
            return Err(Error::Msg(
                "fused_attention: rows must be 16-byte aligned".into(),
            ));
        }
        let q = cuda::view(s1, l1, 0, bhs_strides(l1)?)?;
        let k = cuda::view(s2, l2, 0, bhs_strides(l2)?)?;
        let v = cuda::view(s3, l3, 0, bhs_strides(l3)?)?;
        let dev = s1.device().clone();
        let (o, l) = cuda::forward(&dev, self.scale, self.causal, dims, &q, &k, &v)?;
        let (b, h, s, d) = dims;
        save(&self.lse, cuda::tensor_of(l, &dev, &[b, h, s]))?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(o, dev),
            Shape::from((b, s, h, d)),
        ))
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
        let l = saved(&self.lse)?;
        let grad = aligned_grad(grad_res)?;
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            let dims = q.dims4()?;
            let [gb, gs, gh, _] = grad.stride() else {
                return Err(Error::Msg(
                    "fused_attention: rank-4 gradient expected".into(),
                ));
            };
            let operands = [
                (q, 0, bhs_strides(q.layout())?),
                (k, 0, bhs_strides(k.layout())?),
                (v, 0, bhs_strides(v.layout())?),
            ];
            let g = [*gb, *gh, *gs];
            let dqkv = cuda_backward(
                (self.scale, self.causal),
                dims,
                operands,
                res,
                (&grad, g),
                &l,
            )?;
            let part = |i| dqkv.narrow(2, i, 1)?.squeeze(2)?.transpose(1, 2);
            return Ok((Some(part(0)?), Some(part(1)?), Some(part(2)?)));
        }
        let (o, d_o) = (res.transpose(1, 2)?, grad.transpose(1, 2)?);
        let (dq, dk, dv) = composed_backward((self.scale, self.causal), (q, k, v), &o, &l, &d_o)?;
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

/// The qkv-in, merged-out op behind [`fused_attention_qkv`].
#[derive(Debug)]
struct FusedAttentionQkv {
    /// Number of heads.
    heads: usize,
    /// The softmax scale, usually `1/√head_dim`.
    scale: f32,
    /// Mask keys after the query.
    causal: bool,
    /// The forward's `L`, for the backward.
    lse: Mutex<Option<Tensor>>,
}

impl CustomOp1 for FusedAttentionQkv {
    fn name(&self) -> &'static str {
        "fused-attention-qkv"
    }

    fn cpu_fwd(&self, st: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (dims, strides, [oq, ok, ov]) = qkv_geometry(layout, self.heads)?;
        let start = layout.start_offset();
        let (q, k, v) = (
            gather_cpu(st, start + oq, dims, strides)?,
            gather_cpu(st, start + ok, dims, strides)?,
            gather_cpu(st, start + ov, dims, strides)?,
        );
        let (o, l) = composed_forward(&q, &k, &v, self.scale, self.causal)?;
        save(&self.lse, l)?;
        let (b, h, s, d) = dims;
        cpu_output(&o, Shape::from((b, s, h * d)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        st: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let (dims, strides, [oq, ok, ov]) = qkv_geometry(layout, self.heads)?;
        if layout.start_offset() % 4 != 0 || strides.iter().any(|s| s % 4 != 0) {
            return Err(Error::Msg(
                "fused_attention_qkv: rows must be 16-byte aligned".into(),
            ));
        }
        let q = cuda::view(st, layout, oq, strides)?;
        let k = cuda::view(st, layout, ok, strides)?;
        let v = cuda::view(st, layout, ov, strides)?;
        let dev = st.device().clone();
        let (o, l) = cuda::forward(&dev, self.scale, self.causal, dims, &q, &k, &v)?;
        let (b, h, s, d) = dims;
        save(&self.lse, cuda::tensor_of(l, &dev, &[b, h, s]))?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(o, dev),
            Shape::from((b, s, h * d)),
        ))
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] // the offsets serve CUDA only
    fn bwd(&self, qkv: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        if qkv.dtype() != DType::F32 {
            return Err(Error::Msg("fused_attention_qkv: f32 only".into()));
        }
        let l = saved(&self.lse)?;
        let grad = aligned_grad(grad_res)?;
        let (dims, strides, [oq, ok, ov]) = qkv_geometry(qkv.layout(), self.heads)?;
        let (b, h, s, d) = dims;
        #[cfg(feature = "cuda")]
        if qkv.device().is_cuda() {
            let [gb, gs, _] = grad.stride() else {
                return Err(Error::Msg(
                    "fused_attention_qkv: rank-3 gradient expected".into(),
                ));
            };
            let operands = [(qkv, oq, strides), (qkv, ok, strides), (qkv, ov, strides)];
            let g = [*gb, d, *gs];
            let dqkv = cuda_backward(
                (self.scale, self.causal),
                dims,
                operands,
                res,
                (&grad, g),
                &l,
            )?;
            return Ok(Some(dqkv.reshape((b, s, 3 * h * d))?));
        }
        let heads = |t: &Tensor| t.reshape((b, s, h, d))?.transpose(1, 2);
        let part = |i: usize| heads(&qkv.narrow(2, i * h * d, h * d)?);
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let (o, d_o) = (heads(res)?, heads(&grad)?);
        let (dq, dk, dv) =
            composed_backward((self.scale, self.causal), (&q, &k, &v), &o, &l, &d_o)?;
        let merge = |t: Tensor| t.transpose(1, 2)?.reshape((b, s, h * d));
        Ok(Some(Tensor::cat(&[merge(dq)?, merge(dk)?, merge(dv)?], 2)?))
    }
}
