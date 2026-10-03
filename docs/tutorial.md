# Tutorial: fused fp32 attention in a candle training loop

This tutorial replaces a composed attention with `candle-fused-attn`, checks the result on your own
shapes, checks that the backward is deterministic, and shows the op inside a training step. Every
Rust code block below is taken verbatim from [`examples/tutorial.rs`](../examples/tutorial.rs), which
CI runs (`scripts/check-tutorial.py` fails the build if a block drifts), so what you read here runs:

```bash
cargo run --example tutorial                               # CPU: the composed reference
cargo run --release --features cuda --example tutorial     # CUDA: the fused kernels
```

## 1. Where it runs

The same call works on every device. On CUDA (built with `--features cuda`) it runs the fused
kernels; elsewhere it runs a composed reference, so your tests run on a machine with no GPU.

```rust
let device = Device::cuda_if_available(0)?;
```

The CUDA path needs `head_dim` 64, f32 tensors, and a GPU of compute capability 7.0 (Volta) or
newer.

## 2. The drop-in

Most candle models compute attention from a fused projection `qkv` (`[batch, seq, 3·hidden]`) by
splitting it into heads, multiplying, taking a softmax, multiplying again and merging the heads:

```rust
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
```

The fused op replaces all of it with one call, from the projection to the merged output (the
scale `0.125` is `1/√head_dim`, for `head_dim` 64):

```rust
let (batch, seq, heads, head_dim) = (4, 48, 6, 64);
let qkv = Tensor::randn(0f32, 1.0, (batch, seq, 3 * heads * head_dim), &device)?;
let o = fused_attention_qkv(&qkv, heads, 0.125, false)?;
```

What disappears with it: the head split and merge (q, k, v are read through their strides), the
`[seq, seq]` scores (never written), and the per-head gradient pieces a `narrow` backward would
build and sum (the backward writes one gradient shaped like `qkv`). If your q, k, v are already
separate `[batch, heads, seq, head_dim]` tensors, use `fused_attention(&q, &k, &v, scale, causal)`
instead. `causal = true` masks the keys after each query.

## 3. Check it on your shapes

Before trusting it in your model, compare it with your own composed attention, forward and
gradients. Two small helpers: the largest difference between two tensors, and the gradient with
respect to `qkv` of `sum(attention(qkv) ∘ weights)`, which drives both backwards with the same
upstream gradient:

```rust
fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}
```

```rust
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
```

Then the comparison itself:

```rust
let reference = composed_attention(&qkv, heads, 0.125)?;
let forward_gap = max_abs_diff(&o, &reference)?;
let qkv_var = Var::from_tensor(&qkv)?;
let weights = Tensor::randn(0f32, 1.0, (batch, seq, heads * head_dim), &device)?;
let fused_grad = qkv_gradient(&qkv_var, &weights, heads, true)?;
let composed_grad = qkv_gradient(&qkv_var, &weights, heads, false)?;
let gradient_gap = max_abs_diff(&fused_grad, &composed_grad)?;
```

The two will not be bit-equal: the fused kernels sum in a different order (an online softmax over
key tiles). **Measure the bar rather than assuming one**: your composed attention's own spread
between the CPU and the GPU is the size of an honest rounding difference.

```rust
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
```

Expect gaps of the order of 1e-7 to 1e-6 in fp32; the example prints yours. A gap several times
the measured band is a real difference, worth investigating before training on it.

## 4. Check determinism

The CUDA backward is bit-reproducible: the key blocks add their dQ contributions in a fixed order,
with no atomics. Check it the only way that counts, bit for bit:

```rust
let again = qkv_gradient(&qkv_var, &weights, heads, true)?;
let (first, second) = (
    fused_grad.flatten_all()?.to_vec1::<f32>()?,
    again.flatten_all()?.to_vec1::<f32>()?,
);
let identical = first
    .iter()
    .zip(&second)
    .all(|(a, b)| a.to_bits() == b.to_bits());
```

## 5. Inside a training step

The op is differentiable like any candle op; nothing else in the loop changes. A toy setup — an
input, a qkv projection to learn, a target:

```rust
let hidden = heads * head_dim;
let x = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device)?;
let w_qkv =
    Var::from_tensor(&(Tensor::randn(0f32, 1.0, (hidden, 3 * hidden), &device)? * 0.02)?)?;
let target = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device)?;
```

and three steps of gradient descent through the fused op:

```rust
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
```

## 6. Measure the speed on your card

`bench/compare.py` compares the op with `PyTorch`'s fp32 attentions at a shape of your choice,
speed **and** accuracy against an fp64 reference (it needs `torch` and `safetensors`; `nsys` is
optional):

```bash
python bench/compare.py run --batch 64 --heads 6 --seq 240
```

Three rules make its numbers mean something: alternate the candidates in rounds within one session;
discard the first run of a freshly built binary (the driver compiles its kernels then); and compare
with `PyTorch` only on the same machine, in the same session.

## 7. When not to use it

- `head_dim` other than 64, or dtypes other than f32, on CUDA.
- You need the attention weights themselves (for interpretability hooks, for instance): the fused
  forward never materialises them. [candle-mi](https://github.com/mi-for-the-rust-of-us/candle-mi)
  takes the fused path only when no attention-internal hook is requested.
- Dropout on the attention weights, or an additive mask other than causal.
- A GPU older than Volta (compute capability 7.0).
