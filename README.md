# candle-fused-attn

A fused **fp32** scaled-dot-product attention for [candle](https://github.com/huggingface/candle),
**forward and backward**, as custom ops (`CustomOp1`, `CustomOp3`) over stock candle 0.11.

```rust
// The fast path: straight from a fused qkv projection [batch, seq, 3·heads·64] to the merged
// output [batch, seq, heads·64]; the backward writes ONE gradient, laid out like qkv.
let o = candle_fused_attn::fused_attention_qkv(&qkv, heads, 1.0 / 8.0, /* causal */ false)?;
// The general path: q, k, v f32 [batch, heads, seq, 64], any strides with head_dim contiguous.
let o = candle_fused_attn::fused_attention(&q, &k, &v, 1.0 / 8.0, false)?;
```

- **CUDA**: the FlashAttention-2 algorithm (online softmax, the `[seq, seq]` scores never written,
  only O and the row log-sum-exp saved) in plain fp32 FFMA, register micro-tiled; a
  **deterministic** backward (separate dK/dV and dQ kernels, no atomics: reruns are bit-identical).
- **CPU**: the composed reference, so the gradient tests run anywhere.
- **No copies in, none out**: q, k, v are read through their strides (the head split of a fused
  qkv projection is free), and O is written in the merge-heads layout.
- **No glue on the qkv path**: no head split, no head merge, and none of the zero-padded
  per-view gradients candle's `narrow` backward would build and sum (~3.4 ms per layer at the
  shape below — more than half the kernels' own time).
- **No re-run of the forward in `bwd`**: candle hands `bwd` the same op instance, so the op keeps
  the row log-sum-exp in a field; its output is O alone.

Why: candle has no fused attention that can *train*, and none in f32 on CUDA (`candle-flash-attn`
is f16/bf16 and forward-only). Prior art: kaio-candle 0.2.0 (fp32 fwd+bwd, single-head).

## Status (v0.1, 2026-10-02)

head_dim 64 on CUDA; non-causal and causal; no dropout. Parity against candle's composed attention
(autograd): within 1–2× the composed path's own CPU-vs-CUDA spread (≤ 3.4e-6), forward and all
three gradients. RTX 5060 Ti, b 64 · h 6 · s 240 · d 64, one layer, forward + backward (wall, patched candle):
`fused_attention_qkv` 7.0 ms, `fused_attention` 13.0 ms, composed 42.1 ms. Kernels (nsys):
forward 1.37 ms; backward 0.09 + 2.18 + 1.79 ms.

`cargo test --features cuda` (needs the CUDA toolkit); `cargo test` for the CPU path;
`cargo run --release --features cuda --example bench [b h s]`.

License: MIT OR Apache-2.0.
