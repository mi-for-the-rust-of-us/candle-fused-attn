# candle-fused-attn

A fused **fp32** scaled-dot-product attention for [candle](https://github.com/huggingface/candle),
**forward and backward**, as a `CustomOp3` over stock candle 0.11.

```rust
let o = candle_fused_attn::fused_attention(&q, &k, &v, 1.0 / 8.0, /* causal */ false)?;
// q, k, v: f32 [batch, heads, seq, 64], any strides with head_dim contiguous
```

- **CUDA**: the FlashAttention-2 algorithm (online softmax, the `[seq, seq]` scores never written,
  only O and the row log-sum-exp saved) in plain fp32 FFMA, register micro-tiled; a
  **deterministic** backward (separate dK/dV and dQ kernels, no atomics: reruns are bit-identical).
- **CPU**: the composed reference, so the gradient tests run anywhere.
- **No copies in, none out**: q, k, v are read through their strides (the head split of a fused
  qkv projection is free), and O is written in the merge-heads layout.
- **No re-run of the forward in `bwd`**: the op's single output packs `[O | L]`; the public function
  narrows O out of it for free.

Why: candle has no fused attention that can *train*, and none in f32 on CUDA (`candle-flash-attn`
is f16/bf16 and forward-only). Prior art: kaio-candle 0.2.0 (fp32 fwd+bwd, single-head).

## Status (v0.1, 2026-10-02)

head_dim 64 on CUDA; non-causal and causal; no dropout. Parity against candle's composed attention
(autograd): within 1–3× the composed path's own CPU-vs-CUDA spread (≤ 2e-6), forward and all
three gradients. RTX 5060 Ti, b 64 · h 6 · s 240 · d 64, per call (nsys): forward 1.36 ms; backward
0.11 + 2.12 + 1.82 ms.

`cargo test --features cuda` (needs the CUDA toolkit); `cargo test` for the CPU path;
`cargo run --release --features cuda --example bench [b h s]`.

License: MIT OR Apache-2.0.
