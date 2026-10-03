# candle-fused-attn

> **ἐξ ἐλαχίστου πλεῖστον** · *ex elakhístou pleîston* · from the least, the most<br>
> **ἐξ ὀλίγων πολλά** · *ex olígōn pollá* · from few things, many

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
  **deterministic** backward: one kernel per key block computes dK, dV and a dQ partial, and the
  key blocks add their dQ partials in key-block order (a turn counter per query tile, no atomics):
  reruns are bit-identical, where PyTorch's fp32 backward adds them in arrival order.
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

2026-10-03, the one-kernel backward (S and dP computed once per tile: 5 matmuls, not 7). Kernels
(nsys, same shape): forward 1.44 ms, backward 0.12 + 2.81 ms (was 0.12 + 2.28 + 1.79). Against
PyTorch's fp32 attentions, `py -3.14 bench/compare.py run` (speed AND accuracy vs fp64, stock
candle 0.11, torch 2.10): the backward kernels now beat the memory-efficient SDPA backward (2.93
vs 3.31 ms + 0.35 ms of gradient concatenation), its forward still beats ours (0.92 vs 1.44 ms);
forward + backward net of the loss head, `fused_attention_qkv` 5.3 ms wall against SDPA 4.8 and
`canvas_parity.py`'s composition 5.9. Error vs fp64 (normwise): ours 3.9–4.6e-7, SDPA 5.7–8.0e-7.

`cargo test --features cuda` (needs the CUDA toolkit); `cargo test` for the CPU path;
`cargo run --release --features cuda --example bench [b h s]`.

License: MIT OR Apache-2.0.
