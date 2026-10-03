# candle-fused-attn

> **ἐξ ἐλαχίστου πλεῖστον** · *ex elakhístou pleîston* · from the least, the most<br>
> **ἐξ ὀλίγων πολλά** · *ex olígōn pollá* · from few things, many

A fused **fp32** scaled-dot-product attention for [candle](https://github.com/huggingface/candle),
**forward and backward**, with a **bit-reproducible backward**: custom ops over **stock candle
0.11**, nothing to patch.

```rust
// The fast path: straight from a fused qkv projection [batch, seq, 3·heads·64] to the merged
// output [batch, seq, heads·64]; the backward writes ONE gradient, laid out like qkv.
let o = candle_fused_attn::fused_attention_qkv(&qkv, heads, 1.0 / 8.0, /* causal */ false)?;
// The general path: q, k, v f32 [batch, heads, seq, 64], any strides.
let o = candle_fused_attn::fused_attention(&q, &k, &v, 1.0 / 8.0, false)?;
```

**Why.** candle has no fused attention that can *train*, and none in fp32 on CUDA:
`candle-flash-attn` is f16/bf16 and forward-only. Training in fp32 is the point for work where
the numbers must be reproducible and comparable across runs. Prior art: kaio-candle 0.2.0 (fp32
forward and backward, single-head).

## Table of Contents

- [What it does](#what-it-does)
- [Measured](#measured)
- [Tutorial](#tutorial)
- [Limits](#limits)
- [Building and testing](#building-and-testing)
- [Used by](#used-by)
- [License](#license)
- [Development](#development)

## What it does

- **CUDA**: the FlashAttention-2 algorithm (online softmax; the `[seq, seq]` scores are never
  written; only O and the row log-sum-exp are saved) in plain fp32 FFMA, register micro-tiled.
- **A deterministic backward in one pass**: each key block computes dK, dV and a dQ partial,
  computing S and dP once per tile. The key blocks add their dQ partials **in key-block order**,
  under a turn counter per query tile (acquire/release at GPU scope, the ordered semaphore of
  CUTLASS's serial split-K). Reruns are bit-identical. PyTorch's fp32 backward adds the same
  partials under a spin lock in arrival order.
- **No copies in, none out**: q, k, v are read through their strides (the head split of a fused
  qkv projection is free; a layout the kernels cannot read in place is copied first), and O is
  written in the merge-heads layout.
- **No glue on the qkv path**: no head split or merge, and none of the zero-padded per-view
  gradients candle's `narrow` backward would build and sum. At the shape below, forward + backward
  costs 4.97 ms of kernel time on the RTX 5060 Ti, against 11.99 ms for the same attention through
  `fused_attention` on three `narrow` views of the projection (5090: 0.885 against 1.540 ms; the
  `candle_fused` rows of the reports below).
- **No re-run of the forward in `bwd`**: candle hands `bwd` the op instance that ran the forward,
  so the op keeps the row log-sum-exp in a field.
- **CPU**: the composed reference, so the gradient tests run anywhere.

## Measured

v0.2 at b 64 · h 6 · s 240 · d 64, non-causal. `bench/compare.py`: one unit for every candidate (fused qkv projection → merged output, backward
via `sum(o ∘ dout)`), shared seeded inputs, kernel time per call by nsys. Reports, committed:
[RTX 5090](bench/results/2026-10-03-rtx5090-compare.md) (torch 2.14),
[RTX 5060 Ti](bench/results/2026-10-03-rtx5060ti-compare.md) (torch 2.10), both 2026-10-03.

| kernel ms per call | forward (no-grad call) | forward + backward, net of the loss head |
|---|---|---|
| **RTX 5090**: this crate | 0.226 | **0.885** |
| RTX 5090: PyTorch SDPA (memory-efficient, CUTLASS) | **0.207** | 1.022 |
| RTX 5090: PyTorch, composed (matmul, softmax, matmul) | 0.443 | 1.310 |
| **RTX 5060 Ti**: this crate | 1.419 | 4.973 |
| RTX 5060 Ti: PyTorch SDPA | **0.945** | **4.814** |
| RTX 5060 Ti: PyTorch, composed | 1.843 | 5.760 |

The backward kernels are faster than SDPA's on both cards (5090: 0.597 against 0.767 ms;
5060 Ti: 2.93 against 3.66 ms, SDPA's own gradient concatenation included). The forward is not:
SDPA runs its matmuls on tensor cores as 3xTF32 (fp32-accurate by splitting), where this crate
uses plain FFMA, a cost of ×1.09 on the 5090 and ×1.5 on the 5060 Ti — enough on the 5060 Ti for
SDPA to stay ahead forward + backward.

**Accuracy** against an fp64 reference (normwise relative error; the same on both cards):

| | O | dQ | dK | dV |
|---|---|---|---|---|
| **this crate** | 3.9e-7 | 4.0e-7 | 4.6e-7 | 4.4e-7 |
| PyTorch SDPA (memory-efficient) | 5.7e-7 | 8.0e-7 | 8.0e-7 | 6.2e-7 |
| PyTorch, composed | 3.9e-7 | 4.1e-7 | 4.1e-7 | 3.9e-7 |
| fp64 rounded to fp32 (the floor) | 2.5e-8 | 2.5e-8 | 2.5e-8 | 2.5e-8 |

**In a training step** — a 6-layer, 384-wide masked-diffusion model, RTX 5090, batch 128, same
box, alternated rounds ([report](bench/results/2026-10-03-rtx5090-trainer-ab.md)): candle composed
attention 87 ms, v0.1 79 ms, **v0.2 75 ms per step**.

## Tutorial

[`docs/tutorial.md`](docs/tutorial.md): replacing a composed attention, checking the op on your own
shapes, checking determinism, the op inside a training step, measuring the speed, and when not to
use it. Its code is [`examples/tutorial.rs`](examples/tutorial.rs), which CI runs.

## Limits

- fp32 only. On CUDA, `head_dim` must be 64 (the CPU path takes any).
- No dropout, and no additive mask beyond `causal`.
- The CUDA path needs compute capability **7.0 (Volta) or newer**. Tested on sm_120: RTX 5060 Ti
  and RTX 5090.

## Building and testing

```bash
cargo test                     # the CPU path: no GPU, no CUDA toolkit
cargo test --features cuda     # the kernels: CUDA vs CPU within the measured band, bitwise reruns
bash scripts/ci-local.sh       # before pushing: CI and the release's checks, locally (--no-cuda to skip the GPU)
```

The `cuda` feature compiles the kernels with `nvcc` at build time (the CUDA toolkit must be
present, not just the driver), for the GPU `nvidia-smi` reports, or for `CUDA_COMPUTE_CAP`
(e.g. `120`) when set. The PTX is loaded into candle's own context and launched on candle's own
stream.

```bash
cargo run --release --features cuda --example bench [b h s]   # quick wall-time check
python bench/compare.py run    # vs PyTorch: speed AND accuracy (needs torch, safetensors, nsys optional)
```

## Used by

- [candle-mi](https://github.com/mi-for-the-rust-of-us/candle-mi) — mechanistic interpretability
  toolkit for language models. Its integration is in development, not yet in a release: a
  `fused-attn` feature routes `OthelloGpt`'s attention through `fused_attention_qkv` whenever no
  attention-internal hook is requested at that layer, and keeps the composed attention otherwise.
- askesis (research, not public) — the masked-diffusion planner these kernels were written for.
  Its trainer, through candle-mi, is the training step measured above
  ([report](bench/results/2026-10-03-rtx5090-trainer-ab.md)).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT)
at your option.

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from
  [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s
  [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset
  designed to improve AI coding accuracy — with Grit-FA rules for the CUDA kernels (cited C++
  references, constants twinned across Rust and CUDA, a stated summation order for every reduction)
  and for published claims.
- CI runs the CPU path on the MSRV (1.88) and stable, rustdoc with private items and cargo-deny.
  The CUDA tests need a GPU and run locally before each release. Releases are published from
  GitHub Actions through crates.io Trusted Publishing, after the maintainer's approval.
