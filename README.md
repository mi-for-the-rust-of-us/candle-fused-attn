# candle-fused-attn

[![CI](https://github.com/mi-for-the-rust-of-us/candle-fused-attn/actions/workflows/ci.yml/badge.svg)](https://github.com/mi-for-the-rust-of-us/candle-fused-attn/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/candle-fused-attn.svg)](https://crates.io/crates/candle-fused-attn)
[![docs.rs](https://docs.rs/candle-fused-attn/badge.svg)](https://docs.rs/candle-fused-attn)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://www.rust-lang.org)
[![license](https://img.shields.io/crates/l/candle-fused-attn.svg)](https://github.com/mi-for-the-rust-of-us/candle-fused-attn#license)
[![unsafe: deny](https://img.shields.io/badge/unsafe-deny_(CUDA_launches_only)-blue.svg)](https://github.com/rust-secure-code/safety-dance/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-CUDA_sm__70%2B-76B900.svg?logo=nvidia&logoColor=white)](#limits)

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
the numbers must be reproducible and comparable across runs. Prior art:
[kaio-candle](https://github.com/dmriding/kaio/tree/main/kaio-candle) 0.2.0 (fp32 forward and
backward, single-head).

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

- **CUDA**: the FlashAttention-2 algorithm in plain fp32 FFMA — an online softmax, the `[seq, seq]`
  scores never written, only O and the row log-sum-exp saved.
- **Loads overlap the math**: both kernels copy their next tiles with asynchronous loads
  (`cp.async`, compute capability 8.0+) while the current tile computes.
- **A deterministic backward in one pass**: each key block computes dK, dV and a dQ partial, and
  the key blocks add their dQ partials **in key-block order** (a turn counter per query tile).
  Reruns are bit-identical; PyTorch's fp32 backward adds the same partials in arrival order.
- **No copies, no glue on the qkv path**: q, k, v are read through their strides, O is written in
  the merge-heads layout, and the backward writes one gradient shaped like the projection. On an
  RTX 5060 Ti that path costs 3.98 ms per forward + backward, against 11.25 ms for the same
  attention on three `narrow` views of the projection (RTX 5090: 0.89 against 2.13 ms).
- **No re-run of the forward in `bwd`**: the op keeps the row log-sum-exp from its forward.
- **CPU**: the composed reference, so the gradient tests run anywhere.

## Measured

Kernel time per call, in **milliseconds**, at the canvas trainer's shape (b 64 · h 6 · s 240 ·
d 64, non-causal), 2026-10-05; lower is better, **bold** = fastest. PyTorch SDPA is its fp32
`scaled_dot_product_attention` (memory-efficient backend, 3xTF32 on tensor cores), measured in the
same session. Every number and its report: [RESULTS.md](RESULTS.md).

| ![RTX 5060 Ti](https://img.shields.io/badge/RTX_5060_Ti-76B900?logo=nvidia&logoColor=white) | forward | forward + backward |
|---|--:|--:|
| **candle-fused-attn** 0.4.0 | **0.781** | **3.733** |
| PyTorch SDPA | 0.903 | 4.621 |

| ![RTX 4090](https://img.shields.io/badge/RTX_4090-76B900?logo=nvidia&logoColor=white) | forward | forward + backward |
|---|--:|--:|
| **candle-fused-attn** 0.4.0 | **0.255** | **1.183** |
| PyTorch SDPA | 0.333 | 1.555 |

| ![RTX 5090](https://img.shields.io/badge/RTX_5090-76B900?logo=nvidia&logoColor=white) | forward | forward + backward |
|---|--:|--:|
| **candle-fused-attn** 0.4.0 | **0.178** | **0.813** |
| PyTorch SDPA | 0.206 | 1.022 |

| ![A100](https://img.shields.io/badge/A100_SXM4-76B900?logo=nvidia&logoColor=white) | forward | forward + backward |
|---|--:|--:|
| candle-fused-attn 0.4.0 | 0.625 | 2.856 |
| **PyTorch SDPA** | **0.438** | **1.650** |

**On an A100, use PyTorch's SDPA if speed is the point**: its fp32 path runs 3xTF32 on tensor
cores, which on that card run at 8× the plain fp32 rate. On consumer cards that rate is about the
plain fp32 rate, and this crate's plain fp32 FMAs are faster.

**Accuracy** against fp64 (normwise relative error, dQ): candle-fused-attn 4.0e-7 on every card,
SDPA 8.0e-7 to 1.0e-6 — plain fp32 FMAs keep about half SDPA's error.

**0.3.0 → 0.4.0**, alternated in one session, outputs bit-identical: backward kernel −5.6 to −6.9 %
on the consumer cards, −0.5 % on the A100; the forward unchanged. **0.2.0 → 0.3.0**: forward kernel
−37 to −44 % inside training calls, backward −9 to −16 %. How, with every prediction and every
rejected idea: [`docs/devlog.md`](docs/devlog.md).

**In a training step** (a 6-layer masked-diffusion model through candle-mi) against its PyTorch
reference: 5.1× slower on 2026-07-29, **0.90×** on an RTX 5090 with 0.3.0.

## Tutorial

[`docs/tutorial.md`](docs/tutorial.md): replacing a composed attention, checking the op on your own
shapes, checking determinism, the op inside a training step, measuring the speed, and when not to
use it. Its code is [`examples/tutorial.rs`](examples/tutorial.rs), which CI runs.

## Limits

- fp32 only. On CUDA, `head_dim` must be 64 (the CPU path takes any).
- No dropout, and no additive mask beyond `causal`.
- The CUDA path needs compute capability **7.0 (Volta) or newer**; below 8.0 it uses ordinary
  loads instead of `cp.async`. CUDA 13 toolkits no longer compile for Volta: on a V100, build
  with CUDA 12. Tested on an RTX 5060 Ti, an RTX 4090, an RTX 5090 and an A100.
- Plain fp32 FMAs, no tensor cores: on an A100-class card PyTorch's fp32 SDPA is faster (above).

## Building and testing

```bash
cargo test                     # the CPU path: no GPU, no CUDA toolkit
cargo test --features cuda     # the kernels: CUDA vs CPU within the measured band, bitwise reruns
bash scripts/ci-local.sh       # before pushing: CI + the release's checks (--no-cuda: no GPU)
bash bench/box.sh <label>      # one card's measurement: 0.3.0 vs this checkout, and PyTorch
```

The `cuda` feature compiles the kernels with `nvcc` at build time (the CUDA toolkit must be
present, not just the driver), for the GPU `nvidia-smi` reports, or for `CUDA_COMPUTE_CAP`
(e.g. `120`) when set. The PTX is loaded into candle's own context and launched on candle's own
stream.

```bash
cargo run --release --features cuda --example bench [b h s]   # quick wall-time check
python bench/compare.py run    # vs PyTorch: speed AND accuracy (torch, safetensors; nsys optional)
```

## Used by

- [candle-mi](https://github.com/mi-for-the-rust-of-us/candle-mi) — mechanistic interpretability
  toolkit for language models. Its integration is in development, not yet in a release: a
  `fused-attn` feature routes `OthelloGpt`'s attention through `fused_attention_qkv` whenever no
  attention-internal hook is requested at that layer, and keeps the composed attention otherwise.
- askesis (research, not public) — the masked-diffusion planner these kernels were written for.
  Its trainer, through candle-mi, is the training step measured above
  ([report](bench/results/2026-10-04-rtx5090-trainer-ab-v0.2.0-vs-v0.3.md)).

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
- Plans: [ROADMAP.md](ROADMAP.md) (next: a faster forward); how a release is cut:
  [CLAUDE.md](CLAUDE.md).
- CI runs the CPU path on the MSRV (1.88) and stable, rustdoc with private items and cargo-deny.
  The CUDA tests need a GPU and run locally before each release. Releases are published from
  GitHub Actions through crates.io Trusted Publishing, after the maintainer's approval.
