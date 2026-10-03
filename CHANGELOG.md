# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-03

### Changed

- **The backward is one kernel per key block, and still deterministic.** v0.1 computed dK/dV and
  dQ in two kernels that each recomputed S and dP (7 matmuls per tile). Each key block now computes
  dK, dV and a dQ partial in one pass (5 matmuls), and the key blocks add their dQ partials in
  key-block order under a turn counter per query tile (acquire/release at GPU scope, the ordered
  semaphore of CUTLASS's serial split-K). Every dQ is ((p0 + p1) + p2) + ... whatever the schedule,
  so reruns stay bit-identical; PyTorch's fp32 backward adds the same partials under a spin lock in
  arrival order. Backward kernels at b 64 · h 6 · s 240 · d 64: 4.19 -> 2.93 ms on an RTX 5060 Ti.
  The D kernel also zeroes the turn counters, so the backward is still two launches.
- dQ's error against fp64 dropped from 4.65e-7 to 4.01e-7 (normwise): the scale is now applied
  once, to the full sum.
- **The CUDA path now needs compute capability 7.0 (Volta) or newer** (`ld.acquire` /
  `st.release` at GPU scope). Tested on sm_120: RTX 5060 Ti and RTX 5090.
- MSRV 1.88, the floor candle-core 0.11 sets (`zip` 8.6), and the org's leaf-library MSRV.

### Added

- `bench/compare.py` + `examples/compare.rs`: the crate against PyTorch's fp32 attentions
  (`scaled_dot_product_attention`'s memory-efficient backend and a composed attention), speed AND
  accuracy, one method on both sides: shared seeded inputs, errors against an fp64 reference,
  bitwise-rerun checks, wall time measured identically in both languages over alternating rounds,
  and nsys kernel time per call.
- `docs/tutorial.md` and `examples/tutorial.rs`: the drop-in, checking the op on your own shapes,
  checking determinism, a training step, measuring, and when not to use it. CI runs the example and
  `scripts/check-tutorial.py` fails it if a tutorial code block stops matching the example.
- A runnable crate-level example (a doctest on the CPU path) and `# Shapes` sections on both
  public functions.
- `CONVENTIONS.md` (Grit + Grit-FA): the organisation's conventions, plus rules for CUDA kernels,
  the determinism contract and published claims; `bench/results/`, the reports behind the README's
  numbers.
- CI (CPU path, MSRV 1.88 and stable; rustdoc with private items; cargo-deny) and a Trusted
  Publishing release workflow.

## [0.1.0] - 2026-10-02

**Not published** — an internal milestone (commit `c9400e6`), recorded because the organisation's
notes and `bench/results/` call this version "v0.1". 0.2.0 is the first functional release on
crates.io.

### Added

- `fused_attention` (`CustomOp3` over `[batch, heads, seq, 64]` q, k, v with any strides,
  head_dim contiguous) and `fused_attention_qkv` (`CustomOp1` straight from a fused
  `[batch, seq, 3·heads·64]` projection, one interleaved gradient, no head split or merge).
- CUDA: the FlashAttention-2 algorithm in plain fp32 FFMA (online softmax, the `[seq, seq]`
  scores never written, only O and the row log-sum-exp saved), register micro-tiled; a
  deterministic backward in separate dK/dV and dQ kernels. CPU: the composed reference.
- The log-sum-exp saved in the op instance: candle hands `bwd` the op that ran the forward, so
  the forward is not re-run in `bwd`.

## [0.0.1] - 2026-10-03

**A name reservation, with no functionality**, published so that crates.io Trusted Publishing could
be configured before the first functional release (0.2.0), as for candle-mi and hypomnesis.
