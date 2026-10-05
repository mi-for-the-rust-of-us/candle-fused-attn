# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**Measured on five cards** (RTX 5060 Ti, 3090, 4090, 5090 and a first A100), 0.3.0 against 0.4.0
alternated in one session, outputs bit-identical: backward kernel −5.6 to −6.9 % on the consumer
cards, −0.5 % on the A100. Against PyTorch's fp32 SDPA, this crate stays faster on the consumer
cards; **on the A100, SDPA is 1.7× faster** (3xTF32 on tensor cores), and the README says so.
Every number and its report: `RESULTS.md`.

### Changed

- **The backward splits each paired product across the two halves of the block** (devlog FA14):
  in phase 1 half the threads compute S, the other half dP; in phase 2 half accumulate dV, the
  other half dK. Each thread now computes twice the outputs of one product for half the operand
  loads: shared-load instructions −20.9 %, same tiles, shared memory and occupancy. Outputs are
  **bitwise identical** to 0.3.0. Backward kernel against 0.3.0, alternated: RTX 5060 Ti −6.9 %,
  RTX 5090 −5.7 %, RTX 3090 −5.7 %, RTX 4090 −5.6 %, A100 −0.5 % (devlog FA17).
- `bench/box.sh` measures a released version (`BASE`, default `v0.3.0`) against the checkout,
  with `ab.py` at `ROUNDS` rounds (default 6).

### Added

- `bench/remote_box.sh`: runs `box.sh` on a rented machine from yours — the repository travels as
  a git bundle (nothing published), the run is detached, the results come home without the bulky
  bitwise dumps.
- The RTX 3090 and the A100 in `RESULTS.md` and the README, and a Limits line in the crate docs and the tutorial:
  on A100-class cards PyTorch's fp32 SDPA is the faster attention today.

## [0.3.0] - 2026-10-04

**Measured on three cards** (RTX 5060 Ti, 4090, 5090), 0.2.0 against 0.3.0 alternated in one
session, outputs bit-identical: forward kernel −37 to −44 % inside training calls, backward −9 to
−16 %, an attention training call −13 to −16 %; forward and forward + backward faster than PyTorch's
fp32 SDPA in kernel time on all three. Every number and its report: `RESULTS.md`.

### Changed

- **The forward overlaps its loads with its math** (devlog FA7). K and V tiles are copied with
  `cp.async` (compute capability 8.0+) in FlashAttention-2's order: V(n) arrives while QKᵀ is
  computed, K(n+1) while PV is. Outputs are **bitwise identical** to 0.2.0. RTX 5060 Ti, canvas
  shape, alternated with 0.2.0 in one session: forward kernel 1.480 → 0.850 ms (−42.6 %), now
  below PyTorch's fp32 SDPA forward (0.859 against 0.964 ms in one session). Below 8.0 the
  ordinary loads remain.
- **The backward overlaps its loads with its math too** (devlog FA9): the next query tile's Q, dO,
  L and D are copied with `cp.async` while the current one computes (Q and dO double-buffered;
  shared memory 87,552 B per block). Outputs bitwise identical; backward kernel −14.3 % against
  the forward-only change, alternated, RTX 5060 Ti.

### Added

- `bench/bitwise.py` (a bit-for-bit gate between two builds over a fixed set of shapes) and
  `bench/ab.py` (two builds' kernel times, alternated in one session).
- A Machine section in every `bench/compare.py` report: the crate commit, the `nvcc` that built
  the PTX, candle-core's source, OS and driver model, CPU, GPU limits and states.
- Build switch `CANDLE_FUSED_ATTN_SYNC_LOADS`: compiles the pre-8.0 load path on any card, to test it.
- `bench/box.sh`: one card's whole measurement (0.2.0 against the checkout, bit for bit and in
  time, and against PyTorch), as run on the three cards.
- `RESULTS.md`: every published number, per card and date, with its report; `docs/devlog.md`: every
  measurement registered before it ran, its result, and the rejected ideas.

### Documentation

- Limits: CUDA 13 toolkits no longer compile for Volta (compute capability 7.0); on a V100, build
  with CUDA 12. The code still supports it (ordinary loads below 8.0).

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
  numbers. Applied to the code: `#![deny(warnings)]` with the MSRV lint guard, `REFERENCE` /
  `ORDER` / `DETERMINISM` annotations in the kernels, `__launch_bounds__` on every kernel,
  shared-memory sizes derived from `TWIN` tile constants, and the determinism tests comparing
  `to_bits()`.
- CI (CPU path, MSRV 1.88 and stable; rustdoc with private items; cargo-deny) and a Trusted
  Publishing release workflow. Its `verify` job checks that the tag names `Cargo.toml`'s version
  and that `CHANGELOG.md` has its section; a manual run is a dry run by default (the Trusted
  Publishing token exchange, then `cargo publish --dry-run`; no upload, no GitHub Release).
- `scripts/ci-local.sh`: both workflows' commands run locally before pushing, verbatim, on 1.88 and
  stable, plus what GitHub's runners cannot run (clippy and the tests with `--features cuda`, and
  `cargo publish --dry-run`). It refuses to run if a workflow has a command it does not.

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
