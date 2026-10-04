# v0.3.0 — a faster forward: plan and log

The release's *why* is in [`ROADMAP.md`](../ROADMAP.md) § *Next*. This file holds its plan and
its status; every measurement — registered before it runs, then its result — is an entry of
[`devlog.md`](devlog.md) (moved there 2026-10-04: 0a, 0b, 0c became FA1–FA3).

## Plan (agreed 2026-10-04)

- **Phase 0 — instruments (local, RTX 5060 Ti).** 0a: complete provenance in
  `bench/compare.py`'s report, then the v0.2 baseline report every variant is compared with.
  0b: attention's share of a canvas evaluation (the roadmap's step 0). 0c: a kernel profiler —
  the installed Nsight Compute 2024.3 lists no Blackwell chip, so it needs 2025.x (Éric's call).
- **Phase 1 — C++ first.** PyTorch v2.10's `kernel_forward.h` (memory-efficient attention) and
  CUTLASS's fp32 SIMT tiling; a design section with a predicted gain per lever.
- **Phase 2 — local iteration.** The forward's tile constants as parameters; a few configurations,
  each gated by the bars (bitwise reruns, accuracy against fp64, CPU parity, candle-mi's
  `othello-fused` oracle) and timed with `compare.py`, alternated, one session.
- **Phase 3 — one rented RTX 5090** (and possibly a second card): `compare.py` and the trainer
  A/B, then the release through `CLAUDE.md`'s checklist.
- **Standing decision:** plain fp32 FFMA, no tensor cores (3xTF32 doubles SDPA's error against
  fp64: dQ 8.0e-7 against our 4.0e-7). The levers are occupancy and tiling.
- **Energy:** every GPU job is priced before it runs; local work on the 5060 Ti; one rental at
  the end, not one per idea.

---

## Status

| phase | step | devlog | state |
|---|---|---|---|
| 0 | 0a: provenance + the v0.2 baseline | [FA1](devlog.md#fa1--the-v02-baseline-on-the-rtx-5060-ti-v030-phase-0a) | done — met |
| 0 | 0b: attention's share of an evaluation | [FA2](devlog.md#fa2--attentions-share-of-a-canvas-evaluation-v030-phase-0b) | done — local evals moved to the fused binary |
| 0 | 0c: a profiler for Blackwell | [FA3](devlog.md#fa3--the-profiler-is-ready-v030-phase-0c) | done |
| 1 | reading SDPA's forward | [FA4](devlog.md#fa4--reading-sdpas-fp32-forward-pytorch-v210) | done |
| 1 | profiling both forwards | [FA5](devlog.md#fa5--the-first-profile-both-forward-kernels-at-the-canvas-shape) | done — same occupancy as SDPA; ours waits on global loads |
| 1 | design: a predicted gain per lever | [FA6](devlog.md#fa6--design-the-forwards-levers-ranked-with-predicted-gains) | done — L1 async loads, L2 register tiles, L3 occupancy |
| 2 | L1: asynchronous K/V loads | [FA7](devlog.md#fa7--lever-l1-asynchronous-kv-loads-overlapped-with-the-math) | registered |
| 3 | rented RTX 5090 + RTX 4090, release | — | — |
