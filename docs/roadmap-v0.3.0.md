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
| 2 | L1: asynchronous K/V loads | [FA7](devlog.md#fa7--lever-l1-asynchronous-kv-loads-overlapped-with-the-math) | done — forward −42.6 %, bitwise identical, faster than SDPA's forward on the 5060 Ti |
| 2 | the backward, profiled | [FA8](devlog.md#fa8--the-backward-kernel-profiled-before-any-change) | done — long scoreboard on top, shared memory close |
| 2 | L1 for the backward | [FA9](devlog.md#fa9--lever-l1-for-the-backward-q-do-l-and-d-of-the-next-query-tile-loaded-during-this-one) | done — backward −14.3 %, bitwise identical |
| 2 | L2 design | [FA10](devlog.md#fa10--design-l2-fewer-shared-loads-per-ffma) | done — fragment double-buffering rejected (nvcc already pipelines) |
| 2 | L2, forward: 128 threads, 8 × 4 | [FA11](devlog.md#fa11--l2-forward-128-threads-8--4-outputs-per-thread) | **rejected** — +15.5 % (fewer warps cost more than the loads saved) |
| 3 | three cards (local 5060 Ti; rented 4090 and 5090), training step | [FA12](devlog.md#fa12--v030-on-three-cards-rtx-5060-ti-rtx-4090-rtx-5090) | done — bitwise identical on all; faster than SDPA (kernel time) on all |
| 3 | the 5090's no-grad forward and L2 | [FA13](devlog.md#fa13--does-l2-residency-explain-the-5090s-smaller-no-grad-forward-gain) | done — the gap follows L2 capacity |
| 3 | release 0.3.0 | — | in progress |
