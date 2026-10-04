# v0.3.0 — a faster forward: plan and log

The release's *why* is in [`ROADMAP.md`](../ROADMAP.md) § *Next*. This file holds its plan, every
prediction **written before** the measurement it predicts, and the results, in order. Rule: an
entry's predictions and decision rules are never edited once its measurement has started; a
correction is a new dated paragraph.

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

## 0a — provenance in `compare.py`, and the v0.2 baseline (registered 2026-10-04, before the run)

**Added to the report** (a `## Machine` section in `report.md`, all of it in `report.json`): the
crate's version and commit, with a fallback when the tree has no `.git` (the 5090 report of
2026-10-03 says "unavailable"); the `nvcc` release that produced the PTX (the driver compiles that
PTX to machine code at load, so the driver and `nvcc` versions are both part of the timed code);
`candle-core`'s version and source as `Cargo.lock` resolves it (registry or a patched path);
the OS and the driver model (WDDM / TCC); the CPU; the power limit and maximum clocks; the GPU's
state at the start and at the end of the run (P-state, clocks, temperature, power), beside the
per-process states already in `report.json`.

**The run:** `py -3.14 bench/compare.py run` with the defaults (b 64 · h 6 · s 240 · d 64,
non-causal, 3 rounds, nsys pass), RTX 5060 Ti, nothing else on the GPU (`hmn ps` recorded).
Expected ~2 min (the script's own figure).

**Prediction:** the baseline reproduces 2026-10-03's 5060 Ti report. Bar: every nsys kernel figure
of `candle_fused_qkv` and `torch_sdpa_efficient` within ±5 % of that report (fwd 1.419 / 0.945 ms,
net 4.973 / 4.814 ms per call; ±5 % is the drift seen between captures an hour apart on a warm
card), so the forward ratio ours / SDPA stays ×1.50 ± 0.10. A miss means the environment changed
(driver, toolkit, clocks) and is recorded as such, not averaged away.

---

## 0b — attention's share of a canvas evaluation (registered 2026-10-04, before the run)

**Why both binaries.** Local evaluations run today on the COMPOSED binary; the fused attention is
only in an opt-in build. So the question has two halves: what share of an evaluation is the fused
forward (what v0.3.0 could speed up), and is the fused binary already the faster evaluator (an
energy saving available now).

**Setup.** askesis `reference/canvas` (acsp14), two binaries rebuilt from ONE candle tree (fork
branch `gradstore-accumulate-0.11`, `655244d8`): composed (`--features cuda`) and fused
(`--features cuda,fused-attn`, candle-mi → candle-fused-attn 0.2.0 from crates.io); their md5
recorded. The eval: checkpoint `acspv_sasinv_px_s2_d1/canvas_ema_e80` (Me5 seed 2), the (8, 8)
tower probe (`tower_probe_sasinv`, 200 problems, `passk/tower_8_8.keys`), greedy, left-to-right,
batch 128, 101 rounds — the readout's own command. RTX 5060 Ti.

**Protocol.** Per binary, one warm-up run first, discarded (PTX JIT, cold caches). Then wall
rounds alternated C, F, F, C (the `decode wall-clock` the eval prints, which excludes loading).
Then one nsys capture per binary (`--trace=cuda`, `--force-overwrite=true`), and
`nsys stats --report cuda_gpu_kern_sum` written to a fresh file. Expected cost: ~17 s per eval,
~10 evals → ~3 min of GPU, plus the builds on the CPU.

**Predictions.**
- **P1:** the fused forward kernel (`fattn_fwd_f32_d64`) is **20–30 %** of the fused binary's
  evaluation kernel time. Basis: per layer and decode round at batch 128 · s 240, the linear
  layers are ~109 GFLOP (~7 ms at ~15 TFLOPS) against ~2.8 ms of attention (2 × the 1.42 ms
  measured at batch 64), plus ~1–1.5 ms of LayerNorm, GELU and residuals.
- **P2:** the fused binary evaluates **5–15 % faster** than the composed one (wall, decode only).
- **P3:** both give the same validity count within ±3 problems of 200 (rounding-level
  differences can flip a greedy decision; the composed binary read 154 / 200 on 2026-10-03).

**Decision rules (fixed now).**
- If the forward kernel's share is ≥ 15 %, matching SDPA's forward (÷1.5) saves ≥ 5 % of
  evaluation kernel time: v0.3.0 then pays on evaluations as well as on training on mid-range
  cards. Below 10 %, v0.3.0's priority is training only and is re-discussed.
- If the fused binary is faster (its worst round below the composed binary's best), propose
  running local evaluations on it — a rounding-level change of the readouts, to be disclosed in
  askesis's records before it is adopted.

### 0a — RESULT (2026-10-04, 07:24): the baseline reproduces 2026-10-03; prediction met

`compare.py` with the Machine section, crate `0ed05e7` (clean), RTX 5060 Ti (driver 610.88, WDDM;
nvcc 13.1; Ryzen 9 5950X), nsys 2025.5.2; 2 min 28 s.
Report: [`bench/results/2026-10-04-rtx5060ti-compare-v0.2-baseline.md`](../bench/results/2026-10-04-rtx5060ti-compare-v0.2-baseline.md).

| kernel time per call (ms) | 2026-10-03 | **2026-10-04** | change |
|---|--:|--:|--:|
| candle-fused-attn, forward | 1.419 | **1.387** | −2.3 % |
| candle-fused-attn, forward + backward (net) | 4.973 | **4.874** | −2.0 % |
| SDPA (memory-efficient), forward | 0.945 | **0.949** | +0.4 % |
| SDPA, forward + backward (net) | 4.814 | **4.708** | −2.2 % |

All four within the registered ±5 %; the forward ratio is ×1.46 (registered ×1.50 ± 0.10).
Accuracy against fp64 and the bitwise reruns are identical to 2026-10-03's. The card ran
P1 at 2,775 MHz, 32 → 43 °C, across the timed rounds. **This report is the baseline.**

### 0b — RESULT (2026-10-04, 07:28): P1 missed low (18.5 %), P2 and P3 met; both decision rules fire

Report: [`bench/results/2026-10-04-rtx5060ti-canvas-eval.md`](../bench/results/2026-10-04-rtx5060ti-canvas-eval.md)
(~2.5 min of GPU, as priced).

| | predicted | measured | |
|---|---|---|---|
| P1: fused forward's share of the evaluation's kernel time | 20–30 % | **18.5 %** (2.666 of 14.402 s) | **missed, low** |
| P2: fused evaluation faster (decode wall) | 5–15 % | **9.4 %** (15.67 vs 17.29–17.30 s) | met |
| P3: same validity within ±3 | ±3 | **154 = 154; the 200 plans identical** | met |

**Why P1 missed.** The linear layers' cuBLAS sgemm is 63.6 % of the fused evaluation (9.16 s);
the estimate assumed ~15 TFLOPS of sgemm where the card delivered less. The attention per call,
2.20 ms averaged over the batches of 128 and 72, was close to the estimate.

**Decision rules (as registered).**
- Share ≥ 15 % → **v0.3.0 pays on evaluations too**: at SDPA's forward speed (÷1.5), 2.67 s →
  ~1.78 s, about **−6 % of the evaluation's kernel time**, on top of the training gain.
- The fused binary is faster → **proposed: run local evaluations on the fused binary.** On this
  checkpoint it changes no plan at all; the change must still be disclosed in askesis's records
  before it is adopted (Éric's decision).

**Observed, not registered.** The evaluation is GPU-bound (the wall gain, 1.63 s, equals the
kernel gain, 1.61 s), and its largest consumer is the linear layers' sgemm — outside this crate.
