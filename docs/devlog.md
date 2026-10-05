# candle-fused-attn — devlog

Chronological, across releases. **Rule:** a measurement's *registration* (question, protocol,
predictions, decision rules) is committed **before** it runs, so git dates it; its *result* is
added below it as a new dated part, never by editing the registration. A correction is a new
dated paragraph. Entries are `FA<n>` (fused attention), numbered in order. Plans live in the
per-release roadmaps ([`roadmap-v0.3.0.md`](roadmap-v0.3.0.md)); shipped history in
[`CHANGELOG.md`](../CHANGELOG.md).

## Index

| entry | date | question | verdict |
|---|---|---|---|
| [FA1](#fa1--the-v02-baseline-on-the-rtx-5060-ti-v030-phase-0a) | 2026-10-04 | does the v0.2 baseline reproduce 2026-10-03? | **met** (all within 5 %) |
| [FA2](#fa2--attentions-share-of-a-canvas-evaluation-v030-phase-0b) | 2026-10-04 | what share of an evaluation is the fused forward? is the fused binary faster? | P1 **missed** low (18.5 %); P2, P3 **met** |
| [FA3](#fa3--the-profiler-is-ready-v030-phase-0c) | 2026-10-04 | can Nsight Compute profile the 5060 Ti? | **yes**, after opening the counters |
| [FA4](#fa4--reading-sdpas-fp32-forward-pytorch-v210) | 2026-10-04 | how is SDPA's fp32 forward built? | reading: tensor cores (3xTF32); its "3 blocks per SM" does not hold on the 5060 Ti (FA5) |
| [FA5](#fa5--the-first-profile-both-forward-kernels-at-the-canvas-shape) | 2026-10-04 | why is our forward slower? | P1, P4, P5 **met**; P2 **missed** (SDPA has OUR occupancy); P3 **half missed** (top stall: global loads) |
| [FA6](#fa6--design-the-forwards-levers-ranked-with-predicted-gains) | 2026-10-04 | which levers, in which order? | design |
| [FA7](#fa7--lever-l1-asynchronous-kv-loads-overlapped-with-the-math) | 2026-10-04 | does overlapping the K/V loads with the math close the gap? | P1, P3, P5 **met**; P2 **missed high** (−42.6 %); P4 half missed (114 registers) — **kept** |
| [FA8](#fa8--the-backward-kernel-profiled-before-any-change) | 2026-10-04 | where does the backward kernel's time go? | P1, P2, P4, P5 **met**; P3 **missed** narrowly (barrier 4th) |
| [FA9](#fa9--lever-l1-for-the-backward-q-do-l-and-d-of-the-next-query-tile-loaded-during-this-one) | 2026-10-04 | does prefetching the next query tile speed the backward? | **all met**: backward −14.3 %, bitwise identical — **kept** |
| [FA10](#fa10--design-l2-fewer-shared-loads-per-ffma) | 2026-10-04 | how to cut shared loads per FFMA on a 100 KB/SM card? | design; explicit fragment double-buffering **rejected** before building |
| [FA11](#fa11--l2-forward-128-threads-8--4-outputs-per-thread) | 2026-10-04 | half the warps, twice the work per thread: faster or slower? | **slower** (+15.5 %): P1–P3 met, P4 missed — **rejected, reverted** |
| [FA12](#fa12--v030-on-three-cards-rtx-5060-ti-rtx-4090-rtx-5090) | 2026-10-04 | does v0.3.0 hold on other cards, and in the whole training step? | P1, P3, P4 **met**; P2 half (5090 no-grad −19.7 %); P5 **missed** low (+1.8 % step) — release |
| [FA13](#fa13--does-l2-residency-explain-the-5090s-smaller-no-grad-forward-gain) | 2026-10-04 | is the 5090's smaller no-grad forward gain L2 residency? | P1 **missed** (counters equal — under a profiler that erases the effect); P2, P3 **met**; the capacity pattern supports it |
| [FA14](#fa14--the-backward-split-by-product-half-the-threads-on-each-of-the-two-products) | 2026-10-05 | does splitting each paired product across thread halves cut the backward's shared loads, bit for bit? | **all met**: backward −6.9 % (low end), shared loads −20.9 %, bitwise identical — **kept** |
| [FA15](#fa15--the-backward-skips-its-dead-tail-work) | 2026-10-05 | does skipping the all-padding key groups, query rows and keys of the tail tiles speed the backward, bit for bit? | P1, P2, P4 **met**; P3 **missed** (+14.2 %, slower) — **rejected, reverted** |
| [FA16](#fa16--the-dv-half-stops-waiting-for-ds) | 2026-10-05 | does letting the dV half start as soon as P is written (a 128-thread barrier for the dK half) give back FA14's barrier cost, bit for bit? | P1, P4 **met**; P2, P3 **missed** (+1.4 %, barrier stall up) — **rejected, reverted** |
| [FA17](#fa17--fa14-on-three-rented-cards-rtx-5090-rtx-4090-and-a-first-a100) | 2026-10-05 | does FA14 hold off the 5060 Ti, and where does the crate stand on a datacenter card? | P1–P4 **met** (backward −5.7 / −5.6 / −0.5 %; SDPA 1.73× faster than ours on the A100); P5 not measurable on vast |

---

## FA1 — the v0.2 baseline on the RTX 5060 Ti (v0.3.0 phase 0a)

*Moved verbatim from `roadmap-v0.3.0.md` (registered in `bc4278a`, result in `1f990aa`).*

### 0a — provenance in `compare.py`, and the v0.2 baseline (registered 2026-10-04, before the run)

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

#### 0a — RESULT (2026-10-04, 07:24): the baseline reproduces 2026-10-03; prediction met

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

---

## FA2 — attention's share of a canvas evaluation (v0.3.0 phase 0b)

*Moved verbatim from `roadmap-v0.3.0.md` (registered in `bc4278a`, result in `464d837`).*

### 0b — attention's share of a canvas evaluation (registered 2026-10-04, before the run)

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

#### 0b — RESULT (2026-10-04, 07:28): P1 missed low (18.5 %), P2 and P3 met; both decision rules fire

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

---

## FA3 — the profiler is ready (v0.3.0 phase 0c)

*Moved verbatim from `roadmap-v0.3.0.md` (`9d95d9a`).*

#### 0c — the profiler is ready (2026-10-04)

Nsight Compute **2026.3.1** (winget `Nvidia.Nsight.Compute`; it replaced 2024.3, which listed no
Blackwell chip) lists `gb206` (RTX 5060 Ti) and `gb202` (RTX 5090). GPU performance counters
opened to all users in the NVIDIA Control Panel (Éric). Smoke test: one launch of
`fattn_fwd_f32_d64` at a deliberately tiny shape (b 4: 96 blocks on 36 SMs), `--section
SpeedOfLight`, 9 replay passes, ~3 s — it profiles; its numbers are not a measurement.

**Rule for the phases ahead:** Nsight Compute replays each kernel and locks the clocks (its default
`--clock-control base`), so its durations are not comparable with nsys or `compare.py`. It answers
*why* (achieved occupancy, warp stall reasons, shared-memory traffic, instruction mix); `compare.py`
answers *how fast*. Every profile is taken at the canvas shape (b 64 · h 6 · s 240 · d 64) unless
an entry says otherwise, and is registered before it runs, like any other measurement.

---

## FA4 — reading SDPA's fp32 forward (PyTorch v2.10)

*Phase 1, C++ first. Sources: the headers shipped with torch 2.10.0+cu130,
`ATen/native/transformers/cuda/mem_eff_attention/`; the kernel that ran on 2026-10-04 is named in
FA1's report (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`).*

- **Dispatch** (`kernels/cutlassF.h`): fp32 with `80 <= cc <= 120` — the RTX 5060 Ti and 5090
  (cc 12.0) included — goes to `dispatch_cutlassF_f32_sm80`, whose first kernel is
  `AttentionKernel<float, Sm80, aligned, 64, 64, 64, ...>`: 64 queries × 64 keys per block,
  `kMaxK` 64, so `kSingleValueIteration` and the output accumulator stays in registers (`rf`).
- **Warps and occupancy** (`kernel_forward.h`): `kNumWarpsPerBlock = 64 · 64 / (32 · 32) = 4`
  (128 threads); `kMinBlocksPerSm = getWarpsPerSmFw() / 4 = 12 / 4 = 3` for fp32, used in the
  kernel's `__launch_bounds__` — **at least 3 blocks, 12 warps, per SM by design**.
- **Arithmetic** (`gemm_kernel_utils.h`, `DefaultGemmType`): for fp32 and compute capability ≥ 8.0,
  `OpClassTensorOp`, `InstructionShape 16 × 8 × 8`, `OpMultiplyAddFastF32` — CUTLASS's
  **3xTF32 on tensor cores**. Plain FFMA (`OpClassSimt`) is only the fallback for older cards.
- **Ours (v0.2), for comparison:** the same 64 × 64 tile; 8 warps (256 threads); 69,632 B of
  dynamic shared memory per block, so **1 block, 8 warps, per SM**; plain FFMA, 4 × 4 outputs per
  thread.

**The headroom, from FA1's numbers.** The forward at the canvas shape is 4 · b · h · s² · d =
5.66 GFLOP (QKᵀ and PV). The RTX 5060 Ti has 36 SMs × 128 fp32 lanes = 4,608 lanes (device
query: 36 SMs, 100 KB shared memory and 48 warps per SM); at the 2,775 MHz it ran during FA1, its
FFMA peak is 4,608 × 2 × 2.775 GHz ≈ **25.6 TFLOPS**. Ours: 5.66 GFLOP in 1.387 ms ≈ **4.1
TFLOPS, ~16 % of that peak**. SDPA's 0.949 ms is the equivalent of ~6.0 TFLOPS, **~23 %**; tuned
fp32 matrix multiplies reach 60–80 %. **Parity without tensor cores is therefore within reach on
arithmetic grounds** — the gap is in how the kernel is fed, not in the card.


---

## FA5 — the first profile: both forward kernels at the canvas shape

*Registered 2026-10-04, before the run.*

**Question.** Why is our forward ~1.46× slower than SDPA's at the canvas shape? FA4 found two
differences (occupancy and tensor cores); this profile says where our kernel's time goes.

**Protocol.** RTX 5060 Ti, Nsight Compute 2026.3.1, `--set full` (default clock control: base
clocks locked, kernels replayed — metrics, not durations; FA3's rule). The same inputs for both,
FA1's `inputs.safetensors` (b 64 · h 6 · s 240 · d 64, seed 0), through `compare.py`'s own
`nsys` mode, forward phase only: our `fattn_fwd_f32_d64` from the `compare` example
(`candle_fused_qkv`), SDPA's `fmha_cutlassF_f32_aligned_64x64_rf_sm80` from the PyTorch side
(`torch_sdpa_efficient`); one launch each after the warm-up calls (`--launch-skip` past them).
Reports kept as `.ncu-rep`; the numbers below are read from them and summarised in
`bench/results/`. Cost: a few seconds of GPU per kernel.

**Predictions.**
- **P1 (ours):** theoretical occupancy limited by shared memory to 1 block = 8 of 48 warps
  (16.7 %); achieved occupancy 13–16.7 %.
- **P2 (SDPA):** 128 threads per block, ≤ 34 KB of shared memory per block, theoretical occupancy
  ≥ 12 warps per SM (≥ 25 %).
- **P3 (ours):** the top warp-stall reason is shared-memory traffic or synchronisation (MIO
  throttle, short scoreboard, or barrier), not math dependency; the FP32 (FMA) pipe is busy less
  than 35 % of the time.
- **P4:** SDPA issues tensor-core (MMA) instructions and ours none; our instruction mix is
  dominated by FFMA and shared-memory loads (LDS).
- **P5:** registers per thread: ours 64–128; SDPA ≤ 168 (its `__launch_bounds__(128, 3)`).

**Decision rule for Phase 2 (fixed now).** The levers are ranked by what FA5 shows:
- occupancy shared-memory-limited **and** top stall shared-memory traffic → first lever: cut
  shared memory per block (keep Q or P in registers, smaller staged tiles) to fit ≥ 2 blocks per SM;
- top stall barrier → restructure the synchronisation between phases first;
- FMA pipe already busy ≥ 50 % → occupancy will not help: raise arithmetic intensity first
  (larger per-thread register tiles).

### FA5 — RESULT (2026-10-04, 08:12): same occupancy as SDPA; our warps wait on global loads

Report: [`bench/results/2026-10-04-rtx5060ti-ncu-forward.md`](../bench/results/2026-10-04-rtx5060ti-ncu-forward.md);
raw `.ncu-repz` under `bench/results/profiles/`. ~42 s of profiling, as priced.

| | predicted | measured | |
|---|---|---|---|
| P1: our occupancy | 8 of 48 warps theoretical (16.7 %), 13–16.7 % achieved | 16.67 % / **16.35 %**, shared-memory-limited (1 block) | met |
| P2: SDPA | 128 threads, ≤ 34 KB shared memory, ≥ 12 warps per SM | 128 threads, **36,352 B**, **8 warps per SM** (2 blocks: shared-memory-limited) | **missed** |
| P3: our top stall; FMA pipe | shared memory or sync; FMA < 35 % | **long scoreboard** (global loads) 2.25, then short scoreboard 1.74; FMA pipes 22–25 % | **half missed** |
| P4: tensor cores; our mix | SDPA MMA, ours none; ours FFMA + LDS | SDPA 19.33 G TF32 ops, tensor pipe 42 %; ours 0; ours FMA and LSU pipes ~24 % each | met |
| P5: registers | ours 64–128; SDPA ≤ 168 | 80; 168 | met |

**What it says.**
- **Occupancy is not the difference on this card.** SDPA's block needs 36,352 B of shared memory,
  so only 2 of its "minimum 3" blocks fit in the 5060 Ti's 100 KB: both kernels run 8 warps per SM.
  SDPA turns them into 86 % compute throughput, mostly the tensor pipe (math-pipe throttle is its
  top stall: a saturated pipe, the sign of a well-fed kernel); ours, 30 %.
- **Our warps wait on memory, first global then shared.** Each key tile is loaded from global
  memory into shared memory by ordinary loads, then the block synchronises and computes: nothing
  overlaps the next tile's load with the current tile's math, and 8 warps per SM are too few to
  hide that latency by switching (long scoreboard, 2.25). Then every product reads its operands
  from shared memory (12.6 M shared loads against SDPA's 3.6 M; short scoreboard, 1.74). Our
  shared-memory layout itself is clean: 2.4 K bank conflicts against SDPA's 1.7 M.
- **3xTF32 confirmed to the operation:** 19.33 G TF32 operations = 3 × 6.44 GFLOP, the forward's
  arithmetic on tiles padded from s = 240 to 256.

**The registered decision rule.** Its first branch (occupancy shared-memory-limited **and** top
stall shared-memory traffic) half applies: occupancy is shared-memory-limited, but the top stall is
global-memory latency, a case the rule did not foresee. So the rule does not rank the levers by
itself, and the ranking goes to Phase 1's design entry, to be registered before any code:
latency hiding (asynchronous `cp.async` copies with double buffering, so the next tile loads while
this one computes — what CUTLASS and FlashAttention-2 do), fewer shared-memory loads per FFMA
(larger per-thread register tiles), and occupancy (it would also hide latency).

**Correction to FA4 (2026-10-04, after FA5).** FA4 read `kMinBlocksPerSm = 3` as "at least 3
blocks, 12 warps, per SM by design". It is a `__launch_bounds__` hint to the compiler (it caps
registers so that 3 blocks could fit); on the RTX 5060 Ti, shared memory admits only 2 blocks of
SDPA's forward, so it runs 8 warps per SM, like ours.

---

## FA6 — design: the forward's levers, ranked, with predicted gains

*Registered 2026-10-04, before any code. Phase 1's design entry; FA5's decision rule did not rank
the levers (its top stall, global-memory latency, was a case it had not foreseen).*

**C++ read first.** FlashAttention-2's forward main loop (`flash_fwd_kernel.h`, as vendored in
candle-flash-attn, lines 414–470; `kernel_traits.h`: `Has_cp_async` iff `__CUDA_ARCH__ >= 800`,
copies through `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`). With ONE K buffer and ONE V buffer it
overlaps every load with math: wait for K(n), sync, **issue V(n) asynchronously**, compute
S = Q·K(n)ᵀ while V travels; wait for V(n), sync, **issue K(n+1) asynchronously**, then softmax and
O += P·V(n) while K travels. Below compute capability 8.0 it falls back to ordinary loads.
v0.2's loop instead synchronises, loads K and V with ordinary loads (`ld.global` → registers →
`st.shared`: each thread waits for global memory), synchronises again, computes — every load
exposed, which is FA5's top stall (long scoreboard, 2.25 of 6.50 cycles between issues).

**The levers, in order.**

| | lever | attacks (FA5) | predicted on the forward kernel (5060 Ti, canvas shape) | cost |
|---|---|---|---|---|
| L1 | **asynchronous K/V loads, FA-2's schedule**: `cp.async.cg` 16-byte copies (zero-filled past `seq`), V(n) issued before QKᵀ, K(n+1) before softmax + PV; ordinary loads kept under `__CUDA_ARCH__ < 800` | long scoreboard (2.25) | **−15 to −30 %** (1.387 → 0.97–1.18 ms) | small: the loads and their syncs only; same shared memory, same arithmetic |
| L2 | **larger per-thread register tiles** (8 × 4 outputs instead of 4 × 4), fewer shared loads per FFMA | short scoreboard (1.74), 12.6 M shared loads | −10 to −20 % after L1 | medium: the thread-to-output mapping of both phases |
| L3 | **2 blocks per SM**: P out of shared memory (exchanged by warp shuffles within the 16 lanes that produce a row group) and a ≤ 50 KB block | latency hiding by more warps | uncertain; FA5 shows occupancy is not the gap by itself | large |

**Invariants for every lever.** Plain fp32 FFMA (no tensor cores); bitwise reruns; accuracy
against fp64 no worse than v0.2; CPU parity; candle-mi's `othello-fused` oracle. L1 changes no
arithmetic and no summation order, so it is held to a stronger bar: **bitwise identity with
v0.2** (FA7). L2 changes the per-thread accumulation grouping and is held to the accuracy bar.

**Order and stopping.** L1 first, measured alone (FA7). L2 is registered after L1's result, on
L1's profile. L3 only if L1 + L2 leave the forward short of SDPA's at the canvas shape. The same
L1 applies to the backward kernel's tile loads; that is a separate entry, after the forward.

---

## FA7 — lever L1: asynchronous K/V loads, overlapped with the math

*Registered 2026-10-04, before any code.*

**Change.** In `fattn_fwd_f32_d64` only: K and V tiles loaded with `cp.async.cg.shared.global`
(16 bytes per copy, source size 0 past `seq` so the row is zero-filled as today), in FA-2's order —
K(0) issued in the prologue with Q; per key tile: wait, sync, issue V(n), compute S; wait, sync,
issue K(n+1), softmax and PV. Same shared-memory layout and size (69,632 B), same tile constants,
same arithmetic in the same order. Under `__CUDA_ARCH__ < 800`, the v0.2 loads, unchanged.
`REFERENCE` (FA-2 `flash_fwd_kernel.h`) and `ORDER` annotations as `CONVENTIONS.md` requires.

**Protocol.**
1. **Bitwise identity**: for the canvas-shape inputs of FA1 and the parity tests' shapes (s = 7,
   97, 240; causal and not; both entry points), v0.3-L1's O and L equal v0.2's bit for bit; the
   crate's tests pass (`cargo test --features cuda`), and `scripts/ci-local.sh`.
2. **The fallback path**: built for `CUDA_COMPUTE_CAP=70` (PTX for compute 7.0, JIT-compiled on
   the 5060 Ti), the same bitwise identity with v0.2.
3. **Speed**: `compare.py` on the RTX 5060 Ti, v0.3-L1 against the v0.2 binary in alternated
   rounds of one session (the v0.2 binary kept under another name); nsys kernel time per call.
4. **Profile**: Nsight Compute `--set full` on one forward launch, as FA5.

**Predictions.**
- **P1:** O and L bitwise identical to v0.2 on every input of step 1, and on the fallback build.
- **P2:** forward kernel time per call **−15 to −30 %** against v0.2 in the same session.
- **P3:** long-scoreboard stall from 2.25 to **≤ 0.8** warp-cycles per issued instruction; the
  top stall becomes short scoreboard or barrier.
- **P4:** shared memory per block unchanged (69,632 B); registers per thread ≤ 96 (v0.2: 80).
- **P5:** the backward's kernel time unchanged within ±2 % (the backward is not touched).

**Decision rule (fixed now).** P1 is a gate: any bit difference stops the lever until explained.
If P2 shows ≥ 10 %, L1 is kept and L2 is registered on L1's profile. Below 5 %, L1 is reverted
with a `MEASURED-REVERT` note and the profile is re-read before anything else is tried.

### FA7 — RESULT (2026-10-04, ~09:15): −42.6 % on the forward, bitwise identical; faster than SDPA's forward on the 5060 Ti

Code: `aef44b9`. Reports: [`2026-10-04-rtx5060ti-ab-v0.2-vs-l1.md`](../bench/results/2026-10-04-rtx5060ti-ab-v0.2-vs-l1.md)
(the A/B), [`2026-10-04-rtx5060ti-compare-l1.md`](../bench/results/2026-10-04-rtx5060ti-compare-l1.md)
(against PyTorch, same session); profile `profiles/2026-10-04-rtx5060ti-fwd-candle-fused-attn-l1.ncu-repz`.
GPU used: ~5 min in all, as priced.

| | predicted | measured | |
|---|---|---|---|
| P1: bitwise identity with v0.2 | every input, native and fallback | **identical**, 6 shapes × 2 entry points × (`o`, `dqkv`), native build and forced-fallback build; the gate's negative control catches a one-ulp change | met |
| P2: forward kernel time vs v0.2, same session | −15 to −30 % | **−42.6 %** (1.480 → 0.850 ms; every L1 round below every v0.2 round) | **missed high** |
| P3: long-scoreboard stall | ≤ 0.8; top stall becomes short scoreboard or barrier | **0.18**; top stall short scoreboard (1.07) | met |
| P4: shared memory; registers | 69,632 B; ≤ 96 | 69,632 B; **114** (still 1 block per SM, set by shared memory) | half missed |
| P5: backward kernels | ±2 % | +0.3 % (`fattn_bwd_f32_d64`), −0.1 % (`fattn_bwd_dot_f32_d64`) | met |

Under Nsight Compute (v0.2 → L1): compute throughput 29.7 → **51.2 %**, warp cycles between
issues 6.50 → 3.82, FMA pipe 22 → 38 %, LSU pipe 24 → 44 %; global loads 414,720 → 0 (now
`cp.async`); shared loads unchanged (12.58 M), as the arithmetic is unchanged.

**Against PyTorch, same session** (`compare.py`): forward kernel time **0.859 ms against SDPA's
0.964** (×1.12 in our favour); forward + backward net 4.518 against 5.021 ms. Wall time: forward
0.837 against 0.970 ms (faster by the decision rule, ×1.16); training call, SDPA ahead (×1.29: the
training call includes candle's loss head and glue around the op); net of the head, not separated
(4.838 against 4.872 ms). **On the RTX 5060 Ti, v0.3.0's "done when" is met for the forward**; the
RTX 5090 is measured in Phase 3.

**Decision (the registered rule).** P1 holds and P2 ≥ 10 %: **L1 is kept**. L2 (larger register
tiles, against the remaining short-scoreboard stall) is registered next, on L1's profile.

**Deviations from the registration, recorded.**
- Step 2 registered the fallback as a build for compute capability 7.0. **CUDA 13.1's nvcc no
  longer targets compute 7.0 or 7.2** (`Unsupported gpu architecture`), and candle's own kernels do
  not compile for 7.5 with it (`candle-kernels/src/compatibility.cuh` redefines `__hmax_nan` /
  `__hmin_nan`, which CUDA 13.1's `cuda_fp16.hpp` now provides — an upstream candle issue). The
  fallback was instead compiled on the native target with `-DFATTN_SYNC_LOADS` (a test switch,
  `build.rs`: `$CANDLE_FUSED_ATTN_SYNC_LOADS`), which selects the same code a card below 8.0
  runs; our kernel alone, compiled for compute 7.5 with nvcc, has no `cp.async`, so the guard
  selects that path there.
- Observed, unrelated to L1: nvcc warns `#221-D` on `-INFINITY` (it expands to
  `-((float)(1e+300))` here); the value is still −∞, and the warning predates L1.

**For the release notes.** The README's "compute capability 7.0 (Volta) or newer" holds only with
a CUDA 12 toolkit; with CUDA 13 the buildable floor is 7.5 (Turing).

---

## FA8 — the backward kernel, profiled before any change

*Registered 2026-10-04, before the run. Why now: per training step at batch 64, attention costs
about 18 forward calls × 0.85 ms (after FA7) against 6 backward calls × 2.95 ms — the backward is
now the larger share, and it loads its Q and dO tiles inside its query loop the same exposed way
v0.2's forward did.*

**Protocol.** As FA5: Nsight Compute 2026.3.1, `--set full`, one launch after the warm-up calls,
FA1's inputs (canvas shape), `compare.py`'s `nsys` mode in the `train` phase: our
`fattn_bwd_f32_d64` (crate `26eb8bb`, the backward untouched since v0.2) and SDPA's
`fmha_cutlassB_f32_aligned_64x64_k64_sm80`. A few seconds of GPU each.

**What the kernel does (v0.2, unchanged).** One block per key block of 64 keys (K, V loaded once);
it walks query tiles of 32: per tile it loads Q and dO with ordinary loads (`load_tile4`), reads
L and D, computes S, dP, P, dS, accumulates dK and dV in registers, then adds its dQ partial into
global memory **on its turn** (thread 0 spins on an acquire load until the previous key block has
passed the turn; the block waits at a barrier).

**Predictions.**
- **P1:** occupancy 1 block = 8 warps per SM (69,632 B of shared memory), achieved 14–16.7 %.
- **P2:** long scoreboard is one of the two top stalls (≥ 1.0 warp-cycles per issued
  instruction): the Q / dO tile loads and the dQ read through L2 (`__ldcg`).
- **P3:** barrier is among the top three stalls (≥ 0.5): every thread waits while thread 0 spins
  for the dQ turn.
- **P4:** compute (SM) throughput ≤ 40 %.
- **P5:** SDPA's backward runs on the tensor pipe (TF32), and ours does not.

**Decision rule (fixed now).**
- Long scoreboard the top stall → the next lever is FA7's, for the backward: Q and dO of the next
  query tile copied with `cp.async` while the current one computes (registered as its own entry,
  with the bitwise gate).
- Barrier the top stall → the dQ turn order is the bottleneck; that lever (how blocks hand over
  dQ) is designed first.
- Neither → read the profile before anything is registered.

### FA8 — RESULT (2026-10-04, ~09:35): long scoreboard on top, but shared memory nearly level

Profiles: `profiles/2026-10-04-rtx5060ti-bwd-{candle-fused-attn-v0.2,sdpa-mem-eff}.ncu-repz`.
Convention: "selected" (the cycle a warp issues) is not counted as a stall.

| | predicted | measured | |
|---|---|---|---|
| P1: occupancy | 8 warps per SM; 14–16.7 % achieved | 69,632 B, 1 block; **16.65 %** | met |
| P2: long scoreboard in the top two, ≥ 1.0 | | **1.16, first**; short scoreboard 1.11, second | met |
| P3: barrier in the top three, ≥ 0.5 | | 0.66, **fourth** (MIO throttle 0.68 third) | **missed** (narrowly) |
| P4: compute throughput ≤ 40 % | | 34.9 % | met |
| P5: SDPA on the tensor pipe, ours not | | SDPA 48.3 G TF32 operations, tensor pipe 30.6 %; ours 0 | met |

| | ours `fattn_bwd_f32_d64` | SDPA `fmha_cutlassB_f32_aligned_64x64_k64_sm80` |
|---|--:|--:|
| threads; registers; shared memory per block | 256; 128; 69,632 B | 128; 232; 53,504 B |
| warps per SM (theoretical / achieved) | 8 / 7.99 | **4** / 4.0 |
| duration under ncu (clocks locked) | **2.89 ms** | 3.33 ms |
| compute throughput; issue slots busy | 34.9 %; 33.0 % | 61.8 %; 15.4 % |
| instructions (warp level); shared loads | 357.6 M; **40.9 M** | 192.2 M; 10.7 M |
| top stalls | long sb 1.16, short sb 1.11, MIO throttle 0.68, barrier 0.66 | math pipe throttle 1.98, wait 1.65, long sb 0.77 |

**Reading.** Our backward is already faster than SDPA's, which runs only 4 warps per SM here.
Its stalls are more balanced than v0.2's forward was: global-load latency is ~19 % of the cycles
between issues (forward: ~35 %), and shared memory is close behind — 3.8× SDPA's shared loads,
and a filling MIO queue. **Decision (the registered rule):** long scoreboard is the top stall, so
FA7's lever comes next for the backward (FA9), with a smaller expected gain; fewer shared loads
per FFMA (L2) matters for the backward as much as for the forward.

---

## FA9 — lever L1 for the backward: Q, dO, L and D of the next query tile loaded during this one

*Registered 2026-10-04, before any code.*

**Change.** In `fattn_bwd_f32_d64`, the per-query-tile loads (`load_tile4` of Q and dO, lines
217–218 at `26eb8bb`, and the per-row reads of L and D in phase 1) become `cp.async` copies of
the NEXT query tile, issued at the start of the current one into a second Q / dO buffer (double
buffering: the current tile's Q and dO are read until the end of phase 2, so a single buffer
could only overlap phase 3). L and D of the tile go to shared memory with them. Shared memory
grows by 2 × 32 × 68 floats + 2 × 2 × 32 floats = 17,920 B, to 87,552 B: still one block per SM
(100 KB). The arithmetic and the dQ turn order are unchanged. Below compute capability 8.0 (and
under `-DFATTN_SYNC_LOADS`), the same double-buffered schedule with ordinary loads.
`REFERENCE`: FlashAttention-2's backward (`flash_bwd_kernel.h`, its `Double_buffer` option).

**Predictions.**
- **P1 (gate):** dQ, dK, dV (and O) bitwise identical to `26eb8bb`'s on the 6 shapes of
  `bench/bitwise.py`, native and forced-fallback builds.
- **P2:** backward kernel time **−8 to −18 %** in an alternated A/B against `26eb8bb`
  (`bench/ab.py`, `train` phase), the forward unchanged within ±2 %.
- **P3:** long scoreboard from 1.16 to **≤ 0.5**; the top stall becomes short scoreboard or MIO
  throttle.
- **P4:** shared memory 87,552 B; registers ≤ 168 (at 256 threads, the most that still fits one
  block); occupancy unchanged.

**Decision rule (fixed now).** P1 is a gate. Keep at ≥ 5 % on the backward kernel; below 3 %,
revert with a `MEASURED-REVERT` note. Then L2 (register tiles), designed for both kernels at once,
since both are now limited by shared-memory traffic.

**Correction to FA9's REFERENCE (2026-10-04, read after registering, before any code).** Read
upstream (`Dao-AILab/flash-attention`, `csrc/flash_attn/src/flash_bwd_kernel.h`, main loop):
FA-2's `Double_buffer` applies to **sQ only** (`tQsQ` alternates between two halves every
`m_block`); **dO is single-buffered** and its next tile is copied right after its last use (the
dV GEMM, behind a `__syncthreads` "since we're writing to the same sdO location"), so that copy
overlaps the dQ and dK GEMMs; the next tile's LSE is read into registers in the loop. FA9 keeps its
registered design — both Q and dO double-buffered — because in our kernel dO's last use is phase 2,
so a single dO buffer would overlap phase 3 only; FA-2's choice saves 8,704 B that we do not need
(one block per SM either way). The `REFERENCE` comment in the code will say both.

### FA9 — RESULT (2026-10-04, ~10:00): backward −14.3 %, bitwise identical; every prediction met

Code: `487b4f2`. Report: [`2026-10-04-rtx5060ti-ab-l1-vs-fa9.md`](../bench/results/2026-10-04-rtx5060ti-ab-l1-vs-fa9.md);
profile `profiles/2026-10-04-rtx5060ti-bwd-candle-fused-attn-fa9.ncu-repz`. GPU used: ~2 min.

| | predicted | measured | |
|---|---|---|---|
| P1: bitwise identity | dQ, dK, dV, O; native and fallback | **identical** (6 shapes × 2 entry points; native, forced fallback) | met |
| P2: backward kernel, alternated vs L1 | −8 to −18 %; forward ±2 % | **−14.3 %** (2.950 → 2.527 ms); forward −0.1 % in the training call, +1.5 % alone (not separated) | met |
| P3: long scoreboard | ≤ 0.5; top stall short scoreboard or MIO throttle | **0.43**; top stall short scoreboard (1.28) | met |
| P4: shared memory; registers; occupancy | 87,552 B; ≤ 168; unchanged | 87,552 B; 128; 16.64 % | met |

Also: barrier stall 0.66 → 0.38 (one barrier per tile fewer), compute throughput 34.9 → 42.2 %,
warp cycles between issues 5.96 → 5.07; a whole training call −7.2 % (every FA9 round faster).
The remaining top stall in both kernels is now shared-memory latency (short scoreboard): **L2,
larger register tiles, is next, designed for both kernels**.

**Procedure note.** The build of FA9 overwrote the L1 `compare.exe` that the A/B needed as its
reference; L1 was rebuilt from its commit and checked bitwise identical before use. Binaries are
not identified by md5 across links on Windows (the linker stamps a time in every executable).

---

## FA10 — design: L2, fewer shared loads per FFMA

*2026-10-04, after FA9; before any code. Both kernels' top stall is now short scoreboard (waiting
on shared-memory loads): forward 1.07, backward 1.28 warp-cycles per issued instruction.*

**Rejected before building: explicit register-fragment double-buffering.** CUTLASS's pipelined
main loop (torch 2.10 headers, `mem_eff_attention/gemm/custom_mma_pipelined.h`, lines 294–370:
`warp_frag_A[2]`, the k + 1 fragments loaded before the k-th `warp_mma`) is the textbook answer
to shared-load latency. But nvcc already does it for us: in the SASS of `487b4f2` (`ptxas
-arch=sm_120a`), the distance from each `LDS` to the first instruction that reads its result is
a **median of 29 instructions in the forward and 36 in the backward**; only 12 % and 4 % of the
loads are consumed within 8 instructions. An explicit version would move little; not built.

**Counted, not guessed: shared loads per FFMA.** Per thread and tile (`LDS.128` instructions;
these counts reproduce FA5's and FA8's totals exactly: 12.58 M and 40.89 M):

| kernel, phase | today (256 threads) | LDS / FFMA |
|---|---|--:|
| forward, S = QKᵀ: 4 rows × 4 keys | 16 steps × (4 Q + 4 K) = 128 LDS, 1,024 FFMA | 1 / 8 |
| forward, O += PV: 4 rows × 4 dims | 16 × (4 P + 4 V) = 128 LDS, 1,024 FFMA | 1 / 8 |
| backward, S and dP: 2 rows × 4 keys | 16 × (2 Q + 2 dO + 4 K + 4 V) = 192 LDS, 1,024 FFMA | 1 / 5.3 |
| backward, dV and dK: 4 keys × 4 dims | 32 × (P + dS + dO + Q) = 128 LDS, 1,024 FFMA | 1 / 8 |
| backward, dQ partial: 2 rows × 4 dims | 16 × (2 dS + 4 K) = 96 LDS, 512 FFMA | 1 / 5.3 |

**The constraint.** Outputs per thread = tile area ÷ threads. At today's tiles and 256 threads,
that is 16 per thread (4 × 4). Larger tiles do not fit the 99 KB a block may hold: a forward with
128-query tiles needs Q, K, V, P = (128 + 64 + 64 + 128) rows × 68 floats ≈ 104 KB; a backward
with 64-query tiles, ≈ 139 KB with FA9's double buffers (≈ 104 KB without). So **more work per
thread at the same shared memory means fewer threads: 128 threads, 4 warps per SM instead of 8**
— the configuration SDPA's backward runs in (FA8). Fewer warps hide latency less; more
independent FFMAs per thread hide it more. Which wins on this card is not predictable from
counts; it is measured, on the simpler kernel first.

| candidate | threads, outputs per thread | shared loads per block and tile | warps per SM |
|---|---|--:|--:|
| forward today | 256, 4 × 4 | 65,536 | 8 |
| **forward L2 (FA11)**: rows 8·ty … + 8, keys / dims as today | 128, **8 × 4** | **49,152 (−25 %)** | 4 |
| backward today | 256 | 106,496 | 8 |
| backward L2 (later, if FA11 wins): S, dP 4 × 4; dK, dV 8 × 4; dQ 4 × 4 | 128 | 73,728 (−31 %) | 4 |

**Correction to FA6.** FA6 held L2 to the accuracy bar, assuming larger register tiles regroup the
sums. They need not: each output's accumulation keeps its order (over the head dimension in
phase 1, over keys in phase 2, over queries for dK and dV, over keys for dQ, and the softmax's
16-lane butterfly is unchanged) — only *which thread* computes it changes. **The bitwise gate
applies to L2.**


---

## FA11 — L2, forward: 128 threads, 8 × 4 outputs per thread

*Registered 2026-10-04, before any code.*

**Change.** `fattn_fwd_f32_d64` only, launched with 128 threads (`THREADS`/`NT` split: the
forward's own `FW_THREADS` = 128, a new `TWIN`; the backward and the D kernel keep 256). Thread
(ty = tid / 16, 0–7; tx = tid % 16): phase 1 computes S for rows 8·ty … + 8 and keys tx + 16·jj
(jj 0–3); phase 2 accumulates O for the same 8 rows and dims 4·tx … + 4. Same tiles (64 × 64),
same shared memory (69,632 B), same L1 load schedule (the tile loops stride by 128 threads), same
per-output order everywhere: the dot products over the head dimension, the 16-lane row max / sum
butterflies, PV over keys in order.

**Predictions.**
- **P1 (gate):** O and dqkv bitwise identical to v0.2 on `bench/bitwise.py`'s 6 shapes, native
  and forced-fallback builds.
- **P2:** registers per thread 140–220; shared loads per launch −25 % (12.58 M → 9.44 M
  warp-level); achieved occupancy ≈ 8.3 % (4 warps per SM).
- **P3:** short-scoreboard stall per issued instruction down from 1.07; "no eligible" (no warp
  ready to issue) up from FA7's level — the trade made visible.
- **P4:** forward kernel time against FA9's build, alternated (`bench/ab.py`): **between −15 % and
  +15 %** — the honest range for a trade the counts cannot settle; the backward unchanged ±2 %.

**Decision rule (fixed now).** P1 is a gate. Faster by ≥ 5 % → kept, and the backward's L2 is
registered on the same pattern. Within ±5 % or slower → reverted (`MEASURED-REVERT`), recorded as
rejected with its numbers, and the next design looks at larger tiles with swizzled (unpadded)
shared memory instead, which keeps 8 warps per SM.

### FA11 — RESULT (2026-10-04, ~10:50): bitwise identical, 25 % fewer shared loads, and 15.5 % slower — rejected

Code as measured: `c1dfba2`; reverted by `dbf623a` (sources identical to FA9's `487b4f2`).
Profile: `profiles/2026-10-04-rtx5060ti-fwd-candle-fused-attn-fa11-rejected.ncu-repz`.
A/B against FA9's binary, alternated (`bench/ab.py`, `target/ab/fa11`).

| | predicted | measured | |
|---|---|---|---|
| P1: bitwise identity | native and fallback | **identical** (6 shapes × 2 entry points; both builds) | met |
| P2: registers; shared loads; occupancy | 140–220; −25 % (9.44 M); ≈ 8.3 % | 168; **9,437,184** (−25.0 %); 8.35 % | met |
| P3: short scoreboard down; "no eligible" up | | short scoreboard 1.07 → **0.42**; no eligible 47.8 → **57.9 %** | met |
| P4: forward kernel time vs FA9 | −15 % to +15 % | **+15.5 %** (0.860 → 0.994 ms); training call +2.7 % | **missed** (slower) |

| forward, under Nsight Compute | FA9 (256 threads, 4 × 4) | FA11 (128 threads, 8 × 4) |
|---|--:|--:|
| warp cycles per issued instruction | 3.82 | **2.35** |
| issue slots busy | **51.2 %** | 41.5 % |
| no eligible warp | 47.8 % | 57.9 % |
| FMA pipe; LSU pipe | 38.0 %; 44.2 % | 32.2 %; 29.2 % |

**Reading.** Each warp got better — 38 % fewer cycles between its instructions — but with one
warp per scheduler there is nothing to issue while that warp waits, so the schedulers idle more.
On this card, at this shape, **8 warps per SM are worth more than 25 % of the shared loads**.
Recorded so that nobody re-proposes it blind: the kernel's header now names the variant and its
number. **Next design** (FA10's fallback): larger tiles at 256 threads, keeping 8 warps per SM,
which needs swizzled, unpadded shared memory to fit the 99 KB a block may hold — a larger change,
to be weighed against what the release needs (see the roadmap).

---

## FA12 — v0.3.0 on three cards: RTX 5060 Ti, RTX 4090, RTX 5090

*Registered 2026-10-04, before the rentals; the local dry run of the procedure is reported with it.*

**Protocol.** `bench/box.sh <label>` on each machine, from a clone at the same commit: records the
machine; builds v0.2.0 (its tag, in a worktree) and this checkout; the bitwise gate between them;
the CUDA tests; `compare.py` (this checkout against PyTorch); `bench/ab.py` (v0.2.0 against this
checkout, alternated). On the RTX 5090 only, also the canvas training step through candle-mi:
candle-fused-attn 0.2.0 against this checkout, alternated, and PyTorch's reference step, on one
box (askesis's `torch_vs_candle.sh` with `CANVAS_BIN`).

**Predictions (both rented cards unless said).**
- **P1:** bitwise identity, v0.2.0 against this checkout, on each card; the CUDA tests pass.
- **P2:** forward kernel, v0.3 against v0.2.0: **−30 to −45 %**.
- **P3:** backward kernel: **−8 to −20 %**.
- **P4:** against SDPA in the same session: our forward kernel ≤ SDPA's (on the 5090 it was ×1.09
  slower with v0.2); forward + backward net below SDPA's.
- **P5 (5090):** the whole training step at batch 128, v0.3 against 0.2.0: **−4 to −8 %**
  (attention ≈ a quarter of the step: ≈ 8 ms forward and 8 ms backward of ≈ 67 ms, cut ≈ 40 % and
  ≈ 15 %); against PyTorch, **0.85× to 0.90×** (2026-10-03: 0.93×).

**Decision rule.** A card failing P1 stops the release until explained. Otherwise 0.3.0 is
released with every card's numbers, whatever P2–P5 read, and the README and `RESULTS.md` carry
them.

**Local dry run (RTX 5060 Ti, 2026-10-04 11:14–11:20, 5 min 48 s; reports
`bench/results/2026-10-04-rtx5060ti-{compare-v0.3,ab-v0.2.0-vs-v0.3}.md`).** The procedure runs
end to end. Bitwise identical; tests pass. v0.2.0 against v0.3, one session: forward kernel
**−40.0 %**, backward **−16.4 %**, training call **−16.2 %** (the chained FA7 × FA9 estimate,
−16 %, confirmed directly). Against SDPA: forward 0.895 against 0.978 ms, forward + backward net
**3.982 against 4.963 ms (−20 %)**; in wall time, forward ×1.10 and net ×1.07 faster.

### FA12 — RESULT (2026-10-04, ~10:20): bitwise identical on three cards; the attention gains hold; the whole training step gains less than predicted

Reports (`bench/results/`): `2026-10-04-{rtx4090,rtx5090}-{compare-v0.3,ab-v0.2.0-vs-v0.3}.md`,
`2026-10-04-rtx5090-trainer-ab-v0.2.0-vs-v0.3.md`; the 5060 Ti's are the dry run's. Rentals: askesis
`rentals.md` (4090 $0.16, 5090 $0.20).

| v0.2.0 → v0.3, alternated (`ab.py`) | RTX 5060 Ti | RTX 4090 | RTX 5090 |
|---|--:|--:|--:|
| bitwise identity; CUDA tests | identical; pass | identical; pass | identical; pass |
| forward kernel, no-grad calls | −40.0 % | −36.4 % | **−19.7 %** |
| forward kernel, inside training calls | −43.6 % | −37.3 % | −38.5 % |
| backward kernel | −16.4 % | −11.9 % | −8.9 % |
| training call (forward + head + backward) | −16.2 % | −13.0 % | −12.8 % |

| v0.3 against SDPA, same session (kernel ms) | RTX 5060 Ti | RTX 4090 | RTX 5090 |
|---|--:|--:|--:|
| forward: ours / SDPA | **0.895** / 0.978 | **0.254** / 0.334 | **0.192** / 0.243 |
| forward + backward net: ours / SDPA | **3.982** / 4.963 | **1.200** / 1.559 | **0.891** / 1.104 |
| wall time, net (decision rule) | ours ×1.07 | **SDPA ×1.19** | ours ×1.11 |

| | predicted | measured | |
|---|---|---|---|
| P1 | bitwise identical; tests pass | all three cards | met |
| P2 | forward −30 to −45 % | 4090 −36.4 %; 5090 −19.7 % in no-grad calls, −38.5 % inside training calls | half met |
| P3 | backward −8 to −20 % | 4090 −11.9 %; 5090 −8.9 % | met |
| P4 | forward and net kernel time below SDPA's | met on all three cards (wall time: SDPA still ahead net on the 4090) | met |
| P5 | 5090 step −4 to −8 %; 0.85–0.90× PyTorch | **+1.8 % throughput** (65 against 66–67 ms); ≈ **0.90×** (0.92× with v0.2.0, same session) | **missed** low |

**Reading.** The kernels' gains hold everywhere, bit for bit. Two things limit them. (1) On the
5090, v0.2's forward is itself fast in back-to-back no-grad calls (0.237 against 0.308 ms inside
training calls): the hypothesis is L2 residency — the 70.8 MB `qkv` fits the 5090's L2 (~96 MB)
but not the 4090's (72 MB) or the 5060 Ti's (32 MB); the box blocked the GPU counters
(`ERR_NVGPUCTRPERM`), so it is tested locally next (FA13), at a batch whose `qkv` fits 32 MB.
(2) The whole training step gains +1.8 %, not 4–8 %: attention is a smaller share of the step than
assumed, and the step already waits partly on the host (rentals.md, 2026-10-03). On the 4090,
candle's glue around the op (separate small kernels and their launches) outweighs the kernels in
wall time. **Decision (the registered rule): no card failed P1 — release 0.3.0 with every card's
numbers.**

---

## FA13 — does L2 residency explain the 5090's smaller no-grad forward gain?

*Registered 2026-10-04, before the run (FA12's open question; the rented box blocked the counters).*

**Hypothesis.** In back-to-back no-grad forward calls, the 70.8 MB `qkv` of the canvas shape stays
resident in the RTX 5090's L2 (~96 MB), so v0.2's exposed loads hit L2 and v0.3's asynchronous
loads have less latency to hide; inside training calls, the backward's traffic evicts it. The
RTX 5060 Ti's L2 is 32 MB: the same mechanism should appear at **batch 16** (`qkv` 17.7 MB, fits)
and not at batch 64 (70.8 MB, does not — the control).

**Protocol (local RTX 5060 Ti).** Seeded inputs at b 16 and b 64 (h 6, s 240). (a) Nsight Compute
on v0.2's forward (`target/v02/compare.exe`, `compare` example's `nsys` mode), one launch after 10
warm-up calls, in the `fwd` phase (preceded by forwards on the same `qkv`) and in the `train` phase
(preceded by a backward), with `--cache-control none --replay-mode application` so that each pass
sees the real cache state: L2 hit rate (`lts__t_sector_hit_rate.pct`), DRAM bytes read, the
long-scoreboard stall. (b) `bench/ab.py` v0.2 against v0.3 (FA9's kernels) at b 16.

**Predictions.**
- **P1 (b 16):** v0.2's forward has an L2 hit rate at least 20 points higher, and reads at most
  half the DRAM bytes, in the `fwd` phase than in the `train` phase.
- **P2 (b 64, control):** the two phases' hit rates are within 10 points.
- **P3 (b 16):** v0.3's forward gain over v0.2 is at least 10 points smaller in no-grad calls than
  inside training calls (at b 64 the gap was 3.6 points: −40.0 % against −43.6 %).

**Reading (fixed now).** P1 and P2 together support the hypothesis; with P3 the 5090's smaller
no-grad gain is explained by the card's cache, not by a defect of v0.3. P1 failing refutes it, and
the 5090's number stays unexplained.

### FA13 — RESULT (2026-10-04, ~10:50): the timing pattern follows L2 capacity; the counters cannot see it

Raw data: `target/fa13/` (ncu CSVs) and `target/ab/fa13-b16/` (git-ignored); ~3 min of GPU.

| | predicted | measured | |
|---|---|---|---|
| P1 (b 16): v0.2's L2 hit rate, `fwd` phase ≥ `train` + 20 pts | | **60.23 % against 60.22 %**; long scoreboard 2.18 against 2.19 (DRAM bytes: metric `n/a` on this card) | **missed** |
| P2 (b 64 control): within 10 pts | | 59.96 % against 60.16 % | met |
| P3 (b 16): no-grad gain ≥ 10 pts smaller than in training calls | | **−24.4 % against −43.1 %** (v0.2's forward 280.8 µs no-grad, 386.2 µs in training calls) | met |

**Why P1 is a weak verdict.** Under Nsight Compute the timing difference itself vanishes: v0.2's
forward reads 368.5 µs (`fwd` phase) against 353.0 µs (`train`), where nsys, on the same inputs,
measures 280.8 against 386.2 µs. The profiler intercepts every launch and leaves gaps between
kernels, so it measures a GPU that no longer behaves as the real run does; its equal hit rates
(which look dominated by the kernel's own reuse — each K/V tile is read by 4 query blocks) cannot
settle the question.

**What supports the hypothesis: the gap follows L2 capacity across four conditions.**

| card, batch | `qkv` | L2 | fits | v0.3's forward gain: no-grad / in training calls |
|---|--:|--:|---|---|
| RTX 5060 Ti, b 16 | 17.7 MB | 32 MB | yes | −24.4 % / −43.1 % (gap) |
| RTX 5090, b 64 | 70.8 MB | ~96 MB | yes | −19.7 % / −38.5 % (gap) |
| RTX 4090, b 64 | 70.8 MB | 72 MB | barely | −36.4 % / −37.3 % (no gap) |
| RTX 5060 Ti, b 64 | 70.8 MB | 32 MB | no | −40.0 % / −43.6 % (no gap) |

**Reading.** By the registered rule P1 failed, but its instrument removed the effect it was meant
to see; the capacity pattern, from timings alone, supports L2 residency: when the inputs of
back-to-back no-grad calls stay in L2, v0.2's exposed loads become cheap and v0.3 has less to
hide. **Not a defect of v0.3**, whose no-grad forward is still the faster one everywhere. A
decisive test without a profiler (proposed, not run): flush L2 between no-grad calls (a write
larger than L2 between calls) and check that v0.2's forward slows to its in-training time.

---

## FA14 — the backward, split by product: half the threads on each of the two products

*Registered 2026-10-05, before any code. Why now: in the canvas training step the backward is the
larger attention cost (local 5060 Ti, b64, the Me10 binary: `fattn_bwd_f32_d64` 14.7 of 116.9
kernel ms per step, the forward 5.4), and its top stall since FA9 is short scoreboard (1.28 warp
cycles per issued instruction): waiting on shared-memory loads. The crate is improved on its own —
nothing in candle patched — so that dropping it in is enough.*

**The idea.** FA10 showed that more outputs per thread need larger tiles (no room in 99 KB) or fewer
threads (FA11: 4 warps per SM, 15.5 % slower). There is a third way in the backward, because its
phases 1 and 2 each compute TWO products over the same outputs: S = QKᵀ and dP = dO Vᵀ; dV = Pᵀ dO
and dK = dSᵀ Q. Today each thread computes both products for few outputs, so each step loads four
operands. Giving each product to one half of the block doubles every thread's outputs for the
SAME tiles, shared memory, 256 threads and 8 warps per SM.

**Change** (`fattn_bwd_f32_d64` only; `half = tid / 128`, warp-uniform):
- **Phase 1.** Thread `u = tid % 128` owns query rows 4·(u / 16) … + 4 and keys (u % 16) + 16 jj
  (jj 0–3): half 0 computes S from Q, K; half 1 computes dP from dO, V — 4 × 4 accumulators each,
  per head-dimension step 4 + 4 `LDS.128` for 64 FFMA (today 12 for 64). Half 0 writes
  P = exp(S·scale − L) (0 where not live, as today) to `Ps`; barrier; half 1 reads its 16 P back
  and writes dS = P·(dP − D) to `dSs`; barrier. One barrier and 16 scalar loads more per tile.
- **Phase 2.** Half 0 owns dV, half 1 owns dK: thread u owns keys 4·(u / 8) … + 4 and dims
  4·(u % 8) … + 4 and 32 + 4·(u % 8) … + 4 (two float4 32 floats apart, so each 8-thread phase of a
  128-bit load reads 32 consecutive words: no bank conflict). Per query: 1 + 2 `LDS.128` for 32
  FFMA (today 4 for 32). 32 accumulators per thread, as today (16 dK + 16 dV).
- **Phase 3** (the dQ partial), the dQ turn order, the loads (FA9's double buffers) and the
  forward: unchanged.

**Why the bits cannot move.** Every output keeps its accumulation and its order: S and dP are each
one `dot4` chain over the head dimension in order; P, dS are the same expressions on the same
floats (P is stored and re-read, exact); dV and dK are each one `fmaf` chain over query tiles, then
queries, in order; only *which thread* computes an element changes. **The bitwise gate applies.**

**Counted, per thread and query tile** (`LDS` instructions, FA10's method; phase 1 averaged over
the halves: 128 and 128 + 16):

| phase | today | FA14 |
|---|--:|--:|
| 1: S, dP | 192 | 136 |
| 2: dV, dK | 128 | 96 |
| 3: dQ partial | 96 | 96 |
| total | 416 | **328 (−21 %)** |

**Predictions.**
- **P1 (gate):** O, dQ, dK, dV bitwise identical to v0.3.0 on `bench/bitwise.py`'s 6 shapes, both
  entry points, native and forced-fallback (`CANDLE_FUSED_ATTN_SYNC_LOADS=1`) builds.
- **P2:** backward shared-load instructions −19 to −23 % (Nsight Compute, FA8's protocol; today's
  count from FA9's profile); registers 128–180 (≤ 255 keeps the one block per SM that shared memory
  already sets); shared memory unchanged (87,552 B).
- **P3:** short-scoreboard stall below FA9's 1.28; barrier stall up from 0.38 (one barrier more).
- **P4:** backward kernel time **−6 to −14 %** against v0.3.0, alternated (`bench/ab.py`, `train`
  phase, 6 rounds); the forward unchanged within ±2 %.

**Decision rule (fixed now).** P1 is a gate. ≥ 5 % on the backward kernel: kept. Below 3 %:
reverted with a `MEASURED-REVERT` note. 3–5 %: `ab.py` rerun with 10 rounds; kept if the gain
holds ≥ 3 % with no round overlap. If kept, FA15 = skipping the dead tail work (S = 240: the last
key block has 48 live keys of 64, the last query tile 16 live rows of 32, ~12 % of the FFMAs
compute zeros) with warp-uniform branches, under the same gate. Card-side confirmation (4090, 5090,
and a first A100) at the next rental, never a rental of its own.

**Where / cost.** Local RTX 5060 Ti: bitwise dumps ~15 s each (4), `ab.py` ~1 min, `ncu` a few
seconds; builds ~2 min CPU each.

### FA14 — RESULT (2026-10-05, ~10:15): backward −6.9 %, bitwise identical; every prediction met, the time at the low end

Code: committed with this result. Report:
[`2026-10-05-rtx5060ti-ab-v0.3.0-vs-fa14.md`](../bench/results/2026-10-05-rtx5060ti-ab-v0.3.0-vs-fa14.md);
profiles `profiles/2026-10-05-rtx5060ti-bwd-candle-fused-attn-{v030,fa14}.ncu-repz` (same session).
Reference binaries `target/v030{,-sync}/compare.exe` (the crate at `d7ded1d`, kernels as released).
GPU used: ~3 min.

| | predicted | measured | |
|---|---|---|---|
| P1: bitwise identity | O, dQ, dK, dV; 6 shapes; native and fallback | **identical** (6 shapes × 2 entry points; native, forced fallback) | met |
| P2: shared loads; registers; shared memory | −19 to −23 %; 128–180; 87,552 B | **41.29 M → 32.64 M (−20.9 %)**; 128; 87,552 B | met |
| P3: short scoreboard down; barrier up | below 1.28; above 0.38 | **1.277 → 1.186**; **0.385 → 0.672** | met |
| P4: backward kernel, alternated vs v0.3.0, 6 rounds | −6 to −14 %; forward ±2 % | **−6.9 %** (2.369 → 2.205 ms), separated (2,253 µs worst vs 2,356 best); forward −1.2 % alone, +2.5 % inside the training call with overlapping rounds | met (low end) |

Also: shared-memory bank conflicts on loads 1.72 M → 0.14 M and on stores 1.58 M → 0.01 M (the
new phase-1 and phase-2 mappings happen to be conflict-free where today's P / dS stores and the
phase-2 loads were not); MIO throttle 0.56 → 0.23; instructions −3.6 %. **A trap, recorded:** under
Nsight Compute the two kernels ran at different clocks (2.61 GHz for v0.3.0, 2.28 GHz for FA14), so
ncu's durations say FA14 is slower (2.46 → 2.68 ms) while its cycles say faster (elapsed 6.40 M →
6.11 M, −4.5 %; active −6.6 %). FA3's rule holds: ncu says why, nsys says how fast — and compare
cycles, never ncu durations, across two profiles. The forward kernel's PTX is byte-identical in both
builds. Crate tests (`cargo test --release --features cuda`) pass.

**Reading.** The loads went exactly as counted (−20.9 % for −21 % predicted), but the time gained
only a third of that: the extra barrier doubled the barrier stall (the two halves wait for each
other twice per tile), and short scoreboard fell less than the loads did. The low end, not the
middle. **Decision (the registered rule): kept** (≥ 5 %). Next: FA15, skipping the dead tail work
with warp-uniform branches, registered on the same gate.

---

## FA15 — the backward skips its dead tail work

*Registered 2026-10-05, before any code. Why: at the canvas length S = 240 the backward's tiles
overhang the sequence twice — the last key block holds 48 live keys of 64, the last query tile 16
live rows of 32 — and the kernel computes the overhang in full, on zero-filled rows. Counted below:
**12.1 % of its multiply-adds produce nothing.** Any length that is not a multiple of 64 pays
the same way; the crate pays it, not canvas alone.*

**Change** (`fattn_bwd_f32_d64` only; every bound depends on S, the key block and the query tile
alone, so the branches are uniform across the block or across whole warps at S = 240):
- **Phase 1.** A key group jj (keys k0 + 16 jj … + 16) entirely ≥ S is skipped; a thread whose
  four query rows are all ≥ S skips its dot products. The P / dS writes stay as they are (their
  `live` mask already writes 0).
- **Phase 2.** The query loop runs over the tile's live rows only (`min(32, S − q0)`); a thread whose
  four keys are all ≥ S skips the phase (its dK / dV are never stored).
- **Phase 3.** The key loop runs to `min(64, S − k0)` rounded up to 4; a thread whose two query
  rows are both ≥ S skips its dot products. The dQ turn wait, the barriers and the stores are
  untouched.
- Forward, loads (dead rows stay zero-filled), tiles, shared memory: unchanged.

**Why the bits cannot move.** A skipped term is a product with a zero factor: a zero-filled K, V, Q
or dO row, or a P or dS that the `live` mask set to zero. Adding such a ±0 to an accumulator leaves it
unchanged unless the accumulator is −0, and none ever is: every chain starts at +0, and an `fmaf`
returns −0 only when its addend is already −0. A skipped dot product leaves its accumulator at +0,
which is exactly what the zero rows produced (`+0 + −0 = +0` under round-to-nearest). **The
bitwise gate applies**, causal and the odd lengths (7, 97) included.

**Counted at the canvas shape** (b 64, h 6, S 240, non-causal; per (b, h): 4 key blocks × 8 query
tiles = 32 block-tiles; each phase has the same overhang structure):

| | key block 3, tiles 0–6 | key blocks 0–2, tile 7 | key block 3, tile 7 | saved |
|---|--:|--:|--:|--:|
| per phase, in block-tiles of work | 7 × 0.25 | 3 × 0.5 | 0.625 | **3.875 / 32 = 12.1 %** |

FA14's profile measures 8.288 G fp32 instructions on the FMA pipes per launch (`fmaheavy` +
`fmalite`), against 8.053 G multiply-adds counted by hand: the metric follows the work.

**Predictions.**
- **P1 (gate):** O, dQ, dK, dV bitwise identical to FA14's (`55aea18`) on `bench/bitwise.py`'s 6
  shapes, both entry points, native and forced-fallback builds.
- **P2:** fp32 instructions on the FMA pipes **−10 to −12.5 %** (8.288 G → 7.25–7.46 G); shared
  loads down by a similar share; registers and shared memory unchanged.
- **P3:** backward kernel time **−3 to −8 %** against FA14, alternated (`bench/ab.py`, `train`, 6
  rounds) — sub-linear in the work saved: a warp that skips gives back issue slots, not its whole
  share of the tile, since the block still waits for its slowest warp at each barrier.
- **P4:** the forward unchanged (its PTX byte-identical).

**Decision rule (fixed now).** P1 is a gate. Faster by ≥ 2 % with separated rounds: kept (a small
change, under the bitwise gate). Otherwise reverted with a `MEASURED-REVERT` note. The forward's
own overhang (its last query block holds 48 live rows of 64, its last key tile 48 live keys; the
online softmax makes its bitwise argument more delicate) is FA16's question, not this one's.
Confirmation on the 4090, 5090 and a first A100 at the next rental round (the RTX 3090 joins at the
next release).

**Where / cost.** Local RTX 5060 Ti: 4 bitwise dumps (~15 s each), `ab.py` 6 rounds (~2 min), `ncu`
(seconds); builds ~10 s each.

### FA15 — RESULT (2026-10-05, ~10:45): bitwise identical, 11.8 % less math, and 14.2 % slower — rejected

Code as measured: `95cdd77`; reverted by `c430372` (the backward's PTX byte-identical to FA14's
again, checked). Report:
[`2026-10-05-rtx5060ti-ab-fa14-vs-fa15-rejected.md`](../bench/results/2026-10-05-rtx5060ti-ab-fa14-vs-fa15-rejected.md);
profile `profiles/2026-10-05-rtx5060ti-bwd-candle-fused-attn-fa15-rejected.ncu-repz`. GPU used:
~3 min.

| | predicted | measured | |
|---|---|---|---|
| P1: bitwise identity | vs FA14; 6 shapes; native and fallback | **identical** (and to v0.3.0) | met |
| P2: FMA-pipe fp32 instructions | −10 to −12.5 % | **8.288 G → 7.313 G (−11.8 %)**; shared loads −10.5 %; registers 128 | met |
| P3: backward kernel vs FA14, 6 rounds | −3 to −8 % | **+14.2 %** (2.194 → 2.505 ms), separated: FA15's fastest round above FA14's slowest | **missed** (slower) |
| P4: forward PTX | byte-identical | byte-identical | met |

| backward, under Nsight Compute | FA14 | FA15 |
|---|--:|--:|
| cycles elapsed (compare cycles, not durations: FA14's note) | 6.11 M | **6.84 M (+11.9 %)** |
| instructions | 345.7 M | 340.3 M (−1.6 %) |
| issue slots busy | 40.8 % | **34.9 %** |
| short scoreboard; barrier; branch resolving | 1.19; 0.67; 0.07 | **1.46; 1.02; 0.24** |

**Reading.** The work went exactly as counted, but the instruction count barely moved: the loop
control of runtime-bounded loops (`i < nq`, `j < nk`) and the per-group branch replaced nearly all
the saved arithmetic, and on EVERY tile — the 7 full tiles of 8 now run loops the compiler can no
longer unroll to their fixed trip counts and schedule loads early in, which is what short
scoreboard and the idle issue slots show. The skipping warps then wait at the next barrier for the
others (barrier stall up). The dead work was 12 % of the multiply-adds of the TAIL tiles only; the
cost landed on all of them. **Decision (the registered rule): reverted**, recorded in the kernel's
header as FA11 was.

**What would be needed instead (not registered).** Full tiles must keep today's code exactly: a
separate tail path taken only when the tile overhangs S (`nq < 32` or `k0 + 64 > S`), its loops at
fixed trip counts that are themselves compile-time (a template on the overhang class), so the
compiler can unroll both. At S = 240 that path would run on 11 of 32 block-tiles; the gain is
bounded by the 12.1 % of the multiply-adds it skips, minus the code-size and register cost of a
second copy of each phase — a few percent at best, against a doubled kernel body.

---

## FA16 — the dV half stops waiting for dS

*Registered 2026-10-05, before any code. Why: FA14 bought −20.9 % shared loads with one more
block-wide barrier per query tile, and its barrier stall rose 0.38 → 0.67 warp-cycles per issued
instruction; the backward gained only −6.9 %. Part of that barrier is waiting nobody needs.*

**Today (FA14), per query tile:** barrier (tile landed) → phase 1 (both halves) → half 0 writes P →
**barrier** → half 1 reads P, writes dS → **barrier** → phase 2 (half 0: dV from P, dO; half 1: dK
from dS, Q) → phase 3 (all: dQ partial from dS, K) → the dQ turn (its own barriers).

**The waste.** At the barrier after dS, half 0 (the dV owners) waits for half 1 to write dS, but dV
never reads dS: it needs only P (written before the previous barrier) and dO.

**Change** (`fattn_bwd_f32_d64` only). After the barrier that publishes P:
- half 0 goes straight to its phase 2 (dV);
- half 1 writes dS, syncs **among its own 4 warps** (`bar.sync 1, 128`: a named barrier, id 1 —
  `__syncthreads` is id 0), then accumulates dK;
- one block-wide barrier before phase 3 (which reads all of dS), where today's barrier after dS was.

Same number of block-wide barriers; half 0's dV now overlaps half 1's dS. Hazards checked: phase 2
writes only registers; the next tile's P and dS are written after the next tile's first barrier, so
nobody can still be reading this tile's. `bar.sync` with a thread count orders shared memory among
its participants, and exists on every target the crate builds (the fallback included).

**Why the bits cannot move.** No arithmetic changes, nor which thread computes what; only when a
warp waits. **The bitwise gate applies.**

**Predictions.**
- **P1 (gate):** O, dQ, dK, dV bitwise identical to FA14 (`55aea18`, the code since `c430372`) on
  `bench/bitwise.py`'s 6 shapes, both entry points, native and forced-fallback builds.
- **P2:** barrier stall below FA14's 0.67, between 0.45 and 0.62; instructions within ±1 %;
  registers 128 ± 16; shared memory unchanged (Nsight Compute; cycles compared, not durations).
- **P3:** backward kernel time **−1 to −4 %** against FA14, alternated (`bench/ab.py`, `train`, 6
  rounds). Bounded by the barrier cost FA14 added (+0.29 of ~4 warp-cycles per issue, ~7 %), of
  which only the dS wait is removed.
- **P4:** the forward's PTX byte-identical.

**Decision rule (fixed now).** P1 is a gate. Faster by ≥ 2 % with separated rounds: kept. 1–2 %:
`ab.py` rerun with 10 rounds, kept if still separated. Otherwise reverted (`MEASURED-REVERT`), named
in the kernel header. After FA16, whatever it gives: the rental round (A100, 4090, 5090: FA14 and
FA16 confirmed there, `gemm_autotune` alongside), before any larger redesign is chosen.

**Where / cost.** Local RTX 5060 Ti: 4 bitwise dumps (~15 s each), `ab.py` 6 rounds (~2 min), `ncu`
(seconds); builds ~10 s each.

### FA16 — RESULT (2026-10-05, ~11:10): bitwise identical, no faster — rejected

Code as measured: `b0aa346`; reverted by `9a5a13d` (the backward's PTX byte-identical to FA14's
again, checked). Report:
[`2026-10-05-rtx5060ti-ab-fa14-vs-fa16-rejected.md`](../bench/results/2026-10-05-rtx5060ti-ab-fa14-vs-fa16-rejected.md);
profile `profiles/2026-10-05-rtx5060ti-bwd-candle-fused-attn-fa16-rejected.ncu-repz`. GPU used:
~3 min.

| | predicted | measured | |
|---|---|---|---|
| P1: bitwise identity | vs FA14; 6 shapes; native and fallback | **identical** | met |
| P2: barrier stall; instructions; registers | 0.45–0.62; ±1 %; 128 ± 16 | **0.672 → 0.725 (up)**; +0.2 %; 128 | **missed** |
| P3: backward kernel vs FA14, 6 rounds | −1 to −4 % | **+1.4 %** (2.198 → 2.228 ms), rounds overlapping | **missed** |
| P4: forward PTX | byte-identical | byte-identical | met |

Under Nsight Compute: active cycles 5.902 M → 5.900 M (unchanged); short scoreboard 1.19 → 1.08,
MIO throttle 0.23 → 0.28, issue slots 40.8 → 40.6 %.

**Reading — the registration's model was wrong.** It treated the wait at the barrier after dS as
removable idle time. It is not: each tile's critical path runs through half 1 (dS, then dK over 32
queries) and then phase 3, which needs all of dS. Letting half 0 finish dV earlier shortens nobody's
path; half 0 waits at the barrier before phase 3 instead (barrier stall slightly UP), and the tile
takes as long as before. To gain, the two halves' work would have to be rebalanced (half 1 does dS
on top of the same phase-2 load as half 0), not reordered. **Decision (the registered rule):
reverted**, named in the kernel header.

**Where the backward stands after FA14-FA16.** FA14 (kept): −6.9 %. FA15 and FA16 (rejected): the
remaining small, bitwise levers inside today's structure did not pay. The next step is the
registered one: the rental round (A100, 4090, 5090) confirming FA14 and measuring the crate on
datacenter hardware, before any larger redesign is chosen.

---

## FA17 — FA14 on three rented cards: RTX 5090, RTX 4090, and a first A100

*Registered 2026-10-05, before any rental. Why: FA14 is measured on the 5060 Ti only, and FA12
showed that a gain measured there shrinks on bigger cards (FA9's backward −14.3 % locally became
−8.9 % on the 5090 and −11.9 % on the 4090). And every card in `RESULTS.md` is GeForce: the A100
(sm_80, 108 SMs, 164 KB of shared memory per SM, fp32 19.5 TFLOPS but TF32 tensor cores at 8× that)
is what most people who would drop this crate in actually train on. The RTX 3090 joins at the next
release, not this round.*

**Protocol.** `bench/box.sh <label>` on each card, now against **v0.3.0** (`BASE`, default) with
`ab.py` at **6 rounds** (`ROUNDS`): machine record, both builds on the box, the bitwise gate, the CUDA
tests, `compare.py` (against PyTorch's SDPA on that card), `ab.py` alternated. The source travels as a
git bundle (`bench/remote_box.sh … send`: every branch and tag, nothing published), so the box
measures the local HEAD exactly. Results home with `remote_box.sh … pull`. Validated locally first
(5060 Ti, 6 min 23 s, at `adc8f60` + the uncommitted script): bitwise identical, 3 test binaries ok,
backward −7.5 % (FA14's report: −6.9 %). No Nsight Compute on vast (FA12: `ERR_NVGPUCTRPERM`): times
only.

**Predictions.**
- **P1 (gate):** v0.3.0 and HEAD bitwise identical on each card (6 shapes, both entry points).
- **P2:** backward kernel inside training calls, HEAD vs v0.3.0: **−3 to −7 %** on the 5090 and on
  the 4090 (about 60–80 % of the local −6.9 %, FA9's pattern); **0 to −6 %** on the A100, whose
  SM has half the fp32 lanes of a consumer Ampere/Ada/Blackwell SM (64 against 128), so it is more
  arithmetic-bound and saves less from fewer shared loads.
- **P3:** the forward kernel unchanged within ±3 % on each card (its PTX is byte-identical).
- **P4 (A100 against PyTorch):** SDPA's fp32 path (`fmha_cutlass*_f32`, 3xTF32 on tensor cores)
  is **faster than ours on the A100**: forward + backward kernel time ours ≥ 1.5× SDPA's. On the
  5090 and the 4090 ours stays faster than SDPA's, as in FA12.
- **P5 (A100 occupancy, read from the build, not profiled):** the forward (69,632 B) fits two blocks
  per SM in 164 KB; the backward (87,552 B) still one.

**Decision rule (fixed now).** P1 fails anywhere: stop, no release, investigate. P2 holds on at least
two of three cards: FA14 goes into 0.4.0 with per-card numbers. P4 holds: the README says it plainly
(on an A100-class card, PyTorch's SDPA is the faster fp32 attention today), and an opt-in 3xTF32
path for sm_80/sm_90 becomes a roadmap candidate. P4 fails: we say that instead.

**Where / cost.** Three vast.ai boxes, `PyTorch (Vast)` template, SSH. Per box: ~5 min to reach
Running, `send` ~3 min (rustup + a 21 MB bundle), builds ~3 min, GPU ~6 min, `pull` ~1 min: about
20 min billed. 5090 (~$0.48/h) ≈ $0.16, 4090 (~$0.37/h) ≈ $0.13, A100 ($0.5–1.5/h, market) ≈
$0.20–0.50; **round total ≈ $0.5–0.8**.

### FA17 — RESULT (2026-10-05, 09:10–09:57 UTC): FA14 holds on both GeForce cards; on the A100, SDPA is 1.73× faster than us

Three boxes, one at a time (`rentals.md` in askesis acsp14), `bench/remote_box.sh` at `468c712`
(5090) and `a39eb66` (4090, A100; the two commits between are script fixes only). Results home,
slim (bitwise dumps and safetensors left on the boxes): `target/box-home/{rtx5090,rtx4090,a100}/`.
Cost: 5090 $0.20 (incl. a first launch lost to absent `safetensors`), 4090 $0.06, A100 $0.11:
**$0.37 for the round** (credit $9.76 → $9.39).

| | RTX 5090 | RTX 4090 | A100-SXM4-40GB |
|---|--:|--:|--:|
| machine | driver 580.105, CUDA 13.0, 500 W | driver 595.71, CUDA 13.2, **350 W cap** | driver 595.91, CUDA 13.2, 400 W, MIG off |
| P1: v0.3.0 vs FA14 bitwise; CUDA tests | identical; pass | identical; pass | identical; pass |
| P2: backward kernel in training calls | 511.3 → 482.0 µs, **−5.7 %** | 691.0 → 652.3 µs, **−5.6 %** | 1939.1 → 1929.5 µs, **−0.5 %** |
| rounds (6 each) | no overlap | no overlap | no overlap |
| P3: forward kernel | +0.0 % | +0.0 % | +0.0 % |
| ours vs SDPA, kernel time per training call (net of the head) | **0.813 vs 1.022 ms** | **1.183 vs 1.555 ms** | 2.856 vs **1.650 ms** |
| ours vs SDPA, forward kernel | 0.178 vs 0.206 | 0.255 vs 0.333 | 0.625 vs **0.438** |

| | predicted | measured | |
|---|---|---|---|
| P1 | bitwise identical on each card | identical on all three | met |
| P2 | −3 to −7 % on 5090 and 4090; 0 to −6 % on A100 | −5.7, −5.6; −0.5 | met (A100 at the low edge) |
| P3 | forward ±3 % | +0.0 % everywhere | met |
| P4 | on the A100 SDPA's forward + backward ≥ 1.5× faster; ours faster on the GeForce cards | 1.73× (SDPA backward 0.98 ms vs ours 1.93); ours 0.80× and 0.76× SDPA on 5090 and 4090 | met |
| P5 | forward 2 blocks per SM on the A100 | not measurable: no profiler counters on vast | — |

**Reading.** FA14 cuts shared loads; it pays where shared memory is the limit, on the consumer cards
(128 fp32 lanes per SM), and not on the A100 (64 lanes per SM, arithmetic-bound), as P2's split
predicted. On the A100, SDPA's fp32 kernels run 3xTF32 on tensor cores whose TF32 rate is 8× the
fp32 rate: no FFMA kernel can match that there. **Decision (the registered rule):** FA14 goes into
0.4.0 with these per-card numbers; the README says plainly that on A100-class cards PyTorch's SDPA
is the faster fp32 attention today; an opt-in 3xTF32 path for sm_80/sm_90 becomes a ROADMAP
candidate (it would change summation and so leave the bitwise-to-0.3 guarantee; its own gate
would be accuracy against fp64, as SDPA's is). The RTX 3090 (sm_86) joins at the next release.
