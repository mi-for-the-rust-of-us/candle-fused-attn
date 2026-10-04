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
