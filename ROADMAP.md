# candle-fused-attn — Roadmap

> *fp32 attention for candle that trains, measured. Status snapshot — updated as plans change.*

Shipped history lives in [`CHANGELOG.md`](CHANGELOG.md). How a release is cut lives in
[`CLAUDE.md`](CLAUDE.md) § *Releasing*; how a claim is backed, in
[`CONVENTIONS.md`](CONVENTIONS.md) § *Grit-FA: Claims and Their Evidence*.

**Who this crate serves.** Its main user is the *planner-inside-a-model* project: masked-diffusion
planners trained from scratch with candle, in fp32, where runs must be reproducible and
comparable across seeds, cards and days. Everything below is ranked by what that work needs.

---

## Current state

**v0.2.0** shipped 2026-10-03. *The backward in one kernel, still deterministic.* Each key block
computes dK, dV and a dQ partial in one pass (5 matmuls per tile instead of 7), and the key blocks
add their dQ partials in key-block order under per-tile turn counters, so reruns are
bit-identical. The backward kernels now beat PyTorch's fp32 SDPA (memory-efficient, CUTLASS)
on both cards measured; the forward does not. Kernel time per call at the canvas trainer's shape,
b 64 · h 6 · s 240 · d 64, non-causal (`bench/compare.py`, nsys):

| ms per call | forward | backward kernels | forward + backward, net |
|---|---|---|---|
| RTX 5090: this crate | 0.226 | **0.597** | **0.885** |
| RTX 5090: SDPA | **0.207** | 0.767 | 1.022 |
| RTX 5060 Ti: this crate | 1.419 | **2.93** | 4.973 |
| RTX 5060 Ti: SDPA | **0.945** | 3.66 | **4.814** |

Reports: [5090](bench/results/2026-10-03-rtx5090-compare.md),
[5060 Ti](bench/results/2026-10-03-rtx5060ti-compare.md). In the trainer (6 layers, 384 wide,
RTX 5090, batch 128, alternated rounds): composed attention 87 ms/step, v0.1 79, **v0.2 75**
([report](bench/results/2026-10-03-rtx5090-trainer-ab.md)). Accuracy against fp64 is at the
composed attention's level (dQ 4.0e-7 normwise) and better than SDPA's.

The first downstream integration is candle-mi's `fused-attn` feature (in development, not yet in
a candle-mi release): `OthelloGpt`'s attention through `fused_attention_qkv` at every layer with
no attention-internal hook. Checked there against the PyTorch fixtures of an Othello world model:
logits max-abs-diff 3.1e-5 on CUDA, where the composed attention gives 3.4e-5.

---

## Next: v0.3.0 — a faster forward

**Why.** The forward is the one place this crate is behind: ×1.5 SDPA's time on the 5060 Ti,
×1.09 on the 5090. It also runs more often than it looks. In the canvas trainer it runs about
three times per layer and step — the carry's no-grad pass, the training pass, the validation
share — measured as ~18 forward calls per step at batch 64 (nsys, 2026-10-02), against 6
backward calls. And evaluation, which decodes for 101 rounds per problem, is forward only.

**What it could buy (estimates, not measurements).** At SDPA's forward speed, 18 calls × 0.47 ms
≈ **−8.5 ms per training step** at batch 64 on the 5060 Ti (~7 % of a ~122 ms step), but only
≈ −0.3 ms on the 5090 at the same shape, where training is rented. The larger payoff may be the
local evaluations, run on the 5060 Ti: a registered readout of one run took ~25 min of that
card on 2026-10-03. How much of it is attention is not known yet; step 0 measures it.

**Step 0 — measure first.** One nsys capture of a 200-problem canvas evaluation on the RTX 5060
Ti (fused-attention build, the forward kernel's share of GPU time), with `--force-overwrite=true`
and the binary's first run discarded. If attention is a small share of evaluation, v0.3.0 is
worth its training gain on mid-range cards only, and its priority is decided on that.

**Read first (C++ before code).**
- PyTorch v2.10's memory-efficient forward (`kernel_forward.h`, `mem_eff_attention`): its
  tiling, warps per block and blocks per SM for fp32, head_dim 64.
- cuBLAS's fp32 SIMT kernels: the 8×8-per-thread register tile.
- FlashAttention-2's `flash_fwd_kernel.h` (vendored in candle-flash-attn) for the online-softmax
  bookkeeping it already follows.

**The levers, as measured.** Today's forward block holds Q, K, V and P tiles of 64 rows × 68
floats in **69,632 bytes of dynamic shared memory**, which fits **one block per SM** on the 5060
Ti; 8 warps; each thread a 4×4 register tile. SDPA runs 4 warps × 3 blocks per SM, with larger
per-thread tiles. Candidates: smaller tiles, or K/V tiles shared across query rows, to fit 2–3
blocks per SM; an 8×8 (or 4×8) per-thread tile; fewer warps per block. Each is a change to
`TWIN`-annotated constants on both sides (`src/cuda.rs`, `kernels/fused_attn.cu`).

**Bars that must hold.**
- **Determinism:** `tests/parity.rs::cuda_backward_is_deterministic` passes, and
  `bench/compare.py`'s bitwise rerun check (outputs and gradients) still reads `True`; no atomics
  in any reduction (`CONVENTIONS.md` § *DETERMINISM Annotation*).
- **Accuracy:** against fp64, no worse than v0.2's table, on both cards.
- **Parity:** CUDA vs the CPU reference within the measured band (`cuda_matches_cpu`).
- **Downstream:** candle-mi's `othello-fused` oracle still passes.

**Done when** the forward matches SDPA's at the canvas shape on both cards, measured with
`bench/compare.py` (alternated rounds, same session, report committed under `bench/results/`),
the trainer A/B is re-run on a rented 5090, and the release goes through the `CLAUDE.md`
checklist, rehearsal included.

---

## Later (speculative)

- **head_dim 32 and 128.** The CUDA path takes 64 only; the CPU path takes any. 128 is the
  Llama / Qwen / Mistral head size and doubles the shared memory per tile, so it needs its own
  tiling, and is best designed after v0.3.0's. The tile constants are already named and derived,
  so the kernels can be templated on them.
- **Upstream to candle.** The plan since the start (crate first, upstream second): candle has no
  fused attention that trains, and none in fp32 on CUDA. kaio-candle 0.2.0 is the prior art to cite.

---

## Deliberately out of scope

- **bf16 / f16.** fp32 is the point of this crate: reproducible, comparable training runs.
  `candle-flash-attn` already covers half precision for inference.
- **Dropout on the attention weights.** The planner models do not use it.
- **Arbitrary additive masks.** Only `causal` is supported. The planners train on fixed-length
  blocks; a key-padding mask becomes a candidate if variable-length batches ever appear.

---

## Principles

- **fp32 and bitwise determinism**, both proven by tests, never assumed.
- **Every speed or accuracy claim has a committed report** naming its card, shape, instrument and
  protocol; comparisons are alternated within one session, never across hours or boxes.
- **C++ first**: the reference a kernel follows is read before the kernel is written, and cited
  in a `// REFERENCE:` comment.
- **Measure before optimising**: a lever is priced by a pilot measurement before it is built.
