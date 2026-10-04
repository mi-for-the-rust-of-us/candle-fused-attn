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

**v0.3.0** shipped 2026-10-04. *Loads overlap the math, outputs unchanged bit for bit.* Both
kernels copy their next tiles with asynchronous loads (`cp.async`) while the current tile
computes — FlashAttention-2's load schedule in the forward, double-buffered query tiles in the
backward. Measured on three cards, 0.2.0 against 0.3.0 alternated in one session: forward kernel
−37 to −44 % inside training calls, backward −9 to −16 %, an attention training call −13 to −16 %;
against PyTorch's fp32 SDPA in the same session, forward and forward + backward are faster on the
RTX 5060 Ti, 4090 and 5090 (kernel time). In the canvas training step on an RTX 5090: +1.8 %
throughput, ≈ 0.90× PyTorch. Plan and measurements:
[`docs/roadmap-v0.3.0.md`](docs/roadmap-v0.3.0.md), [`docs/devlog.md`](docs/devlog.md) FA1–FA13;
every number: [`RESULTS.md`](RESULTS.md).

**v0.2.0** shipped 2026-10-03. *The backward in one kernel, still deterministic*: each key block
computes dK, dV and a dQ partial in one pass, the dQ partials added in key-block order.

---

## Next: candidates (not yet chosen)

- **Larger tiles at 8 warps per SM.** Both kernels now wait mostly on shared-memory loads; the
  128-thread remedy was measured slower (devlog FA11: fewer warps cost more than the loads saved).
  The remaining route keeps 256 threads with larger tiles, which needs swizzled, unpadded shared
  memory to fit 99 KB per block (FA10).
- **The L2 question, settled.** FA13 found the 5090's smaller no-grad forward gain follows L2
  capacity; a profiler-free test (flush L2 between calls) would confirm it.
- **The training step's other costs.** The canvas step gains +1.8 % for −13 % on the attention
  call: the step is partly host-bound — candle and candle-mi territory, not this crate's.

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
