# Results

Every number this crate publishes, newest first, each with the committed report it comes from.
The [README](README.md#measured) shows the latest; this page keeps the history. How each
measurement was decided, predicted and read is in [`docs/devlog.md`](docs/devlog.md); the rules
behind the numbers (same box, same session, alternated rounds) are in
[`CONVENTIONS.md`](CONVENTIONS.md#performance-claims).

## The attention op

Kernel time per call (nsys), in **milliseconds**, at the canvas trainer's shape: b 64 · h 6 ·
s 240 · d 64, non-causal. *Forward* is a no-grad call; *fwd + bwd* is a training call net of its
loss head. SDPA is PyTorch's fp32 `scaled_dot_product_attention` (memory-efficient backend,
3xTF32 on tensor cores), measured in the same session.

### RTX 5060 Ti

| date | crate | forward | fwd + bwd | SDPA forward | SDPA fwd + bwd | report |
|---|---|--:|--:|--:|--:|---|
| 2026-10-04 | 0.3.0-dev, async forward loads (`aef44b9`) | **0.859** | **4.518** | 0.964 | 5.021 | [compare](bench/results/2026-10-04-rtx5060ti-compare-l1.md) |
| 2026-10-04 | 0.2.0 (baseline) | 1.387 | 4.874 | 0.949 | 4.708 | [compare](bench/results/2026-10-04-rtx5060ti-compare-v0.2-baseline.md) |
| 2026-10-03 | 0.2 kernels (before release) | 1.419 | 4.973 | 0.945 | 4.814 | [compare](bench/results/2026-10-03-rtx5060ti-compare.md) |

Version against version, alternated in one session ([`bench/ab.py`](bench/ab.py)):

| date | change | forward kernel | backward kernel | training call | report |
|---|---|--:|--:|--:|---|
| 2026-10-04 | 128 threads × 8 × 4 outputs (**rejected**) | +15.5 % | −1.8 % | +2.7 % | devlog FA11 |
| 2026-10-04 | async, double-buffered backward loads | −0.1 % | **−14.3 %** | **−7.2 %** | [A/B](bench/results/2026-10-04-rtx5060ti-ab-l1-vs-fa9.md) |
| 2026-10-04 | async forward loads | **−42.6 %** | +0.3 % | **−9.7 %** | [A/B](bench/results/2026-10-04-rtx5060ti-ab-v0.2-vs-l1.md) |

### RTX 5090

| date | crate | forward | fwd + bwd | SDPA forward | SDPA fwd + bwd | report |
|---|---|--:|--:|--:|--:|---|
| 2026-10-03 | 0.2 kernels (before release) | 0.226 | **0.885** | **0.207** | 1.022 | [compare](bench/results/2026-10-03-rtx5090-compare.md) |

### Accuracy

Against an fp64 reference (normwise relative error), identical on both cards and unchanged by
every 0.3.0 change (their outputs are bit-identical to 0.2.0's):

| | O | dQ | dK | dV |
|---|--:|--:|--:|--:|
| candle-fused-attn | 3.9e-7 | 4.0e-7 | 4.6e-7 | 4.4e-7 |
| PyTorch SDPA (memory-efficient) | 5.7e-7 | 8.0e-7 | 8.0e-7 | 6.2e-7 |
| PyTorch, composed | 3.9e-7 | 4.1e-7 | 4.1e-7 | 3.9e-7 |
| fp64 rounded to fp32 (the floor) | 2.5e-8 | 2.5e-8 | 2.5e-8 | 2.5e-8 |

## The training step

The canvas trainer (a 6-layer, 384-wide masked-diffusion model, through candle-mi): one step,
candle against its PyTorch reference, fp32. Ratio = candle ÷ PyTorch.
Summary with sources: [trainer history](bench/results/2026-10-04-trainer-vs-pytorch-history.md).

| date | card, batch | ratio | what changed |
|---|---|--:|---|
| 2026-10-03 | RTX 5090, 128 | **0.93×** | candle's index_add and fused GELU backward |
| 2026-10-03 | RTX 5090, 128 | 1.01× | this crate's one-kernel backward; LayerNorm statistics |
| 2026-10-02 | RTX 5090, 128 | 1.19× | this crate, v0.1 |
| 2026-10-02 | RTX 5090, 128 | 1.32× | composed attention |
| 2026-10-01 | RTX 5060 Ti, 64 | 1.376× | PyTorch re-measured (129.8 ms) |
| 2026-08-01 | RTX 5060 Ti, 64 | 2.8× | the speed campaign's first rungs |
| 2026-07-29 | RTX 5060 Ti, 64 | 5.1× (≈ 3.9× against the re-measured PyTorch) | the starting point |
