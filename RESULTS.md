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
| 2026-10-05 | **0.4.0** (`7cb1cd8`) | **0.781** | **3.733** | 0.903 | 4.621 | [compare](bench/results/2026-10-05-rtx5060ti-compare-v0.4.md) |
| 2026-10-04 | 0.3.0 (`32f5ec2`) | 0.895 | 3.982 | 0.978 | 4.963 | [compare](bench/results/2026-10-04-rtx5060ti-compare-v0.3.md) |
| 2026-10-04 | 0.3.0-dev, async forward loads (`aef44b9`) | 0.859 | 4.518 | 0.964 | 5.021 | [compare](bench/results/2026-10-04-rtx5060ti-compare-l1.md) |
| 2026-10-04 | 0.2.0 (baseline) | 1.387 | 4.874 | 0.949 | 4.708 | [compare](bench/results/2026-10-04-rtx5060ti-compare-v0.2-baseline.md) |
| 2026-10-03 | 0.2 kernels (before release) | 1.419 | 4.973 | 0.945 | 4.814 | [compare](bench/results/2026-10-03-rtx5060ti-compare.md) |

Version against version, alternated in one session ([`bench/ab.py`](bench/ab.py)):

| date | change | forward kernel | backward kernel | training call | report |
|---|---|--:|--:|--:|---|
| 2026-10-05 | **0.3.0 → 0.4.0** | +0.0 % | **−6.9 %** | **−1.7 %** | [A/B](bench/results/2026-10-05-rtx5060ti-ab-v0.3.0-vs-v0.4.md) |
| 2026-10-05 | backward: dV half skips the dS wait (**rejected**) | +0.1 % | +1.4 % | +0.3 % | [A/B](bench/results/2026-10-05-rtx5060ti-ab-fa14-vs-fa16-rejected.md) |
| 2026-10-05 | backward: skip the tail tiles' dead work (**rejected**) | +0.1 % | +14.2 % | +5.9 % | [A/B](bench/results/2026-10-05-rtx5060ti-ab-fa14-vs-fa15-rejected.md) |
| 2026-10-05 | backward: each product pair split across the block's halves | +2.5 %* | **−6.9 %** | **−2.5 %** | [A/B](bench/results/2026-10-05-rtx5060ti-ab-v0.3.0-vs-fa14.md) |
| 2026-10-04 | 0.2.0 → 0.3.0 | **−40.0 %** | **−16.4 %** | **−16.2 %** | [A/B](bench/results/2026-10-04-rtx5060ti-ab-v0.2.0-vs-v0.3.md) |
| 2026-10-04 | 128 threads × 8 × 4 outputs (**rejected**) | +15.5 % | −1.8 % | +2.7 % | devlog FA11 |
| 2026-10-04 | async, double-buffered backward loads | −0.1 % | **−14.3 %** | **−7.2 %** | [A/B](bench/results/2026-10-04-rtx5060ti-ab-l1-vs-fa9.md) |
| 2026-10-04 | async forward loads | **−42.6 %** | +0.3 % | **−9.7 %** | [A/B](bench/results/2026-10-04-rtx5060ti-ab-v0.2-vs-l1.md) |

\* The forward kernel's PTX is byte-identical in both builds; overlapping rounds (context, not code).

### RTX 5090

| date | crate | forward | fwd + bwd | SDPA forward | SDPA fwd + bwd | report |
|---|---|--:|--:|--:|--:|---|
| 2026-10-05 | **0.4.0** (`468c712`) | **0.178** | **0.813** | 0.206 | 1.022 | [compare](bench/results/2026-10-05-rtx5090-compare-v0.4.md) |
| 2026-10-04 | 0.3.0 (`32f5ec2`) | 0.192 | 0.891 | 0.243 | 1.104 | [compare](bench/results/2026-10-04-rtx5090-compare-v0.3.md) |
| 2026-10-03 | 0.2 kernels (before release) | 0.226 | 0.885 | 0.207 | 1.022 | [compare](bench/results/2026-10-03-rtx5090-compare.md) |

0.3.0 → 0.4.0, alternated: backward kernel −5.7 %, forward +0.0 %, training call −2.4 %
([A/B](bench/results/2026-10-05-rtx5090-ab-v0.3.0-vs-v0.4.md)).
0.2.0 → 0.3.0, alternated: forward kernel −19.7 % in no-grad calls, −38.5 % inside training calls;
backward −8.9 %; training call −12.8 % ([A/B](bench/results/2026-10-04-rtx5090-ab-v0.2.0-vs-v0.3.md)).

### RTX 4090

| date | crate | forward | fwd + bwd | SDPA forward | SDPA fwd + bwd | report |
|---|---|--:|--:|--:|--:|---|
| 2026-10-05 | **0.4.0** (`a39eb66`), power limit 350 W | **0.255** | **1.183** | 0.333 | 1.555 | [compare](bench/results/2026-10-05-rtx4090-compare-v0.4.md) |
| 2026-10-04 | 0.3.0 (`32f5ec2`) | 0.254 | 1.200 | 0.334 | 1.559 | [compare](bench/results/2026-10-04-rtx4090-compare-v0.3.md) |

0.3.0 → 0.4.0, alternated (a different box, capped at 350 W): backward kernel −5.6 %, forward +0.0 %,
training call −2.3 % ([A/B](bench/results/2026-10-05-rtx4090-ab-v0.3.0-vs-v0.4.md)).
0.2.0 → 0.3.0, alternated: forward kernel −36.4 %; backward −11.9 %; training call −13.0 %
([A/B](bench/results/2026-10-04-rtx4090-ab-v0.2.0-vs-v0.3.md)).

### RTX 3090

| date | crate | forward | fwd + bwd | SDPA forward | SDPA fwd + bwd | report |
|---|---|--:|--:|--:|--:|---|
| 2026-10-05 | **0.4.0** (`313005b`) | **0.580** | **2.337** | 0.832 | 3.522 | [compare](bench/results/2026-10-05-rtx3090-compare-v0.4.md) |

0.3.0 → 0.4.0, alternated: backward kernel −5.7 %, forward +0.0 %, training call −2.7 %
([A/B](bench/results/2026-10-05-rtx3090-ab-v0.3.0-vs-v0.4.md)). Consumer Ampere (sm_86): TF32 tensor
throughput about the plain fp32 rate, so SDPA's 3xTF32 path is at its weakest here (1.5× our time).

### A100-SXM4-40GB

| date | crate | forward | fwd + bwd | SDPA forward | SDPA fwd + bwd | report |
|---|---|--:|--:|--:|--:|---|
| 2026-10-05 | 0.4.0 (`a39eb66`) | 0.625 | 2.856 | **0.438** | **1.650** | [compare](bench/results/2026-10-05-a100-compare-v0.4.md) |

**On the A100, PyTorch's SDPA is the faster fp32 attention: 1.7× on a training call.** Its fp32
path runs 3xTF32 on tensor cores, whose TF32 rate is 8× the A100's plain fp32 rate; on the
consumer cards that rate is about the plain fp32 rate, which is why this crate's FMAs win there.
0.3.0 → 0.4.0, alternated: backward kernel −0.5 % (the A100 has half the fp32 lanes per SM of a
consumer card: arithmetic-bound, not shared-memory-bound), forward +0.0 %, training call −0.2 %
([A/B](bench/results/2026-10-05-a100-ab-v0.3.0-vs-v0.4.md)). devlog FA17.

### Accuracy

Against an fp64 reference (normwise relative error): this crate's figures are identical on all
five cards and unchanged since 0.2.0 (0.3.0's and 0.4.0's outputs are bit-identical to 0.2.0's).
SDPA's vary by card: below, the RTX 5090 and 5060 Ti; on the RTX 4090, 3090 and the A100, 7.6e-7 / 1.0e-6
/ 1.0e-6 / 8.0e-7.

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
| 2026-10-04 | RTX 5090, 128 | **0.90×** | this crate's 0.3.0 (+1.8 % throughput against 0.2.0, [A/B](bench/results/2026-10-04-rtx5090-trainer-ab-v0.2.0-vs-v0.3.md)) |
| 2026-10-03 | RTX 5090, 128 | 0.93× | candle's index_add and fused GELU backward |
| 2026-10-03 | RTX 5090, 128 | 1.01× | this crate's one-kernel backward; LayerNorm statistics |
| 2026-10-02 | RTX 5090, 128 | 1.19× | this crate, v0.1 |
| 2026-10-02 | RTX 5090, 128 | 1.32× | composed attention |
| 2026-10-01 | RTX 5060 Ti, 64 | 1.376× | PyTorch re-measured (129.8 ms) |
| 2026-08-01 | RTX 5060 Ti, 64 | 2.8× | the speed campaign's first rungs |
| 2026-07-29 | RTX 5060 Ti, 64 | 5.1× (≈ 3.9× against the re-measured PyTorch) | the starting point |
