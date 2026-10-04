<!-- Source: askesis-acsp14 reference/canvas/data/analysis/v030_0b/ (run.log, kern_sum_{C,F}.csv,
round*.jsonl; git-ignored there), summarised here. Phase 0b of docs/roadmap-v0.3.0.md. -->

# A canvas evaluation, composed vs fused attention — RTX 5060 Ti, 2026-10-04 07:25–07:28

**What ran.** askesis canvas `eval`: checkpoint `acspv_sasinv_px_s2_d1/canvas_ema_e80` (6 layers,
384 wide, 6 heads, head_dim 64), the (8, 8) tower probe (200 problems), greedy, left-to-right,
batch 128, 101 rounds. Two binaries built minutes before from ONE candle tree (fork branch
`gradstore-accumulate-0.11`, `655244d8`): composed (md5 `d84b8140…`) and fused (md5 `37798384…`,
candle-mi → candle-fused-attn 0.2.0 from crates.io).

**Machine.** RTX 5060 Ti, driver 610.88 (WDDM), Windows 11, Ryzen 9 5950X; nsys 2025.5.2. GPU
P0/P1 at 2,632–2,640 MHz, 60–62 °C through the rounds (warm from the run before).

**Protocol.** One warm-up eval per binary, discarded (PTX JIT); four rounds alternated C, F, F, C
(`decode wall-clock`, which excludes loading); one nsys capture per binary (`--trace=cuda`,
`--force-overwrite=true`), `nsys stats --report cuda_gpu_kern_sum` read from its own output.

## Wall time (decode only)

| round | binary | decode wall-clock |
|---|---|--:|
| 1 | composed | 17.288 s |
| 2 | fused | 15.667 s |
| 3 | fused | 15.669 s |
| 4 | composed | 17.302 s |

Fused faster by 9.4 % (worst fused 15.669 s < best composed 17.288 s). Validity 154 / 200 for
both; **the 200 plans are identical** between the binaries, and identical across each binary's
two rounds.

## GPU kernel time (one nsys capture each)

| kernel | composed | fused |
|---|--:|--:|
| `magma_sgemmEx_kernel` (cuBLAS sgemm: linear layers, + attention's QKᵀ / PV when composed) | 9.906 s (61.9 %) | 9.159 s (63.6 %) |
| `fattn_fwd_f32_d64` (this crate's forward, 1,212 calls = 6 layers × 202 forwards) | — | **2.666 s (18.5 %)** |
| `ugelu_erf_f32` | 1.232 s (7.7 %) | 1.250 s (8.7 %) |
| `badd_f32` | 0.822 s (5.1 %) | 0.811 s (5.6 %) |
| `layernorm_stats_f32` | 0.468 s (2.9 %) | 0.457 s (3.2 %) |
| `ucopy_runs_f32` (head split / merge) | 1.312 s (8.2 %) | — |
| `softmax_f32` | 0.923 s (5.8 %) | — |
| CUTLASS SIMT sgemm 64×64 (batched attention matmul) | 0.748 s (4.7 %) | — |
| `affine_f32` (the score scale) | 0.543 s (3.4 %) | — |
| `is_u32_f32` | 0.058 s (0.4 %) | 0.059 s (0.4 %) |
| **total** | **16.012 s** | **14.402 s** |

The wall-time gain (1.63 s) matches the kernel-time gain (1.61 s): the evaluation is GPU-bound.
