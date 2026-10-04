<!-- Source: Nsight Compute 2026.3.1, `--set full`, one launch of each forward kernel after 10
warm-up calls, RTX 5060 Ti (driver 610.88, WDDM; base clocks locked by ncu, ~2.60 GHz), inputs
of target/compare/20261004-baseline (b 64 · h 6 · s 240 · d 64, seed 0) through compare.py's
`nsys` mode on both sides; crate 0ed05e7 (v0.2 kernels), torch 2.10.0+cu130. Raw reports:
profiles/2026-10-04-rtx5060ti-fwd-*.ncu-repz (open with ncu-ui). devlog FA5. -->

# The two forward kernels under Nsight Compute — RTX 5060 Ti, 2026-10-04

| | candle-fused-attn v0.2 `fattn_fwd_f32_d64` | SDPA `fmha_cutlassF_f32_aligned_64x64_rf_sm80` |
|---|--:|--:|
| block size | 256 threads (8 warps) | 128 threads (4 warps) |
| grid | 1,536 blocks | 1,536 blocks |
| registers per thread | 80 | 168 |
| dynamic shared memory per block | 69,632 B | 36,352 B |
| blocks per SM (limit: registers / shared memory / warps) | 3 / **1** / 6 | 3 / **2** / 12 |
| theoretical occupancy | 8 warps per SM (16.67 %) | 8 warps per SM (16.67 %) |
| achieved occupancy | 7.85 warps (16.35 %) | 7.89 warps (16.44 %) |
| duration under ncu (clocks locked; not a timing) | 1.43 ms | 0.97 ms |
| compute (SM) throughput | 29.7 % | **86.4 %** |
| issue slots busy | 29.7 % | 24.7 % |
| instructions executed (warp level) | 158.6 M | 89.4 M |
| shared-memory loads (warp level) | 12.58 M | 3.59 M |
| shared-memory load bank conflicts | 2,406 | 1,669,176 |
| global loads (warp level) | 414,720 | 0 (async copies) |
| FP32 FMA pipes busy (fma / fmaheavy / fmalite) | 22.2 / 23.6 / 24.8 % | 7.3 / 13.1 / 7.4 % |
| tensor pipe (HMMA) busy | 0 % | **42.2 %** |
| TF32 tensor operations | 0 | 19.33 G (= 3 × 6.44 GFLOP: 3xTF32 on the 64-padded tiles) |

**Warp stalls**, in warp-cycles per issued instruction (top reasons):

| reason | ours | SDPA |
|---|--:|--:|
| long scoreboard (waiting on global / L1TEX loads) | **2.25** | 0.56 |
| short scoreboard (waiting on shared-memory / MIO results) | **1.74** | 0.19 |
| math pipe throttle (a math pipe saturated) | 0.02 | **3.63** |
| wait (fixed-latency dependency) | 0.66 | 1.67 |
| barrier | 0.22 | 0.38 |
| MIO throttle | 0.09 | 0.05 |
