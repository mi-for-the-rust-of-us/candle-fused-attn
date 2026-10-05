<!-- Source: bench/ab.py --rounds 6, 2026-10-05 ~10:40 CEST, RTX 5060 Ti (WDDM), nsys 2025.5.2,
inputs target/compare/20261004-baseline (b 64 · h 6 · s 240 · d 64, seed 0). A = FA14 (55aea18), B = FA15
(95cdd77, rejected and reverted by c430372). Raw: target/ab/fa15/ab.json. devlog FA15. -->

# FA14 against FA15 (the backward skips its dead tail work), alternated — RTX 5060 Ti, 2026-10-05

Kernel time per call (nsys), `candle_fused_qkv`, 20 calls per capture after 10 warm-up calls;
round r runs A then B if r is even, B then A if odd.

| phase | FA14 median [range] | FA15 median [range] | FA15 vs FA14 |
|---|--:|--:|--:|
| fwd | 0.803 [0.787–0.832] | 0.806 [0.777–0.847] | +0.3 % |
| train | 5.193 [5.143–5.261] | 5.499 [5.419–5.528] | +5.9 % |

Per kernel, training call, µs per call — per round and median:

| kernel | FA14 rounds | FA15 rounds | FA14 median | FA15 median | change |
|---|---|---|--:|--:|--:|
| `fattn_bwd_f32_d64` | 2197 2163 2223 2183 2191 2228 | 2554 2505 2485 2495 2504 2531 | 2194.3 | 2504.8 | +14.2 % |
| `badd_f32` | 1264 1249 1212 1224 1206 1260 | 1216 1266 1217 1210 1251 1215 | 1236.5 | 1216.5 | -1.6 % |
| `fattn_fwd_f32_d64` | 812 799 785 785 794 788 | 785 798 785 819 785 830 | 790.8 | 791.8 | +0.1 % |
| `bmul_f32` | 575 550 552 617 550 549 | 550 550 550 558 583 567 | 551.1 | 554.0 | +0.5 % |
| `fast_sum_f32` | 265 264 275 266 300 266 | 264 266 265 267 266 266 | 265.6 | 265.5 | -0.0 % |
| `fattn_bwd_dot_f32_d64` | 148 117 117 117 152 117 | 125 117 117 118 117 118 | 117.2 | 117.5 | +0.3 % |
| `const_set_f32` | 1 1 1 1 1 1 | 1 1 1 1 1 1 | 0.8 | 0.7 | -0.9 % |

**Backward kernel +14.2 %, separated the wrong way** (FA15's fastest round 2,485 µs above FA14's slowest,
2,228 µs). Outputs bitwise identical. Rejected, reverted (`MEASURED-REVERT`).
