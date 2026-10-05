<!-- Source: bench/ab.py --rounds 6, 2026-10-05 ~11:05 CEST, RTX 5060 Ti (WDDM), nsys 2025.5.2,
inputs target/compare/20261004-baseline (b 64 · h 6 · s 240 · d 64, seed 0). A = FA14 (55aea18), B = FA16
(b0aa346, rejected and reverted by 9a5a13d). Raw: target/ab/fa16/ab.json. devlog FA16. -->

# FA14 against FA16 (the dV half stops waiting for dS), alternated — RTX 5060 Ti, 2026-10-05

Kernel time per call (nsys), `candle_fused_qkv`, 20 calls per capture after 10 warm-up calls;
round r runs A then B if r is even, B then A if odd.

| phase | FA14 median [range] | FA16 median [range] | FA16 vs FA14 |
|---|--:|--:|--:|
| fwd | 0.816 [0.779–0.838] | 0.845 [0.777–0.907] | +3.5 % |
| train | 5.182 [5.126–5.228] | 5.199 [5.174–5.277] | +0.3 % |

Per kernel, training call, µs per call — per round and median:

| kernel | FA14 rounds | FA16 rounds | FA14 median | FA16 median | change |
|---|---|---|--:|--:|--:|
| `fattn_bwd_f32_d64` | 2197 2200 2191 2225 2167 2254 | 2227 2160 2229 2230 2200 2239 | 2198.3 | 2228.3 | +1.4 % |
| `badd_f32` | 1266 1264 1202 1232 1230 1213 | 1251 1289 1245 1253 1210 1261 | 1231.3 | 1251.7 | +1.7 % |
| `fattn_fwd_f32_d64` | 785 786 784 801 785 796 | 813 803 786 786 787 786 | 785.5 | 786.5 | +0.1 % |
| `bmul_f32` | 556 546 566 552 546 581 | 556 547 544 548 593 608 | 553.9 | 551.7 | -0.4 % |
| `fast_sum_f32` | 266 264 265 266 299 267 | 266 265 275 265 266 265 | 265.9 | 265.7 | -0.1 % |
| `fattn_bwd_dot_f32_d64` | 117 117 117 117 118 117 | 117 117 118 117 117 117 | 117.1 | 117.2 | +0.1 % |
| `const_set_f32` | 1 1 1 1 1 1 | 1 1 1 1 1 1 | 0.8 | 0.8 | +0.2 % |

**Backward kernel +1.4 %, not separated** (rounds 2,167–2,254 µs against 2,160–2,239 µs). Outputs bitwise
identical. Rejected, reverted (`MEASURED-REVERT`).
