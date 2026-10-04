<!-- Source: a summary, not a new measurement. Every row is quoted from the askesis project's
records (not public): reference/canvas/docs/devlog.md (branch acsp14) and rentals.md, at the
entries named in the last column. The trainer is the canvas masked-diffusion model (6 layers,
384 wide, 6 heads, head_dim 64) trained through candle-mi; "PyTorch" is its reference
implementation of the same step, fp32. -->

# The training step against PyTorch, 2026-07-29 → 2026-10-03

One training step of the same model, candle (with candle-mi and, from 2026-10-02, this crate)
against its PyTorch reference, fp32. Ratio = candle's time ÷ PyTorch's: above 1, candle is slower.

| date | card, batch | candle | PyTorch | ratio | what changed | source (askesis) |
|---|---|--:|--:|--:|---|---|
| 2026-07-29 | RTX 5060 Ti, 64 | 505 ms | 99 ms | 5.1× | the starting point | devlog, Phase C |
| 2026-08-01 | RTX 5060 Ti, 64 | | | 3.9× | candle's gradient accumulation patch | devlog, Phase E |
| 2026-08-01 | RTX 5060 Ti, 64 | | | 2.8× | two more rungs of the speed campaign | devlog, Phase E″ |
| 2026-10-01 | RTX 5060 Ti, 64 | | 129.8 ms | 1.376× | Phases E–J; PyTorch re-measured | devlog, "Speed campaign — where it stands" |
| 2026-10-01 | RTX 5090, 64 / 128 | | | 1.27× / 1.29× | first rented-5090 comparison | devlog, `c11496b` |
| 2026-10-02 | RTX 5090, 64 / 128 | 41 / 81 ms | 31.3 / 61.3 ms | 1.31× / 1.32× | composed attention | rentals.md; design note `2a07fdd` |
| 2026-10-02 | RTX 5090, 64 / 128 | 37 / 73 ms | 31.3 / 61.3 ms | 1.18× / 1.19× | this crate, v0.1 | rentals.md; design note `2a07fdd` |
| 2026-10-03 | RTX 5090, 128 | 69 ms | 68.1 ms | 1.01× | v0.2's one-kernel backward; LayerNorm statistics in candle | rentals.md `b26b346` |
| 2026-10-03 | RTX 5090, 64 / 128 | 34 / 67 ms | 36.5 / 72.1 ms | **0.93×** | candle's index_add and fused GELU backward | rentals.md `e1a920b` |

**On the starting ratio.** The 2026-07-29 PyTorch figure (99 ms) had no instrument behind it; the
devlog's finding N1 re-measured PyTorch's step at 129.8 ms (2026-10-01). Against that, the
starting point was ≈ 3.9× (505 ÷ 129.8), not 5.1×. Both are kept, with this note.

**Not in this table.** candle-fused-attn v0.3.0's gains (2026-10-04: forward kernel −42.6 %,
backward kernel −14.3 %) have not yet been measured in the whole training step; that measurement
is part of the v0.3.0 release (`docs/roadmap-v0.3.0.md`, Phase 3). Comparisons across rows on
different boxes or dates are indicative only: a ratio is valid within one box and one session.
