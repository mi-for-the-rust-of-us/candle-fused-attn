# SPDX-License-Identifier: MIT OR Apache-2.0
"""Two builds of the crate against each other: kernel time per call, alternated in one session.

`compare.py` puts ONE candle build beside PyTorch; a lever is judged against the build before it
(devlog FA7 on). This takes nsys captures of two `compare` example binaries in alternating rounds
(round r: A then B if r is even, B then A if odd), on the same seeded inputs, for the `fwd` and
`train` phases of `candle_fused_qkv`, and reports kernel milliseconds per call — the total and
each kernel — with `compare.py`'s decision rule (worst round of one below the best of the other).

Run (GPU, RTX 5060 Ti: ~5 s per capture, 12 captures with the defaults), from the crate root:
  py -3.14 bench/ab.py target/v02/compare.exe target/release/examples/compare.exe \\
     --inputs target/compare/20261004-baseline/inputs.safetensors --nsys <path to nsys.exe>
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402  (the nsys parsing, per-call accounting and decision rule)


def order(r: int) -> tuple[str, str]:
  """Round `r`'s order of the two builds.

  >>> order(0), order(1), order(2)
  (('A', 'B'), ('B', 'A'), ('A', 'B'))
  """
  return ("A", "B") if r % 2 == 0 else ("B", "A")


def change(a: float, b: float) -> str:
  """B relative to A, in percent, signed.

  >>> change(2.0, 1.5)
  '-25.0 %'
  >>> change(1.0, 1.1)
  '+10.0 %'
  """
  return f"{(b / a - 1) * 100:+.1f} %"


def capture(nsys: str, exe: Path, inputs: Path, out: Path, phase: str, calls: int, warmup: int,
            heads: int) -> dict[str, object]:
  """One nsys capture of `calls` calls of `candle_fused_qkv` in `phase`, after `warmup` calls."""
  rep = out / f"{exe.parent.name}_{phase}_{time.strftime('%H%M%S')}"
  subprocess.run([nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                  "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                  "--force-overwrite=true", "-o", str(rep), str(exe), "nsys",
                  "--inputs", str(inputs), "--out", str(out), "--heads", str(heads),
                  "--only", "candle_fused_qkv", "--phase", phase, "--calls", str(calls),
                  "--warmup", str(warmup)], check=True, capture_output=True)
  stats = subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                          str(rep) + ".nsys-rep"], check=True, capture_output=True, text=True)
  return compare.per_call(compare.parse_kern_sum(stats.stdout), calls)


def main() -> None:
  """The alternated captures, then the per-phase and per-kernel summary."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("a", type=Path, help="build A (the reference)")
  p.add_argument("b", type=Path, help="build B (the change)")
  p.add_argument("--inputs", type=Path, required=True)
  p.add_argument("--nsys", type=str, required=True)
  p.add_argument("--heads", type=int, default=6)
  p.add_argument("--rounds", type=int, default=3)
  p.add_argument("--calls", type=int, default=20)
  p.add_argument("--warmup", type=int, default=10)
  p.add_argument("--phases", type=str, default="fwd,train")
  p.add_argument("--out", type=Path, default=Path("target") / "ab" / time.strftime("%Y%m%d-%H%M%S"))
  a = p.parse_args()
  sys.stdout.reconfigure(encoding="utf-8")
  a.out.mkdir(parents=True, exist_ok=True)
  exes = {"A": a.a, "B": a.b}
  phases = a.phases.split(",")
  runs: dict[str, dict[str, list[dict]]] = {k: {ph: [] for ph in phases} for k in exes}
  for r in range(a.rounds):
    for which in order(r):
      for ph in phases:
        state = compare.gpu_state()
        res = capture(a.nsys, exes[which], a.inputs, a.out, ph, a.calls, a.warmup, a.heads)
        res["gpu"] = state
        runs[which][ph].append(res)
        print(f"round {r + 1} {which} {ph}: {res['ms']:.3f} ms/call  [{state}]", flush=True)
  (a.out / "ab.json").write_text(json.dumps({"a": str(a.a), "b": str(a.b), "runs": runs},
                                            indent=1), encoding="utf-8")
  print(f"\nA = {a.a}\nB = {a.b}\n")
  for ph in phases:
    ta = [x["ms"] for x in runs["A"][ph]]
    tb = [x["ms"] for x in runs["B"][ph]]
    print(f"{ph}: A {compare.fmt_range(ta)}  B {compare.fmt_range(tb)}  "
          f"B vs A {change(statistics.median(ta), statistics.median(tb))}  "
          f"({compare.verdict('A', ta, 'B', tb)})")
    names = sorted({k["name"] for x in runs["A"][ph] + runs["B"][ph] for k in x["kernels"]})
    for n in names:
      ka = [next((k["us_per_call"] for k in x["kernels"] if k["name"] == n), 0.0)
            for x in runs["A"][ph]]
      kb = [next((k["us_per_call"] for k in x["kernels"] if k["name"] == n), 0.0)
            for x in runs["B"][ph]]
      if max(ka + kb) >= 5:
        ma, mb = statistics.median(ka), statistics.median(kb)
        print(f"    {n[:40]:40s} A {ma:8.1f} µs  B {mb:8.1f} µs  "
              f"{change(ma, mb) if ma else 'new'}")


if __name__ == "__main__":
  main()
