# SPDX-License-Identifier: MIT OR Apache-2.0
"""Bit-for-bit comparison of two builds of the crate, over a fixed set of shapes.

A change that touches only HOW data reaches the arithmetic (asynchronous loads, a different
synchronisation) must leave every output bit unchanged; this is that gate (devlog FA7). For each
shape below, `dump` runs a `compare` example binary in its `accuracy` mode on seeded inputs and
keeps both entry points' outputs (`o` and the gradient `dqkv`, which is computed from the saved
log-sum-exp L, so equal gradients vouch for L too); `diff` compares two dumps bit for bit.

Run (GPU, RTX 5060 Ti: ~15 s per dump), from the crate root:
  py -3.14 bench/bitwise.py dump target/release/examples/compare.exe target/bitwise/v02
  py -3.14 bench/bitwise.py diff target/bitwise/v02 target/bitwise/l1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402  (the same seeded inputs as the benchmark)

# (batch, heads, seq, causal): the canvas trainer's shape, and the parity tests' odd lengths.
SHAPES = [(64, 6, 240, False), (64, 6, 240, True), (2, 3, 7, False), (2, 3, 7, True),
          (1, 2, 97, False), (1, 2, 97, True)]
OUTPUTS = ("candle_fused_qkv", "candle_fused")
SEED = 0


def shape_name(b: int, h: int, s: int, causal: bool) -> str:
  """The directory name of one shape.

  >>> shape_name(64, 6, 240, False)
  'b64_h6_s240_full'
  >>> shape_name(2, 3, 7, True)
  'b2_h3_s7_causal'
  """
  return f"b{b}_h{h}_s{s}_{'causal' if causal else 'full'}"


def first_difference(a: list[int], b: list[int]) -> tuple[int, int]:
  """`(count, first index)` of the positions where two equal-length bit patterns differ;
  `(0, -1)` when they are identical.

  >>> first_difference([1, 2, 3], [1, 2, 3])
  (0, -1)
  >>> first_difference([1, 2, 3, 4], [1, 9, 3, 7])
  (2, 1)
  """
  diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
  return len(diffs), (diffs[0] if diffs else -1)


def dump(exe: Path, out: Path) -> None:
  """Every shape's inputs, then the binary's `accuracy` mode on them."""
  for b, h, s, causal in SHAPES:
    d = out / shape_name(b, h, s, causal)
    d.mkdir(parents=True, exist_ok=True)
    inputs = d / "inputs.safetensors"
    compare.make_inputs(inputs, b, h, s, SEED)
    cmd = [str(exe), "accuracy", "--inputs", str(inputs), "--out", str(d), "--heads", str(h)]
    subprocess.run(cmd + (["--causal"] if causal else []), check=True, capture_output=True)
    print(f"dumped {d.name}", flush=True)


def diff(a: Path, b: Path) -> int:
  """Compare two dumps bit for bit; the number of differing tensors (0 = identical)."""
  import torch
  from safetensors.torch import load_file
  bad = 0
  for shape in SHAPES:
    name = shape_name(*shape)
    for out in OUTPUTS:
      ta, tb = (load_file(str(p / name / f"{out}.safetensors")) for p in (a, b))
      for key in sorted(ta):
        x, y = ta[key].contiguous(), tb[key].contiguous()
        if x.shape != y.shape:
          print(f"{name} {out} {key}: SHAPE {tuple(x.shape)} vs {tuple(y.shape)}")
          bad += 1
          continue
        n = int((x.view(torch.int32) != y.view(torch.int32)).sum())
        if n:
          first = int((x.view(torch.int32) != y.view(torch.int32)).flatten().nonzero()[0])
          print(f"{name} {out} {key}: {n} of {x.numel()} differ (first at {first})")
          bad += 1
        else:
          print(f"{name} {out} {key}: identical ({x.numel()} values)")
  print("BITWISE IDENTICAL" if bad == 0 else f"{bad} TENSORS DIFFER")
  return bad


def main() -> None:
  """`dump <exe> <dir>` or `diff <dir_a> <dir_b>`."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("command", choices=["dump", "diff"])
  p.add_argument("a", type=Path)
  p.add_argument("b", type=Path)
  args = p.parse_args()
  sys.stdout.reconfigure(encoding="utf-8")
  if args.command == "dump":
    dump(args.a, args.b)
  else:
    sys.exit(1 if diff(args.a, args.b) else 0)


if __name__ == "__main__":
  main()
