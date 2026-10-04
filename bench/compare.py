# SPDX-License-Identifier: MIT OR Apache-2.0
"""candle-fused-attn against PyTorch's fp32 attentions: speed AND accuracy, one method both sides.

The question it answers: is there an fp32 attention, forward AND backward, faster than ours at the
canvas trainer's shape (b 64 · h 6 · s 240 · d 64, non-causal)? The one library to beat is
PyTorch's `scaled_dot_product_attention`, whose fp32 path is the memory-efficient kernel (xFormers /
CUTLASS example 41); its flash and cuDNN backends are probed and their refusals recorded, not
assumed. The trainer's PyTorch reference (`canvas_parity.py`) composes its attention by hand, and
that composition is a candidate too, verbatim.

The unit every candidate computes, so nothing is compared across different work: the fused
projection `qkv` `[b, s, 3·h·64]` to the merged output `[b, s, h·64]` (head split and merge
included, as the trainer pays them); the backward from `dout` to `dqkv`, driven by the loss head
`sum(o ∘ dout)` on both sides.

Method:
1. Seeded inputs (`qkv`, `dout`, N(0, 1)) written once to safetensors; both sides read THEM.
2. Accuracy: each candidate's `o` and `dqkv` against an fp64 reference of the same math (max-abs
   and normwise relative error, per `o`, `dq`, `dk`, `dv`); the floor is the fp64 reference
   rounded to fp32. Determinism: a second forward + backward compared BITWISE with the first.
3. Wall time, the same in both languages: per candidate and phase, warm-up calls, then samples of
   N back-to-back calls between two device syncs (ms per call). Phases: `fwd` (no autograd, as the
   trainer's no-grad forwards), `train` (forward + head + backward), `head` (the head alone, one
   per side) — `net = train − head`, paired within a process. Each side runs in its own process;
   rounds alternate which side goes first and rotate the candidate order; the GPU's clocks and
   temperature are recorded before every process.
4. Kernel time (nsys, both sides the same tool): one process per candidate and phase, the timed
   calls inside a `cudaProfilerApi` capture range; per-call time = kernel total / calls, and every
   kernel's launch count must be a multiple of the calls (else it is listed as irregular).

Decision rule (fixed before any number was seen): X is faster than Y on a phase iff X's WORST
round median is below Y's BEST round median. Overlapping ranges are reported as "not separated".

Run (GPU, RTX 5060 Ti: ~2 min with nsys; ~1 min with `--no-nsys`), from the crate root:
  py -3.14 bench/compare.py run
nsys is found through `--nsys`, then `$NSYS`, then `PATH`; without it the kernel pass is skipped
and the report says so. Outputs go to `target/compare/<timestamp>/` (inputs, per-process JSON,
per-candidate outputs, `report.md`, `report.json`).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
HEAD_DIM = 64
TORCH_CANDIDATES = ("torch_composed", "torch_sdpa_efficient", "torch_sdpa_math")
CANDLE_CANDIDATES = ("candle_fused_qkv", "candle_fused")
PHASES = ("fwd", "train", "net")
PARTS = ("o", "dq", "dk", "dv")


# ---------------------------------------------------------------------------------------------
# Pure helpers (doctested)
# ---------------------------------------------------------------------------------------------

def summarize(xs: list[float]) -> dict[str, float]:
  """Median, quartiles (inclusive method), min and max of `xs`.

  >>> summarize([3.0, 1.0, 2.0, 4.0])
  {'median': 2.5, 'q1': 1.75, 'q3': 3.25, 'min': 1.0, 'max': 4.0}
  >>> summarize([2.0])["q3"]
  2.0
  """
  if len(xs) == 1:
    return {"median": xs[0], "q1": xs[0], "q3": xs[0], "min": xs[0], "max": xs[0]}
  q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
  return {"median": statistics.median(xs), "q1": q1, "q3": q3, "min": min(xs), "max": max(xs)}


def faster(xs: list[float], ys: list[float]) -> bool:
  """The decision rule: every round of `xs` beats every round of `ys` (worst X < best Y).

  >>> faster([1.0, 1.2], [1.3, 1.5])
  True
  >>> faster([1.0, 1.4], [1.3, 1.5])
  False
  """
  return max(xs) < min(ys)


def verdict(name_x: str, xs: list[float], name_y: str, ys: list[float]) -> str:
  """The decision rule, in words.

  >>> verdict("a", [1.0], "b", [2.0])
  'a faster (x2.00)'
  >>> verdict("a", [2.0], "b", [1.0])
  'b faster (x2.00)'
  >>> verdict("a", [1.0, 2.0], "b", [1.5])
  'not separated'
  """
  mx, my = statistics.median(xs), statistics.median(ys)
  if faster(xs, ys):
    return f"{name_x} faster (x{my / mx:.2f})"
  if faster(ys, xs):
    return f"{name_y} faster (x{mx / my:.2f})"
  return "not separated"


def rotate(xs: tuple[str, ...], k: int) -> list[str]:
  """`xs` rotated left by `k`: round `k`'s candidate order (the Rust side does the same).

  >>> rotate(("a", "b", "c"), 1)
  ['b', 'c', 'a']
  >>> rotate(("a", "b"), 3)
  ['b', 'a']
  """
  return [xs[(i + k) % len(xs)] for i in range(len(xs))]


def parse_kern_sum(text: str) -> list[tuple[str, int, int]]:
  """`(name, total_ns, instances)` rows of an `nsys stats --report cuda_gpu_kern_sum` CSV,
  skipping whatever nsys prints before the header.

  >>> parse_kern_sum('Processing x\\n"Time (%)","Total Time (ns)","Instances","Avg (ns)",'
  ...   '"Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"\\n'
  ...   '"90.0","900","20","45.0","45.0","40","50","1.0","k1"\\n'
  ...   '"10.0","100","40","2.5","2.5","2","3","0.1","k2"\\n')
  [('k1', 900, 20), ('k2', 100, 40)]
  """
  lines = text.splitlines()
  start = next(i for i, line in enumerate(lines) if "Total Time (ns)" in line)
  rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
  return [(r["Name"], int(float(r["Total Time (ns)"])), int(r["Instances"])) for r in rows]


def per_call(rows: list[tuple[str, int, int]], calls: int) -> dict[str, object]:
  """Kernel milliseconds per call over a capture of `calls` calls, the per-kernel breakdown, and
  the kernels whose launch count is not a multiple of `calls` (each call should launch the same).

  >>> r = per_call([("k1", 900_000, 20), ("k2", 100_000, 25)], 10)
  >>> r["ms"], r["irregular"]
  (0.1, ['k2'])
  >>> r["kernels"][0]
  {'name': 'k1', 'per_call': 2.0, 'us_per_call': 90.0}
  """
  kernels = [{"name": n, "per_call": inst / calls, "us_per_call": tot / calls / 1e3}
             for n, tot, inst in rows]
  return {
    "ms": sum(tot for _, tot, _ in rows) / calls / 1e6,
    "kernels": kernels,
    "irregular": [n for n, _, inst in rows if inst % calls != 0],
  }


def refusal_reasons(messages: list[str]) -> list[str]:
  """The substantive lines of SDPA's dispatch warnings: without the per-backend "not used because:"
  headers, the "runtime disabled" notes (every backend not forced) and the build-path suffix.

  >>> refusal_reasons(["Flash attention kernel not used because: (Triggered internally at x)",
  ...   "cuDNN attention has been runtime disabled. (Triggered internally at y)",
  ...   "Torch was not compiled with flash attention. (Triggered internally at z)",
  ...   "Torch was not compiled with flash attention. (Triggered internally at z)"])
  ['Torch was not compiled with flash attention.']
  """
  keep = []
  for m in messages:
    line = m.split(" (Triggered internally")[0].strip()
    if "not used because" in line or "runtime disabled" in line or line in keep:
      continue
    keep.append(line)
  return keep


def fmt_range(xs: list[float]) -> str:
  """`median [min–max]`, 3 decimals, for a table cell.

  >>> fmt_range([1.0, 1.25, 1.5])
  '1.250 [1.000–1.500]'
  """
  return f"{statistics.median(xs):.3f} [{min(xs):.3f}–{max(xs):.3f}]"


def package_version(toml_text: str) -> str:
  """The `version` of a `Cargo.toml`'s `[package]` table (another table's `version` is skipped).

  >>> toml = '[package]\\nname = "x"\\nversion = "0.2.0"\\n'
  >>> package_version(toml + '[dependencies]\\nversion = "9"\\n')
  '0.2.0'
  >>> package_version("[workspace]\\n")
  'unknown'
  """
  table = None
  for line in toml_text.splitlines():
    s = line.strip()
    if s.startswith("["):
      table = s
    elif table == "[package]" and s.startswith("version") and "=" in s:
      return s.split("=", 1)[1].strip().strip('"')
  return "unknown"


def lock_entry(lock_text: str, name: str) -> str:
  """How a `Cargo.lock` resolves package `name`: version and source. A `[patch]` to a local path
  leaves the entry without a `source` line, which is how a patched candle fork shows.

  >>> reg = 'source = "registry+https://github.com/rust-lang/crates.io-index"\\n'
  >>> lock = '[[package]]\\nname = "candle-core"\\nversion = "0.11.0"\\n' + reg
  >>> lock_entry(lock, "candle-core")
  '0.11.0 (crates.io)'
  >>> lock_entry(lock.replace(reg, ""), "candle-core")
  '0.11.0 (local path, through [patch])'
  >>> lock_entry(lock, "zip")
  'not in Cargo.lock'
  """
  found = []
  for block in lock_text.split("[[package]]")[1:]:
    fields = {}
    for line in block.splitlines():
      if " = " in line:
        key, value = line.split(" = ", 1)
        fields[key.strip()] = value.strip().strip('"')
    if fields.get("name") == name:
      source = fields.get("source", "")
      where = ("crates.io" if "crates.io-index" in source else source.split("+")[0] if source
               else "local path, through [patch]")
      found.append(f"{fields.get('version', '?')} ({where})")
  return "; ".join(found) or "not in Cargo.lock"


def nvcc_release(text: str) -> str:
  """The release of an `nvcc --version` output; any other text (a capture failure) unchanged.

  >>> nvcc_release("nvcc: NVIDIA (R) Cuda compiler driver\\n"
  ...              "Cuda compilation tools, release 13.1, V13.1.115\\n")
  '13.1 (V13.1.115)'
  >>> nvcc_release("unavailable: no nvcc")
  'unavailable: no nvcc'
  """
  for line in text.splitlines():
    if "release" in line and "," in line:
      parts = [p.strip() for p in line.split(",")]
      release = next((p.split()[-1] for p in parts if p.startswith("release")), "?")
      return f"{release} ({parts[-1]})"
  return text


def cpuinfo_model(text: str) -> str:
  """The first `model name` of a Linux `/proc/cpuinfo`.

  >>> cpuinfo_model("processor\\t: 0\\nmodel name\\t: AMD EPYC 7B13 64-Core Processor\\n")
  'AMD EPYC 7B13 64-Core Processor'
  >>> cpuinfo_model("")
  'unknown'
  """
  for line in text.splitlines():
    if line.startswith("model name") and ":" in line:
      return line.split(":", 1)[1].strip()
  return "unknown"


# ---------------------------------------------------------------------------------------------
# The PyTorch side
# ---------------------------------------------------------------------------------------------

def split_heads(qkv, heads: int):
  """`q, k, v` `[b, h, s, d]` as views of the `[b, s, 3·h·d]` projection (`canvas_parity.py`)."""
  b, s, three_hd = qkv.shape
  hidden = three_hd // 3
  shape = (b, s, heads, hidden // heads)
  q, k, v = qkv.split(hidden, dim=-1)
  return q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2), v.view(shape).transpose(1, 2)


def composed(qkv, heads: int, scale: float, causal: bool):
  """`canvas_parity.py`'s `Attention.forward` between the two projections, verbatim (scale folded
  into q first, rung 1), plus the causal mask the trainer does not use. Any float dtype."""
  import torch
  b, s, three_hd = qkv.shape
  q, k, v = split_heads(qkv, heads)
  scores = (q * scale) @ k.transpose(-2, -1)
  if causal:
    mask = torch.ones(s, s, dtype=torch.bool, device=qkv.device).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
  pattern = torch.softmax(scores, dim=-1)
  return (pattern @ v).transpose(1, 2).reshape(b, s, three_hd // 3)


def sdpa(qkv, heads: int, scale: float, causal: bool):
  """`scaled_dot_product_attention` on the head views; the backend is forced by the caller."""
  import torch
  b, s, three_hd = qkv.shape
  q, k, v = split_heads(qkv, heads)
  o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
  return o.transpose(1, 2).reshape(b, s, three_hd // 3)


def backend_context(name: str) -> contextlib.AbstractContextManager:
  """The forced-backend context of candidate `name` (none for the composition)."""
  from torch.nn.attention import SDPBackend, sdpa_kernel
  backends = {"torch_sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
              "torch_sdpa_math": SDPBackend.MATH}
  return sdpa_kernel([backends[name]]) if name in backends else contextlib.nullcontext()


def probe_backends(qkv, dout, heads: int, scale: float, causal: bool) -> dict[str, str]:
  """Each SDPA backend forced in turn on the real inputs, forward + backward: `ok`, or why not."""
  import torch
  from torch.nn.attention import SDPBackend, sdpa_kernel
  out = {}
  for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                  SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
    x = qkv.detach().clone().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      try:
        with sdpa_kernel([backend]):
          (sdpa(x, heads, scale, causal) * dout).sum().backward()
        torch.cuda.synchronize()
        out[backend.name] = "ok"
      except RuntimeError as e:
        out[backend.name] = "refused: " + " | ".join(refusal_reasons(
          [str(w.message) for w in caught]))
  return out


class TorchBench:
  """The PyTorch half, mirroring `examples/compare.rs` call for call."""

  def __init__(self, inputs: Path, heads: int, causal: bool) -> None:
    import torch
    from safetensors.torch import load_file
    t = load_file(str(inputs), device="cuda")
    self.qkv = t["qkv"].requires_grad_(True)
    self.dout = t["dout"]
    self.heads, self.causal = heads, causal
    self.scale = 1.0 / (self.qkv.shape[-1] // 3 // heads) ** 0.5
    self.standin = torch.zeros_like(self.dout, requires_grad=True)

  def attend(self, name: str, qkv):
    """The merged output of candidate `name`."""
    f = composed if name == "torch_composed" else sdpa
    return f(qkv, self.heads, self.scale, self.causal)

  def train(self, name: str):
    """Forward + head + backward; returns `(o, dqkv)`."""
    self.qkv.grad = None
    o = self.attend(name, self.qkv)
    (o * self.dout).sum().backward()
    return o, self.qkv.grad

  def call(self, name: str, phase: str) -> None:
    """One call of `phase` (`head` ignores `name`)."""
    import torch
    if phase == "fwd":
      with torch.no_grad():
        self.attend(name, self.qkv)
    elif phase == "train":
      self.train(name)
    else:
      self.standin.grad = None
      (self.standin * self.dout).sum().backward()


def torch_side(a: argparse.Namespace) -> None:
  """The PyTorch process: `accuracy`, `time` or `nsys`, outputs as the Rust side writes them."""
  import torch
  from safetensors.torch import save_file
  torch.set_float32_matmul_precision("highest")  # no TF32 in the matmuls: fp32 means fp32
  bench = TorchBench(a.inputs, a.heads, a.causal)
  a.out.mkdir(parents=True, exist_ok=True)
  if a.mode == "accuracy":
    rows = []
    for name in TORCH_CANDIDATES:
      with backend_context(name):
        o1, g1 = bench.train(name)
        o1, g1 = o1.detach().clone(), g1.clone()
        o2, g2 = bench.train(name)
      det = torch.equal(o1, o2) and torch.equal(g1, g2)
      save_file({"o": o1.contiguous().cpu(), "dqkv": g1.contiguous().cpu()},
                str(a.out / f"{name}.safetensors"))
      rows.append({"name": name, "deterministic": det})
    probe = probe_backends(bench.qkv, bench.dout, a.heads, bench.scale, a.causal)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0)),
           "fp32_matmul_precision": torch.get_float32_matmul_precision()}
    (a.out / "torch_accuracy.json").write_text(
      json.dumps({"candidates": rows, "backends": probe, "env": env}, indent=1), encoding="utf-8")
  elif a.mode == "time":
    rows = []

    def timed(name: str, phase: str) -> None:
      for _ in range(a.warmup):
        bench.call(name, phase)
      ms = []
      for _ in range(a.samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.calls):
          bench.call(name, phase)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3 / a.calls)
      rows.append({"name": name, "phase": phase, "ms": ms})

    for name in rotate(TORCH_CANDIDATES, a.round):
      with backend_context(name):  # entered once, outside the timed loop: host cost of no call
        timed(name, "fwd")
        timed(name, "train")
    timed("head", "head")
    (a.out / f"torch_time_r{a.round}.json").write_text(json.dumps(rows), encoding="utf-8")
  else:
    ctx = backend_context(a.only) if a.only != "head" else contextlib.nullcontext()
    with ctx:
      for _ in range(a.warmup):
        bench.call(a.only, a.phase)
      torch.cuda.synchronize()
      torch.cuda.profiler.start()
      for _ in range(a.calls):
        bench.call(a.only, a.phase)
      torch.cuda.synchronize()
      torch.cuda.profiler.stop()


# ---------------------------------------------------------------------------------------------
# The orchestration
# ---------------------------------------------------------------------------------------------

def log(msg: str) -> None:
  """A progress line, flushed (the run is watched from a terminal)."""
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def capture(cmd: list[str]) -> str:
  """`cmd`'s stdout, or a note of its failure (environment records must not abort the run)."""
  try:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError) as e:
    return f"unavailable: {e}"


def gpu_state() -> str:
  """P-state, SM / memory clocks, temperature, power: CPU-side, no CUDA context opened."""
  return capture(["nvidia-smi", "--query-gpu=pstate,clocks.sm,clocks.mem,temperature.gpu,"
                  "power.draw", "--format=csv,noheader"])


def crate_commit() -> str:
  """The crate's commit; on a copy without `.git` (a synced tree on a rented box),
  `$CANDLE_FUSED_ATTN_COMMIT`, else a note saying it is unknown."""
  commit = capture(["git", "-C", str(CRATE), "describe", "--always", "--dirty"])
  if not commit.startswith("unavailable"):
    return commit
  return os.environ.get("CANDLE_FUSED_ATTN_COMMIT",
                        "unknown (no .git; set $CANDLE_FUSED_ATTN_COMMIT)")


def cpu_name() -> str:
  """The host CPU's name (some phases of a run are host-bound)."""
  if os.name == "nt":
    return capture(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor).Name"])
  try:
    return cpuinfo_model(Path("/proc/cpuinfo").read_text(encoding="utf-8"))
  except OSError:
    return platform.processor() or "unknown"


def machine() -> dict[str, object]:
  """Everything about the machine that can change a timing: the crate and the toolchain that
  built its PTX (the driver compiles that PTX at load, so both are part of the timed code), the
  candle it was built against, the OS and driver model, the CPU, the GPU's limits."""

  def smi(query: str) -> str:
    return capture(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])

  lock = CRATE / "Cargo.lock"
  return {
    "crate": f"{package_version((CRATE / 'Cargo.toml').read_text(encoding='utf-8'))} "
             f"at {crate_commit()}",
    "nvcc_on_path": nvcc_release(capture(["nvcc", "--version"])),
    "cuda_env": {k: os.environ[k] for k in ("CUDA_PATH", "CUDA_COMPUTE_CAP", "NVCC")
                 if k in os.environ},
    "candle_core": lock_entry(lock.read_text(encoding="utf-8"), "candle-core")
                   if lock.is_file() else "no Cargo.lock",
    "os": platform.platform(),
    "driver_model": smi("driver_model.current"),
    "cpu": cpu_name(),
    "gpu_limits": smi("power.limit,clocks.max.sm,clocks.max.mem"),
  }


def make_inputs(path: Path, b: int, h: int, s: int, seed: int) -> None:
  """Seeded `qkv` `[b, s, 3·h·64]` and `dout` `[b, s, h·64]`, N(0, 1) fp32, drawn on the CPU."""
  import torch
  from safetensors.torch import save_file
  g = torch.Generator().manual_seed(seed)
  hd = h * HEAD_DIM
  save_file({"qkv": torch.randn(b, s, 3 * hd, generator=g),
             "dout": torch.randn(b, s, hd, generator=g)}, str(path))


def score(out: Path, names: list[str], heads: int, causal: bool) -> dict[str, dict]:
  """Every candidate's `o, dq, dk, dv` against the fp64 reference: max-abs and normwise relative
  error, and the same for the reference rounded to fp32 (the floor no fp32 answer can beat)."""
  import torch
  from safetensors.torch import load_file
  t = load_file(str(out / "inputs.safetensors"), device="cuda")
  qkv = t["qkv"].double().requires_grad_(True)
  dout = t["dout"].double()
  scale = 1.0 / (qkv.shape[-1] // 3 // heads) ** 0.5
  o = composed(qkv, heads, scale, causal)
  (o * dout).sum().backward()

  def parts(o_, g_) -> dict:
    dq, dk, dv = g_.chunk(3, dim=-1)
    return {"o": o_, "dq": dq, "dk": dk, "dv": dv}

  ref = parts(o.detach(), qkv.grad)

  def errors(got: dict) -> dict:
    res = {}
    for p in PARTS:
      diff = got[p].double() - ref[p]
      res[p] = {"max_abs": diff.abs().max().item(),
                "rel_fro": (diff.norm() / ref[p].norm()).item()}
    return res

  scores = {"fp64_rounded_to_fp32": errors({p: ref[p].float() for p in PARTS})}
  for name in names:
    c = load_file(str(out / f"{name}.safetensors"), device="cuda")
    scores[name] = errors(parts(c["o"], c["dqkv"]))
  return scores


def run(a: argparse.Namespace) -> None:
  """Inputs, the Rust build, accuracy, the timed rounds, the nsys pass, the report."""
  out = a.out or CRATE / "target" / "compare" / time.strftime("%Y%m%d-%H%M%S")
  out.mkdir(parents=True, exist_ok=True)
  log(f"output: {out}")
  exe = Path(os.environ.get("CARGO_TARGET_DIR", CRATE / "target")) / "release" / "examples"
  exe = exe / ("compare.exe" if os.name == "nt" else "compare")
  if not a.skip_build:
    log("cargo build --release --features cuda --example compare (CPU)")
    subprocess.run(["cargo", "build", "--release", "--features", "cuda", "--example", "compare"],
                   cwd=CRATE, check=True)
  inputs = out / "inputs.safetensors"
  make_inputs(inputs, a.batch, a.heads, a.seq, a.seed)
  common = ["--inputs", str(inputs), "--out", str(out), "--heads", str(a.heads)]
  common += ["--causal"] if a.causal else []
  side_py = [sys.executable, str(Path(__file__).resolve()), "side"]
  sides = {"torch": side_py, "candle": [str(exe)]}
  env = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "shape": [a.batch, a.heads, a.seq, HEAD_DIM],
         "causal": a.causal, "seed": a.seed, "crate_commit": capture(
           ["git", "-C", str(CRATE), "describe", "--always", "--dirty"]),
         "driver": capture(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
         "gpu_memory_holders": capture(["hmn", "ps"]),
         "machine": machine(), "gpu_start": gpu_state(),
         "timing": {"rounds": a.rounds, "samples": a.samples, "calls": a.calls,
                    "warmup": a.warmup, "nsys_calls": a.nsys_calls}}

  log("accuracy + determinism (GPU, both sides)")
  for name, cmd in sides.items():
    subprocess.run(cmd + ["accuracy"] + common, check=True)
  torch_acc = json.loads((out / "torch_accuracy.json").read_text(encoding="utf-8"))
  candle_acc = json.loads((out / "candle_accuracy.json").read_text(encoding="utf-8"))
  names = list(TORCH_CANDIDATES) + list(CANDLE_CANDIDATES)
  scores = score(out, names, a.heads, a.causal)

  states = []
  for r in range(a.rounds):
    for side in (["torch", "candle"] if r % 2 == 0 else ["candle", "torch"]):
      states.append({"round": r, "side": side, "gpu": gpu_state()})
      log(f"round {r + 1}/{a.rounds}: {side} (GPU) -- {states[-1]['gpu']}")
      subprocess.run(sides[side] + ["time"] + common + [
        "--round", str(r), "--samples", str(a.samples), "--calls", str(a.calls),
        "--warmup", str(a.warmup)], check=True)
  wall = collect_wall(out, a.rounds)

  kernels: dict[str, dict] = {}
  nsys = a.nsys or os.environ.get("NSYS") or shutil.which("nsys")
  if a.no_nsys or not nsys:
    log("nsys pass SKIPPED" + ("" if a.no_nsys else ": no nsys (--nsys, $NSYS or PATH)"))
  else:
    jobs = [(n, p, "torch") for n in TORCH_CANDIDATES for p in ("fwd", "train")]
    jobs += [(n, p, "candle") for n in CANDLE_CANDIDATES for p in ("fwd", "train")]
    jobs += [("head", "head", "torch"), ("head", "head", "candle")]
    for i, (name, phase, side) in enumerate(jobs):
      log(f"nsys {i + 1}/{len(jobs)}: {side} {name} {phase} (GPU)")
      rep = out / f"nsys_{side}_{name}_{phase}"
      subprocess.run([nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                      "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                      "--force-overwrite=true", "-o", str(rep)] + sides[side] + ["nsys"] + common
                     + ["--only", name, "--phase", phase, "--calls", str(a.nsys_calls),
                        "--warmup", str(a.warmup)], check=True, capture_output=True)
      stats = subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                              str(rep) + ".nsys-rep"], check=True, capture_output=True, text=True)
      kernels[f"{side}:{name}:{phase}"] = per_call(parse_kern_sum(stats.stdout), a.nsys_calls)

  env["gpu_end"] = gpu_state()
  report = {"env": env, "torch_env": torch_acc["env"], "backends": torch_acc["backends"],
            "deterministic": {r["name"]: r["deterministic"]
                              for r in torch_acc["candidates"] + candle_acc},
            "accuracy": scores, "wall": wall, "kernels": kernels, "gpu_states": states}
  (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
  md = render(report, names)
  (out / "report.md").write_text(md, encoding="utf-8")
  print(md)


def collect_wall(out: Path, rounds: int) -> dict[str, dict[str, list[float]]]:
  """`wall[name][phase]` = per-round medians (ms per call); `net` = train − head, paired by
  process (same side, same round)."""
  wall: dict[str, dict[str, list[float]]] = {}
  for r in range(rounds):
    for side in ("torch", "candle"):
      rows = json.loads((out / f"{side}_time_r{r}.json").read_text(encoding="utf-8"))
      med = {(x["name"], x["phase"]): statistics.median(x["ms"]) for x in rows}
      head = med[("head", "head")]
      for (name, phase), m in med.items():
        if name == "head":
          wall.setdefault(f"{side}_head", {}).setdefault("head", []).append(m)
          continue
        w = wall.setdefault(name, {})
        w.setdefault(phase, []).append(m)
        if phase == "train":
          w.setdefault("net", []).append(m - head)
  return wall


def render(rep: dict, names: list[str]) -> str:
  """The Markdown report."""
  e, te = rep["env"], rep["torch_env"]
  lines = [f"# candle-fused-attn vs PyTorch — {e['date']}", "",
           f"{te['gpu']} (sm_{te['capability'][0]}{te['capability'][1]}), driver {e['driver']}; "
           f"torch {te['torch']} (CUDA {te['cuda']}, cuDNN {te['cudnn']}, fp32 matmul precision "
           f"`{te['fp32_matmul_precision']}`); crate `{e['crate_commit']}`. Shape b·h·s·d = "
           f"{' · '.join(map(str, e['shape']))}, causal {e['causal']}, seed {e['seed']}. "
           f"Timing: {e['timing']}.", ""]
  m = e.get("machine")
  if m:
    lines += ["## Machine", "",
              f"- GPU: {te['gpu']}, driver {e['driver']} ({m['driver_model']}); power limit, "
              f"max SM / memory clocks: {m['gpu_limits']}",
              f"- PTX built by: nvcc {m['nvcc_on_path']} (the one on PATH; env: "
              f"{m['cuda_env'] or 'none'})",
              f"- crate {m['crate']}; candle-core {m['candle_core']}",
              f"- host: {m['os']}; CPU {m['cpu']}",
              f"- GPU state at the start: {e['gpu_start']}; at the end: {e['gpu_end']}",
              "- GPU state before each timed process:"]
    lines += [f"  - round {s['round'] + 1}, {s['side']}: {s['gpu']}" for s in rep["gpu_states"]]
    lines += [""]
  lines += ["## SDPA backends forced on these inputs (fwd + bwd)", ""]
  lines += [f"- `{k}`: {v}" for k, v in rep["backends"].items()]
  lines += ["", "## Accuracy against fp64 (normwise relative error; max-abs in brackets)", "",
            "| candidate | o | dq | dk | dv | 2 runs bitwise equal |", "|---|---|---|---|---|---|"]
  lines[-2:-2] = ["Two equal runs are an observation, not a guarantee: PyTorch's memory-efficient "
                  "backward adds dQ across key splits in ARRIVAL order (`kernel_backward.h`, "
                  "`AtomicLock`); ours adds them in key-block order.", ""]
  for name, s in rep["accuracy"].items():
    cells = [f"{s[p]['rel_fro']:.2e} [{s[p]['max_abs']:.1e}]" for p in PARTS]
    det = rep["deterministic"].get(name, "—")
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {det} |")
  lines += ["", "## Wall time, ms per call: median of round medians [min–max over rounds]", "",
            "| candidate | fwd | train | net = train − head |", "|---|---|---|---|"]
  for name in names:
    w = rep["wall"][name]
    lines.append(f"| {name} | " + " | ".join(fmt_range(w[p]) for p in PHASES) + " |")
  for side in ("torch", "candle"):
    lines.append(f"| ({side} head) | | {fmt_range(rep['wall'][f'{side}_head']['head'])} | |")
  ours = "candle_fused_qkv"
  lines += ["", f"Decision rule against `{ours}` (worst round of one < best round of the "
            "other):", ""]
  for name in names:
    if name != ours:
      cells = [f"{p}: {verdict(ours, rep['wall'][ours][p], name, rep['wall'][name][p])}"
               for p in PHASES]
      lines.append(f"- vs `{name}` — " + "; ".join(cells))
  if rep["kernels"]:
    lines += ["", "## Kernel time (nsys), ms per call", "",
              "| candidate | fwd | train | net = train − head | irregular kernels |",
              "|---|---|---|---|---|"]
    k = rep["kernels"]
    for name in names:
      side = "torch" if name.startswith("torch") else "candle"
      f, t = k[f"{side}:{name}:fwd"], k[f"{side}:{name}:train"]
      net = t["ms"] - k[f"{side}:head:head"]["ms"]
      irr = sorted(set(f["irregular"]) | set(t["irregular"]))
      lines.append(f"| {name} | {f['ms']:.3f} | {t['ms']:.3f} | {net:.3f} | "
                   f"{', '.join(irr) or '—'} |")
    lines += ["", "Kernels per train call (launches per call, µs per call):", ""]
    for name in names:
      side = "torch" if name.startswith("torch") else "candle"
      ks = sorted(k[f"{side}:{name}:train"]["kernels"], key=lambda x: -x["us_per_call"])
      lines.append(f"- `{name}`: " + "; ".join(
        f"{x['name'][:70]} ×{x['per_call']:g} {x['us_per_call']:.0f}" for x in ks))
  return "\n".join(lines) + "\n"


def main() -> None:
  """`run` (the benchmark) or `side` (the PyTorch process `run` launches)."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("command", choices=["run", "side"])
  p.add_argument("mode", nargs="?", choices=["accuracy", "time", "nsys"], help="side only")
  p.add_argument("--batch", type=int, default=64)
  p.add_argument("--heads", type=int, default=6)
  p.add_argument("--seq", type=int, default=240)
  p.add_argument("--causal", action="store_true")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--rounds", type=int, default=3)
  p.add_argument("--samples", type=int, default=15)
  p.add_argument("--calls", type=int, default=10, help="back-to-back calls per sample")
  p.add_argument("--warmup", type=int, default=10)
  p.add_argument("--nsys-calls", type=int, default=20)
  p.add_argument("--nsys", type=str, default=None, help="path to nsys")
  p.add_argument("--no-nsys", action="store_true")
  p.add_argument("--skip-build", action="store_true")
  p.add_argument("--out", type=Path, default=None)
  # The side process's arguments (the Rust side takes the same).
  p.add_argument("--inputs", type=Path)
  p.add_argument("--round", type=int, default=0)
  p.add_argument("--only", type=str)
  p.add_argument("--phase", type=str)
  a = p.parse_args()
  sys.stdout.reconfigure(encoding="utf-8")  # the report's "−", "µ", "×" on a cp1252 console
  if a.command == "side":
    torch_side(a)
  else:
    run(a)


if __name__ == "__main__":
  main()
