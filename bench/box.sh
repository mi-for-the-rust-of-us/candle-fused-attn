#!/usr/bin/env bash
# The crate's measurements on one machine, rented or local (devlog FA12, FA17): a released
# version against this checkout, on that card, in one session.
#
# Usage, from a clone of the repository (with its tags):
#   [BASE=v0.3.0] [ROUNDS=6] bash bench/box.sh <label>
#   1. records the machine (GPU, driver, nvcc, CPU, commit);
#   2. builds the `compare` example of BASE (a git worktree of the tag) and of this checkout;
#   3. the bitwise gate: both builds' outputs over bench/bitwise.py's shapes, bit for bit;
#   4. the CUDA tests (parity, bitwise reruns);
#   5. bench/compare.py: this checkout against PyTorch (its run also warms this build);
#   6. bench/ab.py: BASE against this checkout, alternated (ROUNDS rounds), forward and training.
# Everything lands in target/box/<label>/ (git-ignored), for the operator to bring home.
# GPU time on an RTX 5060 Ti: ~5 min, after the two builds on the CPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
label="${1:?usage: [BASE=v0.3.0] [ROUNDS=6] bash bench/box.sh <label>}"
BASE="${BASE:-v0.3.0}"  # the baseline release (FA12 measured v0.2.0)
ROUNDS="${ROUNDS:-6}"   # ab.py rounds: FA14-FA16's protocol
out="target/box/$label"
mkdir -p "$out"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$out/box.log"; }

# --- The tools the steps need, found by running them (never by `command -v` alone) ---------------
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
command -v cargo > /dev/null || { echo "cargo missing: install rustup first" >&2; exit 1; }
PY=""
for p in /venv/main/bin/python python3 python py; do
  if "$p" -c "import torch, safetensors" > /dev/null 2>&1; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "no python with torch + safetensors" >&2; exit 1; }
NSYS=""
for n in "$(command -v nsys || true)" /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys \
         "/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"; do
  if [ -n "$n" ] && "$n" --version > /dev/null 2>&1; then NSYS="$n"; break; fi
done
[ -n "$NSYS" ] || { echo "no working nsys (needed by bench/ab.py)" >&2; exit 1; }
# The executable suffix, set by `if` (a failing `[ ]` inside `$( )` in an assignment exits under
# `set -e`: that is how a first version died silently on Linux).
SUF=""
if [ "${OS:-}" = "Windows_NT" ]; then SUF=".exe"; fi

# --- 1. The machine -------------------------------------------------------------------------------
{
  echo "label: $label"; echo "date: $(date -Iseconds)"
  echo "commit: $(git describe --always --dirty --tags)"; echo "base: $BASE"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm \
    --format=csv,noheader
  nvcc --version | tail -1
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  "$NSYS" --version
  (grep -m1 "model name" /proc/cpuinfo 2>/dev/null || echo "cpu: ${PROCESSOR_IDENTIFIER:-?}")
} > "$out/machine.txt" 2>&1
log "machine: $(head -4 "$out/machine.txt" | tail -2 | tr '\n' ' ')"

# --- 2. The two builds ----------------------------------------------------------------------------
git rev-parse -q --verify "$BASE" > /dev/null || git fetch --tags -q
src="target/base-src-$BASE"
if [ ! -d "$src" ]; then git worktree add -q "$src" "$BASE"; fi
log "build $BASE (CPU)"
(cd "$src" && cargo build -q --release --features cuda --example compare)
cp -p "$src/target/release/examples/compare$SUF" "$out/compare-base$SUF"
log "build this checkout (CPU)"
cargo build -q --release --features cuda --example compare
cp -p "target/release/examples/compare$SUF" "$out/compare-head$SUF"
A="$out/compare-base$SUF"
B="$out/compare-head$SUF"

# --- 3. The bitwise gate --------------------------------------------------------------------------
log "bitwise gate (GPU)"
"$PY" bench/bitwise.py dump "$A" "$out/bitwise-base" > /dev/null
"$PY" bench/bitwise.py dump "$B" "$out/bitwise-head" > /dev/null
"$PY" bench/bitwise.py diff "$out/bitwise-base" "$out/bitwise-head" > "$out/bitwise.txt" \
  || { log "BITWISE GATE FAILED -- see $out/bitwise.txt"; exit 1; }
log "$(tail -1 "$out/bitwise.txt")"

# --- 4. The CUDA tests ----------------------------------------------------------------------------
log "cargo test --features cuda (GPU)"
cargo test -q --release --features cuda > "$out/tests.txt" 2>&1 || { log "TESTS FAILED"; exit 1; }
log "$(grep -c 'test result: ok' "$out/tests.txt") test binaries ok"

# --- 5. Against PyTorch ---------------------------------------------------------------------------
log "compare.py (GPU, ~2.5 min)"
"$PY" bench/compare.py run --skip-build --nsys "$NSYS" --out "$out/compare" > "$out/compare.log" 2>&1
log "compare.py done"

# --- 6. BASE against this checkout, alternated ----------------------------------------------------
log "ab.py $BASE vs this checkout, $ROUNDS rounds (GPU, ~20 s per round)"
"$PY" bench/ab.py "$A" "$B" --inputs "$out/compare/inputs.safetensors" --nsys "$NSYS" \
  --rounds "$ROUNDS" --out "$out/ab" > "$out/ab.txt" 2>&1
tail -12 "$out/ab.txt" | tee -a "$out/box.log"
log "done: $out"
