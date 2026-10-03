#!/usr/bin/env bash
# CI and the release's `verify` job, run locally BEFORE pushing: a dry run of both workflows.
#
# Usage: scripts/ci-local.sh [--no-cuda]
#
# Runs, verbatim, every command of .github/workflows/ci.yml (on the MSRV, 1.88, and on stable, as
# its matrix does) and of publish.yml's `verify` job, then what GitHub's runners cannot run:
#   * clippy and the tests with `--features cuda`, on both toolchains (needs the CUDA toolkit and a
#     GPU; the GPU part is the parity and determinism tests, under a minute; `--no-cuda` skips it);
#   * `cargo publish --dry-run` (everything but the upload; skipped while Cargo.toml says
#     `publish = false`, which cargo refuses even for a dry run).
# Every step runs, failed or not, and a summary lists them; the exit code is 1 if any failed. Each
# step's output is in target/ci-local/<n>-<name>.log.
#
# It cannot drift from the workflows unnoticed: before anything runs, every command line of their
# `run:` blocks must appear VERBATIM in this file, either as a step below or in CI_ONLY (the lines
# that only make sense on GitHub). A workflow edit without the matching edit here fails at once.
#
# What it does not reproduce: GitHub's Ubuntu runner (this runs wherever you are, Git Bash on
# Windows included) and the `uses:` steps. cargo-deny is one of those: ci.yml's `deny` job runs
# `EmbarkStudios/cargo-deny-action` with `command: check`, so the step here is `cargo deny check`.
#
# `set -e` is deliberately off: a failing step is recorded and the next one runs, as CI's jobs do.
set -uo pipefail
# This file, by absolute path: the drift check reads it as text.
self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$self")/.." || exit 1

cuda=1
case "${1:-}" in
  "") ;;
  --no-cuda) cuda=0 ;;
  *) echo "usage: scripts/ci-local.sh [--no-cuda]" >&2; exit 2 ;;
esac

# The lines of the workflows' `run:` blocks that only make sense on GitHub (the release job, the
# real publish), listed so that the drift check accounts for them.
# shellcheck disable=SC2016,SC2034  # literal workflow text: read by the drift check, never expanded
CI_ONLY='
cargo publish --locked --no-verify
cargo publish --locked --no-verify --dry-run
set -euo pipefail
scripts/release-notes.sh "${GITHUB_REF_NAME#v}" > release-body.md
gh release create "$GITHUB_REF_NAME" --title "$GITHUB_REF_NAME" \
  --notes-file release-body.md --verify-tag
'

# ---- 1. The drift check ---------------------------------------------------------------------
# Print every command line of a workflow's `run:` values: one-line `run: <cmd>`, and each line of
# a `run: |` block (its indentation removed; blank lines and comments skipped).
run_lines() {
  awk '
    block {
      if ($0 ~ /^[[:space:]]*$/) next
      match($0, /^ */)
      if (ind < 0) ind = RLENGTH
      if (RLENGTH >= ind) { line = substr($0, ind + 1); if (line !~ /^#/) print line; next }
      block = 0
    }
    /^ *run: *\|/ { block = 1; ind = -1; next }
    /^ *run: /    { sub(/^ *run: */, ""); print }
  ' "$1"
}

drift=0
for wf in .github/workflows/ci.yml .github/workflows/publish.yml; do
  while IFS= read -r line; do
    if ! grep -qF -- "$line" "$self"; then
      echo "DRIFT: $wf runs a command this script does not: $line" >&2
      drift=1
    fi
  done < <(run_lines "$wf")
done
if [ "$drift" -ne 0 ]; then
  echo "Add the command above as a step (or to CI_ONLY), then rerun." >&2
  exit 1
fi

# ---- 2. The environment the workflows assume -------------------------------------------------
# On Windows `python3` is often the Microsoft Store stub: fall back to `python`.
if ! python3 --version > /dev/null 2>&1; then
  python3() { python "$@"; }
fi
# The tag a release would carry: Cargo.toml's version.
version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)"
export GITHUB_REF_NAME="v$version"
# CI packages a clean checkout; a dirty tree here is allowed, and said.
allow_dirty=""
if [ -n "$(git status --porcelain)" ]; then
  allow_dirty="--allow-dirty"
  echo "note: uncommitted changes -- package and publish dry run use --allow-dirty (CI's tree is clean)"
fi

# ---- 3. The steps ----------------------------------------------------------------------------
logs=target/ci-local
rm -rf "$logs" && mkdir -p "$logs"
n=0
summary=()
failed=0

# step <name> <toolchain> <command...>: run the command (eval'd, so it reads exactly as in the
# workflow) under that toolchain, log it, and record PASS / FAIL with its duration.
step() {
  local name="$1" toolchain="$2"
  shift 2
  n=$((n + 1))
  local log
  log="$logs/$(printf '%02d' "$n")-$(echo "$name" | tr -cs 'A-Za-z0-9.' '-' | sed 's/-$//').log"
  printf '[%2d] %-52s ' "$n" "$name"
  local start=$SECONDS
  if (export RUSTUP_TOOLCHAIN="$toolchain"; eval "$*") > "$log" 2>&1; then
    summary+=("PASS  $name ($((SECONDS - start)) s)")
    echo "PASS ($((SECONDS - start)) s)"
  else
    summary+=("FAIL  $name ($((SECONDS - start)) s) -> $log")
    echo "FAIL ($((SECONDS - start)) s) -> $log"
    tail -n 15 "$log" | sed 's/^/      /'
    failed=1
  fi
}

skip() {
  summary+=("SKIP  $1 ($2)")
  printf '[--] %-52s SKIP (%s)\n' "$1" "$2"
}

# ci.yml, job `check`, matrix MSRV (1.88) and stable.
for tc in 1.88 stable; do
  step "fmt [$tc]" "$tc" 'cargo fmt --check'
  step "clippy, CPU path [$tc]" "$tc" 'cargo clippy --locked --all-targets -- -D warnings'
  step "tests, CPU path [$tc]" "$tc" 'cargo test --locked'
  step "tutorial, CPU path [$tc]" "$tc" 'cargo run --locked --example tutorial'
  step "tutorial blocks match the example [$tc]" "$tc" '
python3 -m doctest scripts/check-tutorial.py
python3 scripts/check-tutorial.py'
done

# ci.yml, jobs `docs` and `deny` (stable).
step "rustdoc, private items" stable \
  'RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items'
step "cargo-deny" stable 'cargo deny check'

# publish.yml, job `verify` (stable): its fmt / clippy / tests / tutorial / rustdoc steps are the
# ones above; these two are its own.
# The block is publish.yml's, character for character (a quoted heredoc expands nothing).
read -r -d '' version_check << 'EOF'
version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)"
if [ "${DRY_RUN:-}" != "true" ] && [ "${GITHUB_REF_NAME#v}" != "$version" ]; then
  echo "::error::tag ${GITHUB_REF_NAME} does not name the version in Cargo.toml, ${version}" >&2
  exit 1
fi
scripts/release-notes.sh "$version" > /dev/null
EOF
step "tag, version and CHANGELOG section" stable "$version_check"
step "package" stable "cargo package --locked $allow_dirty"

# What GitHub's runners cannot run.
if [ "$cuda" -eq 1 ]; then
  for tc in 1.88 stable; do
    step "clippy, CUDA path [$tc]" "$tc" 'cargo clippy --locked --all-targets --features cuda -- -D warnings'
    step "tests, CUDA path (GPU) [$tc]" "$tc" 'cargo test --locked --features cuda'
  done
else
  skip "clippy + tests, CUDA path" "--no-cuda"
fi
if grep -q '^publish = false' Cargo.toml; then
  skip "publish dry run" "Cargo.toml says publish = false"
else
  step "publish dry run" stable "cargo publish --locked --dry-run $allow_dirty"
fi

# ---- 4. The summary --------------------------------------------------------------------------
echo
printf '%s\n' "${summary[@]}"
if [ "$failed" -ne 0 ]; then
  echo "ci-local: FAILED (logs in $logs/)"
  exit 1
fi
echo "ci-local: all steps passed (v$version)"
