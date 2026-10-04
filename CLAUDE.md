# Claude Code Instructions

## Coding Conventions

Always apply the rules in `CONVENTIONS.md` to all code changes: Grit for the Rust, Grit-FA for the
CUDA kernels, their launches and every published claim. Every annotation pattern, doc-comment rule
and style rule in that file is mandatory.

Every `.rs` and `.cu` file starts with `// SPDX-License-Identifier: MIT OR Apache-2.0` as its first
line.

## Version Control

- Commit directly to `main`; create a branch only when the user asks for one (e.g. for a PR).
- Commit and push **only when the user asks**.
- A pushed tag is never moved or rewritten. A failed rehearsal tag stays; the next one is `-rc.N+1`.

## Pre-commit Checks

Before every commit, run and fix any issues from:
1. `cargo fmt`
2. `cargo clippy --locked --all-targets -- -D warnings`, and the same with `--features cuda`
3. `cargo test --locked`, and `cargo test --locked --features cuda` when a kernel, a launch or
   `src/cuda.rs` changed (GPU, under a minute: CUDA vs CPU within the measured band, and bitwise
   reruns of the backward)
4. If the commit touches a `///` or `//!` comment: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
   --document-private-items`, with and without `--features cuda`
5. Update `CHANGELOG.md`: a bullet under `## [Unreleased]` for any user-visible change, in the
   [Keep a Changelog](https://keepachangelog.com/) categories (Added, Changed, Fixed, Removed)

`scripts/ci-local.sh` runs all of this, on both toolchains (MSRV 1.88 and stable), plus everything
the CI and the release's `verify` job run — see [Releasing](#releasing). The MSRV run is not
optional: clippy 1.88 and stable do not lint identically (on 2026-10-03, 1.88 alone flagged
`float_cmp` in the tests and `similar_names` in an example).

## Kernel Changes and Claims

- **C++ first.** Before writing or changing a kernel, read the reference it follows (`PyTorch`'s
  `kernel_backward.h`, FlashAttention-2's `flash_fwd_kernel.h` / `flash_bwd_kernel.h`, ATen's
  launch path) and cite it in a `// REFERENCE:` comment.
- **A speed or accuracy claim needs its report**: `python bench/compare.py run` on the card, the
  report committed under `bench/results/<date>-<card>-<what>.md`, the claim citing it. Protocol in
  `CONVENTIONS.md` § *Performance claims*: alternate the candidates in one session, discard a fresh
  binary's first run (PTX JIT), never compare across hours or boxes, and make nsys exports
  overwrite (`--force-overwrite=true`).
- **A determinism claim needs the bitwise test**, `tests/parity.rs::cuda_backward_is_deterministic`
  (`to_bits()` equality), never a tolerance.

## Releasing

Cutting a release (a `vX.Y.Z` tag → crates.io through Trusted Publishing). Adapted from anamnesis's
checklist, with candle-mi's order rule — **verify, then bump, then dry-run** — and a rehearsal of the
publish workflow itself. This flow shipped 0.2.0 on 2026-10-03; its first rehearsal (`v0.2.0-rc.1`)
caught a script committed without its executable bit, which would have failed the real release.

1. **Verify green first, before touching any version string**: `scripts/ci-local.sh` (see
   [Shell Environment](#shell-environment) for how to start it). It runs every command of `ci.yml`
   and of `publish.yml`'s `verify` job, verbatim, on 1.88 and stable, plus clippy and the tests with
   `--features cuda` (local GPU). It refuses to run if a workflow has a command it does not have,
   or if a `scripts/*.sh` is not executable in git. If a kernel changed since the last release,
   also refresh the reports the README cites (`bench/compare.py`, see above).
2. **Bump, in one commit**: `version` in `Cargo.toml`; `cargo check` to update `Cargo.lock` (every
   workflow command runs `--locked`); rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` with a
   fresh empty `## [Unreleased]` above it; update any version the README states. Commit as
   `bump version to vX.Y.Z, update changelog date`.
3. **Then `scripts/ci-local.sh` again.** Against the bumped version it now ends with `cargo publish
   --dry-run`, which packages and builds exactly what will ship: missing metadata, `exclude`
   rules, the 10 MiB cap. It does **not** catch a version already on crates.io: it stops before the
   upload, and the registry rejects a taken version only at upload (checked 2026-10-03: the dry
   run passed for 0.2.0 after 0.2.0 was published). The bump in step 2 is what prevents that.
4. **Push `main`, wait for CI to go GREEN.**
5. **Rehearse the publish workflow** on that commit:
   `git tag vX.Y.Z-rc.N; git push origin vX.Y.Z-rc.N`. Pushing a hyphenated tag starts nothing (the
   trigger is `v*` minus `v*-*`). Then **Actions → Publish to crates.io → Run workflow → Tags →
   `vX.Y.Z-rc.N`**, with **Dry run** ticked (the default), or
   `gh workflow run publish.yml --ref vX.Y.Z-rc.N -f dry_run=true`. The user approves the `release`
   job. Green means: every `verify` step passes on the Linux runner; "Authenticate to crates.io"
   logs `Retrieved token successfully` (crates.io accepted the repository, `publish.yml` and the
   `release` environment); the dry-run step ends with `aborting upload due to dry run`; the post
   step logs `Token revoked successfully`; the real publish step and the GitHub Release are
   skipped. If it fails: fix on `main`, push, CI green, rehearse again with `-rc.N+1`.
6. **Tag the rehearsed commit**: `git tag vX.Y.Z; git push origin vX.Y.Z`. `verify` refuses a tag
   that does not name `Cargo.toml`'s version, or a version with no CHANGELOG section.
7. **The user approves the release** (the run → **Review deployments** → `release`). Claude never
   approves a deployment, even when the CLI's account could. `publish` runs
   `cargo publish --locked --no-verify` (it compiles nothing: no dependency's build script ever runs
   in the job that can mint a token); `release` then creates the GitHub Release from the CHANGELOG
   section.
8. **Check all three**: crates.io lists the version; docs.rs built it
   (`https://docs.rs/crate/candle-fused-attn/X.Y.Z/status.json` reads `"doc_status":true`, a few
   minutes after the publish); the GitHub Release appeared with the right notes. If `release`
   failed after `publish` succeeded, **do not re-run the workflow** (the publish would fail on the
   taken version and hide the real error): fix the workflow for next time, and create the Release
   by hand from a checkout of the tag:
   ```bash
   scripts/release-notes.sh X.Y.Z > release-body.md
   gh release create vX.Y.Z --title vX.Y.Z --notes-file release-body.md --verify-tag
   ```

## Traps Already Met

- **Line endings.** `core.autocrlf` is on: files are CRLF on disk, LF in the repository. Edit with
  tools that keep a file's endings; `.gitattributes` forces LF for `*.sh`, which Linux runs.
- **Executable bits.** Windows does not record them, and Git Bash runs any file with a shebang:
  `git update-index --chmod=+x` for every script a workflow runs directly (`ci-local.sh` checks).
- **Synced sources keep old modification times** (`tar`), so cargo does not rebuild: `touch` them.
- **A fresh binary's first run includes kernel JIT**: discard it in any timing.
- **Keep the reference binary before changing the code** (`cp` it out of `target/`): the next
  build overwrites it, and an A/B needs it. An md5 identifies a binary only until it is relinked
  (the Windows linker stamps a time in every executable); a rebuilt reference is re-checked with
  `bench/bitwise.py`.

## Shell Environment

The user runs PowerShell on Windows. Use PowerShell syntax for all suggested commands:
- Use `$env:VAR="value";` instead of `VAR=value` for environment variables
- Use semicolons to chain commands, not `&&`
- Use forward slashes in paths when running Rust/cargo commands

The scripts under `scripts/` are bash. **From PowerShell, a bare `bash` is WSL's**
(`C:\windows\system32\bash.exe`), with its own toolchains and no CUDA: start them with Git Bash,
```powershell
& "C:\Program Files\Git\bin\bash.exe" scripts/ci-local.sh
```
or run `bash scripts/ci-local.sh` from a Git Bash terminal.
