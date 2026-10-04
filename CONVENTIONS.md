# candle-fused-attn Coding Conventions (Grit + Grit-FA Extensions)

This document describes the [Amphigraphic coding](https://github.com/PCfVW/Amphigraphic-Strict)
conventions used in `candle-fused-attn`. It is a superset of the
[Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit)
base, with **Grit-FA** extensions for what no other crate of the organisation has: CUDA C++
kernels (`kernels/*.cu`), their launches from Rust, a determinism contract, and published
accuracy and performance numbers.

## Provenance

The organisation's `CONVENTIONS.md` files share one core but have drifted, each revised after
its own audits. This file takes **anamnesis** as its spine — the most revised and the only one
with the annotation grammar, the lint floor, the full MSRV guard and `MEASURED-REVERT` — and
takes five sections from where they were last deliberately hardened (compared section by
section on 2026-10-03):

| Section | Taken from | Why |
|---|---|---|
| Annotation grammar, lint floor, `INDEX`, MSRV lint guard, `MEASURED-REVERT` | anamnesis | the most complete versions |
| `SAFETY` content rule, FFI patterns, test-module lint allowances | hypomnesis | the content rule (what a call needs, how this site has it) and the 2026-09-26 audit |
| `#[non_exhaustive]` (structs), `# Shapes`, `if let` vs `match`, `# Errors`, pass by value | candle-mi | strict supersets (the 2026-09-20 rework) |
| Backtick hygiene | candle-mi | its "commonly missed words" list, adapted to this crate's vocabulary |
| Field-level docs | candle-mi + hypomnesis | `pub(crate)` fields (candle-mi); units and `None` semantics (hypomnesis) |
| `CAST` | hypomnesis | prefer `From` / `TryFrom`; `as` only as deliberate, in-range intent |
| Intra-doc link safety | hf-fetch-model | adds `redundant_explicit_links`, which CI's `cargo doc -D warnings` enforces |
| Error message wording | written for this crate | the org's two-pattern rule assumes a crate error enum; this crate returns `candle_core::Error::Msg` |
| Python (`bench/compare.py`) | askesis `CONVENTIONS.md` | the organisation's Python sibling |

The doc-section headings are named `Errors Doc Section` / `Shapes Doc Section` rather than
`` `# Errors` Doc Section ``: the leading `#` gives GitHub an anchor starting with a hyphen, so
the sister files' `(#errors-doc-section)` links are broken (a follow-up for those repositories).

Deliberately omitted: anamnesis's untrusted-input parsing and SIMD-loop sections, candle-mi's
weight loading and hook system, and the HashMap grouping idiom (candle-mi, anamnesis,
hf-fetch-model) — this crate parses no files, has no SIMD Rust loop, no hooks, and batches
nothing by key.

---

## Trigger Checklist

**Before writing any line of code, check which triggers apply.**

| You are about to... | Check these rules |
|---|---|
| Write a `///` or `//!` comment | [Backtick hygiene](#backtick-hygiene), [field-level docs](#field-level-docs), [intra-doc link safety](#intra-doc-link-safety) |
| Write a `pub fn` taking or returning a `Tensor` | [`# Shapes`](#shapes-doc-section) |
| Write a `pub fn` or `pub const fn` | [`const fn`](#const-fn), [`#[must_use]`](#must_use-policy), [pass by value](#pass-by-value-vs-reference) |
| Write a `pub fn` returning `Result<T>` | [`# Errors`](#errors-doc-section) |
| Write a `pub enum` or `pub struct` | [`#[non_exhaustive]`](#non_exhaustive-policy) or [`// EXHAUSTIVE:`](#exhaustive-annotation) |
| Write an `as` cast | [`// CAST:`](#cast-annotation) |
| Write `slice[i]` or `slice[a..b]` | [`// INDEX:`](#index-annotation) |
| Write `.as_str()`, `.to_owned()` | [`// BORROW:`](#borrow-annotation) |
| Write an `unsafe` block | [`// SAFETY:`](#safety-annotation), [scope](#accepted-unsafe-scope), [kernel launches](#ffi-pattern-kernel-launches) |
| Write `Box<dyn T>` or `&dyn T` | [`// TRAIT_OBJECT:`](#trait_object-annotation) |
| Write a `match` or `if let` | [`if let` vs `match`](#if-let-vs-match), [`// EXPLICIT:`](#explicit-annotation) if no-op arm |
| Write an error string | [Error message wording](#error-message-wording) |
| Add `#[allow(clippy::...)]` | [Annotation grammar](#annotation-grammar); for a newer lint, [MSRV lint guard](#msrv-lint-guard) |
| Suppress a lint because its advice measured slower | [`// MEASURED-REVERT:`](#measured-revert-annotation) |
| Write or change a CUDA kernel | [CUDA kernels](#grit-fa-when-writing-cuda-kernels): `REFERENCE`, `ORDER`, `DETERMINISM` |
| Change a launch constant (threads, rows, tile, shared memory) | [`// TWIN:`](#twin-annotation) on both sides |
| Keep state from the forward for the backward | [The saved-state pattern](#grit-fa-the-saved-state-pattern) |
| Claim determinism, accuracy or speed (code comment, README, CHANGELOG, commit) | [Claims and their evidence](#grit-fa-claims-and-their-evidence) |
| Edit `bench/compare.py` | [Python](#python) |

---

## Annotation Grammar

The `// CAST:`, `// INDEX:`, `// SAFETY:`, `// EXHAUSTIVE:` and `// MEASURED-REVERT:` comments
pair with `#[allow(...)]` attributes that suppress the crate's [lint floor](#lint-floor): the
comment explains *why*, the attribute is what makes the code build under `#![deny(warnings)]`.
`// EXPLICIT:` pairs with one when it justifies a suppression, and `// BORROW:` and
`// TRAIT_OBJECT:` are documentation-only, as the table below says. The Grit-FA kernel annotations
(`REFERENCE`, `TWIN`, `ORDER`, `DETERMINISM`) live in `.cu` files and in the Rust constants
mirroring them; they are documentation-only and grep-able.

### Comment ↔ attribute pairing

| Annotation | Companion attribute(s) |
|---|---|
| `// CAST:` | `#[allow(clippy::as_conversions)]` plus, as the cast requires, `cast_precision_loss`, `cast_possible_truncation`, `cast_possible_wrap` |
| `// INDEX:` | `#[allow(clippy::indexing_slicing)]` |
| `// EXHAUSTIVE:` | `#[allow(clippy::wildcard_enum_match_arm)]` or `#[allow(clippy::exhaustive_enums)]` |
| `// SAFETY:` | `#[allow(unsafe_code)]` (module, item or block scope) |
| `// MEASURED-REVERT:` | `#[allow(clippy::<the lint whose advice was measured slower>)]` |
| `// EXPLICIT:` | the `#[allow(...)]` it justifies, when one is needed (e.g. `clippy::many_single_char_names` for the papers' notation) |
| `// BORROW:`, `// TRAIT_OBJECT:`, `REFERENCE`, `TWIN`, `ORDER`, `DETERMINISM` | none — documentation-only |

### Per-site vs block-scope

**In library code** (`src/`): a single suppressed line takes the attribute immediately above
it; a loop or function where every site shares one invariant takes one block-scope `#[allow]`
and one comment. The per-site rules ([`CAST`](#cast-annotation), [`INDEX`](#index-annotation))
apply there in full.

**In tests, examples and benchmarks** (`tests/`, `examples/`, `bench/`): a file where the same
reason holds everywhere (a test's `unwrap`, a benchmark's casts) may instead carry one inner
`#![allow(...)]` at its top with its `// EXPLICIT:` reason; per-site annotations are then not
required in that file.

**At the library's crate root** (`src/lib.rs`): an `#![allow]` is a *policy*, because it also
hides code written later. It is allowed only with its reasoning stated beside it and revisited
rather than inherited — today, `clippy::many_single_char_names`, for the attention papers'
notation (`b, h, s, d`, `q, k, v`).

### Lint Floor

The annotations exist because the crate denies the underlying lints. `src/lib.rs` carries
`#![deny(warnings)]`, which promotes every `warn` below to a hard error; `Cargo.toml`'s
`[lints]` sets the levels. Keep this table in sync with it:

| Lint | Level | How to satisfy |
|---|---|---|
| `unsafe_code` | `deny` | only in the [accepted scope](#accepted-unsafe-scope), with `// SAFETY:` |
| `elided_lifetimes_in_paths` | `deny` | write `Foo<'_>` |
| `missing_docs` | `warn` | document every public item |
| `unwrap_used`, `expect_used`, `panic` | `deny` | never in library code |
| `indexing_slicing` | `deny` | `// INDEX:` + `#[allow(clippy::indexing_slicing)]` |
| `wildcard_enum_match_arm`, `match_wildcard_for_single_variants` | `deny` | match every variant, or `// EXHAUSTIVE:` |
| `as_conversions`, `cast_possible_truncation` | `warn` | `// CAST:` + the matching `#[allow]` |
| `missing_docs_in_private_items` | `warn` | document private items too |
| `missing_errors_doc` | `warn` | write an `# Errors` section |
| `must_use_candidate` | `warn` | `#[must_use]` |
| `pedantic`, `nursery` (priority -1) | `warn` | per-lint allow with an explanatory comment |
| `module_name_repetitions` | `allow` | (`cuda::cuda_of` and the like read naturally) |

---

## When Writing Doc Comments (`///`, `//!`)

### Backtick Hygiene

Wrap in backticks every identifier, type, path, file name, crate name and acronym that names a
type or tool: `Tensor`, `CustomOp3`, `Option<f32>`, `candle_core`, `cudarc`, `fused_attn.cu`.
**Commonly missed words here** (backtick these every time): `PyTorch`, `SDPA`, `CUTLASS`,
`FlashAttention-2`, `cuBLAS`, `nvcc`, `PTX`, `nsys`, `F32`, `BF16`, `F16`, `FFMA`, `TF32`.

> ✅ ``/// The backward adds dQ in key-block order, where `PyTorch`'s adds it in arrival order.``
> ❌ `/// The backward adds dQ in key-block order, where PyTorch's adds it in arrival order.`

### Intra-Doc Link Safety

Rustdoc intra-doc links must resolve under every feature combination (enforced by
`#![deny(warnings)]` → `rustdoc::broken_intra_doc_links`, and by CI's
`RUSTDOCFLAGS="-D warnings"`). Two patterns to watch:

1. **Feature-gated items** — the `cuda` module and its items are absent without the `cuda`
   feature, and docs.rs builds without it. Use plain backtick text, not link syntax:

   > ✅ `` /// The launches live in `cuda::backward` (requires the `cuda` feature). ``
   > ❌ `` /// The launches live in [`cuda::backward`](crate::cuda::backward). ``

2. **Redundant explicit targets** — prefer the bare form `` [`fused_attention`] ``; an explicit
   `(crate::...)` target that rustdoc could resolve alone trips
   `rustdoc::redundant_explicit_links`. Use the explicit path only when the bare label is
   genuinely ambiguous.

### Field-Level Docs

Every field of a `pub` or `pub(crate)` struct carries a `///` doc stating:

1. what it holds,
2. its **unit or layout** where one applies (`[b, h, s]` floats, bytes, µs),
3. when it can be `None` or empty, and what that means.

Private fields need a `///` doc too: `missing_docs_in_private_items` is at `warn`, which
`#![deny(warnings)]` makes an error, and a `//` comment does not satisfy it.

> ```rust
> struct FusedAttentionQkv {
>     /// Attention heads; `head_dim` = qkv width / (3 · heads).
>     heads: usize,
>     /// The row log-sum-exp `L` ([b, h, s] f32), set by the forward for the backward;
>     /// `None` until the forward has run.
>     lse: Mutex<Option<Tensor>>,
> }
> ```

### Shapes Doc Section

Every public function that accepts or returns a `Tensor` documents its shapes:

    /// # Shapes
    /// - `qkv`: `[batch, seq, 3·heads·head_dim]` -- the fused projection
    /// - returns: `[batch, seq, heads·head_dim]` -- heads merged

Concrete dimension names (`batch`, `heads`, `seq`, `head_dim`), never `d0`/`d1`; batch first;
every tensor argument and the return value. State layout requirements (contiguous last
dimension, float4-aligned strides) next to the shape they constrain.

### Errors Doc Section

Every public fallible function (`-> Result<T>`) has an `# Errors` section, one bullet per
condition. This crate returns `candle_core::Error` (it has no error enum of its own), so each
bullet names the **condition**, not a variant:

    /// # Errors
    ///
    /// On a dtype other than f32, mismatched shapes or devices, or (CUDA) a head dimension
    /// other than [`CUDA_HEAD_DIM`].

---

## When Writing Function Signatures

### `const fn`

Constructors of plain data, accessors and pure arithmetic helpers are `const fn`. When in doubt,
annotate and let the compiler reject it — do not omit `const` preemptively.

### `#[must_use]` Policy

Every public function or method that returns a value and has no side effect is `#[must_use]`:
constructors, accessors, pure queries. Without it a caller can silently discard the value, which
for such a function is always a bug. `Result<T>` is already `#[must_use]` at the type level.

### Pass by Value vs Reference

| Type | Pass as |
|---|---|
| `Copy` and at most two words (`usize`, `f32`, `bool`, `(usize, usize)`) | by value |
| `Tensor`, `Layout`, storages, anything not mutated | `&T` |
| mutated in place | `&mut T` |

Never accept `&mut T` when the body never writes through it (`needless_pass_by_ref_mut`), nor
`&T` for a small `Copy` type (`trivially_copy_pass_by_ref`).

---

## When Writing Public Enums and Structs

### `#[non_exhaustive]` Policy

- Public enums that may gain variants: `#[non_exhaustive]`.
- Internal dispatch enums matched exhaustively by this crate:
  `#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: <reason>`.
- Public structs with public fields that may gain fields: `#[non_exhaustive]` — adding a field
  to one without it breaks external struct-literal construction. No lint enforces this; it is a
  review item. A marked struct must still be obtainable (`new`, `Default`, a builder), or it
  becomes unusable rather than merely closed.

(The public API is two functions and a constant today; the rule applies when that changes.)

---

## When Writing Expressions

### CAST Annotation

`// CAST: <from> → <to>, <reason>` — required on every `as` cast between numeric types. Prefer
`From`/`Into` for lossless conversions and `TryFrom` with `?` for fallible ones (the launchers'
`i32_of`, `u32_of`, `i64_of` are the house idiom for kernel arguments). Use `as` only when the
conversion is the deliberate intent and provably in range.

> Example: `// CAST: usize → f32, head_dim ≤ 64: exact in f32`

### INDEX Annotation

`// INDEX: <reason>` — required on every direct slice index (`slice[i]`, `slice[a..b]`). Direct
indexing panics on out-of-bounds; prefer `.get(i)` with `?`, a slice pattern
(`let &[sb, sh, ss, 1] = l.stride()`), or an iterator, unless the bound is provably valid and
indexing is clearly more readable.

> Example: `// INDEX: dims has rank 4, checked by dims4() above`

### BORROW Annotation

`// BORROW: <what is converted>` — required on explicit `.as_str()`, `.as_bytes()`,
`.to_owned()`, `.clone()` of a large value, when the reason is not obvious.

### TRAIT_OBJECT Annotation

`// TRAIT_OBJECT: <reason>` — required on `Box<dyn T>` / `&dyn T`, saying why static dispatch
does not fit.

### MEASURED-REVERT Annotation

`// MEASURED-REVERT: <lint>, <measured delta and direction>, <benchmark and shape>,
<significance and reproductions>, <instrument and card>, <what was actually measured>` —
required whenever a lint is suppressed **because taking its advice was measured to be slower**.
It records an experiment: the suggestion was applied, measured against a baseline, and reverted
on the numbers; a reader can re-run it. "Unmeasurable" is a legitimate verdict and must not be
written as "measured slow". When a lint is *inapplicable* rather than costly, say that instead
with a plain reason comment. A crate-wide `#![allow]` is never a `MEASURED-REVERT`.

---

## When Writing `unsafe`

`Cargo.toml` denies `unsafe_code` crate-wide. `deny` rather than `forbid`, because launching a
CUDA kernel is FFI: `unsafe` is intrinsic to the `cuda` feature. Every `unsafe` block is
**scoped, annotated and feature-gated**.

### SAFETY Annotation

`// SAFETY: <invariants>` — required on every `unsafe` block or `unsafe fn` (inline comment, not
a doc comment). It states:

1. what the call requires (a buffer of at least N elements, a live context, an alignment);
2. how this call site establishes it.

> ```rust
> // SAFETY: reads q, k, v through their (float4-aligned) strides inside their storages
> // (shape-checked by the caller), writes all of `o` and `lse`.
> unsafe { builder.launch(grid(dims, FWD_QUERIES, FWD_SMEM)?) }.w()?;
> ```

### Accepted `unsafe` scope

| Gate | Accepted scope | Where |
|---|---|---|
| `cuda` | kernel launches (`LaunchArgs::launch`) and uninitialised device allocations (`alloc`) | `src/cuda.rs` only — `#![allow(unsafe_code)]` at that module's top |
| `cuda` (example) | `cuProfilerStart` / `cuProfilerStop`, bracketing an nsys capture | `examples/compare.rs` |

Each accepted use satisfies all of:

1. it is concentrated in that one module, never scattered;
2. every block carries a `// SAFETY:` comment;
3. it is gated behind the `cuda` feature — a CPU build compiles no `unsafe` at all;
4. in the library, a non-`unsafe` path computes the same result: the CPU reference, against
   which the CUDA path is tested (`tests/parity.rs`). (The example's profiler calls have no
   such twin and need none: they compute nothing.)

Adding a use requires updating this table.

### FFI Pattern: Kernel Launches

- **The argument list mirrors the kernel signature, in order.** A launch pushes arguments
  positionally; one out of order is undefined behaviour the compiler cannot see. Keep the Rust
  builder calls in the order of the `extern "C"` parameters, grouped as they are there, and name
  the kernel in a comment at the launch so the two can be read side by side.
- **Uninitialised allocations only for buffers the kernel writes entirely.** `alloc` (not
  `alloc_zeros`) is acceptable when the `SAFETY:` comment states which kernel writes every
  element — and for a buffer written by one kernel and read by the next, which one.
- **Everything on candle's own stream.** Launch through `dev.cuda_stream()`, so ordering against
  candle's other kernels is the stream's and no synchronisation is needed.
- **Load once.** A module and its functions are loaded once per device (`FUNCTIONS` cache), the
  dynamic shared-memory attribute set at load time.

### MSRV Lint Guard

`src/lib.rs` carries `#![allow(unknown_lints)]` so that an `#[allow(clippy::newer_lint)]` does
not break the MSRV build: `#![deny(warnings)]` implies `deny(unknown_lints)`, and the MSRV
toolchain's clippy may not know lint names added later. Nothing to do when adding a new
`#[allow]`; if the MSRV is bumped, the guard stays as long as the MSRV trails the development
toolchain. The MSRV is 1.88, the organisation's leaf-library MSRV (anamnesis, hypomnesis).

### Test-Module Lint Allowances

A test file or `#[cfg(test)] mod tests` carries an `#[allow]` only for the lints its own code
triggers, each with its `// EXPLICIT:` reason — no blanket preamble. To check one, temporarily
change `#[allow]` to `#[expect]`: the compiler then reports any expectation that is
unfulfilled. Do it on the MSRV toolchain too; some lints fire on only one.

---

## When Writing Control Flow

### `if let` vs `match`

| Situation | Use |
|---|---|
| One pattern, the rest ignored | `if let` / `let ... else` |
| A boolean test of a pattern | `matches!` |
| Dispatch over every variant | `match` |

Never a `match` with a single non-`_` arm and a no-op `_ => {}` (`single_match`); never three
or more chained `if let … else if let …` where a `match` would be exhaustive.

### EXPLICIT Annotation

`// EXPLICIT: <reason>` — required when a match arm is intentionally a no-op, when an imperative
loop replaces an iterator chain for a stateful computation, or when a lint is allowed for a
reason that is a choice rather than a measurement (the attention papers' single-letter
notation `b, h, s, d`, `q, k, v`).

### EXHAUSTIVE Annotation

`// EXHAUSTIVE: <reason>` — required on `#[allow(clippy::exhaustive_enums)]`, and on
`#[allow(clippy::wildcard_enum_match_arm)]` when matching a foreign `#[non_exhaustive]` enum
(candle's `DType`, `Storage`) that cannot be matched exhaustively.

---

## When Writing Error Strings

### Error Message Wording

Errors are `candle_core::Error::Msg` strings that **start with an operation prefix**, so a
message read far from its source still says where it came from: `fused_attention_qkv:` for the
checks of that op alone, `fused_attention:` for everything shared by both ops (the launchers in
`src/cuda.rs`, the saved state, the common validation), which cannot know which op called them.

- **Validation failures** (dtype, shape, layout): `"<prefix>: <noun> <problem> (<context>)"`
  > `"fused_attention: the CUDA kernels take head_dim 64, got {d}"`
- **External failures** (driver, launch, module load): wrap with candle's `.w()?` so cudarc's
  own error is kept, and add context only when the call site is ambiguous.

---

## Grit-FA: When Writing CUDA Kernels

Rules for `kernels/*.cu` and for the Rust that launches them. The kernel file's style is its own
(2-space indentation, at most 100 columns, `// ` comments); these rules are about content.

### REFERENCE Annotation

Every kernel and every non-obvious algorithmic choice cites the C++ it follows or departs from:
`// REFERENCE: <upstream file> (<project> <version>) — <what is followed / why it departs>`.
This is the C++-first rule made checkable: the design was read from the reference before it was
written, and the next reader can check the transfer.

> ```c
> // REFERENCE: kernel_backward.h (PyTorch v2.10, mem_eff_attention) -- one pass per key block,
> // S and dP computed once; departs on dQ: added in key-block ORDER, not under an arrival-order
> // lock, so the backward is deterministic.
> ```

### TWIN Annotation

A constant that exists on both sides — threads per block, rows per block, the query tile, the
padded row stride, a shared-memory size — carries `// TWIN: <file>:<name>` on **both** sides, so
changing one side finds the other. (A kernel's *argument list* is not a constant: it is held by
[the launch rule](#ffi-pattern-kernel-launches), a comment at the launch naming its kernel.)

> ```rust
> /// Threads per block, as `NT` in the kernels.
> const THREADS: u32 = 256; // TWIN: kernels/fused_attn.cu:NT
> ```
> ```c
> #define NT 256  // TWIN: src/cuda.rs:THREADS
> ```

Shared-memory sizes are derived from the **named** tile constants in one expression on the Rust
side — `((2 * BWD_KEYS + 6 * QUERY_TILE) * LD + 4 * QUERY_TILE) * size_of::<f32>()`, each name a
`TWIN` — never typed as a bare byte count nor as bare literals.

### Launch bounds and shared memory

Every kernel declares `__launch_bounds__(<threads>)` matching its launch, and states its shared
memory in a comment: static or dynamic, the total in KB, and what it costs in blocks per SM
(the occupancy it buys or gives up). Dynamic shared memory above 48 KB requires the attribute
set at load time (see [FFI pattern: kernel launches](#ffi-pattern-kernel-launches)).

### Alignment preconditions

A vectorised access (`float4`) states its alignment precondition in the kernel, and the **Rust
side checks it before launching** (`float4_aligned`, falling back to a contiguous copy). A
kernel never assumes an alignment the launcher did not establish.

### ORDER Annotation

Every memory-ordering primitive that synchronises **across blocks** — `ld.acquire` /
`st.release`, `__threadfence()`, a spin-wait — carries `// ORDER: <what it publishes, to whom>`,
at both ends of the handshake. Within a block, `__syncthreads()` needs a comment only when it
guards a shared-memory reuse that is not adjacent.

### DETERMINISM Annotation

Every reduction across threads or blocks states its order: `// DETERMINISM: <fixed order>`.
**No floating-point atomics in a reduction**: an `atomicAdd` sum depends on scheduling and
breaks bit-reproducibility. When blocks must combine results, fix the order (turn counters in
index order, or partials reduced by a second pass in index order), and keep anything that varies
by device — SM count, occupancy, scheduling — out of the order. Here: each dQ tile is
`((p0 + p1) + p2) + …` over key blocks in index order, a function of the shape alone, whatever
the card or the schedule.

### Explicit rounding where bit-identity is claimed

When a kernel claims to reproduce another computation bit for bit (a fused kernel replacing a
composition), the arithmetic that must not be contracted uses explicitly rounded intrinsics
(`__fmul_rn`, `__fadd_rn`), and a test compares the two **bitwise** (see
[bit-identity claims](#bit-identity-claims)).

---

## Grit-FA: The Saved-State Pattern

candle's custom ops have no channel for intermediates kept from the forward for the backward,
but candle hands `bwd` the **same op instance** that ran the forward (`apply_op*` wraps it in the
`Arc` the graph holds). The house pattern:

- the state lives in a field, `Mutex<Option<T>>`, documented with its layout and when it is set;
- the forward sets it; `bwd` **clones** it out (never `take`s it): a graph may be
  differentiated more than once (`tests/parity.rs::backward_twice_over_one_graph`);
- a poisoned lock is an error with the op's name, never a panic;
- a backward that finds no state is an error, never a wrong answer
  (`"fused_attention: backward before forward"`).

---

## Grit-FA: Claims and Their Evidence

Every claim of determinism, accuracy or speed — in a doc comment, the README, the CHANGELOG or
a commit message — is backed by evidence of the matching kind, and says where it came from.

### Determinism claims

"Bit-reproducible" means a test re-runs the computation and compares **bits**
(`to_bits()` equality), several times, over every mode it claims (causal and not, both entry
points) — `tests/parity.rs::cuda_backward_is_deterministic`. Two equal runs of someone else's
kernel are an observation, not a guarantee; say so when reporting one.

### Bit-identity claims

"Bit-identical to X" — a fused kernel reproducing a composition, a refactor preserving a result —
is a different claim from determinism: it compares two *computations*, not two runs of one. It
requires the [explicit rounding](#explicit-rounding-where-bit-identity-is-claimed) above and a
test comparing the two results with `to_bits()` equality on representative shapes, including the
fallback paths (misaligned, non-contiguous).

### Accuracy claims

Errors are stated **against an fp64 reference**, with the fp64-rounded-to-fp32 floor beside
them (`bench/compare.py`). An fp32 composed reference is not a bar: over long reductions it can
be the less accurate side. A tolerance band is measured (CPU-vs-CUDA spread of the reference
itself), never assumed.

### Performance claims

A number states the **card, the shape, the instrument** (nsys kernel time or wall time) **and the
protocol**. The protocol rules, each bought by an incident:

- **Alternate** the candidates in rotating rounds within one session; a capture or a run from
  another hour is not comparable (a warm card ran unchanged kernels 4–5 % slower).
- **Discard first-run JIT samples**: a fresh binary's first process compiles its PTX inside
  whatever it is timing.
- **nsys exports must overwrite** (`--force-overwrite=true`, or delete first): a stale CSV once
  silently stood in for a new capture.
- **A ratio to `PyTorch` is same box, same session**; across boxes it is meaningless.
- Every number in the README or CHANGELOG names its date, card and protocol, and its report is
  **committed under `bench/results/`** (`<date>-<card>-<what>.md`: a `bench/compare.py`
  `report.md`, or an A/B log's summary with its source). A number whose report lives only in a
  git-ignored directory or another repository cannot be checked by a reader of this one.

---

## Python

`bench/compare.py` follows the organisation's Python conventions, `askesis/CONVENTIONS.md`:
2-space indentation, at most 100 columns, double quotes, PEP-604 type hints, a module docstring,
and **doctests on every pure helper** (run with `python -m doctest bench/compare.py`).
