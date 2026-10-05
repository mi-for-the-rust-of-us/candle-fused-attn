// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fused fp32 scaled-dot-product attention, forward AND backward, head_dim 64.
//
// The algorithm is FlashAttention-2's (Dao 2023): the forward walks key tiles with an online
// softmax and never writes the [s, s] scores, saving only O and the per-row log-sum-exp L; the
// backward recomputes P = exp(S - L) tile by tile. Two departures from FA-2's own CUDA, both on
// purpose:
//   - fp32 end to end on CUDA cores (SIMT FFMA), no tensor cores: FA-2 is f16/bf16 only, and
//     PyTorch's fp32 path (the memory-efficient kernels) uses 3xTF32. Plain FFMA keeps the
//     arithmetic that of an fp32 matmul.
//   - a DETERMINISTIC backward: FA-2 accumulates dQ with fp32 atomicAdd from every key block, and
//     PyTorch's memory-efficient backward (`kernel_backward.h`, v2.10) adds the key splits' dQ
//     under a spin lock in ARRIVAL order. Here the key blocks add their dQ partials in KEY-BLOCK
//     order (a turn counter per query tile, the ordered semaphore of CUTLASS's serial split-K), so
//     every sum has a fixed order and reruns are bit-identical; and S, dP are computed once per
//     tile (5 matmuls), not recomputed by a separate dQ kernel (7).
//
// Layouts. q, k, v are read through (batch, head, seq) strides with head_dim contiguous, so the
// views of a fused qkv projection need no copy. O is written [b, s, h, d] (the merge-heads
// layout); dQ, dK, dV are written INTERLEAVED into one [b, s, 3, h, d] buffer, the layout of the
// fused qkv projection's gradient; L and D are [b, h, s].
//
// Every kernel but the small D one is register micro-tiled (each thread a 2x4 or 4x4 block of
// outputs, shared memory read as float4) over DYNAMIC shared memory, 68 KB (69,632 B) per block:
// past candle's 48 KB static limit, so the launcher sets the attribute through cudarc. 256 threads.

#include <math.h>
#include <stdint.h>

#define HD 64          // head_dim; TWIN: src/lib.rs:CUDA_HEAD_DIM
#define NT 256         // threads per block; TWIN: src/cuda.rs:THREADS
#define KB_KEYS 64     // backward: keys per block (one key block); TWIN: src/cuda.rs:BWD_KEYS
#define KB_QRYS 32     // backward: queries per tile (one dQ turn counter each);
                       // TWIN: src/cuda.rs:QUERY_TILE

// ------------------------------------------------------------------------------------------------
// Backward, step 1: D[b, h, i] = sum_c dO[b, i, h, c] * O[b, i, h, c]. One warp per row. It also
// zeroes the dQ turn counters of step 2 ([b, h, ceil(S / KB_QRYS)], one per query tile), which
// saves a memset launch: the stream orders it before step 2.
// REFERENCE: the `dot_do_o` step of FlashAttention-2's backward (`run_flash_bwd_seqk_parallel`,
// Dao-AILab upstream) -- D = rowsum(dO o O), one row per warp; departs by also zeroing the turns.
// Shared memory: none; NT / 32 rows per block (TWIN: src/cuda.rs:WARPS derives it from NT).
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(NT) fattn_bwd_dot_f32_d64(
    const float* __restrict__ o, const float* __restrict__ d_o, float* __restrict__ dsum,
    int* __restrict__ turn,
    int B, int H, int S, int64_t do_sb, int64_t do_sh, int64_t do_ss) {
  const int row = blockIdx.x * (NT / 32) + threadIdx.x / 32;  // row = (b * H + h) * S + i
  const int lane = threadIdx.x % 32;
  if (row >= B * H * S) return;
  const int i = row % S, h = (row / S) % H, b = row / (S * H);
  const float* op = o + (((int64_t)b * S + i) * H + h) * HD;
  const float* gp = d_o + b * do_sb + i * do_ss + h * do_sh;
  float x = op[lane] * gp[lane] + op[lane + 32] * gp[lane + 32];
  // DETERMINISM: each lane its two products, then the xor butterfly 16, 8, 4, 2, 1: one fixed
  // order, whatever the card or the schedule.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
  if (lane == 0) dsum[row] = x;
  if (lane == 0 && i % KB_QRYS == 0)
    turn[row / S * ((S + KB_QRYS - 1) / KB_QRYS) + i / KB_QRYS] = 0;
}

// ------------------------------------------------------------------------------------------------
// Backward, steps 2 and 3: register micro-tiles over DYNAMIC shared memory (the launcher raises
// the per-block limit past 48 KB). Tiles are padded to LD = 68 floats: rows stay 16-byte aligned
// for float4 reads, and 8 consecutive rows start on 8 distinct 4-bank groups, so a float4 read of
// 8 rows by the 8 lanes of a phase is conflict-free.
// ------------------------------------------------------------------------------------------------
#define LD 68  // TWIN: src/cuda.rs:LD

__device__ __forceinline__ float4 ld4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ float dot4(float4 a, float4 b, float acc) {
  acc = fmaf(a.x, b.x, acc);
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  return fmaf(a.w, b.w, acc);
}

// `rows` rows of a strided [seq, HD] slice from `row0` into a [rows][LD] tile, float4 at a time
// (the launcher guarantees 16-byte alignment of every row); rows past `seq` are zero.
__device__ __forceinline__ void load_tile4(float* dst, const float* __restrict__ src,
                                           int64_t s_stride, int row0, int rows, int seq) {
  for (int idx = threadIdx.x; idx < rows * (HD / 4); idx += NT) {
    const int r = idx / (HD / 4), c4 = idx % (HD / 4);
    const int row = row0 + r;
    const float4 val = row < seq ? ld4(src + (int64_t)row * s_stride + 4 * c4)
                                 : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    *reinterpret_cast<float4*>(dst + r * LD + 4 * c4) = val;
  }
}

// Asynchronous tile loads, for the forward's K/V pipeline (devlog FA7). On compute capability 8.0+
// `cp.async.cg` copies 16 bytes from global to shared memory without staging them in registers,
// so the issuing thread does not wait for them; `cp_async_commit` closes this thread's group of
// copies and `cp_async_wait_all` waits for its groups (a `__syncthreads` must follow before other
// threads read the tile). Rows past `seq` are zero-filled with a source size of 0 (nothing is
// read; the address given is row `row0`, always inside the slice). Below 8.0 the copies are
// load_tile4's ordinary loads and commit / wait are empty: the same tile, the same arithmetic.
// `-DFATTN_SYNC_LOADS` (build.rs: $CANDLE_FUSED_ATTN_SYNC_LOADS) forces that path on any card, to
// test it.
// REFERENCE: kernel_traits.h (FlashAttention-2, as vendored in candle-flash-attn) --
// SM80_CP_ASYNC_CACHEGLOBAL<uint128_t> copies when __CUDA_ARCH__ >= 800, ordinary ones below.
__device__ __forceinline__ void load_tile_async(float* dst, const float* __restrict__ src,
                                                int64_t s_stride, int row0, int rows, int seq) {
#if __CUDA_ARCH__ >= 800 && !defined(FATTN_SYNC_LOADS)
  for (int idx = threadIdx.x; idx < rows * (HD / 4); idx += NT) {
    const int r = idx / (HD / 4), c4 = idx % (HD / 4);
    const int row = row0 + r;
    const float* g = src + (int64_t)(row < seq ? row : row0) * s_stride + 4 * c4;
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst + r * LD + 4 * c4));
    const int bytes = row < seq ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(bytes)
                 : "memory");
  }
#else
  load_tile4(dst, src, s_stride, row0, rows, seq);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800 && !defined(FATTN_SYNC_LOADS)
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800 && !defined(FATTN_SYNC_LOADS)
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// `n` consecutive floats of a row vector (L or D) from `src + row0` into `dst`, zero past `seq`;
// 4-byte copies, since these rows are not 16-byte aligned for every sequence length. Same switch
// as load_tile_async: `cp.async` on 8.0+, ordinary loads below or under -DFATTN_SYNC_LOADS.
__device__ __forceinline__ void load_row_async(float* dst, const float* __restrict__ src, int row0,
                                               int n, int seq) {
  for (int i = threadIdx.x; i < n; i += NT) {
    const int row = row0 + i;
#if __CUDA_ARCH__ >= 800 && !defined(FATTN_SYNC_LOADS)
    const float* g = src + (row < seq ? row : row0);
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst + i));
    const int bytes = row < seq ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(g), "r"(bytes)
                 : "memory");
#else
    dst[i] = row < seq ? src[row] : 0.0f;
#endif
  }
}

// The dQ turn of one query tile: wait until `kb` key blocks have added theirs (thread 0 spins,
// acquire at GPU scope), then pass it on (release).
// REFERENCE: `AtomicLock` in kernel_backward.h (PyTorch v2.10, mem_eff_attention) -- the
// acquire/release handshake; departs on who goes next: the turn goes in key-block order, not to
// whoever arrives (the ordered semaphore of CUTLASS's serial split-K).
__device__ __forceinline__ void wait_turn(const int* p, int kb) {
  int seen;
  for (;;) {
    // ORDER: acquire at GPU scope; pairs with pass_turn's release by key block kb - 1, so that
    // block's dQ stores are visible to this thread once it reads kb.
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(seen) : "l"(p) : "memory");
    if (seen == kb) break;
    __nanosleep(32);
  }
}

__device__ __forceinline__ void pass_turn(int* p, int next) {
  // ORDER: release at GPU scope; publishes this block's dQ stores (made visible by the caller's
  // __threadfence) to key block `next`, whose wait_turn acquires them.
  asm volatile("st.release.gpu.global.b32 [%0], %1;" : : "l"(p), "r"(next) : "memory");
}

// dK, dV, dQ. grid (ceil(S / 64), H, B). A block owns key block kb = blockIdx.x (64 keys) and walks
// query tiles of 32. Phases 1 and 2 each compute two products over the same outputs; each product
// goes to one half of the block (half = tid / 128, u = tid % 128; devlog FA14), so a thread does
// twice the outputs of one product for half the operand loads per step.
//   phase 1: S (half 0, from Q and K) or dP (half 1, from dO and V) for [32 queries x 64 keys];
//            thread u owns query rows 4 (u / 16) .. +4 and keys u % 16 + 16 jj. Half 0 writes
//            P to shared memory; after a barrier, half 1 reads it back and writes dS.
//   phase 2: dV (half 0, from P and dO) or dK (half 1, from dS and Q); thread u owns keys
//            4 (u / 8) .. +4 and dims 4 (u % 8) .. +4 and 32 + 4 (u % 8) .. +4 (two float4 32
//            floats apart: each 8-thread phase of a 128-bit load reads 32 consecutive words).
//   phase 3: thread owns queries 2 (tid / 16), +1 and dims 4 (tid % 16) .. +4 of the tile's dQ
//            partial dS K; on its turn the block adds it into dQ: key block 0 stores, the others
//            read (through L2) and add, the last contributor scales. The tile's dQ is thus
//            ((p0 + p1) + p2) + ... in key-block order, whatever the scheduling.
// No deadlock: a block waits only on LOWER key blocks of its own (b, h), which have lower linear
// block indices and so were dispatched first (the assumption CUTLASS's serial split-K makes).
// REFERENCE: flash_bwd_kernel.h (FlashAttention-2, Dao-AILab upstream) -- key blocks in parallel,
// dK and dV in registers, P = exp(S - L) recomputed per query tile, dS = P o (dP - D); and
// kernel_backward.h (PyTorch v2.10, mem_eff_attention) -- one pass per key block, S and dP computed
// once. Departs from both on dQ: added in key-block ORDER (FA-2: fp32 atomicAdd; PyTorch: arrival
// order under a lock), so the backward is deterministic.
// Loads overlap the math (devlog FA9): the NEXT query tile's Q, dO, L and D are copied into a
// second buffer while the current tile computes (double buffering), then the buffers swap.
// REFERENCE: flash_bwd_kernel.h (FlashAttention-2, Dao-AILab upstream) -- its `Double_buffer`
// alternates sQ between two halves; departs by double-buffering dO too (FA-2 reloads its single
// sdO after the dV GEMM): here dO is read until phase 2, so one buffer would overlap phase 3 only.
// Shared memory: dynamic, (2 KB_KEYS + 6 KB_QRYS) LD + 4 KB_QRYS floats = 87,552 B (85.5 KB; TWIN:
// src/cuda.rs:BWD_SMEM): one block per SM on the RTX 5060 Ti.
extern "C" __global__ void __launch_bounds__(NT) fattn_bwd_f32_d64(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ d_o, const float* __restrict__ lse, const float* __restrict__ dsum,
    float* __restrict__ dqkv, int* __restrict__ turn,
    int B, int H, int S, float scale, int causal,
    int64_t q_sb, int64_t q_sh, int64_t q_ss,
    int64_t k_sb, int64_t k_sh, int64_t k_ss,
    int64_t v_sb, int64_t v_sh, int64_t v_ss,
    int64_t do_sb, int64_t do_sh, int64_t do_ss) {
  extern __shared__ float4 smem4[];
  float* Ks = reinterpret_cast<float*>(smem4);  // [64][LD]
  float* Vs = Ks + KB_KEYS * LD;                // [64][LD]
  float* Qb = Vs + KB_KEYS * LD;                // [2][32][LD]: Q, current and next query tile
  float* dOb = Qb + 2 * KB_QRYS * LD;           // [2][32][LD]: dO, the same
  float* Ps = dOb + 2 * KB_QRYS * LD;           // [32][LD], query-major
  float* dSs = Ps + KB_QRYS * LD;               // [32][LD], query-major
  float* LDb = dSs + KB_QRYS * LD;              // [2][2][32]: L then D, current and next tile

  const int b = blockIdx.z, h = blockIdx.y, kb = blockIdx.x, k0 = kb * KB_KEYS;
  const int tid = threadIdx.x, half = tid / 128, u = tid % 128;  // half: warp-uniform
  const int r0 = 4 * (u / 16), kx = u % 16;                       // phase-1 ownership
  const int j0 = 4 * (u / 8), e0 = 4 * (u % 8);                   // phase-2 ownership
  const int i0 = 2 * (tid / 16), d0 = 4 * (tid % 16);             // phase-3 ownership
  int* tp = turn + ((int64_t)b * H + h) * ((S + KB_QRYS - 1) / KB_QRYS);
  const float* qp = q + b * q_sb + h * q_sh;
  const float* gp = d_o + b * do_sb + h * do_sh;
  const float* lp = lse + ((int64_t)b * H + h) * S;
  const float* dp_ = dsum + ((int64_t)b * H + h) * S;

  load_tile4(Ks, k + b * k_sb + h * k_sh, k_ss, k0, KB_KEYS, S);
  load_tile4(Vs, v + b * v_sb + h * v_sh, v_ss, k0, KB_KEYS, S);

  // Half 0: dV; half 1: dK (unscaled until the store). [a][e]: key j0 + a; dim e0 + e for e < 4,
  // 32 + e0 + e - 4 for e >= 4.
  float acc[4][8];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[a][e] = 0.0f;

  // Causal: queries before the block's first key never see it.
  const int q_begin = causal ? (k0 / KB_QRYS) * KB_QRYS : 0;
  // Prologue: the first query tile into buffer 0, one group of copies.
  load_tile_async(Qb, qp, q_ss, q_begin, KB_QRYS, S);
  load_tile_async(dOb, gp, do_ss, q_begin, KB_QRYS, S);
  load_row_async(LDb, lp, q_begin, KB_QRYS, S);
  load_row_async(LDb + KB_QRYS, dp_, q_begin, KB_QRYS, S);
  cp_async_commit();
  for (int q0 = q_begin, buf = 0; q0 < S; q0 += KB_QRYS, buf ^= 1) {
    // This tile has landed for every thread, and every thread is done with the previous
    // iteration (its buffer, Ps, dSs): the next tile may now be copied into the other buffer.
    cp_async_wait_all();
    __syncthreads();
    if (q0 + KB_QRYS < S) {
      const int nb = buf ^ 1, qn = q0 + KB_QRYS;  // travels during this whole iteration
      load_tile_async(Qb + nb * KB_QRYS * LD, qp, q_ss, qn, KB_QRYS, S);
      load_tile_async(dOb + nb * KB_QRYS * LD, gp, do_ss, qn, KB_QRYS, S);
      load_row_async(LDb + nb * 2 * KB_QRYS, lp, qn, KB_QRYS, S);
      load_row_async(LDb + nb * 2 * KB_QRYS + KB_QRYS, dp_, qn, KB_QRYS, S);
      cp_async_commit();
    }
    const float* Qs = Qb + buf * KB_QRYS * LD;
    const float* dOs = dOb + buf * KB_QRYS * LD;
    const float* Lt = LDb + buf * 2 * KB_QRYS;  // L of this tile, 0 past S
    const float* Dt = Lt + KB_QRYS;             // D of this tile, 0 past S

    // Dead tail work skipped (devlog FA15): every bound below depends on S, k0 and q0 alone.
    // A skipped term has a zero factor (a zero-filled row, or a P / dS the `live` mask zeroed),
    // and no accumulator is ever -0 (each chain starts at +0; fmaf returns -0 only from a -0
    // addend), so skipping it leaves every bit unchanged.
    const int ngrp = min(4, (S - k0 + 15) / 16);           // key groups jj holding a key < S
    const int nq = min(KB_QRYS, S - q0);                   // live query rows of this tile
    const int nk = min(KB_KEYS, (S - k0 + 3) / 4 * 4);     // phase-3 keys, in steps of 4

    // Phase 1. Half 0: s = Q K^T; half 1: s = dO V^T (dP). Same dot4 chain per element as when
    // one thread computed both (devlog FA14: the bitwise gate applies).
    const float* As = half ? dOs : Qs;
    const float* Bs = half ? Vs : Ks;
    float s[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) s[r][jj] = 0.0f;
    if (r0 < nq) {  // else all four rows are past S: s stays +0, as the zero rows gave
#pragma unroll 4
      for (int c = 0; c < HD; c += 4) {
        float4 av[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) av[r] = ld4(As + (r0 + r) * LD + c);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          if (jj >= ngrp) continue;  // the whole group is past S
          const float4 bv = ld4(Bs + (kx + 16 * jj) * LD + c);
#pragma unroll
          for (int r = 0; r < 4; ++r) s[r][jj] = dot4(av[r], bv, s[r][jj]);
        }
      }
    }
    if (half == 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int il = r0 + r, i = q0 + il;
        const float li = Lt[il];  // i >= S: 0, as v0.2's guarded reads gave
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int jl = kx + 16 * jj, j = k0 + jl;
          const bool live = i < S && j < S && !(causal && j > i);
          Ps[il * LD + jl] = live ? expf(s[r][jj] * scale - li) : 0.0f;
        }
      }
    }
    __syncthreads();
    if (half == 1) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int il = r0 + r;
        const float di = Dt[il];  // i >= S: 0
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int jl = kx + 16 * jj;
          dSs[il * LD + jl] = Ps[il * LD + jl] * (s[r][jj] - di);
        }
      }
    }
    __syncthreads();

    // Phase 2. Half 0: dV[j] += sum_i P[i][j] dO[i]; half 1: dK[j] += sum_i dS[i][j] Q[i].
    // DETERMINISM: each dK, dV element is one thread's register, summed over query tiles and
    // then queries in index order; no reduction across threads or blocks.
    const float* Xs = half ? dSs : Ps;
    const float* Ys = half ? Qs : dOs;
    if (k0 + j0 < S) {  // else all four keys are past S: never stored
#pragma unroll 4
      for (int i = 0; i < nq; ++i) {
        const float4 x4 = ld4(Xs + i * LD + j0);
        const float4 ya = ld4(Ys + i * LD + e0), yb = ld4(Ys + i * LD + 32 + e0);
        const float xs[4] = {x4.x, x4.y, x4.z, x4.w};
        const float ys[8] = {ya.x, ya.y, ya.z, ya.w, yb.x, yb.y, yb.z, yb.w};
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[a][e] = fmaf(xs[a], ys[e], acc[a][e]);
      }
    }

    // dQ partial: pq[r][e] = sum_j dS[i0 + r][j] K[j][d0 + e], four keys at a time
    float pq[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int e = 0; e < 4; ++e) pq[r][e] = 0.0f;
    if (i0 < nq) {  // else both rows are past S: never stored
#pragma unroll 2
      for (int j = 0; j < nk; j += 4) {
        const float4 s0 = ld4(dSs + i0 * LD + j), s1 = ld4(dSs + (i0 + 1) * LD + j);
        const float sa[2][4] = {{s0.x, s0.y, s0.z, s0.w}, {s1.x, s1.y, s1.z, s1.w}};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const float4 k4 = ld4(Ks + (j + t) * LD + d0);
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            pq[r][0] = fmaf(sa[r][t], k4.x, pq[r][0]);
            pq[r][1] = fmaf(sa[r][t], k4.y, pq[r][1]);
            pq[r][2] = fmaf(sa[r][t], k4.z, pq[r][2]);
            pq[r][3] = fmaf(sa[r][t], k4.w, pq[r][3]);
          }
        }
      }
    }

    // This tile's contributors are key blocks 0 .. last (causal: those holding a key <= the
    // tile's last query); each adds on its turn.
    // DETERMINISM: the tile's dQ is ((p0 + p1) + p2) + ... over key blocks in index order, then
    // x scale by the last: a function of the shape alone, never of the card or the schedule.
    const int t = q0 / KB_QRYS;
    const int last = causal ? min((int)gridDim.x - 1, min(S - 1, q0 + KB_QRYS - 1) / KB_KEYS)
                            : (int)gridDim.x - 1;
    if (kb > 0) {
      if (tid == 0) {
        wait_turn(tp + t, kb);
        // ORDER: with the __syncthreads below, extends thread 0's acquire to the whole block:
        // every thread's __ldcg of dQ reads after key block kb - 1's stores.
        __threadfence();
      }
      __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = q0 + i0 + r;
      if (i >= S) continue;
      // Through L2 (__ldcg / __stcg): L1 is not coherent across SMs.
      float4* dst =
          reinterpret_cast<float4*>(dqkv + ((int64_t)b * S + i) * (3 * H * HD) + h * HD + d0);
      float4 acc = make_float4(pq[r][0], pq[r][1], pq[r][2], pq[r][3]);
      if (kb > 0) {
        const float4 prev = __ldcg(dst);
        acc = make_float4(prev.x + acc.x, prev.y + acc.y, prev.z + acc.z, prev.w + acc.w);
      }
      if (kb == last) acc = make_float4(acc.x * scale, acc.y * scale, acc.z * scale, acc.w * scale);
      __stcg(dst, acc);
    }
    if (kb < last) {
      // ORDER: every thread's dQ stores visible GPU-wide before thread 0 releases the turn to
      // key block kb + 1 (pass_turn).
      __threadfence();
      __syncthreads();
      if (tid == 0) pass_turn(tp + t, kb + 1);
    }
  }

  // Half 0 stores dV, half 1 stores dK x scale.
  float* dst = dqkv + (half ? H * HD : 2 * H * HD);
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int j = k0 + j0 + a;
    if (j >= S) continue;
    const int64_t off = ((int64_t)b * S + j) * (3 * H * HD) + h * HD + e0;
    if (half) {
      *reinterpret_cast<float4*>(dst + off) =
          make_float4(acc[a][0] * scale, acc[a][1] * scale, acc[a][2] * scale, acc[a][3] * scale);
      *reinterpret_cast<float4*>(dst + off + 32) =
          make_float4(acc[a][4] * scale, acc[a][5] * scale, acc[a][6] * scale, acc[a][7] * scale);
    } else {
      *reinterpret_cast<float4*>(dst + off) = make_float4(acc[a][0], acc[a][1], acc[a][2], acc[a][3]);
      *reinterpret_cast<float4*>(dst + off + 32) =
          make_float4(acc[a][4], acc[a][5], acc[a][6], acc[a][7]);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Forward. grid (ceil(S / 64), H, B). A block owns 64 queries and walks key tiles of 64.
//   phase 1: S for [64 queries x 64 keys]; thread (ty = tid / 16, tx = tid % 16) owns query rows
//            4ty .. +4 and keys tx + 16 jj; the online-softmax state (m, l) of its 4 rows is
//            reduced over the 16 lanes sharing ty; P goes to shared memory.
//   phase 2: the SAME thread owns rows 4ty .. +4 and dims 4tx .. +4 of the O accumulator, so the
//            rescale by exp(m_old - m_new) and the final 1/l never cross threads.
//   Measured and rejected (devlog FA11): 128 threads with 8 x 4 outputs each -- 25 % fewer shared
//   loads, bit-identical, but +15.5 % kernel time on an RTX 5060 Ti (4 warps per SM leave the
//   schedulers without a ready warp 58 % of the time).
// Out: o [b, s, h, d] (the merge-heads layout) and lse [b, h, s], the backward's saved state.
// Loads overlap the math (devlog FA7): V(n) travels while phase 1 computes S from K(n), and
// K(n + 1) travels while phase 2 consumes V(n); one K and one V buffer suffice, because each is
// refilled only after the barrier that ends its last read.
// REFERENCE: flash_fwd_kernel.h (FlashAttention-2, as vendored in candle-flash-attn) -- key tiles
// walked with an online softmax, only O and L written, and the main loop's load schedule (lines
// 414-470: V issued before the QK^T gemm, the next K before softmax and PV); departs on the
// arithmetic: fp32 FFMA on CUDA cores, no tensor cores (FA-2 is f16/bf16 only).
// Shared memory: dynamic, (2 FW_QRYS + 2 FW_KEYS) LD floats = 69,632 B (68 KB; TWIN:
// src/cuda.rs:FWD_SMEM): one block per SM on the RTX 5060 Ti (measured).
// ------------------------------------------------------------------------------------------------
#define FW_QRYS 64  // TWIN: src/cuda.rs:FWD_QUERIES
#define FW_KEYS 64  // TWIN: src/cuda.rs:FWD_KEYS

// DETERMINISM (row_max16, row_sum16): the xor butterfly 1, 2, 4, 8 over the 16 lanes of a row;
// the max is order-free and the sum's order is fixed.

__device__ __forceinline__ float row_max16(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 4));
  return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 8));
}

__device__ __forceinline__ float row_sum16(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 4);
  return x + __shfl_xor_sync(0xffffffffu, x, 8);
}

extern "C" __global__ void __launch_bounds__(NT) fattn_fwd_f32_d64(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ o, float* __restrict__ lse,
    int B, int H, int S, float scale, int causal,
    int64_t q_sb, int64_t q_sh, int64_t q_ss,
    int64_t k_sb, int64_t k_sh, int64_t k_ss,
    int64_t v_sb, int64_t v_sh, int64_t v_ss) {
  extern __shared__ float4 smem4[];
  float* Qs = reinterpret_cast<float*>(smem4);  // [64][LD]
  float* Ks = Qs + FW_QRYS * LD;                // [64][LD]
  float* Vs = Ks + FW_KEYS * LD;                // [64][LD]
  float* Ps = Vs + FW_KEYS * LD;                // [64][LD], query-major

  const int b = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * FW_QRYS;
  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  const int r0 = 4 * ty, d0 = 4 * tx;
  const float* kp = k + b * k_sb + h * k_sh;
  const float* vp = v + b * v_sb + h * v_sh;

  // Causal: keys past the block's last query never contribute.
  const int k_end = causal ? min(S, q0 + FW_QRYS) : S;

  // Prologue: Q and the first key tile, one group of copies.
  load_tile_async(Qs, q + b * q_sb + h * q_sh, q_ss, q0, FW_QRYS, S);
  load_tile_async(Ks, kp, k_ss, 0, FW_KEYS, S);
  cp_async_commit();

  float m[4], l[4], acc[4][4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[r][e] = 0.0f;
  }

  for (int k0 = 0; k0 < k_end; k0 += FW_KEYS) {
    // K(k0) (and, the first time, Q) has landed for every thread, and every thread is done with
    // the previous tile's Vs and Ps (its phase 2): V(k0) may now overwrite Vs.
    cp_async_wait_all();
    __syncthreads();
    load_tile_async(Vs, vp, v_ss, k0, FW_KEYS, S);  // travels during phase 1
    cp_async_commit();

    float s[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) s[r][jj] = 0.0f;
#pragma unroll 4
    for (int c = 0; c < HD; c += 4) {
      float4 qv[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) qv[r] = ld4(Qs + (r0 + r) * LD + c);
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float4 kv = ld4(Ks + (tx + 16 * jj) * LD + c);
#pragma unroll
        for (int r = 0; r < 4; ++r) s[r][jj] = dot4(qv[r], kv, s[r][jj]);
      }
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = q0 + r0 + r;
      float tmax = -INFINITY;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int j = k0 + tx + 16 * jj;
        s[r][jj] = (j < S && !(causal && j > i)) ? s[r][jj] * scale : -INFINITY;
        tmax = fmaxf(tmax, s[r][jj]);
      }
      tmax = row_max16(tmax);
      const float m_new = fmaxf(m[r], tmax);
      // A row with every key so far masked keeps m = -inf; exp(-inf - -inf) would be NaN.
      const float alpha = m_new == -INFINITY ? 1.0f : expf(m[r] - m_new);
      float psum = 0.0f;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float p = m_new == -INFINITY ? 0.0f : expf(s[r][jj] - m_new);
        Ps[(r0 + r) * LD + tx + 16 * jj] = p;
        psum += p;
      }
      // DETERMINISM: l and acc accumulate over key tiles in index order, one thread per element.
      l[r] = l[r] * alpha + row_sum16(psum);
      m[r] = m_new;
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][e] *= alpha;
    }
    // V(k0) has landed for every thread, every P is written, and every thread is done with Ks
    // (its phase 1): K(k0 + FW_KEYS) may now overwrite Ks.
    cp_async_wait_all();
    __syncthreads();
    if (k0 + FW_KEYS < k_end) {
      load_tile_async(Ks, kp, k_ss, k0 + FW_KEYS, FW_KEYS, S);  // travels during phase 2
      cp_async_commit();
    }

    // O[i] += sum_j P[i][j] V[j], four keys at a time
#pragma unroll 2
    for (int j = 0; j < FW_KEYS; j += 4) {
      float4 pv[4], vv[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) pv[r] = ld4(Ps + (r0 + r) * LD + j);
#pragma unroll
      for (int t = 0; t < 4; ++t) vv[t] = ld4(Vs + (j + t) * LD + d0);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float pr[4] = {pv[r].x, pv[r].y, pv[r].z, pv[r].w};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          acc[r][0] = fmaf(pr[t], vv[t].x, acc[r][0]);
          acc[r][1] = fmaf(pr[t], vv[t].y, acc[r][1]);
          acc[r][2] = fmaf(pr[t], vv[t].z, acc[r][2]);
          acc[r][3] = fmaf(pr[t], vv[t].w, acc[r][3]);
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = q0 + r0 + r;
    if (i >= S) continue;
    const float inv = 1.0f / l[r];
    *reinterpret_cast<float4*>(o + (((int64_t)b * S + i) * H + h) * HD + d0) =
        make_float4(acc[r][0] * inv, acc[r][1] * inv, acc[r][2] * inv, acc[r][3] * inv);
    if (tx == 0) lse[((int64_t)b * H + h) * S + i] = m[r] + logf(l[r]);
  }
}
