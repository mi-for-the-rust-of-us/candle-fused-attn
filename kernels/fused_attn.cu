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
//   - a DETERMINISTIC backward: FA-2 accumulates dQ with fp32 atomicAdd from every key block;
//     here dK/dV (one block per key tile) and dQ (one block per query tile) are separate kernels
//     that each recompute P, so every sum has a fixed order and reruns are bit-identical.
//
// Layouts. q, k, v are read through (batch, head, seq) strides with head_dim contiguous, so the
// views of a fused qkv projection need no copy. O is written [b, s, h, d] (the merge-heads
// layout); dQ, dK, dV are written INTERLEAVED into one [b, s, 3, h, d] buffer, the layout of the
// fused qkv projection's gradient; L and D are [b, h, s].
//
// Every kernel but the small D one is register micro-tiled (each thread a 2x4 or 4x4 block of
// outputs, shared memory read as float4) over DYNAMIC shared memory, 61-70 KB per block: past
// candle's 48 KB static limit, so the launcher sets the attribute through cudarc. 256 threads.

#include <math.h>
#include <stdint.h>

#define HD 64          // head_dim
#define NT 256         // threads per block

// ------------------------------------------------------------------------------------------------
// Backward, step 1: D[b, h, i] = sum_c dO[b, i, h, c] * O[b, i, h, c]. One warp per row.
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void fattn_bwd_dot_f32_d64(
    const float* __restrict__ o, const float* __restrict__ d_o, float* __restrict__ dsum,
    int B, int H, int S, int64_t do_sb, int64_t do_sh, int64_t do_ss) {
  const int row = blockIdx.x * (NT / 32) + threadIdx.x / 32;  // row = (b * H + h) * S + i
  const int lane = threadIdx.x % 32;
  if (row >= B * H * S) return;
  const int i = row % S, h = (row / S) % H, b = row / (S * H);
  const float* op = o + (((int64_t)b * S + i) * H + h) * HD;
  const float* gp = d_o + b * do_sb + i * do_ss + h * do_sh;
  float x = op[lane] * gp[lane] + op[lane + 32] * gp[lane + 32];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
  if (lane == 0) dsum[row] = x;
}

// ------------------------------------------------------------------------------------------------
// Backward, steps 2 and 3: register micro-tiles over DYNAMIC shared memory (the launcher raises
// the per-block limit past 48 KB). Tiles are padded to LD = 68 floats: rows stay 16-byte aligned
// for float4 reads, and 8 consecutive rows start on 8 distinct 4-bank groups, so a float4 read of
// 8 rows by the 8 lanes of a phase is conflict-free.
// ------------------------------------------------------------------------------------------------
#define LD 68

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

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

// dK, dV. grid (ceil(S / 64), H, B). A block owns 64 keys and walks query tiles of 32.
//   phase 1: S and dP for [32 queries x 64 keys]; thread (ty = tid / 16, tx = tid % 16) owns
//            query rows 2ty, 2ty+1 and keys tx + 16 jj; P and dS go to shared memory.
//   phase 2: thread owns keys 4 (tid / 16) .. +4 and dims 4 (tid % 16) .. +4 of dV and dK.
#define KB_KEYS 64
#define KB_QRYS 32
extern "C" __global__ void __launch_bounds__(NT) fattn_bwd_dkdv_f32_d64(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ d_o, const float* __restrict__ lse, const float* __restrict__ dsum,
    float* __restrict__ dqkv,
    int B, int H, int S, float scale, int causal,
    int64_t q_sb, int64_t q_sh, int64_t q_ss,
    int64_t k_sb, int64_t k_sh, int64_t k_ss,
    int64_t v_sb, int64_t v_sh, int64_t v_ss,
    int64_t do_sb, int64_t do_sh, int64_t do_ss) {
  extern __shared__ float4 smem4[];
  float* Ks = reinterpret_cast<float*>(smem4);  // [64][LD]
  float* Vs = Ks + KB_KEYS * LD;                // [64][LD]
  float* Qs = Vs + KB_KEYS * LD;                // [32][LD]
  float* dOs = Qs + KB_QRYS * LD;               // [32][LD]
  float* Ps = dOs + KB_QRYS * LD;               // [32][LD], query-major
  float* dSs = Ps + KB_QRYS * LD;               // [32][LD], query-major

  const int b = blockIdx.z, h = blockIdx.y, k0 = blockIdx.x * KB_KEYS;
  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  const int j0 = 4 * (tid / 16), d0 = 4 * (tid % 16);  // phase-2 ownership
  const float* qp = q + b * q_sb + h * q_sh;
  const float* gp = d_o + b * do_sb + h * do_sh;
  const float* lp = lse + ((int64_t)b * H + h) * S;
  const float* dp_ = dsum + ((int64_t)b * H + h) * S;

  load_tile4(Ks, k + b * k_sb + h * k_sh, k_ss, k0, KB_KEYS, S);
  load_tile4(Vs, v + b * v_sb + h * v_sh, v_ss, k0, KB_KEYS, S);

  float adk[4][4], adv[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int e = 0; e < 4; ++e) adk[a][e] = adv[a][e] = 0.0f;

  // Causal: queries before the block's first key never see it.
  const int q_begin = causal ? (k0 / KB_QRYS) * KB_QRYS : 0;
  for (int q0 = q_begin; q0 < S; q0 += KB_QRYS) {
    __syncthreads();
    load_tile4(Qs, qp, q_ss, q0, KB_QRYS, S);
    load_tile4(dOs, gp, do_ss, q0, KB_QRYS, S);
    __syncthreads();

    float s[2][4], dpv[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) s[r][jj] = dpv[r][jj] = 0.0f;
#pragma unroll 4
    for (int c = 0; c < HD; c += 4) {
      const float4 q0v = ld4(Qs + (2 * ty) * LD + c), q1v = ld4(Qs + (2 * ty + 1) * LD + c);
      const float4 g0v = ld4(dOs + (2 * ty) * LD + c), g1v = ld4(dOs + (2 * ty + 1) * LD + c);
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float4 kv = ld4(Ks + (tx + 16 * jj) * LD + c);
        const float4 vv = ld4(Vs + (tx + 16 * jj) * LD + c);
        s[0][jj] = dot4(q0v, kv, s[0][jj]);
        s[1][jj] = dot4(q1v, kv, s[1][jj]);
        dpv[0][jj] = dot4(g0v, vv, dpv[0][jj]);
        dpv[1][jj] = dot4(g1v, vv, dpv[1][jj]);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int il = 2 * ty + r, i = q0 + il;
      const float li = i < S ? lp[i] : 0.0f, di = i < S ? dp_[i] : 0.0f;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int jl = tx + 16 * jj, j = k0 + jl;
        const bool live = i < S && j < S && !(causal && j > i);
        const float p = live ? expf(s[r][jj] * scale - li) : 0.0f;
        Ps[il * LD + jl] = p;
        dSs[il * LD + jl] = p * (dpv[r][jj] - di);
      }
    }
    __syncthreads();

    // dV[j] += sum_i P[i][j] dO[i];  dK[j] += sum_i dS[i][j] Q[i]
#pragma unroll 4
    for (int i = 0; i < KB_QRYS; ++i) {
      const float4 p4 = ld4(Ps + i * LD + j0), s4 = ld4(dSs + i * LD + j0);
      const float4 g4 = ld4(dOs + i * LD + d0), q4 = ld4(Qs + i * LD + d0);
      const float pa[4] = {p4.x, p4.y, p4.z, p4.w}, sa[4] = {s4.x, s4.y, s4.z, s4.w};
      const float ga[4] = {g4.x, g4.y, g4.z, g4.w}, qa[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          adv[a][e] = fmaf(pa[a], ga[e], adv[a][e]);
          adk[a][e] = fmaf(sa[a], qa[e], adk[a][e]);
        }
    }
  }

#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int j = k0 + j0 + a;
    if (j >= S) continue;
    const int64_t off = ((int64_t)b * S + j) * (3 * H * HD) + h * HD + d0;
    *reinterpret_cast<float4*>(dqkv + H * HD + off) =
        make_float4(adk[a][0] * scale, adk[a][1] * scale, adk[a][2] * scale, adk[a][3] * scale);
    *reinterpret_cast<float4*>(dqkv + 2 * H * HD + off) = make_float4(adv[a][0], adv[a][1], adv[a][2], adv[a][3]);
  }
}

// dQ. grid (ceil(S / 64), H, B). A block owns 64 queries and walks key tiles of 32.
//   phase 1: S and dP for [64 queries x 32 keys]; thread (ty = tid / 8, tx = tid % 8) owns query
//            rows 2ty, 2ty+1 and keys tx + 8 jj; dS goes to shared memory TRANSPOSED (key-major).
//   phase 2: thread owns queries 4 (tid / 16) .. +4 and dims 4 (tid % 16) .. +4 of dQ.
#define QB_QRYS 64
#define QB_KEYS 32
extern "C" __global__ void __launch_bounds__(NT) fattn_bwd_dq_f32_d64(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ d_o, const float* __restrict__ lse, const float* __restrict__ dsum,
    float* __restrict__ dqkv,
    int B, int H, int S, float scale, int causal,
    int64_t q_sb, int64_t q_sh, int64_t q_ss,
    int64_t k_sb, int64_t k_sh, int64_t k_ss,
    int64_t v_sb, int64_t v_sh, int64_t v_ss,
    int64_t do_sb, int64_t do_sh, int64_t do_ss) {
  extern __shared__ float4 smem4[];
  float* Qs = reinterpret_cast<float*>(smem4);  // [64][LD]
  float* dOs = Qs + QB_QRYS * LD;               // [64][LD]
  float* Ks = dOs + QB_QRYS * LD;               // [32][LD]
  float* Vs = Ks + QB_KEYS * LD;                // [32][LD]
  float* dSt = Vs + QB_KEYS * LD;               // [32][LD], key-major

  const int b = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * QB_QRYS;
  const int tid = threadIdx.x, ty = tid / 8, tx = tid % 8;
  const int i0 = 4 * (tid / 16), d0 = 4 * (tid % 16);  // phase-2 ownership
  const float* kp = k + b * k_sb + h * k_sh;
  const float* vp = v + b * v_sb + h * v_sh;

  load_tile4(Qs, q + b * q_sb + h * q_sh, q_ss, q0, QB_QRYS, S);
  load_tile4(dOs, d_o + b * do_sb + h * do_sh, do_ss, q0, QB_QRYS, S);

  float li[2], di[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = q0 + 2 * ty + r;
    li[r] = i < S ? lse[((int64_t)b * H + h) * S + i] : 0.0f;
    di[r] = i < S ? dsum[((int64_t)b * H + h) * S + i] : 0.0f;
  }

  float adq[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int e = 0; e < 4; ++e) adq[a][e] = 0.0f;

  const int k_end = causal ? min(S, q0 + QB_QRYS) : S;
  for (int k0 = 0; k0 < k_end; k0 += QB_KEYS) {
    __syncthreads();
    load_tile4(Ks, kp, k_ss, k0, QB_KEYS, S);
    load_tile4(Vs, vp, v_ss, k0, QB_KEYS, S);
    __syncthreads();

    float s[2][4], dpv[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) s[r][jj] = dpv[r][jj] = 0.0f;
#pragma unroll 4
    for (int c = 0; c < HD; c += 4) {
      const float4 q0v = ld4(Qs + (2 * ty) * LD + c), q1v = ld4(Qs + (2 * ty + 1) * LD + c);
      const float4 g0v = ld4(dOs + (2 * ty) * LD + c), g1v = ld4(dOs + (2 * ty + 1) * LD + c);
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float4 kv = ld4(Ks + (tx + 8 * jj) * LD + c);
        const float4 vv = ld4(Vs + (tx + 8 * jj) * LD + c);
        s[0][jj] = dot4(q0v, kv, s[0][jj]);
        s[1][jj] = dot4(q1v, kv, s[1][jj]);
        dpv[0][jj] = dot4(g0v, vv, dpv[0][jj]);
        dpv[1][jj] = dot4(g1v, vv, dpv[1][jj]);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int il = 2 * ty + r, i = q0 + il;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int jl = tx + 8 * jj, j = k0 + jl;
        const bool live = i < S && j < S && !(causal && j > i);
        const float p = live ? expf(s[r][jj] * scale - li[r]) : 0.0f;
        dSt[jl * LD + il] = p * (dpv[r][jj] - di[r]);
      }
    }
    __syncthreads();

    // dQ[i] += sum_j dS[i][j] K[j]
#pragma unroll 4
    for (int j = 0; j < QB_KEYS; ++j) {
      const float4 s4 = ld4(dSt + j * LD + i0), k4 = ld4(Ks + j * LD + d0);
      const float sa[4] = {s4.x, s4.y, s4.z, s4.w}, ka[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int e = 0; e < 4; ++e) adq[a][e] = fmaf(sa[a], ka[e], adq[a][e]);
    }
  }

#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int i = q0 + i0 + a;
    if (i >= S) continue;
    const int64_t off = ((int64_t)b * S + i) * (3 * H * HD) + h * HD + d0;
    *reinterpret_cast<float4*>(dqkv + off) =
        make_float4(adq[a][0] * scale, adq[a][1] * scale, adq[a][2] * scale, adq[a][3] * scale);
  }
}

// ------------------------------------------------------------------------------------------------
// Forward. grid (ceil(S / 64), H, B). A block owns 64 queries and walks key tiles of 64.
//   phase 1: S for [64 queries x 64 keys]; thread (ty = tid / 16, tx = tid % 16) owns query rows
//            4ty .. +4 and keys tx + 16 jj; the online-softmax state (m, l) of its 4 rows is
//            reduced over the 16 lanes sharing ty; P goes to shared memory.
//   phase 2: the SAME thread owns rows 4ty .. +4 and dims 4tx .. +4 of the O accumulator, so the
//            rescale by exp(m_old - m_new) and the final 1/l never cross threads.
// Out: o [b, s, h, d] (the merge-heads layout) and lse [b, h, s], the backward's saved state.
// ------------------------------------------------------------------------------------------------
#define FW_QRYS 64
#define FW_KEYS 64

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

  load_tile4(Qs, q + b * q_sb + h * q_sh, q_ss, q0, FW_QRYS, S);

  float m[4], l[4], acc[4][4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[r][e] = 0.0f;
  }

  // Causal: keys past the block's last query never contribute.
  const int k_end = causal ? min(S, q0 + FW_QRYS) : S;
  for (int k0 = 0; k0 < k_end; k0 += FW_KEYS) {
    __syncthreads();  // the previous tile's Ks/Vs/Ps are no longer read
    load_tile4(Ks, kp, k_ss, k0, FW_KEYS, S);
    load_tile4(Vs, vp, v_ss, k0, FW_KEYS, S);
    __syncthreads();

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
      l[r] = l[r] * alpha + row_sum16(psum);
      m[r] = m_new;
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][e] *= alpha;
    }
    __syncthreads();

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
