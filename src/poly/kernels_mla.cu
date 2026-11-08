// src/poly/kernels_mla.cu - GigaChat Audio's absorbed MLA (DeepSeek V2/V3 form: 32 heads, a 512-value latent
// and a 64-value rope key, the latent the value too).  mla_attend is the cuBLAS path. mla_attend_fused is the
// decode kernel. The Q/K packing and the fused kv_b products live here too.
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

cudaStream_t cs(void* s) { return (cudaStream_t) s; }
int grid(int n, int b) { return (n + b - 1) / b; }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Lets `kernel` take as much dynamic shared memory as the card gives a block, once a card.
template <class K>
size_t allow_smem(K kernel, unsigned& done) {
    int dev = 0, most = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || most <= 1024)
        throw std::runtime_error("mla: the card's shared memory a block is unknown");
    const size_t room = size_t(most) - 1024;
    if (dev < 32 && !(done & (1u << dev))) {
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(room)) != cudaSuccess)
            throw std::runtime_error("mla: a kernel cannot be given the card's shared memory");
        done |= 1u << dev;
    }
    return room;
}

constexpr int WARP = 32;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    return x;
}

// One value per thread summed (or the largest of them) over the block; `part` has a float a warp.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float x, float* part) {
    x = warp_max(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float half_at(const uint16_t* p, size_t i) {
    return __half2float(__ushort_as_half(p[i]));
}

// A Q8_0 cache's value: one fp16 scale a 32-value block.
__device__ __forceinline__ float q8_at(const int8_t* code, const uint16_t* scale, size_t i) {
    return float(code[i]) * __half2float(__ushort_as_half(scale[i >> 5]));
}

// A fused kv_b matrix's value: F32 as it is, BF16 as the top half of a float.
template <class W>
__device__ __forceinline__ float weight_at(const W* w, size_t i);
template <>
__device__ __forceinline__ float weight_at<float>(const float* w, size_t i) { return w[i]; }
template <>
__device__ __forceinline__ float weight_at<uint16_t>(const uint16_t* w, size_t i) {
    return __uint_as_float(uint32_t(w[i]) << 16);
}

constexpr int MLA_KEY_BLOCK = 16;   // cache rows one block of mla_keys_kernel copies
constexpr int MLA_THREADS = 256;

// The key rows a token of an MLA call attends: the latent cache rows 0..p, with the rope key after each latent.
// Rows past the token are zeroed, so a pass sized for a recorded launch computes zeros there rather than the
// cache's stale values.
__global__ void mla_keys_kernel(const uint16_t* __restrict__ lat, const int8_t* __restrict__ lat_code,
                                const uint16_t* __restrict__ lat_scale, const uint16_t* __restrict__ kpe,
                                const int8_t* __restrict__ kpe_code, const uint16_t* __restrict__ kpe_scale,
                                float* __restrict__ keys, int rank, int n_rot, int n_keys, long long pos, int first,
                                const long long* __restrict__ at) {
    const int tok = first + blockIdx.y;
    const long long p = at ? at[2 * tok] : pos + tok;
    const int n_sel = int(min((long long) n_keys, p + 1));
    const int kd = rank + n_rot;
    float* dst = keys + size_t(blockIdx.y) * n_keys * kd;
    const int j0 = blockIdx.x * MLA_KEY_BLOCK;
    for (int i = threadIdx.x; i < MLA_KEY_BLOCK * kd; i += blockDim.x) {
        const int j = j0 + i / kd, r = i % kd;
        if (j >= n_keys) continue;
        float v = 0.0f;
        if (j < n_sel) {
            if (r < rank) {
                const size_t idx = size_t(j) * rank + r;
                v = lat_code ? q8_at(lat_code, lat_scale, idx) : half_at(lat, idx);
            } else {
                const size_t idx = size_t(j) * n_rot + (r - rank);
                v = kpe_code ? q8_at(kpe_code, kpe_scale, idx) : half_at(kpe, idx);
            }
        }
        dst[size_t(j) * kd + r] = v;
    }
}

// One head of one token: the softmax over the keys the token sees.  The keys it does not see become zero, and
// the scores past them must not reach the max.
__global__ void mla_softmax_kernel(float* __restrict__ score, int n_keys, long long pos, int first,
                                   const long long* __restrict__ at) {
    __shared__ float part[8];
    const int tok = first + blockIdx.y;
    const long long p = at ? at[2 * tok] : pos + tok;
    const int n_sel = int(min((long long) n_keys, p + 1));
    float* row = score + (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * n_keys;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n_sel; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_sel; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    const float inv = 1.0f / block_sum(sum, part);
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) row[j] = j < n_sel ? row[j] * inv : 0.0f;
}

__global__ void mla_split_kv_kernel(const float* __restrict__ kv, float* __restrict__ lat,
                                    float* __restrict__ kpe, int rank, int n_rot, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int kd = rank + n_rot, t = i / kd, r = i % kd;
    if (r < rank) lat[size_t(t) * rank + r] = kv[i];
    else kpe[size_t(t) * n_rot + (r - rank)] = kv[i];
}

__global__ void mla_split_q_kernel(const float* __restrict__ q, float* __restrict__ qn,
                                   float* __restrict__ qpe, int nope, int n_rot, int n_head, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int kd = nope + n_rot;
    const int t = i / (n_head * kd), rem = i % (n_head * kd), h = rem / kd, r = rem % kd;
    if (r < nope) qn[(size_t(t) * n_head + h) * nope + r] = q[i];
    else qpe[(size_t(t) * n_head + h) * n_rot + (r - nope)] = q[i];
}

__global__ void mla_pack_q_kernel(const float* __restrict__ qabs, const float* __restrict__ qpe,
                                  float* __restrict__ q, int rank, int n_rot, int n_head, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int kd = rank + n_rot;
    const int t = i / (n_head * kd), rem = i % (n_head * kd), h = rem / kd, r = rem % kd;
    q[i] = r < rank ? qabs[(size_t(t) * n_head + h) * rank + r]
                    : qpe[(size_t(t) * n_head + h) * n_rot + (r - rank)];
}

// A block a head of a token.  qabs[t][h][l] = sum_i w[(h * group_stride + i) * rank + l] * qn[t][h][i].
template <class W>
__global__ void mla_absorb_fused_kernel(const float* __restrict__ qn, const W* __restrict__ w,
                                        float* __restrict__ qabs, int nope, int rank, int n_head, int group_stride) {
    extern __shared__ float sq[];
    const int h = blockIdx.x, t = blockIdx.y;
    const float* q = qn + (size_t(t) * n_head + h) * nope;
    for (int i = threadIdx.x; i < nope; i += blockDim.x) sq[i] = q[i];
    __syncthreads();
    float* out = qabs + (size_t(t) * n_head + h) * rank;
    for (int l = threadIdx.x; l < rank; l += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < nope; ++i) acc += sq[i] * weight_at(w, size_t(h * group_stride + i) * rank + l);
        out[l] = acc;
    }
}

// A block a head of a token.  vout[t][h][v] = sum_l w[(h * group_stride + nope + v) * rank + l] * kqv[t][h][l].
template <class W>
__global__ void mla_vb_fused_kernel(const float* __restrict__ kqv, const W* __restrict__ w,
                                    float* __restrict__ vout, int rank, int vd, int n_head, int group_stride,
                                    int nope) {
    extern __shared__ float sk[];
    const int h = blockIdx.x, t = blockIdx.y;
    const float* kv = kqv + (size_t(t) * n_head + h) * rank;
    for (int l = threadIdx.x; l < rank; l += blockDim.x) sk[l] = kv[l];
    __syncthreads();
    float* out = vout + (size_t(t) * n_head + h) * vd;
    for (int v = threadIdx.x; v < vd; v += blockDim.x) {
        const W* wr = w + size_t(h * group_stride + nope + v) * rank;
        float acc = 0.0f;
        for (int l = 0; l < rank; ++l) acc += sk[l] * weight_at(wr, l);
        out[v] = acc;
    }
}

// The decode kernel's shapes.  A 576-half row is 1152 B, a multiple of the 32 banks' 128 B, so every ldmatrix row
// of an unpadded tile lands on the same bank; 584 halfs (1168 B) shifts each row by four banks.
constexpr int MLA_RANK = 512, MLA_ROT = 64, MLA_HEAD = 32, MLA_DIM = MLA_RANK + MLA_ROT;
constexpr int MLA_TILE = 32;              // keys one tile holds
constexpr int MLA_STRIDE = MLA_DIM + 8;   // 584 halfs a query or key row in shared memory
constexpr int MLA_PSTRIDE = 40;           // 40 halfs a head's softmax row
constexpr int MLA_THREADS_F = 256;
constexpr size_t MLA_SC_BYTES = (size_t) MLA_HEAD * MLA_TILE * sizeof(float);
constexpr size_t MLA_SP_BYTES = (size_t) MLA_HEAD * MLA_PSTRIDE * sizeof(__half);
constexpr size_t MLA_SMEM = (size_t) MLA_HEAD * MLA_STRIDE * sizeof(__half) * 2 + MLA_SC_BYTES + MLA_SP_BYTES +
                             MLA_HEAD * 3 * sizeof(float);

__device__ __forceinline__ uint32_t sm_addr(const void* p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ld_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void ld_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}

__device__ __forceinline__ void ld_x4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void mma16816(float& d0, float& d1, float& d2, float& d3, const uint32_t* a,
                                         const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A 16-byte read the compiler leaves where it is: a tile ahead of its use, hidden behind the tile's products.
__device__ __forceinline__ void ld16(uint4& v, const void* p) {
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint16_t ld16b(const void* p) {
    uint16_t v;
    asm volatile("ld.global.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

// 32 int8 codes and their fp16 scale as 32 halfs.
__device__ __forceinline__ void deq32(const uint4& a, const uint4& b, __half scale, __half* dst) {
    const float d = __half2float(scale);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int8_t c0 = (int8_t) (w[i] & 0xff), c1 = (int8_t) ((w[i] >> 8) & 0xff);
        const int8_t c2 = (int8_t) ((w[i] >> 16) & 0xff), c3 = (int8_t) (w[i] >> 24);
        const __half2 h01 = __floats2half2_rn((float) c0 * d, (float) c1 * d);
        const __half2 h23 = __floats2half2_rn((float) c2 * d, (float) c3 * d);
        o[2 * i] = *(const uint32_t*) &h01;
        o[2 * i + 1] = *(const uint32_t*) &h23;
    }
    *(uint4*) dst = *(const uint4*) &o[0];
    *(uint4*) (dst + 8) = *(const uint4*) &o[4];
    *(uint4*) (dst + 16) = *(const uint4*) &o[8];
    *(uint4*) (dst + 24) = *(const uint4*) &o[12];
}

// QK: 32 heads x 576 values a token.  PV: the heads' softmax weights over the tile against the same tile's first
// 512 latent values.  A block covers all heads and one of `splits` key ranges; it writes the split's partial
// unnormalized sum and its (max, sum), and the combine kernel weighs the splits.
template <bool Q8>
__global__ __launch_bounds__(MLA_THREADS_F, 1)
void mla_decode_kernel(const float* __restrict__ q, const int8_t* __restrict__ lat_codes,
                       const uint16_t* __restrict__ lat_scales, const int8_t* __restrict__ kpe_codes,
                       const uint16_t* __restrict__ kpe_scales, const uint16_t* __restrict__ lat16,
                       const uint16_t* __restrict__ kpe16, float* __restrict__ part, float* __restrict__ stats,
                       float scale, int splits, int cap, long long pos, int first,
                       const long long* __restrict__ at) {
    extern __shared__ __align__(16) unsigned char mla_smem[];
    __half* sq = (__half*) mla_smem;                                  // 32 * MLA_STRIDE
    __half* sk = sq + MLA_HEAD * MLA_STRIDE;                          // 32 * MLA_STRIDE
    float* sc = (float*) (sk + MLA_TILE * MLA_STRIDE);                // the tile's raw scores, 32 * 32
    __half* sp = (__half*) (sc + MLA_HEAD * MLA_TILE);                // the softmax weights, 32 * 40
    float* sm = (float*) (sp + MLA_HEAD * MLA_PSTRIDE);
    float* sl = sm + MLA_HEAD;
    float* sa = sl + MLA_HEAD;

    const int split = blockIdx.x, row = blockIdx.y;
    const long long p = at ? at[2 * (first + row)] : pos + first + row;
    const int n_vis = (int) min(p + 1, (long long) cap);
    // The split ranges follow the keys this token actually sees, not the launch's `cap`: a recorded launch at
    // cap 8192 for a token at 4200 would otherwise put half its splits past the last key and leave the wall to
    // the fullest block.  `n_vis` is read from the card, so the grid stays fixed for a graph.
    const int per = (n_vis + splits - 1) / splits;
    const int kb0 = split * per, kb1 = min(n_vis, kb0 + per);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    if (kb0 >= kb1) {   // this split sees nothing: an empty partial for the combine
        if (lane == 0)
            for (int j = 0; j < 4; ++j) {
                const int h = warp * 4 + j;
                stats[((size_t) (split * gridDim.y + row) * MLA_HEAD + h) * 2 + 0] = -INFINITY;
                stats[((size_t) (split * gridDim.y + row) * MLA_HEAD + h) * 2 + 1] = 0.f;
            }
        return;
    }

    // the token's 32 * 576 query values, fp16, shared by every warp; rows padded against bank conflicts
    const float* qrow = q + (size_t) row * MLA_HEAD * MLA_DIM;
    for (int i = threadIdx.x; i < MLA_HEAD * MLA_DIM / 4; i += MLA_THREADS_F) {
        const int h = i / (MLA_DIM / 4), d = i % (MLA_DIM / 4);
        const float4 v = *(const float4*) (qrow + (size_t) i * 4);
        const __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
        uint32_t* o = (uint32_t*) (sq + (size_t) h * MLA_STRIDE + (size_t) d * 4);
        o[0] = *(const uint32_t*) &h0;
        o[1] = *(const uint32_t*) &h1;
    }
    for (int i = threadIdx.x; i < MLA_TILE * MLA_STRIDE / 2; i += MLA_THREADS_F)
        ((uint32_t*) sk)[i] = 0u;   // rows past the last valid key of a short tile stay finite
    for (int h = threadIdx.x; h < MLA_HEAD; h += MLA_THREADS_F) {
        sm[h] = -INFINITY;
        sl[h] = 0.f;
        sa[h] = 0.f;
    }

    // The next tile's codes are read into registers before this tile's products, and dequantized into shared
    // memory at the top of the next pass: the global read is off the critical path, and there is no staging copy.
    uint4 qc[2][2];
    __half qs[2];
    bool lok[2];
    uint4 kc8[2];
    __half ks8;
    bool kok;
    const auto prefetch = [&](int kb) {
        const int nr = min(MLA_TILE, cap - kb);
        if (nr <= 0) return;
        for (int j = 0; j < 2; ++j) {
            const int i = threadIdx.x + j * MLA_THREADS_F;
            lok[j] = i < nr * 16;
            if (!lok[j]) continue;
            const int r = i % nr, blk = i / nr;
            ld16(qc[j][0], lat_codes + (size_t) (kb + r) * MLA_RANK + blk * 32);
            ld16(qc[j][1], lat_codes + (size_t) (kb + r) * MLA_RANK + blk * 32 + 16);
            qs[j] = __ushort_as_half(ld16b(lat_scales + (size_t) (kb + r) * 16 + blk));
        }
        kok = threadIdx.x < nr * 2;
        if (kok) {
            const int r = threadIdx.x % nr, blk = threadIdx.x / nr;
            ld16(kc8[0], kpe_codes + (size_t) (kb + r) * MLA_ROT + blk * 32);
            ld16(kc8[1], kpe_codes + (size_t) (kb + r) * MLA_ROT + blk * 32 + 16);
            ks8 = __ushort_as_half(ld16b(kpe_scales + (size_t) (kb + r) * 2 + blk));
        }
    };
    const auto dequant = [&](int kb) {
        const int nr = min(MLA_TILE, cap - kb);
        for (int j = 0; j < 2; ++j) {
            if (!lok[j]) continue;
            const int i = threadIdx.x + j * MLA_THREADS_F;
            const int r = i % nr, blk = i / nr;
            deq32(qc[j][0], qc[j][1], qs[j], sk + r * MLA_STRIDE + blk * 32);
        }
        if (kok) {
            const int r = threadIdx.x % nr, blk = threadIdx.x / nr;
            deq32(kc8[0], kc8[1], ks8, sk + r * MLA_STRIDE + MLA_RANK + blk * 32);
        }
    };

    const int hg = warp >> 2, kg = warp & 3, h0 = hg * 16;
    const int lr = lane & 15, lc = (lane >> 4) & 1;
    const int nc = warp & 3, dim0 = nc * 128;
    const int r8 = lane >> 2, c8 = (lane & 3) * 2;
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
        for (int j = 0; j < 4; ++j) acc[n][j] = 0.f;

    if constexpr (Q8) prefetch(kb0);
    for (int kb = kb0; kb < kb1; kb += MLA_TILE) {
        const int kv = min(kb1 - kb, MLA_TILE);
        __syncthreads();   // the previous tile's value product is done with sk
        if constexpr (Q8) {
            dequant(kb);
            __syncthreads();
            if (kb + MLA_TILE < kb1) prefetch(kb + MLA_TILE);
        } else {
            for (int i = threadIdx.x; i < kv * 64; i += MLA_THREADS_F) {
                const int r = i >> 6, c = i & 63;
                *(uint4*) (sk + r * MLA_STRIDE + c * 8) =
                    *(const uint4*) (lat16 + (size_t) (kb + r) * MLA_RANK + c * 8);
            }
            for (int i = threadIdx.x; i < kv * 8; i += MLA_THREADS_F) {
                const int r = i >> 3, c = i & 7;
                *(uint4*) (sk + r * MLA_STRIDE + MLA_RANK + c * 8) =
                    *(const uint4*) (kpe16 + (size_t) (kb + r) * MLA_ROT + c * 8);
            }
            __syncthreads();
        }

        // QK: this warp's 16 heads against its 8 keys, 576 values in 16-value steps
        float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
#pragma unroll
        for (int kc = 0; kc < MLA_DIM / 16; ++kc) {
            uint32_t a[4], b[2];
            ld_x4(a[0], a[1], a[2], a[3], sm_addr(&sq[(h0 + lr) * MLA_STRIDE + kc * 16 + lc * 8]));
            ld_x2(b[0], b[1], sm_addr(&sk[(kg * 8 + (lane & 7)) * MLA_STRIDE + kc * 16 + ((lane >> 3) & 1) * 8]));
            mma16816(q0, q1, q2, q3, a, b);
        }
        sc[(h0 + r8) * MLA_TILE + kg * 8 + c8] = q0 * scale;
        sc[(h0 + r8) * MLA_TILE + kg * 8 + c8 + 1] = q1 * scale;
        sc[(h0 + r8 + 8) * MLA_TILE + kg * 8 + c8] = q2 * scale;
        sc[(h0 + r8 + 8) * MLA_TILE + kg * 8 + c8 + 1] = q3 * scale;
        __syncthreads();

        // softmax: a warp four heads, a lane a key of the tile
        for (int j = 0; j < 4; ++j) {
            const int h = warp * 4 + j;
            const float s = (lane < kv) ? sc[h * MLA_TILE + lane] : -INFINITY;
            float tmax = s;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, o));
            const float mold = sm[h];
            const float mnew = fmaxf(mold, tmax);
            const float alpha = mold == -INFINITY ? 0.f : expf(mold - mnew);
            const float pv = (lane < kv) ? expf(s - mnew) : 0.f;
            float sum = pv;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
            if (lane == 0) {
                sm[h] = mnew;
                sl[h] = sl[h] * alpha + sum;
                sa[h] = alpha;
            }
            sp[h * MLA_PSTRIDE + lane] = __float2half(pv);
        }
        __syncthreads();

        // the running sum shrinks with the new maximum, then this tile's weights meet the latent values
        const float a0 = sa[h0 + r8], a1 = sa[h0 + r8 + 8];
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            acc[n][0] *= a0, acc[n][1] *= a0;
            acc[n][2] *= a1, acc[n][3] *= a1;
        }
        uint32_t pa[2][4];
        ld_x4(pa[0][0], pa[0][1], pa[0][2], pa[0][3], sm_addr(&sp[(h0 + lr) * MLA_PSTRIDE + lc * 8]));
        ld_x4(pa[1][0], pa[1][1], pa[1][2], pa[1][3], sm_addr(&sp[(h0 + lr) * MLA_PSTRIDE + 16 + lc * 8]));
        // V is [key][dim] row-major; .trans turns 16 keys x 16 dims into the two dim groups' B fragments,
        // the addressing llama.cpp's fattn-mma-f16 uses for DKQ 576 (16 keys, two 8-dim groups).
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];
                ld_x4t(b[0], b[1], b[2], b[3],
                       sm_addr(&sk[(ks * 16 + (lane & 7) + 8 * ((lane >> 3) & 1)) * MLA_STRIDE + dim0 + n * 8 +
                                  8 * (lane >> 4)]));
                mma16816(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pa[ks], b);
                mma16816(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3], pa[ks], b + 2);
            }
    }

    // this split's unnormalized sum, and the (max, sum) the combine weighs it by
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        const int hA = h0 + r8, hB = h0 + r8 + 8;
        const int dA = dim0 + n * 8 + c8;
        float* pA = part + ((size_t) (split * gridDim.y + row) * MLA_HEAD + hA) * MLA_RANK + dA;
        float* pB = part + ((size_t) (split * gridDim.y + row) * MLA_HEAD + hB) * MLA_RANK + dA;
        pA[0] = acc[n][0];
        pA[1] = acc[n][1];
        pB[0] = acc[n][2];
        pB[1] = acc[n][3];
    }
    if (lane == 0)
        for (int j = 0; j < 4; ++j) {
            const int h = warp * 4 + j;
            stats[((size_t) (split * gridDim.y + row) * MLA_HEAD + h) * 2 + 0] = sm[h];
            stats[((size_t) (split * gridDim.y + row) * MLA_HEAD + h) * 2 + 1] = sl[h];
        }
}

// A warp a (token, head): the splits' partial sums, weighed by exp(max - the largest max) and divided by the same
// weights of their softmax sums.
// One warp a (row, head), a block a warp: a decode token has 32 heads, so 32 blocks spread the split merge over
// every SM instead of the four a 256-thread block left it on.  Each warp reads only the splits with keys (l > 0).
__global__ void mla_combine_kernel(const float* __restrict__ part, const float* __restrict__ stats,
                                   float* __restrict__ out, int rows, int splits) {
    __shared__ float se[64];
    const int lane = threadIdx.x & 31;
    const int wh = blockIdx.x;
    if (wh >= rows * MLA_HEAD) return;
    const int row = wh / MLA_HEAD, h = wh % MLA_HEAD;
    const size_t base = (size_t) row * MLA_HEAD + h;
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        const float l = stats[((size_t) s * rows * MLA_HEAD + base) * 2 + 1];
        if (l > 0.f) M = fmaxf(M, stats[((size_t) s * rows * MLA_HEAD + base) * 2]);
    }
    if (M == -INFINITY) {
        for (int d = lane; d < MLA_RANK; d += 32) out[base * MLA_RANK + d] = 0.f;
        return;
    }
    float den = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t at = (size_t) s * rows * MLA_HEAD + base;
        const float l = stats[at * 2 + 1];
        const float e = l > 0.f ? expf(stats[at * 2] - M) : 0.f;
        if (lane == 0) se[s] = e;
        den += l * e;
    }
    __syncwarp();
    for (int d = lane * 4; d < MLA_RANK; d += 128) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < splits; ++s) {
            const float e = se[s];
            if (e == 0.f) continue;
            const float4 v = *(const float4*) (part + ((size_t) s * rows * MLA_HEAD + base) * MLA_RANK + d);
            acc.x += v.x * e, acc.y += v.y * e, acc.z += v.z * e, acc.w += v.w * e;
        }
        const float inv = 1.f / den;
        *(float4*) (out + base * MLA_RANK + d) =
            make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
    }
}

// A block a head: y[h][o] = sum_i w[h][o][i] * x[h][i], w bf16.  The engine's gemv_kernel takes a block an output
// row; the absorbed MLA's W_UK is 512 rows of 128 values a head, so that is 16,384 blocks of 128 values each and the
// launch, not the weights, is what a token pays for.  Here 32 blocks read the same bytes with 256 threads each.
__global__ void head_gemv_bf16_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x,
                                      float* __restrict__ y, int n_in, int n_out, int parts) {
    extern __shared__ float sx[];   // n_in floats
    const int h = blockIdx.x / parts, part = blockIdx.x % parts;
    const int o0 = part * n_out / parts, o1 = (part + 1) * n_out / parts;
    const uint16_t* wh = w + (size_t) h * n_out * n_in;
    const float* xh = x + (size_t) h * n_in;
    float* yh = y + (size_t) h * n_out;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) sx[i] = xh[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    // Four outputs a warp: the four rows' loads are independent, so a warp has 4x the bytes on their way while it
    // sums, which a decode token's bandwidth needs (one row's 8 bytes a lane did not keep the card fed).
    for (int o = o0 + warp * 4; o < o1; o += nwarp * 4) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < 4 && o + j < o1; ++j) {
            const uint16_t* wr = wh + (size_t) (o + j) * n_in;
            for (int i = lane * 4; i < n_in; i += 128) {   // 8 bytes a lane: four bf16
                const uint2 v = *(const uint2*) (wr + i);
                acc[j] += __uint_as_float(v.x << 16) * sx[i];
                acc[j] += __uint_as_float(v.x & 0xffff0000u) * sx[i + 1];
                acc[j] += __uint_as_float(v.y << 16) * sx[i + 2];
                acc[j] += __uint_as_float(v.y & 0xffff0000u) * sx[i + 3];
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o2);
            if (lane == 0 && o + j < o1) yh[o + j] = acc[j];
        }
    }
}

}  // namespace

void mla_head_gemv_bf16(const void* w, const float* x, float* y, int n_head, int n_in, int n_out, void* stream) {
    if (n_head <= 0 || n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 || n_in > 1024) throw std::invalid_argument("mla_head_gemv_bf16: n_in is not 4 to 1024, in 4s");
    const int parts = n_out >= 4 * 32 ? 4 : 1;
    head_gemv_bf16_kernel<<<n_head * parts, 256, (size_t) n_in * sizeof(float), cs(stream)>>>(
        (const uint16_t*) w, x, y, n_in, n_out, parts);
    ck("mla_head_gemv_bf16");
}

// The same product with the cache in the file's own Q8_0 (a 2-byte scale and 32 codes a block) or Q6_K (a 256-value
// block: ql 128, qh 64, 16 scales, d).  A block a head, four outputs a warp, as the bf16 one.
__global__ void head_gemv_q8_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x,
                                    float* __restrict__ y, int n_in, int n_out, int parts) {
    extern __shared__ float sx[];
    const int h = blockIdx.x / parts, part = blockIdx.x % parts;
    const int o0 = part * n_out / parts, o1 = (part + 1) * n_out / parts;
    const uint8_t* wh = w + (size_t) h * n_out * (n_in / 32) * 34;
    const float* xh = x + (size_t) h * n_in;
    float* yh = y + (size_t) h * n_out;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) sx[i] = xh[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    for (int o = o0 + warp * 4; o < o1; o += nwarp * 4) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < 4 && o + j < o1; ++j) {
            const uint8_t* wr = wh + (size_t) (o + j) * (n_in / 32) * 34;
            for (int b = 0; b < n_in / 32; ++b) {
                const uint8_t* blk = wr + (size_t) b * 34;
                const float d = __half2float(*(const __half*) blk);
                acc[j] += (float) (int8_t) blk[2 + lane] * d * sx[b * 32 + lane];
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o2);
            if (lane == 0 && o + j < o1) yh[o + j] = acc[j];
        }
    }
}

__global__ void head_gemv_q6k_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x,
                                     float* __restrict__ y, int n_in, int n_out, int parts) {
    extern __shared__ float sx[];
    const int h = blockIdx.x / parts, part = blockIdx.x % parts;
    const int o0 = part * n_out / parts, o1 = (part + 1) * n_out / parts;
    const uint8_t* wh = w + (size_t) h * n_out * (n_in / 256) * 210;
    const float* xh = x + (size_t) h * n_in;
    float* yh = y + (size_t) h * n_out;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) sx[i] = xh[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    const int is = lane / 16;
    for (int o = o0 + warp * 4; o < o1; o += nwarp * 4) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < 4 && o + j < o1; ++j) {
            const uint8_t* wr = wh + (size_t) (o + j) * (n_in / 256) * 210;
            for (int b = 0; b < n_in / 256; ++b) {
                const uint8_t* blk = wr + (size_t) b * 210;
                const float d = __half2float(*(const __half*) (blk + 208));
                for (int half = 0; half < 2; ++half) {
                    const uint8_t* ql = blk + half * 64;
                    const uint8_t* qh = blk + 128 + half * 32;
                    const int8_t* sc = (const int8_t*) blk + 192 + half * 8;
                    const int q1 = ((ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
                    const int q2 = ((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
                    const int q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
                    const int q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
                    const int base = b * 256 + half * 128;
                    acc[j] += d * sc[is + 0] * q1 * sx[base + lane];
                    acc[j] += d * sc[is + 2] * q2 * sx[base + lane + 32];
                    acc[j] += d * sc[is + 4] * q3 * sx[base + lane + 64];
                    acc[j] += d * sc[is + 6] * q4 * sx[base + lane + 96];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o2);
            if (lane == 0 && o + j < o1) yh[o + j] = acc[j];
        }
    }
}

void mla_head_gemv_q8(const void* w, const float* x, float* y, int n_head, int n_in, int n_out, void* stream) {
    if (n_head <= 0 || n_in <= 0 || n_out <= 0) return;
    if (n_in % 32) throw std::invalid_argument("mla_head_gemv_q8: n_in is not a multiple of 32");
    const int parts = n_out >= 4 * 32 ? 4 : 1;
    head_gemv_q8_kernel<<<n_head * parts, 256, (size_t) n_in * sizeof(float), cs(stream)>>>(
        (const uint8_t*) w, x, y, n_in, n_out, parts);
    ck("mla_head_gemv_q8");
}

void mla_head_gemv_q6k(const void* w, const float* x, float* y, int n_head, int n_in, int n_out, void* stream) {
    if (n_head <= 0 || n_in <= 0 || n_out <= 0) return;
    if (n_in % 256) throw std::invalid_argument("mla_head_gemv_q6k: n_in is not a multiple of 256");
    const int parts = n_out >= 4 * 32 ? 4 : 1;
    head_gemv_q6k_kernel<<<n_head * parts, 256, (size_t) n_in * sizeof(float), cs(stream)>>>(
        (const uint8_t*) w, x, y, n_in, n_out, parts);
    ck("mla_head_gemv_q6k");
}

void mla_attend_fused(const float* q, const uint16_t* lat, const uint16_t* kpe, int64_t pos, int rows, float* out,
                      float* part, float* stats, int rank, int n_rot, int n_head, float scale, int splits, int cap,
                      void* stream, const long long* at, int first, Kvq8 lat_q8, Kvq8 kpe_q8) {
    if (rows <= 0) return;
    if (rank != MLA_RANK || n_rot != MLA_ROT || n_head != MLA_HEAD)
        throw std::invalid_argument("mla_attend_fused: only the 32-head 512+64 absorbed MLA is fused");
    if (splits < 1) splits = 1;
    if (splits > 64) splits = 64;
    const bool q8 = lat_q8.codes != nullptr && lat_q8.scales != nullptr && kpe_q8.codes != nullptr &&
                    kpe_q8.scales != nullptr;
    if (!q8 && (!lat || !kpe)) throw std::invalid_argument("mla_attend_fused: no cache to read");
    if (cap < 1) cap = 1;

    const dim3 blocks((unsigned) splits, (unsigned) rows);
    if (q8) {
        static unsigned done = 0;
        if (MLA_SMEM > allow_smem(mla_decode_kernel<true>, done))
            throw std::runtime_error("mla_attend_fused: the q8_0 tile does not fit the card's shared memory");
        mla_decode_kernel<true><<<blocks, MLA_THREADS_F, MLA_SMEM, cs(stream)>>>(
            q, lat_q8.codes, lat_q8.scales, kpe_q8.codes, kpe_q8.scales, nullptr, nullptr, part, stats, scale, splits,
            cap, pos, first, at);
    } else {
        static unsigned done = 0;
        if (MLA_SMEM > allow_smem(mla_decode_kernel<false>, done))
            throw std::runtime_error("mla_attend_fused: the fp16 tile does not fit the card's shared memory");
        mla_decode_kernel<false><<<blocks, MLA_THREADS_F, MLA_SMEM, cs(stream)>>>(
            q, nullptr, nullptr, nullptr, nullptr, lat, kpe, part, stats, scale, splits, cap, pos, first, at);
    }
    ck("mla_attend_fused");
    mla_combine_kernel<<<grid(rows * MLA_HEAD, 1), 32, 0, cs(stream)>>>(part, stats, out, rows, splits);
    ck("mla_attend_fused combine");
}


void blas_ck(cublasStatus_t s, const char* what) {
    if (s == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": cuBLAS error " + std::to_string((int) s));
}

void mla_attend(void* blas, const float* q, const uint16_t* lat, const uint16_t* kpe, int64_t pos, int rows,
                float* keys, float* score, int chunk, float* out, int rank, int n_rot, int n_head, float scale,
                void* stream, const long long* at, int room, Kvq8 lat_q8, Kvq8 kpe_q8) {
    if (rows <= 0) return;
    if (rank <= 0 || n_rot <= 0 || n_head <= 0 || chunk <= 0)
        throw std::invalid_argument("mla_attend: bad rank, n_rot, n_head or chunk");
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int kd = rank + n_rot;
    const long long head = (long long) n_head * kd, head_out = (long long) n_head * rank;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        // A recorded launch reads the tokens' positions on the card and is sized for `room` rows; otherwise the
        // last token of the pass sees the most keys.
        const int n_keys = at ? room : int(pos + first + c);
        if (n_keys <= 0) continue;
        const long long key_n = (long long) n_keys * kd, score_n = (long long) n_keys * n_head;
        mla_keys_kernel<<<dim3(grid(n_keys, MLA_KEY_BLOCK), c), MLA_THREADS, 0, cs(stream)>>>(
            lat, lat_q8.codes, lat_q8.scales, kpe, kpe_q8.codes, kpe_q8.scales, keys, rank, n_rot, n_keys,
            (long long) pos, first, at);
        ck("mla_attend keys");
        // score[head][key] = scale * (q[head] . key).  cuBLAS reads a buffer of rows as the transpose.
        blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, n_keys, n_head, kd, &scale, keys, kd, key_n,
                                          q + size_t(first) * n_head * kd, kd, head, &zero, score, n_keys, score_n, c),
                "mla_attend scores");
        mla_softmax_kernel<<<dim3(n_head, c), MLA_THREADS, 0, cs(stream)>>>(score, n_keys, (long long) pos, first,
                                                                          at);
        ck("mla_attend softmax");
        // out[head] = sum over the keys of score[head][key] * the key's first rank values
        blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, rank, n_head, n_keys, &one, keys, kd, key_n,
                                          score, n_keys, score_n, &zero, out + size_t(first) * head_out, rank,
                                          head_out, c),
                "mla_attend values");
    }
}

void mla_split_q(const float* q, float* qn, float* qpe, int nope, int n_rot, int n_head, int rows, void* stream) {
    const size_t total = (size_t) rows * n_head * (nope + n_rot);
    if (total == 0) return;
    mla_split_q_kernel<<<grid((int) total, MLA_THREADS), MLA_THREADS, 0, cs(stream)>>>(
        q, qn, qpe, nope, n_rot, n_head, (int) total);
    ck("mla_split_q");
}

void mla_pack_q(const float* qabs, const float* qpe, float* q, int rank, int n_rot, int n_head, int rows,
                void* stream) {
    const size_t total = (size_t) rows * n_head * (rank + n_rot);
    if (total == 0) return;
    mla_pack_q_kernel<<<grid((int) total, MLA_THREADS), MLA_THREADS, 0, cs(stream)>>>(
        qabs, qpe, q, rank, n_rot, n_head, (int) total);
    ck("mla_pack_q");
}

void mla_split_kv(const float* kv, float* lat, float* kpe, int rank, int n_rot, int rows, void* stream) {
    const size_t total = (size_t) rows * (rank + n_rot);
    if (total == 0) return;
    mla_split_kv_kernel<<<grid((int) total, MLA_THREADS), MLA_THREADS, 0, cs(stream)>>>(kv, lat, kpe, rank, n_rot,
                                                                                       (int) total);
    ck("mla_split_kv");
}

void mla_absorb_fused(const float* qn, const void* w, uint32_t type, float* qabs, int nope, int rank, int n_head,
                      int rows, int group_stride, void* stream) {
    if (rows <= 0) return;
    const dim3 blocks(n_head, rows);
    const size_t smem = (size_t) nope * sizeof(float);
    if (type == 0)
        mla_absorb_fused_kernel<float><<<blocks, MLA_THREADS, smem, cs(stream)>>>(
            qn, (const float*) w, qabs, nope, rank, n_head, group_stride);
    else if (type == 30)
        mla_absorb_fused_kernel<uint16_t><<<blocks, MLA_THREADS, smem, cs(stream)>>>(
            qn, (const uint16_t*) w, qabs, nope, rank, n_head, group_stride);
    else
        throw std::invalid_argument("mla_absorb_fused: the fused kv_b is not F32 or BF16");
    ck("mla_absorb_fused");
}

void mla_vb_fused(const float* kqv, const void* w, uint32_t type, float* vout, int rank, int vd, int n_head,
                  int rows, int group_stride, int nope, void* stream) {
    if (rows <= 0) return;
    const dim3 blocks(n_head, rows);
    const size_t smem = (size_t) rank * sizeof(float);
    if (type == 0)
        mla_vb_fused_kernel<float><<<blocks, MLA_THREADS, smem, cs(stream)>>>(
            kqv, (const float*) w, vout, rank, vd, n_head, group_stride, nope);
    else if (type == 30)
        mla_vb_fused_kernel<uint16_t><<<blocks, MLA_THREADS, smem, cs(stream)>>>(
            kqv, (const uint16_t*) w, vout, rank, vd, n_head, group_stride, nope);
    else
        throw std::invalid_argument("mla_vb_fused: the fused kv_b is not F32 or BF16");
    ck("mla_vb_fused");
}

}  // namespace strata::poly
