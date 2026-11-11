// The 64-key reference for bench/micro/attn128_bench.cu: the production 32-key kernel with KEYS=64, kept here and
// not in the engine.  Built with nvcc; exposes one C entry point.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace bench64 {
constexpr int WARP = 32;
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750)
#define STRATA_POLY_MMA 0
#else
#define STRATA_POLY_MMA 1
#endif

#if STRATA_POLY_MMA
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t half2_bits(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// 16 bytes from global to shared memory, or zeros where `whole` is false; asynchronous from Ampere on (cp.async: a
// thread's copies are a group from flash_commit() on, waited for by flash_wait()), a plain load and store on Turing.
__device__ __forceinline__ void flash_copy(half* dst, const half* src, bool whole) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(whole ? 16 : 0));
#else
    *reinterpret_cast<uint4*>(dst) = whole ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void flash_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until at most `ahead` of the thread's groups are still on their way.
template <int ahead> __device__ __forceinline__ void flash_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(ahead));
#endif
}
#endif

// attend_batch() for heads of 64 values in groups of 4 (LFM2's), the way FlashAttention-2 goes: a launch block takes
// FLASH_TOKENS tokens of a group, 64 rows of queries with a token's 4 heads side by side, a warp 16 of them, and walks
// the keys FLASH_KEYS at a time on the tensor cores, keeping each row's softmax as it goes, so no score leaves the
// registers.  The queries and the weights are rounded to halves and the products sum floats, as with the cuBLAS
// products.  Fragment layouts are PTX's for mma.m16n8k16: a thread holds rows gid and gid + 8, columns 2 tig and
// 2 tig + 1 of each block of 8.  The next block of keys and values is copied in while one is worked on, two buffers
// in turn.
constexpr int FLASH_TOKENS = 16;
__global__ void __launch_bounds__(4 * WARP)
attn_flash64_kernel(const float* __restrict__ q, const half* __restrict__ kcache, const half* __restrict__ vcache,
                    float* __restrict__ out, int rows, int n_head, int kv_heads, long long pos, float scale_log2) {
#if STRATA_POLY_MMA
    // FLASH_ROW: a cache row in shared memory, in halves; 16-byte aligned, and ldmatrix's eight rows on different
    // banks.
    constexpr int DIM = 64, G = 4, FLASH_KEYS = 64, FLASH_ROW = 64 + 8;
    __shared__ __align__(16) half ks[2][FLASH_KEYS][FLASH_ROW], vs[2][FLASH_KEYS][FLASH_ROW];
    const int g = blockIdx.y, t0 = blockIdx.x * FLASH_TOKENS, lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int gid = lane / 4, tig = lane % 4;
    int tok[2], head[2];   // the thread's rows gid and gid + 8 of the warp's
    for (int i = 0; i < 2; ++i) {
        const int r = w * 16 + gid + 8 * i;
        tok[i] = t0 + r / G;
        head[i] = g * G + r % G;
    }
    uint32_t qa[4][4];   // a block of 16 values a step, in log2 units
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (tok[i] < rows)
                    x = *reinterpret_cast<const float2*>(q + (size_t(tok[i]) * n_head + head[i]) * DIM + 16 * k +
                                                         8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = half2_bits(x.x * scale_log2, x.y * scale_log2);
            }
    float o[8][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    const int n_keys = int(pos) + min(t0 + FLASH_TOKENS, rows);   // the block's last token attends rows [0, n_keys)
    const auto fetch = [&](int buf, int k0) {
        for (int i = threadIdx.x; i < FLASH_KEYS * DIM / 8; i += 4 * WARP) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const bool whole = k0 + r < n_keys;
            const size_t at = whole ? (size_t(k0 + r) * kv_heads + g) * DIM + c : 0;
            flash_copy(&ks[buf][r][c], kcache + at, whole);
            flash_copy(&vs[buf][r][c], vcache + at, whole);
        }
        flash_commit();
    };
    fetch(0, 0);
    for (int k0 = 0, buf = 0; k0 < n_keys; k0 += FLASH_KEYS, buf ^= 1) {
        __syncthreads();   // the buffer the last block was worked on in is free
        if (k0 + FLASH_KEYS < n_keys) {
            fetch(buf ^ 1, k0 + FLASH_KEYS);
            flash_wait<1>();
        } else {
            flash_wait<0>();
        }
        __syncthreads();
        float s[8][4];
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < 4; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        for (int i = 0; i < 2; ++i) {
            const int own = int(pos) + tok[i];   // the row's token attends rows [0, own]
            float mx = -INFINITY;
            for (int n = 0; n < 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    if (k0 + 8 * n + 2 * tig + j % 2 > own) s[n][j] = -INFINITY;
                    mx = fmaxf(mx, s[n][j]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = exp2f(m[i] - base);
            m[i] = now;
            l[i] *= f;
            for (int n = 0; n < 8; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = exp2f(s[n][j] - base);
                    l[i] += s[n][j];
                }
            }
        }
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 8; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (tok[i] >= rows) continue;
        float* y = out + (size_t(tok[i]) * n_head + head[i]) * DIM + 2 * tig;
        for (int n = 0; n < 8; ++n)
            *reinterpret_cast<float2*>(y + 8 * n) = make_float2(o[n][2 * i] / l[i], o[n][2 * i + 1] / l[i]);
    }
#else
    (void) q, (void) kcache, (void) vcache, (void) out, (void) rows, (void) n_head, (void) kv_heads, (void) pos,
        (void) scale_log2;   // a card without the tensor cores' steps never launches it (tensor_cores())
#endif
}

// attend_batch() for heads of 128 values in groups of 8 (Flint's), the way attn_flash64_kernel goes, with a window
// over a ring of rows and a sink a head: a launch block takes FLASH128_TOKENS tokens of a group, 64 rows of queries
// with a token's 8 heads side by side, a warp two tokens, and walks the keys a step at a time.  The walk starts at a
// multiple of the step, so a token's keys go in the same blocks whatever tokens the call has beside it: a block it
// attends none of changes none of its sums.  A step of 32 keys takes 34 KiB of the two buffers, so two blocks fit an
// SM; 64 takes 68, past the 48 a block has without asking.
constexpr int FLASH128_TOKENS = 16, FLASH128_KEYS = 64, FLASH128_ROW = 128 + 8;
constexpr int FLASH128_SMEM = 2 * 2 * FLASH128_KEYS * FLASH128_ROW * int(sizeof(half));

// 32 keys a step keeps the registers for two blocks an SM (the second number of __launch_bounds__).
__global__ void __launch_bounds__(8 * WARP, 1)
flash128_64_kernel(const float* __restrict__ q, const half* __restrict__ kcache, const half* __restrict__ vcache,
                     float* __restrict__ out, int rows, int n_head, int kv_heads, long long pos, float scale_log2,
                     int window, int ring, const float* __restrict__ sinks) {
#if STRATA_POLY_MMA
    constexpr int DIM = 128, G = 8, KEYS = FLASH128_KEYS, NS = KEYS / 8, KV = KEYS / 16, ROW = FLASH128_ROW;
    extern __shared__ __align__(16) half flash_sm[];
    half(*ks)[KEYS][ROW] = reinterpret_cast<half(*)[KEYS][ROW]>(flash_sm);
    half(*vs)[KEYS][ROW] = reinterpret_cast<half(*)[KEYS][ROW]>(flash_sm + 2 * KEYS * ROW);
    const int g = blockIdx.y, t0 = blockIdx.x * FLASH128_TOKENS, lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int gid = lane / 4, tig = lane % 4, head = g * G + gid;
    const int tok[2] = {t0 + 2 * w, t0 + 2 * w + 1};   // the thread's rows gid and gid + 8: a head of two tokens
    uint32_t qa[8][4];   // a block of 16 values a step, in log2 units
    for (int k = 0; k < 8; ++k)
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (tok[i] < rows)
                    x = *reinterpret_cast<const float2*>(q + (size_t(tok[i]) * n_head + head) * DIM + 16 * k +
                                                         8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = half2_bits(x.x * scale_log2, x.y * scale_log2);
            }
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = pos + min(t0 + FLASH128_TOKENS, rows);
    const long long lo = window > 0 ? max(0LL, pos + t0 - window + 1) : 0;
    const auto fetch = [&](int buf, long long k0) {
        for (int i = threadIdx.x; i < KEYS * DIM / 8; i += blockDim.x) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;   // the others are zeros, which no weight reads
            const size_t at = whole ? (size_t(ring > 0 ? key % ring : key) * kv_heads + g) * DIM + c : 0;
            flash_copy(&ks[buf][r][c], kcache + at, whole);
            flash_copy(&vs[buf][r][c], vcache + at, whole);
        }
        flash_commit();
    };
    const long long first = lo / KEYS * KEYS;
    fetch(0, first);
    int buf = 0;
    for (long long k0 = first; k0 < hi; k0 += KEYS, buf ^= 1) {
        __syncthreads();   // the buffer the last block was worked on in is free
        if (k0 + KEYS < hi) {
            fetch(buf ^ 1, k0 + KEYS);
            flash_wait<1>();
        } else {
            flash_wait<0>();
        }
        __syncthreads();
        float s[NS][4];
        for (int n = 0; n < NS; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < DIM / 16; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        for (int i = 0; i < 2; ++i) {
            const long long own = pos + tok[i];   // the row's token attends positions (own - window, own]
            float mx = -INFINITY;
            for (int n = 0; n < NS; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const long long key = k0 + 8 * n + 2 * tig + j % 2;
                    if (key > own || (window > 0 && key <= own - window)) s[n][j] = -INFINITY;
                    mx = fmaxf(mx, s[n][j]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = exp2f(m[i] - base);
            m[i] = now;
            l[i] *= f;
            for (int n = 0; n < 16; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
            }
            for (int n = 0; n < NS; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = exp2f(s[n][j] - base);
                    l[i] += s[n][j];
                }
        }
        for (int k = 0; k < KV; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        float f = 1.0f;
        if (sinks) {   // the sink, in log2 units like the scores, as one more weight of no value
            const float sink = sinks[head] * 1.44269504f, now = fmaxf(m[i], sink);
            f = exp2f(m[i] - now);
            l[i] = l[i] * f + exp2f(sink - now);
        }
        if (tok[i] >= rows) continue;
        float* y = out + (size_t(tok[i]) * n_head + head) * DIM + 2 * tig;
        for (int n = 0; n < 16; ++n)
            *reinterpret_cast<float2*>(y + 8 * n) = make_float2(o[n][2 * i] * f / l[i], o[n][2 * i + 1] * f / l[i]);
    }
#else
    (void) q, (void) kcache, (void) vcache, (void) out, (void) rows, (void) n_head, (void) kv_heads, (void) pos,
        (void) scale_log2, (void) window, (void) ring, (void) sinks;   // never launched without the tensor cores
#endif
}


}  // namespace bench64

extern "C" void bench_flash128_64(const float* q, const uint16_t* kcache, const uint16_t* vcache, int rows, int pos,
                                  float* out, int n_head, int kv_heads, float scale, void* stream) {
    using namespace bench64;
    const cudaError_t e =
        cudaFuncSetAttribute(flash128_64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FLASH128_SMEM);
    if (e != cudaSuccess) return;
    flash128_64_kernel<<<dim3((rows + FLASH128_TOKENS - 1) / FLASH128_TOKENS, kv_heads), 8 * WARP, FLASH128_SMEM,
                         (cudaStream_t) stream>>>(q, (const half*) kcache, (const half*) vcache, out, rows, n_head,
                                                  kv_heads, (long long) pos, scale * 1.44269504f, 0, 0, nullptr);
}
