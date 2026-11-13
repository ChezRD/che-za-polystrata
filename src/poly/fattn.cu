// src/poly/fattn.cu - see the header.  The kernel is fattn_mma.cuh's, launched here in place of llama.cpp's
// launch_fattn(): one CUDA block a tile of 64 query rows (8 tokens x 8 heads, or 4 x 16), attending to a whole
// block of keys. The stream-k fixup is never needed; each tile leaves its softmax max and sum behind
// (fattn_mma.cuh, change 2).
#include "fattn.hpp"

#if !defined(GGML_USE_HIP)
#include "fattn_mma.cuh"
#endif

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strata::poly {

#if !defined(GGML_USE_HIP)
namespace {

constexpr int kCols = 64;

struct Setup {
    bool ok = false;
    int nwarps = 0, nbatch_fa = 0;
    size_t smem = 0;
};

int grid(long long n, int per) { return (int) ((n + per - 1) / per); }

// The launch's shape and shared memory as ggml_cuda_flash_attn_ext_mma_f16_case() works them out, for the
// current card.  Ampere or later: on Turing the kernel has no 64-column tile.
template <int D, int Group>
Setup make_setup(int cc) {
    constexpr int kCols1 = kCols / Group, kCols2 = Group;
    const auto kKernel = flash_attn_ext_f16<D, D, kCols1, kCols2, false, false, false>;
    Setup r;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) return r;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(D, D, kCols, cc);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(D, D, kCols, cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(D, D, kCols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(D, D, kCols, cc);
    const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(D, D, kCols, cc);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(D, D, kCols1, kCols2, cc);
    r.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, kCols, cc);
    r.nwarps = nthreads / WARP_SIZE;
    const int cols_per_warp = std::min(kCols, get_cols_per_warp(cc));
    // the tile's max and sum reach dst_meta only with one warp a query column (np == 1, fattn_mma.cuh change 2);
    // with more, the blocks of keys could not be merged
    if (r.nwarps * cols_per_warp != kCols) return r;
    const int stride_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_V = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);
    const size_t kv1 = size_t(r.nbatch_fa) * std::max(stride_K, stride_V) * sizeof(half2);
    const size_t kv2 = size_t(r.nbatch_fa) * (stride_K + stride_V) * sizeof(half2);
    const size_t q = size_t(kCols) * (D / 2 + 4) * sizeof(half2);
    const size_t mask = size_t(kCols1) * (r.nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t combine = size_t(r.nwarps) * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t kv = nstages <= 1 ? kv1 : kv2;
    r.smem = std::max(combine, q_in_reg ? std::max(q, kv + mask) : q + kv + mask);
    r.ok = cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) r.smem) == cudaSuccess;
    cudaGetLastError();
    return r;
}

// Each card's own, worked out the first time it asks: a layer split may put cards of different generations in one
// process, and cudaFuncSetAttribute() holds for the current card only.
template <int D, int Group>
const Setup& setup() {
    static std::mutex mu;
    static Setup s[GGML_CUDA_MAX_DEVICES];
    static bool done[GGML_CUDA_MAX_DEVICES] = {};
    const int d = ggml_cuda_get_device();
    std::lock_guard<std::mutex> lock(mu);
    if (!done[d]) {
        s[d] = make_setup<D, Group>(ggml_cuda_info().devices[d].cc);
        done[d] = true;
    }
    return s[d];
}

// The scratch for a batch of n tokens: a block's results and their max and sum, the merged max and sum, the
// causal block's steps, then K and V side by side, each piece 16-byte aligned (offsets in floats).  rows: the keys
// a block takes, a whole number of steps and at most a grid's 65535 rows (dequant_kernel); 0 when the scratch
// cannot hold a block of n + nbatch_fa rows.
struct Layout {
    size_t part = 0, pmeta = 0, ameta = 0, kv_max = 0, kv = 0;
    long long rows = 0;
};

template <int Group>
Layout layout(int n, int dim, int n_head, int kv_heads, int nbatch_fa, size_t scratch_floats) {
    Layout L;
    const size_t heads = size_t(n) * n_head;
    size_t at = 0;
    auto take = [&](size_t floats) {
        const size_t p = at;
        at += (floats + 3) & ~size_t(3);
        return p;
    };
    L.part = take(heads * dim);
    L.pmeta = take(2 * heads);
    L.ameta = take(2 * heads);
    L.kv_max = take((size_t) grid(n, kCols / Group));
    L.kv = at;
    if (at >= scratch_floats) return L;
    // a row of K and one of V take kv_heads * dim floats together
    const size_t width = size_t(kv_heads) * dim;
    const long long rows =
        std::min<long long>((long long) ((scratch_floats - at) / width), 65535) / nbatch_fa * nbatch_fa;
    if (rows >= (long long) n + nbatch_fa) L.rows = rows;
    return L;
}

template <int D, int Group>
bool fits(int n, int dim, int n_head, int kv_heads, size_t scratch_floats) {
    if (n <= 0 || dim != D || kv_heads <= 0 || n_head != Group * kv_heads) return false;
    const Setup& S = setup<D, Group>();
    return S.ok && layout<Group>(n, dim, n_head, kv_heads, S.nbatch_fa, scratch_floats).rows > 0;
}

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("fattn: ") + what + ": " + cudaGetErrorString(e));
}

// rows [row, row + valid) of a q8_0 cache (or an fp16 one, f16) as fp16, and zeros on to `blockIdx.y`'s last row:
// the kernel reads whole steps of keys, and the causal mask removes the zero rows' weights, but a NaN would get
// through it
__global__ void dequant_kernel(const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                               const half* __restrict__ f16, half* __restrict__ out, int width, long long row,
                               int valid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int r = blockIdx.y;
    half x = __float2half(0.0f);
    if (r < valid) {
        const size_t at = size_t(row + r) * width;
        x = f16 ? f16[at + i]
                : __float2half((float) codes[at + i] * __half2float(__ushort_as_half(scales[at / 32 + i / 32])));
    }
    out[size_t(r) * width + i] = x;
}

// rows [row, row + valid) of a Q4_0 cache as fp16, and zeros on to blockIdx.y's last row: 16 nibble bytes a block
// (element j in the low half of byte j for j < 16, the high half of byte j - 16 otherwise), and the scales in
// their own array.
__global__ void dequant_q4_kernel(const uint8_t* __restrict__ q4, const uint16_t* __restrict__ scales,
                                  half* __restrict__ out, int width, long long row, int valid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int r = blockIdx.y;
    half x = __float2half(0.0f);
    if (r < valid) {
        const size_t nb = (size_t) (width >> 5);
        const uint8_t* blk = q4 + ((size_t) (row + r) * nb + (i >> 5)) * 16;
        const float d = __half2float(__ushort_as_half(scales[(size_t) (row + r) * nb + (i >> 5)]));
        const uint8_t byte = blk[i & 15];
        const int q = (i & 16) ? (byte >> 4) : (byte & 0x0F);
        x = __float2half((float) (q - 8) * d);
    }
    out[size_t(r) * width + i] = x;
}

// The last step of keys a tile of tokens reads in the causal block: token j sees the keys up to j + delta.
template <int Cols1>
__global__ void kv_max_kernel(int* __restrict__ kv_max, int tiles, int n, long long delta, int keys, int step) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tiles) return;
    const int j = t * Cols1 + Cols1 - 1 < n ? t * Cols1 + Cols1 - 1 : n - 1;
    const long long seen = (j + delta + 1 + step - 1) / step * step;
    kv_max[t] = seen < keys ? (int) seen : keys;
}

// A block of keys' results into what the blocks before it made of a token and head: each normalized by its own
// softmax sum, so each weighs in by its sum against the larger max.  One CUDA block a token and head.
__global__ void merge_kernel(float* __restrict__ out, float2* __restrict__ meta, const float* __restrict__ part,
                             const float2* __restrict__ pmeta, int dim) {
    const size_t row = blockIdx.x;
    const float2 a = meta[row], b = pmeta[row];
    const float top = fmaxf(a.x, b.x), wa = a.y * __expf(a.x - top), wb = b.y * __expf(b.x - top);
    const float inv = 1.0f / (wa + wb);
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[row * dim + d] = (out[row * dim + d] * wa + part[row * dim + d] * wb) * inv;
    __syncthreads();
    if (threadIdx.x == 0) meta[row] = make_float2(top, wa + wb);
}

// The q8 cache (kq8, vq8) or the fp16 one (keys, vals): an fp16 cache's whole steps before pos are read where they
// lie, in one block, and only the last block is copied into the scratch.
template <int D, int Group, bool VQ4 = false>
bool run(const float* q, Kvq8 kq8, Kvq8 vq8, const half* keys, const half* vals, Kvq4 vq4, int64_t pos, int n,
         float* out, int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats,
         void* stream) {
    if (n <= 0 || !fits<D, Group>(n, dim, n_head, kv_heads, scratch_floats)) return false;
    constexpr int kCols1 = kCols / Group;
    const auto kKernel = flash_attn_ext_f16<D, D, kCols1, Group, false, false, false>;
    const Setup& S = setup<D, Group>();
    const int width = kv_heads * dim, tiles = grid(n, kCols1);
    const size_t heads = size_t(n) * n_head;
    const Layout L = layout<Group>(n, dim, n_head, kv_heads, S.nbatch_fa, scratch_floats);
    if (L.rows == 0) return false;
    const long long rows = L.rows;
    float* part = scratch + L.part;
    float2* pmeta = (float2*) (scratch + L.pmeta);
    float2* ameta = (float2*) (scratch + L.ameta);
    int* kv_max = (int*) (scratch + L.kv_max);
    half* kw = (half*) (scratch + L.kv);
    half* vw = kw + size_t(rows) * width;

    const cudaStream_t s = (cudaStream_t) stream;
    const dim3 block(WARP_SIZE, S.nwarps);
    const uint3 ne01 = init_fastdiv_values((uint64_t) n);
    const int32_t nb01 = n_head * dim * (int) sizeof(float), nb02 = dim * (int) sizeof(float);
    const int32_t nb11 = width * (int) sizeof(half), nb12 = dim * (int) sizeof(half);
    const long long total = pos + n;
    const long long whole = pos / S.nbatch_fa * S.nbatch_fa;   // an fp16 cache's rows read in place
    for (long long k0 = 0, b = 0; k0 < total; ++b) {
        // Blocks of whole steps from the start, as long as what is left is more than a block; then the rest,
        // which holds the batch's own rows and starts at or before pos, so that every token sees a key of it.
        const bool last = keys ? k0 == whole : total - k0 <= rows;
        const long long m = last ? total - k0
                          : keys ? whole
                                 : std::min<long long>(rows, (pos - k0) / S.nbatch_fa * S.nbatch_fa);
        const int nk = grid(m, S.nbatch_fa) * S.nbatch_fa;
        const long long delta = pos - k0;
        const half* kb = kw;
        const half* vb = vw;
        if (keys && !last) {
            kb = keys, vb = vals;
        } else {
            dequant_kernel<<<dim3(grid(width, 256), nk), 256, 0, s>>>(kq8.codes, kq8.scales, keys, kw, width, k0,
                                                                      (int) m);
            if constexpr (VQ4)
                dequant_q4_kernel<<<dim3(grid(width, 256), nk), 256, 0, s>>>(vq4.q, vq4.scales, vw, width, k0,
                                                                             (int) m);
            else
                dequant_kernel<<<dim3(grid(width, 256), nk), 256, 0, s>>>(vq8.codes, vq8.scales, vals, vw, width, k0,
                                                                          (int) m);
        }
        if (last) kv_max_kernel<kCols1><<<grid(tiles, 128), 128, 0, s>>>(kv_max, tiles, n, delta, nk, S.nbatch_fa);
        float* dst = b == 0 ? out : part;
        float2* meta = b == 0 ? ameta : pmeta;
        kKernel<<<tiles * kv_heads, block, S.smem, s>>>(
            (const char*) q, (const char*) kb, (const char*) vb, nullptr, nullptr, last ? kv_max : nullptr, dst, meta,
            scale, 0.0f, 1.0f, 1.0f, 1u, 0.0f,
            dim, ne01, n_head, 1, nb01, nb02, nb01 * n,
            dim, nk, kv_heads, 1, nb11, nb12, (int64_t) nb11 * nk,
            nb11, nb12, (int64_t) nb11 * nk,
            n, 1, 1, (int32_t) (2 * delta), 0, 0);
        ck("attention");
        if (b > 0) merge_kernel<<<(unsigned) heads, dim, 0, s>>>(out, ameta, part, pmeta, dim);
        k0 += m;
    }
    ck("merge");
    return true;
}

// ---- decode (fdec_q8): a few tokens over the whole cache, split by keys, on the tensor cores ----

constexpr int kSplit = 1024;   // keys a CUDA block takes, counted from the cache's first row
constexpr int kTile = 64;      // keys a block takes at a time: 16 a warp
constexpr int kDecRows = 32;   // 4 tokens x 8 heads, or 2 x 16; the scratch has the same size

// The groups the decode kernel is instantiated for: 8 and 16 were the only two while every model it served was
// one of those; 6 (Qwen3.8-27B dense) and 12 (Qwen3.8-Flash) are the same template with another Group, and 4 keeps
// the dim-64 shapes.
//
// WHAT THE KERNEL ACTUALLY REQUIRES, which is weaker than it looks: rows are numbered `token * Group + head` and a
// launch holds at most Q = MT * 16 of them (MT is 1 while the tokens' rows fit 16, else 2), so a launch carries
// kDecRows / Group tokens and `row < Q <= kDecRows` keeps every split's scratch apart.  So any Group <= kDecRows
// works; the group does NOT have to divide kDecRows, it only has to divide into it at least once.  Groups 6 and 12
// carry 5 and 2 tokens a launch against 8's 4 and 16's 2, which is a slightly shorter batch and nothing else.
bool fdec_group_ok_impl(int n_head, int kv_heads, int tokens) {
    if (kv_heads <= 0 || n_head % kv_heads != 0) return false;
    const int g = n_head / kv_heads;
    return (g == 4 || g == 6 || g == 8 || g == 12 || g == 16) && tokens * g <= 2 * kDecRows && kDecRows / g >= 1;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// Two int8 of `w` (each with its top bit flipped) as a half2, exactly: 0x64XX is 1024 + XX.  sel 0x4140 takes
// bytes 0 and 1, 0x4342 bytes 2 and 3.
__device__ __forceinline__ uint32_t i8pair(uint32_t w, uint32_t sel) {
    uint32_t x = __byte_perm(w, 0x64646464u, sel);
    half2 h = __hsub2(*reinterpret_cast<half2*>(&x), __float2half2_rn(1152.0f));
    return *reinterpret_cast<uint32_t*>(&h);
}

// Two six-bit codes of a 12-bit field as a half2, exactly, by the q8 path's trick: a byte 0x00..0x3F in 0x64XX is
// 1024 + XX, so one HADD2 of -1056 turns a pair into (code - 32), Kvq6's offset binary.  `g` holds the two codes in
// its bits 0..11, so the low byte of g >> 6 is the second one with two junk bits above it, and one LOP3 clears both.
// Five SASS instructions a pair: SHF, PRMT, LOP3, PRMT, HADD2.
__device__ __forceinline__ uint32_t q6pair(uint32_t g) {
    const uint32_t p = __byte_perm(g, g >> 6, 0x4040) & 0x3F3F3F3Fu;   // [c0, c1, c0, c1] with their two bits gone
    const uint32_t m = __byte_perm(p, 0x64646464u, 0x4140);            // [c0, 0x64, c1, 0x64]: two halves of 0x64XX
    half2 h = __hsub2(*reinterpret_cast<const half2*>(&m), __float2half2_rn(1056.0f));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t mul2(uint32_t x, half2 s) {
    half2 h = __hmul2(*reinterpret_cast<half2*>(&x), s);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint4 widen(uint2 v) { return make_uint4(v.x, v.y, 0u, 0u); }

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i < 2 ? (i == 0 ? v.x : v.y) : (i == 2 ? v.z : v.w);
}

// One CUDA block: the keys [split * kSplit, +kSplit) of group blockIdx.y for the launch's tokens (MT * 16 / Group
// at most, row = token * Group + head of the group), as flash attention: per tile of 64 keys, a warp scores 16
// (S = Q K^T, mma m16n8k16, the queries as halves, the key codes exact as halves and each 32-value block's scale
// applied to its sum in floats), the softmax max and sum go on per row, and a warp adds 64 of the head's value
// dimensions (O += P V, the weights and the values as halves).  It leaves the rows' unnormalized results with
// their max and sum; fdec_merge_kernel adds the splits.  A row's arithmetic never depends on the other rows: a
// token comes out the same to the bit whatever tokens the launch has beside it.  The dimensions are numbered as
// suits the reads (a thread's 8 values of a block of K, its 8 dimensions of V), which the sums don't see.  F16: an
// fp16 cache (kc and vc are halves, no scales), read as it is.
// VM: 0 both fp16, 1 both q8_0, 2 q8_0 keys with q4_0 values (K8V4, the values dequantized in registers), 3 six-bit
// keys with q4_0 values (K6V4): the keys' 24-byte block is two uint32 of the lane's 6 bytes and the pairs are
// decoded where q8's codes are, everything below the score loop the same as VM 2.
template <int D, int W, int MT, int VM, int Group>
__global__ void __launch_bounds__(W * WARP_SIZE)
fdec_kernel(const float* __restrict__ q, const void* __restrict__ kc, const uint16_t* __restrict__ ks,
            const void* __restrict__ vc, const uint16_t* __restrict__ vs, long long pos,
            const long long* __restrict__ at, int first, int tokens, int n_head, int kv_heads, float scale,
            float* __restrict__ part, float* __restrict__ ml) {
    constexpr bool F16 = VM == 0, VQ4 = VM == 2 || VM == 3, KQ6 = VM == 3;
    constexpr int NT = 2, NB = D / 32, TILE = D == 256 ? kTile : 32, KT = TILE / 16, Q = MT * 16, QS = D + 8, PS = TILE + 8;
    __shared__ __align__(16) half Qs[Q * QS];
    __shared__ __align__(16) half Ps[Q * PS];
    __shared__ float red[2][W][Q];
    const int w = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE, g8 = lane / 4, t = lane % 4;
    const int grp = blockIdx.y, split = blockIdx.x;
    const long long p0 = (at ? *at : pos) + first, last = p0 + tokens - 1, r0 = (long long) split * kSplit;
    if (r0 > last) return;
    const long long r1 = r0 + kSplit < last + 1 ? r0 + kSplit : last + 1;
    for (int i = threadIdx.x; i < Q * D; i += blockDim.x) {
        const int row = i / D, d = i % D, tok = row / Group;
        const float x = tok < tokens ? q[((size_t) (first + tok) * n_head + grp * Group + row % Group) * D + d] : 0.f;
        Qs[row * QS + d] = __float2half(x);
    }
    constexpr int E = F16 ? 2 : 1;                         // bytes a value
    constexpr uint32_t kZero = F16 ? 0u : (KQ6 ? 0u : 0x80808080u);   // a zero: q6's is 0 (code 0 = -32), q8's 0x80
    const size_t width = (size_t) kv_heads * D, swidth = width / 32;
    const char* kg = (const char*) kc + (size_t) grp * (KQ6 ? (D / 32) * 24 : D * E);
    const size_t krow = KQ6 ? (size_t) (width / 32) * 24 : width;   // bytes a key row of codes
    const int vd0 = 64 * w + 8 * g8;   // the warp's first value of the group: 64 a warp, so 4 warps for 256, 2 for 128
    const char* vg = VQ4 ? (const char*) vc + ((size_t) grp * (D / 32) + vd0 / 32) * 16
                         : (const char*) vc + ((size_t) grp * D + vd0) * E;
    const size_t vstride = VQ4 ? (size_t) kv_heads * (D / 32) * 16 : width * E;
    const uint16_t* ksg = F16 ? nullptr : ks + (size_t) grp * (D / 32);
    const uint16_t* vsg = VM >= 1 ? vs + (size_t) grp * (D / 32) + vd0 / 32 : nullptr;
    const size_t vswidth = (size_t) kv_heads * (D / 32);
    const int vwithin = vd0 % 32;
    float m[MT][2], l[MT][2], o[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        m[mt][0] = m[mt][1] = -INFINITY;
        l[mt][0] = l[mt][1] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.f;
    }
    __syncthreads();
    for (long long tb = r0; tb < r1; tb += TILE) {
        // the reads first, all of them: K of the warp's 16 keys (n-tile nt: key 8 nt + g8; a thread 8 values of
        // each 32-value block), their scales (keys 8 nt + 2t + j), V of all 64 keys for the warp's dimensions.
        // q8 fills .x and .y (8 codes), fp16 all four words (8 halves).
        uint4 kx[NT][8];
        static_assert(NB <= 8, "a key row holds at most 8 scales of 32 values");
        uint16_t ksc[NT][2][NB] = {};
        uint4 vx[KT][4];
        uint16_t vsc[KT][4] = {};
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const long long key = tb + 16 * w + 8 * nt + g8;
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                if constexpr (KQ6) {
                    // the lane's 8 codes are 6 bytes at offset 6t of the 24-byte block, read as two 4-byte words at
                    // the 4-aligned address below with the 0 or 2 leftover bytes shifted out; .x is codes 0..3 in
                    // bits 0..23 and .y codes 4..7, so the score loop's pair split is the same two shifts as q8's
                    const char* kb = kg + key * krow + 24 * b + ((6 * t) & ~3);
                    uint32_t u0 = 0u, u1 = 0u;   // kZero: code 0 is -32, finite, and the mask removes the key
                    if (key <= last) {
                        u0 = *reinterpret_cast<const uint32_t*>(kb);   // 4-byte aligned: 24b and 6t & ~3 are
                        u1 = *reinterpret_cast<const uint32_t*>(kb + 4);
                    }
                    const uint32_t s = (t & 1) ? 16u : 0u;
                    const uint32_t f0 = __funnelshift_r(u0, u1, s), f1 = __funnelshift_r(u1, 0u, s);
                    kx[nt][b] = make_uint4(f0, __funnelshift_r(f0, f1, 24), 0u, 0u);
                } else {
                    kx[nt][b] = key > last ? make_uint4(kZero, kZero, kZero, kZero)
                              : F16 ? *reinterpret_cast<const uint4*>(kg + (key * width + 32 * b + 8 * t) * E)
                                    : widen(*reinterpret_cast<const uint2*>(kg + key * width + 32 * b + 8 * t));
                }
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const long long k2 = tb + 16 * w + 8 * nt + 2 * t + j;
                if (!F16 && k2 <= last)
                    for (int b = 0; b < NB; ++b) ksc[nt][j][b] = ksg[k2 * swidth + b];
            }
        }
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const long long key = tb + 16 * kk + 2 * t + (i & 1) + 8 * (i >> 1);
                if (key > last) {
                    const uint32_t z = VQ4 ? 0u : kZero;
                    vx[kk][i] = make_uint4(z, z, z, z);
                } else if constexpr (VQ4) {
                    // 8 nibbles of the block, one 8-byte aligned read: element j in the low half of byte j
                    // (j < 16), the high half otherwise, so one shift picks the 8 the warp's dimensions need.
                    // Then the q8 read's trick: 0x64XX as a half is 1024 + XX, so XOR 0x80 and one subtract turn
                    // a code into (code - 8) in halves, four at a time.
                    const uint8_t* blk = reinterpret_cast<const uint8_t*>(vg) + key * vstride;
                    const float d = __half2float(__ushort_as_half(vsg[key * vswidth]));
                    const uint2 raw = *reinterpret_cast<const uint2*>(blk + (vwithin & 15));
                    const uint32_t sh = vwithin < 16 ? 0u : 4u;
                    const uint32_t w0 = (raw.x >> sh) & 0x0F0F0F0Fu, w1 = (raw.y >> sh) & 0x0F0F0F0Fu;
                    const half2 d2 = __float2half2_rn(d);
                    auto pair = [d2](uint32_t w, uint32_t sel) {
                        uint32_t x = __byte_perm(w ^ 0x80808080u, 0x64646464u, sel);
                        half2 h = __hsub2(*reinterpret_cast<half2*>(&x), __float2half2_rn(1160.0f));
                        return __hmul2(h, d2);
                    };
                    const half2 h0 = pair(w0, 0x4140), h1 = pair(w0, 0x4342);
                    const half2 h2 = pair(w1, 0x4140), h3 = pair(w1, 0x4342);
                    vx[kk][i] = make_uint4(*reinterpret_cast<const uint32_t*>(&h0),
                                           *reinterpret_cast<const uint32_t*>(&h1),
                                           *reinterpret_cast<const uint32_t*>(&h2),
                                           *reinterpret_cast<const uint32_t*>(&h3));
                } else {
                    vx[kk][i] = F16 ? *reinterpret_cast<const uint4*>(vg + key * width * E)
                                    : widen(*reinterpret_cast<const uint2*>(vg + key * width));
                }
                if constexpr (VM == 1)
                    if (key <= last) vsc[kk][i] = vsg[key * swidth];
            }
        // S = Q K^T, a 32-value block (two k-steps) at a time, then times the keys' scales of the block
        float s[MT][NT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) s[mt][nt][0] = s[mt][nt][1] = s[mt][nt][2] = s[mt][nt][3] = 0.f;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            float c[MT][NT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) c[mt][nt][0] = c[mt][nt][1] = c[mt][nt][2] = c[mt][nt][3] = 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = 32 * b + 8 * t + 4 * h;   // the thread's 4 dimensions of the step
                uint32_t bk[2][2];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    if constexpr (F16) {
                        bk[nt][0] = h ? kx[nt][b].z : kx[nt][b].x;
                        bk[nt][1] = h ? kx[nt][b].w : kx[nt][b].y;
                    } else if constexpr (KQ6) {
                        const uint32_t word = h ? kx[nt][b].y : kx[nt][b].x;   // 4 codes, 24 bits
                        bk[nt][0] = q6pair(word);
                        bk[nt][1] = q6pair(word >> 12);
                    } else {
                        const uint32_t word = (h ? kx[nt][b].y : kx[nt][b].x) ^ 0x80808080u;
                        bk[nt][0] = i8pair(word, 0x4140);
                        bk[nt][1] = i8pair(word, 0x4342);
                    }
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const uint2 lo = *reinterpret_cast<const uint2*>(&Qs[(mt * 16 + g8) * QS + d]);
                    const uint2 hi = *reinterpret_cast<const uint2*>(&Qs[(mt * 16 + g8 + 8) * QS + d]);
                    const uint32_t a[4] = {lo.x, hi.x, lo.y, hi.y};
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) mma16816(c[mt][nt], a, bk[nt][0], bk[nt][1]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float sc = F16 ? 1.f : __half2float(__ushort_as_half(ksc[nt][j][b]));
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        s[mt][nt][j] += c[mt][nt][j] * sc;
                        s[mt][nt][2 + j] += c[mt][nt][2 + j] * sc;
                    }
                }
        }
        // the causal mask, the tile's max per row over the four warps, the new max
        float alpha[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r, tok = row / Group;
                float mx = -INFINITY;
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const long long key = tb + 16 * w + 8 * nt + 2 * t + j;
                        float& x = s[mt][nt][2 * r + j];
                        x = tok < tokens && key <= p0 + tok ? x * scale : -INFINITY;
                        mx = fmaxf(mx, x);
                    }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                if (t == 0) red[0][w][row] = mx;
            }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r;
                float top = m[mt][r];
#pragma unroll
                for (int k = 0; k < W; ++k) top = fmaxf(top, red[0][k][row]);
                alpha[mt][r] = top == -INFINITY ? 1.f : expf(m[mt][r] - top);
                m[mt][r] = top;
                float sum = 0.f;
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const float x0 = s[mt][nt][2 * r], x1 = s[mt][nt][2 * r + 1];
                    const half2 p = __floats2half2_rn(x0 == -INFINITY ? 0.f : expf(x0 - top),
                                                      x1 == -INFINITY ? 0.f : expf(x1 - top));
                    sum += __low2float(p) + __high2float(p);
                    *reinterpret_cast<half2*>(&Ps[row * PS + 16 * w + 8 * nt + 2 * t]) = p;
                }
                sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                if (t == 0) red[1][w][row] = sum;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    o[mt][j][2 * r] *= alpha[mt][r];
                    o[mt][j][2 * r + 1] *= alpha[mt][r];
                }
            }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r;
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < W; ++k) sum += red[1][k][row];
                l[mt][r] = l[mt][r] * alpha[mt][r] + sum;
            }
        // O += P V: k-step kk is keys 16 kk.. of the tile; n-tile j column g8 is dimension 64 w + 8 g8 + j
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            uint32_t a[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const half* p = &Ps[(mt * 16 + g8) * PS + 16 * kk + 2 * t];
                a[mt][0] = *reinterpret_cast<const uint32_t*>(p);
                a[mt][1] = *reinterpret_cast<const uint32_t*>(p + 8 * PS);
                a[mt][2] = *reinterpret_cast<const uint32_t*>(p + 8);
                a[mt][3] = *reinterpret_cast<const uint32_t*>(p + 8 * PS + 8);
            }
            if constexpr (F16 || VQ4) {
                // n-tile j: half j % 2 of word j / 2 of the four keys' 8 halves
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t sel = j & 1 ? 0x7632 : 0x5410;
                    const uint32_t b0 = __byte_perm(word(vx[kk][0], j / 2), word(vx[kk][1], j / 2), sel);
                    const uint32_t b1 = __byte_perm(word(vx[kk][2], j / 2), word(vx[kk][3], j / 2), sel);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) mma16816(o[mt][j], a[mt], b0, b1);
                }
                continue;
            }
            const half2 s01 = __halves2half2(__ushort_as_half(vsc[kk][0]), __ushort_as_half(vsc[kk][1]));
            const half2 s89 = __halves2half2(__ushort_as_half(vsc[kk][2]), __ushort_as_half(vsc[kk][3]));
#pragma unroll
            for (int wd = 0; wd < 2; ++wd) {
                const uint32_t x0 = (wd ? vx[kk][0].y : vx[kk][0].x) ^ 0x80808080u;
                const uint32_t x1 = (wd ? vx[kk][1].y : vx[kk][1].x) ^ 0x80808080u;
                const uint32_t x2 = (wd ? vx[kk][2].y : vx[kk][2].x) ^ 0x80808080u;
                const uint32_t x3 = (wd ? vx[kk][3].y : vx[kk][3].x) ^ 0x80808080u;
                const uint32_t lo01 = __byte_perm(x0, x1, 0x5140), hi01 = __byte_perm(x0, x1, 0x7362);
                const uint32_t lo89 = __byte_perm(x2, x3, 0x5140), hi89 = __byte_perm(x2, x3, 0x7362);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t sel = jj & 1 ? 0x4342 : 0x4140;
                    const uint32_t b0 = mul2(i8pair(jj < 2 ? lo01 : hi01, sel), s01);
                    const uint32_t b1 = mul2(i8pair(jj < 2 ? lo89 : hi89, sel), s89);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) mma16816(o[mt][4 * wd + jj], a[mt], b0, b1);
                }
            }
        }
        __syncthreads();   // Ps and red are the next tile's
    }
    // the rows' results: c0 of n-tile j is dimension 64 w + 16 t + j, c1 the one 8 on
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = mt * 16 + g8 + 8 * r;
            if (row / Group >= tokens) continue;
            const size_t idx = ((size_t) split * kv_heads + grp) * kDecRows + row;
            float4* dst = reinterpret_cast<float4*>(part + idx * D + 64 * w + 16 * t);
            dst[0] = make_float4(o[mt][0][2 * r], o[mt][1][2 * r], o[mt][2][2 * r], o[mt][3][2 * r]);
            dst[1] = make_float4(o[mt][4][2 * r], o[mt][5][2 * r], o[mt][6][2 * r], o[mt][7][2 * r]);
            dst[2] = make_float4(o[mt][0][2 * r + 1], o[mt][1][2 * r + 1], o[mt][2][2 * r + 1], o[mt][3][2 * r + 1]);
            dst[3] = make_float4(o[mt][4][2 * r + 1], o[mt][5][2 * r + 1], o[mt][6][2 * r + 1], o[mt][7][2 * r + 1]);
            if (w == 0 && t == 0) {
                ml[2 * idx] = m[mt][r];
                ml[2 * idx + 1] = l[mt][r];
            }
        }
}

// The splits of a token and head added in their order, each by its sum against the largest max.  One CUDA block
// a head and token, a thread a dimension.  A token reads only the splits up to its own row.
template <int D, int Group>
__global__ void fdec_merge_kernel(const float* __restrict__ part, const float* __restrict__ ml, float* __restrict__ out,
                                  long long pos, const long long* __restrict__ at, int first, int n_head, int kv_heads,
                                  const float* __restrict__ sinks) {
    const int h = blockIdx.x, tok = blockIdx.y, d = threadIdx.x, grp = h / Group;
    const long long p = (at ? *at : pos) + first + tok;
    const int splits = int(p / kSplit) + 1, row = tok * Group + h % Group;
    float top = -INFINITY;
    for (int i = 0; i < splits; ++i) top = fmaxf(top, ml[2 * (((size_t) i * kv_heads + grp) * kDecRows + row)]);
    float sum = 0.f, acc = 0.f;
    for (int i = 0; i < splits; ++i) {
        const size_t idx = ((size_t) i * kv_heads + grp) * kDecRows + row;
        const float e = expf(ml[2 * idx] - top);
        sum += ml[2 * idx + 1] * e;
        acc += part[idx * D + d] * e;
    }
    // a head's sink: one more score with no value of its own, as attend() and the flash kernel count it
    if (sinks) {
        const float sk = sinks[h];
        const float nt = fmaxf(top, sk);
        const float scale = top == -INFINITY ? 0.f : expf(top - nt);
        sum = sum * scale + expf(sk - nt);
        acc *= scale;
    }
    out[((size_t) (first + tok) * n_head + h) * D + d] = acc / sum;
}

// splits of 1,024 keys up to `rows`, each with its results for kDecRows query rows and a max and a sum each
template <int D>
size_t fdec_floats(int rows, int kv_heads) { return (size_t) grid(rows, kSplit) * kv_heads * kDecRows * (D + 2); }

}  // namespace

// fdec is on by default; STRATA_FDEC=0 turns it off (the models' own start position stays with them).
inline bool fdec_allowed() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_FDEC");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}

bool fattn_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats) {
    if ((dim != 128 && dim != 256) || kv_heads <= 0 || rows <= 0) return false;
    if (dim == 128) {
        if (n_head == 8 * kv_heads) return fits<128, 8>(rows, dim, n_head, kv_heads, scratch_floats);
        if (n_head == 16 * kv_heads) return fits<128, 16>(rows, dim, n_head, kv_heads, scratch_floats);
        return false;
    }
    if (n_head == 8 * kv_heads) return fits<256, 8>(rows, dim, n_head, kv_heads, scratch_floats);
    if (n_head == 16 * kv_heads) return fits<256, 16>(rows, dim, n_head, kv_heads, scratch_floats);
    return false;
}

bool fattn_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, float* out, int dim, int n_head, int kv_heads,
              float scale, float* scratch, size_t scratch_floats, void* stream) {
    if (!kq8.codes || !vq8.codes) return false;
    if (dim == 128)
        return n_head == 16 * kv_heads
                   ? run<128, 16>(q, kq8, vq8, nullptr, nullptr, Kvq4{}, pos, n, out, dim, n_head, kv_heads, scale,
                                  scratch, scratch_floats, stream)
                   : run<128, 8>(q, kq8, vq8, nullptr, nullptr, Kvq4{}, pos, n, out, dim, n_head, kv_heads, scale,
                                 scratch, scratch_floats, stream);
    if (dim == 256)
        return n_head == 16 * kv_heads
                   ? run<256, 16>(q, kq8, vq8, nullptr, nullptr, Kvq4{}, pos, n, out, dim, n_head, kv_heads, scale,
                                  scratch, scratch_floats, stream)
                   : run<256, 8>(q, kq8, vq8, nullptr, nullptr, Kvq4{}, pos, n, out, dim, n_head, kv_heads, scale,
                                 scratch, scratch_floats, stream);
    return false;
}

bool fattn_k8v4(const float* q, Kvq8 kq8, Kvq4 vq4, int64_t pos, int n, float* out, int dim, int n_head,
                int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream) {
    if (!kq8.codes || !vq4.q || !vq4.scales) return false;
    if (dim == 128)
        return n_head == 16 * kv_heads
                   ? run<128, 16, true>(q, kq8, Kvq8{}, nullptr, nullptr, vq4, pos, n, out, dim, n_head, kv_heads,
                                        scale, scratch, scratch_floats, stream)
                   : run<128, 8, true>(q, kq8, Kvq8{}, nullptr, nullptr, vq4, pos, n, out, dim, n_head, kv_heads,
                                       scale, scratch, scratch_floats, stream);
    if (dim == 256)
        return n_head == 16 * kv_heads
                   ? run<256, 16, true>(q, kq8, Kvq8{}, nullptr, nullptr, vq4, pos, n, out, dim, n_head, kv_heads,
                                        scale, scratch, scratch_floats, stream)
                   : run<256, 8, true>(q, kq8, Kvq8{}, nullptr, nullptr, vq4, pos, n, out, dim, n_head, kv_heads,
                                       scale, scratch, scratch_floats, stream);
    return false;
}

bool fattn_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, float* out, int dim,
               int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream) {
    if (!keys || !vals) return false;
    if (dim == 128)
        return n_head == 16 * kv_heads
                   ? run<128, 16>(q, Kvq8{}, Kvq8{}, (const half*) keys, (const half*) vals, Kvq4{}, pos, n, out, dim,
                                  n_head, kv_heads, scale, scratch, scratch_floats, stream)
                   : run<128, 8>(q, Kvq8{}, Kvq8{}, (const half*) keys, (const half*) vals, Kvq4{}, pos, n, out, dim,
                                 n_head, kv_heads, scale, scratch, scratch_floats, stream);
    if (dim == 256)
        return n_head == 16 * kv_heads
                   ? run<256, 16>(q, Kvq8{}, Kvq8{}, (const half*) keys, (const half*) vals, Kvq4{}, pos, n, out, dim,
                                  n_head, kv_heads, scale, scratch, scratch_floats, stream)
                   : run<256, 8>(q, Kvq8{}, Kvq8{}, (const half*) keys, (const half*) vals, Kvq4{}, pos, n, out, dim,
                                 n_head, kv_heads, scale, scratch, scratch_floats, stream);
    return false;
}

bool fdec_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats) {
    // The decode kernel's group (query heads a KV head) is a template parameter whose value only has to satisfy
    // any Group <= kDecRows that divides into it at least once (see fdec_group_ok_impl for why that is enough).
    if ((dim != 128 && dim != 256) || kv_heads <= 0 || !fdec_group_ok_impl(n_head, kv_heads, 1) || rows <= 0) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) return false;
    const size_t need = dim == 128 ? fdec_floats<128>(rows, kv_heads) : fdec_floats<256>(rows, kv_heads);
    return need <= scratch_floats;
}

namespace {

template <int D, int VM, int Group>
bool run_dec(const float* q, const void* kc, const uint16_t* ks, const void* vc, const uint16_t* vs, int64_t pos, int n,
             const long long* at, int room, float* out, int dim, int n_head, int kv_heads, float scale, float* scratch,
             size_t scratch_floats, void* stream, const float* sinks) {
    constexpr int W = D == 256 ? 4 : 2;
    const int rows = (int) (at ? room : pos + n);   // a recorded launch is sized for `room` rows
    if (n <= 0 || !kc || !vc || !fdec_fits(dim, n_head, kv_heads, rows, scratch_floats)) return false;
    const int splits = grid(rows, kSplit);
    float* ml = scratch + (size_t) splits * kv_heads * kDecRows * D;
    const cudaStream_t s = (cudaStream_t) stream;
    constexpr int kDecTokens = kDecRows / Group;
    for (int first = 0; first < n; first += kDecTokens) {
        const int c = std::min(kDecTokens, n - first);
        const dim3 blocks(splits, kv_heads);
        const auto kernel = c * Group <= 16 ? fdec_kernel<D, W, 1, VM, Group> : fdec_kernel<D, W, 2, VM, Group>;
        kernel<<<blocks, W * WARP_SIZE, 0, s>>>(q, kc, ks, vc, vs, (long long) pos, at, first, c, n_head, kv_heads,
                                                        scale, scratch, ml);
        fdec_merge_kernel<D, Group><<<dim3(n_head, c), D, 0, s>>>(scratch, ml, out, (long long) pos, at, first, n_head,
                                                         kv_heads, sinks);
    }
    ck("decode attention");
    return true;
}

}  // namespace

bool fdec_group_ok(int n_head, int kv_heads, int tokens) { return fdec_group_ok_impl(n_head, kv_heads, tokens); }

// The group dispatch, written once instead of twice per format: every entry point differs only in vm, and the two
// dimensions and five groups are the same five instantiations the three formats share.
template <int D, int VM>
bool run_dec_grp(int g, const float* q, const void* kc, const uint16_t* ks, const void* vc, const uint16_t* vs,
                 int64_t pos, int n, const long long* at, int room, float* out, int n_head, int kv_heads, float scale,
                 float* scratch, size_t scratch_floats, void* stream, const float* sinks) {
    switch (g) {
        case 4: return run_dec<D, VM, 4>(q, kc, ks, vc, vs, pos, n, at, room, out, D, n_head, kv_heads, scale, scratch,
                                         scratch_floats, stream, sinks);
        case 6: return run_dec<D, VM, 6>(q, kc, ks, vc, vs, pos, n, at, room, out, D, n_head, kv_heads, scale, scratch,
                                         scratch_floats, stream, sinks);
        case 8: return run_dec<D, VM, 8>(q, kc, ks, vc, vs, pos, n, at, room, out, D, n_head, kv_heads, scale, scratch,
                                         scratch_floats, stream, sinks);
        case 12: return run_dec<D, VM, 12>(q, kc, ks, vc, vs, pos, n, at, room, out, D, n_head, kv_heads, scale,
                                           scratch, scratch_floats, stream, sinks);
        case 16: return run_dec<D, VM, 16>(q, kc, ks, vc, vs, pos, n, at, room, out, D, n_head, kv_heads, scale,
                                           scratch, scratch_floats, stream, sinks);
        default: return false;
    }
}

bool fdec_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, const long long* at, int room, float* out,
             int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream, const float* sinks) {
    if (!fdec_allowed()) return false;
    if (!kq8.scales || !vq8.scales) return false;
    if (!fdec_group_ok_impl(n_head, kv_heads, 1)) return false;
    const int g = n_head / kv_heads;
    if (dim == 128) return run_dec_grp<128, 1>(g, q, kq8.codes, kq8.scales, vq8.codes, vq8.scales, pos, n, at, room,
                                               out, n_head, kv_heads, scale, scratch, scratch_floats, stream, sinks);
    if (dim == 256) return run_dec_grp<256, 1>(g, q, kq8.codes, kq8.scales, vq8.codes, vq8.scales, pos, n, at, room,
                                               out, n_head, kv_heads, scale, scratch, scratch_floats, stream, sinks);
    return false;
}

bool fdec_k8v4(const float* q, Kvq8 kq8, Kvq4 vq4, int64_t pos, int n, const long long* at, int room, float* out,
               int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream, const float* sinks) {
    if (!fdec_allowed()) return false;
    if (!kq8.scales || !vq4.q || !vq4.scales) return false;
    if (!fdec_group_ok_impl(n_head, kv_heads, 1)) return false;
    const int g = n_head / kv_heads;
    if (dim == 128) return run_dec_grp<128, 2>(g, q, kq8.codes, kq8.scales, vq4.q, vq4.scales, pos, n, at, room, out,
                                               n_head, kv_heads, scale, scratch, scratch_floats, stream, sinks);
    if (dim == 256) return run_dec_grp<256, 2>(g, q, kq8.codes, kq8.scales, vq4.q, vq4.scales, pos, n, at, room, out,
                                               n_head, kv_heads, scale, scratch, scratch_floats, stream, sinks);
    return false;
}

bool fdec_k6v4(const float* q, Kvq6 kq6, Kvq4 vq4, int64_t pos, int n, const long long* at, int room, float* out,
               int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream, const float* sinks) {
    if (!fdec_allowed()) return false;
    if (!kq6.q6 || !kq6.scales || !vq4.q || !vq4.scales) return false;
    if (!fdec_group_ok_impl(n_head, kv_heads, 1)) return false;
    const int g = n_head / kv_heads;
    if (dim == 128) return run_dec_grp<128, 3>(g, q, kq6.q6, kq6.scales, vq4.q, vq4.scales, pos, n, at, room, out,
                                               n_head, kv_heads, scale, scratch, scratch_floats, stream, sinks);
    if (dim == 256) return run_dec_grp<256, 3>(g, q, kq6.q6, kq6.scales, vq4.q, vq4.scales, pos, n, at, room, out,
                                               n_head, kv_heads, scale, scratch, scratch_floats, stream, sinks);
    return false;
}

bool fdec_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, const long long* at,
              int room, float* out, int dim, int n_head, int kv_heads, float scale, float* scratch,
              size_t scratch_floats, void* stream, const float* sinks) {
    if (!fdec_group_ok_impl(n_head, kv_heads, 1)) return false;
    const int g = n_head / kv_heads;
    if (dim == 128) return run_dec_grp<128, 0>(g, q, keys, nullptr, vals, nullptr, pos, n, at, room, out, n_head,
                                               kv_heads, scale, scratch, scratch_floats, stream, sinks);
    if (dim == 256) return run_dec_grp<256, 0>(g, q, keys, nullptr, vals, nullptr, pos, n, at, room, out, n_head,
                                               kv_heads, scale, scratch, scratch_floats, stream, sinks);
    return false;
}

#else

bool fattn_fits(int, int, int, int, size_t) { return false; }

bool fdec_fits(int, int, int, int, size_t) { return false; }

bool fdec_q8(const float*, Kvq8, Kvq8, int64_t, int, const long long*, int, float*, int, int, int, float, float*, size_t,
             void*) {
    return false;
}

bool fdec_f16(const float*, const uint16_t*, const uint16_t*, int64_t, int, const long long*, int, float*, int, int, int,
              float, float*, size_t, void*) {
    return false;
}

bool fdec_k8v4(const float*, Kvq8, Kvq4, int64_t, int, const long long*, int, float*, int, int, int, float, float*,
               size_t, void*) {
    return false;
}

bool fdec_k6v4(const float*, Kvq6, Kvq4, int64_t, int, const long long*, int, float*, int, int, int, float, float*,
               size_t, void*) {
    return false;
}

bool fattn_q8(const float*, Kvq8, Kvq8, int64_t, int, float*, int, int, int, float, float*, size_t, void*) {
    return false;
}

bool fattn_k8v4(const float*, Kvq8, Kvq4, int64_t, int, float*, int, int, int, float, float*, size_t, void*) {
    return false;
}

bool fattn_f16(const float*, const uint16_t*, const uint16_t*, int64_t, int, float*, int, int, int, float, float*,
               size_t, void*) {
    return false;
}

#endif

}  // namespace strata::poly
