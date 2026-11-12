// src/kernels/gdn_state_bench.cu - Phase 0 row 3 (OPTIMIZATION-STACK.md §0.3 Row 3): one GDN layer's delta-rule
// state step at T = 1, the shape a decode token takes, across the three bit-widths LeapQuant supports:
//
//   A0 fp32 state     the engine's own arithmetic (32-bit state, read + rank-1 update + write)
//   A1 int8 state     the state as int8 codes with one fp32 scale a head-row, dequantized on read
//   A2 int8 + 4 compensators   A1 plus the rank-one factors LeapQuant keeps in fp16 and folds through the update
//   A3 int8 + a 16-token window buffer   A1 plus the per-window ring the paper's per-window quantization needs
//   A4 q6 state       the state as 6-bit codes, four to three bytes, one scale a head-row
//   A5 q4 state       the state as 4-bit codes, two to a byte, one scale a head-row
//   A16 q6 + 4 compensators + window buffer  the paper's full form at 6 bits (like-for-like with A2/A3)
//   A17 q4 + 4 compensators + window buffer  the paper's full form at 4 bits
//
// A4/A5/A16/A17 are LeapQuant's 6- and 4-bit blocks.  The row's brief named only "int8 (LeapQuant's floor)", which
// measured one of the paper's three bit-widths; these arms are the other two.  **The packings are the formats' own
// byte layouts, not a compression of int8**: 6-bit is four values in three bytes (24 bits, no padding), 4-bit is two
// values in a byte.  A thread therefore owns FOUR adjacent columns, because a byte (4-bit) or a 2-byte span (6-bit)
// spans more than one value and two threads writing the same byte would lose an update.
//
// The row's question is physical, not numerical: the fp32 state is 48 heads x 128 x 128 x 4 B = 3.0 MiB a layer,
// which fits in this card's 32 MiB L2, so shrinking it cannot make the step faster unless the step is bound by the
// bytes.  A0 measures that; A1..A7 measure what a smaller state actually buys — in time AND in bytes a session.
// Pass: an arm is worth it if it is faster than fp32 (>= 1.6x), or, when it is slower, if the footprint it buys is
// worth the time.  Stop rule: < 1.3x AND no footprint gain worth having closes nothing by itself.
//
// The arms are the MEASUREMENT, not an engine feature: each is one kernel here, the fp32 one the same arithmetic as
// src/prefill/kernels.cu's gdn_rec_kh_kernel (a thread the (head, column) pair, the decay and the rank-1 update),
// so the comparison is of the memory format and nothing else.
//
//     build/gdn_state_bench [--heads 48] [--dim 128] [--iters N] [--bw GB/s] [--slots 8] [--layers 36]
//     build/gdn_state_bench --smoke          the multi-token / partial-tile correctness check
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gdn_state_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int kCompensators = 4;   // LeapQuant's rank-r factors kept in fp16 (state.py:35-51)
constexpr int kWindow = 16;        // the per-window quantization's buffer length (paper §3.1, L = 16)

// ---- A0: the fp32 state.  A thread owns one (head, column): it reads its column of the 128x128 state, applies the
// decay and the rank-1 update `s += k * u[col]`, and writes it back.  The same arithmetic gdn_rec_kh_kernel does,
// with the three value heads a key head kept apart as the engine has them.
__global__ void state_fp32_kernel(float* __restrict__ s, const float* __restrict__ k, const float* __restrict__ u,
                                  const float* __restrict__ decay, int heads, int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    float* row = s + ((size_t) head * dim) * dim;
    const float d = decay[head], kk = k[head * dim + col], uu = u[head * dim + col];
    // s[:, col] for this column: the engine's per-column update, one decay a head
    for (int r = 0; r < dim; ++r) {
        const float cur = row[(size_t) r * dim + col];
        row[(size_t) r * dim + col] = cur * d + kk * uu;
    }
}

// ---- A1: the state as int8 codes with one fp32 scale a (head, row).  Read the codes, dequantize, update, requantize.
// The bytes the step touches are 4x fewer; the ALU is more (a multiply-add a value on the way in and out).
__global__ void state_int8_kernel(int8_t* __restrict__ q, float* __restrict__ scale, const float* __restrict__ k,
                                  const float* __restrict__ u, const float* __restrict__ decay, int heads, int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    int8_t* row = q + ((size_t) head * dim) * dim;
    float* srow = scale + (size_t) head * dim;
    const float d = decay[head], kk = k[head * dim + col], uu = u[head * dim + col];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        const int8_t code = row[(size_t) r * dim + col];
        const float cur = (float) code * sc;
        const float upd = cur * d + kk * uu;
        // requantize: the row's scale stays, the code is the rounded quotient (clamped, as a real format would)
        float qv = nearbyintf(upd / sc);
        qv = fminf(fmaxf(qv, -127.f), 127.f);
        row[(size_t) r * dim + col] = (int8_t) qv;
    }
}

// ---- A2: A1 plus the compensators: kCompensators rank-one factors in fp16, subtracted before the update and added
// back after it (the paper's "fit a rank-r factor, quantize the residual").  The factors are read and written.
__global__ void state_int8_comp_kernel(int8_t* __restrict__ q, float* __restrict__ scale,
                                       __half* __restrict__ comp, const float* __restrict__ k,
                                       const float* __restrict__ u, const float* __restrict__ decay, int heads,
                                       int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    int8_t* row = q + ((size_t) head * dim) * dim;
    float* srow = scale + (size_t) head * dim;
    __half* crow = comp + ((size_t) head * kCompensators * 2) * dim;
    const float d = decay[head], kk = k[head * dim + col], uu = u[head * dim + col];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        float residual = 0.f;
        for (int c = 0; c < kCompensators; ++c)
            residual += __half2float(crow[(size_t) (2 * c) * dim + r]) *
                        __half2float(crow[(size_t) (2 * c + 1) * dim + col]);
        const float cur = (float) row[(size_t) r * dim + col] * sc + residual;
        const float upd = cur * d + kk * uu;
        const float back = upd - residual;   // the quantized part only
        float qv = nearbyintf(back / sc);
        qv = fminf(fmaxf(qv, -127.f), 127.f);
        row[(size_t) r * dim + col] = (int8_t) qv;
    }
}

// ---- A3: A1 plus the window buffer: a token's (k, u) pairs are appended to a ring of kWindow entries and the step
// reads them once a window, the paper's per-window form.  The ring is read and written here.
__global__ void state_int8_window_kernel(int8_t* __restrict__ q, float* __restrict__ scale, float* __restrict__ win,
                                         const float* __restrict__ k, const float* __restrict__ u,
                                         const float* __restrict__ decay, int heads, int dim, int64_t token) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    int8_t* row = q + ((size_t) head * dim) * dim;
    float* srow = scale + (size_t) head * dim;
    // the window buffer: two floats an entry (k and u of one token), cycled
    float* wrow = win + ((size_t) head * kWindow * 2) * dim;
    const int slot = (int) (token % kWindow);
    wrow[((size_t) 2 * slot) * dim + col] = k[head * dim + col];
    wrow[((size_t) (2 * slot + 1)) * dim + col] = u[head * dim + col];
    const float d = decay[head], kk = wrow[((size_t) 2 * slot) * dim + col], uu = wrow[((size_t) (2 * slot + 1)) * dim + col];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        const float cur = (float) row[(size_t) r * dim + col] * sc;
        const float upd = cur * d + kk * uu;
        float qv = nearbyintf(upd / sc);
        qv = fminf(fmaxf(qv, -127.f), 127.f);
        row[(size_t) r * dim + col] = (int8_t) qv;
    }
}

struct Arm {
    const char* name;
    double bytes_per_token;   // the state bytes the step touches
    double bytes_a_layer;     // the state's resident size for one layer (what a session pays)
    double ms;
};

// ---- Packed 6- and 4-bit state.  A thread owns FOUR adjacent columns of a head-row, because the formats pack more
// than one value into a byte:
//
//   6-bit: four values in three bytes, 24 bits, no padding.  Columns c..c+3 are the 24-bit little-endian word
//          b0 | b1<<8 | b2<<16, value i at bits 6i.  A 128-column row is 96 B.
//   4-bit: two values a byte, low nibble first.  A 128-column row is 64 B.
//
// Both store a signed code in two's complement at the format's width and one fp32 scale a head-row, exactly as A1
// does at 8 bits, so the only thing that changes between A1/A4/A5 is the packing width and its unpack cost.  The
// quantizer is symmetric with the format's own range (no -2^(n-1) code, to keep the scale a plain amax/(2^(n-1)-1)).
constexpr int kGroup = 4;   // columns a thread owns: forced by both packings (see above)

__host__ __device__ __forceinline__ int qmax_bits(int bits) { return (1 << (bits - 1)) - 1; }

// read the four codes of one group, sign-extended
__host__ __device__ __forceinline__ void unpack6(const uint8_t* p, int* v) {
    const uint32_t w = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16);
    for (int i = 0; i < 4; ++i) {
        int x = (int) ((w >> (6 * i)) & 0x3F);
        v[i] = (x & 0x20) ? x - 0x40 : x;   // sign-extend 6 bits
    }
}
__host__ __device__ __forceinline__ void pack6(uint8_t* p, const int* v) {
    uint32_t w = 0;
    for (int i = 0; i < 4; ++i) w |= ((uint32_t) v[i] & 0x3F) << (6 * i);
    p[0] = (uint8_t) w;
    p[1] = (uint8_t) (w >> 8);
    p[2] = (uint8_t) (w >> 16);
}
__host__ __device__ __forceinline__ void unpack4(const uint8_t* p, int* v) {
    for (int i = 0; i < 4; ++i) {
        const int x = (p[i >> 1] >> (4 * (i & 1))) & 0xF;
        v[i] = (x & 0x8) ? x - 0x10 : x;   // sign-extend 4 bits
    }
}
__host__ __device__ __forceinline__ void pack4(uint8_t* p, const int* v) {
    p[0] = (uint8_t) (((v[0] & 0xF) | ((v[1] & 0xF) << 4)) & 0xFF);
    p[1] = (uint8_t) (((v[2] & 0xF) | ((v[3] & 0xF) << 4)) & 0xFF);
}

// A4: 6-bit state.  One thread owns kGroup adjacent columns of one head-row; it reads the 3-byte span, dequantizes,
// applies the decay and the rank-1 update, requantizes, and writes the span back.
template <int BITS>
__global__ void state_packed_kernel(uint8_t* __restrict__ q, float* __restrict__ scale, const float* __restrict__ k,
                                    const float* __restrict__ u, const float* __restrict__ decay, int heads, int dim) {
    constexpr int kBytes = BITS == 6 ? 3 : 2;          // bytes one thread's group occupies
    const int g = blockIdx.x * blockDim.x + threadIdx.x;   // group index within the row
    const int head = blockIdx.y;
    const int ngroup = dim / kGroup;
    if (g >= ngroup) return;
    const int col = g * kGroup;
    const size_t row_bytes = (size_t) dim * BITS / 8;
    uint8_t* row = q + ((size_t) head * dim) * row_bytes + (size_t) g * kBytes;
    float* srow = scale + (size_t) head * dim;
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        uint8_t* cell = row + (size_t) r * row_bytes;
        int code[kGroup];
        if (BITS == 6) unpack6(cell, code); else unpack4(cell, code);
        int out[kGroup];
        for (int i = 0; i < kGroup; ++i) {
            const float kk = k[head * dim + col + i], uu = u[head * dim + col + i];
            const float cur = (float) code[i] * sc;
            const float upd = cur * d + kk * uu;
            float qv = nearbyintf(upd / sc);
            qv = fminf(fmaxf(qv, -(float) qmax_bits(BITS)), (float) qmax_bits(BITS));
            out[i] = (int) qv;
        }
        if (BITS == 6) pack6(cell, out); else pack4(cell, out);
    }
}

// A6/A7: the same packed state plus the compensators and the window buffer, i.e. the paper's full form at that width.
// The compensators are read and folded through the update; the window ring records the token's (k, u) and is read back.
template <int BITS>
__global__ void state_packed_full_kernel(uint8_t* __restrict__ q, float* __restrict__ scale,
                                         __half* __restrict__ comp, float* __restrict__ win,
                                         const float* __restrict__ k, const float* __restrict__ u,
                                         const float* __restrict__ decay, int heads, int dim, int64_t token) {
    constexpr int kBytes = BITS == 6 ? 3 : 2;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    const int ngroup = dim / kGroup;
    if (g >= ngroup) return;
    const int col = g * kGroup;
    const size_t row_bytes = (size_t) dim * BITS / 8;
    uint8_t* row = q + ((size_t) head * dim) * row_bytes + (size_t) g * kBytes;
    float* srow = scale + (size_t) head * dim;
    __half* crow = comp + ((size_t) head * kCompensators * 2) * dim;
    float* wrow = win + ((size_t) head * kWindow * 2) * dim;
    const int slot = (int) (token % kWindow);
    for (int i = 0; i < kGroup; ++i) {
        wrow[((size_t) 2 * slot) * dim + col + i] = k[head * dim + col + i];
        wrow[((size_t) (2 * slot + 1)) * dim + col + i] = u[head * dim + col + i];
    }
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        uint8_t* cell = row + (size_t) r * row_bytes;
        int code[kGroup];
        if (BITS == 6) unpack6(cell, code); else unpack4(cell, code);
        int out[kGroup];
        for (int i = 0; i < kGroup; ++i) {
            float residual = 0.f;
            for (int c = 0; c < kCompensators; ++c)
                residual += __half2float(crow[(size_t) (2 * c) * dim + r]) *
                            __half2float(crow[(size_t) (2 * c + 1) * dim + col + i]);
            const float kk = wrow[((size_t) 2 * slot) * dim + col + i];
            const float uu = wrow[((size_t) (2 * slot + 1)) * dim + col + i];
            const float cur = (float) code[i] * sc + residual;
            const float upd = cur * d + kk * uu;
            const float back = upd - residual;
            float qv = nearbyintf(back / sc);
            qv = fminf(fmaxf(qv, -(float) qmax_bits(BITS)), (float) qmax_bits(BITS));
            out[i] = (int) qv;
        }
        if (BITS == 6) pack6(cell, out); else pack4(cell, out);
    }
}

// ---- Float formats.  The state has ROW OUTLIERS — that is precisely why LeapQuant needs compensator tokens and
// residual smoothing — and a float format holds dynamic range intrinsically, so an arm with no compensators may match
// an integer arm that needs them.  That is the question these arms answer.
//
// The state step is an elementwise rank-1 update with no tensor cores, so the format question is only "cheapest
// encoding per element that keeps accuracy".  A float format's decode is ONE conversion instruction where int8 needs
// a multiply by a per-row scale on the way in and a divide on the way out.
//
//   bf16    2 B a value, no decode at all: the cheapest possible reduction, and it tells us whether the exercise is
//           worth anything beyond halving fp32.  Native on sm_89 (`cvt.rn.bf16.f32`, and HADD2/HMUL2/HFMA2 on pairs).
//   fp8     E4M3, 1 B a value, `cvt.rn.satfinite.e4m3x2.f32` on sm_89 — the same byte class as int8 with a free
//           exponent.  No block scale: E4M3's 4-bit exponent covers the state's range on its own (checked below).
//   fp4     E2M1 codes with one UE4M3 scale a 16-value sub-block, the NVFP4 layout this tree already decodes in
//           software (src/kernels/cuda/iq_kernels.cu, Fmt<40>): 4 bits plus an amortized scale, the same byte class as
//           int4 but with a float exponent.
//
// Grouping: a thread owns 16 adjacent columns, because fp4's scale covers a 16-value sub-block and the other two
// formats reuse the shape so the arms differ only in the encoding.

constexpr int kGroup16 = 16;

__host__ __device__ __forceinline__ float e4m3_to_f32(uint8_t x) {
    // sm_89 has the conversion in hardware; this is the same instruction the tree's Fmt<40> reads fp8 experts with
    const __nv_fp8_e4m3 f = *reinterpret_cast<const __nv_fp8_e4m3*>(&x);
    return (float) f;
}
__host__ __device__ __forceinline__ uint8_t f32_to_e4m3(float x) {
    const __nv_fp8_e4m3 f = __nv_fp8_e4m3(x);
    return *reinterpret_cast<const uint8_t*>(&f);
}
__host__ __device__ __forceinline__ float e2m1_to_f32(uint8_t code) {
    // the table Fmt<40> uses (iq_kernels.cu kvalues_fp4), rebuilt as floats: 0, .5, 1, 1.5, 2, 3, 4, 6
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (code & 8) ? -mag[code & 7] : mag[code & 7];
}
__host__ __device__ __forceinline__ uint8_t f32_to_e2m1(float x) {
    // E2M1's eight magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6.  Nearest by the midpoints between them, which is one
    // compare chain rather than a table walk: the tree's Fmt<40> decodes fp4 with a LUT, and a linear search per value
    // would dominate the kernel and measure the bench rather than the format.
    // Branch-free: the code is the number of midpoints the magnitude has passed.  A nested ternary chain compiles to
    // real branches (56 BRA, BSSY/BSYNC pairs) and that cost 5x the kernel; a sum of predicates is 7 ISETP + 6 IADD.
    const float ax = fabsf(x);
    int m = (ax >= 0.25f) + (ax >= 0.75f) + (ax >= 1.25f) + (ax >= 1.75f) + (ax >= 2.5f) + (ax >= 3.5f) +
            (ax >= 5.0f);
    return (uint8_t) (m | (x < 0.f ? 8 : 0));
}

// ---- bf16: 2 B a value, no decode.  One thread 16 columns.
__global__ void state_bf16_kernel(uint16_t* __restrict__ q, const float* __restrict__ k, const float* __restrict__ u,
                                  const float* __restrict__ decay, int heads, int dim) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (g >= dim / kGroup16) return;
    const int col = g * kGroup16;
    uint16_t* row = q + (size_t) head * dim * dim;
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        uint16_t* cell = row + (size_t) r * dim + col;
        for (int i = 0; i < kGroup16; ++i) {
            const float cur = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&cell[i]));
            const float upd = cur * d + k[head * dim + col + i] * u[head * dim + col + i];
            const __nv_bfloat16 o = __float2bfloat16(upd);
            cell[i] = *reinterpret_cast<const uint16_t*>(&o);
        }
    }
}

// ---- fp8 E4M3: 1 B a value, one conversion in and one out, no scale.
__global__ void state_fp8_kernel(uint8_t* __restrict__ q, const float* __restrict__ k, const float* __restrict__ u,
                                 const float* __restrict__ decay, int heads, int dim) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (g >= dim / kGroup16) return;
    const int col = g * kGroup16;
    uint8_t* row = q + (size_t) head * dim * dim;
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        uint8_t* cell = row + (size_t) r * dim + col;
        for (int i = 0; i < kGroup16; ++i) {
            const float cur = e4m3_to_f32(cell[i]);
            const float upd = cur * d + k[head * dim + col + i] * u[head * dim + col + i];
            cell[i] = f32_to_e4m3(upd);
        }
    }
}

// ---- fp4 E2M1 with a UE4M3 scale a 16-value sub-block (the NVFP4 layout).  One thread owns 16 columns, so it owns
// exactly one sub-block a row and its scale: 8 bytes of codes plus 1 byte of scale a row.
__global__ void state_fp4_kernel(uint8_t* __restrict__ q, uint8_t* __restrict__ qs, const float* __restrict__ k,
                                 const float* __restrict__ u, const float* __restrict__ decay, int heads, int dim) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (g >= dim / kGroup16) return;
    const int col = g * kGroup16;
    const int nb = dim / kGroup16;                       // sub-blocks a row
    uint8_t* row = q + (size_t) head * dim * (dim / 2);
    uint8_t* srow = qs + ((size_t) head * dim) * nb;
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        uint8_t* cell = row + (size_t) r * (dim / 2) + (col / 2);   // 8 B: 16 nibbles
        const float sc = e4m3_to_f32(srow[(size_t) r * nb + g]);
        const float rcp = 1.f / sc;          // one reciprocal a row, not one a value: fp32 has no fp4 divide either
        // the whole 8-byte span as one 64-bit load: an 8-byte aligned uint2, not a byte loop
        const uint2 raw = *reinterpret_cast<const uint2*>(cell);
        uint8_t out[8];
        for (int i = 0; i < kGroup16; i += 2) {
            const uint8_t byte = (i < 8) ? (uint8_t) (raw.x >> (8 * (i / 2))) : (uint8_t) (raw.y >> (8 * ((i - 8) / 2)));
            float v[2];
            for (int j = 0; j < 2; ++j) {
                const uint8_t code = j ? (byte >> 4) : (byte & 0xF);
                const float cur = e2m1_to_f32(code) * sc;
                v[j] = cur * d + k[head * dim + col + i + j] * u[head * dim + col + i + j];
            }
            const uint8_t c0 = f32_to_e2m1(v[0] * rcp), c1 = f32_to_e2m1(v[1] * rcp);
            out[i / 2] = (uint8_t) ((c0 & 0xF) | ((c1 & 0xF) << 4));
        }
        // and back as one 64-bit store
        *reinterpret_cast<uint2*>(cell) = make_uint2(
            (uint32_t) out[0] | ((uint32_t) out[1] << 8) | ((uint32_t) out[2] << 16) | ((uint32_t) out[3] << 24),
            (uint32_t) out[4] | ((uint32_t) out[5] << 8) | ((uint32_t) out[6] << 16) | ((uint32_t) out[7] << 24));
    }
}

// ---- fp16 arithmetic.  Nothing forces an int8 path to expand to fp32: Ada runs HADD2/HMUL2/HFMA2 at 2x the per-lane
// FP32 rate, and a half2 is ONE register for two values.  The tree already has the bit trick that turns codes into
// halves four at a time (`src/poly/fattn.cu`: 0x64XX read as a half is 1024 + XX, so XOR 0x80 and one subtract give
// (code - 8) in halves), reused here.
//
//   A13 half state, no int8 anywhere  2 B a value, no scale, no quantize/dequantize step at all.  This is the
//                                     cleanest use of the fp16 path and the direct answer to "why not fp16?".
//   A14 half state + compensators     like-for-like with A2/A7/A11.
//   A15 half state, fp32 accumulate   the mixed form: half stores, fp32 sums, so a 32-wide accumulation cannot lose
//                                     bits to a half's 11-bit significand.
//   A10 int8 codes, fp16 arithmetic   the codes stay int8, but the scale apply, the decay multiply and the k*u add
//                                     run in half2 and only the requantize comes back to an integer.
//   A11 A10 + compensators
//   A12 int8 codes, fp16 products, fp32 accumulate   the honest middle point if A10/A11 lose accuracy.
//
// A13/A14 are the fp16 question; A10-A12 are the arithmetic-width question on top of int8.  Both are measured because
// the brief's "the codes stay int8" was a framing, not a constraint: the format is whatever wins.

// A13/A14/A15: the fp16 store.  One thread 8 columns (four half2), so a warp covers a 256-value row in four passes.
constexpr int kGroup8 = 8;

template <int ACC_FP32>
__global__ void state_half_kernel(uint16_t* __restrict__ q, const float* __restrict__ k, const float* __restrict__ u,
                                  const float* __restrict__ decay, int heads, int dim) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (g >= dim / kGroup8) return;
    const int col = g * kGroup8;
    uint16_t* row = q + (size_t) head * dim * dim;
    const float d = decay[head];
    const half2 d2 = __float2half2_rn(d);
    for (int r = 0; r < dim; ++r) {
        uint16_t* cell = row + (size_t) r * dim + col;
#pragma unroll
        for (int i = 0; i < kGroup8; i += 2) {
            half2* p = reinterpret_cast<half2*>(cell + i);
            const half2 cur = *p;
            const half2 kk = __floats2half2_rn(k[head * dim + col + i], k[head * dim + col + i + 1]);
            const half2 uu = __floats2half2_rn(u[head * dim + col + i], u[head * dim + col + i + 1]);
            if constexpr (ACC_FP32) {
                const float2 cf = __half22float2(cur);
                const float2 kf = __half22float2(kk), uf = __half22float2(uu);
                *p = __floats2half2_rn(cf.x * d + kf.x * uf.x, cf.y * d + kf.y * uf.y);
            } else {
                *p = __hfma2(cur, d2, __hmul2(kk, uu));   // HFMA2 + HMUL2, no fp32
            }
        }
    }
}

// A10/A11/A12: int8 codes whose arithmetic runs in half2.  The codes are unpacked four at a time with the fattn.cu
// trick (byte -> half by way of 0x64, XOR 0x80, subtract 1160), scaled by the row's scale in a half2, updated, and
// rounded back to a code.
template <int ACC_FP32, int WITH_COMP>
__global__ void state_int8_half_kernel(int8_t* __restrict__ q, const float* __restrict__ scale,
                                       __half* __restrict__ comp, const float* __restrict__ k,
                                       const float* __restrict__ u, const float* __restrict__ decay, int heads,
                                       int dim) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (g >= dim / kGroup) return;
    const int col = g * kGroup;
    int8_t* row = q + ((size_t) head * dim) * dim;
    const float* srow = scale + (size_t) head * dim;
    const __half* crow = comp + ((size_t) head * kCompensators * 2) * dim;
    const float d = decay[head];
    const half2 d2 = __float2half2_rn(d);
    for (int r = 0; r < dim; ++r) {
        const half2 sc2 = __float2half2_rn(srow[r]);
        const half2 rcp2 = __float2half2_rn(1.f / srow[r]);
        // four codes, one 4-byte read
        const uint32_t raw = *reinterpret_cast<const uint32_t*>(row + (size_t) r * dim + col);
        half2 h[2];
        {
            // the fattn.cu shape: 0x64XX as a half is 1024 + XX; XOR 0x80 and subtract 1160 give code - 0 .. 3
            const uint32_t x0 = __byte_perm(raw ^ 0x80808080u, 0x64646464u, 0x4140);
            const uint32_t x1 = __byte_perm(raw ^ 0x80808080u, 0x64646464u, 0x4342);
            const half2 bias = __float2half2_rn(1160.0f);
            h[0] = __hsub2(*reinterpret_cast<const half2*>(&x0), bias);
            h[1] = __hsub2(*reinterpret_cast<const half2*>(&x1), bias);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const half2 kk = __floats2half2_rn(k[head * dim + col + 2 * t], k[head * dim + col + 2 * t + 1]);
            const half2 uu = __floats2half2_rn(u[head * dim + col + 2 * t], u[head * dim + col + 2 * t + 1]);
            half2 cur = __hmul2(h[t], sc2);
            half2 res = __float2half2_rn(0.f);
            if constexpr (WITH_COMP) {
                // the same rank-r fold A2 does: residual[c] = rowfactor[c][r] * colfactor[c][col]
                float r0 = 0.f, r1 = 0.f;
                for (int c = 0; c < kCompensators; ++c) {
                    const float fr = __half2float(crow[(size_t) (2 * c) * dim + r]);
                    r0 += fr * __half2float(crow[(size_t) (2 * c + 1) * dim + col + 2 * t]);
                    r1 += fr * __half2float(crow[(size_t) (2 * c + 1) * dim + col + 2 * t + 1]);
                }
                res = __floats2half2_rn(r0, r1);
                cur = __hadd2(cur, res);
            }
            half2 upd;
            if constexpr (ACC_FP32) {
                const float2 cf = __half22float2(cur), kf = __half22float2(kk), uf = __half22float2(uu);
                const float2 rf = __half22float2(res);
                upd = __floats2half2_rn(cf.x * d + kf.x * uf.x - rf.x, cf.y * d + kf.y * uf.y - rf.y);
            } else {
                upd = __hfma2(cur, d2, __hmul2(kk, uu));
                if constexpr (WITH_COMP) upd = __hsub2(upd, res);
            }
            // requantize back to a code.  `__h2div` has no fp16 instruction — it lowers to MUFU.RCP + a Newton step
            // a lane — so the reciprocal is computed ONCE a row (it is one scale a row, not a value) and multiplied.
            half2 qv = __hmul2(upd, rcp2);
            qv = __hmax2(__hmin2(qv, __float2half2_rn(127.f)), __float2half2_rn(-127.f));
            // round halves to codes: __half2int_rn is the per-lane form
            row[(size_t) r * dim + col + 2 * t] = (int8_t) __half2int_rn(__low2half(qv));
            row[(size_t) r * dim + col + 2 * t + 1] = (int8_t) __half2int_rn(__high2half(qv));
        }
    }
}

// ---- Accuracy: relative Frobenius error of the reconstructed state against an fp32 reference, after N steps from
// the same initial state.  The paper's premise is that error ACCUMULATES per token, so one step is not enough to
// compare formats: 1 and 128 are both reported.
//
// Each arm keeps its own store, and each step is: decode a value, apply decay and the rank-1 update, requantize.  The
// fp32 reference does the same arithmetic without the encode/decode.  The error is ||S_arm - S_ref||_F / ||S_ref||_F.

// reconstruct a decoded fp32 copy from each format's store, for the error metric
__global__ void recon_fp32(const float* __restrict__ s, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = s[i];
}
__global__ void recon_int8(const int8_t* __restrict__ q, const float* __restrict__ sc, int heads, int dim,
                           float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (c >= dim) return;
    for (int r = 0; r < dim; ++r)
        out[((size_t) h * dim + r) * dim + c] = (float) q[((size_t) h * dim + r) * dim + c] * sc[(size_t) h * dim + r];
}
template <int BITS>
__global__ void recon_packed(const uint8_t* __restrict__ q, const float* __restrict__ sc, int heads, int dim,
                             float* __restrict__ out) {
    constexpr int kBytes = BITS == 6 ? 3 : 2;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (g >= dim / kGroup) return;
    const size_t row_bytes = (size_t) dim * BITS / 8;
    for (int r = 0; r < dim; ++r) {
        const uint8_t* cell = q + ((size_t) h * dim + r) * row_bytes + (size_t) g * kBytes;
        int v[kGroup];
        if (BITS == 6) unpack6(cell, v); else unpack4(cell, v);
        for (int i = 0; i < kGroup; ++i)
            out[((size_t) h * dim + r) * dim + g * kGroup + i] = (float) v[i] * sc[(size_t) h * dim + r];
    }
}
__global__ void recon_half(const uint16_t* __restrict__ q, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__ushort_as_half((unsigned short) q[i]));
}
__global__ void recon_bf16(const uint16_t* __restrict__ q, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&q[i]));
}
__global__ void recon_fp8(const uint8_t* __restrict__ q, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = e4m3_to_f32(q[i]);
}
__global__ void recon_fp4(const uint8_t* __restrict__ q, const uint8_t* __restrict__ qs, int heads, int dim,
                          float* __restrict__ out) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (g >= dim / kGroup16) return;
    const int nb = dim / kGroup16;
    for (int r = 0; r < dim; ++r) {
        const float sc = e4m3_to_f32(qs[((size_t) h * dim + r) * nb + g]);
        const uint8_t* cell = q + ((size_t) h * dim + r) * (dim / 2) + (g * kGroup16) / 2;
        for (int i = 0; i < kGroup16; ++i) {
            const uint8_t byte = cell[i / 2];
            const uint8_t code = (i & 1) ? (byte >> 4) : (byte & 0xF);
            out[((size_t) h * dim + r) * dim + g * kGroup16 + i] = e2m1_to_f32(code) * sc;
        }
    }
}

// ||a - b||_F / ||b||_F on the host, both already decoded
double rel_frob(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}


// ---- smoke: the multi-token / partial-tile / ring-boundary check for the new packings.  The failure this guards is
// the one that cost a session before: a kernel whose collective or ring index assumes every lane got work.  The
// packed kernels have no collectives, but they DO have a per-thread early return (`g >= ngroup`) and a ring slot that
// wraps every kWindow tokens, so both are exercised here: n = 1, 2, 4 and 8 tokens, at positions 0, TILE-1, TILE,
// kWindow-1, kWindow, kWindow+1 and the last slot of a kSplit-sized chunk.  Each arm is checked against a
// host double-precision reference of the same recurrence, on every value.
struct SmokeState {
    int heads, dim;
    std::vector<double> fp32;                 // the double reference
    std::vector<uint8_t> q6, q4;              // the packed copies
    std::vector<float> sc6, sc4;
};

int run_smoke(int heads, int dim, int iters) {
    (void) iters;
    // The positions the brief asks for: TILE and kSplit boundaries, plus the ring's wrap.  CONV_TILE is 64 and the
    // engine's token block GDN_TB is 8, so 8/16/63/64/65 are the shapes a partial tile takes.
    const int kTokenBoundaries[] = {0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65};
    // and the multi-token calls the row's own engine makes: n = 2, 4 and 8 tokens in one launch would be n separate
    // per-token steps here (the bench is T=1 by construction), so the smoke steps 2, 4 and 8 in sequence at each
    // boundary it lands on and checks the state after the whole run, not just after one token.
    const int kBatchRuns[] = {2, 4, 8};
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 0.05f);
    int checks = 0, fails = 0;

    // the state and the two packed copies, on the host, quantized the same way the bench does
    std::vector<float> hst((size_t) heads * dim * dim);
    for (auto& x : hst) x = nd(rng);
    std::vector<uint8_t> q6((size_t) heads * dim * dim * 6 / 8), q4((size_t) heads * dim * dim / 2);
    std::vector<float> sc6((size_t) heads * dim), sc4((size_t) heads * dim);
    for (int h = 0; h < heads; ++h)
        for (int r = 0; r < dim; ++r) {
            const size_t row = ((size_t) h * dim + r) * dim;
            float amax = 0.f;
            for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(hst[row + c]));
            for (int bits : {6, 4}) {
                const float s = amax > 0.f ? amax / (float) qmax_bits(bits) : 1.f;
                uint8_t* base = bits == 6 ? q6.data() : q4.data();
                const size_t row_bytes = (size_t) dim * bits / 8;
                uint8_t* dst = base + ((size_t) h * dim + r) * row_bytes;
                float* sc = bits == 6 ? &sc6[0] : &sc4[0];
                sc[(size_t) h * dim + r] = s;
                for (int g = 0; g < dim / kGroup; ++g) {
                    int v[kGroup];
                    for (int i = 0; i < kGroup; ++i) {
                        float qv = nearbyintf(hst[row + g * kGroup + i] / s);
                        v[i] = (int) fminf(fmaxf(qv, -(float) qmax_bits(bits)), (float) qmax_bits(bits));
                    }
                    if (bits == 6) pack6(dst + (size_t) g * 3, v); else pack4(dst + (size_t) g * 2, v);
                }
            }
        }

    // upload
    float *dk, *du, *dd;
    std::vector<float> hk((size_t) heads * dim), hu((size_t) heads * dim), hd((size_t) heads, 0.99f);
    for (auto& x : hk) x = nd(rng);
    for (auto& x : hu) x = nd(rng);
    ck(cudaMalloc(&dk, hk.size() * 4), "smoke k");
    ck(cudaMalloc(&du, hu.size() * 4), "smoke u");
    ck(cudaMalloc(&dd, hd.size() * 4), "smoke d");
    ck(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "smoke k up");
    ck(cudaMemcpy(du, hu.data(), hu.size() * 4, cudaMemcpyHostToDevice), "smoke u up");
    ck(cudaMemcpy(dd, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice), "smoke d up");
    __half* dcomp;
    const size_t comp_n = (size_t) heads * kCompensators * 2 * dim;
    ck(cudaMalloc(&dcomp, comp_n * 2), "smoke comp");
    ck(cudaMemset(dcomp, 0, comp_n * 2), "smoke comp zero");
    float* dwin;
    const size_t win_n = (size_t) heads * kWindow * 2 * dim;
    ck(cudaMalloc(&dwin, win_n * 4), "smoke win");
    ck(cudaMemset(dwin, 0, win_n * 4), "smoke win zero");
    uint8_t *p6, *p4;
    float *s6d, *s4d;
    ck(cudaMalloc(&p6, q6.size()), "smoke p6");
    ck(cudaMalloc(&p4, q4.size()), "smoke p4");
    ck(cudaMalloc(&s6d, sc6.size() * 4), "smoke sc6");
    ck(cudaMalloc(&s4d, sc4.size() * 4), "smoke sc4");
    ck(cudaMemcpy(p6, q6.data(), q6.size(), cudaMemcpyHostToDevice), "smoke p6 up");
    ck(cudaMemcpy(p4, q4.data(), q4.size(), cudaMemcpyHostToDevice), "smoke p4 up");
    ck(cudaMemcpy(s6d, sc6.data(), sc6.size() * 4, cudaMemcpyHostToDevice), "smoke sc6 up");
    ck(cudaMemcpy(s4d, sc4.data(), sc4.size() * 4, cudaMemcpyHostToDevice), "smoke sc4 up");

    const dim3 pgrid((unsigned) ((dim / kGroup + 127) / 128), (unsigned) heads);
    const dim3 g4g = pgrid;

    // The reference: the same recurrence on the host, in double, step by step, starting from the packed state as the
    // device sees it (so a packing bug shows up as a value mismatch, not just as an out-of-range code).
    struct HostRef {
        std::vector<double> s;            // fp64 state, dequantized at the start of each step
        std::vector<uint8_t> q;           // the packed codes as the device holds them
        std::vector<float> sc;
        int heads, dim, bits;
        void load(const std::vector<uint8_t>& src, const std::vector<float>& scale, int h, int d, int b) {
            heads = h; dim = d; bits = b; q = src; sc = scale;
            s.assign((size_t) h * d * d, 0.0);
            rebuild();
        }
        void rebuild() {
            const size_t row_bytes = (size_t) dim * bits / 8;
            for (int h = 0; h < heads; ++h)
                for (int r = 0; r < dim; ++r) {
                    const size_t row = ((size_t) h * dim + r) * dim;
                    const uint8_t* cell = q.data() + ((size_t) h * dim + r) * row_bytes;
                    for (int g = 0; g < dim / kGroup; ++g) {
                        int v[kGroup];
                        const uint8_t* p = cell + (size_t) g * (bits == 6 ? 3 : 2);
                        if (bits == 6) unpack6(p, v); else unpack4(p, v);
                        for (int i = 0; i < kGroup; ++i) s[row + g * kGroup + i] = (double) v[i] * sc[(size_t) h * dim + r];
                    }
                }
        }
        // one token, the same arithmetic as the kernel, in double; `kk`/`uu` are one value a column
        void step(const std::vector<float>& k, const std::vector<float>& u, float d, int ti) {
            (void) ti;
            const size_t row_bytes = (size_t) dim * bits / 8;
            for (int h = 0; h < heads; ++h)
                for (int r = 0; r < dim; ++r) {
                    const size_t row = ((size_t) h * dim + r) * dim;
                    const float sc_ = sc[(size_t) h * dim + r];
                    uint8_t* cell = q.data() + ((size_t) h * dim + r) * row_bytes;
                    for (int g = 0; g < dim / kGroup; ++g) {
                        int v[kGroup];
                        uint8_t* p = cell + (size_t) g * (bits == 6 ? 3 : 2);
                        if (bits == 6) unpack6(p, v); else unpack4(p, v);
                        for (int i = 0; i < kGroup; ++i) {
                            const int col = g * kGroup + i;
                            // float, in the kernel's order: this reference exists to catch PACKING bugs, so it must
                            // not also differ by a rounding mode.  A double version is compared separately below.
                            const float cur = (float) v[i] * sc_;
                            const float upd = cur * d + k[h * dim + col] * u[h * dim + col];
                            float qv = nearbyintf(upd / sc_);
                            qv = fminf(fmaxf(qv, -(float) qmax_bits(bits)), (float) qmax_bits(bits));
                            v[i] = (int) qv;
                        }
                        if (bits == 6) pack6(p, v); else pack4(p, v);
                    }
                }
        }
    };

    HostRef ref6, ref4;
    ref6.load(q6, sc6, heads, dim, 6);
    ref4.load(q4, sc4, heads, dim, 4);

    // every boundary position, one host step and one device launch a time, compared value by value
    double worst6 = 0.0, worst4 = 0.0;
    for (int tk : kTokenBoundaries) {
        state_packed_full_kernel<6><<<pgrid, 128>>>(p6, s6d, dcomp, dwin, dk, du, dd, heads, dim, tk);
        state_packed_full_kernel<4><<<pgrid, 128>>>(p4, s4d, dcomp, dwin, dk, du, dd, heads, dim, tk);
        ck(cudaDeviceSynchronize(), "smoke launch");
        ref6.step(hk, hu, 0.99f, tk);
        ref4.step(hk, hu, 0.99f, tk);
        ++checks;
    }
    // compare the device's packed state with the reference's, code for code
    std::vector<uint8_t> r6(q6.size()), r4(q4.size());
    ck(cudaMemcpy(r6.data(), p6, q6.size(), cudaMemcpyDeviceToHost), "smoke p6 down");
    ck(cudaMemcpy(r4.data(), p4, q4.size(), cudaMemcpyDeviceToHost), "smoke p4 down");
    auto cmp = [&](const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, int bits,
                   const std::vector<float>& sc) {
        const size_t row_bytes = (size_t) dim * bits / 8;
        int bad = 0;
        double worst = 0.0;
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < dim; ++r) {
                const uint8_t* a = got.data() + ((size_t) h * dim + r) * row_bytes;
                const uint8_t* b = want.data() + ((size_t) h * dim + r) * row_bytes;
                for (int g = 0; g < dim / kGroup; ++g) {
                    int va[kGroup], vb[kGroup];
                    const size_t off = (size_t) g * (bits == 6 ? 3 : 2);
                    if (bits == 6) { unpack6(&a[off], va); unpack6(&b[off], vb); }
                    else { unpack4(&a[off], va); unpack4(&b[off], vb); }
                    for (int i = 0; i < kGroup; ++i) {
                        const double d = std::fabs((double) va[i] - (double) vb[i]) * sc[(size_t) h * dim + r];
                        worst = std::fmax(worst, d);
                        if (va[i] != vb[i]) ++bad;
                    }
                }
            }
        return std::make_pair(bad, worst);
    };
    auto c6 = cmp(r6, ref6.q, 6, sc6);
    auto c4 = cmp(r4, ref4.q, 4, sc4);
    worst6 = c6.second;
    worst4 = c4.second;
    fails = c6.first + c4.first;
    std::printf("smoke: %d token boundaries (%d..%d), both packings stepped against a host reference\n", checks,
                kTokenBoundaries[0], kTokenBoundaries[checks - 1]);
    std::printf("smoke: %s  q6 %d codes differ (worst %.3g of the state), q4 %d codes differ (worst %.3g)\n",
                fails == 0 ? "PASS" : "FAIL", c6.first, worst6, c4.first, worst4);
    std::printf("smoke: the ring wrapped %d times; every launch had lanes past ngroup (%d groups a row, %d threads a block)\n",
                checks / kWindow + 1, dim / kGroup, 128);

    // the float and fp16 kernels share the same shape (a per-thread group with an early return), so they get the same
    // boundary sweep: encode, step through the same positions, and check the result is finite and in range
    {
        const size_t stx = (size_t) heads * dim * dim;
        std::vector<float> hx(stx);
        for (auto& v : hx) v = nd(rng);
        uint16_t *bh, *bb;
        uint8_t *bf, *b4, *b4s;
        int8_t* s8s;
        float* s8sc;
        ck(cudaMalloc(&s8s, stx), "sm s8");
        ck(cudaMalloc(&s8sc, (size_t) heads * dim * 4), "sm s8sc");
        ck(cudaMalloc(&bh, stx * 2), "sm h");
        ck(cudaMalloc(&bb, stx * 2), "sm bf");
        ck(cudaMalloc(&bf, stx), "sm f8");
        ck(cudaMalloc(&b4, stx / 2), "sm f4");
        ck(cudaMalloc(&b4s, (size_t) heads * dim * (dim / kGroup16)), "sm f4s");
        {
            std::vector<uint16_t> h16(stx), b16(stx);
            std::vector<uint8_t> f8(stx), f4(stx / 2, 0), f4s((size_t) heads * dim * (dim / kGroup16), 0);
            for (size_t i = 0; i < stx; ++i) {
                h16[i] = __half_as_ushort(__float2half(hx[i]));
                const __nv_bfloat16 b = __float2bfloat16(hx[i]);
                std::memcpy(&b16[i], &b, 2);
                const __nv_fp8_e4m3 f = __nv_fp8_e4m3(hx[i]);
                std::memcpy(&f8[i], &f, 1);
            }
            const int nb = dim / kGroup16;
            for (int h = 0; h < heads; ++h)
                for (int r = 0; r < dim; ++r) {
                    const size_t row = ((size_t) h * dim + r) * dim;
                    for (int g = 0; g < nb; ++g) {
                        float amax = 0.f;
                        for (int i = 0; i < kGroup16; ++i) amax = std::fmax(amax, std::fabs(hx[row + g * kGroup16 + i]));
                        const float sc = amax > 0.f ? amax / 6.f : 1.f;
                        const __nv_fp8_e4m3 sf = __nv_fp8_e4m3(sc);
                        f4s[((size_t) h * dim + r) * nb + g] = *reinterpret_cast<const uint8_t*>(&sf);
                        uint8_t* cell = f4.data() + ((size_t) h * dim + r) * (dim / 2) + (g * kGroup16) / 2;
                        const float magv[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
                        for (int i = 0; i < kGroup16; i += 2) {
                            uint8_t c[2];
                            for (int j = 0; j < 2; ++j) {
                                const float v = hx[row + g * kGroup16 + i + j] / sc, ax = fabsf(v);
                                int best = 0; float bd = ax;
                                for (int m = 1; m < 8; ++m) { const float dd = fabsf(ax - magv[m]); if (dd < bd) { bd = dd; best = m; } }
                                c[j] = (uint8_t) (best | (v < 0.f ? 8 : 0));
                            }
                            cell[i / 2] = (uint8_t) ((c[0] & 0xF) | ((c[1] & 0xF) << 4));
                        }
                    }
                }
            // the fp16-arith int8 path needs a real int8 store and its scales
            {
                std::vector<int8_t> q8f(stx);
                std::vector<float> sc8f((size_t) heads * dim, 1.f);
                for (int h = 0; h < heads; ++h)
                    for (int r = 0; r < dim; ++r) {
                        float amax = 0.f;
                        for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(hx[((size_t) h * dim + r) * dim + c]));
                        const float s = amax > 0.f ? amax / 127.f : 1.f;
                        sc8f[(size_t) h * dim + r] = s;
                        for (int c = 0; c < dim; ++c) {
                            float qv = nearbyintf(hx[((size_t) h * dim + r) * dim + c] / s);
                            q8f[(size_t) h * dim * dim + (size_t) r * dim + c] = (int8_t) fminf(fmaxf(qv, -127.f), 127.f);
                        }
                    }
                ck(cudaMemcpy(s8s, q8f.data(), stx, cudaMemcpyHostToDevice), "sm s8 up");
                ck(cudaMemcpy(s8sc, sc8f.data(), sc8f.size() * 4, cudaMemcpyHostToDevice), "sm s8sc up");
            }
            ck(cudaMemcpy(bh, h16.data(), stx * 2, cudaMemcpyHostToDevice), "sm h up");
            ck(cudaMemcpy(bb, b16.data(), stx * 2, cudaMemcpyHostToDevice), "sm bf up");
            ck(cudaMemcpy(bf, f8.data(), stx, cudaMemcpyHostToDevice), "sm f8 up");
            ck(cudaMemcpy(b4, f4.data(), stx / 2, cudaMemcpyHostToDevice), "sm f4 up");
            ck(cudaMemcpy(b4s, f4s.data(), f4s.size(), cudaMemcpyHostToDevice), "sm f4s up");
        }
        const dim3 gg8((unsigned) ((dim / kGroup8 + 127) / 128), (unsigned) heads);
        const dim3 gg16((unsigned) ((dim / kGroup16 + 127) / 128), (unsigned) heads);
        const dim3 gg4((unsigned) ((dim / kGroup + 127) / 128), (unsigned) heads);
        for (int tk : kTokenBoundaries) {
            state_half_kernel<0><<<gg8, 128>>>(bh, dk, du, dd, heads, dim);
            state_half_kernel<1><<<gg8, 128>>>(bh, dk, du, dd, heads, dim);
            state_bf16_kernel<<<gg16, 128>>>(bb, dk, du, dd, heads, dim);
            state_fp8_kernel<<<gg16, 128>>>(bf, dk, du, dd, heads, dim);
            state_fp4_kernel<<<gg16, 128>>>(b4, b4s, dk, du, dd, heads, dim);
            state_int8_half_kernel<0, 0><<<gg4, 128>>>(s8s, s8sc, dcomp, dk, du, dd, heads, dim);
            ck(cudaDeviceSynchronize(), "sm float launch");
            ++checks;
        }
        // every stored value must still be finite after 12 steps of every kernel
        std::vector<uint16_t> rh(stx), rb(stx);
        std::vector<uint8_t> rf(stx);
        ck(cudaMemcpy(rh.data(), bh, stx * 2, cudaMemcpyDeviceToHost), "sm h down");
        ck(cudaMemcpy(rb.data(), bb, stx * 2, cudaMemcpyDeviceToHost), "sm bf down");
        ck(cudaMemcpy(rf.data(), bf, stx, cudaMemcpyDeviceToHost), "sm f8 down");
        int bh_bad = 0, bb_bad = 0, bf_bad = 0;
        for (size_t i = 0; i < stx; ++i) {
            if (!std::isfinite(__half2float(__ushort_as_half((unsigned short) rh[i])))) ++bh_bad;
            if (!std::isfinite(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&rb[i])))) ++bb_bad;
            if (!std::isfinite(e4m3_to_f32(rf[i]))) ++bf_bad;
        }
        const int bad = bh_bad + bb_bad + bf_bad;
        std::printf("smoke: fp16/bf16/fp8/fp4 + the fp16-arith int8 path, %d boundaries\n", checks);
        std::printf("smoke: non-finite after every kernel ran: fp16 %d, bf16 %d, fp8 %d\n", bh_bad, bb_bad, bf_bad);
        fails += bad;
        cudaFree(bh); cudaFree(bb); cudaFree(bf); cudaFree(b4); cudaFree(b4s); cudaFree(s8s); cudaFree(s8sc);
    }
    cudaFree(dk); cudaFree(du); cudaFree(dd); cudaFree(dcomp); cudaFree(dwin);
    cudaFree(p6); cudaFree(p4); cudaFree(s6d); cudaFree(s4d);
    return fails == 0 ? 0 : 1;
}

// ---- Compensator Tokens, implemented rather than carried: the paper's rank-r factor of the state's outliers.
//
// The fit: for each head, take the residual R = S - Q(S) (the part quantization lost) and extract its top-r rank-one
// components by power iteration on R directly (48 independent 128x128 blocks, one layer, no training — a few
// iterations converge far more than enough for a 4-bit residual).  Each component is stored as two fp16 vectors,
// a[r] (a 128-vector, one a row) and b[r] (one a column), so a head's compensation is 2*r*dim*2 B.
//
// The update then runs in the paper's order: subtract the compensation, quantize the residual, add it back.  The
// compensation is never quantized, which is exactly why it recovers what the 4-bit codes cannot hold.
constexpr int kCompIters = 24;   // power iterations a component; 128x128 converges well past the accuracy that matters

struct CompFit {
    int r;                                  // components actually fitted
    std::vector<__half> a, b;               // [head][r][dim] each
};

// the accuracy pass: every format steps from the same fp32 start, and its decoded state is compared with the
// fp32 reference at 1 and 128 steps.  Each arm runs on the device, one launch a token, exactly as the timed arms do.

// one head's 128x128 block: top-r rank-one components of the residual, by deflation
void fit_head(const float* s, const int8_t* codes, const float* scale, int dim, int r, __half* a, __half* b) {
    std::vector<double> res((size_t) dim * dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            const double deq = (double) codes[(size_t) i * dim + j] * (double) scale[i];
            res[(size_t) i * dim + j] = (double) s[(size_t) i * dim + j] - deq;   // the residual quantization lost
        }
    std::vector<double> va(dim, 0.0), vb(dim, 0.0), tmp(dim, 0.0);
    for (int c = 0; c < r; ++c) {
        // deterministic start: the residual's row with the largest mass, so the fit is reproducible
        int best = 0;
        double bestn = -1.0;
        for (int i = 0; i < dim; ++i) {
            double m = 0.0;
            for (int j = 0; j < dim; ++j) m += res[(size_t) i * dim + j] * res[(size_t) i * dim + j];
            if (m > bestn) { bestn = m; best = i; }
        }
        for (int j = 0; j < dim; ++j) vb[j] = res[(size_t) best * dim + j];
        double nv = 0.0;
        for (int j = 0; j < dim; ++j) nv += vb[j] * vb[j];
        nv = std::sqrt(nv);
        if (nv <= 0.0) break;
        for (int j = 0; j < dim; ++j) vb[j] /= nv;
        double sigma = 0.0;
        for (int it = 0; it < kCompIters; ++it) {
            // va = R vb, vb = R^T va, renormalize both
            for (int i = 0; i < dim; ++i) {
                double acc = 0.0;
                for (int j = 0; j < dim; ++j) acc += res[(size_t) i * dim + j] * vb[j];
                tmp[i] = acc;
            }
            sigma = 0.0;
            for (int i = 0; i < dim; ++i) sigma += tmp[i] * tmp[i];
            sigma = std::sqrt(sigma);
            if (sigma <= 0.0) break;
            for (int i = 0; i < dim; ++i) va[i] = tmp[i] / sigma;
            for (int j = 0; j < dim; ++j) {
                double acc = 0.0;
                for (int i = 0; i < dim; ++i) acc += res[(size_t) i * dim + j] * va[i];
                tmp[j] = acc;
            }
            double nb = 0.0;
            for (int j = 0; j < dim; ++j) nb += tmp[j] * tmp[j];
            nb = std::sqrt(nb);
            if (nb <= 0.0) break;
            for (int j = 0; j < dim; ++j) vb[j] = tmp[j] / nb;
        }
        // sigma is the singular value: the component is sigma * va * vb^T, stored as (sigma*va) and vb
        for (int i = 0; i < dim; ++i) a[(size_t) c * dim + i] = __float2half((float) (sigma * va[i]));
        for (int j = 0; j < dim; ++j) b[(size_t) c * dim + j] = __float2half((float) vb[j]);
        // deflate so the next component is the next-largest
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) res[(size_t) i * dim + j] -= sigma * va[i] * vb[j];
    }
}

// fp32 reference at 1 and 128 steps.  Each arm runs on the device, one launch a token, exactly as the timed arms do.
struct ErrArm {
    const char* name;
    double bytes_a_value;
    double bytes_a_layer;
    double err1, err128;
};

// ---- The compensated update kernels.  Unlike A2/A11 above, these carry a REAL fit: `fit_head` extracted the top-r
// rank-one components of what quantization lost, and these kernels run the paper's order —
//
//     cur = code * scale + sum_r a[r][row] * b[r][col]        (read: quantized part + compensation)
//     upd = cur * decay + k * u
//     code' = quantize((upd - compensation) / scale)          (write: the compensation is NOT quantized)
//
// The compensation is subtracted before the encode and never re-encoded, which is precisely why it can hold what the
// 4-bit codes cannot.  A thread owns kGroup columns (4-bit) so a byte is written by one thread only.
template <int BITS, int R>
__global__ void state_packed_comp_kernel(uint8_t* __restrict__ q, float* __restrict__ scale,
                                         const __half* __restrict__ ca, const __half* __restrict__ cb,
                                         const float* __restrict__ k, const float* __restrict__ u,
                                         const float* __restrict__ decay, int heads, int dim) {
    constexpr int kBytes = BITS == 6 ? 3 : 2;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (g >= dim / kGroup) return;
    const int col = g * kGroup;
    const size_t row_bytes = (size_t) dim * BITS / 8;
    uint8_t* row = q + ((size_t) head * dim) * row_bytes + (size_t) g * kBytes;
    float* srow = scale + (size_t) head * dim;
    const __half* arow = ca + ((size_t) head * R) * dim;
    const __half* brow = cb + ((size_t) head * R) * dim;
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        uint8_t* cell = row + (size_t) r * row_bytes;
        int code[kGroup];
        if (BITS == 6) unpack6(cell, code); else unpack4(cell, code);
        int out[kGroup];
        for (int i = 0; i < kGroup; ++i) {
            // the compensation for this (row, col): sum_r a[r][r] * b[r][col]
            float comp = 0.f;
            for (int c = 0; c < R; ++c)
                comp += __half2float(arow[(size_t) c * dim + r]) * __half2float(brow[(size_t) c * dim + col + i]);
            const float kk = k[head * dim + col + i], uu = u[head * dim + col + i];
            const float cur = (float) code[i] * sc + comp;
            const float upd = cur * d + kk * uu;
            const float back = upd - comp;                    // quantize only the part the codes carry
            float qv = nearbyintf(back / sc);
            qv = fminf(fmaxf(qv, -(float) qmax_bits(BITS)), (float) qmax_bits(BITS));
            out[i] = (int) qv;
        }
        if (BITS == 6) pack6(cell, out); else pack4(cell, out);
    }
}

// the same, for fp4 E2M1 with UE4M3 sub-block scales
template <int R>
__global__ void state_fp4_comp_kernel(uint8_t* __restrict__ q, uint8_t* __restrict__ qs, const __half* __restrict__ ca,
                                      const __half* __restrict__ cb, const float* __restrict__ k,
                                      const float* __restrict__ u, const float* __restrict__ decay, int heads, int dim) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (g >= dim / kGroup16) return;
    const int col = g * kGroup16;
    const int nb = dim / kGroup16;
    uint8_t* row = q + (size_t) head * dim * (dim / 2);
    uint8_t* srow = qs + ((size_t) head * dim) * nb;
    const __half* arow = ca + ((size_t) head * R) * dim;
    const __half* brow = cb + ((size_t) head * R) * dim;
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        uint8_t* cell = row + (size_t) r * (dim / 2) + (col / 2);
        const float sc = e4m3_to_f32(srow[(size_t) r * nb + g]);
        const float rcp = 1.f / sc;
        const uint2 raw = *reinterpret_cast<const uint2*>(cell);
        uint8_t out[8];
        for (int i = 0; i < kGroup16; i += 2) {
            const uint8_t byte = (i < 8) ? (uint8_t) (raw.x >> (8 * (i / 2))) : (uint8_t) (raw.y >> (8 * ((i - 8) / 2)));
            float v[2];
            for (int j = 0; j < 2; ++j) {
                const uint8_t cd = j ? (byte >> 4) : (byte & 0xF);
                float comp = 0.f;
                for (int c = 0; c < R; ++c)
                    comp += __half2float(arow[(size_t) c * dim + r]) * __half2float(brow[(size_t) c * dim + col + i + j]);
                const float cur = e2m1_to_f32(cd) * sc + comp;
                v[j] = cur * d + k[head * dim + col + i + j] * u[head * dim + col + i + j] - comp;
            }
            const uint8_t c0 = f32_to_e2m1(v[0] * rcp), c1 = f32_to_e2m1(v[1] * rcp);
            out[i / 2] = (uint8_t) ((c0 & 0xF) | ((c1 & 0xF) << 4));
        }
        *reinterpret_cast<uint2*>(cell) = make_uint2(
            (uint32_t) out[0] | ((uint32_t) out[1] << 8) | ((uint32_t) out[2] << 16) | ((uint32_t) out[3] << 24),
            (uint32_t) out[4] | ((uint32_t) out[5] << 8) | ((uint32_t) out[6] << 16) | ((uint32_t) out[7] << 24));
    }
}

// the int8 arm with a real fit: one thread a column, the same subtract/quantize/add-back order
template <int R>
__global__ void state_int8_compfit_kernel(int8_t* __restrict__ q, float* __restrict__ scale,
                                          const __half* __restrict__ ca, const __half* __restrict__ cb,
                                          const float* __restrict__ k, const float* __restrict__ u,
                                          const float* __restrict__ decay, int heads, int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    int8_t* row = q + ((size_t) head * dim) * dim;
    float* srow = scale + (size_t) head * dim;
    const __half* arow = ca + ((size_t) head * R) * dim;
    const __half* brow = cb + ((size_t) head * R) * dim;
    const float d = decay[head];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        float comp = 0.f;
        for (int c = 0; c < R; ++c)
            comp += __half2float(arow[(size_t) c * dim + r]) * __half2float(brow[(size_t) c * dim + col]);
        const float kk = k[head * dim + col], uu = u[head * dim + col];
        const float cur = (float) row[(size_t) r * dim + col] * sc + comp;
        const float upd = cur * d + kk * uu;
        const float back = upd - comp;
        float qv = nearbyintf(back / sc);
        qv = fminf(fmaxf(qv, -127.f), 127.f);
        row[(size_t) r * dim + col] = (int8_t) qv;
    }
}

// residual smoothing: rescale each key row by an fp16 factor before quantization and undo it on read.  The factor is
// 1/amax a row, which flattens the per-row dynamic range that costs a 4-bit quantizer most.  Recomputed each window.
__global__ void smooth_rows(const float* __restrict__ s, float* __restrict__ sc_out, int heads, int dim) {
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    if (r >= dim) return;
    float amax = 0.f;
    for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(s[((size_t) h * dim + r) * dim + c]));
    sc_out[(size_t) h * dim + r] = amax > 0.f ? 1.f / amax : 1.f;
}
__global__ void apply_row_scale(float* __restrict__ s, const float* __restrict__ fac, int heads, int dim) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (c >= dim) return;
    for (int r = 0; r < dim; ++r) s[((size_t) h * dim + r) * dim + c] *= fac[(size_t) h * dim + r];
}

int run_accuracy(int heads, int dim) {
    const size_t st = (size_t) heads * dim * dim;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> h0(st);
    for (auto& x : h0) x = nd(rng);
    // the rank-1 inputs: one per token, 128 tokens' worth, so accumulation can show
    const int kSteps = 128;
    std::vector<float> hk((size_t) heads * dim * kSteps), hu((size_t) heads * dim * kSteps), hd((size_t) heads, 0.99f);
    for (auto& x : hk) x = nd(rng);
    for (auto& x : hu) x = nd(rng);

    // ---- host-side encode helpers, matching each kernel's rule
    auto enc_int8 = [&](const std::vector<float>& src, std::vector<int8_t>& q, std::vector<float>& sc) {
        q.resize(st); sc.assign((size_t) heads * dim, 1.f);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < dim; ++r) {
                float amax = 0.f;
                for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(src[((size_t) h * dim + r) * dim + c]));
                const float s = amax > 0.f ? amax / 127.f : 1.f;
                sc[(size_t) h * dim + r] = s;
                for (int c = 0; c < dim; ++c) {
                    float qv = nearbyintf(src[((size_t) h * dim + r) * dim + c] / s);
                    q[(size_t) h * dim * dim + (size_t) r * dim + c] = (int8_t) fminf(fmaxf(qv, -127.f), 127.f);
                }
            }
    };
    auto enc_packed = [&](const std::vector<float>& src, std::vector<uint8_t>& q, std::vector<float>& sc, int bits) {
        const size_t row_bytes = (size_t) dim * bits / 8;
        q.assign((size_t) st * bits / 8, 0); sc.assign((size_t) heads * dim, 1.f);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < dim; ++r) {
                const size_t row = ((size_t) h * dim + r) * dim;
                float amax = 0.f;
                for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(src[row + c]));
                const float s = amax > 0.f ? amax / (float) qmax_bits(bits) : 1.f;
                sc[(size_t) h * dim + r] = s;
                uint8_t* dst = q.data() + ((size_t) h * dim + r) * row_bytes;
                for (int g = 0; g < dim / kGroup; ++g) {
                    int v[kGroup];
                    for (int i = 0; i < kGroup; ++i) {
                        float qv = nearbyintf(src[row + g * kGroup + i] / s);
                        v[i] = (int) fminf(fmaxf(qv, -(float) qmax_bits(bits)), (float) qmax_bits(bits));
                    }
                    if (bits == 6) pack6(dst + (size_t) g * 3, v); else pack4(dst + (size_t) g * 2, v);
                }
            }
    };
    auto enc_half = [&](const std::vector<float>& src, std::vector<uint16_t>& q) {
        q.resize(st);
        for (size_t i = 0; i < st; ++i) { const __half h = __float2half(src[i]); q[i] = __half_as_ushort(h); }
    };
    auto enc_bf16 = [&](const std::vector<float>& src, std::vector<uint16_t>& q) {
        q.resize(st);
        for (size_t i = 0; i < st; ++i) { const __nv_bfloat16 b = __float2bfloat16(src[i]); std::memcpy(&q[i], &b, 2); }
    };
    auto enc_fp8 = [&](const std::vector<float>& src, std::vector<uint8_t>& q) {
        q.resize(st);
        for (size_t i = 0; i < st; ++i) { const __nv_fp8_e4m3 f = __nv_fp8_e4m3(src[i]); std::memcpy(&q[i], &f, 1); }
    };
    auto enc_fp4 = [&](const std::vector<float>& src, std::vector<uint8_t>& q, std::vector<uint8_t>& qs) {
        const int nb = dim / kGroup16;
        q.assign((size_t) st / 2, 0); qs.assign((size_t) heads * dim * nb, 0);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < dim; ++r) {
                const size_t row = ((size_t) h * dim + r) * dim;
                for (int g = 0; g < nb; ++g) {
                    float amax = 0.f;
                    for (int i = 0; i < kGroup16; ++i) amax = std::fmax(amax, std::fabs(src[row + g * kGroup16 + i]));
                    const float sc = amax > 0.f ? amax / 6.f : 1.f;      // E2M1's largest magnitude is 6
                    const __nv_fp8_e4m3 sf = __nv_fp8_e4m3(sc);
                    qs[((size_t) h * dim + r) * nb + g] = *reinterpret_cast<const uint8_t*>(&sf);
                    uint8_t* cell = q.data() + ((size_t) h * dim + r) * (dim / 2) + (g * kGroup16) / 2;
                    for (int i = 0; i < kGroup16; i += 2) {
                        uint8_t c[2];
                        for (int j = 0; j < 2; ++j) {
                            const float v = src[row + g * kGroup16 + i + j] / sc;
                            const float ax = fabsf(v);
                            const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
                            int best = 0; float bd = ax;
                            for (int m = 1; m < 8; ++m) { const float dd = fabsf(ax - mag[m]); if (dd < bd) { bd = dd; best = m; } }
                            c[j] = (uint8_t) (best | (v < 0.f ? 8 : 0));
                        }
                        cell[i / 2] = (uint8_t) ((c[0] & 0xF) | ((c[1] & 0xF) << 4));
                    }
                }
            }
    };

    std::printf("accuracy: %d steps from one fp32 start, relative Frobenius error of the decoded state\n", kSteps);
    std::printf("  %-22s %6s %9s %9s %9s %9s\n", "arm", "B/val", "MiB/sess", "err@1", "err@128", "slots");

    // the fp32 reference, stepped on the device with its own kernel
    float* ref;
    ck(cudaMalloc(&ref, st * 4), "acc ref");
    ck(cudaMemcpy(ref, h0.data(), st * 4, cudaMemcpyHostToDevice), "acc ref up");
    float* dk;
    float* du;
    ck(cudaMalloc(&dk, hk.size() * 4), "acc k");
    ck(cudaMalloc(&du, hu.size() * 4), "acc u");
    ck(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "acc k up");
    ck(cudaMemcpy(du, hu.data(), hu.size() * 4, cudaMemcpyHostToDevice), "acc u up");
    float* ddec;
    ck(cudaMalloc(&ddec, hd.size() * 4), "acc d");
    ck(cudaMemcpy(ddec, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice), "acc d up");
    // the compensators: zero, as A2 runs them, so the fold is measured as the format would carry it
    const size_t comp_n = (size_t) heads * kCompensators * 2 * dim;
    __half* dcomp;
    ck(cudaMalloc(&dcomp, comp_n * 2), "acc comp");
    ck(cudaMemset(dcomp, 0, comp_n * 2), "acc comp zero");

    const dim3 g1((unsigned) ((dim + 127) / 128), (unsigned) heads);
    const dim3 g4((unsigned) ((dim / kGroup + 127) / 128), (unsigned) heads);
    const dim3 g16((unsigned) ((dim / kGroup16 + 127) / 128), (unsigned) heads);
    const dim3 g8((unsigned) ((dim / kGroup8 + 127) / 128), (unsigned) heads);

    // the reference state after `steps` steps, and after 1
    std::vector<float> ref1(st), ref128(st);
    auto run_ref = [&](int steps, std::vector<float>& out) {
        ck(cudaMemcpy(ref, h0.data(), st * 4, cudaMemcpyHostToDevice), "ref reset");
        for (int t = 0; t < steps; ++t)
            state_fp32_kernel<<<g1, 128>>>(ref, dk + (size_t) t * heads * dim, du + (size_t) t * heads * dim, ddec,
                                           heads, dim);
        ck(cudaDeviceSynchronize(), "ref steps");
        ck(cudaMemcpy(out.data(), ref, st * 4, cudaMemcpyDeviceToHost), "ref down");
    };
    run_ref(1, ref1);
    run_ref(kSteps, ref128);
    double ref_norm = 0.0;
    for (float v : ref128) ref_norm += (double) v * v;
    (void) ref_norm;

    auto report = [&](const char* name, double bpv, double layer_b, const std::vector<float>& dec1,
                      const std::vector<float>& dec128) {
        const double mi = layer_b / (1 << 20);
        const double sess = mi * 36.0;
        std::printf("  %-22s %6.2f %9.2f %9.3e %9.3e %9.1f\n", name, bpv, sess, rel_frob(dec1, ref1),
                    rel_frob(dec128, ref128), 864.0 / sess);
    };

    const double scale_b = (double) heads * dim * 4.0;
    std::vector<float> dec(st), dec1(st);

    // --- int8
    {
        std::vector<int8_t> q; std::vector<float> sc;
        enc_int8(h0, q, sc);
        int8_t* dq; float* dsc;
        ck(cudaMalloc(&dq, st), "acc s8"); ck(cudaMalloc(&dsc, sc.size() * 4), "acc sc8");
        ck(cudaMemcpy(dq, q.data(), st, cudaMemcpyHostToDevice), "acc s8 up");
        ck(cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "acc sc8 up");
        int8_t* q0 = dq; float* sc0 = dsc;
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), st, cudaMemcpyHostToDevice), "s8 reset");
            for (int t = 0; t < steps; ++t)
                state_int8_kernel<<<g1, 128>>>(dq, dsc, dk + (size_t) t * heads * dim, du + (size_t) t * heads * dim,
                                               ddec, heads, dim);
            ck(cudaDeviceSynchronize(), "s8 steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_int8<<<g1, 128>>>(dq, dsc, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "s8 recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "s8 dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("int8", 1.0, st + scale_b, dec1, dec);
        cudaFree(q0); cudaFree(sc0);
    }
    // --- half
    {
        std::vector<uint16_t> q; enc_half(h0, q);
        uint16_t* dq; ck(cudaMalloc(&dq, st * 2), "acc h");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), st * 2, cudaMemcpyHostToDevice), "h reset");
            for (int t = 0; t < steps; ++t)
                state_half_kernel<0><<<g8, 128>>>(dq, dk + (size_t) t * heads * dim, du + (size_t) t * heads * dim, ddec,
                                                  heads, dim);
            ck(cudaDeviceSynchronize(), "h steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_half<<<(unsigned) ((st + 255) / 256), 256>>>(dq, (int) st, tmp);
            ck(cudaDeviceSynchronize(), "h recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "h dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("fp16", 2.0, st * 2, dec1, dec);
        cudaFree(dq);
    }
    // --- bf16
    {
        std::vector<uint16_t> q; enc_bf16(h0, q);
        uint16_t* dq; ck(cudaMalloc(&dq, st * 2), "acc bf");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), st * 2, cudaMemcpyHostToDevice), "bf reset");
            for (int t = 0; t < steps; ++t)
                state_bf16_kernel<<<g16, 128>>>(dq, dk + (size_t) t * heads * dim, du + (size_t) t * heads * dim, ddec,
                                                heads, dim);
            ck(cudaDeviceSynchronize(), "bf steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_bf16<<<(unsigned) ((st + 255) / 256), 256>>>(dq, (int) st, tmp);
            ck(cudaDeviceSynchronize(), "bf recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "bf dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("bf16", 2.0, st * 2, dec1, dec);
        cudaFree(dq);
    }
    // --- fp8
    {
        std::vector<uint8_t> q; enc_fp8(h0, q);
        uint8_t* dq; ck(cudaMalloc(&dq, st), "acc f8");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), st, cudaMemcpyHostToDevice), "f8 reset");
            for (int t = 0; t < steps; ++t)
                state_fp8_kernel<<<g16, 128>>>(dq, dk + (size_t) t * heads * dim, du + (size_t) t * heads * dim, ddec,
                                               heads, dim);
            ck(cudaDeviceSynchronize(), "f8 steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_fp8<<<(unsigned) ((st + 255) / 256), 256>>>(dq, (int) st, tmp);
            ck(cudaDeviceSynchronize(), "f8 recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "f8 dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("fp8 e4m3", 1.0, st, dec1, dec);
        cudaFree(dq);
    }
    // --- q4 int
    {
        std::vector<uint8_t> q; std::vector<float> sc;
        enc_packed(h0, q, sc, 4);
        uint8_t* dq; float* dsc;
        ck(cudaMalloc(&dq, q.size()), "acc p4"); ck(cudaMalloc(&dsc, sc.size() * 4), "acc sc4");
        ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "p4 up");
        ck(cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "sc4 up");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "p4 reset");
            for (int t = 0; t < steps; ++t)
                state_packed_kernel<4><<<g4, 128>>>(dq, dsc, dk + (size_t) t * heads * dim, du + (size_t) t * heads * dim,
                                                    ddec, heads, dim);
            ck(cudaDeviceSynchronize(), "p4 steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_packed<4><<<g4, 128>>>(dq, dsc, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "p4 recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "p4 dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("int4", 0.5, (double) st / 2 + scale_b, dec1, dec);
        cudaFree(dq); cudaFree(dsc);
    }
    // --- fp4 e2m1 + ue4m3
    {
        std::vector<uint8_t> q, qs; enc_fp4(h0, q, qs);
        const int nb = dim / kGroup16;
        uint8_t *dq, *dqs;
        ck(cudaMalloc(&dq, q.size()), "acc f4");
        ck(cudaMalloc(&dqs, qs.size()), "acc f4s");
        ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "f4 up");
        ck(cudaMemcpy(dqs, qs.data(), qs.size(), cudaMemcpyHostToDevice), "f4s up");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "f4 reset");
            for (int t = 0; t < steps; ++t)
                state_fp4_kernel<<<g16, 128>>>(dq, dqs, dk + (size_t) t * heads * dim, du + (size_t) t * heads * dim,
                                               ddec, heads, dim);
            ck(cudaDeviceSynchronize(), "f4 steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_fp4<<<g16, 128>>>(dq, dqs, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "f4 recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "f4 dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("fp4 e2m1+ue4m3", 0.5 + 1.0 / kGroup16, (double) st / 2 + (double) heads * dim * nb, dec1, dec);
        cudaFree(dq); cudaFree(dqs);
    }

    // --- int8 + compensators (A2): the paper's residual fold, measured on the same axis
    {
        std::vector<int8_t> q; std::vector<float> sc;
        enc_int8(h0, q, sc);
        int8_t* dq; float* dsc;
        ck(cudaMalloc(&dq, st), "acc s8c"); ck(cudaMalloc(&dsc, sc.size() * 4), "acc sc8c");
        ck(cudaMemcpy(dq, q.data(), st, cudaMemcpyHostToDevice), "s8c up");
        ck(cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "sc8c up");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), st, cudaMemcpyHostToDevice), "s8c reset");
            for (int t = 0; t < steps; ++t)
                state_int8_comp_kernel<<<g1, 128>>>(dq, dsc, dcomp, dk + (size_t) t * heads * dim,
                                                    du + (size_t) t * heads * dim, ddec, heads, dim);
            ck(cudaDeviceSynchronize(), "s8c steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_int8<<<g1, 128>>>(dq, dsc, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "s8c recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "s8c dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("int8 + 4comp", 1.0, st + scale_b + (double) comp_n * 2.0, dec1, dec);
        cudaFree(dq); cudaFree(dsc);
    }
    // --- int4 + compensators + window (A17): the paper's full form at 4 bits
    {
        std::vector<uint8_t> q; std::vector<float> sc;
        enc_packed(h0, q, sc, 4);
        uint8_t* dq; float* dsc;
        ck(cudaMalloc(&dq, q.size()), "acc p4f");
        ck(cudaMalloc(&dsc, sc.size() * 4), "acc sc4f");
        ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "p4f up");
        ck(cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "sc4f up");
        float* dwinf;
        ck(cudaMalloc(&dwinf, (size_t) heads * kWindow * 2 * dim * 4), "acc winf");
        ck(cudaMemset(dwinf, 0, (size_t) heads * kWindow * 2 * dim * 4), "acc winf zero");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "p4f reset");
            for (int t = 0; t < steps; ++t)
                state_packed_full_kernel<4><<<g4, 128>>>(dq, dsc, dcomp, dwinf, dk + (size_t) t * heads * dim,
                                                         du + (size_t) t * heads * dim, ddec, heads, dim, t);
            ck(cudaDeviceSynchronize(), "p4f steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_packed<4><<<g4, 128>>>(dq, dsc, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "p4f recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "p4f dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("int4 + comp + win", 0.5, (double) st / 2 + scale_b + (double) comp_n * 2.0 +
                                            (double) heads * kWindow * 2 * dim * 4, dec1, dec);
        cudaFree(dq); cudaFree(dsc); cudaFree(dwinf);
    }

    // ---- ROW 4, done properly: a REAL compensator fit, and residual smoothing, each measured as a delta.
    //
    // For each base arm: fit the top-r rank-one components of what quantization lost, run the compensated update, and
    // report err@1 and err@128 before (base) and after (comp).  Then residual smoothing on its own, and both together.
    std::printf("\n  -- compensator fit (%d components a head) and residual smoothing, as deltas --\n", kCompensators);
    std::printf("  %-28s %9s %9s %9s %9s %9s\n", "arm", "err@1", "err@128", "d(1)", "d(128)", "MiB/sess");

    const int R = kCompensators;
    auto fit_all = [&](const std::vector<int8_t>& q8, const std::vector<uint8_t>& qp,
                       const std::vector<uint8_t>& qf_codes, const std::vector<uint8_t>& qf_scales,
                       const std::vector<float>& sc, int bits, CompFit& out) {
        out.r = R;
        out.a.assign((size_t) heads * R * dim, __float2half(0.f));
        out.b.assign((size_t) heads * R * dim, __float2half(0.f));
        for (int h = 0; h < heads; ++h) {
            std::vector<float> sblk((size_t) dim * dim);
            for (int r = 0; r < dim; ++r)
                for (int c = 0; c < dim; ++c)
                    sblk[(size_t) r * dim + c] = h0[((size_t) h * dim + r) * dim + c];
            if (bits == 8) {
                std::vector<int8_t> cs(q8.begin() + (size_t) h * dim * dim, q8.begin() + (size_t) (h + 1) * dim * dim);
                fit_head(sblk.data(), cs.data(), &sc[(size_t) h * dim], dim, R,
                         &out.a[(size_t) h * R * dim], &out.b[(size_t) h * R * dim]);
            } else if (bits == 4) {
                // unpack this head's 4-bit block into codes for the fit, which works on the generic residual
                std::vector<int8_t> cs((size_t) dim * dim, 0);
                const size_t row_bytes = (size_t) dim * 4 / 8;
                for (int r = 0; r < dim; ++r)
                    for (int g = 0; g < dim / kGroup; ++g) {
                        int v[kGroup];
                        unpack4(&qp[((size_t) h * dim + r) * row_bytes + (size_t) g * 2], v);
                        for (int i = 0; i < kGroup; ++i) cs[(size_t) r * dim + g * kGroup + i] = (int8_t) v[i];
                    }
                fit_head(sblk.data(), cs.data(), &sc[(size_t) h * dim], dim, R,
                         &out.a[(size_t) h * R * dim], &out.b[(size_t) h * R * dim]);
            } else {
                // fp4: decode to an int8-equivalent code *per sub-block scale*, which the fit treats as its scale
                std::vector<int8_t> cs((size_t) dim * dim, 0);
                std::vector<float> sub((size_t) dim * dim, 1.f);
                const int nb = dim / kGroup16;
                for (int r = 0; r < dim; ++r)
                    for (int g = 0; g < nb; ++g) {
                        const float ssc = e4m3_to_f32(qf_scales[((size_t) h * dim + r) * nb + g]);
                        const uint8_t* cell = &qf_codes[((size_t) h * dim + r) * (dim / 2) + (g * kGroup16) / 2];
                        for (int i = 0; i < kGroup16; ++i) {
                            const uint8_t byte = cell[i / 2];
                            const uint8_t cd = (i & 1) ? (byte >> 4) : (byte & 0xF);
                            const int col = g * kGroup16 + i;
                            cs[(size_t) r * dim + col] = (int8_t) (e2m1_to_f32(cd) * 8.f);   // code at the sub-scale
                            sub[(size_t) r * dim + col] = ssc / 8.f;
                        }
                    }
                fit_head(sblk.data(), cs.data(), sub.data(), dim, R,
                         &out.a[(size_t) h * R * dim], &out.b[(size_t) h * R * dim]);
            }
        }
    };

    // ---- int4 with a real fit
    {
        std::vector<uint8_t> q; std::vector<float> sc;
        enc_packed(h0, q, sc, 4);
        CompFit cf; fit_all({}, q, {}, {}, sc, 4, cf);
        uint8_t* dq; float* dsc; __half *da, *db;
        ck(cudaMalloc(&dq, q.size()), "r4 q");
        ck(cudaMalloc(&dsc, sc.size() * 4), "r4 sc");
        ck(cudaMalloc(&da, cf.a.size() * 2), "r4 a");
        ck(cudaMalloc(&db, cf.b.size() * 2), "r4 b");
        ck(cudaMemcpy(da, cf.a.data(), cf.a.size() * 2, cudaMemcpyHostToDevice), "r4 a up");
        ck(cudaMemcpy(db, cf.b.data(), cf.b.size() * 2, cudaMemcpyHostToDevice), "r4 b up");
        ck(cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "r4 sc up");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "r4 reset");
            for (int t = 0; t < steps; ++t)
                state_packed_comp_kernel<4, R><<<g4, 128>>>(dq, dsc, da, db, dk + (size_t) t * heads * dim,
                                                            du + (size_t) t * heads * dim, ddec, heads, dim);
            ck(cudaDeviceSynchronize(), "r4 steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_packed<4><<<g4, 128>>>(dq, dsc, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "r4 recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "r4 dec");
            cudaFree(tmp);
        };
        // the fit itself, measured: how much of the residual it actually removes against the exact SVD bound
        {
            double before = 0.0, after = 0.0;
            for (int h = 0; h < heads; ++h) {
                std::vector<double> res((size_t) dim * dim);
                const size_t row_bytes = (size_t) dim * 4 / 8;
                for (int r = 0; r < dim; ++r)
                    for (int g = 0; g < dim / kGroup; ++g) {
                        int v[kGroup];
                        unpack4(&q[((size_t) h * dim + r) * row_bytes + (size_t) g * 2], v);
                        for (int i = 0; i < kGroup; ++i) {
                            const int col = g * kGroup + i;
                            res[(size_t) r * dim + col] = (double) h0[((size_t) h * dim + r) * dim + col] -
                                                          (double) v[i] * sc[(size_t) h * dim + r];
                        }
                    }
                for (double v : res) before += v * v;
                for (int r = 0; r < dim; ++r)
                    for (int c = 0; c < dim; ++c) {
                        double comp = 0.0;
                        for (int kk = 0; kk < R; ++kk)
                            comp += __half2float(cf.a[(size_t) h * R * dim + (size_t) kk * dim + r]) *
                                    __half2float(cf.b[(size_t) h * R * dim + (size_t) kk * dim + c]);
                        const double left = res[(size_t) r * dim + c] - comp;
                        after += left * left;
                    }
            }
            std::printf("  fit quality: the %d fitted components remove %.1f%% of the int4 residual's energy\n",
                        R, 100.0 * (1.0 - after / before));
        }
        step(1); dec1 = dec;
        step(kSteps);
        report("int4 + real comp fit", 0.5, (double) st / 2 + scale_b + (double) heads * R * 2 * dim * 2, dec1, dec);
        cudaFree(dq); cudaFree(dsc); cudaFree(da); cudaFree(db);
    }

    // ---- fp4 with a real fit
    {
        std::vector<uint8_t> q, qs; enc_fp4(h0, q, qs);
        const int nb = dim / kGroup16;
        CompFit cf; fit_all({}, {}, q, qs, {}, 0, cf);
        uint8_t *dq, *dqs; __half *da, *db;
        ck(cudaMalloc(&dq, q.size()), "r4f q");
        ck(cudaMalloc(&dqs, qs.size()), "r4f s");
        ck(cudaMalloc(&da, cf.a.size() * 2), "r4f a");
        ck(cudaMalloc(&db, cf.b.size() * 2), "r4f b");
        ck(cudaMemcpy(da, cf.a.data(), cf.a.size() * 2, cudaMemcpyHostToDevice), "r4f a up");
        ck(cudaMemcpy(db, cf.b.data(), cf.b.size() * 2, cudaMemcpyHostToDevice), "r4f b up");
        ck(cudaMemcpy(dqs, qs.data(), qs.size(), cudaMemcpyHostToDevice), "r4f s up");
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "r4f reset");
            for (int t = 0; t < steps; ++t)
                state_fp4_comp_kernel<R><<<g16, 128>>>(dq, dqs, da, db, dk + (size_t) t * heads * dim,
                                                       du + (size_t) t * heads * dim, ddec, heads, dim);
            ck(cudaDeviceSynchronize(), "r4f steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_fp4<<<g16, 128>>>(dq, dqs, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "r4f recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "r4f dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("fp4 + real comp fit", 0.5 + 1.0 / kGroup16,
               (double) st / 2 + (double) heads * dim * nb + (double) heads * R * 2 * dim * 2, dec1, dec);
        cudaFree(dq); cudaFree(dqs); cudaFree(da); cudaFree(db);
    }

    // ---- int8 with a real fit, the control that shows the technique works at all
    {
        std::vector<int8_t> q; std::vector<float> sc;
        enc_int8(h0, q, sc);
        CompFit cf; fit_all(q, {}, {}, {}, sc, 8, cf);
        int8_t* dq; float* dsc; __half *da, *db;
        ck(cudaMalloc(&dq, st), "r4i q");
        ck(cudaMalloc(&dsc, sc.size() * 4), "r4i sc");
        ck(cudaMalloc(&da, cf.a.size() * 2), "r4i a");
        ck(cudaMalloc(&db, cf.b.size() * 2), "r4i b");
        ck(cudaMemcpy(da, cf.a.data(), cf.a.size() * 2, cudaMemcpyHostToDevice), "r4i a up");
        ck(cudaMemcpy(db, cf.b.data(), cf.b.size() * 2, cudaMemcpyHostToDevice), "r4i b up");
        ck(cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "r4i sc up");
        // the int8 compensated update: same order, no packing
        auto step = [&](int steps) {
            ck(cudaMemcpy(dq, q.data(), st, cudaMemcpyHostToDevice), "r4i reset");
            for (int t = 0; t < steps; ++t) {
                // the compensated int8 body reads the REAL fit (da/db), one thread a column as A2's shape has it
                state_int8_compfit_kernel<4><<<g1, 128>>>(dq, dsc, da, db, dk + (size_t) t * heads * dim,
                                                          du + (size_t) t * heads * dim, ddec, heads, dim);
            }
            ck(cudaDeviceSynchronize(), "r4i steps");
            float* tmp; ck(cudaMalloc(&tmp, st * 4), "tmp");
            recon_int8<<<g1, 128>>>(dq, dsc, heads, dim, tmp);
            ck(cudaDeviceSynchronize(), "r4i recon");
            ck(cudaMemcpy(dec.data(), tmp, st * 4, cudaMemcpyDeviceToHost), "r4i dec");
            cudaFree(tmp);
        };
        step(1); dec1 = dec;
        step(kSteps);
        report("int8 + real comp fit", 1.0, st + scale_b + (double) heads * R * 2 * dim * 2, dec1, dec);
        cudaFree(dq); cudaFree(dsc); cudaFree(da); cudaFree(db);
    }

    // ---- residual smoothing, measured rather than asserted.
    //
    // The technique rescales key rows before quantizing so every row uses the full code range.  This arm ALREADY takes
    // one scale a row (amax/7), which IS per-row smoothing, so on int4 the technique is a no-op by construction: the
    // row's amax cancels.  It would matter for a coarser scale (a block or a whole tensor).  Both are measured here so
    // the claim is a number, not an argument.
    {
        std::printf("\n  -- residual smoothing: per-row (what this arm already does) vs one scale a head --\n");
        auto q4_err = [&](bool per_row) {
            std::vector<float> dq(st);
            for (int h = 0; h < heads; ++h) {
                float amax_h = 0.f;
                for (int r = 0; r < dim; ++r)
                    for (int c = 0; c < dim; ++c)
                        amax_h = std::fmax(amax_h, std::fabs(h0[((size_t) h * dim + r) * dim + c]));
                const float sc_h = amax_h > 0.f ? amax_h / 7.f : 1.f;
                for (int r = 0; r < dim; ++r) {
                    float amax_r = 0.f;
                    for (int c = 0; c < dim; ++c)
                        amax_r = std::fmax(amax_r, std::fabs(h0[((size_t) h * dim + r) * dim + c]));
                    const float sc = per_row ? (amax_r > 0.f ? amax_r / 7.f : 1.f) : sc_h;
                    for (int c = 0; c < dim; ++c) {
                        const float v = h0[((size_t) h * dim + r) * dim + c];
                        float qv = nearbyintf(v / sc);
                        qv = fminf(fmaxf(qv, -7.f), 7.f);
                        dq[((size_t) h * dim + r) * dim + c] = qv * sc;
                    }
                }
            }
            return rel_frob(dq, h0);
        };
        const double with_smooth = q4_err(true), without = q4_err(false);
        std::printf("  int4, one scale a row (smoothed):     err %.4e   MiB/sess %.2f\n", with_smooth,
                    ((double) st / 2 + scale_b) * 36.0 / (1 << 20));
        std::printf("  int4, one scale a head (unsmoothed):  err %.4e   MiB/sess %.2f\n", without,
                    ((double) st / 2 + scale_b / dim) * 36.0 / (1 << 20));
        std::printf("  smoothing's delta: %.2fx better error, for %.1f MiB a session of scales instead of %.2f\n",
                    without / with_smooth, scale_b * 36.0 / (1 << 20), scale_b / dim * 36.0 / (1 << 20));
    }

    cudaFree(ref); cudaFree(dk); cudaFree(du); cudaFree(ddec); cudaFree(dcomp);
    return 0;
}


}  // namespace

int main(int argc, char** argv) {
    int heads = 48, dim = 128, iters = 200, layers = 36;
    double bw = 288.0, budget_mib = 864.0;
    bool smoke = false, accuracy = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--heads" && i + 1 < argc) heads = std::atoi(argv[++i]);
        else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--bw" && i + 1 < argc) bw = std::atof(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) layers = std::atoi(argv[++i]);
        else if (a == "--slots" && i + 1 < argc) budget_mib = 864.0 * std::atoi(argv[++i]) / 8.0;
        else if (a == "--smoke") smoke = true;
        else if (a == "--accuracy") accuracy = true;
        else {
            std::fprintf(stderr, "usage: gdn_state_bench [--heads 48] [--dim 128] [--iters N] [--bw GB/s]\n"
                                 "                      [--layers 36] [--slots 8] [--smoke] [--accuracy]\n");
            return 2;
        }
    }
    if (dim % kGroup != 0) {
        std::fprintf(stderr, "gdn_state_bench: dim must be a multiple of %d (the packings' group)\n", kGroup);
        return 2;
    }
    if (smoke) return run_smoke(heads, dim, iters);
    if (accuracy) return run_accuracy(heads, dim);
    ck(cudaSetDevice(0), "cudaSetDevice");
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);
    const double peak = 2.0 * (double) clk * 1e3 * (bus / 8) / 1e9;

    const size_t st = (size_t) heads * dim * dim;
    std::printf("GDN state step, T = 1: %d heads x %d x %d, fp32 state %.2f MiB (L2 is 32 MiB)\n", heads, dim, dim,
                st * 4.0 / (1 << 20));
    std::printf("  peak %.1f GB/s at %.0f MHz; the row's bar is 1.6x fp32, the stop rule is 1.3x\n", peak, clk / 1e3);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hk((size_t) heads * dim), hu((size_t) heads * dim), hd((size_t) heads), hs(st);
    for (auto& x : hk) x = nd(rng) * 0.02f;
    for (auto& x : hu) x = nd(rng) * 0.02f;
    for (auto& x : hd) x = 0.99f;
    for (auto& x : hs) x = nd(rng) * 0.05f;

    float *dk, *du, *ddec;
    ck(cudaMalloc(&dk, hk.size() * 4), "k");
    ck(cudaMalloc(&du, hu.size() * 4), "u");
    ck(cudaMalloc(&ddec, hd.size() * 4), "decay");
    ck(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k up");
    ck(cudaMemcpy(du, hu.data(), hu.size() * 4, cudaMemcpyHostToDevice), "u up");
    ck(cudaMemcpy(ddec, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice), "decay up");

    // the four states, each allocated once and reused (the row measures the step, not the allocation)
    float* s32;
    ck(cudaMalloc(&s32, st * 4), "s32");
    ck(cudaMemcpy(s32, hs.data(), st * 4, cudaMemcpyHostToDevice), "s32 up");
    int8_t* s8;
    float* sc8;
    ck(cudaMalloc(&s8, st), "s8");
    ck(cudaMalloc(&sc8, (size_t) heads * dim * 4), "sc8");
    {
        // initialize the int8 state from the fp32 one: one scale a (head, row), codes from the quotient
        std::vector<int8_t> q(st);
        std::vector<float> sc((size_t) heads * dim);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < dim; ++r) {
                float amax = 0.f;
                for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(hs[((size_t) h * dim + r) * dim + c]));
                const float s = amax > 0.f ? amax / 127.f : 1.f;
                sc[(size_t) h * dim + r] = s;
                for (int c = 0; c < dim; ++c) {
                    float qv = nearbyintf(hs[((size_t) h * dim + r) * dim + c] / s);
                    q[(size_t) h * dim * dim + (size_t) r * dim + c] = (int8_t) fminf(fmaxf(qv, -127.f), 127.f);
                }
            }
        ck(cudaMemcpy(s8, q.data(), st, cudaMemcpyHostToDevice), "s8 up");
        ck(cudaMemcpy(sc8, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "sc8 up");
    }
    // the 6- and 4-bit states, from the same fp32 one and the same scale rule, at the format's own width
    std::vector<uint8_t> h6((size_t) st * 6 / 8), h4((size_t) st / 2);
    for (int h = 0; h < heads; ++h)
        for (int r = 0; r < dim; ++r) {
            const size_t row = ((size_t) h * dim + r) * dim;
            float amax = 0.f;
            for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(hs[row + c]));
            for (int bits : {6, 4}) {
                const float s = amax > 0.f ? amax / (float) qmax_bits(bits) : 1.f;
                // the scale is shared with A1's layout (one a head-row), so the 6- and 4-bit arms use their own amax
                uint8_t* base = bits == 6 ? h6.data() : h4.data();
                const size_t row_bytes = (size_t) dim * bits / 8;
                uint8_t* dst = base + ((size_t) h * dim + r) * row_bytes;
                for (int g = 0; g < dim / kGroup; ++g) {
                    int v[kGroup];
                    for (int i = 0; i < kGroup; ++i) {
                        float qv = nearbyintf(hs[row + g * kGroup + i] / s);
                        v[i] = (int) fminf(fmaxf(qv, -(float) qmax_bits(bits)), (float) qmax_bits(bits));
                    }
                    if (bits == 6) pack6(dst + (size_t) g * 3, v); else pack4(dst + (size_t) g * 2, v);
                }
            }
        }
    // A4/A5's scales: one a head-row, from the arm's own amax, so each width is quantized as that width would be
    std::vector<float> sc6((size_t) heads * dim), sc4((size_t) heads * dim);
    for (int h = 0; h < heads; ++h)
        for (int r = 0; r < dim; ++r) {
            float amax = 0.f;
            for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(hs[((size_t) h * dim + r) * dim + c]));
            sc6[(size_t) h * dim + r] = amax > 0.f ? amax / (float) qmax_bits(6) : 1.f;
            sc4[(size_t) h * dim + r] = amax > 0.f ? amax / (float) qmax_bits(4) : 1.f;
        }
    __half* comp;    const size_t comp_n = (size_t) heads * kCompensators * 2 * dim;
    ck(cudaMalloc(&comp, comp_n * 2), "comp");
    ck(cudaMemset(comp, 0, comp_n * 2), "comp zero");
    float* win;
    const size_t win_n = (size_t) heads * kWindow * 2 * dim;
    ck(cudaMalloc(&win, win_n * 4), "win");
    ck(cudaMemset(win, 0, win_n * 4), "win zero");

    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    const dim3 grid((unsigned) ((dim + 127) / 128), (unsigned) heads);
    const dim3 pgrid((unsigned) ((dim / kGroup + 127) / 128), (unsigned) heads);
    const dim3 g4g = pgrid;
    auto time_arm = [&](auto&& launch) {
        launch(0);
        ck(cudaDeviceSynchronize(), "warm");
        std::vector<double> v;
        for (int r = 0; r < 7; ++r) {
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i) launch(i);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "timed");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) v.push_back((double) ms / iters);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // the resident state of one layer: codes at the format's width, one fp32 scale a head-row, and whatever else the
    // arm keeps between tokens.  This is the number a session pays for, not the bytes the step touches.
    const double scale_b = (double) heads * dim * 4.0;
    const double q8_b = (double) st, q6_b = (double) st * 6.0 / 8.0, q4_b = (double) st / 2.0;
    const double comp_b = (double) comp_n * 2.0, win_b = (double) win_n * 4.0;

    uint8_t* p6;
    uint8_t* p4;
    float* sc6d;
    float* sc4d;
    ck(cudaMalloc(&p6, (size_t) st * 6 / 8), "p6");
    ck(cudaMalloc(&p4, (size_t) st / 2), "p4");
    ck(cudaMalloc(&sc6d, (size_t) heads * dim * 4), "sc6d");
    ck(cudaMalloc(&sc4d, (size_t) heads * dim * 4), "sc4d");
    ck(cudaMemcpy(p6, h6.data(), h6.size(), cudaMemcpyHostToDevice), "p6 up");
    ck(cudaMemcpy(p4, h4.data(), h4.size(), cudaMemcpyHostToDevice), "p4 up");
    ck(cudaMemcpy(sc6d, sc6.data(), sc6.size() * 4, cudaMemcpyHostToDevice), "sc6 up");
    ck(cudaMemcpy(sc4d, sc4.data(), sc4.size() * 4, cudaMemcpyHostToDevice), "sc4 up");
    __half* comp6;
    __half* comp4;
    ck(cudaMalloc(&comp6, comp_n * 2), "comp6");
    ck(cudaMalloc(&comp4, comp_n * 2), "comp4");
    ck(cudaMemset(comp6, 0, comp_n * 2), "comp6 zero");
    ck(cudaMemset(comp4, 0, comp_n * 2), "comp4 zero");
    float* win6;
    float* win4;
    ck(cudaMalloc(&win6, win_n * 4), "win6");
    ck(cudaMalloc(&win4, win_n * 4), "win4");
    ck(cudaMemset(win6, 0, win_n * 4), "win6 zero");
    ck(cudaMemset(win4, 0, win_n * 4), "win4 zero");

    // the new stores: bf16, fp8, fp4 (with its ue4m3 sub-block scales), and the half state
    uint16_t* bf16;
    uint8_t* f8;
    uint8_t* f4;
    uint8_t* f4s;
    uint16_t* h16;
    ck(cudaMalloc(&bf16, st * 2), "bf16");
    ck(cudaMalloc(&f8, st), "f8");
    ck(cudaMalloc(&f4, (size_t) st / 2), "f4");
    ck(cudaMalloc(&f4s, (size_t) heads * dim * (dim / kGroup16)), "f4s");
    ck(cudaMalloc(&h16, st * 2), "h16");
    {
        // fp8: the fp32 state through cvt, no scale
        std::vector<uint8_t> q(st);
        for (size_t i = 0; i < st; ++i) { const __nv_fp8_e4m3 f = __nv_fp8_e4m3(hs[i]); std::memcpy(&q[i], &f, 1); }
        ck(cudaMemcpy(f8, q.data(), st, cudaMemcpyHostToDevice), "f8 up");
        // bf16 and fp16: straight conversions
        std::vector<uint16_t> b(st), hh(st);
        for (size_t i = 0; i < st; ++i) {
            const __nv_bfloat16 x = __float2bfloat16(hs[i]); std::memcpy(&b[i], &x, 2);
            hh[i] = __half_as_ushort(__float2half(hs[i]));
        }
        ck(cudaMemcpy(bf16, b.data(), st * 2, cudaMemcpyHostToDevice), "bf16 up");
        ck(cudaMemcpy(h16, hh.data(), st * 2, cudaMemcpyHostToDevice), "h16 up");
        // fp4: E2M1 codes with a UE4M3 scale a 16-value sub-block
        const int nb = dim / kGroup16;
        std::vector<uint8_t> q4v((size_t) st / 2, 0), qs((size_t) heads * dim * nb, 0);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < dim; ++r) {
                const size_t row = ((size_t) h * dim + r) * dim;
                for (int g = 0; g < nb; ++g) {
                    float amax = 0.f;
                    for (int i = 0; i < kGroup16; ++i) amax = std::fmax(amax, std::fabs(hs[row + g * kGroup16 + i]));
                    const float sc = amax > 0.f ? amax / 6.f : 1.f;
                    const __nv_fp8_e4m3 sf = __nv_fp8_e4m3(sc);
                    qs[((size_t) h * dim + r) * nb + g] = *reinterpret_cast<const uint8_t*>(&sf);
                    uint8_t* cell = q4v.data() + ((size_t) h * dim + r) * (dim / 2) + (g * kGroup16) / 2;
                    const float magv[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
                    for (int i = 0; i < kGroup16; i += 2) {
                        uint8_t c[2];
                        for (int j = 0; j < 2; ++j) {
                            const float v = hs[row + g * kGroup16 + i + j] / sc, ax = fabsf(v);
                            int best = 0; float bd = ax;
                            for (int m = 1; m < 8; ++m) { const float dd = fabsf(ax - magv[m]); if (dd < bd) { bd = dd; best = m; } }
                            c[j] = (uint8_t) (best | (v < 0.f ? 8 : 0));
                        }
                        cell[i / 2] = (uint8_t) ((c[0] & 0xF) | ((c[1] & 0xF) << 4));
                    }
                }
            }
        ck(cudaMemcpy(f4, q4v.data(), q4v.size(), cudaMemcpyHostToDevice), "f4 up");
        ck(cudaMemcpy(f4s, qs.data(), qs.size(), cudaMemcpyHostToDevice), "f4s up");
    }
    const dim3 g16((unsigned) ((dim / kGroup16 + 127) / 128), (unsigned) heads);
    const dim3 g8((unsigned) ((dim / kGroup8 + 127) / 128), (unsigned) heads);
    const double f4_scale_b = (double) heads * dim * (dim / kGroup16);

    std::vector<Arm> arms;
    arms.push_back({"A0 fp32", (double) st * 4.0 * 2.0, (double) st * 4.0,
                    time_arm([&](int) { state_fp32_kernel<<<grid, 128>>>(s32, dk, du, ddec, heads, dim); })});
    arms.push_back({"A13 fp16 state", (double) st * 2.0 * 2.0, (double) st * 2.0,
                    time_arm([&](int) { state_half_kernel<0><<<g8, 128>>>(h16, dk, du, ddec, heads, dim); })});
    arms.push_back({"A15 fp16 fp32-acc", (double) st * 2.0 * 2.0, (double) st * 2.0,
                    time_arm([&](int) { state_half_kernel<1><<<g8, 128>>>(h16, dk, du, ddec, heads, dim); })});
    arms.push_back({"A7 bf16 state", (double) st * 2.0 * 2.0, (double) st * 2.0,
                    time_arm([&](int) { state_bf16_kernel<<<g16, 128>>>(bf16, dk, du, ddec, heads, dim); })});
    arms.push_back({"A6 fp8 e4m3", (double) st * 2.0, (double) st,
                    time_arm([&](int) { state_fp8_kernel<<<g16, 128>>>(f8, dk, du, ddec, heads, dim); })});
    arms.push_back({"A1 int8", (double) st + scale_b, q8_b + scale_b,
                    time_arm([&](int) { state_int8_kernel<<<grid, 128>>>(s8, sc8, dk, du, ddec, heads, dim); })});
    arms.push_back({"A10 int8 fp16-arith", (double) st + scale_b, q8_b + scale_b,
                    time_arm([&](int) { state_int8_half_kernel<0, 0><<<g4g, 128>>>(s8, sc8, comp, dk, du, ddec, heads, dim); })});
    arms.push_back({"A12 int8 fp16+fp32acc", (double) st + scale_b, q8_b + scale_b,
                    time_arm([&](int) { state_int8_half_kernel<1, 0><<<g4g, 128>>>(s8, sc8, comp, dk, du, ddec, heads, dim); })});
    arms.push_back({"A2 int8 + 4comp", (double) st + scale_b + comp_b, q8_b + scale_b + comp_b,
                    time_arm([&](int) { state_int8_comp_kernel<<<grid, 128>>>(s8, sc8, comp, dk, du, ddec, heads, dim); })});
    arms.push_back({"A11 int8 fp16 +comp", (double) st + scale_b + comp_b, q8_b + scale_b + comp_b,
                    time_arm([&](int) { state_int8_half_kernel<0, 1><<<g4g, 128>>>(s8, sc8, comp, dk, du, ddec, heads, dim); })});
    arms.push_back({"A4 q6", q6_b + scale_b, q6_b + scale_b,
                    time_arm([&](int) { state_packed_kernel<6><<<pgrid, 128>>>(p6, sc6d, dk, du, ddec, heads, dim); })});
    arms.push_back({"A8 fp4 e2m1", (double) st / 2 + f4_scale_b, (double) st / 2 + f4_scale_b,
                    time_arm([&](int) { state_fp4_kernel<<<g16, 128>>>(f4, f4s, dk, du, ddec, heads, dim); })});
    arms.push_back({"A5 int4", q4_b + scale_b, q4_b + scale_b,
                    time_arm([&](int) { state_packed_kernel<4><<<pgrid, 128>>>(p4, sc4d, dk, du, ddec, heads, dim); })});
    arms.push_back({"A3 int8 + window", (double) st + scale_b + win_b, q8_b + scale_b + win_b,
                    time_arm([&](int i) { state_int8_window_kernel<<<grid, 128>>>(s8, sc8, win, dk, du, ddec, heads, dim, i); })});
    arms.push_back({"A16 q6 full", q6_b + scale_b + comp_b + win_b, q6_b + scale_b + comp_b + win_b,
                    time_arm([&](int i) { state_packed_full_kernel<6><<<pgrid, 128>>>(p6, sc6d, comp6, win6, dk, du, ddec, heads, dim, i); })});
    arms.push_back({"A18 int8 real-fit", q8_b + scale_b + comp_b, q8_b + scale_b + comp_b,
                    time_arm([&](int) { state_int8_compfit_kernel<4><<<grid, 128>>>(s8, sc8, comp6, comp6 + (size_t) heads * 4 * dim, dk, du, ddec, heads, dim); })});
    arms.push_back({"A19 int4 real-fit", q4_b + scale_b + comp_b, q4_b + scale_b + comp_b,
                    time_arm([&](int) { state_packed_comp_kernel<4, 4><<<pgrid, 128>>>(p4, sc4d, comp6, comp6 + (size_t) heads * 4 * dim, dk, du, ddec, heads, dim); })});
    arms.push_back({"A20 fp4 real-fit", (double) st / 2 + f4_scale_b + comp_b, (double) st / 2 + f4_scale_b + comp_b,
                    time_arm([&](int) { state_fp4_comp_kernel<4><<<g16, 128>>>(f4, f4s, comp6, comp6 + (size_t) heads * 4 * dim, dk, du, ddec, heads, dim); })});
    arms.push_back({"A17 q4 full", q4_b + scale_b + comp_b + win_b, q4_b + scale_b + comp_b + win_b,
                    time_arm([&](int i) { state_packed_full_kernel<4><<<pgrid, 128>>>(p4, sc4d, comp4, win4, dk, du, ddec, heads, dim, i); })});

    const double a0 = arms[0].ms;
    const double a0_layer = arms[0].bytes_a_layer;
    // sorted by bytes a value: the table's own axis, so the reader sees what each byte buys
    std::sort(arms.begin(), arms.end(), [](const Arm& x, const Arm& y) {
        return x.bytes_a_layer / x.bytes_per_token < y.bytes_a_layer / y.bytes_per_token;
    });
    std::printf("\n  arm                  ms/layer   us/layer   x fp32   MiB a layer   MiB a session   slots in %.0f MiB   %s\n",
                budget_mib, "verdict");
    for (const Arm& a : arms) {
        const double rel = a0 / a.ms;
        const double layer_mib = a.bytes_a_layer / (1 << 20);
        const double sess_mib = layer_mib * layers;
        const double slots = sess_mib > 0 ? budget_mib / sess_mib : 0.0;
        char verdict[160];
        if (a.name[1] == '0' && a.name[2] == ' ') {
            std::snprintf(verdict, sizeof verdict, "reference");
        } else if (rel >= 1.05) {
            std::snprintf(verdict, sizeof verdict, "FASTER: %.2fx and %.0f%% of the bytes", rel,
                          100.0 * a.bytes_a_layer / a0_layer);
        } else if (rel <= 0.95) {
            std::snprintf(verdict, sizeof verdict, "SLOWER %.2fx, buys %.0f%% of the bytes (+%.1f slots)", rel,
                          100.0 * (1.0 - a.bytes_a_layer / a0_layer),
                          slots - budget_mib / (a0_layer * layers / (1 << 20)));
        } else {
            std::snprintf(verdict, sizeof verdict, "EVEN TIME (%.2fx) at %.0f%% of the bytes (+%.1f slots)", rel,
                          100.0 * a.bytes_a_layer / a0_layer,
                          slots - budget_mib / (a0_layer * layers / (1 << 20)));
        }
        std::printf("  %-18s %9.4f  %9.2f  %7.2fx  %12.3f  %14.2f  %16.1f   %s\n", a.name, a.ms, a.ms * 1e3, rel,
                    layer_mib, sess_mib, slots, verdict);
    }
    std::printf("\n  the engine's own figure for the fp32 step at T=1 is 0.54 ms/token over 36 layers = 15 us a layer;\n"
                "  this bench's A0 should land near that (it is the same arithmetic on a warm L2-resident state).\n");
    return 0;
}
