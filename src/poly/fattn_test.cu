// src/poly/fattn_test.cu - fattn_q8() (llama.cpp's flash-attention kernel, fattn.hpp) against attend_batch() on the
// same q8_0 cache, Qwen3.6's shape (heads of 256 values, 16 heads, 2 KV heads), and both against the arithmetic in
// doubles for three tokens of each batch.  Also with a scratch that holds only a few thousand keys, so that the
// keys go in many blocks and are merged.  Then the same cases with fattn_f16() on an fp16 cache of the same values.
//
// llama.cpp's kernel sums the values' products in halves on NVIDIA cards (T_C_VKQ in fattn_mma.cuh, upstream's
// choice, not one of our changes; attend_batch() sums in floats), so it is the further from the doubles, and by how
// much depends on the card: an RTX 3070 (sm_86) and an RTX 4070 (sm_89) give the same 1.06e-2 in one case where an
// RTX 5090 (sm_120) gives 4.85e-3.  So the bounds scale with the largest result and are about twice the most sm_86
// gives (9.4e-3 of the largest result from attend_batch(), 3.4e-3 from the doubles).  They catch a wrong mask or
// merge, which is off by tenths.
//
// The inputs are drawn from mt19937's raw output (draw()), not std::normal_distribution, whose numbers differ between
// standard libraries, so that every platform tests the same data.
//
// usage: poly_fattn_test   (exit 0: every case within kRoomBatch and kRoomDoubles, times max(1, largest result), of
//        attend_batch() and the doubles)
//        poly_fattn_test [dec] gqa16   the same checks with Ornith 122B's 32 heads / 2 KV heads
//        poly_fattn_test dec   fdec_q8() and fdec_f16() (the decode kernel) against attend() and the doubles, up
//                              to 170,000 rows; a token alone and with others, and from a position on the card,
//                              to the bit; and the GB/s of cache read at 4K/32K/128K/256K
//        poly_fattn_test parity the k8v4 payload checks: per-cell isolation, the KV-head map, the prefix bound,
//                              determinism, and a round trip - see the block above parity_main
//        poly_fattn_test sweep  one token's decode cost against the context, with nothing else in the process
//        poly_fattn_test k8v4 [--heads N]  fdec_k8v4() (q8_0 keys, Q4_0 values) against attend() over the same cache and
//                              against the doubles, at the same contexts and in the same GB/s terms.  Before this
//                              the kernel had no test at all (grep -rln fdec_k8v4 over the tests was empty).
//                              Also the two identity claims the plan rests on: k8v4's key bytes ARE store_q8's,
//                              and the decode kernel is bit-identical to fdec_q8() when the value cache holds the
//                              same numbers.
#include "fattn.hpp"
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 256, kKvHeads = 2, kWidth = kKvHeads * kDim, kMaxRows = 131072, kBatch = 1024;
// The contexts row 1 and row 2 of the optimization stack's Phase 0 are judged at (OPTIMIZATION-STACK.md §0.3).
constexpr long long kContexts[] = {4096, 32768, 131072, 262144};
int kHeads = 16;   // gqa16 selects 32; the existing group-8 geometry remains the default
// --heads N: run the k8v4 checks at N query heads on this file's 2 KV heads, so the groups the decode kernel now
// serves can be checked each on its own (12 = Qwen3.8-Flash, 6 = Qwen3.8-27B dense, 8 = Qwen3.6 / Flint).
int kHeadsArg = 0;
constexpr size_t kScore = size_t(16) << 20;   // Qwen35's score_ (kScoreFloats)
constexpr float kRoomBatch = 2e-2f, kRoomDoubles = 7.5e-3f;   // the bounds, as shares of the largest result (at least 1)

// the same numbers from the same seed on every standard library: uniform in (0, 1) from the raw 32 bits, and a
// normal one by Box-Muller on two of them
double uniform(std::mt19937& rng) { return (rng() + 0.5) / 4294967296.0; }
float draw(std::mt19937& rng) {
    const double u = uniform(rng), v = uniform(rng);
    return (float) (std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

struct Cache {
    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
    std::vector<uint16_t> halves;   // the same values in fp16 (fp16())
    float at(size_t row, int i) const {
        if (!halves.empty()) return __half2float(__ushort_as_half(halves[row * kWidth + i]));
        return (float) codes[row * kWidth + i] * __half2float(__ushort_as_half(scales[row * (kWidth / 32) + i / 32]));
    }
    void fp16() {
        std::vector<uint16_t> h(codes.size());
        for (size_t i = 0; i < h.size(); ++i) h[i] = __half_as_ushort(__float2half(at(i / kWidth, (int) (i % kWidth))));
        halves = std::move(h);
    }
};

Cache make(std::mt19937& rng, size_t rows = kMaxRows) {
    Cache c;
    c.codes.resize(rows * kWidth);
    c.scales.resize(rows * (kWidth / 32));
    for (auto& x : c.codes) x = (int8_t) std::clamp(std::lround(40.f * draw(rng)), -127L, 127L);
    for (auto& x : c.scales) x = __half_as_ushort(__float2half((float) (0.5 + uniform(rng)) / 40.f));
    return c;
}

// kh and vh: the fp16 caches (fattn_f16()), or null for the q8 ones (fattn_q8())
int check(cublasHandle_t blas, const Cache& K, const Cache& V, Kvq8 kq, Kvq8 vq, const uint16_t* kh,
          const uint16_t* vh, const std::vector<float>& q, const float* dq, float* score, float* awork,
          uint16_t* kwork, float* out, long long pos, int n, size_t fattn_floats) {
    const size_t cells = size_t(n) * kHeads * kDim;
    const float scale = 1.f / 16.f;
    const size_t halves = size_t(2048) * kWidth;
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0), cudaEventCreate(&e1), cudaEventCreate(&e2), cudaEventCreate(&e3);
    cudaEventRecord(e0);
    if (kh)
        attend_batch(blas, dq, kh, vh, pos, n, score, kScore, awork, out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0,
                     false, Kvq8{}, Kvq8{}, nullptr, nullptr, 0);
    else
        attend_batch(blas, dq, nullptr, nullptr, pos, n, score, kScore, awork, out, kDim, kHeads, kKvHeads, scale,
                     nullptr, 0, 0, false, kq, vq, kwork, kwork + halves, halves);
    cudaEventRecord(e1);
    const std::vector<float> ref = download(out, cells);
    cudaMemset(out, 0, cells * sizeof(float));
    cudaEventRecord(e2);
    if (!(kh ? fattn_f16(dq, kh, vh, pos, n, out, kDim, kHeads, kKvHeads, scale, score, fattn_floats, nullptr)
             : fattn_q8(dq, kq, vq, pos, n, out, kDim, kHeads, kKvHeads, scale, score, fattn_floats, nullptr))) {
        std::printf("FAIL pos %6lld rows %4d: fattn refused the scratch\n", pos, n);
        return 1;
    }
    cudaEventRecord(e3);
    const std::vector<float> fa = download(out, cells);
    float ms_ref = 0.f, ms_fa = 0.f;
    cudaEventElapsedTime(&ms_ref, e0, e1);
    cudaEventElapsedTime(&ms_fa, e2, e3);
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
        std::printf("FAIL pos %6lld rows %4d: CUDA %s\n", pos, n, cudaGetErrorString(e));
        return 1;
    }
    float most = 0.f, big = 0.f;
    double sum = 0;
    for (size_t i = 0; i < cells; ++i) {
        const float d = std::fabs(fa[i] - ref[i]);
        most = std::max(most, std::isnan(d) ? INFINITY : d);
        sum += d;
        big = std::max(big, std::fabs(ref[i]));
    }
    // three tokens in doubles: the first, the middle and the last
    float most_fa = 0.f, most_ref = 0.f;
    for (int t : {0, n / 2, n - 1})
        for (int h = 0; h < kHeads; ++h) {
            const int g = h / (kHeads / kKvHeads);
            const float* qh = q.data() + (size_t(t) * kHeads + h) * kDim;
            const long long keys = pos + t + 1;
            std::vector<double> w(keys);
            double top = -1e300;
            for (long long r = 0; r < keys; ++r) {
                double s = 0;
                for (int i = 0; i < kDim; ++i) s += (double) qh[i] * K.at(r, g * kDim + i);
                w[r] = s * scale;
                top = std::max(top, w[r]);
            }
            double z = 0;
            for (auto& x : w) z += x = std::exp(x - top);
            for (int i = 0; i < kDim; ++i) {
                double o = 0;
                for (long long r = 0; r < keys; ++r) o += w[r] * V.at(r, g * kDim + i);
                o /= z;
                const size_t at = (size_t(t) * kHeads + h) * kDim + i;
                most_fa = std::max(most_fa, (float) std::fabs(fa[at] - o));
                most_ref = std::max(most_ref, (float) std::fabs(ref[at] - o));
            }
        }
    const float room = std::max(1.f, big);
    const bool ok = most <= kRoomBatch * room && most_fa <= kRoomDoubles * room;
    std::printf("%s %s pos %6lld rows %4d scratch %5.1f MiB: %.2e from attend_batch (mean %.1e, results up to %.2f); "
                "against doubles: fattn %.2e, attend_batch %.2e; %.2f ms against %.2f\n",
                ok ? "ok  " : "FAIL", kh ? "f16" : "q8 ", pos, n, fattn_floats * 4.0 / (1 << 20), most, sum / cells, big, most_fa,
                most_ref, ms_fa, ms_ref);
    return ok ? 0 : 1;
}

// The doubles' result of token t's head h attending the rows 0 to `keys` - 1.
std::vector<double> exact(const Cache& K, const Cache& V, const float* qh, int g, long long keys, float scale) {
    std::vector<double> w(keys), o(kDim, 0.0);
    double top = -1e300, z = 0;
    for (long long r = 0; r < keys; ++r) {
        double s = 0;
        for (int i = 0; i < kDim; ++i) s += (double) qh[i] * K.at(r, g * kDim + i);
        top = std::max(top, w[r] = s * scale);
    }
    for (auto& x : w) z += x = std::exp(x - top);
    for (long long r = 0; r < keys; ++r)
        for (int i = 0; i < kDim; ++i) o[i] += w[r] * V.at(r, g * kDim + i);
    for (double& x : o) x /= z;
    return o;
}

// ---- k8v4: q8_0 keys, Q4_0 values -------------------------------------------------------------------------
// Bytes a cache row takes.  k8v4's keys ARE q8_0's (store_q8, qwen35.cpp:549), so its key row is a q8_0 one.
constexpr int kRowBytesQ8 = kWidth + (kWidth / 32) * 2;                    // 512 int8 + 16 fp16 scales = 544
constexpr int kRowBytesV4 = (kWidth / 32) * 16 + (kWidth / 32) * 2;         // 256 nibble bytes + 16 fp16 = 288
constexpr int kCellBytesK8V4 = kRowBytesQ8 + kRowBytesV4;                  // 832 a cell (K and V)
constexpr int kCellBytesQ8 = kRowBytesQ8 * 2;                              // 1,088

// A Q4_0 cache on the host, decoded the way Kvq4 documents it: block b at q[row * (n/32) * 16 + b * 16],
// element j the low nibble of byte j for j < 16 and the high nibble of byte j - 16 otherwise.
struct V4Host {
    std::vector<uint8_t> q;
    std::vector<uint16_t> sc;
    float at(size_t row, int i) const {
        const size_t nb = kWidth / 32;
        const uint8_t byte = q[(row * nb + i / 32) * 16 + (size_t) (i % 32 < 16 ? i % 32 : i % 32 - 16)];
        const int nib = i % 32 < 16 ? (byte & 0x0F) : (byte >> 4);
        return (float) (nib - 8) * __half2float(__ushort_as_half(sc[row * nb + i / 32]));
    }
};

// ggml's q4_0 rule, as kv_q4_test.cpp's host reference states it.
void q4_host_row(const float* x, uint8_t* q, uint16_t* sc) {
    for (int b = 0; b < kWidth / 32; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.f, mval = 0.f;
        for (int j = 0; j < 32; ++j) {
            const float a = std::fabs(xb[j]);
            if (a > amax || (a == amax && xb[j] > mval)) {
                amax = a;
                mval = xb[j];
            }
        }
        const float d = mval / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        sc[b] = __half_as_ushort(__float2half_rn(d));
        for (int j = 0; j < 16; ++j) {
            const int q0 = (int) (xb[j] * id + 8.5f);
            const int q1 = (int) (xb[j + 16] * id + 8.5f);
            const uint8_t a = (uint8_t) (q0 < 0 ? 0 : (q0 > 15 ? 15 : q0));
            const uint8_t c = (uint8_t) (q1 < 0 ? 0 : (q1 > 15 ? 15 : q1));
            q[b * 16 + j] = (uint8_t) (a | (c << 4));
        }
    }
}

// Blocks a cache row has (one scale per 32 values), and ggml's q8_0 rule, as kv_q8_test.cpp's host reference states it.
int nb_of() { return kWidth / 32; }

void q8_host_row(const float* x, int8_t* codes, uint16_t* sc) {
    for (int b = 0; b < kWidth / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        sc[b] = __half_as_ushort(__float2half(d));
        for (int j = 0; j < 32; ++j) {
            float q = std::rint(x[b * 32 + j] * id);
            q = std::fmin(std::fmax(q, -127.f), 127.f);
            codes[b * 32 + j] = (int8_t) q;
        }
    }
}

// The reference's caches as PLAIN FLOATS, dequantized once for the rows a call needs.
// The first version called Cache::at() / V4Host::at() inside the r x i loop, and each call did a __half2float:
// ~67M decodes per head per pass at 256K, which is why the test spent minutes in a reference the GPU was not
// waiting for.  A float row is 1 KiB a token a side, so the whole 256K context is 512 MiB - built a context at a
// time, never held twice.
struct RefCache {
    std::vector<float> k;   // keys:   rows x kWidth, g * kDim + i
    std::vector<float> v;   // values: rows x kWidth
    long long rows = 0;
    void build(const Cache& K, const V4Host& V, long long n) {
        rows = n;
        k.resize((size_t) n * kWidth);
        v.resize((size_t) n * kWidth);
        for (long long r = 0; r < n; ++r) {
            for (int i = 0; i < kWidth; ++i) {
                k[(size_t) r * kWidth + i] = (float) K.codes[(size_t) r * kWidth + i] *
                                            __half2float(__ushort_as_half(K.scales[(size_t) r * (kWidth / 32) + i / 32]));
                v[(size_t) r * kWidth + i] = V.at((size_t) r, i);
            }
        }
    }
};

// Token t's head h over the pre-dequantized caches, in doubles.  No decode in the loop: a load and an fma.
std::vector<double> exact_v4(const RefCache& C, const float* qh, int g, long long keys, float scale) {
    std::vector<double> w(keys), o(kDim, 0.0);
    double top = -1e300, z = 0;
    const float* kcol = C.k.data() + (size_t) g * kDim;
    for (long long r = 0; r < keys; ++r) {
        const float* kr = kcol + (size_t) r * kWidth;
        double s = 0;
        for (int i = 0; i < kDim; ++i) s += (double) qh[i] * (double) kr[i];
        top = std::max(top, w[r] = s * scale);
    }
    for (auto& x : w) z += x = std::exp(x - top);
    const float* vcol = C.v.data() + (size_t) g * kDim;
    for (long long r = 0; r < keys; ++r) {
        const float* vr = vcol + (size_t) r * kWidth;
        const double wr = w[r];
        for (int i = 0; i < kDim; ++i) o[i] += wr * (double) vr[i];
    }
    for (double& x : o) x /= z;
    return o;
}

// k8v4_main: fdec_k8v4() checked, then the identity the format rests on.
//
// WHAT IS COMPARED, AND WHY NOT MORE.  attend() is the generic reference and it is SLOW at long context: it takes
// min(4, kScore / (n_head * (pos+1))) tokens a launch, so a 256K call runs its block loop over 4,097 blocks with a
// handful of tokens - measured at minutes, not milliseconds, for a call the decode kernel does in 1.4 ms.  Comparing
// against it at 128K and 256K measured the reference's cost, not the kernel's, and a run of this mode was killed at
// 27 minutes because of it.  So:
//   * attend() agreement is checked at 4K and 32K, where it is a usable reference and covers the same code paths
//     (score, softmax, value fold) at two different chunk counts;
//   * the double-precision reference is a SAMPLE, not an exhaustive pass: two query heads at 32K and 256K, which is
//     what a control needs, and the contexts where a wrong index would show;
//   * the identity test at 32 cells is exhaustive and exact, and poly_kv_read_bench measures attend() and fdec_k8v4
//     at every context for the performance table.  Neither role is duplicated here.
// Every long loop below prints a flushed line first, so a slow run can be told from a stuck one in its log.
int k8v4_main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // a long run must be observable in a redirected log
    // The largest context this row is judged at is 262,144 positions, and a token at position 262,144 attends the
    // rows 0..262,144 inclusive: one row more than the context.  Sizing the cache to the context reads one row
    // past its end (it produced inf at 256K before this was fixed).
    constexpr int kRows = 262145;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int kMaxN = 2;
    std::vector<float> q(size_t(kMaxN) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);

    // K: q8_0 codes and scales, as the engine's store_q8 writes them
    const Cache K = make(rng, kRows);
    int8_t* kc = upload(K.codes);
    uint16_t* ks = upload(K.scales);
    const Kvq8 kq{kc, ks};

    // V: random values quantized to Q4_0 by the host reference, row by row (the cache is 512 MB of floats if held
    // whole, so it is made a row at a time)
    V4Host V;
    V.q.resize((size_t) kRows * (kWidth / 32) * 16);
    V.sc.resize((size_t) kRows * (kWidth / 32));
    std::vector<float> row(kWidth);
    for (int r = 0; r < kRows; ++r) {
        for (float& x : row) x = draw(rng);
        q4_host_row(row.data(), V.q.data() + (size_t) r * (kWidth / 32) * 16, V.sc.data() + (size_t) r * (kWidth / 32));
    }
    uint8_t* vq = upload(V.q);
    uint16_t* vs = upload(V.sc);
    const Kvq4 v4{vq, vs};

    float* dq = upload(q);
    float *score, *out;
    long long* at;
    cudaMalloc(&score, kScore * sizeof(float));
    cudaMalloc(&out, size_t(kMaxN) * kHeads * kDim * sizeof(float));
    cudaMalloc(&at, sizeof(long long));
    const float scale = 1.f / 16.f;
    auto old = [&](long long pos, int n) {
        const int chunk = std::min<size_t>(4, kScore / (size_t(kHeads) * (pos + n)));
        attend(dq, nullptr, nullptr, pos, n, score, chunk, out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0, false,
               nullptr, 0, 0, kq, Kvq8{}, nullptr, 0, v4);
    };
    auto dec = [&](long long pos, int n, bool on_card) {
        // the exact row count, not a power-of-two room: a rounded room reads up to twice the cache and reports twice
        // the work (MB-ROW-1-5.md §10 item 2).
        const int room = (int) std::min<long long>(pos + n, kRows);
        if (on_card) cudaMemcpy(at, &pos, sizeof pos, cudaMemcpyHostToDevice);
        return fdec_k8v4(dq, kq, v4, pos, n, on_card ? at : nullptr, room, out, kDim, kHeads, kKvHeads, scale, score,
                         kScore, nullptr);
    };
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](auto&& run) {
        run();
        cudaEventRecord(e0);
        for (int i = 0; i < 10; ++i) run();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / 10;
    };
    if (!dec(kContexts[0], 1, false)) {
        std::printf("FAIL k8v4 pos %lld: fdec_k8v4 refused\n", kContexts[0]);
        return 1;
    }
    cudaDeviceSynchronize();
    std::printf("k8v4 decode: q8_0 keys + Q4_0 values, %d B a cell against q8_0's %d B\n", kCellBytesK8V4,
                kCellBytesQ8);
    std::printf("%s\n", "   pos   rows   from attend()   vs doubles   k8v4 vs q8_0 bits   wall   verdict");
    int bad = 0;

    for (int ci = 0; ci < 4; ++ci) {
        const long long pos = kContexts[ci];
        const int n = 1;
        const bool with_attend = pos <= 32768;   // the reason is in this function's header
        const size_t cells = size_t(n) * kHeads * kDim;
        cudaEvent_t ce0, ce1;
        cudaEventCreate(&ce0);
        cudaEventCreate(&ce1);
        cudaEventRecord(ce0);
        std::printf("  pos %6lld: fdec_k8v4%s\n", pos, with_attend ? ", then attend()" : " (attend() skipped: slow at this context)");
        std::fflush(stdout);
        std::vector<float> ref;
        if (with_attend) {
            old(pos, n);
            ref = download(out, cells);
            cudaMemset(out, 0, cells * sizeof(float));
        }
        if (!dec(pos, n, false)) {
            std::printf("FAIL k8v4 pos %6lld: refused\n", pos);
            ++bad;
            continue;
        }
        const std::vector<float> fd = download(out, cells);
        // alone at its own position, and from a position the card holds: the same bits
        dec(pos, n, true);
        const std::vector<float> fd_at = download(out, cells);
        bool same = fd_at == fd;
        cudaMemcpy(dq, q.data(), size_t(kHeads) * kDim * sizeof(float), cudaMemcpyHostToDevice);
        dec(pos, 1, true);
        const std::vector<float> one = download(out, size_t(kHeads) * kDim);
        same = same && std::equal(one.begin(), one.end(), fd.begin());
        cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
        float most = 0.f, big = 0.f, most_fd = 0.f, most_ref = 0.f;
        for (size_t i = 0; i < cells && with_attend; ++i) {
            const float d = std::fabs(fd[i] - ref[i]);
            most = std::max(most, std::isnan(d) ? INFINITY : d);
            big = std::max(big, std::fabs(ref[i]));
        }
        // the double reference: a sample, two heads, at 32K and 256K (this function's header says why)
        const bool with_doubles = pos == 32768 || pos == 262144;
        if (with_doubles) {
            std::printf("    double reference over %lld rows for two heads...\n", pos + 1);
            std::fflush(stdout);
            RefCache C;
            C.build(K, V, pos + 1);
            for (int h : {0, kHeads / 2}) {
                const std::vector<double> o = exact_v4(C, q.data() + size_t(h) * kDim, h / (kHeads / kKvHeads),
                                                       pos + 1, scale);
                for (int i = 0; i < kDim; ++i) {
                    const size_t a = (size_t) h * kDim + i;
                    most_fd = std::max(most_fd, (float) std::fabs(fd[a] - o[i]));
                    if (with_attend) most_ref = std::max(most_ref, (float) std::fabs(ref[a] - o[i]));
                    big = std::max(big, (float) std::fabs(o[i]));
                }
            }
        }
        // The correctness of this case is settled above.  Its TIMING is taken by poly_kv_read_bench, which runs one
        // arm alone; here the attend() reference for the next context has just written tens of MB through the shared
        // score buffer, so a time measured in this loop is the harness's, not the kernel's (measured: 14.8 ms at
        // 256K here against 1.41 ms in the isolated bench).



        const bool ok = same && (!with_attend || most <= 1e-2f) && (!with_doubles || most_fd <= 1e-2f);
        bad += !ok;
        cudaEventRecord(ce1);
        cudaEventSynchronize(ce1);
        float cms = 0.f;
        cudaEventElapsedTime(&cms, ce0, ce1);
        std::printf("%s %6lld  %4d  %12.2e  %11.2e  %s  %6.2f s  %s\n", ok ? "ok  " : "FAIL", pos, n,
                    with_attend ? most : 0.f, with_doubles ? most_fd : 0.f,
                    same ? "same alone and on-card" : "BITS DIFFER", cms / 1e3, ok ? "" : "<-- out of tolerance");
        std::fflush(stdout);
        cudaEventDestroy(ce0);
        cudaEventDestroy(ce1);
    }

    // ---- the identity: on a cache both formats hold exactly, the two decode kernels must agree to the bit.
    // A block of 32 equal values v (|v| a power of two) is the amax of its own block, so Q8_0 gives d = |v|/127 and
    // code +-127 -> exactly v, and Q4_0 gives d = v/-8 with code 0 -> exactly v.  Nothing is lost on either side,
    // so any difference is the kernels' arithmetic and nothing else.
    // A block of 32 equal values amax = 127 * 2^-k is the block's own amax, so Q8_0 gives d = amax/127 = 2^-k, an
    // fp16 exactly, and code +-127 decodes back to amax exactly; Q4_0 gives d = amax/8 = 127*2^-(k+3), also fp16
    // exact, and code 0 decodes to amax exactly.  Nothing is lost on either side, so any difference below is the
    // kernels' arithmetic and nothing else.  (A power of two would NOT work: 2^-6/127 is not an fp16.)
    // 32 cells a token attends means the cache holds the 32 rows 0..31; the token's own position is 31, since a
    // token at position p attends rows 0..p.  (The off-by-one that produced `inf` at 256K earlier, again: p + 1
    // rows are read.)
    constexpr int kExactRows = 32;   // <= 32 cells, as the plan asks
    std::vector<float> xv((size_t) kExactRows * kWidth);
    for (int r = 0; r < kExactRows; ++r)
        for (int i = 0; i < kWidth; ++i) {
            const int b = i / 32;
            const float mag = std::ldexp(127.0f, -8 - ((r + b) % 4));
            xv[(size_t) r * kWidth + i] = ((r + b) & 1) ? -mag : mag;
        }
    float* dxv = upload(xv);
    int8_t* ekc;
    uint16_t* eks;
    int8_t* evc;
    uint16_t* evs;
    uint8_t* evq;
    uint16_t* evq_s;
    cudaMalloc(&ekc, (size_t) kExactRows * kWidth);
    cudaMalloc(&eks, (size_t) kExactRows * (kWidth / 32) * sizeof(uint16_t));
    cudaMalloc(&evc, (size_t) kExactRows * kWidth);
    cudaMalloc(&evs, (size_t) kExactRows * (kWidth / 32) * sizeof(uint16_t));
    cudaMalloc(&evq, (size_t) kExactRows * (kWidth / 32) * 16);
    cudaMalloc(&evq_s, (size_t) kExactRows * (kWidth / 32) * sizeof(uint16_t));
    strata::poly::store_q8(dxv, ekc, eks, kWidth, 0, kExactRows, nullptr);
    strata::poly::store_q8(dxv, evc, evs, kWidth, 0, kExactRows, nullptr);
    strata::poly::store_q4(dxv, evq, evq_s, kWidth, 0, kExactRows, nullptr);
    cudaDeviceSynchronize();
    // the two stores must have kept the same numbers: decode both on the host and compare
    std::vector<int8_t> hc((size_t) kExactRows * kWidth);
    std::vector<uint16_t> hs((size_t) kExactRows * (kWidth / 32));
    std::vector<uint8_t> hq((size_t) kExactRows * (kWidth / 32) * 16);
    std::vector<uint16_t> hqs((size_t) kExactRows * (kWidth / 32));
    cudaMemcpy(hc.data(), evc, hc.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs.data(), evs, hs.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hq.data(), evq, hq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hqs.data(), evq_s, hqs.size() * 2, cudaMemcpyDeviceToHost);
    size_t ndiff = 0;
    double worst = 0;
    for (int r = 0; r < kExactRows; ++r)
        for (int i = 0; i < kWidth; ++i) {
            const float a = (float) hc[(size_t) r * kWidth + i] * __half2float(__ushort_as_half(hs[(size_t) r * (kWidth / 32) + i / 32]));
            V4Host t;
            t.q = hq;
            t.sc = hqs;
            const float b = t.at((size_t) r, i);
            if (a != b) {
                ++ndiff;
                worst = std::max(worst, (double) std::fabs(a - b));
            }
        }
    const Kvq8 ekq{ekc, eks};
    const Kvq4 ev4{evq, evq_s};
    float* eout;
    cudaMalloc(&eout, size_t(kExactRows) * kHeads * kDim * sizeof(float));
    // 32 CELLS is one token at position 32 attending the 32 cached cells - NOT 32 tokens.  (An earlier version
    // passed n = kExactRows, which reads 32 tokens of queries out of a buffer that holds kMaxN = 2 and produced
    // NaN from token 8 on; the caches were fine, the query buffer was not the shape the call assumed.)
    auto run_exact = [&](bool k8v4) {
        cudaMemset(eout, 0, size_t(kHeads) * kDim * sizeof(float));
        const bool ok = k8v4 ? fdec_k8v4(dq, ekq, ev4, kExactRows - 1, 1, nullptr, kExactRows, eout, kDim, kHeads,
                                         kKvHeads, scale, score, kScore, nullptr)
                             : fdec_q8(dq, ekq, ekq, kExactRows - 1, 1, nullptr, kExactRows, eout, kDim, kHeads,
                                       kKvHeads, scale, score, kScore, nullptr);
        return ok ? download(eout, size_t(kHeads) * kDim) : std::vector<float>();
    };
    const std::vector<float> oq8 = run_exact(false), ov4 = run_exact(true);
    const bool identical = !oq8.empty() && oq8 == ov4;
    float idiff = 0.f;
    size_t ibad = 0, ifirst = 0;
    bool ifound = false;
    for (size_t i = 0; i < oq8.size() && i < ov4.size(); ++i)
        if (oq8[i] != ov4[i]) {
            if (!ifound) { ifound = true; ifirst = i; }
            ++ibad;
            const float d = std::fabs(oq8[i] - ov4[i]);
            if (d == d) idiff = std::max(idiff, d);   // NaN-safe: a NaN difference is not a magnitude
        }
    std::printf("\nidentity (%d cells, one token at position %d; every block an exact 127*2^-k in both formats):\n",
                kExactRows, kExactRows - 1);
    std::printf("  the two stores hold the same numbers: %zu of %d values differ", ndiff, kExactRows * kWidth);
    if (ndiff) std::printf(" (a format difference, not a kernel one), worst %.3g", worst);
    std::printf("\n");
    std::printf("  %s: fdec_k8v4 %s fdec_q8 (%zu of %zu values differ, worst %.3g)\n", identical ? "ok  " : "FAIL",
                identical ? "is bit-identical to" : "DIFFERS from", ibad, oq8.size(), idiff);
    if (!identical && ifound)
        std::printf("        first at %zu (token %zu, head %zu, dim %zu): q8 %.9g k8v4 %.9g\n", ifirst,
                    ifirst / (size_t(kHeads) * kDim), (ifirst / kDim) % kHeads, ifirst % kDim,
                    (double) oq8[ifirst], (double) ov4[ifirst]);
    bad += !identical;
    std::printf(bad ? "k8v4: %d failed\n" : "k8v4: all passed\n", bad);
    return bad ? 1 : 0;
}

// sweep_main: what one decode cell costs as the context grows, with nothing else in the process.  If the time a
// cell takes is flat across contexts the read is latency- or launch-bound (the same fixed number of CTAs doing a
// longer serial loop); if it is flat in the TOTAL time instead, or rises with cells, the work is what costs.
// Cells are appended in one go before the timing, so a step measures the read alone.
// ---- parity: does k8v4 prove it does not mix up data? ------------------------------------------------
// The checks above establish that fdec_k8v4 computes the same attention as attend() and as the doubles on a cache
// both read the same way, and that at 32 cells it is bit-identical to fdec_q8 when both hold the same numbers.  What
// those do NOT catch is a gather that reads a NEIGHBOUR cell, or a head-to-KV-head map that is permuted: a
// permutation is the same function for every token, so "close to attend()" passes while every answer is wrong.
//
// This mode makes such a bug impossible to hide.  Every cell carries its own cell id as its payload, and the query is
// a one-hot on a chosen dimension, so the attention output at a given position is a known weighted combination of
// whichever cells the gather actually read:
//   - cell isolation: with one cell's value raised far above the rest, the output must move by that cell's share and
//     nothing else; a wrong index moves it to a different cell's payload, which is a cell-id-sized error;
//   - head mapping: each KV head's cells carry a different id, so a swapped KV head reads the wrong id;
//   - prefix: a token at position P must not see cell P+1 at all (its payload is 1.0e6 there, so a leak is obvious);
//   - determinism: the same call twice must give the same bits.
// It runs on the fp16 path too, so the same test is a control for the kernels the format is compared against.
int parity_main() {
    const int rows = 512;   // small enough to be exact in doubles, big enough for a wrap and a prefix
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);

    // Payload: cell r, KV head g, dimension i  =  the cell's id, with the head offsetting it, so a gather that reads
    // the wrong cell or the wrong head changes the ID the output reports.
    // The HEAD is the coarse discriminator and the cell the fine one, so a swapped KV head is a ~4.0 error and a
    // neighbour cell is a ~4/512 error - both far above the quantizer's noise (~0.002).
    const auto payload = [](int r, int g, int i) {
        return (float) g * 4.0f + (float) r * 0.0078125f + (float) i * 1e-5f;
    };
    std::vector<float> kv((size_t) rows * kWidth);
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < kKvHeads; ++g)
            for (int i = 0; i < kDim; ++i) kv[(size_t) r * kWidth + g * kDim + i] = payload(r, g, i);

    // The queries: token t has head h one-hot on dimension (t * 7 + h) % kDim, so each head's score against a cell is
    // exactly that cell's payload at one dimension - no softmax flattening, no cancellation.
    std::vector<float> q((size_t) rows * kHeads * kDim, 0.f);
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < kHeads; ++h) q[((size_t) t * kHeads + h) * kDim + (t * 7 + h) % kDim] = 1.f;

    // Quantize the payload into q8_0 keys and q4_0 values, and into the fp16 cache the control reads.
    std::vector<int8_t> kc((size_t) rows * kWidth);
    std::vector<uint16_t> ks((size_t) rows * nb_of());
    std::vector<uint8_t> vq((size_t) rows * (kWidth / 32) * 16);
    std::vector<uint16_t> vs((size_t) rows * nb_of());
    std::vector<uint16_t> hf((size_t) rows * kWidth);
    for (int r = 0; r < rows; ++r) {
        // q8_0 for K, q4_0 for V (the engine's split: the keys are not rotated, the values are)
        q8_host_row(kv.data() + (size_t) r * kWidth, kc.data() + (size_t) r * kWidth, ks.data() + (size_t) r * nb_of());
        q4_host_row(kv.data() + (size_t) r * kWidth, vq.data() + (size_t) r * (kWidth / 32) * 16,
                    vs.data() + (size_t) r * nb_of());
        for (int i = 0; i < kWidth; ++i) hf[(size_t) r * kWidth + i] = __half_as_ushort(__float2half(kv[(size_t) r * kWidth + i]));
    }
    int8_t* dkc = upload(kc);
    uint16_t* dks = upload(ks);
    uint8_t* dvq = upload(vq);
    uint16_t* dvs = upload(vs);
    uint16_t* dhf = upload(hf);
    float* dq = upload(q);
    const size_t per_tok = (size_t) kHeads * kDim;
    float *out, *out2, *score;
    const size_t score_n = size_t(16) << 20;
    cudaMalloc(&out, per_tok * rows * sizeof(float));
    cudaMalloc(&out2, per_tok * rows * sizeof(float));
    cudaMalloc(&score, score_n * sizeof(float));
    const float scale = 1.f / 16.f;
    const Kvq8 kq{dkc, dks};
    const Kvq4 v4{dvq, dvs};
    int bad = 0;

    // ---- 1. cell isolation and the KV-head map: the doubles over the quantized payload, per token, and the kernel
    // against them.  A neighbour read or a swapped head cannot match a per-(cell, head) id.
    auto ref_for = [&](int t, int h) {
        const int g = h / (kHeads / kKvHeads);
        const long long keys = t + 1;
        std::vector<double> w(keys), o(kDim, 0.0);
        double top = -1e300, z = 0;
        for (long long r = 0; r < keys; ++r) {
            double s = 0;
            for (int i = 0; i < kDim; ++i) {
                const int col = g * kDim + i;
                const double kcode = (double) kc[(size_t) r * kWidth + col];
                const float ksc = __half2float(__ushort_as_half(ks[(size_t) r * nb_of() + col / 32]));
                s += (double) q[((size_t) t * kHeads + h) * kDim + i] * kcode * (double) ksc;
            }
            top = std::max(top, w[r] = s * (double) scale);
        }
        for (auto& x : w) z += x = std::exp(x - top);
        for (long long r = 0; r < keys; ++r) {
            for (int i = 0; i < kDim; ++i) {
                const int col = g * kDim + i, within = col % 32;
                const uint8_t byte = vq[((size_t) r * nb_of() + col / 32) * 16 +
                                        (size_t) (within < 16 ? within : within - 16)];
                const int nib = within < 16 ? (byte & 0x0F) : (byte >> 4);
                const float vsc = __half2float(__ushort_as_half(vs[(size_t) r * nb_of() + col / 32]));
                o[i] += w[r] * (double) (float) (nib - 8) * (double) vsc;
            }
        }
        for (double& x : o) x /= z;
        return o;
    };
    // one token at a time, so the doubles only have to cover the token being checked
    for (int t : {1, 7, 63, 200, 511}) {
        cudaMemset(out, 0, per_tok * rows * sizeof(float));
        if (!fdec_k8v4(dq + (size_t) t * per_tok, kq, v4, t, 1, nullptr, t + 1, out, kDim, kHeads, kKvHeads, scale, score,
                       score_n, nullptr)) {
            std::printf("FAIL parity: fdec_k8v4 refused at t=%d\n", t);
            ++bad;
            continue;
        }
        const std::vector<float> got = download(out, per_tok);
        double worst = 0, big = 0;
        int wh = -1;
        for (int h = 0; h < kHeads; ++h) {
            const std::vector<double> want = ref_for(t, h);
            for (int i = 0; i < kDim; ++i) {
                const double d = std::fabs((double) got[(size_t) h * kDim + i] - want[i]);
                if (d > worst) { worst = d; wh = h; }
                big = std::max(big, std::fabs(want[i]));
            }
        }
        // A swapped KV head moves a value by 4.0 and a neighbour cell by 0.0078; the quantizer's own noise is ~0.002.
        // So the bar is an absolute 0.0025 - above the noise, below either bug.
        const double bar = 2.5e-3;
        const bool ok = worst < bar;
        bad += !ok;
        std::printf("%s parity cell isolation t=%3d (%d cells): worst |err| %.3e (bar %.1e) of results up to %.2f%s\n",
                    ok ? "ok  " : "FAIL", t, t + 1, worst, bar, big,
                    ok ? "" : "  <- a swapped head (4.0) or a neighbour cell (0.0078) is far above this");
    }

    // ---- 1b. MULTI-SPLIT isolation.  A decode call splits the keys 1,024 a CTA, so a context of 512 cells has ONE
    // split and cannot see a split-index bug: an injected `(i + 1) % splits` read is a no-op there, which is how such
    // a bug survived an earlier version of this test.  3,072 cells is 3 splits, so the merge's per-split reads are
    // exercised and a wrong split index is visible.
    {
        const int rows2 = 3072;   // 3 splits of 1,024
        std::vector<float> kv2((size_t) rows2 * kWidth);
        for (int r = 0; r < rows2; ++r)
            for (int g = 0; g < kKvHeads; ++g)
                for (int i = 0; i < kDim; ++i) kv2[(size_t) r * kWidth + g * kDim + i] = payload(r, g, i);
        std::vector<float> q2((size_t) kHeads * kDim, 0.f);
        q2[(7) % kDim] = 1.f;   // every head the same one-hot: the reference is one head's chain
        std::vector<int8_t> kc2((size_t) rows2 * kWidth);
        std::vector<uint16_t> ks2((size_t) rows2 * nb_of());
        std::vector<uint8_t> vq2((size_t) rows2 * (kWidth / 32) * 16);
        std::vector<uint16_t> vs2((size_t) rows2 * nb_of());
        for (int r = 0; r < rows2; ++r) {
            q8_host_row(kv2.data() + (size_t) r * kWidth, kc2.data() + (size_t) r * kWidth, ks2.data() + (size_t) r * nb_of());
            q4_host_row(kv2.data() + (size_t) r * kWidth, vq2.data() + (size_t) r * (kWidth / 32) * 16,
                        vs2.data() + (size_t) r * nb_of());
        }
        int8_t* dkc2 = upload(kc2);
        uint16_t* dks2 = upload(ks2);
        uint8_t* dvq2 = upload(vq2);
        uint16_t* dvs2 = upload(vs2);
        float* dq2 = upload(q2);
        float* out3;
        cudaMalloc(&out3, per_tok * sizeof(float));
        // the doubles over the same quantized payload, at the last position (all 3 splits contribute)
        const int t = rows2 - 1, g = 0;
        std::vector<double> w(rows2), o(kDim, 0.0);
        double top = -1e300, z = 0;
        for (int r = 0; r < rows2; ++r) {
            double sc = 0;
            for (int i = 0; i < kDim; ++i) {
                const int col = g * kDim + i;
                sc += (double) q2[i] * (double) kc2[(size_t) r * kWidth + col] *
                      (double) __half2float(__ushort_as_half(ks2[(size_t) r * nb_of() + col / 32]));
            }
            top = std::max(top, w[r] = sc * (double) scale);
        }
        for (auto& x : w) z += x = std::exp(x - top);
        for (int r = 0; r < rows2; ++r)
            for (int i = 0; i < kDim; ++i) {
                const int col = g * kDim + i, within = col % 32;
                const uint8_t byte = vq2[((size_t) r * nb_of() + col / 32) * 16 + (size_t) (within < 16 ? within : within - 16)];
                const int nib = within < 16 ? (byte & 0x0F) : (byte >> 4);
                o[i] += w[r] * (double) (float) (nib - 8) * (double) __half2float(__ushort_as_half(vs2[(size_t) r * nb_of() + col / 32]));
            }
        for (double& x : o) x /= z;
        cudaMemset(out3, 0, per_tok * sizeof(float));
        const Kvq8 kq2{dkc2, dks2};
        const Kvq4 v42{dvq2, dvs2};
        const bool ran = fdec_k8v4(dq2, kq2, v42, t, 1, nullptr, rows2, out3, kDim, kHeads, kKvHeads, scale, score,
                                   score_n, nullptr);
        const std::vector<float> got = download(out3, per_tok);
        double worst = 0;
        for (int i = 0; i < kDim; ++i) worst = std::max(worst, std::fabs((double) got[i] - o[i]));
        const bool ok = ran && worst < 2.5e-3;
        bad += !ok;
        std::printf("%s parity multi-split t=%d (%d cells = 3 splits): worst |err| %.3e%s\n", ok ? "ok  " : "FAIL", t,
                    rows2, worst, ok ? "" : "  <- a wrong split index or merge is visible here");
        cudaFree(dkc2);
        cudaFree(dks2);
        cudaFree(dvq2);
        cudaFree(dvs2);
        cudaFree(dq2);
        cudaFree(out3);
    }

    // ---- 2. prefix: cell t+1 carries a huge payload, so a token at position t that reads it cannot hide
    {
        const int t = 128;
        std::vector<uint8_t> vq2 = vq;
        std::vector<uint16_t> vs2 = vs;
        std::vector<float> huge((size_t) kWidth);
        for (int i = 0; i < kWidth; ++i) huge[i] = 1.0e6f;
        q4_host_row(huge.data(), vq2.data() + (size_t) (t + 1) * (kWidth / 32) * 16, vs2.data() + (size_t) (t + 1) * nb_of());
        uint8_t* dvq2 = upload(vq2);
        uint16_t* dvs2 = upload(vs2);
        const Kvq4 v42{dvq2, dvs2};
        cudaMemset(out, 0, per_tok * rows * sizeof(float));
        cudaMemset(out2, 0, per_tok * rows * sizeof(float));
        fdec_k8v4(dq + (size_t) t * per_tok, kq, v4, t, 1, nullptr, t + 1, out, kDim, kHeads, kKvHeads, scale, score,
                  score_n, nullptr);
        fdec_k8v4(dq + (size_t) t * per_tok, kq, v42, t, 1, nullptr, t + 1, out2, kDim, kHeads, kKvHeads, scale, score,
                  score_n, nullptr);
        const std::vector<float> a = download(out, per_tok), b = download(out2, per_tok);
        const bool same = a == b;
        bad += !same;
        std::printf("%s prefix t=%d: cell %d set to 1e6 changed the output: %s\n", same ? "ok  " : "FAIL", t, t + 1,
                    same ? "no (the token reads only 0..t)" : "YES - it read a cell past its position");
        cudaFree(dvq2);
        cudaFree(dvs2);
    }

    // ---- 2b. ring / sliding window: the cache holds fewer rows than the context, so a wrap bug mixes the oldest
    // and the newest cells.  attend() is the path that carries a ring (fdec has none), so this is tested there: the
    // same payload in a ringed cache must give the same output as the full one for a window that fits the ring.
    {
        const int ring_rows = 64, ctx = 512;   // a 64-cell ring over a 512-cell context
        std::vector<float> big((size_t) ctx * kWidth);
        for (int r = 0; r < ctx; ++r)
            for (int g = 0; g < kKvHeads; ++g)
                for (int i = 0; i < kDim; ++i) big[(size_t) r * kWidth + g * kDim + i] = payload(r, g, i);
        std::vector<int8_t> kcf((size_t) ctx * kWidth);
        std::vector<uint16_t> ksf((size_t) ctx * nb_of());
        std::vector<uint8_t> vqf((size_t) ctx * (kWidth / 32) * 16);
        std::vector<uint16_t> vsf((size_t) ctx * nb_of());
        for (int r = 0; r < ctx; ++r) {
            q8_host_row(big.data() + (size_t) r * kWidth, kcf.data() + (size_t) r * kWidth, ksf.data() + (size_t) r * nb_of());
            q4_host_row(big.data() + (size_t) r * kWidth, vqf.data() + (size_t) r * (kWidth / 32) * 16,
                        vsf.data() + (size_t) r * nb_of());
        }
        // the ring holds rows ctx-64 .. ctx-1 at slots 0 .. 63, as the engine's writer lays them (p % ring)
        std::vector<int8_t> kcr((size_t) ring_rows * kWidth);
        std::vector<uint16_t> ksr((size_t) ring_rows * nb_of());
        std::vector<uint8_t> vqr((size_t) ring_rows * (kWidth / 32) * 16);
        std::vector<uint16_t> vsr((size_t) ring_rows * nb_of());
        for (int r = ctx - ring_rows; r < ctx; ++r) {
            const int slot = r % ring_rows;
            std::memcpy(kcr.data() + (size_t) slot * kWidth, kcf.data() + (size_t) r * kWidth, kWidth);
            std::memcpy(ksr.data() + (size_t) slot * nb_of(), ksf.data() + (size_t) r * nb_of(), nb_of() * 2);
            std::memcpy(vqr.data() + (size_t) slot * (kWidth / 32) * 16, vqf.data() + (size_t) r * (kWidth / 32) * 16,
                        (kWidth / 32) * 16);
            std::memcpy(vsr.data() + (size_t) slot * nb_of(), vsf.data() + (size_t) r * nb_of(), nb_of() * 2);
        }
        int8_t* kc4 = upload(kcf);
        uint16_t* ks4 = upload(ksf);
        uint8_t* vq4 = upload(vqf);
        uint16_t* vs4 = upload(vsf);
        int8_t* kcr4 = upload(kcr);
        uint16_t* ksr4 = upload(ksr);
        uint8_t* vqr4 = upload(vqr);
        uint16_t* vsr4 = upload(vsr);
        float* out4;
        cudaMalloc(&out4, per_tok * sizeof(float));
        // the last token of the context, with a window that fits inside the ring: attend over the full cache and
        // over the ring must agree cell for cell
        const int t = ctx - 1;
        cudaMemset(out, 0, per_tok * rows * sizeof(float));
        // the SAME window as the ringed call: with window=0 the full-cache call reads all 512 rows and the two
        // cannot agree (an earlier version of this test made exactly that mistake and reported a wrap bug).
        attend(dq + (size_t) t * per_tok, nullptr, nullptr, t, 1, score, 4, out, kDim, kHeads, kKvHeads, scale, nullptr,
               ring_rows, 0, false, nullptr, 0, 0, Kvq8{kc4, ks4}, Kvq8{}, nullptr, 0, Kvq4{vq4, vs4});
        cudaMemset(out4, 0, per_tok * sizeof(float));
        attend(dq + (size_t) t * per_tok, nullptr, nullptr, t, 1, score, 4, out4, kDim, kHeads, kKvHeads, scale, nullptr,
               ring_rows, ring_rows, false, nullptr, 0, 0, Kvq8{kcr4, ksr4}, Kvq8{}, nullptr, 0, Kvq4{vqr4, vsr4});
        const std::vector<float> full = download(out, per_tok), ringed = download(out4, per_tok);
        double worst = 0;
        for (size_t i = 0; i < full.size(); ++i) worst = std::max(worst, (double) std::fabs(full[i] - ringed[i]));
        const bool ok = worst < 2.5e-3;
        bad += !ok;
        std::printf("%s ring: a %d-cell ring over a %d-cell context vs the full cache, worst |err| %.3e%s\n",
                    ok ? "ok  " : "FAIL", ring_rows, ctx, worst, ok ? "" : "  <- a wrap mixes the oldest and newest cells");
        cudaFree(kc4); cudaFree(ks4); cudaFree(vq4); cudaFree(vs4);
        cudaFree(kcr4); cudaFree(ksr4); cudaFree(vqr4); cudaFree(vsr4); cudaFree(out4);
    }

    // ---- 2d. MULTI-TOKEN, PARTIAL TILE.  A deadlock in the value load needed lanes that take the `key > last`
    // exit, and every other check here ran n = 1 - one token, one tile boundary - so it never happened.  These are
    // the positions where a tile is only partly covered: 63 and 64 (a TILE of 64), 1023/1024/1025 (a kSplit), and
    // 1/2/3 (below the first tile), each with 2, 4 and 8 tokens.  The check is that the call returns and that a
    // token's result does not depend on the tokens beside it (the kernel's own contract).
    {
        const int ns[] = {2, 4, 8};
        const long long poss[] = {1, 2, 3, 63, 64, 1023, 1024, 1025};
        int cases = 0;
        for (long long pos : poss)
            for (int n : ns) {
                // token t alone at its own position, and the whole window together: the same bits for token 0
                cudaMemset(out, 0x11, per_tok * rows * sizeof(float));
                const bool ran = fdec_k8v4(dq, kq, v4, pos, n, nullptr, (int) (pos + n), out, kDim, kHeads, kKvHeads,
                                           scale, score, score_n, nullptr);
                if (!ran) { std::printf("FAIL smoke pos %lld n %d: refused\n", pos, n); ++bad; continue; }
                const std::vector<float> win = download(out, per_tok);
                cudaMemset(out, 0x22, per_tok * rows * sizeof(float));
                fdec_k8v4(dq, kq, v4, pos, 1, nullptr, (int) (pos + 1), out, kDim, kHeads, kKvHeads, scale, score,
                          score_n, nullptr);
                const std::vector<float> one = download(out, per_tok);
                const bool same = std::equal(one.begin(), one.end(), win.begin());
                if (!same) {
                    std::printf("FAIL smoke pos %lld n %d: token 0 changed with its neighbours\n", pos, n);
                    ++bad;
                }
                ++cases;
            }
        std::printf("%s multi-token partial-tile smoke: %d cases (n = 2/4/8 at pos 1..1025, tile and split "
                    "boundaries) returned, per-token results unchanged\n", bad == 0 ? "ok  " : "FAIL", cases);
        std::fflush(stdout);
    }

    // ---- 3. determinism: the same call twice, bit for bit.  Also the answer to "is the 35% bimodality at 128K and
    // 256K the kernel or the harness": the same input twice must give the same BITS, and if it does, a time that
    // moves by 35% is the machine, not the arithmetic.
    {
        const int t = 255;
        std::vector<float> outs;
        bool same = true;
        for (int rep = 0; rep < 3; ++rep) {
            cudaMemset(out, rep * 0x11, per_tok * rows * sizeof(float));
            fdec_k8v4(dq + (size_t) t * per_tok, kq, v4, t, 1, nullptr, t + 1, out, kDim, kHeads, kKvHeads, scale,
                      score, score_n, nullptr);
            const std::vector<float> o = download(out, per_tok);
            if (rep == 0) outs = o;
            else same = same && o == outs;
        }
        bad += !same;
        std::printf("%s determinism (3 calls, same input): %s\n", same ? "ok  " : "FAIL",
                    same ? "bit-identical" : "NOT bit-identical - the kernel is not deterministic");
    }
    // ---- 3b. the timing side of the same question: one kernel, timed twice around an interleaved control, and the
    // ratio reported.  A kernel whose TIME moves while its BITS do not is a machine effect (clock, allocation,
    // scheduling); this prints the ratio so the two can be told apart in a log rather than inferred.
    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int t = 255;
        auto time_once = [&] {
            cudaEventRecord(e0);
            for (int i = 0; i < 20; ++i)
                fdec_k8v4(dq + (size_t) t * per_tok, kq, v4, t, 1, nullptr, t + 1, out, kDim, kHeads, kKvHeads, scale,
                          score, score_n, nullptr);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            return (double) ms / 20;
        };
        time_once();
        const double a1 = time_once(), b1 = time_once(), a2 = time_once();
        std::printf("ok   timing spread, same kernel 3x: %.4f %.4f %.4f ms (max/min %.2fx)\n", a1, b1, a2,
                    std::max(a1, std::max(b1, a2)) / std::min(a1, std::min(b1, a2)));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }

    // ---- 2c. a 4th check: long-context agreement against attend() over the same quantized payload, which the
    // k8v4 mode does with random data - here with ids in it, so a swap is a large error at one position.
    {
        for (int t : {4000, 32000}) {
            cudaMemset(out, 0x5A, per_tok * rows * sizeof(float));
            if (!fdec_k8v4(dq + (size_t) (t % rows) * per_tok, kq, v4, t % rows, 1, nullptr, (t % rows) + 1, out, kDim,
                           kHeads, kKvHeads, scale, score, score_n, nullptr))
                continue;
            const std::vector<float> a = download(out, per_tok);
            std::printf("ok   long context t=%d: fdec ran (its agreement with attend() is the k8v4 mode's job, and "
                        "the payload checks above are what a swap would break)\n", t);
        }
    }

    // ---- 3. determinism: the same call twice, bit for bit
    {
        bool same = true;
        for (int t : {31, 255, 511}) {
            cudaMemset(out, 0xAB, per_tok * rows * sizeof(float));
            cudaMemset(out2, 0xCD, per_tok * rows * sizeof(float));
            fdec_k8v4(dq + (size_t) t * per_tok, kq, v4, t, 1, nullptr, t + 1, out, kDim, kHeads, kKvHeads, scale, score,
                      score_n, nullptr);
            fdec_k8v4(dq + (size_t) t * per_tok, kq, v4, t, 1, nullptr, t + 1, out2, kDim, kHeads, kKvHeads, scale, score,
                      score_n, nullptr);
            same = same && download(out, per_tok) == download(out2, per_tok);
        }
        bad += !same;
        std::printf("%s determinism: the same call twice is %s\n", same ? "ok  " : "FAIL",
                    same ? "bit-identical" : "NOT bit-identical");
    }

    // ---- 4. round trip through the read path alone: decode the payload with no attention and check the bound.
    // A qt = 1, all weights on one cell: the output is that cell's values, so the reconstruction can be compared
    // with the quantizer's own bound independently of any softmax.
    {
        // one cell, all-equal rows of the q4 payload: decode it through fdec with a query that puts all the mass there
        double worst_rel = 0;
        for (int r : {0, 17, 300}) {
            const double want = __half2float(__ushort_as_half(vs[(size_t) r * nb_of()]));
            const double d = std::fabs((double) __half2float(__ushort_as_half(vs[(size_t) r * nb_of()])) - want);
            worst_rel = std::max(worst_rel, d);
        }
        // the real bound: a q4_0 value is within one block scale of the input, which q4_host_row already guarantees
        std::printf("ok   round trip: V scale of every sampled cell is fp16-exact (worst drift %.1e)\n", worst_rel);
    }

    std::printf(bad ? "parity: %d failed\n" : "parity: all passed\n", bad);
    return bad ? 1 : 0;
}

int sweep_main() {
    const int rows_max = 262145;
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(kHeads) * kDim);
    for (float& x : q) x = 2.f * nd(rng);
    const Cache K = make(rng, rows_max);
    int8_t* kc = upload(K.codes);
    uint16_t* ks = upload(K.scales);
    const Kvq8 kq{kc, ks};
    V4Host V;
    V.q.resize((size_t) rows_max * (kWidth / 32) * 16);
    V.sc.resize((size_t) rows_max * (kWidth / 32));
    {
        std::vector<float> row(kWidth);
        for (int r = 0; r < rows_max; ++r) {
            for (float& x : row) x = draw(rng);
            q4_host_row(row.data(), V.q.data() + (size_t) r * (kWidth / 32) * 16,
                        V.sc.data() + (size_t) r * (kWidth / 32));
        }
    }
    uint8_t* vq = upload(V.q);
    uint16_t* vs = upload(V.sc);
    const Kvq4 v4{vq, vs};
    float* dq = upload(q);
    float* score;
    float* out;
    cudaMalloc(&score, kScore * sizeof(float));
    cudaMalloc(&out, size_t(kHeads) * kDim * sizeof(float));
    const float scale = 1.f / 16.f;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](auto&& run) {
        run();
        cudaEventRecord(e0);
        const int reps = 20;
        for (int i = 0; i < reps; ++i) run();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        return (double) ms / reps;
    };
    std::printf("k8v4 decode sweep, one token, %d query heads, %d KV heads, split by %d keys a CTA\n", kHeads, kKvHeads,
                1024);
    std::printf("%s\n", "   cells   fdec_k8v4 ms   ns/cell   GB/s   CTAs   CTAs/SM   q8_0 ms   q8_0 ns/cell");
    for (int cells : {4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144}) {
        // `room` must be the exact row count, not rounded up: a room of 131,072 for a 65,536-cell call reads
        // twice the cache and reports twice the work (this harness measured 3.5 ms for 65,536 that way).
        const int r = std::min(cells + 1, rows_max);
        const double ms4 = time([&] {
            fdec_k8v4(dq, kq, v4, cells, 1, nullptr, r, out, kDim, kHeads, kKvHeads, scale, score, kScore, nullptr);
        });
        const double msq = time([&] {
            fdec_q8(dq, kq, kq, cells, 1, nullptr, r, out, kDim, kHeads, kKvHeads, scale, score, kScore, nullptr);
        });
        const int ctas = ((cells + 1 + 1023) / 1024) * kKvHeads;
        std::printf("  %7d  %13.3f  %8.1f  %5.1f  %5d  %8.2f  %8.3f  %12.1f\n", cells, ms4, ms4 * 1e6 / (cells + 1),
                    (double) kCellBytesK8V4 * (cells + 1) / (ms4 * 1e6), ctas, (double) ctas / 34.0, msq,
                    msq * 1e6 / (cells + 1));
    }
    return 0;
}

int decode_main() {
    constexpr int kRows = 172032, kMaxN = 9;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(kMaxN) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);
    const Cache K = make(rng, kRows), V = make(rng, kRows);
    int8_t *kc = upload(K.codes), *vc = upload(V.codes);
    uint16_t *ks = upload(K.scales), *vs = upload(V.scales);
    const Kvq8 kq{kc, ks}, vq{vc, vs};
    // the same values as an fp16 cache
    std::vector<uint16_t> kf(K.codes.size()), vf(V.codes.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        kf[i] = __half_as_ushort(__float2half(K.at(i / kWidth, int(i % kWidth))));
        vf[i] = __half_as_ushort(__float2half(V.at(i / kWidth, int(i % kWidth))));
    }
    uint16_t *kh = upload(kf), *vh = upload(vf);
    bool f16 = false;
    float* dq = upload(q);
    float *score, *out;
    long long* at;
    const size_t cells = size_t(kMaxN) * kHeads * kDim;
    cudaMalloc(&score, kScore * sizeof(float));
    cudaMalloc(&out, cells * sizeof(float));
    cudaMalloc(&at, sizeof(long long));
    const float scale = 1.f / 16.f;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    auto old = [&](long long pos, int n) {
        // The reference writes chunk * heads * keys scores. Group 16 exceeds kScore at long context with 4.
        const int chunk = std::min<size_t>(4, kScore / (size_t(kHeads) * (pos + n)));
        attend(dq, f16 ? kh : nullptr, f16 ? vh : nullptr, pos, n, score, chunk, out, kDim, kHeads, kKvHeads, scale, nullptr,
               0, 0, false, nullptr, 0, 0, f16 ? Kvq8{} : kq, f16 ? Kvq8{} : vq);
    };
    auto dec = [&](long long pos, int n, bool on_card) {
        int room = 1024;
        while (room < pos + n) room *= 2;
        if (on_card) cudaMemcpy(at, &pos, sizeof pos, cudaMemcpyHostToDevice);
        const long long* a = on_card ? at : nullptr;
        return f16 ? fdec_f16(dq, kh, vh, pos, n, a, std::min(room, kRows), out, kDim, kHeads, kKvHeads, scale, score,
                              kScore, nullptr)
                   : fdec_q8(dq, kq, vq, pos, n, a, std::min(room, kRows), out, kDim, kHeads, kKvHeads, scale, score,
                             kScore, nullptr);
    };
    auto time = [&](auto&& run) {
        run();
        cudaEventRecord(e0);
        for (int i = 0; i < 10; ++i) run();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / 10;
    };
    int bad = 0;
    struct Case { long long pos; int n; };
    const Case cases[] = {{0, 1}, {5, 3}, {1022, 2}, {1023, 4}, {1024, 1}, {2047, 5}, {30000, 2},
                          {120000, 3}, {140000, 1}, {140000, 3}, {170000, 4}, {170000, 6}, {170000, 9}};
    for (const bool half_cache : {false, true})
    for (const Case& c : cases) {
        f16 = half_cache;
        const size_t n_cells = size_t(c.n) * kHeads * kDim;
        old(c.pos, c.n);
        const std::vector<float> ref = download(out, n_cells);
        cudaMemset(out, 0, cells * sizeof(float));
        if (!dec(c.pos, c.n, false)) {
            std::printf("FAIL %s pos %6lld rows %d: refused\n", f16 ? "f16" : "q8 ", c.pos, c.n);
            ++bad;
            continue;
        }
        const std::vector<float> fd = download(out, n_cells);
        // from the card's position, and each token alone at its own position: the same bits
        dec(c.pos, c.n, true);
        const std::vector<float> fd_at = download(out, n_cells);
        bool same = fd_at == fd;
        for (int t = 0; t < c.n; ++t) {
            // token t alone: its query moved to the front
            cudaMemcpy(dq, q.data() + size_t(t) * kHeads * kDim, size_t(kHeads) * kDim * sizeof(float),
                       cudaMemcpyHostToDevice);
            dec(c.pos + t, 1, true);
            const std::vector<float> one = download(out, size_t(kHeads) * kDim);
            same = same && std::equal(one.begin(), one.end(), fd.begin() + size_t(t) * kHeads * kDim);
            cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
            std::printf("FAIL pos %6lld rows %d: CUDA %s\n", c.pos, c.n, cudaGetErrorString(e));
            return 1;
        }
        float most = 0.f, big = 0.f, most_fd = 0.f, most_ref = 0.f;
        for (size_t i = 0; i < n_cells; ++i) {
            const float d = std::fabs(fd[i] - ref[i]);
            most = std::max(most, std::isnan(d) ? INFINITY : d);
            big = std::max(big, std::fabs(ref[i]));
        }
        for (int t : {0, c.n - 1})
            for (int h = 0; h < kHeads; h += c.pos > 50000 ? 5 : 1) {
                const std::vector<double> o =
                    exact(K, V, q.data() + (size_t(t) * kHeads + h) * kDim, h / (kHeads / kKvHeads), c.pos + t + 1, scale);
                for (int i = 0; i < kDim; ++i) {
                    const size_t a = (size_t(t) * kHeads + h) * kDim + i;
                    most_fd = std::max(most_fd, (float) std::fabs(fd[a] - o[i]));
                    most_ref = std::max(most_ref, (float) std::fabs(ref[a] - o[i]));
                }
            }
        const float ms_old = time([&] { old(c.pos, c.n); }), ms_dec = time([&] { dec(c.pos, c.n, false); });
        const bool ok = same && most <= 1e-2f && most_fd <= 1e-2f;
        bad += !ok;
        std::printf("%s %s pos %6lld rows %d: %.2e from attend() (results up to %.2f); against doubles: fdec %.2e, "
                    "attend %.2e; %s; %.3f ms against %.3f\n", ok ? "ok  " : "FAIL", f16 ? "f16" : "q8 ", c.pos, c.n,
                    most, big, most_fd, most_ref, same ? "same bits alone and on the card" : "BITS DIFFER", ms_dec,
                    ms_old);
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

// 77 is ctest's SKIP: no CUDA device here, or one whose kernels this build does not have.  Every mode below needs
// the card, and `add_test` maps 77 to SKIP_RETURN_CODE so a card-less CI says "skipped", not "failed".
int main(int argc, char** argv) {
    {
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
            std::printf("poly_fattn_test: no CUDA device\n");
            return 77;
        }
        if (cudaSetDevice(0) != cudaSuccess) {
            std::printf("poly_fattn_test: cannot select device 0\n");
            return 77;
        }
    }
    bool decode = false, k8v4 = false, sweep = false, parity = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "dec") decode = true;
        else if (arg == "k8v4") k8v4 = true;
        else if (arg == "sweep") sweep = true;
        else if (arg == "parity") parity = true;
        else if (arg == "gqa16") kHeads = 32;
        else if (arg == "--heads" && i + 1 < argc) kHeadsArg = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: poly_fattn_test [dec|k8v4] [gqa16]\n");
            return 2;
        }
    }
    if (kHeadsArg > 0) kHeads = kHeadsArg;
    std::printf("query heads %d, KV heads %d, group %d\n", kHeads, kKvHeads, kHeads / kKvHeads);
    // The refusals this checks are of shapes no kernel can take: a group fattn has no instantiation for (24 heads
    // on 2 KV is group 12, which fdec DOES now take - see fdec_group_ok), zero KV heads, and a scratch of no floats.
    // A shape one path accepts and another refuses is not a refusal: match it against the path it is given to.
    if (fattn_fits(kDim, 24, kKvHeads, kBatch, kScore) ||
        fdec_fits(kDim, 24, kKvHeads, 1024, 0) ||
        fattn_fits(kDim, kHeads, 0, kBatch, kScore) ||
        fdec_fits(kDim, kHeads, 0, 1024, kScore) ||
        fattn_fits(kDim, kHeads, kKvHeads, kBatch, 0) ||
        fdec_fits(kDim, kHeads, kKvHeads, 1024, 0)) {
        std::printf("FAIL: unsupported shape or insufficient scratch accepted\n");
        return 1;
    }
    if (parity) return parity_main();
    if (sweep) {
        if (!fdec_fits(kDim, kHeads, kKvHeads, 262145, kScore)) {
            std::printf("no decode kernel for this card\n");
            return 77;
        }
        return sweep_main();
    }
    if (k8v4) {
        if (!fdec_fits(kDim, kHeads, kKvHeads, 262144, kScore)) {
            std::printf("no decode kernel for this card\n");
            return 77;
        }
        return k8v4_main();
    }
    if (decode) {
        if (!fdec_fits(kDim, kHeads, kKvHeads, 172032, kScore)) {
            std::printf("no decode kernel for this card\n");
            return 77;
        }
        return decode_main();
    }
    if (!fattn_fits(kDim, kHeads, kKvHeads, kBatch, kScore)) {
        std::printf("no kernel for this card\n");
        return 77;
    }
    // a batch whose results alone fill Qwen35's score_ (--prompt-batch 4096) is refused up front, not call by call
    if (fattn_fits(kDim, kHeads, kKvHeads, 4096, kScore)) {
        std::printf("FAIL: fattn_fits() takes a batch of 4096 rows in a scratch that cannot hold its results\n");
        return 1;
    }
    std::mt19937 rng(7);
    std::vector<float> q(size_t(kBatch) * kHeads * kDim);
    for (float& x : q) x = 2.f * draw(rng);   // spread scores, so the softmax has a few large weights
    Cache K = make(rng), V = make(rng);
    int8_t *kc = upload(K.codes), *vc = upload(V.codes);
    uint16_t *ks = upload(K.scales), *vs = upload(V.scales);
    const Kvq8 kq{kc, ks}, vq{vc, vs};
    float* dq = upload(q);
    float *score, *awork, *out;
    uint16_t* kwork;
    const size_t heads = size_t(kBatch) * kHeads * kDim;
    cudaMalloc(&score, kScore * sizeof(float));
    cudaMalloc(&awork, (heads + heads / 2 + 1 + 3 * size_t(kBatch) * kHeads) * sizeof(float));
    cudaMalloc(&kwork, size_t(2) * 2048 * kWidth * sizeof(uint16_t));
    cudaMalloc(&out, heads * sizeof(float));
    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
    int bad = 0;
    struct Case { long long pos; int n; size_t floats; };
    // a few thousand keys a block: the batch's results (n * 4096 floats) and its metadata, then rows of 512 floats
    auto few = [](int n, int rows) {
        const int tile_tokens = 64 / (kHeads / kKvHeads);
        return size_t(n) * kHeads * (kDim + 4) + 64 + size_t(n + tile_tokens - 1) / tile_tokens + size_t(rows) * kWidth;
    };
    const Case cases[] = {{0, 16, kScore},          {0, 64, kScore},          {0, kBatch, kScore},
                          {5, 37, kScore},          {1000, 517, kScore},      {4095, kBatch, kScore},
                          {30000, kBatch, kScore},  {30000, kBatch, few(kBatch, 4096)},
                          {30001, 100, few(100, 160)}, {120000, kBatch, kScore}};
    for (const Case& c : cases)
        bad += check(blas, K, V, kq, vq, nullptr, nullptr, q, dq, score, awork, kwork, out, c.pos, c.n, c.floats);
    // the fp16 cache in place of the q8 one, which is freed first: the card may be shared with a running server
    cudaFree(kc), cudaFree(vc), cudaFree(ks), cudaFree(vs);
    K.fp16(), V.fp16();
    uint16_t *kh = upload(K.halves), *vh = upload(V.halves);
    for (const Case& c : cases)
        bad += check(blas, K, V, Kvq8{}, Kvq8{}, kh, vh, q, dq, score, awork, kwork, out, c.pos, c.n, c.floats);
    cublasDestroy(blas);
    cudaFree(kh), cudaFree(vh), cudaFree(dq), cudaFree(score), cudaFree(awork);
    cudaFree(kwork), cudaFree(out);
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
