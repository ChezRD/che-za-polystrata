// src/poly/kv_q4_test.cpp - the Q4_0 KV cache kernels of the poly engine (store_q4 / store_q4_at) and the
// Hadamard rotation (fwht_rows), against a host reference.  No model file and no card beyond CUDA device 0:
//   1. byte layout: for n in {32, 256} the device bytes (fp16 scale, then 16 nibbles a 32-value block) must match
//      the host's own ggml q4_0 arithmetic byte for byte;
//   2. round trip: random floats of mixed magnitude are stored and read back; every dequantized value is within
//      one block scale of the input (the quantizer clips the side away from the largest value at 7/8 of it);
//   3. fwht_rows: H(Hx) == x within a few ulps, and <Hq, Hk> == <q, k> - the rotation is orthonormal;
//   4. store_q4_at with a ring: two rows written from a device `at` land at (*at + r) % ring and equal what plain
//      store_q4 writes from the same position.
// Exit 0 prints "poly_kv_q4_test: PASS"; any failed assertion prints the offending index and values and exits 1.
//
// --bench: the k8v4 store timed as the engine issues it (qwen35.cpp: fwht_rows over a (token, KV head) row of the
// values, then store_q8 for the keys and store_q4 for the values), and the q8_0 store of the same cells as the
// comparison arm, interleaved in this one process so both see the same clock and the same allocation history.
// 832 B a cell for k8v4 (K 512 codes + 16 fp16 scales = 544, V 256 nibble bytes + 16 fp16 scales = 288) against
// 1,088 B for q8_0 (544 + 544).  The time is ns a cell over a write of `cells` rows, the rows cycled so the whole
// cache is touched and a 4K run cannot stay in L2 by luck.  This measures the write path, not the cache's size.
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "poly_kv_q4_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n ? n : 1), "cudaMalloc"); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() { return static_cast<T*>(p); }
};

uint16_t half_bits(__half h) {
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

float half_value(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, sizeof(h));
    return __half2float(h);
}

// A magnitude drawn over five orders (1e-4 .. 1e3) and either sign, so small and large values mix within a block.
float rand_val(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    const float mag = std::pow(10.f, -4.f + 7.f * u(rng));
    return (u(rng) < 0.5f ? -1.f : 1.f) * mag;
}

// ggml's q4_0, the kernel's rule: d = (the value of largest |x|) / -8, q = clamp(trunc(x / d + 8.5), 0, 15);
// ties in |x| resolve to the larger value, as the warp's reduction does.  The poly cache keeps the scale in an
// array of its own and the 16 nibble bytes in another.
void q4_host(const float* x, uint8_t* nib, uint16_t* scale) {
    float amax = 0.f, mval = 0.f;
    for (int j = 0; j < 32; ++j) {
        const float a = std::fabs(x[j]);
        if (a > amax || (a == amax && x[j] > mval)) {
            amax = a;
            mval = x[j];
        }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *scale = half_bits(__float2half_rn(d));
    for (int j = 0; j < 16; ++j) {
        const int q0 = (int) (x[j] * id + 8.5f);
        const int q1 = (int) (x[j + 16] * id + 8.5f);
        const uint8_t a = (uint8_t) (q0 < 0 ? 0 : (q0 > 15 ? 15 : q0));
        const uint8_t b = (uint8_t) (q1 < 0 ? 0 : (q1 > 15 ? 15 : q1));
        nib[j] = (uint8_t) (a | (b << 4));
    }
}

void layout_case(int rows, int n, std::mt19937& rng, int& fails) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);
    Dev dx(x.size() * sizeof(float)), dq4((size_t) rows * nb * 16), dsc((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q4(dx.as<float>(), dq4.as<uint8_t>(), dsc.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaGetLastError(), "store_q4 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q4 sync");
    std::vector<uint8_t> got((size_t) rows * nb * 16);
    std::vector<uint16_t> gsc((size_t) rows * nb);
    ck(cudaMemcpy(got.data(), dq4.p, got.size(), cudaMemcpyDeviceToHost), "download q4");
    ck(cudaMemcpy(gsc.data(), dsc.p, gsc.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t want[16];
            uint16_t wsc = 0;
            q4_host(&x[(size_t) r * n + b * 32], want, &wsc);
            const uint8_t* g = &got[((size_t) r * nb + b) * 16];
            if (std::memcmp(g, want, 16) != 0 || gsc[(size_t) r * nb + b] != wsc) {
                ++fails;
                std::fprintf(stderr, "layout rows=%d n=%d row=%d blk=%d differs (scale %u/%u)\n", rows, n, r, b,
                             gsc[(size_t) r * nb + b], wsc);
                for (int j = 0; j < 16; ++j)
                    if (g[j] != want[j]) std::fprintf(stderr, "  byte %d: got %u want %u\n", j, g[j], want[j]);
                return;
            }
        }
    std::printf("layout rows=%d n=%d: %zu blocks byte-exact\n", rows, n, (size_t) rows * nb);
}

void round_trip(int rows, int n, std::mt19937& rng, int& fails, long& values) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);
    Dev dx(x.size() * sizeof(float)), dq4((size_t) rows * nb * 16), dsc((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q4(dx.as<float>(), dq4.as<uint8_t>(), dsc.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaStreamSynchronize(nullptr), "store_q4 sync");
    std::vector<uint8_t> q4((size_t) rows * nb * 16);
    std::vector<uint16_t> sc((size_t) rows * nb);
    ck(cudaMemcpy(q4.data(), dq4.p, q4.size(), cudaMemcpyDeviceToHost), "download q4");
    ck(cudaMemcpy(sc.data(), dsc.p, sc.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");
    double worst = 0.0;
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = &q4[((size_t) r * nb + b) * 16];
            const float d = half_value(sc[(size_t) r * nb + b]);
            for (int j = 0; j < 32; ++j) {
                const uint8_t byte = blk[j & 15];
                const int q = j < 16 ? (byte & 0x0F) : (byte >> 4);
                const float back = (float) (q - 8) * d;
                const float err = std::fabs(back - x[(size_t) r * n + b * 32 + j]);
                worst = std::max(worst, (double) (err / (std::fabs(d) + 1e-30f)));
                if (err > std::fabs(d) * 1.001f + 1e-9f) {
                    ++fails;
                    std::fprintf(stderr, "round trip rows=%d n=%d row=%d blk=%d j=%d: x=%g back=%g d=%g\n", rows, n, r,
                                 b, j, x[(size_t) r * n + b * 32 + j], back, d);
                    return;
                }
                ++values;
            }
        }
    std::printf("round trip rows=%d n=%d: worst error %.3f of the block scale\n", rows, n, worst);
}

void fwht_case(int rows, int dim, std::mt19937& rng, int& fails) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> x((size_t) rows * dim), back((size_t) rows * dim);
    for (float& v : x) v = u(rng);
    Dev dx(x.size() * sizeof(float));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload fwht");
    strata::poly::fwht_rows(dx.as<float>(), dx.as<float>(), rows, dim, nullptr);
    ck(cudaGetLastError(), "fwht launch");
    ck(cudaStreamSynchronize(nullptr), "fwht sync");
    strata::poly::fwht_rows(dx.as<float>(), dx.as<float>(), rows, dim, nullptr);
    ck(cudaStreamSynchronize(nullptr), "fwht sync 2");
    ck(cudaMemcpy(back.data(), dx.p, back.size() * sizeof(float), cudaMemcpyDeviceToHost), "download fwht");
    double worst = 0.0;
    for (size_t i = 0; i < x.size(); ++i) worst = std::max(worst, (double) std::fabs(back[i] - x[i]));
    if (worst > 2e-5) {
        ++fails;
        std::fprintf(stderr, "fwht rows=%d dim=%d: H(Hx) differs by %g\n", rows, dim, worst);
    } else {
        std::printf("fwht rows=%d dim=%d: H(Hx) error %.3g\n", rows, dim, worst);
    }
    // <Hq, Hk> == <q, k>: one row each, rotated by the same launch.
    std::vector<float> pair((size_t) 2 * dim);
    for (float& v : pair) v = u(rng);
    double dot = 0.0;
    for (int i = 0; i < dim; ++i) dot += (double) pair[i] * pair[dim + i];
    Dev dp(pair.size() * sizeof(float));
    ck(cudaMemcpy(dp.as<float>(), pair.data(), pair.size() * sizeof(float), cudaMemcpyHostToDevice), "upload pair");
    strata::poly::fwht_rows(dp.as<float>(), dp.as<float>(), 2, dim, nullptr);
    ck(cudaStreamSynchronize(nullptr), "fwht pair sync");
    ck(cudaMemcpy(pair.data(), dp.p, pair.size() * sizeof(float), cudaMemcpyDeviceToHost), "download pair");
    double rdot = 0.0;
    for (int i = 0; i < dim; ++i) rdot += (double) pair[i] * pair[dim + i];
    if (std::fabs(rdot - dot) > 1e-4 * (1.0 + std::fabs(dot))) {
        ++fails;
        std::fprintf(stderr, "fwht dim=%d: dot %g -> %g\n", dim, dot, rdot);
    } else {
        std::printf("fwht dim=%d: dot %.6f -> %.6f\n", dim, dot, rdot);
    }
}

void at_case(std::mt19937& rng, int& fails) {
    const int n = 256, rows = 2, ring = 8, nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);
    Dev dx(x.size() * sizeof(float)), dplain((size_t) rows * nb * 16), dring((size_t) ring * nb * 16);
    Dev splain((size_t) rows * nb * sizeof(uint16_t)), sring((size_t) ring * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload at x");
    ck(cudaMemset(dring.p, 0, (size_t) ring * nb * 16), "zero ring");
    ck(cudaMemset(sring.p, 0, (size_t) ring * nb * sizeof(uint16_t)), "zero ring scales");
    strata::poly::store_q4(dx.as<float>(), dplain.as<uint8_t>(), splain.as<uint16_t>(), n, 0, rows, nullptr);
    const long long at = 5;
    Dev dat(sizeof(long long));
    ck(cudaMemcpy(dat.p, &at, sizeof(at), cudaMemcpyHostToDevice), "upload at");
    strata::poly::store_q4_at(dx.as<float>(), dring.as<uint8_t>(), sring.as<uint16_t>(), n, dat.as<long long>(),
                              rows, ring, nullptr);
    ck(cudaGetLastError(), "store_q4_at launch");
    ck(cudaStreamSynchronize(nullptr), "store_q4_at sync");
    std::vector<uint8_t> plain((size_t) rows * nb * 16), got((size_t) ring * nb * 16);
    std::vector<uint16_t> sp((size_t) rows * nb), sg((size_t) ring * nb);
    ck(cudaMemcpy(plain.data(), dplain.p, plain.size(), cudaMemcpyDeviceToHost), "download plain");
    ck(cudaMemcpy(got.data(), dring.p, got.size(), cudaMemcpyDeviceToHost), "download ring");
    ck(cudaMemcpy(sp.data(), splain.p, sp.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");
    ck(cudaMemcpy(sg.data(), sring.p, sg.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download ring scales");
    for (int r = 0; r < rows; ++r) {
        const uint8_t* want = &plain[(size_t) r * nb * 16];
        const uint8_t* have = &got[(size_t) (((at + r) % ring) * nb * 16)];
        const uint16_t* ws = &sp[(size_t) r * nb];
        const uint16_t* hs = &sg[(size_t) (((at + r) % ring) * nb)];
        if (std::memcmp(want, have, (size_t) nb * 16) != 0 ||
            std::memcmp(ws, hs, (size_t) nb * sizeof(uint16_t)) != 0) {
            ++fails;
            std::fprintf(stderr, "store_q4_at: row %d not at (%lld + %d) %% %d\n", r, at, r, ring);
            return;
        }
    }
    std::printf("store_q4_at: rows at (*at + r) %% ring\n");
}

// --bench: the store side of Row 1 of the optimization stack's Phase 0 (OPTIMIZATION-STACK.md §0.3 Row 1).
// The shapes are the engine's: head_dim 256, 2 KV heads, so a cell is 512 values a side.  k8v4's K is q8_0's
// (544 B a cell) and its V is Q4_0's (288 B), 832 B in all against q8_0's 1,088 B.
//
// Two shapes, because they answer two different questions:
//   bulk    the whole cache written in chunks, as a prompt batch writes its own rows.  This is the write
//           bandwidth the row's pass criterion is about.
//   decode  one row a call, back to back, as a decode step appends a token.  This is the per-token cost the
//           engine pays, launch overhead included.
// Both arms run interleaved in this one process (median of several reps), so they see the same clock and the same
// allocation history - the bimodality lesson from MB-ROW-6-7.md §1.
//
// The values come from one source buffer of `cells` rows, so both arms read the same input and the only difference
// between them is the bytes they write.  k8v4 additionally rotates each chunk (fwht_rows) before Q4_0, exactly as
// qwen35.cpp does; the store without the rotation is timed too, so the rotation can be charged separately.
namespace bench {

constexpr int kHeadDim = 256, kKvHeads = 2;
constexpr int kWidth = kKvHeads * kHeadDim;   // 512 values a side of a cell
// K: 512 int8 codes + 16 fp16 scales = 544 B; V k8v4: 256 nibble bytes + 16 fp16 scales = 288 B
constexpr int kCellBytesK8V4 = 544 + 288;
constexpr int kCellBytesQ8 = 544 + 544;
// The store is a read-modify-write: to write a cell's quantized bytes it must READ the cell's floats.  Four bytes
// a value is what the quantizer needs, so a cell's traffic is far more than the bytes it leaves in the cache -
// which is what makes "GB/s of KV bytes written" a denominator with a ceiling well below the card's peak.
//   read  k8v4: kWidth floats for store_q8's keys + kWidth for store_q4's values = 2 * 4 * kWidth = 4,096 B
//   read   q8_0: the same two sides, 4,096 B
//   write k8v4: 832 B;  q8_0: 1,088 B
// and, for k8v4 only, the rotation's own pass: it reads the values and writes the rotated copy, which store_q4 then
// reads.  With the rotated copy in L2 (a chunk is 8 MiB against a 32 MiB L2) that pass adds no DRAM traffic.
constexpr int kReadK8V4 = 2 * 4 * kWidth;
constexpr int kReadQ8 = 2 * 4 * kWidth;
constexpr int kRotateK8V4 = 2 * 4 * kWidth;   // one read of the values + one write of the rotated copy
// The same helpers the engine's cell is: k8v4 writes 832 B of a cell and must move at least 4,096 B to do it.
constexpr double kKvShareK8V4 = (double) kCellBytesK8V4 / (kCellBytesK8V4 + kReadK8V4);
// A chunk of rows: 4,096 rows of 512 values is 8 MiB, which is one prompt chunk's worth of v_ in the engine.
// THE SOURCE IS THIS SMALL ON PURPOSE.  store_q8/store_q4 read 4 B a value to quantize it, so a source read from
// DRAM would cost 4,096 B of traffic for a 832 B cell and cap "KV bytes written" at 17% of the peak no matter how
// good the kernel is.  In the engine the source is the projection's fresh output: one token's k or v is 2 KB at
// decode and a chunk's is 8 MB at prefill, both hot in this card's 32 MB L2.  So the source here is a chunk that
// stays resident and the cache is what reaches DRAM - which is the traffic the row's criterion is about.
constexpr int kChunk = 4096;

struct Arm {
    const char* name;
    int cell_bytes;
    double ns_cell;
};

// k8v4 as the engine issues it: rotate the chunk's (token, KV head) rows, then store_q8 the keys and store_q4 the
// values.  dx is the source chunk, vt the rotated copy.
void store_k8v4(const float* dx, float* vt, int8_t* kc, uint16_t* ks, uint8_t* vq, uint16_t* vs, int64_t first,
                int rows) {
    strata::poly::fwht_rows(dx, vt, rows * kKvHeads, kHeadDim, nullptr);
    strata::poly::store_q8(dx, kc, ks, kWidth, first, rows, nullptr);
    strata::poly::store_q4(vt, vq, vs, kWidth, first, rows, nullptr);
}

// The same stores with no rotation, to charge the ALU separately.
void store_k8v4_norot(const float* dx, int8_t* kc, uint16_t* ks, uint8_t* vq, uint16_t* vs, int64_t first,
                      int rows) {
    strata::poly::store_q8(dx, kc, ks, kWidth, first, rows, nullptr);
    strata::poly::store_q4(dx, vq, vs, kWidth, first, rows, nullptr);
}

void store_q8_pair(const float* dx, int8_t* kc, uint16_t* ks, int8_t* vc, uint16_t* vs, int64_t first, int rows) {
    strata::poly::store_q8(dx, kc, ks, kWidth, first, rows, nullptr);
    strata::poly::store_q8(dx, vc, vs, kWidth, first, rows, nullptr);
}

double median_ns(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace bench

void bench_main() {
    using namespace bench;
    cudaSetDevice(0);
    const int contexts[] = {4096, 32768, 131072, 262144};
    // the card's own peak, for the fraction this row is judged on: 2 * clk * (bus / 8), DDR
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);   // kHz, the card's max memory clock
    const double peak = 2.0 * (double) clk * 1e3 * (bus / 8) / 1e9;
    std::printf("k8v4 store: head_dim %d, %d KV heads, %d values a side; %d B a cell against q8_0's %d B\n", kHeadDim,
                kKvHeads, kWidth, kCellBytesK8V4, kCellBytesQ8);
    std::printf("peak %.1f GB/s (2 x %.0f MHz x %d bit / 8, clocks.max.memory)\n", peak, clk / 1e3, bus);
    std::printf("the source is a %d-row chunk held in L2, as the engine's freshly projected k/v is; the cache\n"
                "write is the DRAM traffic this row is about (%d B a cell for k8v4, %d B for q8_0)\n",
                kChunk, kCellBytesK8V4, kCellBytesQ8);
    const int maxc = contexts[3];
    // the source holds one chunk, cycled for every row of the cache: L2-resident, as the engine's is
    std::mt19937 rng(99);
    float* dx;
    float* vt;
    int8_t* kc;
    uint16_t* ks;
    int8_t* vc;
    uint8_t* vq;
    uint16_t* vs;
    ck(cudaMalloc(&dx, (size_t) kChunk * kWidth * sizeof(float)), "bench x");
    ck(cudaMalloc(&vt, (size_t) kChunk * kWidth * sizeof(float)), "bench vt");
    ck(cudaMalloc(&kc, (size_t) maxc * kWidth), "bench k codes");
    ck(cudaMalloc(&ks, (size_t) maxc * (kWidth / 32) * sizeof(uint16_t)), "bench k scales");
    ck(cudaMalloc(&vc, (size_t) maxc * kWidth), "bench v codes");
    ck(cudaMalloc(&vq, (size_t) maxc * (kWidth / 32) * 16), "bench v q4");
    ck(cudaMalloc(&vs, (size_t) maxc * (kWidth / 32) * sizeof(uint16_t)), "bench v scales");
    {
        std::vector<float> x((size_t) kChunk * kWidth);
        for (float& v : x) v = rand_val(rng);
        ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "bench x up");
    }
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");

    // ---- bulk: the cache written a chunk at a time.  The pass criterion is judged here.
    std::printf("\nbulk: the whole cache written in %d-row chunks (the arm the criterion names is k8v4)\n", kChunk);
    std::printf("  cells   arm          ns/cell   GB/s   %% of %.0f (max clk)   %% of 280.0 (loaded)\n", peak);
    for (int ci = 0; ci < 4; ++ci) {
        const int c = contexts[ci];
        std::vector<double> k8, q8, nr;
        const int reps = c > 100000 ? 5 : 9;
        for (int r = 0; r < reps; ++r) {
            cudaEventRecord(e0);
            for (int row = 0; row < c; row += kChunk)
                store_k8v4(dx, vt, kc, ks, vq, vs, row, std::min(kChunk, c - row));
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "bulk k8v4");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) k8.push_back((double) ms * 1e6 / c);   // the first touch of the pages is not the write
            cudaEventRecord(e0);
            for (int row = 0; row < c; row += kChunk)
                store_q8_pair(dx, kc, ks, vc, vs, row, std::min(kChunk, c - row));
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "bulk q8");
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) q8.push_back((double) ms * 1e6 / c);
            cudaEventRecord(e0);
            for (int row = 0; row < c; row += kChunk)
                store_k8v4_norot(dx, kc, ks, vq, vs, row, std::min(kChunk, c - row));
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "bulk k8v4 nrot");
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) nr.push_back((double) ms * 1e6 / c);
        }
        const double k8ns = median_ns(k8), q8ns = median_ns(q8), nrns = median_ns(nr);
        const double k8g = kCellBytesK8V4 / k8ns, q8g = kCellBytesQ8 / q8ns, nrg = kCellBytesK8V4 / nrns;
        std::printf("  %6d  k8v4         %9.2f  %6.1f  %14.1f%%  %16.1f%%\n", c, k8ns, k8g, 100.0 * k8g / peak,
                    100.0 * k8g / 280.0);
        std::printf("  %6d  k8v4 no-rot  %9.2f  %6.1f  %14.1f%%  %16.1f%%\n", c, nrns, nrg, 100.0 * nrg / peak,
                    100.0 * nrg / 280.0);
        std::printf("  %6d  q8_0         %9.2f  %6.1f  %14.1f%%  %16.1f%%\n", c, q8ns, q8g, 100.0 * q8g / peak,
                    100.0 * q8g / 280.0);
        std::printf("         k8v4 writes %.1f%% of q8_0's bytes and takes %.1f%% of its time (%.0f%% of the byte "
                    "saving reaches memory as time); the rotation is %.1f%% of k8v4's time\n",
                    100.0 * kCellBytesK8V4 / kCellBytesQ8, 100.0 * k8ns / q8ns,
                    100.0 * (1.0 - k8ns / q8ns) / (1.0 - (double) kCellBytesK8V4 / kCellBytesQ8),
                    100.0 * (k8ns - nrns) / k8ns);
        // the same time read against the bytes the operation must actually move
        const double k8traffic = kCellBytesK8V4 + kReadK8V4;
        const double q8traffic = kCellBytesQ8 + kReadQ8;
        std::printf("         as traffic (read %.0f B + write a cell): k8v4 %.1f GB/s, q8_0 %.1f GB/s "
                    "--> %.0f%% and %.0f%% of the peak\n",
                    (double) kReadK8V4, k8traffic / k8ns, q8traffic / q8ns, 100.0 * k8traffic / k8ns / peak,
                    100.0 * q8traffic / q8ns / peak);
    }

    // ---- the control: the same bytes written by cudaMemset.  If this is also far from the peak, the measurement
    // or the card is the limit and the kernels are not; if it is near the peak, the store kernels are.
    std::printf("\ncontrol: the same 832 B a cell written by cudaMemset (no read, no quantize)\n");
    std::printf("  cells   ns/cell   GB/s   %% of %.0f (max clk)   %% of 280.0 (loaded)\n", peak);
    for (int c : contexts) {
        std::vector<double> v;
        for (int r = 0; r < 7; ++r) {
            cudaEventRecord(e0);
            ck(cudaMemset(kc, 0, (size_t) c * kWidth), "memset k");
            ck(cudaMemset(ks, 0, (size_t) c * (kWidth / 32) * sizeof(uint16_t)), "memset ks");
            ck(cudaMemset(vq, 0, (size_t) c * (kWidth / 32) * 16), "memset vq");
            ck(cudaMemset(vs, 0, (size_t) c * (kWidth / 32) * sizeof(uint16_t)), "memset vs");
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "memset");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) v.push_back((double) ms * 1e6 / c);
        }
        const double ns = median_ns(v);
        const double gbs = (double) kCellBytesK8V4 / ns;
        std::printf("  %6d  %9.2f  %6.1f  %14.1f%%  %16.1f%%\n", c, ns, gbs, 100.0 * gbs / peak,
                    100.0 * gbs / 280.0);
    }

    // ---- decode: one row a call.  This is launch cost, so it is reported and not judged.
    const int steps = 65536;
    std::printf("\ndecode: one row a call, %d calls a rep, the cache's own rows cycled\n", steps);
    std::printf("  arm          ns/cell   GB/s   %% of %.0f (max clk)   %% of 280.0 (loaded)\n", peak);
    for (int arm = 0; arm < 2; ++arm) {
        const bool k8 = arm == 0;
        std::vector<double> v;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0);
            for (int i = 0; i < steps; ++i) {
                const int64_t row = ((int64_t) r * 104729 + i) % maxc;
                const float* src = dx + (size_t) (row % kChunk) * kWidth;
                if (k8) store_k8v4(src, vt, kc, ks, vq, vs, row, 1);
                else store_q8_pair(src, kc, ks, vc, vs, row, 1);
            }
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "decode rows");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            v.push_back((double) ms * 1e6 / steps);
        }
        const double ns = median_ns(v);
        const int bytes = k8 ? kCellBytesK8V4 : kCellBytesQ8;
        const double gbs = (double) bytes / ns;
        std::printf("  %-11s  %9.2f  %6.1f  %14.1f%%  %16.1f%%\n", k8 ? "k8v4" : "q8_0", ns, gbs, 100.0 * gbs / peak,
                    100.0 * gbs / 280.0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(1234);
    int fails = 0;
    long values = 0;
    layout_case(3, 32, rng, fails);
    layout_case(4, 256, rng, fails);
    round_trip(3, 32, rng, fails, values);
    round_trip(8, 256, rng, fails, values);
    fwht_case(16, 256, rng, fails);
    fwht_case(16, 128, rng, fails);
    at_case(rng, fails);
    if (fails) {
        std::fprintf(stderr, "poly_kv_q4_test: %d failure(s)\n", fails);
        return 1;
    }
    std::printf("poly_kv_q4_test: PASS (%ld values)\n", values);
    // the timing leg is separate from the pass: `--bench` runs it after the checks have passed
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--bench") == 0) bench_main();
    return 0;
}
