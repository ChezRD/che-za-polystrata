// src/poly/kv_q8_test.cpp - the Q8_0 KV cache kernels of the poly engine, store_q8 and store_q8_at, against a host
// reference.  No model file and no card beyond CUDA device 0:
//   1. round trip: for several row counts and n in {32,128,256}, random floats of mixed magnitude are stored and
//      read back; every dequantized value must be within half a block scale (plus a hair) of the input;
//   2. byte layout: for n=256 the device codes and fp16 scales must match the host's own arithmetic byte for byte,
//      which pins codes[row*n + blk*32 + j] and scales[row*(n/32) + blk];
//   3. store_q8_at with a ring: two rows written from a device `at` land at (*at + r) % ring and equal what plain
//      store_q8 writes from the same position.
// Exit 0 prints "poly_kv_q8_test: PASS"; any failed assertion prints the offending index and values and exits 1.
//
// --bench: store_q8 timed as the q8_0 KV store, the comparison arm of MB-ROW-1-5.md's Row 1 (1,088 B a cell for
// 2 KV heads x 256 values, K and V).  The k8v4 arm is poly_kv_q4_test --bench; this one exists so the bytes and
// the time come from the same two binaries the plan names.
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
        std::fprintf(stderr, "poly_kv_q8_test: %s: %s\n", what, cudaGetErrorString(e));
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

// 1. Round trip: dequantize with the fp16-rounded scale read back from `scales` and bound the error.
void round_trip(int rows, int n, std::mt19937& rng, int& fails, long& values) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    Dev dx(x.size() * sizeof(float)), dcodes((size_t) rows * n), dscales((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q8(dx.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> codes((size_t) rows * n);
    std::vector<uint16_t> scales((size_t) rows * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");

    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float d = half_value(scales[(size_t) r * nb + b]);
            const float half_scale = std::fabs(d) * 0.5f;
            for (int j = 0; j < 32; ++j) {
                const size_t i = (size_t) r * n + (size_t) b * 32 + j;
                const float got = (float) codes[i] * d;
                // A q8_0 value cannot land closer than half a block scale.  The scale is written fp16 from the
                // full-precision d the codes were rounded with (the kernel's id = 1/d), so decoding with the fp16
                // scale can miss by up to 127 * 2^-11 block scales at the rounding boundary: allow |x| * 5e-4 too.
                const float tol = half_scale + std::fabs(x[i]) * 5e-4f + 1e-3f;
                ++values;
                if (!(std::fabs(got - x[i]) <= tol)) {
                    std::fprintf(stderr,
                                 "poly_kv_q8_test: round-trip: n=%d rows=%d value [%d,%d,%d] x=%.9g got=%.9g "
                                 "d=%.9g err=%.9g tol=%.9g\n",
                                 n, rows, r, b, j, (double) x[i], (double) got, (double) d,
                                 (double) std::fabs(got - x[i]), (double) tol);
                    ++fails;
                }
            }
        }
    }
}

// 2. Byte layout: the host's own d, codes and fp16 scale must equal the device's bytes.
void byte_layout(int rows, int n, std::mt19937& rng, int& fails, long& values) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    std::vector<int8_t> ecodes((size_t) rows * n);
    std::vector<uint16_t> escales((size_t) rows * nb);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* xr = x.data() + (size_t) r * n + (size_t) b * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(xr[j]));
            const float d = amax / 127.0f;
            const float id = d > 0.f ? 1.0f / d : 0.0f;
            escales[(size_t) r * nb + b] = half_bits(__float2half(d));
            int8_t* cr = ecodes.data() + (size_t) r * n + (size_t) b * 32;
            for (int j = 0; j < 32; ++j) {
                float q = std::rint(xr[j] * id);
                q = std::fmin(std::fmax(q, -127.0f), 127.0f);
                cr[j] = (int8_t) q;
            }
        }
    }

    Dev dx(x.size() * sizeof(float)), dcodes((size_t) rows * n), dscales((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q8(dx.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> codes((size_t) rows * n);
    std::vector<uint16_t> scales((size_t) rows * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");

    ++values;
    if (std::memcmp(ecodes.data(), codes.data(), codes.size()) != 0) {
        size_t i = 0;
        while (i < codes.size() && ecodes[i] == codes[i]) ++i;
        const int r = (int) (i / n), rem = (int) (i % n);
        std::fprintf(stderr,
                     "poly_kv_q8_test: byte layout: codes differ at row=%d blk=%d j=%d: host=%d device=%d (n=%d "
                     "rows=%d)\n",
                     r, rem / 32, rem % 32, (int) ecodes[i], (int) codes[i], n, rows);
        ++fails;
    }
    ++values;
    if (std::memcmp(escales.data(), scales.data(), scales.size() * sizeof(uint16_t)) != 0) {
        size_t i = 0;
        while (i < scales.size() && escales[i] == scales[i]) ++i;
        const int r = (int) (i / nb), b = (int) (i % nb);
        std::fprintf(stderr,
                     "poly_kv_q8_test: byte layout: scales differ at row=%d blk=%d: host=0x%04x device=0x%04x "
                     "(n=%d rows=%d)\n",
                     r, b, (unsigned) escales[i], (unsigned) scales[i], n, rows);
        ++fails;
    }
}

// 3. store_q8_at with a ring: rows land at (*at + r) % ring and match plain store_q8 from the same position.
void ring_test(std::mt19937& rng, int& fails, long& values) {
    const int n = 128, nb = n / 32, rows = 2, ring = 5;
    const long long at_val = 4;   // rows go to 4 and 0: the second wraps

    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    Dev dx(x.size() * sizeof(float)), d_at(sizeof(long long));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    ck(cudaMemcpy(d_at.p, &at_val, sizeof(at_val), cudaMemcpyHostToDevice), "upload at");

    Dev rcodes((size_t) ring * n), rscales((size_t) ring * nb * sizeof(uint16_t));
    strata::poly::store_q8_at(dx.as<float>(), rcodes.as<int8_t>(), rscales.as<uint16_t>(), n, d_at.as<long long>(),
                              rows, ring, nullptr);
    ck(cudaGetLastError(), "store_q8_at launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8_at sync");

    Dev pcodes((size_t) (at_val + rows) * n), pscales((size_t) (at_val + rows) * nb * sizeof(uint16_t));
    strata::poly::store_q8(dx.as<float>(), pcodes.as<int8_t>(), pscales.as<uint16_t>(), n, at_val, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> rcode((size_t) ring * n), pcode((size_t) (at_val + rows) * n);
    std::vector<uint16_t> rscale((size_t) ring * nb), pscale((size_t) (at_val + rows) * nb);
    ck(cudaMemcpy(rcode.data(), rcodes.p, rcode.size(), cudaMemcpyDeviceToHost), "download ring codes");
    ck(cudaMemcpy(rscale.data(), rscales.p, rscale.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download ring scales");
    ck(cudaMemcpy(pcode.data(), pcodes.p, pcode.size(), cudaMemcpyDeviceToHost), "download plain codes");
    ck(cudaMemcpy(pscale.data(), pscales.p, pscale.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download plain scales");

    for (int r = 0; r < rows; ++r) {
        const size_t rr = (size_t) ((at_val + r) % ring);
        const size_t pr = (size_t) (at_val + r);
        ++values;
        if (std::memcmp(rcode.data() + rr * n, pcode.data() + pr * n, (size_t) n) != 0) {
            size_t i = 0;
            while (i < (size_t) n && rcode[rr * n + i] == pcode[pr * n + i]) ++i;
            std::fprintf(stderr,
                         "poly_kv_q8_test: ring: row %d at %zu != plain row %zu: code [%zu] ring=%d plain=%d\n",
                         r, rr, pr, i, (int) rcode[rr * n + i], (int) pcode[pr * n + i]);
            ++fails;
        }
        ++values;
        if (std::memcmp(rscale.data() + rr * nb, pscale.data() + pr * nb, (size_t) nb * sizeof(uint16_t)) != 0) {
            size_t i = 0;
            while (i < (size_t) nb && rscale[rr * nb + i] == pscale[pr * nb + i]) ++i;
            std::fprintf(stderr,
                         "poly_kv_q8_test: ring: row %d at %zu != plain row %zu: scale [%zu] ring=0x%04x "
                         "plain=0x%04x\n",
                         r, rr, pr, i, (unsigned) rscale[rr * nb + i], (unsigned) pscale[pr * nb + i]);
            ++fails;
        }
    }
}

// --bench: the q8_0 store (store_q8 for the keys and again for the values), the comparison arm of Row 1.
// Same shapes, same two invocation forms as poly_kv_q4_test --bench: bulk (the whole cache in calls of kChunk rows)
// and decode (one row a call).
namespace bench {

constexpr int kHeadDim = 256, kKvHeads = 2;
constexpr int kWidth = kKvHeads * kHeadDim;   // 512 values a side
constexpr int kCellBytes = 544 + 544;         // 512 int8 codes + 16 fp16 scales, twice
constexpr int kChunk = 4096;

double median_ns(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace bench

void bench_main() {
    using namespace bench;
    const int contexts[] = {4096, 32768, 131072, 262144};
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);   // kHz, the card's max memory clock
    const double peak = 2.0 * (double) clk * 1e3 * (bus / 8) / 1e9;
    std::printf("q8_0 store: head_dim %d, %d KV heads, %d values a side; %d B a cell\n", kHeadDim, kKvHeads, kWidth,
                kCellBytes);
    std::printf("peak %.1f GB/s (2 x %.0f MHz x %d bit / 8, clocks.max.memory)\n", peak, clk / 1e3, bus);
    const int maxc = contexts[3];
    // The source is one chunk, cycled for every row: store_q8 reads 4 B a value to quantize it, and in the engine
    // that source is the projection's fresh output, hot in this card's 32 MB L2 - not a DRAM pool.  See
    // poly_kv_q4_test --bench for the same choice and why the criterion is about the write.
    constexpr int kChunkRows = 4096;
    std::mt19937 rng(99);
    float* dx;
    int8_t *kc, *vc;
    uint16_t *ks, *vs;
    ck(cudaMalloc(&dx, (size_t) kChunkRows * kWidth * sizeof(float)), "bench x");
    ck(cudaMalloc(&kc, (size_t) maxc * kWidth), "bench k codes");
    ck(cudaMalloc(&ks, (size_t) maxc * (kWidth / 32) * sizeof(uint16_t)), "bench k scales");
    ck(cudaMalloc(&vc, (size_t) maxc * kWidth), "bench v codes");
    ck(cudaMalloc(&vs, (size_t) maxc * (kWidth / 32) * sizeof(uint16_t)), "bench v scales");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto store_pair = [&](int64_t first, int rows) {
        strata::poly::store_q8(dx, kc, ks, kWidth, first, rows, nullptr);
        strata::poly::store_q8(dx, vc, vs, kWidth, first, rows, nullptr);
    };
    // the whole cache in chunks, each chunk read from the same L2-resident source
    auto store_all = [&](int cells) {
        for (int row = 0; row < cells; row += kChunkRows)
            store_pair(row, std::min(kChunkRows, cells - row));
    };
    std::printf("\nbulk: the whole cache written, %d rows a call\n", kChunk);
    std::printf("  cells   ns/cell   GB/s   %% of %.0f (max clk)   %% of 280.0 (loaded)\n", peak);
    for (int c : contexts) {
        std::vector<double> v;
        const int reps = c > 100000 ? 5 : 9;
        for (int r = 0; r < reps; ++r) {
            cudaEventRecord(e0);
            store_all(c);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "bulk q8");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) v.push_back((double) ms * 1e6 / c);
        }
        const double ns = median_ns(v);
        const double gbs = (double) kCellBytes / ns;
        std::printf("  %6d  %9.2f  %6.1f  %14.1f%%  %16.1f%%\n", c, ns, gbs, 100.0 * gbs / peak,
                    100.0 * gbs / 280.0);
    }
    const int steps = 65536;
    std::printf("\ndecode: one row a call, %d calls a rep, the cache's own rows cycled\n", steps);
    std::printf("  arm     ns/cell   GB/s   %% of %.0f (max clk)   %% of 280.0 (loaded)\n", peak);
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < steps; ++i)
            store_pair(((int64_t) r * 104729 + i) % maxc, 1);
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "decode rows");
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        v.push_back((double) ms * 1e6 / steps);
    }
    const double ns = median_ns(v);
    const double gbs = (double) kCellBytes / ns;
    std::printf("  q8_0  %9.2f  %6.1f  %14.1f%%  %16.1f%%\n", ns, gbs, 100.0 * gbs / peak, 100.0 * gbs / 280.0);
}

}  // namespace

int main(int argc, char** argv) {
    ck(cudaSetDevice(0), "cudaSetDevice");

    std::mt19937 rng(0x51u);
    int fails = 0;
    long values = 0;

    const int ns[3] = {32, 128, 256};
    const int rss[3] = {1, 3, 7};
    // many trials: 20 sweeps of (1+3+7) rows over three n = 660 random rows
    for (int t = 0; t < 20; ++t)
        for (int ri = 0; ri < 3; ++ri)
            for (int ci = 0; ci < 3; ++ci) round_trip(rss[ri], ns[ci], rng, fails, values);

    byte_layout(2, 256, rng, fails, values);
    byte_layout(3, 256, rng, fails, values);
    byte_layout(5, 256, rng, fails, values);

    ring_test(rng, fails, values);

    if (fails) {
        std::fprintf(stderr, "poly_kv_q8_test: FAIL: %d failed assertion(s)\n", fails);
        return 1;
    }
    std::printf("poly_kv_q8_test: PASS (%ld values checked)\n", values);
    // the timing leg is separate from the pass: `--bench` runs it after the checks have passed
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--bench") == 0) bench_main();
    return 0;
}
