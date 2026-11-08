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
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

}  // namespace

int main() {
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
    return 0;
}
