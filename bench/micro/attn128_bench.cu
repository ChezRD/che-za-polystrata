// Decode and prompt attention against attend(), for the audit and the kernel choices.
// The 64-key tile is the reference in attn128_64.cu; the 32-key tile is the only production path, and after the
// k < DIM/16 fix the two differ by 1.3e-6 at 8K.
//   build-poly-local must exist (ninja -C build-poly-local strata-poly).  Build and run:
//   nvcc -O3 -std=c++20 -arch=sm_89 -c bench/micro/attn128_64.cu -o /tmp/attn128_64.o
//   g++ -O2 -std=c++20 -x c++ -I include -I src/poly -I build-poly-local/_deps/strata_llamacpp-src/ggml/include \
//     -isystem /opt/cuda/targets/x86_64-linux/include -isystem /opt/cuda/targets/x86_64-linux/include/cccl \
//     bench/micro/attn128_bench.cu -x none /tmp/attn128_64.o \
//     build-poly-local/CMakeFiles/strata-poly.dir/src/poly/kernels.cu.o \
//     -Lbuild-poly-local -L/opt/cuda/targets/x86_64-linux/lib -L/opt/cuda/targets/x86_64-linux/lib/stubs \
//     -lstrata_poly_mmq -lstrata_kernels -lstrata_spec -lstrata_gdn_mma -lstrata_core \
//     build-poly-local/ggml/src/libggml-cpu.a build-poly-local/ggml/src/libggml-base.a \
//     -lcublas -lcudart -lcublasLt -lcudadevrt -lcudart_static -lpthread -ldl -lrt -lm -o poly_attn128_bench
#include "fattn.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

using namespace strata::poly;

extern "C" void bench_flash128_64(const float* q, const uint16_t* kcache, const uint16_t* vcache, int rows, int pos,
                                  float* out, int n_head, int kv_heads, float scale, void* stream);

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static uint32_t seed = 12345;
static float rnd() {
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}
static uint16_t h16(float x) { return __half_as_ushort(__float2half(x)); }

static float worst(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

static double time_ms(cudaStream_t s, int reps, const std::function<void()>& call) {
    float best = 1e30f;
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "event");
        ck(cudaEventCreate(&b), "event");
        ck(cudaEventRecord(a, s), "record");
        call();
        ck(cudaEventRecord(b, s), "record");
        ck(cudaEventSynchronize(b), "sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        best = std::fmin(best, ms);
        ck(cudaEventDestroy(a), "event");
        ck(cudaEventDestroy(b), "event");
    }
    return best;
}

int main(int argc, char** argv) {
    const int max_rows = argc > 1 ? std::atoi(argv[1]) : 8224;
    const int cache_rows = std::max(max_rows, 10240);
    const int dim = 128, n_head = 32, kv_heads = 4;
    const float scale = 1.0f / std::sqrt((float) dim);
    const size_t width = (size_t) kv_heads * dim;
    const size_t swidth = width / 32;
    const size_t q4row = swidth * 16;
    unsetenv("STRATA_FDEC");

    cudaStream_t s;
    ck(cudaSetDevice(0), "device");
    ck(cudaStreamCreate(&s), "stream");
    float *q, *out, *ref, *score, *sinks;
    uint16_t *k, *v;
    int8_t* kc;
    uint16_t* ks;
    uint8_t* vq;
    uint16_t* vs;
    const size_t rows_max = 4096;
    ck(cudaMalloc(&q, rows_max * n_head * dim * sizeof(float)), "q");
    ck(cudaMalloc(&out, rows_max * n_head * dim * sizeof(float)), "out");
    ck(cudaMalloc(&ref, rows_max * n_head * dim * sizeof(float)), "ref");
    float* out64;
    ck(cudaMalloc(&out64, rows_max * n_head * dim * sizeof(float)), "out64");
    const size_t score_n = std::max(attend_split_floats(2048, n_head, 10240, dim),
                                    attend_split_floats(4096, n_head, 4096, dim));
    ck(cudaMalloc(&score, score_n * sizeof(float)), "score");
    ck(cudaMalloc(&sinks, (size_t) n_head * sizeof(float)), "sinks");
    ck(cudaMalloc(&k, (size_t) cache_rows * width * sizeof(uint16_t)), "k");
    ck(cudaMalloc(&v, (size_t) cache_rows * width * sizeof(uint16_t)), "v");
    ck(cudaMalloc(&kc, (size_t) cache_rows * width), "kc");
    ck(cudaMalloc(&ks, (size_t) cache_rows * swidth * sizeof(uint16_t)), "ks");
    ck(cudaMalloc(&vq, (size_t) cache_rows * q4row), "vq");
    ck(cudaMalloc(&vs, (size_t) cache_rows * swidth * sizeof(uint16_t)), "vs");

    std::vector<uint16_t> hk((size_t) cache_rows * width), hv((size_t) cache_rows * width);
    std::vector<int8_t> hkc((size_t) cache_rows * width);
    std::vector<uint16_t> hks((size_t) cache_rows * swidth), hvs((size_t) cache_rows * swidth);
    std::vector<uint8_t> hvq((size_t) cache_rows * q4row);
    for (auto& x : hk) x = h16(rnd());
    for (auto& x : hv) x = h16(rnd());
    for (auto& x : hkc) x = (int8_t) (rnd() * 8.0f);
    for (auto& x : hks) x = h16(0.01f + std::fabs(rnd()) * 0.01f);
    for (auto& x : hvs) x = h16(0.01f + std::fabs(rnd()) * 0.01f);
    for (auto& x : hvq) x = (uint8_t) (rnd() * 127.0f);
    ck(cudaMemcpy(k, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice), "k");
    ck(cudaMemcpy(v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice), "v");
    ck(cudaMemcpy(kc, hkc.data(), hkc.size(), cudaMemcpyHostToDevice), "kc");
    ck(cudaMemcpy(ks, hks.data(), hks.size() * 2, cudaMemcpyHostToDevice), "ks");
    ck(cudaMemcpy(vq, hvq.data(), hvq.size(), cudaMemcpyHostToDevice), "vq");
    ck(cudaMemcpy(vs, hvs.data(), hvs.size() * 2, cudaMemcpyHostToDevice), "vs");

    // ---- prompt attention: attn_flash128 against attend(), 32 and 2048 tokens ----
    for (const auto& [rows, pos] : {std::pair{32, 0}, std::pair{32, 8192}, std::pair{2048, 0},
                                   std::pair{2048, 8192}, std::pair{4096, 0}}) {
        std::vector<float> hq((size_t) rows * n_head * dim);
        for (auto& x : hq) x = rnd();
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        const size_t keys = (size_t) pos + rows;
        const size_t sfl = attend_split_floats(rows, n_head, keys, dim);
        const auto flash = [&] {
            attend_batch(nullptr, q, k, v, pos, rows, score, sfl, nullptr, out, dim, n_head, kv_heads, scale, s, 0,
                         0, false, Kvq8{}, Kvq8{}, nullptr, nullptr, 0, nullptr, Kvq4{});
        };
        const auto flash64 = [&] {
            bench_flash128_64(q, k, v, rows, pos, out64, n_head, kv_heads, scale, s);
        };
        const auto old = [&] {
            attend(q, k, v, pos, rows, score, rows, ref, dim, n_head, kv_heads, scale, s, 0, 0, false, nullptr, 0,
                   sfl, Kvq8{}, Kvq8{}, nullptr, 0, Kvq4{});
        };
        flash();
        flash64();
        old();
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) rows * n_head * dim), b((size_t) rows * n_head * dim), c((size_t) rows * n_head * dim);
        ck(cudaMemcpy(a.data(), ref, a.size() * 4, cudaMemcpyDeviceToHost), "ref");
        ck(cudaMemcpy(b.data(), out, b.size() * 4, cudaMemcpyDeviceToHost), "out32");
        ck(cudaMemcpy(c.data(), out64, c.size() * 4, cudaMemcpyDeviceToHost), "out64");
        std::printf("flash128 rows %4d pos %5d: 32 |diff| %.3g, 64 |diff| %.3g, 32vs64 %.3g; "
                    "32 %.3f ms, 64 %.3f ms, attend %.3f ms\n", rows, pos,
                    worst(b, a), worst(c, a), worst(b, c), time_ms(s, 20, flash), time_ms(s, 20, flash64),
                    time_ms(s, 20, old));
    }

    // ---- decode: fdec_q8 / fdec_k8v4 / fdec_f16 against attend(), 100 random (pos, n) ----
    const size_t sfl = attend_split_floats(4, n_head, max_rows, dim);
    for (int kind = 0; kind < 3; ++kind) {
        const char* names[3] = {"fdec_f16", "fdec_q8", "fdec_k8v4"};
        double sum = 0;
        float mx = 0;
        int over1 = 0, over2 = 0;
        for (int trial = 0; trial < 100; ++trial) {
            const int n = 1 + (int) (std::fabs(rnd()) * 3.999f);
            const int pos = (int) (std::fabs(rnd()) * (max_rows - n - 1));
            std::vector<float> hq((size_t) n * n_head * dim), hs(n_head);
            for (auto& x : hq) x = rnd();
            for (auto& x : hs) x = rnd();
            ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
            ck(cudaMemcpy(sinks, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice), "sinks");
            bool ok = false;
            if (kind == 0)
                ok = fdec_f16(q, k, v, pos, n, nullptr, 0, out, dim, n_head, kv_heads, scale, score, sfl, s, sinks);
            else if (kind == 1)
                ok = fdec_q8(q, Kvq8{kc, ks}, Kvq8{kc, ks}, pos, n, nullptr, 0, out, dim, n_head, kv_heads, scale,
                             score, sfl, s, sinks);
            else
                ok = fdec_k8v4(q, Kvq8{kc, ks}, Kvq4{vq, vs}, pos, n, nullptr, 0, out, dim, n_head, kv_heads, scale,
                               score, sfl, s, sinks);
            const Kvq8 kq_ref = (kind == 1 || kind == 2) ? Kvq8{kc, ks} : Kvq8{};
            const Kvq8 vq_ref = kind == 1 ? Kvq8{kc, ks} : Kvq8{};
            attend(q, kind == 0 ? k : nullptr, kind == 0 ? v : nullptr, pos, n, score, n, ref, dim, n_head, kv_heads,
                   scale, s, 0, 0, false, nullptr, 0, sfl, kq_ref, vq_ref, sinks, 0,
                   kind == 2 ? Kvq4{vq, vs} : Kvq4{});
            ck(cudaStreamSynchronize(s), "sync");
            if (!ok) {
                std::printf("%s: no kernel\n", names[kind]);
                break;
            }
            std::vector<float> a((size_t) n * n_head * dim), b((size_t) n * n_head * dim);
            ck(cudaMemcpy(a.data(), out, a.size() * 4, cudaMemcpyDeviceToHost), "out");
            ck(cudaMemcpy(b.data(), ref, b.size() * 4, cudaMemcpyDeviceToHost), "ref");
            const float d = worst(a, b);
            mx = std::fmax(mx, d);
            sum += d;
            over1 += d > 1e-3f;
            over2 += d > 2e-3f;
        }
        const int n = 1, pos = max_rows - 2;
        std::vector<float> hq((size_t) n * n_head * dim);
        for (auto& x : hq) x = rnd();
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        const auto call = [&] {
            if (kind == 0)
                fdec_f16(q, k, v, pos, n, nullptr, 0, out, dim, n_head, kv_heads, scale, score, sfl, s, sinks);
            else if (kind == 1)
                fdec_q8(q, Kvq8{kc, ks}, Kvq8{kc, ks}, pos, n, nullptr, 0, out, dim, n_head, kv_heads, scale, score,
                        sfl, s, sinks);
            else
                fdec_k8v4(q, Kvq8{kc, ks}, Kvq4{vq, vs}, pos, n, nullptr, 0, out, dim, n_head, kv_heads, scale, score,
                          sfl, s, sinks);
        };
        std::printf("%-10s 100 random: worst %.3g mean %.3g, >1e-3 %d, >2e-3 %d, %.3f ms at %d keys\n", names[kind],
                    mx, sum / 100.0, over1, over2, time_ms(s, 20, call), pos);
    }
    return 0;
}
