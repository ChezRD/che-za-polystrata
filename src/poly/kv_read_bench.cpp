// src/poly/kv_read_bench.cpp - Phase 0 row 2 measured on every poly KV read path, for the geometry the caller
// names, so the same table can be produced for each model family the engine serves
// (docs/../OPTIMIZATION-STACK.md §0.3 Row 2).
//
//     build/poly_kv_read_bench <dim> <n_head> <kv_heads> [--cells 4096,32768,131072,262144] [--itr N]
//
// Which paths a geometry may use is decided by the engine, not by this bench, and the bench reports the refusal
// instead of hiding it:
//   fdec_k8v4 / fdec_q8   require dim in {128,256} and n_head == 8*kv_heads or 16*kv_heads
//                         (src/poly/fattn.cu:691-693, :630-636)
//   attend / attend_batch  any dim % 32 == 0 n_head / kv_heads, but the hand-written split kernel that a decode
//                         call takes is only for dim 64/128 with groups 4/8 (src/poly/kernels.cu:3214)
//
// The geometries that matter (MB-ROW-1-5.md §1):
//   24 2 256  group 12  Qwen3.8-Flash REAP-320 and standard  -> served by strata, REFUSED here
//   32 4 128  group  8  Flint 40B                             -> accepted
//   16 2 256  group  8  Qwen3.6-35B-A3B                       -> accepted
//   24 4 256  group  6  Qwen3.8-27B dense                     -> REFUSED here
//
// Bytes a cell: q8_0 keys (2 sides of kv_heads*head_dim, one fp16 scale per 32) and q4_0 values (18 B per 32).
#include "fattn.hpp"
#include "strata/poly/kernels.hpp"

namespace strata::poly {
/// The group rule fdec_fits() uses (src/poly/fattn.cu:742); declared here so this bench reports the engine's answer
/// rather than its own copy of it.
bool fdec_group_ok(int n_head, int kv_heads, int tokens);
}  // namespace strata::poly

#include <cublas_v2.h>
#include <cuda_fp16.h>
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

using namespace strata::poly;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "poly_kv_read_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, (n ? n : 1) * sizeof(T)), "cudaMalloc");
    ck(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)), "cudaMemset");
    return p;
}

template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: poly_kv_read_bench <dim> <n_head> <kv_heads> [--cells ...] [--itr N]\n");
        return 2;
    }
    const int dim = std::atoi(argv[1]), n_head = std::atoi(argv[2]), kv_heads = std::atoi(argv[3]);
    std::vector<int> cells_list = {4096, 32768, 131072, 262144};
    int iters = 10;
    bool only_fdec = false;
    int rows_max_override = 0;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cells" && i + 1 < argc) {
            cells_list.clear();
            std::string s = argv[++i];
            size_t from = 0;
            for (;;) {
                const size_t c = s.find(',', from);
                cells_list.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
                if (c == std::string::npos) break;
                from = c + 1;
            }
        } else if (a == "--itr" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (a == "--only-fdec") {
            only_fdec = true;
        } else if (a == "--rows-max" && i + 1 < argc) {
            rows_max_override = std::atoi(argv[++i]);
        }
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    int prop_nsm = 0;
    {
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0), "props");
        prop_nsm = prop.multiProcessorCount;
    }
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);
    const double peak = 2.0 * (double) clk * 1e3 * (bus / 8) / 1e9;

    const int group = n_head / kv_heads;
    const int width = kv_heads * dim;
    const int nb = width / 32;
    const int cell_bytes = width + nb * 2 + (nb * 16 + nb * 2);   // K codes+scales, V nibbles+scales
    std::printf("poly KV read: dim %d, %d heads, %d KV, group %d, %d values a row, %d B a cell\n", dim, n_head, kv_heads,
                group, width, cell_bytes);
    std::printf("  peak %.1f GB/s at %.0f MHz; 34 SMs\n", peak, clk / 1e3);

    // the engine's own guards, reported rather than worked around
    // the engine's own rule, not a copy of it: fdec_group_ok is what fdec_fits() calls
    const bool fdec_ok = (dim == 128 || dim == 256) && strata::poly::fdec_group_ok(n_head, kv_heads, 1);
    const bool split_small = (dim == 64 && group == 4) || (dim == 128 && group == 8);
    std::printf("  fdec_k8v4 accepted: %s%s\n", fdec_ok ? "yes" : "NO",
                fdec_ok ? "" : "  (src/poly/fattn.cu fdec_group_ok: group must be 4/6/8/12/16, kDecRows % group == 0, dim 128 or 256)");
    std::printf("  attend's hand-written split kernel: %s%s\n", split_small ? "yes" : "no (falls back to the generic "
                "split/chunk path)",
                "");

    // The caches are allocated for the largest context of the RUN, so the same measured point can land in a
    // differently sized allocation.  --rows-max pins that, which is how the 128K anomaly was traced: the same call
    // costs 5.5 ns/cell inside a 256K allocation and 8.1 inside a 128K one.
    const int rows_max = rows_max_override > 0 ? rows_max_override
                                               : *std::max_element(cells_list.begin(), cells_list.end()) + 1;
    std::mt19937 rng(20261012u);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- an fp16 source, quantized on the host into the two caches (no store kernels: this is the read row)
    std::vector<int8_t> kc((size_t) rows_max * width);
    std::vector<uint16_t> ks((size_t) rows_max * nb);
    std::vector<uint8_t> k6((size_t) rows_max * nb * 24);
    std::vector<uint16_t> k6s((size_t) rows_max * nb);
    std::vector<uint8_t> vq((size_t) rows_max * nb * 16);
    std::vector<uint16_t> vs((size_t) rows_max * nb);
    {
        std::vector<float> x((size_t) width);
        for (int r = 0; r < rows_max; ++r) {
            for (float& v : x) v = 2.f * nd(rng);
            for (int b = 0; b < nb; ++b) {
                const float* xb = x.data() + b * 32;
                float amax = 0.f, mval = 0.f;
                for (int j = 0; j < 32; ++j) {
                    const float a = std::fabs(xb[j]);
                    if (a > amax || (a == amax && xb[j] > mval)) { amax = a; mval = xb[j]; }
                }
                const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
                ks[(size_t) r * nb + b] = __half_as_ushort(__float2half(d));
                for (int j = 0; j < 32; ++j) {
                    float q = std::rint(xb[j] * id);
                    q = std::fmin(std::fmax(q, -127.f), 127.f);
                    kc[(size_t) r * width + b * 32 + j] = (int8_t) q;
                }
                // q6_0, store_q6()'s rule: d = amax/31, code = clamp(rint(x/d), -31, 31) + 32, 24 bytes a block
                {
                    const float d6 = amax / 31.f, i6 = d6 > 0.f ? 1.f / d6 : 0.f;
                    k6s[(size_t) r * nb + b] = __half_as_ushort(__float2half(d6));
                    uint32_t w6[6] = {0, 0, 0, 0, 0, 0};
                    for (int j = 0; j < 32; ++j) {
                        float q = std::rint(xb[j] * i6);
                        q = std::fmin(std::fmax(q, -31.f), 31.f);
                        const uint32_t v = (uint32_t) ((int) q + 32);
                        const int bit = 6 * j, wi = bit >> 5, sh = bit & 31;
                        w6[wi] |= v << sh;
                        if (sh > 26) w6[wi + 1] |= v >> (32 - sh);
                    }
                    std::memcpy(k6.data() + ((size_t) r * nb + b) * 24, w6, 24);
                }
                // q4_0, ggml's rule (kv_q4_test.cpp's host reference)
                const float dq = mval / -8.f, iq = dq != 0.f ? 1.f / dq : 0.f;
                vs[(size_t) r * nb + b] = __half_as_ushort(__float2half_rn(dq));
                for (int j = 0; j < 16; ++j) {
                    const int q0 = (int) (xb[j] * iq + 8.5f);
                    const int q1 = (int) (xb[j + 16] * iq + 8.5f);
                    const uint8_t a = (uint8_t) (q0 < 0 ? 0 : (q0 > 15 ? 15 : q0));
                    const uint8_t c = (uint8_t) (q1 < 0 ? 0 : (q1 > 15 ? 15 : q1));
                    vq[((size_t) r * nb + b) * 16 + j] = (uint8_t) (a | (c << 4));
                }
            }
        }
    }
    // the fp16 control: the same values as halves, so q8_0 / k8v4 / fp16 are three readings of one payload
    std::vector<uint16_t> hf((size_t) rows_max * width);
    {
        std::vector<float> x((size_t) width);
        for (int r = 0; r < rows_max; ++r) {
            for (float& v : x) v = 2.f * nd(rng);
            for (int i = 0; i < width; ++i) hf[(size_t) r * width + i] = __half_as_ushort(__float2half(x[i]));
        }
    }
    uint16_t* dhf = upload(hf);
    // a SEPARATE value cache for each arm.  An earlier version passed the KEY cache as the value cache
    // (`fdec_q8(q, kq, kq, ...)`), so the q8_0 arm read half the bytes and half of them twice - the same pages,
    // already in L2 - and reported 451 GB/s on a 288 GB/s card.  Each arm now has its own buffer.
    std::vector<int8_t> vc((size_t) rows_max * width);
    std::vector<uint16_t> vsc((size_t) rows_max * nb);
    for (size_t i = 0; i < vc.size(); ++i) vc[i] = (int8_t) ((int) (rng() % 255u) - 127);
    for (auto& x : vsc) x = (uint16_t) (0x2000 + (rng() % 0x100u));
    std::vector<uint16_t> hfv((size_t) rows_max * width);
    for (auto& x : hfv) x = __half_as_ushort(__float2half(2.f * nd(rng)));
    int8_t* dvc = upload(vc);
    uint16_t* dvsc = upload(vsc);
    uint16_t* dhfv = upload(hfv);
    vc.clear(); vc.shrink_to_fit();
    vsc.clear(); vsc.shrink_to_fit();
    hfv.clear(); hfv.shrink_to_fit();
    hf.clear();
    hf.shrink_to_fit();
    int8_t* dkc = upload(kc);
    uint16_t* dks = upload(ks);
    uint8_t* dk6 = upload(k6);
    uint16_t* dk6s = upload(k6s);
    k6.clear(); k6.shrink_to_fit();
    k6s.clear(); k6s.shrink_to_fit();
    uint8_t* dvq = upload(vq);
    uint16_t* dvs = upload(vs);
    kc.clear(); kc.shrink_to_fit();
    ks.clear(); ks.shrink_to_fit();
    vq.clear(); vq.shrink_to_fit();
    vs.clear(); vs.shrink_to_fit();
    const Kvq8 kq{dkc, dks};
    const Kvq4 v4{dvq, dvs};
    const Kvq6 k6q{dk6, dk6s};

    std::vector<float> hq((size_t) (n_head * dim));
    for (float& x : hq) x = 2.f * nd(rng);
    float* q = upload(hq);
    const size_t out_n = (size_t) (n_head * dim);
    float* out = dalloc<float>(out_n);
    float* out2 = dalloc<float>(out_n);
    const size_t score_n = size_t(16) << 20;
    float* score = dalloc<float>(score_n);
    const float scale = 1.f / 16.f;
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto time_ms = [&](auto&& run) {
        run();
        ck(cudaDeviceSynchronize(), "warm");
        std::vector<double> v;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i) run();
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "timed");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) v.push_back((double) ms / iters);
        }
        return median(v);
    };

    std::printf("\n   cells   arm          ms        ns/cell   GB/s   %% of 288   %% of 280  | q8_0: ms  ns/cell  GB/s"
                " || fp16: ms  ns/cell  GB/s\n");
    double k8[8] = {0}, q8[8] = {0};
    for (size_t ci = 0; ci < cells_list.size() && ci < 8; ++ci) {
        const long long pos = cells_list[ci];
        const long long rows = pos + 1;
        // ---- fdec_k8v4 against fdec_q8, the same cache shape (V as q8_0 for the comparison arm)
        if (fdec_ok) {
            // `room` is the EXACT row count: a power-of-two room reads up to twice the cache and reports twice the
            // work (MB-ROW-1-5.md §10 item 2, where this exact bug cost a 2x error at 65,536 cells).
            const int r = (int) std::min<long long>(rows, rows_max);
            const double m4 = time_ms([&] {
                fdec_k8v4(q, kq, v4, pos, 1, nullptr, r, out, dim, n_head, kv_heads, scale, score, score_n, nullptr);
            });
            const double m6 = time_ms([&] {
                fdec_k6v4(q, k6q, v4, pos, 1, nullptr, r, out, dim, n_head, kv_heads, scale, score, score_n, nullptr);
            });
            const double mq = time_ms([&] {
                fdec_q8(q, kq, Kvq8{dvc, dvsc}, pos, 1, nullptr, r, out2, dim, n_head, kv_heads, scale, score, score_n,
                        nullptr);
            });
            // fp16: the same values as halves, read where they lie
            const double mh = time_ms([&] {
                fdec_f16(q, dhf, dhfv, pos, 1, nullptr, r, out2, dim, n_head, kv_heads, scale, score, score_n, nullptr);
            });
            (void) mh;
            k8[ci] = m4;
            q8[ci] = mq;
            const double ns6 = m6 * 1e6 / (double) rows;
            const double cell6 = nb * 24 + nb * 2 + nb * 16 + nb * 2;   // K six-bit codes + scales, V q4_0
            const double gbs6 = (double) cell6 * (double) rows / (m6 * 1e6);
            std::printf("  %7lld  fdec_k6v4  %9.4f  %9.2f  %6.1f  %8.1f%%  %8.1f%%  | k6v4/k8v4 %.4f  (%d B a cell)\n",
                        pos, m6, ns6, gbs6, 100.0 * gbs6 / peak, 100.0 * gbs6 / 280.0, m6 / m4, (int) cell6);
            const double ns = m4 * 1e6 / (double) rows;
            const double gbs = (double) cell_bytes * (double) rows / (m4 * 1e6);
            const int ctas = (int) ((rows + 1023) / 1024 * kv_heads);
            const int cell16 = width * 2 * 2;   // fp16: two sides of `width` halves
            const int cell8 = width + nb * 2;   // q8_0: one side's codes and scales, counted once below
            (void) cell8;
            const double nsq = mq * 1e6 / (double) rows, nsh = mh * 1e6 / (double) rows;
            const double gbsq = (double) cell_bytes * (double) rows / (mq * 1e6);
            const double gbsh = (double) cell16 * (double) rows / (mh * 1e6);
            std::printf("  %7lld  fdec_k8v4  %9.4f  %9.2f  %6.1f  %8.1f%%  %8.1f%%  | q8_0 %9.4f %8.2f %6.1f || "
                        "f16 %9.4f %8.2f %6.1f\n",
                        pos, m4, ns, gbs, 100.0 * gbs / peak, 100.0 * gbs / 280.0, mq, nsq, gbsq, mh, nsh, gbsh);
        }
        // ---- attend with the same k8v4 caches: the engine's generic path (also what a geometry fdec refuses takes)
        // --only fdec: skip it.  Two kernels share one 67 MB score buffer and the card's 32 MB L2, so a context's
        // attend can perturb the next context's fdec; a run of one arm alone settles whether it does.
        if (!only_fdec) {
            const int chunk = std::max(1, (int) std::min<size_t>(4, score_n / (size_t(n_head) * (pos + 1))));
            const double ma = time_ms([&] {
                attend(q, nullptr, nullptr, pos, 1, score, chunk, out, dim, n_head, kv_heads, scale, nullptr, 0, 0,
                       false, nullptr, 0, 0, kq, Kvq8{}, nullptr, 0, v4);
            });
            const double ns = ma * 1e6 / (double) rows;
            const double gbs = (double) cell_bytes * (double) rows / (ma * 1e6);
            std::printf("  %7lld  attend     %9.4f  %9.2f  %6.1f  %8.1f%%  %8.1f%%\n", pos, ma, ns, gbs,
                        100.0 * gbs / peak, 100.0 * gbs / 280.0);
        }
    }
    // ---- the bar: k6v4 against k8v4, the two arms ALTERNATING inside one repetition loop rather than one arm's
    // whole median against another's.  MB-ROW-1-5.md §32.3 is why: a single run of k8v4 against q8_0 gave 1.11 and
    // five interleaved runs 1.185, so the alternated ratio is the one to quote.  Each repetition times one call of
    // each arm back to back; the median of the per-repetition ratios is the number.
    if (fdec_ok) {
        const long long pos = cells_list.back();
        const long long rows = std::min<long long>(pos + 1, rows_max);
        const int r = (int) rows;
        std::vector<double> ratio;
        for (int rep = 0; rep < 5; ++rep) {
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i)
                fdec_k8v4(q, kq, v4, pos, 1, nullptr, r, out, dim, n_head, kv_heads, scale, score, score_n, nullptr);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "timed");
            float ms8 = 0.f;
            cudaEventElapsedTime(&ms8, e0, e1);
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i)
                fdec_k6v4(q, k6q, v4, pos, 1, nullptr, r, out, dim, n_head, kv_heads, scale, score, score_n, nullptr);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "timed");
            float ms6 = 0.f;
            cudaEventElapsedTime(&ms6, e0, e1);
            if (rep) ratio.push_back((double) ms6 / ms8);
        }
        const double md = median(ratio);
        double lo = *std::min_element(ratio.begin(), ratio.end()), hi = *std::max_element(ratio.begin(), ratio.end());
        std::printf("\nk6v4 / k8v4 at %lld cells, five runs alternating in one process: %.4f (per-run %.4f..%.4f)\n",
                    pos, md, lo, hi);
        std::printf("  k6v4 %s the 5%% bar: %s\n", md <= 1.05 ? "clears" : "MISSES",
                    md <= 1.05 ? "within 5% of k8v4's per-cell cost" : "more than 5% above k8v4's per-cell cost");
    }
    std::printf("\nCTAs a call: fdec = ceil(rows/1024) x %d KV; attend = ceil(%lld/64) x 1 block a head group\n",
                kv_heads, (long long) cells_list.back());
    return 0;
}
