// src/kernels/q8_1_quant_bench.cpp - Row 5 of the optimization stack's Phase 0 (OPTIMIZATION-STACK.md §0.3):
// the Q8_1 activation quantizers timed at the shapes the MoE path uses, as GB/s of activations in plus int8 out.
//
//     build/q8_1_quant_bench [--cols 1,2,4,8,16] [--bw GB/s]
//
// The engine quantizes a token's 2,560-value hidden state before every routed-expert call, and the REAP-320 pack
// runs 320 experts a layer over 48 layers, so this launch happens once per MoE layer per token.  Two entry points,
// the two the plan names:
//
//   native_quantize_q8_1   one warp a 32-value block, no launch per row (native_mmvq.cu
//   quantize_q8_1_rows     the generic entry, one thread a value, a warp a block     iq_kernels.cu)
//
// Bytes: 4 B a float in, 36 B an out block (fp16 scale + fp16 sum + 32 int8) per 32 values, i.e. 5.125 B a value
// against 4.  The rate is (4 + 4.125) B a value over the time, which is what a pass against the DRAM roofline
// means for a pass that reads and writes.
//
// The known reference is `quantize_q8_1_kernel 2.30 us / 25.43% DRAM` from the ncu run in docs/PLATFORM_GATES.md:255.
// Pass (the plan): >= 25% of the peak; below 15% it is launch-bound and should be fused (STRATA_QFUSE), not measured
// again.  Both denominators are printed: 288 GB/s (the §0.1 derivation from the card's 9,001 MHz max memory clock)
// and 280.0 GB/s (8,751 MHz, what the card reports while loaded).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

// A token's hidden state is 2,560 values (Flash-Next's n_embd); the REAP-320 pack routes to 320 experts and the
// engine quantizes once a layer, so this is the shape and the frequency.
constexpr int kEmbd = 2560;
constexpr int kExperts = 320;
constexpr int kLayers = 48;

constexpr double kBytesPerValue = 4.0 + 32.0 / 32.0 + 8.0 / 32.0;   // in float, out codes, out scale+sum halves

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> cols = {1, 2, 4, 8, 16};
    double bw = 288.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cols" && i + 1 < argc) cols = ints(argv[++i]);
        else if (a == "--bw" && i + 1 < argc) bw = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: q8_1_quant_bench [--cols 1,2,4,8,16] [--bw GB/s]\n");
            return 2;
        }
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    // native_quantize_q8_1 requires an explicit non-null stream; create one and use it for both entries
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");
    // the card's own peak, for the two fractions: the max memory clock (what §0.1 derives 288 from) and the loaded
    // one (8,751 MHz = 280.0 GB/s).  Both are printed; the pass is judged against `bw`.
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);   // kHz
    const double peak_max = 2.0 * (double) clk * 1e3 * (bus / 8) / 1e9;
    const double peak_loaded = peak_max * 8751.0 / (clk / 1e3);
    std::printf("q8_1 activation quant, %d values a token, %d experts x %d layers the REAP-320 pack\n", kEmbd, kExperts,
                kLayers);
    std::printf("peak %.1f GB/s at clocks.max.memory (%.0f MHz), %.1f GB/s loaded (8,751 MHz); "
                "%.2f B a value in+out\n", peak_max, clk / 1e3, peak_loaded, kBytesPerValue);

    // one buffer holds the largest case's activations; every case reads the same addresses
    const int max_tok = *std::max_element(cols.begin(), cols.end());
    std::vector<float> x((size_t) max_tok * kEmbd);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    for (int t = 0; t < max_tok; ++t) {
        const float m = mag(rng);   // a row-varying magnitude, as a real activation row has
        for (int i = 0; i < kEmbd; ++i) x[(size_t) t * kEmbd + i] = nd(rng) * m;
    }
    float* dx;
    void* dxq;
    ck(cudaMalloc(&dx, x.size() * sizeof(float)), "x");
    ck(cudaMalloc(&dxq, (size_t) max_tok * kEmbd / 32 * 36), "xq");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "x up");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");

    // the two entry points: 0 = quantize_q8_1_rows (iq_kernels.cu), 1 = native_quantize_q8_1 (native_mmvq.cu)
    auto run = [&](int which, int tokens) {
        if (which == 0)
            K::quantize_q8_1_rows(dx, tokens, kEmbd, dxq, stream);
        else
            K::native_quantize_q8_1(dx, dxq, kEmbd, tokens, stream);
    };
    const char* names[2] = {"quantize_q8_1_rows", "native_quantize_q8_1"};

    std::printf("\n%s\n", "  tokens   entry                   ns/row   ns/value   GB/s in+out   % of 288   % of 280   "
                          "us a layer   verdict");
    for (int which = 0; which < 2; ++which)
        for (int tokens : cols) {
            // native_quantize_q8_1 is the MMVQ path's entry and takes at most 8 columns (native_mmvq.cu:2642);
            // quantize_q8_1_rows has no such limit.  A verify window is at most 8 wide on this engine anyway.
            if (which == 1 && tokens > 8) continue;
            std::vector<double> v;
            for (int r = 0; r < 7; ++r) {
                cudaEventRecord(e0);
                const int reps = std::max(1, 200000 / tokens);
                for (int i = 0; i < reps; ++i) run(which, tokens);
                cudaEventRecord(e1);
                ck(cudaEventSynchronize(e1), "quant");
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                if (r) v.push_back((double) ms * 1e6 / ((double) reps * tokens));   // ns a row
            }
            const double ns_row = median(v);
            const double ns_val = ns_row / kEmbd;
            const double gbs = kBytesPerValue * kEmbd / ns_row;
            std::printf("  %6d   %-20s  %8.1f  %8.3f  %12.1f  %8.1f%%  %8.1f%%  %8.2f   %s\n", tokens,
                        names[which], ns_row, ns_val, gbs, 100.0 * gbs / bw, 100.0 * gbs / peak_loaded,
                        ns_row * kExperts / 1e3, 100.0 * gbs / bw >= 25.0 ? "pass" : "below 25%");
        }

    // the fused producer the plan says to compare against when this is launch-bound: SwiGLU and the quantize in one
    // launch, so the same bytes leave with one launch instead of two passes over the activation
    std::printf("\n  (native_quantize_q8_1 takes at most 8 columns - it is the MMVQ path's entry; a verify window is\n"
                "  at most 8 wide on this engine - so it has no 16-token row above.)\n");
    std::printf("\n  the fused form the plan names (swiglu + quantize, one launch):\n");
    float* dgate;
    float* dup;
    ck(cudaMalloc(&dgate, (size_t) max_tok * kEmbd * sizeof(float)), "gate");
    ck(cudaMalloc(&dup, (size_t) max_tok * kEmbd * sizeof(float)), "up");
    std::vector<float> g(x.size()), u(x.size());
    for (size_t i = 0; i < x.size(); ++i) { g[i] = x[i]; u[i] = x[(i + 7) % x.size()]; }
    ck(cudaMemcpy(dgate, g.data(), g.size() * sizeof(float), cudaMemcpyHostToDevice), "gate up");
    ck(cudaMemcpy(dup, u.data(), u.size() * sizeof(float), cudaMemcpyHostToDevice), "up up");
    std::printf("%s\n", "  tokens   entry                   ns/row   ns/value   GB/s in+out");
    for (int tokens : cols) {
        if (tokens > 8) continue;   // the MMVQ entries' column cap, as above
        std::vector<double> v;
        for (int r = 0; r < 7; ++r) {
            cudaEventRecord(e0);
            const int reps = std::max(1, 200000 / tokens);
            for (int i = 0; i < reps; ++i) K::native_swiglu_quantize_q8_1(dgate, dup, dxq, kEmbd, tokens, stream);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "swiglu quant");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) v.push_back((double) ms * 1e6 / ((double) reps * tokens));
        }
        const double ns_row = median(v);
        const double gbs = (8.0 + 4.125) * kEmbd / ns_row;   // two floats in, the same q8_1 out
        std::printf("  %6d   %-20s  %8.1f  %8.3f  %12.1f\n", tokens, "swiglu_quantize_q8_1", ns_row, ns_row / kEmbd,
                    gbs);
    }

    // what the launch costs at the smallest shape, so a launch-bound reading can be told from a bandwidth one
    std::printf("\n  empty-launch cost at the 1-token shape (the floor any launch-bound answer sits on):\n");
    {
        std::vector<double> v;
        for (int r = 0; r < 9; ++r) {
            cudaEventRecord(e0);
            for (int i = 0; i < 20000; ++i) K::quantize_q8_1_rows(dx, 1, kEmbd, dxq, stream);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "launch floor");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            v.push_back((double) ms * 1e6 / 20000);
        }
        std::printf("  quantize_q8_1_rows, 1 row: %.0f ns a launch (%.0f us for a layer's %d experts' worth of "
                    "tokens at 1 token a call)\n", median(v), median(v) * kExperts / 1e3, kExperts);
    }
    return 0;
}
