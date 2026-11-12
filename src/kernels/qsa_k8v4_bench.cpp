// src/kernels/qsa_k8v4_bench.cpp - Phase 0 row 2 on the path the Qwen3.8-Flash packs actually run: strata's QSA
// decode attention over a k8v4 cache (docs/../OPTIMIZATION-STACK.md §0.3 Row 2).
//
// The poly engine's `fdec_k8v4` (src/poly/fattn.cu) is NOT this path.  The REAP-320 pack and the standard
// UD-IQ3_XXS pack are one architecture (`qwen4exp`) served by THIS engine, so their KV cache is read by
// qsa_decode_attn_batch (src/kernels/cuda/qsa_decode_attn.cu), group 12, over the cells a selection names -
// not by a dense split-K walk.  This bench times that kernel over a k8v4 pool and reports it in the same units as
// poly_fattn_test: ns a cell, GB/s, and the fraction of the card's DRAM peak.
//
//     build/qsa_k8v4_bench [--cells N] [--cells 4096,32768,...] [--itr N] [--streams N]
//
// The geometry is the compiled one (core/geometry.hpp): 24 query heads, 2 KV heads, head_dim 256, group 12.  The
// pools are a fully resident identity page table, as a short-context run has them; the selection is a synthetic
// ids list of the width the engine uses (`qsa_selection_width`), because the top-k itself is a separate stage and
// is not what this row measures.
//
// Bytes a cell: K int8 codes + one fp16 scale per KV_Q8_GROUP (64) values, V q4_0 blocks of 18 B per 32 values:
//   2 heads x (256 + 256/64*2) = 528 B of K, 2 heads x 256/32*18 = 288 B of V -> 816 B a cell.
// (poly's is 832: it keeps one K scale per 32 values.  The 16 B difference is the only material format
// difference between the two engines' k8v4 and is accounted for in MB-ROW-1-5.md §6.)
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

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

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_k8v4_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
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
    std::vector<int> cells_list = {4096, 32768, 131072, 262144};
    int iters = 30;
    for (int i = 1; i < argc; ++i) {
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
        } else {
            std::fprintf(stderr, "usage: qsa_k8v4_bench [--cells 4096,32768,...] [--itr N]\n");
            return 2;
        }
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t group = NH / NKV;
    // the compiled group is what this kernel is built for; say so rather than letting a silent mismatch pass
    if (group != 12) {
        std::printf("qsa_k8v4_bench: this build's compiled geometry is %lld heads / %lld KV = group %lld, not the\n"
                    "  Flash-Next 12 this bench reports; the numbers below are for the compiled shape.\n",
                    (long long) NH, (long long) NKV, (long long) group);
    }
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);
    const double peak = 2.0 * (double) clk * 1e3 * (bus / 8) / 1e9;
    const int64_t cell_bytes = NKV * (HD + (HD / k::KV_Q8_GROUP) * 2) + NKV * k::kv_q4_bytes_per_head((int) HD);
    std::printf("strata QSA decode attention, k8v4 pools: %lld query heads, %lld KV, head_dim %lld, group %lld\n",
                (long long) NH, (long long) NKV, (long long) HD, (long long) group);
    std::printf("  %lld B a cell (K int8 + fp16 scale per %d, V q4_0 18 B per 32); peak %.1f GB/s at %.0f MHz\n",
                (long long) cell_bytes, k::KV_Q8_GROUP, peak, clk / 1e3);

    const int max_cells = *std::max_element(cells_list.begin(), cells_list.end());
    const int64_t pages = (max_cells + PS - 1) / PS;
    const int64_t prows = pages * NKV * PS;
    std::mt19937 rng(20261012u);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- k8v4 pools, every page resident with an identity page table
    std::vector<int8_t> kc((size_t) (prows * HD));
    for (auto& x : kc) x = (int8_t) ((int) (rng() % 255u) - 127);
    int8_t* k_q = upload(kc);
    kc.clear();
    kc.shrink_to_fit();
    std::vector<uint16_t> ks((size_t) (prows * (HD / k::KV_Q8_GROUP)));
    for (auto& x : ks) x = (uint16_t) (0x2000 + (rng() % 0x100u));   // a finite fp16 scale, as a real cache has
    uint16_t* k_scale = upload(ks);
    ks.clear();
    ks.shrink_to_fit();
    std::vector<uint8_t> vq((size_t) (prows * k::kv_q4_bytes_per_head((int) HD)));
    for (size_t i = 0; i < vq.size(); i += sizeof(k::block_q4_0)) {
        const uint16_t d = (uint16_t) (0x2000 + (rng() % 0x100u));
        std::memcpy(&vq[i], &d, sizeof d);   // block_q4_0: fp16 scale then 16 nibble bytes
        for (size_t j = 2; j < sizeof(k::block_q4_0); ++j) vq[i + j] = (uint8_t) (rng() & 0xFFu);
    }
    uint8_t* v_q4 = upload(vq);
    vq.clear();
    vq.shrink_to_fit();
    std::vector<int32_t> tab((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
    int32_t* table = upload(tab);
    k::QsaAttnPools pools;
    pools.page_table = table;
    pools.k_q = k_q;
    pools.k_scale = k_scale;
    pools.v_q4 = v_q4;   // kv_mode 3: int8 keys with rotated q4_0 values, which IS k8v4

    // ---- one query, its selection, and the step record
    std::vector<float> hq((size_t) (NH * HD));
    for (float& x : hq) x = nd(rng);
    float* q = upload(hq);
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);   // the widest a selection can be
    float* attn = dalloc<float>((size_t) (NH * HD));
    const int64_t scr = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>((size_t) scr);
    std::vector<int32_t> hids((size_t) cap);
    // a selection of the engine's width: consecutive cells from the start (the top-k's own order is not what this
    // row measures, and the kernel's cost depends on the count, not the order)
    for (int64_t i = 0; i < cap; ++i) hids[(size_t) i] = (int32_t) i;
    int32_t* ids = upload(hids);
    std::vector<int32_t> hstep(k::kStepCount);
    int32_t* step = dalloc<int32_t>(k::kStepCount);

    std::printf("\n  selection width %lld cells (qsa_selection_width(32768)); CTAs = ceil(width/CHUNK) x %lld KV + %lld "
                "merge\n", (long long) cap, (long long) NKV, (long long) NH);
    std::printf("%s\n", "   cells   width   ms        ns/cell   GB/s   % of 288   % of 280   CTAs   CTAs/SM   waves");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    const int CHUNK = 64;   // qsa_decode_attn.cu: cells per block
    for (int cells : cells_list) {
        const int64_t n_kv = cells, width = k::qsa_selection_width(n_kv, s);
        hstep[k::kStepPos] = (int32_t) (n_kv - 1);
        hstep[k::kStepNKv] = (int32_t) n_kv;
        hstep[k::kStepNBid] = (int32_t) (n_kv / s.idx_block);
        hstep[k::kStepWidth] = (int32_t) width;
        ck(cudaMemcpy(step, hstep.data(), hstep.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "step");
        // the ids must name cells the pool holds: re-upload the selection of this width
        for (int64_t i = 0; i < width; ++i) hids[(size_t) i] = (int32_t) ((n_kv - width + i + n_kv) % n_kv);
        ck(cudaMemcpy(ids, hids.data(), (size_t) width * sizeof(int32_t), cudaMemcpyHostToDevice), "ids");
        k::qsa_decode_attn_batch(q, pools, ids, step, cap, s, scratch, attn, 1, st);
        ck(cudaDeviceSynchronize(), "warm");
        std::vector<double> v;
        for (int r = 0; r < 7; ++r) {
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) k::qsa_decode_attn_batch(q, pools, ids, step, cap, s, scratch, attn, 1, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "timed");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) v.push_back((double) ms / iters);
        }
        const double ms = median(v);
        const double ns_cell = ms * 1e6 / (double) width;
        const double gbs = (double) cell_bytes * (double) width / (ms * 1e6);
        const int ctas = (int) ((width + CHUNK - 1) / CHUNK * NKV + NH);
        std::printf("  %7d  %6lld  %8.4f  %9.2f  %6.1f  %8.1f%%  %8.1f%%  %5d  %8.2f  %6.2f\n", cells,
                    (long long) width, ms, ns_cell, gbs, 100.0 * gbs / peak, 100.0 * gbs / 280.0, ctas,
                    (double) ctas / (double) prop.multiProcessorCount, (double) ctas / (double) prop.multiProcessorCount);
    }
    std::printf("\nnote: the selection is capped at qsa_selection_width = %lld cells, so 32K and above all read the\n"
                "  same %lld cells: the rows differ in the POOL's size (what the page table resolves), not in the work.\n",
                (long long) cap, (long long) cap);
    return 0;
}
