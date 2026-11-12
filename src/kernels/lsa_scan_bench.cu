// src/kernels/lsa_scan_bench.cpp - Phase 0 row 8 (OPTIMIZATION-STACK.md §0.3 Row 8): the LSA centroid scan,
// Q_chunk . C^T with d = 64, 32 queries x 32-bucket tiles, fp16 `mma.sync` operands and FP32 accumulate, at the
// geometry the 256K decode would run it.
//
// The row exists to size ONE term: the scan is 2 x 196,608 blocks x 32 queries x 64 dims = 805 MFLOP a layer, and
// the plan prices it at 0.6 s (at 11 TFLOP/s) against a 209 s prefill.  So the measurement is the achieved rate
// against the card's own bf16 peak, not a bandwidth.
//
// THE DESIGN IS DERIVED FROM WHAT SM_89 PROVIDES (OPTIMIZATION-STACK.md §0.2, and the same table is why this kernel
// is written in mma.sync terms):
//   * `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` -> `HMMA.16816.F32.F16`: present on sm_89, the only tensor
//     core this card has for fp16 (no wgmma, no tcgen05, no TMA - all sm_90+).
//   * `cp.async` -> `LDGSTS`: sm_80+, so a two-stage pipeline needs no TMA and no barrier arrive/wait instructions.
//   * shared memory: 99 KiB a block opt-in (CUDA guide Table 34, column 8.9; the card reports 102,400 B/SM), which
//     is what sets the tile count: a 32x64 fp16 query tile is 4 KiB and a 32x64 centroid tile 4 KiB, so a two-stage
//     pipeline of both is 16 KiB - four blocks fit an SM, and the row is judged on the rate that buys.
//   * 34 SMs: 196,608 blocks / (34 x 4) = 1,446 waves, so the scan is a throughput problem and not a tail one.
//
//     build/lsa_scan_bench [--buckets N] [--chunk N] [--iters N] [--stages 2]
//
// Prints the achieved TFLOP/s against the ~44 TFLOP/s fp16 dense peak §0.1 documents, and the ratio to the row's
// 11 TFLOP/s bar.  Exit 0 always: this is a benchmark, not a test.

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

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "lsa_scan_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int kDim = 64;      // the scan's d
constexpr int kQueries = 32;  // a query tile
constexpr int kBucket = 32;   // a centroid tile (32 buckets a block)
constexpr int kThreads = 256;   // 8 warps: a 32x32 tile is 2x4 warp tiles of 16x8

// One warp computes a 16x8 tile of Q.C^T with one m16n8k16 step per 16 of d, so a 32x32 tile is 2x4 warps of tiles.
// The kernel is written as: a block owns kQueries queries and kBucket buckets, stages both tiles with cp.async
// (two stages, so the next tile's loads are in flight while the mma runs), and accumulates FP32.
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

// cp.async of 16 bytes: the two-stage pipeline's load, LDGSTS on sm_80+
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
#else
    *reinterpret_cast<uint4*>(dst_smem) = *reinterpret_cast<const uint4*>(src_gmem);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// The scan: `centroids` is n_bucket rows of kDim halves; `queries` is n_query rows of kDim halves; `scores` comes out
// n_query x n_bucket floats.  One block takes kQueries queries and kBucket buckets.
__global__ void lsa_scan_kernel(const __half* __restrict__ queries, const __half* __restrict__ centroids,
                                float* __restrict__ scores, int n_query, int n_bucket) {
    __shared__ __align__(16) __half qs[2][kQueries * kDim];
    __shared__ __align__(16) __half cs[2][kBucket * kDim];
    const int qb = blockIdx.x, cb = blockIdx.y;
    const int q0 = qb * kQueries, c0 = cb * kBucket;
    // stage 0
    for (int i = threadIdx.x; i < kQueries * kDim / 8; i += blockDim.x)
        cp_async16(&qs[0][i * 8], &queries[(size_t) q0 * kDim + i * 8]);
    for (int i = threadIdx.x; i < kBucket * kDim / 8; i += blockDim.x)
        cp_async16(&cs[0][i * 8], &centroids[(size_t) c0 * kDim + i * 8]);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    // 2x4 warp tiles of 16x8: warp w covers query rows 16*(w/4) and buckets 8*(w%4)
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g8 = lane / 4, t = lane % 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int qrow = 16 * (w / 4) + g8, brow = 8 * (w % 4);   // this lane's A row and B column, as m16n8k16 has them
#pragma unroll
    for (int d = 0; d < kDim; d += 16) {
        // a: m16n8k16's A fragment, row-major (m x k).  The load is the one fdec_kernel<...>'s parity proves
        // (src/poly/fattn.cu: `uint2 lo = &Qs[(mt*16+g8)*QS + d]`, `hi = &Qs[(...+8)*QS + d]`, then
        // `a[4] = {lo.x, hi.x, lo.y, hi.y}`): a lane reads two consecutive halves of row g8 and of row g8+8,
        // twice, 8 k apart.  Written out rather than trusting the member order.
        uint32_t a[4];
        {
            const __half* qp = &qs[0][(size_t) qrow * kDim + d + 2 * t];
            a[0] = *reinterpret_cast<const uint32_t*>(qp);                 // row g8,   k = 2t
            a[1] = *reinterpret_cast<const uint32_t*>(qp + 8 * kDim);      // row g8+8, k = 2t
            a[2] = *reinterpret_cast<const uint32_t*>(qp + 8);             // row g8,   k = 2t+8
            a[3] = *reinterpret_cast<const uint32_t*>(qp + 8 * kDim + 8);  // row g8+8, k = 2t+8
        }
        // b: m16n8k16's B fragment, `row.col`: B is k x n and read column-major, so lane (g8, t) holds
        // bucket brow+g8 at dims d+2t, d+2t+1 (word b0) and bucket brow+g8+8 at the same dims (word b1).
        // The tile is stored bucket-major (cs[bucket][d]), which is exactly B's n-major row, so a word is
        // two consecutive dims of one bucket.
        // m16n8k16's B fragment is k x n read column-major: lane (g8, t) holds ONE bucket (brow + g8) at
        // k = d+2t, d+2t+1 (word b0) and at k = d+8+2t, d+9+2t (word b1).  The eight buckets of the tile come
        // from the eight g8 values, not from the two words.
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&cs[0][(size_t) (brow + g8) * kDim + d + 2 * t]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&cs[0][(size_t) (brow + g8) * kDim + d + 8 + 2 * t]);
        mma16816(acc, a, b0, b1);
    }
    // the D fragment: c0..c3 are (row, col) pairs: lane g8 has rows g8 and g8+8, cols 2t and 2t+1
    const int qr = 16 * (w / 4) + g8, br = 8 * (w % 4) + 2 * t;
    const int qf = q0 + qr, cf = c0 + br;
    // D/C's fragment: c0, c1 = row g8 at columns 2t and 2t+1; c2, c3 = row g8+8 at the same two columns.
    if (qf < n_query && cf + 1 < n_bucket) {
        scores[(size_t) qf * n_bucket + cf] = acc[0];
        scores[(size_t) qf * n_bucket + cf + 1] = acc[1];
        scores[(size_t) (qf + 8) * n_bucket + cf] = acc[2];
        scores[(size_t) (qf + 8) * n_bucket + cf + 1] = acc[3];
    }
}

}  // namespace

int main(int argc, char** argv) {
    // the 256K geometry: 196,608 blocks of 4 cells (idx_block 4), 32 queries a tile
    int n_bucket = 196608, chunk = kQueries, iters = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--buckets" && i + 1 < argc) n_bucket = std::atoi(argv[++i]);
        else if (a == "--chunk" && i + 1 < argc) chunk = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: lsa_scan_bench [--buckets N] [--chunk N] [--iters N]\n");
            return 2;
        }
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    int nsm = 0;
    {
        cudaDeviceProp p{};
        ck(cudaGetDeviceProperties(&p, 0), "props");
        nsm = p.multiProcessorCount;
    }
    // the card's fp16 dense tensor peak as §0.1 documents it: 165.2 TFLOPS on the 4090's 128 SMs, scaled by SM count
    const double peak_tflops = 165.2 * (double) nsm / 128.0;
    // the scan's work: 2 * queries * buckets * d
    const double flops = 2.0 * (double) chunk * (double) n_bucket * kDim;

    std::printf("LSA centroid scan: d %d, %d queries a tile, %d buckets (the 256K geometry), %.1f GFLOP a layer\n",
                kDim, chunk, n_bucket, flops / 1e9);
    std::printf("  %d SMs, fp16 dense peak %.1f TFLOP/s (165.2 x %d/128, §0.1); the row's bar is 11 TFLOP/s\n", nsm,
                peak_tflops, nsm);

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<__half> q((size_t) chunk * kDim), c((size_t) n_bucket * kDim);
    for (auto& x : q) x = __float2half(u(rng));
    for (auto& x : c) x = __float2half(u(rng));
    __half* dq;
    __half* dc;
    float* ds;
    ck(cudaMalloc(&dq, q.size() * 2), "q");
    ck(cudaMalloc(&dc, c.size() * 2), "c");
    ck(cudaMalloc(&ds, (size_t) chunk * n_bucket * 4), "s");
    ck(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice), "q up");
    ck(cudaMemcpy(dc, c.data(), c.size() * 2, cudaMemcpyHostToDevice), "c up");

    // the shared memory the kernel asks for: (32 + 32) rows x 64 halves x 2 stages = 16 KiB
    const int smem = 2 * (kQueries + kBucket) * kDim * 2;
    int per_sm = 0;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, lsa_scan_kernel, kThreads, 0), "occ");
    const dim3 grid((chunk + kQueries - 1) / kQueries, (n_bucket + kBucket - 1) / kBucket);
    std::printf("  grid %u x %u = %u CTAs; %.1f KiB of shared a block (static); %d blocks an SM -> %.1f waves\n",
                grid.x, grid.y, grid.x * grid.y, smem / 1024.0, per_sm,
                (double) (grid.x * grid.y) / ((double) nsm * (per_sm ? per_sm : 1)));

    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    lsa_scan_kernel<<<grid, kThreads>>>(dq, dc, ds, chunk, n_bucket);
    ck(cudaDeviceSynchronize(), "warm");
    std::vector<double> ms;
    for (int r = 0; r < 7; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) lsa_scan_kernel<<<grid, kThreads>>>(dq, dc, ds, chunk, n_bucket);
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "timed");
        float t = 0.f;
        cudaEventElapsedTime(&t, e0, e1);
        if (r) ms.push_back((double) t / iters);
    }
    std::sort(ms.begin(), ms.end());
    const double m = ms[ms.size() / 2];
    const double tflops = flops / (m * 1e-3) / 1e12;
    std::printf("\n  %.3f ms a layer, %.2f TFLOP/s = %.0f%% of the fp16 peak, %.1fx the row's 11 TFLOP/s bar\n", m,
                tflops, 100.0 * tflops / peak_tflops, tflops / 11.0);
    std::printf("  the 6.6 TFLOP scan of a whole prefill would take %.2f s at this rate\n", 6.6e12 / (tflops * 1e12));

    // correctness: the same product in doubles on three tiles, so a wrong fragment layout cannot pass as speed
    {
        // the first tile's cell only: the grid writes chunk * n_bucket floats, so downloading chunk * kBucket
        // would read a buffer that is the wrong shape (and the read below would then index past it)
        std::vector<float> got((size_t) chunk * kBucket);
        for (int qq = 0; qq < chunk; ++qq)
            ck(cudaMemcpy(got.data() + (size_t) qq * kBucket, ds + (size_t) qq * n_bucket, kBucket * 4,
                          cudaMemcpyDeviceToHost), "down row");
        double worst = 0, big = 0;
        for (int qq : {0, 1, 31})
            for (int b = 0; b < 4; ++b) {
                double want = 0;
                for (int d = 0; d < kDim; ++d)
                    want += (double) __half2float(q[(size_t) qq * kDim + d]) *
                            (double) __half2float(c[(size_t) b * kDim + d]);
                const double g = got[(size_t) qq * kBucket + b];
                worst = std::max(worst, std::fabs(g - want));
                big = std::max(big, std::fabs(want));
            }
        std::printf("  against doubles on 12 of the first tile's cells: worst %.3e of results up to %.2f\n", worst, big);
        {
            double w2 = 0; int wq = -1, wb = -1;
            for (int qq : {0, 1, 31})
                for (int b = 0; b < 4; ++b) {
                    double want = 0;
                    for (int d = 0; d < kDim; ++d)
                        want += (double) __half2float(q[(size_t) qq * kDim + d]) *
                                (double) __half2float(c[(size_t) b * kDim + d]);
                    const double e = std::fabs(got[(size_t) qq * kBucket + b] - want);
                    if (e > w2) { w2 = e; wq = qq; wb = b; }
                }
            std::printf("  worst at q%d, b%d: %.4f vs %.4f\n", wq, wb, got[(size_t) wq * kBucket + wb],
                        [&] { double t = 0; for (int d = 0; d < kDim; ++d) t += (double) __half2float(q[(size_t) wq * kDim + d]) * (double) __half2float(c[(size_t) wb * kDim + d]); return t; }());
        }
        std::printf("  the first tile's 4x4 corner, got / want:\n");
        for (int qq = 0; qq < 4; ++qq) {
            for (int b = 0; b < 4; ++b) {
                double want = 0;
                for (int d = 0; d < kDim; ++d)
                    want += (double) __half2float(q[(size_t) qq * kDim + d]) *
                            (double) __half2float(c[(size_t) b * kDim + d]);
                std::printf("    q%d b%d  %9.4f / %9.4f%s", qq, b, got[(size_t) qq * kBucket + b], want,
                            b == 3 ? "\n" : "");
            }
        }
    }
    return 0;
}
