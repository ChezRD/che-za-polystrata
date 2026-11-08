// bench/micro/mla_decode_bench.cu - what one decode token's absorbed MLA costs, without a model: synthetic Q8_0
// latent (512) and rope (64) caches at a range of contexts, the same buffers the engine hands mla_attend().
// Times the whole old call and its four phases (key copy, QK GEMM, softmax, PV GEMM), and the fused kernel, and
// checks the fused output against the old path's.  Calls are timed in batches: one event pair a call lets the
// card's clock fall back between calls, which is not what a decode step sees.  Needs a card; run by hand.
//
// usage: poly_mla_decode_bench [ITERS] [ONLY_CONTEXT]
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kRank = 512, kRot = 64, kDim = kRank + kRot, kHead = 32;
constexpr int kThreads = 256;

std::mt19937 rng(7);

void fill_random(std::vector<float>& v, float s = 1.f) {
    std::normal_distribution<float> nd(0.f, s);
    for (float& x : v) x = nd(rng);
}

// ggml Q8_0 as store_q8() writes it.
void quantize_q8(const std::vector<float>& x, int width, int rows, std::vector<int8_t>& code,
                 std::vector<uint16_t>& scale) {
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < width / 32; ++b) {
            const float* v = x.data() + (size_t) r * width + (size_t) b * 32;
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(v[j]));
            const float d = amax / 127.f, id = d != 0 ? 1.f / d : 0.f;
            scale[(size_t) r * (width / 32) + b] = __half_as_ushort(__float2half(d));
            for (int j = 0; j < 32; ++j) code[(size_t) r * width + (size_t) b * 32 + j] = (int8_t) roundf(v[j] * id);
        }
}

template <class T>
T* to_dev(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

// The phases of mla_attend, repeated here so the microbenchmark can time them apart.  Same math, same shapes.
__global__ void phase_keys(const int8_t* __restrict__ lat_code, const uint16_t* __restrict__ lat_scale,
                           const int8_t* __restrict__ kpe_code, const uint16_t* __restrict__ kpe_scale,
                           float* __restrict__ keys, int n_keys) {
    const int j0 = blockIdx.x * 16;
    for (int i = threadIdx.x; i < 16 * kDim; i += blockDim.x) {
        const int j = j0 + i / kDim, r = i % kDim;
        if (j >= n_keys) continue;
        float v;
        if (r < kRank)
            v = float(lat_code[(size_t) j * kRank + r]) *
                __half2float(__ushort_as_half(lat_scale[(size_t) j * (kRank / 32) + r / 32]));
        else {
            const int q = r - kRank;
            v = float(kpe_code[(size_t) j * kRot + q]) *
                __half2float(__ushort_as_half(kpe_scale[(size_t) j * (kRot / 32) + q / 32]));
        }
        keys[(size_t) j * kDim + r] = v;
    }
}

__global__ void phase_softmax(float* __restrict__ score, int n_keys) {
    __shared__ float part[8];
    float* row = score + (size_t) blockIdx.y * gridDim.x * n_keys + (size_t) blockIdx.x * n_keys;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) mx = fmaxf(mx, row[j]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = part[0];
    for (int k = 1; k < (int) blockDim.x / 32; ++k) mx = fmaxf(mx, part[k]);
    __syncthreads();
    float sum = 0;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sum;
    __syncthreads();
    sum = 0;
    for (int k = 0; k < (int) blockDim.x / 32; ++k) sum += part[k];
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) row[j] *= inv;
}

double ms_of(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::max(1, std::atoi(argv[1])) : 20;
    std::vector<int> ctx = {1024, 8192, 16384, 32768, 45000};
    if (argc > 2) ctx.assign(1, std::atoi(argv[2]));
    const int force_splits = argc > 3 ? std::atoi(argv[3]) : 0;
    const float scale = 0.1447f;
    const size_t max_n = (size_t) *std::max_element(ctx.begin(), ctx.end());

    cublasHandle_t blas = nullptr;
    cublasCreate(&blas);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cublasSetStream(blas, stream);

    std::vector<float> lat_f((size_t) max_n * kRank), kpe_f((size_t) max_n * kRot);
    fill_random(lat_f);
    fill_random(kpe_f);
    // MLA_ROUND_K: the cache holds values an fp16 tile holds exactly, so only the query's rounding is measured.
    if (std::getenv("MLA_ROUND_K"))
        for (float& x : lat_f) x = __half2float(__float2half(x));
    if (std::getenv("MLA_ROUND_K"))
        for (float& x : kpe_f) x = __half2float(__float2half(x));
    std::vector<int8_t> latc((size_t) max_n * kRank), kpec((size_t) max_n * kRot);
    std::vector<uint16_t> lats((size_t) max_n * (kRank / 32)), kpes((size_t) max_n * (kRot / 32));
    quantize_q8(lat_f, kRank, (int) max_n, latc, lats);
    quantize_q8(kpe_f, kRot, (int) max_n, kpec, kpes);

    int8_t* d_latc = to_dev(latc);
    uint16_t* d_lats = to_dev(lats);
    int8_t* d_kpec = to_dev(kpec);
    uint16_t* d_kpes = to_dev(kpes);
    std::vector<float> qf((size_t) kHead * kDim);
    fill_random(qf, 1.3f);
    {   // a few large values, as the absorbed query has (absmax ~46 against an rms of 1.3)
        std::uniform_int_distribution<int> pick(0, (int) qf.size() - 1);
        for (int i = 0; i < 200; ++i) qf[(size_t) pick(rng)] *= 20.f;
    }
    // MLA_ROUND_Q: the query is already fp16, so only the cache's rounding is measured.
    if (std::getenv("MLA_ZERO_Q")) std::fill(qf.begin(), qf.end(), 0.f);
    if (std::getenv("MLA_ROUND_Q"))
        for (float& x : qf) x = __half2float(__float2half(x));
    float* d_q = to_dev(qf);
    float* d_out = nullptr;
    cudaMalloc(&d_out, (size_t) kHead * kRank * sizeof(float));
    float* d_out2 = nullptr;
    cudaMalloc(&d_out2, (size_t) kHead * kRank * sizeof(float));
    float* d_keys = nullptr;
    cudaMalloc(&d_keys, max_n * (size_t) kDim * sizeof(float));
    float* d_score = nullptr;
    cudaMalloc(&d_score, max_n * (size_t) kHead * sizeof(float));
    constexpr int kSplits = 34;
    float* d_part = nullptr;
    cudaMalloc(&d_part, (size_t) kSplits * kHead * kRank * sizeof(float));
    float* d_stats = nullptr;
    cudaMalloc(&d_stats, (size_t) kSplits * kHead * 2 * sizeof(float));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    // One event pair a batch of calls, and a batch long enough to keep the card out of its idle clock.
    const auto timeit = [&](auto&& fn) {
        double total = 0;
        int calls = 0;
        while (total < 250.0 && calls < 200000) {
            cudaEventRecord(e0, stream);
            for (int i = 0; i < 20; ++i) fn();
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            total += ms_of(e0, e1);
            calls += 20;
        }
        return total / calls;
    };

    std::printf("%8s %10s %10s %9s %9s %9s %9s %9s %10s %8s\n", "context", "old ms", "fused ms", "old GB/s",
                "fusedGB/s", "copy ms", "QK ms", "sm ms", "PV ms", "max|d|", "rms|d|");
    for (int n : ctx) {
        const int64_t pos = n - 1;
        const int splits = force_splits > 0 ? force_splits : (n < 768 ? 8 : n < 6144 ? 17 : kSplits);
        const float one = 1.f, zero = 0.f;
        const long long key_n = (long long) n * kDim, score_n = (long long) n * kHead;
        const auto old_call = [&] {
            mla_attend(blas, d_q, nullptr, nullptr, pos, 1, d_keys, d_score, 1, d_out, kRank, kRot, kHead, scale,
                       stream, nullptr, 0, {d_latc, d_lats}, {d_kpec, d_kpes});
        };
        const auto fused_call = [&] {
            mla_attend_fused(d_q, nullptr, nullptr, pos, 1, d_out2, d_part, d_stats, kRank, kRot, kHead, scale,
                             splits, n, stream, nullptr, 0, {d_latc, d_lats}, {d_kpec, d_kpes});
        };
        const double old_ms = timeit(old_call);
        const double fused_ms = timeit(fused_call);
        const double copy_ms = timeit([&] {
            phase_keys<<<dim3((n + 15) / 16, 1), kThreads, 0, stream>>>(d_latc, d_lats, d_kpec, d_kpes, d_keys, n);
        });
        const double qk_ms = timeit([&] {
            cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, kHead, kDim, &scale, d_keys, kDim, key_n,
                                      d_q, kDim, (long long) kHead * kDim, &zero, d_score, n, score_n, 1);
        });
        const double sm_ms = timeit([&] { phase_softmax<<<dim3(kHead, 1), kThreads, 0, stream>>>(d_score, n); });
        const double pv_ms = timeit([&] {
            cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, kRank, kHead, n, &one, d_keys, kDim, key_n,
                                      d_score, n, score_n, &zero, d_out, kRank, (long long) kHead * kRank, 1);
        });

        old_call();
        fused_call();
        cudaStreamSynchronize(stream);
        std::vector<float> ref((size_t) kHead * kRank), got((size_t) kHead * kRank);
        cudaMemcpy(ref.data(), d_out, ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(got.data(), d_out2, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double worst = 0, peak = 0, d2 = 0, r2 = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            worst = std::max(worst, (double) std::fabs(got[i] - ref[i]));
            peak = std::max(peak, (double) std::fabs(ref[i]));
            d2 += (double) (got[i] - ref[i]) * (got[i] - ref[i]);
            r2 += (double) ref[i] * ref[i];
        }
        const double rel_rms = std::sqrt(d2 / std::max(r2, 1e-30));
        const double kv_bytes = (double) n * (kRank + (kRank / 32) * 2 + kRot + (kRot / 32) * 2) + 73728. + 65536.;
        std::printf("%8d %10.4f %10.4f %9.1f %9.1f %9.4f %9.4f %9.4f %9.4f %10.2e %10.2e\n", n, old_ms,
                    fused_ms, kv_bytes / (old_ms * 1e6), kv_bytes / (fused_ms * 1e6), copy_ms, qk_ms, sm_ms, pv_ms,
                    worst / std::max(peak, 1e-6), rel_rms);
        std::fflush(stdout);
    }

    cudaFree(d_latc);
    cudaFree(d_lats);
    cudaFree(d_kpec);
    cudaFree(d_kpes);
    cudaFree(d_q);
    cudaFree(d_out);
    cudaFree(d_out2);
    cudaFree(d_keys);
    cudaFree(d_score);
    cudaFree(d_part);
    cudaFree(d_stats);
    cublasDestroy(blas);
    cudaStreamDestroy(stream);
    return 0;
}
