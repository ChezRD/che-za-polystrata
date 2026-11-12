// src/kernels/gdn_state_bench.cpp - Phase 0 row 3 (OPTIMIZATION-STACK.md §0.3 Row 3): one GDN layer's delta-rule
// state step at T = 1, the shape a decode token takes, in four arms:
//
//   A0 fp32 state     the engine's own arithmetic (32-bit state, read + rank-1 update + write)
//   A1 int8 state     the state as int8 codes with one fp32 scale a head-row, dequantized on read
//   A2 int8 + 4 compensators   A1 plus the rank-one factors LeapQuant keeps in fp16 and folds through the update
//   A3 int8 + a 16-token window buffer   A1 plus the per-window ring the paper's per-window quantization needs
//
// The row's question is physical, not numerical: the state is 48 heads x 128 x 128 x 4 B = 3.0 MiB, which fits in
// this card's 32 MiB L2, so shrinking it 4x cannot make the step faster unless the step is bound by the bytes.  A0
// measures that; A1..A3 measure what a smaller state would actually buy.  Pass: int8 >= 1.6x the fp32 arm.  Stop
// rule: < 1.3x closes the LeapQuant line.
//
// The arms are the MEASUREMENT, not an engine feature: each is one kernel here, the fp32 one the same arithmetic as
// src/prefill/kernels.cu's gdn_rec_kh_kernel (a thread the (head, column) pair, the decay and the rank-1 update),
// so the comparison is of the memory format and nothing else.
//
//     build/gdn_state_bench [--heads 48] [--dim 128] [--iters N] [--bw GB/s]
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
        std::fprintf(stderr, "gdn_state_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int kCompensators = 4;   // LeapQuant's rank-r factors kept in fp16 (state.py:35-51)
constexpr int kWindow = 16;        // the per-window quantization's buffer length (paper §3.1, L = 16)

// ---- A0: the fp32 state.  A thread owns one (head, column): it reads its column of the 128x128 state, applies the
// decay and the rank-1 update `s += k * u[col]`, and writes it back.  The same arithmetic gdn_rec_kh_kernel does,
// with the three value heads a key head kept apart as the engine has them.
__global__ void state_fp32_kernel(float* __restrict__ s, const float* __restrict__ k, const float* __restrict__ u,
                                  const float* __restrict__ decay, int heads, int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    float* row = s + ((size_t) head * dim) * dim;
    const float d = decay[head], kk = k[head * dim + col], uu = u[head * dim + col];
    // s[:, col] for this column: the engine's per-column update, one decay a head
    for (int r = 0; r < dim; ++r) {
        const float cur = row[(size_t) r * dim + col];
        row[(size_t) r * dim + col] = cur * d + kk * uu;
    }
}

// ---- A1: the state as int8 codes with one fp32 scale a (head, row).  Read the codes, dequantize, update, requantize.
// The bytes the step touches are 4x fewer; the ALU is more (a multiply-add a value on the way in and out).
__global__ void state_int8_kernel(int8_t* __restrict__ q, float* __restrict__ scale, const float* __restrict__ k,
                                  const float* __restrict__ u, const float* __restrict__ decay, int heads, int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    int8_t* row = q + ((size_t) head * dim) * dim;
    float* srow = scale + (size_t) head * dim;
    const float d = decay[head], kk = k[head * dim + col], uu = u[head * dim + col];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        const int8_t code = row[(size_t) r * dim + col];
        const float cur = (float) code * sc;
        const float upd = cur * d + kk * uu;
        // requantize: the row's scale stays, the code is the rounded quotient (clamped, as a real format would)
        float qv = nearbyintf(upd / sc);
        qv = fminf(fmaxf(qv, -127.f), 127.f);
        row[(size_t) r * dim + col] = (int8_t) qv;
    }
}

// ---- A2: A1 plus the compensators: kCompensators rank-one factors in fp16, subtracted before the update and added
// back after it (the paper's "fit a rank-r factor, quantize the residual").  The factors are read and written.
__global__ void state_int8_comp_kernel(int8_t* __restrict__ q, float* __restrict__ scale,
                                       __half* __restrict__ comp, const float* __restrict__ k,
                                       const float* __restrict__ u, const float* __restrict__ decay, int heads,
                                       int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    int8_t* row = q + ((size_t) head * dim) * dim;
    float* srow = scale + (size_t) head * dim;
    __half* crow = comp + ((size_t) head * kCompensators * 2) * dim;
    const float d = decay[head], kk = k[head * dim + col], uu = u[head * dim + col];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        float residual = 0.f;
        for (int c = 0; c < kCompensators; ++c)
            residual += __half2float(crow[(size_t) (2 * c) * dim + r]) *
                        __half2float(crow[(size_t) (2 * c + 1) * dim + col]);
        const float cur = (float) row[(size_t) r * dim + col] * sc + residual;
        const float upd = cur * d + kk * uu;
        const float back = upd - residual;   // the quantized part only
        float qv = nearbyintf(back / sc);
        qv = fminf(fmaxf(qv, -127.f), 127.f);
        row[(size_t) r * dim + col] = (int8_t) qv;
    }
}

// ---- A3: A1 plus the window buffer: a token's (k, u) pairs are appended to a ring of kWindow entries and the step
// reads them once a window, the paper's per-window form.  The ring is read and written here.
__global__ void state_int8_window_kernel(int8_t* __restrict__ q, float* __restrict__ scale, float* __restrict__ win,
                                         const float* __restrict__ k, const float* __restrict__ u,
                                         const float* __restrict__ decay, int heads, int dim, int64_t token) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (col >= dim) return;
    int8_t* row = q + ((size_t) head * dim) * dim;
    float* srow = scale + (size_t) head * dim;
    // the window buffer: two floats an entry (k and u of one token), cycled
    float* wrow = win + ((size_t) head * kWindow * 2) * dim;
    const int slot = (int) (token % kWindow);
    wrow[((size_t) 2 * slot) * dim + col] = k[head * dim + col];
    wrow[((size_t) (2 * slot + 1)) * dim + col] = u[head * dim + col];
    const float d = decay[head], kk = wrow[((size_t) 2 * slot) * dim + col], uu = wrow[((size_t) (2 * slot + 1)) * dim + col];
    for (int r = 0; r < dim; ++r) {
        const float sc = srow[r];
        const float cur = (float) row[(size_t) r * dim + col] * sc;
        const float upd = cur * d + kk * uu;
        float qv = nearbyintf(upd / sc);
        qv = fminf(fmaxf(qv, -127.f), 127.f);
        row[(size_t) r * dim + col] = (int8_t) qv;
    }
}

struct Arm {
    const char* name;
    double bytes_per_token;   // the state bytes the step touches
    double ms;
};

}  // namespace

int main(int argc, char** argv) {
    int heads = 48, dim = 128, iters = 200;
    double bw = 288.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--heads" && i + 1 < argc) heads = std::atoi(argv[++i]);
        else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--bw" && i + 1 < argc) bw = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: gdn_state_bench [--heads 48] [--dim 128] [--iters N] [--bw GB/s]\n");
            return 2;
        }
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);
    const double peak = 2.0 * (double) clk * 1e3 * (bus / 8) / 1e9;

    const size_t st = (size_t) heads * dim * dim;
    std::printf("GDN state step, T = 1: %d heads x %d x %d, fp32 state %.2f MiB (L2 is 32 MiB)\n", heads, dim, dim,
                st * 4.0 / (1 << 20));
    std::printf("  peak %.1f GB/s at %.0f MHz; the row's bar is 1.6x fp32, the stop rule is 1.3x\n", peak, clk / 1e3);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hk((size_t) heads * dim), hu((size_t) heads * dim), hd((size_t) heads), hs(st);
    for (auto& x : hk) x = nd(rng) * 0.02f;
    for (auto& x : hu) x = nd(rng) * 0.02f;
    for (auto& x : hd) x = 0.99f;
    for (auto& x : hs) x = nd(rng) * 0.05f;

    float *dk, *du, *ddec;
    ck(cudaMalloc(&dk, hk.size() * 4), "k");
    ck(cudaMalloc(&du, hu.size() * 4), "u");
    ck(cudaMalloc(&ddec, hd.size() * 4), "decay");
    ck(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k up");
    ck(cudaMemcpy(du, hu.data(), hu.size() * 4, cudaMemcpyHostToDevice), "u up");
    ck(cudaMemcpy(ddec, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice), "decay up");

    // the four states, each allocated once and reused (the row measures the step, not the allocation)
    float* s32;
    ck(cudaMalloc(&s32, st * 4), "s32");
    ck(cudaMemcpy(s32, hs.data(), st * 4, cudaMemcpyHostToDevice), "s32 up");
    int8_t* s8;
    float* sc8;
    ck(cudaMalloc(&s8, st), "s8");
    ck(cudaMalloc(&sc8, (size_t) heads * dim * 4), "sc8");
    {
        // initialize the int8 state from the fp32 one: one scale a (head, row), codes from the quotient
        std::vector<int8_t> q(st);
        std::vector<float> sc((size_t) heads * dim);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < dim; ++r) {
                float amax = 0.f;
                for (int c = 0; c < dim; ++c) amax = std::fmax(amax, std::fabs(hs[((size_t) h * dim + r) * dim + c]));
                const float s = amax > 0.f ? amax / 127.f : 1.f;
                sc[(size_t) h * dim + r] = s;
                for (int c = 0; c < dim; ++c) {
                    float qv = nearbyintf(hs[((size_t) h * dim + r) * dim + c] / s);
                    q[(size_t) h * dim * dim + (size_t) r * dim + c] = (int8_t) fminf(fmaxf(qv, -127.f), 127.f);
                }
            }
        ck(cudaMemcpy(s8, q.data(), st, cudaMemcpyHostToDevice), "s8 up");
        ck(cudaMemcpy(sc8, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "sc8 up");
    }
    __half* comp;
    const size_t comp_n = (size_t) heads * kCompensators * 2 * dim;
    ck(cudaMalloc(&comp, comp_n * 2), "comp");
    ck(cudaMemset(comp, 0, comp_n * 2), "comp zero");
    float* win;
    const size_t win_n = (size_t) heads * kWindow * 2 * dim;
    ck(cudaMalloc(&win, win_n * 4), "win");
    ck(cudaMemset(win, 0, win_n * 4), "win zero");

    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    const dim3 grid((unsigned) ((dim + 127) / 128), (unsigned) heads);
    auto time_arm = [&](auto&& launch) {
        launch(0);
        ck(cudaDeviceSynchronize(), "warm");
        std::vector<double> v;
        for (int r = 0; r < 7; ++r) {
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i) launch(i);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "timed");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r) v.push_back((double) ms / iters);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    std::vector<Arm> arms;
    arms.push_back({"A0 fp32 state", (double) st * 4.0 * 2.0,
                    time_arm([&](int) { state_fp32_kernel<<<grid, 128>>>(s32, dk, du, ddec, heads, dim); })});
    arms.push_back({"A1 int8 state", (double) st * 1.0 + (double) heads * dim * 4.0,
                    time_arm([&](int) { state_int8_kernel<<<grid, 128>>>(s8, sc8, dk, du, ddec, heads, dim); })});
    arms.push_back({"A2 int8 + 4 comp", (double) st + (double) comp_n * 2.0 + (double) heads * dim * 4.0,
                    time_arm([&](int) { state_int8_comp_kernel<<<grid, 128>>>(s8, sc8, comp, dk, du, ddec, heads, dim); })});
    arms.push_back({"A3 int8 + window", (double) st + (double) win_n * 4.0 + (double) heads * dim * 4.0,
                    time_arm([&](int i) { state_int8_window_kernel<<<grid, 128>>>(s8, sc8, win, dk, du, ddec, heads, dim, i); })});

    const double a0 = arms[0].ms;
    std::printf("\n  arm                  ms/layer   us/layer   us a token   GB/s touched   x fp32   %s\n", "verdict");
    for (const Arm& a : arms) {
        const double gbs = a.bytes_per_token / (a.ms * 1e6);
        const double rel = a0 / a.ms;
        const char* verdict = &a == &arms[0] ? "reference"
                              : rel >= 1.6 ? "PASS (>=1.6x)"
                              : rel >= 1.3 ? "below 1.6x" : "STOP (<1.3x)";
        std::printf("  %-18s %9.4f  %9.2f  %10.2f  %13.1f  %7.2fx   %s\n", a.name, a.ms, a.ms * 1e3, a.ms * 1e3, gbs,
                    rel, verdict);
    }
    std::printf("\n  the engine's own figure for the fp32 step at T=1 is 0.54 ms/token over 36 layers = 15 us a layer;\n"
                "  this bench's A0 should land near that (it is the same arithmetic on a warm L2-resident state).\n");
    return 0;
}
