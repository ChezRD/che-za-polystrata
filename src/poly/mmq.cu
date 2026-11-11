// src/poly/mmq.cu - see the header.  The kernels are compiled from the llama.cpp checkout the build already takes
// ggml from; src/prefill/ggml_cuda_host.cu supplies the host symbols of ggml-cuda.cu they use.
#include "mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::poly {

#if STRATA_POLY_NVFP4
// mmq_nvfp4.cu
void nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s);
bool nvfp4_w4a8_fits(int64_t w_rows, int cc, size_t smpbo);
bool nvfp4_w4a8_stream_k(int64_t w_rows, int cc);
#endif

namespace {

// The types with kernels in this build: one template instance each in CMakeLists.txt.
#define STRATA_POLY_MMQ_TYPES(X)                                                                                \
    X(Q4_0) X(Q4_1) X(Q5_0) X(Q5_1) X(Q8_0) X(Q2_K) X(Q3_K) X(Q4_K) X(Q5_K) X(Q6_K)                             \
    X(IQ1_S) X(IQ2_XXS) X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S) X(IQ4_NL) X(IQ4_XS) X(MXFP4)

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// NVFP4's prompt products: on RTX 50, with STRATA_PREFILL_NVFP4=w4a4, FP4 x FP4 on its FP4 tensor cores (the
// instance built for 12.0a, as MXFP4's); otherwise, as on every other card, the weights decoded to int8 times Q8_1
// rows (mmq_nvfp4.cu).
bool nvfp4_fp4(int cc) {
    static const bool asked = [] {
        const char* v = std::getenv("STRATA_PREFILL_NVFP4");
        return v != nullptr && std::string(v) == "w4a4";
    }();
    return asked && cc == GGML_CUDA_CC_BLACKWELL;
}

// The float each row of NVFP4's FP4 rows carries (quantize_mmq_fp4_cuda's scale) goes behind the blocks and the
// tiles' tail, in the room q8_bytes() leaves: FP4 blocks take half the bytes of Q8_1's.
float* fp4_scales(void* xq, int64_t rows, int64_t cols) {
    return (float*) ((char*) xq + (size_t) rows * (size_t) (pad512(cols) / QK_FP4_MMQ) * sizeof(block_fp4_mmq) +
                     128 * sizeof(block_q8_1_mmq));
}

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

Mmq::Mmq() { ctx_ = new ggml_backend_cuda_context(ggml_cuda_get_device()); }

Mmq::~Mmq() { delete (ggml_backend_cuda_context*) ctx_; }

// mul_mat_q_case's own choice: a tile size whose configuration exists for this card and fits its shared memory.
// It aborts the process when there is none.
//
// The kernels step through a matrix MMQ_ITER_K columns at a time and read a whole step.  Where a row ends inside a
// step they read on into what lies behind it, and multiply that by the zeros quantize() pads the rows of x with.
// So what lies behind a matrix must be readable and must not be read as an infinite block scale: the engine keeps
// zero bytes there (kTail in runtime.hpp), as ggml-cuda does behind its tensors.
bool Mmq::fits(int type, int64_t w_rows, int64_t w_cols) {
    const ggml_type t = (ggml_type) type;
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    switch (t) {
#define X(T) case GGML_TYPE_##T:
        STRATA_POLY_MMQ_TYPES(X)
#undef X
        break;
#if STRATA_POLY_NVFP4
        case GGML_TYPE_NVFP4:
            if (w_cols % QK_NVFP4 != 0) return false;
            if (!nvfp4_fp4(dev.cc)) return nvfp4_w4a8_fits(w_rows, dev.cc, dev.smpbo);
            break;
#endif
        default: return false;
    }
    if (w_cols % ggml_blck_size(t) != 0) return false;
    const bool fallback = w_rows % 128 != 0;
    // MXFP4's instance is built for 12.0a on Blackwell (CMakeLists.txt): another Blackwell has no image of it
    if (t == GGML_TYPE_MXFP4 && dev.cc >= GGML_CUDA_CC_BLACKWELL && dev.cc != GGML_CUDA_CC_BLACKWELL) return false;
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fallback, dev.cc);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, dev.cc) <= dev.smpbo) return true;
    }
    return false;
}

// the kernels read whole tiles, so the last rows need room behind them
size_t Mmq::q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (pad512(cols) / QK8_1_MMQ) * sizeof(block_q8_1_mmq) + 128 * sizeof(block_q8_1_mmq);
}

bool Mmq::half_sums(int type) {
    const mmq_q8_1_ds_layout layout = mmq_get_q8_1_ds_layout((ggml_type) type);
    return layout == MMQ_Q8_1_DS_LAYOUT_DS4 || layout == MMQ_Q8_1_DS_LAYOUT_D2S6;
}

// launch_mul_mat_q's choice for each tile width: with stream-k the blocks share the tiles' sums out by the number
// of tiles, and a fixup adds the parts
bool Mmq::split_exact(int type, int64_t w_rows) {
    const ggml_type t = (ggml_type) type;
    const bool fallback = w_rows % 128 != 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
#if STRATA_POLY_NVFP4
    if (t == GGML_TYPE_NVFP4 && !nvfp4_fp4(cc)) return !nvfp4_w4a8_stream_k(w_rows, cc);
#endif
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fallback, cc);
        if (c.type != GGML_TYPE_COUNT && c.stream_k) return false;
    }
    return true;
}

void Mmq::quantize(const float* x, const int32_t* ids, void* xq, int type, int64_t cols, int64_t ld, int64_t rows,
                   void* stream) {
    if (rows <= 0) return;
    const ggml_type t = (ggml_type) type;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // On Blackwell llama.cpp's MXFP4 tiles take the rows as FP4 blocks, half the bytes of Q8_1's (mmq.cu's
    // use_native_fp4), and so do NVFP4's FP4 ones; the rows of another type, or another card, are Q8_1.
    const bool nv = t == GGML_TYPE_NVFP4 && nvfp4_fp4(cc);
    if ((t == GGML_TYPE_MXFP4 && blackwell_mma_available(cc)) || nv) {
        const bool aligned = (uintptr_t) x % 32 == 0 && ((size_t) ld * sizeof(float)) % 32 == 0;
        quantize_mmq_fp4_cuda(x, ids, xq, nv ? fp4_scales(xq, rows, cols) : nullptr, t, aligned, cols, ld, rows * ld,
                              rows * ld, pad512(cols), rows, 1, 1, (cudaStream_t) stream);
    } else {
        quantize_mmq_q8_1_cuda(x, ids, xq, t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                               (cudaStream_t) stream);
    }
    ck("mmq quantize");
}

void Mmq::mul(const void* w, int type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
              const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
              int64_t ld_dst, void* stream) {
    if (n <= 0 || max_rows <= 0) return;
    const ggml_type t = (ggml_type) type;
    // the matrices are the channels of x; the tokens' rows are the columns of y and of dst
    mmq_args a = {};
    a.x = (const char*) w;
    a.type_x = t;
    a.y = (const int*) xq;
    a.ids_dst = ids;
    a.expert_bounds = bounds;
    a.dst = dst;
    a.ncols_x = w_cols;
    a.nrows_x = w_rows;
    a.stride_row_x = w_cols / ggml_blck_size(t);
    a.ncols_y = total_rows;
    a.ncols_dst = total_rows;
    a.nrows_dst = ld_dst;
    a.nchannels_x = n;
    a.nchannels_y = n;
    a.stride_channel_x = (int64_t) (stride / ggml_type_size(t));
    a.nsamples_x = 1;
    a.nsamples_y = 1;
    a.ncols_max = max_rows;
    a.ncols_opt = max_rows;

    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
#define X(T) case GGML_TYPE_##T: mul_mat_q_case<GGML_TYPE_##T>(ctx, a, s); break;
        STRATA_POLY_MMQ_TYPES(X)
#undef X
#if STRATA_POLY_NVFP4
        case GGML_TYPE_NVFP4:
            if (nvfp4_fp4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                a.y_scale = fp4_scales(const_cast<void*>(xq), total_rows, w_cols);
                mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
            } else {
                nvfp4_w4a8(ctx, a, s);
            }
            break;
#endif
        default: throw std::runtime_error("mmq: tensor type " + std::to_string(type) + " is not built");
    }
    ck("mmq product");
}

}  // namespace strata::poly
