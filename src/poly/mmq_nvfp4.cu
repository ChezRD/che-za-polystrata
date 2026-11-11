// src/poly/mmq_nvfp4.cu - llama.cpp's NVFP4 tiles with 8-bit activations (W4A8), the same on every card.
//
// On sm_120 mmq.cuh multiplies NVFP4 as FP4 x FP4 (W4A4): the activations are rounded to E2M1 too.  Every other
// card gets its int8 path for NVFP4: weights decoded to int8, activations Q8_1, the precision decode runs at.  This
// unit compiles that path for sm_120 as well by hiding Blackwell's FP4 MMA from mmq.cuh on both sides, the device
// code and the host's tile config, which must agree on the tiles' shared memory (Strata's
// src/prefill/mmq_nvfp4_w4a8.cu).  The two non-static templates are renamed so the linker cannot fold them with
// the W4A4 instance built for 12xa (CMakeLists.txt).
#include "common.cuh"

#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#define mul_mat_q_switch_J strata_poly_w4a8_mul_mat_q_switch_J
#define mul_mat_q_case strata_poly_w4a8_mul_mat_q_case
#include "mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace strata::poly {

void nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    strata_poly_w4a8_mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}

// Mmq::fits() and Mmq::split_exact() for these tiles: the host's config as this unit sees the card
bool nvfp4_w4a8_fits(int64_t w_rows, int cc, size_t smpbo) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, w_rows % 128 != 0, cc);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo) return true;
    }
    return false;
}

bool nvfp4_w4a8_stream_k(int64_t w_rows, int cc) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, w_rows % 128 != 0, cc);
        if (c.type != GGML_TYPE_COUNT && c.stream_k) return true;
    }
    return false;
}

}  // namespace strata::poly
