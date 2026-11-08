// src/poly/gigachat.hpp - GigaChat Audio's decoder (llama.cpp architecture "deepseek2" or "gigachat_audio").
//
// 26 layers of absorbed multi-head latent attention (MLA) and routed experts beside a shared one; layer 0's FFN
// is dense.  The cache holds one 512-value latent and one 64-value rope key a token and layer, never per-head K
// or V: the key side of the latent up-projection is absorbed into the query, and the value side is applied to the
// attention's 512-value output before the output matrix.
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"

#include <vector>

namespace strata::poly {

class GigaChatAudio final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    size_t recurrent_floats() const override { return 0; }
    /// Audio rows arrive through the same `SVE1` path as pictures: one row per prompt token.
    bool sees() const override { return true; }
    size_t extra_bytes() const override { return 0; }
    void load_extra(const uint8_t*, int64_t) override {}
    void save_layer(int, int64_t, uint8_t*) override {}
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int il, int n) const override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        Dev q_a, q_a_norm, q_b;     ///< the low-rank query of a file that has one
        Dev wq;                     ///< else the direct query projection
        Dev kv_a, kv_a_norm, k_b, v_b;   ///< k_b and v_b as the file has them (Q8_0, Q6_K), for a decode token
        Dev k_b_bf, v_b_bf;              ///< the same in bf16, for a prompt batch's matrix product
        Dev wkv_b;                  ///< the legacy fused kv_b, when k_b and v_b are absent
        bool fused_kv = false;
        Dev wo;
        Dev ffn_gate, ffn_up, ffn_down;   ///< a dense layer's FFN
        std::vector<float> probs_b;       ///< what the routing adds to each expert's score before it picks
        Dev probs_b_dev;
        // The latent and the rope key, fp16, or Q8_0 in codes and scales.
        uint16_t* lat = nullptr;
        uint16_t* kpe = nullptr;
        int8_t* lat_codes = nullptr;
        int8_t* kpe_codes = nullptr;
        uint16_t* lat_scales = nullptr;
        uint16_t* kpe_scales = nullptr;
    };

    void attention(Block& B, int64_t pos, int n);
    void dense_ffn(Block& B, int il, int n);
    bool q8() const;
    size_t row_bytes() const;

    std::vector<Block> blocks_;   ///< one a layer of layers_

    int n_head_ = 0, n_rot_ = 0, qk_nope_ = 0, mla_k_ = 0, mla_v_ = 0, kv_rank_ = 0;
    int q_rank_ = 0;              ///< 0: the query is one projection
    int keys_max_ = 0;            ///< rows the keys buffer holds: the context
    int mla_chunk_ = 1;           ///< tokens one pass of the attention takes
    float scale_ = 0;             ///< the softmax's scale, YaRN's mscale included
    Rope rope_;                   ///< the turn of the queries' and keys' last n_rot values
    float w_scale_ = 1;
    bool w_norm_ = true;

    // device scratch
    float *t1_ = nullptr, *qr_ = nullptr;   ///< the low-rank query
    float *q_ = nullptr;                    ///< n_head * (qk_nope + n_rot) a token
    float *qn_ = nullptr, *qpe_ = nullptr;  ///< those per head, split
    float *qabs_ = nullptr;                 ///< n_head * kv_rank a token: the absorbed query
    float *q576_ = nullptr;                 ///< n_head * (kv_rank + n_rot): what mla_attend reads
    float *kv_ = nullptr;                   ///< kv_rank + n_rot a token from kv_a: the latent then the rope key
    float *latv_ = nullptr, *kpev_ = nullptr;
    float *keys_ = nullptr, *score_ = nullptr;
    float *mla_part_ = nullptr, *mla_stats_ = nullptr;   ///< the fused MLA path's split partials
    float *kqv_ = nullptr, *vout_ = nullptr;
};

}  // namespace strata::poly
