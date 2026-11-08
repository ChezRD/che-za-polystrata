// src/poly/gigachat.cpp - GigaChat Audio's decoder.  The reference for the math is llama.cpp's deepseek2 graph and
// its absorbed MLA (src/models/deepseek2.cpp).
#include "gigachat.hpp"

#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr float kPi = 3.14159265358979323846f;
constexpr size_t kKeyBytes = size_t(128) << 20;    // the most the MLA key buffer takes for a pass
constexpr int kMlaSplits = 34;                     // key ranges the fused MLA path splits a decode token into
constexpr uint32_t kTypeQ8_0 = 8, kTypeQ6K = 14;   // the file's own formats for the absorbed k_b and v_b

float get_mscale(float scale, float m) { return scale <= 1.f ? 1.f : 0.1f * m * std::log(scale) + 1.f; }

// ggml_rope_ext's arguments and ggml_rope_yarn_corr_dims(), as llama.cpp's deepseek2 graph gives them.  The
// attn_factor is the one llama.cpp passes to the rope, which the kernel multiplies by (1 + 0.1 log(1/freq_scale))
// again when the extension is on.
Rope make_rope(float base, int n_rot, float freq_scale, float attn_factor, int n_ctx_orig, float beta_fast,
               float beta_slow, bool yarn) {
    Rope r;
    r.theta_scale = std::pow(base, -2.0f / (float) n_rot);
    r.freq_scale = freq_scale;
    r.ext_factor = yarn ? 1.0f : 0.0f;
    r.attn_factor = attn_factor;
    if (yarn) {
        const auto corr = [&](float rot) {
            return (float) n_rot * std::log((float) n_ctx_orig / (rot * 2 * kPi)) / (2 * std::log(base));
        };
        r.corr0 = std::max(0.0f, std::floor(corr(beta_fast)));
        r.corr1 = std::min((float) (n_rot - 1), std::ceil(corr(beta_slow)));
    }
    return r;
}

}  // namespace

void GigaChatAudio::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

void GigaChatAudio::check_draft_block() {
    if (draft_ || block_layer_)
        throw std::runtime_error("GigaChat Audio's draft block is not one this engine runs; --no-draft loads the "
                                 "model without it");
}

bool GigaChatAudio::q8() const { return opt_.kv_format == KvFmt::kQ8_0; }

// One layer's rows: the latent and the rope key, fp16, or Q8_0's codes and a scale a 32-value block.
size_t GigaChatAudio::row_bytes() const {
    if (q8())
        return (size_t) kv_rank_ + (size_t) (kv_rank_ / 32) * sizeof(uint16_t) + (size_t) n_rot_ +
               (size_t) (n_rot_ / 32) * sizeof(uint16_t);
    return (size_t) (kv_rank_ + n_rot_) * sizeof(uint16_t);
}

void GigaChatAudio::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    if (draft_ || block_layer_)
        throw std::runtime_error("the model carries a draft block, which this engine does not run: --no-draft "
                                 "loads the decoder without it");

    n_head_ = shape_i("attention.head_count");
    n_rot_ = shape_i("rope.dimension_count");
    const MetaValue* kvr = find("attention.kv_lora_rank");
    const TensorInfo* kb = gg_->find(blk(0, "attn_k_b.weight"));
    const TensorInfo* vb = gg_->find(blk(0, "attn_v_b.weight"));
    const TensorInfo* kvb = gg_->find(blk(0, "attn_kv_b.weight"));
    const TensorInfo* q0 = gg_->find(blk(0, "attn_q.weight"));
    const TensorInfo* qb = gg_->find(blk(0, "attn_q_b.weight"));
    const TensorInfo* kva = gg_->find(blk(0, "attn_kv_a_mqa.weight"));
    const TensorInfo* wo = gg_->find(blk(0, "attn_output.weight"));

    kv_rank_ = kvr ? (int) kvr->u : kb && kb->shape.size() == 3 ? (int) kb->shape[1]
                                           : vb && vb->shape.size() == 3 ? (int) vb->shape[0]
                                           : kva ? (int) kva->shape[1] - n_rot_ : 0;
    const MetaValue* kl = find("attention.key_length_mla");
    mla_k_ = kl ? (int) kl->u
                : kb && kb->shape.size() == 3 ? (int) kb->shape[0] + n_rot_
                : q0 && q0->shape.size() == 2 ? (int) (q0->shape[1] / std::max(n_head_, 1))
                : qb && qb->shape.size() == 2 ? (int) (qb->shape[1] / std::max(n_head_, 1))
                                              : 0;
    const MetaValue* vl = find("attention.value_length_mla");
    mla_v_ = vl ? (int) vl->u : vb && vb->shape.size() == 3 ? (int) vb->shape[1] : 0;
    if (mla_v_ == 0 && wo && wo->shape.size() == 2 && n_head_ > 0) mla_v_ = (int) (wo->shape[0] / n_head_);
    if (mla_v_ == 0 && kvb && kvb->shape.size() == 2 && n_head_ > 0)
        mla_v_ = (int) (kvb->shape[1] / n_head_) - (mla_k_ - n_rot_);
    qk_nope_ = mla_k_ - n_rot_;

    const MetaValue* qr = find("attention.q_lora_rank");
    const MetaValue* ffl = find("feed_forward_length");
    q_rank_ = qr && qr->u > 0 ? (int) qr->u : 0;
    n_ff_ = ffl ? (int) ffl->u : 0;
    const MetaValue* ws = find("expert_weights_scale");
    const MetaValue* wn = find("expert_weights_norm");
    w_scale_ = ws ? (float) ws->f : 1.f;
    w_norm_ = wn ? wn->u != 0 : true;
    const MetaValue* gating = find("expert_gating_func");
    need(!gating || gating->u == 2, "the expert gate is not a sigmoid");
    need(shape_i("expert_shared_count") == 1, "expert_shared_count is not 1");

    need(n_head_ > 0 && n_head_ <= 256, "attention.head_count is not 1 to 256");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && mla_k_ > n_rot_, "bad rope.dimension_count or MLA key length");
    need(kv_rank_ > 0 && kv_rank_ % 32 == 0, "attention.kv_lora_rank is not a multiple of 32");
    need(mla_v_ > 0 && mla_v_ <= 512, "bad MLA value length");

    // llama.cpp's deepseek2 YaRN: the rope's attn_factor, and the softmax's mscale^2/sqrt(nope + n_rot).
    const MetaValue* kind = find("rope.scaling.type");
    const MetaValue* factor = find("rope.scaling.factor");
    const MetaValue* orig = find("rope.scaling.original_context_length");
    const MetaValue* fast = find("rope.scaling.yarn_beta_fast");
    const MetaValue* slow = find("rope.scaling.yarn_beta_slow");
    const MetaValue* log_mul = find("rope.scaling.yarn_log_multiplier");
    const bool yarn = kind && kind->s == "yarn" && factor && factor->f > 1.f;
    const float f = yarn ? (float) factor->f : 1.f;
    const float freq_scale = yarn ? 1.f / f : 1.f;
    // The converter writes 0.1 * mscale_all_dim here; llama.cpp divides it by 0.1 again.
    const float lm = log_mul && log_mul->f > 0 ? (float) (log_mul->f / 0.1) : 0.f;
    float attn_factor = 1.f;
    if (yarn) {
        // llama.cpp's llama-context: the factor it passes to the rope, then the one deepseek2 folds into the scale.
        if (lm != 0.f) {
            const float ms = lm != 1.f ? lm : 1.f;
            attn_factor = get_mscale(f, ms) / get_mscale(f, lm);
        } else {
            attn_factor = get_mscale(f, 1.f);
        }
        attn_factor *= 1.f / (1.f + 0.1f * std::log(f));
    }
    const float attn_org = attn_factor * (1.f + 0.1f * std::log(f));
    const float mscale = attn_org * (1.f + 0.1f * lm * std::log(f));
    scale_ = mscale * mscale / std::sqrt(float(mla_k_));
    rope_ = make_rope(shape_f("rope.freq_base"), n_rot_, freq_scale, attn_factor,
                      orig ? (int) orig->u : 4096, fast ? (float) fast->f : 32.f,
                      slow ? (float) slow->f : 1.f, yarn);

    clamp_exp_ = clamp_limits(find("swiglu_clamp_exp"), n_layer_);
    clamp_shexp_ = find("swiglu_clamp_shexp") ? clamp_limits(find("swiglu_clamp_shexp"), n_layer_) : clamp_exp_;
}

void GigaChatAudio::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    auto matf = [&](const char* n, int64_t a, int64_t b) {
        return upload(tensor(il, n), blk(il, n), a, b, kFloatMat);
    };
    auto has = [&](const char* n) { return tensor(il, n) != nullptr; };

    L.fixed = true;   // the launches read the position on the card when recorded; head_key() numbers their sizes
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);

    if (q_rank_ > 0 && has("attn_q_a.weight")) {
        B.q_a = mat("attn_q_a.weight", n_embd_, q_rank_);
        B.q_a_norm = f32("attn_q_a_norm.weight", q_rank_);
        B.q_b = mat("attn_q_b.weight", q_rank_, (int64_t) n_head_ * mla_k_);
    } else {
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * mla_k_);
    }
    B.kv_a = mat("attn_kv_a_mqa.weight", n_embd_, kv_rank_ + n_rot_);
    B.kv_a_norm = f32("attn_kv_a_norm.weight", kv_rank_);
    if (has("attn_k_b.weight")) {
        // A decode token reads the file's own bytes (Q8_0 k_b, Q6_K v_b); a prompt batch keeps the bf16 copy the
        // engine's grouped matrix product needs.
        B.k_b = mat("attn_k_b.weight", qk_nope_, (int64_t) kv_rank_ * n_head_);
        B.k_b_bf = matf("attn_k_b.weight", qk_nope_, (int64_t) kv_rank_ * n_head_);
        B.v_b = mat("attn_v_b.weight", kv_rank_, (int64_t) mla_v_ * n_head_);
        B.v_b_bf = matf("attn_v_b.weight", kv_rank_, (int64_t) mla_v_ * n_head_);
    } else if (has("attn_kv_b.weight")) {
        // The legacy fused matrix: per head the nope rows of W_UK then the v rows of W_UV, kv_rank values wide.
        B.wkv_b = matf("attn_kv_b.weight", kv_rank_, (int64_t) n_head_ * (qk_nope_ + mla_v_));
        B.fused_kv = true;
    } else {
        throw std::runtime_error("the GGUF has neither attn_k_b.weight and attn_v_b.weight nor attn_kv_b.weight");
    }
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * mla_v_, n_embd_);

    if (!has("ffn_gate_inp.weight")) {   // layer 0's dense FFN
        if (n_ff_ <= 0) throw std::runtime_error("a dense layer needs feed_forward_length");
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

void GigaChatAudio::alloc_caches() {
    if (q8() && (kv_rank_ % 32 != 0 || n_rot_ % 32 != 0))
        throw std::runtime_error("--kv q8_0 needs the MLA latent and rope key to be multiples of 32 values");
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (q8()) {
            const int lsc = kv_rank_ / 32, ksc = n_rot_ / 32;
            B.lat_codes = (int8_t*) kv_alloc((size_t) kv_rank_, 1);
            B.lat_scales = (uint16_t*) kv_alloc((size_t) lsc * sizeof(uint16_t), 1);
            B.kpe_codes = (int8_t*) kv_alloc((size_t) n_rot_, 1);
            B.kpe_scales = (uint16_t*) kv_alloc((size_t) ksc * sizeof(uint16_t), 1);
        } else {
            B.lat = (uint16_t*) kv_alloc((size_t) kv_rank_ * sizeof(uint16_t), 1);
            B.kpe = (uint16_t*) kv_alloc((size_t) n_rot_ * sizeof(uint16_t), 1);
        }
    }
}

// Per-token buffers hold rows_ tokens, row after row.
void GigaChatAudio::assign_scratch() {
    const size_t n = (size_t) rows_;
    keys_max_ = opt_.max_context;
    const size_t key_bytes = (size_t) (kv_rank_ + n_rot_) * sizeof(float);
    if ((size_t) keys_max_ * key_bytes > kKeyBytes)
        throw std::runtime_error("gigachat: --max-context " + std::to_string(keys_max_) +
                                 " needs more than the " + std::to_string(kKeyBytes >> 20) +
                                 " MiB the MLA key buffer takes");
    mla_chunk_ = (int) std::clamp<size_t>(kKeyBytes / (key_bytes * (size_t) keys_max_), 1, n);

    t1_ = q_rank_ > 0 ? fbuf(n * (size_t) q_rank_) : nullptr;
    qr_ = q_rank_ > 0 ? fbuf(n * (size_t) q_rank_) : nullptr;
    q_ = fbuf(n * n_head_ * (size_t) mla_k_);
    qn_ = fbuf(n * n_head_ * (size_t) qk_nope_);
    qpe_ = fbuf(n * n_head_ * (size_t) n_rot_);
    qabs_ = fbuf(n * n_head_ * (size_t) kv_rank_);
    q576_ = fbuf(n * n_head_ * (size_t) (kv_rank_ + n_rot_));
    kv_ = fbuf(n * (size_t) (kv_rank_ + n_rot_));
    latv_ = fbuf(n * (size_t) kv_rank_);
    kpev_ = fbuf(n * (size_t) n_rot_);
    kqv_ = fbuf(n * n_head_ * (size_t) kv_rank_);
    vout_ = fbuf(n * n_head_ * (size_t) mla_v_);
    keys_ = fbuf((size_t) mla_chunk_ * (size_t) keys_max_ * (size_t) (kv_rank_ + n_rot_));
    score_ = fbuf((size_t) mla_chunk_ * (size_t) keys_max_ * (size_t) n_head_);
    // The fused path: fixed partitions for a token's keys, whatever the context, so a recorded launch keeps its
    // grid.  Eight rows is the most a decode window takes through the kernels.
    if (opt_.mla_fused) {
        mla_part_ = fbuf((size_t) kMlaSplits * 8 * n_head_ * (size_t) kv_rank_);
        mla_stats_ = fbuf((size_t) kMlaSplits * 8 * n_head_ * 2);
    }
}

int GigaChatAudio::widest() const {
    return std::max({n_head_ * mla_k_, n_head_ * (kv_rank_ + n_rot_), n_head_ * mla_v_, q_rank_, 1});
}

int GigaChatAudio::head_key(int il, int n) const { (void) il; return room_key(n); }

void GigaChatAudio::attention(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    const int sized = at ? room(n) : 0;

    if (B.q_a.p) {
        lin(B.q_a, xn_, t1_, n, xq_);
        rms_norm(t1_, fp(B.q_a_norm), qr_, q_rank_, n, rms_eps_, st_);
        lin(B.q_b, qr_, q_, n);
    } else {
        lin(B.wq, xn_, q_, n, xq_);
    }
    // The last n_rot values of every head turn; the first qk_nope do not.
    dsv4_rope(q_, q_, mla_k_, n_head_, qk_nope_, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_, at);
    mla_split_q(q_, qn_, qpe_, qk_nope_, n_rot_, n_head_, n, st_);
    if (B.fused_kv)
        mla_absorb_fused(qn_, B.wkv_b.p, B.wkv_b.type, qabs_, qk_nope_, kv_rank_, n_head_, n,
                         qk_nope_ + mla_v_, st_);
    else if (opt_.mla_fused && n == 1 && B.k_b.type == kTypeQ8_0)   // a decode token: a head a block, not an
        mla_head_gemv_q8(B.k_b.p, qn_, qabs_, n_head_, qk_nope_, kv_rank_, st_);   // output row a block
    else if (opt_.mla_fused && n == 1 && B.k_b.type == kTypeQ6K)
        mla_head_gemv_q6k(B.k_b.p, qn_, qabs_, n_head_, qk_nope_, kv_rank_, st_);
    else if (opt_.mla_fused && n == 1 && B.k_b.type == kTypeBF16)
        mla_head_gemv_bf16(B.k_b.p, qn_, qabs_, n_head_, qk_nope_, kv_rank_, st_);
    else
        lin(B.k_b_bf.p ? B.k_b_bf : B.k_b, qn_, qabs_, n, nullptr, kv_rank_);   // per head, bf16
    mla_pack_q(qabs_, qpe_, q576_, kv_rank_, n_rot_, n_head_, n, st_);
    lin(B.kv_a, xn_, kv_, n, xq_);
    mla_split_kv(kv_, latv_, kpev_, kv_rank_, n_rot_, n, st_);
    rms_norm(latv_, fp(B.kv_a_norm), latv_, kv_rank_, n, rms_eps_, st_);
    dsv4_rope(kpev_, kpev_, n_rot_, 1, 0, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_, at);

    const Kvq8 lq = q8() ? Kvq8{B.lat_codes, B.lat_scales} : Kvq8{};
    const Kvq8 kq = q8() ? Kvq8{B.kpe_codes, B.kpe_scales} : Kvq8{};
    if (q8()) {
        if (at) {
            store_q8_at(latv_, B.lat_codes, B.lat_scales, kv_rank_, at, n, 0, st_);
            store_q8_at(kpev_, B.kpe_codes, B.kpe_scales, n_rot_, at, n, 0, st_);
        } else {
            store_q8(latv_, B.lat_codes, B.lat_scales, kv_rank_, pos, n, st_);
            store_q8(kpev_, B.kpe_codes, B.kpe_scales, n_rot_, pos, n, st_);
        }
    } else if (at) {
        store_f16_at(latv_, B.lat, kv_rank_, at, n, 0, st_);
        store_f16_at(kpev_, B.kpe, n_rot_, at, n, 0, st_);
    } else {
        store_f16(latv_, B.lat, kv_rank_, pos, n, st_);
        store_f16(kpev_, B.kpe, n_rot_, pos, n, st_);
    }

    if (opt_.mla_fused && n <= 8 && kv_rank_ == 512 && n_rot_ == 64 && n_head_ == 32) {
        // A decode token (or a window's few): the cache read as it is, no float key buffer.  The splits scale with
        // the context the launch is sized for, so a recorded graph keeps its grid.
        const int cap = at ? sized : (int) (pos + n);
        // Every SM gets a range even at short context: one block walking the whole cache is latency-bound.  A
        // short cache wants fewer, larger ranges (the combine then reads fewer partials); the microbenchmark
        // picks these breakpoints (bench: poly_mla_decode_bench).
        const int splits = cap < 768 ? 8 : cap < 6144 ? 17 : kMlaSplits;
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr, "strata-poly: MLA decode fused: rows %d, cap %d, splits %d, no key copy\n", n, cap,
                         splits);
        }
        mla_attend_fused(q576_, B.lat, B.kpe, pos, n, kqv_, mla_part_, mla_stats_, kv_rank_, n_rot_, n_head_, scale_,
                         splits, cap, st_, at, 0, lq, kq);
    } else if (opt_.mla_fused) {
        // A prompt batch: the same q8 read, eight rows a launch (the partials' scratch holds eight).  It skips the
        // float key copy the generic path makes for every row.
        const int dim = kv_rank_ + n_rot_, out_head = n_head_ * kv_rank_;
        for (int f = 0; f < n; f += 8) {
            const int c = std::min(8, n - f);
            const int cap = (int) (pos + f + c);
            mla_attend_fused(q576_ + (size_t) f * n_head_ * dim, B.lat, B.kpe, pos, c,
                             kqv_ + (size_t) f * out_head, mla_part_, mla_stats_, kv_rank_, n_rot_, n_head_, scale_,
                             cap < 768 ? 8 : cap < 6144 ? 17 : kMlaSplits, cap, st_, nullptr, f, lq, kq);
        }
    } else {
        mla_attend(blas_, q576_, B.lat, B.kpe, pos, n, keys_, score_, mla_chunk_, kqv_, kv_rank_, n_rot_, n_head_,
                   scale_, st_, at, sized, lq, kq);
    }
    if (B.fused_kv)
        mla_vb_fused(kqv_, B.wkv_b.p, B.wkv_b.type, vout_, kv_rank_, mla_v_, n_head_, n, qk_nope_ + mla_v_, qk_nope_,
                     st_);
    else if (opt_.mla_fused && n == 1 && B.v_b.type == kTypeQ8_0)
        mla_head_gemv_q8(B.v_b.p, kqv_, vout_, n_head_, kv_rank_, mla_v_, st_);
    else if (opt_.mla_fused && n == 1 && B.v_b.type == kTypeQ6K)
        mla_head_gemv_q6k(B.v_b.p, kqv_, vout_, n_head_, kv_rank_, mla_v_, st_);
    else if (opt_.mla_fused && n == 1 && B.v_b.type == kTypeBF16)
        mla_head_gemv_bf16(B.v_b.p, kqv_, vout_, n_head_, kv_rank_, mla_v_, st_);
    else
        lin(B.v_b_bf.p ? B.v_b_bf : B.v_b, kqv_, vout_, n, nullptr, mla_v_);
    lin(B.wo, vout_, ao_, n);
}

void GigaChatAudio::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  A prompt batch's MoE (moe_rows) does
// its own routing.
void GigaChatAudio::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attention(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);

    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void GigaChatAudio::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void GigaChatAudio::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from
// the bare gates, made to sum to one and scaled by expert_weights_scale.
void GigaChatAudio::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool GigaChatAudio::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert.  It runs on the GPU while the host routes.
void GigaChatAudio::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

size_t GigaChatAudio::card_row_bytes() const {
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.lat || B.lat_codes) ++layers;
    return layers * row_bytes();
}

void GigaChatAudio::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t rank = (size_t) kv_rank_, rot = (size_t) n_rot_, lsc = rank / 32, ksc = rot / 32, rows = (size_t) n;
    for (const Block& B : blocks_) {
        if (!B.lat && !B.lat_codes) continue;
        if (q8()) {
            ck(cudaMemcpy(dst, B.lat_codes + (size_t) first * rank, rows * rank, cudaMemcpyDeviceToHost), "save rows");
            dst += rows * rank;
            ck(cudaMemcpy(dst, B.lat_scales + (size_t) first * lsc, rows * lsc * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost), "save rows");
            dst += rows * lsc * sizeof(uint16_t);
            ck(cudaMemcpy(dst, B.kpe_codes + (size_t) first * rot, rows * rot, cudaMemcpyDeviceToHost), "save rows");
            dst += rows * rot;
            ck(cudaMemcpy(dst, B.kpe_scales + (size_t) first * ksc, rows * ksc * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost), "save rows");
            dst += rows * ksc * sizeof(uint16_t);
        } else {
            ck(cudaMemcpy(dst, B.lat + (size_t) first * rank, rows * rank * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost), "save rows");
            dst += rows * rank * sizeof(uint16_t);
            ck(cudaMemcpy(dst, B.kpe + (size_t) first * rot, rows * rot * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost), "save rows");
            dst += rows * rot * sizeof(uint16_t);
        }
    }
}

void GigaChatAudio::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t rank = (size_t) kv_rank_, rot = (size_t) n_rot_, lsc = rank / 32, ksc = rot / 32;
    const size_t at = (size_t) first, skip = (size_t) (first - from), all = (size_t) count, rows = (size_t) n;
    for (const Block& B : blocks_) {
        if (!B.lat && !B.lat_codes) continue;
        if (q8()) {
            ck(cudaMemcpy(B.lat_codes + at * rank, src + skip * rank, rows * rank, cudaMemcpyHostToDevice), "load rows");
            src += all * rank;
            ck(cudaMemcpy(B.lat_scales + at * lsc, src + skip * lsc * sizeof(uint16_t), rows * lsc * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "load rows");
            src += all * lsc * sizeof(uint16_t);
            ck(cudaMemcpy(B.kpe_codes + at * rot, src + skip * rot, rows * rot, cudaMemcpyHostToDevice), "load rows");
            src += all * rot;
            ck(cudaMemcpy(B.kpe_scales + at * ksc, src + skip * ksc * sizeof(uint16_t), rows * ksc * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "load rows");
            src += all * ksc * sizeof(uint16_t);
        } else {
            ck(cudaMemcpy(B.lat + at * rank, src + skip * rank * sizeof(uint16_t), rows * rank * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "load rows");
            src += all * rank * sizeof(uint16_t);
            ck(cudaMemcpy(B.kpe + at * rot, src + skip * rot * sizeof(uint16_t), rows * rot * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "load rows");
            src += all * rot * sizeof(uint16_t);
        }
    }
}

}  // namespace strata::poly
