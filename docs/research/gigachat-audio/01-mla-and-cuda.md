# GigaChat Audio decoder on one RTX 4060 Ti 16 GB

**Scope:** decoder only (26-layer DeepSeek-V3-style MLA/MoE), q3/q4 weights, q8/q4 KV, C++/CUDA engine. BF16 is used only as a parity/reference precision, never as the proposed runtime solution. vLLM and FP8 are explicitly out of scope.

---

## 0. Bottom line

1. **This checkpoint has no Q-LoRA branch.** `q_lora_rank=null`, so `q_a_proj`, `q_a_layernorm`, and `q_b_proj` do not exist. The decoder has a single `q_proj` of shape `[6144, 1536]`. Verified in the HF config and in llama.cpp’s “lite” path, which explicitly names `GigaChat3-10B-A1.8B` ([`deepseek2.cpp:413`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L413)).
2. **The correct absorbed MLA KV is 29.25 KiB/token in BF16**, because the 512-d latent is stored once and V aliases the first 512 dims of K. At 45k tokens that is **1.255 GiB**. q8_0 gives **15.54 KiB/token = 0.667 GiB**; q4_0 gives **8.23 KiB/token = 0.353 GiB**. llama.cpp confirms this layout: for MLA it stores no separate V cache (`const bool has_v = !is_mla;`) ([`llama-kv-cache.cpp:230`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp#L230)).
3. **Materialized MLA is not viable at 45k on 16 GB.** Per-head K/V costs 523.25 KiB/token BF16, i.e. **22.45 GiB at 45k**; q8 11.93 GiB; q4 6.32 GiB. Absorption is mandatory.
4. **W_UV is 192×512 per head, not 512×192.** All 512 latent columns contribute to all 192 output rows. There is no “first 192 rows” truncation. W_UK is 128×512 per head and is absorbed into Q at runtime.
5. **DSA cannot be enabled post-hoc.** It needs per-layer indexer weights and a model trained with sparse attention. A heuristic block indexer is viable only as an approximation, and must be validated against dense MLA.
6. **FlashMLA is unusable on sm_89.** The older release requires SM90/SM100; the current release requires SM100/SM103. It relies on WGMMA/TMA. FlashAttention-2 caps head dim at 256; 576 is unsupported. However, **a custom 576/512 MMA kernel is feasible on Ada**; llama.cpp already compiles one with `DKQ=576, DV=512`, `ncols=64`, 256 threads, 1 CTA/SM ([`fattn.cu:377`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu#L377), [`fattn-mma-f16.cuh:77`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh#L77)).
7. **Decode at 45k is memory-bound; prefill is compute-bound.** Realistic estimates: **50–110 tok/s decode** and **60–90 s dense prefill**, falling to **20–40 s** with 12.5–25% block-sparse attention. These assume 30–50% of the 288 GB/s peak for decode and 50–60% of the 44.1 FP16 TFLOPS peak for prefill.
8. **q8 KV is the safe default; q4 KV is a quality experiment.** MLA forces K and V to share the same quantized latent, so asymmetric “q8 K / q4 V” is not available without a second cache, which costs more memory than q8/q8 shared. q4_K is invalid for KV because its 256-element block does not divide 576.
9. **Do not stream/evict KV for 120-minute audio.** Full 45k KV at q8 is only 0.667 GiB. The model’s temporal grounding depends on periodic 60 s markers; the GigaChat Audio paper reports long-form mIoU collapses from 53.8 to 14.2 when temporal anchors are removed.
10. **Effort:** 15–22 engineer-weeks for dense q4/q8 MLA, +3–5 for block-sparse, +2–4 for audio encoder integration. The dominant unknown is the 576-d attention kernel’s achieved efficiency on Ada.

---

## 1. Model facts and tensor/shape table

### 1.1 Exact config

From [`config.json`](https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B/blob/main/config.json) and [`decoder/config.json`](https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B/blob/main/decoder/config.json):

| Field | Value |
|---|---|
| hidden_size | 1536 |
| layers | 26 |
| heads | 32 |
| q_lora_rank | **null** |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| qk_head_dim | 192 |
| v_head_dim | 192 |
| intermediate_size (dense layer 0) | 8960 |
| moe_intermediate_size | 1280 |
| routed experts / top-k | 64 / 4 |
| shared experts | 1 |
| first_k_dense_replace | 1 |
| n_group / topk_group | 1 / 1 |
| scoring_func | sigmoid |
| norm_topk_prob | true |
| routed_scaling_factor | 1 |
| vocab | 128256 |
| rope_theta | 100000 |
| rope_scaling | YaRN, factor 64, original 4096, beta_fast 32, beta_slow 1, mscale 1, mscale_all_dim 1 |
| rope_interleave | true |
| rms_norm_eps | 1e-6 |
| tie_word_embeddings | false |
| max_position_embeddings | 262144 |

The decoder checkpoint is a plain `DeepseekV3ForCausalLM`; the audio wrapper loads it from `decoder/` ([`modeling_gigachat_audio.py`](https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B/blob/main/modeling_gigachat_audio.py)).

### 1.2 Tensor table

Torch shapes are `[out_features, in_features]` for Linear weights. GGUF names are from [`gguf-py/gguf/constants.py`](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py).

| HF tensor | Torch shape | GGUF name | Parameters | Recommended quant |
|---|---:|---|---:|---|
| `self_attn.q_proj.weight` | `[6144,1536]` | `blk.N.attn_q.weight` | 9.44 M | Q4_K / Q5_K |
| `self_attn.kv_a_proj_with_mqa.weight` | `[576,1536]` | `blk.N.attn_kv_a_mqa.weight` | 0.885 M | Q4_K / Q5_K |
| `self_attn.kv_a_layernorm.weight` | `[512]` | `blk.N.attn_kv_a_norm.weight` | 512 | F32 |
| `self_attn.kv_b_proj.weight` | `[10240,512]` | split into `attn_k_b` + `attn_v_b` | 5.24 M | Q4_K / Q5_K |
| — k part | `[32,128,512]` | `blk.N.attn_k_b.weight` (ggml `{128,512,32}`) | 2.10 M | Q4_K / Q5_K |
| — v part | `[32,192,512]` | `blk.N.attn_v_b.weight` (ggml `{512,192,32}`) | 3.15 M | Q4_K / Q5_K |
| `self_attn.o_proj.weight` | `[1536,6144]` | `blk.N.attn_output.weight` | 9.44 M | Q4_K |
| `mlp.gate.weight` (MoE layers) | `[64,1536]` | `blk.N.ffn_gate_inp.weight` | 0.098 M | F32/F16 |
| `mlp.gate.e_score_correction_bias` | `[64]` | `blk.N.ffn_exp_probs_b.bias` | 64 | F32 |
| `mlp.experts.e.{gate,up}_proj.weight` | `[64,1280,1536]` each | `ffn_gate_exps`, `ffn_up_exps` | 125.8 M each | Q3_K/Q4_K |
| `mlp.experts.e.down_proj.weight` | `[64,1536,1280]` | `ffn_down_exps` | 125.8 M | Q3_K/Q4_K |
| `mlp.shared_experts.{gate,up}_proj.weight` | `[1280,1536]` | `ffn_gate_shexp`, `ffn_up_shexp` | 1.97 M each | Q4_K |
| `mlp.shared_experts.down_proj.weight` | `[1536,1280]` | `ffn_down_shexp` | 1.97 M | Q4_K |
| layer 0 `mlp.{gate,up}_proj.weight` | `[8960,1536]` | `blk.0.ffn_gate/up.weight` | 13.76 M each | Q4_K/Q5_K |
| layer 0 `mlp.down_proj.weight` | `[1536,8960]` | `blk.0.ffn_down.weight` | 13.76 M | Q4_K |
| `model.embed_tokens.weight` | `[128256,1536]` | `token_embd.weight` | 197.0 M | Q6_K/Q8_0 |
| `lm_head.weight` | `[128256,1536]` | `output.weight` | 197.0 M | Q6_K/Q8_0 |
| `model.norm.weight` | `[1536]` | `output_norm.weight` | 1536 | F32 |

**Absent for this config:** `q_a_proj`, `q_a_layernorm`, `q_b_proj`. HF `DeepseekV3Attention` creates them only when `q_lora_rank is not None`; otherwise it uses `q_proj` directly ([`modeling_deepseek_v3.py:381`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L381), [`modeling_deepseek_v3.py:438`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L438)). llama.cpp has the same branch: `layer.wq` instead of `wq_a/wq_b` ([`deepseek2.cpp:103`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L103)).

**Parameter count:** 10.67 B total, 1.63 B active per token. Attention is 25.0 M/layer ×26 = 650 M; dense MLP 41.3 M; MoE 383.5 M/layer ×25 = 9.59 B; embeddings + head 394 M. Active: attention 650 M + dense 41 M + 4 experts/shared/router 740 M + lm_head 197 M = 1.63 B.

### 1.3 Weight-file sizes

Using the bit rates in §5.2:

| Recipe | Effective size (10.67 B) |
|---|---:|
| Q3_K_XL (Unsloth dynamic, ~3.7 bpw) | ~4.8–5.2 GB |
| Q4_K_M (4.5 bpw, some Q6_K) | ~6.1 GB |
| Q5_K_M (5.5 bpw, some Q6_K) | ~7.5 GB |

With q8 KV at 45k (0.667 GiB) and a ~0.6 GB audio encoder, the 16 GB card has ample headroom. The constraint is **bandwidth and attention-kernel efficiency**, not capacity.

---

## 2. MLA mathematics

### 2.1 Notation and exact equations

Let \(h_t\in\mathbb R^{1536}\). For this checkpoint:

\[
c_t=\operatorname{RMSNorm}(W_{DKV}h_t)\in\mathbb R^{512},\qquad W_{DKV}\in\mathbb R^{512\times1536}
\]

\[
k_t^R=\operatorname{RoPE}(W_{KR}h_t)\in\mathbb R^{64},\qquad W_{KR}\in\mathbb R^{64\times1536}
\]

\[
q_t=W_Qh_t\in\mathbb R^{32\times192},\qquad q_t=[q_t^C(128)\;|\;q_t^R(64)]
\]

\[
q_t^R\leftarrow\operatorname{RoPE}(q_t^R)
\]

Per head \(i\):

\[
k_{t,i}^C=W_{UK,i}c_t,\quad W_{UK,i}\in\mathbb R^{128\times512}
\]

\[
v_{t,i}=W_{UV,i}c_t,\quad W_{UV,i}\in\mathbb R^{192\times512}
\]

The score:

\[
s_{t,u,i}=\frac{(q_{t,i}^C+q_{t,i}^R)^T(k_{u,i}^C+k_u^R)}{\sqrt{128+64}}
=\frac{(q_{t,i}^C)^T W_{UK,i}c_u+(q_{t,i}^R)^Tk_u^R}{\sqrt{192}}
\]

Define the absorbed query:

\[
\tilde q_{t,i}^C=W_{UK,i}^Tq_{t,i}^C\in\mathbb R^{512}
\]

Then

\[
s_{t,u,i}=\frac{\tilde q_{t,i}^{C\,T}c_u+q_{t,i}^{R\,T}k_u^R}{\sqrt{192}}
\]

The attention output is a **512-d latent**, not a 192-d value:

\[
o_{t,i}=\sum_{u\le t}p_{t,u,i}\,c_u\in\mathbb R^{512}
\]

and only afterwards

\[
v_{t,i}=W_{UV,i}o_{t,i}\in\mathbb R^{192}
\]

\[
u_t=W_O\bigl[v_{t,1};\dots;v_{t,32}\bigr],\quad W_O\in\mathbb R^{1536\times6144}
\]

This is exactly DeepSeek-V2’s MLA and the absorption statement in Appendix C: \(W^{UK}\) is absorbed into \(W^{UQ}\), \(W^{UV}\) into \(W^O\); the cached tensors are only \(c_t^{KV}\) and \(k_t^R\) ([DeepSeek-V2 §2.1, eq. 37–47](https://arxiv.org/html/2405.04434v5); Table 1 gives \((d_c+d_h^R)l\) elements/token).

### 2.2 RoPE placement and the 512/64 split

`kv_a_proj_with_mqa` outputs **576** values per token:

- **first 512** → \(c_t\), the KV latent; RMSNorm with `kv_a_layernorm`; **not rotated**;
- **last 64** → \(k_t^R\), shared across all 32 heads; **RoPE applied**; not normalized.

`q_proj` outputs **6144 = 32 × 192** per token:

- per head, **first 128** → \(q^C\), no RoPE;
- per head, **last 64** → \(q^R\), RoPE applied.

The 64-d \(k^R\) is a single vector per token, broadcast to all query heads. The 512-d latent is also shared: this is MQA with one KV head and 32 query heads. llama.cpp builds `Kcur` as `{576,1,n_tokens}` and `Vcur` as `{512,1,n_tokens}`, with `k_pe` viewed from offset 512 ([`deepseek2.cpp:511–575`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L511)). HF does the same split ([`modeling_deepseek_v3.py:444`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L444)).

**RoPE style:** this model has `rope_interleave=true`. HF uses `apply_rotary_pos_emb_interleave`, rotating consecutive pairs \((x_0,x_1),(x_2,x_3)\), not the NeoX half-split ([`modeling_deepseek_v3.py:310`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L310), [`modeling_deepseek_v3.py:453`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L453)). llama.cpp maps `LLM_ARCH_DEEPSEEK2` to `LLAMA_ROPE_TYPE_NORM`, which is the consecutive-pair convention ([`llama-model.cpp:3646`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp#L3646)). Using NeoX half-split RoPE is a silent, catastrophic parity bug.

### 2.3 Scaling, softcap, and numerical hazards

**Scale:** DeepSeek-V2 uses \(1/\sqrt{d_h+d_h^R}=1/\sqrt{192}\), not \(1/\sqrt{576}\). With YaRN factor 64 and `mscale_all_dim=1`, HF applies

\[
\text{mscale}=0.1\cdot1\cdot\ln(64)+1=1.4159
\]

\[
\text{scale}=\frac{\text{mscale}^2}{\sqrt{192}}=\frac{2.0047}{13.856}=0.1447
\]

HF computes this in `self.scaling = yarn_apply_mscale(...)` ([`modeling_deepseek_v3.py:405`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L405)); llama.cpp computes the same via `mscale` and `kq_scale = mscale*mscale/sqrtf(192)` ([`deepseek2.cpp:437`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L437)). Using \(1/\sqrt{576}\) makes the logits about 3.47× too small; using \(1/\sqrt{192}\) without the YaRN \(mscale^2\) makes them about 2× too large at this context.

**Softmax cap / logit softcap:** none. This model has no attention logit softcap and no attention temperature scale. Do not clamp logits. (llama.cpp’s `f_attn_temp_scale` is for GLM4-style models, not this one.)

**Absorption hazards:**

- \(\tilde q^C=W_{UK}^Tq^C\) turns a 128-term dot into a 512-term dot. The mathematical result is identical, but the summation order and dynamic range change. Use **FP32 accumulation** for the absorption GEMM and for QK scores.
- The latent \(c_t\) is RMS-normalized, which controls its scale; \(k^R\) is not normalized. The two score terms can have different magnitudes. Do not rescale the absorbed term by \(1/\sqrt{512}\).
- If \(W_{UK}\) rows are not orthonormal, \(\|\tilde q^C\|\) can grow. This is legal, but it makes FP16 accumulation unsafe. FP16 storage with FP32 accumulate is fine.
- Quantized KV adds noise before the 512-d dot. q8_0 error is small; q4_0 error is large enough that softmax and argmax can change. Keep `k_pe` in FP16 if you can (see §4.3) and validate q4 on audio tasks.
- RMSNorm uses \(\epsilon=10^{-6}\). Compute the norm in FP32; do not use a reciprocal-square-root approximation that loses the epsilon at small norms.
- YaRN cos/sin tables should be FP32; the `mscale` factor must be applied consistently to Q and K.

### 2.4 Can all of W_UV be absorbed?

**Yes, all of it, and the “first 192 rows” framing is a shape error.**

- Per head, \(W_{UV,i}\in\mathbb R^{192\times512}\): 192 output rows, 512 input columns.
- The attention produces \(o_{t,i}\in\mathbb R^{512}\). Then \(v_{t,i}=W_{UV,i}o_{t,i}\in\mathbb R^{192}\).
- Every one of the 512 latent dimensions contributes to every one of the 192 value dimensions. There is no truncation to 192 latent rows.
- \(W_{UK,i}\in\mathbb R^{128\times512}\) is the K-side up-projection; its 128 output rows are the per-head no-RoPE key dims.
- You can fold \(W_{UV}\) into \(W_O\) conceptually: \(W_O\operatorname{diag}(W_{UV,1},\dots,W_{UV,32})\). But that produces a \(1536\times16384\) matrix (25.2 M parameters) instead of \(W_O\) (9.44 M) + \(W_{UV}\) (3.15 M) = 12.6 M, and doubles the output-path MACs. **Do not physically fold it.** Apply `wv_b` per head after attention, exactly as llama.cpp does ([`deepseek2.cpp:584–586`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L584)).
- Likewise, do not pre-fold \(W_{UK}\) into `q_proj`: it grows the Q projection from 9.44 M to 25.2 M parameters and triples its output width. The runtime absorption is only \(32\times512\times128=2.1\) M MACs/token, negligible.

So the practical absorption is:

1. cache \(c_t\) (512) + \(k_t^R\) (64), not per-head K/V;
2. compute \(\tilde q^C\) per token with `k_b`;
3. attention over 576-d K and 512-d V latent;
4. apply `v_b` (192×512 per head), then `o_proj`.

### 2.5 Materialized vs absorbed KV cost

Per token, 26 layers:

| Layout | Bytes/token/layer | KiB/token | 45k total |
|---|---:|---:|---:|
| **Absorbed BF16** (K=576, V aliases first 512) | 1,152 | **29.25** | **1.255 GiB** |
| Absorbed q8_0 | 612 | 15.54 | 0.667 GiB |
| Absorbed q4_0 | 324 | 8.23 | 0.353 GiB |
| Naive separate K+V, BF16 | 2,176 | 55.25 | 2.371 GiB |
| Naive separate K+V, q8/q8 | 1,156 | 29.35 | 1.260 GiB |
| Naive separate K+V, q4/q4 | 612 | 15.54 | 0.667 GiB |
| **Materialized BF16** (32×128 + 64 + 32×192) | 20,608 | **523.25** | **22.45 GiB** |
| Materialized q8_0 | 10,948 | 278.0 | 11.93 GiB |
| Materialized q4_0 | 5,796 | 147.2 | 6.32 GiB |

The materialized layout is 17.9× larger than absorbed BF16. At 45k it does not fit with weights and activations in 16 GB, and even q4 materialized (6.32 GiB) plus q4 weights (~6.1 GB) leaves little room for prefill scratch. **Absorption is not an optimization here; it is a requirement.**

The 29.25 KiB figure matches DeepSeek-V2 Table 1: \((d_c+d_h^R)l=(512+64)\times26\) elements × 2 bytes = 29,952 B = 29.25 KiB. llama.cpp confirms it stores only K for MLA, not a separate V ([`llama-kv-cache.cpp:230–233`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp#L230)).

**Important q8/q4 caveat:** because V is the first 512 dims of the same K latent, K and V must use the **same quantization type**. llama.cpp enforces this for MLA: “model does not support different K and V cache types” ([`llama-context.cpp:4328`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp#L4328)). So “q8 K / q4 V” is not available in the absorbed layout. If you insist on asymmetric K/V, you must store a second V cache: q8 K + q4 V costs 22.85 KiB/token = 0.981 GiB at 45k, which is **worse** than q8/q8 shared (0.667 GiB). Do not do it.

### 2.6 Code verification

The derivation triangulates against three independent implementations:

- **HF reference:** direct `q_proj` when `q_lora_rank is None`; `kv_a_proj_with_mqa` split into 512 + 64; `kv_a_layernorm` on the 512; interleaved RoPE on the 64; `kv_b_proj` expanded to 128 + 192 per head; `self.scaling` includes YaRN mscale ([`modeling_deepseek_v3.py:438–476`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L438)).
- **llama.cpp deepseek2 graph:** `wq` direct for lite models; `wkv_a_mqa` 576; split at 512; `k_pe` RoPE; `wk_b` used as `q_nope_absorbed = wk_b × q_nope`; `Kcur = concat(kv_cmpr, k_pe)`; `Vcur = kv_cmpr`; `build_attn(..., wv_b, ...)` ([`deepseek2.cpp:494–586`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L494)). The comment at line 583 states: “MLA with the absorption optimization converts into MQA”.
- **Converter:** `kv_b_proj` is split into `k_b` and `v_b`, and `k_b` is transposed so that the latent dim is contiguous for the absorption matmul ([`conversion/deepseek.py:440–455`](https://github.com/ggml-org/llama.cpp/blob/master/conversion/deepseek.py#L440)). It writes `key_length = kv_lora_rank + qk_rope_head_dim = 576`, `value_length = 512`, `key_length_mla = 192`, `value_length_mla = 192` ([`conversion/deepseek.py:355–359`](https://github.com/ggml-org/llama.cpp/blob/master/conversion/deepseek.py#L355)).

---

## 3. Sparse attention options

### 3.1 DeepSeek Sparse Attention (DSA / “lightning indexer”)

**Exact architecture** (DeepSeek-V3.2, §2.1):

\[
I_{t,s}=\sum_{j=1}^{H_I}w_{t,j}\,\operatorname{ReLU}(q_{t,j}^I\cdot k_s^I)
\]

Then only the top-\(k\) key/value entries by \(I_{t,s}\) go to the main MLA attention:

\[
u_t=\operatorname{Attn}\left(h_t,\{c_s\mid I_{t,s}\in\operatorname{TopK}(I_{t,:})\}\right)
\]

V3.2 uses \(H_I=64\), \(d_I=128\), top-k 2048, per-layer indexers. The indexer has its own `wq_b`, `wk`, `k_norm`, and `weights_proj`, plus a separate FP8 K cache and scale cache ([`inference/model.py:439–487`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L439); paper eq. 1–4).

**Why it cannot be enabled on an untrained model:**

- The indexer weights do not exist in GigaChat Audio’s checkpoint.
- DSA is not a post-hoc mask. The indexer is trained with a KL loss against the dense attention distribution, first in a dense warm-up, then in a sparse stage over 2048 selected tokens ([V3.2 §2.1.1](https://arxiv.org/html/2512.02556v1); eq. 3–4).
- The main model is adapted during sparse training. Switching a dense-trained model to top-2048 at 45k drops 95.5% of the context; the attention distribution is far outside training.
- Each layer needs its own indexer; 26 indexers with separate K caches are not a small add-on.

**Is a heuristic indexer viable?** Yes, but only as an approximation, and it must be validated. Practical options:

1. **Block summaries of the latent \(c\).** Block size 64. Store a 64-d random projection of the block mean, plus the block max norm. Score the query’s absorbed \(\tilde q\) against the summary; always include sink blocks and a local window. Cost: 704 blocks × 64 dims × 26 layers = 1.17 M MACs/token, negligible; summary memory ~2.3 MB at 45k.
2. **LSH / SimHash on \(c\).** The latent is not rotated, so it is position-invariant and hashable. But maintaining buckets with inserts and getting high recall for top-k is harder than block summaries on one GPU. Not recommended for v1.
3. **Sink + local window only.** Cheap and robust for local attention, but cannot retrieve early events in a 120-minute recording. Not sufficient for temporal grounding by itself.

A good approximation is: **4 sink blocks + 1024 local tokens + top-(k−20) blocks by 64-d summary**, then exact scores for all keys in the selected blocks. This is not DSA; call it block-sparse MLA.

**Why DSA’s own indexer is not free at 45k:** with 64 heads × 128 dims, the indexer is still \(O(L^2)\): about 431 TFLOP for 45k × 26 layers, versus ~163 TFLOP for the top-2048 main attention. DeepSeek uses FP8 and specialized kernels. For this engine, a low-dimensional heuristic summary is more appropriate; do not copy the 64×128 indexer without measuring.

### 3.2 NSA and MoBA

**NSA (Native Sparse Attention)** uses three branches: compressed coarse blocks, selected fine blocks, and a sliding window, with learned gates. It is trained end-to-end from scratch; the paper’s model is 27B, trained on 270B tokens. Block sizes: compression 32, selection 64, selected blocks 16, sliding window 512. Compression MLPs and gates are learned, so NSA cannot be applied post-hoc to a dense checkpoint ([NSA §3, §4.1](https://arxiv.org/html/2502.11089v2)).

**MoBA (Mixture of Block Attention)** partitions the context into blocks and applies a learned top-k gate per query block; it introduces no new parameters but the gate is trained. The paper trains MoBA from scratch and uses a MoBA/full hybrid; it can switch between full and sparse during training, not arbitrary post-hoc inference. Block sizes in experiments: 512 (top-3), 2048, 4096; sparsity 81–95% ([MoBA §2, §3](https://arxiv.org/html/2502.13189v1)). A heuristic block scorer is essentially the block-sparse approximation above; MoBA’s learned gate is not available.

**Conclusion:** neither NSA nor MoBA can be dropped into GigaChat Audio without training. The only practical post-hoc option is block-sparse top-k over the latent KV, validated against dense MLA.

### 3.3 Practical block-sparse algorithm

- **Block size:** 64 tokens. This aligns with a paged KV page and contains two q8_0 blocks (32 elements each). For q4_0, also two blocks.
- **Selection:** for each query token, compute a cheap score for every block:
  \[
  \text{score}_b = \tilde q^T \bar c_b + \text{upper bound on } |q^R|\,|k^R|
  \]
  where \(\bar c_b\) is a 64-d random projection of the block mean (or a 128-d projection if recall is poor). Include the first 2 blocks and the last 16 blocks unconditionally.
- **Gather:** sort selected block IDs; for each block, load 64×576 q8/q4 into shared memory, dequantize to FP16, compute exact 576-d scores for all 64 keys, and do online softmax. The V tile is the first 512 dims of the same block.
- **Top-k:** start with 25% of context (11,250 tokens at 45k, 176 blocks). If quality holds, try 12.5% (5,625) and 6.25% (2,812). DSA’s 2048 is 4.5%; MoBA’s 3×512 is 3.4%.
- **Speedup at 45k, batch 1:** decode is dominated by active weights (~0.92 GB) plus KV. Top-25% q8 reduces KV from 0.716 GB to 0.179 GB; total bytes 1.636 → 1.099 GB, i.e. **~1.5× decode**. q4 reduces 1.299 → 1.015 GB, **~1.28×**. Prefill is different because causal attention averages half the context: top-25% of *context* is ~50% of dense causal work, so prefill speedup is only **~1.8×** including projections. Top-12.5% gives **~3.3×**, top-6.25% **~5.3×**. See §4.5 for the table.
- **Quality verification:** compute dense attention weights for a sample of queries and measure the fraction of softmax mass captured by the selected blocks. Then compare logits and generated tokens (thresholds in §3.5).

### 3.4 Streaming / rolling KV for 120-minute audio

The model consumes 6.25 audio tokens/s (160 ms frame rate), so 120 minutes = **45,000 tokens** ([GigaChat Audio §3.1](https://arxiv.org/html/2607.10387v1); the model card says ~33k tokens for 90 min). Full q8 KV is only 0.667 GiB; full q4 is 0.353 GiB. Even at the model’s 262k max position, q8 is 3.79 GiB and q4 is 2.01 GiB. **There is no memory need to stream.**

What breaks temporal grounding if you do stream:

- The model interleaves a time marker every 60 s; removing periodic anchors drops long-form mIoU from 53.8 to 14.2 ([GigaChat Audio §4.1](https://arxiv.org/html/2607.10387v1)). Evicting or compressing those markers is the worst thing you can do.
- Evicting early audio makes questions about early events unanswerable; a local window only preserves the end of the recording.
- A compressed memory (block means) preserves gist but blurs exact timestamps; the model was trained on full-resolution tokens, not learned compression tokens.
- Attention sinks improve stability but do not restore content.

**Recommendation:** keep the full 45k KV. If you ever need 262k, q8/q4 still fits; revisit streaming only after measuring a real memory wall.

### 3.5 Recommendation and acceptance thresholds

**Phase 1 — dense MLA only.** Implement dense absorbed MLA with q4 weights and q8 KV. No sparse approximation. This is the reference and the fallback.

**Phase 2 — optional q4 KV.** Only if it passes the task gates below.

**Phase 3 — optional block-sparse** (top-12.5% to top-25%) after dense parity. Do not attempt DSA/NSA/MoBA; you do not have the weights or the training.

Suggested gates (tighten after collecting data):

| Gate | Dense q8 vs dense FP16 | q4 KV vs q8 | Block-sparse vs dense |
|---|---:|---:|---:|
| Logits max abs diff, p99 | ≤ 0.10 | ≤ 0.50 | ≤ 0.05 |
| Top-1 token agreement | ≥ 99.9% | ≥ 99.0% | ≥ 99.5% |
| 256-token greedy agreement | ≥ 99% | ≥ 95% | ≥ 98% |
| WER Δ (FLEURS/Golos) | ≤ 0.2 pp | ≤ 0.5 pp | ≤ 0.3 pp |
| Temporal mIoU Δ | ≤ 0.5 pp | ≤ 1.0 pp | ≤ 0.5 pp |
| Speedup at 45k | — | 1.15–1.3× decode | ≥1.2× decode, ≥2× prefill |

If block-sparse fails the mIoU gate, fall back to dense. A 1.5× decode gain is not worth a 5-point temporal-grounding loss.

---

## 4. CUDA / kernel feasibility on sm_89

### 4.1 FlashMLA and FlashAttention

**FlashMLA is not usable on Ada.** The last release supporting DeepSeek-V3/V3.2 requires SM90/SM100; dense decoding is SM90-only ([FlashMLA README at ba89a3](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md#L65)). Current master requires SM100/SM103 and explicitly removed Hopper support ([current README](https://github.com/deepseek-ai/FlashMLA/blob/main/README.md#L75)). The kernel uses WGMMA and TMA: the deep-dive describes a 64×576 K block loaded with 9 TMA copies and WGMMA output fragments occupying 32,768 registers ([FlashMLA deep dive](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/docs/20250422-new-kernel-deep-dive.md)). Ada sm_89 has neither WGMMA nor TMA. Do not port it.

**FlashAttention-2** supports Ampere/Ada/Hopper and **head dimensions up to 256** ([FA2 README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#L145)). **FlashAttention-3** is Hopper-only ([FA3 README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#L31)). 576 is outside both.

**Why 576 is a problem:** the online softmax must see the full score \(q_{\text{abs}}\cdot c+q_{\text{pe}}\cdot k_{\text{pe}}\) before normalization. You cannot run two independent FlashAttention calls (one for 512, one for 64) and add the outputs, because
\[
\operatorname{softmax}(s_{512}+s_{64})\neq \operatorname{softmax}(s_{512})+\operatorname{softmax}(s_{64}).
\]
Padding 576→640 does not help; FA2 still rejects >256. The standard workarounds are:

1. **Custom kernel** that computes both dot products into the same score buffer, then softmax. This is what llama.cpp does for `DKQ=576, DV=512`.
2. **Materialize to 192-d per-head K/V** and use FA2. This is mathematically valid, but the KV cache becomes 523.25 KiB/token (22.45 GiB at 45k) and is not viable on 16 GB.
3. **Split the 512 latent into two 256 halves** and run two GEMMs into the same score tile; still custom, but the score accumulation is simple. The 64-d RoPE part is a third small GEMM.

llama.cpp proves option 1 works on Ada: `case 576` asserts `V->ne[0] == 512` and dispatches `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576,512,...>` ([`fattn.cu:377–435`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu#L377)). The Ampere/Ada config for 576/512 is `ncols=64, nthreads=256, occupancy=1, nbatch_fa=32` ([`fattn-mma-f16.cuh:77–85`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh#L77)). That is a concrete starting point.

### 4.2 sm_89 resources and tile shapes

Confirmed by NVIDIA’s Ada tuning guide and CUDA compute-capability table:

- **100 KB shared memory per SM**, **99 KB per thread block** (1 KB reserved), 128 KB unified L1/shared, carveout options 0/8/16/32/64/100 KB ([Ada tuning guide §1.4.1](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html); [CUDA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)).
- **64K 32-bit registers per SM**, **255 registers per thread**, 1024 threads/SM (sm_89).
- RTX 4060 Ti: 34 SMs, 4352 CUDA cores, 136 tensor cores, 288 GB/s, 16 GB GDDR6, PCIe 4.0 x8 ([specs](https://getdeploying.com/gpus/nvidia-rtx-4060-ti)).
- Peak: **22.1 FP32 TFLOPS, 44.1 FP16/BF16 TFLOPS, 176.5 INT8 TOPS** (dense; sparsity doubles).

**Feasible decode tile (batch 1, one token, 32 heads):**

Treat the 32 heads as the M dimension: \(M=32\), \(N=B_c\) keys, \(K=576\), output \(32\times512\).

| Resource | Size |
|---|---:|
| Q tile \(32\times576\times2\) B | 36.9 KB |
| K tile \(32\times576\times2\) B (Bc=32) | 36.9 KB |
| P tile \(32\times32\times4\) B | 4 KB |
| O accumulators \(32\times512\) FP32 | 64 KB in registers (32 floats/thread at 512 threads) |
| Total shared | **77.8 KB** |
| Threads | 512 (16 warps) |
| Registers/thread | ~80–100, 512×100 = 51K < 64K |
| Occupancy | 1 CTA/SM (shared-memory limited) |

With q8 KV, the K tile is 18.4 KB; you can use Bc=64 (36.9 KB) and still fit ~78 KB. With q4, Bc=64 is 18.4 KB; Bc=128 is 36.9 KB. Use **split-K flash decoding** (4–8 splits) to occupy more than 26 SMs: 26 layers × 8 splits = 208 CTAs, ~6 waves on 34 SMs. Each split reads its KV slice once; combine partial outputs and LSE in a small kernel.

**Prefill tile:** the same \(M=32\), \(N=B_c\), \(K=576\) shape works for a group of 4 heads × 8 tokens (32 rows), but then K is re-read per head group. llama.cpp’s `ncols=64` kernel is the proven reference; start there rather than inventing a new prefill schedule. The hard constraint is that \(Q\) and \(O\) for many tokens × many heads exceed the 64K-register file; do not try to hold 32 tokens × 32 heads in registers.

**128-bit loads:** K/Q rows are 1152 B = 72 `float4`; use 16-B vector loads. q8_0 rows are 612 B (not a multiple of 16); q4_0 rows are 324 B. Pad each cache row to a 16-B multiple (q8: 624 or 640 B; q4: 336 B) or use 4-B/2-B loads. The padding costs 2–5% memory but keeps `uint4` loads and `ldmatrix` happy.

**Register pressure:** keep the online-softmax state (\(m\), \(\ell\)) in FP32 per row; the P tile in registers; do not materialize the full \(32\times45k\) score matrix. Use `__launch_bounds__(512, 1)` and cap at ~128 registers. The 576-d QK is split into 36 `mma.m16n8k16` steps per 32-key tile; the 512-d PV is 32 steps. Both fit.

### 4.3 Tensor cores and quantized KV

**Ada MMA shapes** (PTX `mma.sync`):

- FP16/BF16: `m16n8k16`, FP32 accumulate. This is the workhorse.
- INT8: `m16n8k32`, INT32 accumulate.
- INT4: `m16n8k64`, INT32 accumulate.
- FP8 is available on Ada, but out of scope per your constraint.

**Can q8/q4 KV use INT8 tensor cores?** Not directly with FP16 queries. To use `mma.s8`, you must also quantize Q to int8 per row/block, then rescale the int32 accumulator by the Q and K scales. That is possible, but:

- softmax is sensitive to score quantization;
- q4 KV cannot use `mma.s4` with FP16 Q at all; you would have to quantize Q to int4, which is too lossy;
- decode is memory-bound, so the 4× INT8 tensor throughput does not help.

**Recommendation:** dequantize q8/q4 K/V to FP16 in shared memory and use FP16 `mma`. Use INT8 `dp4a`/`mma.s8` only for **weight** GEMV/GEMM (MMVQ/MMQ style), where the activation can be quantized to q8 and the weights are already q4/q3. That is exactly what llama.cpp does: `mmvq.cu` has `vec_dot_q4_K_q8_1`, `vec_dot_q3_K_q8_1`, etc. ([`mmvq.cu:42–55`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu#L42)); `mmq.cuh` uses int8 MMA for prefill ([`mmq.cuh:84–87`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh#L84)).

**Expected effective throughput for decode:** the active weight stream is ~0.92 GB (q4) and the KV stream is 0.35–0.72 GB at 45k. At 30–50% of 288 GB/s, decode is 9–19 ms/token; tensor cores are not the limit. Do not spend effort on INT8 attention.

### 4.4 Quantized KV in attention kernels: what llama.cpp actually does

Two different paths:

- **Vector kernel (`fattn-vec`)**: supports q8_0/q4_0/q4_1/q5_0/q5_1, dequantizes K/V rows on the fly inside the attention loop ([`fattn-vec.cuh`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-vec.cuh)). This is true fused dequant, but it is limited to head sizes ≤256.
- **MMA kernel**: for `DKQ=576`, llama.cpp uses `BEST_FATTN_KERNEL_MMA_F16`, which sets `need_f16_K=true` and `need_f16_V=true` ([`fattn.cu:770–782`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu#L770)). In other words, **quantized KV is upcast to F16 in a separate scratch buffer before the MMA attention**. It saves VRAM, but not attention read bandwidth, and adds a dequant pass. This is why some q4 KV benchmarks show slower long-context processing.

**For your engine:** fuse the dequant into the 576-d attention kernel. Read q8/q4 blocks, dequantize to FP16 in shared memory, then `mma`. That is the only way to realize the 0.667/0.353 GiB bandwidth savings. It is more work than llama.cpp’s current MLA path, and it is the main reason to write a custom kernel rather than wrap `ggml_mul_mat`.

**Can dequant be fused with softmax?** Yes: the score tile is produced by QK^T after K dequant; softmax is elementwise on that tile; V dequant happens before PV. The fusion boundary is natural. Do not try to fuse the entire attention into one monolithic kernel; keep QK-softmax-PV as the online loop.

**Reported accuracy loss:** on a llama.cpp benchmark, q8_0/q8_0 KV gave mean KLD 0.001782 and PPL +0.033; q8_0/q4_0 gave PPL 8.006968 vs 7.960548 (+0.58%), same-top-p 96.885%, RMS Δp 1.762% ([llama.cpp discussion #23470](https://github.com/ggml-org/llama.cpp/discussions/23470)). These are not GigaChat numbers. For MLA you cannot use q8 K/q4 V anyway; you would use q4/q4, which is worse. Treat q4 KV as an experiment, not a default.

### 4.5 Roofline and performance estimates at 45k, batch 1

**Assumptions:**

- 45,000 tokens (120 min at 6.25 tokens/s).
- 26 layers; K=576, V=512; 32 heads.
- Active q4 weights: 0.92 GB (including lm_head, excluding embeddings lookup).
- q8 KV: 0.716 GB; q4 KV: 0.379 GB; BF16: 1.348 GB.
- Decode achieved bandwidth: 30%, 40%, 50% of 288 GB/s = 86.4, 115.2, 144 GB/s.
- Prefill achieved compute: 40%, 50%, 60% of 44.1 FP16 TFLOPS = 17.6, 22.1, 26.5 TFLOPS.
- Excludes audio encoder, sampling, and host overhead.

**Decode FLOPs/token:** attention \(26\times32\times(576+512)\times45{,}000\times2=81.4\) GFLOP; projections \(2\times1.63B=3.26\) GFLOP; total **84.7 GFLOP**. At peak that is 1.92 ms; at 25% efficiency 7.7 ms. Arithmetic intensity for dense q8 is 84.7/1.636 = 51.8 FLOP/B; the ridge point is 44.1 TFLOPS / 288 GB/s = 153 FLOP/B. **Decode is memory-bound.**

| Case | Bytes/token | 30% BW | 40% BW | 50% BW |
|---|---:|---:|---:|---:|
| Dense MLA, q4 W + q8 KV | 1.636 GB | 18.9 ms / 53 tok/s | 14.2 ms / 70 tok/s | 11.4 ms / 88 tok/s |
| Dense MLA, q4 W + q4 KV | 1.299 GB | 15.0 ms / 67 | 11.3 ms / 89 | 9.0 ms / 111 |
| Sparse 25% ctx, q4 W + q8 KV | 1.099 GB | 12.7 ms / 79 | 9.5 ms / 105 | 7.6 ms / 131 |
| Sparse 25% ctx, q4 W + q4 KV | 1.015 GB | 11.7 ms / 85 | 8.8 ms / 114 | 7.0 ms / 142 |
| Sparse 6.25% ctx, q4 W + q8 KV | 0.965 GB | 11.2 ms / 89 | 8.4 ms / 119 | 6.7 ms / 149 |
| Dense MLA, BF16 KV (reference only) | 2.268 GB | 26.2 ms / 38 | 19.7 ms / 51 | 15.8 ms / 63 |

Realistic target: **50–100 tok/s** at 45k with dense q8 and a good kernel; q4 KV might reach 90–130, but only if fused dequant works and quality holds. For sanity: the model card reports 211 tok/s at 60–120 min on an H100 in BF16 ([GigaChat Audio README](https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B)); the 4060 Ti has ~8.6% of H100 bandwidth, so 20–70 tok/s is the plausible envelope, with 50–80 realistic for a tuned q4/q8 engine.

**Prefill FLOPs:** dense causal attention
\[
26\times32\times45{,}000^2\times(576+512)=1.833\ \text{PFLOP}
\]
plus projections \(2\times1.63B\times45k=0.147\) PFLOP → **1.98 PFLOP**.

| Case | Attention fraction | Total PFLOP | 40% TFLOPS | 50% | 60% |
|---|---:|---:|---:|---:|---:|
| Dense | 100% | 1.980 | 112 s | 90 s | 75 s |
| Top-25% of context (11.25k keys) | ~44% of causal | 0.953 | 54 s | 43 s | 36 s |
| Top-12.5% (5.6k keys) | ~25% of causal | 0.604 | 34 s | 27 s | 23 s |
| Top-6.25% (2.8k keys) | ~12.5% of causal | 0.375 | 21 s | 17 s | 14 s |

Why top-25% is only 1.8×: causal attention averages over half the context, so selecting 25% of *context* is ~50% of the dense causal work. To get the 4× people expect from “25% sparse”, you need 25% of the *causal average*, i.e. ~6.25% of context.

**Practical implication:** a 120-minute recording costs **60–90 s of dense prefill** on this card. If the same recording is queried multiple times, **prefill once and persist the KV** (0.667 GiB q8 per recording). That turns the 45k prefill into a one-time cost and makes the interactive path decode-only. This is the single most important product-level optimization.

### 4.6 Paged KV, block tables, and CUDA graphs

**Paged KV:** block size 64. At 45k there are 704 blocks/layer ×26 = 18,304 blocks. Each q8 block is 64×612 B = 39.2 KB; total 716 MB. The block table is an `int32[704]` per sequence. The attention kernel loads block IDs from device memory, then loads contiguous 64-token tiles from the physical blocks.

**CUDA graph compatibility:** yes, if you design for it:

- The block-table pointer and the `seqlen` scalar live in fixed device buffers. Update their **contents** with `cudaMemcpyAsync` before replay; do not reallocate them.
- Capture the whole 26-layer decode graph once per batch size / split count. The graph is ~400–600 kernel nodes; replay overhead is ~5–20 µs, negligible vs 10 ms/token.
- Capture constraints: no `cudaMalloc`/`cudaFree`, no `cudaMemcpy` (use async), no `cudaStreamSynchronize`, no host callbacks, no legacy default stream during capture. Preallocate all workspaces, or use a stream-ordered pool. All kernels must be on the capture stream.
- MoE routing must be device-side. If top-k or expert gather syncs to host, the graph cannot be captured. llama.cpp explicitly disables graphs for `MUL_MAT_ID` when the fallback synchronizes ([`ggml-cuda.cu:3012–3037`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu#L3012)).
- Changing sequence length: use a fixed maximum grid and pass `seqlen` as a device scalar; the kernel early-exits. Or capture separate graphs for buckets (1k, 2k, 4k, 8k, 16k, 32k, 45k). Avoid `cudaGraphExecUpdate` for topology changes; use it only for scalar/pointer updates, and be prepared for update failure. llama.cpp’s implementation updates the executable and re-instantiates on failure ([`ggml-cuda.cu:3080–3125`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu#L3080)).
- Prefill has variable shapes; run it eagerly or capture per bucket. The decode graph is the one that matters.

**Verdict:** paged KV + graphs is feasible and necessary for 26 layers at batch 1. The main constraint is eliminating host synchronization in MoE routing and expert gather.

### 4.7 Triton AOT inside a C++ engine

**Technically acceptable, but not the best choice for the attention kernel.**

- Triton can be compiled ahead of time to a cubin/PTX with `triton.compile`; you can extract `asm["cubin"]` and load it with the CUDA driver API (`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`). PTX can be JIT-compiled with `cuModuleLoadDataEx` and `CU_JIT_TARGET=sm_89`.
- You must freeze `num_warps`, `num_stages`, shared-memory size, and all specialization constants; Triton’s Python autotuner/launcher is not available in the C++ process. You also need to handle argument marshalling yourself.
- Triton on sm_89 uses `mma.sync` + `cp.async`, not TMA/WGMMA. That is fine for 576-d attention, but Triton gives coarse control over shared-memory layout and register pressure, which is exactly where the 576-d kernel is hard.
- Paged KV, fused q8/q4 dequant, and split-K combine are all easier in raw CUDA/CUTLASS.
- AOT Triton is reasonable for the **block indexer / summary scoring** or a prototype; use **raw CUDA (or CUTLASS/CuTe)** for the MLA attention. Load a cubin like:

```cpp
CUmodule mod; CUfunction fn;
cuModuleLoadData(&mod, cubin_bytes);
cuModuleGetFunction(&fn, mod, "mla_attn_decode");
void* args[] = {&q, &k_cache, &block_table, &seqlen, &out};
cuLaunchKernel(fn, grid, block, 1, smem_bytes, stream, args, nullptr);
```

CUDA driver API reference: [`cuModuleLoadData`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html). Triton compile API: [`triton.compile`](https://triton-lang.org/main/python-api/generated/triton.compile.html).

---

## 5. Implementation blueprint

### 5.1 Minimal kernel set

| Kernel | Shapes / work | Dtype | sm_89 feasibility | Effort |
|---|---|---|---:|---:|
| RMSNorm | 1536 and 512; FP32 accum | FP16 in / FP32 | trivial | 0.5 wk |
| `q_proj` + `kv_a_proj` | 1536→6144 and 1536→576 | q4 weights, FP16 act | MMQ/MMVQ | 2–3 wk |
| Q absorption + RoPE | 32×128×512; interleaved RoPE on 32×64 and 1×64 | FP16/FP32 | straightforward | 1 wk |
| MLA decode attention | 32 heads × 576 K / 512 V, split-K, paged q8/q4 | FP16 mma, FP32 softmax | hard but proven | 4–6 wk |
| MLA prefill attention | causal 576/512; Br/Bc tiles | FP16 mma | hard; port llama.cpp ncols=64 | 3–4 wk |
| `v_b` + `o_proj` | 32×192×512, 6144→1536 | q4 | MMQ/MMVQ | 1–2 wk |
| MoE router | 64×1536 sigmoid + bias + top-4 + normalize | FP32 | trivial | 0.5 wk |
| SwiGLU experts | 4 × (1280×1536 gate/up, 1536×1280 down) | q4 | MMVQ batch1; MMQ batch>1 | 2–3 wk |
| Shared expert + dense layer 0 | 1280 / 8960 | q4 | same as above | included |
| `lm_head` | 1536→128256 | q6/q8 | MMVQ | 1 wk |
| KV quantize/write | 512 latent + 64 rope → q8/q4 | int8/4 + scales | simple | 1 wk |
| Block indexer (optional) | 64-d summaries, top-k blocks | FP16 | simple | 2–4 wk |
| CUDA graph capture/replay | 26 layers, paged KV | — | feasible | 1–2 wk |

**Key implementation details:**

- **Q absorption:** `q_abs[h] = sum_d k_b[h,d,latent] * q_nope[h,d]`. Store `k_b` as `[32,128,512]` so the 512 latent dim is contiguous; this is what the converter’s `k_b.transpose(1,2)` achieves.
- **Attention:** K tile is `[64,576]`; V is the first `[64,512]` of the same tile. Compute scores with 576-d Q; softmax in FP32; accumulate latent O `[32,512]`; then `v_b` `[32,192,512]` and `o_proj`.
- **MoE:** `scores = sigmoid(gate(h))`; `choice = scores + e_score_correction_bias`; top-4 by `choice`; weights are `scores[top4]`, normalized to sum 1 (`norm_topk_prob=true`); scale by `routed_scaling_factor=1`; add shared expert. `n_group=1` means no group filtering. This matches llama.cpp’s `build_moe_ffn` call ([`deepseek2.cpp:353–366`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L353)).
- **Batch 1 MoE:** gather 4 expert slices and run MMVQ; fuse gate+up into one 2560×1536 GEMV. Do not launch 64 experts.
- **KV layout:** `[layer][block][64][576]` quantized; block table `int32[704]`. V is a view with row stride 576 and width 512. If you quantize, q8_0 blocks 0–15 cover the latent, blocks 16–17 cover `k_pe`. Keep `k_pe` FP16 if you can: store a separate FP16 `[64]` per token, adding 128 B/token/layer (3.25 KiB/token total, 0.15 GiB at 45k). This protects RoPE precision from q4.
- **Prefill:** process one audio once and persist KV; use llama.cpp’s `ncols=64` MMA prefill as the reference. Do not attempt to hold 32 tokens × 32 heads of Q/O in registers.

### 5.2 GGUF types, block sizes, and feeding the MLA kernel

| Type | Block | Bytes/block | bpw | Scale structure |
|---|---:|---:|---:|---|
| Q3_K | 256 | 110 | 3.4375 | 16 sub-blocks of 16; 6-bit scales packed in 12 B; 2-bit low quants + 1-bit high mask |
| Q4_K | 256 | 144 | 4.5 | 8 sub-blocks of 32; 6-bit scales + 6-bit mins in 12 B; 4-bit quants |
| Q5_K | 256 | 176 | 5.5 | 8 sub-blocks of 32; 6-bit scales + mins; 4-bit low + 1-bit high |
| Q6_K | 256 | 210 | 6.5625 | 16 sub-blocks of 16; 8-bit scales; 6-bit quants |
| Q8_0 | 32 | 34 | 8.5 | one FP16 scale per 32 |
| Q4_0 | 32 | 18 | 4.5 | one FP16 scale per 32 |

Struct definitions: [`ggml-common.h:316–356`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h#L316); q8_0/q4_0 at lines 251–256. `QK_K=256`, `K_SCALE_SIZE=12`.

**Q3_K_XL is not a `ggml_type`.** It is an Unsloth Dynamic label for a per-tensor mix (typically Q3_K for bulk FFN, Q4_K/Q6_K/Q8_0 for sensitive tensors). llama.cpp’s ftype enum has `Q3_K_S/M/L`, `Q4_K_M`, etc., but no `Q3_K_XL` ([`llama.h:130–134`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h#L130)). Inspect the actual per-tensor types with `gguf-dump`; your loader must handle every type present, not assume a uniform block size.

**Feeding without full dequant:**

- **Decode (M=1):** use MMVQ-style q4_K×q8_1 integer dot products; quantize the activation to q8_K/q8_1 once per 256-element block, then `dp4a`. This is what `mmvq.cu` does ([`mmvq.cu:42–55`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu#L42)).
- **Prefill (M large):** use MMQ int8 MMA (`m16n8k32`) with per-block scale correction, or dequantize tiles to FP16 and use FP16 MMA. llama.cpp’s `mmq.cuh` supports Q3_K/Q4_K/Q5_K ([`mmq.cuh:84–87`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh#L84)).
- **Attention:** the KV cache is q8_0/q4_0, not K-quant. Dequantize 32-element blocks to FP16 in shared memory inside the attention kernel. **Do not** use `ggml_mul_mat` for attention; it is a weight GEMM, not a fused attention kernel.
- **Alignment:** Q3_K blocks are 110 B and Q6_K 210 B, neither 16-B aligned. Either pad rows to 16 B in a staging buffer or use 2-B/4-B loads. Q4_K (144 B) and Q5_K (176 B) are 16-B aligned.

**Recommended per-tensor recipe for 16 GB:**

- `token_embd`, `output`: Q6_K or Q8_0 (protect logits).
- `attn_q`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`, `attn_output`: Q4_K, optionally Q5_K for `attn_k_b`/`attn_v_b` (attention is the quality-critical path).
- Experts: Q3_K or Q4_K; shared expert Q4_K.
- Router and norms: F32/F16.
- KV: q8_0 default; q4_0 only after task validation.

### 5.3 Numerical parity plan vs llama.cpp `deepseek2`

**Reference stack:**

1. **HF eager BF16** decoder (`DeepseekV3ForCausalLM`) with `attn_implementation="eager"`, `output_hidden_states=True`; materialized MLA is mathematically equivalent and is the cleanest reference.
2. **llama.cpp master** with the decoder converted to F16 GGUF:
   `python convert_hf_to_gguf.py decoder --outtype f16 --outfile gigachat-decoder-f16.gguf`.
   The converter registers `DeepseekV3ForCausalLM` → `DEEPSEEK2` and splits `kv_b_proj` correctly ([`conversion/deepseek.py:225–241`](https://github.com/ggml-org/llama.cpp/blob/master/conversion/deepseek.py#L225), [`conversion/deepseek.py:440–455`](https://github.com/ggml-org/llama.cpp/blob/master/conversion/deepseek.py#L440)).
3. **Your engine** with the same GGUF and q8/q4 KV.

**Test stages:**

| Stage | What | Pass criterion |
|---|---|---|
| 0. Tensor load | Dequantize each GGUF tensor; compare to HF BF16 | max abs ≤ 2e-3; relative L2 ≤ 1e-4 |
| 1. Layer 0 | 128-token prompt; compare q, k, v, attn_out, MLP out | relative L2 ≤ 1e-2; cosine ≥ 0.999 |
| 2. Full 26 layers | 512-token prompt; compare `attn_out`, `ffn_out`, `l_out` via llama.cpp callback names ([`deepseek2.cpp:487,646,681`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L487)) | relative L2 ≤ 2e-2; cosine ≥ 0.998 |
| 3. Logits, FP16 | 512 / 2k / 8k / 45k prompts; max abs, RMSE, top-1/top-5 | max abs ≤ 0.05; top-1 100% |
| 4. Greedy 256 | temperature 0; token-for-token and edit distance | exact match ≥ 99.8% |
| 5. q8 KV | same prompts vs FP16 engine | max abs ≤ 0.10; top-1 ≥ 99.9%; greedy ≥ 99% |
| 6. q4 KV | same prompts vs q8 engine | max abs ≤ 0.50; top-1 ≥ 99%; greedy ≥ 95% |
| 7. q4 weights | vs BF16 reference | max abs ≤ 1.0; top-1 ≥ 95%; greedy ≥ 85% |
| 8. Audio tasks | FLEURS/Golos WER; TimeGround mIoU; TD/SUM | WER Δ ≤ 0.5 pp; mIoU Δ ≤ 1–2 pp |

**Per-layer activation comparison:** use llama.cpp’s `llama-eval-callback` or a custom callback to dump `attn_out`, `ffn_out`, `l_out`. Compare relative L2 and cosine per layer; a sudden jump localizes a mapping or RoPE bug. Normalize max-abs by the layer’s RMS to avoid scale-dependent thresholds.

**Tolerance rationale:** FP16-vs-BF16 differences are already ~1e-3; q8 KV adds ~1e-3–1e-2; q4 weights dominate (~0.1–1.0 logit). Exact greedy match over 256 tokens is brittle near ties; use token agreement and top-5 overlap as the primary metric, with exact match as a secondary gate.

### 5.4 Risks and effort

**Effort (one strong CUDA engineer):**

| Workstream | Engineer-weeks |
|---|---:|
| GGUF loader, tensor mapping, dequant validation | 1.5–2.5 |
| Quantized GEMM/GEMV (MMQ/MMVQ port) | 3–5 |
| MLA decode kernel (576/512, split-K, fused q8/q4) | 4–6 |
| MLA prefill kernel | 3–4 |
| MoE router + SwiGLU + shared expert | 2–3 |
| Paged KV + CUDA graphs | 2–3 |
| Parity harness + debugging | 1.5–2 |
| Integration/CLI | 1–2 |
| **Dense total** | **15–22** |
| Block-sparse indexer + validation | 3–5 |
| Audio encoder/adapter integration | 2–4 |
| **Total** | **20–31** |

With two engineers, ~10–16 calendar weeks for dense + sparse, assuming the attention kernel lands.

**Ranked risk list:**

1. **576-d attention kernel efficiency (highest).** 100 KB shared / 64K registers is tight; if you only reach 20–25% of peak, decode is 15–20 ms/token and prefill is 100+ s. Mitigation: start from llama.cpp’s `ncols=64` MMA config; decode-first; use split-K; measure before optimizing.
2. **q4 KV quality on audio.** MLA forces q4/q4; q4_0 quantizes the 64-d RoPE slice too. Temporal grounding may degrade more than text. Mitigation: q8 default; keep `k_pe` FP16; task-gate q4.
3. **45k prefill latency.** Dense is 60–90 s; that is unacceptable for interactive use if repeated. Mitigation: prefill once per recording and persist KV; block-sparse 12.5–25%; chunked prefill.
4. **MoE decode efficiency.** Four tiny GEMVs per layer at batch 1 are latency-bound; the 0.92 GB active-weight estimate can degrade to 30% bandwidth. Mitigation: fuse gate+up; group tokens by expert for batch>1; use MMVQ.
5. **Fused q8/q4 dequant.** llama.cpp’s 576 MMA path upcasts KV to F16 first; if you copy that, q8/q4 saves capacity but not bandwidth. Mitigation: custom dequant-in-attention; measure.
6. **RoPE/YaRN parity.** Interleaved pairs + mscale² + 1/sqrt(192) are three easy-to-miss details. Mitigation: unit-test RoPE against HF for positions 0, 100, 4096, 45k; compare q/k before attention.
7. **GGUF conversion of the wrapper model.** The HF checkpoint is a wrapper; convert the `decoder/` subfolder, not the whole model. The converter supports `DeepseekV3ForCausalLM`; verify tensor names and the `kv_b` split.
8. **Q3_K_XL per-tensor mix.** It is not a uniform type; the loader must support Q3_K/Q4_K/Q6_K/Q8_0 in one file. Mitigation: `gguf-dump` and a type-generic dequant path.
9. **CUDA graph capture with dynamic shapes.** Host sync in MoE routing or block-table updates will break capture. Mitigation: device-side top-k, fixed buffers, max grid + early exit.
10. **Triton AOT integration.** Low risk if used only for the indexer; avoid for attention.

---

## 6. References

**Model and paper**

- GigaChat Audio config: https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B/blob/main/config.json
- Decoder config: https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B/blob/main/decoder/config.json
- Model card / throughput: https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B
- GigaChat Audio paper: https://arxiv.org/html/2607.10387v1 (§3.1 160 ms frame rate and 60 s markers; §4.1 mIoU 53.8→14.2 without anchors)
- DeepSeek-V2: https://arxiv.org/html/2405.04434v5 (§2.1, eq. 37–47, Table 1)
- DeepSeek-V3.2: https://arxiv.org/html/2512.02556v1 (§2.1, eq. 1–4)

**llama.cpp**

- DeepSeek2 graph: https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek2.cpp#L413
- Converter / kv_b split: https://github.com/ggml-org/llama.cpp/blob/master/conversion/deepseek.py#L440
- MLA has no V cache: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp#L230
- MLA K/V same type + block-size check: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp#L4328
- 576 MMA kernel: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu#L377
- 576/512 tile config: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh#L77
- Quantized KV upcast to F16 in MMA path: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu#L770
- K-quant structs: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h#L316
- MMQ int8: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh#L84
- MMVQ dp4a: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu#L42
- RoPE type map: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp#L3646
- KV quant accuracy discussion: https://github.com/ggml-org/llama.cpp/discussions/23470

**Kernels / hardware**

- FlashMLA old requirements: https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md#L65
- FlashMLA current requirements: https://github.com/deepseek-ai/FlashMLA/blob/main/README.md#L75
- FlashMLA WGMMA/TMA deep dive: https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/docs/20250422-new-kernel-deep-dive.md
- FlashAttention-2 head dim ≤256: https://github.com/Dao-AILab/flash-attention/blob/main/README.md#L145
- FlashAttention-3 Hopper-only: https://github.com/Dao-AILab/flash-attention/blob/main/README.md#L31
- Ada tuning guide (100 KB SM, 99 KB block, 64K regs): https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- CUDA compute capabilities: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- RTX 4060 Ti specs: https://getdeploying.com/gpus/nvidia-rtx-4060-ti
- CUDA driver module API: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html
- Triton compile API: https://triton-lang.org/main/python-api/generated/triton.compile.html

**Sparse attention**

- NSA: https://arxiv.org/html/2502.11089v2
- MoBA: https://arxiv.org/html/2502.13189v1
- DSA reference inference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L439

**Quantization**

- Unsloth Dynamic GGUF docs: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
- llama.cpp ftype enum: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h#L130

---

### Final recommendation

Build **dense absorbed MLA first**, with **q4_K_M weights and q8_0 KV**, a custom 576/512 split-K decode kernel, llama.cpp’s 576 MMA config as the prefill reference, and a parity harness against llama.cpp `deepseek2`. Target **50–90 tok/s at 45k decode** and **60–90 s one-time prefill**; cache the audio KV per recording so the 45k prefill is paid once. Add **q4 KV** only if it passes the audio task gates. Add **block-sparse top-12.5–25%** only after dense parity, with the explicit acceptance thresholds above. Do not attempt DSA/NSA/MoBA without training, and do not stream/evict KV for 120-minute audio; the full 45k cache is small enough and the temporal markers are essential.
