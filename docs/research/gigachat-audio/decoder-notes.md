# GigaChat Audio decoder: what this backend assumes

Files: `src/poly/gigachat.{hpp,cpp}`, `src/poly/mla_test.cu`, the MLA kernels in
`src/poly/kernels_mla.cu`, their declarations in `include/strata/poly/kernels.hpp`, and the
`deepseek2` / `gigachat_audio` branch in `src/poly/engine.cpp`.

## Tensors read

Metadata (`deepseek2.*`, or `gigachat_audio.*`):

- `attention.head_count`, `attention.kv_lora_rank`, `rope.dimension_count`
- `attention.key_length_mla` (falls back to `attn_k_b`/`attn_q` shapes), `attention.value_length_mla`
  (falls back to `attn_v_b`/`attn_output` shapes)
- `attention.q_lora_rank` (optional; 0 means the direct query)
- `feed_forward_length`, `leading_dense_block_count` (layer 0 is detected by its missing router)
- `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_count`
- `expert_weights_scale`, `expert_weights_norm`, `expert_gating_func` (sigmoid)
- `rope.freq_base`, `rope.scaling.*` (YaRN type, factor, original context, beta fast/slow,
  `yarn_log_multiplier`)

Per layer:

- `attn_norm.weight`, `ffn_norm.weight`
- `attn_q.weight` (or `attn_q_a.weight`, `attn_q_a_norm.weight`, `attn_q_b.weight`)
- `attn_kv_a_mqa.weight`, `attn_kv_a_norm.weight`
- `attn_k_b.weight` + `attn_v_b.weight`, or the legacy fused `attn_kv_b.weight`
- `attn_output.weight`
- `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` (layer 0)
- `ffn_gate_inp.weight`, `exp_probs_b.bias`, `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`
  (or `ffn_gate_up_exps`), `ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp`
- `output_norm.weight`, `output.weight`

## Cache and arithmetic

- One cache row is the 512-value latent then the 64-value rope key.  Values are `lat`; the first 512
  values of a key are its value too.  Nothing is materialised per head.
- fp16 row = 1,152 B; Q8_0 row = 612 B (512 codes + 16 scales + 64 codes + 2 scales).
- The score is the absorbed no-rope dot (512 values) plus the rope dot (64 values), scaled by
  `mscale^2 / sqrt(192)` with YaRN `mscale = 0.1*ln(64)+1 = 1.4159`, i.e. 0.1447.
- RoPE is the consecutive-pair (interleaved) form on the last 64 values of a 192-value query head and
  on the 64-value key.  `mla_test` checks the kernel against a host loop; it does not check a real
  model's logits.
- The legacy fused `attn_kv_b` is read as `[n_head, qk_nope + v_head_dim, kv_lora_rank]` and the
  absorption/value projections run in `mla_absorb_fused()` / `mla_vb_fused()` (F32 or BF16; a
  quantized fused tensor is converted to BF16 at load).

## Graph safety

`Layer::fixed = true`; `dsv4_rope(..., at)`, `store_f16_at`/`store_q8_at`, and
`mla_attend(..., at, room)` read the position on the card, and `head_key()` is `room_key(n)`.
The MLA attention uses two `cublasSgemmStridedBatched` calls per pass.  The engine's cuBLAS handle is
created and warmed at load, but a model-scale run has not verified that cuBLAS captures these two
exact products; if a recorded graph fails, `--no-graphs` is the fallback and the next step is a
custom QK/PV kernel for `mla_attend`.

## What is not done

- This backend implements `--kv fp16` and `--kv q8_0`. It has no `--kv q4_0`.
- Without `--mla-fused`, `mla_attend` copies every key a token sees into a float buffer and runs cuBLAS.
  That is the simple correct path. `--mla-fused` is the absorbed kernel for latent 512, rope 64 and 32
  heads; a decode step uses it when `n` is at most 8.
- The key buffer is capped at 128 MiB for a pass over the full context, so `--max-context` above about
  58k is refused rather than tiled. `assign_scratch` keeps that cap even when `--mla-fused` does not
  read `keys_`.
- MTP / `nextn` draft blocks are refused (`--no-draft`). This decoder has none.
- The quant recipe and the measured GGUF sizes are in `05-quantization-recipe.md`. The CPU encoder is
  `tools/audio/`.
