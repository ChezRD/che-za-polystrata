# Qwen3.8-27B: a dense `qwen35` model on `strata-poly`

Qwen3.8-27B is the dense member of the Qwen3.5/3.6/3.8 layer design: the same gated delta net, the same gated
attention every fourth layer and the same multi-axis rotary embedding as Qwen3.6-35B-A3B, with **one dense SwiGLU
FFN per layer instead of routed experts**. It runs on `strata-poly` as the `qwen35` family, in the same engine class
as Qwen3.6 (`Qwen35`), because nothing but the FFN differs.

The file this page was checked with: `Qwen3.8-27B-UD-IQ3_XXS.gguf` (10,934,860,704 B, Unsloth's UD recipe), with
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf` beside it.

## The shape, from the file's own header

`tools/gguf_reader.py` reads these keys and 866 tensors; not one tensor is an expert tensor.

| key | value |
| --- | --- |
| `general.architecture` | `qwen35` (dense; `qwen35moe` is Qwen3.6/Ornith) |
| `block_count` / `nextn_predict_layers` | 65 = 64 trunk layers + 1 MTP block |
| `embedding_length` | 5120 |
| `feed_forward_length` | 17408 — the **dense** FFN's width |
| `attention.head_count` / `head_count_kv` | 24 / 4 |
| `attention.key_length` / `value_length` | 256 / 256 |
| `full_attention_interval` | 4 → **16 attention layers, 48 GDN layers** |
| `rope.dimension_count` / `rope.freq_base` | 64 / 1e7 |
| `rope.dimension_sections` | `[11, 11, 10, 0]` |
| `ssm.conv_kernel` / `state_size` / `group_count` | 4 / 128 / 16 |
| `ssm.time_step_rank` / `inner_size` | 48 / 6144 |
| vocabulary | 248320 |

There is no `expert_count`, no `expert_used_count`, no `expert_feed_forward_length` and no
`expert_shared_feed_forward_length`, no `hyper_connection.*` and no `attention.indexer.*`: the family's guard requires
the keys a dense file has and the geometry leaves every MoE width at 0.

Per-layer tensors: `attn_norm`, `post_attention_norm` (the FFN's norm), then `attn_qkv` + `attn_gate` + `ssm_*` on
the 48 GDN layers and `attn_q` (query and output gate in one matrix) + `attn_k` + `attn_v` + `attn_output` +
`attn_q_norm` + `attn_k_norm` on the 16 attention layers, and `ffn_gate` / `ffn_up` / `ffn_down` on all 64 and on
the MTP block. `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` is the MTP block. Globals: `token_embd`,
`output`, `output_norm`.

The types are IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS,
and every one is in `native_mmvq_supported` (`src/kernels/cuda/native_mmvq.cu`) and in `STRATA_POLY_MMQ_TYPES`
(`src/poly/mmq.cu`), so no dense matrix is widened to BF16 on its way into VRAM.

## What the engine does differently for it

The dense branch is three places and no kernels:

1. **The family.** `ModelFamily::Qwen35Dense` (`include/strata/artifact/model_family.hpp`) maps `qwen35`, and
   `include/strata/core/family_layout.hpp` reads `feed_forward_length` into `ModelGeometry::n_ff_dense` and zeroes
   `n_expert`, `n_expert_used`, `n_ff` and `n_ff_shexp`. Its layer table names `ffn_gate.up/down` where a MoE layer
   has seven tensors.
2. **The shape.** `Engine::load_shape()` keeps the three expert widths at 0 for a dense file and puts
   `feed_forward_length` in `n_ff_`. That is the number `Engine::shared_scratch()` sizes the FFN buffers from
   (`n * max({n_ff_, n_ff_shexp_, 2*n_ff_exp_})`) and `q_max` sizes the activation quantization from; a dense layer
   whose `n_ff_` is 0 refuses to load rather than run a 17,408-wide SwiGLU into a zero-length buffer.
3. **The FFN.** `Qwen35::load_layer` loads the three matrices instead of `load_moe`, and `Qwen35::layer_head` calls
   `dense_ffn()` — `lin(gate)`, `lin(up)`, `swiglu_clamp`, `lin(down)` — where a MoE layer calls `moe_router`. The
   shape is Flint's (`src/poly/flint.cpp`) and LFM2.5's, unchanged.

Everything else is Qwen3.6's code running unchanged: the GDN recurrence, the gated attention, the k8v4 and q8_0 KV
stores, the M-RoPE positions, the MTP block, the sequence state and parking, vision and the serve protocol.

### The attention entry points were not touched

`Qwen35::attn_store`, `Qwen35::attn` and `Qwen35::attn_rows` are byte-identical to the Qwen3.6 path; the dense branch
does not reach into them. That matters for LSA (below): the selection half of the attention has exactly one place to
change and it is not two.

### No expert cache, and the reporting says so

A dense model has `n_moe_ == 0`. `all_cached()` is then `0 == 0`, which is true, and that would read as "every
expert is in VRAM". The load line therefore prints `dense (no routed experts, no expert cache)` instead of
`N of N routed experts` (`src/poly/main.cpp`, `Engine::has_experts()`). The expert cache, its staging, the CPU
experts, `fill_expert_cache()`, `card_moe()` and `moe_rows()` are all reached only through `Layer::moe`, which a
dense layer never sets, so they stay inert; `route`, `route_card`, `route_block` and `moe_shared` throw rather than
return what a zero expert count would leave in their arrays.

## VRAM: what fits on a 16 GB card

The weights are 10.93 GB and there is no expert cache to fall back on: either the file and the cache fit, or the
load fails in `dmalloc`. Head width 256, 4 KV heads, 17 stacks (16 attention layers + the MTP block). The @256K
column is what the *cache* would cost there, not what this card holds — see the next section for what fits.

| KV format | bytes a token a layer | a token (17 stacks) | @32K | @128K | @256K |
| --- | ---: | ---: | ---: | ---: | ---: |
| k8v4 | 1,664 | 28,288 | 0.86 GiB | 3.45 GiB | 6.91 GiB |
| q8_0 | 2,176 | 36,992 | 1.13 GiB | 4.52 GiB | 9.03 GiB |
| fp16 | 4,096 | 69,632 | 2.12 GiB | 8.50 GiB | 17.00 GiB |

The recurrent state does not grow with the context: 48 GDN layers × (3 × conv channels + 48 × 128 × 128) floats =
**156.9 MB**, held as fp32 today (144 MiB a session; the fp16 follow-up halves it to 72 MiB, and
`docs/QUANT_MICROBENCH.md` is why fp16 and not int8).

On the RTX 4060 Ti this was measured on (16,380 MiB), 10.93 GB of weights + 6.91 GiB of k8v4 KV ≈ **17.8 GiB** does
not fit at 256K. Measured ceilings are in [The context this card serves](#the-context-this-card-serves).

### The KV cache is two formats here, not one

k8v4 is 1,664 B a token a layer and q8_0 is 2,176. `docs/QUANT_MICROBENCH.md` measures the pair: k8v4's store is 28.4%
faster and its read 18.5% slower at 256K (141.5 GB/s against q8_0's 4.96 ns a cell). That is a choice between
footprint and read speed, and both are supported here; neither is "the" cache.

k8v4 is free on this geometry. `head_dim` 256 is one of the six compiled sizes of the values' Hadamard rotation,
`n_kv` is constrained nowhere, and the decode kernel takes a group of 6 heads (`poly_fattn_test k8v4 --heads 24`,
24 query heads over 4 KV heads).

## What loads, and what it produces

Measured on the RTX 4060 Ti this page's numbers come from, `strata-poly --model <the IQ3_XXS file>`:

```
strata-poly: loaded in 1.4 s; dense (no routed experts, no expert cache), 8 CPU threads
strata-poly: k8v4 KV cache: 1664 bytes a row (K 1088, V 576)
```

The geometry the engine derives from the file, printed by `model_family_test --real <file>`:

```
family Qwen3.8-27B: layers 64 (+1 MTP), hidden 5120, attention 24/4 x 256 every 4, GDN K 16 V 48 x 128
   conv 10240 (4), MoE 0 experts top-0 FF 0 shared 0, hc 0, indexer 0, vocab 248320
layout: ok (866 tensors checked)
```

**Token equality with llama.cpp** (`llama.cpp` at `b1049110c`, built CUDA `sm_89`, greedy, the same IQ3_XXS file,
the same prompt token ids). Two prompts, both identical for every token:

| prompt (token ids) | llama.cpp | `strata-poly` |
| --- | --- | --- |
| `814 20139 303 799 11316 3069 279 12515 7701 6105 2261 279 1834 321 2438 506 41564 13` | `271 760 12515 7701 6105 2261 279 1834 1521 22602 6105 89661 314 37728 43000 777 13057 1472 279 16078 318 28232 60845 69377 681 1345 424 10263 2438 506 41564 1521` | identical, 32/32 |
| `7734 264 12654 709 421 16915 287 264 886 11 440 264 4479 889 13` | `271 71093 12305 198 727 9637 3773 1104 25 592 8 1411 592 25 198 262 4071 44174 279 2574 886 13 271 262 17167 25 198 285 274 25 561 886 310 9637 13 271 262 5019 25 198` | identical, 40/40 |

**The MTP block also matches.** The same prompt with the draft block on (`--window 4` and `--window 8`) produces the
same 32 ids as `--no-draft`, so a guess never changes a token, only the speed.

## The context this card serves

Measured on the RTX 4060 Ti (16,380 MiB, ~2.9 GiB held by the desktop), `--kv-whole` and `--vram-reserve-mib 256`,
the largest `--max-context` that loads:

| KV | largest context that loads | KV a layer a token | KV at that context |
| --- | ---: | ---: | ---: |
| k8v4 | **61,440** | 1,664 B | 1.66 GiB |
| q8_0 | **47,104** | 2,176 B | 1.66 GiB |
| fp16 | **26,624** | 4,096 B | 1.77 GiB |

The three ceilings land within 0.11 GiB of each other, which is the check that they are the same limit: what is left
of the card after the 10.9 GB file and the scratch. A token costs `17 x` those bytes, because there are **17
attention stacks** (16 attention layers + the MTP block): 28,288 B, 36,992 B and 69,632 B a token.

**No LRU eviction, no spill: a context above these does not load.** `dmalloc` fails with `no VRAM for scratch` or
`no VRAM for the attention cache`, which is the honest answer for a dense model with no expert cache to give back.
`--prompt-batch` above 512 also costs enough scratch to lose a few thousand tokens of context (the 61,440 figure is
with `--prompt-batch 512`); the default of 1024 balances the other way.

## What is not tested

- **256K.** It does not fit this card: 10.93 GB of weights + 6.91 GiB of k8v4 KV ≈ 17.8 GiB against 16.0. LSA does
  not change that — it cuts the compute of finding keys, not the memory of storing them (see below).
- **The MTP block's own K and V at 256K**, for the same reason.
- **HIP, Metal and SYCL.** Only the CUDA `sm_89` build was compiled and run. The dense branch is engine code with no
  kernels of its own, so it should carry, but that is an inference.
- **Pictures.** The file set includes `mmproj-F16.gguf` and `Qwen35` has the vision path; no image was fed.
- **Multi-card `--layer-split`**, batching (`--batch`), and parking (`--conversation-cache-mib`).
- **The subtree for anything but IQ3_XXS.** Only the one quant was loaded. The type census above says nothing widens
  to BF16, which is a header fact, not a run.
- **A speed baseline.** The numbers a page like this should carry (prefill and decode against a known reference on
  the same card) wait for the dense family to be merged first; there is nothing yet to compare against.

## LSA and the 256K context

LSA is the paper's flagship case on **this** model: arXiv 2610.42147v2 §4.1 demonstrates D-LSA on "a model with full
attention (**Qwen-3.8-27B**)" and on GLM-5.3-Flash's sparse attention, and reports **72.4× prefill and 42.5×
decode** per attention layer at 10M tokens over full attention. Its accuracy on this model holds: D-LSA (Flat)
"stays within 5 points of baseline Full Attention up to 1M tokens and **exceeds** it at 10M" — RULER 63.9 against
59.9, PaperTrail 29.1 against 19.8. The variant that needs no training is validated here too: "D-LSA
(**Training-Free**), without any training, stays close to D-LSA (Flat) on Qwen-3.8-27B, but **trails** it on
GLM-5.3-Flash."

**LSA cuts the cost of finding keys, not of storing them.** The paper's own words: *"It reduces the cost of finding
the keys, not of storing them: the KV cache still grows linearly."* The cache at 256K stays 6.91 GiB (k8v4) or
9.03 GiB (q8_0), so 10.93 GB of weights + 6.91 GiB of KV is still ~17.8 GiB and **LSA does not make 256K fit a 16 GB
card**. What it removes is the attention's arithmetic: at 256K, 16 attention layers over all 262,144 keys with 24
query heads of 256 values is **~13.5 PFLOP** (2·T²·d·H). The paper's Qwen bucket setting is `k = 4,096` long-range
keys plus `n_sink = 4` and `n_w = 512`, so a query attends to at most **4,612 of 262,144 keys = 1.76%**.

The paper's parameters for this model (Table 2, bucket columns): `k = 4,096`, `p = 1,024`, `n_sink = 4`,
`n_w = 512`, `d_idx = 64`, `n_pool = 4`, `n_cluster = 512`, `n̄_b = 8`, `n_b^max = 16`. (The `k = 2,048, p = 512`
column is the *block-sparse* setting and GLM's sub-1M one; it is not this model's.)

**The index reads the RoPE-free part of the key, and this model has it by construction.** The paper: "whose keys and
queries carry RoPE on a quarter of their dimensions, the projections read only the RoPE-free part."
`rope.dimension_count = 64` of `key_length = 256` is exactly one quarter, and `head_norm_rope_kernel`
(`src/poly/kernels.cu`) leaves every dimension at or past `n_rot` unrotated — `if (i >= n_rot)` copies the value
without a turn. So the RoPE-free slice is the *tail*, **dims `[64, 256)`**, 192 values, and `d_idx = 64` is a
further projection of it. (It is not `[64, 128)`: that would be half of a quarter.)

Index overhead, from the paper's parameters (derived arithmetic, not measured): index keys are one per `n_pool = 4`
tokens at `d_idx = 64` fp16 = 8.39 MB a layer at 256K; centroids are one per bucket of ~8 pool keys = 1.05 MB; the
position table is 1.05 MB. **~10.5 MB a layer, ~168 MB over the 16** — an order of magnitude below the KV cache it
indexes. The paper reports 246 MiB a layer at 10M tokens on GLM-5.3-Flash, which is consistent once the context is
38× larger.

The Ward clustering gate is arithmetic on this model: 12.66 µs a 512-key block × 128 blocks a layer × **16 layers =
25.9 ms** of a 256K prefill (the Flash geometry's 12 layers gave 19.45 ms). It clears the 20 µs-a-block bar at 0.63×.

**Where an `--lsa` flag would land.** `Qwen35::attn` (`src/poly/qwen35.cpp`) is the one function that hands the
attention its key set: it takes `pos`/`n` (a contiguous prefix, through `attend`/`attend_batch`/`fattn_*`/`fdec_*`)
and computes `scale`, the k8v4 unrotate and the output gate around it. LSA's hook is a per-query index list `J_i`
in place of that prefix, and the shape it needs is: one `int32_t` list per query row of at most
`n_sink + k + n_w = 4,612` entries, plus the scratch the attention already takes (`score_`, `chunk_`). The kernels
that read keys (`attn_score_*`, `attn_value_*`) index the cache by position, so a gather through such a list is a
kernel change and not a call-site change. `attn_store`, `attn_rows` and the k8v4 store/unrotate branches need no
change to their read path: LSA changes which keys are *read*, never which are *written*. `attn_store` does gain one
extra store — one 64-value fp16 index key a token a layer, ~7.5% of the k8v4 row — which is an added branch beside
the existing `k8v4`/`q8_0` ones and not a change to them.

**The distilled map for this exact checkpoint is available.** It ships as a 6.30 MB release asset of
`Prefrontal-AI/latent-sparse-attention` (`d-lsa-weights-v1`), holding `W_Q_idx` and `W_K_idx` per attention layer as
`[768, 64]` matrices. That means the variant the paper measured at 72.4×/42.5× is the one we can run, not the
training-free fallback the paper says trails it. The details — the exact tensors, the shape checks, the licence
question, and the one accuracy check that should gate the port — are in [LSA-DENSE.md](LSA-DENSE.md).
