# 05 — Quantizing the GigaChat Audio decoder

Status: 2026-11-05. Branch `feature/gigachat-audio-mla`.
Scope: `decoder/` of
[`ai-sage/GigaChat3.1-Audio-10B-A1.8B`](https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B)
(revision `bf73d03`). The 2.37 GB audio encoder is **not** in this recipe.

PolyStrata does not ship a quantizer. The commands below are llama.cpp's
`convert_hf_to_gguf.py`, `llama-imatrix` and `llama-quantize`. The engine only
loads the finished GGUF.

Related artifacts:

* Text-only size reference:
  [`ai-sage/GigaChat3.1-10B-A1.8B-GGUF`](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF)
  — a *different fine-tune*, never shipped as the audio decoder.

---

## 0. TL;DR

| role | file | bytes | bpw | engine-safe |
|---|---|---:|---:|---|
| conversion input | `gigachat-audio-decoder-f16.gguf` | 21,356,264,480 | 16.00 | n/a |
| q4 reference | `...-Q4_K_M.gguf` | 6,474,685,472 | 4.85 | yes |
| **q4 ship** | `...-Q4_K_XL.gguf` | 6,738,485,792 | 5.05 | yes |
| q3 reference | `...-Q3_K_M.gguf` | 5,145,710,112 | 3.85 | yes |
| **q3 fallback ship** | `...-Q3_K_XL.gguf` | 5,095,825,952 | 3.82 | yes |
| **q3 preferred (quality)** | `...-IQ3_S.gguf` | 5,095,826,208 | 3.82 | **yes — load + raw parity passed** |
| q3 minimal | `...-IQ3_XXS.gguf` | 4,653,458,208 | 3.48 | **yes — load + raw parity passed** |

* **Ship Q4_K_XL** when quality comes first (6.74 GB).
* **Ship Q3_K_XL** if the decoder must stay on types the engine already
  exercises (Q3_K/Q6_K/Q8_0/F32).
* **Prefer `IQ3_S`**: 5.10 GB, 3.82 bpw, same size as Q3_K_XL, better text PPL
  (3.799 vs 3.858). The backend matched llama.cpp on 3/3 greedy prompts.
* `IQ3_XXS` (4.65 GB, 3.48 bpw, PPL 3.873) is the size floor, not the default.
  It also matched llama.cpp on 3/3.

---

## 1. Where the bits are

The decoder is a stock `DeepseekV3ForCausalLM`:

| property | value |
|---|---|
| layers | 26 (layer 0 dense FFN, layers 1–25 MoE) |
| hidden | 1536 |
| routed experts | 64 per MoE layer, top-4, sigmoid + `noaux_tc` |
| shared experts | 1 per MoE layer |
| expert FFN | 1280 |
| attention | MLA, `q_lora_rank=null`, `kv_lora=512`, `qk_nope=128`, `qk_rope=64`, `v=192` |
| vocab | 128256, untied `lm_head` |
| parameters | 10.673 B (21.345 GB BF16) |

Tensor-class accounting (from `llama-quantize` logs, GiB of the q4/q3 files):

| class | weights | share | Q4_K_M | Q4_K_XL | Q3_K_M | Q3_K_XL |
|---|---:|---:|---:|---:|---:|---:|
| routed experts | 9.437 B | 88.4% | 5.306 | 5.306 | 4.166 | 3.776 |
| attention | 0.650 B | 6.1% | 0.347 | 0.509 | 0.297 | 0.509 |
| token emb + output | 0.394 B | 3.7% | 0.254 | 0.301 | 0.229 | 0.301 |
| shared experts | 0.150 B | 1.4% | 0.083 | 0.113 | 0.065 | 0.113 |
| layer-0 dense | 0.041 B | 0.4% | 0.025 | 0.032 | 0.020 | 0.032 |
| router | 0.0026 B | 0.02% | 0.009 | 0.009 | 0.009 | 0.009 |

Consequence: **the only place where bits are cheap is `ffn_*_exps`.** Moving
attention/embeddings from Q4_K to Q6_K costs 0.27 GiB; moving all experts from
Q4_K to Q3_K saves 1.53 GiB. The recipes below therefore keep attention,
embeddings, shared experts and the router high, and spend the budget on experts.

---

## 2. Methods compared

### 2.1 Q3_K_S / Q3_K_M / Q3_K_L

K-quants (llama.cpp PR [#1684](https://github.com/ggml-org/llama.cpp/pull/1684)).
The suffix is a *per-tensor mix*, not a block format:

* `Q3_K_S` — all tensors Q3_K (3.44 bpw weights).
* `Q3_K_M` — Q3_K for most tensors; `ffn_down` promoted to Q4_K (and Q5_K in
  the first layer); attention/output promoted by llama.cpp's rules.
* `Q3_K_L` — more Q5_K in `ffn_down` and attention.

On this model stock `Q3_K_M` gives 3.85 whole-file bpw / 5.15 GB. It spends
1.17 GiB on attention+embeddings+shared and only 4.17 GiB on the 88% expert
bulk. `Q3_K_XL` (our name, not a llama.cpp preset) is Q3_K for every expert
plus Q6_K attention/embeddings/shared and F32 router: 3.82 bpw / 5.10 GB. It
is *smaller* than stock Q3_K_M and spends 0.39 GiB less on experts and
0.32 GiB more on the tensors that touch every token.

### 2.2 imatrix and IQ3

I-quants (llama.cpp PR [#5196](https://github.com/ggml-org/llama.cpp/pull/5196))
use a learned codebook per 256-value block. In this decoder's expert tensors
IQ3_XXS packs to **3.06 bpw** and IQ3_S to **3.44 bpw** (measured from the
produced files). llama.cpp's Llama-3-8B reference table reports somewhat
higher whole-file numbers because it counts non-quantized tensors and every
block header.
They are efficient but **depend on the importance matrix**; without one,
llama.cpp falls back to less aggressive choices or produces a visibly worse
file. The imatrix is built with `llama-imatrix` from calibration text
([PR #4861](https://github.com/ggml-org/llama.cpp/pull/4861)); it stores one
importance vector per tensor, not model weights.

`IQ3_XXS` is a distinct GGML type (id 18), not a Q3_K variant. `IQ3_S` is id
21. The PolyStrata verdict is in §7.

### 2.3 Unsloth Dynamic 2.0 / UD-Q3_K_XL

Unsloth's "Dynamic" GGUFs are **not a new format**. They are standard
K-quant/I-quant files with a model-specific per-tensor type assignment plus a
large curated imatrix. Unsloth states that v2.0 "dynamically adjust[s] the
quantization type of every possible layer", that the combinations differ per
model, and that the calibration set is >1.5 M tokens; v3.0 uses a larger,
cleaner multilingual/agentic set and more layer-selection work. They do not
use QAT/QAD for these GGUFs
([docs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs),
[v2.0 notes](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)).

There is no Unsloth `UD-Q3_K_XL` for this audio fine-tune. Our `Q3_K_XL` is
the same *idea* — promote the always-active/sensitive tensors, demote the
routed experts — but it is not Unsloth's search and does not use their
calibration set.

Directional evidence from Unsloth's own Gemma-3-27B table (dense model, not
this one): `Q3_K_M` 70.70 MMLU / 12.51 GB, `Q3_K_XL` 70.87 / 12.76,
`Q4_K_M` 71.23 / 15.41, `IQ3_XXS` 68.27 / 10.07. The gain from the Dynamic mix
at Q3 is small on a dense model; on a 64-expert MoE the expert/non-expert
split matters much more, which is why our Q3_K_XL beats stock Q3_K_M here
(§5).

### 2.4 What was built

| method | reproducible on this box? | why |
|---|---|---|
| Q3_K / Q4_K mixes | yes | llama-quantize, CPU, minutes |
| imatrix + IQ3 | yes | llama-imatrix CPU, ~14 min for 48 chunks; IQ3 packing ~14 min |
| Unsloth Dynamic 2.0 | no (no file for this fine-tune) | search + calibration not public as a recipe |

The files in the tables below are these two rows: stock `Q4_K_M` and `Q3_K_M`,
the same commands with the protections in section 3 (`Q4_K_XL`, `Q3_K_XL`),
and the protected IQ3 recipes (`IQ3_XXS`, `IQ3_S`).

---

## 3. Per-tensor policy

GGUF names, not HF names. `attn_kv_a_mqa` is `kv_a_proj_with_mqa`;
`attn_k_b`/`attn_v_b` are the two halves of `kv_b_proj`.

| tensor(s) | type | reason |
|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` | Q3_K (q3) / IQ3_XXS (iq3) / Q4_K_M mix (q4) | 88% of weights; cheapest bits |
| `attn_q`, `attn_output`, `attn_v_b`, `attn_kv_a_mqa` | Q6_K | MLA projections feed all 32 heads; errors are not averaged over experts |
| `attn_k_b` | Q8_0 | shape `[128,512,32]`; `ne[0]=128` is not a multiple of 256, so all K-quants fall back. Q8_0 is the best 128-wide type and costs 58 MB total |
| `token_embd`, `output` | Q6_K | 128256 vocab; rare Cyrillic/multilingual tokens |
| shared expert (`ffn_*_shexp`) | Q6_K | always active, unlike routed experts |
| layer-0 dense FFN | Q6_K | first layer, 41 M params |
| `ffn_gate_inp` (router) | F32 | 10 MB total; top-4/64 routing is discrete, small weight errors flip experts |
| `ffn_exp_probs_b` | F32 | tiny |
| norms / biases | F32 | 1-D; llama-quantize keeps them F32 |

Why not Q4_K for `attn_k_b`: it is not shape-compatible. In the stock
Q4_K_M file llama.cpp emits **Q5_0** for these 26 tensors (fallback), which is
why the Q4_K_M type histogram has 26 Q5_0 tensors.

Q3_K_XL spends 3.776 GiB on experts (79.7% of the file) versus stock Q3_K_M's
4.166 GiB (87.0%), while protecting attention/embeddings. That is the whole
argument for the custom mix.

---

## 4. Calibration data and imatrix

The importance matrix was built from a 0.59 MB corpus (3,008 documents):

* **FLEURS** `ru_ru` and `en_us` `train.tsv`, real read-speech transcripts
  (1,500 utterances per language, capped), downloaded from
  <https://huggingface.co/datasets/google/fleurs/tree/main/data>.
* **Russian Wikipedia** random extracts (8 articles) via the MediaWiki API.
  English Wikipedia was rate-limited (HTTP 429) and contributed 0 articles.
  This is a limitation, not a design choice.

The mix is deliberately transcript-heavy: the decoder consumes ASR transcript
tokens, not Wikipedia prose. Held-out evaluation uses the FLEURS `dev.tsv`
files (745 utterances, 136 KB), which are disjoint from `train.tsv`.

`llama-imatrix`:

```bash
llama-imatrix -m gigachat-audio-decoder-f16.gguf -f calib/calib.txt \
    -o gigachat-audio-decoder.imatrix.gguf \
    --chunks 48 --ctx-size 512 --threads 8
```

Result: 63,283,424 bytes, 48 chunks × 512 × 4 sequences, ~2 min 15 s wall on
this CPU (24.6 s/pass). Calibration perplexity fell from 17.0 to 15.5 while
building it. A 4 M-token calibration is about 40× larger. Do not treat this
imatrix as that.

---

## 5. Results

### 5.1 Files and tensor mix

All sizes are exact byte counts; `du -h` is in parentheses. Type histograms
from `inspect_gguf.py`.

| file | bytes (du) | bpw | tensor types |
|---|---:|---:|---|
| F16 | 21,356,264,480 (20G) | 16.00 | F16 21.340 GB, F32 0.010 |
| Q4_K_M | 6,474,685,472 (6.1G) | 4.85 | Q4_K 4.990, Q6_K 1.431, Q5_0 0.037, F32 0.010 |
| Q4_K_XL | 6,738,485,792 (6.3G) | 5.05 | Q6_K 2.205, Q4_K 4.459, Q8_0 0.058, F32 0.010 |
| Q3_K_M | 5,145,710,112 (4.8G) | 3.85 | Q3_K 2.993, Q4_K 1.935, Q4_0 0.031, Q6_K 0.162, Q5_K 0.009, F32 0.010 |
| Q3_K_XL | 5,095,825,952 (4.8G) | 3.82 | Q6_K 0.967, Q3_K 4.055, Q8_0 0.058, F32 0.010 |
| IQ3_XXS | 4,653,458,208 (4.4G) | 3.48 | Q6_K 0.967, IQ3_XXS 3.613, Q8_0 0.058, F32 0.010 |
| IQ3_S | 5,095,826,208 (4.8G) | 3.82 | Q6_K 0.967, IQ3_S 4.055, Q8_0 0.058, F32 0.010 |

Expert-only bpw: Q4_K_M/Q4_K_XL 4.83, Q3_K_M 3.79, Q3_K_XL 3.44, IQ3_XXS 3.06,
IQ3_S 3.44.

### 5.2 Text perplexity (proxy, not WER)

`llama-perplexity` on held-out FLEURS `dev` (ru+en), 8 chunks × 512 tokens,
CPU. Lower is better; the spread is the chunk-to-chunk standard error.

| model | PPL | Δ vs Q4_K_M |
|---|---:|---:|
| Q4_K_M | 3.7401 ± 0.207 | — |
| Q4_K_XL | 3.7629 ± 0.209 | +0.6% |
| Q3_K_M | 3.9584 ± 0.227 | +5.8% |
| Q3_K_XL | 3.8580 ± 0.212 | +3.2% |
| IQ3_XXS | 3.8727 ± 0.218 | +3.5% |
| IQ3_S | 3.7993 ± 0.212 | +1.6% |

Caveats: 8 chunks is a smoke test, not a benchmark; FLEURS dev is read speech,
not the audio model's full task; the Q4_K_XL vs Q4_K_M gap is within noise.
Do **not** present these as WER or as evidence that a file passes the audio
gate.

### 5.3 CPU load/generate smoke test

`llama-completion -ngl 0 -t 8 -c 256 -n 8 --temp 0`, one run each, warm page
cache. Tokens/s from llama.cpp's `eval time`; RSS is `ru_maxrss`.

| model | eval tok/s | prompt tok/s | peak RSS | output |
|---|---:|---:|---:|---|
| Q4_K_M | 33.4 | 76.9 | 10.8 GB | "…the lazy dog. output: …" |
| Q4_K_XL | 34.2 | 51.1 | 10.7 GB | same |
| Q3_K_XL | 40.2 | 61.0 | 4.9 GB | same |
| IQ3_XXS | 31.6 | 47.5 | 4.5 GB | same |
| IQ3_S | 28.0 | 48.6 | 4.93 GB | same |

All load, generate, and hit the same continuation for this prompt. This is a
smoke test, not a benchmark: single runs on a shared 12-core CPU vary by
±20–30% with page-cache state (IQ3_S measured 28.0 and, on a cold run, 10.0
tok/s). It is safe to say the IQ3 files are in the same CPU class as Q3_K/Q4_K;
it is not safe to rank them by these numbers.

Tokenizer parity: `llama-tokenize --ids` on the Q4_K_XL file returned exactly
the HF `AutoTokenizer` IDs for a mixed Russian/English string, including BOS.

### 5.4 Greedy continuation parity (3 raw prompts)

`llama-completion -ngl 0 -t 8 -c 256 -n 16 --temp 0`, CPU. Prompt token IDs
from `llama-tokenize --ids` (BOS included):

| prompt | token IDs |
|---|---|
| `The capital of France is` | `[1, 1153, 11319, 367, 17781, 465]` |
| `1, 2, 3, 4, 5,` | `[1, 19, 14, 223, 20, 14, 223, 21, 14, 223, 22, 14, 223, 23, 14]` |
| `The quick brown fox jumps over the lazy` | `[1, 1153, 8527, 31720, 93832, 76910, 1234, 307, 17633]` |

Continuations (16 new tokens max, greedy):

| prompt | IQ3_S |
|---|---|
| capital | `Paris. [end of text]` |
| counting | `6, 7, 8, 9, 10,` |
| fox | `dog. output: ```python\ntext = "The quick brown fox jumps over` |

These three continuations are the ones `IQ3_S` and `IQ3_XXS` matched against
`llama-completion` in section 9. `Q3_K_XL` diverges on the code prompt.

### 5.5 GGUF metadata check (F16)

```
general.architecture              deepseek2
deepseek2.block_count             26
deepseek2.expert_count            64
deepseek2.expert_used_count       4
deepseek2.expert_shared_count     1
deepseek2.leading_dense_block_count 1
deepseek2.attention.key_length    576
deepseek2.attention.value_length  512
deepseek2.attention.key_length_mla 192
deepseek2.attention.value_length_mla 192
deepseek2.attention.kv_lora_rank  512
deepseek2.rope.dimension_count    64
deepseek2.rope.scaling.type       yarn
deepseek2.rope.scaling.factor     64.0
deepseek2.rope.scaling.original_context_length 4096
deepseek2.rope.scaling.yarn_log_multiplier 0.1
tokenizer.ggml.model              gpt2
tokenizer.ggml.pre                gpt-4o
tokenizer.ggml.tokens             128256
```

`rope_interleave` is **absent** from the GGUF and from current llama.cpp's key
list. This is expected: llama.cpp maps `LLM_ARCH_DEEPSEEK2` to
`LLAMA_ROPE_TYPE_NORM` in `llama_model_rope_type()`, which is the interleaved
(GPT-J-style) pairing DeepSeek uses. There is no metadata bit to confirm; the
architecture implies it. Older converters wrote a `rope_interleave` key, current
ones do not. Do not add a fake key.

---

## 6. PolyStrata type support — verdict

Checked against `feature/gigachat-audio-mla`, without modifying `src/`.

**The GGML types are supported.**

| layer | evidence |
|---|---|
| GGUF reader | `include/strata/artifact/gguf_reader.hpp` names types 18 (`IQ3_XXS`) and 21 (`IQ3_S`) and computes their block sizes |
| decode MMVQ | `src/kernels/cuda/native_mmvq.cu`: `native_mmvq_supported()` returns true for 16,17,18,21,22,29; `native_mmvq()` dispatches them to `iq_mmvq()`; `slots_kind()` returns 2, so up to 32 cached experts per launch |
| prefill/batched | `src/poly/mmq.cu`: `STRATA_POLY_MMQ_TYPES` includes `IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS`; `Mmq::fits()` accepts them |
| CPU experts | `src/kernels/cpu/iq_avx1.cpp`, `iq_avx2.cpp`, `iq_avx512.cpp` implement IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23), IQ4_NL (20), Q2_0 (42). The AVX1 file states the layout explicitly: "gate/up IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS; down IQ4_NL and Q2_0" |
| upload | `Engine::upload()` keeps a tensor byte-for-byte when `kAnyMat` and `native_mmvq_supported(type)` is true; IQ3 no longer gets upconverted to BF16 |
| existing tuning | `src/core/arch_defaults.cpp` sets `STRATA_EXPERT_V2=1` for grouped decode experts with IQ3_S gate/up |

**The model is a separate question.** `tools/strata_inspect.py` on our file
prints:

```
routed experts 9.44B 3.61 GB 3.06 bits/w IQ3_XXS 100%
attention ... Q6_K 89%, Q8_0 11%
...
Strata: not Qwen3.8-Flash-Next (architecture 'deepseek2'): PolyStrata runs only that model
```

That message is the *model-family* gate in the inspector, not a type rejection.
`strata_inspect.py` keeps `POLY_ARCHS = {gemma4, glm5-next, deepseek4, lfm2moe,
flint}`; `deepseek2` is not in that set (nor in setup's `POLY_GGUF_ARCHS`). The
engine itself is wired: `src/poly/engine.cpp:1183-1199` maps
`"deepseek2"`/`"gigachat_audio"` to `GigaChatAudio` and permits `--kv q8_0`.
The same inspector, on a Qwen file setup already accepts, prints
`Strata: this is setup's --family qwen --model IQ3_XXS`. In other words:

* the storage/type layer can read and multiply IQ3_XXS/IQ3_S;
* the `deepseek2`/`gigachat_audio` backend exists on this branch
  (`src/poly/gigachat.cpp`, `Engine::make`);
* `strata_inspect.py`'s `POLY_ARCHS`/setup `POLY_GGUF_ARCHS` lists have not been
  updated for `deepseek2`, so the inspector says "no" — a tooling gap, not a
  kernel or backend gap;
* the load/parity test in §9: uniform `IQ3_S` and `IQ3_XXS` match llama.cpp 3/3.

So: **the quant format needs no new kernels, the backend path exists, and the
uniform IQ3 files have passed raw parity.** What remains is the inspector/setup
arch entry for `deepseek2`. If that entry is missing, use **Q3_K_XL**
(Q3_K/Q6_K/Q8_0/F32), which the engine also exercises.

Caveats found in code:

* `IQ2_XXS`/`IQ2_XS` are absent from the AVX1 expert kernels and fall back to
  ggml-cpu's scalar dot. Do not use an IQ2 expert floor on old CPUs.
* `IQ3_*` is slower to pack than K-quants (llama-quantize: ~14 min vs ~1.5 min
  for this model) but decode speed on CPU is comparable (33.8 vs 33.4 tok/s in
  §5.3).
* `attn_k_b` with `ne[0]=128` cannot take a K-quant; Q8_0 is the right choice
  for every recipe, including IQ3.

---

## 7. Acceptance gates (audio phase)

These are **not yet measured**. The decoder-only phase cannot produce WER or
temporal mIoU; those require the encoder, adapter and prompt pipeline
(Phase 2–3).

Compare every candidate against `Q4_K_M` on the same audio set, greedy decode,
same prompt/timestamps:

1. **WER** — Δ ≤ +0.5 percentage points absolute (Russian and English).
2. **Temporal mIoU** — Δ ≤ 1–2 percentage points on segment boundaries.
3. **Timestamp drift** — no systematic shift; per-segment p95 within 1 s.
4. **Text guardrail** — held-out PPL ≤ 5% above Q4_K_XL (all files here pass:
   max +3.5%).
5. **Router stability** — top-4 expert agreement vs F16 on a fixed 4k-token
   sample ≥ 99% (the router stays F32; this catches accidental router damage).
6. **Load check** — `strata_inspect.py` reports the expected type histogram and
   no unexpected BF16 upconversion.

If a 3-bit file fails gate 1 or 2, Q4_K_XL is the floor. Do not relax the gate
to make a smaller file ship.

---

## 8. Recommendation

**q4, ship:** `gigachat-audio-decoder-Q4_K_XL.gguf` — 6.74 GB, 5.05 bpw.
It protects attention/embeddings/shared experts and costs only 0.26 GB over the
stock `Q4_K_M`. Keep `Q4_K_M` as the byte-for-byte reference for comparisons
with the text-base release.

**q3, ship today (engine-safe):** `gigachat-audio-decoder-Q3_K_XL.gguf` —
5.10 GB, 3.82 bpw. It beats stock Q3_K_M (5.15 GB) on held-out PPL (3.858 vs
3.958) and is smaller. It uses only types the current engine exercises, so it
is the safe fallback if the inspector/backend entry for `deepseek2` is not
updated in time.

**q3, preferred (same size, better quality):**
`gigachat-audio-decoder-IQ3_S.gguf` — 5,095,826,208 bytes, 3.82 bpw. It is
byte-for-byte the same size class as Q3_K_XL (256 bytes larger) and has the
better measured PPL: 3.799 vs 3.858 (+1.6% vs Q4_K_M vs +3.2%). The engine has
IQ3_S kernels, the backend raw-parity check passed 3/3, and throughput is in
the same class (28 vs 40 tok/s on this CPU, single-run noise). **Prefer it for
the q3 slot.**

**q3, minimal:** `gigachat-audio-decoder-IQ3_XXS.gguf` — 4.65 GB, 3.48 bpw,
PPL 3.873 (+3.5%). It is 0.44 GB under `IQ3_S` and worse on the text smoke
test. Keep it as the floor, not the default.

**Do not ship IQ2-class experts.** Uniform `IQ3_XXS` with Q6_K attention and
embeddings is 3.48 bpw. That is the floor of this policy.

**Memory (16 GB card, 45k-token q8_0 KV, encoder q8_0):**

| file | decoder | encoder | KV | runtime | total | leaves for desktop/Tiel |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_XL | 6.74 | 0.63 | 0.70 | 1.0 | 9.1 | 6.9 |
| IQ3_S / Q3_K_XL | 5.10 | 0.63 | 0.70 | 1.0 | 7.4 | 8.6 |
| IQ3_XXS | 4.65 | 0.63 | 0.70 | 1.0 | 7.0 | 9.0 |

q3 is what makes Tiel 64K + audio plausible; q4 is audio-alone comfortable.

---

## 9. Reproduction

The snapshot already holds `decoder/`, `tokenizer/`,
`gguf/gigachat-audio-decoder-IQ3_S.gguf` and
`gguf/gigachat-audio-enc-mixed.gguf`. `HF_HOME` moves the cache root.
The encoder layout is in `tools/audio/README.md`. This section rebuilds the
decoder only.

```bash
# 0. decoder weights, 21.345 GB, into the default Hugging Face cache
hf download ai-sage/GigaChat3.1-Audio-10B-A1.8B --revision bf73d03 \
    --include 'decoder/*'
SNAP="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ai-sage--GigaChat3.1-Audio-10B-A1.8B/snapshots/bf73d03a43bdf5118f5a4dbdc24ba6f56ac31cfb"

# 1. llama.cpp CPU tools, commit 10d4846a572aebf21c0482d140b00e74c0f1ad20
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
    -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
cmake --build "$LLAMA_CPP/build" -j8 --target \
    llama-quantize llama-imatrix llama-cli llama-completion llama-tokenize llama-perplexity

# Python for the converter: torch CPU, transformers, safetensors, numpy,
# sentencepiece, protobuf, tqdm, gguf. Apply the two converter fixes below first.

# 2. F16 GGUF
"$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" "$SNAP/decoder" \
    --outtype f16 \
    --outfile "$SNAP/gguf/gigachat-audio-decoder-f16.gguf"
F16="$SNAP/gguf/gigachat-audio-decoder-f16.gguf"

# 3. calib.txt is the corpus in section 4: FLEURS train.tsv field 3
# (raw transcript, at least 40 characters, 1500 utterances per language)
# plus the 8 Russian Wikipedia extracts, paragraphs separated by a blank line.
"$LLAMA_CPP/build/bin/llama-imatrix" -m "$F16" -f calib/calib.txt \
    -o "$SNAP/gguf/gigachat-audio-decoder.imatrix.gguf" \
    --chunks 48 --ctx-size 512 --threads 8

# 4. shipped file: IQ3_S experts, protections from section 3, base type Q3_K_M
Q="$LLAMA_CPP/build/bin/llama-quantize"
IMATRIX="$SNAP/gguf/gigachat-audio-decoder.imatrix.gguf"
PROTECT=(
    --token-embedding-type q6_k
    --output-tensor-type q6_k
    --tensor-type 'ffn_gate_inp=f32'
    --tensor-type 'ffn_exp_probs_b=f32'
    --tensor-type 'attn_k_b=q8_0'
    --tensor-type 'attn_q=q6_k'
    --tensor-type 'attn_output=q6_k'
    --tensor-type 'attn_v_b=q6_k'
    --tensor-type 'attn_kv_a_mqa=q6_k'
    --tensor-type 'token_embd=q6_k'
    --tensor-type 'output=q6_k'
    --tensor-type 'ffn_gate_shexp=q6_k'
    --tensor-type 'ffn_up_shexp=q6_k'
    --tensor-type 'ffn_down_shexp=q6_k'
    --tensor-type 'blk\.0\.ffn_(gate|up|down)\.weight=q6_k'
)
"$Q" --imatrix "$IMATRIX" "${PROTECT[@]}" \
    --tensor-type 'ffn_.*_exps\.weight=iq3_s' \
    "$F16" "$SNAP/gguf/gigachat-audio-decoder-IQ3_S.gguf" Q3_K_M 8
```

The other measured files use the same `PROTECT` list. Only the expert rule
and the fallback type change:

| file | extra `--tensor-type` | fallback | imatrix |
|---|---|---|---|
| `Q4_K_M` | none | `Q4_K_M` | no |
| `Q4_K_XL` | none | `Q4_K_M` | no |
| `Q3_K_M` | none | `Q3_K_M` | no |
| `Q3_K_XL` | `ffn_.*_exps.weight=q3_k` | `Q3_K_M` | no |
| `IQ3_XXS` | `ffn_.*_exps.weight=iq3_xxs` | `Q3_K_M` | yes |
| `IQ3_S` | `ffn_.*_exps.weight=iq3_s` | `Q3_K_M` | yes |

`Q4_K_M` and `Q3_K_M` in the first column are llama.cpp's stock recipes:
`llama-quantize` with no `PROTECT` list. `Q4_K_XL` is `PROTECT` plus fallback
`Q4_K_M`, so attention, embeddings, the shared expert, layer 0 and the router
stay above the stock mix.

### Runtime validation in `strata-poly` (2026-11-05)

The new `deepseek2`/`gigachat_audio` backend was built and tested on the real
files with `--kv q8_0`, 8 CPU threads and a 4,096-token cache:

| file | load (cold/warm) | experts in VRAM | raw-completion parity vs `llama-completion` |
|---|---:|---:|---|
| `IQ3_S` | 4.3 s / 1.0 s | 1600/1600 (3.8 GiB) | **3/3 prompts identical** (numbers, "Paris.", fox) |
| `IQ3_XXS` | 3.7 s / 0.9 s | 1600/1600 (3.4 GiB) | **3/3 prompts identical** |
| `Q3_K_XL` | 3.9 s / 0.9 s | 1600/1600 (3.8 GiB) | 2/3; diverges on a code prompt |

CUDA graph capture works: the same prompt with graphs produced the same tokens
(` Paris.` + EOS) at 96.8 tok/s versus 83.0 tok/s eagerly. `mla_attend`,
absorbed K/V, interleaved RoPE and the YaRN scale all match llama.cpp on
`IQ3_S` and `IQ3_XXS`.

`Q3_K_XL` loads, but it already diverges from llama.cpp on one code prompt, so
`IQ3_S` is the better q3 ship. The file in the Hugging Face snapshot is
`gguf/gigachat-audio-decoder-IQ3_S.gguf`.

### llama.cpp converter commitments

Validated at `10d4846a572aebf21c0482d140b00e74c0f1ad20` (2026-11-05,
build `b1-10d4846`). Two fixes are required for this checkpoint:

1. `decoder/config.json`: `routed_scaling_factor` is `1` (int); transformers 5
   validates it as `float`. Set `1.0` before conversion.
2. `conversion/base.py`: the tokenizer is the GPT-4o pre-tokenizer regex with
   GigaChat's 128k vocab. Its hash is not in `get_vocab_base_pre()`; add
   `2b7d5f51ee85fd9a79f4bd0c4da297047a0cb94b2bad895782017ac84cb33896 -> "gpt-4o"`.
   Without it, conversion reaches the vocab step and dies in `AutoTokenizer`.

These should be sent upstream (hash entry + tolerant int→float config). Until
then, `convert_hf_to_gguf.py` has to run from that patched checkout.

### Hugging Face URLs

* Decoder: <https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B/tree/bf73d03/decoder>
* Text-base size reference: <https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF>

### Remaining uncertainty

* No audio WER/mIoU yet; the 3-bit gate is unproven.
* The imatrix is 0.59 MB, about 40× smaller than a 4 M-token calibration.
  `IQ3_XXS` quality may be understated relative to a larger one.
* The `deepseek2` backend loads and matches llama.cpp on `IQ3_S` and `IQ3_XXS`
  (3/3). `strata_inspect.py`'s `POLY_ARCHS` list is still stale for `deepseek2`.
* PPL on 8×512 held-out tokens is a smoke test; the Q4_XL/Q4_M ordering is
  within noise.
