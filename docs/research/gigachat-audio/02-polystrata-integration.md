# GigaChat Audio on `strata-poly` — integration blueprint

All `file:line` citations are from the tree at commit `2d6a8f9`. Where I had to assume a model shape I say so explicitly and give the exact GGUF key that replaces the assumption.

Two facts up front that shape the whole plan:

1. **`deepseek2` MLA is already implemented in this engine** — not in `deepseek4`, but in `glm5next`'s DSA layer, which is the *absorbed* form of exactly DeepSeek-V2/V3 MLA (`src/poly/glm5next.cpp:200-210`, `:511-543`). GigaChat's decoder is therefore a **new backend of ~500 lines**, not a new attention design.
2. **The `GENI`/`Image` path is model-agnostic and needs no engine change for audio.** `Engine::input()` copies any row the caller supplies (`src/poly/engine.cpp:2665-2673`); `Image{at,n,nx,rows}` is a plain grid of rows (`engine.hpp:179-183`). A Conformer row is a text-like embedding row at 1 position, so `strata-audio` can emit the existing `SVE1` record format and the engine consumes it verbatim.

---

## 0. Scope, and what is verified vs. assumed

| Verified from the tree | Assumed (must be read from the GGUF before step 1) |
|---|---|
| engine architecture, hooks, expert cache, KV, graphs, serve protocol, GENI path, vision helper | decoder geometry: `n_embd`, `n_layer`, `n_head`, `kv_lora_rank`, `q_lora_rank`, `qk_nope`, `qk_rope`, `v_head_dim`, `n_ff_exp`, `n_expert`, `n_used`, shared-expert width |
| llama.cpp `deepseek2` tensor names/shapes (`third_party/llama.cpp/src/models/deepseek2.cpp:89-160`) | GigaChat's exact HF config field names |
| 26 layers × 64 experts (from the brief) | whether the decoder carries a `nextn`/MTP block |

To replace the assumptions in one line:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,"tools")
from gguf_reader import GGUFFile
g=GGUFFile("gigachat-audio-decoder.gguf"); m=g.metadata
for k,v in sorted(m.items()):
    if any(s in k for s in ("block_count","embedding_length","attention.","expert","rope.","nextn","feed_forward")): print(k,"=",v)
PY
```

---

## 1. Architecture of the engine

### 1.1 Backend registration and instantiation

`Engine::open(const Options&)` is the only factory (`src/poly/engine.cpp:1176`):

```
1178  reads general.architecture from the GGUF header
1182  refuses --kv q8_0 unless qwen35moe/clover_moe/kolibri1
1187  refuses --kv int8/int8k unless gemma4
1191  refuses --cache-bias unless qwen35moe
1193  make() -> make_unique<Glm5Next|Qwen35|Gemma4|DeepSeek4|Lfm2Moe|Flint|Kolibri1>
1218  no layer split: engine->load(o); return
1227  layer split: one Engine per card run, prev->next_ chained, rows handed over in host memory
```

There is no registry table, no plugin loading: a new architecture is a `#include`, one `if` in `make()`, and the TU added to the four `add_executable` blocks in `CMakeLists.txt:2094-2131` (`strata-poly` at `:2103`).

`Engine` is a template-method base class. The interface an architecture must implement is `engine.hpp:361-470`; the ones the shared code calls, grouped:

| Hook | Called from | Meaning |
|---|---|---|
| `load_geometry()` | `load()` `:2000` | shape from metadata; `load_shape()` (`:1066`) fills `n_layer_/n_embd_/n_vocab_/n_expert_/n_used_/n_ff_exp_/n_ff_shexp_/rms_eps_` |
| `load_layer(il)` | `load()` `:2064`, twice | weights → VRAM (dense arena) or expert pointers into the mapping |
| `alloc_caches()`, `assign_scratch()`, `widest()`, `streams()` | `alloc()` `:2302,2342` | KV (`kv_alloc`) and per-row scratch (`fbuf`) |
| `layer_head` / `layer_tail` | `run()` `:3873,3883`, `layer()` `:4236,4245` | a layer up to the router, and the fate of the experts' sum |
| `route()`, `route_card()`, `moe_shared()`, `router_scores()` | `moe()` `:3194`, `moe_rows()` `:3473` | routing arithmetic |
| `output_rows()`, `finish_logits()`, `embed_scale()` | `eval()` `:4210` | final norm / logit cap / embedding scale |
| `card_row_bytes/save_card_rows/load_card_rows`, `extra_bytes/save_layer/load_extra` | `save_state()` `:2483`, `bind()` `:831` | conversation state |
| `sees()`, `whole()` | `main.cpp:405`, `run()` | picture rows |

### 1.2 GGUF loading and mapping

`Shards` (`src/poly/runtime.hpp:100-190`) mmaps the file (or the `-0000N-of-0000M` chain) and resolves `tensor name → (TensorInfo*, file)`. Nothing is decompressed at load except what the kernels cannot consume:

* `token_embd.weight` is **never uploaded** — `tok_embd_` is a raw pointer into the mapping, `tok_row_` bytes per row (`engine.cpp:2044-2054`); `embed()` converts a row on demand (`:2640-2643`).
* `upload()` (`:1270`) keeps a tensor's type when a kernel computes with it (`as_is`, `:1306-1307`), otherwise converts to F32 (norms, biases) or BF16 (matrices without a native kernel). It validates the shape against the caller's `[n_in, n_out]` (`:1273-1276`) — this is the single best guard against a wrong GGUF.
* The dense arena is built in two passes: pass 0 only advances `dense_at_`, `dense_bytes_` is then allocated and zeroed with a 256-byte `kTail` between tensors (`:2057-2081`, `kTail` in `runtime.hpp:31`). Routed experts are *not* in it: `load_moe()` records, per layer, the byte offsets of `ffn_gate_exps/ffn_up_exps/ffn_down_exps` inside the mapping and one expert's stride (`:1113-1173`). `ffn_gate_up_exps` (fused) and separate `ffn_gate_exps`+`ffn_up_exps` are both accepted (`:1137-1160`).

### 1.3 The expert cache, generically

`fill_expert_cache()` (`:1357`) is the whole policy:

```
1362  ram_tight_ = file + 5 GiB > available RAM
1364  reserve_mib_ = max(--vram-reserve-mib,256) or default_reserve_mib(total, tight)
1366  reserve = reserve_mib_ << 20;  if free <= reserve -> no cache at all
1385  unit = lcm(type sizes, 8);  slot_bytes = align(gu2 + d_bytes + kTail)   (per layer)
1394  candidate per (layer,expert): value = (prior+use) / bytes
1401  sort by value desc, tie by expert id
1407  arena = min(all, free - reserve, ...)
1409  VMM: reserve address space for `most`, map `arena`      (else one cudaMalloc)
1427  take candidates in value order while they fit
1439  h2d gate, up, down into a contiguous slot per expert
```

So: **capacity is a byte budget**, order is frequency-per-byte, and placement is layer-major so a layer's slots lie back-to-back (that is what makes `native_mmvq_slots` possible in `moe()` `:3284`).

Misses are handled two ways:

* **few tokens (a decode step, `n < kStageRows=32`)**: `moe()` (`:3155`) routes on the host, runs cached experts through `MmvqSlots` and uncached ones through `cpu_experts()` on the `Workers` pool, concurrently with the GPU launches (`:3259-3353`). `cpu_share` decides how many of the uncached experts the CPU takes *while the rest are copied* (`:73-75` of `engine.hpp`, `:3555`).
* **a prompt batch (`n >= 32`)**: `moe_rows()` (`:3465`) routes every token, groups rows by expert, and streams the uncached experts through the staging buffer with MMQ; `plan_ahead`, `stage_ahead` and `stage_read_mib` prefetch before routing (`:1710-1724`, `:2389-2418`, `:1553-1592`).

The cache also *follows* usage: `follow()`/`follow_loop()` (`:1797`, `:1725`) move experts into slots of less-used ones (`--expert-swaps`, default 64), and `mind_vram()` (`:1519-1546`) shrinks/grows the arena between requests to hold a reserve.

`all_cached()` (`engine.hpp:477`) is the important predicate: when every expert of every MoE layer has a slot, `card_moe()` (`:3365`) routes **on the card**, and `run()` records the *entire model* as one CUDA graph (`:3983-4004`) instead of per-layer graphs with a host router round-trip.

### 1.4 KV allocation and quantization

`kv_alloc(bytes_per_row, per)` (`:760`) reserves address space for `max_context` rows, `per` rows to `bytes`:

* **fp16** (default): the architecture asks for `row = n_kv*head_dim*2`. MLA asks for `kv_rank*2` (GLM `glm5next.cpp:288-290`), DeepSeek V4 for 512 fp16 + a 128-row ring (`deepseek4.cpp:222`).
* **`--kv q8_0`** (`KvFmt::kQ8_0`, `engine.hpp:47`): ggml Q8_0 blocks, one fp16 scale per 32 int8 codes, stored as separate code and scale buffers per side, so a reader can vector-load four codes (qwen35 `qwen35.cpp:163-186`; format documented in `docs/Q8_KV.md:14-30`, 544 B/row for `n_kv=2, head=256`).
* **`--kv int8` / `int8k`** (Gemma 4 only): same Q8_0 rule, codes and deltas in two arrays; `int8k` keeps V in fp16. Implemented with an fp16 "twin" cache for the prompt path (`engine.hpp:685-698`, `:801-828`).
* **`--kv q4_0` does not exist in `strata-poly`.** The parser accepts only `fp16|q8_0|int8|int8k` (`main.cpp:1187-1195`). The *other* engine has it, with an orthonormal 256-point Walsh–Hadamard rotation before Q4_0 and a rotated query — see `include/strata/kernels/kv_q4.hpp:1-20` and the parity test `src/kernels/kv_q4_parity.cpp`. That is the reference to port (step 6), not a design question.

Growth is demand-driven: `cover(end)` (`:883`) maps `kKvGrow=4096`-row pieces plus `kKvAhead=1024` (`runtime.hpp:52`), taking the VRAM from the expert cache via `shrink_cache()` and keeping a quarter of the reserve free (`:897-900`). `trim_kv(end)` (`:912`) gives back rows no request reached; with every expert cached `trim_card_kv` is a no-op (`:922`) and `load()` maps the whole context up front (`:2126-2140`). `--kv-whole` (`kv_grow=false`) allocates one block at load (`:764`).

### 1.5 Prefill batches, decode windows, CUDA graphs

* `rows_ = min(prompt_batch, max_context)` (`:2283`); `read_rows_` is the largest read one call may take, bounded by 512 MiB of residual rows (`:2284`, `kReadBytes` in `runtime.hpp:27`).
* `read(n)` (`:4264`) splits a prompt into `max_rows()` calls that respect picture boundaries and checkpoints, and — when experts are outside the cache — runs **layer-major**: all calls through layer 0, then layer 1, …, so each uncached expert is copied once per *read*, not once per *call* (`:4248-4263`, `:4324+`). With everything cached it degenerates into successive `eval()`s (`:4295`).
* A **window** is `opt_.window` guessed tokens fed in one `window()` (`:4469`); with CPU experts they go through the layers whole or in two halves, whichever the timing model prefers (`:3899-3968`, `pick_way()` `:4517`).
* Graphs: `record()` (`:3786`) wraps `cudaStreamBeginCapture`/`Instantiate`; replay is enabled only for `n <= window`, not a picture row, and `Layer::fixed` (`:3851-3852`). A recorded launch cannot take the position as an argument, so it reads `at_` on the card (`:3860-3863`, `engine.hpp:613-617`); recordings are keyed by `head_key(il,n)` which is a row-count bucket (`room_key()` `:3778`) for attention sizes that depend on the context. With all experts cached, `passes_[key]` records **every layer** in one graph (`:3990-4004`).

### 1.6 `Options` — the knobs that matter here

`engine.hpp:49-108`. Relevant subset: `vram_reserve_mib` (:53), `profile` (:54), `prompt_batch` (:59), `stage_mib`/`stage_read_mib`/`stage_ahead` (:60-72), `cpu_share` (:73), `expert_swaps` (:76), `ram_experts_mib` (:78), `graphs` (:83), `kv_grow` (:84), `kv_format` (:86), `window`/`draft` (:87-92), `cache_bias` (:100, qwen35 only), `layer_split` (:103), `batch` (:106).

---

## 2. The image / external-embedding path, end to end (`GENI`)

### 2.1 Trace

| # | Where | What happens |
|---|---|---|
| 1 | `serve/frontend.py:135,185-199` | `image_url`/`input_image`/`image` parts become `{"type":"image","source":…}` items; a string source becomes a URL or path |
| 2 | `serve/server.py:4362` `encode_prompt` | the chat template renders one `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` per image (`image_marks`, `:3351-3352`) |
| 3 | `serve/server.py:4460-4510` `prepare()` | the template's `pad` id and `start` marker are found; each image is downloaded/read; **encoding runs inside the request FIFO** (`self._turn`), never concurrently with the engine — the comment at `:4496-4499` records the deadlock that taught this |
| 4 | `serve/server.py:2547-2578` `Vision.encode` | SHA-256 of normalized bytes → cache hit or `ENC <img> <out>` on stdin; reads `OK <n_tokens> …`; one file per image under the encoder's cwd |
| 5 | `tools/vision/strata_vision.cpp:188-189` | `READY <n_embd>` once, then one `ENC` at a time; `:270-289` writes `SVE1` records `{magic, n, nx, ny, n_embd}` + `float[n][n_embd]` |
| 6 | `serve/server.py:4511-4526` | each `<\|image_pad\|>` run is **expanded to `n` pad ids** (`[pad]*encoded[k][1]`), so the token count of the prompt equals the row count |
| 7 | `serve/server.py:4553-4560` | the request's records are concatenated into one file (`combined_embeddings_path`, `:113`); the path and `pad` are stashed on the thread-local `self.embeddings` |
| 8 | `serve/server.py:4805` / `:1629` | `sampling["image_pad"]=pad` and the head becomes `GENI <max_new> <keys> <file> <id,id,…>` |
| 9 | `src/poly/main.cpp:548-585` `parse_gen` | `GENI` → `pad` key + file; ids parsed |
| 10 | `src/poly/main.cpp:212-262` `take_pictures` | reads every `SVE1` record, finds the next run of `pad` in the ids, **verifies the run length**, replaces those ids with **negative numbers hashed from the picture's bytes and grid** (`:252-254`) so the conversation cache distinguishes two pictures of the same size, and appends `Image{at,n,nx,rows}` |
| 11 | `src/poly/main.cpp:386` | `engine.images(pictures.at.data(), n)` |
| 12 | `src/poly/engine.cpp:2646-2673` | `images()` validates order/grid; `input()` returns `im->rows[row-im->at]` instead of the token embedding, and reports "true = picture row" so `run()` skips the embedding scale (`:3837`) |
| 13 | `src/poly/engine.cpp:3852`, `qwen35.cpp:407-437`, `gemma4.cpp:658-673` | graphs are disabled for a call containing picture rows; M-RoPE positions are derived per row (`turn()`), and `whole()` forces a picture into its own call where rows must see each other |
| 14 | `src/poly/main.cpp:866, 898-904` | token counting is ordinary: `PP <read>/<total>`, `DONE <generated> <prompt> …`; the engine does not count rows differently from tokens |

The **token budget** the server sees is therefore `len(ids)` after step 6 — one prompt token per row — and the same number comes back as the prompt token count in `DONE`. That is why an audio path with 6.25 tokens/s costs almost nothing in prompt length: 5 minutes of audio = 1,875 tokens.

### 2.2 What to copy for a `GENA` path, and what differs

Copy **verbatim**:

* the record format (`SVE1`) — the engine's reader validates `nx*ny == n` and the width (`main.cpp:226-230`), which is all it needs;
* `take_pictures()` — it is already generic over the pad id and the record count (`main.cpp:212`);
* `Engine::images()` / `input()` / `image_at()` — no change;
* the FIFO/`_turn` discipline in `prepare()`; a second GPU process running at the same time as the engine is exactly the failure documented at `:4496-4499`;
* the encoder-class shape: process + line protocol + hashed output cache + timeout-kill + restart (`server.py:2360-2600`).

What differs for audio:

| Concern | Image today | Audio |
|---|---|---|
| token axis | 2-D grid, `nx` used for M-RoPE and tiling | 1-D; `nx = n`, `ny = 1`. **`turn()` must stay `row`** so the default (`engine.hpp:461`) applies — no M-RoPE. |
| rows/year | ~1–4 k rows per picture | 188 rows per 30 s; 1,880 per 5 min |
| `whole()` | Gemma 4 needs one call per picture | default (`m`) is correct: audio rows are causal text-like rows |
| one call per... | a picture may exceed `--prompt-batch` and is refused (`gemma4.cpp:665-668`) | a 30 min clip (11,250 rows) exceeds `prompt_batch`; must be chunked by the **server** into ≤`max_rows` pieces, each its own `SVE1` record — `read()` refuses picture rows (`:4267-4269`) so a long audio span must be prefilled in successive `eval()` calls, which `main.cpp:404-406` already arranges (`text = pictures.at.empty()` → one call at a time up to the checkpoint) |
| timestamps | none | 6.25 tok/s means one row = 160 ms; the decoder needs either (a) ordinary vocabulary timestamp tokens emitted by the decoder, or (b) a time embedding folded into the row **encoder-side**. Prefer (a) — it costs the engine nothing; keep (b) as a fallback if the HF config has a time-embedding for encoder states. |
| precision | rows are fp32 (`SVE1` floats) | same; 188 × 1536 × 4 B = 1.1 MB per 30 s |
| count | one pad per row | identical; the frontend inserts `[pad]*n` between `<\|audio_start\|>`/`<\|audio_end\|>` |

---

## 3. Backend design: `gigachat_audio` (decoder only)

### 3.1 Identity and registration

Register **both** architecture strings to the same class:

```
if (name == "deepseek2" || name == "gigachat_audio") return std::make_unique<GigaChatAudio>();
```

Reason: if the converter writes `general.architecture = "deepseek2"` with DeepSeek-V2/V3 tensor names, **llama.cpp becomes the oracle on the same file** — that is the cheapest possible parity proof, and it is how `deepseek4` was validated (`docs/DEEPSEEK.md:36-40`). GigaChat-specific metadata can live under `gigachat_audio.*` or `deepseek2.*` and be read through `meta_alias()` (`engine.hpp:470`, pattern at `glm5next.cpp:70-88`).

### 3.2 Tensor / shape table

Shapes follow `third_party/llama.cpp/src/models/deepseek2.cpp:98-150` (row-major `[in, out]` as the engine's `upload()` expects; GGUF stores `ne[0] = in`):

| Tensor | GGUF shape (`ne`) | Engine call | Notes |
|---|---|---|---|
| `blk.N.attn_norm.weight` | `[n_embd]` | `f32(…, n_embd_)` | |
| `blk.N.ffn_norm.weight` | `[n_embd]` | `f32(…, n_embd_)` | |
| `blk.N.attn_q_a.weight` | `[n_embd, q_lora_rank]` | `mat(…, n_embd_, q_rank_)` | absent if `q_lora_rank == 0` (then `attn_q`) |
| `blk.N.attn_q_a_norm.weight` | `[q_lora_rank]` | `f32` | |
| `blk.N.attn_q_b.weight` | `[q_lora_rank, n_head*(qk_nope+qk_rope)]` | `mat(…, q_rank_, n_head_*mla_k_)` | `mla_k_ = qk_nope+qk_rope` (=192 for V3) |
| `blk.N.attn_kv_a_mqa.weight` | `[n_embd, kv_lora_rank + qk_rope]` | `mat(…, n_embd_, kv_rank_+n_rot_)` | the extra `n_rot_` columns are `k_pe` |
| `blk.N.attn_kv_a_norm.weight` | `[kv_lora_rank]` | `f32` | RMS over the latent only |
| `blk.N.attn_k_b.weight` | `[qk_nope, kv_lora_rank, n_head]` 3-D | `upload(…, mla_k_ - n_rot_, kv_rank_*n_head_, kFloatMat)` | **per-head** 3-D; the new converter's form |
| `blk.N.attn_v_b.weight` | `[kv_lora_rank, v_head_dim, n_head]` 3-D | `upload(…, kv_rank_, mla_v_*n_head_, kFloatMat)` | |
| `blk.N.attn_kv_b.weight` | `[kv_lora_rank, n_head*(qk_nope+v)]` | same slice as above | **legacy fused form**; accept both, as `load_moe()` accepts fused `gate_up` (`engine.cpp:1137`) |
| `blk.N.attn_output.weight` | `[n_head*v_head_dim, n_embd]` | `mat(…, n_head_*mla_v_, n_embd_)` | |
| `blk.N.ffn_gate_inp.weight` | `[n_embd, n_expert]` | `load_moe()` | |
| `blk.N.ffn_exp_probs_b.bias` | `[n_expert]` | optional F32, like `exp_probs_b` (`deepseek4.cpp:181-188`) | routing bias |
| `blk.N.ffn_{gate,up,down}_exps.weight` | 3-D `[…, n_expert]` | `load_moe()` | or fused `ffn_gate_up_exps` |
| `blk.N.ffn_{gate,up,down}_shexp.weight` | 2-D, `n_ff_shexp` | `load_moe()` | shared expert |
| `blk.N.ffn_gate_tid2eid.weight` | I32 | — | only if a hash-router layer exists (DeepSeek V3 has none) |
| `output_norm.weight`, `output.weight` | | `load()` `:2066-2074` | tied embeddings supported |
| `blk.N.nextn.*` | | `load_drafter()` `:2240` | optional MTP block; `load_shape()` already detects it (`:1067-1080`) |

Metadata (`load_geometry`): `attention.head_count`, `attention.key_length` (= qk_nope) or `attention.key_length_mla`, `attention.value_length` / `value_length_mla`, `attention.q_lora_rank`, `attention.kv_lora_rank`, `rope.dimension_count`, `rope.freq_base`, `rope.scaling.*` (YaRN), `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_feed_forward_length`, `expert_weights_scale`, `expert_weights_norm`, `expert_gating_func`, `swiglu_clamp_*`. Note `glm5next` already reads `key_length_mla`/`value_length_mla` under alias (`glm5next.cpp:53-56`, `:113-115`).

### 3.3 Layer graph

```
x  ──rmsnorm(attn_norm)──►  q_a (n_embd→q_rank) ─ rmsnorm(q_a_norm) ─ q_b (q_rank→n_head*192)
                                                                       │
                                       split per head: [nope 128 | rope 64]
                                       rope(q_pe) with dsv4_rope(off=128)
                                                                       │
        kv_a (n_embd→576) ──┬── latent 512 ─ rmsnorm(kv_a_norm) ─ store_f16(lat)
                            └── k_pe     64  ─ rope ─────────────── store_f16(kpe)
                                                                       │
   q_nope_abs[h] = k_b[h] · q_nope[h]      (128 → 512, per head, rows_per_x=512)
                                                                       │
   attn = softmax( (q_nope_abs·lat + q_pe·kpe)/sqrt(192) ) · lat       (values = latent only)
                                                                       │
   out[h] = v_b[h] · attn[h]               (512 → 128, per head)
   ao = attn_output · out                  (n_head*128 → n_embd)
   x += ao ;  x ──rmsnorm(ffn_norm)──► router → 6 of 64 + 1 shared → x += sum
```

This is *identical* in structure to `glm5next.cpp:225-232` + `:511-543`, with two differences: (a) the extra 64 rope columns in `kv_a` and in the key, (b) no indexer/pool.

### 3.4 MLA attention: absorbed (decode) and not (prefill)

`glm5next` implements **only** the absorbed form:

```cpp
// glm5next.cpp:511-518
lin(B.k_b, q_, qabs_, n, nullptr, kv_rank_);     // absorb W_K into q, per head
dsa_store(B, pos, n);                            // latent → B.lat (fp16, kv_rank/row)
// glm5next.cpp:536-541
dsa_attend(blas_, qabs_, B.lat, sel, ..., rank=kv_rank_, n_head, scale, ...);
lin(B.v_b, kqv_, vout_, n, nullptr, mla_v_);     // un-absorb W_V from the attention output
```

That is the right default and the cheapest decode. For the 64 rope columns the absorbed key is `[latent 512 | k_pe 64]` and the value is `latent 512` only — exactly llama.cpp's `Qcur = concat(q_nope_absorbed, q_pe)`, `Kcur = concat(kv_cmpr, k_pe)`, `Vcur = kv_cmpr` (`deepseek2.cpp:556-586`).

**Recommended implementation**: a `mla_attend` clone of `dsa_attend` (`src/poly/kernels.cu:2830-2857`) with:
* a `keys` buffer of `rank + n_rot` per key, filled by a copy kernel from **two** fp16 caches (latent, kpe);
* `score = cublasSgemmStridedBatched(rank+n_rot)` unchanged;
* `out = cublasSgemmStridedBatched(m = rank, lda = rank+n_rot)` — one line of difference, no extra buffer.

That is ~60 lines of CUDA. `n_head * rank` floats per token is 16k for `n_head=16, rank=576`.

A **non-absorbed** variant is optional and only useful for debugging: `k_nope[h] = k_b[h]·lat`, `v[h] = v_b[h]·lat`, then per-head causal attention. llama.cpp uses it for prefill, absorbed for decode, so last bits differ; keeping only the absorbed path means the parity test compares against llama.cpp's *decode* path (which is what matters) and near-ties on prefill follow the same tolerance already documented for DeepSeek V4 (`docs/DEEPSEEK.md:36-40`).

### 3.5 Router and shared expert

DeepSeek-V2/V3 routing is the *same* as GLM's: sigmoid gate, `+ exp_probs_b`, top-k on the biased score, weights from the **unbiased** gate, normalized to 1 (or not, per `expert_weights_norm`), times `expert_weights_scale`, floored at `6.103515625e-5` (`glm5next.cpp:543-560`, `deepseek4.cpp:411-444`). Copy `Glm5Next::route` verbatim, drop the `by_token` hash table, and copy `route_card` (`:562-567`) with `route_rows(..., gating=1 /*sigmoid*/, ...)`.

`moe_shared()` is `lin → lin → swiglu_clamp → lin` (`glm5next.cpp:571-577`) and needs `clamp_shexp_` from `swiglu_clamp_shexp` if the file has it (GigaChat may not; `clamp_limits` returns `INFINITY` when absent, `runtime.hpp:80-95`).

`layer_head`/`layer_tail` collapse to the single-stream form of `flint.cpp`/`lfm2moe.cpp` (`streams() == 1`, `layer_tail` = `axpy`), *not* GLM's hyper-connection form.

### 3.6 KV layout

* `lat`: `kv_alloc(kv_rank_ * 2, 1)` → 1,024 B/row/layer at `kv_rank_=512`.
* `kpe`: `kv_alloc(n_rot_ * 2, 1)` → 128 B/row/layer.
* total **1,152 B per token per layer**; ×26 layers = **29,952 B/token** (29.3 KiB). 8 K context = 240 MB, 32 K = 960 MB.
* `card_row_bytes()` = 1,152 × `n_moe_`-independent layer count that has attention (= 26) for conversation checkpointing; `save_card_rows`/`load_card_rows` are a straight `n * 1152` copy per layer, much simpler than GLM's pool bookkeeping (`glm5next.cpp:352-400`).
* `extra_bytes()`/`recurrent_floats()` = 0 — MLA's state is entirely in the KV rows.

### 3.7 CUDA graphs

`L.fixed = true` **only if** the attention is written graph-safe like DeepSeek 4's, not GLM's:

* rope via `dsv4_rope(..., const long long* at)` (`include/strata/poly/kernels.hpp:369`);
* store via `store_f16_at` (`:120`, used at `deepseek4.cpp:373-377`);
* `mla_attend(..., const long long* at, int room)` with the launch sized by `room()`/`room_key()` (`engine.cpp:3776-3782`);
* `head_key(il,n) = room_key(n)`.

GLM does **not** do this — `L.fixed = !B.dsa` (`glm5next.cpp:196`) — so its sparse layers cost ~30 launches per token on the CPU thread. For a 26-layer model whose whole point is low latency, this is the single most important code-quality decision: **write the MLA attention graph-safe from day one** and set `L.fixed = true`. With all experts cached, `run()` then folds all 26 layers into one graph (`engine.cpp:3990-4004`).

Consequence for a picture/audio row: `run()` disables replay for a call containing an encoder row (`:3852`), which is exactly what we want for the prefill call.

### 3.8 Generic engine code reused unchanged

`load_shape`, `load_moe`, `upload`, `Shards`, `fill_expert_cache`, `moe`, `moe_rows`, `card_moe`, `stage`/`plan_ahead`, `record`/`run`/`eval`/`read`, `kv_alloc`/`cover`/`trim_kv`, `images`/`input`, `bind`/`step`/`copy_seq`, the conversation cache (`src/poly/conversations.hpp`), the whole serve protocol in `main.cpp`. The only shared-code edits are the `make()` branch, the `--kv` gates, and the `GENA` line.

---

## 4. Ordered patch plan

Estimates are for one engineer familiar with the codebase; "lines" = net new/changed.

| # | What | Files / functions | Lines | Days | Acceptance |
|---|---|---|---|---|---|
| 0 | Registration + build | `src/poly/engine.cpp:1193-1202` (`make`), `:1182-1192` (kv gates), `#include`; `CMakeLists.txt:2094-2131`; `serve/models.py:30-40` (`BACKENDS`); `setup.py:2353` (`POLY_GGUF_ARCHS`) | 25 | 0.5 | unknown arch message no longer names a `deepseek2` file; `--list-models` shows it |
| 1 | Backend skeleton: geometry, tensors, caches, single-stream layer head/tail, route, shared expert, output | new `src/poly/gigachat.hpp` (~110), `src/poly/gigachat.cpp` (~380) | 490 | 2 | loads a `deepseek2` GGUF, allocates, runs one token, prints plausible logits |
| 2 | `mla_attend` + key-copy kernel | `include/strata/poly/kernels.hpp` (decl), `src/poly/kernels_mla.cu` | 130 | 1.5 | unit test vs a numpy reference on random latent/kpe |
| 3 | Graph-safe attention + `L.fixed` | `gigachat.cpp` (`attention()`), `head_key()` | 60 | 0.5 | `--dump-layers` identical with `--no-graphs`; decode tok/s +? |
| 4 | Absorbed/non-absorbed switch for debugging | `gigachat.cpp`, `kernels_mla.cu` | 120 | 1 | both paths agree to fp32 tolerance |
| 5 | **Decoder-only proof** (milestone) | `docs/research/gigachat-audio/05-quantization-recipe.md`, then `llama-completion` against `strata-poly` | 200 | 1 | §9.1 passes |
| 6 | KV quantization for MLA (`q8_0`, then `q4_0` with WHT) | `gigachat.cpp` `alloc_caches`, `store_q8`/`store_q8_at` reuse, `kernels_mla.cu`; port `include/strata/kernels/kv_q4.hpp` | 220 | 2 | Q8_0 answers identical on 20 greedy prompts; Q4_0 measured, documented |
| 7 | `strata-audio` encoder | new `tools/audio/strata_audio.cpp` (~330), `tools/audio/CMakeLists.txt`, `setup.py:3830` sibling | 400 | 3.5 | `ENC` on a 30 s wav → 188 rows, `OK 188 …` |
| 8 | Serve integration | new `serve/audio.py` (`Audio`, ~200), `serve/server.py` (`audio` cfg in `start_model` `:7422`, `prepare()` branch after `:4510`, `GENA` heads beside `:1629/1741/1994`, `audio_marks` beside `:3351`, `SPECIALS` `:3002`), `serve/frontend.py:123-205` (`AUDIO_PARTS`, `_audio_source`, `audio_of`), `serve/models.py:118` (`multimodal`), `serve/runconfig.py:98-100` (lazy_load guard) | 300 | 3 | `/v1/chat/completions` with `input_audio` transcribes |
| 9 | Engine `GENA` | `src/poly/main.cpp:543-587` (`parse_gen`), `:212-262` (reuse `take_pictures`), `--audio-pad` in the usage block `:1123` | 45 | 0.5 | `GENA` accepted; rows appear at the pad runs |
| 10 | setup family + docs | `setup.py` `MODELS`/`FAMILIES` (~472-520 pattern), `poly_setup` `:6703`, `:7005`; `docs/models/gigachat-audio.md`; `docs/BACKENDS.md:13-25` table | 150 | 1 | `./setup.sh --family gigachat-audio` end to end |
| 11 | *Optional*: persistent supervisor (§7.4) | `src/poly/main.cpp`, `src/poly/engine.cpp` | 600 | 5 | warm swap < 2 s |

**Total to a working decoder + audio path: ~15–16 days, ~2,150 lines.** The proof milestone (step 5) lands on day 5.

---

## 5. Expert cache, residency, and the quantified fallback

### 5.1 Can the cache be bypassed/pinned?

There is no explicit "pin" API. `all_cached()` (`engine.hpp:477`) is set implicitly when the byte budget covers every routed expert, and it changes behaviour materially:

| `all_cached()` | effect |
|---|---|
| true | `card_moe()` routes on the card, no host round-trip (`engine.cpp:3365`); `run()` records **all layers as one graph** (`:3983-4004`); `read()` degenerates to plain `eval()`; `fill_expert_cache` never called again unless resized |
| false | host routing, per-layer graphs, CPU experts, staging, swaps |

To *keep* it true under a co-resident neighbour:

| Knob | Set to | Why (code) |
|---|---|---|
| `--gpu-experts` (default on) | on | `fill_expert_cache` returns early when off (`:1358`) |
| `--vram-reserve-mib` | the MiB the neighbour + encoder need (≥256) | with an explicit reserve, `mind_vram`'s `keep = reserve_mib_` (`:1541`), so the cache is only given away when free < keep/2 — deterministic instead of the adaptive `reserve/2` default |
| `--expert-swaps` | `0` | nothing to trade when all cached; prevents churn if it ever shrinks (`:1452`, `follow()`) |
| `--stage-ahead` | `0` | unused with all cached (`moe_rows` not reached; `read()` takes the `all_cached()` branch `:4295`) — documents intent |
| `--cache-bias` | `0` | refused for non-qwen35 (`:1191`) *and* it perturbs routing |
| `--expert-profile` | omit | ordering only matters when short (`:1394-1402`); a profile file is pure I/O here |
| `--ram-experts-mib` | **omit** (`-1`) | if you pass it, 8 pool threads start even with zero outside experts (`:2197`) and `pool_` is sized from an empty list — wasted threads. Leaving it unset also avoids `rekeep()`/`pool_reseed()` in `moe()` (`:3161`) |
| `STRATA_FREEZE_EXPERTS=1` | optional | freezes the split **and blocks `resize_card_cache`/`mind_vram`** (`:1479-1481`, `:1528`) — the strongest pin, at the cost of `POST /v1/vram`. Use it on the model that must not yield (the audio decoder), and let the neighbour be the elastic one |

A real (small) waste to fix: `alloc()` allocates the staging buffer and two pinned 64 MiB chunks **before** `fill_expert_cache()` runs (`:2099` vs `:2101`), so an all-resident model keeps ~130–260 MiB it can never use. ~10-line patch: compute `all_cached()` after fill and `cudaFree`/`vmm_unmap` the stage, or gate the `if (mmq_)` block on `opt_.gpu_experts && !(all_cached())` by moving the size decision after fill. Worth 130 MiB on a 16 GB card.

### 5.2 The 2 GB-cache scenario, quantified

Numbers used: expert slot = (2 × n_embd·n_ff + n_ff·n_embd) weights at 4.5 bit ≈ **4.87 MB/expert** at `n_embd=2048, n_ff_exp=1408` (formula below; replace with the GGUF's own sizes). 2 GiB → **411 of 1,664 experts resident (24.7 %)**; 26 layers × 6 used = **156 expert uses/token**, ~**117 on the CPU**.

Cost anchors from the repo:

* GLM 5.3 (43×256, ~50 % uncached, 129 CPU experts/token) measured **19 ms/token, 11 ms of it the CPU's experts** → ≈ **81 µs per CPU expert**, and **10.5 → 16.4 ms** when the pages landed on the wrong NUMA node (`docs/BACKENDS.md:452-456`, `:465-474`). A single Q2_K expert took **59 µs** (99 µs NUMA-wrong) — those experts are ~2.4 MB, ours are 4.87 MB, so scale to ≈ **110–160 µs**.
* Held to 8 GB with 61–76 % resident, Qwen3.6 wrote **157–162 tok/s** where the all-resident build wrote **368–390** (`docs/BACKENDS.md:490-500`) → ≈ **0.42×**, i.e. 2.4× slower at ~30 % uncached.

Model: `slowdown ≈ 1 / (1 − f + f·(cpu_ms/gpu_ms))`, with `f` the uncached fraction and `cpu_ms/gpu_ms ≈ 20–40` per expert (4.87 MB over PCIe/VRAM vs 81–160 µs of AVX work).

| resident `f` uncached | CPU experts/token | CPU ms/token | projected decode (4060 Ti) |
|---|---|---|---|
| 100 % (all resident) | 0 | 0 | **~110–150 tok/s** (projection, §8) |
| 75 % (6 GB cache) | 39 | 4.3–6.2 ms | ~55–75 tok/s |
| 50 % (4 GB) | 78 | 8.6–12.5 ms | ~40–55 tok/s |
| **25 % (2 GB)** | **117** | **12.9–18.7 ms** | **~30–45 tok/s (3–4× slower)** |
| 10 % (0.8 GB) | 140 | 15.4–22.4 ms | ~25–35 tok/s |

Two qualifications that matter:

* It is **compute-bound, not IO-bound**, as long as RAM ≥ the 6–8 GB of experts outside VRAM — the default `--ram-experts-mib -1` keeps them mapped and the OS page cache holds them (`engine.hpp:78-81`). If RAM is tight (`files + 5 GiB > available`, `:1362`), `ram_tight_` fires and the picture gets much worse.
* Swaps add a second cost: `follow()` moves experts into slots *while* answering (`:1797`), so a churning cache pays PCIe as well. `--expert-swaps 0` removes that but freezes the 24.7 % split.

**Conclusion:** 2 GB of expert cache for a 26×64 q4 model is a bad operating point — it turns a 120 tok/s decoder into a ~35 tok/s one. If co-residency is mandatory, prefer giving *Tiel* the small cache (Tiel is the chat model and tolerates it better at 35B-A3B with a draft block, `docs/BACKENDS.md:490-500`) and GigaChat the resident experts, or run GigaChat at q3 so its 6.2 GB fits whole (§8).

---

## 6. `strata-audio`, the external Conformer encoder

### 6.1 Shape of the process

Mirror `strata-vision` exactly, because every operational lesson in it is already paid for:

| Aspect | `strata-vision` (cited) | `strata-audio` |
|---|---|---|
| binary | `engine-poly/strata-vision`, built by `tools/vision/CMakeLists.txt` (own CMake project, `ggml`/`llama` subdir `EXCLUDE_FROM_ALL`) | `engine-poly/strata-audio`, `tools/audio/CMakeLists.txt` — same pattern, but **no llama.cpp dependency** (a Conformer is conv+attention, not mtmd); link `ggml` only if you reuse its kernels, else plain CUDA |
| protocol | `READY <n_embd>`, `ENC <img> <out>` → `OK <n> <nx> <ny> <ms>` / `ERR msg`, `QUIT` (`strata_vision.cpp:12-17`, `:188-296`) | `READY <n_embd> <hz>`, `ENC <wav> <out> [start_ms] [end_ms]` → `OK <n> <t0> <t1> <ms>`, `QUIT` |
| transport | **files**, one `SVE1` record per tile (`:279-285`); no shared memory, no sockets | same. Bandwidth is trivial (1.5 MB per 30 s), and files give free replay/caching. Only if you later stream 1 h of audio does a `memfd`/`/dev/shm` ring become worth it. |
| startup | loads `mmproj` + vocab-only text model (`:120-146`), prints `model files loaded in %.1f s`, **warms up at the largest input before READY** so GPU buffers are allocated before the engine sizes its cache (`:159-187`) | load weights (~0.3 s page-cache warm), run one 30 s dummy through the encoder before `READY`. Same reasoning: allocate the workspace now, not on a full card |
| CPU mode | `CUDA_VISIBLE_DEVICES=-1` so a CUDA build does not open a 0.4–0.7 GB context and steal ~150-260 expert slots (`:93-101`) | identical, and more important here (16 GB card) |
| place | `serve/server.py:2360-2600` class `Vision`, ctor `:2370-2390` | new `serve/audio.py` class `Audio`, same fields (`dir`, `spawn`, `proc`, `lock`, `cache`, `on_cpu`, `elastic`, `elastic_mib`) |
| failure | `_readline(timeout)` kills the process on silence (`:2414-2431`); `encode()` raises `ValueError` → 400; `alive()` false → next request restarts (`:2433-2459`) | same, plus: a killed encoder mid-request is **retryable** (re-encode the clip) where a lost image is not, so `encode()` retries once on a fresh process |
| cache | SHA-256 of normalized bytes, 64 entries, never evicting a file the request still needs (`:2547-2578`) | SHA-256 of the **decoded PCM** (not the container) so remuxed wav/mp3 of the same audio hit |
| quantization | mtmd/Q8 types from the mmproj file | `--quant q8_0|q5_1|f16` applied to the encoder's GGUF; keep the **output adapter and the last conv in F16** — a quantized adapter moves the decoder's embeddings more than a quantized block stack does |
| GPU vs CPU | GPU `vision_vram_gb` 1.6–2.4 GB (`setup.py:632,685,713`), CPU on cards < 16 GB | GPU ≈ **0.35–0.9 GB** for a ~120 M-param Conformer at f16 with a 30 s activation window; CPU ≈ 0.4 GB RAM. Rule: GPU only when the decoder is not co-resident, else CPU (an audio prompt is 1–3 s of encoding, and the request FIFO serializes it anyway) |

**Startup cost**: `strata-vision` reports its own load time (`:147`); a Conformer GGUF of ~0.25 GB loads in well under a second warm. Keep it *elastic* like the vision encoder (`Vision.elastic_of` `:2355-2365`, `_vision_up` `:3779`, `_vision_idle_down` `:3804`) so its VRAM exists only while a clip is being encoded — that is the pattern that makes co-residency possible at all.

### 6.2 What the encoder must produce

* input: 16 kHz mono PCM, resampled server-side or in-process (`soxr`/`libsamplerate`; do **not** trust the client's rate);
* chunking: window of N seconds with a left context of 1–2 s and a right lookahead, hop chosen so the token rate is exactly **6.25 tokens/s** (160 ms). The chunk boundaries must be deterministic: same audio → same rows, or the conversation cache's hashing (`main.cpp:252`) mismatches. Cap each chunk at `--prompt-batch` rows when it feeds one `eval()` (see §2.2);
* output: `SVE1`-compatible records, `float[n][n_embd]` with `n_embd` = the **decoder's** width (the adapter is inside the encoder, exactly like a vision projector);
* timestamps: either the decoder emits them as vocabulary tokens (preferred — zero engine work) or the encoder writes a sidecar `SVTS` record `int32[n] {t0_ms}` that `serve/audio.py` keeps next to the `.sve` and uses to build the response's `segment`s. Never put them in the rows.

### 6.3 Where it sits, and the exact `serve/` changes

```
serve/
  audio.py            new: class Audio (mirror of Vision), decode/resample/chunk, cache, elastic
  server.py           ~40 lines: cfg "audio" in start_model (7422), Audio(...) construction,
                      prepare() branch after 4510, audio_pad in sampling_keys (1245 area),
                      GENA heads beside 1629/1741/1994, audio_marks beside 3351
  frontend.py         ~40 lines: AUDIO_PARTS = ("input_audio",), _audio_source() (data: URL or path),
                      audio_of(), _parts_of() item {"type":"audio","source":…}, openai_to_messages (557)
  models.py           ~6 lines: multimodal = bool(cfg.get("vision") or cfg.get("audio"))
  runconfig.py        ~3 lines: lazy_load also refused for "audio" (98-100)
```

`input_audio` parsing (OpenAI's shape `{"type":"input_audio","input_audio":{"data":<b64>,"format":"wav"}}`):

1. `frontend._audio_source()` decodes base64 → bytes, or takes a path; `audio_of(messages)` collects them in prompt order.
2. `prepare()` calls `self.audio.encode_all(...)` **inside the FIFO turn**, exactly like images (`:4496-4505`) — a Conformer and the engine on the GPU at once is the documented deadlock.
3. Resampling/chunking happens in `Audio.encode()` before the `ENC` line: decode with `soundfile`/`ffmpeg`, downmix, resample to 16 kHz, cap length (e.g. 10 min → refuse with a clear 400).
4. The template renders `<|audio_start|><|audio_pad|><|audio_end|>` (the template's business, like vision); `prepare()` expands each pad to `n` pads (`[pad]*n`), and writes `self.embeddings.audio_path/pad`, mirroring `:4553-4560`.
5. `run()` adds `audio_pad=<id>` and the file to the `GENA` head; the engine does the rest.
6. Response: `verbose_json`-style `segments` built from the timestamp tokens/sidecar when the client asks for `response_format`, and plain text otherwise — a `serve/transcribe.py` helper (~80 lines) is the cheapest way to serve `/v1/audio/transcriptions` too.

---

## 7. Concurrency and residency

### 7.1 What prevents two processes from sharing the GPU safely today

1. **No coordination at all.** Each of `strata-poly`, `strata-vision`, `strata-audio` is a separate process with its own CUDA context. Nothing takes a lock, nothing uses MPS, nothing reserves. Under `compute_mode=Exclusive_Process` the second context simply fails — the codebase names this failure and its cure (`serve/server.py:2304-2320`).
2. **A second context is not free**: 0.4–0.7 GB and ~150–260 expert slots, measured and documented in `tools/vision/strata_vision.cpp:93-94`.
3. **VRAM accounting is point-in-time and single-tenant.** `fill_expert_cache` reads `cudaMemGetInfo` once at load (`engine.cpp:1359-1366`); `mind_card_vram` reads it again between requests (`:1530-1546`). Neither knows what a neighbour *will* allocate. There is no ledger.
4. **`cover()` steals from the cache silently.** A growing context shrinks the expert arena (`:894-901`) and `trim_card_kv` gives nothing back when all experts are cached (`:922`) — so a long request on model A permanently reduces A's cache unless `mind_vram` later finds free VRAM.
5. **The server serializes GPU work by FIFO**, not by any engine mechanism: `self.fifo`, `_turn`, `self.watch` (`server.py:3779-3800`, `:4496-4505`). That is why vision encoding cannot overlap decoding.

### 7.2 What the engine does at startup (the reserve story)

```
open()  -> pick class, optional layer split                       engine.cpp:1176
load()  -> Shards mmap, load_geometry                             :1998-2000
        -> streams, events, cuBLAS, VMM probe (VMM => elastic)    :2017-2042
        -> token table stays in the mapping                       :2044-2054
        -> two passes: dense arena sized, then filled             :2056-2082
        -> alloc(): rows_=prompt_batch, scratch, alloc_caches(),
           cover(1) => 5,120 KV rows mapped BEFORE the cache      :2283, 2302-2303
        -> read_profile, fill_expert_cache (reserve here)         :2100-2101
        -> sync, drop dense file pages                            :2105-2107
        -> if all_cached and it fits: cover(max_context)          :2126-2140
        -> threads pinned to the experts' NUMA node, pool, swaps  :2141-2233
```

Default reserve: `total/32` clamped to **256–768 MiB**, or `total/16` clamped 512–1024 when RAM is tight (`default_reserve_mib` `:468-471`). `--vram-reserve-mib` is floored at 256 (`:1364`). The reserve exists because recorded launches and cuBLAS workspaces allocate *after* the cache: "180 to 200 MiB of an 8 GB card, most of it in the first request" (`:1512-1513`).

### 7.3 Knobs for a small co-resident model

| Goal | Setting |
|---|---|
| leave room for the neighbour + encoder | `--vram-reserve-mib <sum>`; **do not** rely on the default (it is 256–768 and adaptive) |
| let the neighbour take VRAM back later | neighbour's engine: `POST /v1/vram {"reserve_mib":N}` (`main.cpp:759-795` → `resize_cache`); only meaningful when VMM is available |
| never yield (the pinned model) | `STRATA_FREEZE_EXPERTS=1` + explicit reserve |
| keep the context from eating the cache | `--kv q8_0` (½ KV) or `--kv q4_0` (¼, once step 6 lands); a bounded `--max-context` (audio needs 8–16 K, not 128 K) |
| bound prefill scratch | `--prompt-batch 512`–`1024` (1.4 MiB VRAM/token, `engine.hpp:59`) |
| bound staging | `--stage-mib 128 --stage-read-mib 0` (or the all-cached free, §5.1) |
| avoid the idle 8 pool threads | omit `--ram-experts-mib` |

### 7.4 One process, two models?

**Today: no.** `Engine::open` returns exactly one engine (`:1176-1259`); `main.cpp` holds one `Engine&` in `Session` (`:329-331`) and serves one `GEN` stream. `--batch N` is *N sequences of the same model* (`engine.hpp:347-356`, `bind`/`step`), not N models. Two `Engine` objects in one process is not architecturally impossible (per-engine `blas_`, scratch, `Workers`), but they would fight through `cudaMemGetInfo`, `mind_vram` and `cover`, which all assume single tenancy.

**Process-per-model is therefore mandatory** for q4-gigachat + Tiel today — and that is the design the server already has: `_switch_to` stops the old engine first ("the two models do not fit in memory together", `serve/server.py:3481-3500`), so switches are 8–16 s with warm page cache (`docs/BACKENDS.md:331`) or 6–7 s for a start (`docs/DEEPSEEK.md:20`).

**A persistent supervisor** would need, in order of value:

1. **A VRAM ledger** replacing `cudaMemGetInfo` as the sizing input: each resident model gets an explicit reserve at load and the supervisor owns the sum. Without this, two engines in one process cannot coexist.
2. **A model-slot table** in `main.cpp`: `LOAD <path> [args]`, `BIND <slot>`, `UNLOAD <slot>`, with `GEN` routed to the bound slot — the protocol already tolerates unknown keys (`main.cpp:577`).
3. **Per-slot VMM**: `arena_`, `kv_` and `stage_map_` are already address-stable VMM objects (`engine.hpp:551-565`, `vmm_reserve/map/unmap` `:644-758`). `UNLOAD` = `vmm_unmap` the arena + KV; the weights stay in the OS page cache. This is what makes a swap cheaper than a restart — but note it is a *re-upload* of 6–8 GB over PCIe (~0.5–0.7 s at 12 GB/s) plus graph recapture, not free.
4. **One CUDA context, one cuBLAS handle per slot** (already per-engine), and **graphs captured per slot** (already keyed by buffers, `engine.hpp:609-611`).
5. **Honest expectation**: a warm swap ≈ 1–2 s (context, module load, GGUF parse, weight copy) versus 6–16 s cold. The fast path for a "switch" between GigaChat Audio and Tiel is **not** a swap at all — it is co-residency with GigaChat pinned and Tiel elastic, because a 35B-A3B chat model at 2 GB of expert cache still writes ~150 tok/s (`docs/BACKENDS.md:490-500`) while a 26×64 model at 2 GB collapses to ~35 tok/s (§5.2).

**Parallelism that is actually available**: `--batch N`/`step()` decodes N *requests* together (`engine.hpp:356`, `engine.cpp:4026`). Audio requests are short and bursty (188 audio rows per 30 s, few output tokens), so batching 4–8 transcriptions is worth far more than intra-request parallelism. That is the lever to pull for a transcription service, and it costs no engine change.

---

## 8. Memory budget

Formula (all terms from the GGUF):

```
expert_bytes   = n_layer * n_expert * ((2*n_embd*n_ff_exp + n_ff_exp*n_embd) * bits/8 + slot_pad)
dense_bytes    ≈ n_layer * (n_embd*q_lora + q_lora*n_head*192 + n_embd*(kv_lora+64)
                            + kv_lora*n_head*128*2 + n_head*128*n_embd) * bits/8
                 + n_layer * 3*n_embd*n_ff_shexp * bits/8 + n_vocab*n_embd * out_bits/8
kv_bytes       = n_attn_layers * (kv_lora + 64) * 2 * ctx        (fp16; x0.53 for q8_0, x0.28 for q4_0)
```

Worked at the assumed shape (**n_embd 2048, 26 layers, 64 experts, 6 used, n_ff_exp 1408, 1 shared expert, kv_lora 512, qk_nope 128, qk_rope 64, v 128, n_head 16, q_lora 1536, vocab 150 k, ctx 8192**):

| Item | q4 (≈4.5 bit) | q3 (≈3.44 bit) |
|---|---|---|
| routed experts (1,664 × 4.87 / 3.72 MB) | **8.10 GB** | **6.19 GB** |
| dense arena (attn 224 + k_b/v_b BF16 109 + shared 127 + router 14 + output 173 + pad) | 0.69 GB | 0.56 GB |
| KV fp16 (8,192 × 29,952 B) | 0.24 GB | 0.24 GB |
| KV q8_0 | 0.13 GB | 0.13 GB |
| staging + 2×64 MiB pinned (removable if all cached, §5.1) | 0.26 GB | 0.26 GB |
| scratch (`rows_`=1024: mq/mq2/xq/q8/wf) + graphs + cuBLAS ws | 0.30–0.45 GB | 0.30–0.45 GB |
| CUDA context + modules | 0.25–0.40 GB | 0.25–0.40 GB |
| **decoder total** | **9.9–10.2 GB** | **7.9–8.2 GB** |
| free of 15.9 GB usable (4060 Ti 16 GB) | 5.7–6.0 GB | 7.7–8.0 GB |

Co-resident with Tiel Coder (dense + 8 K KV ≈ 2.1 GB, 2.0 GB expert cache, per the brief):

| | q4-decoder | q3-decoder |
|---|---|---|
| GigaChat decoder | 10.0 | 8.0 |
| `strata-audio` Conformer on GPU (f16, 30 s) | 0.7 | 0.7 |
| Tiel dense + KV 8 K | 2.1 | 2.1 |
| Tiel expert cache | 2.0 | 2.0 |
| second CUDA context + pinned buffers + graphs headroom | 0.9 | 0.9 |
| desktop/compositor headroom | 0.6 | 0.6 |
| **total** | **16.3 GB ✗** | **14.3 GB (~1.6 GB spare)** |

**Verdict: q3 is the co-residency size; q4 is the alone size.** If you must co-reside at q4, the only levers that close the 0.4 GB gap are: encoder on CPU (−0.7), Tiel cache to 1 GB (−1.0), or `--prompt-batch 512` (−0.2). KV at q8_0 buys another −0.11.

Everything in that table except the two anchors (Tiel's 2.1 GB dense+KV, the 0.7 GB Conformer) is **derived from the formula above and must be recomputed from the actual GGUF before committing to a size**.

---

## 9. Parity and test plan

### 9.1 "Decoder-only proof" (step 5, the gate for everything else)

1. **Convert twice, same graph.** `convert_hf_to_gguf.py --outtype bf16` → `gca-bf16.gguf`, then `llama-quantize` to `Q4_K_M` and `Q3_K_M`. Both engines read the *same file*; if the converter writes `general.architecture = "deepseek2"`, llama.cpp is a drop-in oracle.
2. **Logit parity on a fixed token string**, greedy, no draft, no graphs, everything cached:
   ```
   strata-poly --model gca-Q4_K_M.gguf --tokens 1,2,3 --top-probs 8 --no-graphs --kv fp16
   llama-cli -m gca-Q4_K_M.gguf -p "…" -n 1 --temp 0 --top-k 1 -n 8 --logits-all
   ```
   Acceptance: top-1 identical over 200 tokens; top-8 sets identical; max |Δlogit| < 0.05 on q4 (the engine's kernels quantize activations to Q8_1, so exact equality is not on the table — `src/poly/expert_parity.cpp:1-12` states the same tolerance philosophy).
3. **Per-layer parity**: `--dump-layers` prints `pos blk name sum abs first…` per stage (`engine.cpp:2675-2687`); run `llama-eval-callback` (present at `third_party/llama.cpp/examples/eval-callback`) with a matching callback and diff `attn_norm/attn_in/attn/ffn_in/ffn/out_norm` sums layer by layer. This localizes a wrong transpose in one hour instead of one week.
4. **Forced-token divergence test**: `--follow <llama tokens>` (`main.cpp:1127`) forces the engine to replay llama.cpp's exact token sequence, then compare NLL — catches "right answer, wrong distribution".
5. **Gate on determinism**: `STRATA_FREEZE_EXPERTS=1 --window 1` so a greedy answer is a function of the request alone (`engine.cpp:2004-2016`).
6. **Expert-path parity**: `poly_expert_parity gca-Q4_K_M.gguf` (`src/poly/expert_parity.cpp`) validates GPU vs CPU vs dequantized-reference for every tensor type in the file — run it before trusting any speed number.

### 9.2 Engine tests

| Test | Basis |
|---|---|
| MLA unit test vs numpy: absorbed q, two-cache key concat, value over latent only, rope on 64 columns | new `src/poly/attn_mla_test.cu`, sibling of `attn_test.cu`, `dsv4_test.cu` |
| same token with `--no-graphs` and with graphs, 1 / 4 / 128 rows | existing pattern `serve/test_*` + `main.cpp --check-batch` (`:994`) |
| state save/load/rows around an audio span | `Engine::save_state`/`save_rows` round-trip; GLM's `save_card_rows` is the template (`glm5next.cpp:352-400`) |
| KV q8_0/q4_0: 20 greedy prompts identical (q8) / measured + documented (q4) | `docs/Q8_KV.md:36-48` methodology |
| `--kv` refusal messages still correct for other arches | `engine.cpp:1182-1189` unchanged for them |

### 9.3 Serve/path tests (mirror the existing ones)

* `serve/test_frontend.py:89` currently asserts `input_audio` is *dropped* — flip it and add the parsed-audio case.
* `serve/test_images.py` / `test_vision_busy.py` / `test_vision_elastic.py` are the templates for `test_audio.py`: encoder restart, timeout kill, elastic up/down, hash cache hit, pad-count mismatch → 400, `audio_pad` reaching the engine.
* `serve/test_models.py:87-95` — registry says `multimodal: true` for an `"audio"` config.
* `serve/test_context_overflow.py` — a 10-minute clip must be refused with a *count*, not truncated.

### 9.4 Audio accuracy

* 30 fixed clips (short/long, silence, music, code-switching RU/EN) → WER/CER against references, q3/q4 vs the f16 encoder; a regression gate of +0.5 % absolute WER.
* **Timestamp drift**: compare segment boundaries to references; with 6.25 tok/s the quantisation is 160 ms, so the acceptance band is ±1 token (±160 ms) plus model error.
* **Prompt-cache correctness**: send the same clip twice; the second must report `reused > 0` (the negative ids from `take_pictures` are the cache key, `main.cpp:252-254`).

---

## 10. Top risks

| # | Risk | Why it bites | Mitigation / citation |
|---|---|---|---|
| 1 | **GigaChat's decoder is not exactly `deepseek2`** (different rope layout, extra audio-time embedding, no `kv_a_mqa` split, different gating func) | The whole plan rests on the shape table; one surprise (e.g. `qk_rope` interleaved per head, or a learned `k_pe` position table) invalidates the absorbed kernel | step 0's metadata dump before any code; `meta_alias`/`tensor_alias` already exist for converter drift (`glm5next.cpp:26-66`); the non-absorbed path (step 4) is the escape hatch |
| 2 | **16 GB is not enough for q4 + Tiel + encoder** | arithmetic in §8: 16.3 GB vs 15.9 usable; Windows/desktop overhead is worse | q3 for co-residency; encoder on CPU; or `POST /v1/vram` between turns. Decide before buying the config |
| 3 | **`mind_vram` is single-tenant and point-in-time** | two engines each believe the free VRAM is theirs; a co-resident pair can OOM on the *second* request, not the first (`engine.cpp:1519-1546`) | explicit `--vram-reserve-mib` on both, `STRATA_FREEZE_EXPERTS=1` on the pinned one, never `POST /v1/vram` while both are loaded |
| 4 | **A second GPU process collides with the engine** | measured: 0.4–0.7 GB context and 150–260 lost expert slots (`strata_vision.cpp:93-94`); Exclusive_Process refuses outright (`server.py:2304-2320`) | `CUDA_VISIBLE_DEVICES=-1` for the CPU encoder path, elastic encoder, FIFO (`_turn`) so no two GPU jobs overlap |
| 5 | **GLM-style non-recordable attention** | `L.fixed = !B.dsa` (`glm5next.cpp:196`) costs ~30 CPU launches/layer/token; copying that shape for a latency-critical 26-layer model throws away the main win | follow `deepseek4` (`dsv4_rope`/`store_f16_at`/`dsv4_attend` with `at`+`room`), step 3 is a hard requirement, not a polish |
| 6 | **MLA numerics are not bit-identical to llama.cpp** | absorbed decode vs llama.cpp's non-absorbed prefill; Q8_1 activation quantization in every product | compare decode-to-decode, top-1 not logits; `docs/DEEPSEEK.md:36-40` is the precedent (8/8, 64/64, and 15/32 on a long prompt where candidates tie) |
| 7 | **`--kv q4_0` does not exist here and is lossy** | naive Q4_0 KV loses too much; the rotation is what makes it usable (`include/strata/kernels/kv_q4.hpp:1-12`) | port the WHT-rotated variant from `src/kernels/kv_q4*`, or ship fp16/q8_0 first; MLA's 1,152 B/token is already small (§3.6) |
| 8 | **`prompt_batch` vs long audio** | `read()` refuses picture rows (`engine.cpp:4267-4269`); a 30 min clip is 11,250 rows and `whole()`-less rows still prefill one call at a time | server-side chunking into `max_rows()` pieces, one `SVE1` record each, in prompt order (`main.cpp` requires contiguous ordered `Image`s, `engine.cpp:2649`) |
| 9 | **Expert cache churn if it does not fit** | §5.2: 2 GB cache → ~3–4× slower, plus `follow()` PCIe traffic and non-reproducible greedy answers (`engine.cpp:2004-2016`) | size for all-cached or accept and document; `--expert-swaps 0` if it must stay stable |
| 10 | **The Conformer is a new encoder with no mtmd equivalent** | `strata-vision` gets preprocessing, tiling and quantized kernels free from llama.cpp; audio gets none of that | budget 3.5 days (step 7), reuse `ggml` conv/attention kernels from `third_party/ggml` rather than writing CUDA from scratch; keep the adapter in F16 |
| 11 | **Timestamp semantics are model-specific** | if the decoder expects time embeddings in its input rows, a row-per-160 ms assumption is wrong | inspect the HF config for `time_embed`/`timestamp` modules before step 7; the encoder can fold time into the rows at zero engine cost |
| 12 | **Converter drift** | three converters already disagree on GLM tensor names (`glm5next.cpp:26-66`) and on DeepSeek's `k_b/v_b` vs fused `kv_b` (`third_party/llama.cpp/src/models/deepseek2.cpp:119-123`) | accept both forms in `load_layer` the way `load_moe` accepts fused `gate_up` (`engine.cpp:1137-1160`); add the file's names to `tensor_alias()` |

---

## 11. Open questions to close before step 1

1. Does the HF config have `q_lora_rank > 0`? (If 0, `attn_q` is a single matrix — `deepseek2.cpp:106-110`.)
2. Is `k_b/v_b` stored per-head 3-D (new converter) or fused (legacy)? Both, accept both.
3. `expert_gating_func` value: 1 = softmax, 2 = sigmoid, 4 = sqrt(softplus) (`deepseek2.cpp` usage; `deepseek4.cpp:100` asserts 4). Routing must match it exactly or the answers wander.
4. Does the decoder have a `nextn`/MTP block? If yes, implementing `check_draft_block()` + `block_attend()` (a copy of GLM's, `glm5next.cpp:604-605`) gives speculative decoding for free — worth ~1.5× on decode and cheap to add.
5. Audio token rate: is 6.25 Hz the encoder's subsampling ratio or a post-merge? It fixes the chunk hop and therefore the timestamp grid.
6. Does the decoder use YaRN/rope scaling (`rope.scaling.type = yarn`)? If so, port `make_rope()` from `deepseek4.cpp:25-38` and the `attn_factor` handling — otherwise long transcripts drift.
7. Where do timestamps come from: decoder vocabulary, or a separate head? This decides whether `strata-audio` must emit a sidecar at all.
