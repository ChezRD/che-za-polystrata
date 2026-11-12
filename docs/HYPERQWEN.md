# HyperQwen, surveyed against this fork's dense Qwen3.8-27B on sm_89

A survey of **https://github.com/syv-ai/HyperQwen** (shallow clone, `--depth 1`, 32 MB) against the question: is there
anything in it that helps the dense `qwen35` family as we are porting it — 24 Q / 4 KV heads, `head_dim` 256,
`n_embd` 5120, FFN 17408, 16 attention + 48 GDN layers, on Ada sm_89, in `strata-poly`?

**Answer in one line:** one thing is worth taking — the D-LSA (flat) index design, whose distilled map for exactly
this checkpoint is in the repo — and that is a forerunner of our own LSA plan rather than new machinery. Everything
else is either a vLLM-only fix, or needs Hopper/Blackwell, or proposes a KV layout and a state format that our own
measurements already settled.

## 1. What it is

**A patch series against a pinned vLLM, plus a one-time model-preparation pipeline.** Not an engine: it does not
read a GGUF, has no kernels outside optional Triton plugins, and every serving path goes through vLLM 0.33.0. The
README is explicit about the split — the patch series, the KVarN backend and the "SSE keep-alive / engine stall
sentinel" fixes are portable; the int8-QK prefill kernel, the 40k draft vocabulary and the DFlash2 drafter are
model-specific; the Marlin tune tables are sm86-specific.

**What it claims** (README, measured on one RTX 3090 at 250 W, 24 GB): 127 tok/s single-stream on Qwen3.8-27B at 64k
context, ~1,035 tok/s aggregate at 64 concurrent, 381 tok/s while the answer quotes its prompt, 150k context in
"long context" mode and 240k with the KVarN cache.

**What it demonstrates**: those numbers are one card, one vLLM pin, with the protocol in `bench/run_benchmarks.sh`
and its own honesty section (`docs/optimizations.md`, "Things this campaign measured that did NOT pay"). It is
unusually careful about saying what it did not measure — the 40% of the repo's value is that habit.

**The single most relevant thing it ships** is `lsa/`: a port of the paper's **D-LSA (Flat)** design to this exact
checkpoint, with the paper's own distilled index map (`qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt`,
SHA-256 pinned). That is the same LSA our plan is built on, already reduced to a working selection algorithm.

## 2. Per technique

"Cost" is the effort to land it in `strata-poly`, on the assumption that we keep our engine and our formats.

| technique | what it needs | maps onto our dense 27B? | expected win on sm_89 | cost |
| --- | --- | --- | --- | --- |
| **D-LSA (Flat)** — `lsa/`, `core.py` + `kernels.py` | pure Python + Triton; the paper's per-checkpoint distilled map; **its own KV layout** (see §4) | **the algorithm, yes** — the index space, the pooling, Ward, the bucket selection are backend-agnostic. **The code, no**: it is a vLLM attention backend over vLLM's paged cache | prefill +0.3% at 4k → **+3.3% at 51k** standalone on a 3090; +2.7% to +5.3% stacked on int8 GEMMs. Decode a wash. Measured on 24 GB, i.e. contexts below what 16 GB holds | high — it is the LSA follow-up, not a drop-in |
| int8-QK prefill attention (`PREFILL_ATTN=int8`) | Triton; gate is literally `num_heads == 24 && num_kv_heads == 4 && head_size == 256` | **the idea, yes; the kernel, no** — it reads `bf16 KV` only, and its speedup is over FA2's hd256, which we do not use as our baseline | 1.27×–1.35× **of the attention alone** at 4k–51k; +0.4% to +5.3% end-to-end | high: a new int8 QK^T kernel for our `attn` prompt path, and it must be validated against our k8v4 bit-identity |
| int8 tensor-core GEMMs (`INT8_ACT=int8`) | Marlin W4A8 kernels, sm86-tuned; int4 weights (AutoRound/GPTQ) | **no** — our weights are IQ1–Q4 GGUF, not int4 Marlin, and Marlin is an sm86/sm89 tensor-core int4 kernel we do not have | +27–30% prefill in **their** stack (1,437 → 1,845 tok/s at 1k) | not applicable |
| 16-bit recurrent state (`--mamba-ssm-cache-dtype float16`) | vLLM flag | **yes, and it is already decided** — this is exactly `docs/QUANT_MICROBENCH.md` row 3: fp16 is the winner, 0.88× the time, 8 → 16 slots | their framing ("37 of 64 requests running → 64") is the same result | **zero for us now**: it is the separate fp16-state follow-up |
| fp8 KV (`fp8_per_token_head`, `FA2_FP8KV`) | fp8 KV cache; FA2 plugin; Triton fp8 needs **sm89+** | **no** — 1 B a value is 2,176 B/token at our geometry, i.e. exactly q8_0's bytes for worse precision, and it breaks token identity | 0 over q8_0 by bytes | not applicable |
| int4 KV (`int4_per_token_head`, `spec-decode-int4-kv`) | vLLM's Triton backend; per-token-head scales | **no** — 0.5 B a value, but the accuracy and backend cost are documented as 2.3× prefill, 1.5× decode at 100k, and our k8v4 already gives 1,664 B/token at better fidelity | negative at our contexts | not applicable |
| KVarN k4v2 (`kvarn/`) | a **fork of vLLM 0.23** ported to 0.33; 2048-token blocks; Hadamard + variance normalisation | **no** — it is a vLLM cache backend. The *idea* (4-bit K / 2-bit V per 128-token tile) is a different design from k8v4 and would need its own store/read kernels and its own bit-identity gate | −0.16% perplexity, but decode 1.22× slower in batch at 100k and **2.13× slower** single-stream at 112k | not applicable |
| DFlash2 block drafter + lookup drafting | a separate 1.19 GB GPTQ checkpoint, vLLM's spec-decode API, `--no-async-scheduling` | **no** — no such checkpoint for our file set, and it is a vLLM runner feature | +55% where the model copies its prompt, +2–3% on prose | not applicable |
| MTP draft vocabulary (`prepare/build_draft_vocab.py`) | counted over 5.4M tokens of the model's own outputs; a separate 40k-row draft head | **partly** — this *is* our `draft_vocab`/`guess_vocab()` idea (`Qwen35::guess_vocab()` returns 131072) | they claim +10% single-stream from 97.5% coverage vs 92% | low-to-medium: a calibrated id list + the measurement to justify it. **Needs their calibration data**; we cannot reuse theirs (it is their model's output distribution) |
| prefix caching for the hybrid (GDN) model | vLLM's `PREFIX_CACHE`; state resumes at a block boundary | **partly** — we have conversation parking (`--conversation-cache-mib`) and `save_state`/`load_state`, which is the same idea with a different granularity | theirs: 23 s → 1 s on a follow-up turn over 24k | not applicable as a port; useful as a benchmark target |
| split-KV verify attention (`spec-decode-attn`) | Triton, multi-query decode over a paged cache | **not applicable** — our spec decode verifies through `fdec_k8v4`/`fdec_q8`, which already split KV (`fdec_fits`, `poly_fattn_test k8v4 --heads 24`) | — | — |
| sort-free small-k top-k / multi-block softmax | vLLM's sampler | **not applicable** — our sampler is `src/poly/main.cpp`'s `Sampler`, not vLLM's, and greedy needs no sort | — | — |
| tuned Marlin tile tables | sm86 measurements | **no** — Marlin only | 0.4% end-to-end even on their card | — |
| the ~55 vLLM fixes (auth deny-by-default, SSE keep-alive, env registration, KV-group sizing, stall sentinel, …) | a vLLM install | **no** — they fix vLLM; we are not vLLM | — | not applicable |
| vision tower CPU offload, embedding UVA, embed requant | vLLM model code | **no** — different engine, and our embedding is read from the GGUF as its own type | — | not applicable |

## 3. The one thing worth taking

**The D-LSA (Flat) selection design in `lsa/python/hyperqwen_lsa/core.py` and `kernels.py`.** Not as code — as the
worked-out answer to "what shape is the index, and what does a query do with it", which is the part of our LSA plan
that is still prose. Three concrete things it settles for us, and one it confirms:

1. **The RoPE-free slice is the tail, `k[64:256]`, and the repo computes it that way.**
   `core.index_keys`: `flat = key[:, :, rotary_dim:].reshape(t, -1)` → `@ W_K` → L2 normalise. Its README says "the
   four KV heads' 192 position-free dims, past the 64 rotated ones, side by side", i.e. a `[768, 64]` map per layer
   (4 heads × 192 = 768). That is exactly the slice I derived from `head_norm_rope_kernel`'s `if (i >= n_rot)` and
   wrote into `docs/QWEN38.md` — `rope.dimension_count = 64` of `key_length = 256`, the **tail**, not `[64, 128)`.
   Independent confirmation of a thing I would otherwise have had to argue.
2. **The index query sums the query heads of each KV group.** `core.index_queries`: `query.view(t, kv_heads, group,
   dim)[..., rotary_dim:].sum(2)` with `group = 24 / 4 = 6`, then `@ W_Q` (also `[768, 64]`). One index query per
   position, shared by every attention head — which is why one index list `J_i` serves all 24 heads.
3. **Bucket rows live in the KV slots themselves.** `slot(b + 8j) / 8`: bucket *j* of the 512-key block starting at
   *b* is stored in the row of slot `b + 8j`. The allocator, prefix cache and preemption then carry the index with
   the KV pages, and no new bookkeeping exists. That is a real design answer to a question our plan has open — where
   the 168 MB of index lives — and it is *cheap* because it borrows the paging we already have.
4. **It confirms the budget arithmetic I put in `docs/QWEN38.md`.** Their defaults are the paper's: `top_k` 4096,
   `top_p` 1024, window 512, sink 4, pools of 4, buckets of 8 capped at 16, `d_idx` 64 — the numbers from your
   correction, not the `k = 2048` block-sparse column. `dense_below = sink + top_k + window = 4612`. They measure
   ~2.4 KB of index a token over 16 layers against 33 KB of int8 KV (**~7%**), which is the same order as my
   ~10.5 MB a layer at 256K.

**What makes it cheap to take:** it is ~700 lines of Python and Triton whose *algorithm* is separable, and the
selection (score centroids, exact top-`p`, expand best-first to `k`, skip sinks and window) is arithmetic we would
otherwise have to re-derive from the paper. The port itself is the LSA follow-up, not this task.

**Where it would land in our tree**, per the design note already in `docs/QWEN38.md`: `Qwen35::attn`
(`src/poly/qwen35.cpp`) is the single function that hands the attention its key set, and an LSA read needs one
`int32_t` list per query row of at most `n_sink + k + n_w = 4,612` entries. The kernels that gather by such a list
do not exist in our tree; `attn_score_*` and `attn_value_*` index the cache by position today.

**The second thing, with a caveat:** the draft-vocabulary idea is ours already (`guess_vocab()`), but their result —
97.5% coverage buys ~10% single-stream over 92% — is a measurement we have never made. That is a cheap experiment
(no new code, just a calibrated id list and two runs) and it needs *our* model's output distribution, not theirs.

## 4. Traps

**The KV layout conflicts with `Qwen35::alloc_caches`.** HyperQwen's LSA stores index keys and bucket rows *in the
KV cache's own slot space*, one row a slot plus a packed row every 8 slots. Our `alloc_caches` allocates each format
as its own pair of arrays (`k_codes`/`k_scales`, `v_q4`/`v_q4_scales`, or `keys`/`vals`), sized
`n_kv * head_dim` a row. Interleaving 64 fp16 index dims into that would change the row stride of the k8v4 store and
the q8_0 scale positions, i.e. `store_q8`, `store_q4`, `fwht_rows` and the `poly_kv_q8_test` / `poly_kv_q4_test`
gates. **Do not adopt the layout.** The index belongs in its own allocation beside the cache, keyed by position —
which also keeps `attn_store` and `attn_rows` untouched, as the LSA note requires.

**fp8 KV is a precision change that would break k8v4 bit-identity for no bytes.** `fp8_per_token_head` is 1 B a
value = 2,176 B/token at our geometry, *identical* to q8_0's 1,664+512, with one fp32 scale per head on top. It buys
nothing in bytes and costs the bit-identity, and Triton's fp8 path needs sm89 (we have it) but their own table shows
it losing to fp8 FlashInfer on the same card. Not a candidate.

**int4 KV is the same trap with a bigger number.** Their own table: 355k-token pool (attractive) against 2.3× the
prefill time and 1.5× the decode time at 100k on the *same* card, because the Triton backend is that much slower
than FA2. And the accuracy is "needle ok, +0.3% perplexity" — a different bar from our bit-identity gate.

**The int8-QK prefill kernel assumes bf16 KV.** `prefill-attn-int8.patch` gates on
`not is_quantized_kv_cache(self.kv_cache_dtype)`. It cannot read k8v4 or q8_0, so it would need a second
dequantise-then-quantise step over our cache before the QK^T — which is the same trap `attend_batch` already
handles by dequantising to fp16. Its 1.27–1.35× is measured against FA2 at hd256; our prompt path is `attend_batch`
(the cuBLAS path) at group 6, so the baseline it beats is not ours.

**Its prefill win is a 24 GB result and does not transfer down.** Every LSA row is at 31k–137k context on a 3090.
At 16 GB our k8v4 ceiling is 61,440 tokens, where they measure **+19% slower** LSA than dense (the 31k row). Their
own conclusion is "a long-prompt prefill option, not a speed-up across the board" — on our card the crossover is
above what fits.

**Marlin and the drafter both assume weight formats we do not have.** W4A8 Marlin needs int4 weights with group
scales; our file is IQ1_S/IQ2/IQ3/Q4_K/Q5_K/Q6_K/Q8_0. DFlash2 needs a 1.19 GB GPTQ checkpoint that does not exist
for this file set.

**No calibration step we can run.** `prepare/build_draft_vocab.py` and the DFlash2 GPTQ pass both need a corpus of
the model's own outputs. We can generate one, but it is a separate measurement, not a port.

**Their state result and ours agree, so there is nothing to take — only a citation.** `--mamba-ssm-cache-dtype
float16` is their "37 of 64 requests → 64" fix. `docs/QUANT_MICROBENCH.md` row 3 measured the same question across
eight formats on this box and reached the same answer (fp16: 0.88× fp32's time, half the bytes, 25× int8's
accuracy). int8 and 4-bit state are refuted by that row; do not re-test them on HyperQwen's word or mine.

## 5. What I did not do

- Did not run any of it. No Docker image, no vLLM install, no model download. Every number above is quoted from the
  repo's own docs, with the card and protocol it states.
- Did not read all 55 patches; read `PATCHES.md`'s table (one line each) and the full text of the two that touch
  attention, the LSA port, and the long-context doc.
- Did not verify the LSA distilled map's SHA-256 or fetch it.
- The Prefrontal-AI reference implementation was not fetched either; `lsa/` is a port of it and the port's own
  `tests/check_core.py` is what holds it to the reference.
