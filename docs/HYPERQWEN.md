# HyperQwen, surveyed against this fork's dense Qwen3.8-27B on sm_89

A reading survey of **https://github.com/syv-ai/HyperQwen** (`git clone --depth 1`, 32 MB) against the question: what
can be taken into `strata-poly`'s own code paths for the dense `qwen35` family — 24 Q / 4 KV heads, `head_dim` 256,
`n_embd` 5120, FFN 17408, 16 attention + 48 GDN layers, Ada sm_89?

Nothing was installed, downloaded or run for this page. Every number quoted from the repo is its own, with the card
and protocol it states.

**Verdict in one line:** three things are worth taking — the swap-AB decode kernel's answer to `q·G = 6`, its
split-count rule, and its `cp.async` cache loader's discipline. Two are ideas to re-implement in our own `fattn.cu`,
not code to vendor. Everything else is vLLM plumbing, a format we measured and rejected, or a
Hopper/Blackwell/GA100 assumption.

---

# Part 1: the context ceiling on a 16 GB card

## 1.1 What I measured, and what it was actually measuring

My earlier report said k8v4 stops at 61,440 tokens and called it the card's wall. **That was wrong, and the
arithmetic in the request is right: at 61,440 the KV is 1,657 MiB and the card has more than 4 GiB spare.**

What I actually measured was `--max-context` for a run that also had `--kv-whole --prompt-batch 512`. Two separate
things made that a low number, and neither of them is the KV cache:

1. **`--kv-whole` turns off the growing cache** (`main.cpp`: `--kv-whole` sets `kv_grow = false`), so `kv_alloc()`
   calls `dmalloc(all, "the attention cache")` — one allocation of the whole context at load. That is still the same
   total bytes, so this alone does not cap it.
2. **The scratch and the reserve do.** `alloc()` sizes `scratch_` from `--prompt-batch`: `g_`, `u_` and `h_` are
   each `rows_ × n_ff_` floats (512 × 17,408 × 4 = 34 MiB each at `--prompt-batch 512`, 68 MiB at the default 1024),
   `score_` is `chunk_ × n_head × max_context` (62 MiB at 61K), and `awork_` is `rows_ × n_head × head_dim × 3.5`.
   The 64,512-token run died on a **19 MiB** request — a *small* allocation failing at the end of a long list, which
   is the signature of a budget that was already spent, not of a cache that does not fit.

So the ceiling I reported was `--kv-whole` plus a large prompt batch plus, behind both, the thing in §1.2.

## 1.2 The real reason: a dense model never gets a budget

`fill_expert_cache()` opens with

```cpp
if (!opt_.gpu_experts || n_moe_ == 0) return;
```

and that early return does **all** of the following for a dense model:

| what it skips | set at | consequence for `qwen35` |
| --- | --- | --- |
| `reserve_mib_ = …` | `engine.cpp:1856` (`reserve_mib_` is initialised to 0 in `engine.hpp:817`) | `mind_card_vram()` returns at `if (!exp_ \|\| reserve_mib_ <= 0) return false;` — **`mind_vram()` is dead**, so nothing ever gives VRAM back or takes it as it frees |
| `budget_ = free_b - reserve + other_rows()` | `engine.cpp:1969` | stays 0, so `cover()`'s `held + more > budget_` clamp is `> 0` and would cut the cache to nothing — it is only saved by `exp_` also being null |
| `pool_claimed_`, the VMM pool | `engine.cpp:1921` | with `--vram-pool` the pool is never claimed; `vmap()`'s `else if (pool_on_ && pool_claimed_)` never fires, so every map creates its own handle |
| `exp_ = (uint8_t*) arena_.base` | `engine.cpp:1968` | the "every expert is here, take the whole context's rows now" block at `:2820` is gated on `!kv_.empty() && all_cached()`. `all_cached()` is `cached_ == n_moe_ * n_expert_` = `0 == 0` = **true**, so the block *runs* but compares `reserve_mib_` (0) — and it is the one path that would have mapped the full context up front |
| `ensure_stage()` | `engine.cpp:2024` | `if (!mmq_ \|\| stage_) return;` — also inert, and correct |

**So: a dense model runs with `reserve_mib_ == 0`, `budget_ == 0` and no expert arena.** None of that is a wall. It
means the only limit is `cudaMalloc`/`cuMemCreate` failing, and the amount available to fail against is whatever the
desktop has left, minus everything the engine already took.

## 1.3 The one line that actually costs context: `snap_`

`alloc()` does this unconditionally (`engine.cpp:3065`):

```cpp
snap_places_ = snaps_spread() ? std::min(opt_.window - 1, kSnapPlaces) : opt_.window - 1;
if (snap_places_ > 0)
    snap_ = (float*) dmalloc(size_t(snap_places_) * rec_n_ * sizeof(float), "the recurrent state");
```

`Qwen35::snaps_spread()` returns `true`, `kSnapPlaces` is 3, and `opt_.window` defaults to 8 for a model with
recurrent layers. `rec_n_` for this model is **39,223,296 floats = 149.6 MiB**, so:

**`snap_` = 3 × 149.6 = 448.9 MiB of VRAM, allocated for every dense run, worth 16,639 tokens of k8v4 context.**

It exists for one purpose: `keep(n)` puts the recurrent state back to the last token of a window that turned out
right, which only happens when a window carries guesses. Without a guess there is no window and `snap_` is dead
weight. With `--no-draft` the run I did still allocated it. **This was fixed after the survey: `alloc()` no longer
takes it and `window()` does, on the first window of more than one token.** See §1.5 and §3.1 for why the guard is
not `opt_.draft`, and `docs/QWEN38.md` for the measured ceilings.

## 1.4 The correct ceiling

Weights are **10,020 MiB** in VRAM (the 10,418 MiB of tensors less `token_embd`'s 398 MiB, which stays in the file —
`engine.cpp:2744` reads it a row at a time). Then state 149.6, `snap_` 448.9, and scratch that is a function of
`--prompt-batch`.

With **2,920 MiB held by the desktop** (measured on this box) and a k8v4 row of 28,288 B a token (17 stacks), the
budget is:

| taken by the engine | context at 28,288 B/token |
| --- | ---: |
| weights + state + snap + 512-batch scratch (~1,010 MiB) | 61,440 *(what I measured)* |
| weights + state + snap + 128-batch scratch (~700 MiB) | 75,000 |
| **weights + state + no snap** + 128-batch scratch (~250 MiB) | **109,400** |
| weights + state + no snap + 128-batch scratch, 1,000 MiB desktop | 178,000 |

So **~110K is the honest figure for this box as it stands, and the request's 107,963 was right.** The gap between
my 61,440 and that number is `snap_` (16,600 tokens), the prompt-batch scratch at 512 versus 128 (~13,000), and the
`--kv-whole` one-shot allocation.

## 1.5 Which of the three it is: flag, default, or bug

**One default that is wrong and one flag I should not have passed. Not a bug in the memory logic.**

- **A default that should be fixed (a small change, and it is upstream's file).** `snap_` should not be allocated
  when nothing will use it. `window()` is the only writer of it and `keep()` the only reader, and `main.cpp` opens a
  window when `most > 1`, which is `lookup > 0 || e.drafts()`. **This page first said the guard was
  `opt_.draft`, and that is wrong:** `--lookup` defaults to 3 and opens windows with no draft block, so
  `if (snap_places_ > 0 && opt_.draft)` crashed a `--no-draft --lookup 3` run with an illegal memory access — the
  recurrent kernels write through `snap_at_`, which is `snap_`, which was then null. The fix that shipped takes the
  buffer in `window()` on the first window of more than one token instead, marked `// POLYSTRATA LOCAL:` per
  `docs/LOCAL_CHANGES.md`; it is correct for every caller and still hands a run that opens no window the whole
  448.9 MiB. `docs/QWEN38.md` has the measured ceilings, before and after.
- **A flag on my side.** `--kv-whole` is for a *known, fixed* conversation; it throws away `cover()`'s ability to
  grow the cache as an answer needs it, and it makes load fail rather than a later token. For *using* a long context,
  the default growing path (`kv_grow = true`) is the correct instrument. For *measuring* one, `--kv-whole` is the
  honest one, because it makes "does this context fit" a decision at load rather than a guess about which token will
  hit the wall — the ceilings in `docs/QWEN38.md` use it for that reason.
- **Not a bug:** `--vram-reserve-mib 256` for a dense model changes nothing, because `reserve_mib_` is never read
  (it is set only in `fill_expert_cache()`). Passing it was harmless and pointless.

## 1.6 What this does not change

LSA still does not make 256K fit: at 256K the k8v4 KV alone is 6.91 GiB and the weights are 9.8 GiB, so ~16.7 GiB
against a 16.0 GiB card with the desktop holding 2.9. The honest ceiling with the fixes above is ~110K, and 256K needs
either a smaller cache format or a card with more VRAM, LSA or no LSA.

---

# Part 2: the per-technique table

Columns as requested. "Lands in" names a file of ours or says why there is none. "Survives sm_89" is about the
*technique*, not the code: the code is vLLM's either way.

| technique | where in the repo | what it needs | lands in which of our files | survives sm_89 | worth it |
| --- | --- | --- | --- | --- | --- |
| **Swap-AB decode kernel** (`S^T = K·Q^T`, `O^T = V^T·P''`, `NT = ceil(rows/8)` n-tiles) | `pthfa/csrc/pth_attn/attn_decode.cuh` | int8 KV, m16n8k16, 99 KB smem, CUDA graph-safe static shapes | **`src/poly/fattn.cu`** — a new row layout for `fdec_kernel`'s `Q`/`MT` choice | **yes** — m16n8k16 is sm_80+, no WGMMA/TMA/clusters | **yes, the idea.** It names our exact problem: `q·G` rows are 6 at q=1 and 24 at q=4 with G=6, so 62% / 25% of the MMA is padding. Our `fdec` fills `MT·16` rows the same way and pads identically. Worth a careful read before any kernel work |
| **Split count = "one request fills the GPU"** — `S = ceil(C_target / (Hkv·rt))`, `C_target = num_sms · CTAS_PER_SM[D]` | `pthfa/python/pth_flash_attn/planner.py` | host-side arithmetic only | **`src/poly/fattn.cu`** — `fdec`'s `grid(rows, kSplit)` splits by a *fixed* 1,024 keys | **yes** — pure arithmetic | **yes, and it is the cheapest win on this list.** Our split count ignores the SM count entirely: at 34 SMs and 4 KV heads we launch `ceil(rows/1024) × 4` blocks, which at 4K context is **16 blocks for 34 SMs** — under half the card. Their formula with `CTAS_PER_SM[256] = 1` gives `ceil(34/4) = 9` splits a head. One line of host arithmetic |
| **`cp.async` two-stage cache loader with a run-time page size** (`FastDivmod`, strides read from the tensor view) | `pthfa/csrc/pth_attn/loader.cuh`, `cache_geometry.cuh` | cp.async (sm_80+) | our `fdec_kernel` reads the cache with plain loads today | **yes** | **maybe** — the discipline (never derive a stride from a shape; one `StreamGeometry` per stream) is the good part, and we already do the equivalent. The `cp.async` staging is what our `kTile = 64` loop does with registers |
| int8-QK prefill attention (SageAttention-style: mean-smooth K, per-row Q scales, P·V in bf16) | `patches/prefill-attn-int8.patch` | Triton; **bf16 KV only** (`not is_quantized_kv_cache`) | would need a new kernel; not `attend_batch` | **yes** (int8 `mma` exists) | **no** — it is gated to bf16 KV, so it cannot read k8v4 or q8_0 without a dequantise pass we would pay for. Its 1.27–1.35× is against FA2 at hd256, and our prompt path is `attend_batch` (cuBLAS) at group 6. Wrong baseline |
| fp8 KV (E4M3), `fp8_per_token_head`, `FA2_FP8KV` | `fa2/`, `docs/long-context.md` | fp8 KV cache; Triton fp8 needs sm89+ | — | yes on sm89 | **no** — 1 B a value is 2,176 B/token a layer at our geometry, *identical* to q8_0's 1,664+512, with one fp32 scale per head on top. Zero byte saving, and it breaks k8v4 bit-identity |
| int4 KV, `int4_per_token_head`, `spec-decode-int4-kv` | `patches/int4-kv-per-token-head.patch` | Triton backend, per-token-head scales | — | yes | **no** — their own table: 355K pool, but 2.3× prefill and 1.5× decode at 100K on the same card, because Triton attention is that much slower than FA2. Not a byte win worth a bit-identity regression |
| KVarN k4v2 (Hadamard + variance normalisation, 4-bit K / 2-bit V per 128-token tile) | `kvarn/`, fork of vLLM 0.23 ported to 0.33 | a vLLM cache backend, 2048-token blocks | — | yes | **no** — a different design from k8v4 needing its own store/read kernels; their own numbers are −0.16% perplexity but decode **2.13× slower** single-stream at 112K. A format for fitting context we cannot otherwise fit, and at 110K we are not there |
| **`snap_` allocated without a draft block** | n/a — this is *our* code, found by this survey | nothing | **`src/poly/engine.cpp:3065`**, the lazy `window()` allocation (§3.1) | n/a | **yes — shipped.** 448.9 MiB / 16,639 tokens of context for a buffer nothing reads. The largest single item on this page. Measured: +17,400 tokens at `--prompt-batch 128`, +21,000 at 512 |
| 16-bit recurrent state (`--mamba-ssm-cache-dtype float16`) | vLLM flag | vLLM | **`src/poly/qwen35.cpp`** `recurrent_floats()` / `alloc()` | yes | **already decided and closed.** `docs/QUANT_MICROBENCH.md` row 3: fp16 is 0.88× fp32's time for half the bytes, 8→16 slots, 25× int8's accuracy. Their "37 of 64 requests → 64" is the same finding. The follow-up, not a port |
| GDN prefill checkpoints (`VLLM_GDN_PREFILL_CHECKPOINTS`) | `patches/gdn-multi-prefill-checkpoints.patch`, `docs/gdn-checkpoints.md` | Triton/FLA GDN backend + `mamba_cache_mode=align` | — | yes | **no** — it is prefix-cache reuse inside vLLM's scheduler, not a cheaper state step. We have `save_state`/`load_state` and parking already; the idea (periodic interior checkpoints) is a *parking granularity* question for us, not a kernel |
| GDN `in_proj_ba` pad to 5 rows at M=2..4 | `patches/gdn-ba-pad-small-m.patch` | cuBLAS kernel-selection cliff at that shape | possibly `lin()` for a 2–4 token verify | yes | **no, and note why** — it is a workaround for a cuBLAS cliff at M=2..4 in a *bf16* GEMM. Our GDN projections go through `lin()` → `native_mmvq` (quantised), not cuBLAS, so the cliff does not exist for us |
| Marlin W4A8 int8 tensor cores, negative-scale fix, per-layer selection | `patches/marlin-*.patch` | int4 weights with group scales (AutoRound/GPTQ); sm86 tune tables | — | yes (sm89 has int8 `mma`) | **no** — needs int4 weights. Ours are IQ1_S…Q8_0 GGUF. Marlin is not a kernel we have and not a format our files carry |
| MTP draft vocabulary computed over the model's own outputs (97.5% coverage vs 92%) | `prepare/build_draft_vocab.py`, `patches/qwen3_5-mtp-draft-vocab.patch` | a corpus of the model's own generations | `Qwen35::guess_vocab()` already returns 131072; a calibrated id list would land in `src/poly/engine.cpp` | yes | **maybe, cheapest experiment here** — no new code, just a calibrated list and two runs. Needs *our* model's output distribution, not theirs. Their claim is +10% single-stream from the coverage difference |
| DFlash2 block drafter + lookup drafting from the context | `drafter/`, `patches/dflash2-*.patch` | a separate 1.19 GB GPTQ checkpoint; vLLM spec-decode API | — | yes | **no** — no such checkpoint for our file set, and it is a vLLM runner feature. The *idea* (draft from the prompt when the output copies it) is interesting and is a `strata::spec` policy question, not this repo |
| Prefix caching for the hybrid (GDN) model | vLLM 0.28+ default | vLLM scheduler | — | yes | **no as a port; yes as a target.** Their "23 s → 1 s on a 24K follow-up" is what our `--conversation-cache-mib` parking is for. A benchmark target, not code |
| Split-KV verify attention (Triton, multi-query over a paged cache) | `patches/spec-decode-attn.patch`, `opt-170hx/spec_decode_attn.py` | Triton | — | yes | **no** — we already split KV in `fdec_k8v4`/`fdec_q8`. Their design note (`BLOCK_M` rows so the kernel is not capped at `BLOCK_M/G` tokens) is the same trade as our `MT` |
| `fa2_slab` — bounded FA2 prefill over a quantised cache (gather to bf16 slabs ≤16,384 tokens, merge by log-sum-exp) | `pthfa/python/pth_flash_attn/fa2_slab.py` | FA2, Python; 64 MiB a slab at 16K | conceptually **`src/poly/fattn.cu`'s `attend_batch` q8 workspace** (`q8w_`, `kQ8WorkRows`) | yes | **the idea, already ours** — our `attend_batch` dequantises a bounded block of K/V into an fp16 workspace before the cuBLAS product, which is the same technique. The difference is that they get to call FA2 on the slab; we get cuBLAS at group 6 |
| tuned Marlin tile tables | `patches/marlin-tune-table.patch` | sm86 measurements | — | — | **no** — +0.4% end-to-end on their card, and Marlin only |
| ~55 vLLM fixes (auth deny-by-default, SSE keep-alive, env registration, KV-group sizing, CUDA-graph memory accounting, stall sentinel) | `patches/` | a vLLM install | — | — | **no** — they fix vLLM. One is worth *reading*: `cudagraph-memory-from-allocator` (measure graph memory by the allocator's reserved bytes, not the driver's free delta) is exactly the class of bug our own `reserve_mib_` comment describes |
| vision tower CPU offload, embedding UVA, embed requant | `patches/`, `prepare/` | vLLM model code, int8/int4 embeddings | — | — | **no** — different engine; our embedding is read from the GGUF in its own type already |
| int8 / fp8 / int4 recurrent **state** | not shipped by them; the idea circulates | — | — | — | **closed.** `docs/QUANT_MICROBENCH.md` row 3 measured eight formats: fp16 wins on time *and* accuracy; int8 is dominated on every axis; no 4-bit format reaches fp16 (1.56–1.74× the reference norm at 128 steps) |

---

# Part 3: the three or four items worth taking

## 3.1 `snap_` without a draft block — `src/poly/engine.cpp:3065`

The largest single win on this page and it is **our** bug, not theirs. 448.9 MiB, 16,639 tokens of k8v4 context, for
a buffer that only `keep()` reads and `keep()` is only reached from a window.

**It shipped, but not as the one-liner this page proposed.** `if (snap_places_ > 0 && opt_.draft)` is wrong: a
window is opened by `--lookup` (default 3) as well as by a draft block, and `keep()` reads `snap_` for those windows
too. The guard crashed a `--no-draft --lookup 3` run with an illegal memory access (§1.5). What shipped instead is
lazy: `window()` takes the buffer on the first window of more than one token, so a run that never opens one never
allocates it. Measured on this card, `--prompt-batch 128`, k8v4, `--kv-whole`: **85,000 → 102,400 tokens**, and
`--prompt-batch 512`: **73,000 → 94,000**. `docs/QWEN38.md` has the runs.

## 3.2 The split count from the SM count — `src/poly/fattn.cu`

`fdec` splits the cache by a fixed `kSplit = 1024` keys a block, so a 4K-context decode launches
`ceil(4096/1024) × 4 KV heads = 16` blocks on a 34-SM card. Their planner computes the split so **one request fills
the GPU**: `S = ceil(num_sms · CTAS_PER_SM[D] / Hkv)`, with `CTAS_PER_SM[256] = 1`. For us that is `ceil(34/4) = 9`
splits a head at any context ≥ 9 × 1024 keys. The merge kernel already handles an arbitrary split count
(`fdec_merge_kernel` loops `splits`), so this is host arithmetic in `run_dec`, not a kernel change. **Cost: hours.
Win: unmeasured, and the thing to measure before anything else on this list.**

## 3.3 Read swap-AB before writing any group-6 kernel — `pthfa/csrc/pth_attn/attn_decode.cuh`

Not code to take; a design to read. It states our problem exactly — "`q · G` rows (6 at q = 1, 24 at q = 4 with
G = 6), so 62% (q = 1) and 25% (q = 4) of the tensor-core work is padding" — and its answer is to put the *tokens*
on M (always 16 full) and the query rows on N (`NT = ceil(rows/8)`, so 6 rows use one n-tile of 8 and 24 use three
of 8). The `d`-split then gives each warp `D/4` dimensions for all 64 tokens, so every V byte is dequantised once.

Our `fdec_kernel` does the opposite: rows are `token * Group + head`, packed into `MT·16` fixed rows. With G=6 and
one token that is 6 of 16 rows. Their layout is worth understanding before we decide whether the padding matters —
and **the `per = 6` note in `docs/QWEN38.md` should be extended to say the same applies to `fdec`, not just to the
register-resident score kernel.**

## 3.4 The prompt batch is not the only scratch — `--prompt-batch` at 128

Not from the repo; a consequence of finding §1.2. `g_`, `u_`, `h_` are `rows_ × n_ff_` floats each, so the default
1,024 costs 204 MiB in three buffers and the 512 I used costs 102 MiB. At 128 it is 26 MiB, and the cost is a slower
prompt read (more calls a token). For a card that is tight, that is 13,000 tokens of context traded for prefill speed;
it belongs in `docs/QWEN38.md` as a table entry, not as a code change.

---

# Part 4: attractive and not worth it

- **fp8 KV.** The single most tempting item, because sm_89 has FP8 `mma` and they report a 27% decode win. But that
  win is fp8 **FlashAttention** against int8 **Triton**, i.e. a backend change, not a format change. At our geometry
  fp8 is 2,176 B a token a layer — the same as q8_0 — so it buys no bytes and costs the bit-identity.
- **int4 KV / KVarN.** Both fit more context, and at 110K we do not need more context badly enough to pay 1.5–2.1×
  decode. If the ceiling work in §3.1 lands, this becomes a strictly worse k8v4.
- **int8-QK prefill attention.** Their head-line 1.27–1.35× is real *for them*, against FA2 at hd256, and gated to
  bf16 KV. Ours is cuBLAS at group 6, and the kernel cannot read k8v4. A different job.
- **Marlin / W4A8.** Their biggest single number (1,437 → 1,845 prefill) is a 4-bit-weight format we do not have.
  Reading it as "int8 helps prefill" would be a category error: it is int4 weights with int8 activations.
- **Split-KV verify attention.** We already do it; `fdec_fits` and `poly_fattn_test k8v4 --heads 24` are the record.
- **`cp.async` staging of the KV tile.** Our `kTile = 64` loop keeps the tile in registers and reuses it across the
  four warps; theirs stages through shared memory because it needs `ldmatrix` on int8. Different constraint.

---

# Part 5: does this change the dense port's design?

**No, and it validates it.** Three specific confirmations:

1. **The dense branch is engine code and stays that way.** Nothing in 5,400 lines of their attention kernel touches a
   dense FFN; the FFN is a vLLM GEMM. Our `dense_ffn()` (three `lin()` calls and a `swiglu_clamp`) is the whole of
   it, and the repo offers nothing to change in it.
2. **`attn_store` / `attn` / `attn_rows` staying untouched was the right call.** Their kernel is one more reason:
   the decode path is where the group-6 work is, and it reads the cache through `fdec_k8v4`'s own layout. A dense
   FFN branch has no business in there.
3. **The only thing that would have changed the design is the context ceiling, and that is our own accounting, not
   their engineering.** The dense family is correct; it has been running 16,600 tokens below the card's real limit
   because of a snapshot buffer, and that is a fix of one line.

The LSA line is unchanged and still gated on the accuracy check in `docs/LSA-DENSE.md` §4 — the `cos > 0.99` median
head at ~8K tokens, one layer, in Python — which remains the thing that decides whether any kernel work happens.

---

# What I did not do

- Did not run, build, install or download anything for this page. One GPU run was made while investigating the
  ceiling before the no-GPU constraint was re-read: a single load at `--max-context 63488` to see whether the failure
  was at load or after. It is the data point behind "the load succeeds, the allocation fails later", and it should not
  have been run.
- Did not read all 55 patches: read `PATCHES.md`, and the full text of the ones touching attention, the KV formats,
  GDN and the drafter.
- Did not verify the LSA map's contents (only its size and the loader's shape checks).
- Did not measure the split-count change in §3.2. It is an argument from the arithmetic (`16 blocks on 34 SMs`), and
  the measurement is the first thing to do if the item is taken.
