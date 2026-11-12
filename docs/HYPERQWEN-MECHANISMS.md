# HyperQwen: the mechanisms, deep, against our code

A second pass. The first pass answered "what can be ported" against vLLM's shape and was the wrong question: several
of these survive as *mechanisms* even though their implementation does not. This one goes to a number or a
`file:line` for each, with no GPU runs.

---

## The table

| # | mechanism | fits our code? | the number | days | verdict |
| --- | --- | --- | --- | ---: | --- |
| **B** | **swap-AB decode** (`S^T = K·Q^T`, `O^T = V^T·P''`), tokens on MMA's M, query rows on N | the *idea* yes, the code no — `fdec_kernel` would be re-indexed, not rewritten | **6.25% padding at `--window 8`, 25% at `--window 4`, 62.5% at `--no-draft`.** But the kernel is **memory-bound at 11% of the roofline**, so the padding costs nothing measurable | 0 to read, 4–6 to build | **out.** The premise was wrong: the MMA is idle either way |
| **C** | split count from SM count | **yes, as one constant** — `kSplit` has 4 functional sites and none of them is a planner | `CTAS_PER_SM = 2` (register-bound). 1024 fills the card at **17,408** tokens; **256 fills it at 4,352**. Blocks/SM stay 2 either way; scratch 7.6 → 30.2 MiB at 61K | 0.5 | **in, as a `constexpr` experiment.** Helps only ≤17K, worth sub-1% |
| **A** | int8-QK prefill (3-way int8 split, `m16n8k32`) | **no**, on difficulty — the arithmetic leaves it alive | `Q·Kᵀ` is **5.83%** of a 32K dense prefill and `P·V` another 5.83%; attention is **17.7%** total. Three int8 MMAs against fp16-hi/lo's two at 2× rate is **1.33×**, worth **~1.45%** of prefill at 32K, ~2.5–3% at 61K | 8–10 | **out on cost, not share.** Real but small win, large bill |
| **D** | calibrated draft vocabulary | **yes — the flag exists and takes a prefix** (`--draft-vocab N`, `engine.hpp:233`) | head is **52%** of a guess (360 of 694.5 MiB). 131,072 → 65,536 rows is **5.88% → 4.36%** of a 43 ms token. Plus **500 MiB of VRAM** back if `output_` becomes the gathered head — ~18,500 tokens of k8v4 | 1–2 | **in, first.** Above the 2% bar on time, and the VRAM half is bigger |
| **E** | GDN interior checkpoints | yes, onto `save_state`/`load_state` | 149.6 MiB a state; a checkpoint every 4,096 tokens at 61K is **15 × 149.6 = 2.2 GiB** | — | **out, one paragraph below** |
| **F** | prefix caching, as a test | already there | `--conversation-cache-mib` parks a whole conversation; theirs caches a prefix at block granularity | 0.5 | **in, as an experiment** |
| **G** | `cudagraph-memory-from-allocator` | our accounting is **already correct** | we add the graphs to `reserve_mib_`, the same class of fix | 0 | **out — already done** |

---

## B. Swap-AB in `fdec_kernel` — and the 62% claim, which is wrong at our `MT`

### B.1 Our layout, exactly

`fdec_kernel` (`fattn.cu:318`) holds rows as `row = token * Group + head of the group` (`:399`, `:483`, `:520`) in a
`Q = MT × 16` row space (`:326`). One launch carries `kDecTokens = kDecRows / Group = 32/6 = 5` tokens
(`:731`), and `MT` is 1 while `c·Group ≤ 16` else 2 (`:735`). `fdec` is entered only for `n < kStageRows = 32`
(`qwen35.cpp:629`).

So the real cases, with rows used against `Q`:

| call | tokens `c` | rows used | `Q` | padding |
| --- | ---: | ---: | ---: | ---: |
| `--no-draft`, one token a step | 1 | 6 | 16 | **62.5%** |
| `--window 4` | 4 | 24 | 32 | **25.0%** |
| `--window 8`, first launch | 5 | 30 | 32 | **6.25%** |
| `--window 8`, second launch | 3 | 18 | 32 | 43.75% |
| `--window 8`, both launches | 8 | 48 | 64 | **25.0%** |

**This is the answer to your B.4: at our `MT`, "62%" is right only for `--no-draft`.** With a window of 8 — the
default, since `opt_.window = 8` for a model with recurrent layers — the first launch is **6.25%**, and the pair
averages **25%**. The number you quoted to the user is the single-token case, which is the one configuration where
`MT` cannot pack.

### B.2 Their claim, checked

Their `attn_decode.cuh:6` says "`q · G` rows (6 at q = 1, 24 at q = 4 with G = 6), so 62% (q = 1) and 25% (q = 4)
of the tensor-core work is padding". Their row space is `NT × 8` with `NT = ceil(rows/8)`: 6 rows use one n-tile of
8 → **25% padding**, not 62%. 24 rows use three n-tiles of 8 → 0%.

**Their two numbers are the old kernel's, not the swap-AB one's.** The 62% and 25% are what the *pre-swap-AB* kernel
(`attn_kernel.cuh`, 16 rows on M) pays — the paragraph is explaining *why* they wrote a second kernel. The 62% at
q=1 is 6 of 16; 25% at q=4 is 24 of 32. Both are the same arithmetic as our `MT=1` and `MT=2` columns above,
because both kernels put rows on the fp16 MMA's M.

So: their claim is correct *about the kernel they replaced*, and swap-AB's own padding is 25% at q=1 and 0% at q=4.
**Neither number transfers to us**, because our launch carries 5 tokens, not 1 or 4.

### B.3 What swap-AB would cost in our code

Not a re-indexing — a rewrite of the tile loop's core. Four things move:

- **the cache read transposes.** `kg`/`vg`/`vstride` (`:344-349`) address `key * width + d`, a row a key. Swap-AB
  needs K as an MMA A-fragment with tokens on M, i.e. `[d][token]`; our reads are `uint4`/`uint2` gathers a thread
  wide in `d`, and the K half of the tile is read once per warp for its own 16 keys (`:370-380`). Making it a
  fragment means an `ldmatrix`-shaped load, which for int8 means a different lane mapping than our `i8pair`
  (`:293-297`).
- **`jx`/`bk` fragment construction changes.** `:441-460` builds `a[4]` from `Qs` and `b[2]` from the K words; swap-AB
  exchanges those roles.
- **`Ps` becomes `P[row][token]` and needs `movmatrix`.** Ours is already `Ps[row * PS + …]` (`:329`, `:498`) written
  by `t`-lane and read back as an A-fragment (`:519-524`). That part is close to theirs.
- **`part` and the merge change shape.** `part` is `[split][grp][row][D]` (`:583`), and `fdec_merge_kernel`
  (`:602`) reads `part[idx * D + d]` with `row = tok * Group + h % Group`. With rows on N the natural layout is
  `[split][d][row]`, which means the merge reads a transposed `part`. That is the whole merge kernel.

**Cost: 4–6 days**, and the risk is that every one of the 64 `fdec` bit-identity tests
(`poly_fattn_test k8v4 --heads 24`) has to come out identical.

### B.4 The honest bound — the part that decides it

**The kernel is memory-bound, so the padding is free.** From the compiled object
(`cuobjdump -res-usage build-poly/.../fattn.cu.o`): the group-6 `MT=1` kernel is `REG:255 SHARED:11264`. Per CTA it
reads `kSplit = 1024` keys of K (int8 + scales, 272 KiB) and V (Q4_0 + scales, 144 KiB) = **416 KiB**, and does
`2 × (16·1024·256 + 16·256·1024)` = **16.8 MFLOP**. That is **39.4 FLOP/byte**.

The card's roofline is 100 TFLOP/s ÷ 288 GB/s = **347 FLOP/byte**. We are at **11% of it.**

| rows used | FLOP/byte |
| ---: | ---: |
| 6 of 16 | 14.8 |
| 16 of 16 | 39.4 |
| 30 of 32 | 73.8 |

Even at 30 of 32 we are at 21% of the roofline. **The MMA pipe has four-fifths of its capacity idle in every
configuration**, so making the multiplication exact on the used rows buys no time at all.

**So yes: the 62% figure does not support the recommendation, and swap-AB is not worth 4–6 days for us.** The
number you quoted to the user should be corrected to "25% at a window of 8, 6% on the first launch, and it does not
matter because the kernel is DRAM-bound at 11% of the roofline".

---

## C. Split count — and whether it helps at the contexts we serve

### C.1 Where the card is underfilled

`grid(rows, kSplit)` with `kSplit = 1024` (`:261`), `kv_heads = 4` blocks in `y`. Occupancy is **2 CTAs/SM**
(255 registers × 32 lanes × 4 warps = 32,640 of 65,536; shared 11,264 + 1,024 reserved, 8 fit), so **68 CTAs
concurrently**.

| context | splits | CTAs | waves of 68 |
| ---: | ---: | ---: | ---: |
| 1,024 | 1 | 4 | 0.06 |
| 4,096 | 4 | 16 | 0.24 |
| 8,192 | 8 | 32 | 0.47 |
| 16,384 | 16 | 64 | 0.94 |
| 61,440 | 60 | 240 | 3.53 |

**Underfilled below ~17,408 tokens**, saturated above.

### C.2 The right `CTAS_PER_SM`

**2, from the compiled object**, not 1 as theirs (they assume a bigger register budget or a different head size).
`REG:255` is the cap, so the register file gives exactly 2 blocks: `65536 / (255·32·4) = 2.0`.

### C.3 `fdec_merge_kernel` and `fdec_floats`

The merge loops `for (int i = 0; i < splits; ++i)` with `splits = int(p / kSplit) + 1` (`:606`) — **so it does loop
an arbitrary count, but derives it from the same `kSplit` constant.** `kSplit` appears in the kernel (`:333`, `:334`),
the merge (`:606`), and `fdec_floats` (`:630`). `part` is `splits × kv_heads × kDecRows × D` and `fdec_floats` returns
`grid(rows,kSplit) · kv_heads · kDecRows · (D+2)`.

**So this is not a free planner change**: the split count and the split *size* are the same constant. Choosing a
count independently of the size needs a real `split_size` parameter threaded through all three — **1–2 days**, plus
the scratch grows linearly in the count (`part` is `splits × 4 × 32 × 256 × 4 B` = 128 KiB a split).

### C.4 Does it help at the contexts we serve? — **no, barely**

`kSplit = 1024` reaches 68 CTAs at **17,408 tokens**. Below that we are in **0.06–0.94 waves**. But:

- **At `--no-draft`, one token a step at 8K context, the whole attention is 0.05 ms** of a ~43 ms token (memory:
  13 MiB at 288 GB/s). Filling the card could make it 0.02 ms. **0.07% of a decode token.**
- **At 61K the card is already 3.5 waves over** — the fix does nothing there.
- The dense model spends its decode time in the FFN's weight reads (1,121 TFLOP of FFN against 6.6 TFLOP of QK at
  32K), not in attention.

**Verdict: the only version worth anything is `kSplit` 1024 → 256**, which reaches 68 CTAs at **4,352 tokens**
instead of 17,408 — one constant, one rebuild, and it helps *every* context below 17K rather than needing a planner.
Even that is a sub-1% decode change. **It is a one-line experiment to run once the card is free, not a work item.**

### C.5 The `kSplit` 256 change surface, exactly

`kSplit` is `constexpr int kSplit = 1024` at `fattn.cu:261` and appears at **exactly four functional sites**:

| site | use | changes at 256? |
| --- | --- | --- |
| `:333`, `:335` | the kernel's `r0 = split * kSplit`, `r1 = min(r0 + kSplit, last + 1)` | no — it is the loop bound |
| `:728` | `const int splits = grid(rows, kSplit)` | no — recomputed |
| `:630` | `fdec_floats` = `grid(rows,kSplit) · kv_heads · kDecRows · (D+2)` | no — recomputed |
| `:607` | the merge's `splits = int(p / kSplit) + 1` | no — recomputed |

**Nothing else changes, because neither the register count nor the shared memory depends on `kSplit`.** Verified by
reconstructing both from the declarations — `Qs[Q·(D+8)] + Ps[Q·(TILE+8)] + red[2][W][Q]` at `D=256, TILE=64`:
MT=1 gives 8,448 + 2,304 + 512 = **11,264 B** and MT=2 gives **22,528 B**, both matching `cuobjdump`'s `SHARED:`
exactly. `TILE = 64` is the tile inside the loop and is a separate constant from `kSplit`.

**So the resident blocks do NOT double.** They stay **2 a SM**, register-bound at `REG:255` (32,640 of 65,536
registers used by four warps of 255), and 256 keys a block does not touch that.

What *does* change is the split count and therefore the scratch: `part` is
`splits × kv_heads × kDecRows × (D+2)` floats, so

| context | kSplit 1024 | kSplit 256 |
| ---: | ---: | ---: |
| 4,096 | 0.50 MiB | 2.02 MiB |
| 17,408 | 2.14 MiB | 8.57 MiB |
| 61,440 | 7.56 MiB | 30.23 MiB |

`fdec_fits` already checks `need <= scratch_floats` (`:715`) and `score_` is `chunk_ × n_head × max_context`
floats — at 61K that is 62 MiB, so a 30 MiB `part` fits inside it. **One `constexpr` and a rebuild**, with
`poly_fattn_test k8v4 --heads 24` as the gate. The gain is that the card fills at 4,352 tokens instead of 17,408.

---

## A. int8-QK prefill — the recipe, and why it does not fit

### A.1 The recipe, extracted

`qsa_prompt_attn.cu:1086-1113` is not SageAttention's mean-smoothing. It is a **three-way int8 decomposition of the
non-K operand**:

> K and V enter as the int8 codes they are — no int8→fp16 conversion — and the other operand as THREE 8-bit parts:
> q per (head, 64-dim group) and `p·v_scale` per (head, chunk) scaled to 127, then `X = rint(x·65536)` split exactly
> into `h` (signed), `l` (signed) and `m` (unsigned) bytes — 24 bits, as fine as FP32 itself and finer than the FP16
> hi+lo split v2 uses (~22 bits).

`split3` at `:1102` does the split; `imma16832` and `imma16832_us` at `:1094`/`:1099` are `m16n8k32` s8/u8. So the
products are `s8×s8→s32` for the two signed parts and `u8×s8→s32` for the low byte, summed in fp32 as
`h + (l·256 + m)/65536`. Nothing stays fp16: both products are int8, and only the join is float.

**What it is measured to buy, and where.** The note at `:1086` says int8 MMA runs ~3.1× FP16 on **sm_120** (732 vs
233), which is Blackwell and not our card. The measured end-to-end figures are at `:2556`: **+1.6% / +1.8% prompt at
8K / 16K on sm_120**, and **+1.1% / +1.3% on 2× RTX A4000 (sm_86)**. The same comment says "Sm_80, **8.9**, 9.0 and the
datacentre Blackwells run the same kernel **unmeasured**" — so the sm_89 case has no number, and the 3.1× does not
apply to Ada, where int8 is 2× fp16 rather than 3.1×.

### A.2 Can it read k8v4? — yes, and that is the good news

Our k8v4 stores K exactly as `int8` codes plus one fp16 scale per 32 values (`alloc_caches`, `qwen35.cpp:210-215`).
That is the format the v3 kernel consumes: `k_codes` directly, no dequantise pass. The scales fold in after the dot
(`s = (q · k_int8) · k_scale`), which is exact because the scale is constant along the row.

**But the V side is Q4_0 rotated (`fwht_rows` before the store, `qwen35.cpp:578`), and the v3 kernel wants V as
int8** (`:1086` "K and V enter as the int8 codes they are"). Reading k8v4's V would need either a Hadamard
un-rotate and int8 requantise per tile (a real pass over the cache), or a q4 variant of the kernel. **So the byte
compatibility is one product out of two.**

### A.3 What the dense prefill path would have to become

`attend_batch` at group 6, 24 heads, `head_dim` 256: 24 rows of Q against 4 KV heads, i.e. **6 query rows a KV
head** — the same 6 that `fdec_group_ok` handles for decode. `imma16832`'s `m16n8k32` has **32-deep K**, so the
instruction count is `D/32 = 8` steps a head against fp16's `D/16 = 16`, and the row padding is 6 of 16 either way.

**The padding problem does not go away with int8.** `m16n8k32` still has 16 rows on M, so 6 heads use 37.5% of it —
the same fraction. What changes is the *destination*: an int8 path would be a new kernel beside `attend_batch`, and
the group-6 padding is identical.

### A.4 The share of prefill — corrected, and it does not close

**An earlier version of this section said `Q·Kᵀ` was 0.41% of a 32K prefill. That was wrong by the layer count: it
was one layer's 6.6 TFLOP over the whole prefill, and there are 16 attention layers.** The correct table, causal
(`T²/2` pairs), 32,768 tokens:

| component | TFLOP | share |
| --- | ---: | ---: |
| FFN (64 layers × 3 × 5120×17408) | 1,121.5 | 61.98% |
| GDN projections (48 layers) | 364.4 | 20.14% |
| GDN delta rule (48 layers) | 2.5 | 0.14% |
| **`Q·Kᵀ` (16 layers, 24 heads, d=256)** | **105.6** | **5.83%** |
| `P·V` (16 layers) | 105.6 | 5.83% |
| attention projections (16 layers) | 110.0 | 6.08% |
| **total** | **1,809.4** | |

Attention is **17.7%** of a dense prefill, not 1.6%. So the arithmetic does not close the item, and A is **out on
difficulty, not on share**: a group-6 prompt kernel at `m16n8k32` is 8–10 days, and the ceiling it buys is now
computable.

**What int8 would actually recover.** The v3 recipe is three int8 MMAs (`h`, `l`, `m`) for the non-K operand, chosen
because three int8 MMAs cost less than the fp16 hi+lo's two. On Ada int8 `mma` is 2× fp16's rate, not sm_120's 3.1×,
so three int8 ops are **1.5 fp16-equivalents against fp16-hi/lo's 2.0 — 1.33×**. On `Q·Kᵀ` alone that is
`5.83% × (1 − 1/1.33) = **1.45%** of prefill` at 32K, before the group-6 kernel's own padding (6 of 16 rows, so
another 0.63× of that), and `P·V` is a second product the recipe also covers.

At 61,440 tokens the attention share roughly doubles — `Q·Kᵀ` and `P·V` grow as `T²` while the FFN stays `T` — so
the same 1.33× is worth perhaps **2.5–3%** there. That is the same order as the +1.1%/+1.3% measured on sm_86 and the
+1.6%/+1.8% on sm_120, which now makes sense: those were **whole-path** numbers on models whose prefill also has an
indexer taking a share.

**Verdict: out on difficulty, and the trade is honest.** 8–10 days for a new group-6 prefill kernel, to buy ~1.5% of
prefill at 32K and ~2.5–3% at 61K, against an FFN that is 62% of the work and a GDN that is 20%. It is not the
"0.41%, forget it" the first pass claimed — it is a real but small win with a large bill, and it should be ranked
below the draft head (D), which is 1–2 days for a comparable slice of decode.

---

## D. Draft vocabulary — the mechanism is bytes, not hit rate

### D.1 The finding: the machinery exists and takes a prefix

**`--draft-vocab N` already does what this item needs — except that `N` is a count of the first ids.** Two places
say so, and neither of us noticed on the first pass:

```cpp
// engine.hpp:233-236
int draft_rows() const {
    const int v = opt_.draft_vocab >= 0 ? opt_.draft_vocab : guess_vocab();
    return v > 0 ? std::min(v, n_vocab_) : n_vocab_;
}
```

```cpp
// main.cpp:1410, and the usage line at :1243
--draft-vocab N     the block guesses among the vocabulary's first N tokens
```

So the flag exists, it is honoured, and `draft_rows()` is used as **`head.n_out = draft_rows()`** before
`lin(head, …)` (`engine.cpp:5280`, `:5806`, `:5831`) — a row *count* on a contiguous matrix. That is the gap: the
first 131,072 ids are whatever the tokenizer put there, not the ids a model emits. **`guess_vocab()` is
`return 131072` (`qwen35.hpp:34`) with a comment whose provenance is not recorded** — a prefix picked for a reason
nobody wrote down.

`native_mmvq` reads its weight block **contiguously** (`native_mmvq.cu:3908` takes `const void* weights` and
`n_out`; the launch walks rows from the base). A prefix therefore works for free and an arbitrary id set does not.

### D.2 Where the draft's time actually goes

One guess, one row, is a pass through the MTP block plus the head matvec (`draft_rows_of`, `engine.cpp:5224`). Byte
counts from this GGUF's own tensor types, time at 288 GB/s:

| stage | tensor | MiB | µs |
| --- | --- | ---: | ---: |
| `d_proj_` | `blk.64.nextn.eh_proj` Q6_K | 41.0 | 149 |
| `attn_q` | `blk.64.attn_q` Q6_K | 49.2 | 179 |
| `attn_k` / `attn_v` | Q8_0 ×2 | 10.6 | 39 |
| `attn_output` | `blk.64.attn_output` Q6_K | 24.6 | 90 |
| `ffn_gate` / `ffn_up` / `ffn_down` | Q6_K ×3 | 209.1 | 761 |
| **block subtotal** | | **334.5** | **1,218** |
| **head matvec** | `output.weight` Q4_K, `draft_rows()` rows | **360.0** | **1,311** |
| **total, one guess** | | **694.5** | **2,529** |

**The head is 52% of a guess and the block is 48%.** So a vocabulary cut touches **at most half** the cost of a
draft — and the block half is irreducible without quantising the MTP block itself, which is a different item.

The `Likeliest` selection is not a cost: `likeliest` (`kernels.cu:2305`) is one block of 1,024 threads over
`n_out` floats, 512 KiB at 131,072 — 1.8 µs of bandwidth, and it is an argmax with a truncated softmax
(`expf(x - m)` only over `x > m − 20`), not a full-vocabulary reduction. The pinned handoff is one `Likeliest`
struct. **Neither scales with the vocabulary in a way that matters.**

### D.3 A calibrated set: what it costs in our code

Two shapes are possible and one fits.

**(a) Gather the rows into a dense matrix at load.** A `Dev` of `n_in × n_ids` rows, copied out of `output.weight` on
the host at load, then `head = gathered; head.n_out = n_ids`. **This is a drop-in** — `draft_rows()` already returns a
count, `lin()` already takes a `Dev`, and the arithmetic is identical because a row is a row. Cost in VRAM, at a
Q4_K row of 2,880 B:

| ids | MiB |
| ---: | ---: |
| 40,960 | 112.5 |
| 65,536 | 180.0 |
| 131,072 (today) | 360.0 |
| 248,320 (full) | 682.0 |

**(b) An index array consulted by the kernel.** This needs `native_mmvq_ids`-style indirection in the head's matvec —
a new kernel path for one caller — and saves the VRAM. **Not worth it**: at 65,536 ids the gathered copy is 180 MiB,
and the engine already holds 682 MiB of `output_` it could reuse.

**So (a).** And note the accounting: today the engine uploads the **full** `output_` (682 MiB) and then truncates
`n_out` per call, so a gathered 180 MiB head would be **added** to the 682 unless `output_` stops being uploaded
whole. That is a second, separate 500 MiB win — quantified in D.5.

### D.4 Producing the list without training

Their builder (`prepare/build_draft_vocab.py`) is a **frequency count, not a training step**: it counts tokens over
`.txt`/`.jsonl`/`.py`/`.parquet` corpus files, ranks by count with ties by id, appends special tokens, and slices
those rows out of the int8 head. Its own docstring: 8.8M tokens of Danish web text, English Wikipedia, Python and the
model's own chat outputs, **held-out coverage 95%**. It also warns that the result is language-specific — its shipped
list holds **3 of the vocabulary's 55,328 Han ids and 1 of 18,580 Cyrillic** — and offers `--stats` (coverage per
script) and `--add-scripts` to append a script's rows within a budget.

Three sources, ranked by feasibility for us:

1. **The model's own output on a document set (best).** This is the only one that measures the target's distribution,
   and it needs no training: generate a few million tokens, histogram, keep the smallest set covering ~95%. Their
   held-out number is the method's own benchmark. **Feasible**, and it is the same shape as their pipeline.
2. **Frequency from a text corpus (second).** Cheaper — no model run — but it measures the *corpus*, not the model,
   which is exactly the 92%-vs-97.5% gap they report. Needs tokenising with **our** tokenizer, which we can read
   from the GGUF.
3. **A heuristic from `output.weight` row norms (weakest).** Cheap and training-free, but a large-norm row is a token
   the head can score *distinctly*, not a token the model *emits*. Expect materially below the corpus method, and
   there is no measured precedent for it.

**Not viable: the MTP block's weights.** They say which tokens the block can score, not which the target writes.

### D.5 The honest bound — above the 2% bar, and the VRAM half is bigger

A decode token at 61K is ~43 ms measured (23 tok/s), and one model step reads ~10,020 MiB of weights, which at
288 GB/s is 36.5 ms — so the engine is at **85% of peak bandwidth** and the step is weight-read dominated, as
expected.

| one guess | MiB | of a 43 ms token |
| --- | ---: | ---: |
| block + head at 131,072 (today) | 694.5 | **5.88%** |
| block + head at 65,536 | 514.5 | 4.36% |
| block + head at 40,960 | 447.0 | 3.79% |
| block + head at the full vocabulary | 1,016.5 | 8.61% |

**So a cut from 131,072 to 65,536 saves 1.52 percentage points of a decode token — 5.88% → 4.36%.** That is *not*
cosmetic: it is above the 2% bar, and it is per guess, so with a window of 8 it compounds across the drafts a step
proposes. Their "+10% speed" is consistent with that magnitude once a speculating server's accept rate is folded in.

**The larger prize is the double upload.** `output_` is uploaded whole (682 MiB) and `head.n_out` truncates the
*read* per call, so the full head sits in VRAM whether it is used or not. A 65,536-row gathered head would let
`output_` be the gathered matrix instead of the whole thing — **682 → 180 MiB of VRAM, 500 MiB back**, which on a
card where the context ceiling is the binding constraint is worth ~18,500 tokens of k8v4.

**Verdict: in, 1–2 days, and the VRAM half is the bigger half.** The list itself is a Python histogram; the wiring is
a gathered `Dev` in `load_drafter` plus `draft_rows()` returning the list's length. Ranked first because it is the
only item whose win is measured in both decode time *and* the context that fits.

---

## E–G, briefly

**E. GDN interior checkpoints.** Their mechanism is exact states at periodic boundaries inside a *prefill*, so a
later request that diverges mid-prefill can resume there. Ours is `save_state`/`load_state` at *request*
granularity (`engine.hpp:340-342`, `park_bound`/`bind` at `engine.cpp:1191`/`:1246`). The granularity question for our
prompt path is whether a 61K prefill should leave a checkpoint every N tokens: at 4,096 it is
`61440/4096 = 15` boundaries, and a state is 149.6 MiB, so **15 × 149.6 = 2.2 GiB of VRAM or host RAM** — 78,000
tokens of k8v4 context, on a card that cannot spare them. **Out.** Revisit only when host-memory parking is the
default, where the same 2.2 GiB is cheap.

**F. Prefix caching as a test — the design, ready to hand over.** Our `--conversation-cache-mib` parks a whole
conversation between requests; theirs caches a prefix at block granularity inside one. The question is not which is
better, it is whether ours already gives their 23 s → 1 s.

- **Material.** One 24,000-token document of ordinary prose (a Wikipedia article is fine), tokenised with this
  model's tokenizer. `--tokens` takes a comma list, so the second turn's prompt is *document ids + the first answer's
  ids* — no template, no chat formatting, which keeps the two runs comparable.
- **Card settings.** `--max-context 32768 --kv k8v4 --no-draft --prompt-batch 512`, the same in both arms. `--no-draft`
  because the metric is prompt processing, not decode.
- **The two arms.** `--conversation-cache-mib 0` (parking off) and `--conversation-cache-mib auto` (parking on), both
  with `--conversation-cache-slots 4`. Two requests in one process, the second being the 24K document plus ~200
  generated tokens, `--max-new 1`.
- **The metric: the second request's prefill time**, which `strata-poly` already prints as
  `prompt N tokens X tok/s`. Take it twice per arm and use the second run of each (the first pays page faults).
- **Pass looks like** the parked arm's second-request prefill at **≥5× the unparked arm's** — theirs is ~23×
  (23 s → 1 s), and anything near 1× means our parking is not resuming the KV, which is the thing worth knowing.
  The number to record either way is **prompt tok/s on the second request**, both arms, plus `conversation_cache_*`
  from the `INFO` line to prove the slot was actually used.
- **If it passes, close the item.** Their block-granular prefix cache is a refinement of a thing we already do; the
  only gap worth a follow-up would be parking at *token* rather than *conversation* granularity, and that is a
  different design.

**In, 0.5 days of wall clock, two runs, no code.**

**G. `cudagraph-memory-from-allocator`.** Their bug: the CUDA-graph memory estimate subtracted the *driver's*
free-memory delta, which reads zero or collapses under WSL2, and refused the boot. Our accounting is a different
class and already correct: `reserve_mib_` (`engine.cpp:1856`) is sized 256–768 MiB *for* the launches recorded later,
with the comment at `:1841` giving the measured 180 MiB on an 8 GB card, and `mind_vram()` keeps half of it
(`:2169`). We budget a reserve; they tried to measure a delta. **Out — nothing to fix**, and their patch is a good
statement of why measuring a driver delta is the wrong instrument.

---

## What I would do, in order

1. **D, the draft vocabulary (1–2 days).** The flag exists (`--draft-vocab N`) and takes a **prefix**, which is the
   wrong subset; the fix is a gathered matrix and a list. It is worth **5.88% → 4.36%** of a decode token and
   **up to 500 MiB of VRAM** — and on this card the VRAM is the scarcer of the two.
2. **F, the prefix-caching test (0.5 days, two runs, no code).** Design in F above; the metric is the second
   request's prompt tok/s across the two arms.
3. **C's `kSplit` 1024 → 256 (hours).** One `constexpr` at `fattn.cu:261`, four recomputed sites, no register or
   shared-memory change, scratch 7.6 → 30.2 MiB at 61K. Gate on `poly_fattn_test k8v4 --heads 24`.

**Out, with the reason:** **B** because the kernel is DRAM-bound at 11% of the roofline and the padding is 6% on the
launch that matters — swapping M and N cannot buy what the memory system is not offering; **A** because 8–10 days
for ~1.5% of prefill at 32K (and ~2.5–3% at 61K) is a bad trade against a draft head that is 1–2 days for 1.5 points
of *decode* plus 500 MiB; **E** because 15 checkpoints cost 2.2 GiB of a card that has none to spare; **G** because
we already budget for the graphs instead of measuring a driver delta.

## Corrections to what was quoted

- **`Q·Kᵀ` as a share of prefill: 5.83%, not 0.41%.** My 0.41% was one attention layer over the whole prefill; there
  are 16 layers. Attention is 17.7% of a dense prefill, so **A is out on difficulty (8–10 days for a group-6
  kernel), not on arithmetic** — the fix is to say that, and to stop calling it negligible.
- **The 62% padding figure.** Right for `--no-draft` (6 rows of `MT=1`'s 16). At a window of 8, which is the
  default, the first launch uses 30 of 32 rows — **6.25%** — and the two launches average **25%**. And it does not
  matter either way: the kernel is at **11% of the roofline**, so the MMA has idle capacity in every configuration.
  The recommendation that rested on it does not stand.
- **The split-count fix.** It does not help at 61K — 240 CTAs on 68 resident slots is 3.5 waves, already saturated.
  It is worth something only below 17,408 tokens, and there the whole attention is **0.05 ms of a 43 ms token**. The
  version that does help contexts we serve is `kSplit` 1024 → 256, and even that is sub-1%.

## What needs the card

Nothing above. Three things would firm it up if the card is free later, none on the critical path:

- the `kSplit` 256 rebuild measured at 2K / 8K / 16K, which is the only way to confirm the split claim;
- one `fdec` decode measurement against its 11%-of-roofline estimate, to confirm it is memory-bound in practice and
  not only on paper — that is the number B rests on;
- after the gathered head lands, one decode rate at 61K to check the 5.88% → 4.36% prediction, which is arithmetic
  from byte counts and assumes the step stays bandwidth-bound.
