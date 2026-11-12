# HyperQwen: the mechanisms, deep, against our code

A second pass. The first pass answered "what can be ported" against vLLM's shape and was the wrong question: several
of these survive as *mechanisms* even though their implementation does not. This one goes to a number or a
`file:line` for each, with no GPU runs.

---

## The table

| # | mechanism | fits our code? | the number | days | verdict |
| --- | --- | --- | --- | ---: | --- |
| **B** | **swap-AB decode** (`S^T = K·Q^T`, `O^T = V^T·P''`), tokens on MMA's M, query rows on N | the *idea* yes, the code no — `fdec_kernel` would be re-indexed, not rewritten | **6.25% padding at `--window 8`, 25% at `--window 4`, 62.5% at `--no-draft`.** But the kernel is **memory-bound at 11% of the roofline**, so the padding costs nothing measurable | 0 to read, 4–6 to build | **out.** The premise was wrong: the MMA is idle either way |
| **C** | split count from SM count, `S = ceil(num_sms·CTAS_PER_SM/Hkv)` | **partly** — `kSplit` is a `constexpr` baked into three places, so it is a constant change, not a planner | `CTAS_PER_SM = 2` (register-bound, `REG:255`, 11,264 B smem). Split count reaches 68 CTAs at **17,408 tokens**; below that the card is underfilled | 1–2 | **only at ≤16K, and there decode is already fast.** The one real gain is `kSplit` 1024 → 256 |
| **A** | int8-QK prefill (3-way int8 split, `m16n8k32`) | **no**, for a reason that is arithmetic, not porting | `Q·Kᵀ` is **0.41%** of a 32K prefill's FLOPs, **0.76%** at 61K. A 2× faster QK is **0.21% / 0.38%** end-to-end. Measured on sm_86: +1.1/+1.3%; **sm_89 unmeasured** | 8–10 | **out.** Two orders of magnitude below its own 1.6% precedent |
| **D** | calibrated draft vocabulary | **yes**, and it is already half-built | `guess_vocab()` returns **131072 as a constant** (`qwen35.hpp:34`). The head matvec is **360 MiB / 1,311 µs** at 131072 rows against **682 MiB / 2,483 µs** at the full vocabulary. `likeliest` is one 1024-thread block | 1–2 | **in, as a measurement.** The win is bytes off the head matvec, not hit rate |
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

### A.4 The measured ceiling — the arithmetic that closes it

A dense 27B prefill at 32K, in FLOPs (causal, so `T²/2` pairs):

| term | TFLOP | share |
| --- | ---: | ---: |
| FFN (64 layers × 3 matrices × 5120×17408) | 1,121.5 | **69.7%** |
| GDN (48 layers) | 364.4 | 22.7% |
| attention projections (16 layers) | 110.0 | 6.8% |
| `Q·Kᵀ` (16 layers, 24 heads, d=256) | **6.6** | **0.41%** |
| `P·V` | 6.6 | 0.41% |

At **61,440** tokens the attention term grows quadratically: `Q·Kᵀ` is **23.2 TFLOP of 3,039 = 0.76%**.

**So a perfect, free `Q·Kᵀ` — an infinite speedup — is worth 0.41% of prefill at 32K and 0.76% at 61K.** Halving
it, the best int8 can do, is **0.21% and 0.38%**. Our measured 1.6–1.8% on Flash was paid for by the *whole* v3 path
(which also replaced the projections' arithmetic); the dense model has no indexer and a 17,408-wide FFN that is 70%
of the work.

**Verdict: out, and not because of porting difficulty.** The dense model's prefill is an FFN GEMM problem; no
attention kernel can move it. And the one measured datapoint on a card near ours is +1.1% / +1.3% (sm_86) — so even
a working port would land an order of magnitude below what the QSA path already earns, on a model where the FFN is
70% of prefill.

---

## D. Draft vocabulary — the mechanism is bytes, not hit rate

### D.1 What `guess_vocab()` is

`qwen35.hpp:32-34`, with the comment above it:

```cpp
// English answers write no token past 77,000, Chinese prose one in twenty past 131,072, and guesses from the
// first 131,072 still write a Chinese answer faster than from all 248,320
int guess_vocab() const override { return 131072; }
```

**A hardcoded constant with an undocumented provenance** — not a frequency list, not measured on this model, not
per-language. It is a guess about a guess.

### D.2 What a calibrated list needs

It must come from **a corpus of the model's own output**: a token outside the list can never be proposed, so the list
is a statement about the output distribution. It cannot be derived from the MTP block's weights — those say which
tokens the block can *score*, not which the target will *write*, and the two differ exactly where a calibration
would help.

The cheap version: run the model over a few thousand tokens of its own generations, histogram the ids, take the
smallest prefix-plus-set that covers 97.5%. The 131072 constant is a *prefix*, which is the wrong shape as well as
the wrong size — the common tokens are not contiguous in this vocabulary.

### D.3 The mechanism, and whether our bottleneck is theirs

Their framing is "coverage 92% → 97.5%, +10% speed". Ours is different in a way that matters:

- **`draft_rows()` (`engine.hpp:233`) is the head matvec's row count**, not a softmax width.
  `head.n_out = draft_rows(); lin(head, …)` (`engine.cpp:5280`, `:5806`).
- The head is `output.weight`, Q4_K, 5120 × n_out. At 131,072 rows that is **360 MiB read a guess**, at 288 GB/s
  **1,311 µs**; at the full 248,320 rows **682 MiB / 2,483 µs**. **The vocabulary size is the whole cost of a draft**,
  and it is a bandwidth cost, not a softmax cost.
- `likeliest` (`kernels.cu:2305`) is **one block of 1,024 threads** over `n_out` floats — 512 KiB at 131,072 —
  and it does compute a softmax, but a truncated one: it sums `expf(x - m)` only over `x > m - 20` and returns
  `prob = 1/sum` beside the argmax. **Its cost is a single block's read of `n_out` floats, not a full-vocabulary
  reduction**, so it does not scale the way a real softmax would; the matvec is what costs.

**So the mechanism worth taking is "a smaller, calibrated `n_out`", and its win is linear in bytes.** Their +10%
came from their coverage gap; ours would come from shrinking 360 MiB toward the ~92% the constant already covers.

**One caveat on their number transferring.** Their +10% is on a *speculating* server where every draft costs a chain
step; our bottleneck is the same shape (bytes a draft) but the constant is already 131,072 — 53% of the vocabulary —
so the headroom is from 360 MiB down to the ~250 MiB a 92%-coverage set needs, i.e. **a 30% cut in the matvec, not a
10× win**, and only on the drafts that miss. Expected: single-digit percent of decode while drafting.
**This is measurable without the card's scarce time: it is a Python histogram over stored output, and the matvec
cost follows from the byte count.** 1–2 days, and it is the only item here with a clear positive expected value.

---

## E–G, briefly

**E. GDN interior checkpoints.** Their mechanism is exact states at periodic boundaries inside a *prefill*, so a
later request that diverges mid-prefill can resume there. Ours is `save_state`/`load_state` at *request*
granularity (`engine.hpp:340-342`, `park_bound`/`bind` at `engine.cpp:1191`/`:1246`). The granularity question for our
prompt path is whether a 61K prefill should leave a checkpoint every N tokens: at 4,096 it is
`61440/4096 = 15` boundaries, and a state is 149.6 MiB, so **15 × 149.6 = 2.2 GiB of VRAM or host RAM** — 78,000
tokens of k8v4 context, on a card that cannot spare them. **Out.** Revisit only when host-memory parking is the
default, where the same 2.2 GiB is cheap.

**F. Prefix caching as a test.** Our `--conversation-cache-mib` parks a whole conversation, theirs caches a prefix at
block granularity; the question is whether a 24K follow-up turn already costs ~1 s. **The cheapest experiment is
two turns of a chat over the same 24K document, `--max-new 1`, once with `--conversation-cache-mib 0` and once with
`auto`, reading the second turn's prompt-tokens-per-second line.** No code, two runs, and it either closes the item
or shows a gap worth a checkpoint granularity. **In, 0.5 days, after the card is free.**

**G. `cudagraph-memory-from-allocator`.** Their bug: the CUDA-graph memory estimate subtracted the *driver's*
free-memory delta, which reads zero or collapses under WSL2, and refused the boot. Our accounting is a different
class and already correct: `reserve_mib_` (`engine.cpp:1856`) is sized 256–768 MiB *for* the launches recorded later,
with the comment at `:1841` giving the measured 180 MiB on an 8 GB card, and `mind_vram()` keeps half of it
(`:2169`). We budget a reserve; they tried to measure a delta. **Out — nothing to fix**, and their patch is a good
statement of why measuring a driver delta is the wrong instrument.

---

## What I would do, in order

1. **D, the draft vocabulary (1–2 days).** The only item with a positive expected value and it needs no card: a
   calibrated `n_out` shrinks the 360 MiB a draft reads, and `guess_vocab()` is currently a constant nobody
   measured. Start by histogramming stored output over any corpus we already have.
2. **F, the prefix-caching test (0.5 days).** Two runs, closes a question, and it is the one place where their
   23 s → 1 s could mean we are leaving something on the table.
3. **C's `kSplit` 1024 → 256, as a one-line experiment (hours).** One constant, `poly_fattn_test` as the gate, and it
   is the only part of the split-count idea that touches contexts we serve. Expect under 1%.

**Out, with the reason:** **B** because the kernel is DRAM-bound at 11% of the roofline and the padding is 6% on the
launch that matters — swapping M and N cannot buy what the memory system is not offering; **A** because `Q·Kᵀ` is
0.41% of a 32K prefill and a free one is worth 0.41%; **E** because 15 checkpoints cost 2.2 GiB of a card that has
none to spare; **G** because we already budget for the graphs instead of measuring a driver delta.

## Two corrections to what was quoted

- **The 62% padding figure.** Right for `--no-draft` (6 rows of `MT=1`'s 16). At a window of 8, which is the
  default, the first launch uses 30 of 32 rows — **6.25%** — and the two launches average **25%**. And it does not
  matter either way: the kernel is at **11% of the roofline**, so the MMA has idle capacity in every configuration.
  The recommendation that rested on it does not stand.
- **The split-count fix.** It does not help at 61K — 240 CTAs on 68 resident slots is 3.5 waves, already saturated.
  It is worth something only below 17,408 tokens, and there the whole attention is **0.05 ms of a 43 ms token**. The
  version that does help contexts we serve is `kSplit` 1024 → 256, and even that is sub-1%.

## What needs the card

Nothing above. Two things would firm it up if the card is free later, and neither is on the critical path: the
`kSplit` 256 rebuild measured at 2K/8K/16K, and a decode measurement of `fdec` against its 11%-of-roofline estimate
to confirm the kernel is memory-bound in practice and not on paper.
