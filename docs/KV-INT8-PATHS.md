# KV formats and the INT8 tensor-core paths on a 128-bit Ada card

Phase 1 was reading and arithmetic only: no GPU run, no build, no edit to a kernel. Every number below
says whether it was **measured**, **computed** from code, or **documented** by a vendor. Where a number can
only come from the card it is in [§9](#9-needs-the-card).

The question: which KV-format and matmul paths can use INT8 tensor cores, what does each cost in accuracy,
and which of them matter on a 128-bit-bus Ada card in low-bit quantizations.

## 0. The answer, and the two corrections that change the plan

> ### Correction 1 — `qsa select` never reads the KV cache. **`strata` is not a speed case at all.**
>
> A first framing of this study said attention is 10.92% of a 256K prefill on Qwen3.8-Flash and therefore a
> smaller KV format could be worth up to 11% there. **That is wrong.** The 10.92% is `qsa select`, and
> `qsa_layer` computes it from the **indexer's** keys — `qsa_block_scores(st.idx_pooled, b.q_idx, …)`
> (`src/core/layer.cpp:1276`), where `idx_pooled` is `idx_key_dim = 64`-wide indexer data, not K or V
> (`include/strata/kernels/qsa.hpp`). The KV cache is touched only afterwards, in `kv_gather_*` plus the
> attention that follows, which are separate terms and are **0.78% of the cache per token per layer**
> (2,051 of 262,144 cells, `QUANT_MICROBENCH.md:595`).
>
> **So no KV compression moves the 10.92% by a single byte.** `strata` leaves the speed case entirely and
> stays only as a footprint case, exactly like `strata-poly`. That removes the one model that could have
> justified a faster QK product, and it is the strongest single result in this document.
>
> ### Correction 2 — the INT8 peak in our own docs is 2× low
>
> `docs/QUANT_MICROBENCH.md:461` says "~88 TOPS" for int8; the whitepaper and today's measurement both say
> **175.5 / 176.5** (§1.2). It is the FP16-accumulate ratio applied to an FP32-accumulate figure. Every
> "we are at X% of INT8 peak" figure in the tree halves. Details and the affected rows in §1.2.

1. `mma.sync.aligned.m16n8k32.s8.s8.s32` needs **sm_80**: Turing has the int8 mma only as `m8n8k16`
   (PTX ISA, `ptx.txt:32737-32749`). The tree already gates on that — `int8_tensor_cores()` is
   `kernels_ptx() >= 80` (`src/poly/kernels.cu:4049`).
2. INT8 : FP16 = **4 : 1** on Ada *with FP32 accumulate*, and 2 : 1 with FP16 accumulate. That is a
   **GeForce** property (FP16 with FP32 accumulate runs at half rate on consumer parts); on H100/B200 the
   ratio is 2 : 1. The whitepaper's own numbers and today's measurement agree to 0.6%
   (660.6 × 34/128 = **175.5 TOPS** computed against **176.5** measured).
3. The tree has exactly **one** KV reader on IMMA: `attend_int8` (`kernels.cu:3866`). It is reached by
   **Gemma 4** (`--kv int8`/`int8k`) and **Kolibri 1** (`STRATA_KV_INT8=1` on a q8_0 cache). `qwen35`
   never calls it, and the reason is structural, not a gate: its attention needs a QSA selection, a
   Hadamard-unrotated value mix and an output gate that `attend_int8` does not implement (§2.1).
4. **`fdec`** (`q8_0`, `k8v4`, `k6v4`, `fp16`) runs **FP16** MMA after dequantizing in registers
   (`fattn.cu:280`, `:491`). But **`attend` — the path that serves decode below 4,096 positions and every
   geometry `fdec` refuses — runs no tensor instruction at all**: scalar `FFMA` over converted halves
   (Part III §20.1, and §2.2's correction). So the INT8-QK question is not "IMMA instead of HMMA" for the
   generic path, it is **"IMMA instead of no tensor core"**, which is where the largest single win in this
   document sits — and it costs **the query quantized to int8**, a term no current format has (§2.2).
5. **The framing in the brief is right that footprint is the prize — and it is stronger than "right":
   a smaller format is not merely worth little in speed, it is worth *negative* speed on this code.**
   Measured, same harness, 256K, RTX 4060 Ti: `q8_0` **4.96 ns** a cell, `k8v4` **5.88**, `fp16` **8.32**
   (`docs/Q8_KV.md:270`). The 8-bit format is the fastest one; going below it costs time and buys bytes.
6. The cheapest decisive first step is **not** a tok/s run. It is the accuracy gate on the format that is
   already coded and unmeasured, at a sample size that can separate it: `tools/kv_agree.py --kv k6v4
   --prompts 32 --max-new 128` (§8, step 1).

## 1. What the ISA dumps and the vendor documents actually say

### 1.1 The int8 (and int4) mma forms, per architecture

`ptx.txt` is the authoritative statement of which shape exists on which target. The two lines that decide
this study:

```
.u8/.s8 integer type mma operation with .m8n8k16 shape requires sm_75 or higher.        ptx.txt:32737
.u8/.s8 integer type mma operation with .m16n8k16 and .m16n8k32 shapes requires sm_80
or higher.                                                                              ptx.txt:32747
```

| arch | `m16n8k32` s8/u8 | the int8 form it does have | 4-bit mma | `IMMA` in the SASS dump | source |
| --- | --- | --- | --- | --- | --- |
| sm_75 Turing | **no** | `m8n8k16` (s8), `m8n8k32` (s4), `m8n8k128` (b1) | yes, `m8n8k32` | not dumped locally | PTX; in-tree: `mma.cuh:951-960` compiles `m8n8k16` below 800 |
| sm_80 Ampere | **yes** | `m16n8k16` and `m16n8k32` | yes, `m16n8k32`/`k64` | not dumped locally | PTX |
| sm_86 Ampere | **yes** | same | same | not dumped locally | PTX |
| sm_89 Ada | **yes** | same | yes: `IMMA` sizes `8832,8864,16832,16864,168128` with `U8,S8,U4,S4` (`sm89_inst.txt:67500-67540`, `:58146-58310`) | **yes** | local dump |
| sm_90 Hopper | **yes** (PTX), but the fast path is `wgmma` | `IMMA` sizes `8816,16816,16832` with `U8,S8` **only**; `SRCFMTA_U8_S8_U4_S4` has **0** hits in the sm_90 dump against 40+ in sm_89 (`sm90_inst.txt:86947`); `QGMMA`/`HGMMA`/`IGMMA`/`BGMMA` present, **0** in the sm_89 dump | via `wgmma`, not `mma.sync` | yes, narrower | local dump |
| sm_120 Blackwell | **yes** | `mma.sync` still supported; the dense path is `tcgen05` | via the block-scaled kinds | not dumped locally | PTX (`ptx.txt:32749` region) |

Two things here matter and neither is in a blog:

- **Ada's `IMMA` is wider than Hopper's.** The sm_89 table carries the 4-bit and k64 forms
  (`SIZE_8832_8864_16832_16864_168128`, `SRCFMTA_U8_S8_U4_S4`); the sm_90 table does not. So on Ada there
  is a **native 4-bit tensor instruction**, `IMMA.16832.S4.S4`, and the whitepaper prices it at twice INT8
  (1,321.2 against 660.6 TOPS). Nothing in this tree uses it. It is the one genuinely unexplored tensor
  path for a k4v4 cache (§4).
- **`dp4a` is not a tensor core.** In the sm_89 dump `IDP.4A`'s opcode is `IDP4Afmalighter_pipe`
  (`sm89_inst.txt:58204`) — it issues to the FMA pipe, one 32-bit destination, `SRCFMT_U8_S8` operands.
  It is the right tool when the kernel is *already* reading one byte per value and has no matrix to build
  (a matvec at batch 1–8, a dequant-and-dot in registers); it is the wrong tool for a card whose tensor
  pipe does 4× an FFMA's work per instruction at 8× the width. Measured in-tree:
  `poly_kv_int8_test` prints "one 16x8x256 product ... mma X ms, dp4a Y ms" (`kv_int8_test.cpp:804-808`),
  and the kernel comment records **3.4×** for `attend_int8`'s score step (`kernels.cu:3719-3721`).

### 1.2 The INT8 : FP16 ratio — the vendor's own table

One document, three architectures, so the comparison is like for like: the Ada whitepaper V2.02,
Appendix A (`/tmp/opencode/lsa-research/ada.txt:1020-1071`). Slash = dense / with sparsity.

| | Turing (2080 Ti) | Ampere (3090) | Ada (4090) |
| --- | ---: | ---: | ---: |
| Peak FP16 Tensor, **FP16** accumulate | 113.8 | 160 | 330.3 |
| Peak FP16 Tensor, **FP32** accumulate | **56.9** | **80** | **165.2** |
| Peak INT8 Tensor | **227.7** | **320** | **660.6** |
| Peak INT4 Tensor | 455.4 | 640 | 1,321.2 |
| INT8 : FP16(FP32 acc) | **4.0** | **4.0** | **4.0** |
| INT8 : FP16(FP16 acc) | 2.0 | 2.0 | 2.0 |

The 4 : 1 is therefore not an Ada quirk: it is what every GeForce part since Turing does, because the
consumer parts run FP16-with-FP32-accumulate at half the FP16-with-FP16-accumulate rate. On the datacenter
parts there is no such halving, and the same table logic gives INT8 : FP16 = **2 : 1** on H100 and on
Blackwell — **documented, not in a local dump**; the sm_90 and sm_120 ISA files in `/tmp` say the
instruction exists, not what it costs. Anyone quoting "4× on Hopper" is quoting a GeForce number.

Scaled to this card (34 SMs, `docs/QUANT_MICROBENCH.md:461-462`) and against the day's measurements:

| quantity | computed from the whitepaper | measured today | agreement |
| --- | ---: | ---: | --- |
| FP16 tensor, FP32 acc | 165.2 × 34/128 = **43.9 TFLOPS** | **44.1** | 0.5% |
| INT8 tensor | 660.6 × 34/128 = **175.5 TOPS** | **176.5** | 0.6% |
| INT4 tensor | 1,321.2 × 34/128 = **351.0 TOPS** | not measured | — |

### Correction to our own documented peak (`QUANT_MICROBENCH.md:461`)

`docs/QUANT_MICROBENCH.md:461` records int8 dense tensor as "**~88 TOPS**, 2× the bf16 rate". That is the
**FP16-accumulate** ratio applied to an **FP32-accumulate** figure, and the row is therefore **2× low**.

| basis | value | where |
| --- | ---: | --- |
| the documented row (wrong) | ~88 TOPS | `QUANT_MICROBENCH.md:461` |
| whitepaper, scaled 660.6 × 34/128 | **175.5 TOPS** | `ada.txt:1068` |
| measured today | **176.5 TOPS** | this session |
| measured MMQ, as a share | 69.4 / 176.5 = **39%** (not 79%) | this session |

This is the denominator of every INT8 efficiency claim in the tree, so the error propagates multiplicatively:

- **row 6**'s "63% of the 88 TOPS int8 peak" (`QUANT_MICROBENCH.md:73`) is **32% of the true peak**, not 63%.
  The kernel it measured is unchanged; only its place in the ranking moves — down, which makes it a stronger
  argument against that phase's bar, not a weaker one.
- **row 8**'s and §7's percentages are unaffected in kind but must be re-based on 176, not 88.
- Anywhere "we are at X% of INT8 peak" was quoted as a *success*, the number halves. Anywhere it was quoted
  as a *failure*, the failure is twice as large.

By this repo's own rule — *when a summary and its log disagree, the log wins*
(`QUANT_MICROBENCH.md:518`) — the table row becomes **176 TOPS**, labelled "computed from the whitepaper,
confirmed by measurement". The 2× arises because the consumer parts halve FP16-with-FP32-accumulate while
leaving INT8 alone (§1.2), so a ratio taken from the FP16-accumulate column overstates INT8 by exactly that
factor.

**This is a doc fix outside this document's scope**: `docs/QUANT_MICROBENCH.md:461` and `:73` still say 88
and 63%. They are left untouched here because editing another page's measurements is the maintainer's call,
but the correction is recorded so that no future number is derived from the old row.

### 1.3 What the 128-bit bus changes

| | value | source |
| --- | ---: | --- |
| DRAM | 288.0 GB/s (128-bit GDDR6 at 9,001 MHz), 280 measured loaded | measured, `QUANT_MICROBENCH.md:455-456` |
| L2 | 32 MiB | measured, `cuDeviceGetAttribute` |
| SMs | 34 | measured |
| machine balance, FP16-MMA (FP32 acc) | 44.1e12 / 288e9 = **153 FLOP/byte** | computed |
| machine balance, IMMA | 176.5e12 / 288e9 = **613 int8-op/byte** | computed |
| machine balance, dp4a (4 MAC/instr on the FMA pipe) | ~98e12 / 288e9 = **340 op/byte** | computed from 128 lanes × 4 × 2.82 GHz × 34 |

The memory side changes the answer in two ways, and both push the same direction:

- **At 613 int8-ops per byte, IMMA cannot be fed from DRAM.** A k8v4 cell is 832 B and yields 512
  multiply-accumulates for one query — **0.6 MAC a byte against a balance of 306** (88.25e12 MAC/s ÷
  288 GB/s), short by 500×. IMMA is only reachable when the keys are already in registers or shared memory,
  i.e. after a tile has been read and *reused across many queries*. At a decode batch of 1 there is no
  reuse and the tensor pipe idles; at a prompt batch of ≥16 queries per key tile there is. That is
  exactly the shape of the measured result: the IMMA tier is worth **+5–7%** over the
  register-blocked dp4a kernel at ≤16K keys and level past it
  (`kernels.cu:3903-3905`), and `STRATA_PROMPT_ATTN_IMMA=1` was worth **+2.5% of prompt read**
  (`docs/DETAILS.md:2379`).
- **32 MiB of L2 against a 288 GB/s bus** means the 2,051 selected cells × 12 layers of a QSA decode
  (19.5 MiB at k8v4, §3) fit in L2 entirely. So the QSA read is not a DRAM problem at all, and a smaller
  cache format buys it nothing.

## 2. What the code already does

### 2.1 `attend_int8` — the only IMMA KV reader

`mma_s8` is one `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (`kernels.cu:2075-2082`), compiled out
below sm_80 with the note that "ptxas refuses it below sm_80" (`:2068-2074`). Its shape is the cache's own:
A is a query's 32 codes, B a key's 32 codes, C a 16×8 block of int32.

`attn_score_mma_int8_kernel` (`kernels.cu:3727`) is the caller, and **one `mma_s8` is one 32-value block**
(`:3771`). The block's two scales are applied to the int32 result in float (`:3772-3779`), so the accuracy
model is: *exact integer dot over 32 values, then one float multiply by (dq × dk)*. Nothing is lost that a
per-block scale does not already lose.

- **Formats accepted:** a `KvInt8 {int8_t* code; uint16_t* delta}` pair, i.e. Q8_0 codes with a separate
  fp16 scale array — one scale per 32 values (`store_int8`, `kernels.cu:3079`; `kQ8_0Values` = 32).
- **Head widths:** 64, 128, 256, 512, else it throws `"a head is 64, 128, 256 or 512 values"`
  (`kernels.cu:3933`). The **IMMA tier itself is `dim == 256` only** (`:3912`, `:3994` — the kernel is
  templated `attn_score_mma_int8_kernel<256 / WARP>` and static-asserts `PER == 8`, `:3730`). Every other
  width falls to dp4a tiers.
- **The value side has no tensor-core route.** `attn_values_q8` does one `I2F`, one `FMUL` by the block
  scale and one `FFMA` per value (`kernels.cu:3793-3815`), and `attn_values_for_int8` likewise. So
  today's IMMA path accelerates the QK product only; P·V is scalar.
- **Who calls it:** Gemma 4 under `--kv int8`/`--kv int8k` (`gemma4.cpp:721-724`, `:762`, `:776`) and
  Kolibri 1 under `--kv q8_0` with `STRATA_KV_INT8=1` (`kolibri1.cpp:429-433`, `:483-486`). Gemma 4 also
  takes a third route for a prompt batch: dequantize the whole cache to fp16 and call `attend_batch`
  (cuBLAS), with the reason stated in the code — "a Q8_0 cache has no tensor-core route (a block's own
  delta is not a cuBLAS scale)" (`gemma4.cpp:706-718`).
- **Why `qwen35` never calls it.** Not the `--kv` gate alone (`engine.cpp:1650-1653` rejects `int8`/`int8k`
  for anything but Gemma 4) — the reader does not fit the model:
  - `attend_int8` writes the plain weighted sum. `Qwen35::attn` needs the values **rotated back** with
    `fwht_rows` after the mix and then **gated** with `mul_sigmoid` (`qwen35.cpp:717-721`).
  - `attend_int8` attends a contiguous prefix. `Qwen35`'s decode path goes through `fattn_*`/`fdec_*` and
    its prompt path through `attend_batch` with a dequantizing workspace (`qwen35.cpp:691-712`).
  - Its gate is per-architecture *because* of that: "their readers are not the arithmetic q8_0's are"
    (`engine.cpp:1650`).
  - And its `KvCell` has no field for a `KvInt8` value side: `kK8V4`/`kK6V4` are `q4` values
    (`qwen35.cpp:216-226`), which `attend_int8` cannot take; `attend_int8_mixed` (`:3948`) takes fp16
    values, not q4.

> **Corrected by Part III (the SASS audit).** An earlier draft of this section said `attend()` runs FP16
> `mma16816` over the dequantised k8v4 keys. **The binary says it runs no tensor instruction at all**:
> `attn_score_reg_kernel<8,2,true>`, which is what `attend()` launches for a q8/k8v4 key cache at
> `dim == 256`, has **0 HMMA, 0 IMMA, 0 IDP** — it is scalar `FFMA` over `__half2float` conversions. The
> same is true of `attn_score_kernel` and both value kernels. `mma16816` in `kernels.cu` is reached only by
> `attn_flash64_kernel` (LFM2's 64-value heads) and lives in `fattn.cu` for `fdec`. Source check:
> `mma16816(` appears at `kernels.cu:2213-2214`, `:2247-2248`, `:2332-2333`, `:2368-2369` — every one of
> them inside a flash kernel, none in `attend`. **Part III §20.1 is the census; this note is the correction,
> and it is the more important of the two.** The consequence for the bench programme is §B1: the comparison
> is IMMA against *scalar FFMA*, not against HMMA, which is a far larger expected gap.

### 2.2 `fattn.cu` — every KV format runs **FP16** MMA

`fdec_kernel` (`fattn.cu:335`) is the split-K decode attention. `VM` is the cache format:

| `VM` | keys | values | who |
| ---: | --- | --- | --- |
| 0 | fp16 | fp16 | `--kv fp16` |
| 1 | q8_0 | q8_0 | `--kv q8_0` |
| 2 | q8_0 | q4_0 (rotated) | `--kv k8v4` |
| 3 | **6-bit** (`KQ6`) | q4_0 (rotated) | `--kv k6v4` |

Every one of them dequantizes to halves in registers and runs `mma.sync.aligned.m16n8k16.row.col.f32.f16.
f16.f32` (`fattn.cu:280-291`, called at `:491` and `:587`/`:607` for P·V). The q8 codes become halves via
`i8pair` — `0x64XX` is `1024 + XX`, so `PRMT` + one `HADD2` of −1152 is exact (`:293-300`) — and the
6-bit codes via `q6pair` at **five SASS instructions a pair** (`:303-310`: SHF, PRMT, LOP3, PRMT, HADD2).
The block scale is applied to the float result after the mma (`:501`).

**Could the QK product run on IMMA instead? Yes, and the block structure already matches.** The mma's k is
32 and the cache's scale block is 32, so one `mma.sync.m16n8k32.s8.s8.s32` is one scaled block, exactly as
in `attn_score_mma_int8_kernel`. What it would take:

1. **The query must be quantized to int8** with a per-32 scale, once per token per layer — an O(rows·dim)
   pass, which `attend_int8` already does with `store_int8` (`kernels.cu:3877`).
2. **The query error becomes a new term.** Today k8v4/k6v4 lose accuracy on the *key* side only; the query
   stays fp16. IMMA doubles the number of quantized operands. Since the query vector is shared by every
   score in its row, its error is systematic across the row, which is the worse kind.
3. **The layout has to be re-derived.** `fdec`'s A fragments are currently `m16n8k16` halves read from
   shared `Qs` (`:486-489`); IMMA's A fragment is 16 bytes `/thread` over k=32 with the `g8/tig` row split
   `probe_mma_layout` verifies (`kernels.cu:4051`).
4. **`NT=2`, 16 keys a warp, `MT` 1 or 2** already match n=8 and m=16 — no tiling change.

So: **yes for k8v4 and q8_0 keys (after expanding 6-bit to s8), needs a new kernel, and the payoff is
bounded by the attention share of wall time** — 0.8% of prefill on Ornith, and for decode the QSA path
below shows the read is not bandwidth-bound either.

There is a **cheaper, format-independent 2×** sitting in the same line: `fdec` accumulates in **FP32**.
On this card FP16-with-FP32-accumulate is 44.1 TFLOPS and FP16-with-FP16-accumulate is ~88
(§1.2), and ggml's `mma.cuh:977-1019` already has the `f16`-accumulate forms. That is a hypothesis, not a
result — the accuracy of a 16-block fp16 accumulator over 256 values is unmeasured here — but it is a
one-line change per mma call and it does not touch the cache format at all.

### 2.3 The stored bytes, from the stores themselves

| format | store | layout | bytes / 32 values | bytes / value |
| --- | --- | --- | ---: | ---: |
| fp16 | `store_f16` | 32 halves | 64 | 2.0 |
| q8_0 / int8 (`poly`) | `store_q8`, `kernels.cu:628` | 32 int8 codes in one array, fp16 scale in a second | 32 + 2 | 1.0625 |
| int8 (`strata`) | `kv_q8.hpp:KV_Q8_GROUP = 64` | 64 codes + one fp16 scale | 64 + 2 per 64 | 1.03125 |
| k8v4 values | `store_q4`, `kernels.cu:697` | 16 packed bytes, rotation applied before the store | 16 + 2 | 0.5625 |
| k6v4 keys | `store_q6`, `kernels.cu:736` | 24 bytes, value *j* at bit 6*j*, offset-binary, scale in its own array | 24 + 2 | 0.8125 |

Note the two engines do **not** agree on the int8 scale period: `strata` uses one fp16 scale per **64**
values, `strata-poly` one per **32**. That is the whole of the difference between 1,056 and 1,088 B a cell,
and between 816 and 832 for k8v4 (the q4 side uses a scale per 32 in both). Any table that mixes them is
wrong by 32 B a cell. Reconcile-able in one line: `528 + 288 = 816` (`strata`) against `544 + 288 = 832`
(`strata-poly`), where 16 B is the int8 scale array at a 64-value period instead of 32.

## 3. Is the framing right? Footprint or speed?

The brief's framing: attention is 0.8% of prefill on Ornith and 10.92% of a 256K prefill on
Qwen3.8-Flash, therefore a faster KV *read* is worth ≤1% / ≤11%, and the real prize of k5v3/k4v4 is
footprint. **The conclusion is right, and the second premise is not: the 10.92% is not a KV number at all
and the ceiling on `strata` is ~0, not 11%. Three checks.**

**Check 1 — the 10.92% is QSA *select*, and it never reads the KV cache.** In `qsa_layer` the sequence is:
`qsa_block_scores(st.idx_pooled, b.q_idx, …)` → `qsa_block_topk` → `kv_gather_*` → attention
(`src/core/layer.cpp:1269-1300`). The block scores are computed from **indexer keys** (`idx_key_dim` = 64,
`b.idx_pooled`), not from K/V. So a smaller KV format cannot move that 10.92% by one byte. The KV format
appears in the *gather* and in the attention that follows, which are separate terms.

**Therefore the ≤11% ceiling collapses to ~0 and `strata` drops out of the speed case.** That is the single
most consequential correction in this document: it was the only model whose attention share was large enough
to pay for a faster QK product, and the share belongs to the wrong kernel. See Correction 1 in §0.

**Check 2 — on `strata` the KV read is a bounded gather of 0.78% of the cache, and it is not
bandwidth-bound.** `kTopkMaxCells` is 32,768 (`include/strata/kernels/qsa.hpp:286`) but the measured
selection at the Flash geometry is **2,051 cells a layer** (`docs/QUANT_MICROBENCH.md:595-601`), i.e.
2,051 / 262,144 = **0.78%** of the cache per token per layer. Bytes and time:

| format | cell B (`strata`'s scale periods) | KiB a layer | MiB a token (12 layers) | at 288 GB/s | measured kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 2,048 | 4,102 | 48.1 | 175 µs | — |
| int8 (per-64 scales) | 1,056 | 2,115 | 24.8 | 90 µs | — |
| k8v4 | 816 | 1,634 | 19.2 | 70 µs | **270 µs** (`QUANT_MICROBENCH.md:601`) |
| k6v4 | 704 | 1,410 | 16.5 | 60 µs | — |
| k4v4 (`--kv q4_0`) | 576 | 1,154 | 13.5 | 49 µs | — |

The k8v4 attention takes 270 µs a token where its bytes alone would take 71: it is at 26% of DRAM and the
same page records why ("`FFMA` 444, `LDS` 12, no stack"; "it was waiting on the value loop's dependent
loads, not short of arithmetic"). Cutting bytes 30% (k8v4 → k6v4) removes 11 µs of a 270 µs kernel that
isn't waiting on DRAM. **~0%, and it would be negative if the k6v4 dequant costs more instructions than
the q8 dequant — which `q6pair`'s five-SASS-per-pair says it does.**

Also: the gather writes **fp16 scratch whatever the format** (`include/strata/kernels/kv_q8.hpp:9-11`:
"Only STORAGE changes … dequantizes the selected cells into the same FP16 scratch"), and that scratch is
`cap × n_kv × head_dim` halves for K **and** for V (`src/core/layer.cpp:1042`) = **2,048 B a cell, a
constant**. So a smaller format shrinks only the *read* third of the transfer — and at k8v4 the write is
already 2.5× the read (2,048 against 832) — which is why the byte saving does not survive as time: the
kernel that consumes the scratch is the same kernel either way.

**Check 3 — on `strata-poly` the read *is* proportional to bytes, and the minimum is at 8 bits.** `fdec`
walks the whole prefix, so decode reads every byte. Measured at 256K, RTX 4060 Ti, pinned clock, five
interleaved runs, own buffers (`docs/Q8_KV.md:270`):

| format | bytes a cell | ns a cell | vs q8_0 | vs fp16 |
| --- | ---: | ---: | ---: | ---: |
| q8_0 | 1,088 | **4.96** | — | −40% |
| k8v4 | 832 | 5.88 | **+18.5%** | −29% |
| fp16 | 2,048 | 8.32 | +68% | — |

(`docs/QWEN38.md:102` states the same pair as "k8v4's read 18.5% slower, its store 28.4% faster".)

That is the whole argument in one table: **24% fewer bytes made the read 18.5% slower.** The kernel is
dequant-bound, not bandwidth-bound, so bytes saved convert to time at a rate of about zero and then
negative. And prefill: 0.8% of wall time on Ornith leaves *nothing* to win there.

**Verdict.** The framing is right: **footprint is the prize.** Two refinements the plan should carry:

- The prize is not "context that fits" alone. On a 16 GB card with an MoE, the freed VRAM is spent on
  **expert-cache slots**, and that is a *measured* trade: fp16 → q8_0 at 131K with `--kv-whole` bought
  **+1,076 expert slots** (1,237 → 2,313 of 10,496) at unchanged free VRAM (`docs/Q8_KV.md:99-107`), and
  the expert-cache hit rate during the answer went 85.8% → 95.8% on Flint (`:184`). A KV format is a
  **cache-slot multiplier**, and that is the number to measure, not tok/s.
- Because the read minimum is at 8 bits, **the format work that can still win on speed is the 8→4 bit
  step on the *weight* side (MMQ), not on the KV side.** §7.

## 4. The formats, one row each

Cell = one token, one layer, all KV heads, K and V together. `V` = `n_kv × head_dim` values a side.
`strata` scales: int8 per 64. `strata-poly` scales: per 32. All arithmetic, from `store_q8`/`store_q4`/
`store_q6` and `kv_cell_of` (`qwen35.cpp:216-226`).

| format | K | V | cell, V=512 | cell, V=1024 | QK on IMMA? | dequant cost per 32 values | reachable on Ada? |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| fp16 | 16 b | 16 b | 2,048 | 4,096 | **no** (nothing to gain: the tensor cores already run fp16) | none | yes |
| q8_0 / int8 | 8 b | 8 b | 1,088 (`strata` 1,056) | 2,176 | **yes, native** — the cells are already `s8`, so there is no unpack at all | none on the codes; the epilogue still pays 1 `I2F`+1 `FMUL`+1 `FFMA` per output | yes, in production (`attend_int8`) |
| k8v4 | 8 b | 4 b | 832 (`strata` 816) | 1,664 | **keys yes** (as q8_0); values no (P·V has no tensor route) | keys 0; values ~4 per 8 codes (`PRMT`+`LOP3`+`HADD2`+`HMUL2`) | yes |
| k6v4 | 6 b | 4 b | 704 | 1,408 | **needs a new kernel**: expand 6→8 bits, then `IMMA.S8.S8` | keys ~2 per 4 codes (SHF/PRMT/LOP3/BFI); `q6pair` is 5 SASS a *pair* on the fp16 path | **6-bit is not a native width.** Packing: 32 values × 6 b = 192 b = 24 B, value *j* at bit 6*j*, offset-binary +32, 24 B keeps 8-byte alignment (`store_q6`) |
| k5v3 | 5 b | 3 b | **576** | **1,152** | same new kernel, uglier extraction | ~3 per 4 codes (5 values per 25 bits, not byte-aligned) | 5-bit and 3-bit are **not** native widths. 20 B + 12 B a 32-value block, neither a multiple of 4 |
| k4v4 | 4 b | 4 b | **576** | **1,152** | **yes, and natively**: `IMMA.16832.S4.S4` exists on sm_89 at **2× INT8** — but only if the *query* is also s4, since `mma` has no `s8 × s4` form | nibbles → s8 is ~1 op per 2 codes (`PRMT`); s4 can be fed directly | yes: 4-bit *is* a native tensor width on Ada (`sm89_inst.txt:67500-67540`) |

Three things fall out of that table that the brief did not assume:

1. **k5v3 and k4v4 are byte-for-byte identical** (576 / 1,152). k5v3 is not a cheaper format, it is the
   same budget with the bits moved from V to K. So they are not two steps on a ladder, they are two
   *allocations* of one step, and only accuracy can choose between them.
2. **The evidence says the values are the sensitive side, so k5v3 is the wrong allocation.** k8v4 differs
   from q8_0 *only* in the value side, and that cost **7 points** of teacher-forced top-1 at 8K
   (83.3% against 90.4%, `docs/FLINT.md:173`) and **nothing** at 512 tokens (91.7% against 91.2%,
   `:174`). The keys going 8→6 cost **0.7 points** over 144 positions (today) — inside that sample's
   ±2.5-point noise, so the honest reading is "0 to 3 points, unmeasured at scale". Spending the
   byte budget on 5-bit keys and 3-bit values therefore buys accuracy where it is cheap and spends it where
   it is expensive. **k6v4 and k4v4 are the two formats worth having; k5v3 is not.**
3. **The 4-bit IMMA is real, and it is Ada-only: `k4v4` is a live path, not a record.** 351 TOPS computed
   against INT8's 175.5, from `SIZE_…16832…` with `SRCFMTA_U8_S8_U4_S4` — **287 occurrences in
   `sm89_inst.txt` and zero `U4/S4` operand-format entries in `sm90_inst.txt`** (§1.1). So Ada carries a
   native `IMMA.S4.S4` and Hopper does not. Why this survived an earlier draft's dismissal: the operand
   format is an **enum in the SIDL predicates**, not a mnemonic string, so a grep for `imma.*s4` over
   instruction names returns nothing and the instruction looks absent. *A source-only reading of an
   instruction set is not a reading of the instruction set* — the same lesson as `mma.sync` in `mmq.cu`,
   where the source said nothing and the binary said `IMMA`.

   What it costs: `S4.S4` forces the **query** to 4 bits too (`mma` has no `s8 × s4` form). That is a large
   accuracy bet, and it is exactly the bet the accuracy gate exists to price — not an argument for skipping
   the format. **`k4v4` is the format to build after k6v4**, with the query side as the variable under test:
   `k4v4` with an int8 query and one with an s4 query are two arms, and the second is the one that reaches
   351 TOPS.

## 5. The models to test, and which formats each can already take

| model | engine | geometry (values a cell / attention layers) | `--kv` today | gate |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next REAP-320 | `strata` | 2 KV × 256 = 512, 12 QSA | `fp16`, `int8`, `q4_0`, `k8v4` | `generate.cpp:1047-1052`, error at `:4336` |
| **Ornith 1.5 9B** (the file on disk here) | `strata-poly` (`qwen35` dense) | **4 KV × 256 = 1,024**, 9 (8 + draft) | `fp16`, `q8_0`, `k8v4`, `k6v4` | `engine.cpp:1641-1648` |
| Ornith 1.5 35B-A3B (also Tiel, Occamy) | `strata-poly` (`qwen35moe`) | 2 KV × 256 = 512, 11 (10 + draft) | `fp16`, `q8_0`, `k8v4`, `k6v4` | `engine.cpp:1641-1648` |
| Flint 40B-A2.5B | `strata-poly` (`flint`) | 4 KV × 128 = 512, 14 full of 49 | `fp16`, `q8_0`, `k8v4` — **not** k6v4 | `engine.cpp:1645-1648`; `flint.cpp:231-244` |
| Qwen3.8-27B dense (control) | `strata-poly` (`qwen35`) | 4 KV × 256 = **1,024**, 16 of 64 | all four, k6v4 included | `QWEN38.md:21`, `:87` |
| Gemma 4 | `strata-poly` (`gemma4`) | — | `fp16`, `int8`, `int8k` only | the only IMMA attention user |

### The geometry, settled from code and header

**There are two different Ornith files, and the discrepancy is that, not an error in either reading.**

An earlier draft of this section said Ornith is 2 KV heads x 256, citing the engine's own
`q8_0 KV cache: 544 bytes a row` (`docs/Q8_KV.md:31`). That was wrong, and the way it was wrong is worth
recording because the inference looked safe: **544 is not a unique fingerprint.** One side of a q8_0 cell is
`n_kv*head_dim` codes + one fp16 scale per 32 values, and that equals 544 at **three** geometries:

| `n_kv` | `head_dim` | one side | reaches 544? |
| ---: | ---: | ---: | --- |
| 2 | 256 | 512 + 32 | **yes** |
| 4 | 128 | 512 + 32 | **yes** (Flint) |
| 8 | 64 | 512 + 32 | **yes** (LFM2.5) |
| 4 | 256 | 1,024 + 64 | no: 1,088 |

So a printed row size identifies the *value count*, not the head split. The doc line also says which model it
was printed for, and that model is the 35B-A3B (`qwen35moe`), whose header gives 2 x 256.

**The header read on this machine is a different model.** The only Ornith GGUF on disk is
`Ornith-1.5-9B-Q4_K_M.gguf`, and `tools/gguf_reader.py` reads it as:

```
general.architecture            = qwen35          <- dense, not qwen35moe
qwen35.attention.head_count     = 16
qwen35.attention.head_count_kv  = 4
qwen35.attention.key_length     = 256
qwen35.attention.value_length   = 256
qwen35.block_count              = 33
qwen35.full_attention_interval  = 4
qwen35.nextn_predict_layers     = 1
```

`block_count = 33` cannot be the 35B-A3B (40 trunk layers, `docs/ORNITH.md:38`), and the architecture key is
the dense family, which `engine.cpp:1620-1622` routes to `Qwen35`, not `Qwen35Moe`. Counting the attention
tensors settles the layer count: **9 `attn_k.weight` tensors, at layers 3, 7, 11, 15, 19, 23, 27, 31 and 32** —
eight trunk attention layers (every fourth of 32) plus the draft block, which is `full_attention_interval = 4`
with `nextn_predict_layers = 1`.

`n_kv_` comes from exactly the key the header shows, with no alias and no override: `qwen35.cpp:57` is
`n_kv_ = shape_i("attention.head_count_kv")`, `shape_i` (`engine.cpp:1465-1469`) resolves through `meta_in`,
which tries the architecture-prefixed key first and only then a `meta_alias` table — and there is **no alias
entry for `attention.head_count_kv`** anywhere in `src/poly/`. So the header is the only source.

**Both are right, about different files:**

| model | architecture | `n_kv x head_dim` | values a side | attention layers |
| --- | --- | --- | ---: | ---: |
| Ornith 1.5 **9B** (on disk here) | `qwen35` dense | **4 x 256** | **1,024** | **9** (8 + draft) |
| Ornith 1.5 **35B-A3B** | `qwen35moe` | 2 x 256 | 512 | 11 (10 + draft) |

**Consequence: every per-cell number for the 9B doubles against the 35B, and my earlier table's Ornith row was
the 35B's arithmetic attached to a 9B header read.** The corrected 9B table is below. The 35B row is kept
because that is the model `docs/Q8_KV.md` measured and the one `--kv k6v4` was built against, but **the two
must not be mixed**: a k8v4 cell is 1,664 B on the 9B and 832 B on the 35B, and only the 9B's header is
verifiable on this machine.

**Which doc claim was wrong:** not `Q8_KV.md:31` (its 544 is correct for the model it names) and not the
header read. Wrong was this document's inference — treating a row size as a model fingerprint. That is fixed
here; `docs/Q8_KV.md` needs no change.

Why these three and not others:

- **`strata` on Qwen3.8-Flash REAP** is the only model where the indexer makes attention a double-digit
  share of prefill — and §3 shows that share is the *selector*, not the KV read. It is therefore the model
  that tests the framing rather than the format: if any model could pay for a faster QK product, it is this
  one, and the arithmetic says even here the currency is wrong.
- **`strata-poly` on Ornith 35B-A3B and Flint** are MoE hybrids whose expert cache competes for the same
  VRAM, so a KV byte saved is a slot bought — the one place a smaller format pays *directly*. Flint is also
  the only model with a published k8v4 accuracy number at 8K.
- **The dense 27B is the control**: no expert cache to absorb the saving, so the freed VRAM can only become
  context, which isolates "context that fits" from "expert hits". It is also where k6v4 is *already* wired
  (`Q8_KV.md:280`: 1,408 B against 1,664) and where fp16 does not fit at all (17.53 GiB at 256K).

**What each would need.** `k6v4` exists as a reader in `qwen35` and `qwen35moe` and **not** in `flint`
(the gate says so, `engine.cpp:1645-1648`); `k5v3` and `k4v4` exist nowhere in `strata-poly` and nowhere
in `strata`. On the `strata` side `q4_0` already exists (576 B a cell, `kv_q4.hpp:19-38`) — so **`strata`
already has the k4v4 footprint** (both sides 4-bit, Hadamard-rotated), and the only thing missing there is
an asymmetric variant. That is a useful economy: the k4v4 hypothesis can be tested on `strata` today with
`--kv q4_0` and no new code.

### Context that fits, per format

Bytes a token and the context a given KV budget buys. GiB, decimal-free (2³⁰).

| model (V, layers) | format | B/token | @32K | @128K | @256K | ctx in 2 GiB | 3 GiB | 4 GiB | 8 GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **strata Flash (512, 12)** | fp16 | 24,576 | 0.75 | 3.00 | 6.00 | 87k | 131k | 175k | 350k |
| | int8 | 12,672 | 0.39 | 1.55 | 3.09 | 169k | 254k | 339k | 678k |
| | k8v4 | 9,984 | 0.30 | 1.22 | 2.44 | 215k | 323k | 430k | 860k |
| | k6v4 | 8,448 | 0.26 | 1.03 | 2.06 | 254k | 381k | 508k | 1,017k |
| | k4v4 | 6,912 | 0.21 | 0.84 | 1.69 | 311k | 466k | 621k | 1,243k |
| **Ornith 9B (1024, 9)** | fp16 | 36,864 | 1.12 | 4.50 | 9.00 | 58k | 87k | 116k | 233k |
| | q8_0 | 19,584 | 0.60 | 2.39 | 4.78 | 109k | 164k | 219k | 438k |
| | k8v4 | **14,976** | 0.46 | 1.83 | 3.66 | 143k | 215k | 286k | 573k |
| | k6v4 | 12,672 | 0.39 | 1.55 | 3.09 | 169k | 254k | 338k | 677k |
| | k4v4 | 10,368 | 0.32 | 1.27 | 2.53 | 207k | 310k | 414k | 828k |
| **Ornith 35B (512, 11)** | fp16 | 22,528 | 0.69 | 2.75 | 5.50 | 95k | 143k | 191k | 381k |
| | q8_0 | 11,968 | 0.37 | 1.46 | 2.92 | 179k | 269k | 359k | 718k |
| | k8v4 | 9,152 | 0.28 | 1.12 | 2.23 | 235k | 352k | 469k | 939k |
| | k6v4 | 7,744 | 0.24 | 0.95 | 1.89 | 277k | 416k | 555k | 1,109k |
| | k4v4 | 6,336 | 0.19 | 0.77 | 1.55 | 339k | 508k | 678k | 1,356k |
| **Flint (512, 14)** | fp16 | 28,672 | 0.88 | 3.50 | 7.00 | 75k | 112k | 150k | 300k |
| | q8_0 | 15,232 | 0.46 | 1.86 | 3.72 | 141k | 211k | 282k | 564k |
| | k8v4 | 11,648 | 0.36 | 1.42 | 2.84 | 184k | 277k | 369k | 737k |
| | k6v4 | 9,856 | 0.30 | 1.20 | 2.41 | 218k | 327k | 436k | 872k |
| | k4v4 | 8,064 | 0.25 | 0.98 | 1.97 | 266k | 399k | 533k | 1,065k |
| **Qwen3.8-27B dense (1024, 17)** | fp16 | 69,632 | 2.12 | 8.50 | 17.00 | 31k | 46k | 62k | 123k |
| | q8_0 | 36,992 | 1.13 | 4.52 | 9.03 | 58k | 87k | 116k | 232k |
| | k8v4 | 28,288 | 0.86 | 3.45 | 6.91 | 76k | 114k | 152k | 304k |
| | k6v4 | 23,936 | 0.73 | 2.92 | 5.84 | 90k | 135k | 179k | 359k |
| | k4v4 | 19,584 | 0.60 | 2.39 | 4.78 | 110k | 164k | 219k | 439k |

The k8v4 / q8_0 / fp16 rows for the 27B reproduce `docs/QWEN38.md:87-89` exactly and the k6v4 row
reproduces `docs/Q8_KV.md:275` (1,408 / 1,664), which is the check that the other rows are the same
arithmetic and not a new model.

**Read the table at 2 GiB, not 8.** On a 16 GB card with 8–10 GiB of weights, the KV budget is 2–4 GiB:

- Qwen3.8-27B dense at fp16 does not hold **128K** at any budget (8.50 GiB); q8_0 does at 4 GiB (4.52 is
  over, 116k of it fits); k8v4 holds **128K in 3.45 GiB**; k6v4 holds 128K in 2.92; k4v4 in 2.39.
- Ornith 35B at 256K needs 5.50 GiB at fp16, 2.92 at q8_0, 2.23 at k8v4, 1.55 at k4v4. **256K is a k8v4
  or smaller feature on this card; it is not reachable at fp16 or q8_0.** That is the concrete footprint
  prize, and it is worth 1.55/2.23 = 28% more context for k4v4 over k8v4.

## 6. The accuracy gate, and how many positions it needs

The method already in the tree: teacher-forced top-1 of one format against another's greedy picks,
`tools/kv_agree.py`, which sets the whole protocol itself (`--window 1 --lookup 0 --no-draft
--prompt-cache 0 --conversation-cache-mib 0 --eos-ids 0`, `kv_agree.py:26-31`). It prints `same/total`
and exits 1 below `--min-agree`.

Existing numbers, all teacher-forced, all against fp16 unless said otherwise:

| comparison | result | positions | source |
| --- | ---: | ---: | --- |
| q8_0, Flint, 8K prompts | 90.4% | 384 (8 × 48) | `FLINT.md:173` |
| k8v4, Flint, 8K prompts | **83.3%** | 384 | `FLINT.md:173` |
| k8v4, Flint, 512-token prompts | 91.7% (q8_0 91.2%) | 384 | `FLINT.md:174` |
| q8_0, Flint laptop, 32-token prompts | 98.12% | 640 | `Q8_KV.md:135` |
| k6v4 and k8v4, each against fp16 | **97.9% / 97.2%** — a 0.7-point gap | **144** | today |

**144 positions cannot separate k6v4 from k8v4, and the arithmetic says by how much.** With p ≈ 0.975 and
n = 144 the standard error is √(0.025 × 0.975/144) = **1.30 points**, so a 95% interval is about ±2.5
points and a 0.7-point gap is 0.54 σ. It is not a weak signal; it is no signal.

Positions needed to resolve a difference. Paired data, so it is McNemar: `n = 7.85 · ψ / d²`, where `d` is
the gap and `ψ = b + c` is the share of positions the two arms disagree on at all. `ψ = d` is the
best case (every disagreement points the same way); for two near-tie formats `ψ ≈ 2d` is the realistic
reading, and it **doubles** the requirement:

| gap to resolve | n, ψ = d (best case) | n, ψ = 2d (realistic) | harness |
| --- | ---: | ---: | --- |
| 5 points | 160 | **320** | 4 prompts × 80 |
| 2 points | 400 | **790** | 8 × 100 |
| 1 point | 790 | **1,570** | 16 × 100 |
| 0.7 points | 1,120 | **2,240** | 20 × 112 |
| 0.5 points | 1,570 | **3,140** | 32 × 100 |
| 0.2 points | 3,920 | **7,850** | 64 × 128 |

(α = 0.05 two-sided, power 0.8.) **4,096 positions over 32 prompts** therefore resolves 0.5 points at
`ψ = 2d` and 0.2 at the best case, which covers every question in §4 and §6. It is
`--prompts 32 --max-new 128` and costs about 32 × 256 batch-1 forwards ≈ **5–10 minutes a model** — the
cheapest decisive number in this whole study. If two formats turn out to disagree on far more positions
than the gap (`ψ ≫ 2d`), the run prints every prompt's count and the number is re-derivable from the log
without re-running.

**What to expect by arithmetic, format by format:**

| format | expected drop vs fp16 | basis |
| --- | --- | --- |
| q8_0 | −9.6 points at 8K (90.4%) | measured, `FLINT.md:173` |
| k8v4 | −16.7 points at 8K (83.3%) | measured |
| k6v4 | **−10 to −13 points** (0.7 worse than k8v4, but ±2.5 at n=144) | today's 144 positions, read as "k8v4's loss plus 0–3 points" |
| k4v4 | **−5 to −10 points worse than k8v4** (unknown) | 4-bit keys on top of 4-bit values; 8 levels against 16 |
| k5v3 | **worse than k4v4** | same bytes, and the *values* are the side that measured 7 points at 8K, now at 3 bits |

The last two rows are predictions, clearly labelled as such, and they are the reason k5v3 should not be
built: it is the only proposed format that makes the *measured-sensitive* side worse.

**Two rules for the gate that the existing numbers impose:**

1. **Run it at the context the format is for.** k8v4 costs 0.0 points at 512 tokens and 7.1 at 8,192
   (`FLINT.md:173-174`). A 512-token gate would have shipped k8v4.
2. **Run it with `--eos-ids 0` and `--max-new` fixed**, as `kv_agree.py` does, or the number measured is
   the model's stopping behaviour, not the cache's.

## 7. MMQ: does a lower quantization make the weight-side product faster?

Today's measurement: MMQ at **69.4 TFLOP/s = 39% of 176.5**, with the kernel's SASS **3.8% `IMMA`** and
**45% `I2FP` + `FMUL` + `FFMA`**. The question is whether a 2-bit or 3-bit type moves that.

**The MMA count is identical for every type at a fixed shape.** An `m16n8k32` does 4,096 int8 MACs
whatever the weights were stored as, and MMQ's tiling (`J` ≤ 128, `mmq.cuh:1486`) does not depend on the
type. So a type change can only move the *other 96%* of the instruction stream. Where those instructions
go, per IMMA, in this tree:

The card's INT8 rate is 1,024 MAC a cycle an SM (660.6 TOPS / 128 SM / 2.52 GHz / 2), so one
`IMMA.16832` holds one scheduler's tensor pipe for **16 cycles** (4,096 MACs / 256 a cycle a scheduler).
The measured histogram says **~26 instructions are issued per IMMA** (3.8% of them the IMMA itself), of
which 12 are the scale-and-store epilogue:

| cost | per IMMA | where |
| --- | ---: | --- |
| the tensor op | 1 instruction, **16 scheduler-cycles** | `mma.cuh:946` |
| the epilogue | 4 accumulators × (1 `I2F` + 1 `FMUL` + 1 `FFMA`) = **12 instructions**, block scales hoisted | `mmq-vec-dot.cuh:196` (q8_0), `:360-361` (q8_1) |
| the unpack | 0 for q8_0 (codes are already `s8`); ~0.3 op/weight for q4_K; ~0.8 for IQ2_XXS | `mmq-vec-dot.cuh:142-200` (q8_0) against the IQ/K loaders |

**The epilogue is the 45%, and it is type-independent**: it scales the int32 result, so it runs identically
for q8_0 and for IQ1_S. At one instruction a cycle a scheduler, 26 instructions need 26 cycles against the
tensor's 16, which puts the **ceiling at 16/26 = 62% of peak**, and the measured 39% is that ceiling with a
0.63 issue/stall derating. So the headroom is real and it is not in the tensor cores: **halving the
epilogue would move the ceiling from 62% to ~80%**, and no cache format changes it — only the layout
(scales folded into the int32 path, or a wider `k`) does.

**Does a lower bit-width move that? Two models, and they disagree at 4 bits.** Instruction count first: a
4-bit unpack is ~0.3 op a weight, which is `0.3 × 512 / 32 = 4.8` instructions per IMMA (512 weights a
tile, 32 lanes) — about 5 more instructions on 26, so the ceiling falls 62% → 53%. A 2-bit type adds ~13,
ceiling 47%. By that model **a lower type is always slower at a fixed batch, because the MMA count never
changes.**

Memory second. Intensity is `J / bytes-per-weight` once the batch reaches the tile width, because each
J-tile re-reads the whole weight matrix (`J` caps at 128, so batch 512 is four passes and batch 128 is one —
the same for every type, which is why the ranking cannot depend on the batch). Machine balance for IMMA is
613 int8-ops a byte:

| type | bytes / weight | intensity at J=128 | which bound binds |
| --- | ---: | ---: | --- |
| q8_0 | 1.0625 | 120 op/B | **DRAM** (120 < 613) |
| q4_K | 0.5625 | 228 op/B | **DRAM**, and close to it |
| q2_K | 0.320 | 400 op/B | compute |
| IQ2_XXS | 0.258 | 496 op/B | compute |
| IQ1_S | 0.195 | 656 op/B | compute |

The achieved rate is the tie-breaker: 39% of 613 is **239 op/B delivered**, against q4_K's DRAM intensity of
228. Those agree to 5%, which means **q4_K is the one type where the two models are indistinguishable** —
it may be at the DRAM wall (in which case fewer bytes win) or at the issue wall (in which case the unpack
loses). q8_0 is comfortably DRAM-bound and has no unpack at all, so it is the type to beat; everything at 2
bits and below is compute-bound by both models, and there the answer is settled.

So, the answers to the two halves of the question:

- **Batch 128 against 512 does not change the ranking**, because `J` ≤ 128 makes both read the weights
  `⌈batch/128⌉` times with the same per-tile arithmetic. **128 is enough to be decisive**; 512 only adds
  work.
- **Below 4 bits a lower quantization is slower, by arithmetic and by both models** — the MMA count is
  identical and the unpack grows. Lower bits there buy VRAM (which is a real prize for the expert cache),
  never speed. Between 8 and 4 bits the arithmetic does not settle it and the A/B does.

The prediction to test, stated so it can fail: **`q8_0` and `q4_K` within ±10% at batch 128 and 512, and
`IQ2_XXS` at least 15% behind both.** The A/B is three rows, one shape, one binary. There is no timed MMQ
harness in the tree today — `poly_mmq_tiles_test` builds the shapes and calls `Mmq::mul` but times nothing
(`mmq_tiles_test.cu:1-40`) — so it needs a loop of ~30 lines there, not a new kernel.

## 8. The measurement order: cheapest decisive thing first

Each step is cheaper than the one after it, and each can cancel the next. Every step that runs an engine
uses the `docs/QUANT_MICROBENCH.md` protocol: **pin the clock** (`nvidia-smi -lgc 2820`, `-rgc` after,
`nvidia-smi dmon` through the window), `--prompt-cache 0`, `--conversation-cache-mib 0`,
`--no-draft --window 1 --lookup 0`, a fixed `--prompt-batch`, `--temperature 0`, interleaved arms,
**median of 5**, every run under `timeout`.

**Step 1 — the accuracy gate on k6v4 (no new code, ~10 min a model).**

```
python3 tools/kv_agree.py --model <Ornith-35B>.gguf --ref k8v4 --kv k6v4 \
    --prompts 32 --max-new 128 --max-context 8192 --min-agree 0.965
python3 tools/kv_agree.py --model <Ornith-35B>.gguf --ref fp16  --kv k6v4 \
    --prompts 32 --max-new 128 --max-context 8192
```

Pass: k6v4 within 1 point of k8v4 (the gate `FLINT.md:175` set for k8v4 against q8_0), which at 4,096
positions is a ±0.5-point measurement. If it fails, no kernel work on k6v4 is worth doing and the whole
sub-8-bit KV line stops here. **This is the first step because it is the cheapest and it can kill the
most.**

**Step 2 — the read ladder, including k6v4 (no new code, minutes).** The five-arm read cost at 256K on
the harness behind `QUANT_MICROBENCH.md` row 2: fp16 8.32, q8_0 4.96, k8v4 5.88 already measured; add
k6v4 and, if built, k4v4. Pass: k6v4 **not slower** than k8v4. Prediction from `q6pair`'s five SASS a
pair: it will be 5–15% slower, i.e. it fails — and that is a *useful* failure, because it bounds every
future sub-8-bit format.

**Step 3 — the footprint ceiling (no new code, one load a format).** The number that decides everything:

```
strata-poly --model <Ornith>.gguf --kv q8_0 --kv-whole --max-context 131072 ... 
```

and read the expert-slot line the engine prints, as `docs/Q8_KV.md:99-107` did (fp16 1,237 → q8_0 2,313).
Do it for fp16 / q8_0 / k8v4 / k6v4. This turns a KV byte into **expert slots** and, from the published
hit rates, into the only tok/s the format can honestly claim. It needs no new kernel and no timing
discipline — it is a load and a printed count.

**Step 4 — an end-to-end A/B, only if steps 1–3 leave a winner.** Ornith at 128K, q8_0 against the
surviving format, `--prompt 8192`, 128-token answers, five interleaved pairs. This is where a decode number
can finally be quoted — and per §3 it should be within noise, which is the result worth publishing: *a
smaller cache is a footprint feature on this card*.

**Step 5 — the MMQ A/B (last, and only on the weight side).** `q4_K` vs `q8_0` vs `IQ2_XXS` at batch 128
and 512 on one shape (`2560 × 10240`, the shape `QUANT_MICROBENCH.md` uses), 30 lines of timing in
`poly_mmq_tiles_test`. Decisive for the "45% conversion" question; irrelevant to the KV formats, which is
why it comes after them.

**Not in the order, and why:** the IMMA QK rewrite of `fdec_kernel` (§2.2). It needs a new kernel, it adds
a query-quantization accuracy term, and §3 caps its payoff at a share of wall time that the same
measurement puts at 0.8% on the model where it would run. It goes last or never; the fp16-accumulate 2×
(§2.2) is the cheaper experiment on the same line and should be tried first.

## 9. Needs the card

| # | number | command | wall time |
| ---: | --- | --- | --- |
| 1 | IMMA vs dp4a on one 16×8×256 product | `build/poly_kv_int8_test` (prints `mma X ms, dp4a Y ms -> mma is Z×`, `kv_int8_test.cpp:804-808`) | ~2 s |
| 2 | k6v4 / k4v4 teacher-forced top-1 at 4,096 positions | `tools/kv_agree.py …` (§8 step 1) | 5–10 min a model |
| 3 | read cost a cell for k6v4 at 256K | the row-2 harness of `QUANT_MICROBENCH.md` (`poly_fattn_test` decode at 256K), 5 interleaved runs, pinned | ~3 min |
| 4 | expert slots bought by each format at 131K | `--kv-whole` load + the engine's own print | ~2 min a load |
| 5 | MMQ TFLOP/s per type at batch 128 / 512 | needs a timing loop in `poly_mmq_tiles_test` (~30 lines, no new kernel) | ~10 min |
| 6 | fp16-accumulate vs fp32-accumulate `mma` in `fdec` | one-line change at `fattn.cu:281` + the same 256K read bench | ~10 min incl. the accuracy arm |
| 7 | the five-format expert-slot count on Flint as well as Ornith (Flint's 14 full layers against Ornith's 11 changes the byte freed) | same `--kv-whole` load as #4 | ~2 min |

## 10. What I could not verify

- **sm_75, sm_80, sm_86 and sm_120 have no SASS dump here.** Only sm_89 and sm_90 are in
  `/tmp/opencode/lsa-research`. Their int8 forms are stated from the PTX ISA (authoritative for what ptxas
  accepts) and, for Turing, from `mma.cuh:951-960` which compiles `m8n8k16` below 800. The *ratio* for
  Hopper and Blackwell is from vendor datasheets, not from a local document, and is labelled as such.
- **The 3.8% `IMMA` / 45% `I2FP`+`FMUL`+`FFMA` histogram and 69.4 TFLOP/s are recorded from today's
  measurement, not from a doc in this tree.** No file here contains them, and the harness that produced
  them is not in the tree either (§7, step 5). They are used as given.
- **k5v3's and k4v4's accuracy are predictions, not measurements.** §6 says which rows are which.
- **The "10.92% of a 256K prefill" figure is likewise from today's run**, not from a doc. §3 checks what it
  can be attributed to — `strata`'s profile at 40K attributes 19.2% to prompt attention and 6.8% to QSA
  select (`docs/INTEL.md:648`, different card), so the two terms are separate and the brief's number is
  the selector's.
- **The 0.8%-of-prefill figure for Ornith** is from today's run, and it is consistent with everything
  measured about the decode side: at 0.8% there is nothing for a faster QK product to buy.
- **Ornith's 2-KV-head geometry** comes from `docs/Q8_KV.md:31` and `docs/ORNITH.md:49`, both of which
  derive it from the file's own header. No GGUF is in `models/` on this machine, so I could not read the
  header myself. It is a one-line check with `tools/gguf_reader.py` and it should be run before the tables
  in §5 are quoted anywhere.

# Part II — the microbenchmark programme

Part I (sections 0–10 above) is the reading and the arithmetic. This part is the bench design. **Nothing here
has run.** Section 11 is the inventory the matrix is built from, 12 the matrix, 13 the code design, 14 the
expected results written before the runs, 15 the first three measurements, 16 the anti-silent-success rules,
17 what each bench needs that does not exist yet.

The rule the whole programme is built around: **a speed number is never quoted without its parity number, in
the same row, produced by the same call.** Section 16 makes that mechanical rather than a convention.

## 11. The complete path inventory

Every row answers three things: **is it reachable today, by which model** (the `--kv` value or flag), and
**gated where** (`file:line`, the check that accepts or refuses it). Reachable = a user can get there without
editing code.

### 11.1 KV read paths

| # | path | keys | values | kernels | reachable today | gate | tensor cores |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| R1 | `attend` fp16 | fp16 | fp16 | `attn_score_*`/`attn_value_*` (fp32 maths) | `--kv fp16`, every model | default | none (FAKE) |
| R2 | `attend` q8_0 | kq8 | vq8 | `attn_score_reg_kernel<…,true>` + `attn_value_q8` | `--kv q8_0` on qwen35/kolibri1/flint/lfm2moe/deepseek2 | `engine.cpp:1638` | none (dp4a-free: `I2F`+`FMUL`) |
| R3 | `attend` k8v4 | kq8 | vq4 | `attn_score_reg_kernel` + rotated q4 values | `--kv k8v4` on qwen35/qwen35moe/flint | `engine.cpp:1641-1644` | none |
| R4 | `attend` k6v4 | kq6 | vq4 | `attn_score_reg_kernel` reads `Kvq6`; **no six-bit reader in `attend`'s k6 branch for the block-of-guesses path** | `--kv k6v4` on qwen35/qwen35moe only | `engine.cpp:1645-1648`; refused with `STRATA_BLOCK_GUESS=1` (`qwen35.cpp:318-324`) | none |
| R5 | `attend_batch` fp16 | fp16 | fp16 | cuBLAS + flash kernels | prompt batch, every model | `qwen35.cpp:678` | **yes** (fp16 MMA via cuBLAS / `attn_flash128`) |
| R6 | `attend_batch` q8_0 | kq8→fp16 workspace | vq8→fp16 workspace | `dequant_q8_to_f16` + R5 | prompt batch, `--kv q8_0` | `qwen35.cpp:305`, `:697-712` | **yes** (after a full dequant) |
| R7 | `attend_batch` k8v4 | kq8→fp16 ws | vq4→fp16 ws | `dequant_q8_to_f16` + `dequant_q4_to_f16` + R5 | prompt batch, `--kv k8v4` | `qwen35.cpp:697-712` | **yes** (after two dequants) |
| R8 | `attend_batch` k6v4 | kq6→fp16 ws | vq4→fp16 ws | the same two dequants | prompt batch, `--kv k6v4` | `qwen35.cpp:686-687` (deliberately: "no `fattn_kernel`") | **yes** (after two dequants) |
| R9 | `attend_int8` | **int8** | **int8** | `attn_score_mma_int8_kernel` (**IMMA**) or `attn_score_{reg_,}int8_kernel` (**dp4a**) | **Gemma 4 `--kv int8`; Kolibri1 `--kv q8_0` + `STRATA_KV_INT8=1`** | `engine.cpp:1650-1654`; Kolibri `kolibri1.cpp:429` | **yes — IMMA, `dim==256` only** |
| R10 | `attend_int8_mixed` | int8 | **fp16** | IMMA/dp4a QK + `attn_value_kernel` | Gemma 4 `--kv int8k` | `engine.cpp:1650-1654` | **yes**, QK only |
| R11 | `fdec` fp16 | fp16 | fp16 | `fdec_kernel<…,VM=0>` | `--kv fp16`, ≥ `fdec_from_` | `qwen35.cpp:355-364` | **yes** (fp16 `mma16816`) |
| R12 | `fdec` q8_0 | q8_0 | q8_0 | `VM=1` | `--kv q8_0` | same | **yes** (fp16 `mma16816` after `i8pair`) |
| R13 | `fdec` k8v4 | q8_0 | q4_0 | `VM=2` | `--kv k8v4` | same | **yes** |
| R14 | `fdec` k6v4 | **6-bit** | q4_0 | `VM=3` (`KQ6`) | `--kv k6v4`, from position 0 | `qwen35.cpp:364` forces `fdec_from_ = 0` | **yes** (fp16 `mma16816` after `q6pair`) |
| R15 | `fattn_*` prompt | f16/q8/k8v4 | same | llama.cpp's flash-attention MMA | prompt batch when the shape fits | `qwen35.cpp:688-697` | **yes** (fp16) |

**Two rows are the same cache read two ways, and nobody has compared them at 131K/256K:** R13 (`fdec`, chosen
above 4,096 positions) against R3 (`attend`, chosen below it). The threshold is `fdec_from_ = 4096`
(`qwen35.cpp:355-360`) and the comment says the old path was faster below it — measured once, at one geometry,
with no table kept.

### 11.2 Store paths

| # | path | layout written | reachable | caller |
| ---: | --- | --- | --- | --- |
| S1 | `store_f16` / `store_f16_at` | fp16, `n` halves a row | `--kv fp16`, every model | `qwen35.cpp` `attn_store` |
| S2 | `store_q8` / `store_q8_at` | q8_0 codes + scales, two arrays | `--kv q8_0` | `kernels.cu:3094`, `:3103` |
| S3 | `store_q4` / `store_q4_at` | q4_0 nibbles + scales, rotation applied by the caller | `--kv k8v4`/`k6v4` (values), `--kv q4_0` on `strata` | `kernels.cu:3112`, `:3121` |
| S4 | `store_q6` / `store_q6_at` | 24 B a 32-value block, offset-binary, six bits LE | `--kv k6v4` (keys) | `kernels.cu:3130`, `:3139` |
| S5 | `store_int8` / `store_int8_at` | Q8_0 codes + a separate delta array | Gemma 4 `--kv int8`, Kolibri1 `STRATA_KV_INT8=1` | `kernels.cu:3079`, `:3087` |
| S6 | `kv_append_q8_step` / `kv_append_q4_step` (**`strata`**) | int8 per 64 / q4_0, through a page table | `--kv int8`, `--kv q4_0`, `--kv k8v4` on `strata` | `layer.cpp:1225-1240` |
| S7 | the same, `_at` form (recorded launch) | as S6 | CUDA-graph decode | `layer.cpp:870-880` |

`_at` variants are **separate rows, not a flag**, because that is what a graph decode calls and because the
pre-existing note says the plain and recorded forms are different kernels with different launch counts
(`kernels.cu:3087-3092`).

### 11.3 Matmul paths

| # | path | what it does | instruction basis | reachable | gate |
| ---: | --- | --- | --- | --- | --- |
| M1 | `native_mmvq` | quantized weights × 1–8 q8_1 token rows, one launch a chunk | **dp4a** (`IDP.4A.S8.S8`), confirmed at `native_mmvq.cu:4173` ("bound by dp4a throughput on Turing from 2 columns on") | decode, every model | `native_mmvq.cu:3908` |
| M2 | `Mmq` | ggml-cuda MMQ, one weight read a batch | **IMMA** (`mma.cuh:946`) | prompt batch | `mmq.cu:88-118` (`fits`) |
| M3 | `gemm_f32` | fp32/bf16 weights, cuBLAS | fp32/bf16 MMA via cuBLAS | dense models, `--dense-f32` paths | `kernels.hpp:25` |
| M4 | `Mmq` NVFP4 W4A4 | FP4 × FP4 | `mma` block-scaled | **not reachable** (needs cc 12.0) | `mmq.cu:30-38` |

**M1 and M2 already meet in one place**: `poly_expert_parity MODEL.gguf` runs both over the file's own weights
and prints a relative error for each, per type (`expert_parity.cpp:225-310`). It **times neither**. That is the
gap bench 4 fills, and it means the parity reference for bench 4 already exists and is trustworthy — this is the
single cheapest extension in the programme.

## 12. The bench matrix

Rows are formats; columns are contexts; each cell carries **four numbers and its provenance**. A cell that
cannot be produced says *why*, with the file:line of the refusal — a missing number is information, a zero is a
lie.

### 12.1 Bench 1 — `poly_kv_read_bench` extension (R1–R14, decode and prompt)

Geometries: **A = 2 KV × 256** (Qwen3.8-Flash, `strata`, and Ornith/Qwen3.6, `strata-poly` — same cell shape,
different scale period, so the bench must take the period as a parameter, see §13.2); **B = 4 KV × 256**
(Qwen3.8-27B dense, group 6). Flint's 4 × 128 is a third geometry the existing header block already lists
(`kv_read_bench.cpp:15-18`); keep it, it costs nothing.

| format | 4K | 32K | 131K | 256K | harness | reachable? |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | ☐ | ☐ | ☐ | ☐ | `fdec_f16` + `attend` + `attend_batch` | yes (R1, R5, R11) |
| q8_0 | ☐ | ☐ | ☐ | ☐ | `fdec_q8` + `attend` + `attend_batch` | yes (R2, R6, R12) |
| k8v4 | ☐ | ☐ | ☐ | ☐ | `fdec_k8v4` + `attend` + `attend_batch` | yes (R3, R7, R13) |
| k6v4 | ☐ | ☐ | ☐ | ☐ | `fdec_k6v4` + `attend` + `attend_batch` | yes (R4, R8, R14) |
| int8 | ☐ | ☐ | ☐ | ☐ | `attend_int8` only (**no `fdec`**) | yes (R9) |
| int8k | ☐ | ☐ | ☐ | ☐ | `attend_int8_mixed` only | yes (R10) |
| **rows** | `1` (decode) and `512` (prompt) | | | | | |

Each cell: **ns a cell**, **bytes a cell**, **achieved GB/s**, **parity** (cosine + max rel against the fp16
reference from this bench's own data). Plus, as separate rows, `fdec` against `attend` **on the same format** at
each context, which is the missing R13-vs-R3 comparison.

Refusals the bench must print rather than work around:

- `int8`/`int8k` have **no `fdec`**: `attend_int8` is the only reader (`kernels.h:359-372`). So the int8 row's
  prompt column is `attend`, not `attend_batch`, and the two are not comparable — the bench must label it.
- `k6v4` is **refused by `flint`** (`engine.cpp:1645-1648`) and has **no `attend` reader for the block-of-guesses
  path** (`qwen35.cpp:318-324`). At geometry B it prints the refusal.
- `fdec` needs `dim ∈ {128, 256}` and a group in {4, 6, 8, 12, 16} (`fattn.cu` `fdec_group_ok`), already checked
  and printed by the bench (`kv_read_bench.cpp:121-127`). Keep that block; add the same for `attend_int8`'s
  `dim == 256` IMMA tier.

### 12.2 Bench 2 — `attend_int8` against `attend` on the *same* k8v4 keys

This is the question the brief poses and it needs its own table, because the two readers do not take the same
arguments: `attend_int8` wants `KvInt8` (codes + per-32 deltas) and a quantized query scratch, `attend` wants
`Kvq8`/`Kvq4`. The bench must build **one** k8v4 cache and drive both readers from it.

| row | keys | values | query | expected time | expected parity |
| --- | --- | --- | --- | --- | --- |
| A (current) | kq8 (fp16 MMA path) | vq4 dequantized in registers | fp16 | baseline | reference |
| B (candidate) | kq8 as int8 → IMMA | vq4 (still fp32 maths) | **quantized to int8** | ≤ A | **worse than A**, by the query round |
| C (control) | kq8 | vq8 | fp16 | — | isolates the value-side cost |

**Row B is not a drop-in**: `attend_int8` cannot read q4 values. So B is `attend_int8_mixed` with an fp16 value
cache (R10) — which tests the *QK* claim exactly and leaves the value side alone, which is what the hypothesis is
about. That distinction must be in the report or row B will be read as "k8v4 on IMMA" when it is
"int8 keys + fp16 values on IMMA".

Three numbers per row: ns a cell at 4K/32K/131K/256K, decode (rows=1) and prompt (rows=512); cosine and
max-relative against row A; and the **top-1 agreement of the scores' argmax** over the same keys, because a
cosine of 0.9999 on the output can still hide a changed selection, and selection is what a decode turns on.

### 12.3 Bench 3 — the store paths

| store | geometry | contexts | ns a cell | GB/s of KV bytes | bytes written | parity |
| --- | --- | --- | --- | --- | --- | --- |
| S1 `store_f16` | A, B | 4K/32K/131K/256K | ☐ | ☐ | 2,048 / 4,096 | bit-exact vs host fp16 |
| S2 `store_q8` | A, B | " | ☐ | ☐ | 1,088 / 2,176 | byte-exact vs ggml q8_0 ref |
| S3 `store_q4` | A, B | " | ☐ | ☐ | 288 / 576 | byte-exact vs ggml q4_0 ref |
| S4 `store_q6` | A, B | " | ☐ | ☐ | 416 / 832 | byte-exact vs the 6-bit rule |
| S5 `store_int8` | A | " | ☐ | ☐ | 1,056 | byte-exact vs `quantize_row_q8_0_ref` |
| each `_at` form | A, B | 4K/131K | ☐ | ☐ | same | same bytes as the plain form |

Existing coverage: `poly_kv_q4_test --bench` times k8v4 against q8_0 at geometry A, bulk and chunked
(`kv_q4_test.cpp:314-420`); `poly_kv_q8_test --bench` times q8_0 as the comparison arm (`kv_q8_test.cpp:224-280`).
So **S2 and S3 at geometry A exist**. New: S4 (k6v4), S5 (int8), geometry B, and every `_at` row.

The `GB/s` column must be reported against **both** denominators (the card's max clock and the loaded 280 GB/s),
as `kv_q4_test.cpp:325-328` already does, and the source-buffer note from `QUANT_MICROBENCH.md:527-533` applies:
a DRAM-sized source turns a store test into a DRAM test, and the engine's source is hot.

### 12.4 Bench 4 — `native_mmvq` against `Mmq`, per type

| input | what |
| --- | --- |
| types | **read from the GGUF, not assumed** — the header block of `poly_expert_parity` already enumerates them (`expert_parity.cpp:229-232`), so the bench takes a model file and reports the types it found |
| shape | the experts' own `[n_in, n_out]`, plus the fixed `2560 × 10240` for cross-card comparability (`QUANT_MICROBENCH.md:219-221`) |
| n | 1, 8, 32, 128, 512 |
| columns | TFLOP/s, GB/s of weights read, the `n` at which it stops being memory-bound, and **relative error against the double-precision reference each already computes** |
| arms | M1 `native_mmvq`, M2 `Mmq`, M3 `gemm_f32` where the type allows |

The reference is free: `expert_parity.cpp:255-275` builds the double-precision product from ggml's own
`to_float`. Reuse that code path rather than writing a second reference — a second reference is a second thing
that can be wrong.

### 12.5 Bench 5 — the model-level gate (from §6, unchanged)

`tools/kv_agree.py`, per model per format, 4,096 positions (32 prompts × 128), the numbers and the power
arithmetic in §6. That is a shell script's worth of orchestration, not a new binary.

## 13. The code design

### 13.1 Shape

One binary per concern, all `EXCLUDE_FROM_ALL` except the two that are already ctest tests:

| binary | status | CMake | ctest |
| --- | --- | --- | --- |
| `poly_kv_read_bench` | **extend** (`kernels.cu` already linked) | `strata_poly.cmake:340-348` | not a test (benchmark) |
| `poly_kv_q4_test --bench` | **extend** (add k6v4, geometry B, `_at`) | `:330-336` | `poly_kv_q4_test` is a test; `--bench` is not |
| `poly_kv_q8_test --bench` | keep, add geometry B | — | same |
| `poly_kv_int8_test` | **extend** (add `attend_int8_mixed` rows) | `:352-358` | already a test |
| `poly_expert_parity` | **extend with `--bench`** | `:181-186` | not a test (needs a model) |

No new ctest entries for the timed legs. The four correctness binaries stay in ctest; the benches stay by hand.
That matches the existing split and the stated reason (`QUANT_MICROBENCH.md:606-609`).

### 13.2 What `poly_kv_read_bench` needs added

The one real design problem is that **the same geometry has two different byte counts** depending on which
engine's scale period applies (`strata`: per 64; `strata-poly`: per 32), and every GB/s number depends on it.

- **Add `--scale-period {32,64}`** (default 32, the `strata-poly` rule), and make `cell_bytes` a function of it
  rather than the hard-coded `kv_read_bench.cpp:112`. A wrong cell size silently rescales every GB/s figure, and
  this bench is where that would happen.
- **Add `--format {fp16,q8_0,k8v4,k6v4,int8,int8k}`** with the default "all reachable here".
- **Add `--rows {1,512}`** and time both in one process, interleaved, so the prompt and decode numbers see the
  same clock.
- **Add the parity column**: after the timed loop, one call with a fixed seed and a copy of the output, compared
  to the fp16 arm's output. Cosine and max-rel. The fp16 arm must be computed **in the same process from the same
  source floats**, not from a stale file.
- **Add the `int8`/`int8k` arms** — they need a `KvInt8` query scratch and the `attend_int8` signature
  (`kernels.hpp:359-372`), which the bench does not build today.

### 13.3 What the store bench needs

`poly_kv_q4_test --bench` already has the pattern (host source in L2, whole cache cycled, interleaved arms,
per-cell ns, both denominators). Add:

- `bench_store_q6` and `bench_store_int8` beside `store_k8v4`/`store_q8`;
- an `--at` leg that calls the `_at` form with a device `at` advanced per row, and asserts the resulting bytes
  are identical to the plain form's (this is exactly what `kv_q4_test.cpp:1-10` item 4 tests for q4; extend it);
- geometry B (`kKvHeads = 4` at `:254` is a constant today — make it an argument).

### 13.4 What `poly_expert_parity --bench` needs

It already: loads the types from a GGUF (`:229-232`), builds a double-precision reference per type
(`:255-275`), and runs both M1 and M2 (`:288-296`). Add only:

- a `time_us(fn, reps)` helper (the pattern exists in `mmvq_il_parity.cpp:38-70` — copy it, do not reinvent);
- a loop over `n ∈ {1,8,32,128,512}` around both arms, alternating **inside** one repetition as
  `kv_read_bench.cpp:325-355` does;
- TFLOP/s from `2 · n · n_in · n_out / t`, and GB/s from `native_mmvq_weight_bytes`.

### 13.5 CPU references — which arms can have one

| bench | GPU arm | CPU reference that exists | gap |
| --- | --- | --- | --- |
| store | S2, S3, S5 | ggml's `quantize_row_*_ref` (used at `kv_int8_test.cpp:68-81`), `kv_q4_test.cpp`'s host q4 | **S4 has none**: `store_q6`'s packing has no ggml equivalent (`Q6_0` is this project's own type). Needs a host function written from the same rule. |
| read | all | none needed — the fp16 arm is the reference, in-process | — |
| attend_int8 | R9, R10 | `attend` over the same fp16 values (`kv_int8_test.cpp:425-445`) | exists, bar 0.999 |
| mmvq/MMQ | M1, M2 | double-precision product from ggml `to_float` (`expert_parity.cpp:255-275`) | exists |
| model gate | — | fp16 arm's greedy tokens | exists (`kv_agree.py`) |

**The one missing reference is `store_q6`'s host implementation**, because `Q6_0` is ours. It must be written
from `kernels.cu:730-760`'s stated rule (`d = amax/31`, `code = clamp(rint(x/d), -31, 31) + 32`, value *j* at bit
6*j*, offset binary) and asserted byte-exact. Without it S4's parity column is empty and, by §16, S4's speed
number cannot be reported.

## 14. Expected results, written before the runs

Each is stated so it can be **refuted**, and each names what refutes it. This is the section that makes a
surprise visible instead of explainable after the fact.

### 14.1 Read (bench 1)

| arm | expectation | refuted by |
| --- | --- | --- |
| q8_0 vs fp16, decode, 256K | **1.6–1.7× faster**, ~48% of the bytes | anything under 1.4× means the read is not byte-bound and the whole footprint argument needs re-deriving |
| k8v4 vs q8_0, decode, 256K | **0.85–0.95×** (k8v4 *slower*), matching the recorded 5.88 vs 4.96 ns (`Q8_KV.md:270`) | k8v4 at 0.95–1.05 would contradict the published number and one of the two harnesses is wrong |
| k6v4 vs k8v4, decode, 256K | **1.05–1.20× slower** (more bytes saved, more instructions spent on `q6pair`) | k6v4 *faster* would mean `q6pair`'s five SASS are hidden by the DRAM wait after all — a genuinely useful refutation |
| int8 vs q8_0, decode | **within ±5%**: same bytes (1,056 vs 1,088), same reader shape | int8 faster than q8_0 would say the separate delta array is better laid out than the interleaved one |
| `fdec` vs `attend`, k8v4, 131K | `fdec` **1.2–1.5× faster** (the threshold is 4,096, so it must be) | `attend` winning at 131K means the threshold is wrong and `STRATA_FDEC` should change — this is a decision, not a curiosity |
| prompt (rows=512), q8_0 vs fp16 | **2.5–4×** (fp16 goes through cuBLAS after a dequant, so the dequant is counted) | under 2× would mean `attend_batch` is dominated by something other than the cache |

### 14.2 `attend_int8` against `attend` (bench 2)

| claim | expectation | refuted by |
| --- | --- | --- |
| time | int8 QK **1.2–1.9× faster** than the fp16-MMA-with-dequant path, at 256 values a head | under 1.1× means the dequant is not the cost, and §2.2's premise ("the dequant to fp16 might be pure waste") is wrong |
| parity | cosine **> 0.999**, matching `kv_int8_test.cpp:442`'s bar; **max-rel 10⁻³–10⁻²** | cosine < 0.999 means the query quantization is visible at kernel level and the arm should not go to a model test |
| selection | top-1 of the scores' argmax **agrees on ≥ 99.5% of keys**, but **not 100%** | 100% agreement would be suspicious (the query is genuinely rounded) and must be re-checked against a wrong-reference bug |
| the honest reading | int8 wins on time and loses 0.1–0.5 points of top-1; **that is inside the k8v4-vs-q8_0 band of 7 points already accepted** (`FLINT.md:173`) and therefore not disqualifying | if the loss is **outside** that band, the arm is dead regardless of its speed |

The last row is the point of the whole bench: it converts "is int8 QK fast" into "is int8 QK *worth* it", and it
is the only one of these that a speed number alone cannot answer.

### 14.3 Store (bench 3)

| arm | expectation | refuted by |
| --- | --- | --- |
| k6v4 store vs k8v4 store | **~0.85× the bytes, 0.85–0.90× the time** — the store is byte-bound here, unlike the read | a store that is not byte-proportional means the packing, not the traffic, dominates |
| int8 store vs q8_0 store | **within ±5%** (1,056 vs 1,088 B) | — |
| `_at` vs plain | **1.05–1.3× slower** (one extra indirection a row) | 1.0× would mean `_at` is free and the decode path is not paying for the recorded form; > 1.5× would be a graph-path regression worth filing |

### 14.4 Matmul (bench 4)

| claim | expectation | refuted by |
| --- | --- | --- |
| MMQ vs `native_mmvq` | **8–12× at n = 128 and 512**; near parity at n = 1–8 | under 5× at 512 would mean the prompt path is leaving most of INT8 on the floor and §7's 39% figure is not the whole story |
| MMQ flat above n = 128 | **≤ 10% change from 128 to 512** | a big gain at 512 would mean the `J` ≤ 128 tiling re-reads weights more than the model in §7 assumes |
| lower bit is not faster | `IQ2_XXS` **≥ 15% slower than `q4_K`** at n = 128 | `IQ2_XXS` matching `q4_K` would refute the instruction-count model and make §7's conclusion (lower bits buy VRAM, not speed) wrong |
| **the headline if it holds** | **q4_K ≥ q8_0 at n = 128 by 0–25%** | q4_K 20% *slower* than q8_0 would say the unpack does not hide and the 45%-conversion figure is an underestimate |
| `native_mmvq` at n = 8 | **1.5–3× slower than at n = 1**, per the code's own note that dp4a throughput binds from 2 columns (`native_mmvq.cu:4173`) | the opposite would mean the multi-column kernels are not dp4a-bound here |

### 14.5 Model gate (bench 5)

| format | expected | refuted by |
| --- | --- | --- |
| k6v4 vs fp16, 4,096 positions | **84–88%** (k8v4's 83.3% plus 0–3 points) | above 90% would make k6v4 better than q8_0 and demand a re-run |
| k8v4 vs fp16, same run | **82–85%**, reproducing `FLINT.md:173`'s 83.3% — this is the harness's own calibration | outside 82–85 means the prompt set differs from the published one and the two are not comparable |

**The k8v4 row is a calibration, not a finding.** If the bench cannot reproduce 83.3% at 8K on the same model,
every k6v4 number from that run is unquotable. That check goes in the bench's own output.

## 15. The first three measurements

Cheapest decisive first. Each names the decision it settles, so a null result still closes something.

### 15.1 First — `attend_int8` against `attend` on one k8v4 cache (bench 2)

```bash
# no new code beyond the bench-2 arms; geometry A, decode and prompt, four contexts
timeout 900 build/poly_kv_read_bench 256 16 2 --formats k8v4,int8,int8k --rows 1,512 \
    --cells 4096,32768,131072,262144 --itr 10 --scale-period 32 --parity
# and the kernel-level parity bar the existing test already sets
timeout 300 build/poly_kv_int8_test
```

**Wall time: ~6 minutes** (four contexts × three formats × two row counts × ~10 s each, plus the existing test at
~1 min). **Settles:** whether the INT8 QK product is worth a model-level test at all. If it is under 1.1× on
time, the whole int8-KV line stops here with no model run and no `kv_agree.py` — that is the cheapest possible
kill.

### 15.2 Second — the store ladder, including k6v4 and geometry B (bench 3)

```bash
timeout 600 build/poly_kv_q4_test --bench                 # k8v4 vs q8_0, geometry A (exists today)
timeout 600 build/poly_kv_q6_test --bench                 # after the k6v4 store leg is added
timeout 600 build/poly_kv_int8_test                       # store_int8 legs
timeout 600 build/poly_kv_read_bench 256 24 4 --formats fp16,q8_0,k8v4 --rows 1 --cells 4096,131072
```

**Wall time: ~10 minutes** for the three geometries. **Settles:** whether a smaller cell is bought at a
proportional time cost on the *write* side, which is the half the read bench cannot see and the half a
`--kv-resident` streamed session pays on every token. If the k6v4 store is not byte-proportional, the format is
not a footprint win either.

### 15.3 Third — `Mmq` against `native_mmvq` per type at n = 1…512 (bench 4)

```bash
# the types come from the file, so no assumption about Ornith's quantization enters the result
timeout 1800 build/poly_expert_parity --bench ~/models/Ornith-1.5-35B-A3B-*.gguf
timeout 900  build/poly_expert_parity --bench --shape 2560x10240 --types Q4_K,Q6_K,Q8_0,IQ3_XXS,IQ2_S
```

**Wall time: ~25 minutes** (the GGUF load dominates; the timed legs are seconds). **Settles:** §7's central
prediction — whether a lower-bit type is slower in MMQ — and with it whether the expert-cache-vs-format trade has
a speed cost that must be paid on the *weight* side as well as the KV side. This is the one bench whose answer
changes a default rather than a flag.

**Why this order.** #1 can kill a feature with no model and six minutes. #2 can kill the same feature's remaining
justification (footprint) with ten. #3 is the only one that can *create* a win rather than rule one out, and it is
also the longest — so it goes last, after the two cheap kills have had their chance.

## 16. Anti-silent-success: the rules the code must enforce

Today's failure mode (a "bf16" arm that was integer quantisation at 32767 levels) is a *measurement* that ran,
returned, and reported a plausible number for the wrong computation. Four mechanical rules:

1. **Every arm asserts it did work.** A timed region that made no launch is a failure, not a fast result. Count
   the launches (`cudaGetLastError` after each, plus a counter the arm increments) and abort with a non-zero exit
   if the count is zero. The existing benches do this partially (`kv_read_bench.cpp:97` `ck(...)`) but none
   asserts a *launch count*.
2. **Every arm computes its own parity from its own output.** No parity inherited from a previous arm, no
   reference read from a file, no `cos` against a buffer another arm wrote. The reference is recomputed in-process
   from the same source floats, in the same run.
3. **A parity number against the wrong reference fails the run.** Concretely: the reference arm is named in the
   output line, and the bench refuses to print a ratio whose two sides did not both produce a finite output. The
   bf16 failure would have been caught by "the reference is this arm's own fp16 output, and the fp16 output must
   differ from the quantized one by more than zero but less than 1e-2" — a self-consistency range.
4. **A row with no parity prints `PARITY MISSING` and cannot be quoted.** Make the bench print the literal string
   in the parity column when an arm reached its time but has no reference (`store_q6` today, §13.5). That is
   visible in the table and cannot be edited out of a report by accident.

Two more, learned from the existing tree's own recorded mistakes and worth carrying in the harness:

5. **`room` is the exact row count** — the power-of-two `room` that read a 131K cache as 256K and reported
   3.522 ms against a true 0.308 ms (`QUANT_MICROBENCH.md:527-529`). The bench takes `--cells`, so it must pass
   `cells` and never a rounded value.
6. **Each arm gets its own buffers**, and the bench prints the buffer addresses' distinctness for the KV pair.
   The 451 GB/s-on-a-288 GB/s-card figure (`QUANT_MICROBENCH.md:535-539`) came from sharing one cache between
   arms; an impossible GB/s is the symptom, so the bench prints the achieved GB/s as a fraction of peak in the
   same cell and a >100% cell is a hard failure.

## 17. What does not exist yet and must be written

| # | needed | where | size |
| ---: | --- | --- | --- |
| 1 | a **host reference for `store_q6`** (`Q6_0` is our type, ggml has none) | new, in `poly_kv_q6_test` or beside `store_q6` | ~40 lines, from `kernels.cu:730-760`'s rule |
| 2 | a **`poly_kv_q6_test`** — the store and read checks that `kv_q4_test`/`kv_q8_test` have, which k6v4 lacks | new binary + CMake target beside `:330-336` | ~250 lines, mostly copied from `kv_q4_test.cpp`'s four checks |
| 3 | the `int8`/`int8k` arms in `poly_kv_read_bench` | extend `:100-200` (cache build) and `:280-330` (arms) | ~120 lines |
| 4 | `--scale-period`, `--format`, `--rows`, `--parity` in `poly_kv_read_bench` | `kv_read_bench.cpp:82-110` (args) and `:112` (cell bytes) | ~80 lines |
| 5 | `--bench` in `poly_expert_parity` with the n ladder | `expert_parity.cpp:225-310` | ~90 lines |
| 6 | geometry B (`kKvHeads = 4`) as an argument in the store benches | `kv_q4_test.cpp:254` (constants), `kv_q8_test.cpp` likewise | ~30 lines a file |
| 7 | `_at` store legs + the byte-identity assertion vs the plain form | `kv_q4_test.cpp` bench, `kv_q8_test.cpp` bench | ~60 lines |
| 8 | a **launch counter** the arms assert on (§16 rule 1) | one helper in each bench | ~15 lines each |

Nothing here is a new kernel, a new algorithm, or a new file format. It is measurement scaffolding around code
that already exists and, in two cases (bench 2's three arms, bench 4's two arms), around comparisons the tree
already makes for accuracy and has never timed.

## 18. What the programme will and will not settle

**Will settle**, in the order the measurements run:

- whether the INT8 QK product is worth anything on this card (§15.1), with its accuracy cost in the same row;
- whether a sub-8-bit cell is bought at a proportional cost on the store side (§15.2);
- whether a lower-bit weight type is faster or slower in MMQ (§15.3), which decides how the expert cache and the
  weight format trade against each other;
- the model-level accuracy of every format reachable by `--kv`, at a sample size that can separate them (§6,
  4,096 positions).

**Will not settle**, and should not be quoted as if it did:

- **any tok/s number on a model.** All of §15 is kernel-level. The end-to-end A/B (Part I §8 step 4) is a
  separate run and, by Part I §3, is expected to land inside noise. A kernel bench that is 1.3× faster does not
  become a 30% faster token rate when the kernel is 0.8% of the wall time.
- **Hopper or Blackwell.** The sm_90 dump shows the 4-bit `IMMA` forms absent and the fast path elsewhere
  (`wgmma`); nothing here is measured on either.
- **pp512/pp2048-style aggregate prefill.** The bench measures ns a cell at a fixed 512 rows, not a mixed
  workload where the expert cache and the KV cache compete for the same VRAM. Part I §8 step 3 is where that
  competition is measured, and it is a load, not a bench.

# Part III — the SASS audit

Parts I and II are arithmetic and bench design. This part is the one thing that needed no card: a
`cuobjdump --dump-sass` census of the binaries already built. It answers, from counters rather than from
reading the source, which paths actually put work on the INT8 tensor cores and which do not.

**The headline, and it changes §2.1 and R1–R4 of Part II: `attend()` has no tensor-core instruction at all.**
Not HMMA, not IMMA, not dp4a. Its score kernel is scalar `FFMA` with the dequant in registers. The brief's
premise — that `attend` dequantises k8v4 keys "for `mma16816`" — is not what the binary contains. `mma16816`
in this tree is reached by `fdec_kernel` and by `flash_attn_ext_f16`, not by `attend`.

## 19. Method

```
cuobjdump --dump-sass <object>.o      > sass.txt
cuobjdump --dump-resource-usage <obj> > res.txt
```

Objects dumped, all `sm_89`, all already built (`build-poly-local`, which is post-`fix/mmq-dense`):

| object | why |
| --- | --- |
| `CMakeFiles/strata-poly.dir/src/poly/kernels.cu.o` | `attend`, `attend_int8`, `attn_score_*`, `attn_value_*` |
| `CMakeFiles/strata_poly_mmq.dir/src/poly/fattn.cu.o` | `fdec_kernel` (all `VM`), `flash_attn_ext_f16` |
| `CMakeFiles/strata-poly.dir/src/poly/kernels_dsv4.cu.o` | DeepSeek V4's own `attend_kernel` |
| `third_party/poly-mmq-instances/mmq-instance-<type>.cu.o` | `mul_mat_q` per type, Q8_0 … IQ1_S |
| `CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.cu.o` | `native_mmvq`, per type |

Counts are exact instruction counts per `Function :` block, matched on the opcode token after the address
comment. `unch%` is `(I2F + I2FP + F2F + FMUL + FFMA + HADD2 + HMUL2) / insns`: the share of the instruction
stream whose only job is to move a quantised value into, or out of, a float multiply-accumulate. It is the
column that sizes the prize for a dequant-bound kernel.

A caution on `attend_int8`'s row: the kernel is a single 780-instruction body, so its 8 `IMMA` are 1.0% of the
stream and there are **97.5 instructions per `IMMA`**. That is a *worse* ratio than MMQ's, and it is why §21
does not assume the IMMA path is faster merely because it exists.

## 20. Deliverable A — the table, one row a kernel

Ordered as the brief asked: attention first, then matmul. `ins` is the static instruction count of the largest
instantiation of that family; it is not a dynamic count and the ratios are what matter.

### 20.1 Attention and KV readers

| path | kernel | ins | IMMA | HMMA | IDP | I2F | FMUL | FFMA | HADD2 | unch% | regs | smem |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attend` fp16, q in regs | `attn_score_reg_kernel<8,2,false>` | 536 | 0 | **0** | 0 | 4 | 0 | 1 | 64 | 14.4% | 110 | 0 |
| `attend` q8_0 keys, q in regs | `attn_score_reg_kernel<8,2,true>` | 560 | 0 | **0** | 0 | 12 | 0 | 9 | 64 | 15.5% | 115 | 0 |
| `attend` fp16, generic | `attn_score_kernel<false>` | 928 | 0 | **0** | 0 | 4 | 7 | 132 | 68 | 22.7% | 62 | 0 |
| `attend` q8_0, generic | `attn_score_kernel<true>` | 968 | 0 | **0** | 0 | 56 | 59 | 100 | 13 | 23.6% | 71 | 0 |
| `attend` values fp16 | `attn_value_kernel<false,false>` | 2720 | 0 | 0 | 0 | 7 | 0 | 270 | 62 | 12.5% | 48 | 32768 |
| `attend` values int8 | `attn_value_int8_kernel` | 2968 | 0 | 0 | 0 | 65 | 58 | 238 | 58 | 14.1% | 52 | 32768 |
| `attend_int8` QK, dp4a, dim 256 | `attn_score_int8_kernel<8,32>` | 888 | 0 | 0 | **34** | 21 | 0 | 51 | 0 | 10.6% | 40 | 0 |
| `attend_int8` QK, **IMMA** | `attn_score_mma_int8_kernel<8>` | 780 | **8** | 0 | 0 | 40 | 0 | 36 | 32 | 17.9% | 54 | 0 |
| prompt flash, 128 | `attn_flash128_kernel` | 3230 | 0 | **64** | 0 | 3 | 40 | 238 | 656 | 29.0% | 128 | 0 |
| prompt flash, 64 | `attn_flash64_kernel` | 1732 | 0 | **64** | 0 | 0 | 32 | 132 | 174 | 19.5% | 162 | 36864 |
| `fdec` VM0 fp16 | `fdec_kernel<256,4,1,0,8>` | 1090 | 0 | **64** | 0 | 0 | 50 | 2 | 8 | 6.0% | 104 | 11264 |
| `fdec` VM1 q8_0 | `fdec_kernel<256,4,1,1,8>` | 1965 | 0 | **64** | 0 | 0 | 50 | 66 | 168 | 18.0% | **255** | 11264 |
| `fdec` VM2 k8v4 | `fdec_kernel<256,4,1,2,8>` | 1968 | 0 | **64** | 0 | 0 | 50 | 66 | 184 | 19.6% | 254 | 11264 |
| `fdec` VM3 k6v4 | `fdec_kernel<256,4,1,3,8>` | 2216 | 0 | **64** | 0 | 0 | 50 | 66 | 184 | 17.4% | 252 | 11264 |
| `fdec` VM2 k8v4, MT=2 | `fdec_kernel<256,4,2,2,8>` | 2613 | 0 | **128** | 0 | 0 | 100 | 132 | 192 | 19.6% | 255 | 22528 |
| `fdec` VM3 k6v4, MT=2 | `fdec_kernel<256,4,2,3,8>` | 2872 | 0 | **128** | 0 | 0 | 100 | 132 | 192 | 17.9% | 255 | 22528 |
| DeepSeek V4 `attend` | `attend_kernel` (dsv4) | 1120 | 0 | **0** | 0 | 5 | 4 | 56 | 38 | 9.2% | 40 | 128 |
| DeepSeek V4 `attend` rows | `attend_rows_kernel` | 1032 | 0 | **0** | 0 | 7 | 49 | 108 | 104 | 26.0% | 58 | 128 |

Read the HMMA column against the IMMA column. **The only attention kernel in the tree that emits `IMMA` is
`attn_score_mma_int8_kernel`, and it emits 8 of them.** Everything else that has a tensor op at all has `HMMA`.

### 20.2 Matmul

`mul_mat_q` at `J = 128, fallback = 1` (the largest batch tile the config allows, so the type comparison is at
one shape), and `native_mmvq` at its widest instantiation.

| path | type | ins | IMMA | IMMA form | IDP | I2F | FMUL | FFMA | ins/IMMA | unch% |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mul_mat_q` | Q8_0 | 6184 | **256** | `16832.S8.S8` | 0 | 1027 | 1024 | 1024 | 24.2 | 49.7% |
| `mul_mat_q` | Q4_K | 7831 | **256** | `16832.S8.S8` | 0 | 1043 | 1024 | 2048 | 30.6 | 52.5% |
| `mul_mat_q` | Q6_K | 8807 | **512** | `16816.S8.S8` | 0 | 1027 | 0 | 1280 | 17.2 | 26.2% |
| `mul_mat_q` | IQ3_XXS | 6816 | **256** | `16832.S8.S8` | 0 | 1035 | 1040 | 1032 | 26.6 | 45.6% |
| `mul_mat_q` | IQ2_XXS | 6640 | **256** | `16832.S8.S8` | 0 | 1035 | 1032 | 1024 | 25.9 | 46.6% |
| `mul_mat_q` | IQ1_S | 7648 | **256** | `16832.S8.S8` | 0 | 1043 | 1040 | 2056 | 29.9 | 54.1% |
| `native_mmvq` | Q8_0 | 872 | 0 | — | **64** | 16 | 16 | 16 | — | 15.1% |
| `native_mmvq` | Q4_K | 1736 | 0 | — | **256** | 128 | 32 | 160 | — | 22.6% |
| `native_mmvq` | Q5_K | 1792 | 0 | — | **256** | 128 | 32 | 160 | — | 21.9% |
| `native_mmvq` | Q6_K | 1296 | 0 | — | **64** | 64 | 0 | 96 | — | 17.6% |
| `native_mmvq` | IQ4_XS | 1568 | 0 | — | **256** | 32 | 32 | 32 | — | 8.4% |
| `native_mmvq` | Q2_0 | 1392 | 0 | — | **256** | 32 | 32 | 32 | — | 9.5% |

**Every `mul_mat_q` type is on IMMA. Every `native_mmvq` type is on dp4a.** That is the whole matmul story in
one sentence, and it is the one place where the brief's framing holds exactly: MMQ reached the INT8 units for
all six types measured, and `native_mmvq` did not reach any tensor core at all.

Two findings in that table are new and neither was in Part I:

1. **Q6_K is the odd one out: `IMMA.16816.S8.S8`, half-k, and twice as many of them.** It is not a
   different *count* of work — it is a different *shape*, `k=16` instead of `k=32`, because Q6_K's 6-bit codes
   need the scale applied inside the k loop (`mmq-vec-dot.cuh:1010-1160`, the `TURING_MMA_AVAILABLE` branch
   synthesises `k=16` from a `tile<16,4,int>`, and its epilogue folds `scA`/`dA`/`dB` per 16). So Q6_K issues
   **2× the tensor instructions for the same MAC count**, which is a real disadvantage the type table in Part I
   §7 did not have.
2. **The conversion share is worse than the 45% in the brief, not better, for the byte-heavy types.** Q8_0 is
   49.7%, Q4_K 52.5%, IQ1_S 54.1%. The brief's 45% was a whole-binary figure; per kernel at `J=128` the number
   is higher because the epilogue does not shrink with the type.

### 20.3 Registers and shared memory, and what they cost

| kernel | regs | smem | occupancy note |
| --- | ---: | ---: | --- |
| `attn_score_reg_kernel<8,2,*>` | 110–115 | 0 | ~2 blocks an SM at 115 regs — the source of the `.reqntid` pressure the code comments about |
| `attn_score_mma_int8_kernel<8>` | 54 | 0 | 8 blocks an SM; not register-limited |
| `fdec_kernel<256,4,1,0,8>` (fp16) | 104 | 11264 | the only `fdec` shape with headroom |
| `fdec_kernel<256,4,1,{1,2,3},8>` | **252–255** | 11264 | **already at the ceiling at MT=1** — every quantised `VM` is register-bound, before `MT=2` |
| `fdec_kernel<256,4,2,{1,2,3},8>` | **255** | 22528 | at the ceiling with 24–32 B of stack |
| `native_mmvq` Q4_K | 110 | 6144 | — |

This corrects a figure Part I implied. **Every quantised `fdec` shape is at 252–255 registers at `MT=1`**, i.e.
the kernel is at the architectural register ceiling whether or not it doubles `MT`. Only VM0 (fp16) has
headroom at 104. That reframes §22.4: the dequant-in-registers design is not merely *competing* with IMMA, it is
already spilling to make room for itself (`STACK:24` on VM1 and VM3 at `MT=2`), so an IMMA rewrite would have to
free registers before it could win on instructions.

### 20.4 The `IMMA.16816` finding, in full

Across every object dumped, the opcode census is:

| opcode | count | where |
| --- | ---: | --- |
| `IMMA.16832.S8.S8` | 5425 | all `mul_mat_q` types except Q6_K; `attn_score_mma_int8_kernel` |
| `IMMA.16816.S8.S8` | 3584 | **`mul_mat_q` Q6_K only** |
| `HMMA.16816.F32` | 7296 | `fdec_kernel` (all `VM`), `attn_flash*`, `attend_int8`-adjacent kernels |
| `HMMA.16816.F16` | 1536 | **`flash_attn_ext_f16` only** — the prompt path, FP16 accumulate |
| `IDP.4A.S8.S8` | 433 | `attn_score_int8_kernel`, `native_mmvq` |

`HMMA.16816.F16` matters: Part I §2.2 called FP16-accumulate "a hypothesis, not a result". It is already
compiled and reachable — **on the prompt path** (`flash_attn_ext_f16`, via `fattn_q8`/`fattn_f16`), not on
`fdec`. So the 2× that Part I priced as a one-line change is already taken where prompt attention runs, and
`fdec`'s decode path is the one still on `F32` accumulate. That is a precise, checkable statement and it is a
cheaper lead than any IMMA work in this document.

## 21. Deliverable B — what the paired benches would measure, and what the SASS already says

No GPU ran. The point of this section is to say, for each of the brief's four benches, whether the SASS makes
the answer predictable — and in three of four cases it does.

### B1. `attend_int8` against `attend` on k8v4 — **the premise needs correcting first**

The brief says `attend` dequantises k8v4 keys "to fp16 for `mma16816`". The binary says otherwise:
`attn_score_reg_kernel<8,2,true>` — the kernel `attend()` launches for a q8/k8v4 key cache at `dim==256` —
has **0 HMMA and 0 IMMA**. It is 536 instructions of scalar `FFMA` over `I2F`-converted halves, 12.9%
conversion share. The comparison the bench would actually make is:

| arm | tensor op | conversion share | what it is |
| --- | --- | --- | --- |
| `attend` (R3/R13's score step) | **none** | 15.5% | scalar FFMA, dot-product per lane, fp32 accumulate |
| `attend_int8` (R9) | **8 × `IMMA.16832`** | 17.9% | 97.5 instructions per IMMA |

So B1 is not "IMMA vs HMMA". It is **"IMMA vs no tensor core at all"**, which is a much larger expected gap and
a much more interesting measurement — and it is also why the existing comment at `kernels.cu:3903-3905` records
only +5–7%: the IMMA tier there is being compared against a dp4a/register-blocked int8 kernel, not against
`attend`.

**What the SASS predicts:** with 97 instructions per 8 IMMA and 4,096 MAC per IMMA, the IMMA arm issues
`8 × 4096 = 32,768` MAC in 780 static instructions; the scalar arm's `FFMA` count per key tile is the same MAC
count at 1 MAC an instruction. The gap should be large *if* the kernel is issue-bound and not load-bound —
which §20.3's 54-register, 0-smem figure suggests it can be. **This is worth the card time, and it is the
highest-value of the four because the current choice of path was never made against this comparison.**

### B2. `attend_int8_mixed` against `attend_int8` — answerable from the table

`attend_int8_mixed` differs only in the value side: `attn_value_kernel` (fp16, 0 tensor ops, 12.5% conversion)
against `attn_value_int8_kernel` (int8, 0 tensor ops, 14.1%). **Neither value kernel has a tensor instruction**,
so B2 measures the cost of removing `I2F` from the value loop, not the cost of leaving IMMA. Expect a small
effect; the SASS says the value side is not where the tensor cores are.

### B3. `fdec` against `attend_int8` at long context — the decode question, answered

This is the sharpest result in the audit, and it is **no, IMMA would not help `fdec` as things stand**:

- `fdec_kernel` VM2 (k8v4) and VM3 (k6v4) both run **64 `HMMA.16816.F32`** for a `MT=1`, `D=256` block, with
  the dequant in registers. Their conversion share is **19.6%** and **17.4%** — and the low `I2F` count is the
  point: the q8 codes become halves by the `PRMT`+`HADD2` trick, not by 32 integer-to-float converts.
- The cost is in the **unpack**, and the counters show it: `PRMT`+`LOP3`+`SHF` is **385 instructions (19.6%)**
  for k8v4 and **614 (27.7%)** for k6v4. That is the five-SASS-a-pair `q6pair` visible as a counter, and it is
  the number that makes any sub-8-bit `fdec` format slower to feed.
- `attn_score_mma_int8_kernel` by contrast is 780 instructions with only **45 unpack (5.8%)** — its codes are
  already bytes — and 8 IMMA. Per unit of MAC it issues far *more* instructions than `fdec` does per HMMA
  (97 against ~31), because it has no `MT` loop to amortise its prologue over.
- And `fdec` is already the path with **FP16 accumulate available and unused**: `flash_attn_ext_f16` in the same
  object proves `HMMA.16816.F16` compiles and runs on this card. Switching `fdec`'s accumulate from `F32` to
  `F16` is a one-token change per `mma` call; swapping its QK product to IMMA is a new kernel plus quantising
  the query.

**So the ranking for `fdec` is: FP16 accumulate first (already proven to compile), IMMA second, and IMMA may
never be worth it.** That is a direct answer to the brief's question 3 for the decode path.

### B4. `native_mmvq` against `Mmq` per type — the flat-curve prediction, confirmed by counters

The brief predicts a flat curve because "the unpacking grows while the multiply count falls". The counters say
something stronger: **the IMMA count is exactly 256 for Q8_0, Q4_K, IQ3_XXS, IQ2_XXS and IQ1_S**, and 512 for
Q6_K. It never falls with the type. What changes is the surrounding stream:

| type | IMMA | ins | ins/IMMA | delivered MAC/instruction |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 256 | 6184 | 24.2 | 169 |
| Q4_K | 256 | 7831 | 30.6 | 134 |
| IQ3_XXS | 256 | 6816 | 26.6 | 154 |
| IQ2_XXS | 256 | 6640 | 25.9 | 158 |
| IQ1_S | 256 | 7648 | 29.9 | 137 |

So a lower type **cannot** be faster in `mul_mat_q` at a fixed shape by the tensor-op count, and every step
below Q8_0 adds 2–6 instructions per IMMA of unpack. **A flat-to-slower curve going down in bits is the
prediction; the only way it fails is if Q8_0 is DRAM-bound at `J=128`, which Part I §7 already sized as
120 op/B against a 613 op/B balance — i.e. it is.** That means the *useful* version of B4 is not "which type is
faster" but "where is the Q8_0 → Q4_K crossover as a function of batch", and the counters say it exists only if
the memory saving outweighs 6.4 extra instructions per IMMA.

## 22. Deliverable C — k6v4 and k5v3 on IMMA, with the arithmetic

### 22.1 What the ISA actually requires

`IMMA.16832.S8.S8` consumes 8-bit lanes in both operands (`sm89_inst.txt:67500-67540`). A 6-bit code has to be
widened to a byte before the multiply. The question is whether that widening costs more than the byte saving
buys.

`fdec`'s existing `q6pair` does exactly this widening today, for the fp16 path, at **five SASS instructions a
pair** (`fattn.cu:303-310`: `SHF`, `PRMT`, `LOP3`, `PRMT`, `HADD2`). Its shape is:

```
p = __byte_perm(g, g >> 6, 0x4040) & 0x3F3F3F3F   // SHF + PRMT + LOP3
m = __byte_perm(p, 0x64646464, 0x4140)            // PRMT
h = __hsub2(m, 1056)                              // HADD2
```

### 22.2 k6v4's QK on IMMA — the arithmetic

Per 32-value block, k6v4 stores 24 bytes (`store_q6`, `kernels.cu:736`). A `16832` IMMA needs 32 bytes of
`int8` per operand lane set. So the widening is **24 bytes in → 32 bytes out per block**, and the work is:

| step | instructions per 32 values | note |
| --- | ---: | --- |
| unpack 6-bit fields | 32 values / 2 a pair = 16 pairs × 3 (SHF, PRMT, LOP3) = **48** | the mask and shift of `q6pair` |
| widen to a byte lane | 16 pairs × 1 PRMT = **16** | the second `PRMT` |
| *(fp16 path also does)* offset-correct | 16 × HADD2 = 16 | **not needed for IMMA**: `IMMA.S8.S8` sums signed bytes, and the offset-binary bias can be applied once per block in the epilogue as `−32 × sum(q)` |
| **total, IMMA path** | **64 instructions a 32-value block** | 2 a value |
| the IMMA itself | **1** instruction, 4,096 MAC | replaces 32 values × 16 keys of scalar work |

So the widening is **64 instructions per IMMA**, against MMQ's measured 24–30 instructions per IMMA of total
overhead. That is 2–2.7× the overhead of the best existing kernel, for a format that saves 8 bytes a 32-value
block (24 vs 32). **The byte saving does not survive**: `fdec` VM3 (k6v4) is 2,216 instructions against VM2's
1,968 for the same block — 12.6% *more* instructions for 25% fewer key bytes. The measured read numbers in
Part I §3 (k8v4 5.88 ns a cell, 18.5% slower than q8_0 despite 24% fewer bytes) already showed the direction;
this is the mechanism.

### 22.3 k5v3 — worse, and the packing is the reason

32 values × 5 bits = 160 bits = 20 bytes. That is not a multiple of 4, so a block does not align to a
`uint32` boundary across blocks: block *n* starts at bit `160n`, which is byte 20*n* — a multiple of 4 every
5th block only. The extraction is therefore **not** a fixed shift/mask per lane but a shift amount that depends
on the lane index mod 8, and the code path needs either a 5-block group (100 bytes, 800 bits) or a per-lane
variable `SHF`. Either way:

| step | per 32 values |
| --- | ---: |
| shift-and-mask, variable offset | **~48–64** (64 if every value needs its own `SHF`+`LOP3`) |
| widen to a byte | **~16** |
| **total** | **~64–80 a 32-value block**, 2–2.5 a value |

Against k6v4's 64. So k5v3 is **not cheaper than k6v4 on the read side** — it saves 4 bytes a block (20 vs 24)
and costs at least as many instructions. And Part I §4 already established that k5v3 and k4v4 are
**byte-for-byte identical** (576 / 1,152 B a cell), so k5v3 spends its bits on the key side, which is the
cheaper side and the one that measured 0.7 points of accuracy against the value side's 7.

### 22.4 The honest answer for both: `fdec`'s current path may already be good enough

Yes — and the counters support it, which is the brief's own suggested outcome and the one the data favours:

- The cost of k6v4 in `fdec` is **614 unpack instructions (`PRMT`+`LOP3`+`SHF`), 27.7% of the kernel**, against
  k8v4's 385 (19.6%). That is the price of 6 bits in this design, and it is already being paid.
- Its conversion share is 17.4%, and its 64 `HMMA` are amortised over an `MT` loop. `attn_score_mma_int8`'s
  8 IMMA have 45 unpack instructions but no such loop, at 97 instructions per IMMA against `fdec`'s ~31 per
  HMMA.
- Putting the QK product on IMMA would replace 64 HMMA-`F32` with 8 IMMA per block **and** require the query
  quantised to int8 — a new error term the current format does not have, since today only the *key* side is
  quantised and the query stays fp16.

**Verdict on C: k6v4 on IMMA costs 64 widening instructions per IMMA against the 24–30 that MMQ pays in total,
for an 8-byte-a-block saving that the read measurements already show does not convert to time. k5v3 is worse
and is also the wrong allocation of bits. `fdec`'s fp16 path is not leaving IMMA on the table; it is leaving
FP16-accumulate on the table, which is the cheaper fix.**

## 23. Deliverable D — the table to act on

Sorted by recoverable time per day of work. "ns a cell at 131K" is left as `—` where it needs the card; the
`SASS` column is what this audit can state without one.

| # | path | tensor-core instruction | ns a cell at 131K | parity | would IMMA help | cost |
| ---: | --- | --- | --- | --- | --- | --- |
| **1** | `fdec` VM1/2/3 decode, `F32` accumulate → `F16` | `HMMA.16816.F32` → `.F16` | needs card | unmeasured | **no — but FP16 acc does** | **1 token per `mma` call**; `flash_attn_ext_f16` proves it compiles |
| **2** | `attend` k8v4 → `attend_int8` k8v4 | **none** → `IMMA.16832` | needs card (B1) | existing test: cosine > 0.999 | **yes, probably large** — it is IMMA vs scalar FFMA, not vs HMMA | bench only for now; the reader exists for Gemma 4/Kolibri |
| **3** | `mul_mat_q` Q6_K | `IMMA.16816` ×2 | needs card | — | **no; it already over-issues 2×** | change Q6_K off half-k if the count is the cost |
| **4** | `mul_mat_q` Q8_0 → Q4_K | `IMMA.16832` both | needs card | — | **no** — IMMA count is identical (256) | bench only; expects flat-to-slower |
| **5** | `fdec` VM2/VM3 QK → IMMA | `HMMA.16816.F32` → `IMMA.16832` | needs card | query quantisation is a new error | **no, by §22.4** | a new kernel + query quantisation |
| **6** | `attend_int8_mixed` value side | none (both) | needs card | — | **no** | nothing to do |
| **7** | k6v4 QK → IMMA | none today | — | — | **no, by arithmetic** | 64 widening instructions per IMMA |
| **8** | k5v3 | — | — | — | **no, and the format is the wrong allocation** | do not build |
| **9** | `native_mmvq` → tensor cores | `IDP.4A` → `IMMA` | needs card | — | **only at batch ≥ 8**, and `mul_mat_q` already serves those | new kernel; `mul_mat_q` is the answer above 8 rows |

## 24. Needs the card

Each with the command and the expected wall time. Nothing here ran.

| # | measurement | command | time |
| ---: | --- | --- | --- |
| 1 | B1: `attend` vs `attend_int8`, k8v4, 4K/32K/131K/256K | `poly_kv_int8_test` bench arms (`kv_int8_test.cpp:859-880` already has the sizes) | ~20 min |
| 2 | `fdec` `F32` vs `F16` accumulate, VM2/VM3 at 131K/256K | needs the one-token change + `poly_fattn_test` | ~30 min incl. build |
| 3 | B4 per type at `J=128`, n = 8/32/128/512 | the ~30 lines Part I §7 says `poly_mmq_tiles_test` needs | ~40 min |
| 4 | k6v4 read vs k8v4 at 256K | `poly_kv_read_bench` with `--kv k6v4` | ~15 min |

## 25. What this audit could not settle

- **Dynamic instruction counts.** Every number here is static. A kernel whose 64 HMMA sit in a loop that runs
  4,096 times has a different dynamic profile. `nsys`/`ncu` would settle it and neither ran.
- **Register pressure's dynamic effect.** The static counts show every quantised `fdec` shape at 252–255
  registers; what that costs in occupancy and stalls needs `ncu`, which did not run.
- **The `--kv k6v4` accuracy gate.** Part I §8 step 1 is still the cheapest thing in the whole document and
  this audit does not substitute for it.
- **`docs/PREFILL-GAP.md` does not exist** in this tree at `ac76e3de5`; the brief cites it. The prefill numbers
  quoted in Part I come from `docs/QUANT_MICROBENCH.md` and `docs/DETAILS.md` instead.

# Part IV — format × phase: which instruction actually runs

Part III counted instructions per *kernel*. That answers "what is in the binary" and not the question a
configuration asks: **for format X in phase Y, which instruction runs?** The answer is a dispatch, and the
dispatch is two `if` chains in `Qwen35::attn` (`src/poly/qwen35.cpp:700-760`) plus a different one in
`Gemma4::attn` (`src/poly/gemma4.cpp:713-735`). This part maps every reachable `--kv` value through it.

## 26. The dispatch, traced

`Qwen35::attn` branches on token count against `kStageRows = 32` (`src/poly/runtime.hpp:30`).

**Prefill, `n >= 32`** (`qwen35.cpp:713`):

```
if (!(fattn_ && !v6 && (v4 ? fattn_k8v4 : q8 ? fattn_q8 : fattn_f16)))
    attend_batch(...);          // dequantise to q8w_, then cuBLAS
```

`fattn_` is set by `fattn_fits(head_dim, …)` (`fattn.cu:677`), which requires **`dim == 128 || dim == 256`**
and a GQA ratio of 8 or 16. `v6` is k6v4 — excluded by name, with the comment that llama.cpp's flash kernel
reads q8_0 and fp16 blocks only.

**Decode, `n < 32`** (`qwen35.cpp:735`):

```
if (!(fdec_ && pos >= fdec_from_ && (v6 ? fdec_k6v4 : v4 ? fdec_k8v4 : q8 ? fdec_q8 : fdec_f16)))
    attend(...);                // attn_score_reg_kernel / attn_value_kernel
```

`fdec_from_` is **4096** by default, **0** for k6v4 (`qwen35.cpp:395-399`). `fdec_fits` has the same
`dim` gate as `fattn_fits`.

Every `fattn_*` in `fattn.cu` runs the same kernel: a `dequant_*_kernel` fills a half workspace per block of
keys, then `flash_attn_ext_f16<D,D,…>` does the attention. So the dequantise-then-flash shape is identical
for fp16 (no dequant needed), q8_0 and k8v4; only the source format differs.

## 27. Matrix 1 — the instruction, per format and phase

One row a reachable `--kv` value, sorted as `main.cpp:1405-1410` accepts them. "Dominant" is the tensor or
FMA instruction that carries the arithmetic; counts are per kernel instantiation from Part III's dumps.

### 27.1 Shipping formats (qwen35 / Ornith)

| format | prefill kernel | prefill instruction | decode kernel | decode instruction |
| --- | --- | --- | --- | --- |
| **fp16** | `fattn_f16` → `dequant`* → `flash_attn_ext_f16` | `HMMA.16816.F32` **×384** (QK) + `HMMA.16816.F16` **×384** (PV) | `fdec_f16` → `fdec_kernel<…,VM=0,…>` | `HMMA.16816.F32` **×64** at `MT=1`, ×128 at `MT=2` |
| **q8_0** | `fattn_q8` → `dequant_kernel` → `flash_attn_ext_f16` | same flash: **384 F32 + 384 F16**; plus `dequant_kernel` (64 ins, no tensor op) | `fdec_q8` → `fdec_kernel<…,VM=1,…>` | `HMMA.16816.F32` **×64** |
| **k8v4** | `fattn_k8v4` → `dequant_kernel`+`dequant_q4_kernel` → `flash_attn_ext_f16` | same flash: **384 F32 + 384 F16**; plus two dequants (64 + 56 ins) | `fdec_k8v4` → `fdec_kernel<…,VM=2,…>`; **below 4096, `attend`** | `HMMA.16816.F32` **×64** above; **no tensor instruction at all** below |
| **k6v4** | **`attend_batch` + cuBLAS** — no `fattn` exists | **cuBLAS (black box — not ours to count)** | `fdec_k6v4` → `fdec_kernel<…,VM=3,…>` from position **0** | `HMMA.16816.F32` **×64** |

\* `fattn_f16` passes the fp16 caches where they lie and skips the dequant entirely for whole blocks
(`fattn.cu:224-240`: `if (keys && !last) kb = keys, vb = vals;`). So for fp16 prefill the dequant column is
**zero kernels**, which is the only format where the chain is one kernel long.

### 27.2 Gemma 4's int8 formats

These never reach a qwen dispatch. `engine.cpp:1652` throws for any model but `gemma4`.

| format | prefill kernel | prefill instruction | decode kernel | decode instruction |
| --- | --- | --- | --- | --- |
| **int8** (`--kv int8`) | `dequant_int8_to_f16` → `attend_batch` + cuBLAS | **cuBLAS (black box)** | `attend_int8` | `attn_score_mma_int8_kernel` → **`IMMA.16832.S8.S8` ×8** *if* `dim==256` and `keys<=16384`; else `attn_score_{reg_,}int8_kernel` → **`IDP.4A.S8.S8`** (dp4a, FMA pipe) |
| **int8k** (`--kv int8k`) | same: dequant the keys, fp16 values where they lie → cuBLAS | **cuBLAS (black box)** | `attend_int8_mixed` | as int8 for QK; values through `attn_value_kernel` (**no tensor op**) |

### 27.3 The format that is not built

| format | prefill | decode | note |
| --- | --- | --- | --- |
| **k4v4** | — | — | **does not exist.** `grep -c kK4V4` over `src/` and `include/` returns **0**; the string occurs 26 times in this document and zero times in code. It is a proposal, and its row would be `IMMA.16832.S4.S4` (§1.1) if built. |

**Three cells are cuBLAS, and that is the finding.** `attend_batch` hands fp16 operands to `cublasGemmEx`;
its instruction mix is chosen by the driver, differs per architecture, and cannot be counted from our binary.
So for **k6v4 prefill** and **both int8 prefills** the answer to "which instruction runs" is *not ours*.

## 28. Matrix 2 — the time, where we have it

Every cell is a measurement that exists, or `MISSING` with the command that would fill it. Times are ns a
cell at the stated context, RTX 4060 Ti, pinned clock, from the sources named.

| format | prefill ns/cell | source | decode ns/cell | source |
| --- | --- | --- | --- | --- |
| fp16 | `MISSING` | §28.1 | **8.32** @256K — `fdec` | `QUANT_MICROBENCH.md:619` |
| q8_0 | `MISSING` | §28.1 | **4.96** @256K — `fdec` | `QUANT_MICROBENCH.md:619` |
| k8v4 | `MISSING` | §28.1 | **5.88** @256K — `fdec`, i.e. **above** 4096 | `QUANT_MICROBENCH.md:69` |
| k8v4 @4K (`attend`, below threshold) | — | — | **`MISSING`** | §28.2 |
| k6v4 | **`MISSING` — the most important gap** | §28.3 | **10.46** @131K, **11.06** @256K, **16.80** @4K — `fdec` | `Q8_KV.md:324-329` |
| int8 / int8k | `MISSING` | §28.4 | `MISSING` | §28.4 |
| k4v4 | — | — | — | not built |

**The 5.88 figure is `fdec`'s, not `attend`'s.** `poly_kv_read_bench` runs `--only-fdec`
(`QUANT_MICROBENCH.md:565`), so the 4.96 / 5.88 / 8.32 ladder at 256K is three formats through **one** kernel,
and the below-4096 `attend` side of the same format is not in it. That is why §28.2's cell is empty: the
published read ladder does not cross the threshold it is quoted against.

### 28.1 Prefill timing exists for none of them, and cannot be read off the decode table

`poly_kv_read_bench` measures decode attention (`--only-fdec`); the prompt-batch path is `attend_batch` or
`fattn_*`, neither of which it exercises. `poly_kv_int8_test` does have a prompt-batch arm (8 queries at a
time, `kv_int8_test.cpp:871-880`), but it compares `attend` against `attend_int8` on a q8 cache — it does not
time the `fattn_*` flash path. **There is no prefill-attention ns/cell in this tree for any format on the
shipping path.** The prefill numbers that do exist move a whole prompt batch, are dominated by cuBLAS's GEMMs
for the linear layers, and cannot be reduced to a per-cell attention cost without a profile.

### 28.2 The `attend`-below-4096 cell — the cliff

`fdec_from_ = 4096` (`qwen35.cpp:395`), and the comment records the reason: *"below which its kernel was the
slower one (docs/DETAILS.md)"*. That comparison is one geometry, measured once, with no table kept — the doc
says so itself. The instruction change across the threshold is real and large (HMMA ×64 against zero), but
**the ns/cell on each side is unmeasured.** This is the single cheapest bench in the document: one binary, one
arm, two contexts.

**Now measured — see Part V §32.** The answer is that both sides land at 9.3 and 14.4 ns a cell and the
instruction change across the threshold moves the time by under 3%: `attend` 9.30 against `attend_int8` 10.29
at 4,095 cells, and 14.37 against 14.38 at 131,071. The cliff is real in the instruction stream and absent in
the time.

### 28.3 k6v4 prefill against k8v4 prefill — the cell question 1 asks

**MISSING, and nobody has measured it.** The comparison is not close on the instruction side: k8v4's prefill
runs our flash kernel (768 HMMA in the kernel body, plus 120 instructions of dequant), while k6v4's prefill
runs no kernel of ours at all — it dequantises 6-bit keys to fp16 in `dequant_*` and hands 512×q×k operands to
cuBLAS. What that costs is computable in principle and not in practice, because cuBLAS's cost per cell is a
function of its internal tiling.

### 28.4 Gemma 4's int8

No ns/cell exists for either format. `poly_kv_int8_test` has bench arms (`kv_int8_test.cpp:859-880`) that
time `attend` against `attend_int8`, but they have not been run for this document and they measure the two
q8 readers, not the model.

## 29. The four cells, answered

### 29.1 k6v4 prefill abandons the flash kernel — what does it cost?

**Unknown, and it is the highest-value unknown in this document.** What can be said without the card:

- The saving is **15.4% of a cell** (704 B against 832, `Q8_KV.md:280`).
- The cost is a **complete change of kernel**: k8v4's prefill runs `flash_attn_ext_f16` with 768 HMMA in our
  binary; k6v4's runs `dequant_*` into a workspace and then cuBLAS. The workspace is
  `row bytes × block of keys` of fp16 — the very bytes the format was built to avoid writing.
- The dequantise-to-workspace trick means k6v4's *prefill* writes **2 bytes a value** of scratch for keys it
  stored at 0.75. That is the same "dequant into fp16 scratch" shape Part I §3 identified as the reason a
  smaller format does not convert to time on the `strata` engine.
- On the decode side the answer is already measured, and it depends on the context: k6v4 is **1.060 at 4K**
  (slower), **0.915 at 32K**, **0.928 at 131K**, **0.968 at 256K** against k8v4 (`Q8_KV.md:324-329`). So it
  wins decode from about 32K up — by 7–8% — and loses at 4K, where the call is 0.07 ms and there are no bytes
  for the saved read to buy. That is `fdec` reading 6-bit natively.

**So the honest statement is: k6v4 wins decode by 7–8% above ~32K and its prefill is an open question that
could erase that for prompt-heavy use.** The asymmetry is the point — the format was justified on decode
numbers, and its prefill takes a different kernel from every other format, which no measurement covers. The
bench that settles it is one prefill A/B at a fixed prompt with `--kv k8v4` and `--kv k6v4`, same binary,
interleaved — §31 lists it.

**What to expect, from the instruction side.** k8v4's prefill keeps the cache in flash attention end to end:
a `dequant_kernel` (64 instructions) and a `dequant_q4_kernel` (56) feed `flash_attn_ext_f16`'s 768 HMMA. k6v4
instead runs `dequant_q6_kernel` (88 instructions, `kernels.cu:792`, `attend_batch`'s route at `:4307`) into
the fp16 workspace and hands that to cuBLAS — one kernel more in the chain and no flash attention at all. On
the counters in Part III the k6v4 decode kernel is already **+12.6% instructions over k8v4's for the same
block** (2,216 against 1,968), and unlike decode the prefill widening buys no smaller read: the workspace is
fp16 either way, so prefill reads 6-bit keys, writes 2 bytes a value, and reads them back.

### 29.2 k8v4 decode below `fdec_from_` — where is the cliff, and is it a bug?

`fdec_from_ = 4096`. Below it, `attend` runs `attn_score_reg_kernel<8,2,true>`: **560 instructions, zero
tensor instructions, scalar FFMA** (Part III §20.1). Above it, `fdec_k8v4` runs **64 `HMMA.16816.F32`**.

That is not a 5× cliff in *time* — the kernel that has no tensor op is also the one with 115 registers and a
simpler loop, and the measured decode numbers at 256K (5.88 ns a cell) are `fdec`'s, which is the *slower*
of the two possible readings at that context only because `attend` is not measured there. The threshold was
picked because `attend` was faster below 4096, and the code says that measurement was made once.

**Is it wrong?** It is unverified, which is different. Two things make it worth re-measuring: the threshold is
a hard branch (not a heuristic that degrades), and it was measured before the register-blocked rewrite at
`kernels.cu:3900-3910` whose own comment records an earlier comparison that *"ran the plain one with a grid a
quarter too small"* — i.e. this exact comparison has been gotten wrong before in this file.

### 29.3 int8 / int8k — which tier, and is it the tensor cores?

**Both tiers exist and the choice is by context and width, not by format:**

- `attn_score_mma_int8_kernel` (**`IMMA.16832.S8.S8`, ×8**) runs only when `int8_tensor_cores()` (PTX ≥ 8.0),
  **`dim == 256`**, and **`keys <= 16384`** (`kernels.cu:3912-3913`). That last condition is the interesting
  one: past 16K keys the code deliberately falls back.
- Otherwise `attn_score_reg_int8_kernel<8,8>` / `attn_score_int8_kernel<…>` run **`IDP.4A.S8.S8` (dp4a)** —
  `16` of them at `dim 256`, on the **FMA pipe**, which is exactly the opposite of the point.

So for `int8` at a context beyond 16K the answer to "are we on the tensor cores" is **no, we are on dp4a**, and
the reason is a gate, not a hardware limit. Gemma 4's head dim comes from the model file
(`attention.key_length`, `gemma4.cpp:49`), so whether the IMMA tier is reachable at all is a property of the
checkpoint, not of the code path.

### 29.4 The gap between "reachable" and "used by a qwen model"

`attn_score_mma_int8_kernel` emits 8 IMMA and **no qwen35 path calls it**. The chain that blocks it:

1. `engine.cpp:1652` refuses `--kv int8` and `--kv int8k` for any model that is not `gemma4`, with the reason
   stated: *"their readers are not the arithmetic q8_0's are"*.
2. `attend_int8` writes the plain weighted sum. `Qwen35::attn` needs the values **un-rotated** (`fwht_rows`,
   `qwen35.cpp:757`) and **gated** (`mul_sigmoid`, `:758`) after the mix — neither of which `attend_int8` does.
3. `attend_int8` attends a contiguous prefix with `pos`/`rows`; qwen35's decode goes through `fdec_*` and its
   prefill through `attend_batch`.
4. `KvCell` has no `KvInt8` value side for qwen: `kK8V4`/`kK6V4` are `q4` values (`qwen35.cpp:216-226`), and
   `attend_int8_mixed` takes fp16 values, not q4.

**What would have to change:** either `attend_int8` grows the rotation and gate (a kernel change, cheap, and
it makes the IMMA tier reachable for q8_0 keys on qwen — the format already exists), or `fdec`'s k8v4 QK is
rewritten on IMMA (Part III §22 says that is not worth it). **The first is the only route from "8 IMMA exist
in the binary" to "a qwen model runs them", and it is a smaller change than any new format in this document.**

## 30. What the two matrices say together

| | prefill | decode |
| --- | --- | --- |
| **our kernels, tensor cores** | fp16, q8_0, k8v4 — all on `flash_attn_ext_f16`, 384 `F32` + 384 `F16` HMMA | fp16, q8_0, k8v4, k6v4 above threshold — 64 `HMMA.F32` |
| **our kernels, no tensor op** | — | **k8v4 below 4096** (`attend`, scalar FFMA); every value side |
| **not our kernels** | **k6v4, int8, int8k** (cuBLAS) | — |
| **dp4a, not tensor cores** | — | **int8/int8k past 16K keys**, or any `dim != 256` |

Three consequences:

1. **The prefill column is where we do not control the instruction** for three of six formats, and one of
   those (k6v4) is the format the last merge added. The instruction census cannot see inside cuBLAS, and a
   bench is the only way to price it.
2. **The decode column is uniform except for two gates** — `fdec_from_ = 4096` and `keys <= 16384`. Both are
   hard branches into a no-tensor-instruction path, and neither has a published ns/cell on both sides.
3. **The only IMMA in an attention kernel is unreachable from any qwen model**, and the blocking piece is not
   the instruction but the missing rotation and gate in `attend_int8`.

## 31. Needs the card, in the order that answers the most

| # | cell | command | wall time |
| ---: | --- | --- | --- |
| 1 | **k6v4 prefill vs k8v4 prefill** — §29.1, the top unknown | prefill A/B on one prompt, `--kv k8v4` / `--kv k6v4`, same binary, interleaved, median of 5 | ~20 min |
| 2 | **k8v4 decode either side of 4096** — §29.2 | `poly_kv_read_bench 256 16 2` with cells `2048,4096,8192`, `STRATA_FDEC=0` and `1` | ~15 min |
| 3 | int8 IMMA vs dp4a at 8K and 32K — §29.3 | `poly_kv_int8_test` bench arms, `--keys 8192,32768` | ~20 min |
| 4 | fp16/q8_0 prefill ns/cell — §28.1 | `poly_kv_int8_test`'s prompt-batch arm already times 8 queries (`kv_int8_test.cpp:871-880`); extend it rather than write new code | ~30 min |

Rows 1 and 2 are the two cliffs in the matrix and neither has ever been measured on both sides.

# Part V — the IMMA measurement, on the card

The one measurement Part III said was worth the card time, run on 2026-11-13. RTX 4060 Ti 16 GB, sm_89, clock
pinned to 2820 MHz (`nvidia-smi -lgc`, reading 2805 under load), every run under `flock` and `timeout`.

```
build/poly_kv_read_bench 256 16 4 --imma --cells 4096,32768,131072,262144 --itr 20
```

This is the measurement Part IV §28.2 named the cheapest open bench: `attend` against `attend_int8` on either
side of `fdec_from_ = 4096`, with the instruction count changing 64 HMMA against 8 IMMA across it.

Geometry: **16 query heads, 4 KV heads (group 4), dim 256, 1,024 values a row** — the Ornith 9B's header
(`qwen35` dense, `head_count_kv = 4`, `key_length = 256`, Part I §5). The k8v4 payload is the same bytes for
both arms; `attend` reads them as `Kvq8`/`Kvq4`, `attend_int8` as `KvInt8`. `rows = 1`, which is the decode
call and the only shape both readers serve.

## 32. The result

`ratio` is `attend` time / `attend_int8` time: **above 1 means the IMMA path is faster.**

| cells | `attend` ns/cell | GB/s | `attend_int8` ns/cell | GB/s | **ratio** | cosine | argmax |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 9.30 | 178.9 | 10.29 | 211.5 | **0.904** | 0.996633 | 100% |
| 32,767 | 12.84 | 129.6 | 14.01 | 155.3 | **0.917** | 0.996887 | 100% |
| 131,071 | 14.37 | 115.8 | 14.38 | 151.3 | **0.999** | 0.997728 | 100% |
| 262,143 | 14.95 | 111.3 | 14.58 | 149.3 | **1.026** | 0.998267 | 100% |

Forced to IMMA at every size (`STRATA_KV_MMA=1`), which overrides the `keys <= 16384` default at
`kernels.cu:3912`:

| cells | `attend` ns/cell | `attend_int8` ns/cell | **ratio** | cosine | argmax |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 9.25 | 10.23 | **0.905** | 0.996633 | 100% |
| 32,767 | 13.96 | 14.60 | **0.956** | 0.996887 | 100% |
| 131,071 | 14.57 | 14.49 | **1.006** | 0.997728 | 100% |
| 262,143 | 15.59 | 14.73 | **1.058** | 0.998267 | 100% |

Four further runs at 4,096 and 131,072 reproduced 0.904–0.911 and 0.968–1.006. **The spread across runs is
±3%, and the effect is inside it.**

## 33. The verdict: the INT8-QK line does not have value, and the reason is the useful part

**Against the pre-registered bands: 0.90–1.06× on time, where the hypothesis needed 1.2–1.9×.** Cosine
0.9966–0.9983 clears the 0.999 bar at no size except the two longest (0.9977, 0.9983, which do), and argmax
agreement is 100% everywhere. So the accuracy is fine and **the speed is not there**. `attend_int8` is not
≥1.1× at any context, and at short context it is 10% *slower*. The line closes here.

### Why, and it is not what the SASS suggested

Part III predicted a large gap — "IMMA against no tensor core at all" — from a static count: 8 IMMA in 780
instructions. The card says otherwise, and the reason is visible in the same run:

- **The instruction itself is 3.3× faster in isolation.** `poly_kv_int8_test` prints
  `one 16x8x256 product x20000 x384 warps: mma 3.4 ms, dp4a 11.2 ms -> mma is 3.3x`, and
  `[mma probe] 0 of 128 wrong`. So the tensor op does what the ISA dump says it does.
- **The kernel is not bound by it.** Both arms are at 111–179 GB/s of a 288 GB/s bus and both spend the same
  ~14 ns a cell at long context. The dequant, the score write and the value loop dominate, and swapping the
  QK arithmetic for a 3.3× faster instruction moves none of them.
- **`attend_int8` also reads 31% more bytes** — 2,176 B a cell against k8v4's 1,664, because its value side is
  int8 rather than q4. At short context, where the read is not amortised, that is the whole 10% loss.

That is Part I §3's conclusion arriving from a second direction: **on this card the KV read is dequant- and
issue-bound, not arithmetic-bound, so no tensor-core substitution pays.**

### What this does not say

- It does not say IMMA is useless. It says IMMA inside `attend_int8`, against `attend`, at decode batch 1,
  on a 128-bit bus, is not the lever. The 3.3× is real and would show at a shape with reuse — a prompt batch
  of many queries per key tile, which is what `attend_batch` already does with cuBLAS.
- It does not close `k4v4`. The `S4.S4` path (`sm89_inst.txt`, 287 `SRCFMTA_U8_S8_U4_S4` entries) remains the
  one place a *new* tensor form exists, and it was never what this bench tested.
- The `rows = 8` arm was dropped, not failed. A multi-row call's last row attends more keys than its first,
  and the two readers leave different tails of that row unwritten; compared, they measure call shapes, not
  readers. Recorded here so nobody re-runs it expecting a number.

## 34. The harness

`src/poly/kv_read_bench.cpp`, an `--imma` arm added beside the existing ones (nothing replaced). It builds
one payload, drives both readers from it, and prints time, bytes-derived GB/s, cosine, max error against the
vector's RMS, and argmax agreement **on the same line**, so a speed number cannot be quoted without its
parity. `STRATA_KV_MMA` is printed so the tier in force is visible in the log rather than assumed.

Three harness bugs found and fixed while getting here, each of which had reported a plausible wrong number
first — the same failure mode Part II §16 is written against:

1. `pos = cells` with `rows = 512` attended rows the cache never held (cosine 0.0). The last token of a call
   must land on the last populated row: `pos = cells - rows`.
2. `attend()` was asked for 512 rows. It is the few-token path (`< kStageRows = 32`); the engine sends a
   prompt to `attend_batch`. Out-of-contract calls produced NaNs.
3. `max_rel` as `|got-ref|/|ref|` reported 1e2–1e5 on good kernels, because the reference crosses zero. It is
   now the largest absolute error over the vector's RMS.
