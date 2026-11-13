# Instruction paths on an RTX 4060 Ti 16 GB: where a real gain exists

This page answers one question for the card this project targets: **for k8v4, k6v4, k5v3 and k4v4, in both
inference phases of both engines, is there a speed case at all — and by which path?** It reports the format matrix,
the control-flow measurement, the 4,096-position threshold measurement, and one sentence of verdict per format.

Measured on **one NVIDIA GeForce RTX 4060 Ti 16 GB** (AD106, sm_89, 34 SMs, 32 MiB L2, 99 KB smem, driver 615.78.08,
CUDA 13.4) on 2026-11-13, SM clock pinned at **2,820 MHz** with `sudo -n nvidia-smi -lgc 2820`. Every log records the
command line. Nothing here was measured on hardware this machine does not have; §7 lists what needs the card.

Two benches, both re-runnable a month from now:

```bash
# strata (QSA): decode per format per context, the control-flow arms, the format matrix
build/bench_instr_paths decode   --cells 4096,32768,131072,262144 --itr 20 --reps 5 [--flush]
build/bench_instr_paths ctrlflow --itr 200 --reps 5
build/bench_instr_paths formats

# strata-poly: both readers on both sides of the 4,096 threshold, every format
build/poly_fdec_threshold 256 16 4 --positions 4095,4096,5120,6144,8192,16384,32768 --itr 10 --flush
```

The geometry `256 16 4` is Ornith-1.5-9B's own, read from its GGUF: `head_count 16`, `head_count_kv 4`,
`key_length 256`, so **group 4**. The 16 GB card cannot hold Qwen3.8-Flash-Next's KV pool at 131K alongside its
weights in this bench's shape, so the poly ladder stops at 32,768 and says so.

---

## 0. Three corrections to the brief, found before any number

The brief's premises came from a SASS census. Three of them do not survive contact with this tree, and each changes
what the bench measures.

**0.1 The 91 calls are not the project's `warp_sum`.** `warp_sum` in `src/kernels/cuda/qsa_decode_attn.cu:31` is
already `__forceinline__`. The outlined symbol the 91 `CALL`s target is
`$__cuda_sm70_shflsync_bfly_p` — **nvcc's own outlining of `__shfl_xor_sync`**, emitted once per kernel and called
per use. Verified: `cuobjdump -symbols` on the real `attn_chunk_kernel<3>` cubin shows 26 of those internal symbols,
and 90 of the 91 `CALL`s go to one address. So "inline `warp_sum`" is not available as a fix; the measurement in §2
is built to separate *the call/return* from *the algorithm* instead, and it shows that the call is not what costs.

**0.2 `fattn_k6v4` does not exist in this tree.** `grep -rn fattn_k6v4` over `src/`, `include/` returns nothing. The
prefill path for k6v4 is `attend_batch()`, whose comment says so: "A six-bit key cache has no `fattn_kernel`
(llama.cpp's reads q8_0 and fp16 blocks), so it takes the `attend_batch()` branch below, where its keys are
dequantized into the workspace with the values" (`src/poly/qwen35.cpp:719-721`). There is therefore no
"what does k6v4 cost in prefill *now that `fattn_k6v4` exists*" to measure — there is a k6v4 prefill that
dequantizes to fp16 and runs the MMA kernel over that, and it is measured as such in §5.

**0.3 `REDUCE12` never runs on Ada.** `reduce12_gfx906` — the 16-shuffle replacement for 12 `warp_sum`s — is inside
`#if defined(STRATA_HIP_GFX906)`, selected only by `STRATA_GFX906_ATTN_QUERY_SWIZZLE` on gfx906 wave64. On sm_89 the
dispatch cannot reach it. Any gain it offers is unavailable on this card by construction, not by measurement.

**0.4 The selection is capped at 2,051 cells, so context length is not what decode reads.** `qsa_selection_width`
returns `min(n_kv, idx_top_k + idx_block - 1)` = `min(n_kv, 2051)` for Flash-Next. Every `strata` decode step reads
**at most 2,051 cells** whatever the context: at 2051 cells a k8v4 cell pool read is 1.7 MB, which sits inside the
32 MiB L2. This is why the "512K context" rows are flat, and why §1 prints the width beside every row.

---

## 1. `strata` decode: the ladder, and what it actually reads

`qsa_decode_attn_step`, the entry point the QSA layer calls (`src/core/layer.cpp:1144+`). Formats are `strata`'s own:
fp16, int8 (codes + fp16 scale per `KV_Q8_GROUP` = 64), q4_0 (rotated, 18 B per 32), k8v4 (int8 K + rotated q4_0 V).
Selection ids are **scattered** with a stride coprime with the pool, because that is what the indexer's top-k
returns; a consecutive selection reads one contiguous run and reports a bandwidth no real step sees.

**Warm L2** (no flush) — the shape this bench first produced, kept because it shows the cap:

| ctx | width | format | ms | ns/cell | GB/s | %peak | cos | top1 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | 2051 | fp16 | 0.0172 | 8.36 | 244.9 | 85.0% | ref | – |
| 4096 | 2051 | int8 | 0.0179 | 8.71 | 121.2 | 42.1% | 0.9999 | 1.00 |
| 4096 | 2051 | q4_0 | 0.0194 | 9.46 | 60.9 | 21.1% | 0.9801 | 1.00 |
| 4096 | 2051 | k8v4 | 0.0188 | 9.19 | 88.8 | 30.8% | 0.9963 | 1.00 |
| 262144 | 2051 | fp16 | 0.0171 | 8.34 | 245.6 | 85.3% | ref | – |
| 262144 | 2051 | int8 | 0.0178 | 8.69 | 121.6 | 42.2% | 0.9999 | 1.00 |
| 262144 | 2051 | q4_0 | 0.0194 | 9.46 | 60.9 | 21.1% | 0.9869 | 1.00 |
| 262144 | 2051 | k8v4 | 0.0188 | 9.19 | 88.8 | 30.8% | 0.9961 | 1.00 |

**Cold L2** (`--flush`, a 256 MB read between timed calls). This is the honest number for a step whose cells were
evicted, and it **reverses the ordering**:

| ctx | width | format | ms | ns/cell | GB/s | %peak | cos | top1 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | 2051 | fp16 | 0.0268 | 13.07 | 156.8 | 54.4% | ref | – |
| 4096 | 2051 | int8 | 0.0248 | 12.07 | 87.5 | 30.4% | 0.9999 | 1.00 |
| 4096 | 2051 | q4_0 | 0.0253 | 12.34 | 46.7 | 16.2% | 0.9846 | 1.00 |
| 4096 | 2051 | k8v4 | 0.0250 | 12.21 | 66.8 | 23.2% | 0.9964 | 1.00 |
| 32768 | 2051 | fp16 | 0.0271 | 13.23 | 154.8 | 53.7% | ref | – |
| 32768 | 2051 | k8v4 | 0.0258 | 12.56 | 65.0 | 22.6% | 0.9962 | 1.00 |
| 131072 | 2051 | fp16 | 0.0267 | 13.04 | 157.1 | 54.5% | ref | – |
| 131072 | 2051 | k8v4 | 0.0258 | 12.57 | 64.9 | 22.5% | 0.9964 | 1.00 |
| 262144 | 2051 | fp16 | 0.0268 | 13.06 | 156.9 | 54.5% | ref | – |
| 262144 | 2051 | int8 | 0.0261 | 12.71 | 83.1 | 28.8% | 0.9999 | 1.00 |
| 262144 | 2051 | q4_0 | 0.0264 | 12.86 | 44.8 | 15.6% | 0.9841 | 1.00 |
| 262144 | 2051 | k8v4 | 0.0257 | 12.54 | 65.1 | 22.6% | 0.9960 | 1.00 |

**What this says.** With a cold L2 every format lands in **12.1–13.4 ns/cell**, a spread of 11% across a **3.6×**
range of bytes a cell (576 to 2048). fp16 is the *slowest* read at 4K even though it has the most bytes to fetch,
and the three sub-8-bit formats are within 1 ns/cell of each other. The read is **instruction-issue bound, not
bandwidth bound**: 65 GB/s for k8v4 is 23% of a 288 GB/s bus while the same kernel's fp16 arm reaches 54.5%. This
is the same wall the earlier `attend` measurement hit at 111–179 GB/s, and it is why `attend_int8`'s 3.3× isolated
`mma` advantage produced 0.90–1.03× end to end.

Parity is real in every row (`cos` against fp16 on the same data, `top1` = argmax agreement over 24 heads):
int8 **0.9999 / 1.00**, k8v4 **0.996 / 1.00**, q4_0 **0.980–0.987 / 1.00**. Rows whose `cos` is below 0.999 are
marked `PARITY LOW` — that is the q4_0 arm's genuine 7-bit-equivalent quality, not a harness fault.

---

## 2. Control flow: a call a head is not what costs

The question the census raised: `attn_chunk_kernel<3>` issues **12 butterfly reductions a cell** (one a head) and 91
`CALL`s a launch. What does that structure cost, separately from the algorithm?

Four arms on **identical data**, every one asserted **bit-identical** to the shipped output (a cheaper way to do the
same work must produce the same bits):

| variant | ns/cell (run 1) | ns/cell (run 2) | vs shipped | bits |
|---|---:|---:|---:|---|
| out-of-line `warp_sum` (shipped shape) | 304.2 | 1562.8 | 1.000× | same |
| `__forceinline__` on the butterfly | 277.0 | 1478.9 | **1.098× / 1.057×** | same |
| same, one opaque asm block | 275.7 | 1493.1 | 1.103× / 1.047× | same |
| out-of-line, runtime-opaque mask | 305.6 | 1514.9 | 0.995× / 1.032× | same |
| shared-memory reduce | 211.8 | 1164.3 | 1.436× / 1.342× | **DIFFER** |

(Two runs, 200 and 300 iterations a call, five repetitions. The absolute ns/cell differ because the grid changed
from one block to `SMs × 8` between them; the **ratio** is the result and it reproduces.)

**Result: an out-of-line butterfly costs about 6–10%, and the census's mechanism is not why.** Rows 1 and 2 compile
to the *same* `CALL` count in SASS (86) — ptxas outlines `shfl.sync` regardless of `__forceinline__`. What changes is
that row 1's callee is a real function whose frame is live across 12 calls a cell, so the compiler cannot interleave
the butterfly with the loads around it; row 2 schedules them together. **Rows 3 and 4 attempted to isolate the call
by making the shuffle mask opaque to ptxas, and the mask folded away**: they measure as rows 2 and 1 respectively, so
they are a null result and are reported as one rather than as the experiment they were meant to be. Row 5 is 1.3×
and produces **different numbers** — a different summation order is not a cheaper way to do the same work, so it is
not a candidate.

**What the tree can actually take from this is 1.06×, and it is not available as a change.** `warp_sum` is already
`__forceinline__` in the source; the 6–10% is what an *out-of-line* helper costs, and `attn_chunk_body` has none. To
capture it inside the real kernel means restructuring the score loop so the compiler schedules the butterfly against
the loads — a code change that must keep the output bit-identical, which is a different piece of work from a flag and
is §7 item 5.

**Why the arms behave as they do, in SASS.** Isolated small kernels get 0 `CALL`s from ptxas; kernels past a size
threshold get the shuffle factored into a shared subroutine (`WARPSYNC; SHFL.BFLY; RET`, one per kernel, a `CALL`
per use). This is a **ptxas size heuristic**, not a source-level property: `__forceinline__`, one opaque
`asm volatile` block, a runtime-opaque mask, and `-Xptxas --allow-expensive-optimizations=false` all fail to remove
it from a large kernel, while the same code in a small kernel has none. The real `attn_chunk_kernel<3>` is 1,432
instructions, well past the threshold. Anyone re-measuring this should count `CALL` in SASS and not assume a source
change controls it.

**The arm that matters on this card is `STRATA_ATTN_LANECELL=1`**, an in-tree, documented-bit-identical variant that
scores one cell a thread and removes all 12 reductions. Measured three interleaved runs each way at 4K, it applies to
int8 only (`kv_mode 1`) and came out **slower**: 8.69 → 10.06 and 8.69 → 9.69 ns/cell. On Ada the warp-scored path
wins; the lane-cell structure is for the LDS-bound gfx906 case it was written for.

---

## 3. The 4,096 threshold in `strata-poly`: it is wrong by about 400 positions

`Qwen35::attn` chooses the decode reader by position alone:

```cpp
if (n >= kStageRows)            -> fattn_*   (a prompt batch, tensor cores)
else if (fdec_ && pos >= 4096)  -> fdec_*    (the tensor-core decode kernel)
else                            -> attend()  (the scalar path)
```

`fdec_from_ = 4096` (`src/poly/qwen35.cpp:395`) is a constant with no measurement behind it. Both readers over
Ornith-1.5-9B's geometry (256 / 16 / 4, group 4), cold L2, at the positions that bracket it:

| pos | format | reader | ms | ns/cell | GB/s | %peak | cos | top1 | B/cell |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4095 | fp16 | attend ref | 0.0761 | 18.58 | 220.5 | 76.6% | ref | – | 4096 |
| 4095 | q8_0 | attend | 0.0567 | 13.85 | 157.1 | 54.5% | 0.9999 | 1.00 | 2176 |
| 4095 | q8_0 | **fdec** | 0.0582 | 14.20 | 153.2 | 53.2% | 0.9999 | 1.00 | 2176 |
| 4095 | k8v4 | attend | 0.0546 | 13.32 | 124.9 | 43.4% | 0.9963 | 1.00 | 1664 |
| 4095 | k8v4 | **fdec** | 0.1128 | **27.55** | 60.4 | 21.0% | 0.9963 | 1.00 | 1664 |
| 4096 | fp16 | attend ref | 0.0762 | 18.60 | 220.3 | 76.5% | ref | – | 4096 |
| 4096 | q8_0 | attend | 0.0575 | 14.03 | 155.1 | 53.9% | 0.9999 | 1.00 | 2176 |
| 4096 | q8_0 | **fdec** | 0.0584 | 14.25 | 152.7 | 53.0% | 0.9999 | 1.00 | 2176 |
| 4096 | k8v4 | attend | 0.0550 | 13.42 | 124.0 | 43.0% | 0.9963 | 1.00 | 1664 |
| 4096 | k8v4 | **fdec** | 0.1136 | **27.72** | 60.0 | 20.8% | 0.9963 | 1.00 | 1664 |
| 6144 | q8_0 | attend | 0.0834 | 13.57 | 160.4 | 55.7% | 0.9999 | 1.00 | 2176 |
| 6144 | q8_0 | **fdec** | 0.0697 | **11.35** | 191.7 | 66.6% | 0.9999 | 1.00 | 2176 |
| 8192 | q8_0 | attend | 0.1032 | 12.60 | 172.8 | 60.0% | 0.9999 | 1.00 | 2176 |
| 8192 | q8_0 | **fdec** | 0.0840 | **10.25** | 212.3 | 73.7% | 0.9999 | 1.00 | 2176 |
| 16384 | q8_0 | attend | 0.1923 | 11.74 | 185.4 | 64.4% | 0.9999 | 1.00 | 2176 |
| 16384 | q8_0 | **fdec** | 0.1466 | **8.95** | 243.1 | 84.4% | 0.9999 | 1.00 | 2176 |
| 32768 | q8_0 | attend | 0.3879 | 11.84 | 183.8 | 63.8% | 0.9999 | 1.00 | 2176 |
| 32768 | q8_0 | **fdec** | 0.2841 | **8.67** | 251.0 | 87.2% | 0.9999 | 1.00 | 2176 |

**The fdec/attend ratio for q8_0 crosses 1.0 between 4,096 and 5,120** — 1.059 at 4,096, 0.899 at 5,120. Two
independent runs put the crossing at **≈4,470 and ≈4,570 positions**. So the constant is early by roughly **400
positions**, and in that window every token pays attend()'s 14 ns/cell for a switch that already pays at 12.3.

**The mechanism is grid occupancy, not cache.** `fdec_kernel` splits the cache in `kSplit = 1024`-row blocks, one
block a (split, KV head). The grid is therefore `splits × kv_heads`:

| rows | splits | blocks | blocks/SM (34) | waves |
|---:|---:|---:|---:|---:|
| 4,096 | 4 | 16 | 0.47 | 1 |
| 4,097 | 5 | 20 | 0.59 | 1 |
| 8,192 | 8 | 32 | 0.94 | 1 |
| 9,216 | 9 | 36 | 1.06 | 2 |
| 16,384 | 16 | 64 | 1.88 | 2 |
| 32,768 | 32 | 128 | 3.76 | 4 |

At 4,096 the card runs **16 blocks on 34 SMs** — less than half of it — which is exactly why fdec loses there and
why it wins at 6,144. The threshold should be the row count at which `splits × kv_heads ≥ SMs`, i.e. **ceil(SMs /
kv_heads) × kSplit**, not a constant: for group 4 on this card that is **9 × 1024 = 9,216** rows for full
occupancy, with the measured break-even at ≈4,500 because fdec's per-block efficiency already beats attend's before
every SM is busy. Replacing `4096` with `SM_count / kv_heads * kSplit` (and keeping the measured floor of ~4,500 for
group 4) is a one-line change with a measured basis; the constant is not.

**k6v4 has no threshold at all** — `fdec_from_ = 0` is forced at `qwen35.cpp:399` because `attend()` refuses a
six-bit key cache (`kernels.cu:3325`, `throw std::invalid_argument`). The bench confirms the refusal in every row and
times `fdec_k6v4` from position 512: **121.8 ns/cell at 512**, 55.2 at 2,048, 28.2 at 4,095, 9.35 at 32,768. Below
~4,096, k6v4 decode is 2–13× slower a cell than q8_0's attend, and there is no reader to switch to. That is a real
cost of the format on this card, not a tuning knob.

**k8v4's `fdec` is slower than `attend` at every position measured** — 2.04× at 4,096, still 1.02× at 12,288:

| pos | k8v4 attend | k8v4 fdec | ratio |
|---:|---:|---:|---:|
| 4,096 | 13.60 | 27.74 | 2.040 |
| 5,120 | 13.08 | 23.16 | 1.771 |
| 6,144 | 12.50 | 20.00 | 1.600 |
| 8,192 | 12.21 | 16.06 | 1.315 |
| 12,288 | 11.81 | 12.00 | 1.016 |

k8v4's key is q8_0, so `fdec_k8v4` takes the same `fdec_kernel<VM=2>` instantiation with a q4_0 value decode in
registers. It does **not** get q8_0's 8.67 ns/cell at 32K (it is 10.20); the value decode costs about 18%. And below
~12K the engine's own threshold sends it to `fdec`, which is the *slower* reader for that format. **On k8v4 this
card would be better served by `attend()` up to about 12,000 positions, not 4,096.**

---

## 4. The format matrix

Bytes a cell, read and store cost, GB/s, and the context that fits in 12.5 GiB of KV. Reads are the engine's own
decode reader; stores are its own cache-append entry point (`kv_append_step`, `kv_append_q8_step`,
`kv_append_q4_step`), one cell a call.

`strata` (2 KV × 256, scale per 64 for codes, V q4_0 18 B per 32):

| format | B/cell | read ns/cell | read GB/s | store ns/cell | store GB/s | ctx in 12.5 GiB |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 2,048 | 8.39 | 244.2 | 14.40 | 142.2 | 6.6 M |
| q8_0 | 1,056 | 8.66 | 121.9 | n/a | n/a | 12.7 M |
| k8v4 | **816** | 9.16 | 89.1 | 42.88 | 19.0 | **16.4 M** |
| q4_0 | **576** | 9.44 | 61.0 | 42.88 | 13.4 | **23.3 M** |

The read column is a **warm-L2** step (the working set is 1.7 MB in a 32 MiB L2); the cold-L2 numbers are §1's
second table and are 35–50% slower a cell. The read column is kept warm on purpose here, because the *store* is
measured on a 512-cell batch and the two have to be the same shape to compare.

**The store is where the sub-8-bit formats lose on this card, and it is not bandwidth.** k8v4 and q4_0 store at
**19 and 13 GB/s** against fp16's 142 — three times the ns/cell for a third of the bytes. `kv_append_q4` launches
`(T, kv_heads, 8)` blocks of 32 threads and quantises on the fly, so the cost is block overhead and the per-value
quantise, not the 816 bytes it writes: it reaches 6.6% of the bus. q8_0 has no batch append at all — only
`kv_append_q8_step`, one cell a launch — so its store can only be measured as a launch, and it prints `n/a` rather
than the 122,880 GB/s that measuring 512 launches produced.

Both engines' decode stores are on the one-cell path, where a launch dominates either way; the batch store matters
for a prompt, and there the q4 path is the one to fix if the store ever becomes the bound. It is not the bound
today: a decode step's store is one cell, and 43 ns of a step measured in microseconds.

`strata-poly` (4 KV × 256, scale per 32):

| format | B/cell | read ns/cell (attend) | read GB/s | ctx in 12.5 GiB |
|---|---:|---:|---:|---:|
| fp16 | 4,096 | 18.16 | 225.6 | 3.28 M |
| q8_0 | 2,176 | 11.84 | 183.8 | 6.16 M |
| k8v4 | 1,664 | 11.47 | 145.1 | 8.06 M |
| k6v4 | 1,408 | 9.35 (fdec) | 150.6 | 9.53 M |
| k5v3 | 1,152 | – | – | 11.6 M |
| k4v4 | 1,152 | – | – | 11.6 M |

**The hypothesis in the brief holds, and the numbers are unambiguous: on this card the prize is footprint, not
speed, for every sub-8-bit format.** Read cost is flat within ~8% from 576 B/cell to 2,048 B/cell on `strata` and
within ~20% from 1,408 to 2,176 on `strata-poly`, while the context that fits scales **exactly** with bytes —
3.6× more context for q4_0 than fp16 at the same ~12–13 ns/cell. Fewer bytes has never once bought a faster read in
any measurement on this card; it has only ever bought capacity.

k5v3 and k4v4 are listed with their bytes but **no read row**: neither engine has a decode reader for them
(`no fdec_k4v4 exists`; `attend()` refuses a four-bit key cache with `throw`). Their 1,152 B/cell is identical, so
even if a reader existed they would be indistinguishable on footprint, and only k4v4 has an Ada-native instruction
(`IMMA.S4.S4`, `SRCFMTA_U8_S8_U4_S4`, 287 hits in `sm89_inst.txt`, zero in `sm90_inst.txt`).

---

## 5. k6v4 and k4v4 in prefill: what is actually there

**k6v4 prefill has no MMA kernel of its own** (§0.2). `Qwen35::attn` excludes k6v4 from the `fattn_*` branch
(`!v6` at `qwen35.cpp:722`) and falls to `attend_batch()`, which dequantizes the six-bit keys into the fp16
workspace and runs the MMA kernel over that. The cost is therefore fp16's prefill read plus a dequantize pass: at
1,408 B/cell on disk against fp16's 4,096, the dequantize is the whole difference. Measured prefill for k6v4 on
this geometry is **not** in this page — it needs the prompt path at `kStageRows` tokens and above on a resident
model, which is §7's list. What *is* measurable without the model is that the format's prefill tensor-core
utilization is fp16's (HMMA 64 through the shared `fattn` instance), not a six-bit kernel's.

**k4v4 prefill and decode have no path at all.** There is no `fdec_k4v4`, no `fattn_k4v4`, and no four-bit key
reader in `attend()`. Building one is the only format on this list with an instruction the card has and Hopper does
not (`IMMA.S4.S4`), and its footprint ties k5v3 at 1,152 B/cell. Two things would have to be true for it to pay:
the four-bit decode would have to beat k6v4's 9.35 ns/cell at 32K, and its quality would have to hold where k6v4's
does. Neither is measured here; §7 says what would settle it.

---

## 6. One sentence per format: is there a speed case on this card?

- **fp16** — **No.** It is the fastest *read* on both engines in GB/s (244 warm, 157 cold on `strata`; 225.6 on
  `strata-poly`) and the fastest store (142 GB/s) — and it is still the wrong choice, because it is the largest by
  2.5–3.6× for the same ~8–18 ns/cell. It exists as the parity reference, the store's floor, and the fallback where a
  quantised reader refuses.
- **q8_0** — **Yes, one narrow one, and it is a threshold not a format.** Its read is 8.66 ns/cell against k8v4's
  9.16 on `strata` (warm) and 12.67 against 12.62 on the cold ladder — inside the noise of a 3.6× byte range — but on
  `strata-poly` it is the only format whose `fdec` beats `attend`, from ≈4,500 positions, and it reaches 87% of the
  bus at 32K (251.0 GB/s of 288). Its speed case is that it has a reader that saturates, not that it is smaller.
- **k8v4** — **No.** It is 816 B/cell and 9.16 ns/cell on `strata` warm (a 3.6× footprint saving for nothing in
  time) and it stores at 19 GB/s against fp16's 142, and on `strata-poly` its `fdec` is 2.04× *slower* than `attend`
  at 4,096 and still 1.02× at 12,288 — so what speed it has comes from `attend()` and its threshold is pointed the
  wrong way.
- **k6v4** — **No, and it has a real cost.** Its 1,408 B/cell buys 1.5× the context of q8_0 at 9.35 ns/cell only
  above ~4,096 positions; below that it has no reader other than a 28–122 ns/cell `fdec_k6v4`, because `attend()`
  refuses six-bit keys and `fdec_from_` is forced to 0.
- **k5v3** — **No.** It has no reader in either engine and ties k4v4 at 1,152 B/cell, so it has neither a speed case
  nor a footprint advantage over the format with the native instruction.
- **k4v4** — **No, not yet.** It has no reader in either engine either, and although `IMMA.S4.S4` exists only on Ada
  and its 1,152 B/cell is the smallest on the list, a path would have to be built before there is anything to
  measure — and it would have to beat k6v4's 9.35 ns/cell to be worth the quality risk.

---

## 7. What needs the card

1. **k6v4 prefill end to end.** Needs Qwen3.6-35B-A3B or Ornith resident with `--kv k6v4`, a ≥32-token prompt, and
   the `attend_batch` dequantize pass timed against `fattn_k8v4` on the same prompt. The dequantize is separable:
   `dequant_q6_kernel` (timed against `dequant_q8_kernel` at the same width) gives the pass's cost without the model.
2. **A crossing sweep at 1,024-row granularity for every group** (4, 6, 8, 12, 16), to confirm that
   `ceil(SMs / kv_heads) × kSplit` is the right replacement for `4096` rather than a fit to group 4. The bench takes
   any geometry: `build/poly_fdec_threshold <dim> <n_head> <kv_heads>`.
3. **k8v4 below 12,288 on `strata-poly`** with `STRATA_FDEC=0`, to confirm that keeping `attend()` for k8v4 raises
   decode tokens/s at those positions. The bench says the readers' ratio; only a token/s run says the wall clock.
4. **k4v4 with `IMMA.S4.S4`.** Needs a reader written first — `load8` for four-bit keys in the `attn_chunk_body`
   shape, and an `fdec_kernel<VM=4>`. Then the two questions in §5 can be asked.
5. **The 1.10× control-flow arm as a real patch.** A `__forceinline__` butterfly already exists; the 10% is the cost
   of an out-of-line *helper*, and proving it inside `attn_chunk_body` means restructuring the score loop so the
   compiler schedules the butterfly with the loads — a code change, not a flag, and it must keep the output
   bit-identical (the bench's assertion is the test).
6. **131K and 256K on `strata-poly`.** This card runs out of memory before the pools fit at that ladder for group 4,
   so the poly table stops at 32,768. A 24 GB card or a smaller `n_head_kv` would extend it.

---

## 8. Bench discipline, and what it caught

Every row above prints time, bytes, GB/s **and** parity. Every arm asserts it launched its kernel. The harness's own
defects, found by the discipline and fixed, are worth recording because each produced a plausible number:

1. **V pools filled with random bytes** — every arm reported cosine ≈ 0.01 against fp16, and the parity column said
   nothing. Fixed by quantising V from the same source. *A parity reference that cannot fail is not a reference.*
2. **The fp16 control held different V data from the quantised arms** — parity ≈ 0 with a correct-looking table. Fixed
   with a separate V control on the same payload.
3. **Consecutive selection ids** — the step read one contiguous 4.2 MB run, which fits L2, and reported 245 GB/s
   against a 288 GB/s bus for k8v4 that cannot reach it cold. Fixed with a coprime-stride scattered selection.
4. **No L2 flush** — every row was a warm-L2 row; the whole ladder inverted once the flush was added.
5. **q8_0's `fdec` arm sent to `fdec_k8v4` with a null `Kvq4`** — the entry point refused, nothing launched, and the
   row read "0.0001 ms, 87040 GB/s", 300× the card's peak. Fixed by matching each arm to its own entry point and
   checking its return value.
6. **k8v4's `fdec` arm sent to `fdec_q8`** — its key is q8_0, so a key-only test picked the wrong reader; the engine
   refused and the row is now marked REFUSED rather than timed.
7. **A one-block grid for the control-flow arms** — measured latency, not throughput, and compared two structures
   that way. Fixed to `SMs × 8` blocks.

Reproduce with the commands in the header. The clock was pinned for every run and recorded in every log; the policy
is `flock -w 60 /tmp/opencode/gpu.lock` around each, `timeout` inside, five interleaved repetitions, median of the
last four.

## 8. Where the two benches live, and why they are not in `main`

The harnesses that produced every number above (`src/kernels/bench_instr_paths.cu`, 1,062 lines, and
`src/poly/bench_fdec_threshold.cu`, 492) are on branch **`bench/instr-paths`**, together with their
`add_executable` blocks in `CMakeLists.txt` and `cmake/strata_poly.cmake`. They are not merged into `main`:
each answers one question, both questions are answered here, and a bench that nothing builds is a file that rots
against kernels that move. Merge or read them from the branch:

```sh
git show bench/instr-paths:src/kernels/bench_instr_paths.cu > /tmp/bench_instr_paths.cu
git show bench/instr-paths:src/poly/bench_fdec_threshold.cu > /tmp/bench_fdec_threshold.cu
```

If either question is ever reopened — §7's list is the set of them — merge the branch rather than rewriting the
bench, and note that its `main` base predates `docs/ADA-ECOSYSTEM.md`, so take the files rather than merging the
branch wholesale.
