# Tuning `strata-poly` on an RTX 4060 Ti 16 GB: two work lists

Scope: **AD106, sm_89, 34 SM, 288 GB/s DRAM, 32 MiB L2, 99 KB opt-in shared memory, ~12.5 GiB usable VRAM.**
No GPU run was made for this document. Every claim is arithmetic from the numbers in the brief, a `file:line`, or a
measurement already recorded in this tree. Anything the tree cannot answer is in §5.

The engine has two nearly disjoint paths and they rank differently, so this document is two lists:

| | **dense** (`qwen35`, `dense_ = true`) | **MoE** (`qwen35moe`, `flint`, `lfm2moe`, `clover_moe`) |
|---|---|---|
| models here | Ornith-1.5-9B Q4_K_M, Qwen3.8-27B | Qwen3.6-35B-A3B UD-Q3_K_M, Flint |
| geometry (from the GGUF headers, §1) | 32 trunk + 1 draft block, n_embd 4096, n_ff 12288, 16 heads / **4 KV**, hd 256, 8 full-attention layers + the draft's | Qwen3.6: 40 + 1, n_embd 2048, 256 experts top-8, expert n_ff 512, 16 heads / **2 KV**, hd 256, 10 full layers + draft. Flint: 49, 32 heads / **4 KV**, hd 128, 2 shared experts, top-6, sliding window 513 on 35 of 49 layers |
| what a token reads | **every weight**: 5.769 GB (Ornith), 10.93 GB (27B) | ~8 of 256 experts a layer: **12.912 GB of experts of which a token touches ~0.6%**, and the dense part |
| dominant cost | weight bandwidth, per token | **the expert cache**: how many slots, how a miss is served |
| measured today | prefill 40,960 **1,942.2** tok/s, 131,072 **986.4**; decode 16,384 **43.46** (k8v4) / 40.96 (k6v4) | **nothing on this card** |

---

## 1. The geometry, read off the files

Measured with `tools/gguf_reader.py`'s block table on the GGUF headers in the HF cache (no tensor data read):

| model | file | tensor bytes | expert bytes | one expert | full-attn layers |
|---|---|---:|---:|---:|---:|
| Ornith-1.5-9B Q4_K_M | 5.77 GB | **5.769 GB** | — | — | 8 + draft |
| Qwen3.6-35B-A3B UD-Q3_K_M | 17.09 GB | 17.093 GB | **14.619 GB** | **1.33 MiB** | 10 + draft |
| Flint UD-IQ3_XXS | 14.53 GB | 14.530 GB | 12.912 GB | **1.02 MiB** | 49, window 513 on 35 |
| Flint-abliterated-Mix-12G | 11.99 GB | 11.986 GB | 10.723 GB | 0.85 MiB | as above |

Two of those numbers overturn assumptions in the brief:

- **Qwen3.6 has 2 KV heads, not 4** (`qwen35moe.attention.head_count_kv = 2`), so its query/KV group is **8**, not 4.
  Every `fdec` grid calculation below uses that, and the 4-head number the brief quotes is Ornith's.
- **Ornith ships a draft block**: `blk.32.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`, with
  `qwen35.nextn_predict_layers = 1` — so `n_layer_` is 32 trunk layers and the MTP window is live on it
  (`src/poly/engine.cpp:1483-1494`). Ornith's decode is therefore **speculative already**, and its acceptance is the
  unmeasured number the brief suspected.

---

## 2. DENSE — ranked work list

### The budget, decomposed

Two measured prefill points, one model, one binary, clock pinned. Fitting `per-token = a + b·ctx` to
40,960 → 1,942.2 and 131,072 → 986.4 tok/s:

| ctx | per-token | measured | attention share | of which `fdec`/`fattn` at peak |
|---:|---:|---|---:|---:|
| 40,960 | **515 µs** | 1,942.2 tok/s | **44%** (227 µs) | 54% of the 122 µs tensor roofline |
| 131,072 | **1,014 µs** | 986.4 tok/s | **72%** (726 µs) | 54% of the 390 µs roofline |

The attention slope is **5.537 ns/token/cell**, and it holds at **54% of the fp16 tensor roofline** at both points
(4·n_head·hd·ctx·8 layers = 5.4 / 17.2 GFLOP a token against 44.1 TFLOPS). A constant share at two contexts is the
signature of a term that is genuinely quadratic and genuinely compute-bound, so prefill on this model is
**attention majority at 131K** and the weight matmuls are the minority. That inverts the usual instinct and it is
why item 1 below is not `mul_mat_q`.

### 1. **`--prompt-batch` above 128: the weight term is paid per call, and 128 is small**

**Mechanism.** `rows_ = min(prompt_batch, max_context)` (`src/poly/engine.cpp:3057`) is the number of tokens one
`eval()`/`read()` call carries; `Mmq` and the FFN then read **every weight once per call** and apply them to all
`rows_` tokens. At `rows_ = 128` the weight read is amortised over 128 tokens; at 1,024, over 1,024. Nothing else in
the dense path changes — the MMQ gate is `n >= kStageRows = 32` (`src/poly/engine.cpp:3608`,
`src/poly/runtime.hpp:30`), so the path is already the fast one at 128 and only the amortisation improves.

**Arithmetic.** Ornith's weights are 5.769 GB, so at `rows_ = 128` the read costs `5.769/128/288e9 = 156.5 µs` a
token and at `rows_ = 1024` it costs **19.6 µs** — a saving of **137 µs** a token.

| ctx | per-token now | after | tok/s now | after | gain |
|---:|---:|---:|---:|---:|---:|
| 40,960 | 515 µs | 378 µs | 1,942 | **2,646** | **+36%** |
| 131,072 | 1,014 µs | 877 µs | 986 | **1,140** | **+16%** |

**What it costs.** `rows_` drives per-token scratch (`src/poly/qwen35.cpp:318-345`,
`src/poly/engine.cpp:3033`), and the measured 27B ceiling says that scratch is real: `--prompt-batch 512` tops out at
92,160 cells against 128's 102,400 (`docs/Q8_KV.md:347`). That 10,240-cell delta at 16 full-attention layers ×
1,664 B is **260 MiB**, so the **measured** scratch is **0.68 MiB a token** — the header's "about 1.4 MiB a token"
(`src/poly/engine.hpp:75`) is 2× the measured slope, and that discrepancy is worth its own five minutes
(§5). At 0.68 MiB/token, `--prompt-batch 1024` costs **609 MiB** more than 128, which on Ornith at 9 KV-carrying
layers × 1,408 B is **≈50,400 cells** of context. At the header's 1.4 it is **≈103,800 cells**, i.e. the whole
window.

**The measurement.** One binary, clock pinned, `--window 1` (so `snap_` and the draft block do not confound the
scratch), k6v4, one 40,960-token prompt and one 131,072-token prompt built from this repo's docs, three interleaved
runs an arm:

```
flock -w 60 /tmp/opencode/gpu.lock timeout 900 build-poly-local/strata-poly \
  --model <Ornith Q4_K_M> --max-context 262144 --kv k6v4 --window 1 --no-draft --lookup 0 \
  --prompt-batch {128|256|512|1024} --tokens-file <ids> --max-new 1 --temperature 0
```

Wall time **≈25 min** (four arms × two lengths × three runs at 1,942-986 tok/s plus three 30-130 s loads).
**Decision rule:** keep the largest `--prompt-batch` whose 131,072 prefill beats 128 by ≥10%; if 512 and 1,024 are
within 2% of each other, take 512 (it costs less context). **Revert** if any arm writes different tokens on a forced
greedy 64-token run from the same prefix — the batch must not change bits (the MMQ path's own contract,
`docs/PREFILL-GAP.md` §5b's byte-identical gate).

**What could go wrong.** Three things. (a) 44% of the 40K token is already attention, so the +36% is an upper bound
that assumes the weight read is fully overlapped with nothing else — it is not, and the real number will be lower.
(b) The scratch estimate is the tree's, not this model's: if Ornith's per-token buffers are bigger than 0.68 MiB the
context loss is worse and `--prompt-batch 1024` may not load at 262K at all. (c) `read_rows_` is **already 32,768**
(`src/poly/engine.cpp:3058`: `kReadBytes = 512 MiB` / (n_embd × 4) = 32,768, `src/poly/runtime.hpp:37`), so
`read()` may already batch internally — **if it does, this item is worth much less than the arithmetic above and the
measurement will say so immediately.** That is the single most important thing to check before spending the day.

> **Measured 2026-11-13: the gain is +7.2%, not +36%.** Ornith-1.5-9B at 40,960 tokens, `--kv k8v4`, one warm run an
> arm: `--prompt-batch 128` → **1,717 tok/s**, 512 → **1,833**, 1,024 → **1,841** (three repeats an arm, 0.5% spread
> within an arm, 6.7% between arms). That is **+7.2% from 128 to 1,024**, against the +36% predicted above. The
> direction and the mechanism are right — cause (c) above is part of why the magnitude is not — but **treat +7.2% as
> the number and 2,646 tok/s as refuted.**
>
> **Caveat on that measurement, and it is not small:** it ran while a 12-core build was using the same box
> (`loadavg` 24), so **both** the prediction and the measurement are on a loaded machine and neither is clean. The
> relative comparison holds (all arms were equally loaded), the absolute tok/s do not. Re-run it idle before quoting
> a tok/s figure from this row.
>
> Related: the **1,942 tok/s** this item's table starts from did not reproduce at `--prompt-batch 128 --kv k8v4`
> (measured 1,717 / 1,712 / 1,722 — same loaded box). The number is left as published rather than replaced, and
> marked **unverified: format and batch not recorded with it** in `docs/PREFILL-GAP.md`. This is the same
> "arithmetic right, assumption wrong" trap as the three closed ideas in `docs/AUDIT-MERGES.md` §7: check the
> operating point a number was computed at before believing the ratio.

**Ecosystem:** nobody has published this for `strata-poly`. llama.cpp's equivalent knob is `-b`/`-ub`, and it is the
`-ub 512` default in that project for the same reason.

### 2. **`fdec`'s reader threshold as a function of the SM count and the format** — the brief's item, confirmed twice over

**Mechanism.** `fdec_from_ = 4096` is a literal (`src/poly/qwen35.cpp:395`). `fdec_kernel` takes 1,024 keys a block
(`kSplit`, `src/poly/fattn.cu:261`) and launches `dim3(splits, kv_heads)` (`src/poly/fattn.cu:768`), so the grid is
`ceil(cells/1024) × kv_heads`.

**Arithmetic.** On this card the block count is the whole story:

| model | KV heads | cells | blocks | blocks/SM | verdict |
|---|---:|---:|---:|---:|---|
| Ornith (group 4) | 4 | 4,096 | **16** | 0.47 | **half the card idle** |
| Ornith | 4 | 9,216 | 36 | 1.06 | first full wave |
| Ornith | 4 | 40,960 | 160 | 4.7 | fine |
| Qwen3.6 (group 8) | 2 | 4,096 | **8** | 0.24 | **a quarter of the card** |
| Qwen3.6 | 2 | 16,384 | 32 | 0.94 | first full wave |
| Flint (group 8, hd 128) | 4 | 4,096 | 16 | 0.47 | as Ornith |

So the wrong threshold bites **twice as hard on Qwen3.6** as on Ornith, and the brief's 2.04× measurement at
k8v4/4,096 is the Ornith case. The fix is one expression — `ceil(SMs / kv_heads) × kSplit`, with the measured
break-even (~4,500 for group 4) kept as a floor because full occupancy (9,216) is *above* break-even.

**The measurement.** The bench already takes any geometry (`src/poly/kv_read_bench.cpp`, `poly_fdec_threshold`):

```
build/poly_fdec_threshold 256 16 4 --positions 4095,4608,5120,6144,8192,9216,12288,16384 --itr 10 --flush
build/poly_fdec_threshold 256 16 2 --positions ...        # the Qwen3.6 group-8 case
```

Wall time **≈20 min** for both geometries, 8 positions, cold L2. **Decision rule:** replace `4096` with
`max(measured_floor, ceil(SMs/kv_heads) × 1024)` if the crossing for k8v4 moves by more than the bench's own
between-run spread (the tree's bench is ±3%, `docs/BENCH-INSTR-PATHS.md`, branch `bench/instr-paths`, §2). Keep the floor if the two disagree.
**What could go wrong:** k6v4 forces `fdec_from_ = 0` because `attend()` refuses six-bit keys with a `throw`
(`src/poly/qwen35.cpp:399`, `src/poly/kernels.cu`) — a new rule that routes k6v4 to `attend()` below the threshold
would **abort the process**, not run slower. The rule must stay format-aware.

**Ecosystem:** llama.cpp makes exactly this a per-arch, per-`ncols` decision
(`ggml/src/ggml-cuda/fattn.cu:701-709`), and TensorRT-LLM's XQA picks a **different KV tile for sm_89 than sm_90**
(`cpp/kernels/xqa/mha.cu:92-100`). Both treat the reader choice as a function of the card. This tree's is a literal.
**Partly solved externally; the constant is ours to measure.**

### 3. **`fdec`'s split count is itself a tunable, and 1,024 is not obviously right**

**Mechanism.** `kSplit` is a `constexpr` (`src/poly/fattn.cu:261`), so the split count is a compile-time function of
the context. But the kernel's cost per split is not flat: each split writes `kDecRows × (D+2)` floats of scratch and
`fdec_merge_kernel` re-reads every split (`src/poly/fattn.cu:641-660`). Halving `kSplit` doubles the blocks **and**
doubles the merge traffic.

**Arithmetic.** At 4,096 cells the attention is 227 µs a token at 40K. Halving `kSplit` to 512 takes Ornith from 16
blocks to 32 (0.94 waves) — near-full occupancy — and the added merge is `+16 splits × 4 kv_heads × 32 rows × 258
floats = 2.1 MB` a token, i.e. **7 µs at 288 GB/s**, against a kernel that presently runs at 54% of peak. The
optimistic bound is the 54% → 100% on the quadratic term at short context: at 40K, 227 → 122 µs, **+20% of the
token**. It is much smaller at 131K where 128 blocks already fill the card.

**The measurement.** Needs a rebuild with `kSplit` as a template/flag; then the same `poly_fdec_threshold` sweep at
`--positions 4096,4608,5120,6144` with both values. Wall time **≈40 min** including two builds.
**Decision rule:** keep the smaller split only if 4,096-6,144 improves by ≥5% and 16,384+ does not regress by ≥2%.
**What could go wrong:** the merge kernel is one block a (head, token) (`src/poly/fattn.cu:641`), so its launch cost
grows with splits at tiny contexts where the whole call is one launch — the 4,096 point may get *worse*, exactly as
`docs/QUANT_MICROBENCH.md` §8 records for `strata`'s chunk halving ("Halving the chunk doubles the waves and measured
*slower*", `docs/QUANT_MICROBENCH.md:605-607`). **That precedent is a real warning: the same experiment on the same
card, one engine over, lost.** Do this only after item 2, which is cheaper and strictly additive.

**Ecosystem:** no one has published a split-count sweep for a 34-SM card; the general result (flash-decoding's
occupancy-vs-splits trade) is `arXiv 2311.01282`, which does not name its hardware.

### 4. **Speculative decoding on Ornith: measure the acceptance before touching `--window`**

**Mechanism.** Ornith has a trained MTP block (`blk.32.nextn.*`, §1), so `--window 8` (of which the block guesses
3 by default) runs a verify window of up to 8 rows through the same `lin()`/attention path. A window of `w` tokens
costs one weight read for `w` tokens' worth of matmul — so it *should* be nearly free — but each guessed row also
re-reads attention over the whole prefix, and the dense path has no expert cache to thrash.

**Arithmetic.** With 94.7% acceptance and 3.72 tokens a round on `strata` at 256K (`docs/REAP.md:33`) a window is
worth ~3.7×. Ornith's acceptance is **unmeasured**. The bracket from this tree: 61.0% on a 2K code prompt, 73.3% on
a 128K unique prompt, 94.7% on a favourable long prompt (`docs/REAP.md:94-95`). At 3 guesses and 61% acceptance the
expected tokens a round is `1 + 0.61 + 0.61² + 0.61³ = 2.23`, i.e. **+123%** if the window were free; at 94.7% it is
**3.72, +272%**. The cost side: a window of 8 rows makes each of the 32 layers' matmuls an 8-row call rather than a
1-row one, which at 5.769 GB of weights is the *same* read amortised 8×, and 8 rows of attention instead of 1 — at
131K that last term is 8 × 726 µs = **5.8 ms**, which would swamp the 23 ms token. **This is why `--window 8` at long
context is not obviously good on a dense model and is the actual question.**

**The measurement.** `STRATA_SPEC_DEPTH=1` prints acceptance by draft depth (`docs/DETAILS.md:1448`), so one run
gives the number the brief says is missing:

```
STRATA_SPEC_DEPTH=1 build-poly-local/strata-poly --model <Ornith> --max-context 131072 --kv k6v4 \
  --tokens-file <32K ids> --max-new 256 --temperature 0 --window {1|4|8|12}
```

Wall time **≈30 min** for four arms (one warm-up each). **Decision rule:** keep the window whose decode tok/s is
highest at 32K *and* at 131K; if the two disagree, prefer the 131K winner and expose the other as a `--window` the
caller sets. **Revert** if any window changes the greedy answer at temperature 0 — with MTP the answer must be the
target model's (`docs/DETAILS.md:1448`: "the text for a seed does not change"). **What could go wrong:** the
`snap_` buffer is 3 copies of the recurrent state and is taken lazily on the first window of more than one token
(`docs/QWEN38.md:205-207`); on this card that is **448.9 MiB on the 27B** (`docs/QWEN38.md:93`) and it comes
straight out of the context ceiling. On Ornith it is smaller (`s_dim 128 × v_heads 32`), but it is the reason
`docs/Q8_KV.md:356` had to take its ceiling table at `--window 1`. **Measure the ceiling with the window you ship.**

**Ecosystem:** MTP is in the checkpoint, so EAGLE/Medusa-style draft heads add nothing (they solve a problem this
model does not have). EAGLE-2's dynamic trees **widen** the window, which is the direction this card dislikes; the
tree has already measured that raising acceptance 62.1% → 68.0% **did not move decode**
(`docs/DETAILS.md:128`). **Do not build a tree.**

### 5. **`mul_mat_q`'s dequantisation share — real, but the smallest of the dense five**

**Mechanism.** `mul_mat_q` is 39% of INT8 peak with **45% of its instructions on `I2FP`/`FMUL`/`FFMA`** around 3.8%
`IMMA` (`docs/PREFILL-GAP.md` §5b). A schedule that overlaps the dequant of tile *n+1* with tile *n*'s `IMMA` —
Marlin's discipline — cuts that without changing bits.

**Arithmetic.** The matmul is **not** the majority of a dense prefill token here: attention is 44% at 40K and 72% at
131K (§2), and the weight read is 156 µs of the 515 µs at 40K (item 1). That leaves the matmul arithmetic inside
`515 − 227 − 156 = 132 µs` at 40K, of which 45% is dequant, so the **absolute ceiling at 40K is ~59 µs = 11% of the
token** and at 131K it is smaller still. `1/(1 − 0.45/2) = 1.29×` on the matmul is right and it multiplies a term
that is already small. **Ranked fifth for that reason, not because the finding is wrong.**

**The measurement.** An Ada `J`/`occupancy` sweep first, since llama.cpp has **no Ada MMA config table at all** —
`ggml_cuda_mmq_get_config` sends 8.9 to `..._get_config_ampere` (`ggml/src/ggml-cuda/mmq.cuh`). `Mmq::fits` already
picks `J` by `mmq_get_nbytes_shared(c, cc) <= dev.smpbo` (`src/poly/mmq.cu:88-92`), so the sweep is a table edit plus
a prefill A/B. Wall time **≈1 day**. **Decision rule:** keep a table only if the 131,072 prefill gains ≥3% **and**
`native_grouped_parity` stays bitwise. **What could go wrong:** 99 KB of shared memory is the ceiling, and a second
in-flight tile may not fit at the tile size that wins today; a smaller tile raises the launch count, which fights
item 1.

**Ecosystem:** **yes, twice** — QServe's progressive quantisation (measured on **L40S, sm_89**, Apache-2.0) and
Marlin's scheduling (cc ≥ 8.0, "Ampere or Ada", Apache-2.0). Marlin's batch sweet spot (16-32) is a prompt batch,
which is exactly item 1's regime.

### Dense — the `--resident-budget-gib` trap, which ranks above both of the top two

**A configuration bug, found after this list was written, and it costs a REAP run most of its prefill.** It belongs
in the dense section because the same mistake is available to any `--resident-budget-gib` run, and it is worth more
than items 1 and 4 together.

**Mechanism.** `--resident-budget-gib` sets `o.resident_cpu_experts = true`
(`src/program/generate.cpp:3793-3794`). Separately, the **#2090 RAM guard** steps a *bare* `--prefill auto`'s
ceiling **32768 → 16384 → 8192** while `MemAvailable < pin + extra + 6144`
(`src/program/generate.cpp:9907-9935`, the `gfx12_chunk_step_down` loop). `pin_mib` counts every expert the cache
does not hold plus the 8192 loan, so a large model makes `pin` large and the guard trips. It prints
`prompt chunks stay at 8192 … (#2090)` when it does.

**What it cost on the shipped REAP-320 256K run.** The local (untracked) `strata-reap320.json` has
`--prefill auto --resident-budget-gib 34`, so it is a bare auto and the guard applies. With `pin ≈ 54 GiB` against
`MemAvailable ≈ 45 GiB` on this box, the prefill measured in `docs/REAP.md` ran at **chunk 8192**, not 16384 or
32768 — the guard prints `prompt chunks stay at 8192` when it steps down, so a run that shows that line is at 8192.

**The streaming term, from the geometry, with the arithmetic shown so it can be checked.** REAP-320 is 320 experts
× 48 layers = **15,360** experts, and `docs/REAP.md:34` gives **2,522** resident, so **12,838** are not. A chunk
≥ `kStreamAllMinDefault = 1024` tokens streams every non-resident expert of every layer
(`src/prefill/prefill.cpp:271`), so one chunk moves **12,838 × 4.43 MB = 53 GiB** over PCIe. At 256K that is 32
chunks ≈ **1,695 GiB**, and at a PCIe 4.0 ×16 link (≈ 26–32 GB/s in practice) **57–70 s** of pure streaming out of
the 209.8 s prefill. On a 288 GB/s device read the same volume is 6 s, which is why this is a *PCIe* term and not a
bandwidth one.

> **Unverified, and the two figures disagree:** an earlier note on this item gave **30.6 GiB a chunk and ≈ 80 s**,
> which implies about 7,417 non-resident experts — i.e. roughly 7,943 resident, not 2,522. I could not reproduce
> that pair from `docs/REAP.md`'s numbers and I am not going to state it as measured. The **53 GiB / 57–70 s** above
> is arithmetic from the geometry and is what this document should be read as; the **30.6 GiB / 80 s** pair needs
> its own source (a run log with the actual streamed-bytes counter) before it is quoted. Either way the conclusion
> is unchanged: a bigger chunk does not stream fewer bytes per token, it streams them in fewer passes with more
> reuse of what was staged, and that is what the 16K/24K arms would measure.

**An explicit `--prefill N` is not a bare auto** (`src/program/generate.cpp:3598-3599`: `bare_auto = v == "auto" &&
!(o.prefill_auto && env_max != nullptr)`), so the guard does not touch it. That is the way to test 16384 and 24576
on a model with a resident budget.

**Measured 2026-11-14: neither arm reached the prefill, because the budget left no VRAM for a chunk.** Clock pinned
at 2820 MHz, `--prefill <N>` explicit so the #2090 guard does not apply. At `--resident-budget-gib 34
--vram-reserve-mib 2048` the expert cache left **228 MiB (at 16384) and 209 MiB (at 24576)** free, while the prefill
device buffers want **3,714 MiB at 16384 and 5,571 MiB at 24576** (0.2267 MiB a token). Both exited immediately with
`prefill: device buffers for a chunk of N tokens do not fit`.

**No figure was produced at either size, and no conclusion about the ceiling follows.** The failure is the flag pair,
not the card: the cache was sized to take everything the reserve left, which is exactly what starves a chunk. The
arm that answers this question lowers `--resident-budget-gib` (or raises `--vram-reserve-mib`) until the chunk's
buffers fit, and then reports prefill tok/s, decode tok/s and both headrooms together — the decode will drop, and
that trade is the actual answer.

### Dense — what is NOT worth doing

| don't | measured reason |
|---|---|
| **KV `k5v3` / `k4v4` on Ornith** | The ladder is a **tie**: 43.46 (k8v4) vs 40.96 (k6v4) at 16,384 — k6v4 is *slower* in tok/s despite 15% fewer bytes, because at 9 full-attention layers the KV is 198 MiB of a 5,502 MiB model, and the decode is 87% of the weights-only roofline. k5v3/k4v4 (1,152 B/cell) would save another ~35 MiB. `IMMA.S4.S4` being native on Ada is irrelevant when the read is 1.5% of the bytes moved. **Dead on footprint, dead on speed.** |
| **a smaller KV format to speed up decode** | Same measurement: k6v4 reads **slower** than k8v4 at 4,096 cells (16.80 vs 15.85 ns/cell) and only wins 7% from 32K up (`docs/Q8_KV.md:325-331`). At 16,384 the tok/s says 43.46 (k8v4) against 40.96 (k6v4). Fewer bytes has never bought a faster read on this card. |
| **tensor-core `attend`** | `attend` is 560 scalar `FFMA` with zero tensor instructions, but the replacement is measured at **+2.8-3.5%** (`/tmp/opencode/wt-s6/docs/STRATA-QSA-CENSUS.md` §6, not in this worktree) and llama.cpp independently routes Ada decode **away** from MMA (`ggml/src/ggml-cuda/fattn.cu:701-709`). Fix the *threshold* (item 2), not the kernel. |
| **`--vram-reserve-mib`, `--vram-pool`, `--conversation-cache-mib`, `--prompt-cache` on a dense model** | `fill_expert_cache()` returns at `n_moe_ == 0` (`src/poly/engine.cpp:1855`), so the reserve and the pool have **no expert cache to size** — they only bound the attention caches, which `cover()` grows anyway (`src/poly/engine.cpp:1288`). Inert. `--prompt-cache` is a prefix cache: a different win (TTFT), never a per-token one. |
| **`--expert-swaps`, `--tail-skip`, `--cache-bias`, `--skip-weight`, `--stage-*`, `--cpu-share` on dense** | No experts exist. `--cache-bias` is refused outright for anything but `qwen35moe` (`src/poly/engine.cpp:1656`); the others are read only inside `moe()`/`fill_expert_cache()`. Setup does not write them for a dense family. **Zero, not small.** |
| **`--batch N` for throughput on one user** | Each sequence needs its own KV and state: at 131K, k6v4, Ornith needs **1,584 MiB a sequence** (§1 geometry), so `--batch 3` costs 4.6 GiB of a 12.5 GiB card and pushes the ceiling down by ~50K cells. It buys nothing for a single stream, which is what a coding agent is. |

---

## 3. MoE — ranked work list

**Nothing here is measured on this card.** Everything below is arithmetic from the geometry plus the mechanisms in
the code, and each item says what number the card would have to produce. The measured anchors are the tree's own,
from a **laptop with 1 GiB free and 8-12 GB cards** (`docs/BACKENDS.md:1855-1930`) — a different regime, and the
reason the first item is a measurement rather than a change.

### What a miss actually costs — read off the code, not guessed

The brief asks for this first, and the answer is: **depends on which of three things happens**, and they are
different code paths.

1. **A PCIe copy from page-locked RAM**, if `pool_` is up (the adaptive/VMM slot pool). `moe()` writes the missing
   expert's slot through `pool_take()` → `pool_read()` → `read_file()` over the mapped file or the pinned copy
   (`src/poly/engine.cpp:4030-4046`), and the token waits in the `wait` lambda spinning on `pool_state_`
   (`src/poly/engine.cpp:4200-4215`). The copy is `gu2 + d_bytes` ≈ **1.33 MiB** (Qwen3.6) or **1.02 MiB** (Flint).
2. **A CPU compute**, if the expert is not in VRAM and the CPU is asked: `cpu_experts(...)` runs the same three
   matrices on the expert pool with the AVX2/AVX-512 rows, overlapping the GPU's cached experts through the
   `meanwhile` callback (`src/poly/engine.cpp:4196-4216`). This is **not** a copy; it is compute, on DDR5, roughly
   1.33 MiB a miss at the CPU's row rate.
3. **A disk read**, if neither RAM nor the file cache holds it. On this box that is unlikely for Qwen3.6 (17 GB file,
   ≥64 GB RAM) — but Flint is 14.5-31 GB and the tree's `ram_tight_` test is
   `files + 5 GiB > ram_available()` (`src/poly/engine.cpp:1859`).

`--stage-mib`/`--stage-ahead`/`--stage-read-mib` only move **when** case 1's copy starts (before or after the route);
they cannot turn case 2 into case 1. `--cpu-share` decides **how much of case 1 is handed to case 2**
(`src/poly/engine.cpp:4140-4160` and `docs/BACKENDS.md:1928-1936`). **So the first question on this card is not
"which dial" but "is a miss a copy or a compute", and that is one `nsys` away** — it is item 1.

### 1. **Measure the miss path once, because every later decision depends on it**

**Mechanism / arithmetic.** At the levels this card can hold, most experts are resident, so most tokens miss nothing:

| model | dense part | expert blob | experts fitting in 12.5 GiB | resident |
|---|---:|---:|---:|---:|
| Qwen3.6 UD-Q3_K_M | 2.47 GB | 1.33 MiB | **7,558 of 10,496** | **72%** |
| Flint UD-IQ3_XXS | 1.62 GB | 1.02 MiB | **10,607 of 12,032** | **88%** |
| Flint Mix-12G | 1.26 GB | 0.85 MiB | 13,170 of 12,032 | **100% — `all_cached()` is reachable** |

(the arithmetic: 13,422 MiB usable − dense weights × 2⁻²⁰ − 1 GiB reserve, ÷ the per-expert blob. `all_cached()` is
`cached_ == n_moe_ × n_expert_`, `src/poly/engine.hpp:566`, and when true the engine takes `card_moe()`, routing on
the card with no host round trip at all, `src/poly/engine.cpp:4274`.)

So **Flint Mix-12G is the interesting configuration on this card**: it can be *fully resident*, which switches off the
entire expert-miss machinery. Qwen3.6 cannot (72%), and its miss rate is what decides whether `--tail-skip` and
`--cache-bias` matter.

**The measurement.** One run per model, `--max-new 1` after a warm prompt, with the engine's own counters:

```
STRATA_DECODE_TIMING=1 build-poly-local/strata-poly --model <Qwen3.6 UD-Q3_K_M> --max-context 32768 \
  --kv q8_0 --expert-cache 7558 --window 1 --tokens-file <2K ids> --max-new 256 --temperature 0
```

`/metrics` and the `INFO` line report `expert_slots`, `expert_cache_mib`, `vram_free_mib` (`src/poly/main.cpp:879-883`);
`Stats` carries `gpu_experts`, `cpu_experts`, `tail_experts`, `swapped_experts`, `cpu_ms`, `route_wait_ms`
(`src/poly/engine.hpp:196-208`). Wall time **≈20 min** a model. **Decision rule:** if `cpu_experts` is under ~2% of
`gpu_experts`, the miss path is not worth tuning and items 4-6 drop below item 3; if `route_wait_ms` is a large share
of the token, it is a copy problem and item 3 leads. **What could go wrong:** `--expert-cache N` sizes the cache by
*count*, and the count that fits is exactly what is unknown on a 16 GB card — ask for too many and `fill_expert_cache`
clamps silently; ask too few and you measure an artificial miss rate.

### 2. **The KV format *is* a slot-economics dial on a MoE model at long context — the brief's hypothesis, confirmed**

**Mechanism.** On this path the KV and the expert cache compete for the same 12.5 GiB. A smaller KV cell therefore
buys **experts**, and an expert is worth a miss. The engine already has the formats and the gate: `k8v4` is allowed
for `qwen35moe` **and** `flint` (`src/poly/engine.cpp:1641-1643`), `k6v4` for `qwen35moe` and `qwen35` only
(`:1645-1648`).

**Arithmetic.** Qwen3.6's cell is `2 × n_kv × hd × 2` = **2,048 B fp16** and **832 B at k8v4** (`kv_cell_of`,
`src/poly/qwen35.cpp:216-227`), across 10 full-attention layers:

| ctx | fp16 | k8v4 | freed | experts | share of the 7,558 resident |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 640 MiB | 260 MiB | **380 MiB** | 286 | **3.8%** |
| 131,072 | 2,560 MiB | 1,040 MiB | **1,520 MiB** | **1,143** | **15.1%** |
| 262,144 | 5,120 MiB | 2,080 MiB | **3,040 MiB** | **2,286** | **30.2%** |

Qwen3.6 is **28% short of full residency** at default settings (§1 table), so at 131K this closes more than half the
gap and at 262K it closes all of it. **If decode time is roughly linear in miss rate, taking the miss rate from 28%
to 13% is up to ~1.2×, and to 0 is the difference between `moe()`'s host round trip and `card_moe()`'s device-only
route** (`src/poly/engine.cpp:4274`) — a qualitative change, not a percentage.

**Note the sign: this is the opposite of the dense answer.** On Ornith the format is a tie (43.46 k8v4 vs 40.96
k6v4) because the KV is 1.5% of the bytes a token moves. On Qwen3.6 the same change is worth 15% of the cache.
**Same flag, opposite verdict, decided by `n_moe_`.**

**The measurement.** Qwen3.6 at 131,072 and 262,144, `--kv fp16` against `--kv k8v4` against `--kv k6v4`, holding
`--vram-reserve-mib` fixed, 128-token answers, and report **`expert_slots` from the `INFO` line beside tok/s**
(`src/poly/main.cpp:879-883`). Wall time **≈40 min** (three formats × two lengths; the 262K arms need the ceiling
checked first). **Decision rule:** the format wins if `expert_slots` rises by ≥10% **and** decode beats fp16 by ≥3%
at 131K. **What could go wrong:** at 32K the arithmetic says 3.8% and the launch-bound penalty the tree measured at
4,096 cells (k6v4 16.80 vs k8v4 15.85 ns/cell, `docs/Q8_KV.md:325-331`) can eat it — so **do not run this at 32K**.
And k6v4 on Flint would throw at load, so the sweep is Qwen3.6 only.

**Ecosystem:** the same conclusion as this tree's own dense-27B result (`docs/Q8_KV.md:347`: k6v4 buys **+15.6% to
+20% of context**), reached from the opposite direction — there the format bought context, here it buys slots. Nobody
has published the MoE version of that table.

### 3. **`--vram-reserve-mib` and `--vram-pool`: the same slot economics as `strata`, on the model everybody runs**

**Mechanism.** `reserve_mib_ = max(opt_.vram_reserve_mib, 256)` else `default_reserve_mib` =
`clamp(total_b >> 25, 256, 768)` **or** `clamp(total_b >> 24, 512, 1024)` when RAM is tight
(`src/poly/engine.cpp:536-539`, `:1862-1869`). On a 16 GiB card `total_b >> 25` = 512, so the default is **512 MiB**,
and every 512 MiB of reserve is `512 / 1.33 = 385` Qwen3.6 experts = **3.7% of the cache**. `--vram-pool` claims the
card once and lets the cache and the attention caches trade within it (`src/poly/engine.hpp:67-69`,
`src/poly/engine.cpp:1871-1885`), which is the analogue of `strata`'s pool and is **the** mechanism that stops the
KV growth from stealing slots one `cover()` at a time.

**Arithmetic.** On Qwen3.6, dropping the reserve 512 → 256 MiB is **+192 experts (+1.8% of the cache, +2.5% of the
resident set)**. The tree's slot economics from `strata` says a slot is worth **~19.7 µs a token** in the *measured*
1,574 → 2,522 pair, but that slope extrapolates to a negative intercept and must not be used as a per-slot rate
(`docs/ADA-ECOSYSTEM.md` §4.2 item 5). **The honest statement: the reserve is worth low single-digit percent and the
arithmetic cannot say more.** `--vram-pool` cannot be quantified at all without a measurement, because its value is
that it prevents a *cliff* (KV growth evicting experts mid-answer), not that it raises the mean.

**The measurement.** Two arms, Qwen3.6, 32K context, decode 256 tokens: `--vram-reserve-mib 512` (default) against
`256`, then the same with `--vram-pool`; and a 131K re-read to see whether the pool prevents the late-prompt
eviction. Wall time **≈30 min**. **Decision rule:** keep 256 if decode gains ≥3% and `vram_free_mib` never goes
below ~128; keep `--vram-pool` if the 131K arm's decode does not fall off its 32K rate by more than the 512 arm's.
**What could go wrong:** the reserve is also what the **recorded launches** take for their scratch
(`src/poly/engine.cpp:1860`), so cutting it too far converts a slow token into a failed capture or an OOM on the
largest call. And on a card driving a display, the desktop holds ~2 GiB and the whole calculation shifts.

**Ecosystem:** **yes** — KTransformers exposes expert placement as a first-class axis (`--kt-num-gpu-experts` **per
MoE layer**, `frequency`, `--init-expert-location`, `--kt-enable-dynamic-expert-update`), which is more than this
tree's one global count. That is the follow-on if item 2 pays: `fill_expert_cache` already sorts candidates by
`used/bytes` (`src/poly/engine.cpp:1897-1901`), so a per-layer quota is a change to that sort, not a new subsystem.

### 4. **`--tail-skip`: already measured, already chosen, and `--skip-weight` is the better dial**

**Mechanism.** An expert outside VRAM that **every** token of a call ranks at `N+1` or later is not computed
(`src/poly/engine.cpp:4142-4150`). `--skip-weight F` does the same by router weight instead of place
(`:4133-4141`). Both change the answer; the weights of the experts that remain are untouched.

**Verdict: this is not an open question.** The tree measured it on six prompts, two rounds, with KL against a
no-skip reference (`docs/BACKENDS.md:1855-1886`):

| model | `--tail-skip 0` | 7 | 6 | 5 |
|---|---:|---:|---:|---:|
| Qwen3.6 UD-Q3_K_M | 91.2 tok/s | 93.1 (×1.041, KL 0.0038) | **95.6 (×1.089, KL 0.0059)** | 97.8 (×1.116, KL 0.0096) |
| Tiel Coder UD-Q3_K_XL | 72.3 | 74.0 (×1.020) | 79.2 (×1.030) | 79.6 (×1.038) |

So **`--tail-skip 6` is measured, not inherited**: ×1.089 on Qwen3.6 for a KL of 0.0059, which is why `setup.py:630`
writes it (`"tail_skip": 6`, and `setup.py:6754-6763` adds it to configs written before the flag existed). And
`--skip-weight` **beats it at matched KL**: on GLM 5.3 Flash at a mean KL of ~0.031, `--tail-skip 6` is 15.6 tok/s
and `--skip-weight 0.08` is **17.1** (`docs/BACKENDS.md:1898-1912`), i.e. **+10-12% at the same answer movement**,
and at 0.095 it moves the answers *less* than `--tail-skip 5` while leaving out 17% more experts.

**What to do on this card.** Re-measure the pair on Qwen3.6 at 72% residency, where the miss rate is far lower than
the 8 GB laptop's and the tail skip should be worth **less**, not more: `--tail-skip {0,6}` × `--skip-weight
{0,0.06,0.08}` at 32K, 200 tokens, following `docs/BACKENDS.md`'s method (force the no-skip answer's tokens, compare
the 40 most likely). Wall time **≈40 min**. **Decision rule:** switch the default to `--skip-weight` only if it
beats `--tail-skip 6` by ≥5% at a mean KL no worse. **What could go wrong:** at 72% residency most misses are
already rare, so both dials may measure as noise — in which case **the honest answer is to leave setup's 6 alone**
and not spend the day. Accuracy contract: both are documented as answer-changing (`docs/GLM.md:282`), and
`--tail-skip 0` must remain byte-identical.

### 5. **`--cache-bias`: the biggest published MoE win, and the one with a quality bill**

**Mechanism.** Adds `F` to the router score of every expert **already in VRAM** before a decode-time pick, so a
near-tie goes to the cached expert (`biased()`, `src/poly/engine.cpp:4000-4003`; refused for non-`qwen35moe` at
`:1656`). The weights stay the softmax of the router's own scores, so only the *selection* changes
(`src/poly/engine.cpp:4095-4101`).

**Arithmetic.** The tree's numbers are from an RTX PRO 6000 held to 12 GB with Qwen3.6 UD-Q3_K_M — **the closest
existing measurement to this card's regime**: the share of experts read from VRAM went **60% → 80% (F 0.5) → 90%
(F 1.0)**, and decode **196 → 234 → 259 tok/s** on a short prompt and 165 → 227 → 202 after a long one
(`docs/BACKENDS.md:1920-1926`). Top-token agreement stayed 126/128 and 121/128 at F 0.5, 124/128 and 120/128 at F 1.
On **this** card Qwen3.6 is already at **72%** resident, so the headroom is the gap from 72% to 90%, not from 60%.
A crude scaling — the tree's own 60→80 step was +19% decode — puts F 0.5 at **low single digits to ~10%** here, and
that is an estimate, not a measurement.

**The measurement.** Qwen3.6, 32K context, `--cache-bias {0, 0.5, 1.0}`, 200-token answers, forcing F=0's greedy
tokens and comparing the top token each step (the method of `docs/BACKENDS.md:1917-1920`). Wall time **≈45 min**.
**Decision rule:** keep F 0.5 only if decode gains ≥5% **and** top-token agreement stays ≥96%. **What could go
wrong:** this is a **quality** change, not just speed — setup does not write it, and `docs/BACKENDS.md` records that
the idea's provenance is another engine's simulated cache. At 72% residency the bias has less to gain than the
12 GB card had and the same quality bill, so the bar should be higher, not lower.

### 6. **`--stage-ahead` / `--stage-mib` / `--stage-read-mib` for prompt turns — the turn, not the token**

**Mechanism.** A call of fewer tokens than `--stage-ahead` (default 1,024) routes first and copies the experts it
routed to; a call at or above it starts the copies before the route (`src/poly/engine.hpp:84-88`). In a **read of
several calls** the copies are amortised across calls (`src/poly/engine.cpp:5444-5456`). `--stage-mib` (0 → 128 on
CUDA with VMM, else 2048) is the buffer the missed experts pass through, and the wide/narrow staging lets a read
borrow VRAM from the cache and give it back (`widen_stage()`/`narrow_stage()`,
`src/poly/engine.cpp:2188-2228`).

**Arithmetic.** The measured cost on the laptop was **not** the token but the turn: with `--cpu-share 0` vs the
default, a 150-token turn went **808 → 427 ms** on Qwen3.6 and a 300-token turn **1,566 → 826 ms**
(`docs/BACKENDS.md:1941-1953`) — i.e. **the copy dominated a short prompt and the CPU share halved it**. That is the
same mechanism that makes a coding agent feel slow between tool calls. On this card, with 72% resident, the copies
per turn are `0.28 × 8 experts × layers` distinct experts, so the number to beat is the same table measured here.

**The measurement.** The `--cpu-share {0, auto}` × `--stage-ahead {256, 1024, 4096}` grid on a **conversation** as
`docs/BACKENDS.md:1938` describes: a 16,000-token first prompt then turns of 150/300/500/800 tokens, each answered
with 8. Wall time **≈50 min** — the largest item here, and the one that most closely matches how the model is
actually used. **Decision rule:** keep the setting with the lowest **median turn time**, not the highest tok/s.
**What could go wrong:** `--cpu-share` is a **lossy** path in the same sense `--tail-skip` is — the CPU's rows are
computed in a different activation format, so the first token moves (KL 0.006 mean, `docs/DETAILS.md:1795-1800` for
the `strata` equivalent), and `auto` picks from wall-clock timings so the same request can answer differently run to
run. A user who needs determinism must use `0`.

### MoE — what is NOT worth doing

| don't | measured reason |
|---|---|
| **`--kv k6v4` on Flint** | **Refused at load**: `--kv k6v4` is gated to `qwen35moe` and `qwen35` only, because Flint's attention keeps its own caches per layer and "needs a reader this branch does not have" (`src/poly/engine.cpp:1645-1648`). Not a tuning question. |
| **`--batch N` for throughput** | Each sequence's KV is its own (`src/poly/engine.cpp:3086`, `copy_seq`), and the expert cache is **shared**, so N sequences route to N× the distinct experts a token and the 72% residency falls. The tree already requires `--expert-swaps 0 --cpu-share 0 --window 1 --tail-skip 0` to make a batched run reproducible (`docs/BATCHING.md:53`) — i.e. it has to switch off every adaptive mechanism, which is most of what makes the model fast here. |
| **`--conversation-cache-mib` to make tokens faster** | Parking is prefix reuse: it cuts **TTFT on a follow-up**, never the per-token rate, and on a 17-31 GB model the parked state is **RAM** (the store is host-side; `src/poly/main.cpp:326-327` keeps `min_free_mib = 2560` free). On this box with ≥64 GB it is affordable but it is a latency feature, not a throughput one. |
| **`--tail-skip` above 6 on Qwen3.6** | Measured: 5 gives ×1.116 but KL 0.0096 and 93.7% top-token agreement, against 6's ×1.089 at KL 0.0059 and 94.5% (`docs/BACKENDS.md:1862-1866`). The extra 2.7% is not worth 60% more answer movement. And **above 8 there is nothing to skip** — a token has 8 experts. |

---

## 4. Where the two paths need different defaults

This is the part a single flag set cannot express, and it is the same three places every time.

### 4.1 `--prompt-batch`

**Dense wants it large; MoE wants it small, and for opposite reasons.**

- Dense: raising it 128 → 1,024 saves **137 µs a token** of weight re-read (item 1), worth **+36% at 40K**. The
  weights are 5.8-10.9 GB and are re-read by every call; amortising them is nearly pure profit.
- MoE: raising it **costs VRAM that the expert cache wants**, and on Qwen3.6/Flint the weights are *not* re-read per
  call in the same way — a prompt batch stages the experts it routes to, once per layer, and the tree records that
  going from 128 to 512 MiB of staging **traded 1,166 cached experts for 1,013 on a laptop**
  (`src/poly/engine.cpp:3196-3200`). With Qwen3.6 at 72% residency, every MiB of scratch is **0.75 experts**; 128
  MiB of extra scratch is ~96 experts, and the whole Qwen3.6 cache is only 7,558.
- **So: dense → 1024 (measure it); MoE → keep 1024 only if `cached_experts` does not fall.** The number that decides
  is `expert_slots` on the `INFO` line (`src/poly/main.cpp:879`), not tok/s.

### 4.2 `fdec_from_`

**A function of `kv_heads`, and the models differ by 2×.** Ornith is group 4 (4 KV heads), Qwen3.6 is group 8
(2 KV heads), Flint is group 8 at head_dim 128. The grid is `ceil(cells/1024) × kv_heads`, so the *same* context
gives **16 blocks on Ornith and 8 on Qwen3.6** — the threshold that is wrong for one is twice as wrong for the other.
Any replacement for the literal 4,096 must be computed from `ceil(SMs / n_kv_) × kSplit` and not from a constant, and
`fdec_fits` already gates on the group (`src/poly/fattn.cu:743-752`, groups 4/6/8/12/16), so the information is
present at the decision point.

### 4.3 The KV format is a tie on dense and a cache dial on MoE

**The same flag, opposite verdicts, and the reason is `n_moe_`.**

| | dense (Ornith, 9 KV-carrying layers, group 4) | MoE (Qwen3.6, 10 full layers, group 8) |
|---|---|---|
| fp16 cell | 4,096 B | 2,048 B |
| KV at 131K | 3,744 MiB (k8v4) / 3,168 (k6v4) | 2,560 MiB fp16 / 1,040 k8v4 |
| what the saving buys | nothing but context, and the model already fits | **1,143 experts, 15% of the cache** |
| measured decode | 43.46 (k8v4) vs 40.96 (k6v4) — **k6v4 slower** | unmeasured |
| verdict | **leave it on k8v4; the format is not worth tuning** | **tune it: it is a residency dial** |

The asymmetry is structural, not a quirk: on a dense model every token reads all 5.769 GB of weights, so 500 MiB of
KV is 1.5% of the traffic; on a MoE model a token reads ~0.6% of 12.9 GB of experts, so the *miss rate* is what the
KV bytes change. **A single `--kv` default cannot be right for both**, and setup currently has no per-family KV
default at all.

### 4.4 The expert dials, which exist only on one path

`--tail-skip`, `--skip-weight`, `--cache-bias`, `--expert-swaps`, `--stage-*`, `--cpu-share`, `--vram-reserve-mib`,
`--vram-pool` are **all inert on dense** (`fill_expert_cache()` returns at `n_moe_ == 0`,
`src/poly/engine.cpp:1855`; `--cache-bias` is refused outright). Setup already carries this per family
(`setup.py:630`'s `tail_skip: 6` is on `qwen36`, not on a dense family). **The reverse is the gap**: nothing in setup
sets `--prompt-batch` per family, and item 1 says a dense family wants a bigger one than a MoE family. That is the
one place where a per-family default is missing rather than merely different.

---

## 5. Needs the card, in the order that answers the most

| # | what | command | wall time | answers |
|---:|---|---|---|---|
| 1 | **does `read()` already batch to 32,768 internally?** Read the `read()` path first; if yes, item 1's arithmetic collapses and the day moves to item 2. | code only | 10 min | whether the dense top item is real |
| 2 | **`--prompt-batch` 128 vs 512 vs 1024**, `--window 1`, k6v4, 40,960 and 131,072 | §2 item 1 | 25 min | +16..+36% or nothing |
| 3 | **`fdec` crossing for group 4 and group 8**, k8v4 and q8_0 | §2 item 2 | 20 min | the threshold expression |
| 4 | **Ornith's MTP acceptance by depth** at 32K and 131K | `STRATA_SPEC_DEPTH=1`, §2 item 4 | 30 min | whether `--window 8` pays at long context |
| 5 | **MoE miss path**: `cpu_experts` vs `gpu_experts`, `route_wait_ms`, for Qwen3.6 at 7,558 slots and Flint Mix-12G fully resident | §3 item 1 | 20 min a model | whether items 4-6 of the MoE list matter at all |
| 6 | **KV format on Qwen3.6**: fp16 / k8v4 / k6v4 at 131K and 262K, `expert_slots` beside tok/s | §3 item 2 | 40 min | whether the format buys 15-30% of the cache |
| 7 | **`--vram-reserve-mib 512` vs `256`** and `--vram-pool`, Qwen3.6, 32K and 131K | §3 item 3 | 30 min | low single digits, and whether the pool prevents the late-prompt cliff |
| 8 | **`--tail-skip 6` vs `--skip-weight 0.08`** on this card's 72% residency | §3 item 4 | 40 min | whether setup's default should change |
| 9 | **`--cache-bias 0.5`** with the top-token gate | §3 item 5 | 45 min | the biggest MoE win, and its quality bill |
| 10 | **`--cpu-share`/`--stage-ahead` on a conversation**, 150-800-token turns | §3 item 6 | 50 min | turn latency, which is what a user feels |
| 11 | **`kSplit` 512 vs 1,024** at 4,096-6,144 cells | §2 item 3 | 40 min incl. builds | whether occupancy beats the merge |

**What I could not verify:** Ornith's and Qwen3.6's per-token scratch (the tree's 0.68 MiB/token is derived from a
27B ceiling in `docs/Q8_KV.md:347`, not from these models); Ornith's MTP acceptance; every MoE number on this card;
whether `read()`'s 32,768-token window is used for a single prompt or only for multi-call reads
(`src/poly/engine.cpp:3058` sets `read_rows_`, but `main.cpp:1125` feeds the prompt `max_rows()` at a time, so the
two may not meet).
