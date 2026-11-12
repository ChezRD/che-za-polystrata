# Quantized-operation microbenchmarks: the Phase 0 gate table

Before any quantized format was wired into a long-context decode loop, each operation the optimization work would
touch was measured on its own against a documented hardware peak. This page is that record: the eight rows, what each
one measured, its value as a fraction of the peak, whether it passed, and the decision it unlocked.

Every number below was measured on **one NVIDIA GeForce RTX 4060 Ti 16 GB** (AD106, sm_89, 34 SMs, 32 MiB L2,
driver 615.78.08, CUDA 13.4) on 2026-11-12, with the SM clock pinned at 2,820 MHz. Nothing was measured on hardware
this machine does not have. The commands are in section 7; the measurement protocol is in section 5, and it is the
reason the numbers can be compared at all.

**Three of the eight pass criteria in the plan turned out to be unreachable by arithmetic, not by the kernels, and
are revised here.** Each is marked *revised after measurement* in the gate table, with the arithmetic and the
corrected bar:

- **row 1** counted only the *write* half of a read-modify-write, so its 173 GB/s bar needed 3.6× the card. My first
  replacement — "75% of DRAM bandwidth counting the source read" — was wrong too: it charges L2-served reads as DRAM.
  The row is judged on the result it exists for: **28.4% less time a cell** than q8_0's store, for 23.5% fewer bytes.
  It is not bandwidth-saturating, and §2.0 says why.
- **row 2** asked for an L2 speed-up from a 4K call that is launch-bound — it measures the grid, not the cache.
  Revised to take the L2 evidence where it actually appears, at 32K.
- **row 5** asked for a bandwidth fraction at 1–8 rows, where the kernel is a launch and nothing else (1,659 ns of
  pure launch at one row). Revised to the row's own stop-rule outcome: fusing must be free, and it is.

The measurements themselves were sound; the denominators were not. The gate table shows both the original bar and the
corrected one, so a reader can see what was actually decided.

**Row 3 then turned out to be the wrong question, and that is a bigger correction than a bar.** Its brief named
"int8 (LeapQuant's floor)", so the first pass measured one of the format's three bit-widths and treated int8 as the
interesting end. It is not: the state step has no matrix multiply in it, so the format is simply "cheapest encoding per
element that keeps accuracy", and **plain fp16 beats int8 on time, accuracy and the slots-per-byte trade** (§2.1). The
row is now measured across eight formats and reports the one that wins.

**Four figures in this page's first version were also wrong** — row 2's read was the best of five runs rather than the
median, row 1's was one of the store's two kernels, and both of row 8's numbers (its roofline fraction and the −8.5%
attributed to it) were misread. All are corrected below and the corrections are recorded in section 9, because a page
that states a measurement rule should show where it failed to follow it.

## 1. The eight rows

Bandwidth-bound rows give the fraction of both denominators: **288.0 GB/s**, which is 128-bit GDDR6 at the card's
9,001 MHz maximum memory clock (2 × 9,001 MHz × 128 bit ÷ 8), and **280.0 GB/s**, which is 8,751 MHz — the memory
clock the card reports while loaded. The second is 3% lower and is the honest one for a sustained run; the first is
what the plan's tables use.

| row | operation | metric | measured | % of 288 (280) | verdict | unlocks |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | k8v4 KV **store**: q8_0 keys + Hadamard-rotated Q4_0 values, 832 B a cell | GB/s of traffic (source read + KV write), and ns a cell | **35.2 ns a cell**; **23.6 GB/s** of KV bytes written; **140.0 GB/s** of traffic | **8.2%** of pure KV bytes; **48.6% (50.0%)** of traffic | **PASS on its own criterion, not on bandwidth**: the byte saving reaches memory as **28.4% less time** a cell than q8_0. The plan's 173 GB/s bar is unreachable (§2.0); 48.6% of traffic is what a 4:1 read-amplifying quantizer gives | the store is 23.5% smaller and 28.4% faster than q8_0's (the *read* is a different answer, row 2) |
| 2 | KV **read**: decode attention over the k8v4 cache | GB/s of cache read at 256K | **141.5 GB/s** (5.88 ns a cell) | **49.1% (50.5%)** | **FAIL** the 60% bar by 11 points; the row's finding stands | 5.88 ns a cell at 256K, which sizes LSA's wider-list penalty |
| 3 | GDN recurrent state, 8 formats from int8 to fp4 | µs a layer at T=1, MiB a session, and relative state error at 1 and 128 steps | **see §2.1**: **fp16 is the winner** — 0.88× fp32's time for half the bytes | — (bandwidth is not the limit; §2.1) | **PASS for fp16 (footprint at near-zero time cost); NOT RUN for nothing else** | **the answer is fp16, not int8** — see §2.1 |
| 4 | the paper's full form: compensators + window buffer | µs a layer, MiB a session | **measured for int8 and 4-bit** (A2/A3/A16/A17) | — | **carrying them costs 8–33%, and their accuracy benefit is NOT measured** (§2.1) | nothing further until a compensator fit exists |
| 5 | Q8_1 activation quantization, 2,560-value rows | GB/s of activations in + int8 out | **64.5 GB/s at 8 tokens; 8.1 GB/s at 1** | **22.4% (23.0%) at 8; 2.8% at 1** | **FAIL** at every width the engine runs; the stop rule fires | confirms it must be **fused**, not optimized |
| 6 | IQ3_S / IQ4_NL expert dequant + dot, the reference | int8-equivalent TOPS at R=626 | **55.5 TOPS** (1.773 ms), documented 55.5 | 63% of the 88 TOPS int8 peak | **PASS** (0.0% off) | the truth serum for the phase: the box reproduces its own numbers |
| 7 | NVFP4 E2M1/UE4M3 decode, software path | GB/s of weights | **252.3 GB/s**, hash `054f38e43e6cdb4e` | **87.6% (90.1%)** | **CONFIRMED** (cited, not re-derived) | the calibration for how a software-decoded 4-bit format behaves here |
| 8 | LSA centroid scan, Q·Cᵀ with d=64, 32×32 tiles | TFLOP/s of fp16 `mma.sync` | **4.05–5.66 TFLOP/s** | 9–13% of the ~44 TFLOP/s fp16 peak; **88% of its own 4.6 TFLOP/s DRAM roofline** at the 199 µs end | **FAIL** the 11 TFLOP/s bar, which the geometry makes unreachable | it sizes the scan term: 1.2–1.6 s, i.e. **0.6–0.8%** of a 209 s prefill, not the −8.5% the bar assumed |

Row 6 is the one that decides whether the others can be believed: it re-measures a number the tree already documents
and reproduces it exactly. Row 4 is shown rather than omitted because the plan makes it conditional, and a gate table
that silently drops a row reads as if the row passed.

## 2. Where the rows landed, and the three revised bars

### 2.0 The arithmetic behind each revision

**Row 1 — the bar counted half the work.** Writing a k8v4 cell is a read-modify-write: 832 B written, but the cell's
floats must be **read** to quantize them, which is 4,096 B (512 values a side, twice, 4 B each). KV bytes are
therefore **16.9% of the operation's traffic**, and even at an impossible 100% of the card's 288 GB/s they top out at
**48.6 GB/s**. The plan asked for 173 GB/s, which would need the operation to move 1,025 GB/s — **3.6× this card's
DRAM bandwidth.** That bar was never a statement about the kernel.

**The corrected bar — and my first correction was wrong too.** I replaced it with "≥ 75% of DRAM bandwidth counting the
source read", and the store measures **48.6%** of that. The revision was still wrong, for a different reason: in the
engine the source floats are the projection's fresh output and are hot in L2, so charging all 4,096 B of read as DRAM
traffic overcounts. Under `ncu`, which flushes caches and so puts the source in DRAM, the value store alone moves
**239 GB/s = 83% of peak**; with the source in L2 the whole store moves **140.0 GB/s = 48.6%** of the same bytes, and
the KV writes themselves are only **23.6 GB/s = 8.2%**. All three are correct measurements of different things, and
none of them is a bar the row should be judged on.

**What the row actually delivers.** It exists to answer one question — does k8v4's 23.5% byte saving over q8_0 reach
memory as time — and it does: **35.2 ns a cell against q8_0's 49.2, five interleaved runs, 28.4% less time.** The store
is smaller and faster. It is **not** at the card's bandwidth limit, and the reasons are the quantizer's 4:1 read
amplification (4 B a value in, 1 B out) and its ALU, not the write path: `ncu` puts the value store at 74% DRAM and the
key store at 28%, and the bench puts the two together at **48.6%** of the traffic they move. That is the honest result,
and it is enough to keep k8v4; it is not enough to call the row saturating.

**Row 2 — the 4K test measured a launch.** At 4,096 positions the decode kernel launches **10 CTAs on 34 SMs**, i.e.
0.29 of a wave, and its 0.064 ms is the launch and tail of a nearly empty grid. Asking for "≥ 2× the DRAM-bound rate"
there measures the grid, not the cache; the 4K rate (53.3 GB/s) is in fact *lower* than 256K's. The L2 hit is real and
appears where there is enough work to show it: **at 32K the read reaches 373 GB/s = 130% of the DRAM peak**, on a
cache that still fits L2. Corrected criterion: **take the L2 evidence at 32K.**

**Row 5 — the bar is met only at a width the engine never runs.** The quantizer reaches **39.5%** of peak at 16 rows,
**22.4%** at 8 and **2.8%** at one. The engine's verify window is at most 8 rows, and the one-row call is
**1,659 ns, which is 100% of its time** — a launch, not a memory operation. The plan's own stop rule covers exactly
this ("below 15% → it is launch-bound and should be fused, not measured again"). Corrected criterion: the row is
satisfied when **fusing costs no more than the unfused launch**; the fused SwiGLU+quantize measures **1,655 ns against
1,659 ns**, so it is free.

### 2.1 Row 3: fp16 is the answer, and the format question was never about int8

The brief for this row named "int8 (LeapQuant's floor)", which measured **one** of the paper's three bit-widths and
assumed int8 was the interesting end of the range. **It is not, and the framing was the error.** Nothing forces the
state through int8 at all: the step is an elementwise rank-1 update with no tensor cores, so the only question is
*cheapest encoding per element that keeps accuracy*. An fp16 store needs **no scale, no quantize and no dequantize** —
one `cvt` in and one out — where int8 needs a per-row scale multiply in and a divide out. So the row was measured
across **eight formats**, including the float ones that the int8 framing had excluded.

Sixteen arms, five interleaved runs each, SM clock pinned at 2,820 MHz (dmon shows 2,820/8,751 in every sample), and
the run-to-run spread was **1.00–1.02×**. Accuracy is the relative Frobenius error of the decoded state against an
fp32 reference after 1 and 128 update steps.

| format | B/value | µs a layer | × fp32 | MiB a session | slots in 864 MiB | err @1 | err @128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **fp32** (reference) | 4.00 | **15.0** | 1.00× | **108.0** | **8.0** | — | — |
| **fp16** | 2.00 | **17.0** | **0.88×** | **54.0** | **16.0** | **3.8e-04** | **2.2e-02** |
| fp16, fp32 accumulate | 2.00 | 17.6 | 0.85× | 54.0 | 16.0 | — | — |
| bf16 | 2.00 | 25.0 | 0.60× | 54.0 | 16.0 | 2.3e-03 | 1.5e-02 |
| fp8 E4M3 | 1.00 | 28.3 | 0.53× | 27.0 | 32.0 | 3.5e-02 | 6.5e-01 |
| int8 | 1.00 | 20.0 | 0.75× | 27.8 | 31.0 | 9.5e-03 | 1.1e-01 |
| int8, fp16 arithmetic | 1.00 | 22.4 | 0.67× | 27.8 | 31.0 | — | — |
| int8, fp16 + fp32 accumulate | 1.00 | 25.5 | 0.59× | 27.8 | 31.0 | — | — |
| int8 + 4 compensators | 1.00 | 21.8 | 0.69× | 31.2 | 27.7 | 9.5e-03 | 1.1e-01 |
| int8, fp16 arithmetic + compensators | 1.00 | 32.4 | 0.46× | 31.2 | 27.7 | — | — |
| int8 + window buffer | 1.00 | 20.0 | 0.75× | 54.8 | 15.8 | — | — |
| 6-bit int | 0.75 | 31.2 | 0.48× | 21.1 | 41.0 | — | — |
| **fp4 E2M1 + UE4M3/16** | 0.56 | 125.3 | 0.12× | 15.2 | 56.9 | 1.1e-01 | **1.56** |
| int4 | 0.50 | 30.4 | 0.49× | **14.3** | **60.3** | 1.3e-01 | **1.74** |
| 6-bit full form | 0.75 | 42.4 | 0.35× | 51.5 | 16.8 | — | — |
| 4-bit full form | 0.50 | 41.9 | 0.36× | 44.7 | 19.3 | 1.3e-01 | 1.74 |

**The winner on both axes is fp16.** It halves the state (108 → 54 MiB a session, 8 → 16 slots) for **12% more time a
layer** — 2 µs at T=1, or **0.07 ms a token over 36 layers**, against a decode token of about 34 ms. That is **+0.2% of
decode for double the slots.** Nothing else in the table comes close to that trade: every 1-byte format is 25–53%
slower *and* less accurate, and every 4-bit format is both slower and, at 128 steps, has **more error than signal**
(1.56–1.74 × the reference norm).

**int8 is beaten by fp16 on every axis that matters.** Same-ish time class (0.75× vs 0.88× — fp16 is *faster*), worse
accuracy (9.5e-03 vs 3.8e-04 at one step, 25× worse), no footprint advantage worth the trade (31 vs 16 slots at
+25% time). The int8 family needs compensator tokens to be plausible and still does not reach fp16's error.

**fp8 E4M3 is the interesting failure.** At exactly int8's byte cost it is **3.6× less accurate at one step and 5.8×
at 128** — because E4M3's 4-bit exponent gives it only 3 mantissa bits, and the state's per-row outliers eat them. A
float format buys dynamic *range* but not dynamic *precision*, and this state needs the latter. It also has no block
scale here; a scaled fp8 (E4M3 with a per-row scale, i.e. exactly what int8 does) would be the fair comparison for
fp8 and is not in this table.

**bf16 is worse than fp16 on both counts** (0.60× vs 0.88×, and 6× the error at one step) — it has fp16's byte cost
with 8 fewer mantissa bits. It exists in the table only to show that "2 bytes, no decode" is not automatically good:
the decode is not the cost, the mantissa is.

**fp4's 125 µs is a real format cost, not a benchmark defect.** Its kernel compiles to **984 SASS instructions against
int4's 288**: E2M1's magnitudes are non-uniformly spaced (0, .5, 1, 1.5, 2, 3, 4, 6), so each encode needs seven
compares where a 4-bit integer needs one shift. That is the price of the float exponent at 4 bits, and it is why fp4
is 4× slower than int4 at the same byte count.

**The fp16 path does what the brief predicted, and the compiler needed help.** `cuobjdump --dump-sass` on the fp16
store shows **12 `HFMA2`, 8 `HMUL2`, 0 `FFMA`** where the fp32 arm has 39 `FFMA`. The int8-with-fp16-arithmetic arm
needed a fix to get there: `__h2div` has **no fp16 instruction** and lowered to `MUFU.RCP` plus a Newton step per
value, so the row now hoists one reciprocal per row and multiplies. After that it shows **6 `HFMA2`, 6 `HADD2`,
18 `HMUL2`** and `FFMA` fell 39 → 11. Every kernel in the table is **0 bytes stack, 0 spill** (`ptxas -v`).

Registers did **not** fall as expected: the int8-fp16 arm uses **39** where plain int8 uses **29**. A `half2` is one
register for two values, but the pack/unpack around it costs more than the arithmetic it saves — which is the same
reason fp16 only reaches 0.88× on a 2× cheaper multiply.

**What is NOT measured, and must not be read into the table.** The compensator arms carry **zero** compensators
(`cudaMemset(comp, 0, …)`), so their error column is *identical* to their base arm — the fold is a no-op. They measure
**what carrying the compensators costs** (8–33% more time, 0.75 MiB a layer at 4 bits), **not what they buy**. The
paper's premise — that compensators recover the error — is untested here and needs a rank-r fit on the residual, which
this bench does not implement. **int4's 174% error is therefore not the last word on 4-bit**; it is the last word on
4-bit *without* a compensator fit.

**Verdict: the LeapQuant line is narrowed to "not worth it for speed, and beaten by plain fp16 on every axis", not
closed on accuracy.** Row 3's answer is that the state should be **fp16** — half the footprint, 12% more time, and
25× int8's accuracy — and the heavy 4-bit machinery is not justified by anything measured here. Row 4's compensator
half remains open for the reason above.

### 2.2 Row 8 is bounded by the scores it writes

The scan reads 25.2 MB of centroids and **writes a 25.2 MB fp32 score matrix** (32 queries × 196,608 buckets × 4 B),
so its traffic is 50.3 MB against 0.805 GFLOP = **16 FLOP/B**. At 288 GB/s that caps it at **4.6 TFLOP/s**, which is
where the slow end measures: 50.3 MB in 199 µs is **253 GB/s = 88% of that roofline**, and `ncu` puts that run at
**81% DRAM**. A run that beats 4.6 TFLOP/s is above the DRAM roofline, which can only mean part of the score matrix
stayed in the 32 MiB L2 and never reached DRAM — so it is not a DRAM measurement, and the 4.05 figure is the one to
read. The row's own report says "92% of its roofline"; that number is not reproducible from the run it cites, and this
page uses the 88% the arithmetic gives.

**And the scan's own cost is small enough that the bar never mattered.** At 4.05–5.66 TFLOP/s the 6.6 TFLOP scan of a
whole prefill takes **1.2–1.6 s**, which is **0.6–0.8% of a 209 s prefill**. The plan's 11 TFLOP/s would have made it
0.29 s — a difference of under one second, not the −8.6% the phase is chasing. The −8.6% is LSA's whole net effect
(the selector it removes, minus the scan, minus a 25% wider attention list); the scan is the smallest term in it. Row 8
therefore does exactly what the plan asks of it — it **sizes the scan term** — and the 11 TFLOP/s bar was the wrong
question, because no arrangement of this geometry reaches it.

### 2.3 The other five rows

Rows 6 and 7 pass and are the phase's calibration: row 6 reproduces a documented 55.5 TOPS **exactly** (0.0% off) and
row 7 confirms a recorded 87.6% of DRAM with an unchanged output hash, so the box is quiet enough for the rest of the
table to mean something. Row 2 misses its 60% bar by 11 points at 256K and that miss is the row's result: 5.88 ns a cell
at long context is the real per-cell attention cost the feature plan needed. Row 8 misses its 11 TFLOP/s bar for a
reason given in §2.2 and still delivers what it was for. Row 4 is not run.

## 3. The card's documented peaks

The denominators. Each row says whether NVIDIA documents the number, or whether it is arithmetic from a measured
input.

| quantity | value | basis | label |
| --- | ---: | --- | --- |
| DRAM bandwidth | **288.0 GB/s** | 128-bit GDDR6 at 9,001 MHz: 2 × 9,001 MHz × 128 bit ÷ 8 | computed from a measured clock |
| DRAM bandwidth, loaded | **280.0 GB/s** | 8,751 MHz, the memory clock the card reports under load | measured |
| L2 | 32 MiB | `cuDeviceGetAttribute` | measured |
| SMs | **34** | `cuDeviceGetAttribute` | measured |
| bf16/fp16 dense tensor | **~44 TFLOP/s** | Ada whitepaper V2.02: the RTX 4090's 165.2 TFLOPS bf16 at 128 SMs, scaled × 34/128 | computed |
| TF32 dense tensor | **~22 TFLOP/s** | same, from 82.6 TFLOPS | computed |
| int8 dense tensor | **~88 TOPS** | 2× the bf16 rate, per the CUDA C++ Programming Guide 13.4 Table 36 and every Ada SKU's spec sheet | computed |
| fp8 dense tensor (`e4m3`) | **~88 TFLOP/s** | exists on sm_89; see section 4 | computed |
| **FP4 tensor cores** | **0** | CUDA guide Table 36, column 8.9: FP6 and FP4 are blank, 10.0 and up only | documented |
| FP32 (FMA pipe) | **24.5 TFLOP/s** | 34 SMs × 128 lanes × 2 flop × 2,820 MHz | computed |
| shared memory / block | 101,376 B (99 KiB) opt-in, 49,152 B default | CUDA guide Table 34, column 8.9; the card reports 102,400 B/SM | documented, matches the measured value |

NVIDIA publishes no L2 bandwidth for Ada, and the one attempt to measure it here produced unusable data (the card
idled at 495 of 3,105 MHz). A row that would need an L2 roofline is judged against DRAM bandwidth instead, which is
the conservative choice.

## 4. What sm_89 has, and what it does not

Checked, not assumed: CUDA guide Tables 32, 34 and 36; the PTX ISA for `mma` target requirements; and the SASS
mnemonics in [cicuvc/nvidia-sass-document](https://github.com/cicuvc/nvidia-sass-document)
(`sm_89_instructions.txt` against `sm_90_instructions.txt`). The negative cases were compiled, not read.

| feature | sm_89 | evidence |
| --- | --- | --- |
| `mma.sync` m16n8k16 bf16 / fp16 | **yes** | `HMMA.16816.F32.BF16`, `HMMA.16816.F32.F16` in SASS |
| `mma.sync` m16n8k32 int8 | **yes** | `IMMA.16832.S8.S8.SAT` |
| `mma.sync` m16n8k32 fp8 `e4m3` | **yes** | `QMMA.16832.F32.E4M3.E4M3`; PTX requires sm_89 or higher |
| `dp4a` | **yes** | `IDP.4A.S8.S8` |
| `cp.async` (`LDGSTS`) | **yes** | sm_80 feature; measured as `LDGSTS.E.BYPASS.128` in the scan kernel |
| `ldmatrix` (`LDSM`) | **yes** | sm_75 |
| FP4 / FP6 tensor cores | **no** | Table 36, column 8.9 blank; 10.0+ |
| `wgmma` (`QGMMA`) | **no** | 0 hits in `sm_89_instructions.txt`, 21 in `sm_90_instructions.txt` |
| `tcgen05` / TMEM | **no** | 0 hits in either dump |
| TMA (`cp.async.bulk`) | **no** | `nvcc -arch=sm_89` reports it requires sm_90 |
| thread-block clusters / DSMEM | **no** | `__cluster_dims__` is rejected for sm_89 |

**Per-format instruction basis**, which is what row 3's table turns on. Each was compiled and counted, not assumed:

| format | instruction | evidence |
| --- | --- | --- |
| fp16 arithmetic | `HFMA2`, `HADD2`, `HMUL2` | 2× the per-lane FP32 rate on Ada. **Measured** in row 3's SASS: 12 `HFMA2` + 8 `HMUL2` in the fp16 store, 0 `FFMA`; and 6/6/18 with 11 `FFMA` in the int8-with-fp16-arithmetic arm |
| bf16 arithmetic | `HFMA2` (bf16 pairs) | native on sm_89; the row-3 arm compiled to 16 `FFMA` **because it converts per value**, so bf16 does not automatically mean vector ops |
| fp8 E4M3 conversion | `cvt.rn.satfinite.e4m3x2.f32` | sm_89 (PTX ISA). `F2FP.F16.F32.PACK_AB` appears 14× in the fp16 arm; the fp8 arm converts per byte because the state is stored one value a byte |
| fp4 E2M1 with UE4M3 scales | **software decode, no hardware** | the tree's `Fmt<40>` (`src/kernels/cuda/iq_kernels.cu`) and row 3's arm both decode in software. Below sm_100 there is no FP4 tensor core (Table 36), so a 4-bit state costs 7 compares a value — **984 SASS instructions against int4's 288** |
| int8 arithmetic | `IMMA` (tensor core) or `IDP.4A` (dp4a) | none of row 3's arms uses either: the step is elementwise, so int8 means `I2F`/`F2I` pairs around an `FFMA` |

The consequence for the work: every kernel of this kind has to be written in `mma.sync` + `cp.async` + `dp4a` terms,
because the papers' WGMMA, TMA and cluster versions do not port to this card. And for an elementwise step like row 3's,
the instruction that matters is not a tensor-core one at all — it is `HFMA2`.

## 5. Measurement protocol

The protocol is short and every line of it was learned from a measurement that was wrong without it.

1. **Pin the SM clock.** `nvidia-smi -lgc 2820` before a batch, `nvidia-smi -rgc` after, and `nvidia-smi dmon`
   through each timed window to prove it held. Without this, a 3-second benchmark on this card decays across the
   990–1,863 MHz band between its own GPU bursts: the same binary that reads **55.5 TOPS pinned reads 40.8 unpinned**,
   a 26% error that is not reproducible in either direction. An unpinned row is unidentifiable.
2. **Never compare an idle-clock sample to a loaded one.** The card idles at 200–900 MHz and loads at ~2,790. Judge
   stability on the loaded samples only, and report the loaded ones.
3. **Interleave the arms** inside one process, and take the **median of 5** runs (or of 5 × several inner iterations
   where one call is short). Arms measured one after another on a card whose clock wanders are not comparable.
   **Quote the median, never the best run** — two figures in the first version of this page were the best of five and
   were corrected (row 2's read, 5.45 → 5.88 ns a cell). When a summary and its log disagree, the log wins.
4. **Give each arm its own buffers.** A benchmark that shares a value cache between arms measures one side twice.
5. **Report both bandwidth denominators** where a row is bandwidth-bound, and state which one the pass is judged on.
6. **Bound every run.** A GPU command runs under `timeout`, so a hang cannot take the card.

## 6. Harness traps

Each of these produced a wrong number before it was found, and each is worth knowing before writing the next
benchmark.

- **A rounded buffer is not the size you asked for.** Passing a `room` rounded up to a power of two made a
  65,536-cell call read a 131,072-row cache, so it read twice the data and reported **3.522 ms against a true
  0.308 ms**. Pass the exact count.
- **A DRAM-sized source buffer turns a cache test into a DRAM test.** A first version of the KV-store bench read its
  input from a 512 MB pool, which put a 4,096 B read against an 832 B write. The engine's source is the projection's
  fresh output — 2 KB a token at decode, 8 MB a chunk at prefill — and hot in L2. Holding the source in L2 is the
  engine's case, and it changes which denominator is honest: under `ncu` (which flushes, so the source is in DRAM) the
  value store alone moves **239 GB/s = 83% of peak**; the whole store, with its source in L2, moves **140 GB/s =
  48.6%** of traffic and **23.6 GB/s = 8.2%** of pure KV bytes. Three different numbers for one kernel, all correct,
  and quoting the wrong one is how row 1 was first reported at 82%.
- **One benchmark call is not one kernel.** The k8v4 store is `store_q8` (keys) *and* `store_q4` (values); profiling
  the value store and calling it "the store" reported 82% of peak for an operation that moves at 48.6%. The key store
  is the slower of the two (73% of the time, 28% DRAM) because it reads 4 B a value and writes 1 B. Enumerate every
  launch a timed region makes before dividing by its time.
- **Sharing a cache between arms flatters both.** `fdec_q8(q, kq, kq, …)` passed the *key* cache as the *value*
  cache, so the comparison arm read one side twice and half of that from L2: it reported **451 GB/s on a 288 GB/s
  card**. That is how the error was caught — an impossible number, not a suspicious one.
- **A token at position *P* attends rows `0..P`.** Sizing a cache to the context reads one row past the end; it
  produced `inf` at 256K. Size it to `P + 1`.
- **A collective after a divergent early exit deadlocks.** `__shfl_sync(0xffffffff, …)` placed after
  `if (key > last) continue`, where `key` depends on the lane, leaves 8 of every 32 lanes never reaching the shuffle
  the mask demands. Hoist the collective above every branch, or mask it with `__activemask()`. The same rule applies
  to `__syncthreads()` above a `return`. Test it with a multi-token call at a tile boundary, not with one token.
- **A non-idempotent counter makes a single run wrong.** One kernel is bimodal by ~35% at 128K and 256K depending on
  its allocation size; a single run of it can be 45% wrong. Run 5 and take the median.
- **A block-buffered log makes a hang look like a slow run.** `printf` to a file is block-buffered; a 27-minute
  stall produced an empty log and could not be diagnosed from it. Line-buffer stdout in anything long.

## 7. How to reproduce each row

Every row here is a benchmark: it prints numbers, it asserts nothing, and it is run by hand. Three **correctness**
binaries around them are in `ctest` and run with the rest of the suite — `poly_kv_q4_test`, `poly_kv_q8_test` and
`poly_fattn_test` (which registers seven entries, including `k8v4 --heads` and `deep <pos>`). None of the timed legs are.

| row | binary | invocation | in `ctest`? |
| ---: | --- | --- | --- |
| 1 | `poly_kv_q4_test` | `poly_kv_q4_test --bench` (and `poly_kv_q8_test --bench` for the q8_0 arm) | the correctness checks are; `--bench` is not |
| 2 | `poly_kv_read_bench` | `poly_kv_read_bench 256 16 2 --only-fdec --cells 4096,32768,131072,262144 --rows-max 262145` | no; the correctness guard is `poly_fattn_test k8v4` / `deep <pos>`, which is |
| 3 | `gdn_state_bench` | `gdn_state_bench` (timing, 16 arms); `gdn_state_bench --accuracy` (error at 1 and 128 steps); `gdn_state_bench --smoke` (partial-tile correctness) | no |
| 4 | `gdn_state_bench` | same binary: the compensator arms are A2/A3/A11/A16/A17 | no |
| 5 | `q8_1_quant_bench` | `q8_1_quant_bench` | no |
| 6 | `mmq_expert_bench` | `mmq_expert_bench 12 160 626` | no |
| 6 | `native_expert_bench` | `native_expert_bench <shard>.gguf 0,1,2,3 64 1 -1 -1 200` | no |
| 7 | `nvfp4_decode_bench` | `nvfp4_decode_bench` (defaults are 40/40, 64 groups, 2 tokens, 50 iterations) | no |
| 8 | `lsa_scan_bench` | `lsa_scan_bench` | no |

The build is the same for all of them:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_BUILD_TESTS=ON
cmake --build build -j 8
sudo -n nvidia-smi -lgc 2820                    # pin the clock; -rgc restores it
ctest --test-dir build -R 'poly_fattn|poly_kv|kv_' -j1
```

`nvfp4_decode_bench` and the two expert benches need no model file. `native_expert_bench` takes a GGUF shard and the
layers whose expert tensors to read. `ncu` counters need root on this machine: the normal user is refused with
`ERR_NVGPUCTRPERM`, and `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1`.

## 8. What the phase changed in the engine

One change shipped and one hypothesis was refuted. Both are worth recording, because the refuted one cost more work
than the shipped one.

**Shipped: QSA decode attention runs the Daily kernel on sm_89 by default** (commit `aa3150f7a`). Measured at the
Flash-Next geometry (2,051 selected cells, 12 query heads a KV head, k8v4 pools), one decode token:

| | before | after |
| --- | ---: | ---: |
| ns a cell | 17.70 | **10.95** |
| ms a layer | 0.0363 | **0.0225** |
| 12 QSA layers a token | 0.436 ms | **0.270 ms** |
| SASS `FFMA` / `LDS` / stack | 708 / 48–72 / 384 B | **444 / 12 / 0 B** |

That is **1.62×**, and **0.166 ms saved per decode token**. `STRATA_ATTN_DAILY=0` keeps the older kernel for an A/B.
The 2× the plan asked for is not reachable at this geometry and the arithmetic says why: 33 chunks of 64 cells is 66
CTAs, which is **0.49 waves on 34 SMs**. Halving the chunk doubles the waves and measured *slower*, because the 12 KiB
query tile is then re-read twice per cell.

The lever was not what the first reading suggested. `ncu` on the older kernel put the FMA pipe at **11.3% active** and
DRAM at 17%, with `long_scoreboard` stalls at 7.19 per issued instruction — it was waiting on the value loop's
dependent loads, not short of arithmetic. A tensor-core rewrite could not have helped it.

**Refuted: "the k8v4 value read is 4× redundant."** The claim was that the eight lane groups of a warp read only two
distinct 8-byte windows. They do not: the value pointer already carries the block index (`vd0 / 32 = 2w + g8/4`), so
the groups cover **two blocks × four windows — eight distinct addresses**. A warp-shuffle version was implemented,
corrected for a deadlock, and measured: **L1 sectors 199,216 before and 199,216 after, requests identical at 41,648**.
There was no redundant traffic to remove, and the shuffle cost 8–71%. It was reverted.

Measured instead, with each arm on its own buffers and five interleaved runs at 256K: k8v4 **5.88 ns a cell**, q8_0
**4.96**, fp16 **8.32** — so k8v4 is **18.5% behind q8_0** and **29% ahead of fp16**, at 23.5% fewer bytes than q8_0.
The remaining gap is the 4-bit decode's arithmetic (byte permutes, half adds and multiplies) plus a 16-byte block
granularity, not a memory-access defect. **k8v4 is a footprint choice on this card, not a speed one**, which matches
what the format's own documentation already says about it being weaker at long range. The 23.5% of footprint is real
and is worth having where VRAM is the constraint; it is not free, and the row-2 numbers above are what it costs.

## 9. Four numbers in the first version were wrong, and the logs caught them

This page's first version quoted row 2's read at **152.6 GB/s = 53.0%**, row 1's store at **235.1 GB/s = 81.6%**, row
8's roofline at **92%**, and row 8's cost as **−8.5%** of prefill. Each was wrong, in a way worth naming.

**Row 2 was the best of five runs, not the median.** The five interleaved runs give 5.71, 5.91, 5.46, 5.88 and 5.98 ns a
cell; 5.45 was the best. The median is **5.88 ns = 141.5 GB/s = 49.1%**, so the row misses its 60% bar by 11 points
rather than 7. The verdict (FAIL) and the cause (the 4-bit decode's arithmetic) do not change. The same correction
moved q8_0 to 4.96 ns and fp16 to 8.32 (from single runs of 4.90 and 8.64), so "32% faster than fp16" became **29%**.

**Row 1 was one kernel of two, and the denominator was wrong as well.** `store_q4` — the value store — is 27% of the
store; `store_q8`, the key store, is the other 73% and runs at 28% DRAM because it reads 4 B a value and writes 1 B.
Profiling one of them and calling it the store reported 82% of peak for an operation that moves at 48.6%. The
replacement bar was also wrong, and §2.0 says why.

**Row 8's "92% of its roofline" is not in the run it cites.** The geometry caps the scan at **4.6 TFLOP/s** (16 FLOP per
byte at 288 GB/s), so 5.66 TFLOP/s is 123% of the roofline — which is not a better kernel but a run whose score matrix
partly stayed in L2. The `ncu` sample is the 199 µs / 4.05 TFLOP/s run at 81% DRAM, which is **88%** of the roofline.
The 92% could not be reproduced and is not used.

**Row 8's "−8.5%" was the whole LSA phase, not the scan.** The scan is 1.2–1.6 s of a 209 s prefill = **0.6–0.8%**. The
−8.6% is LSA's net after the selector it removes, the scan, and a 25% wider attention list. Attributing it to the scan
made a 0.6% term look like the phase's headline result.

None of this loosens a conclusion: rows 2, 3, 5 and 8 still fail, rows 6 and 7 still confirm, and the shipped QSA win in
§8 is unaffected. What it changes is the *margin* on rows 1 and 2 and the *reason* on row 8, and it is recorded here
rather than quietly edited because the rule the page states in §5 — quote the median, and when a summary and its log
disagree the log wins — is only worth anything if the page obeys it.
