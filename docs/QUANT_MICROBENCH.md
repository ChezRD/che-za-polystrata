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

- **row 1** counted only the *write* half of a read-modify-write, so its 173 GB/s bar needed 3.6× the card. Revised to
  a fraction of the operation's real traffic.
- **row 2** asked for an L2 speed-up from a 4K call that is launch-bound — it measures the grid, not the cache.
  Revised to take the L2 evidence where it actually appears, at 32K.
- **row 5** asked for a bandwidth fraction at 1–8 rows, where the kernel is a launch and nothing else (1,659 ns of
  pure launch at one row). Revised to the row's own stop-rule outcome: fusing must be free, and it is.

The measurements themselves were sound; the denominators were not. The gate table shows both the original bar and the
corrected one, so a reader can see what was actually decided.

## 1. The eight rows

Bandwidth-bound rows give the fraction of both denominators: **288.0 GB/s**, which is 128-bit GDDR6 at the card's
9,001 MHz maximum memory clock (2 × 9,001 MHz × 128 bit ÷ 8), and **280.0 GB/s**, which is 8,751 MHz — the memory
clock the card reports while loaded. The second is 3% lower and is the honest one for a sustained run; the first is
what the plan's tables use.

| row | operation | metric | measured | % of 288 (280) | verdict | unlocks |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | k8v4 KV **store**: q8_0 keys + Hadamard-rotated Q4_0 values, 832 B a cell | GB/s of DRAM traffic, counting the source read | **235.1 GB/s** (82% of peak) | **81.6% (84.0%)** | **PASS**, with the bar revised | the store is 23.5% smaller and 29% faster than q8_0's (the *read* is a different answer, row 2) |
| 2 | KV **read**: decode attention over the k8v4 cache | GB/s of cache read at 256K | **152.6 GB/s** | **53.0% (54.5%)** | **FAIL** the 60% bar by 7 points; the row's finding stands | 5.4 ns a cell at 128K–256K, which sizes LSA's wider-list penalty |
| 3 | int8 GDN recurrent state (LeapQuant's floor) | µs a layer at T=1, int8 against fp32 | **20.2 µs int8 vs 15.1–16.8 µs fp32 = 0.70–0.83×** | — (the state is L2-resident; see §2.1) | **FAIL**: below the 1.3× stop rule | **closes the LeapQuant line** on this card |
| 4 | int8 state + compensator tokens + window buffer | — | **not run** | — | **NOT RUN — gated by row 3's FAIL** | nothing; the plan runs it only if row 3 passes |
| 5 | Q8_1 activation quantization, 2,560-value rows | GB/s of activations in + int8 out | **64.5 GB/s at 8 tokens; 8.1 GB/s at 1** | **22.4% (23.0%) at 8; 2.8% at 1** | **FAIL** at every width the engine runs; the stop rule fires | confirms it must be **fused**, not optimized |
| 6 | IQ3_S / IQ4_NL expert dequant + dot, the reference | int8-equivalent TOPS at R=626 | **55.5 TOPS** (1.773 ms), documented 55.5 | 63% of the 88 TOPS int8 peak | **PASS** (0.0% off) | the truth serum for the phase: the box reproduces its own numbers |
| 7 | NVFP4 E2M1/UE4M3 decode, software path | GB/s of weights | **252.3 GB/s**, hash `054f38e43e6cdb4e` | **87.6% (90.1%)** | **CONFIRMED** (cited, not re-derived) | the calibration for how a software-decoded 4-bit format behaves here |
| 8 | LSA centroid scan, Q·Cᵀ with d=64, 32×32 tiles | TFLOP/s of fp16 `mma.sync` | **4.05–5.66 TFLOP/s** | 9–13% of the ~44 TFLOP/s fp16 peak | **FAIL** the 11 TFLOP/s bar; meets **92% of its own DRAM roofline** | the scan costs 1.2 s of a 209 s prefill, i.e. −8.5% |

Row 6 is the one that decides whether the others can be believed: it re-measures a number the tree already documents
and reproduces it exactly. Row 4 is shown rather than omitted because the plan makes it conditional, and a gate table
that silently drops a row reads as if the row passed.

## 2. Where the rows landed, and the three revised bars

### 2.0 The arithmetic behind each revision

**Row 1 — the bar counted half the work.** Writing a k8v4 cell is a read-modify-write: 832 B written, but the cell's
floats must be **read** to quantize them, which is 4,096 B (512 values a side, twice, 4 B each). KV bytes are
therefore **16.9% of the operation's traffic**, and at an impossible 100% of the card's 288 GB/s they top out at
**48.6 GB/s**. The plan asked for 173 GB/s, which would require moving 1,025 GB/s — **3.6× this card's DRAM
bandwidth**. Corrected bar: **≥ 75% of DRAM bandwidth counting the source read.** Measured: 235.1 GB/s of traffic,
**81.6%**, PASS.

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

### 2.1 Row 3 closed a line on physics, not on a percentage

The GDN state is 48 heads × 128 × 128 × 4 B = **3.0 MiB**, which fits in the card's 32 MiB L2. The fp32 arm moves it
at **374–416 GB/s — above the card's 288 GB/s DRAM peak**, which can only happen if it is being served from L2. A
format that is 4× smaller cannot make a step faster when the step is not paying for its bytes; it only adds a
dequantize and a requantize a value. That is why int8 measured **0.70–0.83× of fp32** and why the stop rule, not the
pass bar, is the row's real result.

### 2.2 Row 8 is bounded by the scores it writes

The scan reads 25.2 MB of centroids and **writes a 25.2 MB fp32 score matrix** (32 queries × 196,608 buckets × 4 B),
so its traffic is 50.3 MB against 0.805 GFLOP = **16 FLOP/B**. At 288 GB/s that caps it at **4.6 TFLOP/s**, which is
where it measures. `ncu` puts it at **81% DRAM / 92% of its own roofline**, so the kernel is not the problem. The
plan already says this row does not gate the feature: at the measured rate the 6.6 TFLOP scan of a whole prefill takes
1.2 s, i.e. **−8.5%** against the −8.6% quoted for a scan 2.7× faster.

### 2.3 The other five rows

Rows 6 and 7 pass and are the phase's calibration: row 6 reproduces a documented 55.5 TOPS **exactly** (0.0% off) and
row 7 confirms a recorded 87.6% of DRAM with an unchanged output hash, so the box is quiet enough for the rest of the
table to mean something. Row 2 misses its 60% bar by 7 points at 256K and that miss is the row's result: 5.4 ns a cell
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

The consequence for the work: every kernel of this kind has to be written in `mma.sync` + `cp.async` + `dp4a` terms,
because the papers' WGMMA, TMA and cluster versions do not port to this card.

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
  fresh output — 2 KB a token at decode, 8 MB a chunk at prefill — and hot in L2. With the source held in L2 the same
  kernel measures 82% of peak instead of 8%.
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

Two of these are registered in `ctest` and run with the rest of the suite. The rest are benchmarks: they print
numbers, they assert nothing, and they are run by hand.

| row | binary | invocation | in `ctest`? |
| ---: | --- | --- | --- |
| 1 | `poly_kv_q4_test` | `poly_kv_q4_test --bench` (and `poly_kv_q8_test --bench` for the q8_0 arm) | the correctness checks are; `--bench` is not |
| 2 | `poly_kv_read_bench` | `poly_kv_read_bench 256 16 2 --only-fdec --cells 4096,32768,131072,262144 --rows-max 262145` | no; `poly_fattn_test k8v4`, `... deep <pos>` and `... parity` are the correctness guards and are |
| 3 | `gdn_state_bench` | `gdn_state_bench` | no |
| 4 | - | not run (see the table) | - |
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
**4.96**, fp16 **8.64** — so k8v4 is **18.5% behind q8_0** and **32% ahead of fp16**, at 23.5% fewer bytes than q8_0.
The remaining gap is the 4-bit decode's arithmetic (byte permutes, half adds and multiplies) plus a 16-byte block
granularity, not a memory-access defect. **k8v4 is a footprint choice on this card, not a speed one**, which matches
what the format's own documentation already says about it being weaker at long range. The 23.5% of footprint is real
and is worth having where VRAM is the constraint; it is not free, and the row-2 numbers above are what it costs.
