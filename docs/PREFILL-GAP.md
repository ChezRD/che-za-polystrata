# The prefill gap: a dense 27B's matmuls run at 7.5% of this card

**Status: measured, not fixed.** This page is the record of a finding that came out of the LSA work
(`docs/LSA.md` §9b) and is bigger than it: the dense Qwen3.8-27B's prefill multiplies its weights at **4–6 TFLOP/s on
a card that does 55.5**, while its attention runs at or above peak. Everything that follows is a number from a run,
with the command that produced it.

## 1. The observation

A 14,336-token prefill of the dense 27B on one RTX 4060 Ti at 2,820 MHz:

```
strata-poly --model Qwen3.8-27B-UD-IQ3_XXS.gguf --tokens-file <14,336 ids> --max-new 1 \
  --kv k8v4 --no-draft --window 1 --max-context 88064 --prompt-batch 128 --threads 8 \
  --prompt-cache 0 --lookup 0 --temperature 0 --seed 1
```

| arm | runs (tok/s) | median |
|---|---|---:|
| `--lsa` off | 95.0, 94.7, 94.7 | **94.8** |
| `--lsa` on | 94.1, 94.0, 94.1 | **94.1** |

174 s for 14,336 tokens. The FLOPs in that call, from the geometry the GGUF declares (`embedding_length` 5,120,
`feed_forward_length` 17,408, 16 attention layers, 48 recurrent, 24 heads × 256, 4 KV heads):

| component | TFLOP | share | at the card's 55.5 TOPS |
|---|---:|---:|---:|
| dense FFN, 64 layers | 490.7 | 67.4% | 8.8 s |
| GDN projections, 48 layers | 163.0 | 22.4% | 2.9 s |
| attention projections, 16 | 33.7 | 4.6% | 0.6 s |
| **Q·Kᵀ + PV, 16 layers** | **40.4** | **5.6%** | **0.73 s** |
| **total** | **727.7** | | **13.1 s** |

The wall is 174 s against a 13.1 s roofline. **The aggregate is 4.2 TFLOP/s, 7.5% of the card.** Attention's own
40.4 TFLOP would take 0.73 s at peak, so if attention is at peak then everything else — 687.3 TFLOP — takes 173.3 s at
**4.0 TFLOP/s, 7.2% of peak.**

That is the finding. A 29% cut of a 5.6% component is 1.6% of the total, which is why the LSA prefill A/B moved nothing
either way.

## 2. The localisation: the FFN is run as an 8-token GEMV

`tools/gemm_probe.cu` (target `poly_gemm_probe`) times the engine's **two** matrix paths on one real FFN matrix —
`[5120 × 17408]` of IQ3_XXS, 34.1 MB, 178 MFLOP a token — at the batch sizes a prefill has. Both arms link the same
`strata_poly_mmq` the engine links, so neither is a stand-in.

```
flock -w 60 /tmp/opencode/gpu.lock timeout 178 ./build/poly_gemm_probe
```

```
clock 2820 MHz (pin: sudo -n nvidia-smi -lgc 2820)
iq3_xxs 98 B a 256-value block = 0.383 B a weight; n_embd 5120, n_ff 17408
one [5120 x 17408] matrix: 34.1 MB, 178 MFLOP a token
  intensity at batch   1:    5.2 FLOP/B (memory-bound: the balance is 193)
  intensity at batch  128:  668.7 FLOP/B
  intensity at batch 1024: 5349.9 FLOP/B
Mmq::fits(IQ3_XXS, 17408, 5120) = true   <- what lin_raw checks before it would use mmq_
Mmq::fits(IQ3_XXS, 5120, 17408)  = true  (the down matrix)
n=   8  mmvq    0.232 ms   6.2 TFLOP/s | mmq    0.121 ms  11.8 TFLOP/s  (1.92x)
n=  64  mmvq    1.860 ms   6.1 TFLOP/s | mmq    0.245 ms  46.6 TFLOP/s  (7.60x)
n= 128  mmvq    3.721 ms   6.1 TFLOP/s | mmq    0.349 ms  65.3 TFLOP/s  (10.66x)
n= 256  mmvq    7.453 ms   6.1 TFLOP/s | mmq    0.659 ms  69.3 TFLOP/s  (11.31x)
n= 512  mmvq   15.117 ms   6.0 TFLOP/s | mmq    1.305 ms  69.9 TFLOP/s  (11.58x)
```

**`Mmq::fits` is printed because the localisation rests on it.** It is the condition `lin_raw` checks before it would
use `mmq_`, and it is **true** for both FFN shapes — so the missing piece is the null pointer's gate, not a kernel the
card cannot run.

* `mmvq` = `native_mmvq`, which is what `Engine::lin_raw` falls back to: **8 tokens a launch**.
* `mmq` = `Mmq::mul`, llama.cpp's matrix-matrix kernel: **the whole batch in one launch**, weights read once.

**At batch 128 the gap is 10.7×, and it is the entire 7.5%.** 6.0 TFLOP/s is the measured `mmvq` rate; the prefill's
4.0 TFLOP/s for non-attention work is that same path plus the elementwise work between the matrices.

The arithmetic says why. IQ3_XXS is 0.383 B a weight, so one FFN matrix has an arithmetic intensity of 5.2 FLOP/B at
batch 1 — memory-bound, against this card's 193 FLOP/B balance. At batch 128 it is 668.7 FLOP/B, comfortably
compute-bound. **`native_mmvq` eight at a time is 42 FLOP/B: still memory-bound, which is exactly the 6 TFLOP/s it
measures.** The kernel that would fix it exists, is built, and is faster by an order of magnitude.

## 3. Why a dense model never reaches it

`Engine::lin_raw` (`src/poly/engine.cpp:3591`) picks the MMQ path on two conditions:

```cpp
if (mmq_ && n >= kStageRows && Mmq::fits((int) w.type, w.n_out, w.n_in)) {
    ...  mmq_->mul(w.p, (int) w.type, w.n_out, w.n_in, 0, 1, mqx_, nullptr, nullptr, n, n, y, w.n_out, st_);
```

and `mmq_` is built in `load()` (`engine.cpp:3104-3116`):

```cpp
for (int il = 0; il < n_layer_; ++il) {
    Layer& L = layers_[(size_t) il];
    if (!L.moe) continue;                        // <-- the loop is over MoE LAYERS
    ...
    L.mmq = opt_.gpu_experts && rows_ >= kStageRows && gu_fits && Mmq::fits(...);
    tiles = tiles || L.mmq;
}
if (tiles) mmq_ = std::make_unique<Mmq>();
```

**`mmq_` is created only if some MoE layer has MMQ tiles.** A dense model has no MoE layers (`L.moe` is false
throughout), so `tiles` stays false, `mmq_` stays null, and every dense FFN and GDN projection takes the `mmq_ ==
nullptr` branch to `native_mmvq` 8 tokens at a time. `Layer::mmq` is documented as "the card has MMQ tiles for its
**routed experts'** types" — it was never meant to describe a dense matrix.

The kernel is not missing: `IQ3_XXS` is in `STRATA_POLY_MMQ_TYPES` (`src/poly/mmq.cu:25-27`) and `Mmq::fits` returns
true for it at these shapes. Nothing but the `mmq_` gate stands between a dense layer and a 10.7× faster matrix
product.

## 4. What a fix would have to change

Stated as requirements, because none of it is done here — LSA is this branch's work and the matmul path is a line of
its own with its own measurements.

1. **Create `mmq_` for a dense model's own matrices**, not only for routed experts. The gate has to consult the layer's
   dense FFN and projection types as well as `L.moe`, and `Layer::mmq` stops being an expert-only flag or gets a
   sibling.
2. **Keep the state a second sequence would need in step.** `alloc_caches()` runs again per bound sequence
   (`engine.cpp:1236`) and `mmq_` is one object; the probe used a single stream and one matrix, so per-sequence
   correctness is untested.
3. **Check the half-sum guard.** `lin_raw`'s MMQ branch divides rows by a power of two for types whose block sums live
   in halves (`Mmq::half_sums`), and undoes it after. That path has never run for IQ3_XXS in this engine, so it needs
   the same bit-comparison against `native_mmvq` that `fattn_test.cu` gives the attention kernels — a faster matmul
   that changes an answer is not a fix.
4. **Re-measure the ceiling.** MMQ's shared-memory configuration is chosen per card (`Mmq::fits`); at 128 tokens it
   could change what fits beside the cache, and `docs/LSA.md` §9a is exactly the kind of number that moves.
5. **Expect the ceiling, not the estimate, to bound the win.** 10.7× on the FFN is 80 s → 7.5 s of a 174 s prefill; if
   the GDN projections move the same way, the prefill is 4× faster and attention becomes 8% of it. That would make LSA
   matter — which is the second reason to fix this before judging LSA's prefill claim.

## 5. What is not measured

* **The GDN projections were not probed.** They take the same `lin()` path, so the mechanism is the same, but their
  shapes are different and the ×10.7 is not theirs to claim.
* **The elementwise work is not separated.** `swiglu_clamp`, `rms_norm`, `native_quantize_q8_1` and `axpy` are inside
  the 93 s that is not the FFN and not attention. `--dump-layers` synchronises on every stage, so it cannot time
  anything; a per-stage timing needs an instrument this branch does not have.
* **One card, one model, one format.** sm_89, IQ3_XXS, 14,336 tokens.
* **MMQ's own correctness at these shapes was not checked**, only its speed — item 3 above is the gate that would.

## 5a. The gates this branch still passes

Two, re-run after the probe landed, because a new target is a new binary:

* **`--lsa` off is byte-identical: PASS.** 4/4 prompts, 24 greedy tokens, against `integrate/dense-and-fixes`'s binary.
* **`lsa_parity --gpu`: PASSED, 0 failures** — key 0 wrong of 8 rows in both spaces, Ward's 64 buckets summing to 512
  of 512 keys, and the scan's worst |got − want| 2.254e-07 at n_kv 8,192.
* **The spill gate: the same 22 violations as the base binary, and none of them LSA's.** 1,989 kernels against 1,977 —
  the 12 new ones are the LSA kernels, and every one is STACK 0 / LOCAL 0. The 22 are pre-existing
  (`fdec_kernel`'s instantiations, `rope_kernel`, `gdn_state_reg_kernel` and others) and are byte-for-byte the same
  set on the base build, so this branch adds nothing to the spill debt and clears none of it.

## 5b. The fix, measured on `fix/mmq-dense`

**The gate is fixed** (`c8c6723c9`): `Engine::dense_products` asks the architecture for the matrices it multiplies
outside the experts' path, `Mmq::fits` is asked about each, and any yes creates `mmq_`. The MoE loop that decided
`tiles` is untouched, so a MoE model answers nothing.

### The SASS answer: it IS INT8 tensor cores, and my "read once instead of sixteen times" account was only half

`cuobjdump --dump-sass` on the built binary, then every `mul_mat_q` instantiation counted by opcode. **`HMMA` appears in
none of them; every quantized type uses `IMMA.16832.S8.S8`.** IQ3_XXS's tile-128 kernel:

| | count | share of 6,816 |
|---|---:|---:|
| `IMMA.16832.S8.S8` | **256** | 3.8% |
| `FFMA` | 1,032 | 15.1% |
| `FMUL` | 1,040 | 15.3% |
| `I2FP` | 1,032 | 15.1% |
| `IMAD` | 652 | 9.6% |
| `LOP3` | 475 | 7.0% |
| `LDS`/`LDG` | 664 | 9.7% |
| `HMMA` | **0** | — |

Across all 21 quantized types: **`IMMA` present in every one, `HMMA` in none.** So the user's account is right that
the win is INT8 tensor cores — but the `I2FP`/`FMUL`/`FFMA` thirds say the kernel spends most of its instructions
converting and scaling, not multiplying. **Both accounts describe part of it:** MMQ reaches the INT8 units *and* reads
each weight once per batch, where `native_mmvq` reads it once per eight tokens. `src/poly/mmq.cu` has no `mma.sync` of
its own — it is a wrapper over llama.cpp's `mul_mat_q_case`, which is why the SASS has to be read from the binary rather
than from the source.

### The A/B, interleaved

131,072 tokens, k6v4, `--prompt-batch 128`, one run per lock, alternating:

| | run 1 | run 2 | run 3 | median |
|---|---:|---:|---:|---:|
| **MMQ dense** | 986.4 | 1,001.0 | 964.4 | **986.4 tok/s** |
| **`native_mmvq`** | 233.7 | 229.1 | 225.3 | **229.1 tok/s** |
| ratio | 4.22× | 4.37× | 4.28× | **4.31×** |

Spread: **1.5% on the fast arm, 2.0% on the slow one** — the three pairs agree to within noise, so a warm L2 is not
carrying the result.

At 40,960 tokens the same A/B was **1,932 / 1,935 / 2,008 tok/s** against **266 / 257 / 260**, i.e. **7.45×** — larger
because the shorter context lets the fast path do more of its work per weight read. **The ratio falls with context
length, which is the signature of a win in the matmul and not in attention.**

### The MoE regression gate

Both MoE models on this box, greedy, 32 tokens, `STRATA_MMQ_DENSE` unset against `=0`:

| model | result |
|---|---|
| Qwen3.6-35B-A3B (UD-Q3_K_M) | **32/32 byte-identical** |
| Flint-abliterated-Mix-12G | **32/32 byte-identical** |

And Ornith, which the gate actually changes, is **byte-identical with the gate on and off** — so the fast path is
free, not a quality/latency trade.

## 6. The command line, for the record


Prefill A/B (`docs/LSA.md` §9b), clock pinned with `sudo -n nvidia-smi -lgc 2820`:

```
flock -w 60 /tmp/opencode/gpu.lock timeout 178 strata-poly \
  --model Qwen3.8-27B-UD-IQ3_XXS.gguf --tokens-file <14336 ids> --max-new 1 --kv k8v4 \
  --no-draft --max-context 88064 --prompt-batch 128 --window 1 --threads 8 \
  --prompt-cache 0 --prompt-cache-every 0 --lookup 0 --temperature 0 --seed 1 [--lsa --lsa-map …]
```

The probe:

```
cmake --build build --target poly_gemm_probe
flock -w 60 /tmp/opencode/gpu.lock timeout 178 ./build/poly_gemm_probe
```
