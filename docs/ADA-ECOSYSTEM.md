# The Ada ecosystem around this fork: what exists, what transfers, and what to build

**Scope.** External survey plus a ranking, for one card: **RTX 4060 Ti 16 GB, AD106, sm_89, 34 SM, 288 GB/s DRAM,
32 MiB L2, 99 KB opt-in shared memory, ~12.5 GiB usable VRAM.** No GPU run was made for this document. Every claim
cites a URL or a `file:line` / doc section in this tree. Anything I could not check is marked **unverified**.

**The two engines.** This tree holds two independent engines and the ranking below is meaningless without the split:

> **Which docs live where.** `docs/REAP.md`, `docs/PREFILL-GAP.md`, `docs/DETAILS.md`, `docs/QUANT_MICROBENCH.md`,
> `docs/EXPERTDEC.md`, `docs/FLINT.md`, `docs/LOCAL_CHANGES.md`, `docs/MMVQ_IL_TABLE.md`, `docs/ORNITH.md` and
> `docs/QWEN38.md` are on `main`. **`docs/BENCH-INSTR-PATHS.md` is on branch `bench/instr-paths`** and
> **`docs/KV-INT8-PATHS.md` on branch `study/imma-paths`**, and **`docs/STRATA-QSA-CENSUS.md` is not in this
> worktree at all** — it is in `/tmp/opencode/wt-s6` (read-only). Read them with
> `git show bench/instr-paths:docs/BENCH-INSTR-PATHS.md` and `git show study/imma-paths:docs/KV-INT8-PATHS.md`.

| | **Engine A — `strata`** | **Engine B — `strata-poly`** |
|---|---|---|
| Binary | `build-poly-local/strata` | `strata-poly` |
| Model | Qwen3.8-Flash-Next REAP-320 (512→320 experts, 48 layers, 262,144 window, 62.37 GiB) | `qwen35` family: Ornith-1.5-9B / 35B / 122B, Qwen3.8-27B, Qwen3.6-35B-A3B, Flint |
| Attention | **QSA**: indexer + selector + page table + fixed-width id list (2,051 cells) + KV streaming | **dense/contiguous** `fattn` / `fdec` / `attend` |
| Decode attention (hot) | `attn_chunk_kernel_pre75<M,2,false>` — the fork's "Daily" kernel, `daily_attn()` on for `(8,9)` at `src/kernels/cuda/qsa_decode_attn.cu:892` | `fdec_kernel` ≥ 4,096 cells (`HMMA 64`), `attend` below (`560` scalar `FFMA`) |
| Expert cost | streams from **page-locked host RAM**, 2,522 slots × 4.43 MB, 0 file reads during decode | streams from RAM/SSD, `native_mmvq`/`Mmq` per layer |
| 256K baseline | prefill 1,249 tok/s / 209.8 s, decode **50.19 tok/s** (`docs/REAP.md:29-32`) | not run at 256K on this card; 131K: 1,945 read / 46.4 write (Flint, q8_0) (`docs/FLINT.md:110-113`) |

Two facts that decide most of Part 4's ranking and that are easy to get backwards:

1. **The "Daily" kernel is a fork-only default.** Upstream `daily_attn()` returns `sm120_default(maj,mnr)` only
   (`https://github.com/Niko1221/Strata/blob/main/src/kernels/cuda/qsa_decode_attn.cu:886`); the `|| (maj == 8 && mnr == 9)`
   that puts it on Ada is **this fork's** commit `aa3150f7a` ("perf(qsa): run the Daily decode-attention kernel on
   sm_89 by default"). Anything learned about the Daily kernel is therefore not portable upstream as a default.
2. **The Daily kernel is the faster path, and the profile numbers in the brief belong to the slower one.**
   `docs/QUANT_MICROBENCH.md:594-604` gives 17.70 → 10.95 ns/cell (1.62×) for the ship of the Daily kernel; the
   `stalled_long_scoreboard 45.5% / short_scoreboard 29.0% / occupancy 32.1% / DRAM 19%` profile was taken on the
   **upstream** kernel that no longer runs (`/tmp/opencode/SUMMARY.md`, a dynamic `ncu` profile). So the current
   hot kernel's stall mix is **unverified** — nobody has profiled `attn_chunk_kernel_pre75` at 256K. The pre-Daily
   kernel's 14.31 → 10.95 ns/cell spread in the tree also means the "19%" and "1.62×" figures are different runs of
   the same change; treat 1.62× as the measured one because it is the one with a before/after table.

---

# Part 1 — forks, descendants, and sibling engines, per engine

Licences: **MIT is the only licence this fork can take code from without a change** (`LICENSE`, and
`docs/DETAILS.md:2260`). Apache-2.0 is compatible with MIT for our purposes but adds a NOTICE obligation; **AGPL and
"personal & non-commercial" are not compatible** with an MIT fork.

## 1.1 The Strata family

### `Niko1221/Strata` — upstream, the engine's source of truth

- **URL** https://github.com/Niko1221/Strata · **active**: pushed 2026-11-12, 43,827 stars, 4,810 forks, 1,412 open
  issues (GitHub API, 2026-11-13) · **licence MIT**.
- **Architectures**: `cmake/backends/cuda.cmake:23` defaults to `"86-real;89-real;120"`, so **Ada is a first-class
  build target**, and `include/strata/core/sm120_series.hpp:101` carries `mmvq_il_89_default()` — an Ada-only
  small-batch matvec row table, `kIlRows_89`, measured on an RTX 4070 and an RTX 4080 and enabled by default on
  cc 8.9 (`src/kernels/cuda/native_mmvq.cu:4142-4150`, `:4242-4258`). This is the one place upstream has
  **tuned specifically for Ada**, and it is already in this tree.
- **What it has that this tree does not**: the QSA decode-attention *upstream* kernel (`attn_chunk_kernel<3>`), and
  the DeltaNet/`GR` families. Not useful: this fork measured the Daily kernel faster at this geometry.
- **On a path we run**: yes, but it is the *unmodified* path — this fork is ahead of it on sm_89 attention.
- **Engine A and B**: both (it is the parent).

### `Hardin22/Strata-DualGPU` — the "Daily" lineage (the name in the brief)

- **URL** https://github.com/Hardin22/Strata-DualGPU · **active**: pushed 2026-10-06, 84 stars, 6 forks, 5 open
  issues · **licence MIT** · **fork of** `Niko1221/Strata`.
- **Attribution resolved.** The two SHAs in this tree are **commits by `imanu <imanu86@gmail.com>` in upstream
  Strata**, not in Hardin22's repo:
  - `be4cf1c` — "OPUS-KERNEL-Accendi di default l'MMVQ multi-colonna di B6", 2026-10-06.
  - `3effa9d5` — "OPUS-PORT-Porta da Hardin22 il merge dell'attenzione limitato alla larghezza", 2026-10-05, which
    says it ports `Hardin22/Strata-DualGPU b1f03d33` (upstream PR #910).

  So "the Daily" is a **collaborator's kernel line inside upstream**, and Hardin22's repo is where the
  width-bounded attention merge came from, not where the chunk kernel came from. The name `Hardin22` in
  `src/kernels/cuda/qsa_decode_attn.cu:500,755` is accurate for the merge and for the port, not for
  `attn_chunk_kernel_pre75`.
- **Architectures**: releases are built for `sm_75, 86, 89, 120` with PTX for newer (release notes). **Ada
  present.**
- **What it does that this tree does not**: (a) `--pipeline-windows`, pipelined verify windows across a layer
  split; (b) async adaptive expert tier so swaps do not stop decode; (c) resident-RAM mode **on** a layer split;
  (d) programmatic dependent launch on RTX 50. Readme numbers: 29 → 143 tok/s on Swift IQ2_XS, RTX 5080 + 4060 Ti.
- **Relevance**: (a)-(c) are two-GPU features; this card is the flagship **single**-GPU target, so they are
  **Engine A, not on this path**. (d) is sm_90+ PDL. The genuinely reusable idea is `attn_merge_v2_kernel`'s
  **bounded merge** — merge only up to the selection width instead of the padded chunk count — which this tree
  already has (`3effa9d5`).
- **Verdict**: nothing left to take for one Ada card.

### `DevDock-AI/Speedy` — tuned on Ampere, Ada is a build target only

- **URL** https://github.com/DevDock-AI/Speedy · **active**: pushed 2026-11-13, 62 stars · **licence
  "Personal & Non-Commercial"** (README badge; GitHub reports `NOASSERTION`) → **code cannot be taken**.
- **Architectures**: needs cc 8.0+; the README tells the user to set `89` for RTX 40 — i.e. Ada compiles but
  **nothing is tuned for it**; every number is RTX 3080 10 GB.
- **What it has**: `gpu-unlock` FP16 tensor-rate unlock (locked 62 → unlocked 106 TFLOPS on a 3080, +10-12% prefill,
  **decode unchanged**), 229K/1M context tiers, `STRATA_VERIFY_DEFER_CPU`, and per-quant presets.
- **On our path**: `STRATA_VERIFY_DEFER_CPU` is Engine A's expert path (defer ranking to CPU). The unlock result is
  a useful negative: **decode did not move**, which is what a bandwidth-bound path should do.
- **Licence verdict**: **ideas only, no code** — "Personal & Non-Commercial" is stricter than MIT.

### `Asofwar/Strata-Turbo` — long-context agent fork, 16 GB card

- **URL** https://github.com/Asofwar/Strata-Turbo · **active** (2026-11) · **licence MIT**.
- **Hardware**: RTX 5080 16 GB + i9-14900K, Windows; claims 16-18× upstream at 66K on Qwen3.6-35B-A3B.
- **What it does**: int8 K/V, own long-context kernels, conversation parking on NVMe, per-family presets.
- **On our path**: **Engine B** (Qwen3.6/Ornith, `fdec`/`fattn`). The 16 GB budget is the same class as ours, but
  the numbers are on a 5080 (sm_120, 448 GB/s + 48 GB/s of L2 the AD106 does not have) — **not transferable
  without argument**.
- **Verdict**: worth reading its long-context kernel notes; no measurement of ours is reproduced there.

### `sangharshadhyeta/Strata-HotRod` — SSD streaming on 12 GB / 30 GB RAM

- **URL** https://github.com/sangharshadhyeta/Strata-HotRod · **active**: pushed 2026-11-13 · **licence MIT**.
- **What it does**: three-tier expert placement (VRAM / RAM / SSD) with a look-ahead predictor; 50-70 tok/s
  IQ3_S on an RTX 3060 12 GB, 30 GB RAM, 262K.
- **On our path**: **Engine A's expert cache**, the exact mechanism. But this fork's 256K config already has
  **0 file reads during decode** (`docs/REAP.md:36`) with 31.42 GiB page-locked — SSD tiering is what you need
  when RAM is short, and at 62 GB RAM it is not the binding constraint.
- **Verdict**: no transfer at 62 GB RAM. Relevant only if the resident budget drops.

### `rocktboy1982/pancake-strata` — the ecosystem's own survey, and a large patch ledger

- **URL** https://github.com/rocktboy1982/pancake-strata · **active** (2026-11) · licence MIT (inherits).
- **Why it matters even though it targets gfx1100**: it carries
  [`pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md`](https://github.com/rocktboy1982/pancake-strata/blob/main/pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md),
  a dated (2026-11-09) audit of ~30 derivatives with PR numbers. It is the best available map of this ecosystem and
  I used it as a starting index, then verified each entry independently.
- **What it lists that bears on Ada**: `gputier/StrataGP` (16 perf patches, +5-15% decode / +42-49% prompt on an
  RTX 5090 — **sm_120, not Ada**), `havok2-htwo/StrataQ38FN` (43 patches, 4×3090), `Juud_engine` (+9.6-15.6%
  decode on a 4090 by shortening the worker spin and skipping activation quantisation for GPU-resident tokens —
  **this one is Ada and Engine A**).
- **Verdict**: the ledger is more valuable than the code.

### Others in the family, one line each (all MIT unless noted)

| repo | what it is | engine | Ada relevance |
|---|---|---|---|
| `simplexai-labs/PolyStrata` (1,022★, 2026-11-13) | this fork's direct parent; `src/poly` second engine | B | the base of everything here |
| `jmnargi/Strata-V100` (143★) | sm_70 port | both | none (sm_70) |
| `MistVVK/XeStrata` (347★) | SYCL/Level Zero, no CUDA | both | none |
| `eddoursul/Lamina` (63★) | NVIDIA-only, own course, cc 8.0+ | A+B | `--prompt-split` two-stage prompts, 1.24-1.55× on long prompts |
| `CrashxZ/MoEmentum` (4★) | CPU expert kernel work | A | CPU-side only |
| `gputier/StrataGP` (9★) | 16 perf patches | A | measured on 5090 → **not Ada** |
| `havok2-htwo/StrataQ38FN` | 43 patches, serving scale | A | agent-switch re-read 25 s → 1.5 s |
| `cutlerbenjamin1-cmd/qwen38-megakernel` | **AGPL** | A | ideas only, **no code** |

## 1.2 Sibling engines (not Strata forks) with Ada kernels

### `ggml-org/llama.cpp` — the source this tree already borrows from, and it has **explicit Ada routing**

- **URL** https://github.com/ggml-org/llama.cpp · **active** (daily) · **licence MIT** · `ggml-cuda` is what
  `src/poly/mmq.cu` wraps.
- **Architectures**: `89-real` in the default arch list (`ggml/src/ggml-cuda/CMakeLists.txt`), and
  `GGML_CUDA_CC_ADA_LOVELACE 890` is a named constant (`ggml/src/ggml-cuda/common.cuh:56`).
- **What it does that this tree does not — three things, all measured, all on our path:**

  1. **Decode attention is deliberately routed *away* from tensor cores on Ada.**
     `ggml/src/ggml-cuda/fattn.cu:701-704`: for a single query (`Q->ne[1] == 1`) with unquantised K/V, Ada takes
     `BEST_FATTN_KERNEL_VEC`; and at `:706-709` with **quantised** K/V, Ada takes the vector kernel up to
     `Q->ne[1] <= 2` where Turing takes it only at `== 1`. So llama.cpp's authors concluded that at decode on Ada
     the **non-MMA** kernel wins. That is the same conclusion this tree reached independently (`attend` scalar vs
     `fdec` tensor, `docs/BENCH-INSTR-PATHS.md` §3), and it is a strong external confirmation that "put decode
     attention on tensor cores" is the wrong lever on this card.
  2. **A sparse-gather flash attention with an index list** — the mechanism Engine A's QSA needs, and it is
     **merged** upstream: PR #27970 (`CUDA + ggml: add sparse-fa for DSV4/GLM`, merged 2026-09-02), built on
     PR #25917. The kernel takes `KV_max_ptr`/`KV_rows_ptr` and a per-tile index list, gathers the selected rows
     and runs dense attention over them; the gate is `K->ne[1] >= max(4096, 2*n_gather)`
     (`ggml/src/ggml-cuda/fattn.cu`, `shall_use_sparse`). The QSA-specific follow-ups (#28213, #28130) were
     **closed unmerged**, but #28213's body carries the number that matters: on dual RTX A6000, IQ4_XS, q8_0 KV,
     masking the full 130K context gave 15.7 tok/s and gathering the ~2K selected cells gave **23.6 tok/s (+50%)**,
     growing with depth (+19% at 62K, +6% at 31K).
  3. **`get_mmvq_mmid_max_batch` special-cases Ada**: for `GGML_OP_MUL_MAT_ID` on Ada it returns the full
     `MMVQ_MAX_BATCH_SIZE` (always MMVQ) rather than the Turing+ per-type table
     (`ggml/src/ggml-cuda/mmvq.cu:281-292`), and `ggml_cuda_should_use_mmvq` has an **Ada-only** switch tuned on an
     RTX 4090 (`:321-330`): Q2_K → MMVQ up to 4 columns, Q3_K → up to 6, everything else → the full batch. That
     is a per-type decode-matvec crossover table measured on **Ada consumer silicon**, i.e. exactly the class of
     thing `docs/MMVQ_IL_TABLE.md` asks a volunteer to produce for sm_89.

- **On a path we run**: (1) is Engine A *and* B decode attention; (3) is Engine A and B decode matvec; (2) is
  Engine A only.
- **Licence**: MIT → **no licence change**.

### `ikawrakow/ik_llama.cpp` — the most aggressive low-bit CUDA kernel work in the ecosystem

- **URL** https://github.com/ikawrakow/ik_llama.cpp · **active**: pushed 2026-11-13, 3,393 stars, 118 open issues ·
  **licence MIT**.
- **Architectures**: no `sm_89`-specific guards found in `ggml/src/ggml-cuda/` — it inherits llama.cpp's
  `ampere_mma_available`/`turing_mma_available` routing, so Ada runs the Ampere configs.
- **What it does that this tree does not**:
  - **`iqk_mul_mat`**: a completely separate quantised GEMM/GEMV family with its own repacking (`IQ4_KS`, `IQ5_KS`,
    `IQ4_KS_R4`, ternary `TQ`), on top of an `iqk_mmvq.cu` decode matvec.
  - **PR #2831** (merged 2026-11-04) — "one block per row tile and expert in MoE MMQ": replaces stream-k with
    `nty*ntzw` blocks when the grid fills ≥90% of block slots, and **skips the fixup kernel entirely**. Measured on
    GB10 at 64/128/512/1024/2048 columns: **+8.3% / +15.1% / +8.0% / +9.1% / +9.7%** on Qwen3.8-Flash-Next Q2_K.
    This is the *batch* regime, i.e. a verify window (T=4) and prefill chunks — not decode T=1.
  - **PR #2824** — CUDA graphs also when Hadamard rotations are present (relevant to a k8v4-style rotated-V cache).
  - `MTP decoding` support and GLM-5.3/glm5next paths.
- **On our path**: the MoE-MMQ grid change is **Engine A's expert matmul** (verify windows), and the "no fixup
  kernel" part removes a launch from a 380-launch token (`docs/FLINT.md:130`). `iqk_mul_mat` is a whole second
  implementation — too large to take, but its *repack* argument (never dequantise the same weight twice per batch)
  is the same argument `docs/PREFILL-GAP.md` §2 makes.
- **Licence**: MIT → no change.

### `kvcache-ai/ktransformers` — the reference for Engine A's dominant cost

- **URL** https://github.com/kvcache-ai/ktransformers · **active**: pushed 2026-11-13, 19,666 stars · **licence
  Apache-2.0** (compatible; keep their NOTICE).
- **Architecture**: CUDA, cc 8.0+; their published table has **1× RTX 4090** rows.
- **What it does that this tree does not**: **CPU-GPU expert scheduling as a first-class placement problem** —
  `--kt-num-gpu-experts`, `--kt-gpu-experts-ratio`, and four placement strategies (`uniform`, `frequency`,
  `front-loading`, `random`) plus `--kt-init-expert-location` from an activation-statistics file and
  `--kt-enable-dynamic-expert-update`
  (`doc/en/kt-kernel/experts-sched-Tutorial.md`). Their Qwen3-30B-A3B row is **8+ it/s on 1× RTX 4090** at ~24 GB
  total, i.e. experts split between VRAM and host.
- **On our path**: **Engine A, directly.** This fork's equivalent knobs are `--expert-cache auto`,
  `--adapt-every/--adapt-swaps/--adapt-decay` and `--expert-profile`; KTransformers adds *placement by measured
  frequency* as a separate axis from *replacement policy*. Note the honest counter: this fork already has a
  trace-built `expert-profile.bin` and an adaptive tier, so the delta is the **per-layer count** (`--kt-num-gpu-experts`
  is per MoE layer, not one global budget), which this fork does not expose.
- **Licence**: Apache-2.0 → code usable, keep NOTICE.

### `SJTU-IPADS/PowerInfer` — hot/cold neuron split

- **URL** https://github.com/SJTU-IPADS/PowerInfer · **active**: pushed 2026-05-11, 9,884 stars · **licence MIT**.
- **What it claims**: SOSP 2024, arXiv 2312.12456; hot neurons preloaded to GPU, cold computed on CPU, **on a single
  RTX 4090**; up to 11.69× over llama.cpp, OPT-30B at 82% of an A100's generation rate. **Ada, single consumer card.**
- **On our path**: the *idea* is **Engine A's expert cache** — but PowerInfer splits by **neuron** (dense models),
  this engine splits by **expert** (MoE). The transferable part is the predictor: an *adaptive* predictor choosing
  what to keep resident, which this fork does with `--adapt-*`. Their measured claim (82% of A100 on a 4090) is a
  sanity anchor for how much a 4090 can do at low batch, not a target for a 128-bit card.
- **Licence**: MIT.

### `MaxDevv/lean-qwenflash` — the closest external competitor to Engine A's exact problem

- **URL** https://github.com/MaxDevv/lean-qwenflash · **active**: pushed 2026-10-26, 67 stars · **licence: none
  declared** (GitHub API returns `null`) → **treat as all-rights-reserved, no code**.
- **What it is**: Qwen3.8-Flash-Next, **512 experts**, full 262K, on **16 GB VRAM and 8 GB system RAM**, by
  streaming experts **directly from NVMe with `O_DIRECT`** into a bounded GPU cache, with the whole 51B n-gram
  table also on disk. All expert math on the GPU; built on ExLlamaV3 + `glm53-flash-offload`.
- **Measured**: RTX 4090 capped to 16 GB VRAM / 8 GB RAM — 8k: 1,080 prefill / 28-33 decode (prose), 19-22 (code);
  100k: 1,536 / 20.8; 262k: 1,627 / **20.0**. RTX 3080 Laptop 16 GB, 100k: 647 / 17.6 (8 GB cache) to 20.9
  (16 GB cache).
- **Why it is the single most important external data point in this document**: it is **a 512-expert MoE with
  partial residency on 16 GB**, which is exactly the question the brief asks in Part 3, and it is **on a 4090**
  (same 16 GB class, 2.5× our bandwidth, 72 SM). At 262K it gets **20.0 tok/s**; this fork gets **50.19** on the
  same model size class at the same context on a card with **44% of the bandwidth**. See Part 3 for what that
  means.
- **Licence**: none → **no code, ideas only**.

### `sybil-solutions/glm53-flash-offload` and `turboderp-org/exllamav3` — the machinery under it

- `glm53-flash-offload` https://github.com/sybil-solutions/glm53-flash-offload · 352★ · **MIT** · active 2026-10-18.
  GLM-5.3-Flash EXL3 3.05 bpw on **one 24 GB RTX 3090** + DDR4: `fast` 28.2 tok/s (218 GiB free), **`nvme` 19.5
  tok/s at a 55 GiB RAM cap**, 16 GiB cap 14.7, bit-exact mode 8.3. Elastic GPU expert cache, zero-copy experts,
  AVX2 CPU tier. **This is the strongest published "experts on NVMe, 24 GB card" number I found**, and its
  architecture (elastic cache + staging) is Engine A's, done outside Strata.
- `exllamav3` https://github.com/turboderp-org/exllamav3 · 1,821★ · **MIT** · active 2026-11-11. EXL3/QTIP quant,
  2-8 bit **KV cache** quantisation, and a GEMM/GEMV family written for consumer cards; its README's benchmark table
  is 3090 Ti vs **4090**. **Engine B and Engine A both** could read its KV-quant layout, but its kernels are
  torch/CUDA-graph based, not a drop-in.

### `IST-DASLab/marlin` — the dequantisation-overhead paper kernel

- **URL** https://github.com/IST-DASLab/marlin · 1,175★ · **Apache-2.0** · last push 2024-09-04 (dormant).
- **What it does**: FP16×INT4 matmul that keeps **near-ideal 4× speedup up to batch 16-32** by ordering
  dequantisation against tensor-core issue, async global loads with an evict-first L2 policy, double-buffered
  smem, conflict-free layouts, and a striped tile partition. Explicitly "**NVIDIA GPU with compute capability
  >= 8.0 (Ampere or Ada; not yet optimised for Hopper)**".
- **On our path**: this is the **dequantisation-overhead** answer for Engine A and B's `mul_mat_q` — the tree
  measured 45-54% of `mul_mat_q`'s instructions on `I2FP`/`FMUL`/`FFMA` around 3.8% `IMMA`
  (`docs/PREFILL-GAP.md` §5b). Marlin's technique list is a checklist of exactly those wasted instructions.
  Caveat: Marlin is **W4A16** (fp16 accumulate), not the **W2A8/W3A8 → IMMA** shape this tree uses, and its
  batch sweet spot (16-32) is a verify window, not T=1.
- **Licence**: Apache-2.0.

### `flashinfer-ai/flashinfer`

- **URL** https://github.com/flashinfer-ai/flashinfer · 6,918★ · **Apache-2.0** · active 2026-11-13.
- **Ada**: explicitly in the support matrix — "Ada Lovelace | SM 8.9 | L4, L40, RTX 40 series" (README). Has
  **fused MoE kernels**, **grouped GEMM for multi-expert routing**, sorting-free sampling, and chain speculative
  sampling.
- **On our path**: Engine A's **expert dispatch** and Engine B's **MoE layer** — but its kernels are
  serving-shaped (batch ≫ 1) and its dependency surface is large. The transferable item is *grouped GEMM as one
  launch for a routing pattern*, which this tree already does (`native_expert_grouped`).

### `vllm-project/vllm`, `sgl-project/sglang`, `NVIDIA/TensorRT-LLM`, `microsoft/MInference`, `ModelTC/lightllm`

- **All large, all active, none of them tuned for a 34-SM 128-bit Ada card.** Their attention and MoE kernels
  assume batch ≫ 1 and, for TensorRT-LLM, sm_90+ for the fast paths.
- **One exception worth naming: TensorRT-LLM's XQA** (`cpp/kernels/xqa/`, Apache-2.0). It is a generation-phase
  MQA/GQA kernel family with an explicit **per-architecture tile choice** and an `#error` for unknown archs:
  `mha.cu:92-100` picks `preferedKHeadPartBytes = 64, cacheVTileSeqLen = 32` for **`__CUDA_ARCH__ == 890`** and
  `128 / 64` for 800/870/900. So NVIDIA ships an **Ada-specific KV tile width** for decode attention — half the
  Hopper one — and documents paged-KV support (`PAGED_KV_CACHE_LAYOUT=1`, "VLLM-style KV pool … split-wise
  KV-pool and sequence-first memory layout"). `gen_cubins.py` builds for `arch_options = [80, 86, 90]` — **no 89
  cubin in that list**, so Ada runs the 86 cubin.
- **On our path**: the **64-byte K head part / 32-cell V tile for sm_89** is the one concrete, Ada-specific
  decode-attention constant I found outside llama.cpp, and it is a **direct argument** for Engine A's and B's
  chunk/tile width choice (this tree uses a 64-cell chunk, `CHUNK` in `qsa_decode_attn.cu`). **Engine A and B.**

### `mit-han-lab/omniserve` (QServe) — W4A8KV4

- **URL** https://github.com/mit-han-lab/omniserve · 867★ · **Apache-2.0** · dormant since 2025-03.
- **What it does**: names the exact disease this tree measured — "existing INT4 methods suffer **20-90% runtime
  overhead dequantising** weights or partial sums" — and treats it with **progressive quantisation** (a low-bit
  high-order part plus a high-bit low-order correction, so the *dequantisation* is cheap), compute-aware weight
  reordering, and register-level parallelism.
- **Measured on**: **A100 and L40S** (`arXiv 2405.04532`). L40S is **Ada, sm_89**, 48 GB, 864 GB/s — an Ada part,
  but a datacentre one with 3× our bandwidth and 4× our SMs.
- **On our path**: **both engines' `mul_mat_q`**. This is the closest published match to the tree's 45-54%
  dequantisation finding, and it is Ada silicon (L40S), which makes it the one quantisation paper that transfers
  with an argument rather than a caveat.

### `Mozilla-Ocho/llamafile` — `tinyBLAS`

- **URL** https://github.com/Mozilla-Ocho/llamafile · 26,632★ · licence `NOASSERTION` (Apache-2.0 with a
  vendored-licence bundle) · active 2026-11-09.
- **What it does**: `tinyBLAS`, a small set of fp16/bf16 GEMM kernels for **AMD and NVIDIA consumer cards** that
  beat cuBLAS on the small, flat matrices of token generation.
- **On our path**: Engine B's prompt GEMM (`attend_batch` → cuBLAS, `src/poly/qwen35.cpp`); the tree already
  measured that path as the k6v4 regression (`docs/BENCH-INSTR-PATHS.md` §0.2). A tinyBLAS-style kernel is an
  alternative to cuBLAS for exactly those flat shapes.
- **Caveat**: `docs/PREFILL-GAP.md` shows the fix for the same problem was **`mul_mat_q` itself** (7.45×/4.31×), so
  tinyBLAS is a *second* option, not the first.

## 1.3 The table, corrected, with the engine column

Ranked by what it would take to move a number on **this** card. "A" = Engine A (`strata`), "B" = `strata-poly`.

| project | active | sm_89? | engine | what it has that we do not | on our hot path? | licence | take code? |
|---|---|---|---|---|---|---|---|
| **llama.cpp** `ggml-cuda` | daily | **yes, explicit** (`CC_ADA_LOVELACE`, `89-real`, wired into fattn + mmvq routing) | **A + B** | Ada routes decode attention to **VEC not MMA**; sparse-KV **gather** FA (merged #27970); Ada-tuned MMVQ/`MUL_MAT_ID` batch table | **A**: decode attn shape, sparse gather, decode matvec. **B**: decode attn, matvec | MIT | **yes** |
| **ik_llama.cpp** | 2026-11-13 | inherits Ampere configs | **A** (+B) | MoE MMQ **one block per row tile per expert, no fixup kernel** (+8-15% at 64-2048 cols); CUDA graphs with Hadamard; `iqk_mul_mat` | **A** expert verify-window matmul | MIT | **yes** |
| **KTransformers** | 2026-11-13 | 4090 rows | **A** | per-MoE-layer GPU expert **count** + frequency placement + dynamic update, as a placement axis | **A** expert cache (dominant cost) | Apache-2.0 | yes (NOTICE) |
| **PowerInfer** | 2026-05-11 | **4090** | **A** (idea) | adaptive hot/cold predictor; neuron-level, not expert-level | **A** predictor | MIT | yes |
| **lean-qwenflash** | 2026-10-26 | 4090 16 GB | **A** (benchmark) | 512-expert Qwen3.8, **NVMe `O_DIRECT`** streaming, everything-GPU math, 262K on 16 GB | comparison only | **none** | **no** |
| **glm53-flash-offload** | 2026-10-18 | 3090 | **A** (design) | elastic GPU expert cache + NVMe staging + AVX2 CPU tier, measured at 4 RAM caps | **A** | MIT | yes |
| **exllamav3** | 2026-11-11 | 4090 table | **A + B** | 2-8 bit **KV** quant + EXL3; consumer-card GEMMs | B: KV, A: idea | MIT | code hard to lift |
| **Marlin** | 2024-09 (dormant) | **Ampere or Ada, stated** | **A + B** | dequant-ordered INT4 GEMM, near-4× up to batch 16-32 | **both**: `mul_mat_q` dequant 45-54% | Apache-2.0 | yes (NOTICE) |
| **QServe / omniserve** | 2025-03 | **L40S (sm_89)** | **A + B** | progressive quantisation to cut dequantisation 20-90% | **both**: same | Apache-2.0 | yes |
| **TensorRT-LLM XQA** | 2026-11-13 | **`__CUDA_ARCH__ == 890` branch**, no 89 cubin | **A + B** | sm_89 picks **64 B K-head part / 32-cell V tile** vs Hopper's 128/64; paged-KV layout | **A**: chunk width; **B**: `fdec` tiles | Apache-2.0 | yes (NOTICE) |
| **FlashInfer** | 2026-11-13 | listed | **A + B** | fused MoE, grouped GEMM for routing, sorting-free sampling | same shape as existing code | Apache-2.0 | heavy |
| **Hardin22/DualGPU** | 2026-10-06 | releases 75/86/89/120 | **A** | pipelined windows, async tier, RAM mode **on a split** | **no** (single-GPU card) | MIT | nothing left |
| **DevDock-AI/Speedy** | 2026-11-13 | builds 89, tunes 86 | **A** | `VERIFY_DEFER_CPU`, gpu-unlock (decode unchanged) | **A** expert path | **Personal & Non-Commercial** | **no** |
| **Asofwar/Strata-Turbo** | 2026-11 | 5080 | **B** | long-context kernels on Qwen3.6/Ornith | B | MIT | read only |
| **Strata-HotRod** | 2026-11-13 | 3060 | **A** | SSD 3-tier + lookahead | only if RAM-starved | MIT | no |
| **vLLM / SGLang / MInference / LightLLM** | active | generic | B mostly | serving-scale kernels | no | Apache-2.0 | no |
| **pancake-strata** | 2026-11 | gfx1100/3060 | A+B | **the ecosystem map** + a large rejected-item ledger | indirect | MIT | read only |

**The five that matter, in order:** llama.cpp (Ada routing + sparse gather + Ada MMVQ table), ik_llama.cpp
(MoE-MMQ grid without fixup), KTransformers (expert placement), Marlin/QServe (dequantisation overhead), and
lean-qwenflash (the only 512-expert-on-16 GB measurement that exists).

---

# Part 2 — papers, split by which engine they can move

Rule applied throughout: a paper is **transferable** if it measured on consumer Ada (4060/4070/4080/4090) or on a
128-bit-bus part, or if its mechanism is a property of the *card class* (few SMs, small L2, low bandwidth) rather
than of a datacentre part. Everything measured only on A100/H100/B200 is marked **NOT TRANSFERABLE WITHOUT
ARGUMENT** and says what argument would be needed.

## 2.1 Decode attention on bandwidth-limited consumer cards — **Engine A and B**

| paper | venue/year | what it claims | hardware measured | transfers? |
|---|---|---|---|---|
| **FlashDecoding++** — arXiv [2311.01282](https://arxiv.org/abs/2311.01282) | 2023 | asynchronised softmax with a **unified max** removes the inter-block sync (~20% of attention); flat-GEMM double buffering; per-shape dataflow | "NVIDIA and AMD GPUs"; the abstract does not name a part and the HTML has no A100/4090 string — **part unverified** | **Partly, and the part that transfers is the negative one.** Our decode reads **at most 2,051 cells** (`qsa_selection_width` = `min(n_kv, idx_top_k + idx_block - 1)`, `docs/BENCH-INSTR-PATHS.md` §0.4), so every block already sees the whole selection and **there are no partial softmaxes to unify** in Engine A. In Engine B `fdec` does split (1,024-row blocks) — there the unified-max trick applies, but at 34 SMs the split is 16-66 blocks and the *merge* is not the cost (`attn_merge_kernel` = 496 instructions, census §1). **Verdict: not a lever here.** |
| **S²ANTA** (Stochastic Additive No-mulT Attention) — arXiv [2605.01910](https://arxiv.org/abs/2605.01910) | 2026 | sample S ≪ n_k value rows from the post-softmax distribution; unbiased estimator; gather-and-add instead of MAC | **NVIDIA RTX 6000 Ada** — 1.5× decode-attention kernel speedup over FlashInfer/FlashDecoding, 1.25× end-to-end | **Yes — this is the one attention paper measured on a real Ada part.** And it lands exactly on Engine A's measured wall: the tree's decode kernel is latency-bound on the **value loop's dependent loads** (`docs/QUANT_MICROBENCH.md:609-611`, `long_scoreboard` 7.19/instruction), and S²ANTA replaces the value-stage MACs with **gather-and-add over a sampled subset** — fewer dependent loads, not fewer bits. Engine A only (it needs a sampling distribution over the selection; Engine B's `attend` is contiguous). Quality risk is real: it is a *biased-free but sampled* estimator, so it needs the same cosine/top-1 gate the tree already applies (`docs/BENCH-INSTR-PATHS.md` §1). |
| **SparQ Attention** — arXiv [2312.04985](https://arxiv.org/abs/2312.04985) | 2024 | approximate top-k of key blocks by a **low-rank sketch of K**, then gather; 6.4× theoretical at S=4096, r=32, k=128 | **A100 (40 GB) and A10G** (Triton kernel: 49 → 38 µs on A100, 128 µs on A10G) | **Not transferable without argument.** A10G is Ampere 24 GB / 600 GB/s; the mechanism is "skip bytes you don't need", and on this card **fewer bytes has never bought a faster read** (q8_0 4.96 < k8v4 5.88 < fp16 8.32 ns/cell, `docs/QUANT_MICROBENCH.md:619`). Would need a measurement showing the sketch's extra pass is cheaper than the saved reads at 34 SMs and 288 GB/s. Engine A only. |
| **TidalDecode** — arXiv [2410.05076](https://arxiv.org/abs/2410.05076) | 2024 | keep a small set of **"positional" tokens** whose KV is cached and attended every step, chosen by attention-score stability across layers | **one A100 80 GB SXM4**, CUDA 12.2, vs FlashInfer | **No.** Needs a stable-score argument and a spare-SM budget; on 34 SMs the persistent-token blocks compete with the expert kernels for the same SMs. Engine A only (needs a selection). |
| **MInference 1.0** — https://github.com/microsoft/MInference (NeurIPS'24 spotlight) | 2024 | vertical-slash / block-sparse patterns **for prefill**, 10× on 1M context | A100 | **No for this card** — prefill-side, and this fork's prefill is 1,249 tok/s on a 209.8 s budget whose binding term is the **matmul**, not attention (`docs/PREFILL-GAP.md` §1: attention is 5.6% of the FLOPs and its 40.4 TFLOP would take 0.73 s of 174 s). |
| **llama.cpp sparse-FA (PR #27970, merged)** — not a paper, but the mechanism | 2026-09 | pass a per-token max-KV count and an index list; gather selected rows, run dense attention over them | DGX Spark (GB10) in the PR table; **A6000 in #28213** | **Yes, and it is the strongest single external finding for Engine A.** #28213 on 2× A6000: 15.7 → 23.6 tok/s at 130K (**+50%**), 26.5 → 31.6 at 62K, 36.5 → 38.5 at 31K. Engine A already *has* an id list and a page table, so the mechanism is present; the *number* says the gap between "mask the full cache" and "gather 2K cells" is large when the kernel is masked. Engine A does not mask — it gathers through the page table already (`qsa_attn_pools`, `src/core/layer.cpp:1012`). **So the finding is confirmation that the current design is right, not a change.** |

**Engine A's decode attention in one line:** 2,051 cells × 13 stacks × 816 B ≈ **20.7 MiB a token**, and at
50.19 tok/s a token is 19,924 µs — so the KV read is **0.38% of a token at peak bandwidth and 1.7% at the
65 GB/s the k8v4 kernel actually reaches** (`docs/BENCH-INSTR-PATHS.md` §1). Nothing that makes
that read cheaper can matter. What matters is the **0.270 ms of 12 QSA layers** (`docs/QUANT_MICROBENCH.md:601`)
and, above all, **whether the expert slots stay fed**.

## 2.2 MoE inference with experts split between VRAM and host RAM — **Engine A only**

| paper | venue/year | what it claims | hardware measured | transfers? |
|---|---|---|---|---|
| **PowerInfer** — arXiv [2312.12456](https://arxiv.org/abs/2312.12456) | SOSP 2024 | hot/cold **neuron** split, adaptive predictor, neuron-aware sparse operators; 11.69× over llama.cpp; OPT-30B at 82% of an A100's rate | **single RTX 4090** | **Yes as a design, no as a kernel.** Ada consumer, single card. But it splits **neurons in a dense FFN**; Engine A splits **experts in a 320-expert MoE**, and its "hot set" is already measured (`expert-profile.bin`, trace-built). The transferable claim is the *shape* of the win: a consumer card with a smart residency policy reaches a large fraction of a datacentre card at batch 1. Engine A. |
| **HOBBIT** — arXiv [2411.01433](https://arxiv.org/abs/2411.01433) | 2024 | **mixed-precision** expert offload: a cache-miss expert is replaced by a **low-precision copy** instead of being loaded at full precision; token-level dynamic loading, layer-level prefetch, sequence-level cache policy; up to 9.93× decode | **RTX 4090** ("edge server") and **Jetson AGX Orin**; they measure that **expert loading is 85.5% of layer time on the 4090** | **Yes, and it is the most directly actionable paper for Engine A.** 85.5% of layer time in expert loading is the same regime as this fork's resident-RAM mode. The mixed-precision idea maps onto something this tree already has but does not combine: `--resident-budget-gib` decides **how many** slots, and REAP decides **which experts exist**; HOBBIT says decide **at what precision** per slot, so a miss costs a smaller read. Engine A. |
| **"Not All Models Suit Expert Offloading"** — arXiv [2505.16056](https://arxiv.org/abs/2505.16056) | 2025 | defines **SRP** (segment routing best performance) and **SCH** (segment cache best hit rate) over 20 MoE LLMs; a strong local-consistency vs local-load-balance trade-off; most models are best served at a cache of **≈2× the active experts** | analysis + toy models; **no consumer-GPU tok/s** | **Yes as a diagnostic, not as code.** It gives the right question to ask of this model: with 320 experts and 10 active, is the cache at 2,522 slots above or below the "2× active" knee? 2,522 slots over 48 layers ≈ 52 slots a layer against 10 active — **5.2× active**, i.e. above the knee the paper identifies. That is a *prediction that more slots still pay*, which the two measured points support (1,574 → 25.92, 2,522 → 50.19). Marked **arithmetic from the paper's rule, not measured here**. |
| **FineMoE** — arXiv [2502.05370](https://arxiv.org/abs/2502.05370) | 2025 | fine-grained expert selection patterns + **semantic hints from the prompt** to drive prefetch/cache/offload; −47% latency, +39% hit rate | **six-GPU testbed** | **No.** Six GPUs, HuggingFace-transformers prototype; the mechanism (prompt semantics) is orthogonal to a 34-SM single card and would need a new input path. |
| **Ostinato** — arXiv [2611.11533](https://arxiv.org/abs/2611.11533) | 2026-11-08 | a short prefill sends **84% of its expert routes to a stable set that fits in VRAM**; compute those on the GPU every pass, split the rest; one fixed-shape CUDA graph per GPU per decode step; **−22-55% median wait after a tool output**, sessions 1.4-2.5× sooner | **4× RTX 4090D**, DeepSeek-V4-Flash, vs llama.cpp and KTransformers | **Yes for the *split* rule, no for the hardware.** The 84% stable-route observation is a routing statistic of a 512-expert-class model and should be checked on Flash-Next; the "one fixed-shape CUDA graph per decode step" is what this fork already does — the tree records ~380 launches a token with graphs against 1,296 without (`docs/FLINT.md:130`). Four GPUs is not our shape. Engine A. |
| **HybriMoE** — arXiv [2504.05897](https://arxiv.org/abs/2504.05897) | DAC'25 | dynamic intra-layer CPU/GPU **workload balance**, impact-driven inter-layer prefetch, score-based caching; 1.33× prefill / 1.70× decode over SOTA hybrid | **NVIDIA RTX A6000**, 10 CPU cores | **Partly.** A6000 is Ampere 48 GB / 768 GB/s; the *balance* rule is what this fork already ships as `STRATA_PREFILL_CPU_SHARE=auto` (`docs/DETAILS.md:1781-1810`, −19…−35% prompt time on 512-1,000-token chunks) with a measured caveat: it **lost 2% at 2,048 tokens on a 5070**. So the mechanism is present and already tuned; HybriMoE's contribution is the inter-layer prefetch, which `STRATA_LOOKAHEAD` covers. |
| **AdaOffload** — arXiv [2611.10219](https://arxiv.org/abs/2611.10219) | 2026-11-06 | multi-GPU adaptive expert offloading, GPU-CPU co-execution, heterogeneous hardware; 1.9-15.5× throughput | **L4, Pro6000, B200**; 284B-753B models | **No.** Multi-GPU, datacentre parts, 100B+ models. |
| **FluxMoE** — arXiv [2604.02715](https://arxiv.org/abs/2604.02715) | 2026-04 | expert **paging** abstraction (`PagedTensor`) decoupling experts from residency | large-batch offline serving | **No** — the stated target is "large-batch, long-output offline workloads", the opposite of batch 1. |
| **OLED-MoE** — arXiv [2609.33385](https://arxiv.org/abs/2609.33385) | 2026-09 | inter-iteration locality for **diffusion** LLMs | diffusion LMs | **No** — different decoding model. |
| **KTransformers** (system, not paper) | 2026 | expert placement as a policy axis | 1×4090 row: Qwen3-30B-A3B 8+ it/s | **Yes** — see Part 1. Engine A. |

## 2.3 Quantised matmul on INT8 tensor cores — **Engine A and B** (the shared `mul_mat_q`)

The tree's numbers to beat: `IMMA.16832.S8.S8` present in **every** quantised `mul_mat_q` instantiation and
`HMMA` in none; IQ3_XXS's tile-128 kernel is **3.8% IMMA** with `I2FP` 15.1% + `FMUL` 15.3% + `FFMA` 15.1% =
**45% dequantisation** around it; MMQ measures **69.4 TOPS = 39% of the card's 176.5 TOPS**
(`docs/PREFILL-GAP.md` §5b, `docs/KV-INT8-PATHS.md` §1.2 Correction).

| paper | venue/year | claim | hardware | transfers? |
|---|---|---|---|---|
| **QServe / QoQ (W4A8KV4)** — arXiv [2405.04532](https://arxiv.org/abs/2405.04532) | MLSys 2024 | existing INT4 methods waste **20-90% of runtime dequantising**; **progressive quantisation** (low-bit high-order + high-bit low-order correction) makes dequantisation cheap because it is *integer* work, not float; compute-aware reordering + register-level parallelism; 1.4× (Llama-3-8B) and **3.5× (Qwen1.5-72B) on L40S** vs TensorRT-LLM | **A100 and L40S** | **Yes, with an argument.** L40S is **sm_89** — same instruction set, same 4:1 INT8:FP16 ratio, same `IMMA.S8.S8`. What differs is size (4× SMs) and bandwidth (3×). The claim is about *instruction mix*, which does not depend on SM count, so it transfers directly to the 45% dequant share. **Both engines.** |
| **Marlin** — arXiv [2408.11743](https://arxiv.org/abs/2408.11743) | 2024 | FP16×INT4 at near-ideal 4× up to batch **16-32**; the technique list is dequant/tensor-core interleaving, async loads with evict-first, double-buffered smem, striped partitioning | "ideally partitioned on an **A10**"; README states cc ≥ 8.0, **"Ampere or Ada"** | **Yes for the *ordering* idea, partly for the shape.** Marlin is W4A16 (FP16 accumulate, HMMA); this tree is W2/3A8 → IMMA with int32 accumulate. The transferable part is the *scheduling discipline*: issue the dequant for tile n+1 while tile n's IMMA runs. The tree's 45% dequant share is a scheduling symptom, not an arithmetic one. **Both engines.** |
| **Atom** — arXiv [2310.19102](https://arxiv.org/abs/2310.19102) | MLSys 2024 | low-bit **weight** quant with a mixed-precision *activation* split (outlier channels in higher bits) to keep the GEMM on INT4 tensor cores | A100 / V100 class | **Partly.** The idea (keep the hot channels at higher precision, everything else at the tensor core's native width) applies to `mul_mat_q`'s activation side, which the tree quantises to q8_1 for every column. Unverified on Ada. |
| **llama.cpp `mmq-config-ampere.cuh`** (code, not paper) | 2026 | per-type `(nthreads, occupancy, I, J, K_vram, stream_k, fallback)` tables; **Ada uses the Ampere table verbatim** (`ggml_cuda_mmq_get_config` → `..._get_config_ampere` for anything ≥ Volta that is not Blackwell) | tuned per architecture by the project | **Yes, and this is a finding in itself**: there is **no Ada MMQ table** in llama.cpp either. The Ampere tile/occupancy choice is what runs on our card. Any `J`/`I`/occupancy retune is unexplored territory for both projects. **Both engines.** |

## 2.4 KV cache compression where the win is **footprint** — **Engine A** (and B's k8v4/k6v4)

The tree's measurement that forbids the usual framing: at 256K on this card **q8_0 4.96 ns/cell < k8v4 5.88 <
fp16 8.32** (`docs/QUANT_MICROBENCH.md:619`), and a direct IMMA-vs-scalar A/B came out **0.90 / 0.92 / 1.00 /
1.03×** across 4,095 → 262,143 cells even though the instruction is **3.3× faster in isolation**
(`docs/KV-INT8-PATHS.md` §32-33). Any paper claiming a *speed* win from fewer KV bytes must explain why its card
differs.

| paper | claim | hardware | transfers as **footprint**? |
|---|---|---|---|
| **KIVI** — arXiv [2402.02750](https://arxiv.org/abs/2402.02750), ICML 2024 | keys quantised **per channel**, values **per token**, 2-bit; 2.6× less peak memory, 4× batch, 2.35-3.47× throughput | A100-class serving, batch ≫ 1 | **As footprint: yes. As speed: no.** The per-channel-vs-per-token asymmetry is a *quality* result and is format-independent. The tree already applies the same split (k8v4 = int8 K + rotated q4_0 V, `include/strata/kernels/kv_stream.hpp:48`). The throughput claim is a batching claim, and this card runs batch 1. Engine A. |
| **SmoothAttention** (inside QServe) | rotation to make 4-bit K/V quantise without outliers | L40S / A100 | **Yes as a quality method**, and the tree already does it (the k8v4 V is `FWHT`-rotated, `docs/QUANT_MICROBENCH.md` §10). Engine A. |
| **k6v4 / k4v4** (this tree's own) | 688 / 560 B a cell vs k8v4's 816 | this card | **Footprint only**, and measured so: k6v4's decode is **2-13× slower** a cell below ~4,096 and only at parity above (`docs/BENCH-INSTR-PATHS.md` §6). `IMMA.S4.S4` is **Ada-only** (287 `SRCFMTA_U8_S8_U4_S4` hits in the sm_89 dump, **0** in sm_90 — `docs/KV-INT8-PATHS.md` §1.1), so k4v4 is the only format with an instruction Hopper lacks; nobody has written its reader. Engine A (QSA) and B (`attend`/`fdec`). |

**A quantitative bound on the whole section.** 2,051 selected cells × 13 stacks × 816 B = **20.7 MiB of KV read a token**: **0.38%** of a 19,924 µs token at the 288 GB/s peak, or **1.7%** at the 65 GB/s
the k8v4 kernel actually reaches. So no change to how the KV is *read* can be worth more than ~1%. Even a perfect 4× compression frees 2.59 → 0.65 GiB at 256K, i.e. **1.94 GiB ≈ 935 slots (+37% of the 2,522)**, and the token-time value of those slots is **unmeasured** — the only two published points (1,574 → 25.92, 2,522 → 50.19) change `--spec` as well as the slot count. So the KV-format prize on this card is expressed **in slots**, and its time value is the open question, not a number any paper supplies.

## 2.5 Speculative decoding on a single consumer GPU — **Engine A (MTP) and B**

The tree's baseline: MTP **5,987 of 6,319 accepted (94.7%), 3.72 tokens a round** at 256K
(`docs/REAP.md:33`), and on a different prompt 61.0% (2K code) / 73.3% (128K unique)
(`docs/REAP.md:94-95`). So acceptance is prompt-dependent by 34 points, which is the number a paper has to beat.

| paper | venue | claim | hardware | transfers? |
|---|---|---|---|---|
| **EAGLE** — arXiv [2401.15077](https://arxiv.org/abs/2401.15077) | ICML 2024 | draft at the **feature** level (second-to-top-layer), 2.7-3.5× latency on LLaMA2-70B | training on 4×A100 or a **3090 node**; evaluation parts not stated on the abs page | **The training recipe no, the tree shape yes.** Its value here is negative: this model already ships an MTP head *inside* the checkpoint, so the draft-quality problem EAGLE solves is already solved by the model. Engine A. |
| **EAGLE-2** — arXiv [2406.16858](https://arxiv.org/abs/2406.16858) | EMNLP 2024 | **context-aware dynamic draft tree**; 3.05-4.26× | not stated on the abs page (**unverified**); comparison is via Spec-Bench | **No, probably.** A dynamic tree multiplies the **verify window width**, and this fork's verify window is exactly where the expert cache hurts (T=1..4 gate/up times, `docs/EXPERTDEC.md:96`; `--route-tail-skip` exists for this). A wider window reads more distinct experts from RAM. The tree has already rejected draft trees once for this reason (`docs/DETAILS.md` on sampled drafting and the coupled/Gumbel experiments: acceptance rose, **decode did not move**). Engine A. |
| **Medusa** — arXiv [2401.10774](https://arxiv.org/abs/2401.10774) | 2024 | multiple decoding heads on the last hidden state, tree attention, no separate draft model | A100-class | **No** — same window-width objection, and MTP already provides a trained head. |
| **Mirror-SD** — arXiv [2510.13161](https://arxiv.org/abs/2510.13161) | 2025 | breaks the serial draft barrier, "mirror" speculative decoding | not Ada | **No.** |
| **VOCABTRIM** — arXiv [2506.22694](https://arxiv.org/abs/2506.22694) | 2025 | prune the drafter's LM head vocabulary | — | **Yes, and it is already done here**: the fork ships a **draft vocabulary** (`--draft-vocab`, `draft_vocab*.bin`) and measured an Italian-built vocabulary at 54.4 vs 51.9 tok/s (`bench/results/2026-10-05-community-2x-rtx-4060ti/README.md`). Nothing to take. |

---

# Part 3 — published numbers on this card class, and where this engine sits

## 3.1 The envelope

All rows are **one** RTX 4060 Ti 16 GB unless marked. "This fork" rows are this tree's own measurements and are
included to place the external ones.

| model | quant | ctx | prefill tok/s | decode tok/s | engine | VRAM/RAM split | source |
|---|---|---:|---:|---:|---|---|---|
| **Qwen3.8-Flash-Next REAP-320** (320/512 experts) | pack, k8v4 KV | 262,144 | **1,249** | **50.19** | **this fork (`strata`)** | 2,522 slots (5.22 GiB VRAM) + 31.42 GiB page-locked RAM, 0 file reads | `docs/REAP.md:29-36` |
| Qwen3.8-Flash-Next REAP-320 | pack, plain mmap | 262,144 | — | 25.92 | this fork | 1,574 slots, reads 52.6 MB/round | `docs/REAP.md:48` |
| **Qwen3.8-Flash-Next** (full 512 experts) | 3.05 bpw EXL3 | 262,144 | 1,627 | **20.0** | lean-qwenflash | 16 GB VRAM capped, **8 GB RAM**, rest NVMe `O_DIRECT` | [README](https://github.com/MaxDevv/lean-qwenflash) |
| Qwen3.8-Flash-Next | 3.05 bpw EXL3 | 100,000 | 1,536 | 20.8 | lean-qwenflash | as above | same |
| Qwen3.8-Flash-Next | 3.05 bpw EXL3 | 8,192 | 1,080 | 28-33 prose / 19-22 code | lean-qwenflash | 16 GB / 8 GB | same |
| Qwen3.8-Flash-Next | UD-IQ4_XS | 131,072 | 880 / 1,917 / 2,221 (4K/32K/128K) | **54.4 / 56.6 / 54.7** | **this fork, 2× 4060 Ti** | 4,083 + 3,678 slots, layer split | `bench/results/2026-10-05-community-2x-rtx-4060ti/README.md` |
| Qwen3.8-Flash-Next | Q2_0 | 65,536 | 716 (3.5K) | **48.1 → 55.7** | this fork (0.1.39 / improved) | 7,475 slots, 9.62 GiB | `bench/results/2026-10-05-community-e5-2673v3-rtx-4060ti/README.md` |
| Qwen3.8-Flash-Next | IQ3_S | 65,536 | 439 (3.5K) | **30.5 → 35.0** | this fork | 3,531 slots, 9.65 GiB | same |
| Qwen3.8-Flash-Next | IQ2_XS | 100,000 cells | 28,912-token prompt | **48.23** (5-run median) | modified Strata fork | 3,796 slots, 20 GiB host RAM, q4_0 KV | `bench/results/2026-10-08-community-rtx4070-iq2-xs/README.md` (**RTX 4070 12 GB**) |
| Qwen3.8-Flash-Next | IQ3_S | 131K | 3,451 / 4,277 / 4,166 | **146.0 / 132.2 / 124.9** | Strata 0.1.41 | **RTX 5080 16 GB + 4060 Ti 8 GB helper** | `bench/results/2026-10-09-community-rtx-5080-4060ti-0.1.41/README.md` |
| **Qwen3.6-35B-A3B** | UD-Q4_K_XL | 262,144 | 76.7 | **122.5** | not stated | 2× 4060 Ti | [llamaperf](https://llamaperf.com/gpu/rtx-4060-ti-16gb/) (Oct 29 2026) |
| Qwen3.6-35B-A3B | Q5_K_XL, q8_0 KV | 131,072 | 132.6 | **19.53** | llama.cpp | 1× 4060 Ti, i7 7th-gen, 48 GB DDR4 | same |
| Qwen3.8-27B (dense) | UD-Q4_K_XL | 200,000 | 55.5 | 34.3 | not stated | 2× 4060 Ti | same |
| **Qwen3.8 27B (dense)** | 3-bit | 4K | 800 | **20.0** (median, one run) | mixed community, incl. 1 Strata report | 1× 4060 Ti | [llamaperf GPU page](https://llamaperf.com/gpu/rtx-4060-ti-16gb/) |
| Qwen3 14B (dense) | Q4_K | 16K | 918 | **22.4** | llama.cpp-class | 1× 4060 Ti, full VRAM | [hardware-corner](https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-4060-ti-16gb/) |
| Qwen3 8B (dense) | Q4_K | 64K | 392 | **13.0** | llama.cpp-class | 1× 4060 Ti, full VRAM | same |
| gpt-oss 20B (MoE) | MXFP4 | 128K | 780 | **31.1** | llama.cpp-class | 1× 4060 Ti, full VRAM | same |
| GLM-5.3-Flash 320B | UD-IQ3_XXS | 131,072 | 165-166 (2K) / 385-386 (8K) | **13.0** (14.0 with `=2`) | Strata v1.1.42 | 2× 4060 Ti, SATA SSD | llamaperf, Nov 1 2026 |
| **GLM-5.3-Flash** | EXL3 3.05 bpw | 131,072 | 664-969 | **19.5** (55 GiB RAM cap) / 14.7 (16 GiB cap) | glm53-flash-offload | **1× RTX 3090 24 GB**, NVMe | [README](https://github.com/sybil-solutions/glm53-flash-offload) |
| **Ornith-1.5-9B** | — | 131,072 | **~1,050** | — | **this fork (`strata-poly`, post-MMQ gate)** | — | brief; pre-gate 229.1 (`docs/PREFILL-GAP.md` §5b) |
| Flint-abliterated-12G | q8_0 / k8v4 | 131K | **1,945 / 1,948** | **46.4 / 48.1** | this fork (`strata-poly`) | 2 GiB reserve | `docs/FLINT.md:110-113` |
| Flint-abliterated-12G | q8_0 / k8v4 | 256K | 1,268 / 1,277 | **30.7 / 30.2** | this fork | 2 GiB reserve | same |

**Does anything here beat us?** Two rows, and both are explained:

- **Qwen3.6-35B-A3B at 122.5 tok/s on 2× 4060 Ti** (llamaperf) is a **3B-active** MoE — 1/3 the active
  parameters of a 320-expert Flash-Next layer set and, critically, **fully VRAM-resident** at UD-Q4_K_XL across
  two 16 GB cards. Not comparable.
- **Qwen3.8-Flash-Next at 146.0 tok/s on a 5080 + 4060 Ti** is a **5080** (sm_120, 960 GB/s, 84 SM): 3.3× our
  bandwidth. Scaled by bandwidth alone that is 44 tok/s equivalent, i.e. **below** our 50.19.

**Does anything here suggest we are far from the wall?** No. Every comparable-row is **at or below** this fork:
- Same 16 GB Ada class, same model size class, **more** RAM pressure: lean-qwenflash gets **20.0** at 262K with
  8 GB RAM. We get **50.19** with 31.42 GiB page-locked and 0 file reads. Our number is 2.5× on **0.44×** the
  bandwidth.
- Same model, smaller quant, **half the card**: Q2_0 at 48.1-55.7 and IQ3_S at 30.5-35.0 on a single 4060 Ti with
  a DDR3 Xeon. Our 50.19 with REAP-320 (a 320-expert prune) and a 34 GiB resident budget is in the same band, from
  a bigger model, at 4× the context.

**The published envelope for this card class is roughly 20-56 tok/s decode for a 30B-125B MoE at long context,
and this engine is at the top of it.** There is no external row showing 100+ tok/s on one 4060 Ti for a model of
this class at 256K. That is the single most important calibration in this document: **the remaining headroom is
not in someone else's published kernel; it is in the parts of the token this tree has not yet removed.**

## 3.2 Anyone running a 512-expert MoE with partial residency on 16 GB?

**Yes — exactly one project, and it is the row above.** `MaxDevv/lean-qwenflash` runs the **full 512-expert**
Qwen3.8-Flash-Next at 262K on **16 GB VRAM and 8 GB system RAM**, everything else on NVMe, and reports **20.0
tok/s at 262K** and **28-33 at 8K**. Two things follow:

1. **Partial residency of 512 experts on 16 GB is solved, and 20 tok/s is where it lands** when the miss path is
   NVMe. This fork's 50.19 comes from the miss path being **page-locked RAM with 0 reads** (`docs/REAP.md:36`) —
   i.e. **the resident-RAM mode is worth 2.5× over the best published NVMe-streaming engine**, on the same model
   and the same context. That is the fork's most defensible result.
2. **Nobody publishes a 512-expert partial-residency number on a 4060 Ti.** lean-qwenflash's cards are 4090 and a
   3080 Laptop; KTransformers' consumer row is a 4090; glm53-flash-offload is a 3090. The 4060 Ti rows in the
   community set are all *pruned* models (REAP-320, Coder, 256-expert). So the specific configuration in
   `docs/REAP.md` has **no external peer** — which is why the envelope has to be built from model-class matches
   rather than card matches.

---

# Part 4 — why this engine performs as it does, and the top five

## 4.1 The two budgets, written out

**Engine A, one decode token at 256K = 19,924 µs** (50.19 tok/s). What is inside it:

| term | value | source |
|---|---|---|
| QSA decode attention, 12 layers | **0.270 ms = 1.4%** | 0.0225 ms a layer, `docs/QUANT_MICROBENCH.md:594-601` |
| QSA KV read | 20.7 MiB = 76 µs at the 288 GB/s peak (**0.38%**); 335 µs at the measured 65 GB/s (**1.7%**) | §2.4 arithmetic; `docs/BENCH-INSTR-PATHS.md` §1 |
| Expert slots held | 2,522 × 4.43 MB = **5.22 GiB VRAM** | `docs/REAP.md:34,81` |
| Everything else (expert path, dense projections, head, host gaps) | **~19.5 ms = 98%+** | remainder |

A token is **3.72 accepted** (`docs/REAP.md:33`), so a verify round is ~74 ms for ~3.72 tokens. The engine is
therefore not attention-bound and not KV-bound at decode: **it is bound by the expert path** — the per-slot matvecs,
the routing, and whatever non-kernel time the token carries (Engine B's equivalent on this box is **4.4 ms of a
13.4 ms token**, `docs/FLINT.md:130-131`; Engine A's own figure does not exist and is item 2 below).

**Engine B, one decode token at 131K on Flint**: `fdec` is **16.2%** of a 105 s / 8,192-token window at 20K, and
at 250K `fdec` is **18.3 ms of 31.8 ms = 57.7%**, reading **3.83 GB a token at 209 GB/s** (`docs/FLINT.md:136-141`).
The expert matvecs are 5.3 ms. So **Engine B at long context is attention-bound, and Engine A is not** — because
Engine A reads 2,051 cells and Engine B reads the whole prefix. That is the whole structural difference and it
justifies separate rankings.

## 4.2 The top five, ranked by expected gain per day of work

The ranking is **Engine A first**, because `strata` is the model the fork exists for. Engine B appears where it has
a cheaper win. Each item states the mechanism, the arithmetic, what to build, what breaks, and whether external work
already solved it.

| # | engine | item | expected gain | cost | kill test |
|---|---|---|---|---|---|
| 1 | **A** | per-layer expert-slot allocation | single-digit to ~10% decode | ~1 day | read `expert-profile.bin` (1 hour) — if the per-layer distribution is flat, stop |
| 2 | **A** | instrument, then remove, non-kernel time per token | +5…+20% decode (bracketed) | 1 day to measure, 2-5 to fix | `nsys` one 256K token: if gaps < 5%, stop |
| 3 | **both** | `mul_mat_q` dequantisation share (Ada table + pipeline) | 1.29× on the matmul → prefill +16-28% | 3-5 days | an Ada `J`/`I`/occupancy sweep — if no cell beats Ampere's, the table is not the lever |
| 4 | **B** | reader choice as a function of SM count | a few % to ~15% for k8v4 at 4K-12K | ~1 day | `poly_fdec_threshold` sweep; the rule is one expression |
| 5 | **A** | KV bytes → expert slots (k6v4 port) | single-digit %, by slots only | 5+ days | the 4× compression bound: does 1.9 GiB of slots buy a measurable tok/s? |

Numbers below are arithmetic from the cited measurements. Every estimate is labelled; nothing here was run.

---

### 1. **A** — Make every expert slot pay: per-layer slot allocation

**Mechanism.** Engine A's decode token is **19,924 µs** and its QSA attention is **270 µs = 1.4%**
(`docs/QUANT_MICROBENCH.md:601`); the KV read is **0.38-1.7%** (§4.1). So **98%+ of the token is the rest of the
model** — the routed expert matvecs, the dense projections, the head, and the host gaps. The expert path is the
largest addressable sub-term, and the fork feeds it from a **uniform** allocation: `ExpertCache::layer_slot_range`
gives layer `l` the range `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`
(`src/core/expert_cache.cpp:749-757`) — **52.5 slots a layer for all 48 layers**, whatever each layer's routing
actually does. KTransformers exposes precisely this as an axis (`--kt-num-gpu-experts`, **per MoE layer**, plus
`frequency` placement from an activation-statistics file and `--kt-enable-dynamic-expert-update`,
`doc/en/kt-kernel/experts-sched-Tutorial.md`), and the routing literature puts the useful cache at ≈2× the active
experts (`arXiv 2505.16056`).

**Arithmetic.** 2,522 slots / 48 layers = **52.5 slots a layer against 10 active experts** = 5.25× active, above
the 2× knee the paper identifies — which predicts more capacity still pays. If the hottest quarter of the layers
carries most of the routing mass, moving 20% of the slots out of the coldest layers into the hottest raises
hot-layer capacity from 52 to ~84 (**+60%**) at identical bytes; on a hit-rate curve that is worth **single-digit
percent to ~10%** of decode. The honest ceiling is the measured **25.92 → 50.19 tok/s** spread between 1,574 and
2,522 slots (`docs/REAP.md:48-49`) — but that pair also changes `--spec`, so the slope is **confounded** and must
not be used as a per-slot rate (§4.2 item 5 shows why it cannot be extrapolated).

**What has to be built.** The storage side **already exists**: `ExpertCache::open_sized` takes
`std::vector<int64_t> slot_bytes` (`src/core/expert_cache.cpp:683`, declared at
`include/strata/core/expert_cache.hpp:100`). The work is (a) a per-layer vector derived from `expert-profile.bin`
instead of `slots_/n_layers_`, (b) a flag, (c) keeping the adaptive tier coherent, since `--adapt-swaps 96` is a
global count and a non-uniform map changes what "swap" means.

**What could go wrong.** Routing may be load-balanced by training — the paper above says *global* balance and
*local* consistency coexist, and if Flash-Next is uniformly balanced the whole item is worth zero. **That is why it
is first and why its first hour is a kill test**: read the profile and look at the per-layer concentration; if it
is flat, stop and move to item 2.

**Already solved externally?** Yes, as a *policy*, by KTransformers. No external project publishes this for a
34-SM Ada card, and none has a 320-expert REAP prune.

---

### 2. **A** — Measure, then remove, the non-kernel time in a decode token

**Mechanism.** There is **no `nsys`/`ncu` timeline of one Engine A decode token in this tree** — the profile in the
brief is of the *retired* attention kernel. On the same card and the same model family, Engine B's 20K decode spends
**4.4 ms of a 13.4 ms token outside kernels** (`docs/FLINT.md:130-131`) and its 105 s window is only **70.8%
GPU-busy** (`:136`). Engine A carries *more* per-step host work than Engine B — routing, QSA selection, KV-stream
resolution, the adaptive tier — while having *fewer* launches thanks to graphs (~380 a token against 1,296,
`docs/FLINT.md:130`). The class of fix is proven on Ada: `Juud_engine` reports **+9.6-15.6% decode on an RTX 4090**
from shortening the worker spin after the expert batch and skipping activation quantisation for GPU-resident tokens
(cited in `pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md` §3.3); Hardin22's "fewer host waits and kernel launches
per step" is the same lever; and this fork already ships `--adapt-async`, `STRATA_ADAPT_NOWAIT` and
`--pipeline-windows`.

**Arithmetic.** Bracketed, because the number does not exist yet. If Engine A's gap is **5%** of 19,924 µs, halving
it is **+2.6%**. If it is Engine B's **33%**, halving it is **+20%**. The 4090 measurement (+9.6-15.6% from two
small changes) sits between those, which is why the honest expectation is **+5…+20%** and why the measurement comes
first: it costs one day and it decides whether the next three days are spent here or on item 3.

**What has to be built.** (a) An `nsys` timeline of one 256K decode token, split kernel / gap, with the clock pinned
— the tree's bench discipline (`flock -w 60 /tmp/opencode/gpu.lock`, `nvidia-smi -lgc 2820`) already does the rest.
(b) Then, in order of likely size: move KV-stream resolution and adaptive-swap decisions onto the device (the layer
split already uses device-side flags); collapse the remaining per-layer launches; and check whether the
`--adapt-every 1` setting is paying for itself at 256K.

**What could go wrong.** A graph that pins the miss path defeats adaptation — the engine already treats these as
coupled (`docs/EXPERTDEC.md:42`: "captured graphs keep the kernels they captured"), so a change here can silently
disable the adaptive tier and look fast in a short run. And if the gap turns out to be DRAM latency inside kernels
rather than host gaps, this item collapses into item 1.

**Already solved externally?** Partly. `Juud_engine` (+9.6-15.6% on a 4090) and Hardin22's host-wait work both
report results without a timeline. llama.cpp's answer to the same problem is CUDA graphs, which this tree already
has.

---

### 3. **both** — Cut `mul_mat_q`'s dequantisation share: an Ada config table, then a pipeline

**Mechanism.** The tree measured `mul_mat_q` at **69.4 TOPS = 39% of the card's 176.5 INT8 TOPS**
(`docs/KV-INT8-PATHS.md` §1.2, which also corrects the tree's own documented peak from 88 to 176 TOPS), with
**45% of its instructions** on `I2FP`/`FMUL`/`FFMA` around **3.8% `IMMA`** (`docs/PREFILL-GAP.md` §5b, SASS table). Those instructions
exist to turn packed weights into int8 operands, and two published systems attack exactly that: **QServe** names
the overhead at **20-90%** of runtime and removes it with *integer* progressive quantisation (measured on **L40S,
sm_89** — the same instruction set as this card), and **Marlin** removes it by *scheduling* — issue the dequant for
tile *n+1* while tile *n*'s tensor op runs, with async global loads and double-buffered shared memory. Neither
needs new hardware. This is the only item in the top five that moves **both** engines, and the only one that moves
**prefill**.

**Arithmetic.** The gate fix that made a dense model use this kernel was **4.31× at 131,072** (986.4 vs 229.1 tok/s,
`docs/PREFILL-GAP.md` §5b) and **7.45× at 40,960**; the brief records ~**1,050 tok/s at 131K** as the current
Ornith number. If dequantisation is 45% of the instructions and issues on the same pipe as the `I2FP`/`FMUL`
sequence, hiding half of it is **1/(1 − 0.45/2) ≈ 1.29×** on the matmul. Applied to the measured numbers:
**Engine B Ornith ~1,050 → ~1,200-1,350 tok/s at 131K**, at byte-identical output (the tree gates MMQ changes that
way: `native_grouped_parity`, the `STRATA_MMQ_DENSE=0` A/B). For **Engine A, prefill 1,249 → ~1,450-1,600 tok/s**
(209.8 s → ~165-180 s) is an **extrapolation, not a measurement**: the 45% dequant share and the 1.29× were both
measured on a *dense* 27B matmul (`docs/PREFILL-GAP.md` §5b), and Engine A's 256K prefill has a routed-expert
term whose type mix and batch shape differ. The direction is right (the same `mul_mat_q` runs), the magnitude is
not verified for the MoE prefill.

**What has to be built.** (a) An **Ada entry** in the MMQ config table. There is none in llama.cpp either —
`ggml_cuda_mmq_get_config` sends everything from Volta to pre-Blackwell to `..._get_config_ampere`
(`ggml/src/ggml-cuda/mmq.cuh`), so Ada runs Ampere's `(nthreads, occupancy, I, J, K_vram, stream_k)` choices
unmodified. A sweep over `J` ∈ 8…128 and occupancy at this card's 99 KB of shared memory is unexplored for both
projects. (b) A dequant/MMA software pipeline. (c) The parity gate.

**What could go wrong.** 99 KB of opt-in shared memory is the ceiling. `Mmq::fits` already picks `J` by
`mmq_get_nbytes_shared(c, cc) <= dev.smpbo` (`src/poly/mmq.cu:88-92`), so a second in-flight tile may not fit at the
tile size that wins today, and a smaller tile raises the launch count — which fights item 2. Also: **this is a
prefill win on a card whose product is decode.** It cuts the 209.8 s cold read, not the 50.19 tok/s.

**Already solved externally?** Partly, and this is the one item where the external work is directly usable:
QServe's arithmetic (L40S, Apache-2.0), Marlin's scheduling (cc ≥ 8.0, Ampere **or Ada**, Apache-2.0), and
ik_llama.cpp PR #2831's grid change (**+8.3% / +15.1% / +8.0% / +9.1% / +9.7%** at 64 / 128 / 512 / 1024 / 2048
columns on Qwen3.8-Flash-Next Q2_K, MIT) — which is exactly a verify-window batch size.

---

### 4. **B** — Choose the decode reader from the SM count, not the literal 4,096

**Mechanism.** `Qwen35::attn` picks the reader by position alone: `fdec_*` from `fdec_from_ = 4096`
(`src/poly/qwen35.cpp:395`). `fdec_kernel` splits the cache into 1,024-row blocks, one per (split, KV head), so the
grid is `ceil(rows/1024) × kv_heads` — at 4,096 rows with group 4 that is **16 blocks on 34 SMs**
(`docs/BENCH-INSTR-PATHS.md` §3). Measured: q8_0's `fdec`/`attend` ratio is **1.059 at 4,096** and **0.899 at
5,120**, crossing at **≈4,470-4,570**; and **k8v4's `fdec` is 2.04× slower than `attend` at 4,096**, still 1.02×
at 12,288. So for k8v4 the engine runs the **slower** reader across a wide band, and for q8_0 across a narrow one.

**Arithmetic.** The fix is `ceil(SMs / kv_heads) × kSplit`, with the measured floor: for group 4 on 34 SMs that is
**9 × 1024 = 9,216** rows for full occupancy, but break-even is at ~4,500, so the rule keeps the floor. The band
where k8v4 currently pays 1.02-2.04× runs from 4,096 to ~12,288, i.e. a **3× window**. Attention is **16-58%** of
Engine B's token at 20K-250K (`docs/FLINT.md:136-141`), so a 2× on attention inside that window is **a few percent
end to end**, more for k8v4 at the short end of it, near zero above 20K where `fdec` is correct. Modest, but it is
**one expression plus a sweep**, and the sweep doubles as the evidence that the rule is right for groups 4/6/8/12/16.

**What has to be built.** The expression, plus `poly_fdec_threshold <dim> <n_head> <kv_heads>` at 1,024-row
granularity (`docs/BENCH-INSTR-PATHS.md` §7 item 2). The bench already takes any geometry.

**What could go wrong.** Break-even (~4,500) is **below** full occupancy (9,216), so a naive "enough blocks" rule
overshoots into splits that do not pay. And **k6v4 has `fdec_from_ = 0` forced** (`qwen35.cpp:399`) because
`attend()` refuses six-bit keys with a `throw` — the rule must not route k6v4 to a reader that throws.

**Already solved externally?** The *shape* of the answer, twice: llama.cpp routes Ada decode per-arch and per-`ncols`
(`ggml/src/ggml-cuda/fattn.cu:701-709`), and TensorRT-LLM's XQA picks **different KV tile widths for sm_89
(64 B / 32 cells) than for sm_90 (128 B / 64 cells)** (`cpp/kernels/xqa/mha.cu:92-100`). Both treat the reader
choice as a function of the card; this tree's is a literal.

---

### 5. **A** — Compress the KV for **slots**, and only for slots

**Mechanism.** The tree proved that fewer KV bytes reads **slower** here (q8_0 **4.96**, k8v4 **5.88**, fp16
**8.32** ns/cell at 256K — `docs/QUANT_MICROBENCH.md:619`) and that substituting a 3.3×-faster `IMMA` for the QK
product moves nothing end to end (**0.90-1.06×**, `docs/KV-INT8-PATHS.md` §33). So the only KV question left is
**how many expert slots the freed bytes buy** — i.e. it is item 1's currency, not a speed lever of its own.

**Arithmetic.** k8v4 is **816 B a cell** on `strata`, k6v4 would be **688** — **128 B a cell**
(`docs/STRATA-QSA-CENSUS.md` §7, which corrects an earlier 16 B error). At 262,144 cells × 13 stacks: 2.59 GiB →
2.18 GiB, so **0.40 GiB at 256K** (0.81 GiB at 512K), and the same bytes buy **+9.5% context** at fixed VRAM.
**The token-time value cannot be quoted from existing data.** The 1,574 → 2,522 pair gives a local slope of
19.7 µs a slot a token, but 2,522 × 19.7 µs = 49.7 ms against a 19.9 ms token — the slope extrapolates to a
negative intercept, so it is not a production function and a naive linear use would over-claim ~+19%. The defensible
statement is **single-digit percent**, and the 4×-compression ceiling (2.59 → 0.65 GiB, freeing 1.94 GiB ≈ **935 slots**, **+37%** of the cache) is
the number an experiment should be designed around.

**What has to be built.** One pointer (`k_q6`), one arm in `attn_kv_mode`, the `kv_stream` arm (three runs, one
length change — §5.2 of the census says the mechanism already carries it), and a prompt-side `load8_q6`. It is a
dozen format-protocol sites across every reader (`docs/STRATA-QSA-CENSUS.md` §5.1).

**What could go wrong.** Two things, both measured elsewhere in the tree: it has **no decode gain** by construction,
and the sub-8-bit **store** is slow — k8v4 and q4_0 append at **19 and 13 GB/s** against fp16's 142
(`docs/BENCH-INSTR-PATHS.md` §4) because the append quantises per value. k6v4 inherits that shape. Plus the prompt
path: k6v4's kernel exists and already emits the 64 `HMMA` that k8v4 does, so the risk is a sixth-bit unpack
feeding `mma16816` worse than `i8x2_to_h2` does — a measurable risk, not a missing tensor path.

**Already solved externally?** No. No published six-bit KV beats eight-bit on a 128-bit Ada card, and the two
papers that claim KV speed wins (KIVI, SmoothAttention) claim them at **batch ≫ 1** on A100/L40S. The tree's own
measurement is the state of the art for this card.

### Runners-up, with the reason each is sixth

- **B — `attend`'s scalar score loop below 4,096.** 560 scalar `FFMA`, zero tensor instruction. A real defect, but
  the tensor replacement is already measured at **+2.8-3.5%** (`docs/STRATA-QSA-CENSUS.md` §6), so do it inside
  item 4 rather than as its own project.
- **A — the `block_scores_tc_kernel` selector.** Already 48 `HMMA`, already on tensor cores; it is the **10.92%**
  term of a 256K *prefill* only because the prefill is long, and no KV format can move it
  (`docs/STRATA-QSA-CENSUS.md` §2).
- **Both — a wider verify window or a draft tree.** Wider windows read more distinct experts from page-locked RAM;
  the T=1→4 gate/up slope is already **+0.86 ms a row** (`docs/EXPERTDEC.md:96`) and `--route-tail-skip` exists to
  cut it. EAGLE-2/Medusa add width — the wrong direction on this card.

## 4.3 What is **not** worth doing, with the measured reason

1. **Putting Engine A's decode attention on tensor cores.** The only tensor-core decode arm is behind
   `STRATA_QSA_SPLIT_ATTN=1`, int8-int8 only, and its published value is **+2.8% / +3.5%**
   (`docs/STRATA-QSA-CENSUS.md` §6). The kernel is **0 `HMMA`, 0 `IMMA`, 0 `IDP`** in all twelve instantiations,
   and it still wins, because it is latency-bound on the value loop's dependent loads, not arithmetic. llama.cpp
   reached the same conclusion independently and routes Ada decode **away** from MMA
   (`ggml/src/ggml-cuda/fattn.cu:701-709`). Do not fund a tensor-core rewrite.
2. **A smaller KV format for decode speed.** Measured three ways on this card: q8_0 **4.96** < k8v4 **5.88** <
   fp16 **8.32** ns/cell at 256K; the IMMA A/B at **0.90-1.06×** despite a 3.3× faster instruction; and the KV
   read is **0.38-1.7% of a token** in Engine A. Fewer bytes has never bought a faster read here, only more capacity.
3. **k4v4 with `IMMA.S4.S4`.** It is the only format with an instruction Ada has and Hopper does not (287 vs 0
   `SRCFMTA_U8_S8_U4_S4`), and its footprint **ties k5v3 at 1,152 B/cell**, so even a perfect reader buys nothing
   over a format nobody has written either. It would need to beat k6v4's 9.35 ns/cell **and** hold quality.
   `docs/BENCH-INSTR-PATHS.md` §6 verdict: "No, not yet."
4. **`REDUCE12` / the gfx906 lane-cell path on Ada.** `reduce12_gfx906` is inside
   `#if defined(STRATA_HIP_GFX906)` and `STRATA_ATTN_LANECELL=1` measured **slower** on this card (8.69 → 10.06 and
   9.69 ns/cell, `docs/BENCH-INSTR-PATHS.md` §2). The 6-10% the control-flow experiment found is the cost of an
   out-of-line helper and **is not available as a change**: `warp_sum` is already `__forceinline__`, and ptxas
   outlines `shfl.sync` regardless.
5. **Wider draft trees / EAGLE-2-style dynamic trees.** They widen the verify window, which is the exact shape
   that costs experts: +0.86 ms a row from T=1 to T=4. And the tree already measured that raising acceptance
   (coupled/Gumbel drafting, 62.1% → 68.0%) **did not move decode** (`docs/DETAILS.md:128`).
6. **`STRATA_PREFILL_CPU_SHARE` beyond 1,024 tokens.** It is already the default on one CUDA GPU below 1,024
   tokens (−19…−35% prompt time), and the extension to 3,072 **lost 2% at 2,048 tokens on a 5070**
   (`docs/DETAILS.md:1789-1791`).
7. **Copying from Speedy or `qwen38-megakernel`.** "Personal & Non-Commercial" and AGPL respectively — a licence
   change for this MIT fork, for code whose measurements are on Ampere or 4×3090.
8. **A second GPU feature set (pipelined windows, tensor-parallel decode).** The flagship target is one card; the
   dual-GPU work in the family is real and measured but does not touch this configuration.

---

# Part 5 — what I could not verify

- **The hot decode kernel's stall mix at 256K.** The `long_scoreboard 45.5% / short 29.0% / occupancy 32.1% /
  DRAM 19%` profile is from `/tmp/opencode/SUMMARY.md` and was taken on `attn_chunk_kernel<3>` (1,432
  instructions), **not** on the shipped `attn_chunk_kernel_pre75<3,2,false>` (2,416 instructions). No profile of
  the shipped kernel exists in the tree.
- **Engine A's kernel-vs-gap split.** No `nsys`/`ncu` timeline of one 256K decode token for `strata`. Engine B's
  equivalent exists (`docs/FLINT.md:130-141`) and I borrowed its shape, not its numbers.
- **The slot slope.** 19.7 µs a slot a token is derived from two runs that differ in `--spec` as well as slot
  count; the brief says it is being re-measured. Every arithmetic use above is labelled accordingly.
- **`expert-profile.bin`'s per-layer concentration.** Not read for this document. Item 1's "+60% capacity on the
  hot layers" is an illustration of the mechanism, not a measurement of this model.
- **FlashDecoding++'s hardware** is not stated in the abstract or the intro; the paper would have to be read in
  full. **EAGLE-2's evaluation hardware** is likewise unstated on the abstract page.
- **`lean-qwenflash`'s licence**: GitHub reports none. Treated as no-code.
- **The KV share of a token has two values, and the tree only ever quoted one.** 20.7 MiB at the 288 GB/s peak is
  0.38%; at the 65 GB/s the k8v4 decode kernel actually reaches (`docs/BENCH-INSTR-PATHS.md` §1) it is **1.7%**. Both
  are small, but the second is the honest one, and any statement that "KV is 0.4% of a token" is a peak-bandwidth
  statement.
- **Star counts, PR states and last-push dates** are from the GitHub API on 2026-11-13 and will drift.
- **The `Hardin22` attribution**: `be4cf1c` and `3effa9d5` are commits authored by `imanu <imanu86@gmail.com>` in
  **upstream** Strata, and `3effa9d5`'s message says it ports Hardin22's `b1f03d33` (upstream PR #910). So
  "the Daily" is upstream's name for a kernel line whose *merge* came from Hardin22's fork. I could not find a
  commit that shows the chunk kernel itself originating in a repo named "Daily", and the search hits for that name
  (`PolyStrata-2080Ti Daily`, `docs/PROMPT_CACHE_TAIL.md:30`) point at a **2080 Ti** project, not an Ada one.
