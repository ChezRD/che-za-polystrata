# Audit: merged, deleted, and what not to revisit

2026-11-13/14. Scope: **the engine only** — `src/`, `include/`, `cmake/`, `CMakeLists.txt`,
`third_party/strata-ggml`, `metal/`/`sycl/` where a change reaches them. `serve/`, `setup.py`, `tools/`, `docs/`
and `bench/results/` were not merged from either upstream. Derivations are in the commit messages; this keeps the
decisions.

## 1. The MMQ row-id over-read — fixed in three places upstream, once here

llama.cpp's MMQ reads its row-id table in whole J-column tiles, `J` up to 128 (`mmq.cuh:1037`, `:1131`, and the
stream-k fixup `:1353`), so an exact-sized table is read up to `roundup(max_rows,J) + J` entries past its end. The
value read becomes an index into `dst` (`mmq.cuh:1374`): **an arbitrary offset write** — silent wrong numbers, or
an illegal address when the table ends a `cudaMalloc` chunk.

| site | allocation | worst overrun | by |
|---|---|---:|---|
| `src/qwen35/q36_engine.cu:1396` | `ne = T*Kp` ints | 320 B | the sync |
| `src/prefill/gemm.cu:1301` | `kRows = 1024` ints | 384 B | the sync |
| `src/prefill/prefill.cpp:2344` | `B*hc` ints | ~508 B | the sync |
| `src/poly/engine.cpp:3186` | `rr = n*n_used_` ints | 512 B | **here** (`iota_`) |

**`ac76e3de5` did not expose this.** That dense gate passes **null** for bounds and ids (`engine.cpp:3621`), and
with `ids_dst == nullptr` the kernels never load a table. What *was* exposed is the poly **MoE** path's `iota_`,
exact-sized at `engine.cpp:3186` and passed at `:4632`, `:4674`, `:4725` — unfixable from `strata/main`, because
`src/poly/` does not exist upstream. It now takes `mmq_ids_table_rows(rr, rr)` with the identity written over the
whole table (an unwritten margin entry is an arbitrary row index — the failure the margin exists to prevent).
`max_rows = rr` at all three sites: each passes `most` / `r.most` / `top`, all `<= rr`. The **Qwen3.6-MoE** path was
exposed with no flag at all (`use_mmq` is on unless `STRATA_Q36_NO_MMQ` is set, `q36_engine.cu:1375`).

**Guard:** `mmq_ids_margin_test` pins the helper and **can fail** (reverted: 95,765 failures). `poly_mmq_ids_test`
(new, host-only) pins `src/poly/`'s table: exact size gives **49,764 failures**, an unwritten margin **9,859**. **No
test runs the Qwen3.6 engine's MMQ path** — the one gap; it needs a GPU fixture.

## 2. The strata/main sync: three of eight commits

`strata/main` was 8 ahead of snapshot `c1ff07f1a`. Taken: `d7f7d9677` (qwen35), `14dd746f6` (dense MMQ), `697b6d67a` (DeepSeek MTP). Skipped as not engine: `13eedf552` (the #4677 stop bound is serve behaviour that reaches
`src/program/generate.cpp`; our tree keeps the HIP-only guards at `:18219`, `:18782`, `:19326`), `5f0ae5b55` and
`b90d38aa8`/`a95046690`/`3b3c41e40` (serve tests, `vision_reader.py`, `serve/`). The borderline pair
`serve_stop_bound.hpp` + its test was **not imported**: the test compiles here (`cmake/backends/cpu.cmake:94-97`)
but tests the serve header, and importing it without the `generate.cpp` hunks registers a test for a function
nothing calls.

Snapshot branch `upstream-engine-2026-11-13` = one commit, three files, +24/−7.

```
previous merge base:  b4d53e636  upstream: engine snapshot from Strata main c1ff07f1a
this sync's snapshot: 14dc921e6  upstream: engine snapshot from Strata main 13eedf552
```
The next sync uses `b4d53e636` until this reaches `main`; after that, `14dc921e6`.

## 3. origin/main: 26 commits, zero needed

`83a7a3327` is not an ancestor of `main`, but `77ff78930` (which is) carried its content rewritten.

| group | commits | verdict |
|---|---|---|
| engine | `5336fb437`, `dbf803bea`, `6578f7648`, `f0d2a6ca3`, `f0aab9975` | **already in `main`.** `5336fb437` present **and a superset** (`main` also has type 19); `dbf803bea`'s files are the **same blobs**; `f0aab9975` is a **proven no-op** (`git merge-tree` returns `main`'s tree); every `f0d2a6ca3` symbol exists (`batch_mtp` at `engine.hpp:138`, `draft_many`, `steps_guesses`, `rows_snaps`). Merging any of them would re-introduce a narrower version. |
| installer, serve, docs, benches, community | 21 commits, incl. 47 `bench/results/` files | leave — `setup.py`, `serve/`, `docs/` and `bench/results/` are ours or out of scope; other people's hardware belongs to the project that ran it. |

**NVFP4 on sm_89 is a non-issue.** `5336fb437` is the *portable* half — type 40 through `iq_mmvq`, a software
E2M1/UE4M3 decode times Q8_1 rows, no FP4 tensor core (`native_mmvq.cu:3927`) — and we have it. The FP4×FP4 half
is gated to cc 12.0 (`src/poly/mmq.cu:41-47`). An NVFP4 GGUF on Ada does not fail and does not fall back to BF16.

## 4. Branches and worktrees

Nothing deleted. `git branch -d` refuses an unmerged branch, so the merged list is safe.

| branch / worktree | ahead | verdict |
|---|---:|---|
| 24 fully merged: `analysis/*` (8), `bench/quant-micro-0`, `feat/{k6v4,qwen38-dense}`, `fix/mmq-dense`, `integrate/dense-and-fixes`, `perf/qsa-decode-control-flow`, `probe/kv-stream`, `research/ada-ecosystem`, `sync/strata-main-2026-11-11`, `test/tokens-file`, `upstream-engine-0.2.{12,13}`, `upstream-engine-0.2.13-main`, `main` | 0 | **clutter; delete.** `upstream-engine-0.2.13-main` is `b4d53e636` (§2). |
| `research/poly-tuning` | 1 | **merged** (`da63d03ac`), plus annotations (`e186166d7`). |
| `fix/k6v4-prompt` | 2 | **separate job, with a GPU run.** `fattn_k6v4` is nowhere in `main`; its claimed 5.05× is unverified here. |
| `feat/lsa-phase1-wt` | 23 | **do not merge** — LSA dropped (§6); harnesses were in `main` and are deleted, the rest is parked. |
| `fix/state-per-window` | 16 | **do not merge** — LeapQuant (§6). |
| `test/ornith-main` | 3 | **stale; `-D`.** `--tokens-file` is in `main` and strictly better (`main.cpp:1320`, `:1458`: names the file in every error, refuses a NUL byte). Its only novel file builds in no target. |
| `bench/instr-paths` | 1 | **doc taken** (`a236d0d34`), harness left on the branch (its §8 says where). |
| `study/imma-paths` | 7 | **doc is the deliverable**, restored in the primary checkout (§5). Harness has one open question (`k4v4`). |
| `wt-ctl` | 0 + uncommitted | **discard the edit.** A stale copy of `qsa_decode_attn.cu`; applying it would revert the Daily arm. |
| `feat/k6v4-strata`, `bench/slot-economics`, `wt-s6` | 1 each | docs only, leave. |

**k6v4 and the group-4/6/12 decode kernel survive**: `src/poly/fattn.cu:830`/`:876` (`fdec_k6v4`) and `:277`
(`g == 4 || g == 6 || g == 8 || g == 12 || g == 16`). Origin's `fattn.cu` is 357 lines shorter and has neither.

## 5. What was deleted, and what was corrected

**Deleted** — each `add_executable`-only (no `add_test`), the fork's own, its question already answered in a doc
that stays: `src/kernels/lsa_scan_bench.cu`, `lsa_ward_bench.cu`, `gdn_state_bench.cu`, their three
`CMakeLists.txt` blocks, and `tools/polystrata_names.py` with its test.
**The name script went rather than being narrowed:** on `docs/ADA-ECOSYSTEM.md` it rewrote 19 occurrences, every
one naming someone else — `Hardin22/Strata-DualGPU` and three other forks into repositories that do not exist,
"not Strata forks" into "not PolyStrata forks", and two benchmark rows' provenance restamped as ours.
`git log --all -S'PolyStrata-DualGPU'` shows only the reverted attempt: no history is damaged.

**Corrected** — one line each, with the number that replaces the old one:

- `src/kernels/cuda/qsa_decode_attn.cu:887`: "19% faster, 14.31 vs 17.70 ns" → **1.95×, 9.10/9.11 vs 17.91/18.45**
  (2,051 cells, group 12, parity 7.82e-08).
- `docs/PREFILL-GAP.md`: the 40,960 A/B marked **unverified — KV format and batch not recorded**, number left as
  published (a loaded-box reproduction gave 1,717/1,712/1,722, not evidence against it).
- `docs/REAP.md`: `--prefill auto` ran at **chunk 8192** — `--resident-budget-gib` sets `resident_cpu_experts`
  (`generate.cpp:3793-3794`), the #2090 guard steps the ceiling down (`:9907-9935`). `STRATA_NV_CHUNK=1` forces it.
  **16K and 24K are not measured.** One attempt at `--resident-budget-gib 34 --vram-reserve-mib 2048` never reached
  the prefill: the cache left 228 MiB free against the 3,714 MiB a 16K chunk's buffers want, so it exited with
  "device buffers ... do not fit". That is a budget that has no room for a chunk, not a property of the card — the
  flag pair has to change before the sizes mean anything.
- `docs/QUANT_MICROBENCH.md` rows 3, 4, 8, W1: the binary is gone, the measurement stays. `AGENTS.md`,
  `docs/LOCAL_CHANGES.md`, `docs/SYNC-2026-11-11.md`: the naming rule.
- **`docs/KV-INT8-PATHS.md` restored.** Untracked in the primary checkout and **332 lines short** — one append-only
  hunk (`1487a1488,1819`, **0 lines removed**) — so it stopped at §25 and lacked **§33's verdict**: the INT8-QK
  line measures 0.90–1.06× against a 1.2–1.9× hypothesis and **closes**. The complete 1819-line version is in
  place, still untracked; `git add` makes it survive a checkout.

## 6. What not to revisit

| idea | measured | verdict | harness (deleted) | measurement (kept) |
|---|---|---|---|---|
| **k6v4** in **`strata`** | decode 43.6 vs 43.5 (a tie), prefill +1.4% | **closed:** **26 slots**, ~+1.7%, inside ±1.2 noise — `strata` is *streamed* (`--kv-resident 32768`) | — | `docs/QUANT_MICROBENCH.md` |
| **k6v4** in **`strata-poly`** | at 131,072 cells: k8v4 **208.0 MiB** vs k6v4 **176.0 MiB** | **kept, wanted:** **32 MiB saved**, the ceiling at `--prompt-batch 128` rises **102,400 → 122,880 cells** — `poly` is *fully resident* (`grep -rn kv_resident src/poly/` → **0**) | — | the code, and `21caa097e` |
| **LeapQuant** per-window state | fp16 **0.74×** for half the bytes, worst error 5.28e-04; int8 **50× worse at every horizon**, flat | **closed:** **36 slots** | `gdn_state_bench.cu` | `QUANT_MICROBENCH.md` rows 3-4 (**3** LeapQuant mentions survive) |
| **LSA** | **1.008×** (27B) / **1.028×** (Ornith), post-MMQ | **closed:** attention is **0.8%** of Ornith's prefill | `lsa_scan_bench.cu`, `lsa_ward_bench.cu` | `docs/LSA-DENSE.md` (HyperQwen's maps, + a "dropped" line), `QUANT_MICROBENCH.md` rows 8/W1 |

**Why:** on a 16 GB Ada card at 288 GB/s what binds is **bytes moved and PCIe**, not bytes saved from a cache a few
percent of the model. A 1.02× win inside a ±3% band is not a win — **unless the KV is fully resident, and then the
byte saved is a byte that was going to be read.**

**Three cautions so this is not misread:**

- **LeapQuant's paper is not refuted.** `QUANT_MICROBENCH.md:272` says our bench "did not implement the paper it
  was written to test". The compensator arms carried **zero** compensators (`:189`); §2.1a then fit them for real
  and measured **−24% error at one step but only −2% at 128** (`:214`) — **the compensation works and evaporates
  over context.** A paper's central claim can be true and still not help us.
- **k6v4's first arithmetic had a wrong premise, not wrong maths** — full KV residency (206 slots) against
  streamed KV (26). Same format, different engine, opposite answer.
- **Each of the three was first measured with a number that was wrong in its operating point:** k6v4's slots (above),
  LeapQuant's int8-as-the-whole-format, LSA's comment claiming 1.24× where the measurement said 1.95×. Check the
  operating point, and check which engine, before believing a number.

## 7. Junk on disk — commands only, nothing here was run

```sh
# ~4.6 G of dead build trees (regenerable).  KEEP build-poly-local (3.3 G), engine-poly, engine -- the local
# strata-*.json configs name them as their exe.  The llama.cpp checkouts are only for -DSTRATA_GGML_DIR builds.
rm -rf build-sync build-parity build-poly build-sync-checkout build \
       build-audio-cuda build-audio-cuda-local build-vision build-vision-check \
       third_party/llama.cpp third_party/llama.cpp-69f201a

# untracked pre-edit copies of tracked files
rm strata-coder-iq1_m.json.bak-20261108-2010 strata-iq2_xs.json.bak-20261108-2010 \
   strata-swift-iq3_xxs.json.bak-20261108-1955 tools/gguf_reader.py.bak-20261109-1338

git add docs/KV-INT8-PATHS.md   # the complete doc is untracked; this makes it survive a checkout
```

`strata-*.json` / `*.log` / `run-*.sh` are live local configs — keep. 10 of the json files point at
`/home/chez/llm/PolyStrata/engine-poly`, which does not exist, so those entries are already stale.

## 8. Do not touch

`docs/ADA-ECOSYSTEM.md` and every measured number in `docs/` (a refuted row with its retraction is evidence; a
deleted row is a hole). `docs/{LSA-DENSE,QUANT_MICROBENCH,POLY-TUNING,KV-INT8-PATHS}.md`. `src/poly/gigachat.*`
and `tools/gigachat/` — a feature, wanted. `src/poly/fattn.cu`'s `fdec_k6v4` and groups 4/6/12.
`upstream-engine-2026-11-13`. Any default path: every deletion here removed a target nothing tested or called.
