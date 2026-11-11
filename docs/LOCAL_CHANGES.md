# Local changes and upstream sync

The engine is taken from upstream Strata through `upstream-engine-<tag>` snapshot commits;
`main` merges them. Local work stays in new files and small hooks. This page lists the
fork-local change set and the sync checklist.

## Flint

- IQ1_S (ggml type 19) in native MMVQ and Slots: `src/kernels/cuda/iq_kernels.cu` (dot,
  dequant, `Fmt<19>`, `STRATA_MMVQ_FMTS`, `is_iq`, `iq_row_bytes`), `src/kernels/cuda/native_mmvq.cu`
  (`native_mmvq_supported`, weight bytes, dispatch, `slots_kind`), the IQ parity tests and
  `tools/iq_fixture.py`. Only `Flint-abliterated-Mix-12G` uses type 19 (layers 2-7 gate/up);
  the 24G, UD-IQ3_XXS and UD-Q3_K_XL files have none. 20K decode 79.8 -> 92.7 tok/s; docs in
  `docs/FLINT.md`.
- Prefill window ring dequantized once to fp16 and read through `attn_flash128`; the tail
  `axpy` folded into the next RMS norm; k8v4 and `fdec` paths. `stage_out` and `stage_in` are
  plain `cudaMemcpy`: the pinned conversation cache was reverted after its restore gate
  measured 27.6 against 36.5 tok/s live. Files: `src/poly/flint.cpp/.hpp`,
  `src/poly/fattn.cu/.hpp`, `src/poly/kernels.cu`, `src/poly/engine.cpp/.hpp`,
  `src/poly/main.cpp`, `bench/micro/attn128_bench.cu`, `bench/micro/attn128_64.cu`.
  Docs: `docs/FLINT.md`, `docs/DETAILS.md`, `docs/Q8_KV.md`.

## Qwen3.8-Flash-Next REAP-320

- IQ4_NL gate/up: `X(20)` in `STRATA_GU_FMTS` and the `gu_ipb`/`gu_rows` fix in
  `src/kernels/cuda/iq_kernels.cu`. Without them layer 1 is refused; with only `X(20)` and the
  old grid, the up half is never written. `native_grouped_parity` now covers `(20,20)` and
  `(20,8)` at 2560x640; 111/111 bitwise, 0 failures.
- `tools/embd_extract.py` copies `token_embd.weight` from the base Qwen IQ2_XS shard (IQ4_XS)
  into a one-tensor `strata-embd` GGUF for `--embd-gguf`. The REAP file's own Q2_K embedding is
  not read by the native path.
- `docs/REAP.md`; `strata-reap320.json` is a local server config and is not tracked by
  design, like the other run configs (`/strata-*.json` is ignored). The flags in `docs/REAP.md`
  are the record.
- Pack and profile are data under `packs/reap320`; MTP uses the existing `mtp/rt`.
- 256K with the resident-budget mode: 50.19 tok/s, MTP 94.7%, 31.42 GiB page-locked, 2,522
  slots. Plain mmap without the resident budget gives 25.92 tok/s and is not the model's speed.

## Upstream

- IQ1_S and IQ4_NL gate/up are generic type additions and candidates for upstream PRs with
  their parity tests. The `gu_rows` fix belongs with the IQ4_NL PR.
- `tools/embd_extract.py` is a tool candidate; the config and docs are fork-local.
- The prefill and tail changes are fork-local unless measured and accepted upstream.
- If an upstream file must change, keep the edit to one marked block
  `// POLYSTRATA LOCAL: <id> begin/end` and check it after each sync.
- After a sync: `python tools/polystrata_names.py`, then build, `ctest -R 'local_hooks|iq_|poly_'`,
  `native_grouped_parity`, and the 12G/REAP smoke tests.

## Verification, 2026-11-11

- `native_grouped_parity`: 111 cases, 111 bitwise equal, 0 failures.
- `local_hooks_test`: ok.
- `ctest -R 'iq_|poly_'`: 12/12 (3 HIP tests skipped).
- `git diff --check`: clean.
