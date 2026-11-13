# AGENTS.md

Che'za-PolyStrata is a research fork of PolyStrata and Strata, tuned for one RTX 4060 Ti 16 GB (sm_89).

## Architecture

The project builds two binaries:

- `src/` -> `strata` (upstream: Strata). QSA attention, KV streaming. Primary model: Qwen3.8-Flash-Next.
- `src/poly/` -> `strata-poly` (upstream: PolyStrata). Dense attention, resident KV. Models: Flint, Ornith, Qwen3.6, GLM 5.3 Flash, DeepSeek V4 Flash.

Do not transfer benchmark numbers between these two engines.

## Build and Test

```sh
cmake -S . -B build-poly-local -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_BUILD_TESTS=ON
cmake --build build-poly-local -j8
ctest --test-dir build-poly-local --output-on-failure -j1
```

- Leave `STRATA_GGML_DIR` empty (uses vendored `third_party/strata-ggml`).
- Run `ctest` with `-j1` to prevent VRAM overcommit.
- Local machine builds only CUDA (sm_89). Mark any Metal, HIP, or SYCL changes as untested.

## GPU Rules

Any timing or benchmark run on GPU must follow these rules:

1. **Lock clock:** `sudo -n nvidia-smi -lgc 2820` before the run, `nvidia-smi -rgc` after.
2. **Lock GPU:** Wrap executions in `flock -w 60 /tmp/opencode/gpu.lock timeout <sec> <cmd>`.
3. **No background runs:** Run in foreground only. Never benchmark while compiling.
4. **Report conditions:** State model, quant, KV type, context length, and verify parity/correctness.

## Working Rules

- **Local patches in upstream code:** Wrap edits in `// POLYSTRATA LOCAL: <id> begin` and `// POLYSTRATA LOCAL: <id> end`.
- **Naming:** Keep original upstream names in engine sources and docs. Do not run bulk search-and-replace on "Strata" or "PolyStrata".
- **Local files:** Files like `strata-*.json`, `strata-*.log`, `run-*.sh`, and `build*/` at the root are gitignored local state. Do not delete them and do not commit them.
- **Defaults:** Change engine defaults only if verified faster with pinned clocks on this RTX 4060 Ti.
