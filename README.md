# Che'za-PolyStrata

Personal research fork combining [Strata](https://github.com/Niko1221/Strata) and [PolyStrata](https://github.com/simplexai-labs/PolyStrata). Tuned for one **RTX 4060 Ti 16 GB** (AD106, sm_89, 34 SM, 288 GB/s, 32 MiB L2, 99 KB smem, ~12.5 GiB usable VRAM).

## Architecture

The project builds two binaries:

| Binary | Source | Attention & KV | Models |
|---|---|---|---|
| `strata` | `src/`, `include/`, `cmake/`, `metal/`, `sycl/` (synced from Strata) | QSA (indexer, selector, page table), KV streaming | Qwen3.8-Flash-Next, Swift 1.5 / Unsloth variants; Qwen3.6 via `src/qwen35` |
| `strata-poly` | `src/poly/` (synced from PolyStrata) | Dense `fattn` / `fdec` / `attend`, resident KV | Ornith 1.5, Qwen3.6-35B-A3B, Flint, GLM 5.3 Flash, DeepSeek V4 Flash, Gemma 4, Tiel Coder, MiniCPM-V, Clover, Kolibri 1, LFM2.5 |

Performance numbers and settings from one engine do not transfer to the other.

## Measured on this card

"Reads" is prefill, "writes" is decode (tok/s); each row indicates the context length tested.

| Model | Quant | Context | Reads | Writes |
|---|---|---:|---:|---:|
| Qwen3.8-Flash-Next | native pack | 262,144 | 1,249 | **40.19** |
| Flint 40B-A2.5B abliterated | Mix-12G | 131,072 | 1,945 | 46.4 |
| Flint 40B-A2.5B abliterated | Mix-12G | 262,144 | 1,268 | 30.7 |
| Ornith-1.5-9B, dense | Q4_K_M | 40,960 | 1,942.2 | — |
| Ornith-1.5-9B, dense | Q4_K_M | 131,072 | 986.4 | — |

The two Ornith rows were measured with SM clock pinned at 2,820 MHz (`docs/POLY-TUNING.md`).

## Build

```sh
cmake -S . -B build-poly-local -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_BUILD_TESTS=ON
cmake --build build-poly-local -j8
ctest --test-dir build-poly-local --output-on-failure -j1
```

- Leave `STRATA_GGML_DIR` empty to compile vendored `third_party/strata-ggml`.
- Both `strata` and `strata-poly` are built in one pass.

## Run

### CLI

```sh
build-poly-local/strata-poly \
  --model <path-to-file.gguf> \
  --max-context 131072 \
  --kv k8v4 \
  --threads 8
```

### Server

```sh
./start.sh strata-<model>.json
```

Default address: `http://127.0.0.1:8080/v1` (OpenAI and Anthropic compatible). `strata-<model>.json` specifies the engine, model, and port.

## Documentation

- **Tuning & Optimization:** [docs/POLY-TUNING.md](docs/POLY-TUNING.md), [docs/DETAILS.md](docs/DETAILS.md), [docs/BACKENDS.md](docs/BACKENDS.md)
- **Quantization & Microbenchmarks:** [docs/QUANT_MICROBENCH.md](docs/QUANT_MICROBENCH.md), [docs/Q8_KV.md](docs/Q8_KV.md)
- **Model Logs:** [docs/FLINT.md](docs/FLINT.md), [docs/ORNITH.md](docs/ORNITH.md), [docs/REAP.md](docs/REAP.md), [docs/QWEN36.md](docs/QWEN36.md)
- **Engine Sync & Local Changes:** [docs/SYNC-2026-11-11.md](docs/SYNC-2026-11-11.md), [docs/LOCAL_CHANGES.md](docs/LOCAL_CHANGES.md)
- **Troubleshooting:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), [SECURITY.md](SECURITY.md)

## License

MIT License ([LICENSE](LICENSE)). Original copyright Niko1221 (Strata) and VecSzn (PolyStrata). Model licenses: `docs/HOW_IT_WORKS.md#license`.
