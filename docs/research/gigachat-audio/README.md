# GigaChat Audio in PolyStrata — research and implementation

Branch: `feature/gigachat-audio-mla`
Base: `a82999fd` (`main` after pull on 2026-11-05)
Research snapshot: `2d6a8f9` (the tree the subagents read; code files below are
unchanged, `serve/server.py` moved by +61/−13 lines)

## Documents

| File | What it covers |
|---|---|
| [`01-mla-and-cuda.md`](01-mla-and-cuda.md) | MLA math (absorption, RoPE split, YaRN scale), sparse-attention options, CUDA kernels on sm_89, roofline, GGUF types, parity plan, effort |
| [`02-polystrata-integration.md`](02-polystrata-integration.md) | Engine architecture, backend hooks, GGUF loading, expert cache, KV, graphs, `GENI` path, ordered patch plan, memory budgets |
| [`decoder-notes.md`](decoder-notes.md) | Tensors the decoder backend reads, the MLA cache row, and the limits it still refuses |
| [`03-audio-pipeline.md`](03-audio-pipeline.md) | Mel front end, Conformer graph, adapter, 6.25 tok/s arithmetic, prompt/timestamp format, streaming, quantization, test plan, risks |
| [`PLAN.md`](PLAN.md) | Ordered implementation plan with acceptance criteria and file ownership |

## Decisions

1. **No vLLM, no BF16, no FP8.** Weights are q3/q4; KV is q8_0 (q4_0 only as an
   experiment); the encoder is q8_0.
2. **Decoder first.** GigaChat's decoder is a stock `DeepseekV3ForCausalLM`
   (`q_lora_rank = null`), so it maps to `deepseek2` in llama.cpp. Prove the
   decoder alone before any audio work.
3. **Absorbed MLA, not materialized.** Cache the 512-d latent + 64-d RoPE key;
   never cache 32 per-head K/V. At 45k tokens this is 0.67 GiB q8 vs 22.45 GiB
   materialized BF16.
4. **Dense MLA first.** DSA/NSA/MoBA cannot be enabled without training. Optional
   heuristic block-sparse is a Phase-3 prefill accelerator, not a foundation.
5. **Audio is another model in the same server.** The user's chosen integration
   point is the existing `--switch-on-request` in `serve/server.py`; no separate
   router or supervisor. `_switch_to` must be made safe (preflight, wait for
   VRAM, atomic rollback, timeouts, cooldown, last-good).
6. **One model per engine process.** `Engine::open` returns one engine; `--batch`
   is N sequences of the same model. Concurrent Tiel + audio is only possible by
   cutting Tiel's context/expert cache or by running a small ASR instead.
7. **CPU `strata-audio`.** The Conformer is 585M params / 0.63 GB q8; running it
   on CPU preserves VRAM for 256K KV and avoids a second CUDA context while the
   decoder is co-resident.

## Memory reality (one RTX 4060 Ti 16 GB)

| Item | GB |
|---|---:|
| Decoder q4_K_M | 6.47 |
| Encoder q8_0 | 0.63 |
| KV q8_0, 120 min / 45,001 tokens | 0.70 |
| Runtime + graphs + scratch | 0.7–1.2 |
| **Audio total** | **8.8–9.8** |
| Tiel 256K fixed (dense + ctx + KV) | ~6.5 GiB |
| Desktop baseline | ~3.6 GiB |

- Audio **alone** fits comfortably.
- Tiel 256K + audio **does not fit** (≈19 GB).
- Tiel 64K + q3 audio leaves ~2.6 GiB of expert cache → Tiel ~30–40 tok/s.
- True parallelism at 256K requires a small resident ASR (0.5–1.5 GB), not the
  10B audio LLM.

## Status

| Workstream | State |
|---|---|
| Pull + branch | done |
| Research artifacts | done |
| Quantization recipe (Q4_K, Q3_K, IQ3_S, IQ3_XXS, imatrix) | done |
| Lifecycle P0 in `switch_on_request` | done |
| `gigachat` backend + `mla_attend` | done |
| Decoder GGUF conversion + llama.cpp parity | done for IQ3_S / IQ3_XXS |
| `strata-audio` Conformer (CPU, q8/mixed) | done |
| `GENA` + `input_audio` front end | pending |
| 200-frame streaming with halos | pending |
| KV checkpoint (`DUMPKV`/`LOADKV`) | pending |

### Measured on 2026-11-05

* `IQ3_S` loads cold in **4.3 s**, warm in **1.0 s**; 1600/1600 experts in
  VRAM (3.8 GiB).
* Raw greedy continuation matches llama.cpp `llama-completion` on three
  prompts for `IQ3_S` and uniform `IQ3_XXS`.
* CUDA graphs capture and replay for the MLA layer: 96.8 tok/s vs 83.0 eager
  on a 6-token prompt.
* Tiel → GigaChat Audio switch through `switch_on_request`: **6.2 s**,
  including unload, preflight, load and a 16-token answer at **128.8 tok/s**.
* GigaChat Audio → Tiel: **4.6 s**, VRAM returned, Tiel reloaded at 262K.
* Audio decoder alone at 49K context: 10,269 MiB used with the ~3.6 GiB
  desktop, so 5.7 GiB stays free.
* `strata-audio` (CPU, `mixed` 807 MB): 10 s clip → **63 rows**, rows
  cosine **0.99993** vs PyTorch, RTF ≈ 0.2; 120 s → 750 rows in 22 s,
  RSS 1.1 GiB. Mel is within 1e-6 of a float64 run of the same front end.
