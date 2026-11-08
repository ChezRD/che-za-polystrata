# Implementation plan

Ordered by dependency. Each item names its file ownership so parallel agents do
not collide.

## Phase 0 — repository and safety (1–2 days)

| # | Task | Files | Acceptance |
|---|---|---|---|
| 0.1 | Linux parent-death containment: `prctl(PR_SET_PDEATHSIG, SIGKILL)`, `start_new_session=True`, `killpg` on shutdown | `serve/server.py` (`popen`, `StrataEngine`, `close`), `serve/winjob.py` | `kill -9` of the server leaves no `strata-poly`; VRAM returns ≤2 s |
| 0.2 | Free-VRAM preflight before eviction in `_switch_to`; estimate `dense + kv + cache_floor + reserve` | `serve/server.py` (`_switch_to`, `ensure_loaded`, `free_vram_mib`) | refusing a switch leaves the old model serving |
| 0.3 | Wait for release: `waitpid` + NVML delta, short TERM/KILL timeouts, confirm per-PID | `serve/server.py` (`unload`, `close`) | no load starts into unconfirmed memory; 100 cycles, VRAM ±128 MiB |
| 0.4 | Per-model ready timeout, cooldown, `last_good`, 503 + `Retry-After` | `serve/server.py`, `serve/model_gate.py` | failed load → 503 retryable; old model restored in background; no retry storm |

**Do not** change the `switch_on_request` architecture. Fix it in place.

## Phase 1 — decoder-only proof (5–6 days)

| # | Task | Files | Acceptance |
|---|---|---|---|
| 1.1 | Download `decoder/` (21.35 GB) from `ai-sage/GigaChat3.1-Audio-10B-A1.8B`; convert to F16 GGUF with `convert_hf_to_gguf.py`; quantize Q4_K_M and Q3_K_XL | `docs/research/gigachat-audio/05-quantization-recipe.md` | GGUF loads in llama.cpp; `general.architecture = deepseek2` |
| 1.2 | Backend registration: `deepseek2` + `gigachat_audio` → `GigaChatAudio`; allow `--kv q8_0` | `src/poly/engine.cpp` (`make`, kv gate), `CMakeLists.txt`, `serve/models.py` | unknown-arch error gone; model listed |
| 1.3 | Backend skeleton: geometry, tensor loading, MLA caches, router top-4, shared expert, layer head/tail, output | new `src/poly/gigachat.{hpp,cpp}` | one token runs; logits plausible |
| 1.4 | `mla_attend` + key-copy (absorbed form: K = latent 512 + rope 64, V = latent 512) | `include/strata/poly/kernels.hpp`, `src/poly/kernels_mla.cu` | unit test vs numpy reference |
| 1.5 | Graph-safe attention (`dsv4_rope` with `at`, `store_f16_at`, `mla_attend(..., room)`); `L.fixed = true` | `src/poly/gigachat.cpp` | `--dump-layers` identical with/without graphs |
| 1.6 | Parity vs llama.cpp `deepseek2` on the same GGUF | `llama-completion` against `strata-poly`, recorded in `05-quantization-recipe.md` | top-1 100% over 200 tokens; max Δlogit < 0.05 |

Parity details that must be right: `1/sqrt(192)` × YaRN `mscale²` = 0.1447;
interleaved RoPE; K/V share the latent so KV quant type is the same for both;
`k_b` stored as `[128,512,32]` with latent contiguous; `v_b` as `[512,192,32]`.

## Phase 2 — audio encoder (4–5 days)

| # | Task | Files | Acceptance |
|---|---|---|---|
| 2.1 | Encoder GGUF arch `gigachat_audio_enc`: 24 Conformer layers, q8_0 matrices, F32 norms | `tools/audio/` | loads, reports 585M params |
| 2.2 | Mel kernel: 16 kHz mono, 64 HTK mels, 400/160, Hann, power 2, log clamp, center=True | `tools/audio/` | max|Δ| < 1e-4 vs torchaudio |
| 2.3 | Conformer: conv subsampling ×4, macaron FFN, depthwise conv k=5 + LayerNorm (eps 1e-5), 200-frame chunk attention, **RoPE before Q/K projection**, base 5000 | `tools/audio/` | layer cosine > 0.9999 |
| 2.4 | Stacking ×4 + projector 4096→1536; emit `SVE1` rows, `nx = n`, `ny = 1` | `tools/audio/` | 30 s wav → 188 rows |
| 2.5 | 200-frame streaming with 6-frame mel + 2-frame/layer conv halo; chunk by encoder frame index | `tools/audio/` | streaming vs offline < 1e-5 |
| 2.6 | `strata-audio` process: `READY`, `ENC`, `QUIT`, elastic VRAM, CPU default | `tools/audio/`, `engine-poly/` | restart on silence; no VRAM leak |

First verify the mel discrepancy: config says 320/center=False, code says
400/center=True. Build both, compare against `encode_audio`, keep the code path.

## Phase 3 — server integration (3–4 days)

| # | Task | Files | Acceptance |
|---|---|---|---|
| 3.1 | `input_audio` parsing (base64/path), decode, downmix, resample, chunk | `serve/frontend.py`, new `serve/audio.py` | 400 with a clear message on bad input |
| 3.2 | `GENA` head in the engine, reuse `take_pictures` / `SVE1` | `src/poly/main.cpp` | `GENA` accepted; rows at pad runs |
| 3.3 | Prompt: `<|file|>…[audio_token]…<|/file|>`, 1,019-token template, timestamps every 60 s + duration | `serve/server.py`, `serve/audio.py` | token count matches reference |
| 3.4 | Audio model config entry, `switch_on_request`, vision/audio cleanup | `strata-gigachat-audio.json`, `serve/server.py` | `/v1/models` shows it; switch in/out works |
| 3.5 | `response_format=verbose_json` segments; `/v1/audio/transcriptions` | `serve/transcribe.py` | timestamps in response |

## Phase 4 — residency and switching (1–2 weeks)

| # | Task | Files | Acceptance |
|---|---|---|---|
| 4.1 | `home_model`; shrink-before-admit via `POST /v1/vram`; grow debounce | `serve/server.py` | Tiel→Audio→Tiel p95 ≤8 s warm |
| 4.2 | Page-cache warming (`madvise(WILLNEED)`), persistent worker with context/graphs | `serve/server.py` | 15 GB swap 3–6 s warm |
| 4.3 | one `strata-poly` process loads the next GGUF (`SWITCH`); parked chats stay in that process | `src/poly/main.cpp`, `src/poly/conversations.hpp`, `serve/server.py` | the same model comes back with its parked chats; no cache file |

## Phase 5 — optional accelerators

- q4_0 KV with Walsh–Hadamard rotation (port from `include/strata/kernels/kv_q4.hpp`).
- Heuristic block-sparse MLA for prefill (blocks 64, sink + local + top-k by
  64-d summary); gate on temporal mIoU.
- MPS static SM partitions + cgroup `dmem` hard caps.
- Resident small ASR for true 256K parallelism.

## Quantization workstream

Goal: smallest viable decoder while preserving audio quality. Candidate
families: Unsloth Dynamic (UD) Q3_K_XL / Q4_K_M, IQ3_XXS/IQ3_XS with imatrix,
Q3_K_M/L. Rules:

- experts q3/q4; attention `q6_K`; `token_embd`/`output` `q6_K`; router,
  biases, norms **F32**; `attn_k_b`/`attn_v_b` `q6_K`.
- calibration imatrix from audio transcripts + Russian/English text, not only
  wikitext.
- acceptance: WER Δ ≤ 0.5 pp, temporal mIoU Δ ≤ 1–2 pp vs Q4_K_M; if Q3 fails,
  Q4_K_M is the floor.

## File ownership (parallel agents)

| Workstream | Owns |
|---|---|
| Lifecycle | `serve/server.py`, `serve/model_gate.py`, `serve/winjob.py`, lifecycle tests |
| Backend | `src/poly/gigachat.*`, `include/strata/poly/kernels.hpp`, `src/poly/kernels_mla.cu`, `src/poly/engine.cpp` (make/kv gate), `CMakeLists.txt` |
| Audio | `tools/audio/`, `serve/audio.py`, `serve/frontend.py`, `src/poly/main.cpp` (`GENA`) |
| Quantization | `docs/research/gigachat-audio/05-quantization-recipe.md` only. No quantizer in the tree. |
| Artifacts | `docs/research/gigachat-audio/` |
