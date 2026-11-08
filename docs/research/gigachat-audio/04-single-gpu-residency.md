# Single-GPU residency, switching, and the `switch_on_request` path

Hardware: one RTX 4060 Ti 16 GB (Ada sm_89), 62 GB RAM, Linux. Measured
2026-11-05 on the running machine; `[M]` = measured here, `[L]` = measured in
PolyStrata logs/docs, `[E]` = estimated.

## 0. Decision recorded after this research

The user's chosen integration point is **PolyStrata's existing
`--switch-on-request`**, not a separate router or supervisor. Consequences:

- GigaChat Audio becomes another model in the same `strata-*.json` list and the
  same `serve/server.py` on `127.0.0.1:8081`.
- Audio load/unload is `_switch_to` / `Service.use_model`, with the same FIFO and
  `ModelGate`.
- Safety work goes into `_switch_to`, `start_model`, `unload`, `Vision`/`Audio`,
  and `switch_on_request` — no sidecar service.
- The one-process-per-model rule stays: today `Engine::open` returns exactly one
  engine, and `--batch N` means N sequences of the same model, not N models.

## 1. CUDA memory reality

A CUDA context is per-process; `cudaMalloc` reserves virtual space in that
context and backs it with the single physical VRAM pool. Two processes:

- share the physical pool first-come-first-served, with no reservation and no
  fairness;
- cannot see each other's pointers; sharing needs an explicit IPC/VMM handle.

**Measured:** with 502 MiB free, `cuMemAlloc(2 GiB)` returned
`CUDA_ERROR_OUT_OF_MEMORY` in 0.3 ms; the resident 11.6 GB engine was untouched.
There is no Linux sysmem fallback for `cudaMalloc`.

| Mechanism | Cross-process | Shares pages | Useful here |
|---|---|---|---|
| `cudaMalloc` in two processes | coexists only | no | only if both fit |
| `cudaIpcGetMemHandle` | yes (Linux) | **yes** | no — different architectures |
| VMM `cuMemCreate` → POSIX fd → `cuMemImportFromShareableHandle` | yes | **yes** | only with co-designed engines |
| `cudaMallocAsync` pools | export/import | yes, but **never release physical blocks** | harmful for eviction accounting |
| MPS | yes | no | scheduling only |
| Green contexts | one process | no | inside one process |
| MIG | — | — | **not available on 4060 Ti** |

**VMM works here [M]:** `cuMemGetAllocationGranularity` (2 MiB) →
`cuMemCreate(64 MiB, POSIX_FILE_DESCRIPTOR)` → `cuMemAddressReserve` →
`cuMemMap` → `cuMemSetAccess` → `cuMemExportToShareableHandle` all returned
`CUDA_SUCCESS`. PolyStrata already uses this: `resize_card_cache()` calls
`vmm_unmap(arena_, 0)` then refills (`src/poly/engine.cpp:1499-1509`), and the
elastic KV uses the same primitive (`src/core/layer.cpp:570`).

**MPS works on this GeForce [M]:** the daemon starts, the client reports
`MPS_ENABLED=1`; static SM partitioning is supported on Ampere+ and driver
r610+ gives partial fault isolation. But memory is still per-client unless MPS
v3 memory partitioning / cgroup `dmem` limits are configured. This machine has
kernel 7.2.9 with the `dmem` cgroup controller and open kernel modules, so hard
isolation is technically available — a Phase-3 hardening item, not the base.

**Green contexts work on sm_89 [M]:** a 2-group split (8 + 18 SMs) and
`cuGreenCtxCreate` succeeded. They isolate SMs inside one process, not across
processes.

**Unified memory oversubscription works [M]:** with 502 MiB free,
`cudaMallocManaged(2 GiB)` succeeded. But oversubscription is not weight
streaming: every weight is touched every token, so UVM would migrate pages at
~12 GB/s with fault overhead.

**Measured PCIe 4.0 ×8 bandwidth [M]** (128 MiB, 10 reps):

| Transfer | GiB/s | GB/s |
|---|---:|---:|
| H2D pinned | **12.07** | 12.96 |
| D2H pinned | **12.08** | 12.97 |
| H2D pageable | 11.56 | 12.41 |

So the dense-streaming ceiling is arithmetic:

| Resident weight traffic/token | tok/s ceiling at 12.9 GB/s |
|---|---:|
| 5 GB | **2.6** |
| 8 GB | **1.6** |
| 15 GB | **0.9** |
| 0.12 GB (Tiel MoE misses at 80% hit) | **107** |

Measured Tiel: **77.2 tok/s at 85% cache hit, 7.2 GiB cache [M]**. That is why
expert streaming works and dense streaming does not.

## 2. Fast model switching

Measured components:

| Stage | 5 GB model | 15 GB model | Source |
|---|---:|---:|---|
| process spawn + link (72 MB binary) | ~0.19 s | ~0.19 s | [M] |
| `cuInit` | 12 ms | 12 ms | [M] |
| `cuCtxCreate` + module load | 94–118 ms | 94–118 ms | [M] |
| GGUF mmap + page-cache copy | 0.5–1.5 s warm; 8–15 s cold | **1.5–4.3 s warm; 44.6–49.8 s cold** | [L] |
| weight H2D (if copied) | 0.4 s | 1.2 s | [E] from [M] |
| CUDA graph capture | 0.1–0.5 s | 0.3–1.5 s | [E] |
| tokenizer load (70–240 MB) | 0.3–1 s | 0.3–1 s | [L] |
| vision encoder start (CPU) | 0.9–2.2 s | 0.9–2.2 s | [L] |
| first-request warmup | 1–3 s | 3–8 s | [L] |

Observed end-to-end switches [L]:

| Transition | Time |
|---|---:|
| Tiel → Flint-12G (warm) | ~6 s |
| Tiel → Flint-24G (warm page cache) | 25–30 s |
| Flint-24G → Tiel (warm) | ~30 s |
| Tiel → Cyber-Tiel (cold) | 40+ s |
| server start → READY, Tiel | 4.3 s warm / 44.6 s cold |

The swap is **not CUDA-bound**; page cache and engine init dominate.

Make a swap take seconds, ranked:

1. **Do not reload — shrink in place.** `POST /v1/vram {"reserve_mib": N}`
   unmaps the expert arena via VMM and refills later (`main.cpp:776-800`,
   `engine.cpp:1473-1510`). Sub-second shrink; grow at ~1.4 GiB/s.
2. **Keep GGUF pages hot.** `mmap` + `madvise(MADV_WILLNEED)` on the next model
   while the current one serves; `MADV_DONTNEED` on the evicted one. 62 GB RAM
   holds Tiel (22.7 GB) + audio (6–8 GB) + page cache, but not all four LLMs.
3. **Persistent worker** with context + graphs + tokenizer, weights unmapped:
   removes 118 ms context and 0.3–1.5 s graph capture; costs ~150–400 MB host
   RSS and 0 VRAM when dormant.
4. **Pin a 1–2 GB staging buffer** once.
5. **Double-buffer:** read the new GGUF into page cache before evicting.
6. **Lazy tokenizer** in parallel.
7. Overlap H2D with graph capture on separate streams.

Realistic target: **3–6 s warm** for 15 GB, **15–25 s cold**, **1.5–3 s** for
5 GB.

### Eviction and proving VRAM came back

| Method | Latency | Guarantee |
|---|---:|---|
| `QUIT` line | 0.3–2 s [L] | graceful |
| `SIGTERM` | ms + teardown | process exit |
| `SIGKILL` | immediate | driver reclaims at teardown |
| `cudaFree` | 0.00 ms [M] | accounting synchronous, physical release deferred |
| `cudaFreeAsync` | async | **returns to pool, not OS** |
| `cuMemUnmap` + `cuMemRelease` | sub-ms [M] | returns physical pages |
| `cudaDeviceReset` | destroys context | frees everything |

Guarantee requires: `waitpid`, then poll NVML until `used` drops by the expected
amount (2 s typical, 10 s max), then `SIGTERM` → 3 s → `SIGKILL` → `waitpid`.
`StrataEngine` does that: `QUIT` for 2 s, then `SIGTERM` for 3 s, then
`SIGKILL` for 3 s.

On Linux `spawn()` / `spawn_kwargs()` starts the engine in its own session with
`PR_SET_PDEATHSIG=SIGKILL`. `signal_group()` sends the signal to that process
group, so a child the engine started dies with it. `contain()` is true when
the child's process-group id is its own pid.

Also check fd/handle leaks: all engine pipes/log fds must be `CLOEXEC`; the
`Vision` class passes a log file as `stderr` (`serve/server.py:2357`) — verify
`close_fds=True`. Attribute memory per PID (`nvidia-smi --query-compute-apps`),
not just the total.

### KV checkpoint: the biggest win

| Item | Number |
|---|---|
| Tiel KV at 256K, q8_0 | **3.0–3.35 GiB** |
| D2H copy at 12.0 GiB/s | **0.27 s** |
| H2D restore | **0.27 s** |
| Park to host RAM (engine already does) | 576.7 ms for 119,547 tokens (3.2 GB/s) [L] |
| Re-prefill if lost, 640–1,300 tok/s | **160–310 s for 200K tokens** [L] |

Checkpointing KV is ~500× cheaper than re-prefilling a 200K conversation. The
conversation cache already lives in the `strata-poly` process. A switch to
another model that uses the same binary does not exit that process: the engine
copies the live conversation onto the host shelf, frees the weights, and opens
the next GGUF. The shelf is keyed by the model file, context and KV format.
Coming back to that model puts the shelf into the new engine. There is no
cache file. `SAVE` / `RESTORE` stay the Strata project's engine. `strata-poly`
still refuses them. GigaChat Audio is not parked: its run config passes
`--conversation-cache-mib 0`, and a positive value is an error.

## 3. Residency budgets

Measured baseline:

| Line | MiB | Source |
|---|---:|---|
| `nvidia-smi` total | 16,380 | [M] |
| usable (`cuMemGetInfo` total) | 15,997 | [M] |
| desktop (kwin, Zed, Brave, Firefox, plasmashell, VPN, Telegram, Xwayland) | 2,929 | [M] |
| driver/context overhead | ~700 | [M] |
| **desktop baseline to reserve** | **~3,600** | [M] |
| Tiel dense + MTP head | ~2,600 | [E] |
| Tiel context/graphs/scratch | ~700 | [E] |
| Tiel KV @256K q8_0 | 3,200 | [L] |
| Tiel expert slot | 1.82 MiB | [L] |
| Tiel full expert set (10,496) | ~19.1 GiB | [L] |

Cross-check at 67K: 2,600 + 700 + 837 + 7,373 = 11,510 vs 11,640 measured
(within 1%) [M].

Tiel fixed cost (no expert cache) = 3,300 + KV:

| Context | Fixed MiB | Cache available with 3.6 GiB desktop |
|---|---:|---:|
| 32K | 3,700 | 9,080 |
| 64K | 4,100 | 8,680 |
| 128K | 4,900 | 7,880 |
| 256K | 6,500 | 6,280 (6.1 GiB) |

### Strategy A — exclusive single owner (recommended default)

- Tiel alone: 12.8 GiB engine + 3.6 desktop = 16.4 → cache ~6.1 GiB.
- Audio alone: ~8.8–9.8 + desktop 3.6 = 12.4–13.4 GiB.
- Downtime: 4–8 s warm Tiel, 25–50 s Flint-24G, plus 160–310 s if a long
  conversation must re-prefill (unless KV checkpoint exists).
- In-flight requests drain under `ModelGate`; new ones queue or get 503.
- Zero OOM by construction if preflight is done. Full 256K.

### Strategy B — co-residency with reduced Tiel cache

Available for `cache + audio` = 16,380 − 3,600 − fixed:

| Tiel context | Audio 5.0 GB | Audio 6.3 GB | Audio 8.0 GB |
|---|---:|---:|---:|
| **256K** | 1,672 MiB | **272 MiB** | **−920 (no fit)** |
| 128K | 3,272 | 1,872 | 872 |
| 64K | 4,072 | **2,672** | 1,672 |
| 32K | 4,472 | 3,072 | 2,072 |

Predicted Tiel decode, anchored on measured 77 tok/s @85% hit / 7.2 GiB and the
44→33 tok/s drop for a 4.3 GiB shrink in `docs/VRAM_ELASTIC.md`:

| Cache | Experts resident | Hit rate (Zipf fit) | Decode |
|---|---:|---:|---:|
| 7.2 GiB | 4,049 | 85% [M] | 77 tok/s [M] |
| 4.0 GiB | 2,250 | ~72% | 50–60 |
| 2.6 GiB | 1,460 | ~62% | 30–40 |
| 1.0 GiB | 560 | ~40% | 18–25 |
| 0.25 GiB (floor 128) | 128 | ~5% | **8–15** |

**256K + audio is impossible.** 64K + q3 audio leaves ~2.6 GiB cache →
**30–40 tok/s**. The good news: this can be done without unloading Tiel, via
`POST /v1/vram {"reserve_mib": 6000}`, which unmaps the arena and refills on
grow-back. But `--vram-elastic` and `--kv-grow` are mutually exclusive in the
Strata engine (`docs/VRAM_ELASTIC.md`); in `strata-poly` a shrink+grow cycle
costs a full refill (~3.5 s for 5 GiB). Debounce grow-back (e.g. 30 s).

### Strategy C — time-slice inside one process

Green contexts and streams exist, but both models must be resident to alternate
at token granularity, so the budget is the same as B. **Not viable for
Tiel+Audio.** Useful only for small models.

"Preemptible layers" is streaming with extra steps; at 12.9 GB/s it caps below
1 tok/s for dense weights.

### Strategy D — Conformer on CPU, decoder on GPU

Encoder for 120 min: ~18–36 TFLOP → **1.5–6 min on CPU**, 4–10 s on GPU [E].
Decoder q3 ~6.0 + KV 0.1–0.5 + ctx 0.4 ≈ 6.5 GiB. With Tiel 256K:
6,500 + 3,600 + 6,650 = 16,750 > 16,380 → **no fit**. At 64K: 14,350 → 2.0 GiB
cache, Tiel ~25–35 tok/s. **Good for batch transcription, bad for live voice.**

### Strategy E — small ASR resident

Parakeet/Whisper-small class: 0.5–1.5 GiB + 0.3 GB runtime. With Tiel 256K:
6,500 + 3,600 + 1,500 = 11,600 → **4.6 GiB cache**, Tiel ~60–70 tok/s [E].
Real-time transcription. Loses audio QA, summarization, emotion/event
reasoning, instruction-following over audio; keeps transcription + timestamps.
**The only way to get true parallelism with Tiel at 256K without hurting it.**

| Strategy | VRAM peak | Tiel | Audio | Fits 256K? | Verdict |
|---|---|---:|---:|---|---|
| A exclusive | 1 model + desktop | 77 tok/s | 5–8 s switch | ✅ | **default** |
| B co-resident 256K | >16 GB | 8–15 tok/s | 0 | ❌ | impossible |
| B co-resident 64K | 14.4 GB | 30–40 tok/s | 0 | ❌ | viable fallback |
| C time-slice | both resident | shared | request granularity | ❌ | reject |
| D CPU encoder + GPU decoder | 14.4 GB @64K | 25–35 tok/s | 1.5–6 min/2 h | ❌ | batch only |
| E small ASR resident | 11.6 GB | 60–70 tok/s | real-time | ✅ | **best parallelism** |

## 4. Manager inside `switch_on_request`

### State machine

```
ABSENT ──load──► LOADING ──ready──► READY ──evict──► DRAINING ──exit──► ABSENT
   ▲                │                   │                                   │
   │                └──fail──► FAILED ◄─┘ (crash)                           │
   │                            │  cooldown, retry_count, last_good          │
   └────────────────────────────┴───────────────────────────────────────────┘
READY ──shrink──► READY_DEGRADED ──grow──► READY     (elastic cache, same process)
```

Per model: `state, pid, pgid, started_at, last_request_at, last_ok_at,
fail_count, cooldown_until, vram_expected_mib, kv_tokens, config_mtime`.
Global: `owner`, `budget_mib`, `load_lock`, `switch_epoch`,
`reserved_for_desktop_mib`, `home_model`.

### Control loop

1. **Admission.** Resolve model; if ready → lease; else enqueue with class and
   deadline. Estimate need: `dense + kv(prompt + max_tokens) + cache_floor + 512`.
   If it cannot fit even after eviction → 503 + `Retry-After: 30` immediately.
2. **Preflight before eviction.** Read NVML free + per-process usage; require
   `free_after_evict ≥ need + reserve`. Today `_switch_to` skips this: it
   unloads the old engine then calls `start_model` directly, bypassing
   `ensure_loaded`, `min_free_vram_mib` and `before_load`.
3. **Single load lock.** `self.fifo` + `load_lock`; `ModelGate` already
   serialises switches while requests hold leases.
4. **Eviction order.** idle non-home → shrink home cache (`POST /v1/vram`) →
   evict home only if policy allows → refuse.
5. **Wait for release.** `SIGTERM` → `waitpid` → poll NVML until `used` drops by
   expected or 5 s → `SIGKILL` → `waitpid` → re-poll.
6. **Start + verify.** Per-model `ready_timeout_s` (currently global 900 s,
   `ENGINE_READY_S`); probe `/health` and a 1-token generate; record `last_good`.
7. **Post-start.** Restore KV if implemented; grow cache to target; background
   warm/prefill.
8. **Idle/TTL.** Demand-driven eviction preferred; `idle_unload_s` only for
   non-home models. Home is never timer-unloaded.
9. **Watchdog.** No token for `engine_silence_s` (300 s) → kill and mark FAILED.
10. **Metrics.** Per model: state, `load_seconds`, `switch_seconds`,
    `vram_used_mib`, `vram_returned_mib`, `oom_total`, `evictions_total`,
    `queue_depth`, `503_total`, `kv_park_restore_seconds`, p50/p95 TTFT.

### Priority and preemption

| Class | Example | Policy |
|---|---|---|
| `interactive` | chat, voice | may preempt `batch` at a request boundary, never mid-token |
| `batch` | 2 h transcription | runs when interactive idle; yields at token boundary |
| `background` | summaries, evals | only when GPU otherwise idle; first evicted |

Preemption is cooperative: send `STOP` (`serve/server.py:2076`), wait for the
engine to end the turn, then unload. A running kernel cannot be preempted from
another process.

Fairness: FIFO within class, aging across classes (a batch waiter older than
60 s is promoted once); explicit reservation for the home model;
`max_consecutive_switches` (e.g. 2) to prevent starvation.

### API behaviour

- **503 + `Retry-After`** when admission is refused. Today only `GpuBusy` carries
  `retry_after` (`serve/server.py:388-396`, mapped at 6251); `SwitchFailed`
  (6246) and `ModelBusy` (409) do not.
- Queue keepalive for waits under ~10 s (SSE comments).
- Never switch during a stream. If forced by a crash, emit a terminal `error`
  event with `retryable: true` and close cleanly.
- Idempotent retry: stateless server, so a retry re-prefills; document
  `Retry-After`.
- Warm pool: one persistent worker per frequent model with context/graphs/
  tokenizer, weights unmapped.
- "Keep main loaded": `home_model` in config; after any borrow, grow cache back
  or reload and restore KV; expose `home_ready` in `/v1/status`.

### Patch points in the current tree

| Location | Symbol | Change |
|---|---|---|
| `serve/model_gate.py:24` | `ModelGate` | add priority, deadline, `home` reservation, `Retry-After`; typed `Busy(retry_after)` |
| `serve/server.py:3481` | `_switch_to` | split into `preflight → evict → start → verify → post`; call the missing free-VRAM preflight; per-model ready timeout |
| `serve/server.py:3442` | `switch_model` | accept `mode=graceful\|force`, `deadline_s`; 503 + `Retry-After`; restore old model in background |
| `serve/server.py:3571` | `use_model` | pass request class; batch may yield at token boundary |
| `serve/server.py:3860` | `ensure_loaded` | generalise to `ensure_resident(model, need_mib)` |
| `serve/server.py:3845` | `free_vram_mib` | per-PID attribution + `wait_for_free(target, deadline)` |
| `serve/server.py:1096/1119` | `StrataEngine.unload/restart` | `PDEATHSIG`, `start_new_session`, short timeouts, confirm VRAM |
| `serve/server.py:2254` | `close` | 20 s → 3 s per TERM/KILL stage; `killpg` |
| `serve/server.py:79`, `serve/winjob.py:83` | `popen`/`contain` | Linux `prctl(PR_SET_PDEATHSIG, SIGKILL)` |
| `serve/server.py:2330,2436` | `Vision`/unload | one encoder process shared across models; elastic on GPU; CPU default for Tiel |
| `serve/server.py:3960` | `Service.vram` | manager shrink primitive; `grow_debounce_s` |
| `serve/server.py:7422` | `start_model` | `madvise(WILLNEED)` preload; `ready_timeout_s` |
| `serve/server.py:7790` | `main()` | `home_model`, `residency_policy`, `max_switches_per_min` |
| `serve/server.py:6246` | 503 mapping | `Retry-After` for `SwitchFailed`; retryable vs permanent |
| `serve/prometheus.py:109` | `strata()` | per-model residency/queue/switch metrics |

> Line numbers are from the research tree. `serve/server.py` changed by +61/−13
> lines between the research commit `2d6a8f9` and `a82999fd`; re-locate symbols
> before patching.

### Test plan

| Test | Method | Pass |
|---|---|---|
| Concurrency | 20 clients, two models, 10 min | no wrong-model answer; no overlapping loads; no gate deadlock |
| OOM injection | MockEngine allocates until failure; fake NVML 200 MiB free | 503 + `Retry-After`; no kill; old model still serves |
| Crash mid-load | engine exits before READY | 503; FAILED; cooldown; old model restored in background |
| Switch during stream | long SSE + switch | stream intact; switch after; no interleaved tokens |
| Repeated cycles | 500 × (Tiel→Air→Audio→Tiel) | 0 OOM; VRAM ±128 MiB; no fd growth |
| Orphan | `kill -9` server during load | engine gone ≤2 s; VRAM returned |
| KV checkpoint (phase 2) | 200K conversation, switch away/back | TTFT ≤2 s; tokens identical |
| Soak | 24 h mixed traffic | p95 switch ≤8 s warm; p95 TTFT ≤1.2× baseline; 0 OOM |

Extend `serve/test_model_gate.py`, `test_lifecycle.py`, `test_vram.py`,
`test_restart_waiters.py`, `test_vision_elastic.py`, `test_fatal_recovery.py`.

## 5. Verdict and phased plan

**Architecture (now inside the existing server):**

- **Home:** Tiel 256K, pinned, never timer-unloaded; expert cache is the shock
  absorber.
- **Borrowers:** audio decoder (exclusive, ~9 GB) or, if co-resident, q3 audio
  and Tiel at 32–64K with a small cache.
- **Exclusive:** Air / Flint 12G / Flint 24G; demand switch, never during a
  stream, with KV checkpoint/restore.
- **Encoder:** CPU `strata-audio`, elastic, started only while encoding.

**Physically impossible here:** Tiel 256K + 6–8 GB audio + desktop (~20 GB);
two 15 GB models; dense streaming >2.5 tok/s for 5 GB; sub-2 s cold swap of
15 GB; MIG; preempting a running kernel from another process.

| Phase | Work | Effort | Acceptance |
|---|---|---|---|
| **P0** | PDEATHSIG + killpg; preflight in `_switch_to`; shorter timeouts; `Retry-After`; cooldown/last_good; per-PID metrics | in this tree | `kill -9` leaves no engine; 100 cycles 0 OOM; VRAM ±128 MiB |
| **P1** | `home_model`; shrink-before-admit via `/v1/vram`; page-cache warm; 503 policy | 3–5 d | warm Tiel→Audio→Tiel p95 ≤8 s; Tiel ≥30 tok/s if co-resident |
| **P2** | one `strata-poly` process loads the next model and keeps the conversation shelf in RAM | in this tree | the same model, context and KV format come back in that process |
| **P3** | optional MPS static partitions + cgroup `dmem` caps | 1 wk | 24 h hostile-allocator soak: 0 OOM for Tiel |
| **P4** | resident small ASR + on-demand audio LLM | 1 wk | Tiel stays 256K and ≥60 tok/s; transcription real-time |
