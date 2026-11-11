# Qwen3.8-Flash-Next REAP-320 on `strata`

`Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf` is a community prune of Qwen3.8-Flash-Next:
320 of 512 experts per layer, 48 layers, 262,144-token window, 62.37 GiB. The main `strata`
engine runs it through a native pack; `strata-poly` does not.

## What the pack needs

- `tools/iq_pack.py --gguf <REAP>.gguf --out packs/reap320 --compat-bf16`
  (1079 tensors, 303 served natively, 1.44 GB `dense.bin`, 320 experts per layer).
- `token_embd.weight` in the REAP file is Q2_K, which the native embedding does not read.
  Take the same vocabulary table from the base Qwen3.8 IQ2_XS shard, where it is IQ4_XS:
  `tools/embd_extract.py --gguf <IQ2_XS-00001>.gguf --out packs/reap320/embd-iq4xs.gguf`.
- The file has 8 layers with IQ4_NL gate/up; `STRATA_GU_FMTS` must include type 20, otherwise
  the engine refuses layer 1. The local tree has that fix and `native_grouped_parity` covers
  `(20,20)` and `(20,8)` at 2560x640. The gate/up grid also has to use 8-row blocks for IQ4_NL:
  `gu_rows` now follows `launch_gu`'s 80-dot condition through `gu_ipb`, otherwise the up half
  is never written.
- PLE (`per_layer_token_embd.weight`) stays in the GGUF and is read with `--ple-io direct`,
  the default. The table is 28.8 GB; `--ple-io ram` does not fit this machine.

## 256K, this box

RTX 4060 Ti 16 GB, Ryzen 5 8600G, 62 GB RAM, NVMe. Config: `strata-reap320.json`.
One run, unique 262,144-token prompt built from this repository's docs, 8,192-token answer:

| metric | value |
| --- | ---: |
| prefill | 1,249 tok/s |
| prefill time | 209.8 s |
| time to first token, CLI | 261.4 s (includes the 50 s page-lock of the resident copy) |
| decode | **50.19 tok/s** |
| MTP | 5,987 of 6,319 accepted (94.7%), 3.72 tokens/round |
| GPU expert slots | 2,522 (5.22 GiB) |
| resident RAM | 31.42 GiB page-locked |
| file reads during decode | 0 |
| free VRAM with everything loaded | 1,617 MiB |
| swap | unchanged (about 11.1 GiB free before and after) |
| lowest MemAvailable | 9.7 GiB |

In `--serve` the 31.42 GiB page-lock happens once at startup, so a request's TTFT is the
prefill time, not the 261.4 s CLI figure.

The mode matters more than any flag:

| 256K mode | decode |
| --- | ---: |
| plain `--mmap-experts`, no resident budget, 1,574 slots | 25.92 tok/s |
| `--resident-budget-gib 34`, 2,522 slots, `--spec 4` | **50.19 tok/s** |

The first row is not the model's speed; it reads 52.6 MB of experts per round. The second
keeps 31.42 GiB of experts in page-locked RAM and reads nothing during decode.

## Config

The working flags, also kept in the local (untracked) `strata-reap320.json`, are:

- `--resident-budget-gib 34`, `--mmap-experts`, `--vram-reserve-mib 2048`;
- `--spec 4 --spec-min-p 0.5`, `--mtp .../mtp/rt` (canonical 512-expert head);
- `--kv k8v4 --kv-resident 32768`, `--max-context 270336`;
- `--adapt-async 1 --adapt-every 1 --adapt-swaps 96 --adapt-decay 0.97`;
- `STRATA_RESIDENT_HEADROOM_GIB=4`, `STRATA_KV_PREFETCH=1`, `STRATA_EXPDEC=1`.

A one-shot CLI run, with `$PACK` and `$REAP` set to the pack and the REAP file:

```sh
build-poly-local/strata --pack "$PACK" --native "$REAP" \
  --embd-gguf "$PACK/embd-iq4xs.gguf" --ple-gguf "$REAP" \
  --expert-profile "$PACK/expert-profile.bin" --expert-cache auto \
  --prefill auto --spec 4 --spec-min-p 0.5 --mtp .../mtp/rt \
  --max-context 270336 --kv k8v4 --kv-resident 32768 \
  --vram-reserve-mib 2048 --mmap-experts --resident-budget-gib 34 \
  --adapt-async 1 --adapt-every 1 --adapt-swaps 96 --adapt-decay 0.97 \
  --tokens-file <ids> --max-new 8192
```

Notes:

- `--adapt-async` and `--expert-profile-save` act in `--serve`; the CLI accepts them but
  prints that asynchronous adaptation is off.
- The REAP file's expert blobs are larger than Swift's (4.43 MB against 2.33 MB), so the same
  reserve gives 2,522 slots here against 3,132 for Swift. The desktop held about 2.1 GiB of
  VRAM during the run.
- The engine in the tree is 0.1.58; the Swift config points at 0.1.45.
- `--resident-cpu-experts` with `--ram-keep-gib` is not the same mode. It copies the whole
  expert complement, reaches 30.62 tok/s at 2K, and swaps other applications at 256K. Use
  `--resident-budget-gib` instead.

## Quality

REAP removes about 40% of the experts. The model card reports HumanEval 95.1% for the Q2
variant and 96.3% for the Q3 variant, and 25% fabrication on its 12-question general-knowledge
probe. Treat it as a coding and agent model, not as a general-knowledge replacement. The
94.7% MTP acceptance above is a favorable prompt; a 2K code prompt measured 61.0% and a
128K unique prompt in the file-tier run measured 73.3%.

## Artifacts

- Pack: `/home/chez/.cache/huggingface/polystrata-data/packs/reap320`
- Embedding: `packs/reap320/embd-iq4xs.gguf`
- Profile: `packs/reap320/expert-profile.bin` (trace-built, 320 wide)
- MTP: `/home/chez/.cache/huggingface/polystrata-data/mtp/rt`
- Config: `strata-reap320.json` (local, untracked; the flags above are the record)
