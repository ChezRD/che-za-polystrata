# tools/audio — `strata-audio`, the GigaChat Audio Conformer encoder

`strata-audio` turns a WAV into the embeddings the GigaChat audio decoder reads in place of its
`[audio_token]` (id 128006) ids. It is the audio twin of `tools/vision/strata-vision`: a resident
helper process driven over stdin/stdout, one `SVE1` record per clip, the same record layout the
engine's `--serve GENI` path already reads for images.

CPU only for now (`ggml-cpu`, q8_0/f16 matmuls). The 4060 Ti is busy serving, so this allocates no
GPU memory at all and hides the card from ggml (`CUDA_VISIBLE_DEVICES=-1`).

Reference: [`ai-sage/GigaChat3.1-Audio-10B-A1.8B`](https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B)
@ `bf73d03`, `modeling_gigachat_audio.py`. Graph, parameters and risks:
[`docs/research/gigachat-audio/03-audio-pipeline.md`](../../docs/research/gigachat-audio/03-audio-pipeline.md).

## What it computes

```
wav -> downmix (mean) -> 16 kHz -> STFT 400/160 Hann periodic, center=True, reflect, power 2
    -> 64-bin HTK mel (norm=None, 0..8 kHz) -> log(clamp(x, 1e-9, 1e9))      (100 fps, 64 ch)
    -> Conv1d(64->1024, k5, s2, p2) + ReLU -> Conv1d(1024->1024, k5, s2, p2) + ReLU   (25 fps)
    -> 24 x Conformer layer, d=1024, 16 heads, d_ff=4096, conv k=5, LayerNorm eps 1e-5
         x += 0.5*FFN1(LN1(x)); x += Attn(LN2(x)); x += Conv(LN3(x));
         x += 0.5*FFN2(LN4(x)); x = LN5(x)
       Attn: block-diagonal over hard 200-frame (8 s) blocks, non-causal, no cross-block keys;
             RoPE base 5000 applied to the LN2 output *before* the Q/K projections, shared by all
             16 heads, split-half rotation, phase = (frame_index mod 200) * 5000^(-2i/64).
             linear_v gets the *unrotated* LN2 output.
    -> stacking x4 (4 x 1024 = 4096, zero pad to a multiple of 4) -> Linear(4096->1536) + bias
    -> (rows, 1536), one row per 160 ms
```

Row arithmetic (all of it load-bearing, off by one = a broken prompt):

```
T_mel  = 1 + floor(n_samples / 160)
T_enc  = floor((floor((T_mel-1)/2) + 1 - 1)/2) + 1      # two k=5/s=2 convs
n_rows = ceil(T_enc / 4)                                # stacking

10 s  (160000 samples):  T_mel 1001 -> T_enc 251 -> 63 rows   (NOT 62: see below)
30 s:                    T_mel 3001 -> T_enc 751 -> 188 rows
120 s:                   T_mel 12001 -> 3000 -> 750 rows
120 min (7200001):       T_mel 720001 -> 180001 -> 45001 rows
```

`03-audio-pipeline.md` §3 says 10 s -> 62 and 30 s -> 187. The shipped code (`_encoder_output_lengths`
+ `StackingSubsampler.calc_lengths_fn`) and the checkpoint itself give **63** and **188**; the
PyTorch reference in this directory (which runs the shipped code) agrees with the binary.

## Building

```sh
cmake -S tools/audio -B build-audio -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
cmake --build build-audio --target strata-audio        # -> build-audio/bin/strata-audio
cmake --build build-audio --target strata-audio -j     # same; portable AVX2 baseline by default
cp build-audio/bin/strata-audio engine-poly/strata-audio
```

`STRATA_PORTABLE=OFF` (what `setup.py` does for `strata-vision`) builds for the host CPU. The build
above is native: `-march=native`, OpenMP on, ggml-cuda off.

## Model file

`strata-audio` reads a finished GGUF. The file used here is
`gguf/gigachat-audio-enc-mixed.gguf` in the Hugging Face snapshot
`models--ai-sage--GigaChat3.1-Audio-10B-A1.8B`, revision `bf73d03`.
The engine does not quantize this checkpoint.

The file was packed once from `audio_encoder.safetensors` (2.37 GB F32) into ggml's layout:

| tensor | layout | type |
|---|---|---|
| `encoder.layers.N.*.*.weight` (Linear, 1x1 Conv1d) | bytes stay `[out, in]`; declared `ne = [in, out]` | q8_0 / f16 |
| `encoder.pre_encode.conv.{0,2}.weight` | `ne = [k*in, out]`, tap-major/channel-minor | q8_0 / f16 |
| `encoder.layers.N.conv.depthwise_conv.weight` | `ne = [ch, k]`, one contiguous 1024-float row per tap | **F32** |
| biases, LayerNorm weights | `ne = [ch]` | **F32** |
| `modality_adapter.projector.linear.weight` | `ne = [4096, 1536]` | q8_0 / f16 |
| `audio.mel_fb` | `ne = [64, 201]` = torchaudio's f32 filterbank, `[bin][mel]` | **F32** |

`mixed` is the file in the snapshot: 807 MB. q8_0 for every matrix would be 631 MB; f16 for every
matrix would be 1185 MB. In `mixed`, a matrix whose name contains `pre_encode`, `self_attn`,
`conv.pointwise`, `conv.depthwise` or `modality_adapter` stays f16. Every other matrix is ggml
q8_0: blocks of 32 values, float16 scale `amax/127`, int8 codes `round(x/scale)` clamped to
[-127, 127]. The projector is in the f16 set. The depthwise conv is stored F32, not q8_0 and not q4.

The mel filterbank ships inside the GGUF because torchaudio builds it in float32 from a float32
`linspace`; a float64 rebuild differs by up to 5e-6 per coefficient, which is visible on small mel
values. `strata-audio` still has the generator and uses it when the file has no `audio.mel_fb`.

## Process protocol

```
$ engine-poly/strata-audio --model gigachat-audio-enc-mixed.gguf [--threads N] [--group-blocks N]
strata-audio: weights loaded in 0.2 s (12 threads)
strata-audio: warmed up on 1.0 s in 2.1 s (101 mel frames, 7 rows)
READY 1536
ENC /path/a.wav /path/a.sve1
OK 63 0 9970 1898                    # rows, t0_ms, t1_ms, elapsed_ms
QUIT
```

* `READY <n_embd>` after the weights load and one warm-up encode, as `strata-vision` does.
* `ENC <wav> <out>` -> `OK <n_rows> <t0_ms> <t1_ms> <ms> [key=value ...]` or `ERR <message>`; the process stays up.
  The key=values are what the encode was: `device=cuda:0|cpu`, `gpu_util=N` (mean, NVML, sampled every 200 ms),
  `vram_mib` (this process's peak), `device_vram_mib` (the card's), `workspace_mib` (graph buffers),
  `weights_mib`, `chunks` (graph executions), and `mel_ms` / `pre_ms` / `enc_ms` / `adapter_ms`.
* `PROGRESS <rows_done> <rows_total> <ms> [gpu_util=N vram_mib=N]` after every 200-frame group while an `ENC`
  runs (never during the warm-up): the rows finished so far, for the server's live status.  A reader of the
  protocol must skip lines starting with `PROGRESS` until `OK`/`ERR`.
* `QUIT` exits 0.
* The output file is one record `int32 {0x31455653 'SVE1', n, nx, ny, n_embd}` with `nx = n`,
  `ny = 1`, followed by `float32 [n][n_embd]`, row *i* at audio time `i*160 ms`.
  (Same magic and header as `strata-vision`'s records; the engine's existing `SVE1` reader applies.)
* WAV in: PCM 8/16/24/32-bit and float 32/64, any channel count (downmixed by arithmetic mean),
  any rate (resampled to 16 kHz).

Integration (`GENA`, `input_audio`, the 1,019-token template, the `HH:MM:SS` anchors) is not part of
this step.

## Tests

This directory is the encoder binary, the same shape as `tools/vision/`. The checks live in
`tools/gigachat/`, with the other model harnesses (`tools/qwen35/`).

`tools/gigachat/reference.py` is the oracle: it imports the shipped `modeling_gigachat_audio.py`,
loads the same safetensors, and dumps the mel, the conv-subsampling output, every layer, the
encoder output and the adapter rows. It is CPU PyTorch, float32.

```sh
# oracle (needs torch/torchaudio/soundfile/numpy/safetensors on CPU)
python tools/gigachat/reference.py --wav clip10s.wav --out /tmp/ref10s.npz --layers 0 11 23

# the binary with intermediates
mkdir -p /tmp/dump10 && printf 'ENC %s /tmp/out10.sve1\nQUIT\n' clip10s.wav | \
    build-audio/bin/strata-audio --model ...-mixed.gguf --threads 12 --dump-dir /tmp/dump10

python tools/gigachat/test_audio.py --ref /tmp/ref10s.npz --dump /tmp/dump10 --sve1 /tmp/out10.sve1 \
    --layers 0 11 23
python tools/gigachat/test_protocol.py --bin engine-poly/strata-audio --model ...-mixed.gguf \
    --wav clip2s.wav --wav clip10s.wav --wav clip5s_44k_stereo.wav
```

### Measured (this PC, Ryzen-class 12-thread host, 2026-11-05)

`clip10s.wav` = 159531 samples @ 16 kHz (9.97 s), `--threads 12`, gate `cos > 0.9999`:

| quant | size | pre_encode cos | layer 0 | layer 23 (encoder) | rows cos | rows max abs | mean abs |
|---|---|---|---|---|---|---|---|
| q8_0 | 631 MB | 0.9999999 | 0.99982 | 0.99988 | **0.99984** | 1.12e-2 | 6.7e-4 |
| mixed (default) | 807 MB | 0.9999999 | 0.9999985 | 0.99993 | **0.99993** | 5.5e-3 | 3.8e-4 |
| f16 | 1185 MB | 0.9999999 | 0.9999998 | 0.9999999 | **0.99999986** | 3.2e-4 | 2.3e-5 |

Only `f16` clears `cos > 0.9999` on *every* layer; `mixed` clears it on the encoder and on the rows
(0.99993) but layer 0 is 0.999998 and layer 11 0.99997; `q8_0` (0.63 GB, the type
`03-audio-pipeline.md` §2.5 recommends) lands at 0.99984 on the rows, i.e. just under the gate.
That is why `mixed` is the default: 0.8 GB, above the gate, f16 on the boundary modules.

Mel, `clip10s.wav`, 998x64 frames, against the shipped float32 front end: `max|Δ| 2.17e-4`,
`mean 1.62e-6`. Against a float64 computation of the same front end (same filterbank, same window,
same resampler, STFT in f64): `max|Δ| 9.5e-7`, `mean 1.7e-7` — while the *reference* is `2.18e-4`
away from that float64 answer. So the port is at the exact front end's rounding floor and what is
left of the difference is the reference's own float32 STFT. Same on the other clips: 2 s `9.6e-5` vs
the reference and `8.7e-7` vs f64; 120 s `3.3e-4` vs the reference (its own f64 error is `3.33e-4`)
and `9.5e-7` vs f64; 44.1 kHz stereo `5.3e-3` vs the reference (`5.3e-3` of that is again its own
f32 resampler) and `6.1e-5` vs f64. `test_audio.py` therefore passes when the port matches the
reference *or* is itself within the 1e-4 gate of f64 — the latter is the case above 2 s.

Row counts: 2 s -> 13, 5 s @ 44.1 kHz stereo -> 32, 10 s -> 63, 30 s -> 188, 120 s -> 750 — all equal
to the reference's `T_enc`/`ceil(/4)`, checked by `test_protocol.py`.

Other measured runs:

* `clip5s_44k_stereo.wav` (44100 Hz, 2 ch): downmix + resample vs `torchaudio.functional.resample`
  on the samples: `cos 0.9999999`, `max|Δ| 3.9e-7` (RMS 0.0144). Rows `cos 0.99993`.
* `clip120s.wav` (2 min): 750 rows, `cos 0.99993` vs the reference, **22.4 s** wall, peak RSS
  **1101 MiB** (807 MB of weights). 15 encoder blocks -> 2 groups, i.e. the halo path.
* Speed on this Ryzen 5 8600G (6C/12T, AVX-512), with the box also running the decoder server
  (load average 15-22 during these runs): 10 s of audio in 1.9-4.8 s, 120 s in 21-28 s; RTF 0.19
  best case, ~0.2. 120 min is ~212 TFLOP of Conformer, so ~20-25 min at this rate — the reason a
  GPU path matters later.

## Limits / what remains

* **Grouped vs. one-shot windows are not bit-identical.** Files up to `--group-blocks N` blocks
  (default 8 = 1600 frames = 64 s; `0` = never split) run in one ggml graph; longer ones run in block-aligned groups
  with a **one-block (200-frame) halo** on each side — 200 >= the 2 frames per layer x 24 layers the
  depthwise conv needs, so the grouped result is *mathematically* exact (verified: a 2-block file
  gives bit-identical encoder output grouped vs. whole-file). It still differs by up to
  `max|Δ| 5e-3` on row values ~0.7, because ggml's quantized-activation dot products are
  discontinuous (a 1e-7 change in an activation can flip a q8/f16 code) and 24 layers amplify it.
  Making this bit-exact needs fixed-shape windows (pad every window to the same frame count) —
  the design to use when streaming lands.
* **Streaming** is not implemented. The pieces are known: a 6-frame mel ring, a 48-frame conv halo
  per window, the stacking carry, and `frame_index mod 200` for the RoPE phase; no attention KV is
  needed (blocks never talk to each other). The 200-frame grouping above is the offline version.
* **120 min** works arithmetically (45,001 rows) but was not run: at RTF 0.19 it is ~21 minutes of
  CPU. Memory would be dominated by the mel (184 MB) and the pre-encoded frames (180001 x 1024 x 4 =
  737 MB) — `AudioEncoder::encode` keeps both for the whole file; chunking the mel and the
  pre-encode per group would bring that down to a few MB and is the first thing to do for long
  inputs.
* **CUDA** is a second build of the same binary (`-DSTRATA_AUDIO_CUDA=ON`, target `strata-audio-cuda`):
  the same graph, `ggml_backend_cuda_init(0)` plus `ggml_backend_sched` with the CPU backend as the
  op-offload fallback, the q8_0 weights copied into one CUDA buffer at load.  `--gpu [--device N]` picks
  the card; without it the binary hides the card and runs on the CPU exactly as the CPU build does.
  Measured on the 4060 Ti (sm_89) beside the IQ3_S decoder: 494.51 s of audio -> 2.28 s wall (mel 0.29 s,
  Conformer 1.76 s, pre-encode 0.11 s), peak 1232 MiB (769 weights + 463 workspace), vs 95.15 s on the CPU.
  GPU-vs-CPU rows: cos 0.999955, max |d| 0.0105, same 3091 rows (CUDA q8_0 MMQ rounds differently).
* The adapter runs per group on `inner` frames; groups are 200-frame aligned, so stacking (x4)
  always lands on a group boundary.
* `--dump-dir` writes `mel.f32`, `pre_encode.f32`, `encout_<frame0>.f32`, `rows.f32`, and — when the
  whole file is one window — `layer_NN.f32` plus layer-0 stage dumps (`s0_n1`, `s0_ffn1`,
  `s0_attn_ctx`, `s0_attn`, `s0_dw`, `s0_conv`, `s0_ffn2`). Layouts are ggml's, i.e. `[frame][ch]`
  row-major for the 1024-wide dumps; the dump sizes are in the code.
* `STRATA_AUDIO_DUMP_PCM=<path>` writes the downmixed/resampled 16 kHz samples, for resampler tests.
* `STRATA_AUDIO_DUMP_FB=<path>` writes the 201x64 mel filterbank the run used (`[bin][mel]`); with
  the shipped table in the file this must equal `audio.mel_fb` byte for byte.
* `STRATA_AUDIO_LAYERS=N` runs only the first N layers (debugging the graph against the oracle).

## Files

| file | what it is |
|---|---|
| `strata_audio.cpp` | the encoder, mel front end, WAV reader, resampler, protocol |
| `CMakeLists.txt` | ggml-only build (no llama, no mtmd), `-DSTRATA_PORTABLE` like `tools/vision` |
| `reference.py` | PyTorch CPU oracle: mel, per-layer activations, adapter rows |
| `test_audio.py` | parity gates against the oracle |
| `test_protocol.py` | READY/ENC/OK/ERR/QUIT, SVE1 header, row-count arithmetic |
