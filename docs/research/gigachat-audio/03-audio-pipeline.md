# GigaChat Audio: encoder, prompt pipeline, quantization

Source: [`ai-sage/GigaChat3.1-Audio-10B-A1.8B`](https://huggingface.co/ai-sage/GigaChat3.1-Audio-10B-A1.8B), revision `bf73d03`.
Citations are `file#Lx-Ly` against the HF repo. Research done 2026-11-05, before the
`feature/gigachat-audio-mla` branch was created.

## 0. Bottom line

| Question | Verdict |
|---|---|
| Front end | torchaudio, **16 kHz mono, 64-bin HTK mel, win/fft 400, hop 160, Hann, power 2, log(clamp(1e-9,1e9)), center=True** |
| Encoder | 24-layer Conformer, d=1024, 16 heads, FFN 4096, conv k=5, **chunk-bidirectional attention in hard 200-frame (8.0 s) blocks**, RoPE base **5000**, applied **before** Q/K projection |
| Rate | 100 mel fps → /4 conv → 25 fps → /4 stack → **6.25 tok/s (160 ms/token)** |
| Tokens | 376 / 3,751 / 22,501 / **45,001** for 1 / 10 / 60 / 120 min |
| Decoder | stock `DeepseekV3ForCausalLM`, 26 layers, 64 experts top-4 + 1 shared, KV = 576 values/token/layer |
| KV @ 120 min | **713 MiB q8_0**, 377 MiB q4_0, 1,342 MiB f16 |
| Streaming | encoder is **not causal**; it is 8 s-block-bidirectional. Exact streaming needs ~100 KB of state and **no attention KV** |
| Quant | encoder **q8_0 everywhere (0.63 GB)**; decoder **Q4_K_M = 6.47 GB measured** |
| Fit on 4060 Ti | **~9.4 GB peak** at 120 min with q4_K_M + q8 encoder + q8 KV |

> **Critical discrepancy.** `config.json` advertises `win_length=320, n_fft=320,
> center=false` (`config.json#L167-L175`), but the shipped extractor ignores those
> keys and hardcodes `win_length = sr//40 = 400`, `n_fft=400`, `center=True`
> (`modeling_gigachat_audio.py#L916-L925`). The code produces the reference
> outputs. Resolve this first (risk R1).

## 1. Audio front end

From `AudioFeatureExtractor`, `modeling_gigachat_audio.py#L898-L973`:

```python
SAMPLE_RATE = 16000
self.num_mels    = 64
self.hop_length  = 16000 // 100      # 160
win_length       = 16000 // 40       # 400
MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=400, win_length=400,
               hop_length=160, center=True)
mel = torch.log(mel.clamp_(1e-9, 1e9))
```

| Parameter | Value |
|---|---|
| sample rate | 16,000 Hz; any input resampled with `torchaudio.functional.resample` |
| channels | mono by arithmetic mean |
| n_mels | 64 |
| n_fft / win | **400 / 400** (25 ms) |
| hop | 160 (10 ms) |
| window | Hann, periodic |
| power | 2.0 |
| mel scale | HTK, `norm=None`, `f_min=0`, `f_max=8000` |
| centering | `center=True`, `pad_mode='reflect'` |
| log | natural log of `clamp(x, 1e-9, 1e9)` |
| dither / pre-emphasis / CMVN | none |
| SpecAugment | none |

Frame count: `T_mel = 1 + floor(n_samples/160)` → **100·t + 1**. The `+1` is
load-bearing for the final token count.

Shapes per second: waveform 16,000 → log-mel **(100, 64)** → conv ×4 **(25, 1024)**
→ stacking ×4 **(6.25, 4096)** → projector **(6.25, 1536)**.

## 2. Encoder graph

Config: `config.json#L12-L33` (`d_model 1024`, `n_heads 16`, `n_layers 24`,
`ff_expansion_factor 4`, `conv_kernel_size 5`, `conv_norm_type layer_norm`,
`attn_chunk_size 200`, `subsampling striding_conv1d`,
`subsampling_conv_channels 1024`). `pos_emb_max_len=5000` → **RoPE base 5000**
(`L294`, used at `L603-L608`). `subs_kernel_size=None` → 5 (`L342`).

### 2.1 Subsampling

```
(B,T,64) -> (B,64,T)
Conv1d(64 -> 1024, k=5, s=2, p=2) + ReLU
Conv1d(1024 -> 1024, k=5, s=2, p=2) + ReLU
-> (B, 1024, T/4) -> (B, T/4, 1024)
```

Length recurrence: `floor((L-1)/2 + 1)` twice (`L366-L368`, `L384-L390`).
No norm inside the conv stack.

### 2.2 Conformer layer

Order (macaron, `fc_factor=0.5`), `L542-L586`:

```
x = x + 0.5*FFN1(LN1(x))
x = x + Attn(LN2(x))
x = x + Conv(LN3(x))
x = x + 0.5*FFN2(LN4(x))
x = LN5(x)
```

- FFN: `Linear(1024→4096) → SiLU → Linear(4096→1024)`, biases present.
- Conv module: `Conv1d(1024→2048,k=1)` → GLU → pad-mask → depthwise
  `Conv1d(1024,k=5,groups=1024,pad=2)` → **LayerNorm over channels** → SiLU →
  `Conv1d(1024→1024,k=1)`.
- **LayerNorm eps = 1e-5**, not 1e-6.
- Dropout is 0 at inference.

### 2.3 Attention — the part that will bite

RoPE table (`L447-L455`):

```python
inv_freq = 1.0 / (5000 ** (arange(0, 64, 2)/64))
t = arange(200)                     # 0..199 only
freqs = t[:,None] * inv_freq[None,:]
coss = cat([cos]*blocks)            # repeats every 200 frames
```

Phase is `(frame_index mod 200) · 5000^(-2i/64)`: it hard-resets at every chunk
boundary and never accumulates.

And then (`L490-L497`):

```python
query = apply_rotary(query.view(B,T,16,64), cos, sin).view(B,T,-1)
key   = apply_rotary(key.view(B,T,16,64), cos, sin).view(B,T,-1)
q = self.linear_q(query); k = self.linear_k(key); v = self.linear_v(value)
```

**RoPE is applied to the LayerNorm output before the Q/K projections**, and
`query == key == value == normed`. A port that rotates Q/K after projection
produces different logits. Rotate the 1024-d vector viewed as `(T,16,64)`,
split-half (`x1=…[:32]`, `x2=…[32:]`), same 32 angles for all 16 heads.

Attention is **block-diagonal, non-causal**: `(B·n_blocks, h, 200, 64)`, mask
`pos < valid_in_block` (`L517-L539`); Flash path uses varlen `cu_seqlens`
segments of 200 (`L646-L659`). No attention across chunks. Complexity O(T·200).

Output: `feat_out=-1` → no `out_proj` (`L636-L639`); output `(B,E,1024)`.

### 2.4 Weight table (822 tensors, F32 on disk)

Per layer, prefix `encoder.layers.i.`, 34 tensors, 24,153,088 params:

| Tensor | Shape | Params | Quant |
|---|---|---:|---|
| `self_attn.linear_{q,k,v,out}.weight` | [1024,1024] | 1,048,576 each | **q8_0** |
| `self_attn.linear_{q,k,v,out}.bias` | [1024] | 1,024 each | F16/F32 |
| `feed_forward{1,2}.linear1.weight` | [4096,1024] | 4,194,304 each | q4_K ok, q8 preferred |
| `feed_forward{1,2}.linear1.bias` | [4096] | 4,096 | F16 |
| `feed_forward{1,2}.linear2.weight` | [1024,4096] | 4,194,304 each | q4_K ok |
| `feed_forward{1,2}.linear2.bias` | [1024] | 1,024 | F16 |
| `conv.pointwise_conv1.weight` | [2048,1024,1] | 2,097,152 | **q8_0** |
| `conv.pointwise_conv1.bias` | [2048] | 2,048 | F16 |
| `conv.depthwise_conv.weight` | [1024,1,5] | 5,120 | **F16/F32** |
| `conv.depthwise_conv.bias` | [1024] | 1,024 | F32 |
| `conv.pointwise_conv2.weight` | [1024,1024,1] | 1,048,576 | **q8_0** |
| `conv.pointwise_conv2.bias` | [1024] | 1,024 | F16 |
| `conv.batch_norm.{weight,bias}` | [1024] | 1,024 each | **F32** (LayerNorm) |
| `norm_{feed_forward1,conv,self_att,feed_forward2,out}.{weight,bias}` | [1024] | 1,024 each | **F32** |

Non-layer:

| Tensor | Shape | Params |
|---|---|---:|
| `encoder.pre_encode.conv.0.weight` | [1024,64,5] | 327,680 |
| `encoder.pre_encode.conv.2.weight` | [1024,1024,5] | 5,242,880 |
| `modality_adapter.projector.linear.weight` | [1536,4096] | 6,291,456 |
| `modality_adapter.projector.linear.bias` | [1536] | 1,536 |

Totals: encoder **585,246,720**, adapter **6,292,992**, grand total
**591,539,712**. F32 file = 2,366,158,848 B.

FLOPs: 24.53 MMAC per frame per layer → **29.4 GFLOP/s of audio**; 120 min =
**212 TFLOP** for the encoder.

### 2.5 Quantization sensitivity

| Module | Params | Keep |
|---|---:|---|
| pre_encode convs | 5.57 M | q8_0 or F16 |
| attn q/k/v/out | 100.8 M | **q8_0** |
| FFN1/FFN2 | 402.9 M | q4_K acceptable, q8 preferred |
| conv pw1/pw2/dw | 75.7 M | **q8_0**, dw F16 |
| norms / biases | 0.25 M | **F32** |
| projector | 6.29 M | **q8_0 or F16** |

**Recommendation: q8_0 for the whole encoder (0.629 GB).** q4 saves only
0.30 GB and risks the mel→token boundary.

## 3. Adapter and token arithmetic

`ModalityAdapter` does **stacking first, then linear** (`L684-L693`).
`StackingSubsampler` zero-masks, pads T to a multiple of 4, reshapes to
`(B,T/4,4·1024=4096)`, returns `ceil(L/4)` (`L197-L219`). `LinearProjector`
is `nn.Linear(4096,1536)` with bias, no activation, no norm (`L238-L244`).

100 fps ÷ 4 ÷ 4 = **6.25 tok/s = 160 ms/token**.

| Duration | mel frames | encoder frames | audio tokens | anchors | audio block | full prompt* |
|---|---:|---:|---:|---:|---:|---:|
| 1 min | 6,001 | 1,501 | **376** | 1 | 392 | 1,416 |
| 10 min | 60,001 | 15,001 | **3,751** | 10 | 3,839 | 4,863 |
| 60 min | 360,001 | 90,001 | **22,501** | 60 | 22,989 | 24,013 |
| 90 min | 540,001 | 135,001 | **33,751** | 90 | 34,479 | 35,503 |
| 120 min | 720,001 | 180,001 | **45,001** | 120 | 45,969 | 46,993 |

\* measured with the real tokenizer: 1,019 fixed template tokens + user text +
audio block. README's "~90 min is 33k tokens" matches 33,751.

Short-clip counts verified against the shipped code: **10 s = 63 rows**,
**30 s = 188 rows** (not 62/187). The `+1` in `T_mel = 1 + floor(n/160)` and
`ceil` in the stacking subsampler account for it; the 120-minute figure,
45,001, is unchanged.

KV (MLA, K only; `has_v = !is_mla`):

| KV type | B/token/layer | B/token | 1 min | 10 min | 60 min | 120 min |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 1,152 | 29,952 | 40.4 MiB | 138.9 | 685.9 | 1,342.3 MiB |
| **q8_0** | 612 | **15,912** | 21.5 MiB | 73.8 | 364.4 | **713.1 MiB** |
| q4_0 | 324 | 8,424 | 11.4 MiB | 39.1 | 192.9 | 377.5 MiB |

## 4. Prompt / chat format

Special tokens (`tokenizer_config.json#L20-L123`, `config.json#L163-L166`):

| id | surface | role |
|---:|---|---|
| 1 / 2 | `<s>` / `</s>` | BOS/EOS |
| 128000 | `<|role_sep|>\n` | after role name |
| 128001 | `<|message_sep|>\n\n` | end of turn |
| 128002 / 128003 | `<|file|>` / `<|/file|>` | file envelope |
| **128006** | **`[audio_token]`** | `audio_token_id` |
| 128012 | `<|function_call|>` | tools |

`_to_template_messages` replaces the audio part with `<file_placeholder_{id}>`
(`L1143-L1171`); the template renders:

```
<|file|>{"id": "audio_0", "name": "a.wav", "type": "audio"}[audio_token]<|/file|>
```

Token count: 1 + JSON **21** + 1 + 1 = **24**. `expand_audio_tokens`
(`L859-L890`) replaces every id 128006 with `_audio_block(length, …)`:

```python
block = full((length + 8*k,), audio_token_id)     # L850
stamps = _hms_token_ids(secs, lut)                 # L852, 8 tokens "HH:MM:SS"
starts = positions + 8*arange(k)                   # L853
block[idx] = stamps.reshape(-1)                    # L855
parts.append(_hms_token_ids(dur, ...))             # L885-L887, duration after run
```

Timestamp rule (`L840-L845`, `L830-L837`, `L177-L183`):
- anchors = `arange(0, total_seconds, 60)`, i.e. 0, 60, 120, … strictly < total;
- each anchor is 8 tokens (`h h : m m : s s`);
- token index of anchor *j* is `secs_j*6.25 + 8j`;
- total duration is appended as one more 8-token `HH:MM:SS` **after** the audio
  run, before `<|/file|>`;
- `HH:MM:SS` restricted to < 100 h.

A 1-min clip is literally:

```
00:00:00 [367 audio tokens] 00:01:00 [1 audio token] 00:01:00
```

Full template for `user: "Transcribe the audio."`:

```
<s>developer system<|role_sep|>\n
{DEVSYSTEM}                                   # 5,128 chars, template L365-L424
<|message_sep|>\n\n
user<|role_sep|>\n
<|file|>{"id": "audio_0", "name": "a.wav", "type": "audio"}[audio_token]<|/file|>Transcribe the audio.<|message_sep|>\n\n
available functions<|role_sep|>\n[]<|message_sep|>\n\n
assistant<|role_sep|>\n
```

Prefix **987**, file envelope **24**, suffix **14** → fixed overhead **1,019**
tokens. **Do not skip the `available functions … []` block**: the reference
processor uses `use_suggester=True`, which injects it after every user turn.

There is **no task-specific system prompt**. Only the user text changes:

| Task | User text |
|---|---|
| ASR (ru) | `Transcribe the audio.` |
| ASR (en) | `Transcribe the audio in English.` |
| Translation | `Translate the audio into English.` |
| Audio QA | `What is the main topic of this recording?` |
| Temporal localization | `When does the speaker mention Star Wars midi-chlorians?` |
| Temporal description | `What issue is raised from 50:20 to 50:35?` |
| Summarization | `Summarize the English audio with segment start times.` |

Answers are plain text with `HH:MM:SS`; `<point>`/`</point>` exist but the
released model is the plain-text timing-token variant.

## 5. Streaming

**Not causal.** Attention is full within each 200-frame block and hard-reset at
boundaries (`L517-L539`), i.e. 8.0 s of intrinsic lookahead.

But the encoder is not strictly block-independent: the depthwise conv (k=5) runs
over the whole time axis at every layer, so information crosses chunk boundaries
with a per-layer receptive field of ±2 frames (±80 ms), ±48 frames (±1.92 s)
after 24 layers. Exact streaming needs a **2-frame per-layer halo**.

State to cache:

| State | Size | Why |
|---|---|---|
| mel ring, last 6 frames | 1.5 KB | two k=5/s=2 convs need mel 4m−6…4m+6 |
| 2-frame halo per layer of conv input | 24×2×1024×2 B = **98 KB** | depthwise k=5 |
| stacking carry | ≤3×4096 | pad to multiple of 4 |
| chunk phase | `frame_index mod 200` | RoPE repeats every 200 |
| **attention KV** | **0** | no cross-chunk attention |

**Chunk by encoder frame index** (`floor(enc_frame/200)`), not by wall clock.
Feeding 8.000 s wall-clock blocks with `center=True` yields 801 mel frames →
201 encoder frames; resetting every 8 s drifts **+1 frame per chunk** (900
frames = 36 s after 2 h).

The decoder **can** consume partial output (it is causal). Caveats: the trailing
duration marker is only known at end-of-audio, and timestamps are positional
(`secs*6.25 + 8j`), so live mic cannot place future anchors.

RTF estimates on one 4060 Ti:

| Stage | Work | Effective | 120 min |
|---|---|---:|---:|
| Encoder | 212 TFLOP | 8–12 TF/s | **20–35 s** |
| Decoder prefill | 172 TFLOP linear + ~65 TFLOP attn | 8–12 TF/s | **20–30 s** |
| Decode 500 tok | ~1.1 GB weights + 0.71 GB KV/token | 40–70 tok/s | 7–13 s |
| **Total** | | | **~50–80 s (RTF 0.007–0.011)** |

Per 8 s chunk: ~40–80 ms compute + mandatory 8.0 s lookahead (+80 ms halo) →
~8.1 s latency, RTF ≈ 1.0 by construction.

What breaks temporal grounding: boundary drift, per-chunk zero padding instead
of halo, wrong anchor index, dropping markers (paper: removing inter-timings
drops mIoU from 50.9 to 24.5), re-anchoring to wall clock.

## 6. Quantization / conversion

| Component | Params | FP32 | BF16 | q8_0 | q4_K | q3_K |
|---|---:|---:|---:|---:|---:|---:|
| Encoder | 585,246,720 | 2.341 GB | 1.170 GB | **0.622 GB** | 0.329 | 0.252 |
| Adapter | 6,292,992 | 0.025 | 0.013 | 0.007 | 0.004 | 0.003 |
| **Encoder total** | 591,539,712 | **2.366 GB** | 1.183 | **0.629** | 0.333 | 0.254 |
| Decoder | 10,672,535,616 | 42.690 GB | **21.345 GB** | **11.34** | 6.00 | 4.59 |

Decoder measured GGUF (`ai-sage/GigaChat3.1-10B-A1.8B-GGUF`, same parameter
count): **bf16 21.36**, **q8_0 11.35**, **q6_K 8.78**, **q4_K_M 6.47 GB**.

Decoder group breakdown:

| Group | Params | BF16 | q8_0 | q4_K | q3_K |
|---|---:|---:|---:|---:|---:|
| `embed_tokens` | 197,001,216 | 0.394 | 0.209 | 0.111 | 0.085 |
| `lm_head` | 197,001,216 | 0.394 | 0.209 | 0.111 | 0.085 |
| attention ×26 | 650,144,768 | 1.300 | 0.691 | 0.366 | 0.280 |
| dense FFN L0 | 41,287,680 | 0.083 | 0.044 | 0.023 | 0.018 |
| **routed experts ×25** | **9,437,184,000** | 18.874 | 10.027 | **5.308** | **4.058** |
| shared expert + router ×25 | 149,915,200 | 0.300 | 0.159 | 0.084 | 0.064 |

88.4% of the decoder is routed experts → **quantize experts, protect
attention/embeddings**.

Recommended types:

| Type | Size | Use |
|---|---:|---|
| Q4_K_M | **6.47 GB** | default; 2 h + q8 KV + q8 encoder still ~9.4 GB |
| Q5_K_M | ~7.6 GB (est.) | +0.5–1 pt quality; still fits |
| Q6_K | 8.78 GB | near-lossless; fits with q4 KV |
| Q3_K_XL | ~5.0 GB (est.) | only if 1.5 GB must be freed |
| Q8_0 | 11.35 GB | won't fit at 46k ctx + encoder |

Per-tensor policy: experts `q4_K`; `attn_q/kv_a/kv_b/attn_output` `q6_K`;
`token_embd`/`output` `q6_K`; `ffn_gate_inp` + `exp_probs_b` + norms **F32**
(router flips if quantized); `kv_b` (split `attn_k_b`/`attn_v_b`) **q6_K**.

**Existing converter: yes.** `decoder/config.json` is stock
`DeepseekV3ForCausalLM`, no `auto_map`; `decoder/` has 5 safetensors shards
(21.345 GB) + tokenizer. llama.cpp `conversion/deepseek.py` registers
`DeepseekV2Model → DEEPSEEK2`, forces `num_key_value_heads=1`, writes
`key_length=576`, `value_length=512`, `key_length_mla=192`,
`value_length_mla=192`, splits `kv_b_proj` into `k_b`/`v_b`.
`src/models/deepseek2.cpp` explicitly names `GigaChat3-10B-A1.8B`.

Safetensors → GGUF mapping:

| safetensors | GGUF |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` |
| `model.layers.N.self_attn.q_proj.weight` | `blk.N.attn_q.weight` |
| `model.layers.N.self_attn.kv_a_proj_with_mqa.weight` | `blk.N.attn_kv_a_mqa.weight` |
| `model.layers.N.self_attn.kv_a_layernorm.weight` | `blk.N.attn_kv_a_norm.weight` |
| `model.layers.N.self_attn.kv_b_proj.weight` | `blk.N.attn_k_b.weight` + `blk.N.attn_v_b.weight` (or legacy `blk.N.attn_kv_b.weight`) |
| `model.layers.N.self_attn.o_proj.weight` | `blk.N.attn_output.weight` |
| `model.layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` |
| `model.layers.0.mlp.{gate,up,down}_proj.weight` | `blk.0.ffn_{gate,up,down}.weight` |
| `model.layers.N.mlp.gate.weight` | `blk.N.ffn_gate_inp.weight` |
| `model.layers.N.mlp.gate.e_score_correction_bias` | `blk.N.exp_probs_b.bias` |
| `model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}_exps.weight` |
| `model.layers.N.mlp.shared_experts.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}_shexp.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` |

Metadata: `block_count 26`, `embedding_length 1536`, `feed_forward_length 8960`,
`head_count 32`, `head_count_kv 1`, `key_length 576`, `value_length 512`,
`key_length_mla 192`, `value_length_mla 192`, `expert_count 64`,
`expert_used_count 4`, `expert_shared_count 1`,
`expert_feed_forward_length 1280`, `leading_dense_block_count 1`,
`expert_weights_scale 1.0`, `expert_weights_norm true`,
`expert_gating_func sigmoid`, `rope.dimension_count 64`,
`rope.freq_base 100000`, `rope.scaling.type yarn, factor 64, original 4096,
beta_fast 32, beta_slow 1`, `rope_interleave true`.

## 7. End-to-end memory, 120 min

q4_K_M decoder, q8 encoder, q8_0 KV, 46,993-token prompt:

| Item | GB |
|---|---:|
| Decoder Q4_K_M | 6.47 |
| Encoder + adapter q8 | 0.63 |
| KV q8_0 (713 MiB) | 0.70 |
| CUDA context + cuBLAS | 0.4–0.8 |
| CUDA graphs / graph pool | 0.3–0.6 |
| Prefill activations (ubatch 512) | 0.3–0.8 |
| Encoder working set (streamed) | 0.02 |
| **Total** | **~8.8–9.8** |
| Headroom on 16 GB | **6–7 GB** |

Peaks: end of prefill (KV materialized + largest activation slab) and first
decode step, both ≈ 9.5 GB.

Do **not** materialize all 45k encoder frames. Chunked 200-frame blocks: <5 MB
working set; full materialization would be ~3.4 GB f16.

Latency: encoder 20–35 s + prefill 20–30 s + decode 7–13 s ≈ **50–80 s**
(RTF 0.007–0.011). Use `n_ubatch=512`; a 46k single ubatch will OOM.

## 8. Test plan

**A. Mel parity.** 20 clips (5 s–10 min, 16/44.1/48 kHz, mono/stereo), reference
vs C++. Gate: max|Δ| < 1e-4, mean < 1e-5, identical `T_mel = 100t+1`. Run both
400/center=True (code) and 320/center=False (config).

**B. Encoder layer parity.** Hook every layer. f32: cosine > 0.99999,
max abs < 1e-4; q8 kernel: cosine > 0.9999, max rel < 1e-3; adapter cosine >
0.9999.

**C. Logits / greedy.** 20 clips × 64 greedy steps. Gate: 100% top-1, max
|Δlogit| < 0.05 on argmax, KL < 0.01.

**D. ASR WER.** FLEURS-en 200, FLEURS-ru 200, Golos-crowd 100. Targets from the
model card: ≤ 6.5 / 4.4 / 14.7 (+0.5 absolute for q8 encoder, +1.0 for q4_K
experts).

**E. Timestamp accuracy.** TimeGround-1M `tl` test, 200 rows across 4 duration
buckets. Expected mIoU ≈ 40.3 / 46.8 / 48.3 / 48.9; MAE ≈ 3.0 s. Reject if
MAE > 3.5 s or mIoU drops > 2 pts vs f32.

**F. Long-audio drift.** 120 min must produce exactly 45,001 audio tokens, 120
anchors at `secs*6.25 + 8j`, duration `02:00:00`. Encode streaming vs offline;
require bit-identical (or < 1e-5).

**G. Memory / leak.** 50 sequential 10-min requests; VRAM returns ±100 MB,
RSS ±200 MB; one 120-min then 1-min fragmentation check.

**H. Quant matrix.** q8 encoder × {Q4_K_M, Q5_K_M, Q3_K_XL}; ship Q4_K_M unless
Q3 loses > 1.5 pts mIoU.

## 9. Ranked risks

| # | Risk | Impact | Evidence | Effort |
|---|---|---|---|---|
| **R1** | Mel config vs code: 320/center=False vs 400/center=True | silent quality loss / wrong frame count | `config.json#L167-L175` vs `modeling…#L916-L925` | 2–3 d |
| **R2** | RoPE before Q/K projection, shared heads, base 5000, chunk reset | every logit wrong if "fixed" | `#L447-L455`, `#L490-L497` | 2–3 d |
| **R3** | Streaming chunk drift (+1 frame per 8 s) | timestamps off by minutes after 2 h | `#L384-L390` vs `center=True` | 3–5 d |
| **R4** | Token-count off-by-one (`ceil` stack, `floor` conv) | hard failure at prompt build | `#L988-L1001`, `#L1372-L1375` | 1 d |
| **R5** | Conv halo omitted | 3–5 frame artifacts every 8 s | `#L406-L433` | 2–3 d (part of R3) |
| **R6** | Template omissions (`available functions []`, 1,019 prefix, JSON spacing) | distribution shift | `chat_template.jinja#L290-L305, L493-L500` | 1 d |
| **R7** | Timestamp expansion (`+8j`, `secs*6.25`, trailing duration) | grounding collapse | `#L840-L890` | 1 d |
| **R8** | MLA GGUF layout, YaRN mscale, `rope_interleave` | load fails or long-context drift | `conversion/deepseek.py#L439-L462`; `deepseek2.cpp#L65-L113` | 2–4 d |
| **R9** | q4 on attention/projector/depthwise conv | WER/mIoU regression | §2.5 | 2 d |
| **R10** | 46k prefill: all-position logits, unfused attention | OOM | — | 2 d |
| **R11** | Resampler mismatch (torchaudio sinc vs libsamplerate) | small drift on 44.1/48 kHz | `#L953-L954` | 0.5 d |
| **R12** | Encoder weights F32 on disk, bf16 cast in reference | slight divergence | header F32; `#L1350` | 0.5 d |
| **R13** | Encoder q8 GEMM throughput on 4060 Ti | RTF above estimate | — | 2 d |

## Pipeline

```
file / mic ──► downmix mean ──► resample 16k ──► STFT 400/160 Hann, power=2
                                                   │
                                          HTK 64-mel [0,8k], no norm
                                                   ▼
                                        log(clamp(x,1e-9,1e9))  (T,64) 100 fps
                                                   ▼
                         Conv1d(64→1024,k5,s2,p2)+ReLU ─► Conv1d(1024→1024,k5,s2,p2)+ReLU
                                                   ▼
                                          (T/4,1024)  25 fps / 40 ms
                     ┌──────────────────────────────────────────────────┐
                     │ 24 × [ x+=.5·FFN1(LN x); x+=Attn(LN x);          │
                     │        x+=Conv(LN x);   x+=.5·FFN2(LN x); LN ]   │
                     │ Attn: 200-frame blocks, no cross-block,          │
                     │       RoPE(base 5000) applied PRE-projection     │
                     └──────────────────────────────────────────────────┘
                                                   ▼
                              StackingSubsampler ×4 (concat 4×1024)  25→6.25 fps
                                                   ▼
                                     Linear 4096→1536 (+bias)  ──► (L,1536)
                                                   ▼
             scatter into positions where input_ids==128006 (count must equal L)
                                                   ▼
       DeepSeek-V3 MLA MoE: 26 layers, d=1536, 32 heads, latent 576, 64 experts top-4
                                                   ▼
                                     text (ASR / MT / QA / timestamps)
```

**What PolyStrata needs to add:** (1) torchaudio-exact mel kernel; (2) a Conformer
encoder GGUF arch with q8_0 2-D tensors and F32 norms; (3) the pre-projection
chunk-RoPE attention kernel; (4) stacking + linear adapter fused into the
embedding scatter; (5) the 1,019-token template renderer with timestamp/duration
expander; (6) a 200-frame streaming driver with 6-frame mel + 2-frame/layer conv
halos. The decoder itself is a supported `deepseek2` graph — only the attention
KV quant policy (q8_0 K, no V) and Q4_K_M expert packing need care.
