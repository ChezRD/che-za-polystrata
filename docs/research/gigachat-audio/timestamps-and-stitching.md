# Long-audio cues and stitching, measured on Russian ASR datasets

Date: 2026-11-06. GPU: one RTX 4060 Ti 16 GB. Model: GigaChat Audio 10B-A1.8B, decoder IQ3_S, KV q8_0, CUDA
Conformer, `serve/segment.py` 300 s parts / 15 s overlap, `CUE_SECONDS = 6`.

Datasets: Podlodka (conversational, close microphone), FLEURS `ru_ru` validation (clean read speech),
Golos `sberdevices_golos_100h_farfield` test (spontaneous, far microphone, low level).  No personal audio.

## What was wrong

The model writes no timestamps unless its prompt asks for them, so `verbose_json` returned one segment for the whole
file and `srt`/`vtt` one cue.  Asked for stamps (`timestamps=true`), it writes `M:SS — text`, but those stamps drift
(median 6 s, up to 33 s) and can put a boundary in the wrong order, so they are opt-in.

Two more failures showed up on the new datasets:

* digital silence and steady non-speech (a tone, stationary noise) made the model answer a refusal, a music tag or a
  loop.  A 3 s noise clip generated `— А ты? — А я? …` until the token cap.
* on far-field speech the two parts phrase the 15 s overlap differently, so the word alignment found no run reaching
  the end of the previous part and kept the whole head: 24 words said twice at one boundary.

## Timestamps are opt-in

By default the endpoint answers with the assembled text alone: `json`/`text` return `{"text": ...}`, and
`verbose_json` returns task, language, duration and text, with no `segments` key.  Times are added only when asked
for:

| request | answer |
|---|---|
| `timestamps=true` | `verbose_json.segments`, cues of about 6 s |
| `timestamp_granularities[]=segment` | the same (OpenAI's own field) |
| `response_format=srt` / `vtt` | subtitles; the format implies times, `timestamps=false` is a 400 |
| `model_timestamps=true` | use the model's own `M:SS` spans as anchors instead of the estimate |

## What the cues are now

Each part's kept words are spread over the audio the part actually covers - the 15 s shared head excluded - in cues
of about 6 s, with a preferred break at a sentence end.  With `timestamps=true` the model's spans are used as
anchors and each span is itself spread into 6 s cues, so an anchor every 20-30 s keeps the clock honest.

## Non-speech and loops

* `serve.transcribe.is_silent`: peak below 0.1 % of full scale, or frame-level level variation (20 ms frames) below
  0.15.  Speech measures 0.6-1.2, a tone 0.006, white noise 0.03.  Such a clip returns an empty transcript and never
  reaches the model.
* `serve.transcribe.clean_answer`: a refusal, a markdown commentary or a lone `[Музыка]` tag is not a transcript
  (returns empty); a block of 1-40 words repeated four or more times keeps two copies, which caps a decoding loop.
* A part's answer is bounded by its own length (8 tokens a second + 128, twice the measured rate), so a loop cannot
  run to the context limit.  `max_tokens` from the client still wins.

## Stitching on far-field speech

`align` used to require a run of at least 28 % of the expected overlap words.  Golos far-field phrases the same audio
as "салют йоко" in one part and "салют фильм" in the next, so only the last seven words agreed (`мерфи как он искал
невесту в америке`) and the cut was refused.  A run that reaches the end of the previous transcript is now enough at
four words (three for a tiny transcript).  Boundary duplicates on the 7-minute Golos concat: 3 runs / 24 words -> 0.

## Podlodka test material

`tools/gigachat/asr_concat.py` joins a dataset's own clips with 0.4 s silence and records each clip's exact start (it
converts float WAV, as FLEURS ships, to PCM16).  `tools/gigachat/check_segments.py` compares a saved `verbose_json`
answer with that manifest.

| file | audio | clips | cues | stitched duplicates | cut losses |
|---|---:|---:|---:|---:|---:|
| Podlodka 8 min | 494.51 s | 20 | 64 | 0 | 0 |
| Podlodka 15 min | 915.12 s | 38 | 119 | 0 | 0 |
| Golos farfield 7 min | 422.80 s | 133 | 58 | 0 | 0 |
| FLEURS 7 min | 420.90 s | 39 | 54 | 0 | 0 (1 digit written as a word: "40" -> "сорока") |

Duplicates are 8-word runs that occur twice within 30 s and are absent from the reference.  Losses are reference
words from each 15 s shared region that the whole transcript does not hold.

## Cue accuracy against the known clip starts

Each clip's words are searched only inside their own part, so a repeated clip set cannot match the wrong pass.

| file | median cue start error | p90 | max | cue/clip IoU | clip words in a cue touching the clip (+-3 s) | 5-gram coverage |
|---|---:|---:|---:|---:|---:|---:|
| Podlodka 8 min | 2.07 s | 8.07 s | 10.74 s | 0.753 | 99.54% | 0.725 |
| Podlodka 15 min | 3.36 s | 9.80 s | 15.31 s | 0.701 | 99.33% | 0.743 |
| Golos farfield 7 min | 2.70 s | 7.02 s | 8.41 s | 0.209 | 91.12% | 0.587 |
| FLEURS 7 min | 3.31 s | 12.44 s | 28.81 s | 0.422 | 85.69% | 0.744 |

Golos clips are 1-5 s, so a cue covers several of them and IoU against one short clip is small by construction; the
per-clip start error is the meaningful figure.  FLEURS loses about 6 % of the words (the model skips a stretch of
dense read speech), and since cue times are spread over the part, that skip shifts the following cues early - the
28.81 s outlier.  This is a model-coverage limit, not a stitching one.

## Word accuracy (short clips, one request each)

`tools/gigachat/eval_asr_ru.py` scores a sample of clips against their reference; normalization is case, ё and
punctuation.  60 clips per dataset, 2-20 s.

| dataset | WER | CER |
|---|---:|---:|
| FLEURS ru_ru (clean read) | 4.74% | 1.72% |
| Golos 100h farfield (spontaneous, far mic) | 15.53% | 6.91% |

## The model's own timestamps are opt-in

`timestamps=true` asks for the timed transcript.  Measured on the same 7-minute files:

| file | cue start median | p90 | words in place | 5-gram |
|---|---:|---:|---:|---:|
| FLEURS, estimated cues | 3.31 s | 12.44 s | 85.7% | 0.744 |
| FLEURS, model stamps | **2.08 s** | **4.92 s** | **93.7%** | **0.838** |
| Golos, estimated cues | 2.70 s | 7.02 s | 91.1% | 0.587 |
| Golos, model stamps | 17.15 s | 23.09 s | 37.7% | 0.535 |

The timed prompt helps clean read speech and hurts far-field speech badly, so it stays off by default.

## Reproduction

FLEURS, Golos and Podlodka are Hub datasets. `hf://datasets/...` reads the file from
`${HF_HOME:-$HOME/.cache/huggingface}/hub` and downloads it there if it is missing.
Podlodka is `bond005/podlodka_speech`. The test split is 20 clips; episode 132,
«Дисфункции организаций», is one of them. Answers and the concatenated WAV are
written under `out/`.

```bash
python tools/gigachat/asr_concat.py \
  --parquet hf://datasets/bond005/podlodka_speech/data/test-00000-of-00001.parquet \
  --seconds 900 --out out/podlodka15
curl -sS http://127.0.0.1:8081/v1/audio/transcriptions \
  -F file=@out/podlodka15/audio.wav -F model=gigachat-audio-10b-a1.8b \
  -F response_format=verbose_json -F language=ru -F max_tokens=2048 \
  -F segment_seconds=300 -F overlap_seconds=15 -o out/p15.json
python tools/gigachat/check_segments.py --manifest out/podlodka15/manifest.json \
  --answer out/p15.json

python tools/gigachat/eval_asr_ru.py \
  --data fleurs=hf://datasets/google/fleurs/parquet-data/ru_ru/validation-00000-of-00001.parquet \
  --data golos=hf://datasets/bond005/sberdevices_golos_100h_farfield/data/test-00000-of-00001-7fb58d4cda8ff9fe.parquet \
  --limit 60 --out out/eval-asr-ru.json
```

## Keeping Tiel's chat prefix across a voice request

A voice message needs GigaChat Audio, and the two models do not fit in 16 GB together, so the server switches Tiel
out for the transcription.  When both models run in the same `strata-poly` process, that process stays up: it shelves
Tiel's parked conversations in RAM, loads GigaChat, and puts the shelf back when Tiel returns.  There is no cache file.

Measured on the 4060 Ti (Tiel Coder 35B A3B Q4_K_XL, 262144 context, q8_0 KV, 4-slot conversation cache):

| step | result |
|---|---|
| chat, 2891 prompt tokens | prefill 2662.9 ms (1085.7 tok/s), 24 generated |
| voice request -> switch | Tiel cache saved: 1 conversation, 2914 tokens, 298 MB, 2415 ms |
| GigaChat transcription | 17.2 s clip in 15.8 s |
| same chat -> switch back | Tiel cache restored: 1 conversation, 2914 tokens, 1394 ms |
| the same chat again | `cached_tokens` 2890, `prompt_n` 1, prefill 219.5 ms |

So the prefix came back: a 12x shorter prefill on the request after the voice message.  Those timings were taken
while the cache was copied through a file.  The process now keeps the same rows in RAM and does not write that file.
