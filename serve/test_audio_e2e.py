"""End to end: the OpenAI audio endpoints and chat `input_audio` against the real GigaChat audio model.

Opt-in. A plain unittest run skips it, so a clone with no weights and no server still passes.

    STRATA_AUDIO_E2E=1 .venv/bin/python -m unittest serve.test_audio_e2e -v

Clips come from the Hub dataset google/fleurs, config ru_ru, split validation
(`hf://datasets/google/fleurs/parquet-data/ru_ru/validation-00000-of-00001.parquet`).
The reference text is that row's `transcription`. `STRATA_AUDIO_PARQUET` overrides the file.
The tokenizer is the HF snapshot `tokenizer/` directory
(`models--ai-sage--GigaChat3.1-Audio-10B-A1.8B`, revision bf73d03), or `STRATA_GIGACHAT_TOKENIZER`.

The server is already `STRATA_AUDIO_MODEL` (default `gigachat-audio-10b-a1.8b`).
"""
import base64
import json
import os
import sys
import time
import unittest
import urllib.request
import uuid
from pathlib import Path

from serve import audio as A
from serve.frontend import ChatTemplate, audios_of, openai_to_messages
from serve.server import MockEngine, Service, load_tokenizer

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "gigachat"
sys.path.insert(0, str(_TOOLS))
from hf_path import resolve_data  # noqa: E402

URL = os.environ.get("STRATA_AUDIO_URL", "http://127.0.0.1:8081")
KEY = os.environ.get("STRATA_API_KEY", "local")
MODEL = os.environ.get("STRATA_AUDIO_MODEL", "gigachat-audio-10b-a1.8b")
FLEURS = ("hf://datasets/google/fleurs/parquet-data/ru_ru/"
          "validation-00000-of-00001.parquet")
SNAPSHOT = "bf73d03a43bdf5118f5a4dbdc24ba6f56ac31cfb"
TEXT = "Transcribe the audio."
# One clip may miss more words than the 60-clip mean (4.74% on this split). Above this the
# answer is a different text, not a transcription of the row.
WER_CEILING = 0.25


def tokenizer_dir() -> Path:
    raw = os.environ.get("STRATA_GIGACHAT_TOKENIZER", "")
    if raw:
        return Path(raw)
    root = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    return (root / "hub" / "models--ai-sage--GigaChat3.1-Audio-10B-A1.8B" / "snapshots"
            / SNAPSHOT / "tokenizer")


def enabled():
    return os.environ.get("STRATA_AUDIO_E2E") == "1"


def wav_info(data: bytes) -> tuple[int, int]:
    """(frames, sample rate) of a PCM or float WAV."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV clip")
    pos, rate, align = 12, 0, 0
    frames = 0
    while pos + 8 <= len(data):
        tag, n = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if tag == b"fmt ":
            rate = int.from_bytes(data[pos + 12:pos + 16], "little")
            align = int.from_bytes(data[pos + 20:pos + 22], "little") or 1
        elif tag == b"data":
            frames = n // max(1, align)
            break
        pos += 8 + n + (n & 1)
    if rate <= 0 or frames <= 0:
        raise ValueError("WAV has no samples")
    return frames, rate


def encoder_rows(n_samples: int) -> int:
    """Rows the encoder emits for 16 kHz samples. Same arithmetic as tools/gigachat/test_protocol.py."""
    t_mel = 1 + n_samples // 160
    t1 = (t_mel - 1) // 2 + 1
    t2 = (t1 - 1) // 2 + 1
    return (t2 + 3) // 4


def norm(text: str) -> list[str]:
    import re
    strip = re.compile(r"[^\w]+", re.UNICODE)
    return [w for w in (strip.sub("", x.lower().replace("ё", "е")) for x in text.split()) if w]


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    words = norm(ref)
    return edit(words, norm(hyp)) / max(1, len(words))


def cue_text(body: str) -> str:
    """The spoken lines of an srt or vtt body, without cue numbers and timestamps."""
    lines = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s == "WEBVTT" or s.isdigit() or "-->" in s:
            continue
        lines.append(s)
    return " ".join(lines)


def load_clips(spec: str, n: int = 2) -> list[dict]:
    """The first `n` distinct 16 kHz clips of 4-12 s, in id order.

    FLEURS repeats an id for several recordings of one sentence. One recording per id.
    The reference text is that row's transcription.
    """
    path = resolve_data(spec)
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["id", "audio", "transcription"])
    picked = {}
    for i, row in enumerate(table.to_pylist()):
        data = row["audio"]["bytes"]
        frames, rate = wav_info(data)
        if rate != 16000 or not (4 * rate <= frames <= 12 * rate):
            continue
        text = (row.get("transcription") or "").strip()
        if not text:
            continue
        key = row.get("id", i)
        if key in picked:
            continue
        picked[key] = {"id": key, "bytes": data, "text": text, "frames": frames, "rate": rate}
    clips = [picked[k] for k in sorted(picked)]
    if len(clips) < n:
        raise RuntimeError(f"{path}: fewer than {n} clips of 4-12 s at 16 kHz")
    return clips[:n]


def chat(body: dict, timeout: float = 1200.0) -> dict:
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def multipart(fields: dict, filename: str, data: bytes):
    boundary = "----strata" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
             f"Content-Type: application/octet-stream\r\n\r\n").encode()
    body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(body)


def post(path: str, fields: dict, filename: str, data: bytes, timeout: float = 1200.0):
    ctype, body = multipart(fields, filename, data)
    req = urllib.request.Request(URL + path, data=body, method="POST",
                                 headers={"Content-Type": ctype, "Authorization": f"Bearer {KEY}"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8"), time.time() - started


def prompt_tokens(rows: int, duration: int, name: str, tok_dir: Path) -> int:
    """What the prompt must come to: the model's template with the audio file form, then the block expansion."""
    tok = load_tokenizer(tok_dir)
    template = ChatTemplate(tok_dir / "chat_template.jinja")
    messages, tools, kwargs = openai_to_messages({"messages": [{"role": "user", "content": [
        {"type": "audio", "source": "x.wav", "format": "wav", "name": name},
        {"type": "text", "text": TEXT},
    ]}]})
    svc = Service(MockEngine(tok, "ok"), tok, template)
    form, _ = svc.audio_messages(messages, audios_of(messages))
    ids = svc.encode_prompt(form, tools, kwargs)
    pad = tok.encode(A.AUDIO_TOKEN, parse_special=True)[0]
    out, _ = A.expand_audio_tokens(ids, pad, [rows], [duration], A.timing_lut(tok))
    return len(out)


@unittest.skipUnless(enabled(), "set STRATA_AUDIO_E2E=1; clips are google/fleurs ru_ru validation")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = os.environ.get("STRATA_AUDIO_PARQUET", FLEURS)
        cls.tok_dir = tokenizer_dir()
        try:
            cls.clips = load_clips(spec)
        except Exception as e:
            raise unittest.SkipTest(f"FLEURS ru_ru validation is not available ({e})") from None

    def assert_transcript(self, hyp: str, ref: str):
        score = wer(ref, hyp)
        self.assertLessEqual(score, WER_CEILING, f"WER {score:.3f} over the dataset text\nref {ref}\nhyp {hyp}")

    def test_chat_input_audio_returns_the_transcript_in_content(self):
        clip = self.clips[0]
        wav = clip["bytes"]
        answer = chat({"model": MODEL, "temperature": 0, "max_tokens": 128, "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
            {"type": "text", "text": TEXT},
        ]}]})
        message = answer["choices"][0]["message"]
        print(f"\nchat id={clip['id']}: content={message.get('content')!r} "
              f"reasoning={message.get('reasoning_content')!r} usage={answer['usage']}", flush=True)
        self.assertTrue((message.get("content") or "").strip(), "the transcript is not in content")
        self.assertFalse(message.get("reasoning_content"), "the transcript must not be reasoning_content")
        self.assert_transcript(message["content"], clip["text"])
        if (self.tok_dir / "vocab.json").is_file() and (self.tok_dir / "chat_template.jinja").is_file():
            dur_ms = int(clip["frames"] * 1000 / clip["rate"])
            rows = encoder_rows(clip["frames"])
            self.assertEqual(answer["usage"]["prompt_tokens"],
                             prompt_tokens(rows, dur_ms // 1000, "audio_0.wav", self.tok_dir))

    def test_transcriptions_json(self):
        clip = self.clips[0]
        status, ctype, body, elapsed = post("/v1/audio/transcriptions",
                                           {"model": MODEL, "response_format": "json"},
                                           f"fleurs-{clip['id']}.wav", clip["bytes"])
        print(f"\njson id={clip['id']} ({elapsed:.1f} s): {body}", flush=True)
        self.assertEqual((status, ctype), (200, "application/json"))
        self.assert_transcript(json.loads(body)["text"], clip["text"])

    def test_transcriptions_text(self):
        clip = self.clips[0]
        status, ctype, body, elapsed = post("/v1/audio/transcriptions",
                                           {"model": MODEL, "response_format": "text"},
                                           f"fleurs-{clip['id']}.wav", clip["bytes"])
        print(f"\ntext id={clip['id']} ({elapsed:.1f} s): {body!r}", flush=True)
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/plain"))
        self.assert_transcript(body, clip["text"])

    def test_transcriptions_verbose_json(self):
        clip = self.clips[0]
        duration = round(clip["frames"] / clip["rate"], 3)
        status, ctype, body, elapsed = post("/v1/audio/transcriptions",
                                           {"model": MODEL, "response_format": "verbose_json"},
                                           f"fleurs-{clip['id']}.wav", clip["bytes"])
        print(f"\nverbose_json id={clip['id']} ({elapsed:.1f} s): {body}", flush=True)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual((data["task"], data["language"]), ("transcribe", "ru"))
        self.assertAlmostEqual(data["duration"], duration, places=3)
        self.assert_transcript(data["text"], clip["text"])
        self.assertNotIn("segments", data)                        # by default: the text alone

        status, _, body, _ = post("/v1/audio/transcriptions",
                                  {"model": MODEL, "response_format": "verbose_json", "timestamps": "true"},
                                  f"fleurs-{clip['id']}.wav", clip["bytes"])
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertGreaterEqual(len(data["segments"]), 1)         # asked for: cue times
        self.assertEqual(data["segments"][0]["start"], 0.0)
        self.assertGreater(data["segments"][-1]["end"], duration - 1.5)

    def test_transcriptions_language_and_prompt(self):
        clip = self.clips[0]
        status, _, body, _ = post("/v1/audio/transcriptions",
                                  {"model": MODEL, "language": "ru", "prompt": "Transcribe the audio."},
                                  f"fleurs-{clip['id']}.wav", clip["bytes"])
        self.assertEqual(status, 200)
        self.assert_transcript(json.loads(body)["text"], clip["text"])

    def test_transcriptions_srt_and_vtt(self):
        clip = self.clips[0]
        for fmt in ("srt", "vtt"):
            with self.subTest(fmt=fmt):
                status, ctype, body, elapsed = post("/v1/audio/transcriptions",
                                                    {"model": MODEL, "response_format": fmt},
                                                    f"fleurs-{clip['id']}.wav", clip["bytes"])
                print(f"\n{fmt} id={clip['id']} ({elapsed:.1f} s):\n{body}", flush=True)
                self.assertEqual(status, 200)
                if fmt == "srt":
                    self.assertTrue(body.startswith("1\n00:00:00,000 --> "))
                else:
                    self.assertTrue(body.startswith("WEBVTT\n\n00:00:00.000 --> "))
                self.assert_transcript(cue_text(body), clip["text"])

    def test_translations(self):
        clip = self.clips[0]
        status, ctype, body, elapsed = post("/v1/audio/translations",
                                           {"model": MODEL, "response_format": "json"},
                                           f"fleurs-{clip['id']}.wav", clip["bytes"])
        print(f"\ntranslations id={clip['id']} ({elapsed:.1f} s): {body}", flush=True)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["text"].strip())

    def test_a_second_sample_through_verbose_json(self):
        clip = self.clips[1]
        duration = round(clip["frames"] / clip["rate"], 3)
        status, _, body, elapsed = post("/v1/audio/transcriptions",
                                        {"model": MODEL, "response_format": "verbose_json"},
                                        f"fleurs-{clip['id']}.wav", clip["bytes"])
        print(f"\nsecond id={clip['id']} verbose_json ({elapsed:.1f} s): {body}", flush=True)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assert_transcript(data["text"], clip["text"])
        self.assertAlmostEqual(data["duration"], duration, places=3)


if __name__ == "__main__":
    unittest.main()
