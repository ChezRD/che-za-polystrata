"""Audio input parsing and the audio-token expansion (the GigaChat audio decoder's prompt side).

`expand_audio_tokens` mirrors the reference processor's `_audio_block`/`expand_audio_tokens`: `n` audio rows where
one `[audio_token]` stood, an `HH:MM:SS` stamp every 60 s at `secs*6.25 + 8j`, and the duration's stamp after the
block.  The stamps split a clip's rows into contiguous runs, which is what serve/audio.py's `split_records` writes
for the engine's one-record-per-run reader.
"""
import base64
import io
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from serve import audio as A
from serve.frontend import ChatTemplate, anthropic_to_messages, audio_wav, audios_of, openai_to_messages
from serve.server import MockEngine, Service, load_tokenizer


def _env_path(name: str) -> Path:
    raw = os.environ.get(name, "")
    return Path(raw) if raw else Path("/no/such/strata/model")


TOKENIZER = _env_path("STRATA_GIGACHAT_TOKENIZER")
BINARY = Path(__file__).resolve().parents[1] / "engine-poly" / "strata-audio"
MODEL = _env_path("STRATA_GIGACHAT_ENCODER")
PAD = 128006
LUT = {**{d: 900 + d for d in range(10)}, 10: 910}      # a stand-in for the tokenizer's digits and colon


def real_tokenizer():
    if not (TOKENIZER / "vocab.json").exists():
        return None
    try:
        return load_tokenizer(TOKENIZER)
    except ImportError:                                 # `regex`, only in the server's environment
        return None


def wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class RawAudioParts(unittest.TestCase):
    def test_openai_input_audio(self):
        data = base64.b64encode(wav_bytes(0.1)).decode()
        messages, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
            {"type": "text", "text": "Transcribe the audio."},
        ]}]})
        items = audios_of(messages)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["format"], "wav")
        self.assertTrue(items[0]["source"].startswith("data:audio/wav;base64,"))
        # prompt order: the clip where it stood, then the text
        self.assertEqual([p["type"] for p in messages[0]["content"]], ["audio", "text"])

    def test_anthropic_audio_block(self):
        data = base64.b64encode(b"ID3notreallymp3").decode()
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            {"type": "audio", "source": {"type": "base64", "format": "mp3", "data": data}},
            {"type": "text", "text": "What is this?"},
        ]}]})
        items = audios_of(messages)
        self.assertEqual([i["format"] for i in items], ["mp3"])
        self.assertTrue(items[0]["source"].startswith("data:audio/mp3;base64,"))

    def test_local_path_and_named_source(self):
        messages, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "audio", "path": "/tmp/a.wav", "name": "a.wav"},
        ]}]})
        item = audios_of(messages)[0]
        self.assertEqual((item["source"], item["format"], item["name"]), ("/tmp/a.wav", "wav", "a.wav"))


class ToWav(unittest.TestCase):
    def test_wav_passes_through(self):
        data = wav_bytes(0.5)
        part = {"type": "audio", "source": "data:audio/wav;base64," + base64.b64encode(data).decode()}
        with tempfile.TemporaryDirectory() as d:
            path = audio_wav(part, Path(d))
            self.assertEqual(path.read_bytes(), data)

    def test_declared_wav_that_is_not(self):
        part = {"type": "audio", "format": "wav", "source": "data:audio/wav;base64," + base64.b64encode(b"nope").decode()}
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, "RIFF"):
            audio_wav(part, Path(d))

    def test_other_format_without_ffmpeg(self):
        part = {"type": "audio", "format": "mp3", "source": "data:audio/mp3;base64," + base64.b64encode(b"ID3").decode()}
        with tempfile.TemporaryDirectory() as d, mock.patch("serve.frontend.shutil.which", return_value=None), \
                self.assertRaisesRegex(ValueError, "ffmpeg"):
            audio_wav(part, Path(d))

    def test_bytes_are_converted_with_ffmpeg_when_present(self):
        import shutil
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed")
        raw = wav_bytes(0.2)
        part = {"type": "audio", "format": "wav", "source": "data:audio/wav;base64," + base64.b64encode(raw).decode()}
        with tempfile.TemporaryDirectory() as d:
            path = audio_wav(part, Path(d))
            with wave.open(str(path), "rb") as w:
                self.assertEqual(w.getnframes(), int(0.2 * 16000))


class Timestamps(unittest.TestCase):
    def test_hms(self):
        self.assertEqual(A.hms_ids(0, LUT), [900, 900, 910, 900, 900, 910, 900, 900])
        self.assertEqual(A.hms_ids(3661, LUT), [900, 901, 910, 900, 901, 910, 900, 901])   # 01:01:01
        with self.assertRaisesRegex(ValueError, "100 hours"):
            A.hms_ids(100 * 3600, LUT)

    def test_anchor_seconds(self):
        self.assertEqual(A.anchor_seconds(10), [0])
        self.assertEqual(A.anchor_seconds(120), [0, 60])
        self.assertEqual(A.anchor_seconds(240, ["60s", "30s"]), [0, 30, 60, 90, 120, 150, 180, 210])
        self.assertEqual(A.anchor_seconds(7200), list(range(0, 7200, 60)))

    def test_ten_seconds(self):
        # 10 s = 63 rows (tools/audio/README.md): 00:00:00, 63 pads, then the duration 00:00:09
        block, runs = A.audio_block(63, PAD, A.anchor_seconds(10), 9, LUT)
        self.assertEqual(len(block), 63 + 8 + 8)
        self.assertEqual(block[:8], A.hms_ids(0, LUT))
        self.assertEqual(block[8:71], [PAD] * 63)
        self.assertEqual(block[71:], A.hms_ids(9, LUT))
        self.assertEqual(runs, [(0, 63)])

    def test_thirty_seconds(self):
        block, runs = A.audio_block(188, PAD, A.anchor_seconds(30), 29, LUT)
        self.assertEqual(len(block), 188 + 8 + 8)
        self.assertEqual(runs, [(0, 188)])

    def test_two_minutes_splits_at_sixty(self):
        # 120 s = 750 rows: 00:00:00, 375 pads, 00:01:00 at token 60*6.25 + 8 = 383, 375 pads, duration
        block, runs = A.audio_block(750, PAD, A.anchor_seconds(120), 120, LUT)
        self.assertEqual(len(block), 750 + 16 + 8)
        self.assertEqual(block[:8], A.hms_ids(0, LUT))
        self.assertEqual(block[383:391], A.hms_ids(60, LUT))
        self.assertEqual(runs, [(0, 375), (375, 375)])
        self.assertEqual(block[-8:], A.hms_ids(120, LUT))

    def test_expand_in_prompt_order(self):
        ids = [1, 2, PAD, 3, PAD, 4]
        out, runs = A.expand_audio_tokens(ids, PAD, [63, 188], [9, 29], LUT)
        self.assertEqual(out[:2], [1, 2])
        self.assertEqual(out[2:10], A.hms_ids(0, LUT))
        self.assertEqual(out[10:73], [PAD] * 63)
        self.assertEqual(out[73:81], A.hms_ids(9, LUT))
        self.assertEqual(out[81], 3)
        self.assertEqual(runs, [[(0, 63)], [(0, 188)]])
        self.assertEqual(out[-1], 4)

    def test_mismatched_counts_are_a_400(self):
        with self.assertRaisesRegex(ValueError, "audio place"):
            A.expand_audio_tokens([PAD, 2], PAD, [63, 63], [9, 9], LUT)

    def test_records_are_split_at_the_stamps(self):
        """split_records writes one record per run, and their rows are the clip's rows in order."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "clip.sve1"
            rows = [[float(i), 1.5, -2.0] for i in range(750)]
            body = b"".join(struct.pack("<3f", *r) for r in rows)
            path.write_bytes(struct.pack("<5i", A.SVE1, 750, 750, 1, 3) + body)
            _, runs = A.audio_block(750, PAD, A.anchor_seconds(120), 120, LUT)
            data = A.split_records(path, runs)
            self.assertEqual(len(data), (20 + 375 * 3 * 4) * 2)
            header = 20 + 375 * 12
            first, second = data[:20], data[header:header + 20]
            self.assertEqual(struct.unpack("<5i", first), (A.SVE1, 375, 375, 1, 3))
            self.assertEqual(struct.unpack("<5i", second), (A.SVE1, 375, 375, 1, 3))
            self.assertEqual(struct.unpack("<3f", data[20:32]), (0.0, 1.5, -2.0))
            self.assertEqual(struct.unpack("<3f", data[20 + 375 * 12 - 12:20 + 375 * 12]), (374.0, 1.5, -2.0))


class ServiceAudio(unittest.TestCase):
    """prepare() with the model's real template and tokenizer: the clip's block in the prompt and the request's
    combined SVE1 file, cut where a stamp interrupts the rows."""

    def setUp(self):
        self.tok = real_tokenizer()
        if self.tok is None:
            self.skipTest("the GigaChat audio tokenizer (or `regex`) is not available")
        self.template = ChatTemplate(TOKENIZER / "chat_template.jinja")
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    class FakeAudio:
        dir = None
        rows = 63

        def encode_all(self, paths, cancel=None):
            out = []
            for k, _ in enumerate(paths):
                f = Path(self.dir) / f"clip{k}.sve1"
                f.write_bytes(struct.pack("<5i", A.SVE1, self.rows, self.rows, 1, 1536) + b"\0" * (self.rows * 6144))
                out.append((f, self.rows, 0, 9970, 100))
            return out

    def service(self, rows=63, audio=True):
        fake = self.FakeAudio()
        fake.dir, fake.rows = self.dir, rows
        svc = Service(MockEngine(self.tok, "ok"), self.tok, self.template, audio=fake if audio else None)
        self.addCleanup(svc.drop_embeddings)
        return svc

    def request(self):
        data = base64.b64encode(wav_bytes(10)).decode()
        return openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
            {"type": "text", "text": "Transcribe the audio."},
        ]}]})

    def test_prepare_expands_the_audio_token(self):
        svc = self.service()
        messages, tools, kwargs = self.request()
        ids, thinking, _ = svc.prepare(messages, tools, kwargs, max_new=16)
        # The template wrote one [audio_token]; the block replaced it with 63 rows, 00:00:00 and the duration.
        self.assertEqual(ids.count(PAD), 63)
        self.assertFalse(thinking)      # the audio model has no reasoning: the transcript is the answer
        with open(svc.embeddings.path, "rb") as f:
            magic, n, nx, ny, n_embd = struct.unpack("<5i", f.read(20))
        self.assertEqual((magic, n, nx, ny, n_embd), (A.SVE1, 63, 63, 1, 1536))
        self.assertEqual(svc.embeddings.kind, "audio")
        self.assertEqual(svc.embeddings.pad, PAD)

        # The template's file form for the clip: the envelope and one [audio_token].
        form, _ = svc.audio_messages(messages, audios_of(messages))
        single = self.template.render(form, tools=None)
        self.assertIn('<|file|>{"id": "audio_0", "name": "audio_0.wav", "type": "audio"}[audio_token]<|/file|>',
                      single)

    def test_a_stamp_splits_the_request_record(self):
        svc = self.service(rows=750)
        messages, tools, kwargs = self.request()
        ids, _, _ = svc.prepare(messages, tools, kwargs, max_new=16)
        self.assertEqual(ids.count(PAD), 750)
        with open(svc.embeddings.path, "rb") as f:
            records = []
            while (hdr := f.read(20)):
                magic, n, nx, ny, n_embd = struct.unpack("<5i", hdr)
                self.assertEqual((magic, nx, ny, n_embd), (A.SVE1, n, 1, 1536))
                records.append(n)
                f.seek(n * n_embd * 4, 1)
        self.assertEqual(records, [375, 375])

    def test_no_encoder_is_a_clear_error(self):
        svc = self.service(audio=False)
        messages, tools, kwargs = self.request()
        with self.assertRaisesRegex(ValueError, "without the audio encoder"):
            svc.prepare(messages, tools, kwargs, max_new=16)

    def test_images_and_audio_together_are_refused(self):
        svc = self.service()
        messages, tools, kwargs = self.request()
        messages[0]["content"].insert(0, {"type": "image", "source": "x.png"})
        with self.assertRaisesRegex(ValueError, "both images and audio"):
            svc.prepare(messages, tools, kwargs, max_new=16)

    def test_audio_messages_writes_the_template_file_form(self):
        svc = self.service()
        messages, _, _ = self.request()
        out, names = svc.audio_messages(messages, audios_of(messages))
        self.assertEqual(names, ["audio_0.wav"])
        self.assertEqual(out[0]["content"], "<file_placeholder_audio_0>Transcribe the audio.")
        self.assertEqual(out[0]["embed_files"], [{"id": "audio_0", "name": "audio_0.wav", "type": "audio"}])
        self.assertTrue(out[0]["file_placeholders"])


class Stats(unittest.TestCase):
    """The `audio` block of /v1/status and the facts the Prometheus families are built from."""

    def encoder(self):
        enc = A.Audio({"exe": "/nonexistent/strata-audio", "model": "/models/gigachat-audio-enc-mixed.gguf"})
        self.addCleanup(enc.shutdown)
        return enc

    def test_idle_before_the_first_clip(self):
        s = self.encoder().status()
        self.assertEqual((s["enabled"], s["encoder"], s["model"]), (True, "idle", "gigachat-audio-enc-mixed.gguf"))
        self.assertEqual(s["requests"], {"transcribe": 0, "translate": 0, "errors": 0})
        self.assertEqual((s["audio_seconds"], s["rows"], s["queue"]), (0, 0, 0))
        self.assertEqual(s["latency_ms"], {"p50": None, "p95": None})
        self.assertIsNone(s["last"])

    def test_counters_last_and_percentiles(self):
        enc = self.encoder()
        enc.last_batch = {"device": "cpu", "encode_ms": 800.0, "rows": 51, "chunks": 1, "mel_ms": 20.0}
        last = enc.note("transcribe", "ru", 8.12, 51, 1200.0, prefill_ms=300.0, decode_ms=100.0)
        self.assertEqual((last["device"], last["encode_ms"], last["prefill_ms"], last["decode_ms"]), ("cpu", 800, 300, 100))
        self.assertEqual((last["rtf"], last["total_ms"]), (0.099, 1200))
        self.assertNotIn("gpu_util_pct", last)                  # the CPU path reports no GPU numbers
        enc.note("translate", None, 11.87, 75, 2400.0, prefill_ms=600.0, decode_ms=400.0)
        enc.note_error()
        s = enc.status()
        self.assertEqual(s["requests"], {"transcribe": 1, "translate": 1, "errors": 1})
        self.assertEqual((s["audio_seconds"], s["rows"]), (19.99, 126))
        self.assertEqual((s["latency_ms"]["p50"], s["latency_ms"]["p95"]), (1200, 2400))
        self.assertEqual(s["device"], "cpu")
        self.assertFalse(s["encoding"])
        self.assertNotIn("gpu_util_pct", s)                     # and not in the status block either
        last = s["last"]
        self.assertEqual((last["task"], last["language"], last["duration_s"], last["rows"]), ("translate", None, 11.87, 75))
        self.assertEqual((last["encode_ms"], last["prefill_ms"], last["decode_ms"]), (800, 600, 400))
        self.assertEqual(last["at"], s["last"]["at"])
        self.assertEqual((s["encode_ms_total"], s["encode_count"]), (1600.0, 2))

    def test_gpu_status_reports_sampled_numbers_only(self):
        enc = A.Audio({"exe": "/nonexistent/strata-audio-cuda", "model": "/models/enc.gguf", "gpu": True,
                       "device": 0, "exe_gpu": "/nonexistent/strata-audio-cuda"})
        self.addCleanup(enc.shutdown)
        self.assertEqual(enc.device_label, "cuda:0")
        self.assertEqual(enc.spawn[0][0], "/nonexistent/strata-audio-cuda")
        self.assertIn("--gpu", enc.spawn[0])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", enc.spawn[2])   # the card stays visible to the encoder
        enc.last_batch = {"device": "cuda:0", "encode_ms": 3492.0, "rows": 3091, "chunks": 8,
                          "gpu_util_pct": 87, "vram_mib": 1232, "workspace_mib": 463}
        enc.note("transcribe", "ru", 494.51, 3091, 30920.0, prefill_ms=6300.0, decode_ms=20100.0)
        s = enc.status()
        self.assertEqual((s["device"], s["gpu_util_pct"], s["vram_mib"], s["workspace_mib"]), ("cuda:0", 87, 1232, 463))
        self.assertEqual((s["last"]["rtf"], s["last"]["chunks"]), (0.007, 8))

    def test_cpu_config_hides_the_card(self):
        enc = A.Audio({"exe": "/nonexistent/strata-audio", "model": "/models/enc.gguf"})
        self.addCleanup(enc.shutdown)
        self.assertEqual(enc.device_label, "cpu")
        self.assertEqual(enc.spawn[2]["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertNotIn("--gpu", enc.spawn[0])


@unittest.skipUnless(BINARY.exists() and MODEL.exists(), "strata-audio or its GGUF is not here")
class Encoder(unittest.TestCase):
    """The resident helper over its real protocol: READY, one clip, the cache, QUIT."""

    def test_one_clip_and_the_cache(self):
        enc = A.Audio({"exe": str(BINARY), "model": str(MODEL), "threads": 4})
        self.addCleanup(enc.shutdown)
        self.assertFalse(enc.alive())                    # lazy: nothing starts at load
        with tempfile.TemporaryDirectory() as d:
            one = Path(d) / "one.wav"
            one.write_bytes(wav_bytes(1.0))
            rows, t0, t1, ms = enc.encode(one)
            self.assertEqual((rows, t0, t1), (7, 0, 1000))
            self.assertGreaterEqual(ms, 0)
            self.assertTrue(enc.alive())
            again = enc.encode(one)                      # the cache: the same answer, no second encode
            self.assertEqual(again, (rows, t0, t1, ms))
            self.assertEqual(len(enc.cache), 1)


class PhaseProgress(unittest.TestCase):
    """activity_view(): the phase's real counters, None where there is no known end, and nothing left over from
    the previous request."""

    def view(self, phase, status=None, audio=None, progress=None):
        import threading
        import time as _t
        import types
        stub = types.SimpleNamespace(status=status or {}, phase=phase, phase_at=_t.time() - 12.5,
                                     status_lock=threading.Lock(), audio=audio,
                                     engine=types.SimpleNamespace(progress=progress))
        return Service.activity_view(stub)

    class AudioStub:
        def __init__(self, **kw):
            self.live = kw

        def progress(self):
            return dict(self.live)

    def test_encode_counts_rows_and_reports_gpu(self):
        v = self.view("encode", audio=self.AudioStub(encoding=True, rows_done=1200, rows_total=3091,
                                                     gpu_util_pct=87, vram_mib=1232))
        self.assertEqual((v["phase"], v["progress_label"]), ("encode", "Encoding audio"))
        self.assertAlmostEqual(v["progress"], 0.3882, places=4)
        self.assertEqual(v["progress_detail"], "1200 / 3091 rows, GPU 87%")
        self.assertEqual((v["rows_done"], v["rows_total"]), (1200, 3091))
        self.assertIsNone(v["prompt_read"])                 # the decoder has not started: no prefill number
        self.assertIsNone(v["prompt_total"])
        self.assertEqual(v["phase_elapsed_s"], 12.5)

    def test_encode_never_reports_full_while_running(self):
        v = self.view("encode", audio=self.AudioStub(encoding=True, rows_done=3091, rows_total=3091))
        self.assertEqual(v["progress"], 0.999)

    def test_prefill_uses_the_requests_prompt(self):
        v = self.view("prefill", status={"prompt_tokens": 4199}, progress=(2100, 4199))
        self.assertEqual((v["progress_label"], v["progress_detail"]),
                         ("Prefill (after audio)", "2100 / 4199 tokens"))
        self.assertAlmostEqual(v["progress"], 0.5001, places=4)
        self.assertIsNone(v["rows_done"])                   # the encoder's counters do not leak into prefill

    def test_prefill_without_engine_counters_starts_at_zero(self):
        v = self.view("prefill", status={"prompt_tokens": 300}, progress=None)
        self.assertEqual((v["prompt_read"], v["prompt_total"], v["progress"]), (0, 300, 0.0))

    def test_generate_needs_a_cap_the_client_asked_for(self):
        v = self.view("generate", status={"generated": 350, "max_tokens": 1714, "max_tokens_requested": True})
        self.assertEqual((v["progress_label"], v["progress_detail"]), ("Generating", "350 / 1714 tokens"))
        self.assertAlmostEqual(v["progress"], 0.2042, places=4)
        v = self.view("generate", status={"generated": 350, "max_tokens": 48047})   # the context ceiling, not a cap
        self.assertIsNone(v["progress"])                    # a transcript has no length until it ends
        self.assertEqual(v["progress_detail"], "350 tokens")
        v = self.view("generate", status={"generated": 350, "max_tokens": 0})
        self.assertIsNone(v["progress"])                    # indeterminate, not 0
        self.assertEqual(v["progress_detail"], "350 tokens")

    def test_completed_keeps_no_counters(self):
        v = self.view("completed", status={"generated": 350, "max_tokens": 1714})
        self.assertEqual((v["phase"], v["progress"], v["generated"], v["max_tokens"]), ("completed", None, None, None))


if __name__ == "__main__":
    unittest.main()
