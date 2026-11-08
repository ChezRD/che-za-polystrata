"""The OpenAI audio endpoints' form parsing, instructions and response formats (serve/transcribe.py)."""
import io
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from serve import audio as A
from serve import transcribe as T


def wav_bytes(seconds: float = 0.2, rate: int = 16000, amplitude: int = 0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        sample = int(amplitude).to_bytes(2, "little", signed=True)
        w.writeframes(sample * int(seconds * rate))
    return buf.getvalue()


def form(fields: dict, files: dict) -> tuple[str, bytes]:
    """A multipart/form-data body as curl sends it."""
    boundary = "----strataTestBoundary"
    out = bytearray()
    for name, value in fields.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()
    for name, (filename, data) in files.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(out)


class Multipart(unittest.TestCase):
    def test_fields_and_file(self):
        data = wav_bytes()
        ctype, body = form({"model": "gigachat-audio-10b-a1.8b", "response_format": "text",
                            "timestamp_granularities[]": "segment"},
                           {"file": ("sample0.wav", data)})
        fields, files = T.parse_multipart(ctype, body)
        self.assertEqual(fields, {"model": "gigachat-audio-10b-a1.8b", "response_format": "text",
                                  "timestamp_granularities[]": "segment"})
        self.assertEqual(files["file"], ("sample0.wav", data))

    def test_not_multipart(self):
        with self.assertRaisesRegex(ValueError, "multipart/form-data"):
            T.parse_multipart("application/json", b"{}")


class Instructions(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(T.instruction("transcribe"), "Transcribe the audio.")
        self.assertEqual(T.instruction("transcribe", "ru"), "Transcribe the audio in Russian.")
        self.assertEqual(T.instruction("translate"), "Translate the audio into English.")

    def test_translate_targets(self):
        self.assertEqual(T.instruction("translate", None, None, "de"), "Translate the audio into German.")
        self.assertEqual(T.instruction("translate", "ru", None, None),
                         "Translate the audio from Russian into English.")
        self.assertEqual(T.instruction("translate", "ru", None, "de"),
                         "Translate the audio from Russian into German.")
        self.assertEqual(T.instruction("translate", "kk", None, "ru"),
                         "Translate the audio from Kazakh into Russian.")
        self.assertEqual(T.instruction("transcribe", "de"), "Transcribe the audio in German.")

    def test_language_validation(self):
        self.assertIsNone(T.check_language(None))
        self.assertIsNone(T.check_language(""))
        self.assertEqual(T.check_language("RU"), "ru")
        self.assertEqual(T.check_language("uz"), "uz")
        for bad in ("russian", "r1", "e", "en-US"):
            with self.assertRaisesRegex(ValueError, "ISO-639-1"):
                T.check_language(bad)
        with self.assertRaisesRegex(ValueError, "target_language"):
            T.check_language("russian", "target_language")

    def test_max_tokens(self):
        self.assertEqual(T.parse_max_tokens(None), 0)
        self.assertEqual(T.parse_max_tokens(""), 0)
        self.assertEqual(T.parse_max_tokens("0"), 0)
        self.assertEqual(T.parse_max_tokens("4096"), 4096)
        for bad in ("-1", "abc", "40000"):
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                T.parse_max_tokens(bad)

    def test_prompt_wins(self):
        self.assertEqual(T.instruction("transcribe", "ru", "Write every word."), "Write every word.")
        self.assertEqual(T.instruction("translate", "ru", "Write every word.", "de"), "Write every word.")

    def test_timed_prompt_keeps_the_language_sentence(self):
        self.assertEqual(T.instruction("transcribe", "ru", timestamps=True),
                         "Transcribe the audio in Russian. Start each line with its beginning time as HH:MM:SS.")
        self.assertEqual(T.instruction("translate", None, None, "de", timestamps=True),
                         "Translate the audio into German. Start each line with its beginning time as HH:MM:SS.")
        self.assertEqual(T.instruction("transcribe", timestamps=False), "Transcribe the audio.")

    def test_timestamps_are_opt_in(self):
        self.assertIsNone(T.parse_timestamps(None))                # no times by default: the text alone
        self.assertIsNone(T.parse_timestamps(""))
        self.assertIs(T.parse_timestamps("true"), True)
        self.assertIs(T.parse_timestamps("0"), False)
        for bad in ("maybe", "2"):
            with self.assertRaisesRegex(ValueError, "timestamps"):
                T.parse_timestamps(bad)

    def test_when_cues_are_wanted(self):
        self.assertFalse(T.wants_cues(None, "json", []))
        self.assertFalse(T.wants_cues(None, "verbose_json", []))
        self.assertFalse(T.wants_cues(False, "verbose_json", []))
        self.assertTrue(T.wants_cues(True, "json", []))
        self.assertTrue(T.wants_cues(None, "srt", []))              # a subtitle format implies times
        self.assertTrue(T.wants_cues(None, "vtt", []))
        self.assertTrue(T.wants_cues(None, "verbose_json", ["segment"]))
        self.assertFalse(T.wants_cues(False, "srt", []))            # an explicit false still yields text

    def test_model_timestamps_are_a_separate_opt_in(self):
        self.assertFalse(T.parse_model_timestamps(None))
        self.assertTrue(T.parse_model_timestamps("true"))
        with self.assertRaisesRegex(ValueError, "model_timestamps"):
            T.parse_model_timestamps("maybe")


class CleanAnswer(unittest.TestCase):
    def test_refusal_and_markdown_are_not_transcripts(self):
        self.assertEqual(T.clean_answer("I cannot fulfill this request."), "")
        self.assertEqual(T.clean_answer("## Transcription of Audio The audio contains"), "")
        self.assertEqual(T.clean_answer("[Музыка]"), "")
        self.assertEqual(T.clean_answer("И поэтому использовать их."), "И поэтому использовать их.")

    def test_decoding_loop_is_cut(self):
        loop = " — А ты? — А я?" * 30
        out = T.clean_answer(loop)
        self.assertLessEqual(len(out.split()), 14)
        self.assertIn("А ты?", out)
        real = "да да да да мы это сделали"
        self.assertEqual(T.clean_answer(real), real)          # a short natural repeat stays

    def test_silence_gate(self):
        with tempfile.TemporaryDirectory() as d:
            silent, speech = Path(d) / "s.wav", Path(d) / "l.wav"
            silent.write_bytes(wav_bytes(0.5, amplitude=0))
            with wave.open(str(speech), "wb") as w:          # speech-like: loud 20 ms frames, quiet 20 ms frames
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                loud = (8000).to_bytes(2, "little", signed=True) * 320
                quiet = (300).to_bytes(2, "little", signed=True) * 320
                w.writeframes((loud + quiet) * 12)
            self.assertTrue(T.is_silent(silent))
            self.assertTrue(T.is_silent(Path(d) / "s.wav"))
            self.assertFalse(T.is_silent(speech))
            self.assertEqual(T.wav_peak(silent), 0.0)
            self.assertGreater(T.wav_peak(speech), 0.1)


class Upload(unittest.TestCase):
    def test_wav_goes_in_as_it_is(self):
        data = wav_bytes(0.5)
        with tempfile.TemporaryDirectory() as d:
            path = T.upload_to_wav(data, "sample.wav", Path(d))
            self.assertEqual(path.read_bytes(), data)
            self.assertAlmostEqual(A.wav_seconds(path), 0.5)

    def test_non_wav_without_ffmpeg_is_a_clear_400(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("serve.transcribe.shutil.which", return_value=None), \
                self.assertRaisesRegex(ValueError, "ffmpeg"):
            T.upload_to_wav(b"ID3not-really-mp3", "a.mp3", Path(d))

    def test_non_wav_with_ffmpeg(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.wav"
            src.write_bytes(wav_bytes(0.3))
            mp3 = Path(d) / "a.mp3"
            import subprocess
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), str(mp3)], check=True)
            path = T.upload_to_wav(mp3.read_bytes(), "a.mp3", Path(d))
            self.assertTrue(T.is_wav(path.read_bytes()))
            self.assertAlmostEqual(A.wav_seconds(path), 0.3, delta=0.1)


class Formats(unittest.TestCase):
    STAMPED = ("00:00:00 И поэтому использовать их в повседневности не получается. "
               "00:00:04 Мы вынуждены поступать зачастую интуитивно.")
    PLAIN = "И поэтому использовать их в повседневности не получается. Мы вынуждены поступать зачастую интуитивно."

    def test_stamps_are_stripped_for_json_and_text(self):
        self.assertEqual(T.plain_text(self.STAMPED), self.PLAIN)

    def test_no_stamp_is_one_cue(self):
        self.assertEqual(T.cues(self.PLAIN, 8.12), [(0.0, 8.12, self.PLAIN)])

    def test_stamps_are_segments(self):
        cues = T.cues(self.STAMPED, 8.12)
        self.assertEqual([(round(a, 2), round(b, 2)) for a, b, _ in cues], [(0.0, 4.0), (4.0, 8.12)])
        self.assertEqual(cues[0][2], "И поэтому использовать их в повседневности не получается.")

    def test_minutes_seconds_stamps_with_a_dash(self):
        raw = "0:00 — И поэтому использовать их. 0:27 — Человек недавно стал. 1:00: Вот так."
        self.assertEqual(T.plain_text(raw), "И поэтому использовать их. Человек недавно стал. Вот так.")
        cues = T.cues(raw, 83.3)
        self.assertEqual([(a, b) for a, b, _ in cues], [(0.0, 27.0), (27.0, 60.0), (60.0, 83.3)])
        self.assertEqual(cues[1][2], "Человек недавно стал.")
        self.assertEqual(cues[2][2], "Вот так.")

    def test_hours_minutes_seconds_stamps(self):
        cues = T.cues("00:01:05 hello 00:02:10 world", 200.0)
        self.assertEqual([(a, b) for a, b, _ in cues], [(65.0, 130.0), (130.0, 200.0)])
        self.assertEqual([c[2] for c in cues], ["hello", "world"])

    def test_verbose_json(self):
        body = T.verbose_json("transcribe", self.STAMPED, 8.121, None)
        self.assertEqual((body["task"], body["language"], body["duration"]), ("transcribe", "ru", 8.121))
        self.assertNotIn("target_language", body)
        self.assertEqual(body["text"], self.PLAIN)
        self.assertEqual([(s["id"], s["start"], s["end"]) for s in body["segments"]],
                         [(0, 0.0, 4.0), (1, 4.0, 8.121)])

    def test_verbose_json_target_language(self):
        body = T.verbose_json("translate", self.STAMPED, 8.121, "ru", target_language="de")
        self.assertEqual((body["task"], body["language"], body["target_language"]), ("translate", "ru", "de"))
        body = T.verbose_json("translate", self.STAMPED, 8.121, None, target_language=None)
        self.assertEqual(body["target_language"], "en")

    def test_srt(self):
        self.assertEqual(T.srt(self.STAMPED, 8.12).splitlines()[:4],
                         ["1", "00:00:00,000 --> 00:00:04,000",
                          "И поэтому использовать их в повседневности не получается.", ""])

    def test_vtt(self):
        out = T.vtt(self.PLAIN, 8.12)
        self.assertTrue(out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:08.120\n"))
        self.assertTrue(out.endswith("интуитивно.\n"))

    def test_language(self):
        self.assertEqual(T.language_of(self.PLAIN, None), "ru")
        self.assertEqual(T.language_of("Hello there.", None), "en")
        self.assertEqual(T.language_of(self.PLAIN, "de"), "de")

    def test_temperature(self):
        self.assertEqual(T.parse_temperature(None), 0.0)
        self.assertEqual(T.parse_temperature("0.2"), 0.2)
        with self.assertRaisesRegex(ValueError, "0 to 1"):
            T.parse_temperature("2")


if __name__ == "__main__":
    unittest.main()
