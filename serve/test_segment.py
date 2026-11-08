"""serve/segment.py - fixed overlapping parts, fuzzy boundary stitching and global cues (no model, no GPU)."""
from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from serve import segment as S
from serve import transcribe as T


def wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    w = wave.open(str(path), "wb")
    try:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    finally:
        w.close()
    return path


class Plan(unittest.TestCase):
    def test_fixed_parts_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p = wav(Path(d) / "a.wav", 900)
            parts = S.plan(p, 300, 15)
        self.assertEqual(len(parts), 3)
        self.assertEqual((parts[0].start, parts[0].end), (0.0, 300.0))
        self.assertAlmostEqual(parts[1].start, 285.0, delta=0.16)
        self.assertAlmostEqual(parts[1].end, 585.0, delta=0.16)
        self.assertAlmostEqual(parts[2].start, 570.0, delta=0.16)
        self.assertAlmostEqual(parts[2].end, 900.0)
        for a, b in zip(parts, parts[1:]):
            self.assertAlmostEqual(b.start, a.end - 15.0, delta=0.16)
        for p in parts:
            self.assertLessEqual(p.seconds, S.HARD_CAP + 1e-6)

    def test_grid_and_hard_cap(self):
        with tempfile.TemporaryDirectory() as d:
            p = wav(Path(d) / "a.wav", 914.72)
            parts = S.plan(p, 300, 15)
        for part in parts:
            self.assertLessEqual(part.seconds, S.HARD_CAP + 1e-6)
            self.assertAlmostEqual(part.start / S.GRID, round(part.start / S.GRID))
        self.assertAlmostEqual(parts[-1].end, 914.72, places=3)
        self.assertGreaterEqual(parts[-1].seconds, S.MIN_PART)

    def test_short_tail_joins_the_part_before(self):
        with tempfile.TemporaryDirectory() as d:
            p = wav(Path(d) / "a.wav", 915.12)              # a 60 s tail: one more boundary would cost a cut
            parts = S.plan(p, 300, 15)
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(parts[-1].end, 915.12, places=2)
        self.assertLessEqual(parts[-1].seconds, S.HARD_CAP + 1e-6)

    def test_short_audio_is_one_part(self):
        with tempfile.TemporaryDirectory() as d:
            p = wav(Path(d) / "a.wav", 100)
            parts = S.plan(p, 300, 15)
        self.assertEqual(len(parts), 1)
        self.assertEqual((parts[0].start, parts[0].end), (0.0, 100.0))

    def test_empty_audio(self):
        with tempfile.TemporaryDirectory() as d:
            p = wav(Path(d) / "a.wav", 0)
            self.assertEqual(S.plan(p, 300, 15), [])

    def test_validation(self):
        self.assertEqual(S.check_segment_seconds(None), 300.0)
        self.assertEqual(S.check_overlap_seconds(None, 300.0), 15.0)
        for bad in ("10", "400", "x"):
            with self.assertRaisesRegex(ValueError, "segment_seconds"):
                S.check_segment_seconds(bad)
        for bad in ("-1", "61", "300"):
            with self.assertRaisesRegex(ValueError, "overlap_seconds"):
                S.check_overlap_seconds(bad, 300.0)


class Stitch(unittest.TestCase):
    def test_exact_duplicate_overlap(self):
        text, audit = S.stitch(["a b c d e f", "d e f g h"], 15)
        self.assertEqual(text, "a b c d e f g h")
        self.assertEqual((audit[0]["method"], audit[0]["cut"]), ("words", 3))
        self.assertEqual(audit[0]["before"], 0)

    def test_fuzzy_asr_spelling(self):
        prev = "the quick brown fox jumps over the lazy dog"
        nxt = "brown fox jumpz over teh lazy dog and then some"
        text, audit = S.stitch([prev, nxt], 15)
        self.assertEqual(audit[0]["method"], "words")
        self.assertTrue(text.startswith(prev))
        for w in ("and", "then", "some"):
            self.assertIn(w, text.split())
        # the shared words appear once
        self.assertEqual(text.split().count("brown"), 1)

    def test_no_match_falls_back_and_warns(self):
        msgs = []
        text, audit = S.stitch(["alpha beta gamma delta", "completely unrelated words here"], 15, warn=msgs.append)
        self.assertIn(audit[0]["method"], ("duration", "characters", "none"))
        self.assertTrue(msgs)
        self.assertIn("alpha", text)
        self.assertIn("completely", text)

    def test_no_words_lost(self):
        prev = "one two three four five six seven eight"
        nxt = "five six seven eight nine ten eleven"
        text, audit = S.stitch([prev, nxt], 15)
        words = text.split()
        for w in prev.split() + ["nine", "ten", "eleven"]:
            self.assertIn(w, words)
        self.assertEqual(words.count("five"), 1)

    def test_interior_insertion_is_recovered(self):
        # "x" falls between the shared "b" and "c d e"; the previous part missed it and it belongs in the merge.
        text, audit = S.stitch(["a b c d e", "b x c d e f"], 15)
        self.assertEqual(text, "a b x c d e f")
        self.assertEqual(audit[0]["recovered"], 1)

    def test_farfield_paraphrased_overlap_cuts_on_a_short_tail(self):
        # Golos farfield: both parts say the same 15 s differently ("салют йоко" vs "салют фильм"), and only the last
        # seven words agree.  A run that reaches the end of the previous part is enough evidence to drop the head.
        prev = "раз два три салют йоко салют фильм седой мерфи как он искал невесту в америке"
        nxt = ("сбер выйти из навыка афина сделай звонок ларисе романовской салют фильм сэди мерфи как он искал "
               "невесту в америке афина лучший курс")
        text, audit = S.stitch([prev, nxt], 15)
        self.assertEqual(audit[0]["method"], "words")
        self.assertEqual(audit[0]["matched"], 7)
        self.assertEqual(text.split().count("мерфи"), 1)
        self.assertEqual(text.split()[-2:], ["лучший", "курс"])

    def test_leading_paraphrase_is_dropped(self):
        # "tool" is the next part's rendering of the shared audio the previous part called "old product"; keeping it
        # would say the same span twice.
        text, audit = S.stitch(["old product for logs and more", "tool for logs and more stuff"], 15)
        self.assertEqual(text, "old product for logs and more stuff")
        self.assertEqual(audit[0]["recovered"], 0)

    def test_punctuation_and_case(self):
        text, _ = S.stitch(["Hello, world!", "world! How are you?"], 15)
        self.assertEqual(text, "Hello, world! How are you?")


class Cues(unittest.TestCase):
    def test_shifted_and_deduplicated(self):
        parts = [S.Part(0.0, 300.0), S.Part(285.0, 585.0)]
        texts = ["00:00:01 first words here", "00:00:02 second words here"]
        cues = S.global_cues(texts, parts)
        self.assertEqual(cues[0][0], 0.0)          # a part's first stamp is not its start; the cue covers it
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a[1], b[0] + 1e-9)
        self.assertEqual(" ".join(c[2] for c in cues).split(),
                         "first words here second words here".split())

    def test_no_stamps_cues_per_part(self):
        parts = [S.Part(0.0, 300.0), S.Part(285.0, 585.0)]
        cues = S.global_cues(["first words", "second words"], parts)
        self.assertEqual(len(cues), 2)
        self.assertEqual((cues[0][0], cues[0][1]), (0.0, 300.0))
        self.assertEqual((cues[1][0], cues[1][1]), (300.0, 585.0))       # never before the cue it follows
        self.assertEqual(" ".join(c[2] for c in cues), "first words second words")

    def test_no_stamps_times_are_real_audio_time(self):
        from pathlib import Path                      # noqa: F401 - parts are enough, no audio read
        parts = [S.Part(0.0, 300.0), S.Part(285.0, 585.0), S.Part(570.0, 914.72)]
        texts = ["alpha beta gamma delta epsilon", "delta epsilon zeta eta theta",
                 "eta theta iota kappa lambda"]
        text, audit, kept = S.stitch_parts(texts, 15)
        cues = S.timed_cues(texts, parts, kept, audit)
        self.assertGreater(len(cues), 2)
        self.assertEqual(cues[0][0], 0.0)
        self.assertAlmostEqual(cues[-1][1], 914.72, places=3)
        self.assertEqual(" ".join(c[2] for c in cues).split(), text.split())
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a[1], b[0] + 1e-9)          # no cue overlaps the next
        self.assertTrue(all(b > a for a, b, _ in cues))       # and none has zero length

    def test_srt_vtt_use_global_times(self):
        cues = [(1.0, 2.0, "hello"), (287.0, 288.0, "world")]
        self.assertIn("00:00:01,000 --> 00:00:02,000", T.srt_cues(cues))
        self.assertIn("00:04:47.000 --> 00:04:48.000", T.vtt_cues(cues))


if __name__ == "__main__":
    unittest.main()
