"""serve/segment.py - splitting a long clip into fixed overlapping parts and stitching their transcripts.

The decoder ends its answer after roughly seven minutes of one clip (EOS, not the output cap), so a longer file is
cut into parts of `segment_seconds` (default 300 s, never over 360), each part sharing `overlap_seconds` (default
15) with the next, on the audio token's 160 ms grid.  The parts go through the existing one-at-a-time audio turn.

The work is at the boundaries: the tail of the previous transcript is aligned with the head of the next one at the
word level - normalized, fuzzy (an ASR word or two may differ) - and the duplicated head is dropped.  A confident
alignment drops exactly the shared words; otherwise a character or duration fallback cuts and logs.  Timestamps are
put on the file's clock two ways: the model's own stamps when it wrote them (shifted by each part's start and
de-overlapped), else its words spread over the audio the part really covers, in CUE_SECONDS cues.  The second way is
the default: the model's stamps drift by seconds and can reorder a boundary.

No model, no GPU: this module reads and writes WAV and edits text.
"""
from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

import serve.transcribe as transcribe_api

GRID = 0.160          # seconds of one audio token: every cut snaps to it
HARD_CAP = 360.0      # no part is longer than this
MIN_PART = 90.0       # a tail shorter than this joins the part before it (when that stays under HARD_CAP)
TAIL_WORDS = 140      # words of the previous transcript aligned at a boundary
HEAD_WORDS = 200      # words of the next transcript aligned at a boundary
CUE_SECONDS = 6.0    # a cue's target length when the model wrote no stamps of its own


@dataclass
class Part:
    start: float
    end: float
    snapped: bool = True

    @property
    def seconds(self) -> float:
        return self.end - self.start


def check_segment_seconds(value) -> float:
    """The optional `segment_seconds` form field: a part's target length, 30 to 360 s."""
    if value in (None, ""):
        return 300.0
    try:
        s = float(str(value).strip())
    except ValueError:
        raise ValueError(f"segment_seconds={value!r}: a number from 30 to 360") from None
    if not 30.0 <= s <= HARD_CAP:
        raise ValueError(f"segment_seconds={value!r}: a number from 30 to 360")
    return s


def check_overlap_seconds(value, segment_seconds: float) -> float:
    """The optional `overlap_seconds` form field: 0 to 60 s, and less than a part."""
    if value in (None, ""):
        return 15.0
    try:
        s = float(str(value).strip())
    except ValueError:
        raise ValueError(f"overlap_seconds={value!r}: a number from 0 to 60") from None
    if not 0.0 <= s <= 60.0:
        raise ValueError(f"overlap_seconds={value!r}: a number from 0 to 60")
    if s >= segment_seconds:
        raise ValueError(f"overlap_seconds={value!r} must be less than segment_seconds={segment_seconds:g}")
    return s


def _pcm(path: Path):
    w = wave.open(str(path), "rb")
    try:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes(), w.readframes(w.getnframes())
    finally:
        w.close()


def _snap(t: float, grid: float = GRID) -> float:
    return round(t / grid) * grid


def plan(wav: str | Path, segment_seconds: float = 300.0, overlap_seconds: float = 15.0,
         hard_cap: float = HARD_CAP, grid: float = GRID) -> list[Part]:
    """Fixed parts of `segment_seconds`, at most `hard_cap`, sharing `overlap_seconds`, cut on the 160 ms grid.  A
    file no longer than one part is one part; a tail shorter than MIN_PART joins the part before it."""
    _, _, rate, frames, _ = _pcm(Path(wav))
    duration = frames / float(rate) if rate else 0.0
    if duration <= 0:
        return []
    if duration <= segment_seconds:
        return [Part(0.0, duration)]
    parts: list[Part] = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + segment_seconds, duration)
        end = min(_snap(end, grid), start + hard_cap, duration)
        if duration - end < MIN_PART and duration - start <= hard_cap:
            end = duration
        parts.append(Part(_snap(start, grid), end))
        if end >= duration - 1e-6:
            break
        nxt = max(start + 1.0, _snap(end - overlap_seconds, grid))
        if nxt <= start:
            nxt = min(duration, start + grid)
        start = nxt
    return parts


def write_parts(wav: str | Path, parts: list[Part], directory: str | Path) -> list[Path]:
    """One WAV a part, at the part's exact frames."""
    channels, width, rate, frames, data = _pcm(Path(wav))
    out = []
    for i, p in enumerate(parts):
        path = Path(directory) / f"part-{i:03d}.wav"
        w = wave.open(str(path), "wb")
        try:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(rate)
            a = max(0, int(round(p.start * rate)))
            b = min(frames, int(round(p.end * rate)))
            w.writeframes(data[a * channels * width:b * channels * width])
        finally:
            w.close()
        out.append(path)
    return out


_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def normalize(word: str) -> str:
    return _STRIP.sub("", word.lower().replace("ё", "е"))


def _words(text: str) -> tuple[list[str], list[str]]:
    words = text.split()
    return words, [normalize(w) for w in words]


def _swap(a: str, b: str) -> bool:
    """One adjacent transposition apart ("the"/"teh")."""
    if len(a) != len(b) or len(a) < 3:
        return False
    d = [i for i in range(len(a)) if a[i] != b[i]]
    return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]


def _edit(a: str, b: str) -> int:
    """Levenshtein distance, two rows."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def same_word(a: str, b: str) -> bool:
    """Two normalized words the same, or one ASR slip apart (short words must agree exactly)."""
    if a == b:
        return True
    if not a or not b:
        return False
    n = min(len(a), len(b))
    if n >= 9:
        return _edit(a, b) <= 2
    if n >= 3:
        return _edit(a, b) <= 1 or _swap(a, b)
    return False


def _run(prev: list[str], nxt: list[str], i: int, j: int) -> tuple[int, int, int, int]:
    """The matched run around an anchor at (i, j): (i0, i1, j0, j1), expanding both ways."""
    i0 = j0 = 0
    while i - i0 > 0 and j - j0 > 0 and same_word(prev[i - i0 - 1], nxt[j - j0 - 1]):
        i0 += 1
        j0 += 1
    i1 = j1 = 0
    while i + i1 < len(prev) and j + j1 < len(nxt) and same_word(prev[i + i1], nxt[j + j1]):
        i1 += 1
        j1 += 1
    return i - i0, i + i1, j - j0, j + j1


def align(prev_words: list[str], next_words: list[str], overlap_seconds: float) -> dict:
    """Where the next transcript starts saying new words.  Returns cut (words of the head to drop), the matched
    count, the head words before the match (variants of the shared audio), the tail words after it, and the method.

    A confident alignment matches a run that reaches the end of the previous transcript; the cut is the end of
    that run in the head.  Otherwise a character suffix/prefix, then the overlap's expected word count, decide."""
    prev, nextw = prev_words[-TAIL_WORDS:], next_words[:HEAD_WORDS]
    best = None
    for i in range(len(prev)):
        for j in range(len(nextw)):
            if not same_word(prev[i], nextw[j]):
                continue
            i0, i1, j0, j1 = _run(prev, nextw, i, j)
            if j1 - j0 < 3:
                continue
            # prefer a run that reaches the end of the previous transcript, then the longest, then a head start
            score = (i1 == len(prev)) * 1000 + (i1 - i0) - 0.05 * j0
            if best is None or score > best[0]:
                best = (score, i0, i1, j0, j1)
    expected = max(3, min(int(round(overlap_seconds * 2.2)), len(prev), len(nextw), 60))
    if best is not None:
        _, i0, i1, j0, j1 = best
        matched = i1 - i0
        # A run that reaches the end of the previous transcript is the evidence that matters: far-field speech makes
        # the two parts phrase the shared audio differently, so a correct cut often has only 5-8 words in a row
        # ("мерфи как он искал невесту в америке"), not the tenth of the overlap a clean recording gives.  A very
        # short transcript (a test, a tiny clip) keeps the old floor of three words.
        need = 3 if len(prev) < 20 else max(4, int(0.12 * expected))
        if matched >= need and i1 == len(prev):
            return {"cut": j1, "matched": matched, "before": j0, "after": len(prev) - i1,
                    "method": "words", "tail_from": i0, "head_to": j1}
    # character fallback over the normalized words
    sa, sb = "".join(prev), "".join(nextw)
    top = min(len(sa), len(sb), 500)
    for k in range(top, 3, -1):
        if sa[-k:] == sb[:k]:
            take, have = 0, 0
            while take < len(nextw) and have + len(nextw[take]) <= k:
                have += len(nextw[take])
                take += 1
            if take:
                return {"cut": take, "matched": 0, "before": take, "after": 0, "method": "characters",
                        "tail_from": len(prev), "head_to": take}
    # No confident match: keep every head word and log it.  Dropping an estimate here would lose words the
    # previous part did not have; a repeated phrase is the lesser fault and the boundary audit names it.
    return {"cut": 0, "matched": 0, "before": 0, "after": 0, "method": "none",
            "tail_from": len(prev), "head_to": 0}


def stitch(texts: list[str], overlap_seconds: float = 15.0, warn=None) -> tuple[str, list[dict]]:
    """Join parts, dropping the head words of each part that repeat the part before.  -> (text, boundary audit)."""
    text, audit, _ = stitch_parts(texts, overlap_seconds, warn)
    return text, audit


def stitch_parts(texts: list[str], overlap_seconds: float = 15.0,
                 warn=None) -> tuple[str, list[dict], list[list[str]]]:
    """`stitch`, and the words each part contributed to the final text.  -> (text, boundary audit, words a part).

    The per-part words are what the cues are built from, so cue text and stitched text cannot drift apart: the head
    word of a part repeats the part before and is not here, and a word the previous part missed is put back into the
    previous part's list at the boundary, where it was said."""
    if not texts:
        return "", [], []
    kept = [_words(texts[0])[0]]
    joined, joined_norm = _words(texts[0])
    audit: list[dict] = []
    for text in texts[1:]:
        words, norm = _words(text)
        a = align(joined_norm, norm, overlap_seconds)
        if a["method"] in ("duration", "none") and warn:
            warn(f"no confident word overlap at a boundary: {a['method']} cut of {a['cut']} words")
        # The head words before the matched run are shared audio: the previous part already transcribed it.  A word
        # there that is not in the previous text is one of two things - an insertion the previous part missed
        # ("как раз вот"), which belongs in the merge, or a paraphrase of what the previous part did say ("Инструмент"
        # for "отдельный продукт"), which would duplicate.  The start of a part is where a decoder is least anchored,
        # so a novel word before the first word shared with the previous text is treated as a paraphrase and dropped;
        # novel words after it are inserted before the matched suffix.
        recovered: list[str] = []
        if a["method"] == "words" and a["before"] and a["matched"]:
            window = joined_norm[-max(60, 3 * a["matched"]):]
            prefix = words[:a["before"]]
            prefix_norm = [normalize(w) for w in prefix]
            first_seen = next((k for k, w in enumerate(prefix_norm)
                               if any(same_word(w, p) for p in window)), None)
            if first_seen is not None:
                for w in prefix[first_seen + 1:]:
                    if not any(same_word(normalize(w), p) for p in window):
                        recovered.append(w)
            if recovered:
                m = a["matched"]
                joined = joined[:-m] + recovered + joined[-m:]
                joined_norm = joined_norm[:-m] + [normalize(w) for w in recovered] + joined_norm[-m:]
                kept[-1] = kept[-1][:-m] + recovered + kept[-1][-m:]
        audit.append({**a, "recovered": len(recovered), "recovered_text": " ".join(recovered),
                      "prev_tail": " ".join(joined[-8:]), "next_head": " ".join(words[:8]),
                      "cut_text": " ".join(words[:a["cut"]]), "kept_head": " ".join(words[a["cut"]:a["cut"] + 6])})
        joined += words[a["cut"]:]
        joined_norm += norm[a["cut"]:]
        kept.append(words[a["cut"]:])
    return " ".join(joined), audit, kept


def _drop_words(cues: list[tuple[float, float, str]], cut: int) -> list[tuple[float, float, str]]:
    """Remove the first `cut` words across a part's cues - the shared head the previous part already has."""
    out, left = [], max(0, int(cut))
    for start, end, body in cues:
        words = body.split()
        if left:
            if len(words) <= left:
                left -= len(words)
                continue
            words = words[left:]
            left = 0
        out.append((start, end, " ".join(words)))
    return out


def _spread(body: str, start: float, end: float, prev_end: float) -> list[tuple[float, float, str]]:
    """One block of words as ~CUE_SECONDS cues inside [start, end], proportional to their length, never before the
    cue that came before."""
    words = body.split()
    if not words:
        return []
    total = sum(len(w) + 1 for w in words)
    span = max(0.05, end - start)
    groups = _groups(words, max(1.0, total * CUE_SECONDS / span))
    out: list[tuple[float, float, str]] = []
    done = 0
    for g in groups:
        chars = sum(len(w) + 1 for w in g)
        a = max(start + span * (done / total), prev_end)
        b = max(start + span * ((done + chars) / total), a + 0.05)
        out.append((a, b, " ".join(g)))
        prev_end = b
        done += chars
    return out


def stamped_cues(texts: list[str], parts: list[Part],
                 audit: list[dict] | None = None) -> list[tuple[float, float, str]]:
    """The model's own `HH:MM:SS` spans as anchors on the file's clock.  The shared head the previous part already
    transcribed is dropped, and each stamped span is itself spread into ~CUE_SECONDS cues, so an anchor every 20-30 s
    keeps the clock honest without cue text jumping a whole span at a time."""
    out: list[tuple[float, float, str]] = []
    for i, (text, part) in enumerate(zip(texts, parts)):
        blocks = transcribe_api.cues(text, part.seconds)
        cut = int((audit[i - 1] or {}).get("cut") or 0) if i and audit and i - 1 < len(audit) else 0
        if cut:
            blocks = _drop_words(blocks, cut)
        for n, (start, end, body) in enumerate(blocks):
            if not body:
                continue
            start, end = start + part.start, end + part.start
            if n == 0 and start > part.start:
                start = max(part.start, out[-1][1] if out else part.start)
            prev_end = out[-1][1] if out else 0.0
            if start < prev_end:
                if end <= prev_end + 1e-6:
                    continue
                start = prev_end
            out += _spread(body, start, max(end, start + 0.2), prev_end)
    return out


_END = re.compile(r"[.!?…][\"'»)\]]?$")


def _groups(words: list[str], target_chars: float) -> list[list[str]]:
    """Words in reading order cut into groups of about `target_chars`, preferring to end on a sentence mark."""
    total = sum(len(w) + 1 for w in words)
    n = max(1, int(round(total / max(1.0, target_chars))))
    if n <= 1:
        return [words]
    target = total / n
    out, cur, have = [], [], 0
    for i, w in enumerate(words):
        cur.append(w)
        have += len(w) + 1
        left = len(words) - i - 1
        if have >= target and left >= 2 and len(out) < n - 1 and (_END.search(w) or have >= target * 1.3):
            out.append(cur)
            cur, have = [], 0
    if cur:
        out.append(cur)
    return out


def part_seconds(part_text: str, part: Part, cut: int, overlap_seconds: float) -> tuple[float, float]:
    """Where a part's kept words were spoken: the head the previous part already covered is not theirs."""
    if cut <= 0 or not part_text:
        return part.start, part.end
    all_words = part_text.split()
    if cut >= len(all_words):
        return part.end, part.end
    head = len(" ".join(all_words[:cut]))
    total = len(" ".join(all_words))
    share = head / total if total else 0.0
    head_s = min(max(0.0, share) * part.seconds, max(0.0, overlap_seconds), 0.75 * part.seconds)
    return part.start + head_s, part.end


def timed_cues(texts: list[str], parts: list[Part], kept: list[list[str]] | None = None,
               audit: list[dict] | None = None) -> list[tuple[float, float, str]]:
    """Cues on the file's clock.  A part with the model's own stamps uses them as anchors; a part without is spread
    over the audio it really covers (the shared head excluded).  Either way the cue text joins back to the stitched
    transcript and every cue time is real audio time, not one cue for the whole file."""
    if not texts or not parts:
        return []
    if kept is None:
        kept = [_words(t)[0] for t in texts]
    overlap = 0.0
    if len(parts) > 1:
        overlap = max(0.0, min(parts[0].end - parts[1].start, parts[0].seconds))
    out: list[tuple[float, float, str]] = []
    for i, (text, part) in enumerate(zip(texts, parts)):
        cut = int((audit[i - 1] or {}).get("cut") or 0) if i and audit and i - 1 < len(audit) else 0
        if transcribe_api.STAMP.search(text):
            blocks = transcribe_api.cues(text, part.seconds)
            if cut:
                blocks = _drop_words(blocks, cut)
            for n, (start, end, body) in enumerate(blocks):
                if not body:
                    continue
                start, end = start + part.start, end + part.start
                if n == 0 and start > part.start:
                    start = max(part.start, out[-1][1] if out else part.start)
                prev_end = out[-1][1] if out else 0.0
                if start < prev_end:
                    if end <= prev_end + 1e-6:
                        continue
                    start = prev_end
                out += _spread(body, start, max(end, start + 0.2), prev_end)
            continue
        words = kept[i] if i < len(kept) else _words(text)[0]
        if not words:
            continue
        begin, end = part_seconds(text, part, cut, overlap)
        if end <= begin:
            begin, end = part.start, part.end
        prev_end = out[-1][1] if out else 0.0
        out += _spread(" ".join(words), begin, end, prev_end)
    return out


def global_cues(texts: list[str], parts: list[Part], kept: list[list[str]] | None = None,
                audit: list[dict] | None = None) -> list[tuple[float, float, str]]:
    """`timed_cues` under its old name: model cues when the model wrote times, per-part cues otherwise."""
    return timed_cues(texts, parts, kept, audit)
