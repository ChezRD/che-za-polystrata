"""serve/transcribe.py - the OpenAI audio endpoints' request/response side (text, no model, no GPU).

`POST /v1/audio/transcriptions` and `POST /v1/audio/translations` are multipart/form-data in OpenAI's shape.  This
module parses the form, turns the upload into a WAV `strata-audio` reads, and builds the answer in the requested
format (`json`, `text`, `verbose_json`, `srt`, `vtt`) from the model's plain text.  By default the answer is just the
assembled text: times are opt-in (`timestamps`, or OpenAI's `timestamp_granularities[]`, or an `srt`/`vtt` format),
and `model_timestamps` asks the model for its own `M:SS` spans instead of the estimated ones.

serve/server.py runs the model; nothing here starts it or touches the engine.
"""
from __future__ import annotations

import email.policy
import re
import shutil
import subprocess
from email.parser import BytesParser
from pathlib import Path

# How a language code is said in the instruction.  The model card's own ASR prompts: "Transcribe the audio." (the
# audio's language), "Transcribe the audio in English."; unknown codes are passed through as given.
LANGUAGES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "kk": "Kazakh", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "tr": "Turkish", "uk": "Ukrainian", "uz": "Uzbek", "zh": "Chinese",
}

DEFAULT_ASR = "Transcribe the audio."
DEFAULT_TRANSLATE = "Translate the audio into English."
FORMATS = ("json", "text", "verbose_json", "srt", "vtt")

# A timestamp the model writes when a timed transcript is asked for: "12:34" (minutes:seconds) or "1:12:34"
# (hours:minutes:seconds), optionally with a fraction and a dot or a comma, and the separator it puts after it
# ("0:27 — ", "1:00: text").  The model card's own timed format is minutes:seconds; the parser takes both.
STAMP = re.compile(r"(?<![\d:])(?:(\d{1,2}):([0-5]\d):([0-5]\d)|(\d{1,2}):([0-5]\d))"
                   r"(?:[.,](\d{1,3}))?(?!\d)(?!:(?=\d))")
_AFTER_STAMP = re.compile(r"^\s*[—–\-:|·]\s*")
_STAMP_WITH_SEP = re.compile(STAMP.pattern + r"(?:\s*[—–\-:|·]\s*)?")
TIMED_ASR = "Start each line with its beginning time as HH:MM:SS."


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """A multipart/form-data body -> (plain fields, {name: (filename, bytes)}).  ValueError when it is not one or
    when it holds no part.  The standard library's parser, so repeated fields (timestamp_granularities[]) arrive under
    their own names."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("send multipart/form-data (curl: -F file=@audio.wav -F model=...)")
    raw = (b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    try:
        message = BytesParser(policy=email.policy.default).parsebytes(raw)
    except Exception as e:                                        # noqa: BLE001 - a malformed form is a 400
        raise ValueError(f"the multipart body could not be read ({e})") from None
    if not message.is_multipart():
        raise ValueError("the multipart body has no parts")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None or name in ("file", "audio"):
            files[name] = (str(filename or "audio"), payload)
        else:
            fields[str(name)] = payload.decode("utf-8", "replace")
    if not fields and not files:
        raise ValueError("the multipart body has no fields")
    return fields, files


def check_language(value, field: str = "language") -> str | None:
    """An ISO-639-1 code from a form field, lowercased; empty or absent is None.  A known code is said by its
    English name in the instruction, an unknown one as the client wrote it (the model card's own rule)."""
    if value in (None, ""):
        return None
    code = str(value).strip().lower()
    if not re.fullmatch(r"[a-z]{2,3}", code):
        raise ValueError(f"{field}={value!r}: an ISO-639-1 code (ru, en, de, ...)")
    return code


def parse_max_tokens(value) -> int:
    """The optional `max_tokens` form field: empty or absent is 0, the endpoint's own bound (the context room)."""
    if value in (None, ""):
        return 0
    try:
        n = int(str(value).strip())
    except ValueError:
        raise ValueError(f"max_tokens={value!r}: a whole number of tokens") from None
    if n < 0 or n > 32768:
        raise ValueError(f"max_tokens={value!r}: 0 to 32768")
    return n


def instruction(task: str, language: str | None = None, prompt: str | None = None,
                target_language: str | None = None, timestamps: bool = False) -> str:
    """The user text sent with the clip: the request's own `prompt`, else the model card's ASR/translation sentence.
    A translation's `language` is the source hint (absent: the audio's own) and `target_language` the target (default
    English).  `timestamps` adds the model's own timed-transcript sentence, for the formats whose purpose is timing."""
    if prompt:
        return prompt
    if task == "translate":
        target = LANGUAGES.get((target_language or "en").lower(), target_language or "en")
        if language:
            source = LANGUAGES.get(language.lower(), language)
            text = f"Translate the audio from {source} into {target}."
        else:
            text = f"Translate the audio into {target}."
    elif language:
        text = f"Transcribe the audio in {LANGUAGES.get(language.lower(), language)}."
    else:
        text = DEFAULT_ASR
    return f"{text} {TIMED_ASR}" if timestamps else text


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def upload_to_wav(data: bytes, filename: str, directory: Path) -> Path:
    """The upload as a WAV file the encoder reads.  WAV goes in as it is (the encoder downmixes and resamples); any
    other container is converted to 16 kHz mono PCM with ffmpeg, and a 400 that says what to install when that is
    missing."""
    if not data:
        raise ValueError("the audio file is empty")
    out = Path(directory) / "audio.wav"
    if is_wav(data):
        out.write_bytes(data)
        return out
    suffix = Path(filename).suffix.lower().lstrip(".") or "bin"
    suffix = re.sub(r"[^a-z0-9]", "", suffix) or "bin"
    src = Path(directory) / f"upload.{suffix}"
    src.write_bytes(data)
    exe = shutil.which("ffmpeg")
    if not exe:
        raise ValueError(f"the audio is not WAV (.{suffix}): this server converts mp3/flac/m4a/ogg with ffmpeg, "
                         "which is not installed - send a WAV file, or install ffmpeg")
    try:
        done = subprocess.run([exe, "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                               "-acodec", "pcm_s16le", "-f", "wav", str(out)],
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(f"the audio could not be converted to WAV (ffmpeg: {e})") from None
    if done.returncode != 0 or not out.is_file() or out.stat().st_size <= 44:
        detail = (done.stderr or "").strip().splitlines()
        raise ValueError("the audio could not be converted to WAV" + (f" (ffmpeg: {detail[-1]})" if detail else ""))
    return out


def _seconds(match: re.Match) -> float:
    """A stamp's time: "12:34" is minutes:seconds, "1:12:34" hours:minutes:seconds."""
    a, b, c, d, e, frac = match.groups()
    if c is not None:
        whole = int(a) * 3600 + int(b) * 60 + int(c)
    else:
        whole = int(d) * 60 + int(e)
    return whole + (int((frac or "0").ljust(3, "0")) / 1000.0)


def _cut_stamp(text: str, match: re.Match, upto: int) -> str:
    """The words between a stamp and `upto`, without the separator the timed format puts after a stamp."""
    body = text[match.end():upto]
    return plain_text(_AFTER_STAMP.sub("", body, count=1))


def plain_text(text: str) -> str:
    """The answer with the `HH:MM:SS` marks (and the separator after each) taken out, as `json`/`text` have it."""
    return " ".join(_STAMP_WITH_SEP.sub(" ", text).split())


def cues(text: str, duration: float) -> list[tuple[float, float, str]]:
    """The answer as (start, end, text) cues: one per stamp the model wrote, or one covering the whole file when it
    wrote none.  A stamp opens a cue at its time and the next stamp closes it; the file's duration closes the last."""
    matches = list(STAMP.finditer(text))
    if not matches:
        words = plain_text(text)
        return [(0.0, float(duration), words)] if words else []
    out: list[tuple[float, float, str]] = []
    lead = plain_text(text[:matches[0].start()])
    if lead:
        out.append((0.0, min(float(duration), _seconds(matches[0])), lead))
    for i, m in enumerate(matches):
        start = _seconds(m)
        upto = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _cut_stamp(text, m, upto)
        end = _seconds(matches[i + 1]) if i + 1 < len(matches) else float(duration)
        end = min(float(duration), end) if end > start else start
        if body:
            out.append((start, end, body))
    return out


def language_of(text: str, given: str | None) -> str:
    """The `verbose_json` language: the request's, else a guess from the letters (Cyrillic: ru, else en).  A guess,
    said as one - the model is not asked to detect it."""
    if given:
        return given.lower()
    letters = [c for c in text if c.isalpha()]
    cyrillic = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return "ru" if letters and cyrillic / len(letters) > 0.2 else "en"


SILENCE_PEAK = 0.001      # |sample| below this (of full scale) everywhere: a muted microphone, not quiet speech
SPEECH_CV = 0.15          # frame-level variation below this is a tone/stationary noise, not speech (speech: 0.6-1.2)
_META = re.compile(
    r"^\s*(?:#{1,6}\s|\*\*|I (?:cannot|can't|am unable|am sorry)|I'm sorry|"
    r"Я не могу|К сожалению, я не могу|Вот транскрипц|Ниже транскрипц|"
    r"(?:The|This) audio (?:contains|does not)|В этом аудио|Аудио (?:не )?содержит)", re.I)
_TAG_ONLY = re.compile(r"^\s*[\[(][^\])]{1,40}[\])]\s*[.!]?\s*$")


def wav_peak(path: str | Path, stride: int = 16) -> float:
    """The loudest sample of a RIFF/WAVE file, 0..1, for PCM16 and IEEE float32.  Every 16th sample is enough for a
    silence gate and keeps a 15-minute clip's check cheap.  1.0 when the file is not a WAV this reads."""
    from array import array

    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 1.0
    pos, fmt = 12, 1
    while pos + 8 <= len(data):
        tag, n = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if tag == b"fmt ":
            fmt = int.from_bytes(data[pos + 8:pos + 10], "little")
        elif tag == b"data":
            body = data[pos + 8:pos + 8 + n]
            if fmt == 3:                             # IEEE float32
                samples = array("f", body[:len(body) // 4 * 4])
                return min(1.0, max((abs(v) for v in samples[::stride]), default=0.0))
            if fmt == 1:                             # PCM16
                samples = array("h", body[:len(body) // 2 * 2])
                return min(1.0, max((abs(v) for v in samples[::stride]), default=0) / 32768.0)
            return 1.0
        pos += 8 + n + (n & 1)
    return 1.0


def _frame_variation(path: str | Path, ms: int = 20, stride: int = 8) -> tuple[float, float]:
    """(coefficient of variation, mean) of 20 ms frame levels, 0..1.  Speech swings by an order of magnitude between
    frames; a tone, white noise and a muted microphone do not."""
    from array import array

    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 1.0, 1.0
    pos, fmt, rate, channels = 12, 1, 16000, 1
    while pos + 8 <= len(data):
        tag, n = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if tag == b"fmt ":
            fmt = int.from_bytes(data[pos + 8:pos + 10], "little")
            channels = int.from_bytes(data[pos + 10:pos + 12], "little") or 1
            rate = int.from_bytes(data[pos + 12:pos + 16], "little") or 16000
        elif tag == b"data":
            body = data[pos + 8:pos + 8 + n]
            kind, scale = ("f", 1.0) if fmt == 3 else ("h", 32768.0) if fmt == 1 else (None, 1.0)
            if kind is None:
                return 1.0, 1.0
            samples = array(kind, body[:len(body) // (4 if kind == "f" else 2) * (4 if kind == "f" else 2)])
            step = max(1, rate * ms // 1000 * channels)
            levels = []
            for i in range(0, len(samples) - step + 1, step):
                chunk = samples[i:i + step:stride]
                levels.append((sum((v / scale) ** 2 for v in chunk) / max(1, len(chunk))) ** 0.5)
            levels = [v for v in levels if v > 1e-6]
            if not levels:
                return 0.0, 0.0
            mean = sum(levels) / len(levels)
            sd = (sum((v - mean) ** 2 for v in levels) / len(levels)) ** 0.5
            return (sd / mean if mean else 0.0), mean
        pos += 8 + n + (n & 1)
    return 1.0, 1.0


def is_silent(path: str | Path, floor: float = SILENCE_PEAK, variation: float = SPEECH_CV) -> bool:
    """True when the clip has no speech: digital silence, a steady tone or stationary noise.  The model answers those
    with a refusal, a music tag or a loop, so the route returns an empty transcript instead of asking it."""
    if wav_peak(path) < floor:
        return True
    cv, mean = _frame_variation(path)
    return mean < 1e-5 or cv < variation


def collapse_repeats(text: str, keep: int = 2) -> str:
    """Cut a decoding loop: a block of 1-40 words repeated four or more times in a row keeps `keep` copies."""
    words = text.split()
    keys = [normalize_word(w) for w in words]
    n = len(words)
    for p in range(1, min(40, n // 4) + 1):
        need = 6 if p < 3 else 4
        i = 0
        while i + need * p <= n:
            block = keys[i:i + p]
            k = 1
            while i + (k + 1) * p <= n and keys[i + k * p:i + (k + 1) * p] == block:
                k += 1
            if k >= need:
                return " ".join(words[:i + keep * p])
            i += 1
    return text


def normalize_word(word: str) -> str:
    return re.sub(r"[^\w]+", "", word.lower().replace("ё", "е"))


def clean_answer(text: str) -> str:
    """The answer as a transcript: a refusal, a markdown commentary or a lone non-speech tag is not one, and a
    decoding loop is cut.  Returns "" when there is nothing to transcribe."""
    t = text.strip()
    if not t:
        return ""
    if _TAG_ONLY.match(t) or _META.match(t):
        return ""
    return collapse_repeats(t)


def _clock(seconds: float, sep: str) -> str:
    total = max(0, int(round(float(seconds) * 1000)))
    hh, rest = divmod(total, 3600_000)
    mm, rest = divmod(rest, 60_000)
    ss, ms = divmod(rest, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ms:03d}"


def srt(text: str, duration: float) -> str:
    out = []
    for i, (start, end, body) in enumerate(cues(text, duration), 1):
        out.append(f"{i}\n{_clock(start, ',')} --> {_clock(end, ',')}\n{body}\n")
    return "\n".join(out)


def vtt(text: str, duration: float) -> str:
    out = ["WEBVTT", ""]
    for start, end, body in cues(text, duration):
        out.append(f"{_clock(start, '.')} --> {_clock(end, '.')}")
        out.append(body)
        out.append("")
    return "\n".join(out)


def verbose_json(task: str, text: str, duration: float, language: str | None,
                 granularities=("segment",), target_language: str | None = None) -> dict:
    """OpenAI's `verbose_json`.  `word` timestamps fall back to the segments (the model gives none finer than
    segments).  A translation names its target language; a transcription does not have one."""
    body = {"task": task, "language": language_of(text, language), "duration": round(float(duration), 3),
            "text": plain_text(text)}
    if task == "translate":
        body["target_language"] = (target_language or "en").lower()
    body["segments"] = [{"id": i, "start": round(start, 3), "end": round(end, 3), "text": words}
                        for i, (start, end, words) in enumerate(cues(text, duration))]
    return body


def verbose_json_cues(task: str, text: str, duration: float, language: str | None,
                      cues_list, target_language: str | None = None) -> dict:
    """`verbose_json` from cues already on the file's clock (a long clip's stitched parts)."""
    body = {"task": task, "language": language_of(text, language), "duration": round(float(duration), 3),
            "text": plain_text(text)}
    if task == "translate":
        body["target_language"] = (target_language or "en").lower()
    body["segments"] = [{"id": i, "start": round(float(a), 3), "end": round(float(b), 3), "text": words}
                        for i, (a, b, words) in enumerate(cues_list)]
    return body


def srt_cues(cues_list) -> str:
    out = []
    for i, (start, end, body) in enumerate(cues_list, 1):
        out.append(f"{i}\n{_clock(start, ',')} --> {_clock(end, ',')}\n{body}\n")
    return "\n".join(out)


def vtt_cues(cues_list) -> str:
    out = ["WEBVTT", ""]
    for start, end, body in cues_list:
        out.append(f"{_clock(start, '.')} --> {_clock(end, '.')}")
        out.append(body)
        out.append("")
    return "\n".join(out)


def _flag(value, name: str) -> bool | None:
    if value in (None, ""):
        return None
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={value!r}: true or false")


def parse_timestamps(value) -> bool | None:
    """The optional `timestamps` form field.  Empty or absent is None (off): the answer is the assembled text alone,
    as the endpoint returns by default.  `true` adds segment times (`verbose_json.segments`, one cue every 6 s),
    estimated from the parts' real audio spans."""
    return _flag(value, "timestamps")


def parse_model_timestamps(value) -> bool:
    """The optional `model_timestamps` form field: ask the model for its own timed transcript.  Off by default -
    the model's stamps help clean read speech and hurt far-field speech - and only meaningful with `timestamps`."""
    return bool(_flag(value, "model_timestamps"))


def wants_cues(timestamps: bool | None, fmt: str, granularities) -> bool:
    """Whether the answer carries times: asked for by `timestamps`, implied by a subtitle format, or by OpenAI's
    `timestamp_granularities[]`.  A plain `json`/`text`/`verbose_json` without them is only text."""
    if timestamps is not None:
        return timestamps
    return fmt in ("srt", "vtt") or bool(granularities)


def parse_temperature(value) -> float:
    if value in (None, ""):
        return 0.0
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"temperature={value!r}: a number from 0 to 1") from None
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"temperature={value!r}: a number from 0 to 1")
    return t
