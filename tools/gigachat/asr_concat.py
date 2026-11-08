#!/usr/bin/env python3
"""tools/gigachat/asr_concat.py - a long test clip from a Russian ASR dataset's own clips.

Joins the clips of a Hugging Face ASR parquet (Podlodka, Golos, ...), 0.4 s of silence between, repeating the set in
reference order until the asked length, and writes a manifest of where every clip really is.  That manifest is the
ground truth for timestamp and boundary checks, so no personal audio is read.

    python tools/gigachat/asr_concat.py --seconds 900 --out out/podlodka15
    -> out/podlodka15/audio.wav, manifest.json, reference.txt

The default parquet is the Hub dataset bond005/podlodka_speech, split test (20 clips).
`hf://datasets/...` reads the file from `${HF_HOME:-$HOME/.cache/huggingface}/hub`
and downloads it there if it is missing. Episode 132, «Дисфункции организаций», is
in that split. FLEURS and Golos are the same kind of URI (see eval_asr_ru.py).

Every clip is converted to 16 kHz mono PCM16 when it is not already that (FLEURS is float32), with numpy for float
WAV and ffmpeg for any other container.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from hf_path import resolve_data

# bond005/podlodka_speech, split test: 20 clips. Episode 132 is one of them.
PODLODKA = "hf://datasets/bond005/podlodka_speech/data/test-00000-of-00001.parquet"
GAP = 0.4
RATE = 16000


def _pcm16(data: bytes) -> tuple[bytes, float]:
    """A WAV clip as 16 kHz mono PCM16 bytes, and its length."""
    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as w:
        rate, frames = w.getframerate(), w.getnframes()
        channels, width = w.getnchannels(), w.getsampwidth()
        if w.getcomptype() != "NONE":
            raise ValueError("compressed WAV")
        raw = w.readframes(frames)
    seconds = frames / float(rate)
    if channels == 1 and width == 2 and rate == RATE:
        return raw, seconds
    if width == 4:                                   # IEEE float32, as FLEURS ships
        samples = np.frombuffer(raw, dtype="<f4")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if rate != RATE:
            raise ValueError(f"{rate} Hz float WAV")
        return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes(), seconds
    raise ValueError(f"{channels}ch/{width * 8}bit/{rate} Hz WAV")


def _ffmpeg_pcm16(data: bytes, suffix: str) -> tuple[bytes, float]:
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / f"in.{suffix}", Path(d) / "out.wav"
        src.write_bytes(data)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(RATE),
                        "-acodec", "pcm_s16le", str(out)], check=True, capture_output=True)
        raw = out.read_bytes()
    with wave.open(io.BytesIO(raw), "rb") as w:
        return w.readframes(w.getnframes()), w.getnframes() / float(w.getframerate())


def read_clips(parquet: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    names = table.schema.names
    if "audio" not in names or "transcription" not in names:
        raise SystemExit(f"{parquet}: no audio/transcription columns ({names})")
    clips = []
    for row in table.to_pylist():
        data, seconds = row["audio"]["bytes"], None
        try:
            data, seconds = _pcm16(data)
        except (ValueError, wave.Error):
            data, seconds = _ffmpeg_pcm16(data, "wav")
        if not seconds:
            continue
        clips.append({"data": data, "seconds": seconds, "text": row["transcription"],
                      "episode": row.get("episode"), "title": row.get("title")})
    return clips


def build(parquet: Path, seconds: float, out: Path) -> dict:
    clips = read_clips(parquet)
    if not clips:
        raise SystemExit("the parquet has no clips")
    out.mkdir(parents=True, exist_ok=True)
    silence = b"\x00\x00" * int(round(GAP * RATE))
    body = bytearray()
    placed, total, k = [], 0.0, 0
    while total < seconds:
        for i, clip in enumerate(clips):
            if total >= seconds:
                break
            start = len(body) / 2 / RATE
            body += clip["data"]
            end = len(body) / 2 / RATE
            placed.append({"index": k, "clip": i, "start": round(start, 3), "end": round(end, 3),
                           "text": clip["text"], "episode": clip["episode"], "title": clip["title"]})
            body += silence
            total = len(body) / 2 / RATE
            k += 1
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(body))
    manifest = {"audio": str(out / "audio.wav"), "rate": RATE, "gap": GAP,
                "seconds": round(len(body) / 2 / RATE, 3), "clips": placed}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "reference.txt").write_text(" ".join(c["text"] for c in placed), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", default=PODLODKA,
                    help="a parquet path, or hf://datasets/<org>/<name>/<file> "
                         "(default: bond005/podlodka_speech test, downloaded into the Hub cache)")
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--out", default="out/podlodka15")
    args = ap.parse_args()
    m = build(resolve_data(args.parquet), args.seconds, Path(args.out))
    print(f"{m['audio']}: {m['seconds']} s, {len(m['clips'])} clips")
    for c in m["clips"][:3]:
        print(f"  [{c['start']:7.2f},{c['end']:7.2f}] clip {c['clip']}: {c['text'][:60]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
