#!/usr/bin/env python3
"""tools/gigachat/eval_asr_ru.py - WER/CER of the GigaChat audio route on Russian ASR datasets.

Reads a parquet of audio plus transcription (a local file, or `hf://datasets/<org>/<name>/<file>`
from the Hugging Face Hub cache), sends clips to a running server's `/v1/audio/transcriptions`, and scores them
against the reference text.  Two Hub datasets cover a different domain than the Podlodka slice: FLEURS ru_ru
validation (clean read speech) and Golos farfield test (spontaneous, far microphone).

    python tools/gigachat/eval_asr_ru.py \
        --data fleurs=hf://datasets/google/fleurs/parquet-data/ru_ru/validation-00000-of-00001.parquet \
        --data golos=hf://datasets/bond005/sberdevices_golos_100h_farfield/data/test-00000-of-00001-7fb58d4cda8ff9fe.parquet \
        --limit 60 --out out/eval-asr-ru.json

The score is word/character edit distance after normalization (case, ё, punctuation).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

from hf_path import resolve_data

URL = "http://127.0.0.1:8081/v1/audio/transcriptions"
STRIP = re.compile(r"[^\w]+", re.UNICODE)


def norm(text: str) -> list[str]:
    return [w for w in (STRIP.sub("", x.lower().replace("ё", "е")) for x in text.split()) if w]


def chars(words: list[str]) -> list[str]:
    return list(" ".join(words))


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: list[str], hyp: list[str]) -> float:
    return edit(ref, hyp) / max(1, len(ref))


def read_rows(path: Path, limit: int, lo: float, hi: float, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["audio", "transcription"])
    rows = table.to_pylist()
    random.Random(seed).shuffle(rows)
    out = []
    for row in rows:
        data = row["audio"]["bytes"]
        secs = wav_seconds(data)
        if not lo <= secs <= hi:
            continue
        out.append({"bytes": data, "name": row["audio"].get("path") or "", "ref": row["transcription"],
                    "seconds": secs})
        if len(out) >= limit:
            break
    return out


def wav_seconds(data: bytes) -> float:
    """Seconds of a RIFF/WAVE clip, PCM or IEEE float (FLEURS is float32)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV clip")
    pos, rate, align, size = 12, 16000, 2, len(data) - 44
    while pos + 8 <= len(data):
        tag, n = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if tag == b"fmt ":
            rate = int.from_bytes(data[pos + 12:pos + 16], "little")
            align = int.from_bytes(data[pos + 20:pos + 22], "little") or 2
        elif tag == b"data":
            size = n
            break
        pos += 8 + n + (n & 1)
    return size / float(max(1, align) * max(1, rate))


def transcribe(data: bytes, name: str, language: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = requests.post(URL, files={"file": (name or "clip.wav", data, "audio/wav")},
                      data={"model": "gigachat-audio-10b-a1.8b", "response_format": "json",
                            "language": language, "max_tokens": str(max_tokens)}, timeout=1200)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return {"text": r.json().get("text", ""), "wall_s": round(elapsed, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", action="append", required=True,
                    help="name=path or name=hf://datasets/<org>/<name>/<file>")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--language", default="ru")
    ap.add_argument("--out", default="out/eval-asr-ru.json")
    args = ap.parse_args()

    report = {}
    for spec in args.data:
        name, _, path = spec.partition("=")
        path = resolve_data(path)
        rows = read_rows(path, args.limit, args.min_seconds, args.max_seconds, args.seed)
        print(f"\n== {name}: {len(rows)} clips from {path}", flush=True)
        scored, total_w, total_ref = [], 0.0, 0
        for i, row in enumerate(rows, 1):
            secs = wav_seconds(row["bytes"])
            got = transcribe(row["bytes"], row["name"], args.language, max(64, int(secs * 8) + 64))
            ref, hyp = norm(row["ref"]), norm(got["text"])
            w = wer(ref, hyp)
            scored.append({"i": i, "seconds": round(secs, 2), "ref": row["ref"], "hyp": got["text"],
                           "wer": round(w, 4), "cer": round(edit(chars(ref), chars(hyp)) / max(1, len(chars(ref))), 4),
                           "wall_s": got["wall_s"]})
            total_w += w * len(ref)
            total_ref += len(ref)
            print(f"  {i:3d}/{len(rows)} {secs:5.1f} s  WER {w:5.3f}  {got['wall_s']:6.2f} s  "
                  f"ref {len(ref):3d} hyp {len(hyp):3d}", flush=True)
        report[name] = {"clips": len(scored), "wer": round(total_w / max(1, total_ref), 4),
                        "cer": round(sum(s["cer"] for s in scored) / max(1, len(scored)), 4),
                        "rows": scored}
        print(f"  -> {name}: WER {report[name]['wer']:.4f}, CER {report[name]['cer']:.4f} "
              f"over {len(scored)} clips", flush=True)
        worst = sorted(scored, key=lambda s: -s["wer"])[:5]
        for s in worst:
            print(f"     worst WER {s['wer']:.2f}: ref {s['ref'][:70]!r}\n"
                  f"                       hyp {s['hyp'][:70]!r}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
