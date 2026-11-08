#!/usr/bin/env python3
"""The process protocol of strata-audio, without torch.

    python tools/gigachat/test_protocol.py --bin engine-poly/strata-audio \
        --model .../gigachat-audio-enc-mixed.gguf --wav clip10s.wav [--wav more.wav ...]

Checks, against a resident process like the server drives it:
  * "READY <n_embd>" is printed after the weights load
  * each "ENC <wav> <out>" answers "OK <n_rows> <t0_ms> <t1_ms> <ms>" and writes one SVE1 record
    {magic, n, nx, ny, n_embd} + float32 [n][n_embd]
  * n_rows equals the arithmetic the decoder's prompt builder relies on:
        T_mel = 1 + floor(n_samples/160); T1 = floor((T_mel-1)/2)+1; T2 = floor((T1-1)/2)+1;
        n_rows = ceil(T2/4)
  * a bad path answers "ERR ..." and the process stays alive
  * QUIT exits cleanly
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
import time

MAGIC = 0x31455653


def wav_samples(path: str) -> tuple[int, int]:
    """(frames, sample rate) of a PCM/float WAV, read straight from the chunks."""
    with open(path, "rb") as f:
        b = f.read()
        assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", path
        off, nch, bits, fmt, data, rate = 12, 0, 0, 0, None, 0
        while off + 8 <= len(b):
            cid, sz = b[off:off + 4], struct.unpack("<I", b[off + 4:off + 8])[0]
            body = off + 8
            if cid == b"fmt ":
                fmt, nch = struct.unpack("<H", b[body:body + 2])[0], \
                           struct.unpack("<H", b[body + 2:body + 4])[0]
                rate = struct.unpack("<I", b[body + 4:body + 8])[0]
                bits = struct.unpack("<H", b[body + 14:body + 16])[0]
                if fmt == 0xFFFE:
                    fmt = struct.unpack("<H", b[body + 24:body + 26])[0]
            elif cid == b"data":
                data = sz
            off += 8 + sz + (sz & 1)
        return data // (nch * bits // 8), rate


def expected_rows(n_samples: int) -> tuple[int, int, int]:
    t_mel = 1 + n_samples // 160
    t1 = (t_mel - 1) // 2 + 1
    t2 = (t1 - 1) // 2 + 1
    return t_mel, t2, (t2 + 3) // 4


class Client:
    def __init__(self, exe: str, model: str, threads: int | None = None):
        args = [exe, "--model", model]
        if threads:
            args += ["--threads", str(threads)]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)

    def line(self, timeout: float = 600.0) -> str:
        # the protocol lines are short; stderr is left for the log
        assert self.p.stdout is not None
        return self.p.stdout.readline().rstrip("\n")

    def send(self, s: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def close(self) -> None:
        try:
            self.send("QUIT")
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="engine-poly/strata-audio")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", action="append", required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    fails: list[str] = []
    c = Client(args.bin, args.model, args.threads)
    t0 = time.time()
    ready = c.line()
    print(f"startup    {ready!r} in {time.time() - t0:.1f} s")
    if not ready.startswith("READY "):
        print("FAIL: no READY")
        c.close()
        return 1
    n_embd = int(ready.split()[1])
    if n_embd != 1536:
        fails.append(f"n_embd {n_embd}")

    outdir = tempfile.mkdtemp(prefix="strata-audio-")
    for wav in args.wav:
        out = os.path.join(outdir, os.path.basename(wav) + ".sve1")
        n_samples, rate = wav_samples(wav)
        # the reference resamples to 16 kHz with torchaudio, whose target length is ceil(new*len/orig)
        n16 = n_samples if rate == 16000 else math.ceil(n_samples * 16000 / rate)
        t_mel, t_enc, rows = expected_rows(n16)
        c.send(f"ENC {wav} {out}")
        line = c.line()
        parts = line.split()
        print(f"ENC        {os.path.basename(wav)} ({n_samples} @ {rate} Hz -> {n16} @ 16 kHz): "
              f"{line!r}  (expect OK {rows} 0 {n_samples * 1000 // rate} ...)")
        if parts[:1] != ["OK"]:
            fails.append(f"{wav}: {line}")
            continue
        n, t_a, t_b = int(parts[1]), int(parts[2]), int(parts[3])
        if n != rows:
            fails.append(f"{wav}: {n} rows, formula says {rows} (T_mel {t_mel}, T_enc {t_enc})")
        if t_a != 0:
            fails.append(f"{wav}: t0 {t_a}")
        if not os.path.exists(out):
            fails.append(f"{wav}: no output file")
            continue
        raw = open(out, "rb").read()
        magic, rn, nx, ny, ne = struct.unpack("<5i", raw[:20])
        if magic != MAGIC or rn != n or nx != n or ny != 1 or ne != n_embd:
            fails.append(f"{wav}: header {magic:#x} {rn} {nx} {ny} {ne}")
        if len(raw) != 20 + n * n_embd * 4:
            fails.append(f"{wav}: {len(raw)} bytes, expected {20 + n * n_embd * 4}")

    c.send("ENC /nonexistent/nope.wav " + os.path.join(outdir, "nope.sve1"))
    line = c.line()
    print(f"bad path   {line!r}")
    if not line.startswith("ERR "):
        fails.append("bad path did not answer ERR")

    c.send("FROB")
    line = c.line()
    print(f"bad cmd    {line!r}")
    if not line.startswith("ERR "):
        fails.append("bad command did not answer ERR")

    c.send("QUIT")
    try:
        rc = c.p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        c.p.kill()
        rc = -9
    print(f"exit       {rc} (expect 0)")
    if rc != 0:
        fails.append(f"exit {rc}")

    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
