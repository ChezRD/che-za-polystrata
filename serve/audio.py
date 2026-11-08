"""serve/audio.py - the resident audio encoder: `strata-audio` (tools/audio/strata_audio.cpp) reads `ENC <wav> <out>`
lines and writes the GigaChat audio decoder's `[audio_token]` embeddings, one `SVE1` record per clip.  The record is
the same one strata-vision writes (`{magic, n, nx, ny, n_embd}` + rows), with `nx = n, ny = 1`: one row per 160 ms.

CPU is the default, and the only mode of the CPU build: it allocates no VRAM, so it never competes with the decoder
for the card.  Setting the audio config's `"gpu": true` (and optionally `"device": N`) instead runs the CUDA build
(`engine-poly/strata-audio-cuda`, or `"exe_gpu"`), with the same graph on the card and the CPU backend kept as the
fallback for ops that have no CUDA kernel.  It is lazy, like the elastic vision encoder: nothing starts at model load,
the first clip starts it, and one process serves every request (its results are cached by the clip's hash).

`expand_audio_tokens()` mirrors the reference processor's function of the same name: the `[audio_token]` id is
replaced by `n` audio rows, with an `HH:MM:SS` anchor every 60 s at token `secs*6.25 + 8j`, and the clip's duration
appended as one more stamp after the block.
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import math
import os
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from serve.winjob import contain, signal_group, spawn, spawn_kwargs

AUDIO_TOKEN = "[audio_token]"
SVE1 = 0x31455653
AUDIO_READY_S = float(os.environ.get("STRATA_AUDIO_READY_S") or 300)     # weights + one warm-up encode
AUDIO_ENCODE_S = float(os.environ.get("STRATA_AUDIO_ENCODE_S") or 600)   # plus 4x the clip's length
QUIT_TIMEOUT_S = 5.0
TERM_TIMEOUT_S = 5.0
KILL_TIMEOUT_S = 5.0
FRAMERATE_MS = 160                                                       # 6.25 rows a second


def audio_start_error(line: str) -> str:
    """What to say when `strata-audio` printed something else than READY."""
    return "the audio encoder did not start: " + (line.strip() or "(it said nothing)")


def wav_seconds(path: str | Path) -> float | None:
    """A RIFF/WAVE file's length in seconds from its own header (no decoding, so it is cheap on a big clip), or None
    for anything this does not read.  The encoder resamples and downmixes; this is the reference processor's
    `frames / samplerate` for the file as it was sent."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            rate = block = data = None
            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    return None
                cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
                if cid == b"fmt ":
                    fmt = f.read(size)
                    if len(fmt) < 14:
                        return None
                    rate, block = struct.unpack("<I", fmt[4:8])[0], struct.unpack("<H", fmt[12:14])[0]
                    if size & 1:
                        f.seek(1, 1)
                elif cid == b"data":
                    data = size
                    break
                else:
                    f.seek(size + (size & 1), 1)
            if not rate or not block:
                return None
            return data / block / rate
    except OSError:
        return None


def timing_lut(tok) -> dict[int, int]:
    """The `HH:MM:SS` stamps are written with the tokenizer's own single-token digits and colon, as the reference
    builds its LUT (`_build_timing_lut`): {0..9, 10 for ':'} -> token id."""
    lut = {}
    for surface in [str(d) for d in range(10)] + [":"]:
        ids = tok.encode(surface, parse_special=True)
        if len(ids) != 1:
            raise ValueError(f"the timing token {surface!r} does not encode to a single id ({ids}); "
                             "audio timestamps need single-token digits and a colon")
        lut[10 if surface == ":" else int(surface)] = ids[0]
    return lut


def hms_ids(secs: int, lut: dict[int, int]) -> list[int]:
    """One `HH:MM:SS` stamp as 8 token ids (`h h : m m : s s`), the reference's `_hms_token_ids`."""
    s = int(secs)
    if s < 0 or s >= 100 * 3600:
        raise ValueError("HH:MM:SS timestamps only support durations under 100 hours")
    hh, mm, ss = s // 3600, (s % 3600) // 60, s % 60
    codes = (hh // 10, hh % 10, 10, mm // 10, mm % 10, 10, ss // 10, ss % 10)
    return [lut[c] for c in codes]


def anchor_seconds(total_seconds: int, specs=("60s",)) -> list[int]:
    """The timestamps of the reference's `_inter_mark_seconds`: every `spec` seconds strictly below the total, sorted.
    A spec is `<n><s|m|h>` (the checkpoint's `boundary_timings.inter_timings` is `["60s"]`)."""
    out: set[int] = set()
    for spec in specs:
        spec = str(spec).strip()
        unit = spec[-1:] if spec else ""
        if unit not in ("s", "m", "h") or not spec[:-1].isdigit():
            raise ValueError(f"bad inter-timing {spec!r}: expected <n>s, <n>m or <n>h")
        out.update(range(0, int(total_seconds), int(spec[:-1]) * {"s": 1, "m": 60, "h": 3600}[unit]))
    return sorted(out)


def audio_block(length: int, pad: int, secs: list[int], dur: int, lut: dict[int, int],
                framerate_ms: int = FRAMERATE_MS) -> tuple[list[int], list[tuple[int, int]]]:
    """One clip's block: `length` audio pads with an 8-token stamp at `positions + 8j`, then the duration's stamp.
    -> (tokens, [(row offset, rows), ...]): the runs of pads, i.e. where a contiguous `SVE1` record can be read.
    """
    starts = [(s * 1000) // framerate_ms + 8 * j for j, s in enumerate(secs)]
    keep = [(s, st) for s, st in zip(secs, starts) if (s * 1000) // framerate_ms < length]
    block = [pad] * (length + 8 * len(keep))
    stamps: list[int] = []
    for s, st in keep:
        block[st:st + 8] = hms_ids(s, lut)
        stamps.extend(range(st, st + 8))
    runs, i = [], 0
    while i < len(block):
        if block[i] == pad:
            j = i
            while j < len(block) and block[j] == pad:
                j += 1
            runs.append((i - bisect.bisect_left(stamps, i), j - i))
            i = j
        else:
            i += 1
    if sum(n for _, n in runs) != length:
        raise ValueError(f"the audio block's stamps split {length} rows into {sum(n for _, n in runs)}")
    return block + hms_ids(dur, lut), runs


def expand_audio_tokens(ids: list[int], pad: int, lengths: list[int], durations: list[int],
                        lut: dict[int, int], framerate_ms: int = FRAMERATE_MS,
                        inter_specs=("60s",)) -> tuple[list[int], list[list[tuple[int, int]]]]:
    """`ids` with each `pad` (one per clip, in prompt order) replaced by its expanded audio block.
    -> (tokens, [runs per clip]).  `runs` says which rows of the clip's `SVE1` record sit in which contiguous run
    of the prompt, which is what the engine's one-record-per-picture reader can take."""
    places = [i for i, t in enumerate(ids) if t == pad]
    if len(places) != len(lengths):
        raise ValueError(f"the prompt has {len(places)} audio place(s) and the request {len(lengths)} clip(s)")
    out: list[int] = []
    prev, all_runs = 0, []
    for at, length, dur in zip(places, lengths, durations):
        out += ids[prev:at]
        secs = anchor_seconds(framerate_ms * int(length) // 1000, inter_specs)
        block, runs = audio_block(int(length), pad, secs, int(dur), lut, framerate_ms)
        out += block
        all_runs.append(runs)
        prev = at + 1
    out += ids[prev:]
    return out, all_runs


def record_parts(path: str | Path) -> tuple[int, bytes]:
    """A `SVE1` record's (row count, its float rows as bytes); ValueError for anything else."""
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path}: not an SVE1 record (short file)")
    magic, n, nx, ny, n_embd = struct.unpack("<5i", data[:20])
    if magic != SVE1 or n < 1 or nx < 1 or ny < 1 or nx * ny != n:
        raise ValueError(f"{path}: not an SVE1 record of one grid")
    if len(data) != 20 + n * n_embd * 4:
        raise ValueError(f"{path}: SVE1 record of {n} rows, {len(data) - 20} bytes")
    return n, data[20:]


def split_records(path: str | Path, runs: list[tuple[int, int]]) -> bytes:
    """The clip's rows as one `SVE1` record per contiguous run: `{magic, n, n, 1, n_embd}` + those rows.
    `runs` is each run's (row offset, rows), in order; together they are the whole record."""
    n, rows = record_parts(path)
    n_embd = len(rows) // (n * 4)
    row_bytes = n_embd * 4
    out = bytearray()
    for off, count in runs:
        if off < 0 or count < 1 or off + count > n:
            raise ValueError(f"{path}: run ({off}, {count}) is outside the record's {n} rows")
        out += struct.pack("<5i", SVE1, count, count, 1, n_embd)
        out += rows[off * row_bytes:(off + count) * row_bytes]
    return bytes(out)


class Audio:
    """The resident audio encoder process and its cache.  `encode()` takes a WAV (or its bytes) and returns the row
    count and the encoder's own timings; `encode_all()` is one request's clips, keeping the files it still needs.

    While a clip is encoded the helper prints `PROGRESS <rows_done> <rows_total> <ms> [gpu_util=N vram_mib=N]`
    lines, and a short-lived sampler reads `nvidia-smi` every 500 ms: together they are what the status page shows
    (live utilization, VRAM, rows done) while the request waits.  The final numbers come from the helper's `OK`
    line, which sampled its own encode window - never from the sampler's last guess."""

    @staticmethod
    def work_dir() -> Path:
        """The encoder's directory.  The engine reads the file named in a `GENA` line and the encoder the output of
        an `ENC <wav> <out>` line, both split at spaces, so a space in the path broke the line (#480).  The system
        temp directory is used while its path has no space, then the working directory, then the system drive's
        root; the plain temp directory is the last resort."""
        for parent in (None, Path.cwd(), Path(os.environ.get("SystemDrive", "C:") + os.sep)):
            try:
                d = Path(tempfile.mkdtemp(prefix="strata-audio-", dir=parent))
            except OSError:
                continue
            if " " not in str(d):
                return d
            try:
                d.rmdir()
            except OSError:
                pass
        return Path(tempfile.mkdtemp(prefix="strata-audio-"))

    def __init__(self, cfg: dict, log=None, env: dict | None = None, lazy: bool = True):
        absolute = lambda p: os.path.abspath(p) if os.path.dirname(str(p)) else p   # noqa: E731
        gpu, device = bool(cfg.get("gpu")), cfg.get("device")
        exe = (cfg.get("exe_gpu") or cfg["exe"]) if gpu else cfg["exe"]
        args = [absolute(exe), "--model", absolute(cfg["model"])]
        if gpu:
            args.append("--gpu")
            if device is not None and str(device) != "":
                args += ["--device", str(device)]
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("group_blocks") is not None:
            args += ["--group-blocks", str(cfg["group_blocks"])]
        env = dict(env or {})
        if not gpu:                                  # CPU path: it must not take a card from the decoder
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            env["HIP_VISIBLE_DEVICES"] = "-1"
        self.gpu, self.device = gpu, (0 if (gpu and device in (None, "")) else device)
        self.device_label = f"cuda:{int(self.device)}" if gpu else "cpu"
        self.dir = self.work_dir()
        self.spawn = (args, log, env)
        self.proc = None
        self.stopped = True
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int, int, int, int]] = {}
        self.n_embd = 1536
        self.encode_s = float(cfg.get("timeout_s") or 0) or AUDIO_ENCODE_S
        self.model = Path(str(cfg["model"])).name
        self.live_lock = threading.Lock()                 # what the last/current encode is doing
        self.live: dict = {"encoding": False}
        self.last_batch: dict | None = None               # the encoder's own account of the last encode_all()
        # What GET /v1/status and the Prometheus metrics report: finished requests by task, failures, the audio and
        # rows they covered, the last one, and a ring of their end-to-end times (after the model was loaded).
        self.stats_lock = threading.Lock()
        self.started_once = False
        self.requests = {"transcribe": 0, "translate": 0}
        self.errors = 0
        self.audio_seconds = 0.0
        self.rows = 0
        self.encode_ms_total = 0.0
        self.encode_count = 0
        self.last: dict | None = None
        self.latencies_ms: collections.deque[float] = collections.deque(maxlen=128)   # whole request, waited
        self.encode_ms_hist: collections.deque[float] = collections.deque(maxlen=128)  # strata-audio alone
        if not lazy:
            self._start()

    def _start(self):
        self.started_once = True
        args, log, env = self.spawn
        self.proc = spawn(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL,
                          text=True, encoding="utf-8", bufsize=1, env=env, cwd=self.dir, **spawn_kwargs())
        contain(self.proc)
        try:
            line = self._readline(AUDIO_READY_S, "the audio encoder did not start")
        except RuntimeError as e:
            self.proc, self.stopped = None, True
            raise RuntimeError(str(e)) from None
        if not line.startswith("READY"):
            proc, self.proc = self.proc, None
            self.stopped = True
            try:
                signal_group(proc, signal.SIGKILL)
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise RuntimeError(audio_start_error(line))
        parts = line.split()
        self.n_embd = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1536
        self.stopped = False

    def _readline(self, timeout: float, what: str) -> str:
        """One line from the encoder, waiting at most `timeout` s.  On a timeout the encoder is killed, so the next
        request starts a fresh one, and the read raises RuntimeError instead of holding the request turn."""
        proc = self.proc
        got: queue.Queue = queue.Queue()

        def read():
            try:
                got.put(proc.stdout.readline())
            except (OSError, ValueError):               # killed under the read
                got.put("")

        threading.Thread(target=read, daemon=True).start()
        try:
            return got.get(timeout=timeout)
        except queue.Empty:
            try:
                signal_group(proc, signal.SIGKILL)
            except OSError:
                pass
            raise RuntimeError(f"{what}: it said nothing for {timeout:.0f} s") from None

    def alive(self) -> bool:
        return self.proc is not None and not self.stopped and self.proc.poll() is None

    def unload(self):
        """Stop the encoder process (its ~800 MB of RAM goes back); the encoded clips stay cached on disk."""
        self.close()
        self.stopped = True

    def restart(self):
        """Start the encoder again after an unload (or if it died); the cache of encoded clips is kept."""
        if self.proc is not None:
            signal_group(self.proc, signal.SIGKILL)
        self.proc = None
        self.stopped = True
        self._start()

    def encode_all(self, sources: list, cancel=None) -> list[tuple[Path, int, int, int, int]]:
        """encode() for every clip of one request -> [(SVE1 file, rows, t0_ms, t1_ms, ms)].  The files of this
        request's earlier clips are never evicted to make room for a later one; `cancel` stops between clips."""
        out: list[tuple[Path, int, int, int, int]] = []
        clips: list[dict] = []
        for src in sources:
            if cancel is not None and cancel.is_set():
                raise AudioCancelled()
            out.append(self.encode_file(src, keep=[p for p, *_ in out]))
            if self.last_batch is not None:
                clips.append(dict(self.last_batch))
        if clips:
            self.last_batch = self.combine(clips)
        return out

    @staticmethod
    def combine(clips: list[dict]) -> dict:
        """One request's clips as one account: times and rows summed, utilization time-weighted, VRAM peaked."""
        ms = sum(float(c.get("encode_ms") or 0) for c in clips)
        util = [(c.get("gpu_util_pct"), c.get("encode_ms") or 0) for c in clips if c.get("gpu_util_pct") is not None]
        weight = sum(w for _, w in util)
        vrams = [int(c["vram_mib"]) for c in clips if c.get("vram_mib") is not None]
        works = [int(c["workspace_mib"]) for c in clips if c.get("workspace_mib") is not None]
        out = {"device": clips[0].get("device"), "rows": sum(int(c.get("rows") or 0) for c in clips),
               "encode_ms": round(ms, 1), "chunks": sum(int(c.get("chunks") or 0) for c in clips),
               "vram_mib": max(vrams) if vrams else None,
               "workspace_mib": max(works) if works else None,
               "mel_ms": round(sum(float(c.get("mel_ms") or 0) for c in clips), 1),
               "graph_ms": round(sum(float(c.get("graph_ms") or 0) for c in clips), 1)}
        if util and weight > 0:
            out["gpu_util_pct"] = round(sum(u * w for u, w in util) / weight)
        elif util:
            out["gpu_util_pct"] = round(sum(u for u, _ in util) / len(util))
        else:
            out["gpu_util_pct"] = None
        return out

    @staticmethod
    def parse_ok(line: str) -> tuple[int, int, int, int, dict]:
        """`OK <rows> <t0> <t1> <ms> [key=value ...]` -> (rows, t0, t1, ms, the key=values the helper sampled)."""
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"the audio encoder's answer is not OK <rows> <t0> <t1> <ms>: {line!r}")
        rows, t0, t1, ms = (int(x) for x in parts[1:5])
        meta: dict = {}
        for p in parts[5:]:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            try:
                meta[k] = float(v) if "." in v else int(v)
            except ValueError:
                meta[k] = v
        return rows, t0, t1, ms, meta

    def encode(self, path: str | Path | bytes) -> tuple[int, int, int, int]:
        """-> (rows, t0_ms, t1_ms, ms) for one clip: a WAV path, or its bytes."""
        return self.encode_file(path)[1:]

    def encode_file(self, path: str | Path | bytes, keep=()) -> tuple[Path, int, int, int, int]:
        """-> (SVE1 file, rows, t0_ms, t1_ms, ms).  `keep`: files the caller still needs (never evicted here)."""
        data = path if isinstance(path, bytes) else Path(path).read_bytes()
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                self.cache[key] = self.cache.pop(key)                  # most recently used last
                return self.cache[key]
            if not self.alive():
                self._start()
            wav, out = self.dir / f"{key}.wav", self.dir / f"{key}.sve1"
            wav.write_bytes(data)
            seconds = wav_seconds(wav)
            timeout = self.encode_s + (4.0 * seconds if seconds else 0.0)
            started = time.time()
            with self.live_lock:
                self.live = {"encoding": True, "device": self.device_label, "rows_done": 0,
                             "rows_total": None, "started": started, "encode_ms_live": 0}
            sampler = None
            if self.gpu:
                sampler = GpuSampler(int(self.device))
                if not sampler.start():
                    sampler = None
            try:
                self.proc.stdin.write(f"ENC {wav.name} {out.name}\n")   # relative to the encoder's cwd (#480)
                self.proc.stdin.flush()
                try:
                    while True:                                         # PROGRESS lines come first, then OK/ERR
                        line = self._readline(timeout, "the clip could not be read").strip()
                        if line.startswith("PROGRESS"):
                            self.note_progress(line, started, sampler)
                            continue
                        break
                except RuntimeError as e:                               # silent for the timeout: it was ended
                    self.stopped = True
                    raise ValueError(str(e)) from None
            finally:
                with self.live_lock:
                    self.live = {"encoding": False, "device": self.device_label,
                                 "rows_done": self.live.get("rows_done"),
                                 "rows_total": self.live.get("rows_total"),
                                 "encode_ms_live": round((time.time() - started) * 1000)}
                if sampler is not None:
                    sampler.stop()                                      # the OK line's own numbers are the record
                wav.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the clip could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                  "the audio encoder stopped"))
            try:
                rows, t0, t1, ms, meta = self.parse_ok(line)
            except ValueError as e:
                raise ValueError(str(e)) from None
            with self.live_lock:
                # the last sample of the encode phase: every row the clip has is done (the bar fills only here)
                self.live.update(rows_done=rows, rows_total=rows)
                self.last_batch = {"device": str(meta.get("device") or self.device_label), "rows": rows,
                                   "encode_ms": ms, "chunks": meta.get("chunks"), "mel_ms": meta.get("mel_ms"),
                                   "graph_ms": meta.get("enc_ms"), "gpu_util_pct": meta.get("gpu_util"),
                                   "vram_mib": meta.get("vram_mib"), "workspace_mib": meta.get("workspace_mib"),
                                   "weights_mib": meta.get("weights_mib")}
            self.cache[key] = (out, rows, t0, t1, ms)
            held = {str(p) for p in keep} | {str(out)}
            for old in list(self.cache):                               # oldest first, but never one still in use
                if len(self.cache) <= 64:
                    break
                if str(self.cache[old][0]) not in held:
                    self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def note_progress(self, line: str, started: float, sampler: GpuSampler | None) -> None:
        """One `PROGRESS <rows_done> <rows_total> <ms> [gpu_util=N vram_mib=N]` line: the live rows and time the
        status page shows, plus the sampler's current numbers when it has any."""
        parts = line.split()
        done = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None
        total = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        values = {}
        for p in parts[3:]:
            if "=" in p:
                k, v = p.split("=", 1)
                if v.isdigit():
                    values[k] = int(v)
        live = {"rows_done": done, "rows_total": total, "encode_ms_live": round((time.time() - started) * 1000)}
        if "gpu_util" in values:
            live["gpu_util_pct"] = values["gpu_util"]
        if "vram_mib" in values:
            live["vram_mib"] = values["vram_mib"]
        if sampler is not None:
            sample = sampler.now()
            if sample:
                live.update(sample)
        with self.live_lock:
            if self.live.get("encoding"):
                self.live.update(live)

    def note(self, task: str, language: str | None, duration_s: float, rows: int,
             total_ms: float | None = None, *, prefill_ms: float | None = None,
             decode_ms: float | None = None, encode_ms: float | None = None,
             chunks: int | None = None, update: bool = True) -> dict:
        """A finished audio request.  `encode_ms` is `strata-audio`'s own time for the request's clips (the helper's
        `OK` line), `prefill_ms` the decoder reading the prompt with the audio rows, `decode_ms` its token loop, and
        `total_ms` what a caller waited (model load and queue included).  `rtf` is encode against the clip's length."""
        batch = dict(self.last_batch or {})
        encode_ms = float(batch.get("encode_ms") or 0.0) if encode_ms is None else float(encode_ms)
        duration_s = max(0.0, float(duration_s))
        total_ms = float(total_ms) if total_ms is not None else encode_ms
        last = {"task": task, "language": language, "duration_s": round(duration_s, 3), "rows": int(rows),
                "device": batch.get("device") or self.device_label,
                "encode_ms": int(round(encode_ms)),
                "prefill_ms": int(round(prefill_ms)) if prefill_ms is not None else None,
                "decode_ms": int(round(decode_ms)) if decode_ms is not None else None,
                "rtf": round(encode_ms / 1000.0 / duration_s, 3) if duration_s > 0 else None,
                "at": int(time.time())}
        for k in ("gpu_util_pct", "vram_mib", "workspace_mib"):
            if batch.get(k) is not None:
                last[k] = batch[k]
        if chunks is not None:
            last["chunks"] = int(chunks)
        elif batch.get("chunks") is not None:
            last["chunks"] = batch["chunks"]
        if total_ms is not None:
            last["total_ms"] = int(round(total_ms))
        if update:
            with self.stats_lock:
                self.requests[task] = self.requests.get(task, 0) + 1
                self.audio_seconds += duration_s
                self.rows += int(rows)
                self.encode_ms_total += encode_ms
                self.encode_count += 1
                self.last = last
                self.latencies_ms.append(total_ms)
                self.encode_ms_hist.append(encode_ms)
        return last


    def note_error(self) -> None:
        """An audio request that failed after its form was read (a conversion, the encoder, the engine)."""
        with self.stats_lock:
            self.errors += 1

    def progress(self) -> dict:
        """What the clip being encoded has done so far, from the helper's own PROGRESS lines: the rows finished of
        the clip's rows (never an estimate from the clock) and the sampler's current GPU numbers."""
        with self.live_lock:
            live = dict(self.live)
        return {"encoding": bool(live.get("encoding")), "rows_done": live.get("rows_done"),
                "rows_total": live.get("rows_total"), "gpu_util_pct": live.get("gpu_util_pct"),
                "vram_mib": live.get("vram_mib"), "encode_ms_live": live.get("encode_ms_live")}

    def state(self) -> str:
        """`running` (the helper process is up), `stopped` (it ran and ended, or was unloaded) or `idle` (lazy: the
        first clip starts it)."""
        if self.alive():
            return "running"
        return "stopped" if self.started_once else "idle"

    def status(self, active: bool = False) -> dict:
        """The `audio` block of GET /v1/status, as the dashboard reads it.  `active`: a request that uses the
        encoder is in flight, so the previous request's GPU numbers must not stand in for this one's."""
        with self.stats_lock:
            lat = sorted(self.latencies_ms)
            enc_lat = sorted(self.encode_ms_hist)
            requests = dict(self.requests, errors=self.errors)
            seconds, rows, last = self.audio_seconds, self.rows, (dict(self.last) if self.last else None)
            encode_ms_total, encode_count = self.encode_ms_total, self.encode_count

        def percentile(p: float, xs=None):
            xs = lat if xs is None else xs
            if not xs:
                return None
            return int(round(xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))]))

        with self.live_lock:
            live = dict(self.live)
        encoding = bool(live.get("encoding"))
        out = {"enabled": True, "encoder": self.state(), "model": self.model, "requests": requests,
               "device": self.device_label, "encoding": encoding,
               "encode_ms": (last or {}).get("encode_ms", 0),
               # live only while the helper prints PROGRESS: the rows it has finished and the time so far
               "encode_ms_live": live.get("encode_ms_live") if encoding else None,
               "rows": rows,
               "rows_done": live.get("rows_done") if encoding else None,
               "rows_total": live.get("rows_total") if encoding else None,
               "audio_seconds": round(seconds, 2),
               "rtf": (last or {}).get("rtf"),
               "chunks": (last or {}).get("chunks"),
               "prefill_ms": (last or {}).get("prefill_ms"),
               "decode_ms": (last or {}).get("decode_ms"),
               "last": last,
               "latency_ms": {"p50": percentile(0.5), "p95": percentile(0.95)},
               "encode_latency_ms": {"p50": percentile(0.5, enc_lat), "p95": percentile(0.95, enc_lat)},
               "queue": 0,
               "encode_ms_total": round(encode_ms_total, 1), "encode_count": encode_count}
        if self.gpu:                                  # the CPU path reports no GPU fields at all
            prev = {} if (active and not encoding) else (last or {})
            for k in ("gpu_util_pct", "vram_mib", "workspace_mib"):
                v = live.get(k) if encoding else prev.get(k)
                if v is not None:
                    out[k] = v
            if encoding and live.get("gpu_util_pct") is not None:
                out["gpu_util_pct"] = live["gpu_util_pct"]
        return out

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            self.stopped = True
            return
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=QUIT_TIMEOUT_S)
        except Exception:
            signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=TERM_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                signal_group(proc, signal.SIGKILL)
                try:
                    proc.wait(timeout=KILL_TIMEOUT_S)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            except (OSError, subprocess.SubprocessError):
                signal_group(proc, signal.SIGKILL)
        self.stopped = True

    def shutdown(self):
        """close() for good: the server ends, so the encoder's directory (one .wav input and one .sve1 per clip, on a
        tmpfs /tmp that is RAM) goes with it instead of piling up with every start."""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class AudioCancelled(Exception):
    """The client left while its clips were being encoded (a 499, as ClientGone in serve/server.py)."""


class GpuSampler:
    """`nvidia-smi` in a short-lived thread for the length of one encode: live utilization and memory while the
    request waits.  It samples the card (as `nvidia-smi` does), never the process; on a machine without the tool it
    starts nothing and reports nothing, and the CPU path never starts it at all."""

    def __init__(self, device: int = 0, every_ms: int = 500):
        self.device, self.every_ms = device, every_ms
        self.proc = None
        self.thread = None
        self.samples: list[tuple[float, int, int]] = []
        self.lock = threading.Lock()
        self.t0 = 0.0

    def start(self) -> bool:
        args = ["nvidia-smi", "-i", str(self.device), "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits", "-lms", str(self.every_ms)]
        try:
            self.proc = spawn(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except (OSError, ValueError):
            self.proc = None
            return False
        self.t0 = time.time()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return True

    def _read(self):
        try:
            for line in self.proc.stdout:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
                    with self.lock:
                        self.samples.append((time.time() - self.t0, int(parts[0]), int(parts[1])))
                        del self.samples[:-120]
        except (OSError, ValueError):
            pass

    def now(self) -> dict:
        with self.lock:
            if not self.samples:
                return {}
            _, util, mem = self.samples[-1]
            return {"gpu_util_pct": util, "vram_mib": mem}

    def stop(self) -> dict:
        """Mean/peak of what it saw, or {} when it saw nothing."""
        if self.proc is None:
            return {}
        try:
            self.proc.terminate()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)
        with self.lock:
            s = list(self.samples)
        self.proc, self.thread = None, None
        if not s:
            return {}
        return {"gpu_util_pct": round(sum(x[1] for x in s) / len(s)), "gpu_util_max": max(x[1] for x in s),
                "vram_mib": max(x[2] for x in s), "samples": len(s)}
