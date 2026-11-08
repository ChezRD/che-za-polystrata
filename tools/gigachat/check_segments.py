#!/usr/bin/env python3
"""tools/gigachat/check_segments.py - does a long transcript match the Podlodka clips it was built from?

The concat of `tools/gigachat/asr_concat.py` knows where every dataset clip really starts.  This compares a saved
`verbose_json` answer with that manifest:

* coverage - how much of the reference text the answer holds (words and 5-grams);
* timestamps - for every clip, the cue that carries its words (searched only inside that clip's own part, so a
  repeated clip set cannot match the wrong pass): start error, overlap with the clip's real span, and the share of
  the clip's words whose cue covers the clip at all;
* boundaries - at every part boundary, words duplicated or lost around the cut.

    python tools/gigachat/check_segments.py --manifest out/podlodka15/manifest.json \
        --answer out/p15.json [--answer out/p15-interp.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from serve import segment as seg  # noqa: E402

STRIP = re.compile(r"[^\w]+", re.UNICODE)


def norm(word: str) -> str:
    return STRIP.sub("", word.lower().replace("ё", "е"))


def words(text: str) -> list[str]:
    return [w for w in (norm(x) for x in text.split()) if w]


def grams(seq: list[str], n: int) -> list[tuple]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def ngram_coverage(ref: list[str], hyp: list[str], n: int) -> float:
    want: dict[tuple, int] = {}
    for g in grams(ref, n):
        want[g] = want.get(g, 0) + 1
    have: dict[tuple, int] = {}
    for g in grams(hyp, n):
        have[g] = have.get(g, 0) + 1
    total = sum(want.values())
    return sum(min(c, have.get(g, 0)) for g, c in want.items()) / total if total else 1.0


def locate(clip: list[str], part_words: list[str], n: int | None = None) -> int | None:
    """First word of the part that starts the clip, by the rarest shared n-gram; None when nothing is shared.  A
    two-word clip cannot offer a four-word anchor, so the shingle shortens with the clip."""
    if n is None:
        n = min(4, len(clip))
    if n < 2 or len(part_words) < n:
        return None
    want = {}
    for g in grams(clip, n):
        want[g] = want.get(g, 0) + 1
    best, best_score = None, 0
    for j, g in enumerate(grams(part_words, n)):
        s = want.get(g, 0)
        if s > best_score:
            best, best_score = j, s
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--answer", action="append", required=True)
    ap.add_argument("--segment-seconds", type=float, default=300.0)
    ap.add_argument("--overlap-seconds", type=float, default=15.0)
    ap.add_argument("--proximity", type=float, default=30.0)
    ap.add_argument("--tol", type=float, default=3.0)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    clips = man["clips"]
    ref = words((Path(args.manifest).parent / "reference.txt").read_text(encoding="utf-8"))
    parts = seg.plan(man["audio"], args.segment_seconds, args.overlap_seconds)
    print(f"audio {man['seconds']} s, {len(clips)} clips, parts " +
          ", ".join(f"[{p.start:.1f},{p.end:.1f}]" for p in parts))

    for path in args.answer:
        ans = json.loads(Path(path).read_text(encoding="utf-8"))
        segments = ans.get("segments") or []
        hyp, hyp_cue = [], []
        for si, s in enumerate(segments):
            for w in words(s.get("text") or ""):
                hyp.append(w)
                hyp_cue.append(si)
        print(f"\n== {path}")
        print(f"{len(segments)} segments / {len(hyp)} words; reference {len(ref)} words")
        print(f"coverage: words {ngram_coverage(ref, hyp, 1):.4f}, 5-gram {ngram_coverage(ref, hyp, 5):.4f}, "
              f"10-gram {ngram_coverage(ref, hyp, 10):.4f}")

        # ---- timestamps: every clip searched inside its own part only
        errors, ious, inside, worst, missed = [], [], [], [], []
        for k, c in enumerate(clips):
            part = next((p for p in parts if p.start <= c["start"] < p.end), parts[-1])
            lo = next((j for j, si in enumerate(hyp_cue) if float(segments[si].get("start", 0)) >= part.start), None)
            hi = next((j for j, si in enumerate(hyp_cue) if float(segments[si].get("start", 0)) >= part.end),
                      len(hyp))
            if lo is None:
                continue
            clip_words = words(c["text"])
            j = locate(clip_words, hyp[lo:hi])
            if j is None:
                missed.append(k)
                continue
            j += lo
            cue = segments[hyp_cue[j]]
            start, end = float(cue.get("start", 0)), float(cue.get("end", 0))
            errors.append(start - c["start"])
            spans = [(float(segments[hyp_cue[x]].get("start", 0)), float(segments[hyp_cue[x]].get("end", 0)))
                     for x in range(j, min(hi, j + len(clip_words)))]
            pred_end = max(b for _, b in spans)
            inter = max(0.0, min(pred_end, c["end"]) - max(start, c["start"]))
            ious.append(inter / max(1e-6, (c["end"] - c["start"]) + (pred_end - start) - inter))
            ok = sum(1 for a, b in spans if a <= c["end"] + args.tol and b >= c["start"] - args.tol)
            inside.append(ok / len(spans))
            worst.append((abs(start - c["start"]), k, c, start, pred_end, ious[-1], inside[-1]))
        if errors:
            ae = sorted(abs(e) for e in errors)
            print(f"cue start vs clip start: median {ae[len(ae) // 2]:.2f} s, "
                  f"p90 {ae[min(len(ae) - 1, int(0.9 * len(ae)))]:.2f} s, max {ae[-1]:.2f} s "
                  f"over {len(errors)} clips")
            print(f"cue/clip IoU mean {sum(ious) / len(ious):.4f} min {min(ious):.4f}; "
                  f"clip words whose cue touches the clip (+-{args.tol:g} s) "
                  f"{sum(inside) / len(inside):.4f}")
        if missed:
            print(f"clips whose text was not found in their part: {missed}")

        # ---- boundaries: what the shared audio region says on both sides of a cut
        def ref_slice(lo: float, hi: float) -> list[str]:
            """Reference words spoken in [lo, hi], taken proportionally from each clip's word count."""
            out_words: list[str] = []
            for c in clips:
                a, b = max(lo, c["start"]), min(hi, c["end"])
                if b <= a:
                    continue
                cw = words(c["text"])
                n0 = int(round(len(cw) * (a - c["start"]) / max(1e-6, c["end"] - c["start"])))
                n1 = int(round(len(cw) * (b - c["start"]) / max(1e-6, c["end"] - c["start"])))
                out_words += cw[n0:max(n0 + 1, n1)]
            return out_words

        def fuzzy_counts(ref_win: list[str], hyp_win: list[str]) -> tuple[int, int]:
            used = [False] * len(hyp_win)
            hit = 0
            for rw in ref_win:
                for k, hw in enumerate(hyp_win):
                    if not used[k] and seg.same_word(rw, hw):
                        used[k] = True
                        hit += 1
                        break
            return len(ref_win) - hit, len(hyp_win) - hit      # lost, extra

        # A stitching duplicate is the same 8-word run twice within PROXIMITY seconds of the clock - two passes of a
        # part saying the same thing.  A phrase repeated far apart is the recording's own repeat (the clip set is
        # played twice), so distance, not the raw count, is what separates them.
        times = [float(segments[hyp_cue[j]].get("start", 0)) for j in range(len(hyp))]
        at: dict[tuple, list[float]] = {}
        for j, g in enumerate(grams(hyp, 8)):
            at.setdefault(g, []).append(times[j])
        ref8 = set(grams(ref, 8))
        dup, dup_words = 0, 0
        for g, ts in at.items():
            if len(ts) < 2 or g in ref8:
                continue
            ts = sorted(ts)
            if any(b - a <= args.proximity for a, b in zip(ts, ts[1:])):
                dup += 1
                dup_words += 8
        print(f"stitching duplicates (8-grams twice within {args.proximity:g} s, absent from the reference): "
              f"{dup} runs / {dup_words} words")

        lost_total = 0
        for nxt in parts[1:]:
            lo, hi = nxt.start, nxt.start + args.overlap_seconds
            want = ref_slice(lo, hi)
            lost = fuzzy_counts(want, hyp)[0]
            lost_total += lost
            print(f"cut {lo:7.2f} s, shared [{lo:.1f},{hi:.1f}]: {len(want)} reference words, "
                  f"{lost} of them absent from the whole transcript")
        print(f"cut losses: {lost_total} reference words over {len(parts) - 1} boundaries")
        for err, k, c, start, pred_end, iou, ok in sorted(worst, reverse=True)[:5]:
            print(f"  worst clip {k:2d} true [{c['start']:7.2f},{c['end']:7.2f}] predicted "
                  f"[{start:7.2f},{pred_end:7.2f}] start {start - c['start']:+7.2f} s "
                  f"IoU {iou:.3f} words in place {ok:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
