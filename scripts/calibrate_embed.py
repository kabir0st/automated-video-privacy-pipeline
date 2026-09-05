"""Calibrate the appearance channel's veto threshold on real footage.

libs/embed.py is wired into the bridge stage as a one-directional veto: a
cosine below ``DIFF_ID_COS`` blocks a join geometry proposed, and a high score
never buys a join geometry rejected. That asymmetry makes it safe to ship
uncalibrated — if the embedder is uninformative for your footage the veto
simply never fires — but it also means the threshold is currently a guess.

This measures the two distributions that decide the right value:

  * **same-track** pairs — two crops from the *same* tracklet, which are the
    same person by construction. These should score high.
  * **cross-track** pairs — crops from tracklets that overlap in time, so they
    cannot be the same person. These should score low.

If the two distributions overlap heavily, the embedder is not discriminating
on this footage (a common outcome with heavy occlusion, extreme angles, or
faces too small to resolve) and the veto should stay conservative or off
(``AVPP_EMBED=0``). If they separate cleanly, set ``DIFF_ID_COS`` between
them — nearer the cross-track side, since a wrong veto only splits a track
while a wrong merge smears two people together.

    uv run python scripts/calibrate_embed.py CLIP.mp4

Requires a project sidecar (``CLIP.mp4.avpp2.json``) from a previous analysis,
so the tracklets being compared are the real ones the pipeline produced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from libs.embed import DIFF_ID_COS, FaceEmbedder, cosine   # noqa: E402
from pipeline import project as proj                        # noqa: E402


def read_tracklets(video: str):
    """Tracklets straight out of the project sidecar, ignoring the
    fingerprint: calibration only needs real tracklets from this video, not
    ones recorded under a particular parameter set. Step-space tracklets
    are mapped to frames with the recorded stride."""
    p = proj.load(video)
    if p is None:
        raise SystemExit(f"no readable project sidecar at {proj.sidecar_path(video)}")
    out = []
    for t in p.tracklets:
        t.start = t.start * p.stride
        out.append((t, p.stride))
    return out


def sidecar_path(video: str):
    return proj.sidecar_path(video)


def _crops_for(t, cap, k=8, stride=1):
    """Up to k evenly spaced hit frames of one tracklet, as (frame, box)."""
    hits = np.nonzero(np.asarray(t.hits, bool))[0]
    if not len(hits):
        return []
    picks = hits[np.linspace(0, len(hits) - 1, min(k, len(hits))).astype(int)]
    fb = t.fb()
    out = []
    for i in sorted(set(int(x) for x in picks)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t.start + i * stride)
        ok, frame = cap.read()
        if not ok:
            continue
        cx, cy, w, h = fb[i]
        out.append((frame, np.array(
            [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]], np.float32)))
    return out


def _pct(a, qs=(1, 5, 25, 50, 75, 95, 99)):
    return {q: float(np.percentile(a, q)) for q in qs} if len(a) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--samples", type=int, default=8,
                    help="crops sampled per tracklet (default 8)")
    args = ap.parse_args()

    side = sidecar_path(args.video)
    if not Path(side).is_file():
        raise SystemExit(
            f"no sidecar at {side} — analyse this file first so there are "
            f"real tracklets to compare.")

    tracklets = read_tracklets(args.video)
    if not tracklets:
        raise SystemExit("sidecar has no tracklets")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    emb = FaceEmbedder(on_status=lambda m: print(f"  {m}"))

    vecs = {}
    for t, stride in tracklets:
        vs = [emb.embed(f, b)[0] for f, b in _crops_for(t, cap, args.samples, stride)]
        vs = [v for v in vs if np.any(v)]
        if len(vs) >= 2:
            vecs[t.tid] = (t, np.stack(vs))
    cap.release()
    if len(vecs) < 2:
        raise SystemExit("not enough embeddable tracklets to calibrate")

    same, cross = [], []
    for tid, (_t, vs) in vecs.items():
        m = cosine(vs, vs)
        same.extend(m[np.triu_indices(len(vs), k=1)].tolist())
    items = list(vecs.items())
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ta, va = items[a][1]
            tb, vb = items[b][1]
            # Overlapping in time ⇒ two different people, by construction.
            if ta.start <= tb.start + len(tb.boxes) - 1 \
                    and tb.start <= ta.start + len(ta.boxes) - 1:
                cross.extend(cosine(va, vb).ravel().tolist())

    same, cross = np.array(same), np.array(cross)
    print(f"\ntracklets embedded: {len(vecs)}")
    print(f"same-track pairs:  {len(same)}")
    print(f"cross-track pairs: {len(cross)}  (time-overlapping ⇒ different)")
    print(f"\n{'pct':>5} {'same-track':>12} {'cross-track':>12}")
    ps, pc = _pct(same), _pct(cross)
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"{q:>4}% {ps.get(q, float('nan')):>12.3f} "
              f"{pc.get(q, float('nan')):>12.3f}")

    if len(same) and len(cross):
        lo, hi = float(np.percentile(same, 1)), float(np.percentile(cross, 99))
        print(f"\ncurrent DIFF_ID_COS = {DIFF_ID_COS}")
        if lo > hi:
            print(f"Distributions separate cleanly ({hi:.3f} .. {lo:.3f}).")
            print(f"A threshold near {hi + (lo - hi) * 0.25:.3f} would veto "
                  f"cross-track joins while sparing same-track ones.")
        else:
            print(f"Distributions OVERLAP (cross p99 {hi:.3f} >= same p1 "
                  f"{lo:.3f}).")
            print("The embedder is not separating identities on this footage. "
                  "Keep the\nthreshold low (or set AVPP_EMBED=0); raising it "
                  "would split real tracks.")
    else:
        print("\nNot enough pairs to draw a conclusion.")


if __name__ == "__main__":
    main()
