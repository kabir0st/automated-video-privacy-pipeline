"""Per-stage timing for pass 1 on a real video.

    uv run python scripts/benchmark.py CLIP.mp4
    uv run python scripts/benchmark.py CLIP.mp4 --frames 120 --stride 1 2 3 --preset Fast

Reports mean milliseconds per *decoded* frame for each detector and the
non-model remainder (fusion, tracking, recording), plus the effective rate.
Nothing is written and no blur is rendered.

One set of sessions for the whole run, built once and never rebuilt: creating
a fresh session per configuration crashes the MIGraphX EP outright ("HIP
failure 700", process aborted) and corrupts DirectML device state.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from libs.utils import best_onnx_providers                     # noqa: E402
from pipeline.analysis import Models, detect_frame            # noqa: E402
from pipeline.presets import PRESETS                           # noqa: E402
from pipeline.record import TrackRecorder                      # noqa: E402
from pipeline.tracker import Tracker                           # noqa: E402

_MODELS: dict = {}


def _models() -> Models:
    if "m" not in _MODELS:
        _MODELS["m"] = Models(on_status=print)
    return _MODELS["m"]


def run(path: str, n_frames: int, cfg) -> dict:
    models = _models()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = Tracker(fps=fps / cfg.stride, spawn_conf=cfg.spawn_conf,
                      sustain_conf=cfg.sustain_conf, min_hits=cfg.min_hits,
                      max_age_s=cfg.max_age_s)
    rec = TrackRecorder()
    timings: dict = {}
    analysed = decoded = 0
    total_ms = 0.0
    while decoded < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        if (decoded - 1) % cfg.stride:
            continue
        t0 = time.perf_counter()
        c, _src = detect_frame(models, frame, cfg, timings)
        obs = tracker.update(c.heads, c.flags, frame.shape, faces=c.faces)
        rec.observe(analysed, obs)
        total_ms += (time.perf_counter() - t0) * 1e3
        analysed += 1
    cap.release()
    if not analysed:
        raise SystemExit("no frames analysed")
    per = {k: v / analysed for k, v in timings.items()}
    per["other"] = max(0.0, total_ms / analysed - sum(per.values()))
    per["per_decoded"] = total_ms / max(decoded, 1)
    per["analysed"] = analysed
    return per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--stride", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--preset", default="Balanced", choices=list(PRESETS))
    args = ap.parse_args()
    base = PRESETS[args.preset].analysis
    print(f"providers: {best_onnx_providers()}")
    print("warming sessions (first run compiles; this can take a minute)…")
    run(args.video, 2, replace(base, stride=1))
    print(f"video: {args.video}   ({args.frames} frames per config, preset {args.preset})\n")
    hdr = f"{'stride':>6} {'headdet':>8} {'wholebody':>10} {'scrfd':>7} {'other':>6} {'analysed':>9} {'per-frame':>10} {'fps':>6}"
    print(hdr)
    print("-" * len(hdr))
    for stride in args.stride:
        r = run(args.video, args.frames, replace(base, stride=stride))
        print(f"{stride:>6} {r.get('headdet', 0):>7.1f}m {r.get('wholebody', 0):>9.1f}m "
              f"{r.get('scrfd', 0):>6.1f}m {r['other']:>5.1f}m {r['analysed']:>9} "
              f"{r['per_decoded']:>9.1f}m {1000.0 / max(r['per_decoded'], 1e-6):>6.1f}")
    print("\nColumns are ms per analysed frame (all rotation passes summed), except "
          "per-frame,\nwhich is amortised over every decoded frame — that sets export wall-clock.")


if __name__ == "__main__":
    main()
