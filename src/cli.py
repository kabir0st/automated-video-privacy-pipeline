"""Headless pipeline driver — batch use and validation without the GUI.

    uv run python src/cli.py analyse VIDEO [--preset P] [--stride N] [--frames N] [--no-cache]
    uv run python src/cli.py inspect VIDEO [--preset P]
    uv run python src/cli.py export  VIDEO OUT.mp4 [--preset P] [--region head|face]
    uv run python src/cli.py dump    VIDEO OUT.json [--preset P]     # tracks + table

``analyse`` runs (or reuses) pass 1 and stores the project sidecar next to
the video; ``inspect`` prints the refined tracks sorted by suspicion;
``export`` renders using the review decisions stored in the sidecar.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pipeline import project as proj  # noqa: E402
from pipeline.analysis import Models  # noqa: E402
from pipeline.presets import DEFAULT_PRESET, PRESETS, with_stride  # noqa: E402
from pipeline.render import build_table, coverage, render_video  # noqa: E402
from pipeline.session import ensure_analysed, run_offline  # noqa: E402
from dataclasses import replace  # noqa: E402


def _status(msg: str) -> None:
    print(f"[avpp] {msg}", flush=True)


def _progress_printer(label: str):
    last = {"t": 0.0}

    def cb(done: int, total: int) -> None:
        now = time.time()
        if now - last["t"] >= 2.0 or done == total:
            last["t"] = now
            print(f"[avpp] {label} {done}/{total}", flush=True)
    return cb


def _preset(args):
    p = PRESETS[args.preset]
    if getattr(args, "stride", None):
        p = with_stride(p, args.stride)
    if getattr(args, "no_wholebody", False):
        p = replace(p, analysis=replace(p.analysis, use_wholebody=False))
    if getattr(args, "rots", None):
        r = tuple(int(x) for x in args.rots.split(","))
        p = replace(p, analysis=replace(p.analysis, headdet_rots=r,
                                        wb_rots=tuple(x for x in r if x in (0, 180)),
                                        scrfd_rots=tuple(x for x in r if x in (0, 180))))
    return p


def cmd_analyse(args) -> int:
    preset = _preset(args)
    models = Models(on_status=_status, headdet_size=preset.analysis.headdet_size,
                    scrfd_size=preset.analysis.scrfd_size)
    p = ensure_analysed(args.video, preset, models, use_cache=not args.no_cache,
                        progress=_progress_printer("pass 1"), on_status=_status,
                        max_frames=args.frames)
    if p is None:
        return 1
    tracks = run_offline(p, preset, models, on_status=_status,
                         verify=not args.no_verify)
    _print_tracks(tracks, p.fps)
    return 0


def _print_tracks(tracks, fps: float) -> None:
    print(f"{'tid':>5} {'start':>7} {'end':>7} {'dur_s':>6} {'hits':>5} "
          f"{'susp':>5} {'kept':>4}  reasons")
    for tr in tracks:
        t = tr.t
        top = sorted(((k, v) for k, v in tr.reasons.items()
                      if isinstance(v, float)), key=lambda kv: -kv[1])[:3]
        rs = " ".join(f"{k}={v:.2f}" for k, v in top)
        why = tr.reasons.get("why", "")
        print(f"{t.tid:>5} {t.start:>7} {t.end:>7} {len(t.boxes) / fps:>6.1f} "
              f"{int(t.hits.sum()):>5} {tr.suspicion:>5.2f} "
              f"{'yes' if tr.kept else 'no':>4}  {rs} {why}")


def cmd_inspect(args) -> int:
    preset = _preset(args)
    p = proj.load(args.video)
    if p is None:
        print("no project sidecar; run analyse first")
        return 1
    models = Models(on_status=_status) if not args.no_verify else None
    tracks = run_offline(p, preset, models, on_status=_status,
                         verify=not args.no_verify, identify=not args.no_verify)
    _print_tracks(tracks, p.fps)
    return 0


def cmd_export(args) -> int:
    preset = _preset(args)
    models = Models(on_status=_status)
    # Like the GUI, prefer whatever analysis already exists next to the video
    # (review decisions live in it); only analyse when there is none.
    p = None if args.no_cache else proj.load(args.video)
    if p is not None:
        _status(f"Using existing analysis ({len(p.tracklets)} tracklets, "
                f"stride {p.stride})")
    else:
        p = ensure_analysed(args.video, preset, models, on_status=_status,
                            progress=_progress_printer("pass 1"),
                            use_cache=not args.no_cache)
    if p is None:
        return 1
    tracks = run_offline(p, preset, models, on_status=_status,
                         verify=not args.no_verify)
    rcfg = replace(preset.render, region=args.region)
    try:
        from libs.labels import record as record_labels
        record_labels(p.video, p.fps, tracks, p.enabled)
    except Exception:  # noqa: BLE001 — never block an export on the corpus
        pass
    table = build_table(tracks, p.manual, p.n_frames, p.enabled, rcfg)
    cov = coverage(table)
    _status(f"blur on {int((cov > 0).sum())}/{p.n_frames} frames")
    ok, msg = render_video(p.video, args.out, table, rcfg,
                           progress=_progress_printer("pass 2"),
                           on_status=_status)
    _status(msg)
    return 0 if ok else 1


def cmd_dump(args) -> int:
    preset = _preset(args)
    p = proj.load(args.video)
    if p is None:
        print("no project sidecar; run analyse first")
        return 1
    models = Models(on_status=_status) if not args.no_verify else None
    tracks = run_offline(p, preset, models, on_status=_status,
                         verify=not args.no_verify, identify=not args.no_verify)
    table = build_table(tracks, p.manual, p.n_frames, p.enabled, preset.render)
    doc = {
        "meta": {"video": p.video, "fps": p.fps, "n_frames": p.n_frames,
                 "stride": p.stride, "preset": preset.name,
                 "settings": p.settings},
        "tracks": [{"tid": tr.tid, "start": int(tr.t.start), "end": int(tr.t.end),
                    "hits": int(tr.t.hits.sum()), "suspicion": round(tr.suspicion, 3),
                    "kept": tr.kept, "verified": tr.verified,
                    "reasons": {k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in tr.reasons.items()},
                    "src_any": int(np.bitwise_or.reduce(tr.t.src_arr()))
                    if len(tr.t.boxes) else 0}
                   for tr in tracks],
        "table": [[[int(tid), np.asarray(b).round(1).tolist()] for tid, b in row]
                  for row in table],
        "raw": {str(k): np.asarray(v).round(2).tolist() for k, v in p.raw.items()},
    }
    Path(args.out).write_text(json.dumps(doc))
    _status(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
        sp.add_argument("--stride", type=int, default=0)
        sp.add_argument("--no-verify", action="store_true")
        sp.add_argument("--no-wholebody", action="store_true")
        sp.add_argument("--rots", default="",
                        help="comma list of rotations for the head detector, e.g. 0,180")

    a = sub.add_parser("analyse"); a.add_argument("video"); common(a)
    a.add_argument("--frames", type=int, default=0); a.add_argument("--no-cache", action="store_true")
    i = sub.add_parser("inspect"); i.add_argument("video"); common(i)
    e = sub.add_parser("export"); e.add_argument("video"); e.add_argument("out"); common(e)
    e.add_argument("--region", default="head", choices=["head", "face"])
    e.add_argument("--no-cache", action="store_true", help="re-analyse even if a sidecar exists")
    d = sub.add_parser("dump"); d.add_argument("video"); d.add_argument("out"); common(d)
    args = ap.parse_args(argv)
    return {"analyse": cmd_analyse, "inspect": cmd_inspect,
            "export": cmd_export, "dump": cmd_dump}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
