"""Project sidecar: ``<video>.avpp2.json`` next to the source.

Holds the raw pass-1 output (step-space tracklets + per-frame fused
candidates), keyed by a fingerprint of the file and every analysis-affecting
setting, plus everything review produces: per-track enable overrides and
manual tracks. Refinement/scoring/rendering settings are *not* part of the
fingerprint — changing them re-runs the cheap offline stages from the cached
raw data, never pass 1.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .types import ManualTrack, Tracklet

SUFFIX = ".avpp2.json"
SCHEMA = 2


def sidecar_path(video: "str | Path") -> Path:
    return Path(str(video) + SUFFIX)


def fingerprint(video: "str | Path", analysis_params: dict) -> str:
    st = Path(video).stat()
    key = {"size": st.st_size, "mtime": int(st.st_mtime),
           "params": analysis_params}
    blob = json.dumps(key, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Project:
    video: str
    fps: float
    n_frames: int
    stride: int
    fingerprint: str
    tracklets: list[Tracklet] = field(default_factory=list)   # raw, step space
    raw: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> (N,6)
    raw_faces: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> (M,5)
    raw_sources: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    enabled: dict[int, bool] = field(default_factory=dict)    # review overrides
    manual: list[ManualTrack] = field(default_factory=list)
    settings: dict = field(default_factory=dict)              # last-used knobs
    width: int = 0
    height: int = 0

    @property
    def frame_hw(self) -> tuple[int, int]:
        return (self.height, self.width)

    def next_manual_id(self) -> int:
        used = [m.tid for m in self.manual]
        return (min(used) - 1) if used else -1000


def _t2d(t: Tracklet) -> dict:
    return {"tid": int(t.tid), "start": int(t.start),
            "boxes": t.boxes.round(2).tolist(),
            "scores": t.scores.round(3).tolist(),
            "hits": t.hits.astype(int).tolist(),
            "fboxes": t.fb().round(2).tolist(),
            "src": t.src_arr().astype(int).tolist(),
            "fvalid": t.fvalid_arr().astype(int).tolist()}


def _d2t(d: dict) -> Tracklet:
    return Tracklet(int(d["tid"]), int(d["start"]),
                    np.asarray(d["boxes"], np.float32).reshape(-1, 4),
                    np.asarray(d["scores"], np.float32),
                    np.asarray(d["hits"], bool),
                    np.asarray(d["fboxes"], np.float32).reshape(-1, 4),
                    np.asarray(d["src"], np.uint32),
                    np.asarray(d["fvalid"], bool))


def save(p: Project) -> Path:
    doc = {
        "schema": SCHEMA, "fingerprint": p.fingerprint,
        "fps": p.fps, "n_frames": p.n_frames, "stride": p.stride,
        "width": p.width, "height": p.height,
        "tracklets": [_t2d(t) for t in p.tracklets],
        "raw": {str(k): np.asarray(v, np.float32).round(2).tolist()
                for k, v in p.raw.items()},
        "raw_faces": {str(k): np.asarray(v, np.float32).round(2).tolist()
                      for k, v in p.raw_faces.items()},
        "raw_sources": {str(k): {n: np.asarray(a, np.float32).round(2).tolist()
                                 for n, a in v.items() if len(a)}
                        for k, v in p.raw_sources.items()},
        "enabled": {str(k): bool(v) for k, v in p.enabled.items()},
        "manual": [{"tid": int(m.tid), "start": int(m.start),
                    "boxes": np.asarray(m.boxes, np.float32).round(1).tolist()}
                   for m in p.manual],
        "settings": p.settings,
    }
    path = sidecar_path(p.video)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    tmp.replace(path)
    return path


def load(video: "str | Path",
         analysis_params: Optional[dict] = None) -> Optional[Project]:
    """The project for ``video``, or ``None`` when absent, unreadable, or
    (when ``analysis_params`` is given) recorded under different analysis
    settings — the caller re-runs pass 1 then."""
    path = sidecar_path(video)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("schema") != SCHEMA:
            return None
        if analysis_params is not None and \
                doc.get("fingerprint") != fingerprint(video, analysis_params):
            return None
        return Project(
            video=str(video), fps=float(doc["fps"]),
            n_frames=int(doc["n_frames"]), stride=int(doc.get("stride", 1)),
            fingerprint=str(doc["fingerprint"]),
            tracklets=[_d2t(d) for d in doc.get("tracklets", [])],
            raw={int(k): np.asarray(v, np.float32).reshape(-1, 6)
                 for k, v in doc.get("raw", {}).items()},
            raw_faces={int(k): np.asarray(v, np.float32).reshape(-1, 5)
                       for k, v in doc.get("raw_faces", {}).items()},
            raw_sources={int(k): {n: np.asarray(a, np.float32).reshape(-1, 5)
                                  for n, a in v.items()}
                         for k, v in doc.get("raw_sources", {}).items()},
            enabled={int(k): bool(v) for k, v in doc.get("enabled", {}).items()},
            manual=[ManualTrack(int(m["tid"]), int(m["start"]),
                                np.asarray(m["boxes"], np.float32).reshape(-1, 4))
                    for m in doc.get("manual", [])],
            settings=dict(doc.get("settings", {})),
            width=int(doc.get("width", 0)), height=int(doc.get("height", 0)))
    except (OSError, ValueError, KeyError, TypeError):
        return None
