"""Analysis sidecar: persist pass-1 tracklets + review decisions next to the
source video so a re-export can skip the expensive analyse pass.

Pass 1 (detect → pose → evidence gate → track, once per frame, see
ui.py._analyse_pass) is the costly stage — three model passes per frame plus
tracking. Pass 2's offline cleanup (:func:`libs.tracklets.clean_tracklets`)
is comparatively cheap pure-numpy work (plus a handful of cropped re-inference
calls during VERIFY, run only on the few tracklets that survive the prune).
So the split this module persists is exactly the pass-1/pass-2 boundary: the
*raw* tracklets :class:`libs.tracklets.TrackRecorder` produced, keyed by a
fingerprint of the video file plus every param that can change what pass 1
itself records (confidence floors, evidence profile, rotation assist, ...).

On the next export of the same file with an unchanged fingerprint, the raw
tracklets load straight from disk and pass 1 is skipped entirely; cleanup-only
param changes (bridge gap, smoothing window, face hold) then re-run
``clean_tracklets`` — including its VERIFY step — from the cached raw data.
This is a deliberate simplification over caching VERIFY's own outcome
separately: re-running cropped re-inference on the surviving tracklets is
cheap relative to pass 1, so there is no need for a second fingerprint to
invalidate it.

Also carries :class:`ManualRegion` (a user-drawn blur region, Phase 4's
review UI) and per-track enable/disable decisions — the schema exists now so
Phase 4 has a persistence layer ready; both are empty until that UI writes
them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .tracklets import Tracklet

_SUFFIX = ".avpp.json"
_SCHEMA_VERSION = 1


@dataclass
class ManualRegion:
    """A user-drawn blur region (Phase 4's review UI) spanning
    ``[start, end]`` frames, linearly interpolated between an xyxy box at
    each end — a static region is just ``box0 == box1``."""
    start: int
    end: int
    box0: tuple[float, float, float, float]
    box1: tuple[float, float, float, float]

    def box_at(self, frame: int) -> Optional[np.ndarray]:
        """Interpolated xyxy box at ``frame``, or ``None`` outside range."""
        if not (self.start <= frame <= self.end):
            return None
        t = 0.0 if self.end == self.start \
            else (frame - self.start) / (self.end - self.start)
        b0 = np.asarray(self.box0, dtype=np.float32)
        b1 = np.asarray(self.box1, dtype=np.float32)
        return b0 + (b1 - b0) * t


@dataclass
class ReviewDecisions:
    """Per-track enable/disable + manual regions from the review UI (Phase
    4). ``enabled`` maps ``tid -> False`` for a track the user disabled; a
    track absent from the map defaults to enabled — so an empty
    :class:`ReviewDecisions` (the only kind Phase 3 ever writes) changes
    nothing."""
    enabled: dict[int, bool] = field(default_factory=dict)
    manual_regions: list[ManualRegion] = field(default_factory=list)


def sidecar_path(video_path: "str | Path") -> Path:
    return Path(str(video_path) + _SUFFIX)


def fingerprint(video_path: "str | Path", analysis_params: dict) -> str:
    """Stable id for "this exact video file + these exact analysis-affecting
    params". Keyed on file size + mtime (not a content hash — re-hashing a
    multi-GB video on every export would defeat the point of caching) plus
    whatever pass-1-affecting params the caller passes (confidence floors,
    evidence profile, rotation assist, ...; cleanup-only knobs like bridge
    gap or smoothing window must NOT be included here — a re-export that
    only changed those should still hit the cache)."""
    st = Path(video_path).stat()
    key = {"size": st.st_size, "mtime": int(st.st_mtime),
          "params": analysis_params}
    blob = json.dumps(key, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _tracklet_to_dict(t: Tracklet) -> dict:
    fboxes = t.fboxes if t.fboxes is not None else t.boxes
    ev = t.ev if t.ev is not None else np.zeros(len(t.boxes), np.uint32)
    fvalid = (t.fvalid if t.fvalid is not None
             else np.zeros(len(t.boxes), bool))
    return {
        "tid": t.tid, "start": t.start,
        "boxes": t.boxes.tolist(), "scores": t.scores.tolist(),
        "hits": t.hits.tolist(), "fboxes": fboxes.tolist(),
        "ev": ev.tolist(), "fvalid": fvalid.tolist(),
    }


def _tracklet_from_dict(d: dict) -> Tracklet:
    return Tracklet(
        tid=int(d["tid"]), start=int(d["start"]),
        boxes=np.asarray(d["boxes"], np.float32).reshape(-1, 4),
        scores=np.asarray(d["scores"], np.float32),
        hits=np.asarray(d["hits"], bool),
        fboxes=np.asarray(d["fboxes"], np.float32).reshape(-1, 4),
        ev=np.asarray(d["ev"], np.uint32),
        fvalid=np.asarray(d["fvalid"], bool))


def _review_to_dict(r: ReviewDecisions) -> dict:
    return {
        "enabled": {str(k): v for k, v in r.enabled.items()},
        "manual_regions": [
            {"start": m.start, "end": m.end,
             "box0": list(m.box0), "box1": list(m.box1)}
            for m in r.manual_regions
        ],
    }


def _review_from_dict(d: dict) -> ReviewDecisions:
    return ReviewDecisions(
        enabled={int(k): bool(v) for k, v in d.get("enabled", {}).items()},
        manual_regions=[
            ManualRegion(start=int(m["start"]), end=int(m["end"]),
                        box0=tuple(m["box0"]), box1=tuple(m["box1"]))
            for m in d.get("manual_regions", [])
        ],
    )


def save(
    video_path: "str | Path",
    *,
    fps: float,
    n_frames: int,
    analysis_params: dict,
    tracklets: list[Tracklet],
    review: Optional[ReviewDecisions] = None,
) -> Path:
    """Write the sidecar next to ``video_path`` (``.part`` + atomic rename).
    Returns the path written."""
    path = sidecar_path(video_path)
    doc = {
        "schema": _SCHEMA_VERSION,
        "fingerprint": fingerprint(video_path, analysis_params),
        "fps": fps, "n_frames": n_frames,
        "tracklets": [_tracklet_to_dict(t) for t in tracklets],
        "review": _review_to_dict(review or ReviewDecisions()),
    }
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    tmp.replace(path)
    return path


def load(
    video_path: "str | Path", analysis_params: dict,
) -> Optional[tuple[float, int, list[Tracklet], ReviewDecisions]]:
    """``(fps, n_frames, tracklets, review)`` from the sidecar next to
    ``video_path``, or ``None`` when no sidecar exists, it's unreadable, or
    its fingerprint doesn't match this exact video + these exact
    analysis-affecting params (a different file, or a param that changes
    what pass 1 itself would record) — the caller should re-run pass 1 in
    that case, exactly as if no sidecar existed."""
    path = sidecar_path(video_path)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("schema") != _SCHEMA_VERSION:
            return None
        if doc.get("fingerprint") != fingerprint(video_path, analysis_params):
            return None
        tracklets = [_tracklet_from_dict(d) for d in doc["tracklets"]]
        review = _review_from_dict(doc.get("review", {}))
        return float(doc["fps"]), int(doc["n_frames"]), tracklets, review
    except (OSError, ValueError, KeyError, TypeError):
        # Corrupt/partial/foreign-schema sidecar — degrade to "no sidecar",
        # never crash the export over a stale cache file.
        return None
