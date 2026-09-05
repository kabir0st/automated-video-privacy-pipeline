"""Harvest review decisions as training labels.

Every enable/disable a reviewer makes is a human judgement on a track the
pipeline was unsure about: "this really is a head, blur it" or "this is a
misread, leave it alone". Recorded here, one JSON line per reviewed track,
so a learned replacement for the hand-tuned suspicion score has data by the
time anyone wants to fit one.

**What is deliberately NOT recorded.** This is an anonymisation tool, usually
pointed at footage its user would not want indexed, so the corpus is designed
to be worthless to anyone who obtains it:

* no imagery, ever — no crops, no thumbnails, no frames;
* no file path, name, or parent directory; videos are identified only by a
  keyed BLAKE2 hash of their *content* fingerprint (size + mtime), under a
  256-bit salt generated once on this machine and stored ``0600`` alongside
  the corpus. Without that salt the ids are not linkable to any file, and the
  salt never leaves the machine;
* no absolute timestamps — only offsets within the clip.

What is recorded is the suspicion components, the track's geometry and score
statistics, what the verifier concluded, and the user's verdict. All numbers.

Set ``AVPP_LABELS=0`` to switch the whole thing off.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pipeline.types import Track

_DIR = Path.home() / ".cache" / "avpp"
CORPUS = _DIR / "labels.jsonl"
_SALT_FILE = _DIR / "labels.salt"
_SCHEMA = 2


def enabled() -> bool:
    """``AVPP_LABELS=0`` disables label harvesting."""
    return os.environ.get("AVPP_LABELS", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _salt() -> bytes:
    """Per-machine secret keying the video ids. Created once, mode 0600."""
    try:
        if _SALT_FILE.is_file():
            raw = _SALT_FILE.read_bytes().strip()
            if len(raw) >= 32:
                return raw
    except OSError:
        pass
    salt = secrets.token_bytes(32)
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(_SALT_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(salt)
    except OSError:
        pass
    return salt


def video_id(video_path: "str | Path") -> str:
    """Keyed hash of the video's content fingerprint — never its name."""
    try:
        st = Path(video_path).stat()
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        stamp = "unknown"
    return hashlib.blake2b(stamp.encode(), key=_salt(),
                           digest_size=12).hexdigest()


def features(track: "Track", fps: float) -> dict:
    """Numeric feature vector for one track — no imagery, no identifiers."""
    t = track.t
    boxes = np.asarray(t.boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(t.scores, np.float32).reshape(-1)
    hits = np.asarray(t.hits, bool).reshape(-1)
    src = t.src_arr()
    top3 = float(np.mean(np.sort(scores)[-3:])) if len(scores) else 0.0
    step = (np.linalg.norm(np.diff(boxes[:, :2], axis=0), axis=1)
            if len(boxes) > 1 else np.zeros(1, np.float32))
    diag = float(np.median(np.hypot(boxes[:, 2], boxes[:, 3]))) or 1.0
    out = {
        "suspicion": round(float(track.suspicion), 4),
        "n_frames": int(len(boxes)),
        "n_hits": int(hits.sum()),
        "duration_s": round(len(boxes) / max(fps, 1e-6), 3),
        "hit_ratio": round(float(hits.mean()) if len(hits) else 0.0, 4),
        "score_mean": round(float(scores.mean()) if len(scores) else 0.0, 4),
        "score_top3": round(top3, 4),
        "box_w_med": round(float(np.median(boxes[:, 2])), 2),
        "box_h_med": round(float(np.median(boxes[:, 3])), 2),
        "aspect_med": round(float(np.median(
            boxes[:, 2] / np.maximum(boxes[:, 3], 1e-3))), 4),
        "motion_med": round(float(np.median(step)) / diag, 5),
        "motion_max": round(float(np.max(step)) / diag, 5),
        "size_cv": round(float(np.std(boxes[:, 2]) /
                               max(float(np.mean(boxes[:, 2])), 1e-3)), 5),
        "fvalid_ratio": round(float(t.fvalid_arr().mean()) if len(boxes) else 0.0, 4),
        "src_any": int(np.bitwise_or.reduce(src)) if len(src) else 0,
        "verified": (None if track.verified is None else round(float(track.verified), 3)),
    }
    for k, v in track.reasons.items():
        if isinstance(v, float):
            out[f"c_{k}"] = round(v, 4)
    return out


def record(
    video_path: "str | Path",
    fps: float,
    tracks: "list[Track]",
    enabled_ids: Optional[dict] = None,
    corpus: Optional[Path] = None,
) -> int:
    """Append one labelled row per track. Returns rows written.

    The label is the user's *final* verdict — whether the track ends up
    blurred: the pipeline's default (``Track.kept``) unless overridden in
    ``enabled_ids``. Best-effort: any failure is swallowed; a corpus write
    must never fail an export."""
    if not enabled():
        return 0
    overrides = enabled_ids or {}
    path = Path(corpus) if corpus is not None else CORPUS
    vid = video_id(video_path)
    stamp = int(time.time())
    rows = []
    for tr in tracks:
        label = bool(overrides.get(tr.tid, tr.kept))
        rows.append({
            "schema": _SCHEMA, "video": vid, "t": stamp, "tid": int(tr.tid),
            "start_s": round(tr.t.start / max(fps, 1e-6), 3),
            "pipeline_kept": bool(tr.kept),
            "blur": label,
            "overridden": label != bool(tr.kept),
            **features(tr, fps),
        })
    if not rows:
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        return 0
    return len(rows)
