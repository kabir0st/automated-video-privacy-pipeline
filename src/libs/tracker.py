"""Per-person subject tracker for the segmentation-first pipeline.

Replaces the old Kalman face tracker. In the new design every frame already
yields a reliable head region for *every* detected person (see libs/pipeline.py
``build_subjects``), so the tracker has only three small jobs:

  * give each person a stable id across frames (for the overlay and for the
    landmark smoother keyed on it),
  * smooth the head box so the blur does not jitter, and
  * hold a person's last head box for a short while when RF-DETR drops them for
    a frame or two, so the blur does not flicker off.

There is deliberately no motion model, no coasting across the frame, and no
"only blur a face we have seen" rule — a head that leaves the frame simply ages
out. Association is greedy IoU on the head box (with a centre-distance fallback
for fast motion); at most a couple of people are ever in frame, so nothing
fancier is needed.
"""

from typing import Any, NamedTuple, Optional

import numpy as np

_MATCH_IOU = 0.30
# Head-box exponential smoothing: new = a*old + (1-a)*measured. Enough to kill
# per-frame jitter without the blur lagging a moving head.
_HEAD_SMOOTH = 0.5
# Privacy cap — only bounds pathological detector spam; every real person tracks.
_MAX_TRACKS = 16


class TrackedFace(NamedTuple):
    track_id: int
    face: Any | None          # refining Face when one was found this frame
    bbox: np.ndarray          # xyxy float32 — smoothed head region, always present
    source: str               # "face" | "head" | "hold"
    mask: Any | None          # body silhouette (uint8) for this person, or None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def _centre(box: np.ndarray) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)


class _Track:
    def __init__(self, track_id: int, subject) -> None:
        self.id = track_id
        self.head = np.asarray(subject.head, dtype=np.float32).copy()
        self.face = subject.face
        self.mask = subject.mask
        self.source = "face" if subject.face is not None else "head"
        self.misses = 0

    def update(self, subject) -> None:
        meas = np.asarray(subject.head, dtype=np.float32)
        self.head = (_HEAD_SMOOTH * self.head + (1.0 - _HEAD_SMOOTH) * meas
                     ).astype(np.float32)
        self.face = subject.face
        self.mask = subject.mask
        self.source = "face" if subject.face is not None else "head"
        self.misses = 0

    def coast(self) -> None:
        """A frame with no matching subject: hold position, drop the face."""
        self.face = None
        self.source = "hold"
        self.misses += 1


class SubjectTracker:
    """Tracks the handful of people in frame; smooths and briefly holds heads."""

    def __init__(
        self,
        fps: float = 30.0,
        match_iou: float = _MATCH_IOU,
        hold_secs: float = 0.5,
        max_tracks: int = _MAX_TRACKS,
    ) -> None:
        self._fps = max(fps, 1.0)
        self._match_iou = match_iou
        self._hold_secs = hold_secs
        self._max_tracks = max_tracks
        self._tracks: list[_Track] = []
        self._next_id = 1

    def configure(
        self,
        match_iou: Optional[float] = None,
        hold_secs: Optional[float] = None,
    ) -> None:
        if match_iou is not None:
            self._match_iou = match_iou
        if hold_secs is not None:
            self._hold_secs = hold_secs

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    @property
    def _hold_frames(self) -> int:
        return max(1, int(round(self._hold_secs * self._fps)))

    def update(self, subjects: list, frame_shape: tuple) -> list[TrackedFace]:
        fh, fw = frame_shape[:2]
        unmatched = list(subjects)
        matched: dict[_Track, Any] = {}

        # Greedy IoU matching of subjects to existing tracks.
        while unmatched:
            best: tuple[float, _Track, Any] | None = None
            for t in self._tracks:
                if t in matched:
                    continue
                for s in unmatched:
                    score = _iou(t.head, np.asarray(s.head, dtype=np.float32))
                    if score >= self._match_iou and (best is None or score > best[0]):
                        best = (score, t, s)
            if best is None:
                break
            _, t, s = best
            matched[t] = s
            unmatched = [x for x in unmatched if x is not s]

        # Centre-distance fallback: fast motion can drop IoU to zero between
        # frames; accept a subject whose head centre is within one head-diagonal.
        for s in list(unmatched):
            scx, scy = _centre(np.asarray(s.head, dtype=np.float32))
            best_d: tuple[float, _Track] | None = None
            for t in self._tracks:
                if t in matched:
                    continue
                tcx, tcy = _centre(t.head)
                diag = float(np.hypot(t.head[2] - t.head[0], t.head[3] - t.head[1]))
                d = float(np.hypot(tcx - scx, tcy - scy))
                if d < diag and (best_d is None or d < best_d[0]):
                    best_d = (d, t)
            if best_d is not None:
                matched[best_d[1]] = s
                unmatched = [x for x in unmatched if x is not s]

        for t, s in matched.items():
            t.update(s)

        # Age unmatched tracks; hold their last head box up to hold_frames.
        survivors: list[_Track] = []
        for t in self._tracks:
            if t not in matched:
                t.coast()
            cx, cy = _centre(t.head)
            inside = 0 <= cx < fw and 0 <= cy < fh
            if t.misses <= self._hold_frames and inside:
                survivors.append(t)
        self._tracks = survivors

        # Births — subjects that matched no track start fresh.
        for s in unmatched:
            if len(self._tracks) >= self._max_tracks:
                break
            t = _Track(self._next_id, s)
            self._next_id += 1
            matched[t] = s
            self._tracks.append(t)

        return [TrackedFace(t.id, t.face, t.head.copy(), t.source, t.mask)
                for t in self._tracks]


# Back-compat alias: ui.py and any callers may still import the old name.
KalmanFaceTracker = SubjectTracker
