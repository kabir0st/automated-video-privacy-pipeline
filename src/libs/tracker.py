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
out.

Association is keyed on the **person box** first, not the head box. The body box
barely shifts between frames, so a head darting across the frame stays a single
track that follows the head — instead of (the old head-box-only matching) failing
the IoU test on the jump, spawning a *new* track at the new position while the
old track coasts at the stale spot, which painted a trail of ghost heads down the
motion path. Head-box IoU and a head-centre fallback are kept as lower tiers so
the face-only path (no person box) and momentary segmentation dropouts still
re-acquire. At most a couple of people are ever in frame, so greedy matching is
plenty.
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


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


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
        self.head = _arr(subject.head).copy()
        self.person = _arr(subject.person_box).copy()
        self.face = subject.face
        self.mask = subject.mask
        self.source = "face" if subject.face is not None else "head"
        self.misses = 0

    def update(self, subject) -> None:
        meas = _arr(subject.head)
        # Adaptive smoothing: snap toward the measurement on fast head motion so
        # the blur never lags an exposed head, but smooth hard when the head is
        # still so the blur does not jitter. motion = 1 ⇒ the head centre moved a
        # full head-diagonal this frame (genuinely fast) → a≈0 (take the new box);
        # motion = 0 ⇒ still → a = _HEAD_SMOOTH (heavy smoothing).
        diag = float(np.hypot(self.head[2] - self.head[0],
                              self.head[3] - self.head[1])) or 1.0
        mcx, mcy = _centre(meas)
        tcx, tcy = _centre(self.head)
        motion = min(1.0, float(np.hypot(mcx - tcx, mcy - tcy)) / diag)
        a = _HEAD_SMOOTH * (1.0 - motion)
        self.head = (a * self.head + (1.0 - a) * meas).astype(np.float32)
        self.person = _arr(subject.person_box).copy()
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

    def _greedy_match(
        self,
        unmatched: list,
        matched: dict,
        score_fn,
        thr: float,
    ) -> list:
        """Greedily pair leftover tracks↔subjects by ``score_fn`` (≥ ``thr``).

        Highest score wins each round; a matched track/subject is removed from
        contention. Returns the still-unmatched subjects.
        """
        while True:
            best: tuple[float, _Track, Any] | None = None
            for t in self._tracks:
                if t in matched:
                    continue
                for s in unmatched:
                    sc = score_fn(t, s)
                    if sc >= thr and (best is None or sc > best[0]):
                        best = (sc, t, s)
            if best is None:
                return unmatched
            _, t, s = best
            matched[t] = s
            unmatched = [x for x in unmatched if x is not s]

    @staticmethod
    def _centre_score(t: "_Track", s) -> float:
        """1 − (head-centre distance / track head-diagonal): ≥0 within one
        diagonal, used as the fast-motion fallback when IoU has dropped to 0."""
        diag = float(np.hypot(t.head[2] - t.head[0], t.head[3] - t.head[1]))
        if diag <= 0:
            return -1.0
        scx, scy = _centre(_arr(s.head))
        tcx, tcy = _centre(t.head)
        return 1.0 - float(np.hypot(tcx - scx, tcy - scy)) / diag

    def update(self, subjects: list, frame_shape: tuple) -> list[TrackedFace]:
        fh, fw = frame_shape[:2]
        unmatched = list(subjects)
        matched: dict[_Track, Any] = {}

        # 1) Stable person-box IoU — the body barely moves between frames, so a
        #    head darting across the frame stays one track that follows the head
        #    rather than spawning a ghost and leaving the old spot blurred.
        unmatched = self._greedy_match(
            unmatched, matched,
            lambda t, s: _iou(t.person, _arr(s.person_box)), self._match_iou)
        # 2) Head-box IoU — the face-only path (person_box == head) and re-acquiry
        #    across a momentary segmentation dropout, where the body box vanished.
        unmatched = self._greedy_match(
            unmatched, matched,
            lambda t, s: _iou(t.head, _arr(s.head)), self._match_iou)
        # 3) Head-centre proximity — fast motion that drops every IoU to zero.
        unmatched = self._greedy_match(
            unmatched, matched, self._centre_score, 0.0)

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
