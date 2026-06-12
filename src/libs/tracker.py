"""Kalman-filter face tracker with detection-gap coasting.

Replaces the previous boxmot ByteTrack wrapper, which only ever returned
tracks that matched a detection in the current frame — the moment the
detector lost a face (profile view, looking down, partial occlusion) the
track vanished from the output and the blur switched off.

Here each face owns a constant-velocity Kalman filter (state
[cx, cy, w, h, vx, vy]; size is a random walk). Every frame the filter
predicts the head position; measurements then correct it:

  * a matched InsightFace detection is a strong correction (low noise),
  * a pose-estimated head box (see libs/pose_head.py) is a weak one —
    it pins position but barely moves the size, since pose head boxes
    are coarser than face detections,
  * with no measurement at all the track coasts on prediction (velocity
    damped each frame so the box cannot sail away) and is dropped only
    after `hold_secs` without any correction or when it leaves the frame.

Tracks are only ever *created* from a face detection, so the pose model
cannot start blurring someone whose face was never seen.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np

_MATCH_IOU = 0.3
# Pose head boxes are coarse; accept them on loose overlap with the
# predicted box, or on centre distance when overlap fails entirely.
_HEAD_MATCH_IOU = 0.1
# Per-coast-frame velocity decay — keeps an undetected box from drifting
# off across the frame at its last observed speed.
_COAST_VEL_DAMP = 0.92

# ByteTrack-style noise weights, relative to box height.
_W_POS = 1.0 / 20.0
_W_VEL = 1.0 / 160.0


class TrackedFace(NamedTuple):
    track_id: int
    face: Any | None          # InsightFace Face when matched this frame
    bbox: np.ndarray          # xyxy float32 — Kalman state, always present
    source: str               # "face" | "head" | "coast"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def _xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


class _Track:
    _F = np.eye(6)
    _F[0, 4] = 1.0
    _F[1, 5] = 1.0
    _H = np.eye(4, 6)

    def __init__(self, track_id: int, bbox: np.ndarray) -> None:
        self.id = track_id
        self.misses = 0          # frames since last correction of any kind
        self.source = "face"
        z = _xyxy_to_z(bbox)
        h = max(z[3], 1.0)
        self.x = np.concatenate([z, [0.0, 0.0]])
        self.P = np.diag([
            (2 * _W_POS * h) ** 2, (2 * _W_POS * h) ** 2,
            (2 * _W_POS * h) ** 2, (2 * _W_POS * h) ** 2,
            (10 * _W_VEL * h) ** 2, (10 * _W_VEL * h) ** 2,
        ])

    @property
    def bbox(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w = max(w, 2.0)
        h = max(h, 2.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        dtype=np.float32)

    def predict(self) -> None:
        h = max(self.x[3], 1.0)
        q_pos = (_W_POS * h) ** 2
        q_vel = (_W_VEL * h) ** 2
        Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel])
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + Q

    def damp_velocity(self) -> None:
        self.x[4:] *= _COAST_VEL_DAMP

    def correct(self, bbox: np.ndarray, *, weak: bool = False) -> None:
        z = _xyxy_to_z(bbox)
        h = max(self.x[3], 1.0)
        if weak:
            # Pose head box: trust position loosely, size barely at all.
            r = [(2.5 * _W_POS * h) ** 2] * 2 + [(0.5 * h) ** 2] * 2
        else:
            r = [(_W_POS * h) ** 2] * 4
        R = np.diag(r)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P
        self.misses = 0


class KalmanFaceTracker:
    """Tracks a handful of faces (the app targets at most two people).

    update() takes this frame's filtered InsightFace detections plus an
    optional lazy `head_provider` callable. The provider is only invoked
    when at least one live track failed to match a face detection, so the
    (comparatively expensive) pose model runs exclusively on the frames
    that actually need it.
    """

    def __init__(
        self,
        fps: float = 30.0,
        match_iou: float = _MATCH_IOU,
        hold_secs: float = 2.0,
        # Privacy: every detected face must get a track (and therefore a
        # blur) — the cap only bounds pathological detector spam.
        max_tracks: int = 16,
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
        """Live-update thresholds without resetting track IDs."""
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

    def update(
        self,
        faces: list[Any],
        frame_shape: tuple[int, ...],
        head_provider: Optional[Callable[[], list[tuple[np.ndarray, float]]]] = None,
    ) -> list[TrackedFace]:
        fh, fw = frame_shape[:2]

        for t in self._tracks:
            t.predict()

        face_of: dict[_Track, Any] = {}
        unmatched_faces = list(faces)

        # Greedy IoU matching — with at most a couple of tracks/faces the
        # Hungarian algorithm buys nothing.
        while unmatched_faces:
            best: tuple[float, _Track, Any] | None = None
            for t in self._tracks:
                if t in face_of:
                    continue
                for f in unmatched_faces:
                    s = _iou(t.bbox, f.bbox[:4])
                    if s >= self._match_iou and (best is None or s > best[0]):
                        best = (s, t, f)
            if best is None:
                break
            _, t, f = best
            face_of[t] = f
            # Identity-based removal: Face is a dict subclass holding numpy
            # arrays, so list.remove()'s == comparison raises on it.
            unmatched_faces = [x for x in unmatched_faces if x is not f]

        # Distance fallback: fast motion can drop IoU to zero between
        # consecutive frames; accept a face whose centre is within one
        # box-diagonal of the prediction and of comparable size.
        for f in list(unmatched_faces):
            best_d: tuple[float, _Track] | None = None
            fz = _xyxy_to_z(f.bbox[:4])
            for t in self._tracks:
                if t in face_of:
                    continue
                diag = float(np.hypot(t.x[2], t.x[3]))
                d = float(np.hypot(t.x[0] - fz[0], t.x[1] - fz[1]))
                area_ratio = (fz[2] * fz[3]) / max(t.x[2] * t.x[3], 1.0)
                if d < diag and 0.25 < area_ratio < 4.0 and (
                        best_d is None or d < best_d[0]):
                    best_d = (d, t)
            if best_d is not None:
                face_of[best_d[1]] = f
                unmatched_faces = [x for x in unmatched_faces if x is not f]

        for t, f in face_of.items():
            t.correct(f.bbox[:4])
            t.source = "face"

        # Pose assist: only consulted when a live track has no face this
        # frame. Head boxes overlapping an already-matched track are
        # discarded so one person's head cannot correct the other's track.
        head_corrected: set[_Track] = set()
        lost = [t for t in self._tracks if t not in face_of]
        if lost and head_provider is not None:
            heads = [
                hb for hb, _score in head_provider()
                if not any(_iou(hb, t.bbox) > 0.3 for t in face_of)
            ]
            for t in lost:
                if not heads:
                    break
                cand = max(heads, key=lambda hb: _iou(hb, t.bbox))
                score = _iou(cand, t.bbox)
                diag = float(np.hypot(t.x[2], t.x[3]))
                hz = _xyxy_to_z(cand)
                dist = float(np.hypot(t.x[0] - hz[0], t.x[1] - hz[1]))
                if score >= _HEAD_MATCH_IOU or dist < 0.7 * diag:
                    t.correct(cand, weak=True)
                    t.source = "head"
                    head_corrected.add(t)
                    heads = [hb for hb in heads if hb is not cand]

        # Coast uncorrected tracks; drop them once the hold expires or the
        # predicted box has left the frame entirely.
        survivors: list[_Track] = []
        for t in self._tracks:
            if t not in face_of and t not in head_corrected:
                t.source = "coast"
                t.damp_velocity()
                t.misses += 1
            bb = t.bbox
            inside = bb[2] > 0 and bb[0] < fw and bb[3] > 0 and bb[1] < fh
            if t.misses <= self._hold_frames and inside:
                survivors.append(t)
        self._tracks = survivors

        # Births — face detections that matched nothing start new tracks.
        for f in unmatched_faces:
            if len(self._tracks) >= self._max_tracks:
                break
            t = _Track(self._next_id, f.bbox[:4])
            self._next_id += 1
            face_of[t] = f
            self._tracks.append(t)

        return [TrackedFace(t.id, face_of.get(t), t.bbox, t.source)
                for t in self._tracks]
