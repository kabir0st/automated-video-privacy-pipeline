"""Kalman head tracker with BYTE-style two-stage association.

Replaces the greedy IoU/static-hold SubjectTracker. Three properties the old
tracker lacked, each mapped to a failure it caused:

  * **Motion model** — a constant-velocity Kalman filter per track. A lost
    head coasts *along its trajectory* instead of freezing in place (the old
    static hold painted ghost blurs at stale positions).
  * **BYTE association** — low-score detections (0.1–0.5) may *sustain* an
    existing track through partial occlusion but can never spawn one. This is
    the anti-flicker mechanism: a head half-hidden behind a shoulder keeps
    scoring ~0.2 and keeps its track alive, while background noise at 0.2
    never becomes a blur.
  * **Confirmation** — a track must be matched ``min_hits`` consecutive
    frames before it is rendered, so a one-frame false positive never
    flashes a blur.

Assignments are solved with the Hungarian method (scipy) instead of greedy
matching, so two heads close together (the common case in this footage)
cannot steal each other's detection just because one pair was scored first.

Coasting tracks are *reported* (the offline pass in libs/tracklets.py uses
them as gap-bridging candidates) but callers must only render confirmed,
recently-hit tracks — see ``TrackObs.hit``/``coast_frames``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ByteTrack's noise convention: std devs proportional to box height.
_STD_POS = 1.0 / 20.0
_STD_VEL = 1.0 / 160.0
# Velocity damping per coasted frame — bounds drift over a long coast.
_COAST_DAMP = 0.95

# Association gates.
_IOU_GATE_HIGH = 0.20    # stage 1: track prediction × high-score dets
_IOU_GATE_LOW = 0.30     # stage 2 (BYTE): stricter — low dets only sustain
_CENTRE_GATE = 0.80      # stage 3: centre distance / track diagonal
_SIZE_GATE = (0.5, 2.0)  # stage 3: w and h ratio bounds

_MAX_TRACKS = 16         # bounds pathological detector spam


class TrackObs(NamedTuple):
    track_id: int
    box: np.ndarray          # xyxy float32 — KF posterior (hit) or prediction
    score: float             # score of the matched detection (0.0 on coast)
    hit: bool                # matched a detection this frame
    confirmed: bool
    coast_frames: int        # consecutive frames without a detection


class _KalmanBox:
    """Constant-velocity Kalman filter over ``[cx, cy, w, h]``.

    State is 8-D ``[cx, cy, w, h, vcx, vcy, vw, vh]``; w/h are filtered
    directly (heads keep near-constant aspect, and w/h states smooth and
    interpolate trivially downstream). Noise scales with box height per
    ByteTrack so the filter is resolution-independent.
    """

    def __init__(self, z: np.ndarray) -> None:  # z = [cx, cy, w, h]
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = z
        h = max(float(z[3]), 1.0)
        self.P = np.diag([(2 * _STD_POS * h) ** 2] * 4
                         + [(10 * _STD_VEL * h) ** 2] * 4)
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0

    def predict(self) -> None:
        h = max(float(self.x[3]), 1.0)
        Q = np.diag([(_STD_POS * h) ** 2] * 4 + [(_STD_VEL * h) ** 2] * 4)
        self.x = self._F @ self.x
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.P = self._F @ self.P @ self._F.T + Q

    def damp_velocity(self) -> None:
        self.x[4:] *= _COAST_DAMP

    def update(self, z: np.ndarray) -> None:
        h = max(float(self.x[3]), 1.0)
        R = np.diag([(_STD_POS * h) ** 2] * 4)
        y = z - self.x[:4]
        S = self.P[:4, :4] + R
        K = self.P[:, :4] @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.P = self.P - K @ self.P[:4, :]

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        dtype=np.float32)


def _to_z(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                     box[2] - box[0], box[3] - box[1]], dtype=np.float64)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def _hungarian(cost: np.ndarray, gate: np.ndarray) -> list[tuple[int, int]]:
    """Optimal assignment on ``cost`` where ``gate`` is True; others excluded.

    Gated-out pairs get an infinite-ish cost, and assignments landing on them
    are discarded afterwards, so an all-invalid row/column simply stays
    unmatched.
    """
    if cost.size == 0:
        return []
    BIG = 1e6
    c = np.where(gate, cost, BIG)
    rows, cols = linear_sum_assignment(c)
    return [(r, col) for r, col in zip(rows, cols) if c[r, col] < BIG]


class _Track:
    def __init__(self, tid: int, det: np.ndarray, min_hits: int) -> None:
        self.id = tid
        self.kf = _KalmanBox(_to_z(det))
        self.hits = 1
        self.coast_frames = 0
        self.confirmed = min_hits <= 1
        self.last_score = float(det[4])
        self.hit = True

    def mark_hit(self, det: np.ndarray, min_hits: int) -> None:
        self.kf.update(_to_z(det))
        self.hits += 1
        self.coast_frames = 0
        self.last_score = float(det[4])
        self.hit = True
        if self.hits >= min_hits:
            self.confirmed = True

    def mark_miss(self) -> None:
        self.kf.damp_velocity()
        self.coast_frames += 1
        self.last_score = 0.0
        self.hit = False


class HeadTracker:
    """Tracks head boxes across frames; see the module docstring."""

    def __init__(
        self,
        fps: float = 30.0,
        det_conf: float = 0.50,
        det_conf_low: float = 0.10,
        min_hits: int = 3,
        max_age_s: float = 1.75,
        max_tracks: int = _MAX_TRACKS,
    ) -> None:
        self._fps = max(fps, 1.0)
        self.det_conf = det_conf
        self.det_conf_low = det_conf_low
        self.min_hits = max(1, int(min_hits))
        self.max_age_s = max_age_s
        self._max_tracks = max_tracks
        self._tracks: list[_Track] = []
        self._next_id = 1

    def configure(self, det_conf: float | None = None,
                  det_conf_low: float | None = None,
                  min_hits: int | None = None,
                  max_age_s: float | None = None) -> None:
        if det_conf is not None:
            self.det_conf = det_conf
        if det_conf_low is not None:
            self.det_conf_low = det_conf_low
        if min_hits is not None:
            self.min_hits = max(1, int(min_hits))
        if max_age_s is not None:
            self.max_age_s = max_age_s

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    @property
    def _max_age(self) -> int:
        return max(1, int(round(self.max_age_s * self._fps)))

    def update(self, heads: np.ndarray, frame_shape: tuple) -> list[TrackObs]:
        """Advance one frame with (N,5) head candidates; return live tracks."""
        fh, fw = frame_shape[:2]
        heads = np.asarray(heads, dtype=np.float32).reshape(-1, 5)

        for t in self._tracks:
            t.kf.predict()

        high = heads[heads[:, 4] >= self.det_conf]
        low = heads[(heads[:, 4] >= self.det_conf_low)
                    & (heads[:, 4] < self.det_conf)]

        track_boxes = np.array([t.kf.box for t in self._tracks],
                               dtype=np.float32).reshape(-1, 4)

        matched_t: set[int] = set()
        matched_d: set[int] = set()

        # Stage 1 — every live track × high-score detections, IoU-gated.
        ious = _iou_matrix(track_boxes, high[:, :4])
        for ti, di in _hungarian(1.0 - ious, ious >= _IOU_GATE_HIGH):
            self._tracks[ti].mark_hit(high[di], self.min_hits)
            matched_t.add(ti)
            matched_d.add(di)

        # Stage 2 (BYTE) — leftover recently-alive tracks × low-score dets.
        # Stricter IoU: a low det may sustain a track, never yank it far.
        rem_t = [i for i in range(len(self._tracks))
                 if i not in matched_t and self._tracks[i].coast_frames <= 3]
        if rem_t and len(low):
            boxes_t = track_boxes[rem_t]
            ious = _iou_matrix(boxes_t, low[:, :4])
            for ri, di in _hungarian(1.0 - ious, ious >= _IOU_GATE_LOW):
                self._tracks[rem_t[ri]].mark_hit(low[di], self.min_hits)
                matched_t.add(rem_t[ri])

        # Stage 3 — fast motion: IoU already zero, so match unmatched
        # *confirmed* tracks to leftover high dets by normalised centre
        # distance, with a size-similarity gate. Hungarian keeps two close
        # heads from stealing each other's detection.
        rem_t = [i for i in range(len(self._tracks))
                 if i not in matched_t and self._tracks[i].confirmed]
        rem_d = [i for i in range(len(high)) if i not in matched_d]
        if rem_t and rem_d:
            cost = np.full((len(rem_t), len(rem_d)), 10.0, dtype=np.float32)
            gate = np.zeros_like(cost, dtype=bool)
            for a, ti in enumerate(rem_t):
                tb = track_boxes[ti]
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                diag = float(np.hypot(tw, th)) or 1.0
                tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                for b, di in enumerate(rem_d):
                    d = high[di]
                    dw, dh = d[2] - d[0], d[3] - d[1]
                    dist = np.hypot((d[0] + d[2]) / 2 - tcx,
                                    (d[1] + d[3]) / 2 - tcy) / diag
                    ok_w = _SIZE_GATE[0] <= dw / max(tw, 1e-3) <= _SIZE_GATE[1]
                    ok_h = _SIZE_GATE[0] <= dh / max(th, 1e-3) <= _SIZE_GATE[1]
                    cost[a, b] = dist
                    gate[a, b] = dist < _CENTRE_GATE and ok_w and ok_h
            for a, b in _hungarian(cost, gate):
                self._tracks[rem_t[a]].mark_hit(high[rem_d[b]], self.min_hits)
                matched_t.add(rem_t[a])
                matched_d.add(rem_d[b])

        # Lifecycle: misses, deaths.
        survivors: list[_Track] = []
        for i, t in enumerate(self._tracks):
            if i not in matched_t:
                t.mark_miss()
                # An unconfirmed track gets no benefit of the doubt: one miss
                # during warm-up and it dies (kills one-frame false blurs).
                if not t.confirmed:
                    continue
                if t.coast_frames > self._max_age:
                    continue
                cx = float(t.kf.x[0])
                cy = float(t.kf.x[1])
                if not (0 <= cx < fw and 0 <= cy < fh):
                    continue    # coasted out of frame
            survivors.append(t)
        self._tracks = survivors

        # Births — only leftover *high* detections confident enough to spawn.
        spawn_thr = min(self.det_conf + 0.10, 0.95)
        for di in range(len(high)):
            if di in matched_d or len(self._tracks) >= self._max_tracks:
                continue
            if high[di, 4] >= spawn_thr:
                self._tracks.append(
                    _Track(self._next_id, high[di], self.min_hits))
                self._next_id += 1

        return [TrackObs(t.id, t.kf.box, t.last_score, t.hit, t.confirmed,
                         t.coast_frames)
                for t in self._tracks]
