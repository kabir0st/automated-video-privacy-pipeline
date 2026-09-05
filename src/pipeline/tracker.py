"""Online head tracker: constant-velocity Kalman per track, three-stage
association (Hungarian on IoU for confident candidates, BYTE-style second
pass so weak candidates *sustain* a track, centre-distance rescue for fast
motion), long coasting memory.

Policy differences from a precision-first tracker, all deliberate:

* the spawn bar is low (``spawn_conf``) and one confirmation hit is enough
  by default — a missed head costs more than a spurious one;
* a lost track coasts for ``max_age_s`` as a *bridge candidate* (it is not
  blurred while coasting; ``refine`` decides whether the gap joins);
* face evidence is remembered per track relative to the head box, so the
  face-tight blur mode can follow the face between face detections.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geom import iou_matrix
from .types import Src

_STD_POS = 1.0 / 20.0
_STD_VEL = 1.0 / 160.0
_COAST_DAMP = 0.95

_IOU_GATE_HIGH = 0.15
_IOU_GATE_LOW = 0.25
_CENTRE_GATE = 1.0          # centre distance / track diagonal (stage 3)
_SIZE_GATE = (0.4, 2.5)     # stage 3 (fast motion)
# Size sanity for IoU matches. A torso-sized box *contains* a head, so its
# IoU with the head track can clear the gate; without this a big weak box
# would "sustain" the track and drag the Kalman box up to torso size.
_SIZE_GATE_HIGH = (0.5, 2.0)
_SIZE_GATE_LOW = (0.6, 1.7)
_MAX_TRACKS = 24
FACE_NEVER = 1 << 30


class TrackObs(NamedTuple):
    track_id: int
    box: np.ndarray          # xyxy float32 — posterior (hit) or prediction
    score: float
    hit: bool
    confirmed: bool
    coast_frames: int
    face_age: int = FACE_NEVER
    face_box: "np.ndarray | None" = None
    flags: int = 0


class _KalmanBox:
    """CV Kalman over ``[cx, cy, w, h, vcx, vcy, vw, vh]``; noise scales with
    box height so it is resolution-independent."""

    def __init__(self, z: np.ndarray) -> None:
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


def _size_ok(tracks: np.ndarray, dets: np.ndarray,
             lo: float, hi: float) -> np.ndarray:
    """(T, D) bool — det w and h within [lo, hi] × the track's."""
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)), bool)
    tw = np.maximum(tracks[:, 2] - tracks[:, 0], 1e-3)[:, None]
    th = np.maximum(tracks[:, 3] - tracks[:, 1], 1e-3)[:, None]
    dw = (dets[:, 2] - dets[:, 0])[None, :]
    dh = (dets[:, 3] - dets[:, 1])[None, :]
    rw, rh = dw / tw, dh / th
    return (rw >= lo) & (rw <= hi) & (rh >= lo) & (rh <= hi)


def _hungarian(cost: np.ndarray, gate: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    BIG = 1e6
    c = np.where(gate, cost, BIG)
    rows, cols = linear_sum_assignment(c)
    return [(r, col) for r, col in zip(rows, cols) if c[r, col] < BIG]


class _Track:
    def __init__(self, tid: int, det: np.ndarray, min_hits: int,
                 flags: int) -> None:
        self.id = tid
        self.kf = _KalmanBox(_to_z(det))
        self.hits = 1
        self.coast_frames = 0
        self.confirmed = min_hits <= 1
        self.last_score = float(det[4])
        self.hit = True
        self.last_flags = flags
        self.face_age = FACE_NEVER
        self.face_rel: np.ndarray | None = None

    def mark_hit(self, det: np.ndarray, min_hits: int, flags: int) -> None:
        self.kf.update(_to_z(det))
        self.hits += 1
        self.coast_frames = 0
        self.last_score = float(det[4])
        self.hit = True
        self.last_flags = flags
        if self.hits >= min_hits:
            self.confirmed = True

    def mark_miss(self) -> None:
        self.kf.damp_velocity()
        self.coast_frames += 1
        self.last_score = 0.0
        self.hit = False
        self.last_flags = 0


class Tracker:
    def __init__(self, fps: float = 25.0, spawn_conf: float = 0.35,
                 sustain_conf: float = 0.10, min_hits: int = 2,
                 max_age_s: float = 2.5, max_tracks: int = _MAX_TRACKS) -> None:
        self._fps = max(fps, 1.0)
        self.spawn_conf = spawn_conf
        self.sustain_conf = sustain_conf
        self.min_hits = max(1, int(min_hits))
        self.max_age_s = max_age_s
        self._max_tracks = max_tracks
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    @property
    def _max_age(self) -> int:
        return max(1, int(round(self.max_age_s * self._fps)))

    def update(self, cands: np.ndarray, flags: np.ndarray,
               frame_shape: tuple, faces: np.ndarray | None = None
               ) -> list[TrackObs]:
        """Advance one frame with fused head candidates ``(N, 5)`` and their
        ``Src`` bits; return every live track (coasting ones included)."""
        fh, fw = frame_shape[:2]
        cands = np.asarray(cands, np.float32).reshape(-1, 5)
        flags = (np.zeros(len(cands), np.int64) if flags is None
                 else np.asarray(flags, np.int64).reshape(-1))

        for t in self._tracks:
            t.kf.predict()
            t.face_age = min(t.face_age + 1, FACE_NEVER)

        high_m = cands[:, 4] >= self.spawn_conf
        low_m = (cands[:, 4] >= self.sustain_conf) & ~high_m
        high, high_f = cands[high_m], flags[high_m]
        low, low_f = cands[low_m], flags[low_m] | int(Src.SUSTAIN)

        track_boxes = np.array([t.kf.box for t in self._tracks],
                               np.float32).reshape(-1, 4)
        matched_t: set[int] = set()
        matched_d: set[int] = set()

        # Stage 1 — live tracks × confident candidates, IoU-gated.
        ious = iou_matrix(track_boxes, high[:, :4])
        gate1 = (ious >= _IOU_GATE_HIGH) & _size_ok(track_boxes, high, *_SIZE_GATE_HIGH)
        for ti, di in _hungarian(1.0 - ious, gate1):
            self._tracks[ti].mark_hit(high[di], self.min_hits,
                                      int(high_f[di]))
            matched_t.add(ti)
            matched_d.add(di)

        # Stage 2 (BYTE) — recently-alive leftovers × weak candidates.
        rem_t = [i for i in range(len(self._tracks))
                 if i not in matched_t and self._tracks[i].coast_frames <= 5]
        if rem_t and len(low):
            ious = iou_matrix(track_boxes[rem_t], low[:, :4])
            gate2 = (ious >= _IOU_GATE_LOW) & _size_ok(track_boxes[rem_t], low,
                                                       *_SIZE_GATE_LOW)
            for ri, di in _hungarian(1.0 - ious, gate2):
                self._tracks[rem_t[ri]].mark_hit(low[di], self.min_hits,
                                                 int(low_f[di]))
                matched_t.add(rem_t[ri])

        # Stage 3 — fast motion: unmatched confirmed tracks × leftover
        # confident candidates by normalised centre distance + size gate.
        rem_t = [i for i in range(len(self._tracks))
                 if i not in matched_t and self._tracks[i].confirmed]
        rem_d = [i for i in range(len(high)) if i not in matched_d]
        if rem_t and rem_d:
            tb = track_boxes[rem_t]
            tw, th = tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]
            diag = np.maximum(np.hypot(tw, th), 1.0)
            tcx, tcy = (tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2
            d = high[rem_d]
            dw, dh = d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]
            dcx, dcy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
            dist = np.hypot(dcx[None, :] - tcx[:, None],
                            dcy[None, :] - tcy[:, None]) / diag[:, None]
            rw = dw[None, :] / np.maximum(tw[:, None], 1e-3)
            rh = dh[None, :] / np.maximum(th[:, None], 1e-3)
            lo, hi = _SIZE_GATE
            gate = ((dist < _CENTRE_GATE) & (rw >= lo) & (rw <= hi)
                    & (rh >= lo) & (rh <= hi))
            for a, b in _hungarian(dist.astype(np.float32), gate):
                self._tracks[rem_t[a]].mark_hit(high[rem_d[b]], self.min_hits,
                                                int(high_f[rem_d[b]]))
                matched_t.add(rem_t[a])
                matched_d.add(rem_d[b])

        # Lifecycle.
        survivors: list[_Track] = []
        for i, t in enumerate(self._tracks):
            if i not in matched_t:
                t.mark_miss()
                if not t.confirmed and t.coast_frames > 1:
                    continue
                if t.coast_frames > self._max_age:
                    continue
                cx, cy = float(t.kf.x[0]), float(t.kf.x[1])
                if not (0 <= cx < fw and 0 <= cy < fh):
                    continue
            survivors.append(t)
        self._tracks = survivors

        # Births from leftover confident candidates.
        for di in range(len(high)):
            if di in matched_d or len(self._tracks) >= self._max_tracks:
                continue
            self._tracks.append(_Track(self._next_id, high[di], self.min_hits,
                                       int(high_f[di])))
            self._next_id += 1

        # Face memory: any raw face whose centre sits in a track's box (or
        # vice versa) refreshes that track's face, stored relative to the
        # head box so it rides the Kalman motion between sightings.
        if faces is not None and len(faces):
            pool = np.asarray(faces, np.float32).reshape(-1, 5)
            fcx, fcy = (pool[:, 0] + pool[:, 2]) / 2, (pool[:, 1] + pool[:, 3]) / 2
            for t in self._tracks:
                b = t.kf.box
                bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                in_t = ((b[0] <= fcx) & (fcx <= b[2]) & (b[1] <= fcy)
                        & (fcy <= b[3]))
                t_in = ((pool[:, 0] <= bcx) & (bcx <= pool[:, 2])
                        & (pool[:, 1] <= bcy) & (bcy <= pool[:, 3]))
                m = in_t | t_in
                if bool(m.any()):
                    t.face_age = 0
                    best = pool[m][np.argmax(pool[m][:, 4])]
                    bw = max(b[2] - b[0], 1e-3)
                    bh = max(b[3] - b[1], 1e-3)
                    t.face_rel = np.array(
                        [(best[0] - b[0]) / bw, (best[1] - b[1]) / bh,
                         (best[2] - b[0]) / bw, (best[3] - b[1]) / bh],
                        dtype=np.float32)

        def face_box(t: _Track) -> np.ndarray | None:
            if t.face_rel is None:
                return None
            b = t.kf.box
            bw, bh = b[2] - b[0], b[3] - b[1]
            return np.array([b[0] + t.face_rel[0] * bw, b[1] + t.face_rel[1] * bh,
                             b[0] + t.face_rel[2] * bw, b[1] + t.face_rel[3] * bh],
                            dtype=np.float32)

        return [TrackObs(t.id, t.kf.box, t.last_score, t.hit, t.confirmed,
                         t.coast_frames, t.face_age, face_box(t), t.last_flags)
                for t in self._tracks]
