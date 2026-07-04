"""Kalman tracker, BYTE-style two-stage association — over face-anchored
tracks.

Tracks are driven by gated face candidates (libs/evidence.gate_face_candidates)
via their head-scale *anchor* box (a covering head/pose box, or a face grown
to head proportions); the anchor is Kalman motion food, never a blur target
by itself — only a track with real face evidence behind it renders. Three
properties this tracker has that a plain greedy/static-hold tracker lacks,
each mapped to a failure it caused:

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

# face_age for a track that has never shown a face. Face evidence (the
# detector's face class cross-checked/supplemented by SCRFD) is matched to
# tracks every frame; ``face_age`` counts frames since the last match, so
# callers can require a *recent* face rather than a face-this-frame — the
# temporal smoothing that keeps a blur steady while a head briefly turns or
# the face detector flickers, as long as the track itself stays coherent
# (the association gates are the "body position hasn't changed" test).
FACE_NEVER = 1 << 30


class TrackObs(NamedTuple):
    track_id: int
    box: np.ndarray          # xyxy float32 — KF posterior (hit) or prediction
    score: float             # score of the matched detection (0.0 on coast)
    hit: bool                # matched a detection this frame
    confirmed: bool
    coast_frames: int        # consecutive frames without a detection
    face_age: int = FACE_NEVER   # frames since face evidence matched this track
    # Last matched face box, xyxy in frame coords, re-anchored to the current
    # head box every frame (it is stored relative to the head box, so it rides
    # the Kalman motion between face detections). None until a face matches;
    # check ``face_age`` for freshness before trusting it.
    face_box: "np.ndarray | None" = None
    # Ev bits (libs/evidence.py) of the candidate that drove this frame's
    # hit; 0 on a coast or a sustain-pool (ungated) hit. Feeds the offline
    # evidence ledger — see libs/tracklets.clean_tracklets.
    flags: int = 0


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
    def __init__(self, tid: int, det: np.ndarray, min_hits: int,
                flags: int = 0) -> None:
        self.id = tid
        self.kf = _KalmanBox(_to_z(det))
        self.hits = 1
        self.coast_frames = 0
        self.confirmed = min_hits <= 1
        self.last_score = float(det[4])
        self.hit = True
        self.last_flags = flags
        self.face_age = FACE_NEVER
        # Last matched face box relative to the head box (x1, y1, x2, y2 as
        # fractions of the box), so it follows the Kalman motion between
        # face detections instead of freezing at a stale absolute position.
        self.face_rel: np.ndarray | None = None

    def mark_hit(self, det: np.ndarray, min_hits: int, flags: int = 0) -> None:
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

    def update(self, cands: np.ndarray, frame_shape: tuple,
               faces: np.ndarray | None = None,
               flags: np.ndarray | None = None,
               sustain: np.ndarray | None = None,
               weak_faces: np.ndarray | None = None) -> list[TrackObs]:
        """Advance one frame with (N,5) gated face-anchor candidates; return
        live tracks.

        ``cands`` (xyxy + face score) come from
        ``libs.evidence.gate_face_candidates`` and are the *only* thing that
        may spawn or drive a track — a candidate scoring at or above
        ``det_conf`` may do either; one scoring in ``[det_conf_low,
        det_conf)`` may only sustain (BYTE stage 2). ``flags`` (N,) are each
        candidate's ``Ev`` bits (see libs/evidence.py), aligned 1:1 with
        ``cands``, stored on the matching track and surfaced as
        ``TrackObs.flags`` for the offline evidence ledger.

        ``sustain`` is a *separate* pool of ungated low-score boxes
        (``libs.evidence.sustain_pool``, veto-passed but not gated) that may
        also sustain — never spawn — a track through a confidence dip; it
        carries no flags (an ungated hit contributes no ledger evidence).

        ``faces``/``weak_faces`` are flat face-box pools, spatially matched
        against each track's *current* box after this frame's position
        updates (not index-aligned with ``cands`` — a track's box already
        tells us where to look). ``faces`` is this frame's own gated
        candidates' face boxes and may refresh any track, confirmed or not;
        ``weak_faces`` is raw, ungated face-ish evidence and only refreshes
        already-*confirmed* tracks — an established track's own identity is
        what makes that weaker evidence trustworthy there. Either match
        resets ``face_age`` to 0 and stores the face box relative to the
        head box, so it rides the Kalman motion between face detections.
        """
        fh, fw = frame_shape[:2]
        cands = np.asarray(cands, dtype=np.float32).reshape(-1, 5)
        cand_flags = (np.zeros(len(cands), dtype=np.int64) if flags is None
                     else np.asarray(flags, dtype=np.int64).reshape(-1))
        sustain = (np.empty((0, 5), np.float32) if sustain is None
                  else np.asarray(sustain, dtype=np.float32).reshape(-1, 5))

        for t in self._tracks:
            t.kf.predict()
            t.face_age = min(t.face_age + 1, FACE_NEVER)

        high_mask = cands[:, 4] >= self.det_conf
        high, high_flags = cands[high_mask], cand_flags[high_mask]

        low_mask = (cands[:, 4] >= self.det_conf_low) & ~high_mask
        low_gated, low_gated_flags = cands[low_mask], cand_flags[low_mask]
        if len(sustain):
            s_mask = ((sustain[:, 4] >= self.det_conf_low)
                     & (sustain[:, 4] < self.det_conf))
            sustain = sustain[s_mask]
        low = np.concatenate([low_gated, sustain])
        low_flags = np.concatenate(
            [low_gated_flags, np.zeros(len(sustain), dtype=np.int64)])

        track_boxes = np.array([t.kf.box for t in self._tracks],
                               dtype=np.float32).reshape(-1, 4)

        matched_t: set[int] = set()
        matched_d: set[int] = set()

        # Stage 1 — every live track × high-score candidates, IoU-gated.
        ious = _iou_matrix(track_boxes, high[:, :4])
        for ti, di in _hungarian(1.0 - ious, ious >= _IOU_GATE_HIGH):
            self._tracks[ti].mark_hit(high[di], self.min_hits,
                                      int(high_flags[di]))
            matched_t.add(ti)
            matched_d.add(di)

        # Stage 2 (BYTE) — leftover recently-alive tracks × low-score pool
        # (gated candidates below det_conf, unioned with the ungated sustain
        # pool). Stricter IoU: a low hit may sustain a track, never yank it.
        rem_t = [i for i in range(len(self._tracks))
                 if i not in matched_t and self._tracks[i].coast_frames <= 3]
        if rem_t and len(low):
            boxes_t = track_boxes[rem_t]
            ious = _iou_matrix(boxes_t, low[:, :4])
            for ri, di in _hungarian(1.0 - ious, ious >= _IOU_GATE_LOW):
                self._tracks[rem_t[ri]].mark_hit(low[di], self.min_hits,
                                                 int(low_flags[di]))
                matched_t.add(rem_t[ri])

        # Stage 3 — fast motion: IoU already zero, so match unmatched
        # *confirmed* tracks to leftover high candidates by normalised centre
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
                self._tracks[rem_t[a]].mark_hit(high[rem_d[b]], self.min_hits,
                                                int(high_flags[rem_d[b]]))
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

        # Births — every leftover high candidate is already gated (see the
        # docstring); det_conf is the sole spawn bar now — the evidence gate
        # replaced the old score-margin's job of keeping noise from spawning.
        for di in range(len(high)):
            if di in matched_d or len(self._tracks) >= self._max_tracks:
                continue
            self._tracks.append(
                _Track(self._next_id, high[di], self.min_hits,
                      int(high_flags[di])))
            self._next_id += 1

        def _refresh_faces(pool: np.ndarray | None,
                           confirmed_only: bool) -> None:
            if pool is None or len(pool) == 0:
                return
            pool = np.asarray(pool, dtype=np.float32).reshape(-1, 5)
            fcx = (pool[:, 0] + pool[:, 2]) / 2
            fcy = (pool[:, 1] + pool[:, 3]) / 2
            for t in self._tracks:
                if confirmed_only and not t.confirmed:
                    continue
                b = t.kf.box
                bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                in_t = ((b[0] <= fcx) & (fcx <= b[2])
                        & (b[1] <= fcy) & (fcy <= b[3]))
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

        # Face evidence → per-track memory (spatial join against each
        # track's current box — see the docstring for why ``faces`` may
        # touch any track but ``weak_faces`` only confirmed ones).
        _refresh_faces(faces, confirmed_only=False)
        _refresh_faces(weak_faces, confirmed_only=True)

        def face_box(t: _Track) -> np.ndarray | None:
            if t.face_rel is None:
                return None
            b = t.kf.box
            bw, bh = b[2] - b[0], b[3] - b[1]
            return np.array([b[0] + t.face_rel[0] * bw,
                             b[1] + t.face_rel[1] * bh,
                             b[0] + t.face_rel[2] * bw,
                             b[1] + t.face_rel[3] * bh], dtype=np.float32)

        return [TrackObs(t.id, t.kf.box, t.last_score, t.hit, t.confirmed,
                         t.coast_frames, t.face_age, face_box(t),
                         t.last_flags)
                for t in self._tracks]
