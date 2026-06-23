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
    damped each frame so the box cannot sail away).

A track is dropped after `hold_secs` without a *face* correction (a pose head
box may reposition a momentarily-lost track but does not prolong it) or when it
leaves the frame. Were head boxes to reset the expiry clock, a body that stays
in frame would revive a faceless track forever — blurring a bare body long
after the face was gone.

Tracks are normally only ever *created* from a face detection, so the pose
model cannot start blurring someone whose face was never seen.

The one deliberate exception is the **privacy safety-net anchor** (opt-in via
``anchor_provider``). RF-DETR person boxes (see libs/person_detector.py) carry
head-region boxes that *may* spawn a track and sustain a blur even when SCRFD
and RTMW both fail on a person who is plainly present (head turned fully away,
motion blur). For a privacy tool a missed face is the failure that matters, so
this trades a small over-blur risk for fewer leaks — and it is bounded:

  * an anchor only spawns where it overlaps no existing track (no double-blur);
  * the instant a real face matches an anchor track it becomes an ordinary face
    track (SCRFD's tight hull takes over);
  * a track stays alive while a *face* was seen within ``hold_secs`` OR a person
    anchor within ``anchor_hold_secs`` — so a lost-face track is sustained only
    as long as RF-DETR still sees that person, then it expires.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np

_MATCH_IOU = 0.3
# Pose head boxes are coarse, but reviving a lost track on a near-miss let a
# stray/oversized head box drag a track onto empty space. Require a real
# overlap, and only fall back to centre distance when it is tight AND the sizes
# are comparable.
_HEAD_MATCH_IOU = 0.3
# Cap how much a single weak (pose head box) correction may grow a track's box.
_WEAK_GROW_MAX = 1.3
# Per-coast-frame velocity decay — keeps an undetected box from drifting
# off across the frame at its last observed speed.
_COAST_VEL_DAMP = 0.92
# A person anchor must overlap a track this much to sustain it (same person),
# and must overlap *no* track this much before it may spawn a fresh anchor
# track (so an anchor never duplicates an existing blur).
_ANCHOR_MATCH_IOU = 0.2
_ANCHOR_SPAWN_IOU = 0.2
# Sentinel for "never anchor-corrected" — large enough that the OR-aliveness
# rule ignores it until an anchor actually resets it to 0.
_NEVER = 1 << 30

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
        self.face_misses = 0     # frames since last *face* correction
        # Frames since last person-anchor correction. Starts "never" so a
        # face-born track is held purely by the face rule unless an anchor
        # actually sustains it.
        self.anchor_misses = _NEVER
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
        w0, h0 = float(self.x[2]), float(self.x[3])
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P
        if weak:
            # A coarse head box must never balloon a face-sized track.
            self.x[2] = min(self.x[2], w0 * _WEAK_GROW_MAX)
            self.x[3] = min(self.x[3], h0 * _WEAK_GROW_MAX)
        if not weak:        # only a real face match resets the expiry clock
            self.face_misses = 0


class KalmanFaceTracker:
    """Tracks a handful of faces (the app targets at most two people).

    update() takes this frame's filtered InsightFace detections plus an
    optional `head_provider` callable. The provider is invoked whenever
    there is a live track to assist: a matched track takes a weak nudge from
    its own head box, a track that lost its face is revived by one. It is
    skipped only when there are no tracks at all.
    """

    def __init__(
        self,
        fps: float = 30.0,
        match_iou: float = _MATCH_IOU,
        hold_secs: float = 0.6,
        # A track sustained only by a person anchor (no face) is held this long
        # after the person also disappears — longer than hold_secs because the
        # whole point is to keep covering a person whose face we never see.
        anchor_hold_secs: float = 2.0,
        # Privacy: every detected face must get a track (and therefore a
        # blur) — the cap only bounds pathological detector spam.
        max_tracks: int = 16,
    ) -> None:
        self._fps = max(fps, 1.0)
        self._match_iou = match_iou
        self._hold_secs = hold_secs
        self._anchor_hold_secs = anchor_hold_secs
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

    @property
    def _anchor_hold_frames(self) -> int:
        return max(1, int(round(self._anchor_hold_secs * self._fps)))

    def update(
        self,
        faces: list[Any],
        frame_shape: tuple[int, ...],
        head_provider: Optional[Callable[[], list[tuple[np.ndarray, float]]]] = None,
        anchor_provider: Optional[Callable[[], list[tuple[np.ndarray, float]]]] = None,
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

        # Pose assist: head boxes correct *all* live tracks. A face-matched
        # track takes a weak secondary nudge from its own overlapping head box
        # (steadier size/position heading into a detection gap; its "face" tag
        # is kept). Remaining heads revive tracks that had no face this frame.
        # Heads are claimed one per track so one person's head can never
        # correct another's track.
        head_corrected: set[_Track] = set()
        if self._tracks and head_provider is not None:
            heads = [hb for hb, _score in head_provider()]
            # Matched tracks first: only a clearly-overlapping head is theirs.
            for t in face_of:
                if not heads:
                    break
                cand = max(heads, key=lambda hb: _iou(hb, t.bbox))
                if _iou(cand, t.bbox) > 0.3:
                    t.correct(cand, weak=True)
                    heads = [hb for hb in heads if hb is not cand]
            # Lost tracks: revive on loose overlap or centre proximity.
            lost = [t for t in self._tracks if t not in face_of]
            for t in lost:
                if not heads:
                    break
                cand = max(heads, key=lambda hb: _iou(hb, t.bbox))
                score = _iou(cand, t.bbox)
                diag = float(np.hypot(t.x[2], t.x[3]))
                hz = _xyxy_to_z(cand)
                dist = float(np.hypot(t.x[0] - hz[0], t.x[1] - hz[1]))
                area_ratio = (hz[2] * hz[3]) / max(t.x[2] * t.x[3], 1.0)
                close = dist < 0.35 * diag and 0.3 < area_ratio < 3.0
                if score >= _HEAD_MATCH_IOU or close:
                    t.correct(cand, weak=True)
                    t.source = "head"
                    head_corrected.add(t)
                    heads = [hb for hb in heads if hb is not cand]

        # Privacy safety-net: person anchors sustain or spawn blur where faces
        # are missing. An anchor that overlaps a face-less track keeps it alive
        # (a person we see but whose face we lost); a leftover anchor overlapping
        # no track spawns a fresh anchor track. Anchors never touch a track that
        # matched a face this frame — its own face already positions it.
        anchor_corrected: set[_Track] = set()
        if anchor_provider is not None:
            remaining = list(anchor_provider())  # [(box, score)]
            for t in self._tracks:
                if t in face_of or not remaining:
                    continue
                best = max(remaining, key=lambda bs: _iou(bs[0], t.bbox))
                if _iou(best[0], t.bbox) > _ANCHOR_MATCH_IOU:
                    t.correct(best[0], weak=True)
                    t.anchor_misses = 0
                    t.source = "anchor"
                    anchor_corrected.add(t)
                    remaining = [bs for bs in remaining if bs is not best]
            for box, _score in remaining:
                if len(self._tracks) >= self._max_tracks:
                    break
                # Skip if it overlaps a live track or a face that is about to
                # spawn its own track this frame — anchors never double-blur.
                if any(_iou(box, t.bbox) > _ANCHOR_SPAWN_IOU for t in self._tracks):
                    continue
                if any(_iou(box, np.asarray(f.bbox[:4], dtype=np.float32))
                       > _ANCHOR_SPAWN_IOU for f in unmatched_faces):
                    continue
                nt = _Track(self._next_id, box)
                self._next_id += 1
                nt.anchor_misses = 0
                nt.source = "anchor"
                self._tracks.append(nt)
                anchor_corrected.add(nt)

        # Age every track that saw no face this frame (head-corrected ones
        # included — a head box repositions but must not prolong); coast the
        # ones with no correction at all. A track survives while a face was seen
        # within hold_frames OR a person anchor within anchor_hold_frames; drop
        # it once both lapse or the predicted box has left the frame entirely.
        survivors: list[_Track] = []
        for t in self._tracks:
            if t not in face_of:
                t.face_misses += 1
                if t not in head_corrected and t not in anchor_corrected:
                    t.source = "coast"
                    t.damp_velocity()
            if t not in anchor_corrected:
                t.anchor_misses += 1
            bb = t.bbox
            inside = bb[2] > 0 and bb[0] < fw and bb[3] > 0 and bb[1] < fh
            alive = (t.face_misses <= self._hold_frames
                     or t.anchor_misses <= self._anchor_hold_frames)
            if alive and inside:
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
