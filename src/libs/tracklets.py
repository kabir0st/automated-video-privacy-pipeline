"""Pass-1 track recording and offline tracklet cleanup for two-pass export.

The export runs in two passes (see ui.py). Pass 1 detects and tracks but
renders nothing; every frame's ``TrackObs`` land here. Between passes,
:func:`postprocess` turns the raw tracklets into a per-frame render table
using knowledge a streaming tracker can never have — the future:

  * **Trim** — coasted tails are cut; a Kalman prediction that never met
    another detection was a guess, and guesses aren't blurred (the old
    static-hold ghost fix).
  * **Prune** — tracklets too short or never confident are detector noise; a
    one-second-late prune here beats a three-frame-late blur onset in a
    streaming tracker, because pass 2 rewinds time.
  * **Verify** — every tracklet that survives the prune must *reproduce*
    under re-inference: sample a few of its hit frames, crop around its box
    with context, and ask the detector again on the magnified crop
    (:func:`verify_tracklets`). A real head re-detects stronger when it
    fills the input; a hallucination (bed corner, curtain fold) doesn't
    reproduce at the same spot. This kills persistent false positives that
    score-and-length gates can never catch — a hallucination on a static
    scene is long *and* confident — without touching thresholds, so recall
    is untouched.
  * **Bridge** — a track that vanishes and reappears nearby (head briefly
    buried in a pillow / behind a shoulder) is re-joined and the gap is
    *interpolated along the path*, so the blur follows the head instead of
    flickering off and on. Gates guard against joining two different heads:
    velocity-extrapolated distance, size similarity, a corridor test (never
    bridge through a region another surviving track occupies), and an
    ambiguity test (two plausible predecessors → bridge neither).
  * **Smooth** — zero-phase Savitzky-Golay on cx/cy/w/h. Offline smoothing has
    no lag, so the blur is both steady *and* on target.

Boxes are recorded in full-resolution frame coordinates so pass 2 needs no
scale bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.signal import savgol_filter

# Prune gates (see postprocess).
_MIN_HIT_RATIO = 0.30
# Verification gates (see verify_tracklets).
_VERIFY_SAMPLES = 5      # hit frames sampled per tracklet
_VERIFY_MIN_FRAC = 0.4   # fraction of tested samples that must re-detect
# Crop side = max(box w, h) × this. Must be ≥ ~2.5: the detector's rotated
# passes reject heads above ~20 % of the *image* area, and the image here is
# the crop — at ×3 a true head is ~11 % of it, safely under that cap even for
# sideways heads that only the rotated pass can re-detect.
_VERIFY_MARGIN = 3.0
_VERIFY_MIN_CROP = 96    # px — never crop tighter than this
_VERIFY_SCORE = 0.50     # re-detection floor when face evidence agrees
# Floor without face evidence in the crop. Skin misread as a head (chest,
# thigh) *reproduces* under magnification — unlike texture hallucinations —
# but rarely this confidently; a real back-of-head magnified ×3 does. This
# is the safe form of "check it has a face": face evidence lowers the bar,
# its absence raises it, it never zeroes a tracklet by itself.
_VERIFY_LONE = 0.65
# Bridge gates.
_BRIDGE_SIZE_RATIO = (0.6, 1.67)
_CORRIDOR_IOU = 0.20
_AMBIGUITY_MARGIN = 0.20
# Interpolated segments get up to this much extra size mid-gap (uncertainty).
_GAP_PAD = 0.15
# Head-enters-frame lead-in: extend each tracklet by this many seconds of
# held first/last box (covers detector spin-up; blurring early is safe).
_EXTEND_S = 0.12


@dataclass
class Tracklet:
    tid: int
    start: int              # first frame index
    boxes: np.ndarray       # (T, 4) float32 cx, cy, w, h — contiguous frames
    scores: np.ndarray      # (T,)
    hits: np.ndarray        # (T,) bool — False = coasted or interpolated
    # (T,) bool — face evidence matched the track that frame; None = no
    # evidence was collected (face gating must then not judge this tracklet).
    faces: Optional[np.ndarray] = None

    @property
    def end(self) -> int:
        return self.start + len(self.boxes) - 1


@dataclass
class PostParams:
    """Offline-cleanup knobs, mirrored from the GUI Params."""
    det_conf: float = 0.50
    min_hits: int = 3
    min_track_s: float = 0.25
    bridge_gap_s: float = 1.5
    smooth_win_s: float = 0.5
    # Drop tracklets that never once showed a face across their whole
    # (bridged) lifetime — one sighting anywhere marks the entire track,
    # turned-away spans included. The caller must only enable this while
    # face-evidence collection is actually live (SCRFD up), or real heads
    # would be judged on missing data.
    require_face: bool = False


class TrackRecorder:
    """Pass-1 sink: collects per-frame TrackObs into contiguous tracklets."""

    def __init__(self) -> None:
        # tid -> [start, [boxes cxcywh], [scores], [hits]]
        self._open: dict[int, list] = {}
        self._done: list[Tracklet] = []
        self._last_frame = -1

    def observe(self, frame_idx: int, obs: list, inv_scale: float = 1.0) -> None:
        """Record one frame of tracker output (boxes scaled by ``inv_scale``
        back to full resolution). Tracks absent this frame are closed."""
        seen: set[int] = set()
        for o in obs:
            b = np.asarray(o.box, dtype=np.float32) * inv_scale
            z = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                          b[2] - b[0], b[3] - b[1]], dtype=np.float32)
            rec = self._open.get(o.track_id)
            if rec is None:
                self._open[o.track_id] = [frame_idx, [z], [o.score], [o.hit]]
            else:
                # Guard against a recycled id after the tracker dropped the
                # track for exactly one frame boundary we didn't see: pad any
                # hole with the previous box so tracklets stay contiguous.
                expect = rec[0] + len(rec[1])
                while expect < frame_idx:
                    rec[1].append(rec[1][-1].copy())
                    rec[2].append(0.0)
                    rec[3].append(False)
                    expect += 1
                rec[1].append(z)
                rec[2].append(o.score)
                rec[3].append(bool(o.hit))
            seen.add(o.track_id)
        for tid in list(self._open):
            if tid not in seen:
                self._close(tid)
        self._last_frame = frame_idx

    def _close(self, tid: int) -> None:
        start, boxes, scores, hits = self._open.pop(tid)
        self._done.append(Tracklet(
            tid, start,
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32),
            np.asarray(hits, dtype=bool)))

    def finalize(self) -> list[Tracklet]:
        for tid in list(self._open):
            self._close(tid)
        return self._done


def _trim(t: Tracklet) -> Tracklet | None:
    """Cut everything after the last hit (and before the first)."""
    idx = np.nonzero(t.hits)[0]
    if len(idx) == 0:
        return None
    a, b = int(idx[0]), int(idx[-1]) + 1
    return Tracklet(t.tid, t.start + a, t.boxes[a:b], t.scores[a:b],
                    t.hits[a:b])


def _boxes_at(t: Tracklet, frame: int) -> np.ndarray | None:
    if t.start <= frame <= t.end:
        return t.boxes[frame - t.start]
    return None


def _iou_cxcywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return float(inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9))


def _tail_velocity(t: Tracklet, n: int = 5) -> np.ndarray:
    """Mean per-frame (dcx, dcy) over the last up-to-n frames."""
    k = min(n, len(t.boxes) - 1)
    if k < 1:
        return np.zeros(2, dtype=np.float32)
    d = t.boxes[-1, :2] - t.boxes[-1 - k, :2]
    return d / k


def _bridge_candidates(
    tracklets: list[Tracklet], gap_max: int
) -> list[tuple[float, int, int]]:
    """Admissible (cost, pred_idx, succ_idx) bridge pairs, all gates applied."""
    cands: list[tuple[float, int, int]] = []
    for i, a in enumerate(tracklets):
        va = _tail_velocity(a)
        diag = float(np.hypot(a.boxes[-1, 2], a.boxes[-1, 3])) or 1.0
        for j, b in enumerate(tracklets):
            if i == j:
                continue
            g = b.start - a.end - 1
            if g < 0 or g > gap_max:
                continue
            # Distance gate on the velocity-extrapolated landing point; the
            # allowance widens with gap length (uncertainty grows).
            pred = a.boxes[-1, :2] + va * (g + 1)
            dist = float(np.hypot(*(pred - b.boxes[0, :2])))
            if dist > diag * (0.5 + 0.5 * g / max(gap_max, 1)):
                continue
            rw = b.boxes[0, 2] / max(a.boxes[-1, 2], 1e-3)
            rh = b.boxes[0, 3] / max(a.boxes[-1, 3], 1e-3)
            lo, hi = _BRIDGE_SIZE_RATIO
            if not (lo <= rw <= hi and lo <= rh <= hi):
                continue
            # Corridor gate: never bridge across a gap that another surviving
            # track passes through — that's how two entangled heads would get
            # smeared into one blur path.
            blocked = False
            for k, c in enumerate(tracklets):
                if k in (i, j):
                    continue
                for f in range(a.end + 1, b.start):
                    cb = _boxes_at(c, f)
                    if cb is None:
                        continue
                    w = (f - a.end) / (g + 1)
                    p = (1 - w) * a.boxes[-1] + w * b.boxes[0]
                    if _iou_cxcywh(p, cb) > _CORRIDOR_IOU:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                cands.append((dist / diag + 0.05 * g, i, j))
    cands.sort()
    return cands


def _merge_pair(a: Tracklet, b: Tracklet) -> Tracklet:
    """Join ``a``→``b`` with the gap linearly interpolated and size-padded."""
    g = b.start - a.end - 1
    gap_boxes = np.empty((g, 4), dtype=np.float32)
    for f in range(g):
        w = (f + 1) / (g + 1)
        gap_boxes[f] = (1 - w) * a.boxes[-1] + w * b.boxes[0]
        # Uncertainty padding, strongest mid-gap.
        gap_boxes[f, 2:] *= 1.0 + _GAP_PAD * np.sin(np.pi * (f + 1) / (g + 1))
    return Tracklet(
        a.tid, a.start,
        np.concatenate([a.boxes, gap_boxes, b.boxes]),
        np.concatenate([a.scores, np.zeros(g, np.float32), b.scores]),
        np.concatenate([a.hits, np.zeros(g, bool), b.hits]))


def _bridge(tracklets: list[Tracklet], gap_max: int) -> list[Tracklet]:
    """Merge tracklets across detection gaps; chains bridge over repeated
    rounds (A→B this round, (AB)→C the next)."""
    tracklets = list(tracklets)
    while True:
        cands = _bridge_candidates(tracklets, gap_max)
        if not cands:
            return tracklets
        merged: dict[int, Tracklet] = {}   # pred idx -> merged tracklet
        consumed: set[int] = set()         # succ idxs absorbed this round
        used: set[int] = set()
        for cost, i, j in cands:
            if i in used or j in used:
                continue
            # Ambiguity gate: another live candidate sharing this pred or
            # succ at near-equal cost means the join is a coin flip between
            # two heads — bridge neither.
            near = [c for c, x, y in cands
                    if (x, y) != (i, j) and (x == i or y == j)
                    and x not in used and y not in used
                    and c <= cost * (1 + _AMBIGUITY_MARGIN)]
            if near:
                used.add(i)
                used.add(j)
                continue
            merged[i] = _merge_pair(tracklets[i], tracklets[j])
            consumed.add(j)
            used.add(i)
            used.add(j)
        if not merged:
            return tracklets
        tracklets = [merged.get(k, t) for k, t in enumerate(tracklets)
                     if k not in consumed]


def _face_backed(cand: np.ndarray, faces: np.ndarray) -> bool:
    """A face box centre inside the candidate, or vice versa — the candidate
    head has face evidence behind it."""
    if len(faces) == 0:
        return False
    fcx = (faces[:, 0] + faces[:, 2]) / 2
    fcy = (faces[:, 1] + faces[:, 3]) / 2
    in_c = ((cand[0] <= fcx) & (fcx <= cand[2])
            & (cand[1] <= fcy) & (fcy <= cand[3]))
    ccx, ccy = (cand[0] + cand[2]) / 2, (cand[1] + cand[3]) / 2
    c_in = ((faces[:, 0] <= ccx) & (ccx <= faces[:, 2])
            & (faces[:, 1] <= ccy) & (ccy <= faces[:, 3]))
    return bool((in_c | c_in).any())


def verify_tracklets(
    tracklets: list[Tracklet],
    frame_at: Callable[[int], "np.ndarray | None"],
    detect_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    min_score: float = _VERIFY_SCORE,
) -> tuple[list[Tracklet], list[Tracklet]]:
    """Re-inference gate: keep tracklets whose head reproduces on crops.

    For up to ``_VERIFY_SAMPLES`` evenly spaced *hit* frames per tracklet,
    crop the frame around the tracklet box (side ``max(w, h) ×
    _VERIFY_MARGIN``) and run ``detect_fn`` on the crop. ``detect_fn``
    returns ``(candidates, faces)`` — head blur candidates plus every piece
    of face evidence seen in the crop (both as ``[x1, y1, x2, y2, score]``
    in crop coords). A sample verifies when some candidate lands on the
    tracklet box (its centre inside the box, or the box centre inside it — a
    close-up head can out-grow its recorded box) and either scores ≥
    ``min_score`` with face evidence behind it, or scores ≥ ``_VERIFY_LONE``
    alone (the back-of-head path — face evidence is never required, only
    rewarded). The tracklet survives when at least ``_VERIFY_MIN_FRAC`` of
    its *tested* samples verify.

    Fail-open by design: unreadable frames or degenerate crops don't count
    as tested, and a tracklet with zero tested samples is kept — for a
    privacy tool a spurious blur is cheaper than an unblurred head.

    Returns ``(kept, dropped)`` so the caller can log what died.
    """
    kept: list[Tracklet] = []
    dropped: list[Tracklet] = []
    for t in tracklets:
        hit_idx = np.nonzero(t.hits)[0]
        if len(hit_idx) == 0:
            kept.append(t)
            continue
        n = min(_VERIFY_SAMPLES, len(hit_idx))
        picks = hit_idx[np.unique(
            np.round(np.linspace(0, len(hit_idx) - 1, n)).astype(int))]
        tested = accepted = 0
        for k in picks:
            frame = frame_at(t.start + int(k))
            if frame is None:
                continue
            fh, fw = frame.shape[:2]
            cx, cy, w, h = (float(v) for v in t.boxes[k])
            side = max(w, h) * _VERIFY_MARGIN
            side = max(side, float(_VERIFY_MIN_CROP))
            x0 = int(max(0, cx - side / 2))
            y0 = int(max(0, cy - side / 2))
            x1 = int(min(fw, cx + side / 2))
            y1 = int(min(fh, cy + side / 2))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            tested += 1
            raw_cands, raw_faces = detect_fn(frame[y0:y1, x0:x1])
            cands = np.asarray(raw_cands, dtype=np.float32).reshape(-1, 5)
            faces = np.asarray(raw_faces, dtype=np.float32).reshape(-1, 5)
            # Tracklet box in crop coordinates.
            bx0, by0 = cx - w / 2 - x0, cy - h / 2 - y0
            bx1, by1 = cx + w / 2 - x0, cy + h / 2 - y0
            bcx, bcy = cx - x0, cy - y0
            for c in cands:
                if c[4] < min_score:
                    continue
                ccx, ccy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
                if not ((bx0 <= ccx <= bx1 and by0 <= ccy <= by1)
                        or (c[0] <= bcx <= c[2] and c[1] <= bcy <= c[3])):
                    continue
                if c[4] >= _VERIFY_LONE or _face_backed(c, faces):
                    accepted += 1
                    break
        if tested == 0 or accepted / tested >= _VERIFY_MIN_FRAC:
            kept.append(t)
        else:
            dropped.append(t)
    return kept, dropped


def postprocess(
    tracklets: list[Tracklet],
    *,
    fps: float,
    n_frames: int,
    p: PostParams,
    verify: Optional[Callable[[list[Tracklet]], list[Tracklet]]] = None,
) -> list[list[tuple[int, np.ndarray]]]:
    """Raw tracklets → per-frame render table ``frame -> [(tid, xyxy)]``."""
    fps = max(fps, 1.0)

    # 1. TRIM coasted tails/heads.
    trimmed = [t for t in (_trim(t) for t in tracklets) if t is not None]

    # 2. PRUNE noise.
    min_len = max(p.min_hits, int(round(p.min_track_s * fps)))
    kept: list[Tracklet] = []
    for t in trimmed:
        n_hit = int(t.hits.sum())
        if n_hit < min_len:
            continue
        top = np.sort(t.scores[t.hits])[-3:]
        if float(top.mean()) < p.det_conf:
            continue
        if n_hit / len(t.boxes) < _MIN_HIT_RATIO:
            continue
        kept.append(t)

    # 2½. VERIFY — re-inference second opinion (export wires cropped
    # re-detection here; None in tests/preview). Before BRIDGE so a fake
    # tracklet can never be interpolated into a real one.
    if verify is not None:
        kept = verify(kept)

    # 3. BRIDGE across gaps.
    kept = _bridge(kept, gap_max=int(round(p.bridge_gap_s * fps)))

    # 4. EXTEND ends (hold first/last box briefly).
    ext = max(0, int(round(_EXTEND_S * fps)))
    extended: list[Tracklet] = []
    for t in kept:
        pre = min(ext, t.start)
        post = min(ext, max(0, n_frames - 1 - t.end))
        boxes = np.concatenate([np.repeat(t.boxes[:1], pre, axis=0),
                                t.boxes,
                                np.repeat(t.boxes[-1:], post, axis=0)])
        scores = np.concatenate([np.zeros(pre, np.float32), t.scores,
                                 np.zeros(post, np.float32)])
        hits = np.concatenate([np.zeros(pre, bool), t.hits,
                               np.zeros(post, bool)])
        extended.append(Tracklet(t.tid, t.start - pre, boxes, scores, hits))

    # 5. SMOOTH (zero-phase; offline so no lag).
    win = int(round(p.smooth_win_s * fps)) | 1
    for t in extended:
        n = len(t.boxes)
        if n >= 5:
            w = min(win, n if n % 2 else n - 1)
            if w >= 5:
                for ch in range(4):
                    t.boxes[:, ch] = savgol_filter(
                        t.boxes[:, ch], w, polyorder=2, mode="interp")

    # 6. EMIT render table.
    table: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_frames)]
    for t in extended:
        for k in range(len(t.boxes)):
            f = t.start + k
            if 0 <= f < n_frames:
                cx, cy, w, h = t.boxes[k]
                table[f].append((t.tid, np.array(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    dtype=np.float32)))
    return table
