"""Pass-1 track recording and offline tracklet cleanup for two-pass export.

The export runs in two passes (see ui.py). Pass 1 detects and tracks but
renders nothing; every frame's ``TrackObs`` land here. Between passes,
:func:`clean_tracklets` turns the raw tracklets into cleaned, graded
tracklets using knowledge a streaming tracker can never have — the future —
and :func:`build_table` turns those into a per-frame render table:

  * **Trim** — coasted tails are cut; a Kalman prediction that never met
    another detection was a guess, and guesses aren't blurred (the old
    static-hold ghost fix).
  * **Prune** — a *composite* gate: tracklets too short, never confident, or
    too sparse are detector noise (as before); on top of that, a tracklet
    whose per-frame evidence ledger (libs/evidence.summarize/grade) never
    accumulates real anatomical/part backing is graded "C" and dropped too —
    this is what catches a static skin/fabric misread that reproduces every
    frame at high confidence (long *and* confident, so score-and-length
    gates alone can't tell it from a real head) but never earns the
    independent evidence a real face does.
  * **Verify** — every tracklet that survives the prune must *reproduce*
    under re-inference: sample a few of its hit frames, crop around its box
    with context, and ask an INDEPENDENT model (SCRFD and/or NudeNet — never
    the primary detector that produced the tracklet in the first place) for
    a face at the same spot (:func:`verify_tracklets_xmodel`). This kills
    persistent false positives the ledger's per-frame view can miss, without
    touching thresholds, and closes the "verifier re-runs the same model
    that hallucinated" hole a same-model verifier would leave open.
  * **Bridge** — a track that vanishes and reappears nearby (head briefly
    buried in a pillow / behind a shoulder) is re-joined and the gap is
    *interpolated along the path*, so the blur follows the head instead of
    flickering off and on. Gates guard against joining two different heads:
    velocity-extrapolated distance, size similarity, a corridor test (never
    bridge through a region another surviving track occupies), and an
    ambiguity test (two plausible predecessors → bridge neither).
  * **Fill face gaps** — separately from bridging (which concerns *any*
    detection gap), short runs where face evidence itself was momentarily
    missing get their face-target box interpolated between the flanking
    real sightings instead of falling back to the wider head box — a
    detector flicker should not visibly balloon the face-only blur. Longer
    runs are left alone: a genuine, extended loss of face evidence (the
    subject turned away) is reported as such (``fvalid=False``), not papered
    over.
  * **Smooth** — zero-phase Savitzky-Golay on cx/cy/w/h. Offline smoothing has
    no lag, so the blur is both steady *and* on target.

Boxes are recorded in full-resolution frame coordinates so pass 2 needs no
scale bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    # Only for type hints — tracklets.py must not import libs.sidecar at
    # runtime (sidecar.py imports Tracklet from here, so a real import would
    # be circular). build_table only ever calls .start/.end/.box_at() on
    # these, so any duck-typed object works at runtime.
    from .sidecar import ManualRegion

from .evidence import EvidenceSummary, PROFILES, GateThresholds, grade, summarize

# Prune gates (see _prune).
_MIN_HIT_RATIO = 0.30
# Verification gates (see verify_tracklets_xmodel).
_VERIFY_SAMPLES = 5      # hit frames sampled per tracklet
_VERIFY_MIN_FRAC = 0.4   # fraction of tested samples that must re-detect
# Crop side = max(box w, h) × this. Must be ≥ ~2.5: the detector's rotated
# passes reject heads above ~20 % of the *image* area, and the image here is
# the crop — at ×3 a true head is ~11 % of it, safely under that cap even for
# sideways heads that only the rotated pass can re-detect.
_VERIFY_MARGIN = 3.0
_VERIFY_MIN_CROP = 96    # px — never crop tighter than this
_VERIFY_SCORE = 0.50     # witness re-detection floor to count as a match
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
    boxes: np.ndarray       # (T, 4) float32 cx, cy, w, h — head/anchor channel
    scores: np.ndarray      # (T,)
    hits: np.ndarray        # (T,) bool — False = coasted or interpolated
    # (T, 4) float32 cx, cy, w, h — the *face-only* blur target per frame
    # (the tracker's remembered face box, or the head/anchor box when no
    # face has ever matched). None = not recorded; falls back to ``boxes``.
    fboxes: Optional[np.ndarray] = None
    # (T,) uint32 — Ev bits (libs/evidence.py) credited to this tracklet's
    # hit that frame; 0 on a coast/interpolated frame or an evidence-free
    # (sustain-pool) hit. Feeds the evidence ledger (summarize/grade).
    ev: Optional[np.ndarray] = None
    # (T,) bool — face evidence was a literal match *this exact frame*
    # (before gap-filling). None = not recorded.
    fvalid: Optional[np.ndarray] = None

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
    # How long a run of missing face evidence may be interpolated across
    # (see _fill_face_gaps) before it's reported as a genuine coverage gap.
    face_gap_bridge_s: float = 1.0
    # Key into libs.evidence.PROFILES — how much evidence a tracklet needs
    # to clear the composite prune (see _prune).
    evidence_profile: str = "balanced"


@dataclass
class CleanTracklet:
    """One tracklet after cleanup, with its evidence grade attached — the
    unit :func:`build_table` (kept) and a future review UI (kept + rejected)
    both consume."""
    t: Tracklet
    summary: EvidenceSummary
    grade: str
    verify: Optional[object] = None      # VerifyResult — wired in Phase 3
    enabled_default: bool = True         # review-UI seed — wired in Phase 4


class TrackRecorder:
    """Pass-1 sink: collects per-frame TrackObs into contiguous tracklets."""

    def __init__(self) -> None:
        # tid -> [start, [boxes cxcywh], [scores], [hits], [fboxes cxcywh],
        #         [ev], [fvalid]]
        self._open: dict[int, list] = {}
        self._done: list[Tracklet] = []
        self._last_frame = -1

    def observe(self, frame_idx: int, obs: list, inv_scale: float = 1.0,
               face_boxes: Optional[list] = None,
               face_valid: Optional[list] = None,
               ev_flags: Optional[list] = None) -> None:
        """Record one frame of tracker output (boxes scaled by ``inv_scale``
        back to full resolution). Tracks absent this frame are closed.

        ``face_boxes`` is an optional parallel list of xyxy boxes — the
        raw face-target position per observation (the tracker's own best
        guess; the caller does not pre-decide freshness here — see
        ``face_valid``); each defaults to the head box when omitted.
        ``face_valid`` marks which of those are a literal match this frame
        (vs. a stale carried-over position); ``ev_flags`` is each
        observation's ``Ev`` bits. Both default to "no evidence" when
        omitted."""
        seen: set[int] = set()
        for k, o in enumerate(obs):
            b = np.asarray(o.box, dtype=np.float32) * inv_scale
            z = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                          b[2] - b[0], b[3] - b[1]], dtype=np.float32)
            fb = b if face_boxes is None \
                else np.asarray(face_boxes[k], dtype=np.float32) * inv_scale
            fz = np.array([(fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2,
                           fb[2] - fb[0], fb[3] - fb[1]], dtype=np.float32)
            ev = 0 if ev_flags is None else int(ev_flags[k])
            fv = False if face_valid is None else bool(face_valid[k])
            rec = self._open.get(o.track_id)
            if rec is None:
                self._open[o.track_id] = [frame_idx, [z], [o.score], [o.hit],
                                          [fz], [ev], [fv]]
            else:
                # Guard against a recycled id after the tracker dropped the
                # track for exactly one frame boundary we didn't see: pad any
                # hole with the previous box so tracklets stay contiguous.
                expect = rec[0] + len(rec[1])
                while expect < frame_idx:
                    rec[1].append(rec[1][-1].copy())
                    rec[2].append(0.0)
                    rec[3].append(False)
                    rec[4].append(rec[4][-1].copy())
                    rec[5].append(0)
                    rec[6].append(False)
                    expect += 1
                rec[1].append(z)
                rec[2].append(o.score)
                rec[3].append(bool(o.hit))
                rec[4].append(fz)
                rec[5].append(ev)
                rec[6].append(fv)
            seen.add(o.track_id)
        for tid in list(self._open):
            if tid not in seen:
                self._close(tid)
        self._last_frame = frame_idx

    def _close(self, tid: int) -> None:
        start, boxes, scores, hits, fboxes, ev, fvalid = self._open.pop(tid)
        self._done.append(Tracklet(
            tid, start,
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32),
            np.asarray(hits, dtype=bool),
            np.asarray(fboxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(ev, dtype=np.uint32),
            np.asarray(fvalid, dtype=bool)))

    def finalize(self) -> list[Tracklet]:
        for tid in list(self._open):
            self._close(tid)
        return self._done


def _fb(t: Tracklet) -> np.ndarray:
    """Face-target channel, falling back to the head boxes when absent."""
    return t.fboxes if t.fboxes is not None else t.boxes


def _ev_arr(t: Tracklet) -> np.ndarray:
    """Evidence-flag channel, falling back to "no evidence recorded"."""
    return (t.ev if t.ev is not None
            else np.zeros(len(t.boxes), dtype=np.uint32))


def _fvalid_arr(t: Tracklet) -> np.ndarray:
    """Face-freshness channel, falling back to "never fresh"."""
    return (t.fvalid if t.fvalid is not None
            else np.zeros(len(t.boxes), dtype=bool))


def _trim(t: Tracklet) -> Tracklet | None:
    """Cut everything after the last hit (and before the first)."""
    idx = np.nonzero(t.hits)[0]
    if len(idx) == 0:
        return None
    a, b = int(idx[0]), int(idx[-1]) + 1
    return Tracklet(t.tid, t.start + a, t.boxes[a:b], t.scores[a:b],
                    t.hits[a:b], _fb(t)[a:b], _ev_arr(t)[a:b],
                    _fvalid_arr(t)[a:b])


def _prune(
    trimmed: list[Tracklet], p: PostParams, fps: float, thr: GateThresholds,
) -> tuple[list[Tracklet], list[Tracklet]]:
    """Composite prune: the original length/confidence/density gates, *and*
    the evidence ledger (see the module docstring). Both must pass. Returns
    ``(kept, rejected)`` — rejected tracklets are not silently discarded by
    the caller (see :func:`clean_tracklets`)."""
    min_len = max(p.min_hits, int(round(p.min_track_s * fps)))
    kept: list[Tracklet] = []
    rejected: list[Tracklet] = []
    for t in trimmed:
        n_hit = int(t.hits.sum())
        top = np.sort(t.scores[t.hits])[-3:]
        basic_ok = (n_hit >= min_len
                   and float(top.mean()) >= p.det_conf
                   and n_hit / len(t.boxes) >= _MIN_HIT_RATIO)
        g = grade(summarize(_ev_arr(t), t.hits), thr)
        if basic_ok and g != "C":
            kept.append(t)
        else:
            rejected.append(t)
    return kept, rejected


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
                    pred_box = (1 - w) * a.boxes[-1] + w * b.boxes[0]
                    if _iou_cxcywh(pred_box, cb) > _CORRIDOR_IOU:
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
    gap_fboxes = np.empty((g, 4), dtype=np.float32)
    fa, fb = _fb(a), _fb(b)
    for f in range(g):
        w = (f + 1) / (g + 1)
        pad = 1.0 + _GAP_PAD * np.sin(np.pi * (f + 1) / (g + 1))
        gap_boxes[f] = (1 - w) * a.boxes[-1] + w * b.boxes[0]
        # Uncertainty padding, strongest mid-gap.
        gap_boxes[f, 2:] *= pad
        gap_fboxes[f] = (1 - w) * fa[-1] + w * fb[0]
        gap_fboxes[f, 2:] *= pad
    return Tracklet(
        a.tid, a.start,
        np.concatenate([a.boxes, gap_boxes, b.boxes]),
        np.concatenate([a.scores, np.zeros(g, np.float32), b.scores]),
        np.concatenate([a.hits, np.zeros(g, bool), b.hits]),
        np.concatenate([fa, gap_fboxes, fb]),
        np.concatenate([_ev_arr(a), np.zeros(g, np.uint32), _ev_arr(b)]),
        np.concatenate([_fvalid_arr(a), np.zeros(g, bool), _fvalid_arr(b)]))


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


def _fill_face_gaps(t: Tracklet, max_gap: int) -> Tracklet:
    """Interpolate the face-target box across runs of ``fvalid=False`` no
    longer than ``max_gap`` frames, flanked on both sides by a real sighting
    — a detector flicker should not visibly balloon the face-only blur out
    to the head box and back. Leading/trailing runs (no flanking sighting on
    one side) and longer runs are left alone: a genuine, extended loss of
    face evidence is reported as such, not papered over."""
    fvalid = _fvalid_arr(t).copy()
    fboxes = _fb(t).copy()
    n = len(fvalid)
    i = 0
    while i < n:
        if fvalid[i]:
            i += 1
            continue
        j = i
        while j < n and not fvalid[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n:
            a, b = fboxes[i - 1], fboxes[j]
            for k in range(gap_len):
                w = (k + 1) / (gap_len + 1)
                fboxes[i + k] = (1 - w) * a + w * b
                fvalid[i + k] = True
        i = j
    return Tracklet(t.tid, t.start, t.boxes, t.scores, t.hits, fboxes,
                    _ev_arr(t), fvalid)


def verify_tracklets_xmodel(
    tracklets: list[Tracklet],
    frame_at: Callable[[int], "np.ndarray | None"],
    witness_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    thr: GateThresholds,
    *,
    min_score: float = _VERIFY_SCORE,
) -> tuple[list[Tracklet], list[Tracklet]]:
    """Cross-model re-inference gate: keep tracklets whose face reproduces
    on crops, according to a model that had no part in producing them.

    ``witness_fn(crop) -> faces`` must come from an INDEPENDENT model — SCRFD
    and/or NudeNet, unioned by the caller (see ui.py's ``_make_verifier``) —
    never the primary Wholebody17 detector that fed the evidence gate and
    produced the tracklet in the first place. The primary has no vote here
    at all: it may have been the thing that hallucinated, so it cannot also
    be the thing that confirms itself. This is the Phase 3 replacement for
    the interim, single-model verifier the evidence-gate rework shipped with
    in Phase 1, and it closes exactly that self-correlated-confirmation hole.

    For up to ``_VERIFY_SAMPLES`` evenly spaced *hit* frames per tracklet,
    crop the frame around the tracklet box (side ``max(w, h) ×
    _VERIFY_MARGIN``) and ask ``witness_fn`` for every face
    (``[x1, y1, x2, y2, score]``, crop coords) it sees there. A sample
    verifies when some witness face's centre lands in the tracklet box (or
    vice versa — a close-up face can out-grow its recorded box) scoring ≥
    ``min_score``. The tracklet survives when at least ``_VERIFY_MIN_FRAC``
    of its *tested* samples verify.

    When nothing could be tested for a tracklet — ``witness_fn`` is ``None``
    (every independent model is unavailable; the verifier genuinely "can't
    run") or every sampled frame/crop was unreadable — the decision falls
    back to the tracklet's own evidence grade (:func:`grade`): only grades in
    ``thr.verify_fail_open_grades`` survive unverified. This is tighter than
    a blanket fail-open, per the pipeline's precision-first mandate: under
    the "balanced" profile a "B"-graded track is dropped rather than trusted
    on faith when no independent model could check it; "strict" trusts
    nothing unverified at all.

    Returns ``(kept, dropped)`` so the caller can log what died.
    """
    kept: list[Tracklet] = []
    dropped: list[Tracklet] = []

    def _fail_open(t: Tracklet) -> None:
        g = grade(summarize(_ev_arr(t), t.hits), thr)
        (kept if g in thr.verify_fail_open_grades else dropped).append(t)

    for t in tracklets:
        hit_idx = np.nonzero(t.hits)[0]
        if len(hit_idx) == 0:
            kept.append(t)
            continue
        if witness_fn is None:
            _fail_open(t)
            continue

        n = min(_VERIFY_SAMPLES, len(hit_idx))
        picks = hit_idx[np.unique(
            np.round(np.linspace(0, len(hit_idx) - 1, n)).astype(int))]
        tested = verified = 0
        for k in picks:
            frame = frame_at(t.start + int(k))
            if frame is None:
                continue
            fh, fw = frame.shape[:2]
            cx, cy, w, h = (float(v) for v in t.boxes[k])
            side = max(max(w, h) * _VERIFY_MARGIN, float(_VERIFY_MIN_CROP))
            x0 = int(max(0, cx - side / 2))
            y0 = int(max(0, cy - side / 2))
            x1 = int(min(fw, cx + side / 2))
            y1 = int(min(fh, cy + side / 2))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            tested += 1
            faces = np.asarray(witness_fn(frame[y0:y1, x0:x1]),
                               dtype=np.float32).reshape(-1, 5)
            if len(faces) == 0:
                continue
            # Tracklet box in crop coordinates.
            bx0, by0 = cx - w / 2 - x0, cy - h / 2 - y0
            bx1, by1 = cx + w / 2 - x0, cy + h / 2 - y0
            bcx, bcy = cx - x0, cy - y0
            for f in faces:
                if f[4] < min_score:
                    continue
                fcx, fcy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
                if ((bx0 <= fcx <= bx1 and by0 <= fcy <= by1)
                        or (f[0] <= bcx <= f[2] and f[1] <= bcy <= f[3])):
                    verified += 1
                    break
        if tested == 0:
            _fail_open(t)
        elif verified / tested >= _VERIFY_MIN_FRAC:
            kept.append(t)
        else:
            dropped.append(t)
    return kept, dropped


def clean_tracklets(
    tracklets: list[Tracklet],
    *,
    fps: float,
    n_frames: int,
    p: PostParams,
    verify: Optional[Callable[[list[Tracklet]], list[Tracklet]]] = None,
) -> tuple[list[CleanTracklet], list[CleanTracklet]]:
    """Raw tracklets → ``(kept, rejected)`` graded :class:`CleanTracklet`
    lists — ``kept`` is ready for :func:`build_table`; ``rejected`` is
    logged today and will feed the review UI once one exists (Phase 4).
    See the module docstring for what each stage does."""
    fps = max(fps, 1.0)
    thr = PROFILES.get(p.evidence_profile, PROFILES["balanced"])

    # 1. TRIM coasted tails/heads.
    trimmed = [t for t in (_trim(t) for t in tracklets) if t is not None]

    # 2. PRUNE — composite length/confidence/density + evidence ledger.
    kept, rejected = _prune(trimmed, p, fps, thr)

    # 2½. VERIFY — re-inference second opinion (export wires cropped
    # re-detection here; None in tests/preview). Before BRIDGE so a fake
    # tracklet can never be interpolated into a real one.
    if verify is not None:
        kept = verify(kept)

    # 3. BRIDGE across gaps.
    kept = _bridge(kept, gap_max=int(round(p.bridge_gap_s * fps)))

    # 3½. FILL short face-evidence gaps (see the module docstring).
    fill_gap = max(0, int(round(p.face_gap_bridge_s * fps)))
    kept = [_fill_face_gaps(t, fill_gap) for t in kept]

    # 4. EXTEND ends (hold first/last box briefly).
    ext = max(0, int(round(_EXTEND_S * fps)))
    extended: list[Tracklet] = []
    for t in kept:
        pre = min(ext, t.start)
        post = min(ext, max(0, n_frames - 1 - t.end))
        boxes = np.concatenate([np.repeat(t.boxes[:1], pre, axis=0),
                                t.boxes,
                                np.repeat(t.boxes[-1:], post, axis=0)])
        fb = _fb(t)
        fboxes = np.concatenate([np.repeat(fb[:1], pre, axis=0), fb,
                                 np.repeat(fb[-1:], post, axis=0)])
        scores = np.concatenate([np.zeros(pre, np.float32), t.scores,
                                 np.zeros(post, np.float32)])
        hits = np.concatenate([np.zeros(pre, bool), t.hits,
                               np.zeros(post, bool)])
        ev = _ev_arr(t)
        ev_ext = np.concatenate([np.repeat(ev[:1], pre), ev,
                                 np.repeat(ev[-1:], post)])
        fvalid = _fvalid_arr(t)
        fvalid_ext = np.concatenate([np.repeat(fvalid[:1], pre), fvalid,
                                     np.repeat(fvalid[-1:], post)])
        extended.append(Tracklet(t.tid, t.start - pre, boxes, scores, hits,
                                 fboxes, ev_ext, fvalid_ext))

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
                    t.fboxes[:, ch] = savgol_filter(
                        t.fboxes[:, ch], w, polyorder=2, mode="interp")

    def _wrap(t: Tracklet) -> CleanTracklet:
        s = summarize(_ev_arr(t), t.hits)
        return CleanTracklet(t=t, summary=s, grade=grade(s, thr))

    return [_wrap(t) for t in extended], [_wrap(t) for t in rejected]


def apply_review(
    kept: list[CleanTracklet], rejected: list[CleanTracklet],
    enabled_overrides: dict[int, bool],
) -> list[CleanTracklet]:
    """Resolve the final render set from :func:`clean_tracklets`' ``(kept,
    rejected)`` plus the review UI's per-tid enable/disable overrides
    (Phase 4; see ``libs.sidecar.ReviewDecisions.enabled``).

    ``kept`` tracklets are enabled by default (they already cleared the
    gate/ledger/composite-verify pipeline); ``rejected`` ones are disabled
    by default (they already failed it) but are never silently dropped —
    the caller can re-enable a wrongly-rejected track by tid, and disable a
    wrongly-kept one, either way via ``enabled_overrides``. ``kept`` and
    ``rejected`` are disjoint tid sets by construction (a tid can only ever
    be in one of the two clean_tracklets output lists)."""
    out = [ct for ct in kept if enabled_overrides.get(ct.t.tid, True)]
    out.extend(ct for ct in rejected if enabled_overrides.get(ct.t.tid, False))
    return out


def build_table(
    kept: list[CleanTracklet], n_frames: int,
    manual_regions: "Sequence[ManualRegion]" = (),
) -> list[list[tuple[int, np.ndarray, np.ndarray, bool]]]:
    """Cleaned tracklets → per-frame render table ``frame -> [(tid,
    head_xyxy, face_xyxy, face_ok)]``. ``face_ok`` is whether the face
    channel is trustworthy that frame (see ``Tracklet.fvalid`` /
    ``_fill_face_gaps``) — pass 2 picks head vs. face live from
    ``Params.blur_region``, and in face mode skips any entry where
    ``face_ok`` is False rather than silently falling back to the head box,
    per the "face only" blur-region contract.

    ``manual_regions`` (Phase 4's review UI) are user-drawn blur regions —
    each becomes a synthetic entry per frame in its range, keyed by a
    negative tid so it can never collide with a real track id (the tracker's
    ids are always non-negative). A manual region has no separate head/face
    channel (there's nothing to hold vs. blur) and no freshness gap to
    report, so both box slots are the same interpolated box and
    ``face_ok`` is always ``True``."""
    def xyxy(z: np.ndarray) -> np.ndarray:
        cx, cy, w, h = z
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        dtype=np.float32)

    table: list[list[tuple[int, np.ndarray, np.ndarray, bool]]] = [
        [] for _ in range(n_frames)]
    for ct in kept:
        t = ct.t
        fb = _fb(t)
        fvalid = _fvalid_arr(t)
        for k in range(len(t.boxes)):
            f = t.start + k
            if 0 <= f < n_frames:
                table[f].append((t.tid, xyxy(t.boxes[k]), xyxy(fb[k]),
                                 bool(fvalid[k])))
    for i, region in enumerate(manual_regions):
        tid = -(1000 + i)
        lo, hi = max(0, region.start), min(n_frames - 1, region.end)
        for f in range(lo, hi + 1):
            box = region.box_at(f)
            if box is None:
                continue
            box = box.astype(np.float32)
            table[f].append((tid, box, box, True))
    return table
