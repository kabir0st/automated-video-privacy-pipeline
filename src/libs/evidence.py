"""Face-candidate acceptance gate — the pipeline's precision mechanism.

Detector classes (head/face/eye/nose/mouth/hand/foot, from libs/detector.py),
SCRFD's landmark-checked witness faces (libs/scrfd.py) and, from Phase 2
onward, pose-derived head anchors + torso axis (libs/pose.py) are all
*evidence*. None of it is a blur target by itself. A face claim only becomes
a :class:`FaceCandidate` — the one thing downstream that may spawn or drive a
blur track — when it clears :func:`gate_face_candidates`'s combined test:

  * **anatomical anchoring** — the claim sits inside a head box the detector
    independently flagged (``Ev.HEAD_ANCHOR``), or inside a pose-derived head
    region on the correct side of the body's torso axis (``Ev.POSE_ANCHOR`` /
    ``Ev.AXIS_OK``, Phase 2). A face-shaped misread of skin/fabric with no
    independent head-class or pose corroboration at the same spot fails here.
  * **part consensus** — an eye/nose/mouth detection lands inside the claim,
    or it came from SCRFD with landmarks that already passed
    ``scrfd.kps_plausible`` (the caller is expected to pre-filter SCRFD faces
    through that gate — see ``scrfd_faces`` below), or (Phase 2) confident
    pose facial keypoints land inside it.
  * **no negative veto** — a hand or foot detection substantially covering
    the claim (the sock/knee case) rejects it, *unless* the candidate has
    strong independent backing (≥2 distinct part hits, or a pose torso axis
    that confirms the claim sits on the head side of the body) — a hand
    genuinely resting over someone's face is a real scene, not a false
    positive, and should not lose its blur.

A claim that fails all of the above may still be accepted through the
**extreme-close-up path** (``Ev.CLOSEUP``): when no head/pose anchor exists
at all (the face fills the frame; there is no body context to anchor to),
independent-model consensus substitutes for it — the primary detector's face
class and SCRFD must both confidently claim the same region, at a size that
actually defeats a whole-head detector, with part evidence behind it.

Every accepted candidate's :class:`Ev` flags travel onto the tracker
(``TrackObs.flags``) and then into the offline evidence ledger
(:func:`summarize`/:func:`grade`, consumed by libs/tracklets.py's composite
prune) — so a track's *history* of evidence, not just its score and length,
decides whether it survives to render. This is what closes the failure this
module exists to fix: a static skin/fabric misread reproduces every frame at
high confidence (score-and-length gates can't tell it from a real head), but
it does not repeatedly earn independent anatomical + part evidence the way a
real face does.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .detector import (Detections, NEG_FOOT, NEG_HAND, PART_EYE, PART_MOUTH,
                       PART_NOSE, _centres_inside, _iou_matrix)


class Ev(enum.IntFlag):
    """Bits recording why a face candidate was accepted or rejected.
    Carried as a plain ``int`` on ``TrackObs``/``Tracklet`` so numpy arrays
    of flags need no object dtype."""
    NONE = 0
    # Source / cross-model agreement.
    YOLO_FACE = enum.auto()
    SCRFD_FACE = enum.auto()
    CONSENSUS = enum.auto()        # YOLO and SCRFD claims describe the same face
    # Part / landmark consensus.
    PART_EYE_HIT = enum.auto()
    PART_NOSE_HIT = enum.auto()
    PART_MOUTH_HIT = enum.auto()
    SCRFD_KPS = enum.auto()        # SCRFD source already passed kps_plausible
    POSE_KPT = enum.auto()         # confident pose facial keypoints in-box (Phase 2)
    # Anatomical anchoring.
    HEAD_ANCHOR = enum.auto()      # covered by an independent YOLOv9 head box
    BODY_MEMBER = enum.auto()      # that head also sits inside a body box (bonus)
    POSE_ANCHOR = enum.auto()      # inside a pose-derived head box (Phase 2)
    AXIS_OK = enum.auto()          # on the head side of the torso axis (Phase 2)
    CLOSEUP = enum.auto()          # accepted via the extreme-close-up path
    # Vetoes — informational once resolved; see VETO_OVERRIDDEN.
    VETO_HAND = enum.auto()
    VETO_FOOT = enum.auto()
    VETO_AXIS = enum.auto()        # wrong side of the torso axis (Phase 2)
    VETO_OVERRIDDEN = enum.auto()  # a veto fired but strong evidence beat it


_PART_HIT_MASK = int(Ev.PART_EYE_HIT | Ev.PART_NOSE_HIT | Ev.PART_MOUTH_HIT
                    | Ev.SCRFD_KPS | Ev.POSE_KPT)
_ANCHOR_MASK = int(Ev.HEAD_ANCHOR | Ev.POSE_ANCHOR | Ev.CLOSEUP)


def _part_hit_count(flags: int) -> int:
    return bin(int(flags) & _PART_HIT_MASK).count("1")


@dataclass(frozen=True)
class GateThresholds:
    """Tunable bars for :func:`gate_face_candidates`, bundled into named
    profiles below. ``det_conf``/``det_conf_low`` stay in ``Params`` — the
    profile changes how much evidence a candidate needs, never the raw
    confidence floor, so "catch more" is bought with evidence leniency and
    longer holds instead of trusting the detector more."""
    consensus_iou: float = 0.30    # YOLO x SCRFD claims agreeing on one face
    anchor_iou: float = 0.20       # face-in-head-box overlap (or centre containment)
    veto_cover: float = 0.60       # area(face ∩ hand/foot) / area(face) that vetoes
    closeup_frac: float = 0.25     # face's longest side / frame short side
    require_parts: bool = True     # part/landmark hit required at spawn
    prefer_pose: bool = True       # Phase 2: pose anchor outranks head-box anchor
    # Offline ledger ratios (see summarize/grade) — over a tracklet's hit
    # frames, either ratio clearing its bar is enough for grade "B".
    min_part_ratio: float = 0.20
    min_consensus_ratio: float = 0.10
    # Grades whose tracklets survive VERIFY when the verifier itself can't
    # run (Phase 3) — kept here since it's part of the same leniency knob.
    verify_fail_open_grades: tuple[str, ...] = ("A",)


PROFILES: dict[str, GateThresholds] = {
    "balanced": GateThresholds(),
    "max": GateThresholds(
        require_parts=False, veto_cover=0.75, consensus_iou=0.25,
        min_part_ratio=0.10, min_consensus_ratio=0.05,
        verify_fail_open_grades=("A", "B")),
    "strict": GateThresholds(
        consensus_iou=0.40, veto_cover=0.50, min_part_ratio=0.35,
        verify_fail_open_grades=()),
}


@dataclass
class FaceCandidate:
    """One accepted face claim: ``face`` is the blur-target seed (fed to
    ``bloom_face_box`` downstream); ``anchor`` is a head-scale box for the
    Kalman motion model (a covering head/pose box when one exists, else a
    face grown to head proportions — see ``_grow_to_head``); ``flags`` is
    the ``Ev`` bitmask explaining why it was accepted, kept as ``int`` for
    numpy-friendliness."""
    face: np.ndarray
    anchor: np.ndarray
    flags: int


@dataclass
class GateDebug:
    """Rejected claims for the TRACKING overlay — every claim that didn't
    become a :class:`FaceCandidate`, with the flags explaining why not."""
    rejected: list[tuple[np.ndarray, int]] = field(default_factory=list)


@dataclass
class EvidenceSummary:
    n_hits: int
    part_ratio: float
    consensus_ratio: float
    anchor_ratio: float


# ── geometry helpers ─────────────────────────────────────────────────────────

def _boxes_match(a: np.ndarray, b: np.ndarray, iou_thr: float) -> np.ndarray:
    """(N, M) bool — a[i] and b[j] describe the same region: IoU beyond
    ``iou_thr``, or either box's centre inside the other (two detectors, or a
    detector and a pose anchor, box the same face at very different scales)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), bool)
    return ((_iou_matrix(a, b) >= iou_thr)
            | _centres_inside(a, b)
            | _centres_inside(b, a).T)


def _veto_coverage(face_box: np.ndarray, negatives: np.ndarray,
                   cls_id: int) -> float:
    """Max, over negatives of one class, of area(face ∩ negative)/area(face)."""
    if len(negatives) == 0:
        return 0.0
    sel = negatives[negatives[:, 5].astype(int) == cls_id]
    if len(sel) == 0:
        return 0.0
    fx1, fy1, fx2, fy2 = (float(v) for v in face_box[:4])
    farea = max((fx2 - fx1) * (fy2 - fy1), 1e-6)
    ix1 = np.maximum(fx1, sel[:, 0]); iy1 = np.maximum(fy1, sel[:, 1])
    ix2 = np.minimum(fx2, sel[:, 2]); iy2 = np.minimum(fy2, sel[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return float((inter / farea).max())


def _veto_mask(boxes: np.ndarray, negatives: np.ndarray,
              cover_thr: float) -> np.ndarray:
    """(N,) bool — box[i] is substantially covered by some hand/foot
    negative of either class. Used by :func:`sustain_pool`, which has no
    per-candidate flags to carry an override through, so it is a flat veto."""
    if len(boxes) == 0 or len(negatives) == 0:
        return np.zeros(len(boxes), bool)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum((x2 - x1) * (y2 - y1), 1e-6)
    ix1 = np.maximum(x1[:, None], negatives[None, :, 0])
    iy1 = np.maximum(y1[:, None], negatives[None, :, 1])
    ix2 = np.minimum(x2[:, None], negatives[None, :, 2])
    iy2 = np.minimum(y2[:, None], negatives[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return ((inter / area[:, None]) >= cover_thr).any(axis=1)


def _grow_to_head(face: np.ndarray) -> np.ndarray:
    """Approximate head-scale box from a face — only reached via the
    close-up path, where by definition no head/pose box covers the
    candidate. Direct port of the legacy pseudo-head geometry: ×1.7 wide,
    ×1.9 tall, centre shifted up a quarter face-height for forehead/hair."""
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    w, h = face[2] - face[0], face[3] - face[1]
    pcx, pcy = cx, cy - 0.25 * h
    pw, ph = 1.7 * w, 1.9 * h
    return np.array([pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2,
                     face[4]], dtype=np.float32)


# ── per-claim evidence ───────────────────────────────────────────────────────

def _merge_face_claims(
    yolo_faces: np.ndarray, scrfd_faces: np.ndarray, consensus_iou: float,
) -> list[tuple[np.ndarray, "Ev", float]]:
    """Merge same-face claims from the two independent face sources.

    Returns one ``(box, flags, other_score)`` per distinct face: ``box`` is
    the higher-scoring source's box, ``other_score`` is the co-source's score
    when they agree (0.0 when the claim is solo) — the close-up path needs
    both scores, not just the winner's. ``scrfd_faces`` must already be
    filtered through ``scrfd.kps_plausible`` by the caller (see the module
    docstring); every claim from it carries ``SCRFD_KPS``.
    """
    claims: list[tuple[np.ndarray, "Ev", float]] = []
    used_scrfd = np.zeros(len(scrfd_faces), bool)
    same = _boxes_match(yolo_faces, scrfd_faces, consensus_iou) \
        if len(yolo_faces) and len(scrfd_faces) \
        else np.zeros((len(yolo_faces), len(scrfd_faces)), bool)
    for i in range(len(yolo_faces)):
        yf = yolo_faces[i]
        flags = Ev.YOLO_FACE
        box = yf[:5].copy()
        other_score = 0.0
        js = np.nonzero(same[i])[0] if same.size else np.empty(0, dtype=int)
        if len(js):
            j = int(js[np.argmax(scrfd_faces[js, 4])])
            used_scrfd[j] = True
            flags |= Ev.SCRFD_FACE | Ev.SCRFD_KPS | Ev.CONSENSUS
            if scrfd_faces[j, 4] > box[4]:
                other_score = float(yf[4])
                box = scrfd_faces[j, :5].copy()
            else:
                other_score = float(scrfd_faces[j, 4])
        claims.append((box, flags, other_score))
    for j in range(len(scrfd_faces)):
        if used_scrfd[j]:
            continue
        claims.append((scrfd_faces[j, :5].copy(),
                      Ev.SCRFD_FACE | Ev.SCRFD_KPS, 0.0))
    return claims


def _anchor_evidence(
    box: np.ndarray, flags: "Ev", dets: Detections,
    poses: Sequence[object], thr: GateThresholds,
) -> tuple[np.ndarray, "Ev"]:
    """Anatomical anchoring: HEAD_ANCHOR/BODY_MEMBER from the detector's own
    head class, POSE_ANCHOR/AXIS_OK/VETO_AXIS from pose (Phase 2 — ``poses``
    is always empty until libs/pose.py lands, so that half is inert today).
    Returns the head-scale anchor box the tracker's Kalman filter should
    follow, preferring a real head/pose box over the face-grown fallback.
    The anchor's score column is always overwritten with the *face*
    candidate's own score before returning — a head/pose box's incidental
    confidence must never stand in for how confident we are this is a face,
    since that score is what drives the tracker's spawn/high-low banding.
    """
    anchor_box: "np.ndarray | None" = None
    if len(dets.heads):
        covered = _boxes_match(box[None], dets.heads, thr.anchor_iou)[0]
        if covered.any():
            flags |= Ev.HEAD_ANCHOR
            idx = np.nonzero(covered)[0]
            anchor_box = dets.heads[idx[np.argmax(dets.heads[idx, 4])]].copy()
            if len(dets.bodies) and bool(
                    _centres_inside(dets.heads[idx], dets.bodies).any()):
                flags |= Ev.BODY_MEMBER
    for p in poses:
        p_anchor = getattr(p, "anchor", None)
        if p_anchor is None:
            continue
        if not _boxes_match(box[None], p_anchor[None], thr.anchor_iou)[0, 0]:
            continue
        flags |= Ev.POSE_ANCHOR
        if anchor_box is None:
            anchor_box = np.asarray(p_anchor[:5], np.float32).copy()
        torso = getattr(p, "torso", None)
        if torso is not None:
            shoulder_mid, hip_mid, _sw = torso
            face_c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            up = shoulder_mid - hip_mid
            if float(np.dot(face_c - shoulder_mid, up)) <= 0:
                flags |= Ev.VETO_AXIS      # wrong side of the body — legs, not head
            else:
                flags |= Ev.AXIS_OK
        break
    if anchor_box is None:
        anchor_box = _grow_to_head(box)
    anchor_box[4] = box[4]
    return anchor_box, flags


def _part_evidence(box: np.ndarray, flags: "Ev", parts: np.ndarray) -> "Ev":
    """PART_EYE_HIT/PART_NOSE_HIT/PART_MOUTH_HIT: does an eye/nose/mouth
    detection's centre land inside this face claim? (Ear is diagnostic only
    — see PART_EAR in detector.py — and is not consulted here.)"""
    if len(parts) == 0:
        return flags
    cls = parts[:, 5].astype(int)
    inside = _centres_inside(parts, box[None])[:, 0]
    if bool((inside & (cls == PART_EYE)).any()):
        flags |= Ev.PART_EYE_HIT
    if bool((inside & (cls == PART_NOSE)).any()):
        flags |= Ev.PART_NOSE_HIT
    if bool((inside & (cls == PART_MOUTH)).any()):
        flags |= Ev.PART_MOUTH_HIT
    return flags


def _veto_evidence(
    box: np.ndarray, flags: "Ev", negatives: np.ndarray, thr: GateThresholds,
) -> tuple["Ev", bool]:
    """VETO_HAND/VETO_FOOT from hand/foot coverage, folded with any VETO_AXIS
    already set by :func:`_anchor_evidence`. A live veto is overridden (never
    rejects the candidate, but is flagged ``VETO_OVERRIDDEN`` for the ledger)
    when the claim has ≥2 independent part hits or a pose-confirmed torso
    axis — a hand genuinely resting on a face, or a face the axis test
    itself already vouches for, must not lose its blur. Returns
    ``(flags, vetoed)``."""
    hand_cover = _veto_coverage(box, negatives, NEG_HAND)
    foot_cover = _veto_coverage(box, negatives, NEG_FOOT)
    live = bool(flags & Ev.VETO_AXIS)
    if hand_cover >= thr.veto_cover:
        flags |= Ev.VETO_HAND
        live = True
    if foot_cover >= thr.veto_cover:
        flags |= Ev.VETO_FOOT
        live = True
    if not live:
        return flags, False
    strong = _part_hit_count(int(flags)) >= 2 or bool(flags & Ev.AXIS_OK)
    if strong:
        flags |= Ev.VETO_OVERRIDDEN
        return flags, False
    return flags, True


def _closeup_ok(
    box: np.ndarray, flags: "Ev", other_score: float,
    frame_hw: tuple[int, int], thr: GateThresholds, det_conf: float,
    n_parts: int,
) -> bool:
    """Extreme-close-up fallback: reachable only when the normal anchored
    path failed (see ``gate_face_candidates``) — typically because the face
    fills the frame and there is no head/body/pose context to anchor to.
    Independent-model consensus (both sources agreeing, both confident)
    plus scale plus parts substitutes for the missing anatomical context."""
    if not (flags & Ev.CONSENSUS):
        return False
    fh, fw = frame_hw
    side = max(box[2] - box[0], box[3] - box[1])
    if side < thr.closeup_frac * min(fh, fw):
        return False
    floor = max(det_conf, 0.45)
    if box[4] < floor or other_score < floor:
        return False
    return n_parts >= 1


# ── public API ───────────────────────────────────────────────────────────────

def gate_face_candidates(
    dets: Detections,
    poses: Sequence[object],
    scrfd_faces: np.ndarray,
    frame_hw: tuple[int, int],
    thr: GateThresholds,
    det_conf: float,
) -> tuple[list[FaceCandidate], GateDebug]:
    """Raw detector/SCRFD face claims → gated :class:`FaceCandidate` list.

    ``scrfd_faces`` must already be filtered through ``scrfd.kps_plausible``
    by the caller (matching the existing ``scrfd_view()`` convention in
    ui.py) — this module only consumes faces that have already cleared that
    landmark-geometry check. ``poses`` is a sequence of pose-like objects
    with optional ``.anchor`` ((5,) xyxy+score or ``None``) and ``.torso``
    ((shoulder_mid, hip_mid, shoulder_width) or ``None``) attributes; pass
    ``[]`` until libs/pose.py (Phase 2) is wired in — every pose-only code
    path below is then simply inert.

    Returns ``(accepted, debug)``; ``debug.rejected`` carries every rejected
    claim's box and flags for the TRACKING overlay.
    """
    scrfd_faces = np.asarray(scrfd_faces, np.float32).reshape(-1, 5)
    accepted: list[FaceCandidate] = []
    rejected: list[tuple[np.ndarray, int]] = []

    for box, flags, other_score in _merge_face_claims(
            dets.faces, scrfd_faces, thr.consensus_iou):
        anchor_box, flags = _anchor_evidence(box, flags, dets, poses, thr)
        flags = _part_evidence(box, flags, dets.parts)
        flags, vetoed = _veto_evidence(box, flags, dets.negatives, thr)

        n_parts = _part_hit_count(int(flags))
        anchored = bool(flags & (Ev.HEAD_ANCHOR | Ev.POSE_ANCHOR))
        parts_ok = (not thr.require_parts) or n_parts >= 1
        accept = anchored and parts_ok and not vetoed

        if not accept and not vetoed:
            if _closeup_ok(box, flags, other_score, frame_hw, thr, det_conf,
                           n_parts):
                flags |= Ev.CLOSEUP
                accept = True

        if accept:
            accepted.append(FaceCandidate(face=box, anchor=anchor_box,
                                          flags=int(flags)))
        else:
            rejected.append((box, int(flags)))

    return accepted, GateDebug(rejected=rejected)


def sustain_pool(dets: Detections, thr: GateThresholds,
                 det_conf_low: float, det_conf: float) -> np.ndarray:
    """Raw head/face boxes scoring in ``[det_conf_low, det_conf)`` — BYTE's
    stage-2 sustain-only pool (``HeadTracker.update``'s ``sustain`` arg): may
    keep an existing track alive through a confidence dip, never spawn one.
    Vetoed boxes (substantially hand/foot-covered) are excluded even here —
    a sustained sock is still a sock."""
    parts = [a for a in (dets.heads, dets.faces) if len(a)]
    if not parts:
        return np.empty((0, 5), np.float32)
    boxes = np.concatenate(parts)
    m = (boxes[:, 4] >= det_conf_low) & (boxes[:, 4] < det_conf)
    boxes = boxes[m]
    if len(boxes) == 0:
        return boxes
    return boxes[~_veto_mask(boxes, dets.negatives, thr.veto_cover)]


def weak_faces(dets: Detections, scrfd_faces: np.ndarray) -> np.ndarray:
    """All raw face-shaped evidence (primary + landmark-plausible SCRFD), no
    gating at all — used only to refresh ``face_age`` on tracks that already
    exist (``HeadTracker.update``'s ``weak_faces`` arg). A confirmed track's
    own identity is what already established "this location is a face";
    weak evidence there just keeps it fresh and can never spawn or drive a
    track by itself."""
    scrfd_faces = np.asarray(scrfd_faces, np.float32).reshape(-1, 5)
    if len(dets.faces) and len(scrfd_faces):
        return np.concatenate([dets.faces, scrfd_faces])
    return dets.faces if len(dets.faces) else scrfd_faces


def summarize(ev: np.ndarray, hits: np.ndarray) -> EvidenceSummary:
    """Per-tracklet evidence ledger from its per-frame ``Ev`` flags — the
    composite prune's input (see libs/tracklets.clean_tracklets). Ratios are
    computed over *hit* frames only; coasted/interpolated frames carry no
    evidence and must not dilute or inflate the ratios."""
    hits = np.asarray(hits, dtype=bool)
    n = int(hits.sum())
    if n == 0:
        return EvidenceSummary(0, 0.0, 0.0, 0.0)
    hit_ev = np.asarray(ev, dtype=np.uint32)[hits]
    part_ratio = float(((hit_ev & _PART_HIT_MASK) != 0).sum()) / n
    consensus_ratio = float(((hit_ev & int(Ev.CONSENSUS)) != 0).sum()) / n
    anchor_ratio = float(((hit_ev & _ANCHOR_MASK) != 0).sum()) / n
    return EvidenceSummary(n, part_ratio, consensus_ratio, anchor_ratio)


def grade(summary: EvidenceSummary, thr: GateThresholds) -> str:
    """"A" — strong, independent evidence across most of the tracklet's
    life. "B" — clears the (profile-tunable) minimum bar. "C" — fails it: a
    long, confident-*looking* track with no anatomical backing (the
    sock/skin-misread signature) lands here regardless of score or length,
    which is the whole point of keeping a ledger instead of pruning on
    score and length alone."""
    if summary.n_hits == 0:
        return "C"
    if summary.part_ratio >= 0.5 and summary.anchor_ratio >= 0.5:
        return "A"
    if (summary.part_ratio >= thr.min_part_ratio
            or summary.consensus_ratio >= thr.min_consensus_ratio):
        return "B"
    return "C"
