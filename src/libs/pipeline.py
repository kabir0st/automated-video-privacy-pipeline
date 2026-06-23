"""Shared detection front-end: SCRFD faces ⊕ RTMW pose, ensembled.

The GUI's preview and export paths both need identical "detect faces, refine
close-ups, ask the pose model for head boxes" logic. This module centralises it
and adds the ensemble that fixes the two failures the project hit on its target
footage
(two people in bed, cuddling, top-down close-ups):

  1. *Missed faces at odd angles.* SCRFD only fires near-frontal, so a face
     looking up/down/away was never detected and never blurred. RTMW estimates
     the head from whole-body context and contributes a face detection there.

  2. *Skin blurred as a face.* SCRFD occasionally fired on bare skin (a real
     problem on nude footage). We now drop any SCRFD detection that overlaps no
     RTMW head region — skin on a torso has no head keypoints near it — while
     keeping high-confidence SCRFD boxes so genuine faces are never lost.

Where both agree, SCRFD's 106-point mesh wins (a tighter hull than the 68 pose
points); RTMW fills in everywhere SCRFD is silent.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .pose_rtmw import _KPT_THR, PoseFrame, RTMWPoseEstimator, _torso_axis
from .utils import crop_face_patch, unproject_landmark

# An SCRFD detection this confident is kept even with no corroborating pose
# head region — strong frontal faces must never be gated away.
_HIGH_CONF = 0.70
# A pose face this close to a kept SCRFD face is the same person; SCRFD's
# denser landmarks win, so the pose duplicate is dropped.
_DUP_IOU = 0.40

# ── privacy safety-net anchor tuning ─────────────────────────────────────────
# An RF-DETR person box must clear this score before it may anchor a blur.
_ANCHOR_PERSON_SCORE_MIN = 0.55
# Fraction of a person box's height taken as the head region when nothing else
# located the head. Deliberately generous — for a privacy tool over-blur is the
# safe error — but kept head-ish (see _person_head_region). NOTE: this assumes
# the head sits at the top of the box, which is wrong for a lying/top-down
# subject; it is a last resort that only fires when pose found no head at all.
_ANCHOR_TOP_FRAC = 0.33
# Suppress an anchor whose head region a real face or pose head already covers —
# only a genuinely unseen head earns a safety-net blur.
_ANCHOR_COVERED_IOU = 0.10
# When no pose orients a person box, the top-of-box head guess is only trusted
# for a clearly-upright (tall) box; a top-down/lying box has its head elsewhere.
_ANCHOR_UPRIGHT_ASPECT = 1.3


# ── geometry helpers ─────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def _overlaps_region(face_box: np.ndarray, region: np.ndarray) -> bool:
    """True if the face box plausibly belongs to this head region.

    Tight on purpose: a real IoU overlap, or the face centre landing inside the
    region grown by only a small margin. A loose test let an oversized/offset
    head region validate SCRFD hits on bare skin far from any actual head.
    """
    if _iou(face_box, region) > 0.20:
        return True
    cx = (face_box[0] + face_box[2]) / 2
    cy = (face_box[1] + face_box[3]) / 2
    mx = (region[2] - region[0]) * 0.15
    my = (region[3] - region[1]) * 0.15
    return (region[0] - mx <= cx <= region[2] + mx
            and region[1] - my <= cy <= region[3] + my)


# ── SCRFD detection + close-up refinement ────────────────────────────────────

def scrfd_detect(
    app,
    frame: np.ndarray,
    *,
    det_score: float,
    face_aspect: float,
    close_up_ratio: float,
    close_up_target: int = 1024,
) -> list:
    """Run SCRFD, filter, and re-detect on an upscaled crop for close-ups."""
    fh, fw = frame.shape[:2]
    raw = app.get(frame)
    faces = [
        f for f in raw
        if (f.bbox[3] - f.bbox[1]) > 0
        and (f.bbox[2] - f.bbox[0]) / (f.bbox[3] - f.bbox[1]) >= face_aspect
        and float(f.det_score) >= det_score
    ]

    refined: list = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox[:4]
        if ((x2 - x1) * (y2 - y1)) / (fh * fw) > close_up_ratio:
            crop, (ox, oy, sc) = crop_face_patch(
                frame, face.bbox[:4], target_size=close_up_target)
            cfs = app.get(crop)
            if cfs:
                cf = cfs[0]
                if cf.landmark_2d_106 is not None:
                    cf.landmark_2d_106 = np.array(
                        [unproject_landmark(x, y, ox, oy, sc)
                         for x, y in cf.landmark_2d_106], dtype=np.float32)
                    cf.bbox[:4] = [
                        ox + cf.bbox[0] / sc, oy + cf.bbox[1] / sc,
                        ox + cf.bbox[2] / sc, oy + cf.bbox[3] / sc]
                refined.append(cf)
                continue
        refined.append(face)
    return refined


# ── ensemble ─────────────────────────────────────────────────────────────────

def merge_detections(scrfd_faces: list, pose: PoseFrame) -> list:
    """Gate SCRFD against pose head regions, then add the pose faces it missed."""
    regions = pose.head_regions
    if not regions:
        # Pose unavailable/absent → preserve legacy SCRFD-only behaviour.
        return list(scrfd_faces) + list(pose.faces)

    kept = [
        f for f in scrfd_faces
        if float(f.det_score) >= _HIGH_CONF
        or any(_overlaps_region(np.asarray(f.bbox[:4], dtype=np.float32), r)
               for r in regions)
    ]

    merged = list(kept)
    for pf in pose.faces:
        pb = np.asarray(pf.bbox[:4], dtype=np.float32)
        if not any(_iou(pb, np.asarray(kf.bbox[:4], dtype=np.float32)) > _DUP_IOU
                   for kf in kept):
            merged.append(pf)
    return merged


# ── privacy safety-net anchors ───────────────────────────────────────────────

def _person_head_region(pbox: np.ndarray, frac: float = _ANCHOR_TOP_FRAC) -> np.ndarray:
    """Coarse head region (xyxy) from a whole-person box: its top ``frac``,
    centred horizontally and kept no wider than ~1.4× its height so it stays
    head-shaped rather than a full-width band."""
    x1, y1, x2, y2 = (float(v) for v in pbox[:4])
    hh = (y2 - y1) * frac
    hw = min(x2 - x1, hh * 1.4)
    cx = (x1 + x2) / 2
    return np.array([cx - hw / 2, y1, cx + hw / 2, y1 + hh], dtype=np.float32)


def _best_pose_in_box(
    poses: list, box: np.ndarray, *, min_frac: float = 0.30
) -> Optional[np.ndarray]:
    """The pose with the largest share of its confident keypoints inside ``box``.

    Lets a person box borrow orientation from the pose that belongs to it; None
    when no pose overlaps the box well enough to trust."""
    best: Optional[np.ndarray] = None
    best_frac = min_frac
    for pose in poses:
        pose = np.asarray(pose, dtype=np.float32)
        conf = pose[:, 2] > _KPT_THR
        c = int(conf.sum())
        if c == 0:
            continue
        inside = (conf
                  & (pose[:, 0] >= box[0]) & (pose[:, 0] <= box[2])
                  & (pose[:, 1] >= box[1]) & (pose[:, 1] <= box[3]))
        frac = int(inside.sum()) / c
        if frac > best_frac:
            best_frac, best = frac, pose
    return best


def _head_from_torso(
    torso: tuple[np.ndarray, np.ndarray, float], pbox: np.ndarray
) -> Optional[np.ndarray]:
    """Head region placed a head-height beyond the shoulders along the body axis,
    clipped to the person box. None when the torso is implausibly proportioned."""
    sh, hp, sw = torso
    axis = sh - hp
    tlen = float(np.hypot(axis[0], axis[1]))
    if not (0.6 * sw <= tlen <= 3.5 * sw):
        return None
    up = axis / tlen
    hw, hh = 0.5 * sw, 0.6 * sw
    hc = sh + up * (0.6 * sw)
    region = np.array([hc[0] - hw, hc[1] - hh, hc[0] + hw, hc[1] + hh],
                      dtype=np.float32)
    region[[0, 1]] = np.maximum(region[[0, 1]], pbox[[0, 1]])
    region[[2, 3]] = np.minimum(region[[2, 3]], pbox[[2, 3]])
    if region[2] - region[0] < 4 or region[3] - region[1] < 4:
        return None
    return region


def _oriented_head_region(pbox: np.ndarray, poses: list) -> Optional[np.ndarray]:
    """Locate a person box's head using overlapping pose keypoints.

    Places the region at the *actual* head end via the body axis (handles
    sideways/inverted subjects). Falls back to the box's top slice only for a
    confidently-upright box, and returns None when nothing orients it — so a
    top-down/lying person never blurs its legs as a head."""
    pose = _best_pose_in_box(poses, pbox)
    if pose is not None:
        torso = _torso_axis(pose[:, :2], pose[:, 2])
        if torso is not None:
            region = _head_from_torso(torso, pbox)
            if region is not None:
                return region
    x1, y1, x2, y2 = (float(v) for v in pbox[:4])
    if (y2 - y1) < _ANCHOR_UPRIGHT_ASPECT * max(x2 - x1, 1.0):
        return None
    return _person_head_region(pbox)


def anchor_regions(
    pose: PoseFrame,
    faces: list,
    *,
    person_score_min: float = _ANCHOR_PERSON_SCORE_MIN,
) -> list[tuple[np.ndarray, float]]:
    """Head regions for people RF-DETR saw but no face/pose head covers.

    Returns [(xyxy, score)] the tracker may spawn a safety-net blur from. The
    region is oriented from the person's own pose keypoints (never a blind
    top-of-box guess), and a person already covered by a kept face or a pose head
    region is skipped — the anchor exists only to catch a head nothing else found.
    """
    if not pose.person_boxes:
        return []
    face_boxes = [np.asarray(f.bbox[:4], dtype=np.float32) for f in faces]
    regions = [np.asarray(r, dtype=np.float32) for r in pose.head_regions]
    out: list[tuple[np.ndarray, float]] = []
    for pbox, pscore in pose.person_boxes:
        if pscore < person_score_min:
            continue
        head = _oriented_head_region(np.asarray(pbox, dtype=np.float32), pose.poses)
        if head is None:
            continue
        if any(_iou(head, fb) > _ANCHOR_COVERED_IOU for fb in face_boxes):
            continue
        if any(_iou(head, r) > _ANCHOR_COVERED_IOU for r in regions):
            continue
        out.append((head, float(pscore)))
    return out


# ── pose backend factory + uniform interface ─────────────────────────────────

class _MediaPipeBackend:
    """Adapts the legacy MediaPipe PoseHeadEstimator to the PoseFrame API.

    MediaPipe yields head boxes (for tracker assist) but no face landmarks, so
    it contributes no face detections and gates nothing — SCRFD behaves exactly
    as before, just with head-box revival of lost tracks.
    """

    def __init__(self, on_status: Optional[Callable[[str], None]]) -> None:
        from .pose_head import PoseHeadEstimator
        self._mp = PoseHeadEstimator(on_status=on_status)
        self.kpt_format = "blazepose"

    def estimate(self, frame: np.ndarray) -> PoseFrame:
        head_boxes, poses = self._mp.estimate(frame)
        return PoseFrame(faces=[], head_boxes=head_boxes, head_regions=[],
                         poses=poses, kpt_format="blazepose")


def make_pose_backend(
    backend: str,
    mode: str = "performance",
    on_status: Optional[Callable[[str], None]] = None,
):
    """Construct a pose backend exposing ``estimate(frame) -> PoseFrame``.

    backend: "rtmw" (default, robust at odd angles), "mediapipe" (legacy),
    or "none"/"" to disable pose assist entirely.
    """
    b = (backend or "none").lower()
    if b in ("none", "off", ""):
        return None
    if b == "mediapipe":
        return _MediaPipeBackend(on_status)
    return RTMWPoseEstimator(mode=mode, on_status=on_status)


# ── one-call detection front-end ─────────────────────────────────────────────

def detect(
    app,
    frame: np.ndarray,
    pose,
    *,
    det_score: float,
    face_aspect: float,
    close_up_ratio: float,
    close_up_target: int = 1024,
) -> tuple[list, list, list, PoseFrame]:
    """Return (faces, head_boxes, anchors, pose_frame) for one frame.

    faces  — ensembled detections (each carries .bbox/.det_score and, when
             available, .landmark_2d_106) ready for the Kalman tracker.
    head_boxes — [(xyxy, score)] weak corrections for the tracker.
    anchors    — [(xyxy, score)] privacy safety-net head regions for people
                 RF-DETR saw but no face/pose head covers; pass to the tracker's
                 ``anchor_provider`` (opt-in) to spawn/sustain a blur there.
    pose_frame — raw PoseFrame (poses for overlay, regions for debugging).
    """
    scrfd = scrfd_detect(
        app, frame, det_score=det_score, face_aspect=face_aspect,
        close_up_ratio=close_up_ratio, close_up_target=close_up_target)
    pf = pose.estimate(frame) if pose is not None else PoseFrame()
    faces = merge_detections(scrfd, pf)
    anchors = anchor_regions(pf, faces)
    return faces, pf.head_boxes, anchors, pf
