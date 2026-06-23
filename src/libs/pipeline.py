"""Shared detection front-end: SCRFD faces ⊕ RTMW pose, ensembled.

cli.py and ui.py both ran near-identical "detect faces, refine close-ups, ask
the pose model for head boxes" code. This module centralises it and adds the
ensemble that fixes the two failures the project hit on its target footage
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

from .pose_rtmw import PoseFrame, RTMWPoseEstimator
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


def anchor_regions(
    pose: PoseFrame,
    faces: list,
    *,
    person_score_min: float = _ANCHOR_PERSON_SCORE_MIN,
) -> list[tuple[np.ndarray, float]]:
    """Head regions for people RF-DETR saw but no face/pose head covers.

    Returns [(xyxy, score)] the tracker may spawn a safety-net blur from. A
    person already covered by a kept face or a pose head region is skipped — the
    anchor exists only to catch a head nothing else found.
    """
    if not pose.person_boxes:
        return []
    face_boxes = [np.asarray(f.bbox[:4], dtype=np.float32) for f in faces]
    regions = [np.asarray(r, dtype=np.float32) for r in pose.head_regions]
    out: list[tuple[np.ndarray, float]] = []
    for pbox, pscore in pose.person_boxes:
        if pscore < person_score_min:
            continue
        head = _person_head_region(np.asarray(pbox, dtype=np.float32))
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
