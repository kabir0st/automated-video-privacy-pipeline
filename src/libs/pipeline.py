"""Segmentation-first detection front-end: body is the basis of truth.

Rewritten around the observation that RF-DETR's per-person *segmentation* is the
most reliable signal on this project's footage (two people, intimate, top-down
close-ups, odd angles). The old SCRFD-first pipeline leaked faces (the detector
only fires near-frontal) and blurred bare skin (it fired where no head was),
patched over with a brittle "anchor" safety-net and a coasting Kalman tracker.

The new flow is **segmentation as the spine, pose as the compass**:

  1. RF-DETR -seg gives every person a box + a silhouette mask (the body truth).
  2. RTMW pose, run inside each person box, gives the head keypoints and the
     torso axis — the orientation needed to know *which end is the head*.
  3. For each person we locate a head region: the pose head box when it lands on
     the body, else the head end of the silhouette itself. The head is therefore
     never placed off the body.
  4. When SCRFD or RTMW found a *face inside that head region* we blur its tight
     landmark hull; otherwise we blur the head region. A face detection that
     overlaps no located head region is skin/background and is ignored — the old
     false-positive simply cannot happen.

Every detected person is blurred (no "the face must have been seen" rule), and
the blur is clipped to each person's own silhouette downstream (see
libs/utils.MaskBuilder), so it never paints background. Over-blur is the only
remaining error mode — the right error for a privacy tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .pose_rtmw import (
    PoseFrame,
    RTMWPoseEstimator,
    _head_box_from_anchors,
    _KPT_THR,
)
from .utils import crop_face_patch, unproject_landmark

# ── head-from-mask tuning ─────────────────────────────────────────────────────
# Smallest silhouette (px in the largest connected blob) we will locate a head
# from — below this the mask is noise/sliver and we skip the person.
_MIN_MASK_PX = 64
# A pose-derived head box must cover at least this many body-mask pixels to be
# trusted; otherwise it floats off the body and we relocate from the silhouette.
_MIN_HEAD_MASK_PX = 12
# Eigenvalue ratio above which a silhouette is "elongated" enough that its
# principal axis is a trustworthy body axis (head end = the narrower end).
_PCA_ELONGATION = 1.5
# Head-ward slab: the top fraction (by projection onto the head direction) of
# the silhouette whose centroid becomes the head centre.
_HEAD_TIP_PCTL = 88
# A face detection counts as "this head's face" when it overlaps the head region
# this much, or its centre lands inside the (slightly grown) region.
_FACE_IN_HEAD_IOU = 0.15


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


def _centre_in_region(box: np.ndarray, region: np.ndarray, margin: float = 0.15) -> bool:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    mx = (region[2] - region[0]) * margin
    my = (region[3] - region[1]) * margin
    return (region[0] - mx <= cx <= region[2] + mx
            and region[1] - my <= cy <= region[3] + my)


def _mask_px_in_box(box: np.ndarray, mask: np.ndarray) -> int:
    """Count body-mask pixels under a box — used to test a head box is on-body."""
    fh, fw = mask.shape[:2]
    x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
    x2 = min(fw, int(box[2])); y2 = min(fh, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(np.count_nonzero(mask[y1:y2, x1:x2]))


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
    """Run SCRFD, filter, and re-detect on an upscaled crop for close-ups.

    Unchanged from before, but its role is now *refinement only*: a face is used
    only when it lands inside a head region already located from the body, so a
    stray hit on skin/background is harmless (it matches no head).
    """
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


# ── head localisation: body silhouette + pose orientation ────────────────────

def _largest_blob_points(
    mask: np.ndarray, pbox: np.ndarray
) -> Optional[tuple[np.ndarray, int, int]]:
    """Float (x, y) pixels of the largest connected silhouette blob in ``pbox``.

    Returns ``(pts_local, x_off, y_off)`` in coordinates local to the person
    box, or None when the mask is too small / empty. Cropping to the box keeps
    the PCA over a few thousand pixels and drops any stray blob outside it.
    """
    fh, fw = mask.shape[:2]
    x1 = max(0, int(pbox[0])); y1 = max(0, int(pbox[1]))
    x2 = min(fw, int(pbox[2])); y2 = min(fh, int(pbox[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = np.ascontiguousarray(mask[y1:y2, x1:x2]).astype(np.uint8)
    if int(sub.sum()) < _MIN_MASK_PX:
        return None
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    if num <= 1:
        return None
    # Largest non-background component (skip label 0).
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lbl == largest)
    if xs.size < _MIN_MASK_PX:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    return pts, x1, y1


def _head_from_mask(mask: np.ndarray, pbox: np.ndarray) -> Optional[np.ndarray]:
    """Head region (xyxy) at the head end of the body silhouette.

    Orientation comes from the silhouette's principal axis (PCA): an elongated
    body's head is its *narrower* extreme (a head is thinner than shoulders /
    hips / legs). A near-round blob (curled / top-down) has no reliable axis, so
    we fall back to its topmost slab — a bounded guess that the per-person blur
    clip keeps on the body either way. This is the last resort used only when
    pose gave no usable head; with pose present the pose head box wins.
    """
    found = _largest_blob_points(mask, pbox)
    if found is None:
        return None
    pts, xo, yo = found

    mean, eigvec, eigval = cv2.PCACompute2(pts, mean=None)
    major = eigvec[0].astype(np.float32)
    elong = float(eigval[0, 0]) / max(float(eigval[1, 0]), 1e-6)
    perp = np.array([-major[1], major[0]], dtype=np.float32)

    proj_major = pts @ major
    perp_proj = pts @ perp
    if elong >= _PCA_ELONGATION:
        lo = proj_major <= np.percentile(proj_major, 25)
        hi = proj_major >= np.percentile(proj_major, 75)
        w_lo = float(perp_proj[lo].std()) if lo.any() else 0.0
        w_hi = float(perp_proj[hi].std()) if hi.any() else 0.0
        up = major if w_hi <= w_lo else -major
        scale = 2.0 * min(w_lo, w_hi) + 1.0
    else:
        # No trustworthy axis: head at the top of the blob, size from its width.
        up = np.array([0.0, -1.0], dtype=np.float32)
        scale = float(pts[:, 0].max() - pts[:, 0].min()) * 0.5 + 1.0

    proj = pts @ up
    slab = pts[proj >= np.percentile(proj, _HEAD_TIP_PCTL)]
    hc = slab.mean(axis=0) if slab.size else pts[np.argmax(proj)]

    hw = max(0.55 * scale, 6.0)
    hh = max(0.65 * scale, 6.0)
    cx, cy = float(hc[0]) + xo, float(hc[1]) + yo
    box = np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float32)
    fh, fw = mask.shape[:2]
    box[[0, 2]] = box[[0, 2]].clip(0, fw - 1)
    box[[1, 3]] = box[[1, 3]].clip(0, fh - 1)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    return box


def locate_head(
    mask: Optional[np.ndarray],
    pbox: np.ndarray,
    pose_kpts: Optional[np.ndarray],
    fw: int,
    fh: int,
) -> Optional[np.ndarray]:
    """Locate one person's head region (xyxy), kept on the body.

    Pose head box first (it points at the actual face via head keypoints / the
    torso axis), but only when it sits on the body silhouette; otherwise the
    head end of the silhouette itself. Returns None only when neither pose nor a
    usable mask is available (a face-only fallback handles that upstream).
    """
    pose_head: Optional[np.ndarray] = None
    if pose_kpts is not None and len(pose_kpts) >= 5:
        pk = np.asarray(pose_kpts[:, :2], dtype=np.float32)
        ps = np.asarray(pose_kpts[:, 2], dtype=np.float32)
        hb = _head_box_from_anchors(pk, ps, fw, fh)
        if hb is not None:
            pose_head = hb[0]

    # A pose head box is trusted only when it actually overlaps the body.
    if pose_head is not None and mask is not None:
        if _mask_px_in_box(pose_head, mask) < _MIN_HEAD_MASK_PX:
            pose_head = None

    if pose_head is not None:
        return pose_head
    if mask is not None:
        return _head_from_mask(mask, pbox)
    return None


# ── face refinement (tighten a head region to a detected face) ───────────────

def _best_face_for_head(head: np.ndarray, faces: list) -> Optional[object]:
    """The detected face that best belongs to this head region, or None.

    Among faces overlapping the head region (IoU or centre-inside), prefer the
    one with the most landmarks (SCRFD's 106-pt mesh over RTMW's 68) then the
    highest detection score — a tighter, better hull. A face overlapping no head
    region is never returned, so skin/background hits are dropped.
    """
    best = None
    best_key = (-1, -1.0)
    for f in faces:
        fb = np.asarray(f.bbox[:4], dtype=np.float32)
        if _iou(fb, head) < _FACE_IN_HEAD_IOU and not _centre_in_region(fb, head):
            continue
        lm = getattr(f, "landmark_2d_106", None)
        n_lm = 0 if lm is None else len(lm)
        key = (n_lm, float(f.det_score))
        if key > best_key:
            best_key, best = key, f
    return best


# ── subjects: one per detected person ────────────────────────────────────────

@dataclass
class Subject:
    """One person to anonymise, derived from the body segmentation."""

    person_box: np.ndarray              # xyxy float32 (RF-DETR)
    mask: Optional[np.ndarray]          # body silhouette (uint8, 1 inside) or None
    head: np.ndarray                    # head region xyxy float32, on the body
    face: Optional[object]              # refining Face w/ landmark_2d_106, or None
    score: float


def _face_only_subjects(faces: list, fh: int, fw: int) -> list[Subject]:
    """Degrade path: no RF-DETR people available → blur detected faces directly.

    Keeps a baseline blur when segmentation is unavailable (model missing or a
    failed frame) instead of going dark. Each face becomes its own maskless
    subject whose head region is the face box.
    """
    out: list[Subject] = []
    for f in faces:
        fb = np.asarray(f.bbox[:4], dtype=np.float32)
        if fb[2] - fb[0] < 4 or fb[3] - fb[1] < 4:
            continue
        out.append(Subject(fb.copy(), None, fb.copy(), f, float(f.det_score)))
    return out


def build_subjects(pf: PoseFrame, scrfd_faces: list, frame_shape: tuple) -> list[Subject]:
    """Turn a PoseFrame (RF-DETR people + RTMW poses) into anonymisation subjects."""
    fh, fw = frame_shape[:2]
    if not pf.person_boxes:
        # No bodies detected — fall back to whatever faces we have.
        return _face_only_subjects(list(scrfd_faces) + list(pf.faces), fh, fw)

    faces = list(scrfd_faces) + list(pf.faces)
    n = len(pf.person_boxes)
    poses_aligned = len(pf.poses) == n  # RF-DETR path: poses run per person box
    subjects: list[Subject] = []
    for i, (pbox, pscore) in enumerate(pf.person_boxes):
        pbox = np.asarray(pbox, dtype=np.float32)
        mask = pf.person_masks[i] if i < len(pf.person_masks) else None
        pose_kpts = pf.poses[i] if poses_aligned else _best_pose_in_box(pf.poses, pbox)
        head = locate_head(mask, pbox, pose_kpts, fw, fh)
        if head is None:
            continue
        face = _best_face_for_head(head, faces)
        subjects.append(Subject(pbox, mask, head, face, float(pscore)))
    return subjects


def _best_pose_in_box(
    poses: list, box: np.ndarray, *, min_frac: float = 0.30
) -> Optional[np.ndarray]:
    """The pose with the largest share of its confident keypoints inside ``box``.

    Used only on the YOLOX fallback where poses are not index-aligned with the
    person boxes; on the RF-DETR path poses are aligned by index instead.
    """
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


# ── pose backend factory + uniform interface ─────────────────────────────────

class _MediaPipeBackend:
    """Adapts the legacy MediaPipe PoseHeadEstimator to the PoseFrame API.

    MediaPipe yields head boxes and poses but no per-person segmentation, so the
    segmentation-first path degrades to pose-only head localisation (head box
    from keypoints, no silhouette clip) when this backend is selected.
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

    backend: "rtmw" (default — RF-DETR seg + RTMW pose, the segmentation-first
    path), "mediapipe" (legacy, pose-only), or "none"/"" to disable.
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
) -> tuple[list[Subject], PoseFrame]:
    """Return ``(subjects, pose_frame)`` for one frame.

    subjects   — one per detected person: body box, silhouette, on-body head
                 region, and the refining face detection (when one lands on the
                 head). Feed to the SubjectTracker.
    pose_frame — raw PoseFrame (people, masks, poses) for the tracking overlay.
    """
    scrfd = scrfd_detect(
        app, frame, det_score=det_score, face_aspect=face_aspect,
        close_up_ratio=close_up_ratio, close_up_target=close_up_target,
    ) if app is not None else []
    pf = pose.estimate(frame) if pose is not None else PoseFrame()
    subjects = build_subjects(pf, scrfd, frame.shape)
    return subjects, pf
