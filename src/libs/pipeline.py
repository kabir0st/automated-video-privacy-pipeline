"""Detection front-end: blur only where there is real head/face evidence.

Built around RF-DETR's per-person *segmentation* (box + silhouette) as the body
truth on this project's footage (two people, intimate, top-down close-ups, odd
angles), with RTMW pose as the orientation compass and SCRFD/RTMW faces as the
fine detail. The silhouette's job is to *clip* the blur to a person's body, not
to invent a head where none was found.

The flow is **evidence-based head localisation**:

  1. RF-DETR -seg gives every person a box + a silhouette mask (the body truth).
  2. RTMW pose, run inside each person box, gives head keypoints and the torso
     axis — the orientation needed to know *which end is the head*.
  3. For each person we locate a head region from evidence only: a pose head box
     when it lands on the body, else a SCRFD/RTMW face detection sitting on that
     person. A person with neither is *not blurred* — there is nothing we can
     confidently point at as a head.
  4. When a *face* was found we blur its tight landmark hull; otherwise (pose
     head, face turned away) we blur the head region. The blur is clipped to the
     person's own silhouette downstream (see libs/utils.MaskBuilder) so it never
     paints background.

Earlier revisions deliberately over-blurred — every detected person got a head
*guessed* from the silhouette shape (PCA / topmost slab) so a missed face still
got covered. That guess fired on legs/torso/groin whenever the real face was out
of frame or undetected (the top-down false positive), so it was removed. The
trade-off is explicit: a face neither the detector nor pose can find is left
unblurred, rather than smearing the blur across whatever body part the silhouette
happened to point at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .pose_rtmw import (
    PoseFrame,
    RTMWPoseEstimator,
    _head_box_from_anchors,
    _KPT_THR,
)
from .utils import crop_face_patch, unproject_landmark

# ── head localisation tuning ──────────────────────────────────────────────────
# A pose-derived head box must cover at least this many body-mask pixels to be
# trusted; otherwise it floats off the body and the person is left unblurred
# unless a face detection lands on them.
_MIN_HEAD_MASK_PX = 12
# A face detection counts as "this head's face" when it overlaps the head region
# this much, or its centre lands inside the (slightly grown) region.
_FACE_IN_HEAD_IOU = 0.15
# A head region covering more than this fraction of its person box is a skeleton/
# landmark misfit (a "head" spread over most of the body), not a head — reject it
# so it cannot anchor a blur over the torso/legs.
_HEAD_MAX_PERSON_FRAC = 0.6
# Face-only degrade path (no RF-DETR person boxes): a single face box covering
# more than this fraction of the frame is almost certainly a detector misfire,
# not a real close-up — skip it so the maskless path can't smear the frame.
_FACE_ONLY_MAX_FRAME_FRAC = 0.5


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


def _head_fits_person(head: np.ndarray, person_box: np.ndarray) -> bool:
    """True when ``head`` is a plausible size for a head on ``person_box``.

    Rejects a "head" that spans most of the body box — the skeleton/landmark
    misfit (legs read as a torso, scattered keypoints) that fabricated a blur
    over the whole person. A head genuinely larger than its body box is never
    real, so this never drops a legitimate head."""
    pw = float(person_box[2] - person_box[0]); ph = float(person_box[3] - person_box[1])
    if pw <= 0 or ph <= 0:
        return True
    hw = float(head[2] - head[0]); hh = float(head[3] - head[1])
    return (hw * hh) <= _HEAD_MAX_PERSON_FRAC * pw * ph


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


# ── head localisation: pose orientation, validated on the body ───────────────

def locate_head(
    mask: Optional[np.ndarray],
    pbox: np.ndarray,
    pose_kpts: Optional[np.ndarray],
    fw: int,
    fh: int,
) -> Optional[np.ndarray]:
    """Locate one person's head region (xyxy) from pose evidence only.

    The pose head box points at the actual face via head keypoints / the torso
    axis. It is trusted only when it sits on the body silhouette (so a head
    guessed over background is rejected). Returns None when pose gives no usable
    head — the silhouette is *not* used to guess a head from its shape, because
    that guess fired on legs/torso whenever the real face was out of frame. A
    person with no pose head is picked up by a face detection upstream
    (``build_subjects``) or left unblurred.
    """
    if pose_kpts is None or len(pose_kpts) < 5:
        return None
    pk = np.asarray(pose_kpts[:, :2], dtype=np.float32)
    ps = np.asarray(pose_kpts[:, 2], dtype=np.float32)
    hb = _head_box_from_anchors(pk, ps, fw, fh)
    if hb is None:
        return None
    pose_head = hb[0]

    # A pose head box is trusted only when it actually overlaps the body.
    if mask is not None and _mask_px_in_box(pose_head, mask) < _MIN_HEAD_MASK_PX:
        return None
    return pose_head


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
    frame_area = float(fh * fw)
    for f in faces:
        fb = np.asarray(f.bbox[:4], dtype=np.float32)
        if fb[2] - fb[0] < 4 or fb[3] - fb[1] < 4:
            continue
        # A face box eating half the frame in the no-body degrade path is a
        # detector misfire; blurring it (maskless) is the random huge smear.
        if (fb[2] - fb[0]) * (fb[3] - fb[1]) > _FACE_ONLY_MAX_FRAME_FRAC * frame_area:
            continue
        out.append(Subject(fb.copy(), None, fb.copy(), f, float(f.det_score)))
    return out


def build_subjects(pf: PoseFrame, scrfd_faces: list, frame_shape: tuple) -> list[Subject]:
    """Turn a PoseFrame (RF-DETR people + RTMW poses) into anonymisation subjects.

    One subject per person, but *only* when there is head evidence for that
    person: a pose head box sitting on the body, or a detected face on the body.
    A person with neither is skipped (not blurred) — the body silhouette is never
    used to fabricate a head from its shape.
    """
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
        # A pose head spanning most of the body is a skeleton misfit, not a head.
        if head is not None and not _head_fits_person(head, pbox):
            head = None
        if head is not None:
            # Pose found a head on the body; a face on it refines the blur.
            face = _best_face_for_head(head, faces)
        else:
            # No pose head — blur this person only if a face landed on them; the
            # face box becomes the head region. No face → no evidence → skip.
            face = _best_face_for_head(pbox, faces)
            if face is None:
                continue
            head = np.asarray(face.bbox[:4], dtype=np.float32).copy()
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
