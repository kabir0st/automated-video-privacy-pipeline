"""Whole-body pose estimation (RTMDet/YOLOX → RTMW) for robust head finding.

The frontal SCRFD face detector collapses on the footage this app targets —
two people lying in bed, cuddling, top-down close-ups where only the eyes,
nose and upper lip are visible. RTMW estimates 133 COCO-WholeBody keypoints
*per detected person*, and 68 of those are dense face landmarks (indices
23–90). Because they are anchored to a coherent body skeleton they keep
landing on the actual face at angles that kill a frontal detector — and, just
as importantly, they do **not** fire on a random patch of bare skin the way a
face detector can, which is exactly the false-positive that plagued the
previous pipeline on nude footage.

This module wraps rtmlib's ``Wholebody`` (YOLOX person detector + RTMW pose)
and turns each person into:

  * a *face detection* carrying the confident 68 face keypoints in the
    ``landmark_2d_106`` slot, so it flows through the existing Kalman tracker
    and landmark-hull MaskBuilder with no changes there;
  * a coarse *head box* used as a weak tracker correction (revives a track
    whose face the detector lost) and as the region SCRFD detections are
    gated against (a face box overlapping no head region is skin, not a face).

Design mirrors libs/pose_head.py: the model is created lazily on first use and
never destroyed, and any failure (rtmlib missing, model download blocked,
native init error) flips ``available`` to False so the pipeline degrades to
SCRFD-only instead of crashing.

DirectML: rtmlib hard-codes a single ONNX Runtime provider per ``device`` and
has no DirectML option. We let it build its CPU session, then swap in a
session built from this project's ``best_onnx_providers()`` (DirectML → CUDA →
CPU) — but only when a GPU provider is actually available, so the CPU path
pays nothing. The original sessions are retained, never destroyed, per the
never-destroy-sessions rule the ORT/DirectML side of this app lives by (see
libs/face_app.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ── COCO-WholeBody-133 keypoint layout ───────────────────────────────────────
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_FACE = slice(23, 91)               # 68 dense face landmarks
_HEAD_ANCHORS = (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR)

# Per-keypoint confidence to trust a point, and how many confident face points
# we need before trusting a tight face hull. RTMW happily scatters a handful of
# low-confidence "face" keypoints across bright/textured background (a sunlit
# window, bedding) when it mis-reads an odd pose — e.g. legs raised toward a
# window read as an upper body. Six points was far too few; require a real
# chunk of the 68-point mesh so noise can't fabricate a face.
_KPT_THR = 0.30
_MIN_FACE_KPTS = 12
# Mean confidence over the kept dense face landmarks before we emit a face.
_FACE_SCORE_MIN = 0.45
# A dense face is only trusted when the independent sparse-anchor head box
# (nose/eyes/ears) lands over it — co-location is strong corroboration that a
# real head is there, and it rejects scattered points that have no anchor.
_FACE_HEAD_IOU = 0.20
# A head box (anchor- or shoulder-derived) must clear this confidence before it
# may gate SCRFD, correct a track, or blur — keeps a low-confidence head guessed
# over scenery from doing any of those.
_HEAD_SCORE_MIN = 0.35

# rtmlib mode → effective model. "performance" = RTMW-x @384×288 (best
# accuracy, the project default for offline runs); "lightweight" = RTMW-l for
# fast CPU preview.
DEFAULT_MODE = "performance"


@dataclass
class PoseFrame:
    """One frame's pose output, consumed by the detection pipeline."""

    faces: list = field(default_factory=list)         # synthetic Face dets
    head_boxes: list = field(default_factory=list)    # [(xyxy float32, score)]
    head_regions: list = field(default_factory=list)  # [xyxy] for SCRFD gating
    poses: list = field(default_factory=list)         # [(K, 3) x,y,score]
    # Whole-person boxes from the RF-DETR detector that fed this pose pass — the
    # source of the privacy safety-net anchors (see libs/pipeline.py). Empty
    # when RF-DETR is unavailable (rtmlib's YOLOX ran instead).
    person_boxes: list = field(default_factory=list)  # [(xyxy float32, score)]
    # Per-person silhouettes from the RF-DETR ``-seg`` export, aligned with
    # ``person_boxes`` (full-frame uint8, or None). Drawn on the TRACKING panel
    # only; never used for the blur. Empty when the export has no mask output.
    person_masks: list = field(default_factory=list)  # [uint8 mask | None]
    kpt_format: str = "coco133"


def _new_face(bbox: np.ndarray, score: float,
              landmarks: Optional[np.ndarray]):
    """A synthetic InsightFace ``Face`` so pose detections are drop-in for the
    tracker (reads ``.bbox``/``.det_score``) and MaskBuilder (``.landmark_2d_106``)."""
    from insightface.app.common import Face

    f = Face(bbox=np.asarray(bbox, dtype=np.float32), kps=None,
             det_score=float(score))
    f.landmark_2d_106 = (None if landmarks is None
                         else np.asarray(landmarks, dtype=np.float32))
    return f


def _torso_axis(
    pts: np.ndarray, scr: np.ndarray
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """``(shoulder_mid, hip_mid, shoulder_width)`` for a confident torso, else None.

    Requires both shoulders and both hips above ``_KPT_THR``: the body axis
    (hip_mid → shoulder_mid → head) is the only reliable cue for which end of a
    person is the head on a sideways / inverted / lying subject.
    """
    if (scr[_L_SHOULDER] <= _KPT_THR or scr[_R_SHOULDER] <= _KPT_THR
            or scr[_L_HIP] <= _KPT_THR or scr[_R_HIP] <= _KPT_THR):
        return None
    sh = (pts[_L_SHOULDER] + pts[_R_SHOULDER]) / 2.0
    hp = (pts[_L_HIP] + pts[_R_HIP]) / 2.0
    sw = float(np.hypot(pts[_L_SHOULDER, 0] - pts[_R_SHOULDER, 0],
                        pts[_L_SHOULDER, 1] - pts[_R_SHOULDER, 1]))
    if sw < 4.0:
        return None
    return sh.astype(np.float32), hp.astype(np.float32), sw


def _head_box_from_anchors(
    pts: np.ndarray, scr: np.ndarray, fw: int, fh: int
) -> Optional[tuple[np.ndarray, float]]:
    """Coarse head box from nose/eyes/ears, oriented by the torso.

    Used when the dense face keypoints are not confident (head turned far away)
    so a strong turn-away still produces a head region to blur/track. The body
    axis (hip_mid → shoulder_mid → head) both *validates* an anchor-derived head
    (one landing on the hip side of the shoulders is a skeleton misfit onto
    legs/torso, not a face) and *places* the shoulders-only guess — and we refuse
    to guess a head we cannot orient rather than fabricate one over empty space.
    """
    def pt(i: int) -> tuple[float, float, float]:
        return float(pts[i, 0]), float(pts[i, 1]), float(scr[i])

    torso = _torso_axis(pts, scr)

    head = [(x, y, s) for x, y, s in (pt(i) for i in _HEAD_ANCHORS) if s > _KPT_THR]
    if head:
        cx = float(np.mean([p[0] for p in head]))
        cy = float(np.mean([p[1] for p in head]))
        score = float(np.mean([p[2] for p in head]))
        # A real head sits on the far side of the shoulders from the hips. When a
        # torso is visible, reject a "head" on the hip side — the legs-read-as-
        # upper-body misfit that fabricated a blur over the legs.
        if torso is not None:
            sh, hp, _sw = torso
            up = sh - hp                       # hips → shoulders → head
            if float(np.dot(np.asarray([cx, cy], dtype=np.float32) - sh, up)) <= 0:
                return None
        lex, ley, lev = pt(_L_EAR)
        rex, rey, rev = pt(_R_EAR)
        if lev > _KPT_THR and rev > _KPT_THR:
            base = float(np.hypot(lex - rex, ley - rey))
        else:
            lo, ro = pt(_L_EYE), pt(_R_EYE)
            if lo[2] > _KPT_THR and ro[2] > _KPT_THR:
                base = float(np.hypot(lo[0] - ro[0], lo[1] - ro[1])) * 1.8
            else:
                xs = [p[0] for p in head]
                ys = [p[1] for p in head]
                base = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 1.6
        w, h = 1.6 * base, 2.0 * base
    else:
        # No head anchors — guess the head only from a confident torso, placed a
        # head-height beyond the shoulders *along the body axis*. Data-driven, so
        # a sideways/inverted subject is handled; a torso we cannot establish
        # (bare shoulders, legs misread as shoulders) yields no guess at all,
        # where the old "above the shoulder line" hardcode fabricated one.
        if torso is None:
            return None
        sh, hp, sw = torso
        axis = sh - hp
        torso_len = float(np.hypot(axis[0], axis[1]))
        # A real torso is ~0.6×–3.5× shoulder width; a knee-to-knee or
        # shoulder-to-elbow "torso" misfit falls outside this and is rejected.
        if not (0.6 * sw <= torso_len <= 3.5 * sw):
            return None
        up = axis / torso_len
        w, h = 0.5 * sw, 0.6 * sw
        hc = sh + up * (0.6 * h)
        cx, cy = float(hc[0]), float(hc[1])
        score = float(min(scr[_L_SHOULDER], scr[_R_SHOULDER])) * 0.5

    if w < 4 or h < 4:
        return None
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                   dtype=np.float32)
    box[[0, 2]] = box[[0, 2]].clip(0, fw - 1)
    box[[1, 3]] = box[[1, 3]].clip(0, fh - 1)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    return box, score


class RTMWPoseEstimator:
    """Lazy rtmlib Wholebody wrapper producing face detections + head boxes."""

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        on_status: Optional[Callable[[str], None]] = None,
        use_rfdetr: bool = True,
    ) -> None:
        self._mode = mode
        self._on_status = on_status
        self._wb = None
        self._retired: list = []   # never-destroy: keep old sessions alive
        self.available: Optional[bool] = None  # None = not yet attempted
        self.kpt_format = "coco133"
        # Optional RF-DETR person detector: replaces rtmlib's bundled YOLOX as
        # the box source feeding RTMW pose (better recall on hard poses) and
        # supplies the person boxes the privacy safety-net anchors are built
        # from. Lazy — no model load until the first estimate(). Degrades to
        # YOLOX when RF-DETR is unavailable.
        self._person = None
        if use_rfdetr:
            from .person_detector import PersonDetector
            self._person = PersonDetector(on_status=on_status)

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    def _ensure(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            from rtmlib import Wholebody

            self._status(f"Loading RTMW pose model ({self._mode})…")
            wb = Wholebody(mode=self._mode, backend="onnxruntime", device="cpu")
            self._maybe_use_gpu(wb)
            self._wb = wb
            self.available = True
            self._status("RTMW pose ready")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash pipeline
            self._status(f"RTMW pose unavailable: {exc!r}")
            self.available = False
        return self.available

    def _maybe_use_gpu(self, wb) -> None:
        """Swap rtmlib's CPU sessions for DirectML/CUDA/ROCm ones when available.

        No-op on a CPU-only host (the providers list is just CPU), so the
        common dev path pays nothing and avoids building a throwaway session.
        Each swap is independent and best-effort: if a provider rejects a model
        (e.g. DirectML's strict shape validation), that model keeps its working
        CPU session instead of taking down the pipeline.
        """
        from .utils import best_onnx_providers, make_session

        providers = best_onnx_providers()
        if not providers or providers[0] == "CPUExecutionProvider":
            return

        for name, tool in (("detector", wb.det_model), ("pose", wb.pose_model)):
            try:
                # make_session adds the fp16 derivative (RDNA2/DirectML ~2×) and
                # DirectML-friendly session options; on any failure it raises and
                # the model keeps its working CPU session below.
                sess = make_session(tool.onnx_model, providers)
                self._retired.append(tool.session)  # keep alive; never destroy
                tool.session = sess
                self._status(f"RTMW {name} on {sess.get_providers()[0]}")
            except Exception as exc:  # noqa: BLE001 — keep CPU session, never crash
                self._status(f"RTMW {name} stays on CPU ({exc!r})")

    def estimate(self, frame_bgr: np.ndarray) -> PoseFrame:
        if not self._ensure():
            return PoseFrame()

        # Prefer RF-DETR person boxes (better recall than rtmlib's YOLOX, and
        # the safety-net anchor source); fall back to the bundled YOLOX when
        # RF-DETR is unavailable so behaviour degrades to the old path.
        if self._person:
            person_boxes, person_masks = self._person.detect(frame_bgr)
        else:
            person_boxes, person_masks = [], []
        if person_boxes:
            bboxes = [b.tolist() for b, _s in person_boxes]
            kpts, scores = self._wb.pose_model(frame_bgr, bboxes=bboxes)
        else:
            kpts, scores = self._wb(frame_bgr)
        kpts = np.asarray(kpts, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        out = PoseFrame(person_boxes=person_boxes, person_masks=person_masks)
        if kpts.ndim != 3 or kpts.shape[0] == 0:
            return out

        fh, fw = frame_bgr.shape[:2]
        for p in range(kpts.shape[0]):
            pk, ps = kpts[p], scores[p]
            out.poses.append(np.concatenate([pk, ps[:, None]], axis=1))

            face_pts = pk[_FACE]
            face_scr = ps[_FACE]
            keep = face_scr > _KPT_THR
            head = _head_box_from_anchors(pk, ps, fw, fh)
            # Drop a head box whose own anchors aren't confident enough to gate,
            # correct or blur — otherwise a head guessed over scenery still votes.
            if head is not None and head[1] < _HEAD_SCORE_MIN:
                head = None

            emit_face = False
            if int(keep.sum()) >= _MIN_FACE_KPTS:
                lm = face_pts[keep]
                x1, y1 = lm.min(0)
                x2, y2 = lm.max(0)
                fbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                fscore = float(face_scr[keep].mean())
                # Corroboration gate: trust the dense face only when it is
                # confident on average AND the independent sparse-anchor head box
                # lands over it. A real face satisfies both; RTMW's scattered
                # background "face" has no confident, overlapping anchor head and
                # is rejected here instead of blurring empty scenery.
                emit_face = (fscore >= _FACE_SCORE_MIN and head is not None
                             and _iou(fbox, head[0]) >= _FACE_HEAD_IOU)

            if emit_face:
                out.faces.append(_new_face(fbox, fscore, lm))
                # Head region for gating spans the dense face hull unioned with
                # the corroborating anchor head, so a slightly-offset SCRFD box
                # still counts as covered.
                out.head_boxes.append((fbox, fscore))
                out.head_regions.append(_union(fbox, head[0]))
            elif head is not None:
                hbox, hscore = head
                # No confident dense face this frame: contribute the coarse head
                # box *only* as a weak tracker correction (revives/steadies a
                # track for a person whose face was already seen) and as an SCRFD
                # gating region. We deliberately do NOT emit a synthetic face
                # here — a head box guessed from sparse anchors or bare shoulders
                # must never spawn a brand-new track and blur, or the pipeline
                # blurs bodies/empty space where no face was ever detected. This
                # preserves the tracker's "never blur a face never seen" rule.
                out.head_boxes.append((hbox, hscore))
                out.head_regions.append(hbox)

        return out


def _union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([min(a[0], b[0]), min(a[1], b[1]),
                     max(a[2], b[2]), max(a[3], b[3])], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))
