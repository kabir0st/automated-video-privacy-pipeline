"""Head-region estimation from human pose, used to keep face tracks alive.

When InsightFace loses a face (person turns away, looks down, partial
occlusion) the Kalman tracker coasts on prediction alone. MediaPipe's
PoseLandmarker still sees the *person* in those frames — ears, nose and
shoulders survive head turns that kill a frontal face detector — so we
derive a coarse head box from the pose keypoints and feed it to the
tracker as a weak correction.

Design notes:
  * The landmarker is created lazily on first use and never destroyed,
    mirroring the never-destroy-sessions rule the ORT/DirectML side of
    this app lives by (and model load is slow enough to do exactly once).
  * IMAGE running mode: the inspector scrubs non-monotonically, so the
    VIDEO mode timestamp contract would be violated; our own Kalman
    filter supplies the temporal smoothing VIDEO mode would have added.
  * num_poses=2 — the footage this app targets has at most two people.
  * The .task model (~5 MB) is downloaded once into ~/.faceblur.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_DIR = Path.home() / ".faceblur"

# BlazePose landmark indices.
_HEAD_PTS = range(0, 11)        # nose, eyes (inner/outer), ears, mouth
_L_EAR, _R_EAR = 7, 8
_L_EYE_O, _R_EYE_O = 3, 6
_L_SHOULDER, _R_SHOULDER = 11, 12

_VIS_THRESHOLD = 0.4


def _model_path(on_status: Optional[Callable[[str], None]]) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / "pose_landmarker_lite.task"
    if not path.exists():
        if on_status:
            on_status("Downloading pose model (one-time, ~5 MB)…")
        tmp = path.with_suffix(".task.part")
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        tmp.rename(path)
    return path


class PoseHeadEstimator:
    """Lazy wrapper around MediaPipe PoseLandmarker producing head boxes.

    head_boxes() returns [(xyxy ndarray, score), …] — one entry per person
    whose head region could be estimated. Any failure (mediapipe missing,
    model download blocked, native init error) flips `available` to False
    and the estimator degrades to returning no boxes; the tracker then
    simply coasts on Kalman prediction alone.
    """

    def __init__(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        self._on_status = on_status
        self._landmarker = None
        self._mp = None
        self.available: Optional[bool] = None  # None = not yet attempted

    def _ensure(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            base = mp_tasks.BaseOptions(
                model_asset_path=str(_model_path(self._on_status)),
                delegate=mp_tasks.BaseOptions.Delegate.CPU,
            )
            opts = vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=2,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(opts)
            self._mp = mp
            self.available = True
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the pipeline
            if self._on_status:
                self._on_status(f"Pose assist unavailable: {exc!r}")
            self.available = False
        return self.available

    def head_boxes(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
        if not self._ensure():
            return []
        import cv2

        fh, fw = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(mp_img)

        boxes: list[tuple[np.ndarray, float]] = []
        for lms in result.pose_landmarks:
            box = self._head_box(lms, fw, fh)
            if box is not None:
                boxes.append(box)
        return boxes

    @staticmethod
    def _head_box(
        lms: list, fw: int, fh: int
    ) -> Optional[tuple[np.ndarray, float]]:
        def pt(i: int) -> tuple[float, float, float]:
            lm = lms[i]
            return lm.x * fw, lm.y * fh, lm.visibility

        head = [(x, y, v) for x, y, v in (pt(i) for i in _HEAD_PTS)
                if v > _VIS_THRESHOLD]

        if head:
            cx = float(np.mean([p[0] for p in head]))
            cy = float(np.mean([p[1] for p in head]))
            score = float(np.mean([p[2] for p in head]))
            # Head width baseline: ear-to-ear when available (≈ true head
            # width even from behind), else eye span, else point spread.
            lex, ley, lev = pt(_L_EAR)
            rex, rey, rev = pt(_R_EAR)
            if lev > _VIS_THRESHOLD and rev > _VIS_THRESHOLD:
                base = float(np.hypot(lex - rex, ley - rey))
            else:
                lo, ro = pt(_L_EYE_O), pt(_R_EYE_O)
                if lo[2] > _VIS_THRESHOLD and ro[2] > _VIS_THRESHOLD:
                    base = float(np.hypot(lo[0] - ro[0], lo[1] - ro[1])) * 1.6
                else:
                    xs = [p[0] for p in head]
                    ys = [p[1] for p in head]
                    base = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 1.5
            w, h = 1.5 * base, 1.9 * base
        else:
            # Head fully turned/occluded: hang an estimate above the
            # shoulders so a strong turn-away still yields a correction.
            ls, rs = pt(_L_SHOULDER), pt(_R_SHOULDER)
            if ls[2] < _VIS_THRESHOLD or rs[2] < _VIS_THRESHOLD:
                return None
            sw = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
            if sw < 4.0:
                return None
            cx = (ls[0] + rs[0]) / 2
            cy = (ls[1] + rs[1]) / 2 - 0.55 * sw
            w, h = 0.65 * sw, 0.8 * sw
            score = float(min(ls[2], rs[2])) * 0.5

        if w < 4 or h < 4:
            return None
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                       dtype=np.float32)
        box[[0, 2]] = box[[0, 2]].clip(0, fw - 1)
        box[[1, 3]] = box[[1, 3]].clip(0, fh - 1)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return None
        return box, score
