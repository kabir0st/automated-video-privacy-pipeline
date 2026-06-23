"""RF-DETR person detection (ONNX Runtime) — the privacy safety-net anchor.

RF-DETR (Roboflow's real-time DETR; the Nano–Large variants are Apache-2.0)
detects whole-*person* boxes, not faces. It therefore cannot decide *what* to
blur on its own — but a confident person box is a strong privacy anchor:

  * fed to RTMW as the person detector it improves pose recall on people
    rtmlib's bundled YOLOX misses (cuddling, top-down, heavy occlusion), which
    in turn recovers faces the ensemble would otherwise never see;
  * when SCRFD *and* RTMW both fail to find a face on a person who is plainly
    present (head turned fully away, motion blur), the person box lets the
    tracker keep a head region blurred instead of leaking an un-anonymised face
    (see libs/tracker.py anchor tracks).

Design mirrors libs/pose_rtmw.py exactly:
  * the ONNX session is built lazily on first use and never destroyed;
  * any failure (model file absent, ORT init error, unexpected I/O) flips
    ``available`` to False so the pipeline degrades to SCRFD ⊕ RTMW-with-YOLOX
    instead of crashing;
  * GPU execution uses this project's ``best_onnx_providers()`` (DirectML →
    CUDA → ROCm → CPU), the same order every other model here follows, and the
    original CPU session is retired (kept alive), never destroyed.

The exported ONNX model is loaded from the first path that exists:
  1. the ``AVPP_RFDETR_ONNX`` environment variable, or
  2. ``~/.cache/avpp/rfdetr/rf-detr.onnx``.
Export one once with the ``rfdetr`` package, e.g.::

    from rfdetr import RFDETRNano
    RFDETRNano().export(format="onnx")   # writes output/inference_model.onnx

then move/point it at one of the paths above. We deliberately do not pull the
``rfdetr``/torch runtime into this project — only the exported ``.onnx`` is
needed, keeping the strict ONNX-Runtime-only architecture intact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ImageNet normalisation — RF-DETR preprocesses with these (RGB).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# COCO "person" class. RF-DETR keeps COCO's category ids in its label output,
# where person == 1. Overridable for a custom export.
_PERSON_CLASS = int(os.environ.get("AVPP_RFDETR_PERSON_CLASS", "1"))
# A person box must clear this score before it may anchor a blur.
_PERSON_SCORE_MIN = 0.50

_DEFAULT_CACHE = Path.home() / ".cache" / "avpp" / "rfdetr" / "rf-detr.onnx"


def _model_path() -> Optional[Path]:
    env = os.environ.get("AVPP_RFDETR_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    if _DEFAULT_CACHE.is_file():
        return _DEFAULT_CACHE
    return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class PersonDetector:
    """Lazy RF-DETR ONNX wrapper → confident COCO-person boxes."""

    def __init__(
        self,
        score_min: float = _PERSON_SCORE_MIN,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._score_min = score_min
        self._on_status = on_status
        self._sess = None
        self._retired: list = []          # never-destroy: keep old sessions alive
        self._in_name: str = ""
        self._in_hw: tuple[int, int] = (0, 0)
        self.available: Optional[bool] = None  # None = not yet attempted

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    def _ensure(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            path = _model_path()
            if path is None:
                raise FileNotFoundError(
                    "no RF-DETR ONNX found (set AVPP_RFDETR_ONNX or place it at "
                    f"{_DEFAULT_CACHE})")
            import onnxruntime as ort

            from .utils import best_onnx_providers

            self._status(f"Loading RF-DETR person detector ({path.name})…")
            sess = ort.InferenceSession(str(path), providers=best_onnx_providers())
            inp = sess.get_inputs()[0]
            self._in_name = inp.name
            # Static export shape is [N, 3, H, W]; fall back to 560² if dynamic.
            shape = inp.shape
            h = shape[2] if isinstance(shape[2], int) else 560
            w = shape[3] if isinstance(shape[3], int) else 560
            self._in_hw = (int(h), int(w))
            self._sess = sess
            self.available = True
            self._status(
                f"RF-DETR ready on {sess.get_providers()[0]} @ {w}×{h}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash pipeline
            self._status(f"RF-DETR unavailable: {exc!r}")
            self.available = False
        return self.available

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        h, w = self._in_hw
        img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])  # 1×3×H×W

    @staticmethod
    def _split_outputs(outs: list[np.ndarray]):
        """Return (boxes_cxcywh, logits) from the model outputs by shape.

        RF-DETR emits a boxes tensor (last dim 4, normalised cxcywh) and a
        class-logits tensor (last dim = num classes); identify by the last dim
        rather than by name so any export variant works.
        """
        boxes = logits = None
        for o in outs:
            a = np.asarray(o)
            if a.ndim == 3 and a.shape[-1] == 4:
                boxes = a[0]
            elif a.ndim == 3:
                logits = a[0]
        return boxes, logits

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return [(xyxy float32 in frame pixels, score)] for confident people."""
        if not self._ensure():
            return []
        try:
            inp = self._preprocess(frame_bgr)
            outs = self._sess.run(None, {self._in_name: inp})
            boxes, logits = self._split_outputs(outs)
            if boxes is None or logits is None:
                return []

            scores = _sigmoid(logits)
            ncls = scores.shape[-1]
            col = _PERSON_CLASS if _PERSON_CLASS < ncls else int(scores.argmax(1).max())
            person = scores[:, col] if _PERSON_CLASS < ncls else scores.max(1)
            keep = person >= self._score_min
            if not np.any(keep):
                return []

            fh, fw = frame_bgr.shape[:2]
            out: list[tuple[np.ndarray, float]] = []
            for (cx, cy, bw, bh), sc in zip(boxes[keep], person[keep]):
                x1 = (cx - bw / 2) * fw
                y1 = (cy - bh / 2) * fh
                x2 = (cx + bw / 2) * fw
                y2 = (cy + bh / 2) * fh
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                box[[0, 2]] = box[[0, 2]].clip(0, fw - 1)
                box[[1, 3]] = box[[1, 3]].clip(0, fh - 1)
                if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
                    out.append((box, float(sc)))
            return out
        except Exception as exc:  # noqa: BLE001 — one bad frame must not crash
            self._status(f"RF-DETR detect failed, degrading: {exc!r}")
            self.available = False
            return []
