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
If neither exists, :func:`download_model` fetches a pre-exported ``.onnx`` from
``RFDETR_URL`` (overridable via ``AVPP_RFDETR_URL``) into the default cache path
above — this is what the startup preflight in :mod:`libs.models` calls. You can
also export one yourself with the ``rfdetr`` package, e.g.::

    from rfdetr import RFDETRNano
    RFDETRNano().export(format="onnx")   # writes output/inference_model.onnx

then move/point it at one of the paths above. We deliberately do not pull the
``rfdetr``/torch runtime into this project — only the exported ``.onnx`` is
needed, keeping the strict ONNX-Runtime-only architecture intact.
"""

from __future__ import annotations

import os
import urllib.request
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

# Pre-exported RF-DETR ONNX, fetched on first run when no local copy exists.
# Overridable so a different export (e.g. the detection variant, or a self-hosted
# mirror) can be swapped in without code changes.
RFDETR_URL = os.environ.get(
    "AVPP_RFDETR_URL",
    "https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/"
    "rf-detr-seg-xxlarge.onnx",
)


def _model_path() -> Optional[Path]:
    env = os.environ.get("AVPP_RFDETR_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    if _DEFAULT_CACHE.is_file():
        return _DEFAULT_CACHE
    return None


def download_model(on_status: Optional[Callable[[str], None]] = None) -> Path:
    """Ensure the RF-DETR ONNX exists locally, downloading it if missing.

    Returns the path to the model. A custom ``AVPP_RFDETR_ONNX`` that already
    exists is returned untouched; otherwise the file is fetched from
    :data:`RFDETR_URL` into :data:`_DEFAULT_CACHE` via a ``.part`` temp file
    (atomic rename), mirroring the download pattern in ``pose_head.py``. Any
    failure propagates to the caller (the preflight logs it and degrades)."""
    existing = _model_path()
    if existing is not None:
        return existing

    _DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _DEFAULT_CACHE.with_suffix(".onnx.part")

    def _report(block: int, block_size: int, total: int) -> None:
        if on_status and total > 0:
            done = min(block * block_size, total)
            on_status(
                f"Downloading RF-DETR… {done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if on_status:
        on_status(f"Downloading RF-DETR from {RFDETR_URL}")
    urllib.request.urlretrieve(RFDETR_URL, tmp, reporthook=_report)
    tmp.replace(_DEFAULT_CACHE)
    return _DEFAULT_CACHE


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

            from .utils import best_onnx_providers, make_session

            self._status(f"Loading RF-DETR person detector ({path.name})…")
            providers = best_onnx_providers()
            try:
                # fp16 derivative (RDNA2/DirectML ~2×) + DML session options.
                sess = make_session(str(path), providers)
            except Exception:  # noqa: BLE001 — fp16/DML rejected → plain float32
                sess = ort.InferenceSession(str(path), providers=providers)
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
        """Return (boxes_cxcywh, logits, masks) from the model outputs by shape.

        RF-DETR emits a boxes tensor (last dim 4, normalised cxcywh) and a
        class-logits tensor (last dim = num classes); the ``-seg`` export adds a
        per-query mask-logits tensor (4-D: ``[1, Q, h, w]``). Identify each by
        rank / last dim rather than by name so any export variant works;
        ``masks`` is None for a detection-only export.
        """
        boxes = logits = masks = None
        for o in outs:
            a = np.asarray(o)
            if a.ndim == 4:
                masks = a[0]
            elif a.ndim == 3 and a.shape[-1] == 4:
                boxes = a[0]
            elif a.ndim == 3:
                logits = a[0]
        return boxes, logits, masks

    @staticmethod
    def _mask_to_frame(
        mask_logits: np.ndarray, box: np.ndarray, fw: int, fh: int
    ) -> np.ndarray:
        """Per-query mask logits → full-frame uint8 silhouette, clipped to its box.

        The mask comes out at the model's internal resolution in the same
        squashed-to-(W,H) space the boxes live in, so a direct resize to the
        frame aligns it with the box. Clipping to the person box drops any stray
        activation the seg head leaves outside the detection.
        """
        import cv2

        prob = _sigmoid(np.asarray(mask_logits, dtype=np.float32))
        prob = cv2.resize(prob, (fw, fh), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((fh, fw), dtype=np.uint8)
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        out[y1:y2, x1:x2] = (prob[y1:y2, x1:x2] > 0.5).astype(np.uint8)
        return out

    def detect(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[tuple[np.ndarray, float]], list[Optional[np.ndarray]]]:
        """Confident people as ``([(xyxy float32, score)], [mask | None])``.

        The second list is aligned with the first: each entry is a full-frame
        uint8 silhouette from the ``-seg`` model (1 inside the person), or None
        when the loaded export has no mask output. Masks are for the TRACKING
        panel overlay only — they never drive the blur.
        """
        if not self._ensure():
            return [], []
        try:
            inp = self._preprocess(frame_bgr)
            outs = self._sess.run(None, {self._in_name: inp})
            boxes, logits, masks = self._split_outputs(outs)
            if boxes is None or logits is None:
                return [], []

            scores = _sigmoid(logits)
            ncls = scores.shape[-1]
            col = _PERSON_CLASS if _PERSON_CLASS < ncls else int(scores.argmax(1).max())
            person = scores[:, col] if _PERSON_CLASS < ncls else scores.max(1)
            keep = person >= self._score_min
            if not np.any(keep):
                return [], []

            kept_masks = masks[keep] if masks is not None else None
            fh, fw = frame_bgr.shape[:2]
            persons: list[tuple[np.ndarray, float]] = []
            out_masks: list[Optional[np.ndarray]] = []
            for j, ((cx, cy, bw, bh), sc) in enumerate(
                    zip(boxes[keep], person[keep])):
                x1 = (cx - bw / 2) * fw
                y1 = (cy - bh / 2) * fh
                x2 = (cx + bw / 2) * fw
                y2 = (cy + bh / 2) * fh
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                box[[0, 2]] = box[[0, 2]].clip(0, fw - 1)
                box[[1, 3]] = box[[1, 3]].clip(0, fh - 1)
                if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
                    persons.append((box, float(sc)))
                    out_masks.append(
                        self._mask_to_frame(kept_masks[j], box, fw, fh)
                        if kept_masks is not None else None)
            return persons, out_masks
        except Exception as exc:  # noqa: BLE001 — one bad frame must not crash
            self._status(f"RF-DETR detect failed, degrading: {exc!r}")
            self.available = False
            return [], []
