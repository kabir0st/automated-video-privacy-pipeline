"""Dedicated head detector — the primary blur source in the union pipeline.

A head is an angle-invariant blob: it does not stop being a head when the
person turns away, looks down, or lies upside down, which is exactly where
face detectors fail on this footage. The pipeline therefore blurs *heads*,
and uses face detections only as a second vote (see pipeline/fuse.py).

Model: deepghs/real_head_detection (YOLO11, single ``head`` class, Ultralytics
raw export: output ``(1, 5, N)`` = cx, cy, w, h, score in input-pixel
space, no NMS baked in). Variants ``head_detect_v0_{n,s,m,l}_yv11``; the
default is picked for the GPU target and can be swapped with ``AVPP_HEADDET``.

Resolution order (first hit wins): ``AVPP_HEADDET_ONNX`` -> the frozen bundle
(``sys._MEIPASS/models/<variant>.onnx``) -> ``~/.cache/avpp/headdet/`` (auto-
download). ``AVPP_HEADDET_URL`` overrides the download source.

One session per process, never rebuilt (session churn corrupts DirectML
device state and aborts the MIGraphX EP — the same rule every model wrapper
in libs/ follows).
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

DEFAULT_VARIANT = "head_detect_v0_l_yv11"
VARIANTS = ("head_detect_v0_n_yv11", "head_detect_v0_s_yv11",
            "head_detect_v0_m_yv11", "head_detect_v0_l_yv11")
_HF = "https://huggingface.co/deepghs/real_head_detection/resolve/main"
_CACHE_DIR = Path.home() / ".cache" / "avpp" / "headdet"

# Parse floor. Deliberately low: the tracker's spawn bar and the offline
# temporal prune are the real gates, and BYTE association needs weak boxes
# to sustain a track through a dip. deepghs' own F1-optimal threshold for
# these models is ~0.2.
FLOOR = 0.10
_NMS_IOU = 0.50
# Rotated passes are merged with the upright one at this IoU.
_MERGE_IOU = 0.55
_PAD_GREY = 114


def variant() -> str:
    v = os.environ.get("AVPP_HEADDET", DEFAULT_VARIANT).strip()
    return v if v in VARIANTS else DEFAULT_VARIANT


def _url() -> str:
    return os.environ.get("AVPP_HEADDET_URL") or f"{_HF}/{variant()}/model.onnx"


def default_cache() -> Path:
    return _CACHE_DIR / f"{variant()}.onnx"


def model_path() -> Optional[Path]:
    """First existing model file per the resolution order in the module doc."""
    env = os.environ.get("AVPP_HEADDET_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        p = Path(bundle) / "models" / f"{variant()}.onnx"
        if p.is_file():
            return p
    p = default_cache()
    return p if p.is_file() else None


def download_model(on_status: Optional[Callable[[str], None]] = None) -> Path:
    """Ensure the ONNX exists locally (``.part`` + atomic rename)."""
    existing = model_path()
    if existing is not None:
        return existing
    cache = default_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = _url()

    def _report(block: int, block_size: int, total: int) -> None:
        if on_status and total > 0:
            done = min(block * block_size, total)
            on_status(f"Downloading head detector… "
                      f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if on_status:
        on_status(f"Downloading head detector from {url}")
    tmp = cache.with_suffix(".onnx.part")
    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    tmp.replace(cache)
    return cache


def letterbox(frame_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Top-left anchored aspect-preserving resize onto a ``size``×``size``
    grey canvas → ``(canvas_bgr, scale)``. Top-left (not centred) so the
    inverse is a plain divide by ``scale``."""
    import cv2

    fh, fw = frame_bgr.shape[:2]
    s = size / float(max(fh, fw))
    nw, nh = max(1, int(round(fw * s))), max(1, int(round(fh * s)))
    canvas = np.full((size, size, 3), _PAD_GREY, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh),
                                  interpolation=cv2.INTER_LINEAR)
    return canvas, s


def decode(output: np.ndarray, scale: float, frame_hw: tuple[int, int],
           floor: float) -> np.ndarray:
    """Raw ``(1, 5, N)``/``(5, N)`` export output → ``(K, 5)`` xyxy+score in
    frame coordinates, clipped, NMS applied. Pure numpy."""
    from .detector import _nms

    arr = np.asarray(output, np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.T                                     # (N, 5)
    if arr.ndim != 2 or arr.shape[1] < 5 or len(arr) == 0:
        return np.empty((0, 5), np.float32)
    arr = arr[arr[:, 4] >= floor]
    if len(arr) == 0:
        return np.empty((0, 5), np.float32)
    fh, fw = frame_hw
    cx, cy, w, h, sc = (arr[:, i] for i in range(5))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                     axis=1) / max(scale, 1e-9)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, fw - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, fh - 1)
    ok = (boxes[:, 2] - boxes[:, 0] >= 4) & (boxes[:, 3] - boxes[:, 1] >= 4)
    rows = np.concatenate([boxes[ok], sc[ok, None]], axis=1)
    return _nms(rows.astype(np.float32), _NMS_IOU)


class HeadDetectorV2:
    """deepghs head detector. ``available`` is ``None`` until first use, then
    True/False; every failure degrades to "no boxes" and is reported once."""

    def __init__(self, size: int = 640,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
        self.size = int(size)
        self.available: Optional[bool] = None
        self.last_ms = 0.0
        self._sess = None
        self._in_name = ""
        self._on_status = on_status

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    def _ensure(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            path = model_path()
            if path is None:
                raise FileNotFoundError(
                    f"no head detector ONNX found (set AVPP_HEADDET_ONNX or "
                    f"place it at {default_cache()})")
            import onnxruntime as ort

            from .detector import _pinned_model
            from .utils import best_onnx_providers, make_session

            self._status(f"Loading head detector ({path.name})…")
            pinned = _pinned_model(str(path), (self.size, self.size))
            providers = best_onnx_providers()
            try:
                sess = make_session(pinned, providers)
            except Exception:  # noqa: BLE001 — fp16/DML rejected → plain fp32
                sess = ort.InferenceSession(pinned, providers=providers)
            self._in_name = sess.get_inputs()[0].name
            self._sess = sess
            self.available = True
            self._status(f"Head detector ready on {sess.get_providers()[0]} "
                         f"@ {self.size}×{self.size}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._status(f"Head detector unavailable: {exc!r}")
            self.available = False
        return self.available

    def _infer(self, frame_bgr: np.ndarray, floor: float) -> np.ndarray:
        canvas, s = letterbox(frame_bgr, self.size)
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0      # BGR → RGB
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        out = self._sess.run(None, {self._in_name: blob})[0]
        return decode(out, s, frame_bgr.shape[:2], floor)

    def detect(self, frame_bgr: np.ndarray,
               rotations: tuple[int, ...] = (0,),
               floor: float = FLOOR) -> np.ndarray:
        """Heads on ``frame_bgr`` → ``(K, 5)`` xyxy+score, frame coords.

        ``rotations`` beyond ``(0,)`` also run 90°/180°/270° copies and merge
        the unrotated boxes — cheap recall insurance for sideways and
        upside-down heads. No hallucination gate here: the union layer and
        the offline temporal prune decide what survives."""
        if not self._ensure():
            return np.empty((0, 5), np.float32)
        import cv2

        from .detector import _nms, _unrotate_boxes

        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        fh, fw = frame_bgr.shape[:2]
        try:
            t0 = time.perf_counter()
            parts = []
            for rot in rotations:
                img = frame_bgr if rot == 0 else cv2.rotate(frame_bgr,
                                                            codes[rot])
                b = self._infer(img, floor)
                parts.append(b if rot == 0 else _unrotate_boxes(b, rot, fw, fh))
            self.last_ms = (time.perf_counter() - t0) * 1e3
            rows = np.concatenate(parts) if parts else np.empty((0, 5),
                                                                np.float32)
            return _nms(rows.astype(np.float32), _MERGE_IOU)
        except Exception as exc:  # noqa: BLE001 — one bad frame ≠ crash
            self._status(f"head detect failed, degrading: {exc!r}")
            self.available = False
            return np.empty((0, 5), np.float32)
