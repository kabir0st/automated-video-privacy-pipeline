"""SCRFD witness face detector — independent corroboration, every frame.

The primary Wholebody17 detector owns heads (all 360° orientations); SCRFD is
a dedicated face detector with 5-point landmarks that stays reliable where the
primary struggles, notably extreme close-ups where only part of a face fills
the frame — no whole head, no whole body, just an eye/cheek/forehead.

It runs on **every frame** as a first-class evidence source for
``libs/evidence.py``'s gate, and again offline as one of the cross-model
VERIFY witnesses. (It was a close-up-only fallback originally; it was promoted
because independent corroboration is what separates a real face from
face-shaped skin, and a witness that only speaks when the primary is silent
can't corroborate anything.) Its faces are never blur targets by themselves —
``kps_plausible`` cuts landmark-implausible claims before the gate sees them,
and the gate decides the rest. It never replaces the primary, which sees
back-of-heads it cannot.

This is a standalone ONNX wrapper, not a return of the insightface package
(dropped in the single-detector rewrite — it dragged heavy deps into the exe
and its FaceAnalysis created-and-destroyed sessions, which corrupts DirectML).
Same DirectML laws as libs/detector.py: one session, created lazily, never
destroyed, input shape-pinned. Any failure flips ``available`` to False and
the assist silently switches off.

Model file resolution order (first hit wins):
  1. ``AVPP_SCRFD_ONNX`` — explicit path to a local ``.onnx``;
  2. the frozen bundle (``sys._MEIPASS/models/det_10g.onnx``, PyInstaller);
  3. ``~/.cache/avpp/scrfd/det_10g.onnx``.
:func:`download_model` fetches the official insightface ``buffalo_l`` zip and
extracts only ``det_10g.onnx``; ``AVPP_SCRFD_URL`` may point at either a
direct ``.onnx`` or another ``.zip`` containing one.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_CACHE_DIR = Path.home() / ".cache" / "avpp" / "scrfd"
_MEMBER = "det_10g.onnx"
_URL = ("https://github.com/deepinsight/insightface/releases/download/"
        "v0.7/buffalo_l.zip")
_INPUT_HW = (640, 640)   # pinned into the graph for DirectML

# Why this is square, and why shrinking it is not available:
#
# Letterboxing a 16:9 frame into 640x640 leaves ~44 % of the tensor as black
# padding that every convolution still pays for, and SCRFD is not cheap — on
# this project's CPU path it costs about as much per frame as one full pass of
# the primary detector. Sizing the canvas to the source's aspect ratio (640x384
# for 16:9) would be *lossless*, since the letterbox scale min(W/fw, H/fh) is
# unchanged by shrinking the axis the content doesn't occupy — the same pixels
# at the same scale, just less padding. It measured at ~1.67x on that stage.
#
# The det_10g export makes it impossible. Its input is [1, 3, '?', '?'] where
# both spatial dims share ONE symbolic name, so fixing H also fixes W and the
# graph can only ever be square; and its outputs are hard-coded to 12800/3200/
# 800 rows (80^2 + 40^2 + 20^2, times _ANCHORS_PER_CELL) — the anchor counts for
# 640x640 specifically. Pinning any other shape fails outright in
# onnxruntime's fix_output_shapes:
#     ValueError: Can't replace existing fixed size of 384 with 640
# _decode() is written against (h, w) and would handle a rectangular grid fine,
# so this is purely a property of the published weights. Re-exporting SCRFD from
# insightface with dynamic axes would unlock it; swapping to a lighter variant
# (det_2.5g, det_500m) is the cheaper way to buy the same time.

# Faces below this never leave the module. SCRFD on bare skin is a known
# false-positive source on this footage, and every face here becomes a blur
# candidate via a pseudo-head, so the floor is deliberately higher than the
# primary detector's 0.05 parse floor — the tracker's spawn gate and the
# export's tracklet verification are the backstops, not substitutes.
_FLOOR = 0.45
_NMS_IOU = 0.45

# Corroboration floor: when SCRFD is asked "does a face exist here, even in
# profile?" to second-opinion the primary's face claims, a weaker answer
# counts than when its detection must stand alone as a blur candidate —
# witnesses gate other evidence, they never become boxes themselves.
WITNESS_FLOOR = 0.30

# The assist exists solely for faces big enough to defeat the primary —
# "partial face fills the frame". A small SCRFD-only face on a frame where
# the 360°-trained primary saw nothing confident is almost always skin or
# blanket texture misread as a face (the failure mode that got insightface
# dropped from this project once already), and it fires on *every* frame of
# head-free footage, so sub-close-up boxes must die before they can seed a
# track. Longest side vs the frame's short side.
_CLOSEUP_MIN_FRAC = 0.25

# SCRFD head geometry: anchor-free FPN, 2 anchors per cell at each stride;
# bbox outputs are centre-to-edge distances in stride units.
_STRIDES = (8, 16, 32)
_ANCHORS_PER_CELL = 2

# 5-point landmark plausibility gates (see kps_plausible). Deliberately
# loose — profile and upside-down faces are the norm in this footage — the
# check only has to kill the degenerate/scattered layouts that skin and
# fabric misreads produce, not grade real faces.
_KPS_RATIO = (0.4, 4.0)     # |eye-mid→mouth-mid| / |eye→eye| bounds
_KPS_MAX_COS = 0.8          # eye axis vs face axis: ≥ ~37° apart
_KPS_NOSE_T = (-0.3, 1.3)   # nose projection along the face axis


def model_path() -> Optional[Path]:
    """First existing model file per the resolution order in the module doc."""
    env = os.environ.get("AVPP_SCRFD_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        p = Path(bundle) / "models" / _MEMBER
        if p.is_file():
            return p
    cache = default_cache()
    if cache.is_file():
        return cache
    return None


def default_cache() -> Path:
    return _CACHE_DIR / _MEMBER


def download_model(
    on_status: Optional[Callable[[str], None]] = None,
) -> Path:
    """Ensure ``det_10g.onnx`` exists locally, downloading it if missing.

    The default source is the official insightface ``buffalo_l.zip``; only the
    detection member is extracted (matched by name suffix so the archive's
    internal layout doesn't matter). ``.part`` temp + atomic rename, same as
    the primary detector's downloader."""
    existing = model_path()
    if existing is not None:
        return existing

    url = os.environ.get("AVPP_SCRFD_URL", _URL)
    cache = default_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)

    def _report(block: int, block_size: int, total: int) -> None:
        if on_status and total > 0:
            done = min(block * block_size, total)
            on_status(f"Downloading SCRFD close-up detector… "
                      f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if on_status:
        on_status(f"Downloading SCRFD close-up detector from {url}")

    if url.endswith(".zip"):
        with tempfile.NamedTemporaryFile(
                suffix=".zip", dir=cache.parent, delete=False) as tf:
            archive = Path(tf.name)
        try:
            urllib.request.urlretrieve(url, archive, reporthook=_report)
            with zipfile.ZipFile(archive) as zf:
                names = [n for n in zf.namelist() if n.endswith(_MEMBER)]
                if not names:
                    raise FileNotFoundError(f"{_MEMBER} not in {url}")
                tmp = cache.with_suffix(".onnx.part")
                with zf.open(names[0]) as src, open(tmp, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                tmp.replace(cache)
        finally:
            archive.unlink(missing_ok=True)
    else:
        tmp = cache.with_suffix(".onnx.part")
        urllib.request.urlretrieve(url, tmp, reporthook=_report)
        tmp.replace(cache)
    return cache


def closeup_filter(boxes: np.ndarray, frame_hw: tuple[int, int],
                   min_score: float = _FLOOR) -> np.ndarray:
    """Keep only faces at close-up scale (see _CLOSEUP_MIN_FRAC) that also
    clear ``max(min_score, _FLOOR)`` — callers pass their confidence setting
    so the assist is never trusted below the bar the primary must clear.
    Pure numpy; applied by the *fallback* caller, not detect(), because the
    export verifier re-runs SCRFD on magnified crops where a real face is
    legitimately below this fraction."""
    if len(boxes) == 0:
        return boxes
    fh, fw = frame_hw
    side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    keep = ((boxes[:, 4] >= max(min_score, _FLOOR))
            & (side >= _CLOSEUP_MIN_FRAC * min(fh, fw)))
    return boxes[keep]


def _decode(outs: list, in_hw: tuple[int, int], floor: float) -> np.ndarray:
    """Raw SCRFD outputs → per-face rows in input pixels.

    ``outs`` is the session's output list: per-stride score tensors first,
    then per-stride bbox distance tensors, then (when the export carries
    them — ``det_10g`` does) per-stride 5-point landmark tensors. Rows are
    ``[x1, y1, x2, y2, score]`` — (K, 5) — without landmarks, or (K, 15)
    with the five ``x, y`` landmark pairs appended (left eye, right eye,
    nose, left/right mouth corner). Pure numpy so it unit-tests without a
    model."""
    h, w = in_hw
    fmc = len(_STRIDES)
    has_kps = len(outs) >= 3 * fmc
    ncol = 15 if has_kps else 5
    rows: list[np.ndarray] = []
    for i, stride in enumerate(_STRIDES):
        scores = np.asarray(outs[i], dtype=np.float32).reshape(-1)
        bbox = np.asarray(outs[i + fmc],
                          dtype=np.float32).reshape(-1, 4) * stride
        keep = scores >= floor
        if not keep.any():
            continue
        gh, gw = h // stride, w // stride
        xs, ys = np.meshgrid(np.arange(gw), np.arange(gh))
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
        centers = np.repeat(centers, _ANCHORS_PER_CELL,
                            axis=0).astype(np.float32)
        c, b, s = centers[keep], bbox[keep], scores[keep]
        cols = [c[:, 0] - b[:, 0], c[:, 1] - b[:, 1],
                c[:, 0] + b[:, 2], c[:, 1] + b[:, 3], s]
        if has_kps:
            kps = np.asarray(outs[i + 2 * fmc],
                             dtype=np.float32).reshape(-1, 10)[keep] * stride
            for j in range(5):
                cols.append(c[:, 0] + kps[:, 2 * j])
                cols.append(c[:, 1] + kps[:, 2 * j + 1])
        rows.append(np.stack(cols, axis=1).astype(np.float32))
    if not rows:
        return np.empty((0, ncol), np.float32)
    return np.concatenate(rows)


def kps_plausible(kps: np.ndarray) -> np.ndarray:
    """(K, 5, 2) landmark sets → (K,) bool — does the layout read as a face?

    Rotation-invariant on purpose (faces in this footage lie sideways and
    upside down): eyes and mouth must sit a sane distance apart relative to
    the eye spacing, the eye axis must cross the eye→mouth axis at a real
    angle (a collinear smear is fabric, not a face), and the nose must fall
    between the eye line and the mouth line along the face axis. NaN sets
    (a model without landmark outputs) pass — the gate fails open."""
    kps = np.asarray(kps, dtype=np.float32).reshape(-1, 5, 2)
    if len(kps) == 0:
        return np.zeros(0, bool)
    ok = np.ones(len(kps), bool)
    nan = np.isnan(kps).any(axis=(1, 2))
    e1, e2, nose = kps[:, 0], kps[:, 1], kps[:, 2]
    eye_mid = (e1 + e2) / 2
    mouth_mid = (kps[:, 3] + kps[:, 4]) / 2
    eye_ax = e2 - e1
    face_ax = mouth_mid - eye_mid
    d_eye = np.hypot(eye_ax[:, 0], eye_ax[:, 1])
    d_face = np.hypot(face_ax[:, 0], face_ax[:, 1])
    lo, hi = _KPS_RATIO
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = d_face / np.maximum(d_eye, 1e-6)
        ok &= (ratio >= lo) & (ratio <= hi)
        cos = np.abs((eye_ax * face_ax).sum(axis=1)) \
            / np.maximum(d_eye * d_face, 1e-6)
        ok &= cos <= _KPS_MAX_COS
        t = ((nose - eye_mid) * face_ax).sum(axis=1) \
            / np.maximum(d_face ** 2, 1e-6)
        ok &= (t >= _KPS_NOSE_T[0]) & (t <= _KPS_NOSE_T[1])
    ok[nan] = True
    return ok


class ScrfdDetector:
    """Lazy single-session SCRFD wrapper → (K, 5) face boxes, frame coords.

    Mirrors HeadDetector's lifecycle: built once on first use, never
    destroyed; any failure flips ``available`` to False and ``detect``
    returns empty forever after (the assist degrades, the pipeline doesn't
    notice)."""

    def __init__(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        self._on_status = on_status
        self._sess = None
        self._in_name = ""
        self.available: Optional[bool] = None
        self.last_ms = 0.0
        self._in_hw = _INPUT_HW

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
                    f"no SCRFD ONNX found (set AVPP_SCRFD_ONNX or place it "
                    f"at {default_cache()})")
            import onnxruntime as ort

            from .detector import _pinned_model
            from .utils import best_onnx_providers, make_session

            self._status(f"Loading SCRFD witness detector ({path.name})…")
            # Pinned once, on the first frame's aspect ratio, and never
            # rebuilt — the DirectML law forbids replacing a live session, so
            # a mid-video resolution change keeps the original shape (still
            # correct, just back to letterboxing).
            pinned = _pinned_model(str(path), self._in_hw)
            providers = best_onnx_providers()
            try:
                sess = make_session(pinned, providers)
            except Exception:  # noqa: BLE001 — fp16/DML rejected → plain fp32
                sess = ort.InferenceSession(pinned, providers=providers)
            self._in_name = sess.get_inputs()[0].name
            self._sess = sess
            self.available = True
            h, w = self._in_hw
            self._status(f"SCRFD witness detector ready on "
                         f"{sess.get_providers()[0]} @ {w}×{h}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._status(f"SCRFD witness detector unavailable: {exc!r}")
            self.available = False
        return self.available

    def detect(self, frame_bgr: np.ndarray,
               floor: float = _FLOOR) -> np.ndarray:
        """Detect faces on ``frame_bgr`` → (K, 5) xyxy+score, frame coords.

        Pass ``floor=WITNESS_FLOOR`` when the result corroborates other
        evidence instead of standing alone."""
        return self.detect_full(frame_bgr, floor)[0]

    def detect_full(
        self, frame_bgr: np.ndarray, floor: float = _FLOOR,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect faces → ``(boxes, kps)``: (K, 5) xyxy+score plus the
        matching (K, 5, 2) 5-point landmarks, all in frame coords (landmarks
        are NaN when the model export carries none).

        Aspect-preserving letterbox into the pinned input (top-left anchored,
        insightface convention) so close-up faces aren't distorted."""
        if not self._ensure():
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
        import cv2

        from .detector import _nms

        fh, fw = frame_bgr.shape[:2]
        h, w = self._in_hw
        scale = min(w / fw, h / fh)
        rw, rh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
        try:
            t0 = time.perf_counter()
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:rh, :rw] = cv2.resize(frame_bgr, (rw, rh),
                                          interpolation=cv2.INTER_LINEAR)
            blob = canvas[:, :, ::-1].astype(np.float32)   # BGR → RGB
            blob = (blob - 127.5) / 128.0
            blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
            outs = self._sess.run(None, {self._in_name: blob})
            # _nms keys on column 4 and carries any landmark columns along.
            rows = _nms(_decode(outs, self._in_hw, floor), _NMS_IOU)
            self.last_ms = (time.perf_counter() - t0) * 1e3
            if len(rows) == 0:
                return (np.empty((0, 5), np.float32),
                        np.empty((0, 5, 2), np.float32))
            rows[:, :4] /= scale
            rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, fw - 1)
            rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, fh - 1)
            ok = ((rows[:, 2] - rows[:, 0] >= 4)
                  & (rows[:, 3] - rows[:, 1] >= 4))
            rows = rows[ok]
            if rows.shape[1] >= 15:
                kps = (rows[:, 5:15] / scale).reshape(-1, 5, 2)
            else:
                kps = np.full((len(rows), 5, 2), np.nan, np.float32)
            return rows[:, :5].copy(), kps
        except Exception as exc:  # noqa: BLE001 — one bad frame ≠ crash
            self._status(f"SCRFD detect failed, degrading: {exc!r}")
            self.available = False
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
