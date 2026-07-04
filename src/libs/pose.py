"""RTMPose-m body7 (SimCC, COCO-17) — pose-derived anatomical anchor + torso
axis for libs/evidence.py's gate (Phase 2).

The gate (evidence.py) already has a pose code path wired and inert: it
duck-types ``.anchor`` ((5,) xyxy+score head box, or ``None``) and ``.torso``
((shoulder_mid, hip_mid, shoulder_width), or ``None``) off whatever objects
``poses`` contains, and ``ui.py`` has been passing ``poses=[]`` since Phase 1.
This module supplies the real thing: :class:`PoseEstimator` runs a top-down
RTMPose pass on each of the frame's (up to two) YOLOv9 body boxes and returns
a :class:`PersonPose` per person.

Unlike the removed RTMW-133 pipeline (see git history), body7 has no dense
face mesh — only 17 sparse COCO keypoints. That's enough: the gate only ever
needs a coarse head-anchor box and the shoulder→hip axis, never a face hull,
so a much smaller/cheaper model suffices. ``anchor``/``torso`` are direct
ports of the legacy ``pose_rtmw._head_box_from_anchors``/``_torso_axis`` —
same anatomical reasoning (reject a "head" on the hip side of the torso;
place a head guess along the body axis when no anchor keypoint is confident),
adapted to a 17-point layout.

Same DirectML laws as libs/scrfd.py: one session, created lazily, never
destroyed, input shape-pinned via ``detector._pinned_model``. Any failure
flips ``available`` to False and the gate simply never sees a pose anchor —
``HEAD_ANCHOR`` (the detector's own head class) carries recall instead, per
the module docstring in evidence.py.

Model file resolution order (first hit wins):
  1. ``AVPP_POSE_ONNX`` — explicit path to a local ``.onnx``;
  2. the frozen bundle (``sys._MEIPASS/models/rtmpose_m_body7.onnx``, PyInstaller);
  3. ``~/.cache/avpp/pose/rtmpose_m_body7.onnx``.
:func:`download_model` fetches the official mmpose ONNX SDK zip and extracts
only ``end2end.onnx``; ``AVPP_POSE_URL`` may point at either a direct
``.onnx`` or another ``.zip`` containing one (e.g. the smaller ``rtmpose-s``
export, if the exe size budget is ever tight).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_CACHE_DIR = Path.home() / ".cache" / "avpp" / "pose"
_LOCAL_NAME = "rtmpose_m_body7.onnx"
_MEMBER = "end2end.onnx"
_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-"
        "e48f03d0_20230504.zip")
_INPUT_WH = (192, 256)   # (w, h) — matches the SDK export's preprocess spec
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_SIMCC_RATIO = 2.0
_BBOX_PADDING = 1.25

# COCO-17 keypoint indices (body7's output layout).
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_HEAD_ANCHORS = (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR)

# Per-keypoint confidence to trust a point (same bar the legacy RTMW pipeline
# used — permissive on purpose, this footage has plenty of odd poses/motion
# blur, and a low-confidence anchor still has to clear _HEAD_SCORE_MIN below).
_KPT_THR = 0.30
# A head box (anchor- or shoulder-derived) must clear this confidence before
# it may anchor a face candidate in the gate — keeps a low-confidence guess
# over scenery/background from voting there. Mirrors scrfd.py's WITNESS_FLOOR
# pattern: this is a witness bar, not the raw per-keypoint threshold.
_HEAD_SCORE_MIN = 0.35

MAX_PEOPLE = 2


@dataclass
class PersonPose:
    """One person's pose this frame — what libs/evidence.py's gate consumes.

    ``kpts`` is (17, 3) ``x, y, score`` in frame coordinates, kept for the
    TRACKING panel's skeleton overlay only (the gate never reads it directly,
    duck-typing ``.anchor``/``.torso`` instead — see evidence.py). ``anchor``
    is a (5,) xyxy+score head-scale box, or ``None`` when no anchor cleared
    ``_HEAD_SCORE_MIN`` (pose *abstains*, never vetoes). ``torso`` is
    ``(shoulder_mid, hip_mid, shoulder_width)``, or ``None`` when shoulders or
    hips weren't both confident."""
    kpts: np.ndarray
    anchor: Optional[np.ndarray]
    torso: Optional[tuple[np.ndarray, np.ndarray, float]]


def model_path() -> Optional[Path]:
    """First existing model file per the resolution order in the module doc."""
    env = os.environ.get("AVPP_POSE_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        p = Path(bundle) / "models" / _LOCAL_NAME
        if p.is_file():
            return p
    cache = default_cache()
    if cache.is_file():
        return cache
    return None


def default_cache() -> Path:
    return _CACHE_DIR / _LOCAL_NAME


def download_model(
    on_status: Optional[Callable[[str], None]] = None,
) -> Path:
    """Ensure the pose ONNX exists locally, downloading it if missing.

    The default source is the official mmpose ONNX SDK zip; only the
    ``end2end.onnx`` member is extracted (matched by suffix, like
    scrfd.py's downloader). ``.part`` temp + atomic rename."""
    existing = model_path()
    if existing is not None:
        return existing

    url = os.environ.get("AVPP_POSE_URL", _URL)
    cache = default_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)

    def _report(block: int, block_size: int, total: int) -> None:
        if on_status and total > 0:
            done = min(block * block_size, total)
            on_status(f"Downloading pose estimator… "
                      f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if on_status:
        on_status(f"Downloading pose estimator from {url}")

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


# ── pure-numpy pre/post-process (unit-testable without a model) ─────────────

def _bbox_xyxy2cs(bbox: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], dtype=np.float32)
    return center, scale


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _get_warp_matrix(center: np.ndarray, scale: np.ndarray,
                     output_wh: tuple[int, int]) -> np.ndarray:
    """2x3 affine mapping the (center, scale) box in the source image to
    ``output_wh`` — direct port of mmpose/rtmlib's ``get_warp_matrix`` with
    ``rot`` fixed at 0 (pose crops are never rotated, unlike the detector's
    rotation-assist passes)."""
    import cv2

    dst_w, dst_h = output_wh
    src_dir = np.array([0.0, -float(scale[0]) * 0.5], dtype=np.float32)
    dst_dir = np.array([0.0, -dst_w * 0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    return cv2.getAffineTransform(src, dst)


def _top_down_affine(
    input_wh: tuple[int, int], scale: np.ndarray, center: np.ndarray,
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``img``'s (center, scale) box to ``input_wh``, reshaping ``scale``
    to the model's fixed aspect ratio first (so a non-matching bbox aspect
    ratio doesn't squash the person). Returns ``(warped, adjusted_scale)`` —
    the caller needs the adjusted scale to invert keypoints back to frame
    coordinates."""
    import cv2

    w, h = input_wh
    aspect_ratio = w / h
    b_w, b_h = float(scale[0]), float(scale[1])
    if b_w > b_h * aspect_ratio:
        scale = np.array([b_w, b_w / aspect_ratio], dtype=np.float32)
    else:
        scale = np.array([b_h * aspect_ratio, b_h], dtype=np.float32)
    warp_mat = _get_warp_matrix(center, scale, (w, h))
    warped = cv2.warpAffine(img, warp_mat, (int(w), int(h)),
                            flags=cv2.INTER_LINEAR)
    return warped, scale


def _get_simcc_maximum(simcc_x: np.ndarray,
                       simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(1, K, Wx)/(1, K, Wy) SimCC logits → per-keypoint ``(locs (K,2), vals
    (K,))`` via argmax. Pure numpy so it unit-tests against hand-built logits
    with no model. ``locs`` is -1 where the peak value is <= 0 (dead output)."""
    n, k, _ = simcc_x.shape
    sx = simcc_x.reshape(n * k, -1)
    sy = simcc_y.reshape(n * k, -1)
    x_locs = np.argmax(sx, axis=1)
    y_locs = np.argmax(sy, axis=1)
    locs = np.stack([x_locs, y_locs], axis=-1).astype(np.float32)
    vals = 0.5 * (np.amax(sx, axis=1) + np.amax(sy, axis=1))
    locs[vals <= 0.0] = -1
    return locs.reshape(n, k, 2)[0], vals.reshape(n, k)[0]


def _torso_axis(
    kpts: np.ndarray, scr: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """``(shoulder_mid, hip_mid, shoulder_width)`` for a confident torso, else
    ``None``. Direct port of the legacy ``pose_rtmw._torso_axis``: the body
    axis (hip_mid → shoulder_mid → head) is the only reliable cue for which
    end of a person is the head on a sideways/inverted/lying subject."""
    if (scr[_L_SHOULDER] <= _KPT_THR or scr[_R_SHOULDER] <= _KPT_THR
            or scr[_L_HIP] <= _KPT_THR or scr[_R_HIP] <= _KPT_THR):
        return None
    sh = (kpts[_L_SHOULDER] + kpts[_R_SHOULDER]) / 2.0
    hp = (kpts[_L_HIP] + kpts[_R_HIP]) / 2.0
    sw = float(np.hypot(kpts[_L_SHOULDER, 0] - kpts[_R_SHOULDER, 0],
                        kpts[_L_SHOULDER, 1] - kpts[_R_SHOULDER, 1]))
    if sw < 4.0:
        return None
    return sh.astype(np.float32), hp.astype(np.float32), sw


def _head_anchor(
    kpts: np.ndarray, scr: np.ndarray, fw: int, fh: int,
) -> Optional[np.ndarray]:
    """Coarse head-anchor box from nose/eyes/ears, oriented by the torso, or
    ``None`` when no confident anchor could be placed. Direct port of the
    legacy ``pose_rtmw._head_box_from_anchors`` (sans the dense-face-mesh
    corroboration branch, which body7 has no keypoints for): the body axis
    both *validates* an anchor-derived head (one landing on the hip side of
    the shoulders is a skeleton misfit onto legs/torso, not a face) and
    *places* a shoulders-only guess when no anchor keypoint is confident —
    and refuses to guess a head it cannot orient rather than fabricate one
    over empty space. Returns ``[x1, y1, x2, y2, score]`` gated at
    ``_HEAD_SCORE_MIN`` (pose abstains below that, it never vetoes)."""
    def pt(i: int) -> tuple[float, float, float]:
        return float(kpts[i, 0]), float(kpts[i, 1]), float(scr[i])

    torso = _torso_axis(kpts, scr)
    head = [(x, y, s) for x, y, s in (pt(i) for i in _HEAD_ANCHORS)
            if s > _KPT_THR]

    if head:
        cx = float(np.mean([p[0] for p in head]))
        cy = float(np.mean([p[1] for p in head]))
        score = float(np.mean([p[2] for p in head]))
        if torso is not None:
            sh, hp, _sw = torso
            up = sh - hp
            if float(np.dot(np.array([cx, cy], dtype=np.float32) - sh,
                            up)) <= 0:
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
        if torso is None:
            return None
        sh, hp, sw = torso
        axis = sh - hp
        torso_len = float(np.hypot(axis[0], axis[1]))
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
    if box[2] - box[0] < 4 or box[3] - box[1] < 4 or score < _HEAD_SCORE_MIN:
        return None
    return np.array([box[0], box[1], box[2], box[3], score], dtype=np.float32)


# ── model wrapper ─────────────────────────────────────────────────────────────

class PoseEstimator:
    """Lazy single-session RTMPose wrapper → up to ``max_people`` PersonPoses.

    Mirrors ScrfdDetector's lifecycle: built once on first use, never
    destroyed; any failure flips ``available`` to False and ``estimate``
    returns ``[]`` forever after (the gate degrades to head-anchor-only, the
    pipeline doesn't notice)."""

    def __init__(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        self._on_status = on_status
        self._sess = None
        self._in_name = ""
        self.available: Optional[bool] = None
        self.last_ms = 0.0

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
                    f"no pose ONNX found (set AVPP_POSE_ONNX or place it "
                    f"at {default_cache()})")
            import onnxruntime as ort

            from .detector import _pinned_model
            from .utils import best_onnx_providers, make_session

            self._status(f"Loading pose estimator ({path.name})…")
            w, h = _INPUT_WH
            pinned = _pinned_model(str(path), (h, w))
            providers = best_onnx_providers()
            try:
                sess = make_session(pinned, providers)
            except Exception:  # noqa: BLE001 — fp16/DML rejected → plain fp32
                sess = ort.InferenceSession(pinned, providers=providers)
            self._in_name = sess.get_inputs()[0].name
            self._sess = sess
            self.available = True
            self._status(f"Pose estimator ready on "
                         f"{sess.get_providers()[0]} @ {w}×{h}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._status(f"Pose estimator unavailable: {exc!r}")
            self.available = False
        return self.available

    def _infer_one(self, frame_bgr: np.ndarray,
                   bbox_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center, scale = _bbox_xyxy2cs(bbox_xyxy, _BBOX_PADDING)
        warped, scale = _top_down_affine(_INPUT_WH, scale, center, frame_bgr)
        blob = warped[:, :, ::-1].astype(np.float32)   # BGR → RGB
        blob = (blob - _MEAN) / _STD
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        simcc_x, simcc_y = self._sess.run(None, {self._in_name: blob})
        locs, vals = _get_simcc_maximum(np.asarray(simcc_x, np.float32),
                                        np.asarray(simcc_y, np.float32))
        w, h = _INPUT_WH
        kpts = (locs / _SIMCC_RATIO) / np.array([w, h], np.float32) \
            * scale + center - scale / 2
        return kpts.astype(np.float32), vals.astype(np.float32)

    def estimate(self, frame_bgr: np.ndarray, bodies: np.ndarray,
                max_people: int = MAX_PEOPLE) -> list[PersonPose]:
        """Top-down pose on the ``max_people`` highest-scoring ``bodies``
        boxes ((K,5) xyxy+score, frame coords) → one :class:`PersonPose` per
        person. Degrades to ``[]`` on any failure (never raises) or when no
        bodies are given."""
        bodies = np.asarray(bodies, np.float32).reshape(-1, 5)
        if not self._ensure() or len(bodies) == 0:
            return []
        try:
            fh, fw = frame_bgr.shape[:2]
            order = np.argsort(-bodies[:, 4])[:max_people]
            t0 = time.perf_counter()
            out: list[PersonPose] = []
            for b in bodies[order]:
                kpts, scores = self._infer_one(frame_bgr, b[:4])
                out.append(PersonPose(
                    kpts=np.concatenate([kpts, scores[:, None]], axis=1),
                    anchor=_head_anchor(kpts, scores, fw, fh),
                    torso=_torso_axis(kpts, scores)))
            self.last_ms = (time.perf_counter() - t0) * 1e3
            return out
        except Exception as exc:  # noqa: BLE001 — one bad frame ≠ crash
            self._status(f"Pose estimate failed, degrading: {exc!r}")
            self.available = False
            return []
