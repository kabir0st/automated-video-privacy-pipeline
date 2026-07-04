"""NudeNet (YOLOv8n, 18-class) — independent offline face-verify witness.

The evidence gate (libs/evidence.py) already anchors and corroborates face
claims per-frame, but every witness in that gate — the primary detector's own
face/part classes, SCRFD — is a face detector trained on ordinary photos
(WIDER FACE and similar). NudeNet is trained specifically on adult content
with explicit ``FACE_FEMALE``/``FACE_MALE`` classes alongside its nudity
classes, so it reads a face on *this* footage's skin-heavy, close-up, unusual
poses differently than a WIDER-FACE-trained model does — a genuinely
independent second opinion, not a second run of the same training
distribution. This is why libs/tracklets.py's cross-model verify (Phase 3)
requires it (or SCRFD) as the agreeing witness and never lets the primary
detector alone confirm its own claim.

Offline-only: this runs during the export's VERIFY step on magnified crops of
already-pruned tracklets, never in the live per-frame path (see
libs/tracklets.verify_tracklets_xmodel) — the cost of a second model pass is
paid once per candidate track, not once per frame.

Model: the upstream project publishes ``320n.onnx`` (YOLOv8n, 320×320, 18
classes) as a GitHub release asset, but that specific repository's release
downloads currently sit behind a GitHub login wall (confirmed: an anonymous
fetch 302s to /login) and can't be fetched by an unattended downloader. The
identical file ships inside the official ``nudenet`` PyPI wheel instead
(``nudenet/320n.onnx``, no login required, a stable content-addressed
files.pythonhosted.org URL), which is what :func:`download_model` fetches.

Same DirectML laws as libs/scrfd.py: one session, created lazily, never
destroyed, input shape-pinned via ``detector._pinned_model``. Any failure
flips ``available`` to False and the export's verify step degrades to
SCRFD-only (see tracklets.py) rather than crashing.

Model file resolution order (first hit wins):
  1. ``AVPP_NUDENET_ONNX`` — explicit path to a local ``.onnx``;
  2. the frozen bundle (``sys._MEIPASS/models/nudenet_320n.onnx``, PyInstaller);
  3. ``~/.cache/avpp/nudenet/nudenet_320n.onnx``.
``AVPP_NUDENET_URL`` may point at either a direct ``.onnx`` or a ``.zip``/
``.whl`` (both are zip archives) containing one.

Licensing: NudeNet is AGPL-3.0. Bundled here for personal/offline use in this
project's own frozen build; see README for the licence note if this module
is ever redistributed.
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

_CACHE_DIR = Path.home() / ".cache" / "avpp" / "nudenet"
_LOCAL_NAME = "nudenet_320n.onnx"
_MEMBER = "320n.onnx"
# Stable, content-addressed PyPI file URL (immutable once published) for the
# official `nudenet` 3.4.2 wheel, which bundles this exact ONNX file.
_URL = ("https://files.pythonhosted.org/packages/1c/ee/"
        "1aa02d44ba958cc77e16ff1e41a0aac5e721037db7bf62b9c9d124917f87/"
        "nudenet-3.4.2-py3-none-any.whl")
_INPUT_HW = (320, 320)   # pinned into the graph for DirectML

# 18-class label layout (fixed order in the model's export — see upstream
# nudenet/nudenet.py). Only the two face classes are ever consumed by this
# project; the rest of the columns are decoded (for testability/debugging)
# but never surfaced through detect_faces().
LABELS = (
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)
FACE_FEMALE = LABELS.index("FACE_FEMALE")
FACE_MALE = LABELS.index("FACE_MALE")
_FACE_IDS = (FACE_FEMALE, FACE_MALE)

# Matches upstream's combined pre-filter (0.2) + NMSBoxes score threshold
# (0.25) — a witness only has to clear the bar the model's own author judged
# reliable, not any project-specific bar (this is a corroborating second
# opinion, never a blur target by itself).
_FLOOR = 0.25
_NMS_IOU = 0.45


def model_path() -> Optional[Path]:
    """First existing model file per the resolution order in the module doc."""
    env = os.environ.get("AVPP_NUDENET_ONNX")
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
    """Ensure the NudeNet ONNX exists locally, downloading it if missing.

    The default source is the official ``nudenet`` PyPI wheel (a zip
    archive); only the ``320n.onnx`` member is extracted (matched by suffix,
    like scrfd.py's downloader). ``.part`` temp + atomic rename."""
    existing = model_path()
    if existing is not None:
        return existing

    url = os.environ.get("AVPP_NUDENET_URL", _URL)
    cache = default_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)

    def _report(block: int, block_size: int, total: int) -> None:
        if on_status and total > 0:
            done = min(block * block_size, total)
            on_status(f"Downloading NudeNet verify witness… "
                      f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if on_status:
        on_status(f"Downloading NudeNet verify witness from {url}")

    if url.endswith(".zip") or url.endswith(".whl"):
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


# ── pure-numpy decode (unit-testable without a model) ────────────────────────

def _decode_yolov8(output: np.ndarray, floor: float) -> np.ndarray:
    """Raw ``(1, 22, N)`` or ``(22, N)`` YOLOv8 export output → ``(K, 6)``
    ``[x1, y1, x2, y2, score, cls]`` rows in the *pinned model input's* pixel
    space (caller rescales to frame coordinates). The export bakes box
    decoding in already (Ultralytics' onnx export includes the DFL head), so
    columns ``[0:4]`` are plain ``cx, cy, w, h`` pixel values and ``[4:]`` are
    18 per-class scores with no separate objectness column — argmax picks
    the winning class per anchor point. Pure numpy, no NMS (see _nms in
    libs/detector.py, applied by the caller)."""
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.T   # (N, 22)
    if arr.shape[0] == 0:
        return np.empty((0, 6), np.float32)
    boxes_cxcywh = arr[:, :4]
    class_scores = arr[:, 4:]
    cls = np.argmax(class_scores, axis=1).astype(np.float32)
    score = np.amax(class_scores, axis=1)
    keep = score >= floor
    if not keep.any():
        return np.empty((0, 6), np.float32)
    cx, cy, w, h = (boxes_cxcywh[keep, i] for i in range(4))
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return np.stack([x1, y1, x2, y2, score[keep], cls[keep]],
                    axis=1).astype(np.float32)


class NudeNetDetector:
    """Lazy single-session NudeNet wrapper → (K, 6) xyxy+score+cls boxes,
    frame coordinates. Mirrors ScrfdDetector's lifecycle: built once on first
    use, never destroyed; any failure flips ``available`` to False and
    ``detect_faces`` returns empty forever after (verify degrades to
    SCRFD-only, the pipeline doesn't notice)."""

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
                    f"no NudeNet ONNX found (set AVPP_NUDENET_ONNX or place "
                    f"it at {default_cache()})")
            import onnxruntime as ort

            from .detector import _pinned_model
            from .utils import best_onnx_providers, make_session

            self._status(f"Loading NudeNet verify witness ({path.name})…")
            pinned = _pinned_model(str(path), _INPUT_HW)
            providers = best_onnx_providers()
            try:
                sess = make_session(pinned, providers)
            except Exception:  # noqa: BLE001 — fp16/DML rejected → plain fp32
                sess = ort.InferenceSession(pinned, providers=providers)
            self._in_name = sess.get_inputs()[0].name
            self._sess = sess
            self.available = True
            h, w = _INPUT_HW
            self._status(f"NudeNet verify witness ready on "
                         f"{sess.get_providers()[0]} @ {w}×{h}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._status(f"NudeNet verify witness unavailable: {exc!r}")
            self.available = False
        return self.available

    def detect_full(self, frame_bgr: np.ndarray,
                    floor: float = _FLOOR) -> np.ndarray:
        """All 18 classes → (K, 6) xyxy+score+cls, frame coords. Aspect-
        preserving letterbox into the pinned square input (top-left
        anchored), same pattern as scrfd.py's detect_full."""
        if not self._ensure():
            return np.empty((0, 6), np.float32)
        import cv2

        from .detector import _nms

        fh, fw = frame_bgr.shape[:2]
        h, w = _INPUT_HW
        scale = min(w / fw, h / fh)
        rw, rh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
        try:
            t0 = time.perf_counter()
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:rh, :rw] = cv2.resize(frame_bgr, (rw, rh),
                                         interpolation=cv2.INTER_LINEAR)
            blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
            blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
            out = self._sess.run(None, {self._in_name: blob})[0]
            rows = _nms(_decode_yolov8(out, floor), _NMS_IOU)
            self.last_ms = (time.perf_counter() - t0) * 1e3
            if len(rows) == 0:
                return np.empty((0, 6), np.float32)
            rows[:, :4] /= scale
            rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, fw - 1)
            rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, fh - 1)
            ok = ((rows[:, 2] - rows[:, 0] >= 4)
                  & (rows[:, 3] - rows[:, 1] >= 4))
            return rows[ok].copy()
        except Exception as exc:  # noqa: BLE001 — one bad frame ≠ crash
            self._status(f"NudeNet detect failed, degrading: {exc!r}")
            self.available = False
            return np.empty((0, 6), np.float32)

    def detect_faces(self, frame_bgr: np.ndarray,
                     floor: float = _FLOOR) -> np.ndarray:
        """``FACE_FEMALE``/``FACE_MALE`` only → (K, 5) xyxy+score, frame
        coords — the independent witness libs/tracklets.verify_tracklets_
        xmodel matches against a tracklet's face box."""
        rows = self.detect_full(frame_bgr, floor)
        if len(rows) == 0:
            return np.empty((0, 5), np.float32)
        keep = np.isin(rows[:, 5].astype(int), _FACE_IDS)
        return rows[keep, :5].copy()
