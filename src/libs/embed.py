"""Face appearance embeddings — the identity channel the tracker never had.

Everything upstream of this module reasons about *where* a face is; nothing
reasons about *who* it is. ``head_tracker.py`` associates on IoU, centre
distance and size ratio; ``tracklets.py`` re-joins a head that vanished and
reappeared using velocity extrapolation and a corridor test. Both are pure
geometry, and geometry cannot tell two people apart. That is why
``_bridge_candidates`` needs an *ambiguity gate* at all: when two tracklets
are equally plausible continuations, the only safe answer available was to
join neither, and a real person's blur got split into two tracks (or, worse,
two people's tracks got smeared into one path).

An appearance vector answers the question geometry can't. This wraps an
ArcFace-style embedder: a 112x112 aligned face crop in, a 512-d L2-normalised
vector out, where cosine similarity between two crops of the same person is
high (typically > 0.4) and between different people is near zero.

Cost is deliberately small. It runs only on boxes that already cleared the
evidence gate — usually one to three crops per analysed frame at 112x112,
around 1 GFLOP total against the ~93 GFLOPs the detector/SCRFD/pose chain
already spends on that frame. Offline it runs a handful of times per tracklet.

Default weights are ``w600k_mbf.onnx`` (MobileFaceNet, ~13 MB) from the
official insightface ``buffalo_s.zip`` — the same release family this project
already pulls SCRFD from, so it adds a model but not a new trust root. The
interface is just crop -> vector, so a stronger backbone (a ViT/TransFace
export, or ``w600k_r50`` from ``buffalo_l``) is a drop-in via
``AVPP_EMBED_ONNX``/``AVPP_EMBED_URL`` with no code change.

Model resolution order: ``AVPP_EMBED_ONNX`` -> PyInstaller bundle
(``sys._MEIPASS/models``) -> ``~/.cache/avpp/embed`` -> download.

Session lifecycle follows the same DirectML law as every other model here:
built once, never destroyed, input shape-pinned. Any failure flips
``available`` to False and every caller degrades to the original
geometry-only behaviour — an embedder that won't load must never be able to
stop the pipeline, or block a blur.
"""

import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_CACHE_DIR = Path.home() / ".cache" / "avpp" / "embed"
_MEMBER = "w600k_mbf.onnx"
_URL = ("https://github.com/deepinsight/insightface/releases/download/"
        "v0.7/buffalo_s.zip")
_INPUT_HW = (112, 112)   # pinned into the graph for DirectML
_DIM = 512

#: Below this cosine, two tracklets are treated as positively *different*
#: people and a geometric bridge between them is vetoed.
#:
#: This is used **only as a veto** — appearance can reject a join that
#: geometry proposed, never enable one geometry would not already allow. That
#: asymmetry is deliberate and it is what makes the feature safe to ship
#: before the threshold has been calibrated on real footage:
#:
#: * If the embedder is working, a low score means two different people and
#:   the veto prevents an identity smear the corridor/ambiguity gates miss.
#: * If the embedder is uninformative for this footage (an out-of-distribution
#:   backbone returns high similarity for everything), the veto simply never
#:   fires and behaviour is identical to the geometry-only pipeline.
#:
#: The failure mode of the opposite design — using a *high* score to justify
#: bridging further than geometry allows — is merging two people's blur paths,
#: which is exactly what the ambiguity gate exists to prevent. Do not switch
#: to that without calibrating on real footage first (see
#: ``scripts/calibrate_embed.py``).
#:
#: Deliberately low: it should fire only when appearance is *strongly*
#: dissimilar, not merely unconvincing.
DIFF_ID_COS = 0.18

#: Similarity above which two crops would be considered the same identity.
#: Not used for any decision yet — recorded so the calibration script has a
#: reference point and so a future, validated positive-evidence path (longer
#: bridges, cross-video track stitching) has somewhere to hang.
SAME_ID_COS = 0.42


def enabled() -> bool:
    """``AVPP_EMBED=0`` disables the appearance channel entirely."""
    return os.environ.get("AVPP_EMBED", "1").strip().lower() not in (
        "0", "false", "no", "off")


def model_path() -> Optional[Path]:
    """First existing model file per the resolution order in the module doc."""
    env = os.environ.get("AVPP_EMBED_ONNX")
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
    """Ensure the embedder ONNX exists locally, downloading it if missing.

    Only the recognition member is extracted from the archive (matched by name
    suffix, so the zip's internal layout doesn't matter). ``.part`` temp +
    atomic rename, same as every other downloader here."""
    existing = model_path()
    if existing is not None:
        return existing

    url = os.environ.get("AVPP_EMBED_URL", _URL)
    cache = default_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)

    def _report(block: int, block_size: int, total: int) -> None:
        if on_status and total > 0:
            done = min(block * block_size, total)
            on_status(f"Downloading face embedder… "
                      f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if on_status:
        on_status(f"Downloading face embedder from {url}")

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


# ── similarity helpers (pure numpy — usable with no model present) ───────────

def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between ``(N, D)`` and ``(M, D)`` rows.

    Inputs are assumed L2-normalised (:meth:`FaceEmbedder.embed` guarantees
    it), so this is a plain dot product; it re-normalises defensively anyway
    because callers also feed it running averages of several vectors."""
    a = np.asarray(a, np.float32).reshape(-1, _DIM)
    b = np.asarray(b, np.float32).reshape(-1, _DIM)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return (a @ b.T).astype(np.float32)


def merge_embedding(mean: Optional[np.ndarray], new: np.ndarray,
                    momentum: float = 0.9) -> np.ndarray:
    """Exponential-moving-average update of a track's identity vector.

    A single frame's crop can be motion-blurred, half-occluded or badly lit; a
    running mean over the track's history is a far more stable identity than
    the newest observation, and costs one multiply-add."""
    new = np.asarray(new, np.float32).reshape(_DIM)
    if mean is None:
        out = new.copy()
    else:
        out = momentum * np.asarray(mean, np.float32).reshape(_DIM) \
            + (1.0 - momentum) * new
    return (out / max(float(np.linalg.norm(out)), 1e-9)).astype(np.float32)


class FaceEmbedder:
    """ONNX ArcFace-style embedder. Degrades to "unavailable" on any failure.

    Session lifecycle follows the DirectML law (module docstring): built once
    on first use, never destroyed, never rebuilt.
    """

    dim = _DIM

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
                    f"no face-embedder ONNX found (set AVPP_EMBED_ONNX or "
                    f"place it at {default_cache()})")
            import onnxruntime as ort

            from .detector import _pinned_model
            from .utils import best_onnx_providers, make_session

            self._status(f"Loading face embedder ({path.name})…")
            h, w = _INPUT_HW
            pinned = _pinned_model(str(path), (h, w))
            providers = best_onnx_providers()
            try:
                sess = make_session(pinned, providers)
            except Exception:  # noqa: BLE001 — fp16/DML rejected → plain fp32
                sess = ort.InferenceSession(pinned, providers=providers)
            self._in_name = sess.get_inputs()[0].name
            self._sess = sess
            self.available = True
            self._status(f"Face embedder ready on "
                         f"{sess.get_providers()[0]} @ {w}×{h}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._status(f"Face embedder unavailable: {exc!r}")
            self.available = False
        return self.available

    def embed(self, frame_bgr: np.ndarray,
              boxes: "np.ndarray | list") -> np.ndarray:
        """``(N, 512)`` L2-normalised embeddings for xyxy ``boxes``.

        Returns an all-zero ``(N, 512)`` when the model is unavailable or a
        crop is degenerate — a zero vector has cosine 0 against everything, so
        callers that use similarity as *supporting* evidence naturally fall
        back to geometry alone rather than needing an availability check at
        every site.
        """
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4) \
            if len(boxes) else np.empty((0, 4), np.float32)
        out = np.zeros((len(boxes), _DIM), np.float32)
        if len(boxes) == 0 or not self._ensure():
            return out
        import cv2

        h, w = _INPUT_HW
        fh, fw = frame_bgr.shape[:2]
        try:
            t0 = time.perf_counter()
            for i, b in enumerate(boxes):
                # Square-expand around the face centre: ArcFace expects a
                # centred face at a consistent scale, and a non-square crop
                # squashed to 112x112 shifts the geometry the model keys on.
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                side = max(b[2] - b[0], b[3] - b[1]) * 1.15
                if side < 8.0:
                    continue
                x1 = int(round(cx - side / 2)); x2 = int(round(cx + side / 2))
                y1 = int(round(cy - side / 2)); y2 = int(round(cy + side / 2))
                # Clamp to the frame, then letterbox what survives so a face
                # at the edge keeps its aspect ratio instead of stretching.
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(fw, x2), min(fh, y2)
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue
                crop = frame_bgr[cy1:cy2, cx1:cx2]
                canvas = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
                canvas[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1] = crop
                blob = cv2.resize(canvas, (w, h),
                                  interpolation=cv2.INTER_LINEAR)
                blob = blob[:, :, ::-1].astype(np.float32)   # BGR → RGB
                blob = (blob - 127.5) / 127.5
                blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
                vec = np.asarray(
                    self._sess.run(None, {self._in_name: blob})[0],
                    np.float32).reshape(-1)[:_DIM]
                n = float(np.linalg.norm(vec))
                if n > 1e-9:
                    out[i] = vec / n
            self.last_ms = (time.perf_counter() - t0) * 1e3
            return out
        except Exception as exc:  # noqa: BLE001 — one bad frame ≠ crash
            self._status(f"Face embedding failed, degrading: {exc!r}")
            self.available = False
            return np.zeros((len(boxes), _DIM), np.float32)
