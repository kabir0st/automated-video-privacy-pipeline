import os
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

# ── windowed-safe debug log ───────────────────────────────────────────────────
# The frozen build is --windowed: sys.stdout/stderr are routed to devnull
# (rth_windowed_stdio.py), so print() debug vanishes. Mirror every status line
# into this file (beside the existing FaceBlurInspector-error.log) so model
# locations, downloads and provider/model-load info are inspectable in the .exe.
DEBUG_LOG = Path(tempfile.gettempdir()) / "FaceBlurInspector-debug.log"


def debug_log(msg: str) -> None:
    """Append a timestamped line to DEBUG_LOG and echo to stdout.

    Best-effort: never raises (a logging failure must not take down startup or a
    worker frame). print() is harmless in dev and lands in devnull in the
    windowed .exe, so the file is the source of truth there."""
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    try:
        print(line)
    except Exception:  # noqa: BLE001 — devnull/closed stream in some builds
        pass

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── anonymisation constants ──────────────────────────────────────────────────
BLUR_EXPAND = 0.45       # uniform outward expansion from hull centroid
BLUR_HAIR_EXTRA = 0.90   # additional upward expansion to cover hairline
BLUR_K_GAUSSIAN = 71     # Gaussian kernel (must be odd)
BLUR_PIXELATE_BLOCK = 10 # downsample factor for pixelation pass

# A blur stack is an ordered tuple of (kind, strength) layers applied in
# sequence; "gaussian" strength is the kernel size, "pixelate" strength is
# the mosaic block size. The default reproduces the original hardwired
# Gaussian → pixelate pipeline.
BlurLayer = tuple[str, int]
DEFAULT_BLUR_LAYERS: tuple[BlurLayer, ...] = (
    ("gaussian", BLUR_K_GAUSSIAN),
    ("pixelate", BLUR_PIXELATE_BLOCK),
)


def best_onnx_providers() -> list[str]:
    """Pick GPU execution providers when available, in preference order:
    DirectML (any Windows GPU incl. AMD Radeon) > CUDA (NVIDIA) > ROCm (AMD on
    Linux) > CPU. On an AMD RX 6800 the winning provider is DirectML on Windows
    (install ``onnxruntime-directml``) or ROCm on native Linux."""
    import onnxruntime as ort

    preferred = (
        "DmlExecutionProvider",      # Windows, any GPU incl. AMD Radeon
        "CUDAExecutionProvider",     # NVIDIA
        "ROCMExecutionProvider",     # AMD on native Linux
        "CPUExecutionProvider",
    )
    available = ort.get_available_providers()
    return [p for p in preferred if p in available] or list(available)


def has_gpu_provider() -> bool:
    """True when a non-CPU ONNX execution provider is available."""
    prov = best_onnx_providers()
    return bool(prov) and prov[0] != "CPUExecutionProvider"


# ── float16 acceleration for GPU inference ───────────────────────────────────
# RDNA2 (e.g. RX 6800) runs float16 convolutions/matmuls at roughly twice the
# float32 rate, and DirectML honours an fp16 graph. Converting each model's
# *internal* compute to fp16 while keeping its float32 inputs/outputs
# (``keep_io_types=True``) is a near-lossless ~2× inference speedup that needs
# no change to the numpy plumbing in insightface/rtmlib that feeds these
# sessions float32 arrays. Output deviation measured on this project's models is
# ~0.03 % (SCRFD det) / ~0.3 % (106-pt landmarks) — sub-pixel, imperceptible.

# Shared on-disk cache for derived model files (also used by face_app for the
# DirectML shape-pinned det variants).
DERIVED_MODEL_CACHE = Path(tempfile.gettempdir()) / "FaceBlurInspector-models"


def fp16_enabled() -> bool:
    """fp16 GPU inference is on by default; ``AVPP_FP16=0`` disables it.

    A single env kill-switch so a model that misbehaves under DirectML's strict
    fp16 validation can be turned off without rebuilding the .exe."""
    return os.environ.get("AVPP_FP16", "1").strip().lower() not in (
        "0", "false", "no", "off")


def fp16_model_path(src_path: str) -> str:
    """Return a float16-internal (float32 I/O) copy of an ONNX model, cached on
    disk. Returns ``src_path`` unchanged when fp16 is disabled, no GPU provider
    is active (fp16 is slower on CPU), or conversion fails for any reason —
    callers always get a usable path."""
    if not fp16_enabled() or not has_gpu_provider():
        return src_path
    try:
        src = Path(src_path)
        DERIVED_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        # Key on size+mtime so a re-downloaded/updated model invalidates its
        # cached fp16 derivative instead of silently reusing a stale one.
        st = src.stat()
        dst = DERIVED_MODEL_CACHE / f"{src.stem}_fp16_{st.st_size}_{int(st.st_mtime)}.onnx"
        if not dst.exists():
            import onnx
            from onnxconverter_common import float16

            model = onnx.load(str(src))
            with warnings.catch_warnings():
                # convert_float_to_float16 warns per tiny denormal it clamps to
                # the fp16 range; harmless and very noisy, so silence it.
                warnings.simplefilter("ignore")
                model16 = float16.convert_float_to_float16(
                    model, keep_io_types=True, disable_shape_infer=True)
            onnx.save(model16, str(dst))
        return str(dst)
    except Exception:  # noqa: BLE001 — any failure → fall back to fp32 model
        return src_path


def make_session(model_path: str, providers: list[str]):
    """Build an ONNX Runtime session with fp16 (GPU) and DirectML-friendly
    options. Use for sessions this project creates directly (RTMW, RF-DETR);
    insightface/rtmlib that build their own sessions get fp16 via the model
    file from :func:`fp16_model_path` instead."""
    import onnxruntime as ort

    path = fp16_model_path(model_path)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers and providers[0] == "DmlExecutionProvider":
        # DirectML does not support ORT's memory-pattern planner; leaving it on
        # forces a fallback path. Disabling is required for DML, harmless else.
        so.enable_mem_pattern = False
    return ort.InferenceSession(path, sess_options=so, providers=providers)


# ── crop / unproject helpers ─────────────────────────────────────────────────

def crop_face_patch(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    target_size: int = 640,
    pad: float = 0.25,
) -> tuple[np.ndarray, tuple[int, int, float]]:
    """Crop + pad a face bbox and resize to target_size on the long edge."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(frame.shape[1], int(x2 + bw * pad))
    y2 = min(frame.shape[0], int(y2 + bh * pad))
    crop = frame[y1:y2, x1:x2]
    scale = target_size / max(crop.shape[:2])
    resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    return resized, (x1, y1, scale)


def unproject_landmark(
    lx: float, ly: float,
    origin_x: int, origin_y: int,
    scale: float,
) -> tuple[float, float]:
    return origin_x + lx / scale, origin_y + ly / scale


# ── mask builders ─────────────────────────────────────────────────────────────

def _expanded_hull(
    landmarks: np.ndarray,
    frame_h: int,
    frame_w: int,
    expand: float = BLUR_EXPAND,
    hair_extra: float = BLUR_HAIR_EXTRA,
) -> np.ndarray | None:
    """Return an expanded convex hull polygon (N, 2) int32, or None on failure."""
    pts = landmarks.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    if hull is None or len(hull) < 3:
        return None
    M = cv2.moments(hull)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    expanded = []
    for pt in hull[:, 0]:
        dx, dy = pt[0] - cx, pt[1] - cy
        # Points above centroid (dy < 0) get extra upward push for hairline coverage
        sy = 1 + expand + (hair_extra if dy < 0 else 0)
        ex = int(np.clip(cx + dx * (1 + expand), 0, frame_w - 1))
        ey = int(np.clip(cy + dy * sy, 0, frame_h - 1))
        expanded.append([ex, ey])
    return np.array(expanded, dtype=np.int32)


def add_face_mask(
    mask: np.ndarray,
    landmarks: np.ndarray,
    expand: float = BLUR_EXPAND,
    hair_extra: float = BLUR_HAIR_EXTRA,
) -> np.ndarray | None:
    """Fill the expanded hull into mask (uint8) in-place. Returns the polygon."""
    fh, fw = mask.shape[:2]
    poly = _expanded_hull(landmarks, fh, fw, expand=expand, hair_extra=hair_extra)
    if poly is not None:
        cv2.fillPoly(mask, [poly], 255)
    return poly


def add_bbox_mask(mask: np.ndarray, bbox: np.ndarray) -> None:
    """Fill a rectangular region into mask — fallback when landmarks are absent."""
    fh, fw = mask.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(fw, int(bbox[2]))
    y2 = min(fh, int(bbox[3]))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255


def add_ellipse_mask(mask: np.ndarray, bbox: np.ndarray) -> None:
    """Fill an ellipse inscribed in bbox — head-shaped blur for a head region
    that has no detected face landmarks (the blur is clipped to the body next)."""
    fh, fw = mask.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax, ay = max(1.0, (x2 - x1) / 2), max(1.0, (y2 - y1) / 2)
    if 0 <= cx < fw and 0 <= cy < fh:
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)),
                    0, 0, 360, 255, -1)


def clip_to_body(scratch: np.ndarray, body: "np.ndarray | None", bbox: np.ndarray) -> None:
    """Zero out scratch (uint8 0/255) outside the person's silhouette, in-place.

    The body silhouette (uint8, 1 inside) is grown a little — proportional to the
    head height — to cover the hairline and absorb mask-edge error, then anything
    outside it is dropped. No-op when no silhouette is available (face-only /
    MediaPipe path), so the blur is never erased for want of a mask.
    """
    if body is None or not np.any(body):
        return
    k = max(3, int(round((float(bbox[3]) - float(bbox[1])) * 0.12)))
    k |= 1  # odd kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    grown = cv2.dilate(body.astype(np.uint8), kernel)
    scratch[grown == 0] = 0


def _map_landmarks(
    landmarks: np.ndarray, src_bbox: np.ndarray, dst_bbox: np.ndarray
) -> np.ndarray:
    """Translate + scale landmarks from src_bbox's frame to dst_bbox's.

    Scale is clamped so a degenerate Kalman box cannot explode or collapse
    the hull.
    """
    sc = np.array([(src_bbox[0] + src_bbox[2]) / 2, (src_bbox[1] + src_bbox[3]) / 2])
    dc = np.array([(dst_bbox[0] + dst_bbox[2]) / 2, (dst_bbox[1] + dst_bbox[3]) / 2])
    sw = max(float(src_bbox[2] - src_bbox[0]), 1.0)
    sh = max(float(src_bbox[3] - src_bbox[1]), 1.0)
    scale = np.clip(
        [float(dst_bbox[2] - dst_bbox[0]) / sw, float(dst_bbox[3] - dst_bbox[1]) / sh],
        0.7, 1.4,
    )
    return ((landmarks - sc) * scale + dc).astype(np.float32)


class MaskBuilder:
    """Builds the combined blur mask from tracker output, with continuity.

    Each person's blur is built in a scratch layer and then clipped to that
    person's body silhouette before compositing, so the blur never paints
    background (``clip_to_body``). When a refining face was found this frame the
    smoothed 106-pt hull is used (and remembered with the box it was seen at);
    when a tracked person has no face this frame the remembered hull is
    translated/scaled onto its current head box so the blur follows the head;
    a person who never had a face falls back to a head-region ellipse. The
    silhouette clip keeps every one of those on the body.
    """

    def __init__(self, smoother: "LandmarkSmoother | None" = None) -> None:
        from .smoother import LandmarkSmoother
        self._smoother = smoother or LandmarkSmoother()
        # track_id -> (smoothed landmarks, bbox at observation time)
        self._mem: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def add(
        self,
        mask: np.ndarray,
        tracked,  # TrackedFace from libs.tracker
        *,
        expand: float = BLUR_EXPAND,
        hair_extra: float = BLUR_HAIR_EXTRA,
    ) -> np.ndarray | None:
        """Stamp one person into mask, clipped to their body; return the hull."""
        tid, face, bbox = tracked.track_id, tracked.face, tracked.bbox
        body = getattr(tracked, "mask", None)
        scratch = np.zeros_like(mask)
        poly: np.ndarray | None = None

        if face is not None and getattr(face, "landmark_2d_106", None) is not None:
            smoothed = self._smoother.update(
                tid, face.landmark_2d_106, float(face.det_score))
            self._mem[tid] = (smoothed, np.array(face.bbox[:4], dtype=np.float32))
            poly = add_face_mask(scratch, smoothed, expand=expand, hair_extra=hair_extra)
        elif tid in self._mem:
            lm0, bb0 = self._mem[tid]
            moved = _map_landmarks(lm0, bb0, bbox)
            poly = add_face_mask(scratch, moved, expand=expand, hair_extra=hair_extra)
        else:
            add_ellipse_mask(scratch, bbox)

        clip_to_body(scratch, body, bbox)
        mask[scratch == 255] = 255
        return poly

    def evict(self, active_ids: set[int]) -> None:
        """Drop state for tracks the tracker no longer reports."""
        for tid in list(self._mem):
            if tid not in active_ids:
                del self._mem[tid]
                self._smoother.drop(tid)

    def reset(self) -> None:
        self._mem.clear()
        self._smoother.drop_all()


# ── GPU-accelerated stacked blur pipeline ────────────────────────────────────

class BlurPipeline:
    """Applies a configurable stack of blur layers to a masked region.

    Layers (Gaussian / pixelate, in any order and multiplicity) are applied
    in sequence to the whole frame once, then composited where the mask is set,
    in a single GPU round-trip per frame regardless of how many faces present.

    Backend is picked once at construction:
      * "cuda"   — PyTorch CUDA (NVIDIA).
      * "opencl" — OpenCV Transparent-API / UMat on any OpenCL device. This is
                   the path that lights up an AMD Radeon (e.g. RX 6800) where
                   torch-CUDA never applies, so the blur runs on the GPU too.
      * "cpu"    — OpenCV on CPU (fallback).
    """

    def __init__(self) -> None:
        self.backend = self._select_backend()
        self._layers: tuple[BlurLayer, ...] = DEFAULT_BLUR_LAYERS
        if self.backend == "cuda":
            self._device = torch.device("cuda")  # type: ignore[possibly-undefined]
            self._kernels: dict[int, "torch.Tensor"] = {}
        label = {"cuda": "GPU (CUDA/PyTorch)",
                 "opencl": "GPU (OpenCL/UMat)",
                 "cpu": "CPU"}[self.backend]
        print(f"[BlurPipeline] {label} blur active")

    @staticmethod
    def _select_backend() -> str:
        if _TORCH_AVAILABLE and torch.cuda.is_available():  # type: ignore[possibly-undefined]
            return "cuda"
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    return "opencl"
        except Exception:  # noqa: BLE001 — any OpenCL probe failure → CPU
            pass
        return "cpu"

    def reconfigure(self, layers: tuple[BlurLayer, ...]) -> None:
        """Update the blur layer stack without re-constructing the pipeline."""
        self._layers = tuple(
            (kind, (max(3, s) | 1) if kind == "gaussian" else max(2, s))
            for kind, s in layers
        )

    def _gaussian_kernel(self, k: int) -> "torch.Tensor":
        kernel = self._kernels.get(k)
        if kernel is None:
            kernel = self._make_gaussian_kernel(k, k / 6.0, self._device)
            self._kernels[k] = kernel
        return kernel

    # ── public API ────────────────────────────────────────────────────────────

    # Below this masked-area fraction of the frame, blur only the mask's
    # bounding-box ROI instead of the whole frame — at 4K a few small faces
    # then cost a tiny upload/blur/download instead of a full-frame one. Above
    # it the ROI covers most of the frame and the crop overhead isn't worth it.
    _ROI_MAX_FRAC = 0.6

    def apply(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Apply the blur stack to frame in-place, only where mask == 255."""
        if not self._layers:
            return
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return

        fh, fw = frame.shape[:2]
        # Pad the ROI so the blur has valid context around the masked region
        # (a Gaussian reads its kernel's worth of neighbours; pixelate a block).
        gmax = max((s for k, s in self._layers if k == "gaussian"), default=0)
        pmax = max((s for k, s in self._layers if k == "pixelate"), default=0)
        pad = gmax + pmax + 4
        x1 = max(0, int(xs.min()) - pad); x2 = min(fw, int(xs.max()) + 1 + pad)
        y1 = max(0, int(ys.min()) - pad); y2 = min(fh, int(ys.max()) + 1 + pad)

        if (x2 - x1) * (y2 - y1) >= self._ROI_MAX_FRAC * fw * fh:
            f_roi, m_roi = frame, mask          # ROI ≈ whole frame; skip the crop
        else:
            f_roi, m_roi = frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]

        # The backends write the result back through f_roi, which is a view onto
        # frame, so the in-place contract holds for both the ROI and full paths.
        if self.backend == "cuda":
            self._apply_gpu(f_roi, m_roi)
        elif self.backend == "opencl":
            self._apply_opencl(f_roi, m_roi)
        else:
            self._apply_cpu(f_roi, m_roi)

    # ── OpenCL path (UMat / Transparent API — AMD, Intel, any OpenCL GPU) ──────

    def _apply_opencl(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Run the layer stack on the GPU via OpenCV UMat, composite by mask.

        UMat operations are dispatched to the OpenCL device transparently; the
        single upload/download pair keeps the host round-trip to one per frame.
        """
        h, w = frame.shape[:2]
        src = cv2.UMat(np.ascontiguousarray(frame))
        out = src
        for kind, strength in self._layers:
            if kind == "gaussian":
                out = cv2.GaussianBlur(out, (strength, strength), 0)
            else:
                b = strength
                small = cv2.resize(out, (max(1, w // b), max(1, h // b)),
                                   interpolation=cv2.INTER_AREA)
                out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        # Composite blurred pixels into the original where the mask is set.
        src = cv2.copyTo(out, cv2.UMat(np.ascontiguousarray(mask)), src)
        np.copyto(frame, src.get())

    # ── CPU path ──────────────────────────────────────────────────────────────

    def _apply_cpu(self, frame: np.ndarray, mask: np.ndarray) -> None:
        h, w = frame.shape[:2]
        out = frame
        for kind, strength in self._layers:
            if kind == "gaussian":
                out = cv2.GaussianBlur(out, (strength, strength), 0)
            else:
                # Mosaic — INTER_AREA averages each block to a flat colour,
                # giving true solid-square mosaic rather than a blended
                # downsample.
                b = strength
                small = cv2.resize(out, (max(1, w // b), max(1, h // b)),
                                   interpolation=cv2.INTER_AREA)
                out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        frame[mask == 255] = out[mask == 255]

    # ── GPU path ──────────────────────────────────────────────────────────────

    def _apply_gpu(self, frame: np.ndarray, mask: np.ndarray) -> None:
        device = self._device
        h, w = frame.shape[:2]

        # Upload once: (1, 3, H, W) float32
        t = (
            torch.from_numpy(frame)  # type: ignore[possibly-undefined]
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )

        out = t
        for kind, strength in self._layers:
            if kind == "gaussian":
                pad = strength // 2
                out = F.conv2d(  # type: ignore[possibly-undefined]
                    F.pad(out, [pad] * 4, mode="reflect"),  # type: ignore[possibly-undefined]
                    self._gaussian_kernel(strength),
                    groups=3,
                )
            else:
                # Mosaic — avg_pool2d averages each block to a flat colour
                # (GPU equivalent of INTER_AREA).
                b = strength
                out = F.avg_pool2d(out, kernel_size=b, stride=b, padding=0)  # type: ignore[possibly-undefined]
                out = F.interpolate(out, size=(h, w), mode="nearest")  # type: ignore[possibly-undefined]

        # Composite: use blurred result where mask is set, original elsewhere
        mask_t = (
            torch.from_numpy(mask)  # type: ignore[possibly-undefined]
            .to(device)
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, H, W)
        result = torch.where(mask_t, out, t)  # type: ignore[possibly-undefined]

        # Write back to the original numpy buffer
        result_np = result.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        np.copyto(frame, result_np)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_gaussian_kernel(
        k: int, sigma: float, device: "torch.device"
    ) -> "torch.Tensor":
        ax = torch.arange(k, device=device, dtype=torch.float32) - k // 2  # type: ignore[possibly-undefined]
        gauss = torch.exp(-ax**2 / (2 * sigma**2))  # type: ignore[possibly-undefined]
        gauss = gauss / gauss.sum()
        kernel_2d = gauss.outer(gauss).view(1, 1, k, k).repeat(3, 1, 1, 1)
        return kernel_2d
