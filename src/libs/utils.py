import cv2
import numpy as np

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

# ── face-detection filter ────────────────────────────────────────────────────
_MIN_FACE_ASPECT = 0.4
_MIN_DET_SCORE = 0.55


def best_onnx_providers() -> list[str]:
    """Pick GPU execution providers when available, in preference order:
    DirectML (any Windows GPU incl. AMD) > CUDA (NVIDIA) > CPU."""
    import onnxruntime as ort

    preferred = ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    available = ort.get_available_providers()
    return [p for p in preferred if p in available] or list(available)


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


# ── detection filter ─────────────────────────────────────────────────────────

def is_likely_face(bbox: np.ndarray, det_score: float) -> bool:
    x1, y1, x2, y2 = bbox[:4]
    bw, bh = x2 - x1, y2 - y1
    if bh <= 0:
        return False
    return (bw / bh) >= _MIN_FACE_ASPECT and det_score >= _MIN_DET_SCORE


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


# ── GPU-accelerated stacked blur pipeline ────────────────────────────────────

class BlurPipeline:
    """Applies stacked Gaussian + pixelation blur to a masked region.

    Uses CUDA via PyTorch when available (one GPU round-trip per frame
    regardless of how many faces are present). Falls back to CPU cv2.
    """

    def __init__(self) -> None:
        self.use_gpu = _TORCH_AVAILABLE and torch.cuda.is_available()  # type: ignore[possibly-undefined]
        self._k = BLUR_K_GAUSSIAN
        self._block = BLUR_PIXELATE_BLOCK
        if self.use_gpu:
            self._device = torch.device("cuda")  # type: ignore[possibly-undefined]
            self._kernel = self._make_gaussian_kernel(
                self._k, self._k / 6.0, self._device)
        print(
            f"[BlurPipeline] {'GPU (CUDA)' if self.use_gpu else 'CPU'} blur active"
        )

    def reconfigure(self, k_gaussian: int, pixelate_block: int) -> None:
        """Update blur parameters without re-constructing the pipeline."""
        k_gaussian = k_gaussian | 1  # enforce odd
        if k_gaussian == self._k and pixelate_block == self._block:
            return
        self._k = k_gaussian
        self._block = pixelate_block
        if self.use_gpu:
            self._kernel = self._make_gaussian_kernel(
                k_gaussian, k_gaussian / 6.0, self._device)

    # ── public API ────────────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Apply stacked blur to frame in-place, only where mask == 255."""
        if not np.any(mask):
            return
        if self.use_gpu:
            self._apply_gpu(frame, mask)
        else:
            self._apply_cpu(frame, mask)

    # ── CPU path ──────────────────────────────────────────────────────────────

    def _apply_cpu(self, frame: np.ndarray, mask: np.ndarray) -> None:
        h, w = frame.shape[:2]
        k = self._k
        b = self._block

        # Pass 1: Gaussian blur
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        # Pass 2: mosaic — INTER_AREA averages each block to a flat colour,
        # giving true solid-square mosaic rather than a blended downsample.
        small = cv2.resize(blurred, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        frame[mask == 255] = mosaic[mask == 255]

    # ── GPU path ──────────────────────────────────────────────────────────────

    def _apply_gpu(self, frame: np.ndarray, mask: np.ndarray) -> None:
        device = self._device

        # Upload once: (1, 3, H, W) float32
        t = (
            torch.from_numpy(frame)  # type: ignore[possibly-undefined]
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )

        # Pass 1: Gaussian blur using precomputed kernel (depthwise conv)
        h, w = frame.shape[:2]
        pad = self._k // 2
        blurred = F.conv2d(  # type: ignore[possibly-undefined]
            F.pad(t, [pad] * 4, mode="reflect"),  # type: ignore[possibly-undefined]
            self._kernel,
            groups=3,
        )

        # Pass 2: mosaic — avg_pool2d averages each block to a flat colour
        # (GPU equivalent of INTER_AREA), giving true solid-square mosaic.
        b = self._block
        mosaic = F.avg_pool2d(blurred, kernel_size=b, stride=b, padding=0)  # type: ignore[possibly-undefined]
        mosaic = F.interpolate(mosaic, size=(h, w), mode="nearest")  # type: ignore[possibly-undefined]

        # Composite: use mosaic where mask is set, original elsewhere
        mask_t = (
            torch.from_numpy(mask)  # type: ignore[possibly-undefined]
            .to(device)
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, H, W)
        result = torch.where(mask_t, mosaic, t)  # type: ignore[possibly-undefined]

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
