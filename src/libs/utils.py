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


# ── GPU runtime discovery ────────────────────────────────────────────────────
# ONNX Runtime advertises an execution provider whenever its plugin .so is
# present in the wheel — it does NOT check that the plugin's own dependencies
# resolve. On a ROCm box that gap is easy to hit: distro packages install the
# HIP/HSA runtime under /opt/rocm (a symlink to the newest tree) while
# repo.radeon.com's math libraries (rocBLAS, MIOpen, MIGraphX) land in a
# *different*, version-suffixed tree that nothing adds to the linker path. The
# provider then lists fine and fails at session creation with
#     Failed to load library ... libmigraphx_c.so.3: cannot open shared object
# and ORT silently falls back to CPU.
#
# That silent fallback is worse than it looks here, because fp16_model_path()
# has already decided to convert the graph based on a GPU being "available" —
# so the fallback runs an *fp16 graph on the CPU*, which measured 71 ms/pass
# against 26 ms for plain fp32. A broken GPU provider made things 2.7x slower
# than having no GPU at all.
#
# So: preload the math libraries into the global symbol namespace, then verify
# each GPU provider's plugin actually loads, and drop the ones that don't.
_ROCM_LIB_GLOBS = ("/opt/rocm/lib", "/opt/rocm-*/lib", "/opt/rocm*/lib64")

#: EP → its plugin filename. Providers absent from this map (DirectML, CPU)
#: live inside the main runtime library and need no separate check.
_PROVIDER_LIBS = {
    "MIGraphXExecutionProvider": "libonnxruntime_providers_migraphx.so",
    "ROCMExecutionProvider": "libonnxruntime_providers_rocm.so",
    "CUDAExecutionProvider": "libonnxruntime_providers_cuda.so",
}

_preloaded = False
_provider_cache: "dict[str, bool]" = {}


def _preload_gpu_runtime() -> None:
    """dlopen ROCm's math libraries with RTLD_GLOBAL, once per process.

    Loading them here satisfies the provider plugin's dependencies without the
    user having to set LD_LIBRARY_PATH (which the dynamic loader only reads at
    exec, so a process can't set it for itself). Repeated passes handle
    inter-library ordering without hardcoding a dependency graph; anything that
    never loads is simply not needed. Entirely best-effort and silent — on a
    machine with no ROCm this is a no-op."""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    import ctypes
    import glob as _glob

    pending = []
    for pattern in _ROCM_LIB_GLOBS:
        for d in _glob.glob(pattern):
            pending.extend(_glob.glob(os.path.join(d, "lib*.so*")))
    pending = sorted(set(pending))
    if not pending:
        return
    for _ in range(4):
        progressed = False
        for path in list(pending):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            pending.remove(path)
            progressed = True
        if not progressed:
            break


def _provider_usable(name: str) -> bool:
    """True when this EP's plugin library actually loads.

    ORT must be imported first: the plugin resolves ``Provider_GetHost`` out of
    the main runtime library, so probing it standalone reports a spurious
    undefined-symbol failure."""
    lib = _PROVIDER_LIBS.get(name)
    if lib is None:
        return True                      # built into the main runtime
    if name in _provider_cache:
        return _provider_cache[name]
    ok = True
    try:
        import ctypes
        import onnxruntime as ort

        cand = Path(ort.__file__).parent / "capi" / lib
        if cand.is_file():               # absent ⇒ monolithic build, assume ok
            _preload_gpu_runtime()
            try:
                ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                debug_log(f"{name} advertised but its plugin will not load "
                          f"({exc}); dropping it — a silent CPU fallback would "
                          f"run an fp16 graph on the CPU and be slower than "
                          f"never having claimed a GPU")
                ok = False
    except Exception:  # noqa: BLE001 — probing must never break startup
        ok = True
    _provider_cache[name] = ok
    return ok


def _configure_migraphx_cache() -> None:
    """Point MIGraphX at a persistent compiled-model cache.

    MIGraphX compiles the whole graph ahead of time, which is where its speed
    comes from and also why a cold session costs ~25 s per model — five models
    would be minutes of startup, every launch. With the cache that drops to
    ~0.8 s. Only set when the user hasn't chosen a location themselves.

    Lives under ~/.cache/avpp beside the models rather than in DERIVED_MODEL_
    CACHE (which is under the system temp dir): the fp16 and shape-pinned
    derivatives kept there regenerate in seconds, whereas these cost ~25 s per
    model, so losing them to a reboot is a materially worse trade."""
    if os.environ.get("ORT_MIGRAPHX_MODEL_CACHE_PATH"):
        return
    try:
        cache = Path.home() / ".cache" / "avpp" / "migraphx"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = str(cache)
    except OSError:
        pass


def best_onnx_providers() -> list[str]:
    """Pick GPU execution providers when available, in preference order:
    DirectML (any Windows GPU incl. AMD Radeon) > CUDA (NVIDIA) > MIGraphX /
    ROCm (AMD on Linux) > CPU.

    Only providers whose plugin library actually loads are returned — see
    :func:`_provider_usable` for why advertising is not enough.

    On an AMD RX 6800 the winning provider is DirectML on Windows (install
    ``onnxruntime-directml``) or, on native Linux, **MIGraphX** — AMD ships
    ``onnxruntime_migraphx`` rather than ``onnxruntime_rocm`` from ROCm 7.1
    onward, so both names are listed and whichever the installed wheel provides
    is used. MIGraphX is preferred over the plain ROCm EP because it compiles
    and fuses the graph ahead of time (closer to TensorRT than to a per-op
    dispatcher), which matters a lot for the batch-1 shapes this pipeline runs.
    """
    import onnxruntime as ort

    preferred = (
        "DmlExecutionProvider",        # Windows, any GPU incl. AMD Radeon
        "CUDAExecutionProvider",       # NVIDIA
        "MIGraphXExecutionProvider",   # AMD on native Linux (ROCm >= 7.1)
        "ROCMExecutionProvider",       # AMD on native Linux (ROCm <= 7.0)
        "CPUExecutionProvider",
    )
    available = ort.get_available_providers()
    usable = [p for p in preferred if p in available and _provider_usable(p)]
    if usable and usable[0] == "MIGraphXExecutionProvider":
        _configure_migraphx_cache()
    return usable or ["CPUExecutionProvider"]


#: Providers that actually run compute on a GPU. Checked by name rather than
#: "anything that isn't CPU" because ORT also advertises non-compute providers
#: (AzureExecutionProvider ships in the stock CPU wheel), and treating one of
#: those as a GPU silently enables the fp16 path on a CPU-only box.
GPU_PROVIDERS = frozenset({
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
})


def has_gpu_provider() -> bool:
    """True when the winning ONNX execution provider runs compute on a GPU."""
    prov = best_onnx_providers()
    return bool(prov) and prov[0] in GPU_PROVIDERS


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


def _cpu_threads() -> int:
    """Physical-core count for ORT's intra-op pool (SMT siblings excluded).

    ``os.cpu_count()`` reports logical CPUs; ORT's default pool of that size
    oversubscribes SMT pairs on convolution graphs. Falls back to half the
    logical count, which is right for every SMT-2 machine this runs on."""
    override = os.environ.get("AVPP_CPU_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    try:  # Linux: count distinct physical core ids
        import re
        txt = Path("/proc/cpuinfo").read_text()
        ids = set(re.findall(r"^core id\s*:\s*(\d+)", txt, re.M))
        pkgs = set(re.findall(r"^physical id\s*:\s*(\d+)", txt, re.M))
        if ids:
            return max(1, len(ids) * max(1, len(pkgs)))
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def make_session(model_path: str, providers: list[str]):
    """Build an ONNX Runtime session with fp16 (GPU) and provider-specific
    options. Use for every session this project creates directly (detector,
    SCRFD, pose, NudeNet, face embedder)."""
    import onnxruntime as ort

    path = fp16_model_path(model_path)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers and providers[0] == "DmlExecutionProvider":
        # DirectML does not support ORT's memory-pattern planner; leaving it on
        # forces a fallback path. Disabling is required for DML, harmless else.
        so.enable_mem_pattern = False
    if providers and providers[0] == "CPUExecutionProvider":
        # ORT's default is one intra-op thread per *logical* core, which
        # oversubscribes SMT pairs and costs ~10-20 % on convolution-heavy
        # graphs like these. Pin to physical cores instead. AVPP_CPU_THREADS
        # overrides for benchmarking.
        so.intra_op_num_threads = _cpu_threads()
        so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so, providers=providers)


# ── mask rendering ──────────────────────────────────────────────────────────

# Face-only blur: how far the detected face box grows before it becomes the
# blur target. The top gets most of the bloom so the mask catches the
# forehead/fringe ("a bit of hair"), the sides a little, the chin barely —
# this is deliberately much tighter than a head box (~1.7–1.9× the face),
# which is the whole point of the face-only mode.
_FACE_BLOOM_TOP = 0.35
_FACE_BLOOM_SIDE = 0.10
_FACE_BLOOM_BOTTOM = 0.05


def bloom_face_box(box: np.ndarray) -> np.ndarray:
    """Grow a face box ``[x1, y1, x2, y2, ...]`` into its blur target: the
    face plus a fringe of hair (see the _FACE_BLOOM_* constants). Score and
    any trailing columns pass through untouched."""
    out = np.asarray(box, dtype=np.float32).copy()
    w = out[2] - out[0]
    h = out[3] - out[1]
    out[0] -= _FACE_BLOOM_SIDE * w
    out[2] += _FACE_BLOOM_SIDE * w
    out[1] -= _FACE_BLOOM_TOP * h
    out[3] += _FACE_BLOOM_BOTTOM * h
    return out


def render_head_mask(
    shape_hw: tuple[int, int],
    boxes: "list[np.ndarray] | np.ndarray",
    *,
    pad: float = 0.18,
    feather: float = 0.12,
) -> np.ndarray:
    """Soft blur mask (uint8 0..255) — one feathered ellipse per head box.

    Every head is rendered the same way every frame: an axis-aligned ellipse
    inscribed in the box expanded by ``pad`` per side, edge-feathered by a
    single Gaussian over the whole mask (kernel ∝ mean head diagonal). One
    consistent shape means no popping between evidence types, and the feather
    both hides residual per-frame jitter and reads far less harsh than a hard
    mask edge. The ellipse is pre-grown by the feather radius so the fully
    opaque core still covers the padded box (feathering never shrinks
    coverage).
    """
    mask = np.zeros(shape_hw, dtype=np.uint8)
    boxes = list(boxes)
    if not boxes:
        return mask
    diags = [float(np.hypot(b[2] - b[0], b[3] - b[1])) for b in boxes]
    k = 0
    if feather > 0:
        k = max(3, int(round(feather * (sum(diags) / len(diags)))) | 1)
    r = k / 2.0
    ex1 = ey1 = float("inf")
    ex2 = ey2 = float("-inf")
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax = max(1.0, w / 2.0 * (1.0 + 2.0 * pad) + r)
        ay = max(1.0, h / 2.0 * (1.0 + 2.0 * pad) + r)
        # cv2.ellipse clips to the frame itself — no centre-in-bounds guard,
        # so a head half out of frame keeps its sliver of blur.
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                    (int(round(ax)), int(round(ay))), 0, 0, 360, 255, -1)
        ex1, ey1 = min(ex1, cx - ax), min(ey1, cy - ay)
        ex2, ey2 = max(ex2, cx + ax), max(ey2, cy + ay)
    if k >= 3:
        # Feather only the sub-rect the ellipses touch, grown by a full kernel
        # width. Every nonzero pixel is then at least k from that sub-rect's
        # border, so a border pixel's kernel sees nothing but zeros and the
        # result is bit-identical to blurring the whole frame — while a 4K
        # frame with one head stops paying for a 3840x2160 Gaussian.
        mh, mw = shape_hw
        bx1 = max(0, int(np.floor(ex1)) - k)
        by1 = max(0, int(np.floor(ey1)) - k)
        bx2 = min(mw, int(np.ceil(ex2)) + k + 1)
        by2 = min(mh, int(np.ceil(ey2)) + k + 1)
        if bx2 > bx1 and by2 > by1:
            roi = mask[by1:by2, bx1:bx2]
            mask[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (k, k), 0)
    return mask


# ── GPU-accelerated stacked blur pipeline ────────────────────────────────────

def _is_soft(mask: np.ndarray) -> bool:
    """True when the mask has intermediate values (a feathered alpha mask).

    ``cv2.inRange`` + ``countNonZero`` is one C++ pass over one uint8
    temporary; the equivalent ``((mask > 0) & (mask < 255)).any()`` builds two
    full-size bool arrays and a third for the AND."""
    return cv2.countNonZero(cv2.inRange(mask, 1, 254)) > 0


def _blend(frame: np.ndarray, blurred: np.ndarray, mask: np.ndarray) -> None:
    """Alpha-composite ``blurred`` over ``frame`` in-place, weighted by mask."""
    a = mask.astype(np.float32)[..., None] / 255.0
    np.copyto(frame, (frame.astype(np.float32) * (1.0 - a)
                      + blurred.astype(np.float32) * a).astype(np.uint8))


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
        """Apply the blur stack to frame in-place where the mask is set.

        A binary mask (0/255) composites hard, exactly as before; a soft mask
        (any value in between, e.g. from ``render_head_mask``'s feathering)
        alpha-blends, so blur fades out over the feather band."""
        if not self._layers:
            return
        # boundingRect is a single C++ pass returning the box directly.
        # np.nonzero allocated two int64 arrays holding *every* set pixel's
        # coordinate — tens of MB per frame on a 4K feathered mask — purely to
        # take four min/max values off them.
        bx, by, bw, bh = cv2.boundingRect(mask)
        if bw == 0 or bh == 0:
            return

        fh, fw = frame.shape[:2]
        # Pad the ROI so the blur has valid context around the masked region
        # (a Gaussian reads its kernel's worth of neighbours; pixelate a block).
        gmax = max((s for k, s in self._layers if k == "gaussian"), default=0)
        pmax = max((s for k, s in self._layers if k == "pixelate"), default=0)
        pad = gmax + pmax + 4
        x1 = max(0, bx - pad); x2 = min(fw, bx + bw + pad)
        y1 = max(0, by - pad); y2 = min(fh, by + bh + pad)

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
        # Soft masks blend on the host — the blur itself (the heavy part)
        # already ran on the GPU, and the ROI crop keeps this cheap.
        if _is_soft(mask):
            _blend(frame, out.get(), mask)
        else:
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

        if _is_soft(mask):
            _blend(frame, out, mask)
        else:
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

        # Composite: alpha-blend by the mask (a binary mask degenerates to
        # the old hard select; a feathered one fades the blur out).
        mask_t = (
            torch.from_numpy(mask)  # type: ignore[possibly-undefined]
            .to(device)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        ) / 255.0  # (1, 1, H, W)
        result = t * (1.0 - mask_t) + out * mask_t

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
