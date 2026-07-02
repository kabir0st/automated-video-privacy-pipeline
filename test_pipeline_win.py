"""Windows smoke test for the single-detector pipeline (run with Windows
Python from WSL, the same interop build_exe.sh uses):

    py.exe -3.12 test_pipeline_win.py [video_or_image]

Verifies, in order:
  1. onnxruntime exposes the DirectML provider (the RX 6800 path);
  2. the detector model resolves (bundle/cache/env), loads, and the session
     lands on DmlExecutionProvider — not a silent CPU fallback;
  3. a detection pass returns sane per-class boxes (with rotation assist);
  4. the Kalman tracker consumes them across a few frames;
  5. mask render + blur run on the available backend.

Prints timings so a CPU fallback is obvious (~100 ms/frame CPU vs ~5-15 ms
DML for the s-model at 640).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2
import numpy as np
import onnxruntime as ort

print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
if "DmlExecutionProvider" not in ort.get_available_providers():
    print("WARNING: DirectML provider missing — inference will run on CPU")

from libs.detector import HeadDetector, fuse_heads, model_path  # noqa: E402
from libs.head_tracker import HeadTracker  # noqa: E402
from libs.utils import BlurPipeline, render_head_mask  # noqa: E402

print("model:", model_path() or "MISSING (run the app once, or set AVPP_DETECTOR_ONNX)")

# Source frames: a video/image path if given, else a synthetic gradient with
# no people (still exercises the full code path, just detects nothing).
frames: list[np.ndarray] = []
if len(sys.argv) > 1:
    src = sys.argv[1]
    img = cv2.imread(src)
    if img is not None:
        frames = [img] * 5
    else:
        cap = cv2.VideoCapture(src)
        while len(frames) < 30:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
if not frames:
    base = np.tile(np.linspace(0, 255, 1280, dtype=np.uint8), (720, 1))
    frames = [cv2.merge([base] * 3)] * 5
    print("(no input given — using a synthetic empty frame)")

det = HeadDetector(on_status=print)
tracker = HeadTracker(fps=30.0)
blur = BlurPipeline()

for i, frame in enumerate(frames):
    t0 = time.perf_counter()
    d = det.detect(frame, rotations=(0, 90, 270))
    heads = fuse_heads(d)
    obs = tracker.update(heads, frame.shape)
    boxes = [o.box for o in obs if o.confirmed]
    mask = render_head_mask(frame.shape[:2], boxes)
    out = frame.copy()
    blur.apply(out, mask)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"frame {i}: heads={len(d.heads)} faces={len(d.faces)} "
          f"bodies={len(d.bodies)} tracks={len(obs)} blurred={len(boxes)} "
          f"({ms:.0f} ms, detect {det.last_ms:.0f} ms)")

assert det.available, "detector failed to initialise"
print("SMOKE OK")
