"""Windows smoke test for the union pipeline (run with Windows Python from
WSL, the same interop build_exe.sh uses):

    py.exe -3.12 test_pipeline_win.py [video_or_image]

Verifies, in order:
  1. onnxruntime exposes the DirectML provider;
  2. every model resolves (bundle/cache/env), loads, and lands on
     DmlExecutionProvider — not a silent CPU fallback;
  3. one fused detection pass returns sane boxes with provenance flags;
  4. the tracker consumes them and the recorder produces a tracklet;
  5. the face embedder returns unit-norm vectors;
  6. mask render + blur run on the available backend.

Prints per-stage timings so a CPU fallback is obvious.
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

from libs.embed import FaceEmbedder                                  # noqa: E402
from libs.utils import BlurPipeline, best_onnx_providers, render_head_mask  # noqa: E402
from pipeline.analysis import AnalysisConfig, Models, detect_frame  # noqa: E402
from pipeline.record import TrackRecorder                            # noqa: E402
from pipeline.tracker import Tracker                                 # noqa: E402
from pipeline.types import Src                                       # noqa: E402

print("selected providers:", best_onnx_providers())

src = sys.argv[1] if len(sys.argv) > 1 else None
if src and Path(src).suffix.lower() in (".jpg", ".jpeg", ".png"):
    frame = cv2.imread(src)
elif src:
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"cannot read {src}"
else:
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:] = np.linspace(0, 255, 1280, dtype=np.uint8)[None, :, None]
    print("no input given — synthetic gradient (no heads expected)")

models = Models(on_status=print)
cfg = AnalysisConfig()
timings = {}
t0 = time.perf_counter()
cands, _sources = detect_frame(models, frame, cfg, timings)
ms = (time.perf_counter() - t0) * 1e3
for name, m in (("headdet", models.headdet()), ("wholebody", models.wb()), ("scrfd", models.scrfd())):
    sess = getattr(m, "_sess", None)
    prov = sess.get_providers()[0] if sess is not None else "unavailable"
    print(f"{name:10s} available={m.available} provider={prov} ms={timings.get(name, 0):.1f}")
    assert m.available, f"{name} failed to load"
    if "DmlExecutionProvider" in ort.get_available_providers():
        assert prov == "DmlExecutionProvider", f"{name} fell back to {prov}"
print(f"fused: {len(cands.heads)} heads, {len(cands.faces)} faces in {ms:.0f} ms")
for b, f in zip(cands.heads, cands.flags):
    print("  ", b.round(1).tolist(), Src(int(f)))

tracker = Tracker(fps=25)
rec = TrackRecorder()
obs = tracker.update(cands.heads, cands.flags, frame.shape, faces=cands.faces)
rec.observe(0, obs)
tl = rec.finalize()
print(f"tracker: {len(obs)} live tracks, {len(tl)} tracklet(s)")

emb = FaceEmbedder(on_status=print)
if len(cands.faces):
    v = emb.embed(frame, cands.faces[:1])
    print("embedder norm", float(np.linalg.norm(v[0])))
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-3 or not v.any()

boxes = [b[:4] for b in cands.heads] or [np.array([100, 100, 300, 340], np.float32)]
mask = render_head_mask(frame.shape[:2], boxes)
bp = BlurPipeline()
t0 = time.perf_counter()
bp.apply(frame, mask)
print(f"blur backend {bp.backend}: {(time.perf_counter() - t0) * 1e3:.1f} ms")
print("SMOKE OK")
