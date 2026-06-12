"""Headless reproduction of the ui.py processing pipeline (no Qt).

Run with Windows Python from WSL2:
    py.exe -3.12 test_pipeline_win.py
"""
import sys
from pathlib import Path

def trace(msg: str) -> None:
    print(msg, flush=True)

trace("[0] interpreter up")
sys.path.insert(0, str(Path(__file__).parent / "src"))

trace("[1] importing numpy")
import numpy as np
trace("[2] importing cv2")
import cv2
trace("[3] importing onnxruntime")
import onnxruntime
trace("[4] importing torch")
import torch
trace("[5] importing mediapipe")
import mediapipe

print("=== 1. onnxruntime providers ===")
from libs.utils import best_onnx_providers, BlurPipeline, MaskBuilder
providers = best_onnx_providers()
print("providers:", providers)

print("=== 2. FaceApp load (downloads buffalo_l on first run) ===")
from libs.face_app import FaceApp
app = FaceApp(providers=providers)
app.prepare(ctx_id=0, det_size=(640, 640))
print("models loaded: detection + landmark_2d_106")

print("=== 3. detection on real faces (insightface sample image) ===")
import insightface.data
frame = insightface.data.get_image("t1")  # group photo with several faces
fh, fw = frame.shape[:2]
faces = app.get(frame)
print(f"faces detected: {len(faces)}")
for f in faces:
    lm = f.landmark_2d_106
    print(f"  bbox={f.bbox[:4].astype(int)} score={f.det_score:.2f} "
          f"landmarks={'None' if lm is None else lm.shape}")

print("=== 4. close-up refinement path ===")
from libs.utils import crop_face_patch, unproject_landmark
crop, (ox, oy, sc) = crop_face_patch(frame, faces[0].bbox[:4], target_size=1024)
cfs = app.get(crop)
print(f"crop {crop.shape} -> {len(cfs)} faces")

print("=== 5. Kalman tracker update (with detection-gap coast) ===")
from libs.tracker import KalmanFaceTracker
tracker = KalmanFaceTracker(fps=25.0, match_iou=0.3, hold_secs=2.0)
tracked = tracker.update(faces, frame.shape)
print("tracked:", len(tracked))
for t in tracked:
    print(f"  track {t.track_id}: source={t.source}")
# Detector goes blind: every track must keep coasting on prediction.
coasted = tracker.update([], frame.shape)
assert len(coasted) == len(tracked), "tracks vanished on detection gap"
assert all(t.source == "coast" for t in coasted)
print(f"coasting OK: {len(coasted)} tracks held without detections")

print("=== 6. pose head boxes (mediapipe, downloads model on first run) ===")
from libs.pose_head import PoseHeadEstimator
pose = PoseHeadEstimator(on_status=print)
heads = pose.head_boxes(frame)
print(f"pose available={pose.available} head boxes={len(heads)}")

print("=== 7. blur with face polygons (stacked layers) ===")
blur = BlurPipeline()
blur.reconfigure((("gaussian", 71), ("pixelate", 10), ("gaussian", 31)))
masks = MaskBuilder()
mask = np.zeros((fh, fw), dtype=np.uint8)
for t in coasted:
    masks.add(mask, t)
blur.apply(frame, mask)
print("blur OK")

print("=== ALL STAGES PASSED ===")
