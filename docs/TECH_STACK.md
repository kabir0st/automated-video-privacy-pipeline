# Technology Stack & Architecture

## Overview

The Automated Video Privacy Pipeline is built as a modular detection → tracking → smoothing → masking → blurring system. Each component uses a specific technology chosen for accuracy, performance, and reliability.

```
Video Frame
    ↓
[Detection] → InsightFace SCRFD (face detection + landmarks)
    ↓
[Tracking] → Kalman Filter (per-face state estimation)
    ↓ (if lost)
[Recovery] → MediaPipe Pose (body-pose head boxes)
    ↓
[Smoothing] → Savitzky-Golay Filter (landmark noise reduction)
    ↓
[Masking] → Convex Hull + Expansion (landmark-fitted mask)
    ↓
[Blurring] → PyTorch / NumPy (Gaussian + Pixelate layers)
    ↓
Anonymized Frame
```

---

## Component Breakdown

### 1. **Detection: InsightFace SCRFD**

**What it does:** Finds all faces in a frame and extracts 106 facial landmarks (eyes, nose, mouth, jaw, etc.).

**Where:** `libs/face_app.py`

**Why InsightFace?**
- SCRFD is fast (384×640 at 30+ fps on CPU) and accurate even on small/tilted faces
- 106-point landmarks let us fit a precise mask around the face contour, not just a bounding box
- Works offline (models ship locally)
- Mature codebase with DirectML/CUDA/CPU backends

**How it's used:**
```python
# Get face detections on the frame
faces = face_app.get(frame, target_size=640)
# Each face has: bbox, landmarks (106 points), confidence score
```

---

### 2. **Tracking: Kalman Filter**

**What it does:** Tracks each detected face across frames so the blur stays attached even if detection temporarily fails.

**Where:** `libs/tracker.py` → `KalmanFaceTracker`

**Why Kalman?**
- Per-frame detection is fragile: occlusions, head turns, and motion blur cause detections to drop out
- A Kalman filter predicts where a face *should be* based on its motion, so the blur coasts smoothly instead of disappearing
- Simple, fast, and battle-tested (used in robotics, autonomous vehicles, etc.)

**How it works:**
- **State:** Position (cx, cy), size (w, h), velocity (vx, vy) — 6 numbers per face
- **Prediction:** "If this face was moving right at 10 px/frame, it's probably 10 px further right now"
- **Correction:** When a new detection arrives, the filter updates the state to match observed position
- **Coasting:** If detection drops, the filter keeps predicting for up to `hold_secs` (default 2.0), then the track dies. Velocity is damped on each coasting frame so a lost box cannot drift across the frame
- **Fallbacks:** Detections are matched to tracks by IoU, with a centre-distance fallback so fast motion (which drops IoU to zero between frames) doesn't break the association

**Example:**
```
Frame 1: Face detected at (100, 200), velocity (5, 0)
Frame 2: No detection, but Kalman predicts (105, 200) and blur follows
Frame 3: Still no detection, Kalman predicts (110, 200)
Frame 4: After 2 seconds, track is dropped if no new detection arrives
```

---

### 3. **Recovery: MediaPipe Pose**

**What it does:** When a face is lost (head turned away, looking down), MediaPipe detects the person's body and estimates where their head is.

**Where:** `libs/pose_head.py` → `PoseHeadEstimator`

**Why MediaPipe Pose?**
- Solves the "blur vanishes when someone looks away" problem
- Works even when the face is completely turned around (side view, back view)
- Lightweight (~5 MB model) and fast

**How it's used:**
- MediaPipe detects 33 body landmarks (nose, eyes, ears, shoulders, etc.)
- A coarse head box is built primarily from the head landmarks — ear-to-ear width
  when visible (robust even from behind), falling back to eye span, and finally to
  an estimate hung above the shoulders for a full turn-away — then fed to the
  Kalman filter as a *weak correction* (it pins position but barely moves size)
- Strong corrections (new face detection) override weak ones, so if the face comes back into view, the filter snaps to the real detection
- The inspector's Tracking panel overlays the pose skeleton and this head box, so you can confirm pose assist is alive frame by frame

**Example:**
```
Person turns their head (face detection drops)
  ↓
Kalman coasts for a few frames
  ↓
MediaPipe Pose sees the person's shoulders and body
  ↓
Estimated head box sent to Kalman filter
  ↓
Blur stays on as the person walks out of frame
```

---

### 4. **Smoothing: Savitzky-Golay Filter**

**What it does:** Reduces jitter in the 106 landmarks so the blur mask doesn't wiggle frame-to-frame.

**Where:** `libs/smoother.py` → `LandmarkSmoother`

**Why Savitzky-Golay?**
- Preserves the shape of the landmark cloud while removing high-frequency noise
- Handles occlusion (if a landmark is missing, hold the last known value)
- Better than simple averaging because it doesn't blur edges

**How it's used:**
- Runs over a sliding window of recent frames (default: 15-frame window, polynomial order 2)
- Fits the polynomial to each landmark's trajectory and reads off the smoothed latest point
- Outputs smoothed positions that track real motion without jitter

---

### 5. **Masking: Convex Hull + Expansion**

**What it does:** Converts the 106 landmarks into a polygon mask that covers the face and ears.

**Where:** `libs/utils.py` → `MaskBuilder`

**How it works:**
- Compute the convex hull of the 106 landmarks (outermost points form a polygon)
- Expand the polygon outward by `hull_expand` pixels in all directions (catch ears, jawline)
- Expand additional pixels *upward* by `hair_extra` to cover hair and hats
- Render the polygon as a binary mask

**Why this approach?**
- Landmark-fitted masks are tighter than axis-aligned boxes
- You can tune coverage without changing detection or tracking
- Expansion parameters are intuitive: bigger = more coverage

**Example:**
```
Face landmarks: 106 points scattered around the face
    ↓
Convex hull: Outline of the face perimeter
    ↓
Expand by 0.45: Outward in all directions
    ↓
Hair extra 0.9: Additional upward growth
    ↓
Mask polygon: A shape that covers face + ears + hair
```

---

### 6. **Blurring: PyTorch + NumPy**

**What it does:** Applies stackable Gaussian and pixelate blur layers to the masked region.

**Where:** `libs/utils.py` → `BlurPipeline`

**Tech choice:**
- **Gaussian blur:** PyTorch (CUDA) for GPU, OpenCV/SciPy for CPU
- **Pixelate:** NumPy (always CPU, since it's just averaging pixels)

**Why stack layers?**
- Gaussian alone is reversible (image forensics can sometimes recover faces)
- Pixelate (mosaic) is hard to reverse, but can look crude on its own
- Gaussian → Pixelate gives you a soft wash + hard mosaic = maximum privacy with good aesthetics

**How to use:**
```python
# Stack layers: Gaussian 71 kernel, then Pixelate 10 px blocks
layers = [("gaussian", 71), ("pixelate", 10)]
pipeline.reconfigure(layers)
pipeline.apply(frame, mask)  # blurs the masked region of `frame` in place
```

---

### 7. **UI: PyQt6**

**What it does:** Provides the inspector with three live panels (Before / Tracking / After), preset chips, and export controls.

**Where:** `src/ui.py`

**Key pattern: Preview Mailbox**
- Don't emit numpy frames through Qt signals (GUI can't keep up with 3×1080×1920 pixmap builds)
- Instead, `ProcessWorker` writes the latest preview to a shared memory slot (`_emit_preview`)
- GUI reads it (`take_preview`) on demand, avoiding the queue backlog

**GPU acceleration:**
- ONNX Runtime auto-selects DirectML (Windows) → CUDA (NVIDIA) → CPU

---

## Dependency Graph

```
InsightFace (det + landmarks)
    ↓
    ├→ Kalman Filter (tracking)
    │    ↓
    │    └→ Smoothing (Savitzky-Golay)
    │        ↓
    │        └→ MaskBuilder (convex hull)
    │            ↓
    │            └→ BlurPipeline (Gaussian + Pixelate)
    │
    └→ MediaPipe Pose (recovery)
        ↓
        └→ Kalman Filter (weak correction)
```

---

## Performance Notes

| Component | GPU | CPU (WSL2) | Notes |
|-----------|-----|-----------|-------|
| InsightFace detection | ~6 fps (1080p) | ~1 fps (1080p) | DirectML/CUDA accelerated |
| Kalman tracking | — | Real-time | No GPU needed |
| MediaPipe pose | ~10 fps | ~2 fps | Optional; bottleneck if enabled |
| Blur (Gaussian) | Real-time | ~3 fps | PyTorch CUDA speeds this up |
| Blur (Pixelate) | — | Real-time | NumPy, no GPU needed |
| **Overall** | ~6 fps | ~4 fps | Limited by detection |

---

## Key Design Decisions

### Why not ByteTrack?
We initially used ByteTrack (popular in sports) but it only returns tracks *matched this frame*. On detection dropout, tracks vanish immediately. Kalman filtering solves this by predicting position when detection fails.

### Why Kalman + MediaPipe Pose?
Kalman alone coasts until detection returns. Adding MediaPipe pose keeps the track *corrected* as long as the person is visible, even from behind. This is especially useful in scenarios where faces turn away but blur must remain.

### Why convex hull, not a tight ellipse?
Ellipses can miss ears and asymmetric hairstyles. Convex hulls fit the actual landmark cloud shape and let you tune coverage with simple parameters (expand, hair extra).

### Why Savitzky-Golay smoothing?
Simple averaging blurs edges. Savitzky-Golay fits a polynomial, preserving landmark shape while removing jitter. This prevents the blur mask from wiggling frame-to-frame.

---

## File Organization

```
src/
  main.py                 entry point
  cli.py                  headless pipeline
  ui.py                   PyQt6 inspector
  
libs/
  face_app.py            InsightFace loader (DirectML-safe)
  tracker.py             Kalman filter
  pose_head.py           MediaPipe pose → head boxes
  smoother.py            Savitzky-Golay smoothing
  utils.py               MaskBuilder + BlurPipeline
  video_writer.py        streaming ffmpeg exporter (handles >4 GiB output)
```

---

## Getting Started

1. **Install:** `uv sync`
2. **Run GUI:** `uv run python src/main.py`
3. **Run CLI:** `uv run python src/main.py --input video.mp4 --output blurred.mp4`

All models download on first run. Everything else is offline.
