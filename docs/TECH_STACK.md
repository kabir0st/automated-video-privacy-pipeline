# Technology Stack & Architecture

## Overview

The Automated Video Privacy Pipeline is built as a modular detection → tracking → smoothing → masking → blurring system. Each component uses a specific technology chosen for accuracy, performance, and reliability.

> For the end-to-end dataflow diagram (Mermaid) and a per-stage tech table, see
> [DATAFLOW.md](DATAFLOW.md).

```
Video Frame
    ↓
[Detection] → InsightFace SCRFD (faces + 106 landmarks)
              ⊕ RTMW Wholebody pose (faces + head boxes)
              → merge_detections() ensemble
    ↓
[Tracking] → Kalman Filter (per-face state estimation)
              ↑ pose head box = weak correction (revives lost tracks)
    ↓
[Smoothing] → Savitzky-Golay Filter (landmark noise reduction)
    ↓
[Masking] → Convex Hull + Expansion (landmark-fitted mask)
    ↓
[Blurring] → PyTorch (CUDA) / OpenCV UMat (OpenCL) / CPU (Gaussian + Pixelate)
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

### 1b. **Detection ensemble: SCRFD ⊕ RTMW**

**What it does:** Combines the SCRFD face detector with RTMW pose so the two cover
each other's blind spots before anything reaches the tracker.

**Where:** `libs/pipeline.py` → `detect()` / `merge_detections()`

**Why an ensemble?** On the footage this app targets (two people, odd top-down
angles, nude scenes) SCRFD has two failure modes the pose model fixes:

- **Missed faces at odd angles** — SCRFD only fires near-frontal, so a face looking
  up/down/away is never detected. RTMW estimates the head from whole-body context
  and contributes a face detection there.
- **Skin blurred as a face** — SCRFD occasionally fires on bare skin. Any SCRFD box
  overlapping *no* RTMW head region is dropped (skin on a torso has no head
  keypoints near it), while high-confidence SCRFD boxes (≥ 0.70) are always kept so
  genuine frontal faces are never lost.

Where both agree, SCRFD's denser 106-point mesh wins; RTMW fills in everywhere
SCRFD is silent.

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

### 3. **Pose assist & recovery: RTMW (default) / MediaPipe (legacy)**

**What it does:** Whole-body pose estimation finds heads that the face detector
can't — both to *contribute face detections* (the ensemble above) and to *recover
lost tracks* (when a head turns away, looking down). It also supplies the head
regions SCRFD detections are gated against.

**Where:** `libs/pose_rtmw.py` → `RTMWPoseEstimator` (**default**); legacy
`libs/pose_head.py` → `PoseHeadEstimator` (MediaPipe). The backend is chosen by
`make_pose_backend()` in `libs/pipeline.py` and the `--pose-backend
{rtmw,mediapipe,none}` flag.

**Why RTMW (rtmlib `Wholebody`: YOLOX person detector → RTMW pose)?**
- Estimates 133 COCO-WholeBody keypoints *per person*; 68 of those are dense face
  landmarks that keep landing on the real face at angles that kill a frontal
  detector
- Anchored to a coherent body skeleton, it does **not** fire on random bare skin —
  exactly the false-positive that plagued the old pipeline on nude footage
- Degrades gracefully: if rtmlib/model load fails, it flips to SCRFD-only instead
  of crashing
- `--pose-mode performance` (RTMW-x, best) / `balanced` / `lightweight` (RTMW-l,
  fast CPU preview)

**MediaPipe (legacy `--pose-backend mediapipe`):** detects 33 body landmarks and
builds a coarse head box (ear-to-ear width when visible, falling back to eye span,
then an estimate above the shoulders). It yields head boxes for tracker assist but
**no** face landmarks, so it contributes no face detections and gates nothing —
SCRFD behaves as before, just with head-box revival of lost tracks.

**How it feeds the tracker:**
- A head box is fed to the Kalman filter as a *weak correction* (pins position,
  barely moves size)
- Strong corrections (a new face detection) override weak ones, so when the face
  returns the filter snaps to the real detection
- The inspector's Tracking panel overlays the pose skeleton and head box, so you
  can confirm pose assist is alive frame by frame

**Example:**
```
Person turns their head (SCRFD face detection drops)
  ↓
RTMW still estimates the head from the body skeleton
  ↓
Head box → Kalman filter (weak correction), track revived
  ↓
Blur stays on as the person turns / walks out of frame
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

**Tech choice (backend picked once at construction):**
- **CUDA / PyTorch** — NVIDIA GPUs (`F.conv2d` Gaussian, `avg_pool2d` mosaic)
- **OpenCL / OpenCV UMat** — any OpenCL device; this is the path that lights up an
  AMD Radeon (e.g. RX 6800) where torch-CUDA never applies, so the blur still runs
  on the GPU
- **CPU / OpenCV** — fallback

Both Gaussian and pixelate run as one stacked composite pass per frame, regardless
of how many faces are present.

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
InsightFace SCRFD (faces + 106 landmarks)
    │                                   RTMW Wholebody pose (faces + head boxes)
    └────────────┬──────────────────────────────┘
                 ↓
        merge_detections() ensemble (libs/pipeline.py)
                 ↓
            Kalman Filter (tracking)  ←─ pose head box (weak correction)
                 ↓
            Smoothing (Savitzky-Golay)
                 ↓
            MaskBuilder (convex hull)
                 ↓
            BlurPipeline (Gaussian + Pixelate · CUDA / OpenCL / CPU)
```

---

## Performance Notes

| Component | GPU | CPU (WSL2) | Notes |
|-----------|-----|-----------|-------|
| InsightFace detection | ~6 fps (1080p) | ~1 fps (1080p) | DirectML/CUDA accelerated |
| Kalman tracking | — | Real-time | No GPU needed |
| RTMW pose (default) | ~10 fps | ~2 fps | rtmlib YOLOX→RTMW; bottleneck if enabled. MediaPipe legacy backend is lighter/faster |
| Blur (Gaussian) | Real-time | ~3 fps | PyTorch CUDA speeds this up |
| Blur (Pixelate) | — | Real-time | NumPy, no GPU needed |
| **Overall** | ~6 fps | ~4 fps | Limited by detection |

---

## Key Design Decisions

### Why not ByteTrack?
We initially used ByteTrack (popular in sports) but it only returns tracks *matched this frame*. On detection dropout, tracks vanish immediately. Kalman filtering solves this by predicting position when detection fails.

### Why Kalman + whole-body pose (RTMW)?
Kalman alone coasts until detection returns. Adding pose estimation keeps the track *corrected* as long as the person is visible, even from behind. RTMW is the default (it also contributes face detections and suppresses skin false-positives via the ensemble); MediaPipe remains a lighter legacy backend that only supplies head boxes. This is especially useful in scenarios where faces turn away but blur must remain.

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
  pipeline.py            shared detection front-end + SCRFD⊕RTMW ensemble
  face_app.py            InsightFace loader (DirectML-safe)
  tracker.py             Kalman filter
  pose_rtmw.py           RTMW whole-body pose → faces + head boxes (default)
  pose_head.py           MediaPipe pose → head boxes (legacy backend)
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
