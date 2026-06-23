# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release rebuilds face finding around **whole-body pose** so the blur holds
on faces at the hard angles the previous frontal detector missed — two people
in bed, cuddling, top-down close-ups — and pushes the heavy work onto the GPU.

### Detection & estimation

- **RTMW whole-body pose backend** (`src/libs/pose_rtmw.py`). A RTMDet/YOLOX
  person detector feeds RTMW, producing 133 COCO-WholeBody keypoints per
  person; the 68 dense face keypoints (indices 23-90) become a tight face hull
  that keeps landing on the face when it looks up, down, or away. Built on
  `rtmlib` (pure ONNX Runtime — no mmcv/mmengine), lazy-loaded and degrading to
  SCRFD-only if unavailable.
- **SCRFD ⊕ RTMW ensemble** (`src/libs/pipeline.py`, shared by the CLI and
  inspector). SCRFD's 106-point mesh still wins on near-frontal faces; RTMW
  fills in every angle SCRFD misses (in test footage SCRFD found 1 face where
  the ensemble found 8).
- **Skin false-positive gating.** SCRFD detections that overlap no RTMW head
  region — bare skin mistaken for a face, a real problem on nude footage — are
  dropped instead of blurred. High-confidence detections are always kept so
  genuine faces are never lost.
- `--pose-backend {rtmw,mediapipe,none}` and `--pose-mode
  {performance,balanced,lightweight}` on the CLI; matching fields on the
  inspector's `Params`/presets. MediaPipe is retained as a legacy backend.

### GPU acceleration

- **All ONNX inference targets the GPU.** Provider preference is now DirectML →
  CUDA → ROCm → CPU, so an **AMD Radeon RX 6800** runs SCRFD, YOLOX and RTMW on
  DirectML (Windows). The RTMW session swap is best-effort per model and keeps a
  working CPU session if a provider rejects a graph.
- **GPU blur on AMD/Intel.** New OpenCV OpenCL/UMat blur path (`BlurPipeline`)
  runs the blur stack on any OpenCL GPU, where the previous PyTorch-CUDA path
  only ever fired on NVIDIA.
- Startup diagnostics print the active ONNX provider and blur backend.

### Packaging (Windows .exe)

- `build_exe.sh` now installs and bundles **rtmlib** (`--collect-all rtmlib`),
  fixing the `No module named 'rtmlib'` crash in the frozen `.exe` that dropped
  it to SCRFD-only. rtmlib is installed with `--no-deps` so its plain
  `onnxruntime` dependency can't overwrite the DirectML build (which would
  silently force CPU inference); `onnxruntime-directml` is force-reinstalled
  last. The build smoke test now asserts `DmlExecutionProvider` is present, so a
  CPU-only bundle fails the build instead of shipping. Verified end-to-end: a
  frozen test exe imports `rtmlib.Wholebody` and reports DirectML active.

### Fixes

- `LandmarkSmoother` now resets a track's history when its landmark count
  changes (SCRFD's 106 points ⇄ RTMW's 68), which would otherwise index out of
  range as a head turned between backends.

## [0.2.0] — 2026-06-17

This release is about **tracking and estimating faces more reliably** — keeping
the blur locked on a face through head turns, glances down, and partial
occlusion, instead of flickering off the moment the detector loses a frame.

### Tracking & estimation

- **Kalman-filter face tracker** (`src/libs/tracker.py`). Each face owns a
  constant-velocity Kalman filter with state `[cx, cy, w, h, vx, vy]` (size is a
  random walk). Every frame it *predicts* the head's position and then *corrects*
  from measurements, so when the detector drops a face the track coasts on
  prediction rather than disappearing. Velocity is damped on every coasting frame
  so a lost box can't sail across the frame, and a track is dropped only after
  `Hold (s)` seconds without any correction or once it leaves the frame. This
  replaced the earlier boxmot/ByteTrack wrapper, which only ever returned tracks
  matched in the current frame.
- **Pose-estimation head-box recovery** (`src/libs/pose_head.py`). MediaPipe
  Pose still sees the *person* when a frontal face detector gives up, so a coarse
  head box is derived from the pose keypoints — ear-to-ear width when visible,
  falling back to eye span, then to an estimate hung above the shoulders for a
  full turn-away. That box revives a lost track as a *weak* correction (it pins
  position but barely nudges size), keeping the blur on even when the face is
  turned away or seen from behind.
- **Skeleton & head-box overlay in the Tracking panel.** The inspector now draws
  the MediaPipe pose skeleton and the coarse head box that is fed to the tracker,
  so you can see pose assist working frame by frame.
- **Other estimation improvements:**
  - Distance-fallback matching attaches a detection to a track by centre
    proximity when fast motion drops IoU to zero between consecutive frames.
  - Close-up faces (filling most of the frame) are re-detected on an upscaled
    crop for tighter 106-point landmarks.
  - Savitzky-Golay landmark smoothing (`src/libs/smoother.py`) removes
    frame-to-frame jitter so the blur mask doesn't wiggle.
  - Strong (face detection) corrections override weak (pose) ones, so the filter
    snaps back to the real face the moment it reappears.

### Added

- Skeleton overlay and head-box visualization in the inspector's Tracking panel.
- Streaming, ffmpeg-based video writer (`src/libs/video_writer.py`).
- `docs/TECH_STACK.md` — technology stack and architecture documentation.

### Fixed

- Exports failing or corrupting past ~4 GiB: replaced the previous writer with a
  streaming ffmpeg-based one (`src/libs/video_writer.py`).
- Documentation drift — corrected the smoother class name and defaults, the blur
  pipeline example, the pose head-box description, and the file listings in
  `docs/TECH_STACK.md`.

### Docs

- README and `docs/TECH_STACK.md` refreshed to match the current code.

## [0.1.0-beta] — 2026-06-12

Initial public preview.

- InsightFace SCRFD face detection with 106-point landmarks (`src/libs/face_app.py`).
- Per-face Kalman tracker with detection-gap coasting (`src/libs/tracker.py`).
- MediaPipe pose head-region recovery for lost faces (`src/libs/pose_head.py`).
- Stackable Gaussian + pixelate blur over an expanded landmark-hull mask
  (`src/libs/utils.py`).
- PyQt6 three-panel inspector (Before / Tracking / After) with presets and live
  tuning (`src/ui.py`).
- Headless CLI for batch jobs (`src/cli.py`).
- Standalone Windows `.exe` build via PyInstaller (`build_exe.sh`).

[0.2.0]: https://github.com/kabir0st/automated-video-privacy-pipeline/releases/tag/v0.2.0
[0.1.0-beta]: https://github.com/kabir0st/automated-video-privacy-pipeline/releases/tag/0.1.0-beta-0.1
