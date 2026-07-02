# Technology Stack & Architecture

## Overview

The pipeline is a **single-detector, two-pass** anonymiser: one ONNX model
finds body/head/face boxes, a Kalman tracker links them across frames, an
offline cleanup pass turns the raw tracklets into a per-frame render table,
and the render pass paints feathered ellipse masks and blurs them.

> For the end-to-end dataflow diagram (Mermaid) and a per-stage tech table,
> see [DATAFLOW.md](DATAFLOW.md).

```
Video ──▶ Pass 1 (analyse)                       Pass 2 (render)
          Detection  → YOLOv9-Wholebody17        RenderTable lookup
          Fusion     → fuse_heads()              render_head_mask()
          Tracking   → Kalman + BYTE             BlurPipeline (GPU)
          Recording  → TrackRecorder             FFmpegWriter (+audio copy)
                   ╲                            ╱
                    ▶ tracklets.postprocess() ─▶
                      trim · prune · bridge · interpolate · smooth
```

---

## Components and why they were chosen

### Detection — PINTO 457_YOLOv9-Wholebody17 (`src/libs/detector.py`)

One post-processed ONNX graph (NMS, BGR handling and normalisation embedded)
emits `[N, 7]` rows of `[batchno, classid, score, x1, y1, x2, y2]`. Only the
Body (0), Head (7) and Face (8) classes are consumed.

- **Why a head detector, not a face detector.** The head class is annotated
  for all 360° orientations — "Head does not mean Face". Faces vanish when a
  person looks away or lies face-down; heads don't. The previous stack
  (SCRFD faces + RTMW pose + RF-DETR persons, ensembled) missed exactly those
  frames and cost 4 model passes per frame; this costs one.
- **Rotation assist.** Detection recall drops on in-plane-rotated (sideways)
  heads, which bed-angle footage is full of. `detect(rotations=(0, 90, 270))`
  re-runs the same session on rotated copies and merges unrotated results.
  Rotated passes are *assist-only*: stricter score floor, and any rotated
  "head" that contains an upright-detected head is discarded (a rotated scene
  can hallucinate one giant frame-sized head — verified and gated in tests).
- **Face fusion.** A face detection with no covering head box grows a
  head-proportioned pseudo-head (`fuse_heads`) — the recall backstop when the
  head class misses but the face class fires.
- **Swappable spec.** `AVPP_DETECTOR=yolox_bhhf` switches to the Apache-2.0
  434_YOLOX-Body-Head-Hand-Face model; `AVPP_DETECTOR_ONNX`/`_URL` override
  the file or mirror. Specs are read once at startup.

### Tracking — `src/libs/head_tracker.py`

Constant-velocity Kalman filter per head (`[cx, cy, w, h]` + velocities,
ByteTrack noise convention) with a three-stage association cascade, all solved
by the Hungarian method (`scipy.optimize.linear_sum_assignment`):

1. all tracks × high-score detections, IoU-gated;
2. **BYTE**: leftover tracks × low-score detections (sustain-only — the
   anti-flicker mechanism through occlusion);
3. fast-motion recovery: centre-distance + size-ratio gated (never a bare
   nearest-neighbour grab — that caused the old detection-stealing ghosts).

Lifecycle: births only from confident detections, `min_hits` consecutive hits
to confirm (one-frame false positives never render), confirmed tracks coast on
prediction up to `max_age_s` as bridge candidates only — coasted frames are
never blurred directly.

### Offline cleanup — `src/libs/tracklets.py`

The export's second brain. With the whole timeline recorded,
`postprocess()`:

1. **trims** coasted tails (predictions that never met a detection are not
   blurred — the old static-hold ghost fix);
2. **prunes** tracklets that are too short / never confident / mostly coasted;
3. **bridges** gaps ≤ `bridge_gap_s`: velocity-extrapolated distance gate,
   size-ratio gate, a **corridor gate** (never bridge through a region another
   surviving track occupies — the identity-smear guard) and an **ambiguity
   gate** (two near-equal candidates → bridge neither), then linear
   interpolation with mid-gap size padding;
4. **extends** each tracklet ~0.12 s at both ends (covers detector spin-up);
5. **smooths** cx/cy/w/h with zero-phase Savitzky-Golay — steady blur, no lag.

### Masking & blur — `src/libs/utils.py`

`render_head_mask()` draws one padded axis-aligned ellipse per head box and
feathers the whole mask with a single Gaussian — one consistent shape every
frame (no hull/ellipse popping), pre-grown by the feather radius so feathering
never shrinks coverage. `BlurPipeline` applies the configurable
Gaussian/pixelate stack once per frame (CUDA → OpenCL/UMat → CPU) and
alpha-composites soft masks; binary masks keep the fast hard path.

### Encoding — `src/libs/video_writer.py`

Raw BGR frames pipe to an FFmpeg subprocess (libx264, source-matched bitrate,
`+faststart`, co64-safe past 4 GiB). The source's audio track is
**stream-copied** into the output (`-map 1:a:0? -c:a copy`) — a remux, zero
cost. No `-shortest`: AAC priming makes audio fractionally shorter and it
would drop the final video frame.

### Runtime — ONNX Runtime with DirectML

`best_onnx_providers()` resolves **DirectML → CUDA → ROCm → CPU**. Two
DirectML survival rules shape the code:

1. **Never destroy a session.** Destroying one corrupts the provider's device
   state; the next inference dies with a native access violation. The detector
   session is created lazily, exactly once, per process.
2. **Pin graph shapes.** DirectML validates strictly; the detector input is
   fixed to `1×3×640×640` (`make_input_shape_fixed`) before the session is
   built.

fp16 conversion (`fp16_model_path`, ~2× on RDNA2) exists but the detector
ships fp32 first — fp16 across the embedded-NMS partition boundary is a known
risk; `AVPP_FP16=0` is the kill switch.

### GUI — PyQt6 (`src/ui.py`)

Three-panel inspector (Before / Tracking / After), preset chips + advanced
sliders, live preview on a worker thread (latest-wins mailbox), and the
two-pass export with pass-aware progress. Detection/tracking sliders drive
pass 1 and the preview; mask padding, feather and blur layers stay live even
during pass 2.

---

## Packaging

`build_exe.sh` (WSL2 → Windows Python interop → PyInstaller onefile/windowed):

- installs the trimmed dependency set (PyQt6, scipy, opencv-python, numpy,
  onnx, onnxconverter-common, imageio-ffmpeg);
- force-installs **onnxruntime-directml last** so no CPU-only wheel can
  clobber it, and asserts `DmlExecutionProvider` before building;
- **bundles the detector ONNX** (`--add-data … models`) so the .exe's first
  run downloads nothing — `libs.detector.model_path()` checks
  `sys._MEIPASS/models/` first, then `~/.cache/avpp/detector/`, then env
  overrides.

## Environment variables

| Variable | Effect |
| --- | --- |
| `AVPP_DETECTOR` | Spec name: `wholebody17` (default) or `yolox_bhhf` |
| `AVPP_DETECTOR_ONNX` | Absolute path to a local detector ONNX |
| `AVPP_DETECTOR_URL` | Mirror URL (`.onnx` or PINTO `.tar.gz`) |
| `AVPP_FP16` | `0` disables the fp16 model derivative |
| `AVPP_SKIP_MODEL_DOWNLOAD` | Preflight checks locations only |

## File organization

```
src/
  main.py            entry point; splash → GUI
  splash.py          loading splash
  ui.py              PyQt6 inspector + two-pass export worker
  libs/
    detector.py      single ONNX detector (body/head/face) + rotation assist
    head_tracker.py  Kalman + BYTE + Hungarian tracker
    tracklets.py     pass-1 recorder + offline cleanup → render table
    models.py        startup preflight (bundle/cache/download)
    utils.py         providers, fp16, feathered masks, GPU blur stack
    video_writer.py  streaming ffmpeg exporter (>4 GiB safe, audio copy)
```

## Debug artifacts (Windows)

- `%TEMP%\FaceBlurInspector-debug.log` — status lines, resolved providers,
  per-stage timings every 30 frames (`[analyse f…]` / `[render f…]`), and the
  between-pass tracklet summary (`N raw tracklets → M heads, blur on X/Y
  frames`).
- `%TEMP%\FaceBlurInspector-error.log` — tracebacks + faulthandler dumps.
