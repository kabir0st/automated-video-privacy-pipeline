# Technology Stack & Architecture

## Overview

The pipeline is an **evidence-gated, two-pass** anonymiser: three lightweight
ONNX models (a body/head/face/parts detector, a landmark-checked witness face
detector, and a body pose estimator) all feed one acceptance gate — no
detection is ever a blur target by itself. Only a face claim that clears
anatomical anchoring, independent corroboration, and an unresolved-veto check
becomes a candidate a Kalman tracker may follow. Offline, a fourth model (an
independent, adult-content-trained witness) re-verifies every surviving
track, a review dialog gives the user the final say, and only then does the
render pass paint feathered ellipse masks and blur them.

> For the end-to-end dataflow diagram (Mermaid) and a per-stage tech table,
> see [DATAFLOW.md](DATAFLOW.md).

```
Video ──▶ Pass 1 (analyse)                              Pass 2 (render)
          Detection    → YOLOv9-Wholebody17             RenderTable lookup
                          (body/head/face/parts/neg)     render_head_mask()
          Witness      → SCRFD (every frame)             BlurPipeline (GPU)
          Pose         → RTMPose-m body7                 FFmpegWriter (+audio copy)
          Evidence gate→ gate_face_candidates()               ▲
          Tracking     → Kalman + BYTE                        │
          Recording    → TrackRecorder                        │
                   ╲                                           │
                    ▶ clean_tracklets() ──▶ ReviewDialog ──────┘
                      trim · ledger prune · cross-model verify ·
                      bridge · interpolate · smooth
```

---

## Components and why they were chosen

### The core problem: a confidence threshold can't separate "face" from "face-shaped skin"

This footage has heavy skin exposure, extreme close-ups and unusual poses —
exactly the conditions where a face detector's raw confidence score stops
being a useful precision knob. A sock, a knee, a shoulder can all score
highly on a model trained to recognise face-shaped regions; raising the
threshold only trades false positives for missed real faces, it never
separates the two. The single fix that works (and the one this project's own
history validates — an earlier pose+multi-model pipeline had this problem
solved, then a since-removed single-detector rewrite lost it) is **anatomical
and cross-model corroboration**: a face claim is only trusted once something
*independent* backs it up spatially. That corroboration is the entire reason
detector.py, scrfd.py and pose.py all exist side by side feeding one gate.

### Detection — PINTO 457_YOLOv9-Wholebody17 (`src/libs/detector.py`)

One post-processed ONNX graph (NMS, BGR handling and normalisation embedded)
emits `[N, 7]` rows of `[batchno, classid, score, x1, y1, x2, y2]` across 17
classes. Body/head/face are the primary boxes; eye/nose/mouth/ear ("parts")
and hand/foot ("negatives") are parsed too — **all of it is evidence for the
gate, none of it is a blur target by itself.**

- **Why a head detector, not just a face detector.** The head class is
  annotated for all 360° orientations — "Head does not mean Face". Faces
  vanish when a person looks away or lies face-down; heads don't. A head box
  is what anchors a face claim anatomically (`Ev.HEAD_ANCHOR`).
- **Rotation assist.** Detection recall drops on in-plane-rotated (sideways)
  heads, which bed-angle footage is full of. `detect(rotations=(0, 90, 270))`
  re-runs the same session on rotated copies and merges unrotated results.
  Rotated passes are *assist-only*: stricter score floor, absolute area cap,
  and any rotated box that contains an upright-detected box of half its area
  or less is discarded (a rotated scene can hallucinate one giant
  frame-sized box — verified and gated in tests).
- **Swappable spec.** `AVPP_DETECTOR=yolox_bhhf` switches to the Apache-2.0
  434_YOLOX-Body-Head-Hand-Face model (only hand parts, no eye/nose/mouth);
  `AVPP_DETECTOR_ONNX`/`_URL` override the file or mirror.

### Witness face detection — SCRFD det_10g (`src/libs/scrfd.py`)

A standalone ONNX wrapper (no insightface package) around SCRFD, run on
**every** frame — a first-class consensus/corroboration source, not a
close-up-only fallback. `kps_plausible()` checks its 5-point landmark layout
(eye/mouth spacing, eye-axis-vs-face-axis angle, nose position along the face
axis) and rejects the scattered/degenerate layouts a skin or fabric misread
produces, *before* the gate ever sees the claim. `AVPP_SCRFD_ONNX`/`_URL`
override the model.

### Pose — RTMPose-m body7 SimCC (`src/libs/pose.py`)

Top-down pose on the frame's top-2 highest-scoring body boxes → 17 COCO
keypoints per person. Two things feed the gate:

- **Head anchor**: a coarse box from the nose/eye/ear keypoints (or, if none
  are confident, placed a head-height beyond the shoulders along the body
  axis) — a second, independent way to anatomically anchor a face claim
  besides the detector's own head class.
- **Torso axis**: the shoulder-mid → hip-mid vector. A face claim on the hip
  side of this axis is a skeleton misfit (legs read as a face) and is
  vetoed (`Ev.VETO_AXIS`); on the head side, it's confirmed (`Ev.AXIS_OK`)
  and strong enough on its own to override a hand-veto (a hand genuinely
  resting on a face, confirmed by body geometry, is not a false positive).

Pre/post-processing is a direct NumPy/OpenCV port of the standard RTMPose SDK
pipeline (bbox→center/scale, affine warp, SimCC argmax decode) — no rtmlib or
mmpose package dependency. `AVPP_POSE_ONNX`/`_URL` override the model (a
smaller `rtmpose-s` export is a drop-in swap if the exe size budget is ever
tight).

### Evidence gate — `src/libs/evidence.py`

The precision mechanism everything above feeds. `gate_face_candidates()`
merges the primary's face claims with SCRFD's landmark-checked ones
(`Ev.CONSENSUS` when they agree), then requires, for each claim:

1. **Anatomical anchoring** — inside an independent head box, or inside a
   pose-derived head region on the correct side of the torso axis. A
   face-shaped misread of skin/fabric with no independent head-class or pose
   corroboration at the same spot fails here.
2. **Part/landmark corroboration** — an eye/nose/mouth detection lands inside
   the claim, or it came from a landmark-plausible SCRFD witness, or
   confident pose facial-region keypoints agree.
3. **No unresolved negative veto** — a hand/foot detection substantially
   covering the claim is rejected, *unless* ≥2 independent part hits or a
   pose-confirmed torso axis override it (`Ev.VETO_OVERRIDDEN`) — a hand
   genuinely resting on someone's face is a real scene, not a false positive.

A claim that fails all of the above may still clear an **extreme-close-up
path** (`Ev.CLOSEUP`): when there's no body/pose context to anchor to at all
(the face fills the frame), the primary and SCRFD must mutually agree at a
size that actually defeats a whole-head detector, with part evidence behind
it — independent-model consensus substituting for missing anatomical context.

Every accepted candidate's `Ev` flags travel onto the tracker and then into
the offline evidence ledger (`summarize`/`grade`), so a track's *history* of
evidence — not just its score and length — decides whether it survives
cleanup. Three named profiles (`PROFILES["balanced"/"max"/"strict"]`) tune how
much evidence is required; the raw confidence floor (`det_conf`) never moves
for coverage — that's the whole point of separating the two knobs.

### Tracking — `src/libs/head_tracker.py`

Constant-velocity Kalman filter per track (`[cx, cy, w, h]` + velocities,
ByteTrack noise convention) with a multi-stage association cascade, all
solved by the Hungarian method (`scipy.optimize.linear_sum_assignment`).
Critically, **only gated candidates may spawn or drive a track** — heads,
bodies and poses are motion anchors and evidence, never inputs to the
association cascade directly:

1. all tracks × high-score *gated* candidates, IoU-gated;
2. **BYTE**: leftover tracks × low-score, veto-passed detections
   (sustain-only — the anti-flicker mechanism through occlusion, still never
   a sock/skin misread since the veto still applies);
3. fast-motion recovery: centre-distance + size-ratio gated (never a bare
   nearest-neighbour grab — that caused the old detection-stealing ghosts).

Lifecycle: births only from gated candidates, `min_hits` consecutive hits to
confirm (one-frame false positives never render), confirmed tracks coast on
prediction up to `max_age_s` as bridge candidates only — coasted frames are
never blurred directly. A separate weak-faces channel refreshes an
already-confirmed track's face-freshness without needing to re-clear the
gate (identity was already established; this just says "still fresh").

### Offline cleanup — `src/libs/tracklets.py`

The export's second brain, run between the two passes with the whole
timeline (and every frame's evidence flags) available. `clean_tracklets()`:

1. **trims** coasted tails (predictions that never met a detection are not
   blurred — the old static-hold ghost fix);
2. **composite prunes** — the original length/score/hit-ratio gates *and* the
   evidence ledger (`grade() != "C"`). This is what catches a static
   skin/fabric misread that reproduces every frame at high confidence (long
   *and* confident, so score-and-length gates alone can't tell it from a real
   face) but never earns real anatomical/part evidence the way a real face
   does;
3. **cross-model VERIFIES** every survivor (`verify_tracklets_xmodel`):
   sample a few hit frames, crop around the track with context, and ask an
   **independent** witness (SCRFD and/or NudeNet — never the primary
   detector that produced the track) for a face at the same spot. When no
   witness could run at all (both unavailable, or every sampled frame was
   unreadable), the tracklet's own evidence grade decides whether it
   survives unverified (`thr.verify_fail_open_grades`) — tighter than
   trusting everything, looser than requiring live verification always;
4. **bridges** gaps ≤ `bridge_gap_s`: velocity-extrapolated distance gate,
   size-ratio gate, a **corridor gate** (never bridge through a region
   another surviving track occupies) and an **ambiguity gate** (two
   near-equal candidates → bridge neither), then linear interpolation with
   mid-gap size padding;
5. **fills short face-evidence gaps** separately from head bridging — a
   detector flicker in the face channel shouldn't visibly balloon a
   face-only blur out to the head box and back;
6. **extends** each tracklet ~0.12 s at both ends (covers detector spin-up);
7. **smooths** cx/cy/w/h (both head and face channels) with zero-phase
   Savitzky-Golay — steady blur, no lag.

`apply_review()` then resolves the final render set against the review
dialog's per-track enable/disable overrides (kept tracks enabled by default,
rejected ones disabled by default, either overridable by tid), and
`build_table()` folds in any manual regions before pass 2 renders.

### Cross-model verify witness — `src/libs/nudenet.py`

NudeNet's YOLOv8n export (`320n.onnx`, 18 nudity/anatomy classes trained on
adult content) supplies `FACE_FEMALE`/`FACE_MALE` detections as the
independent witness VERIFY needs. It never runs per-frame — only during the
export's offline VERIFY step, on magnified crops of tracklets that already
survived the composite prune. Sourced from the official `nudenet` PyPI wheel
rather than the upstream GitHub release, which sits behind a login wall for
anonymous downloads and (at `640m.onnx`) is ~8× larger than the exe's size
budget allows; see the module's docstring for the full story.

### Analysis cache — `src/libs/sidecar.py`

A JSON file (`<video>.avpp.json`) next to the source, keyed on a fingerprint
of the file (size + mtime) plus every `Params` field that changes what pass 1
itself records (confidence floors, evidence profile, rotation assist —
deliberately *not* cleanup-only knobs like bridge gap or smoothing). A
re-export that only changes a cleanup knob loads the cached raw tracklets and
skips pass 1's expensive detect/pose/gate loop entirely; `clean_tracklets`
(including a fresh VERIFY pass — cheap relative to pass 1) still re-runs from
the cached data. Also carries `ReviewDecisions` (per-track enable/disable +
`ManualRegion`s) so review choices persist across re-exports of the same file.

### Review dialog — `src/review_ui.py`

A modal PyQt6 dialog between cleanup and render: every kept/rejected track
(worst grade first) with a checkbox, time range, grade and thumbnail; a
second panel to scrub the video, rubber-band a rectangle, and mark it as a
manual region's start/end keyframe (linearly interpolated between them).
Runs on the GUI thread — `ProcessWorker`'s `QThread` can't construct or exec
PyQt widgets — and blocks the worker via a `threading.Event` handshake
(`review_ready`/`take_review_request`/`submit_review_decision` in `ui.py`),
the same block-and-signal pattern already used for pause/resume.

### Masking & blur — `src/libs/utils.py`

`render_head_mask()` draws one padded axis-aligned ellipse per blur-target
box and feathers the whole mask with a single Gaussian — one consistent
shape every frame (no hull/ellipse popping), pre-grown by the feather radius
so feathering never shrinks coverage. `BlurPipeline` applies the configurable
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
DirectML survival rules shape the code, applied identically to all four
models (detector, SCRFD, pose, NudeNet):

1. **Never destroy a session.** Destroying one corrupts the provider's device
   state; the next inference dies with a native access violation. Each
   model's session is created lazily, exactly once, per process.
2. **Pin graph shapes.** DirectML validates strictly; every model's input is
   fixed to its working resolution (`make_input_shape_fixed`) before the
   session is built.

fp16 conversion (`fp16_model_path`, ~2× on RDNA2) exists but every model ships
fp32 first — fp16 across an embedded-NMS/DFL partition boundary is a known
risk; `AVPP_FP16=0` is the kill switch.

### GUI — PyQt6 (`src/ui.py`, `src/review_ui.py`)

Three-panel inspector (Before / Tracking / After), preset chips + advanced
sliders (including the Evidence profile combo and Face Hold slider), live
preview on a worker thread (latest-wins mailbox), and the two-pass export
with pass-aware progress. Detection/tracking sliders drive pass 1 and the
preview; mask padding, feather and blur layers stay live even during pass 2.
Between the two passes, the review dialog (`review_ui.py`) blocks the export
worker until the user enables/disables tracks and/or adds manual regions.

---

## Packaging

`build_exe.sh` (WSL2 → Windows Python interop → PyInstaller onefile/windowed):

- installs the trimmed dependency set (PyQt6, scipy, opencv-python, numpy,
  onnx, onnxconverter-common, imageio-ffmpeg);
- force-installs **onnxruntime-directml last** so no CPU-only wheel can
  clobber it, and asserts `DmlExecutionProvider` before building;
- **bundles all four ONNX models** (`--add-data … models`, ~82 MB combined)
  so the .exe's first run downloads nothing — each model's `model_path()`
  checks `sys._MEIPASS/models/` first, then its `~/.cache/avpp/<name>/`, then
  env overrides.

## Environment variables

| Variable | Effect |
| --- | --- |
| `AVPP_DETECTOR` | Spec name: `wholebody17` (default) or `yolox_bhhf` |
| `AVPP_DETECTOR_ONNX` / `AVPP_DETECTOR_URL` | Local path / mirror URL for the detector |
| `AVPP_SCRFD_ONNX` / `AVPP_SCRFD_URL` | Local path / mirror URL for the SCRFD witness |
| `AVPP_POSE_ONNX` / `AVPP_POSE_URL` | Local path / mirror URL for the pose estimator |
| `AVPP_NUDENET_ONNX` / `AVPP_NUDENET_URL` | Local path / mirror URL for the NudeNet verify witness |
| `AVPP_FP16` | `0` disables the fp16 model derivative |
| `AVPP_SKIP_MODEL_DOWNLOAD` | Preflight checks locations only |

## File organization

```
src/
  main.py            entry point; splash → GUI
  splash.py           loading splash
  ui.py               PyQt6 inspector + two-pass export worker
  review_ui.py        review dialog: track enable/disable, manual regions
  libs/
    detector.py       ONNX detector — body/head/face/parts/hand-foot, rotation assist
    evidence.py       face-candidate acceptance gate + offline evidence ledger
    scrfd.py          landmark-checked witness face detector (every frame)
    pose.py           RTMPose body estimator — anatomical anchor + torso axis
    nudenet.py        independent, offline-only cross-model verify witness
    sidecar.py        analysis cache + review-decision persistence
    head_tracker.py   Kalman + BYTE + Hungarian tracker
    tracklets.py      pass-1 recorder + offline cleanup → render table
    models.py         startup preflight (bundle/cache/download, all 4 models)
    utils.py          providers, fp16, feathered masks, GPU blur stack
    video_writer.py   streaming ffmpeg exporter (>4 GiB safe, audio copy)
```

## Debug artifacts (Windows)

- `%TEMP%\FaceBlurInspector-debug.log` — status lines, resolved providers,
  per-stage timings every 30 frames (`[analyse f…]` / `[render f…]`), and the
  between-pass tracklet summary (`N raw tracklets → M kept (grades …), K
  rejected — J enabled after review, R manual region(s) — blur on X/Y
  frames`).
- `%TEMP%\FaceBlurInspector-error.log` — tracebacks + faulthandler dumps.
