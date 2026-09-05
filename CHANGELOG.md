# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — pipeline and UI rewrite (recall-first)

The evidence-gated pipeline is replaced end to end. Measured on the reference
clip against an independent all-detector consensus (342 corroborated heads
on every 20th frame), the previous design covered **70 %** of heads and
**93 %** of faces with 30 blur on/off transitions; the new *Balanced* preset
covers **93 %** and **98 %** with 14, *Max Privacy* **96 %** and **98 %**
with 6. The old gate required a visible facial part on every frame, so heads
turned away, in profile or under hair were never blurred, and its face-only
blur timed out while a head turned. The price is precision: blurred area
rises from 8 % to 16 % of the frame (real heads on this footage are large and
now all covered) and 95 of 547 sampled blur boxes have no detector support
(held through occlusion, or false tracks the review timeline ranks by
suspicion). See `docs/TECH_STACK.md` for the bake-off behind every number.

- **Union detection.** Three sources vote on every analysed frame — a new
  deepghs YOLO11-L head detector (`libs/headdet.py`, the primary source),
  the Wholebody17 head and face classes, and landmark-checked SCRFD faces —
  and both head sources run on 0°/90°/180°/270° copies (Wholebody head
  recall 64 % → 94 % on the clip). `pipeline/fuse.py` clusters the claims,
  rewards agreement and *down-weights* unsupported claims by source trust
  and by how little they look like a head (aspect outside 0.45–2.2,
  rotated-only, frame-filling) instead of gating them; a face inside a box
  corroborates it only while the box is under 8× the face's area; nothing
  is dropped per frame except a rotated-only, single-source box over 30 % of
  the frame. Both tracker association stages carry size gates so a
  torso-sized weak box cannot inflate a head track. Heads define geometry
  and faces only corroborate (a face grown to head scale is larger than a
  close-up head and spawned torso-sized duplicates); agreement is counted
  per model family; and NudeNet's non-face classes flag candidates that
  coincide with a body part for the suspicion score (a spawn veto in the
  *Strict* preset only — measured from cache, vetoing everywhere cost three
  points of head coverage for a dozen fewer unsupported boxes).
- **Heads are the unit of blur.** Face mode is optional and falls back to
  the head box, never to nothing.
- **Long-memory tracking + offline refinement** (`pipeline/tracker.py`,
  `pipeline/refine.py`): spawn at 0.35, sustain from 0.10, coast 2.5 s;
  offline soft prune (set aside, never deleted), bridging up to 2.5 s with
  an appearance veto, 0.3 s end holds, zero-phase smoothing, upsampling with
  `INTERP` provenance.
- **Suspicion scoring replaces verification-as-deletion**
  (`pipeline/score.py`): eight components (lonely, unverified, weak, short,
  sparse, static, rotated, oversized) rank tracks for review; only the
  *Strict* preset auto-disables aggressively, *Max Privacy* never does.
- **Detection cached, everything else cheap.** The sidecar
  (`<video>.avpp2.json`, `pipeline/project.py`) stores per-source detections
  per analysed frame; fusion and tracker knobs re-run from it
  (`analysis.retrack`) in seconds and are excluded from the fingerprint.
- **Timeline editor** (`src/app/`) replaces the three-panel inspector and
  the modal review dialog: suspicion-sorted lanes, an alert strip marking
  confident detections with no blur, live blur preview, split/trim at the
  playhead, and drawn boxes that **follow the head** both ways
  (`pipeline/propagate.py`) instead of interpolating between two keyframes.
- **Velocity-aware masks**: boxes grow along their per-frame motion so fast
  heads never outrun the blur.
- **Headless CLI** (`src/cli.py`): analyse / inspect / export / dump.

### Removed

- `libs/evidence.py` (gate + ledger), `libs/head_tracker.py`,
  `libs/tracklets.py`, `libs/pose.py` (RTMPose is no longer used),
  `libs/sidecar.py` (schema 1), `src/ui.py`, `src/review_ui.py`, and their
  tests. NudeNet is no longer a face witness; its body-part classes are a
  default fusion source (`use_bodypart`).

### Earlier unreleased work (pre-rewrite)

The entries below predate the rewrite. Those about providers, the MIGraphX
cache, CPU threads, the ROI-limited mask feather, the embedder and the label
corpus still apply; those about the evidence gate, pose, `tracklets.py` or
the review dialog describe code that has since been replaced.


### Performance

Pass 1 dominates export time: six ONNX calls per frame (three detector passes
with rotation assist, SCRFD, up to two pose), none batchable — the published
detector graph has its batch dimension baked to 1, so batching the rotation
passes is not available. Measured end to end on the reference RX 6800 box,
CPU path, 1080p: **133 ms/frame → 27 ms/frame (4.9×)**.

- **MIGraphX execution provider.** `best_onnx_providers()` now resolves
  DirectML → CUDA → **MIGraphX** → ROCm → CPU. AMD ships `onnxruntime_migraphx`
  rather than `onnxruntime_rocm` from ROCm 7.1 onward, so a Linux AMD box had
  no GPU path at all through the pinned CPU-only wheel. Verified end to end on
  an RX 6800 (gfx1030, ROCm 7.2.3): all five models run on MIGraphX, detector
  26.2 → 9.0 ms/pass, SCRFD 35 → 4.5 ms, pass 1 **133 → 32.6 ms/frame** at
  default settings and **4.6 ms/frame** with the Fast preset — **29×** against
  the CPU default.
- **Providers are verified, not trusted.** ORT advertises an EP whenever its
  plugin ships in the wheel, without checking the plugin's own dependencies
  resolve — which routinely fails on ROCm, where the HIP runtime and the math
  libraries install into different trees and only one is on the linker path.
  MIGraphX then listed, died at session creation, and ORT fell back to CPU
  *after* `fp16_model_path()` had converted the graph — running an fp16 graph
  on the CPU at 71 ms/pass against 26 ms for plain fp32. A broken GPU provider
  was 2.7× worse than no GPU. `_preload_gpu_runtime()` now dlopens the ROCm
  math libraries with `RTLD_GLOBAL` (a process can't set its own
  `LD_LIBRARY_PATH`) and `_provider_usable()` drops any EP whose plugin won't
  load.
- **MIGraphX compiled-model cache.** MIGraphX compiles ahead of time: ~25 s
  per model cold, so five models meant minutes of startup on every launch.
  Cached under `~/.cache/avpp/migraphx` (~30 MB/model), which takes that to
  ~0.8 s. Kept beside the models rather than in the temp-dir derived-model
  cache, since fp16/shape-pinned derivatives regenerate in seconds and these
  do not.
- **CPU thread pool pinned to physical cores.** ORT defaults to one intra-op
  thread per *logical* CPU, which oversubscribes SMT pairs; on a 6-core/12-
  thread machine that measured 36.8 ms vs 26.2 ms per detector pass — the
  default was **1.4× slower**. `AVPP_CPU_THREADS` overrides.
- **Analysis stride** (`Params.analysis_stride`, "Analyse every N frames").
  Runs the model chain on every Nth frame; the tracker and every cleanup stage
  run in step space at `fps / N` so all second-based knobs keep their meaning,
  and `upsample_tracklets` maps the result back onto every frame. Interpolated
  frames are never counted as detector hits, and face-only blur stays off
  across a span unless face evidence was fresh at both ends. Striding the
  recorder directly would not have worked: `TrackRecorder.observe` pads unseen
  frames by repeating the previous box, manufacturing the static-hold ghosts
  this pipeline exists to eliminate and pushing hit-ratio under the prune
  threshold at stride ≥ 4.
- **New "Fast" preset** — stride 3, rotation assist off, wider bridging to
  compensate. Same evidence bar, so precision is unchanged; the trade is
  recall on sideways heads and on anything entering and leaving within ~3
  frames.
- **`render_head_mask` feathers only the ROI the ellipses touch**, grown by a
  full kernel width so the result is bit-identical to blurring the whole frame
  (asserted over 72 cases, including off-frame and frame-filling heads). 4K,
  one head: 3.41 ms → 0.44 ms.
- **`BlurPipeline.apply` uses `cv2.boundingRect`** instead of `np.nonzero`,
  which materialised two int64 arrays holding every set pixel's coordinate —
  tens of MB per frame on a 4K feathered mask — to take four min/max values
  off them. `_is_soft` likewise uses one `cv2.inRange` pass instead of three
  full-size bool temporaries.
- **Bridge candidate gates vectorised.** Gap/distance/size are evaluated as
  whole `(N, N)` matrices and the corridor test is vectorised across the gap's
  frames; only pairs surviving the cheap gates pay for it. Byte-identical
  output, 3–5× faster (183 ms → 36 ms on an adversarial 200-tracklet case).
  Note this stage was *not* a significant cost — a few ms at realistic
  tracklet counts.

### Added

- **Appearance channel** (`libs/embed.py`) — ArcFace MobileFaceNet embeddings,
  512-d, bundled (~14 MB). Nothing in the pipeline reasoned about identity:
  the tracker associates on IoU/distance/size and the bridge on velocity plus
  a corridor test, which is exactly why `_bridge_candidates` needs an
  ambiguity gate. Wired in as a **one-directional veto** — a low cosine blocks
  a join geometry proposed, a high one never buys a join geometry rejected —
  so an uncalibrated or uninformative embedder can only ever make cleanup more
  conservative, never less. `scripts/calibrate_embed.py` measures the
  same-track vs cross-track similarity distributions on real footage so the
  threshold can be set on evidence rather than a guess. `AVPP_EMBED=0`
  disables.
- **Review-decision corpus** (`libs/labels.py`) — every review checkbox is a
  human judgement on a tracklet, and nothing was recording them. One JSONL row
  per reviewed tracklet now accumulates in `~/.cache/avpp/labels.jsonl`: the
  dataset a learned replacement for `evidence.py`'s ~60 hand-tuned constants
  needs. **No imagery, no paths, no filenames** — videos appear only as a
  keyed BLAKE2 hash of size+mtime under a per-machine 256-bit salt stored
  `0600`; an unkeyed digest could be recomputed from any candidate file.
  `AVPP_LABELS=0` disables. `tests/test_labels.py` asserts these properties.
- **`scripts/benchmark.py`** — per-stage pass-1 timing on a real clip across
  stride and rotation settings, without running a full export.

### Fixed

- **Face bloom was applied in the preview but not the export.**
  `bloom_face_box` (face + hair fringe) ran only in `_blur_target_box`, which
  the live preview uses; pass 2 fed `render_head_mask` the raw face box. The
  exported blur therefore covered **less** hair than the AFTER panel showed —
  the wrong direction for a privacy tool, and the opposite of the documented
  "the export is strictly better than the preview". Now applied at render
  time, so the table keeps raw geometry for the offline smoother and both
  paths agree. Manual regions are drawn at the size the user chose and are not
  bloomed.
- **The per-stage timing log only reported the detector.** SCRFD, pose and the
  non-model remainder were never logged despite every model exposing
  `last_ms`, and in pass 1 the `blur` field is always 0 — so the line showed
  `detect` and `frame` with an unexplained gap between them.
  `self._last_detect_ms` was declared and never assigned. All stages are now
  logged plus an `other` remainder, so the frame total is fully accounted for.
- **`_pinned_model`'s cache key omitted a file fingerprint.** Overriding the
  detector via `AVPP_DETECTOR_ONNX` with a different file of the same basename
  silently reused the previously pinned graph — you would benchmark the old
  model and never find out. Now keyed on size+mtime, matching
  `fp16_model_path`.
- **`has_gpu_provider()` treated any non-CPU provider as a GPU.** ORT's stock
  CPU wheel advertises `AzureExecutionProvider`, so on a CPU-only box the fp16
  path could be enabled. Now checks an explicit `GPU_PROVIDERS` set.
- **`test_pipeline_win.py` did not import.** It pulled `fuse_heads` from
  `libs.detector`, removed in the single-detector rewrite — the one script
  that prints timings failed at import. Rewritten against the current chain
  (detect → SCRFD → pose → evidence gate → track) and extended to cover the
  embedder.

### Documentation

- Documented that **session churn crashes the MIGraphX EP** — `HIP failure
  700: an illegal memory access was encountered`, a process abort rather than
  a degrade. The "never destroy a session" rule was recorded as
  DirectML-specific; it is not. `scripts/benchmark.py` builds one set of
  models and reuses it across configurations for this reason.
- README's ROCm instructions use `--find-links`, not `--index-url`:
  repo.radeon.com serves a wheel directory, not a PEP 503 index, and
  `--index-url` fails with "not found in the package registry".

- Corrected drift found while working: `scrfd.py` still described itself as a
  close-up *fallback* that "costs nothing on frames the primary handles"
  (it has run on every frame since the witness promotion); `evidence.py`
  described pose as not yet wired in; `utils.make_session` referenced RTMW and
  RF-DETR, neither of which exists here — **no DETR has ever been in this
  codebase**, contrary to what that docstring implied; `main.py` listed
  torch/insightface as the slow imports, neither being a dependency.
- Recorded why SCRFD's input cannot be shrunk to the source aspect ratio,
  which would otherwise be a free ~1.67× on that stage: `det_10g` declares
  both spatial dims under one symbolic name (so the graph can only be square)
  and hard-codes its output row counts to the 640×640 anchor totals. Measured
  and reverted rather than left as a plausible-looking idea.


### Fixed — false-positive blurs on Max Privacy

- **Rotation-assist hallucinations with no upright witness.** The ±90° recall
  passes could hallucinate "heads" — frame-spanning or bed-sized — on scenes
  with *no real head in frame*; the containment gate only fired when the
  upright pass had found one, so on empty scenes the fake sailed through,
  formed a confident static track, survived every offline gate and blurred
  the furniture. Two gates in `detector._filter_rotated` now close this: an
  absolute area cap (rotated-only heads > 20 % of the frame are fake), and a
  **witness rule** — the head class is 360°-trained, so a real sideways head
  never leaves the upright pass completely blind at its spot; a rotated box
  with no upright corroboration (weak head/face overlap or a body containing
  its centre) must clear 0.75 instead of 0.50 to survive.
- **Tracklet verification by cropped re-inference**
  (`tracklets.verify_tracklets`). Between the export passes, every tracklet
  that survives the prune must *reproduce*: up to 5 of its hit frames are
  re-read, cropped around its box with context, and re-detected. A real head
  re-detects stronger when magnified; a persistent hallucination (long *and*
  confident, so score/length gates can't touch it) doesn't reproduce and is
  dropped before rendering. Fail-open: unreadable frames or a downed detector
  never remove a blur.

### Added — extreme-close-up recall assist

- **SCRFD close-up fallback** (`src/libs/scrfd.py`). When the primary
  detector finds no confident head or face — the extreme-close-up signature —
  a standalone SCRFD det_10g wrapper (no insightface package; same
  one-session-never-destroyed DirectML rules) supplies face boxes that grow
  pseudo-heads. Bundled into the exe (~17 MB), preflighted at startup,
  overridable via `AVPP_SCRFD_ONNX` / `AVPP_SCRFD_URL`, and it degrades to
  off on any failure. Gated to its actual mission (`closeup_filter`): only
  faces at close-up scale (longest side ≥ 25 % of the frame's short side)
  that clear the user's confidence bar are accepted — SCRFD misreading bare
  skin or blanket texture as a face is the failure that got insightface
  dropped from this project once already, and the fallback state holds on
  every frame of head-free footage.

---

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
