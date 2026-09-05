# Technology Stack & Architecture

## Overview

```
video ──► decode ──► [head det ×4 rot | Wholebody ×4 rot | SCRFD ×2 rot] ──► fuse ──► track ──► sidecar
                                                                                           │
          blurred video ◄── render ◄── table ◄── review ◄── score ◄── refine ◄── retrack ◄──┘
```

Two tiers: detection is expensive and cached; everything downstream is pure
numpy and re-runs from the cache in seconds. The GUI and the CLI share one
orchestration module (`pipeline/session.py`).

## Components and why they were chosen

### The core problem: recall on bodies at every angle

A face detector's confidence cannot separate "face" from "face-shaped skin",
and on footage with bodies at every angle it also cannot *find* a face that is
in profile, upside down, under hair or turned away. The previous design
answered the first problem with a per-frame AND-gate (anchor + facial part +
no hand/foot veto) and paid for it with the second: measured against an
independent consensus on the reference clip it covered **69 % of heads** —
exactly the heads without a visible face were the ones it dropped.

This pipeline inverts the loss. The unit of blur is the **head** (an
angle-invariant blob), any source may propose one, and precision comes from
time and from ranking rather than from per-frame proof.

### Detector bake-off (reference clip, 314 sampled frames, 357 consensus heads)

Consensus = a head box two model families agree on, or one with a
landmark-plausible SCRFD face inside. Recall is against that set; "lonely"
counts boxes no other source supports (a false-positive proxy).

| Source | Recall, upright only | Recall, all 4 rotations | Rotation-only finds | Lonely boxes | CPU ms/call |
| --- | --- | --- | --- | --- | --- |
| Wholebody17-S head class | 63.6 % | **93.6 %** | 107 | 266 | 36 |
| deepghs YOLO11-M head | 54.1 % | 75.9 % | 78 | 16 | 65 |
| deepghs YOLO11-L head | 64.2 % | 83.8 % | 70 | 22 | 78 |

Conclusions baked into the defaults:

- **Rotation is not optional** on this footage: +30 points for Wholebody, +20
  for the YOLO11 models. Both head sources run at 0/90/180/270 in every
  preset except *Fast*.
- **No single source is enough**; the union is. Wholebody has the best
  recall but hallucinates heads on torsos and limbs (the lonely column);
  YOLO11-L is tight. Both are kept; L beats M for a small cost.
- **Wholebody's hallucinations are shaped, not sized.** Real heads here are
  large (median 10 % of the frame, 95th percentile 40 %) so area separates
  little, but unsupported boxes run to 3:1 aspect and beyond (75th percentile
  3.3 vs 1.4 for real heads) and 85 % of them come from rotated passes. Hence
  the fusion trust rules: a single-source claim is scaled by its source's
  measured trust (Wholebody ×0.7), then ×0.5 outside 0.45–2.2 aspect, ×0.7
  when rotated-only, ×0.5 over 30 % of the frame — it may still sustain a
  track, it can no longer start one.
- **A face inside a box does not make it a head.** The first full run
  blurred torsos and whole scenes: a Wholebody torso box containing the
  person's face counted as face-corroborated, and a big weak box overlapping
  a head track by IoU 0.25 could "sustain" it and inflate the Kalman box.
  Now a face validates a box only while the box is under 8× the face's
  area, containment merging stops at a 1.8× area ratio, and both tracker
  association stages carry size gates (0.5–2× and 0.6–1.7×); heads define
  geometry and faces only corroborate. Blurred area on the clip fell from
  23 % to 16 % of the frame with head coverage at 93 % (see the README).
- **SCRFD at 1280** finds 18 % more confident faces than at 640 for 3.2× the
  cost; 180° adds 8 %. 640 + 0/180 is the default, 1280 in *Max Privacy*.

### Head detector — deepghs real_head_detection (`src/libs/headdet.py`)

YOLO11 single-class head model published as ONNX with a raw `(1, 5, N)`
output (cx, cy, w, h, score; no NMS). Letterboxed top-left onto a 640 grey
canvas so the inverse is one divide; decoded and NMS'd in numpy. Variant via
`AVPP_HEADDET` (`head_detect_v0_{n,s,m,l}_yv11`), default L.

### Second head vote — PINTO Wholebody17 (`src/libs/detector.py`)

Kept from the previous design for its head and face classes (its parts and
hand/foot classes are no longer consulted). Called once per rotation with
`rotations=(0,)` so its own rotated-pass hallucination gates are bypassed —
the union layer's trust rules and the offline prune replace them.

### Faces — SCRFD det_10g (`src/libs/scrfd.py`)

Anchor-free FPN face detector with 5-point landmarks. The landmark
plausibility check (`kps_plausible`) kills the degenerate layouts that skin
and fabric misreads produce. Faces feed fusion as head-sized claims and the
tracker's per-track face memory (face-tight blur mode).

### Fusion — `src/pipeline/fuse.py`

Heads define geometry; faces corroborate. Head claims cluster greedily
(score-weighted box mean); each face attaches to the head cluster containing
its centre when that cluster is under 8× its area, contributing its source
bit and — only if it comes from another model family — `FACE_IN_HEAD`. A
face never moves a cluster's box: the second full run showed that faces
grown to "head scale" with the legacy ×1.7/×1.9 factor are *larger* than a
close-up head and spawned torso-sized duplicates (a 9 % face became a 33 %
"head"). Orphan faces become modest head claims (×1.35/×1.55). Agreement is
counted per model family — head detector, Wholebody (its head and face
classes are one network), SCRFD — so one network's rotated-pass mistake no
longer earns the bonus. NudeNet's non-face classes act as a per-frame
**body-part veto**: a candidate that coincides with one is scaled ×0.3 and
flagged, whatever the head models said, because on this footage all three
fire together on a harness-covered groin and only a model trained on adult
content tells the difference. Fusion runs again offline from the cached
per-source detections (`analysis.retrack`), so every constant here can be
tuned without re-detecting.

### Tracking — `src/pipeline/tracker.py`

Constant-velocity Kalman over `[cx, cy, w, h]` with ByteTrack-style noise
scaling; three association stages (Hungarian on IoU, BYTE sustain from weak
candidates, centre-distance rescue for fast motion), each IoU stage with a
size gate so a containing box cannot hijack a head track; spawn at 0.35,
sustain from 0.10, confirm at two hits, coast 2.5 s. Tracker knobs are
excluded from the sidecar fingerprint: `analysis.retrack` re-fuses and
re-tracks from cached detections whenever the offline stages run.

### Refinement — `src/pipeline/refine.py`

Soft prune (only tracklets with no temporal support are set aside, and they
stay in the review list), bridging with a velocity-extrapolated landing
point, size ratio, corridor and ambiguity gates and an ArcFace veto,
face-gap fill, end extension, zero-phase Savitzky–Golay smoothing, and
upsampling from step space with `INTERP` flags on every non-measured frame.

### Scoring — `src/pipeline/score.py`

Eight components, each 0–1, weighted mean. `lonely` (no second source ever)
carries the most weight because it is the single strongest tell for a skin
misread; `unverified` comes from re-running SCRFD and the head detector on
magnified crops of five hit frames. Nothing here deletes: `kept` is a default
verdict, one key to flip.

### Rendering — `src/pipeline/render.py`, `src/libs/utils.py`

Per-frame boxes grow along their motion (`motion_lead ×` centre displacement)
so smoothing lag never leaves a fast head half-blurred; the feathered-ellipse
mask, ROI-limited feathering, OpenCL/CUDA blur stack and ffmpeg writer are
unchanged from the previous design.

### Manual propagation — `src/pipeline/propagate.py`

A drawn box snaps to cached fused candidates near its prediction, falls back
to normalised cross-correlation template matching (original + latest
template), holds up to six frames, then stops. Backward runs decode in
chunks so nothing seeks per frame.

### Body-part veto — `src/libs/nudenet.py`

YOLOv8n, 320 px, 18 classes trained on adult content. Its face classes were
once a cross-model verify witness; now its *non-face* classes
(genitalia, buttocks, belly, breasts, feet, armpits — exposed or covered)
are a per-frame source whose boxes veto head candidates at fusion and feed
the `bodypart` suspicion component. One pass per analysed frame, ~15 ms on
CPU. `use_bodypart=False` in `AnalysisConfig` switches it off. AGPL-3.0 —
see the README licence note.

### Appearance channel — `src/libs/embed.py`

ArcFace MobileFaceNet, 512-d. Wired as a **veto only** (`DIFF_ID_COS`): low
similarity blocks a bridge, high similarity never buys one; missing vectors
mean "no opinion". `scripts/calibrate_embed.py` measures same-track vs
cross-track cosine distributions on your footage.

### Review-decision corpus — `src/libs/labels.py`

One JSON line per reviewed track: suspicion components, geometry, motion,
the verifier's ratio and the user's verdict. No imagery, no paths; video ids
are keyed BLAKE2 hashes under a machine-local `0600` salt.

### GUI — PyQt6 (`src/app/`)

Timeline editor. `PipelineWorker` (QThread) owns every model session for the
process lifetime and runs analyse / offline / export / propagate jobs; the
main window keeps the project, the scored tracks and the blur table and
rebuilds table → lanes → alerts on each edit. Split/trim edits are recorded
in the project and replayed after any offline re-run.

### Runtime — ONNX Runtime

`best_onnx_providers()` resolves **DirectML → CUDA → MIGraphX → ROCm → CPU**.
MIGraphX is listed ahead of the plain ROCm EP because AMD ships
`onnxruntime_migraphx` rather than `onnxruntime_rocm` from ROCm 7.1 onward,
and because it compiles and fuses the graph ahead of time — closer to
TensorRT than to a per-op dispatcher, which matters at the batch-1 shapes this
pipeline runs. `GPU_PROVIDERS` names the compute providers explicitly rather
than testing "not CPU", since ORT's stock CPU wheel also advertises
`AzureExecutionProvider` and treating that as a GPU would silently enable the
fp16 path on a CPU-only box.

**Advertised ≠ usable.** ORT lists an EP whenever its plugin ships in the
wheel, without checking that the plugin's dependencies resolve — routinely
false on ROCm, where the HIP runtime installs under `/opt/rocm` (a symlink to
the newest tree) while repo.radeon.com's math libraries land in a different
version-suffixed tree that nothing puts on the linker path. MIGraphX then
lists, fails at session creation with `libmigraphx_c.so.3: cannot open shared
object file`, and ORT falls back to CPU — after `fp16_model_path()` has
already converted the graph on the strength of a GPU being "available", so the
fallback runs an **fp16 graph on the CPU**: 71 ms/pass against 26 ms for plain
fp32. A broken GPU provider was 2.7× worse than no GPU at all.
`_preload_gpu_runtime()` dlopens ROCm's math libraries with `RTLD_GLOBAL`
(a process cannot set its own `LD_LIBRARY_PATH` — the loader reads it at exec)
and `_provider_usable()` drops any EP whose plugin will not load.

**MIGraphX compiles ahead of time**, which is where its speed comes from and
why a cold session costs ~25 s per model — minutes of startup across five
models, every launch. `_configure_migraphx_cache()` points
`ORT_MIGRAPHX_MODEL_CACHE_PATH` at `~/.cache/avpp/migraphx` (~30 MB/model),
taking that to ~0.8 s. It lives beside the models rather than in
`DERIVED_MODEL_CACHE` under the system temp dir, because the fp16 and
shape-pinned derivatives kept there regenerate in seconds while these do not.

**Session churn crashes it.** The "never destroy a session" rule is not
DirectML-specific: building and dropping sessions across configurations
aborts the MIGraphX EP with `HIP failure 700: an illegal memory access was
encountered` — a process abort, not a degrade. `scripts/benchmark.py` builds
one set of models and reuses it for exactly this reason.

Measured on the reference RX 6800 (ROCm 7.2.3, MIGraphX, 1080p): detector
26.2 → 9.0 ms/pass, SCRFD 35 → 4.5 ms. End to end, pass 1 goes from 133 ms to
32.6 ms per frame at default settings, or 4.6 ms with the Fast preset.

On the CPU path the intra-op pool is pinned to *physical* cores
(`_cpu_threads()`); ORT's default of one thread per logical CPU oversubscribes
SMT pairs and measured ~1.4× slower on these graphs (36.8 ms vs 26.2 ms per
detector pass on a 6-core/12-thread machine). `AVPP_CPU_THREADS` overrides. Two
DirectML survival rules shape the code, applied identically to every
model wrapper (head detector, Wholebody, SCRFD, NudeNet, embedder):

1. **Never destroy a session.** Destroying one corrupts the provider's device
   state; the next inference dies with a native access violation. Each
   model's session is created lazily, exactly once, per process.
2. **Pin graph shapes.** DirectML validates strictly; every model's input is
   fixed to its working resolution (`make_input_shape_fixed`) before the
   session is built.

fp16 conversion (`fp16_model_path`, ~2× on RDNA2) exists but every model ships
fp32 first — fp16 across an embedded-NMS/DFL partition boundary is a known
risk; `AVPP_FP16=0` is the kill switch.


---

## Packaging

`build_exe.sh` (WSL2 → Windows Python interop → PyInstaller onefile/windowed):

- installs the trimmed dependency set (PyQt6, scipy, opencv-python, numpy,
  onnx, onnxconverter-common, imageio-ffmpeg);
- force-installs **onnxruntime-directml last** so no CPU-only wheel can
  clobber it, and asserts `DmlExecutionProvider` before building;
- **bundles all five ONNX models** (`--add-data … models`, ~172 MB combined)
  so the .exe's first run downloads nothing — each wrapper's `model_path()`
  checks `sys._MEIPASS/models/` first, then `~/.cache/avpp/<name>/`, then
  env overrides;
- collects the `pipeline` and `app` packages with `--collect-submodules`.

## Environment variables

| Variable | Effect |
| --- | --- |
| `AVPP_HEADDET` | Head detector variant: `head_detect_v0_{n,s,m,l}_yv11` (default `l`) |
| `AVPP_HEADDET_ONNX` / `AVPP_HEADDET_URL` | Local path / mirror URL for the head detector |
| `AVPP_DETECTOR` | Wholebody spec: `wholebody17` (default) or `yolox_bhhf` |
| `AVPP_DETECTOR_ONNX` / `AVPP_DETECTOR_URL` | Local path / mirror URL for the Wholebody detector |
| `AVPP_SCRFD_ONNX` / `AVPP_SCRFD_URL` | Local path / mirror URL for SCRFD |
| `AVPP_NUDENET_ONNX` / `AVPP_NUDENET_URL` | Local path / mirror URL for the NudeNet witness |
| `AVPP_EMBED_ONNX` / `AVPP_EMBED_URL` | Local path / mirror URL for the face embedder |
| `AVPP_EMBED` | `0` disables the appearance channel (bridging becomes geometry-only) |
| `AVPP_LABELS` | `0` disables review-decision label harvesting |
| `AVPP_FP16` | `0` disables the fp16 model derivative |
| `AVPP_CPU_THREADS` | Override the CPU intra-op thread count (default: physical cores) |
| `AVPP_SKIP_MODEL_DOWNLOAD` | Preflight checks locations only |
| `AVPP_SKIP_PREFLIGHT` | `1` skips the startup model check entirely (headless testing) |
| `ORT_MIGRAPHX_MODEL_CACHE_PATH` | MIGraphX compiled-graph cache (default `~/.cache/avpp/migraphx`) |

## File organization

See the project structure in the README. On disk at runtime:

| Artifact | Path |
| --- | --- |
| Project sidecar (detections, tracklets, decisions, manual heads, edits) | `<video>.avpp2.json` |
| Model cache | `~/.cache/avpp/{headdet,detector,scrfd,nudenet,embed}/` |
| MIGraphX compiled graphs | `~/.cache/avpp/migraphx/` |
| fp16 / shape-pinned derivatives | `$TMPDIR/FaceBlurInspector-models/` |
| Label corpus + salt (`0600`) | `~/.cache/avpp/labels.jsonl`, `labels.salt` |
| Debug / crash logs | `$TMPDIR/FaceBlurInspector-debug.log`, `-error.log` |

## Debug artifacts (Windows)

The frozen build is `--windowed`, so `print()` goes nowhere: `utils.debug_log`
mirrors every status line to `%TEMP%\FaceBlurInspector-debug.log`, and
`sys.excepthook` / `threading.excepthook` / `faulthandler` write tracebacks to
`%TEMP%\FaceBlurInspector-error.log`.
