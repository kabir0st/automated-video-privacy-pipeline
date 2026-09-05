# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Recall-first head anonymisation for video: union detection → long-memory tracking → offline refinement → suspicion scoring → timeline review → blur. Fully on-device. Two entry points: the PyQt6 editor (`src/main.py`) and a headless CLI (`src/cli.py`). There is no runtime config file; configuration is the `Preset` bundles in `src/pipeline/presets.py` plus `AVPP_*` environment variables (table in `docs/TECH_STACK.md`).

## Commands

Package manager is `uv` (Python 3.12). The project is not installable as a package: entry scripts and every test file insert `src/` into `sys.path` and import `from pipeline.X` / `from libs.X` / `from app.X` (never `from src....`). No `conftest.py`, no pytest config.

```bash
uv sync                                          # install (CPU-only onnxruntime wheel)
uv run python src/main.py [CLIP.mp4]             # editor
uv run python src/cli.py analyse CLIP.mp4 --preset Fast --frames 900   # headless pass 1 + scoring
uv run python src/cli.py export CLIP.mp4 OUT.mp4 # render honouring saved decisions
uv run pytest tests/                             # all tests: synthetic, fast, no models/GPU/network
uv run pytest tests/test_pipeline_refine.py::TestBridge::test_joins_across_gap_with_interp_flags
uv run pytest tests/ -k fuse
```

Always pass `tests/` explicitly: `test_pipeline_win.py` at the repo root matches `test_*.py` and would load every model on import. No linter, formatter, type-checker, or CI is configured.

Other scripts (need real weights and a real video):
- `uv run python scripts/benchmark.py CLIP.mp4 [--stride 1 2 3] [--preset Fast]` — per-stage pass-1 timings.
- `uv run python scripts/calibrate_embed.py CLIP.mp4` — needs an existing `CLIP.mp4.avpp2.json`.
- `QT_QPA_PLATFORM=offscreen uv run python scripts/capture_screenshots.py CLIP.mp4` — drives the editor headless. The same trick (offscreen platform, `AVPP_SKIP_PREFLIGHT=1`, `MainWindow.open_video`, pump `processEvents` while `win.busy`, `win.grab()`) is how to verify UI changes without a display.
- `./build_exe.sh` — Windows exe from WSL2; `py.exe -3.12 test_pipeline_win.py [video]` — DirectML smoke.

GPU providers are opt-in by swapping the onnxruntime wheel (see README).

## Architecture

Read `docs/DATAFLOW.md` first, then `docs/TECH_STACK.md` for the measurements behind the constants.

**Thesis.** On this footage a missed head is a harm and a spurious blur is cosmetic. So `pipeline/fuse.py` unions every detector's output (any single source may propose; agreement adds an `agree_bonus`; unsupported claims that look unlike a head — aspect outside 0.45–2.2, rotated-only, frame-filling — are *down-weighted*, never dropped), `pipeline/tracker.py` spawns at a low bar and coasts for seconds, and false positives are handled offline by temporal support and by ranking (`pipeline/score.py`), not by per-frame gates.

**Two passes, cheap middle.** `pipeline/analysis.py:analyse` decodes once and stores per-frame fused candidates (`raw`, `raw_faces`) plus live tracklets in the sidecar (`pipeline/project.py`, `<video>.avpp2.json`). Everything after detection re-runs from that cache: `session.run_offline` → `analysis.retrack` (tracker knobs are *not* in the detection fingerprint) → `refine.refine` (trim, soft prune, bridge, fill, extend, smooth, upsample) → `score.score_tracks` → `Track` list sorted by suspicion. `render.build_table` turns enabled tracks + manual heads into per-frame boxes with velocity padding; `render.render_video` paints them.

**Data shapes.** Sources are `(K,5)` xyxy+score. `FrameCands` = fused heads `(N,5)` + `Src` flag bits + raw face pool. `Tracklet` channels are `cx,cy,w,h` (`boxes`, `fboxes`) with `hits`, `scores`, `src` (uint32 `Src` bits, `INTERP` on interpolated frames), `fvalid`; step space until `refine.upsample`. `Track` = frame-space tracklet + suspicion/reasons/verified/kept. `ManualTrack` = dense xyxy boxes with a negative tid.

**Step space.** With stride `s`, pass 1 records at `frame // s`; `retrack`/`refine` run at `fps / s` and `n_steps`; `upsample` maps back, marking non-measured frames `Src.INTERP` and never as hits. Anything that reads frames for a step-space tracklet must seek `idx * s` (`session.make_identify`); the verifier runs after upsample, in frame space.

**UI** (`src/app/`). `worker.PipelineWorker` (QThread) owns the `Models` registry and runs one job at a time (analyse / offline / export / propagate); results come back as signals, preview frames through a latest-wins mailbox. `window.MainWindow` holds the `Project`, the scored tracks and the blur table, and rebuilds table → lanes → alerts on every edit; split/trim edits are stored in `project.settings["edits"]` and replayed after any offline re-run so they survive.

## Invariants

- **Never create and discard an ONNX session.** Every model is built once per process (`Models`, `PipelineWorker.models()`) and never rebuilt; churn aborts MIGraphX (`HIP failure 700`) and corrupts DirectML.
- Pin graph input shapes before session creation (`detector._pinned_model`, reused by every wrapper).
- The identity channel is a veto only (`embed.DIFF_ID_COS`): it may split, never join.
- `AnalysisConfig.TRACKING_FIELDS` stay out of the sidecar fingerprint; adding a detection-affecting field means adding it to `fingerprint()`.
- A live track is always blurred while it exists; face mode falls back to the head box, never skips.
- Manual heads use negative tids (`Project.next_manual_id`); split tracks get tids ≥ 100000 in replay order.
- Everything is BGR. The Wholebody graph takes raw BGR float32; headdet/SCRFD/embed convert to RGB and normalise themselves.
- Labels (`libs/labels.py`) must never contain imagery, paths, or filenames; `tests/test_labels.py` enforces it.

## Frozen build

`--windowed` leaves stdio `None`: `rth_windowed_stdio.py` redirects it and `utils.debug_log` mirrors status to `$TMPDIR/FaceBlurInspector-debug.log`. Model resolution order everywhere: `AVPP_<NAME>_ONNX` → `sys._MEIPASS/models/` → `~/.cache/avpp/<name>/`. `build_exe.sh` must bundle every model `libs/models.py:preflight` knows about.

## Changelog

`CHANGELOG.md` follows Keep-a-Changelog; current work goes under `[Unreleased]`.
