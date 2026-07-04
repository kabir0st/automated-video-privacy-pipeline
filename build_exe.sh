#!/usr/bin/env bash
# Build a standalone Windows .exe for the Face Blur Pipeline Inspector.
#
# Run this from WSL2. It invokes the Windows Python interpreter via WSL2
# interop so PyInstaller produces a native Windows executable.
#
# Usage:
#   ./build_exe.sh
#
# Output: dist/FaceBlurInspector.exe  (single-file, no console window)
#
# Requirements:
#   - WSL2 with interop enabled (default)
#   - Python for Windows installed on the host (python.org installer or
#     Microsoft Store), accessible as py.exe or python.exe from WSL2
#   - All project dependencies must be installable on the Windows Python
#     (onnxruntime-directml, pyqt6, scipy, opencv-python, etc.)
#
# Notes:
#   - All four models — the PINTO YOLOv9-Wholebody17 detector (~28 MB), the
#     SCRFD close-up assist (~17 MB), the RTMPose-m body7 pose estimator
#     (~25 MB, supplies the evidence gate's anatomical anchor + torso axis)
#     and the NudeNet YOLOv8n verify witness (~12 MB, offline-only —
#     independent cross-model confirmation for tracklet verification) — ARE
#     bundled into the exe (--add-data below), so a first run needs no
#     downloads at all. The startup preflight (libs/models.py) still reports
#     each one's location and can download to %USERPROFILE%\.cache\avpp\ if a
#     bundle is bypassed via the AVPP_*_ONNX/AVPP_*_URL env overrides.
#   - NudeNet is AGPL-3.0 licensed (the model weights, bundled here for this
#     project's own offline/personal-use build; see README before ever
#     redistributing the exe).
#   - GPU acceleration uses the DirectML execution provider (onnxruntime-directml),
#     which runs on any Windows GPU including the AMD Radeon RX 6800. The build
#     asserts DirectML is active so a silent CPU-only bundle can't ship; the app
#     still falls back to CPU at runtime if no GPU is present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── locate Windows Python ─────────────────────────────────────────────────────
# onnxruntime-directml ships wheels for 3.10–3.13; stay off 3.14+ until it does.
echo ">>> Locating Windows Python (3.10–3.13)…"
WIN_PY=()
if command -v py.exe &>/dev/null; then
  for ver in 3.12 3.13 3.11 3.10; do
    if py.exe -"$ver" --version &>/dev/null; then
      WIN_PY=(py.exe -"$ver")
      break
    fi
  done
fi
if [[ ${#WIN_PY[@]} -eq 0 ]]; then
  echo ""
  echo "ERROR: No Windows Python 3.10–3.13 found."
  echo "  Install Python 3.12 for Windows from https://python.org, then re-run."
  echo "  Check installed versions with:  py.exe --list"
  exit 1
fi
echo "    Using: ${WIN_PY[*]}"
"${WIN_PY[@]}" --version

# ── install / upgrade pyinstaller in Windows Python ──────────────────────────
echo ""
echo ">>> Installing dependencies into Windows Python…"
"${WIN_PY[@]}" -m pip install --no-cache-dir --upgrade pip

# imageio-ffmpeg ships a static ffmpeg binary that the export path uses for
# rate-controlled, co64-safe H.264 encoding + audio stream-copy (replacing
# cv2.VideoWriter, which blew exports past 4 GiB into unplayable files). It is
# collected into the bundle below so no system ffmpeg install is required.
# onnxconverter-common provides the float16 graph conversion that gives the
# RX 6800 its ~2× fp16 inference speedup (see libs/utils.fp16_model_path).
# onnx is needed at runtime to shape-pin the detector graph for DirectML.
"${WIN_PY[@]}" -m pip install --no-cache-dir \
  pyinstaller pyqt6 scipy opencv-python imageio-ffmpeg numpy onnx \
  onnxconverter-common

# GPU: onnxruntime-directml ships the DirectML execution provider, which
# accelerates inference on any Windows GPU including the AMD Radeon RX 6800. It
# installs into the same `onnxruntime` package directory as the CPU-only build
# that other packages may pull in, so remove BOTH and force-reinstall directml.
# This MUST be the last pip operation that touches onnxruntime.
"${WIN_PY[@]}" -m pip uninstall --quiet -y onnxruntime onnxruntime-directml || true
"${WIN_PY[@]}" -m pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-directml

# ── smoke test: catch missing modules before the slow PyInstaller run ────────
echo ""
echo ">>> Smoke-testing imports in Windows Python…"
SRC_WIN_DIR=$(wslpath -w "$SCRIPT_DIR/src")
"${WIN_PY[@]}" -c "
import sys
sys.path.insert(0, r'$SRC_WIN_DIR')
import ui  # pulls in PyQt6, cv2, scipy, libs.*
import onnxruntime as ort
from libs.utils import best_onnx_providers
prov = best_onnx_providers()
print('imports OK | onnx providers:', prov)
assert 'DmlExecutionProvider' in ort.get_available_providers(), (
    'DirectML provider missing — onnxruntime-directml is not active, the .exe '
    'would run inference on CPU. Re-check the force-reinstall step above.')
print('OK: DirectML provider present — GPU inference will be used on the RX 6800')
"

# ── ensure the detector model exists, for bundling ───────────────────────────
# The single ONNX (~28 MB) is bundled into the exe so first run downloads
# nothing. libs.detector resolves sys._MEIPASS/models/<name> first at runtime.
echo ""
echo ">>> Ensuring detector model for bundling…"
MODEL_PATH=$("$SCRIPT_DIR/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR/src')
from libs.detector import download_model
print(download_model(on_status=lambda m: print(m, file=sys.stderr)))
" | tail -1)
echo "    Model: $MODEL_PATH"
MODEL_WIN=$(wslpath -w "$MODEL_PATH")

# SCRFD close-up assist model (~17 MB) — same treatment: bundled so the
# frozen exe never downloads. libs/scrfd.py resolves _MEIPASS/models first.
echo ""
echo ">>> Ensuring SCRFD close-up model for bundling…"
SCRFD_PATH=$("$SCRIPT_DIR/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR/src')
from libs.scrfd import download_model
print(download_model(on_status=lambda m: print(m, file=sys.stderr)))
" | tail -1)
echo "    Model: $SCRFD_PATH"
SCRFD_WIN=$(wslpath -w "$SCRFD_PATH")

# Pose estimator (RTMPose-m body7, ~25 MB) — supplies the evidence gate's
# anatomical anchor + torso axis (Phase 2). Same bundling treatment;
# libs/pose.py resolves _MEIPASS/models first.
echo ""
echo ">>> Ensuring pose estimator model for bundling…"
POSE_PATH=$("$SCRIPT_DIR/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR/src')
from libs.pose import download_model
print(download_model(on_status=lambda m: print(m, file=sys.stderr)))
" | tail -1)
echo "    Model: $POSE_PATH"
POSE_WIN=$(wslpath -w "$POSE_PATH")

# NudeNet verify witness (YOLOv8n, ~12 MB) — independent, offline-only
# cross-model witness for tracklet verification (Phase 3). Same bundling
# treatment; libs/nudenet.py resolves _MEIPASS/models first.
echo ""
echo ">>> Ensuring NudeNet verify witness model for bundling…"
NUDENET_PATH=$("$SCRIPT_DIR/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR/src')
from libs.nudenet import download_model
print(download_model(on_status=lambda m: print(m, file=sys.stderr)))
" | tail -1)
echo "    Model: $NUDENET_PATH"
NUDENET_WIN=$(wslpath -w "$NUDENET_PATH")

# ── convert WSL paths → Windows paths ────────────────────────────────────────
# Entry is main.py (NOT ui.py): main.py shows the loading splash before the heavy
# cv2/onnxruntime/insightface imports, then hands the splash to ui.main() which
# runs the model preflight on it. Building from ui.py skips the splash entirely.
SRC_WIN=$(wslpath -w "$SCRIPT_DIR/src/main.py")
PATHS_WIN=$(wslpath -w "$SCRIPT_DIR/src")
RTH_WIN=$(wslpath -w "$SCRIPT_DIR/rth_windowed_stdio.py")

# PyInstaller's heavy write I/O must happen on the native Windows filesystem:
# writing the ~300 MB bundle over the \\wsl.localhost\ 9P share intermittently
# produces empty/corrupt files. Build under C:\ and copy the exe back at the end.
WIN_BUILD_ROOT="/mnt/c/Temp/FaceBlurInspector-build"
mkdir -p "$WIN_BUILD_ROOT"
DIST_WIN=$(wslpath -w "$WIN_BUILD_ROOT/dist")
WORK_WIN=$(wslpath -w "$WIN_BUILD_ROOT/build")
SPEC_WIN=$(wslpath -w "$WIN_BUILD_ROOT")

# ── clean previous build artefacts ───────────────────────────────────────────
echo ""
echo ">>> Cleaning previous build…"
rm -rf build dist FaceBlurInspector.spec "$WIN_BUILD_ROOT/build" "$WIN_BUILD_ROOT/dist"

# ── generate the bootloader splash PNG ───────────────────────────────────────
# In --onefile mode the bootloader unpacks the whole bundle before any Python
# runs; the Qt splash (src/splash.py) can't show during that gap, so on Windows
# the user saw nothing for several seconds. This native --splash image covers it
# and is handed off to the Qt splash (pyi_splash.close() in show_splash). Painted
# from cv2 so no image asset lives in the repo.
echo ""
echo ">>> Generating bootloader splash image…"
SPLASH_PNG="$WIN_BUILD_ROOT/splash.png"
"${WIN_PY[@]}" "$(wslpath -w "$SCRIPT_DIR/make_splash.py")" "$(wslpath -w "$SPLASH_PNG")"
SPLASH_WIN=$(wslpath -w "$SPLASH_PNG")

# ── run pyinstaller via Windows Python ───────────────────────────────────────
echo ""
echo ">>> Building FaceBlurInspector.exe (--onefile --windowed)…"

"${WIN_PY[@]}" -m PyInstaller \
  --onefile \
  --windowed \
  --splash "$SPLASH_WIN" \
  --runtime-hook "$RTH_WIN" \
  --name "FaceBlurInspector" \
  --distpath "$DIST_WIN" \
  --workpath "$WORK_WIN" \
  --specpath "$SPEC_WIN" \
  --paths "$PATHS_WIN" \
  \
  --hidden-import "libs.utils" \
  --hidden-import "libs.detector" \
  --hidden-import "libs.evidence" \
  --hidden-import "libs.scrfd" \
  --hidden-import "libs.pose" \
  --hidden-import "libs.nudenet" \
  --hidden-import "libs.sidecar" \
  --hidden-import "libs.head_tracker" \
  --hidden-import "libs.tracklets" \
  --hidden-import "libs.models" \
  --hidden-import "libs.video_writer" \
  --hidden-import "splash" \
  --hidden-import "review_ui" \
  --hidden-import "ui" \
  \
  --hidden-import "PyQt6" \
  --hidden-import "PyQt6.QtWidgets" \
  --hidden-import "PyQt6.QtCore" \
  --hidden-import "PyQt6.QtGui" \
  --hidden-import "PyQt6.sip" \
  \
  --collect-all "onnxruntime" \
  --collect-all "imageio_ffmpeg" \
  --add-data "$MODEL_WIN;models" \
  --add-data "$SCRFD_WIN;models" \
  --add-data "$POSE_WIN;models" \
  --add-data "$NUDENET_WIN;models" \
  \
  \
  `# utils.py optionally imports torch for the CUDA blur path; on this` \
  `# DirectML/OpenCL target that path never activates, and leftover torch in` \
  `# the build env would add ~200 MB. Excluding it flips the runtime to the` \
  `# same OpenCL blur it would pick anyway.` \
  --exclude-module "torch" \
  --exclude-module "torchvision" \
  --exclude-module "torchaudio" \
  --exclude-module "pandas" \
  --exclude-module "matplotlib" \
  --exclude-module "sklearn" \
  --exclude-module "IPython" \
  \
  --hidden-import "onnx" \
  --hidden-import "onnxruntime.tools.onnx_model_utils" \
  --collect-all "onnxconverter_common" \
  --hidden-import "scipy.signal" \
  --hidden-import "scipy.optimize" \
  \
  --hidden-import "cv2" \
  --hidden-import "numpy" \
  \
  "$SRC_WIN"

# ── copy result back to the project dir ──────────────────────────────────────
echo ""
echo ">>> Copying exe back to project dist/…"
mkdir -p dist
cp "$WIN_BUILD_ROOT/dist/FaceBlurInspector.exe" dist/

# ── report ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
OUT="dist/FaceBlurInspector.exe"
if [[ -f "$OUT" ]]; then
  SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1 || echo "?")
  echo "  Built:  $OUT  ($SIZE)"
  echo "  Run:    copy to Windows and double-click, or:"
  echo "          $(wslpath -w "$SCRIPT_DIR/$OUT")"
else
  echo "  WARNING: $OUT not found — check PyInstaller output above."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

