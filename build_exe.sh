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
#     (insightface, boxmot, onnxruntime, pyqt6, scipy, opencv-python, etc.)
#
# Notes:
#   - InsightFace models (~300 MB) and the RTMW/YOLOX pose models (~300-400 MB)
#     are downloaded at first launch to %USERPROFILE%\.insightface\ and
#     %USERPROFILE%\.cache\rtmlib\ respectively, and are NOT bundled (intentional).
#   - GPU acceleration uses the DirectML execution provider (onnxruntime-directml),
#     which runs on any Windows GPU including the AMD Radeon RX 6800. The build
#     asserts DirectML is active so a silent CPU-only bundle can't ship; the app
#     still falls back to CPU at runtime if no GPU is present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── locate Windows Python ─────────────────────────────────────────────────────
# boxmot 21.0.0 (the version this project uses) requires Python >=3.10,<3.14,
# so we need a 3.10–3.13 interpreter — NOT 3.14+.
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
  echo "ERROR: No Windows Python 3.10–3.13 found (boxmot does not support 3.14+)."
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

# boxmot pinned to the same version as the Linux venv (see uv.lock) so the
# import paths and tracker API match. Its full dep tree (torch CPU, pandas,
# pyyaml, regex, yacs, …) resolves to pre-built wheels on Python 3.10–3.13.
# imageio-ffmpeg ships a static ffmpeg binary that the export path uses for
# rate-controlled, co64-safe H.264 encoding (replacing cv2.VideoWriter, which
# blew exports past 4 GiB into unplayable files). It is collected into the
# bundle below so no system ffmpeg install is required on the target machine.
# tqdm is rtmlib's download-progress dependency; pulled in explicitly so the
# rtmlib --no-deps install below leaves nothing missing.
"${WIN_PY[@]}" -m pip install --no-cache-dir \
  "boxmot==21.0.0" \
  pyinstaller insightface pyqt6 scipy opencv-python scikit-learn imageio-ffmpeg tqdm

# rtmlib is the default pose backend (RTMDet/YOLOX → RTMW whole-body); it must
# be installed here so PyInstaller can bundle it, or the .exe dies with
# "No module named 'rtmlib'" and silently falls back to SCRFD-only.
#
# CRITICAL: install it with --no-deps. rtmlib declares a plain `onnxruntime`
# dependency, which pip would resolve to the CPU-only wheel and silently
# overwrite the DirectML build below (they share the same `onnxruntime` package
# directory) — the .exe would then run inference on CPU. All of rtmlib's real
# runtime deps (numpy, opencv-python, onnxruntime, tqdm) are provided by the
# other install lines, so --no-deps yields a fully working rtmlib.
"${WIN_PY[@]}" -m pip install --no-cache-dir --no-deps rtmlib

# GPU: onnxruntime-directml ships the DirectML execution provider, which
# accelerates inference on any Windows GPU including the AMD Radeon RX 6800. It
# installs into the same `onnxruntime` package directory as the CPU-only build
# that insightface/rtmlib pull in, so remove BOTH and force-reinstall directml.
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
import rtmlib  # default pose backend — must be importable for bundling
import ui  # pulls in PyQt6, insightface, boxmot, libs.*
import onnxruntime as ort
from libs.utils import best_onnx_providers
prov = best_onnx_providers()
print('imports OK | rtmlib', getattr(rtmlib, '__version__', '?'),
      '| onnx providers:', prov)
assert 'DmlExecutionProvider' in ort.get_available_providers(), (
    'DirectML provider missing — onnxruntime-directml is not active, the .exe '
    'would run inference on CPU. Re-check the force-reinstall step above.')
print('OK: DirectML provider present — GPU inference will be used on the RX 6800')
"

# ── convert WSL paths → Windows paths ────────────────────────────────────────
SRC_WIN=$(wslpath -w "$SCRIPT_DIR/src/ui.py")
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

# ── run pyinstaller via Windows Python ───────────────────────────────────────
echo ""
echo ">>> Building FaceBlurInspector.exe (--onefile --windowed)…"

"${WIN_PY[@]}" -m PyInstaller \
  --onefile \
  --windowed \
  --runtime-hook "$RTH_WIN" \
  --name "FaceBlurInspector" \
  --distpath "$DIST_WIN" \
  --workpath "$WORK_WIN" \
  --specpath "$SPEC_WIN" \
  --paths "$PATHS_WIN" \
  \
  --hidden-import "libs.utils" \
  --hidden-import "libs.tracker" \
  --hidden-import "libs.smoother" \
  --hidden-import "libs.face_app" \
  --hidden-import "ui" \
  \
  --hidden-import "PyQt6" \
  --hidden-import "PyQt6.QtWidgets" \
  --hidden-import "PyQt6.QtCore" \
  --hidden-import "PyQt6.QtGui" \
  --hidden-import "PyQt6.sip" \
  \
  --collect-all "insightface" \
  --collect-all "onnxruntime" \
  --collect-all "boxmot" \
  --collect-all "imageio_ffmpeg" \
  --collect-all "rtmlib" \
  --hidden-import "tqdm" \
  --hidden-import "libs.pose_rtmw" \
  --hidden-import "libs.pipeline" \
  --hidden-import "libs.video_writer" \
  \
  --hidden-import "onnx" \
  --hidden-import "onnxruntime.tools.onnx_model_utils" \
  --hidden-import "scipy.signal" \
  --hidden-import "scipy.ndimage" \
  --hidden-import "scipy.spatial" \
  --hidden-import "sklearn.utils._cython_blas" \
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

