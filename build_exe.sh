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
#   - InsightFace models (~300 MB) are downloaded at first launch to
#     %USERPROFILE%\.insightface\models\ and are NOT bundled (intentional).
#   - CUDA/GPU acceleration requires matching CUDA drivers on the Windows host.
#     The app falls back to CPU automatically when CUDA is unavailable.

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
"${WIN_PY[@]}" -m pip install --no-cache-dir \
  "boxmot==21.0.0" \
  pyinstaller insightface pyqt6 scipy opencv-python scikit-learn

# GPU: onnxruntime-directml ships the DirectML execution provider, which
# accelerates inference on any Windows GPU (AMD/Intel/NVIDIA). It installs
# into the same `onnxruntime` package directory as the CPU-only build that
# insightface pulls in, so remove BOTH and force-reinstall directml — a plain
# uninstall of one corrupts the other's shared files.
"${WIN_PY[@]}" -m pip uninstall --quiet -y onnxruntime onnxruntime-directml || true
"${WIN_PY[@]}" -m pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-directml

# ── smoke test: catch missing modules before the slow PyInstaller run ────────
echo ""
echo ">>> Smoke-testing imports in Windows Python…"
SRC_WIN_DIR=$(wslpath -w "$SCRIPT_DIR/src")
"${WIN_PY[@]}" -c "
import sys
sys.path.insert(0, r'$SRC_WIN_DIR')
import ui  # pulls in PyQt6, insightface, boxmot, libs.*
from libs.utils import best_onnx_providers
print('imports OK, onnx providers:', best_onnx_providers())
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

