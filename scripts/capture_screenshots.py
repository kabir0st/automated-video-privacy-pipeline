"""Generate README screenshots by driving the real inspector window headlessly.

Runs the actual src/ui.py MainWindow under the Qt 'offscreen' platform, feeds it
a short clip built from InsightFace's bundled multi-face sample image, and grabs
the window under a few tunable settings. No display, webcam, or manual video
needed; the buffalo_l model pack must already be available (~/.insightface).

    QT_QPA_PLATFORM=offscreen uv run python scripts/capture_screenshots.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import insightface.data

from PyQt6.QtWidgets import QApplication

import ui as ui_module
from ui import MainWindow

OUT_DIR = ROOT / "docs" / "screenshots"
SAMPLE = Path("/tmp/_avpp_sample.mp4")
N_FRAMES = 30
FPS = 25.0


def build_sample_clip() -> None:
    """Write a short clip from the 't1' group photo (several faces)."""
    img = insightface.data.get_image("t1")  # BGR group photo
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(SAMPLE), fourcc, FPS, (w, h))
    for _ in range(N_FRAMES):
        writer.write(img)
    writer.release()
    print(f"[sample] wrote {N_FRAMES} frames → {SAMPLE}  ({w}x{h})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_sample_clip()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(ui_module.STYLE)
    win = MainWindow()
    win.resize(1360, 860)
    win.show()

    # Count worker renders without consuming the preview mailbox (MainWindow's
    # own _on_preview_ready slot, connected first, does the actual UI update).
    rendered = {"n": 0}
    win._worker.preview_ready.connect(lambda: rendered.__setitem__("n", rendered["n"] + 1))

    def settle(rounds: int = 8) -> None:
        for _ in range(rounds):
            app.processEvents()
            time.sleep(0.02)

    def wait_render(target: int, timeout: float = 120.0) -> None:
        t0 = time.time()
        while rendered["n"] < target and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.01)
        settle()

    # Load the sample clip directly (bypass the file dialog in _open_video).
    win._cap = cv2.VideoCapture(str(SAMPLE))
    win._video_path = str(SAMPLE)
    win._total_frames = int(win._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win._frame_slider.setMaximum(max(0, win._total_frames - 1))
    win._play_btn.setEnabled(True)
    win._export_btn.setEnabled(True)
    win._on_status(f"{SAMPLE.name}   {win._total_frames} frames")
    settle()

    def load_frame(idx: int) -> None:
        win._current_frame_idx = idx
        win._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = win._cap.read()
        assert ret, "could not read sample frame"
        win._pending_frame = frame
        win._frame_slider.blockSignals(True)
        win._frame_slider.setValue(idx)
        win._frame_slider.blockSignals(False)
        win._frame_lbl.setText(f"{idx} / {win._total_frames - 1}")

    def submit_and_grab(name: str) -> None:
        win._debounce.stop()  # avoid a stray delayed re-submit
        target = rendered["n"] + 1
        win._submit_current_frame()
        wait_render(target)
        path = OUT_DIR / name
        ok = win.grab().save(str(path))
        print(f"[shot] {'ok ' if ok else 'FAIL'} {name}  (render #{rendered['n']})")

    load_frame(0)

    # 1. Hero — Balanced preset, simple view (advanced collapsed).
    submit_and_grab("hero.png")

    # 2. Max Privacy preset selected, still the simple view.
    win._apply_preset("Max Privacy")
    submit_and_grab("preset-max-privacy.png")

    # 3. Advanced panel revealed, back on Balanced.
    win._apply_preset("Balanced")
    win._adv_btn.setChecked(True)
    settle()
    submit_and_grab("advanced.png")

    # 4. Soft Gaussian wash (large kernel, minimal mosaic).
    win._set_blur_layers((("gaussian", 151), ("pixelate", 2)))
    win._on_param_change()
    submit_and_grab("tunables-blur-gaussian.png")

    # 5. Heavy pixelation (chunky mosaic, minimal Gaussian).
    win._set_blur_layers((("gaussian", 3), ("pixelate", 40)))
    win._on_param_change()
    submit_and_grab("tunables-blur-pixelate.png")

    # 6. Wide mask coverage (large hull expand + hairline extra). Reset blur.
    win._set_blur_layers((("gaussian", 71), ("pixelate", 10)))
    win._on_param_change()
    win._expand_sl.set_value(1.50)
    win._hair_sl.set_value(2.50)
    submit_and_grab("tunables-hull-expand.png")

    win._cap.release()
    win._worker.stop()
    win._worker.wait(3000)
    print(f"[done] screenshots in {OUT_DIR}")


if __name__ == "__main__":
    main()
