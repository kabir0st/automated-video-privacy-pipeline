"""Generate documentation screenshots by driving the real editor headlessly.

    QT_QPA_PLATFORM=offscreen uv run python scripts/capture_screenshots.py CLIP.mp4

Opens ``CLIP.mp4`` in the actual ``app.window.MainWindow`` under the Qt
'offscreen' platform (analysing it first if no project sidecar exists), then
grabs the window in a few states into docs/screenshots/. Pick a clip you are
happy to publish — the frames end up in the README.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AVPP_SKIP_PREFLIGHT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PyQt6.QtWidgets import QApplication   # noqa: E402

from app import theme                     # noqa: E402
from app.window import MainWindow         # noqa: E402

OUT_DIR = ROOT / "docs" / "screenshots"


def _wait(app: QApplication, win: MainWindow, timeout: float = 3600) -> None:
    t0 = time.time()
    while win.busy and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)


def _settle(app: QApplication, n: int = 8) -> None:
    for _ in range(n):
        app.processEvents()
        time.sleep(0.03)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    video = sys.argv[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLE)
    win = MainWindow()
    win.resize(1480, 920)
    win.show()
    win.open_video(video)
    _wait(app, win)
    if win.project is None:
        win.analyse()
        _wait(app, win)
    if not win.tracks:
        raise SystemExit("no tracks found in this clip")
    longest = max(win.tracks, key=lambda t: len(t.t.boxes))
    win.show_frame(longest.t.start + len(longest.t.boxes) // 2)
    _settle(app)
    win.grab().save(str(OUT_DIR / "hero.png"))
    win.select(longest.tid)
    _settle(app)
    win.grab().save(str(OUT_DIR / "inspector.png"))
    win.chk_blur.setChecked(True)
    _settle(app)
    win.grab().save(str(OUT_DIR / "blur-preview.png"))
    win.chk_blur.setChecked(False)
    win.chk_dets.setChecked(True)
    _settle(app)
    win.grab().save(str(OUT_DIR / "detections.png"))
    print(f"wrote 4 screenshots to {OUT_DIR}")
    win.close()


if __name__ == "__main__":
    main()
