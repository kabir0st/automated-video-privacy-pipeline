"""Generate the PyInstaller bootloader splash PNG (build-time only).

The ``--onefile`` bootloader extracts the whole bundle to a temp dir *before*
any Python runs, so the Qt splash in ``src/splash.py`` cannot appear during that
gap — on Windows the user sees nothing for several seconds, then the window
pops. This static PNG, passed to PyInstaller via ``--splash``, is shown by the
bootloader to cover the gap; ``splash.show_splash`` calls ``pyi_splash.close()``
to hand off to the live Qt splash. Painted here with cv2 (already a build dep)
so no image asset has to live in the repo — matching splash.py's runtime paint.

Usage:
    python make_splash.py path/to/splash.png
"""

import sys

import cv2
import numpy as np

_W, _H = 420, 240
# BGR equivalents of the Nordic design tokens in src/splash.py / src/ui.py.
_BG = (49, 40, 35)            # #232831
_BORDER = (84, 70, 62)        # #3E4654
_TEXT = (244, 239, 236)       # #ECEFF4
_TEXT_DIM = (148, 132, 122)   # #7A8494
_ACCENT = (208, 192, 136)     # #88C0D0


def make(out_path: str) -> None:
    img = np.empty((_H, _W, 3), dtype=np.uint8)
    img[:] = _BG
    cv2.rectangle(img, (0, 0), (_W - 1, _H - 1), _BORDER, 1)
    cv2.rectangle(img, (32, 92), (32 + 44, 92 + 3), _ACCENT, -1)
    cv2.putText(img, "Automated Video Privacy Pipeline", (32, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, _TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "Loading...", (32, 188),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_DIM, 1, cv2.LINE_AA)
    if not cv2.imwrite(out_path, img):
        raise SystemExit(f"failed to write splash PNG: {out_path}")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "splash.png")
