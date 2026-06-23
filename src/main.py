"""Automated Video Privacy Pipeline — GUI inspector launcher.

Usage:
    uv run python src/main.py        # open the GUI inspector
"""

import sys


def main() -> None:
    # Show the splash before importing ui, whose module-level imports
    # (cv2/torch/insightface/onnxruntime) are what make startup slow — this file
    # must stay stdlib-only at module level so the splash appears instantly.
    from splash import show_splash  # PyQt6 only

    splash = show_splash()
    from ui import main as ui_main

    ui_main(splash=splash)


if __name__ == "__main__":
    main()
