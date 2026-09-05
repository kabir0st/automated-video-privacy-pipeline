"""Automated Video Privacy Pipeline — launcher.

Usage:
    uv run python src/main.py [VIDEO]     # open the timeline editor
    uv run python src/cli.py --help       # headless analyse / export
"""

import sys


def main() -> None:
    # Show the splash before importing the app, whose module-level imports
    # (cv2/numpy/scipy/onnxruntime/PyQt6 widgets) are what make startup slow —
    # this file must stay stdlib-only at module level so the splash appears
    # instantly.
    from splash import show_splash  # PyQt6 only

    splash = show_splash()
    from app.window import main as app_main

    app_main(splash=splash)


if __name__ == "__main__":
    main()
