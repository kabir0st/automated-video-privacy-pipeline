"""Pipeline B — Highest Accuracy Offline face landmark tracking.

Usage:
    uv run python src/main.py                        # open GUI inspector
    uv run python src/main.py --input path/to/video.mp4
    uv run python src/main.py --input video.mp4 --output out.mp4
"""

import sys


def main() -> None:
    # No args → GUI inspector. Show the splash before importing ui, whose
    # module-level imports (cv2/torch/insightface/onnxruntime) are what make
    # startup slow — this file must stay stdlib-only at module level.
    if len(sys.argv) == 1:
        from splash import show_splash  # PyQt6 only

        splash = show_splash()
        from ui import main as ui_main

        ui_main(splash=splash)
    else:
        from cli import main as cli_main

        cli_main()


if __name__ == "__main__":
    main()
