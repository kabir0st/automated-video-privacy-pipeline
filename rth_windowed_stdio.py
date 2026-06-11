"""PyInstaller runtime hook: give the windowed (no-console) app usable stdio.

In a --windowed build sys.stdout/sys.stderr are None, which crashes
libraries that write to them at import time (boxmot configures loguru
with `logger.add(sys.stderr)`). Point them at the null device instead.
"""
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
