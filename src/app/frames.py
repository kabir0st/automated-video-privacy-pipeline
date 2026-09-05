"""Random-access frames for scrubbing and playback.

One ``VideoCapture`` per source, read sequentially whenever the requested
frame is the next one (playback), seeking otherwise (scrub). A small LRU
cache of recent frames makes stepping back and forth free. Not thread-safe
by design: the GUI thread is the only reader; the worker opens its own.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np


class FrameSource:
    def __init__(self, path: str, cache: int = 48) -> None:
        import cv2
        self.path = path
        self._cap = cv2.VideoCapture(path)
        self.ok = self._cap.isOpened()
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_n = cache

    def frame(self, idx: int) -> Optional[np.ndarray]:
        import cv2
        if not self.ok or idx < 0 or (self.n_frames and idx >= self.n_frames):
            return None
        hit = self._cache.get(idx)
        if hit is not None:
            self._cache.move_to_end(idx)
            return hit
        if idx != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self._cap.read()
        if not ok:
            self._next = -1
            return None
        self._next = idx + 1
        self._cache[idx] = fr
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return fr

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:  # noqa: BLE001
            pass
