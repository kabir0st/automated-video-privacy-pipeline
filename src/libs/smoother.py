from collections import deque

import numpy as np
from scipy.signal import savgol_filter

WINDOW = 15
CONF_THRESHOLD = 0.5
SG_POLY_ORDER = 2
MIN_FRAMES_FOR_SG = 5


class LandmarkSmoother:
    """Per-track Savitzky-Golay smoother with occlusion hold.

    Maintains a rolling deque of (x, y) observations per landmark per track.
    When a landmark's confidence is below CONF_THRESHOLD the last valid value
    is held — the deque is not updated — so the smoother output stays stable
    rather than jumping during occlusion.
    """

    def __init__(
        self,
        window: int = WINDOW,
        conf_threshold: float = CONF_THRESHOLD,
        poly_order: int = SG_POLY_ORDER,
    ) -> None:
        self.window = window
        self.conf_threshold = conf_threshold
        self.poly_order = poly_order
        # track_id -> list[ [deque_x, deque_y] ]  (one entry per landmark)
        self._histories: dict[int, list[list[deque]]] = {}
        # track_id -> last smoothed array  (fallback when window too short)
        self._last: dict[int, np.ndarray] = {}

    def _init_track(self, track_id: int, n: int) -> None:
        self._histories[track_id] = [
            [deque(maxlen=self.window), deque(maxlen=self.window)]
            for _ in range(n)
        ]

    def update(
        self,
        track_id: int,
        landmarks: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        """Return smoothed (N, 2) landmark array for this track.

        landmarks: (N, 2) raw positions from InsightFace.
        confidence: scalar det_score from InsightFace (proxy for all-landmark
                    confidence; use per-landmark scores if available).
        """
        n = len(landmarks)
        if track_id not in self._histories:
            self._init_track(track_id, n)

        history = self._histories[track_id]
        smoothed = np.zeros((n, 2), dtype=np.float32)

        for i, (x, y) in enumerate(landmarks):
            hx, hy = history[i]
            if confidence >= self.conf_threshold:
                hx.append(float(x))
                hy.append(float(y))

            if len(hx) >= MIN_FRAMES_FOR_SG:
                wlen = len(hx) if len(hx) % 2 == 1 else len(hx) - 1
                wlen = max(wlen, self.poly_order + 1)
                sx = savgol_filter(list(hx), wlen, self.poly_order)[-1]
                sy = savgol_filter(list(hy), wlen, self.poly_order)[-1]
            elif len(hx) > 0:
                sx, sy = hx[-1], hy[-1]
            else:
                sx, sy = float(x), float(y)

            smoothed[i] = [sx, sy]

        self._last[track_id] = smoothed
        return smoothed

    def drop(self, track_id: int) -> None:
        """Forget a track's history (call when the tracker drops it)."""
        self._histories.pop(track_id, None)
        self._last.pop(track_id, None)

    def drop_all(self) -> None:
        self._histories.clear()
        self._last.clear()
