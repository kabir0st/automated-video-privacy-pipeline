"""Manual-box propagation: the user draws a head on one frame and the box
follows it forward (or backward) without any model call.

Per frame, in order of preference:
  1. snap to a recorded raw candidate (pass 1 kept every fused head down to
     a low floor, including those too weak to spawn a track) that overlaps
     the prediction;
  2. normalised cross-correlation template match in a search window around
     the prediction, against both the original template and the most
     recent one;
  3. hold the prediction for a few frames, then stop.

Backward propagation decodes in chunks so it never seeks per frame and
never holds more than a chunk of frames in memory.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .geom import iou_matrix


@dataclass(frozen=True)
class PropagateConfig:
    max_frames: int = 300
    snap_iou: float = 0.25
    snap_centre: float = 0.6       # centre distance / box diagonal
    match_min: float = 0.45        # NCC floor to accept a template match
    search_scale: float = 1.0      # search window = box grown by this per side
    hold_frames: int = 6
    chunk: int = 48


def _crop(gray: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    x1, y1 = int(max(0, b[0])), int(max(0, b[1]))
    x2, y2 = int(min(w, b[2])), int(min(h, b[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return gray[y1:y2, x1:x2]


def _match(gray: np.ndarray, templates: list[np.ndarray], pred: np.ndarray,
           cfg: PropagateConfig) -> tuple[Optional[np.ndarray], float]:
    import cv2

    h, w = gray.shape[:2]
    bw, bh = pred[2] - pred[0], pred[3] - pred[1]
    sx1 = int(max(0, pred[0] - bw * cfg.search_scale))
    sy1 = int(max(0, pred[1] - bh * cfg.search_scale))
    sx2 = int(min(w, pred[2] + bw * cfg.search_scale))
    sy2 = int(min(h, pred[3] + bh * cfg.search_scale))
    win = gray[sy1:sy2, sx1:sx2]
    best, best_val = None, -1.0
    for tpl in templates:
        th, tw = tpl.shape[:2]
        if th < 8 or tw < 8 or win.shape[0] < th or win.shape[1] < tw:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, loc = cv2.minMaxLoc(res)
        if mx > best_val:
            best_val = float(mx)
            best = np.array([sx1 + loc[0], sy1 + loc[1],
                             sx1 + loc[0] + tw, sy1 + loc[1] + th], np.float32)
    return best, best_val


def _step(gray: np.ndarray, pred: np.ndarray, raw_here: Optional[np.ndarray],
          templates: list[np.ndarray], cfg: PropagateConfig,
          ) -> tuple[Optional[np.ndarray], str]:
    if raw_here is not None and len(raw_here):
        cand = raw_here[:, :4]
        ious = iou_matrix(pred[None, :], cand)[0]
        diag = float(np.hypot(pred[2] - pred[0], pred[3] - pred[1])) or 1.0
        dcx = (cand[:, 0] + cand[:, 2]) / 2 - (pred[0] + pred[2]) / 2
        dcy = (cand[:, 1] + cand[:, 3]) / 2 - (pred[1] + pred[3]) / 2
        dist = np.hypot(dcx, dcy) / diag
        ok = (ious >= cfg.snap_iou) | (dist <= cfg.snap_centre)
        if ok.any():
            k = int(np.argmax(np.where(ok, ious - 0.1 * dist, -1e9)))
            return cand[k].astype(np.float32), "snap"
    box, val = _match(gray, templates, pred, cfg)
    if box is not None and val >= cfg.match_min:
        return box, "match"
    return None, "miss"


def propagate(
    video_path: str, frame: int, box: np.ndarray, *, direction: int = 1,
    raw: Optional[dict[int, np.ndarray]] = None,
    cfg: PropagateConfig = PropagateConfig(),
    cancel: Optional[threading.Event] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> list[tuple[int, np.ndarray]]:
    """Boxes for ``frame`` and the frames after (``direction=1``) or before
    (``-1``) it, until the head is lost or ``max_frames`` is reached.
    Returns ``[(frame_idx, xyxy), ...]`` in the order visited (the caller
    reverses for backward runs)."""
    import cv2

    box = np.asarray(box, np.float32).reshape(4)
    raw = raw or {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [(frame, box)]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: list[tuple[int, np.ndarray]] = [(frame, box)]
    templates: list[np.ndarray] = []
    pred = box.copy()
    vel = np.zeros(2, np.float32)
    misses = 0

    def visit(idx: int, bgr: np.ndarray) -> bool:
        """Process one frame; False when propagation should stop."""
        nonlocal pred, vel, misses, templates
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if not templates:
            t = _crop(gray, pred)
            if t is not None:
                templates = [t]
            return True
        guess = pred.copy()
        guess[[0, 2]] += vel[0]
        guess[[1, 3]] += vel[1]
        nb, how = _step(gray, guess, raw.get(idx), templates, cfg)
        if nb is None:
            misses += 1
            if misses > cfg.hold_frames:
                return False
            out.append((idx, guess.copy()))
            pred = guess
            return True
        misses = 0
        vel = 0.6 * vel + 0.4 * np.array([(nb[0] + nb[2]) / 2 - (pred[0] + pred[2]) / 2,
                                          (nb[1] + nb[3]) / 2 - (pred[1] + pred[3]) / 2],
                                         np.float32)
        pred = nb
        out.append((idx, nb.copy()))
        t = _crop(gray, nb)
        if t is not None:
            templates = [templates[0], t]
        if progress:
            progress(idx)
        return True

    try:
        if direction >= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                return out
            visit(frame, bgr)
            idx = frame
            while idx + 1 < total and idx + 1 <= frame + cfg.max_frames:
                if cancel is not None and cancel.is_set():
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                idx += 1
                if not visit(idx, bgr):
                    break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                return out
            visit(frame, bgr)
            lo_limit = max(0, frame - cfg.max_frames)
            hi = frame - 1
            stopped = False
            while hi >= lo_limit and not stopped:
                lo = max(lo_limit, hi - cfg.chunk + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
                chunk: list[np.ndarray] = []
                for _ in range(hi - lo + 1):
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    chunk.append(bgr)
                for k in range(len(chunk) - 1, -1, -1):
                    if cancel is not None and cancel.is_set():
                        stopped = True
                        break
                    if not visit(lo + k, chunk[k]):
                        stopped = True
                        break
                hi = lo - 1
    finally:
        cap.release()
    # Drop trailing held (unconfirmed) frames so a lost head does not leave
    # a static blur hanging in space.
    while len(out) > 1 and misses > 0:
        out.pop()
        misses -= 1
    return out
