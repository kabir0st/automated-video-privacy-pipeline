"""Pass 2: per-frame blur table and the export loop. No inference here.

The table is built once from the enabled tracks (and manual tracks) with
velocity-aware padding: a box moving fast is grown along its motion so the
blur never trails the head between two smoothed positions. Rendering then
reuses the proven mask/blur/writer stack in libs/.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .geom import rows_to_xyxy
from .types import ManualTrack, Track

BlurLayer = tuple[str, int]
DEFAULT_BLUR_LAYERS: tuple[BlurLayer, ...] = (("gaussian", 71), ("pixelate", 12))


@dataclass(frozen=True)
class RenderConfig:
    region: str = "head"             # "head" | "face"
    mask_pad: float = 0.20
    mask_feather: float = 0.12
    motion_lead: float = 0.6         # box grows by this × per-frame motion
    blur_layers: tuple[BlurLayer, ...] = DEFAULT_BLUR_LAYERS
    crf: int = 18


def _motion_pad(xyxy: np.ndarray, lead: float) -> np.ndarray:
    """Grow each box by ``lead`` × its centre displacement to the previous
    and next frame (symmetric, so smoothing lag in either direction is
    covered)."""
    if len(xyxy) < 2 or lead <= 0:
        return xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    dx = np.abs(np.gradient(cx)) * lead
    dy = np.abs(np.gradient(cy)) * lead
    out = xyxy.copy()
    out[:, 0] -= dx
    out[:, 2] += dx
    out[:, 1] -= dy
    out[:, 3] += dy
    return out


def track_boxes(tr: Track, region: str) -> np.ndarray:
    """Per-frame xyxy blur target for one track. Face mode uses the bloomed
    face box where face evidence exists and falls back to the head box
    elsewhere — it never leaves a frame of a live track uncovered."""
    from libs.utils import bloom_face_box

    head = rows_to_xyxy(tr.t.boxes)
    if region != "face":
        return head
    fb = rows_to_xyxy(tr.t.fb())
    fv = tr.t.fvalid_arr()
    out = head.copy()
    for k in np.nonzero(fv)[0]:
        out[k] = bloom_face_box(fb[k])[:4]
    return out


def build_table(
    tracks: list[Track], manual: list[ManualTrack], n_frames: int,
    enabled: Optional[dict[int, bool]] = None, cfg: RenderConfig = RenderConfig(),
) -> list[list[tuple[int, np.ndarray]]]:
    """``frame -> [(tid, xyxy), ...]`` for every enabled track."""
    enabled = enabled or {}
    table: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_frames)]
    for tr in tracks:
        if not enabled.get(tr.tid, tr.kept):
            continue
        boxes = _motion_pad(track_boxes(tr, cfg.region), cfg.motion_lead)
        for k in range(len(boxes)):
            f = tr.t.start + k
            if 0 <= f < n_frames:
                table[f].append((tr.tid, boxes[k]))
    for m in manual:
        if not enabled.get(m.tid, True):
            continue
        boxes = _motion_pad(np.asarray(m.boxes, np.float32).reshape(-1, 4),
                            cfg.motion_lead)
        for k in range(len(boxes)):
            f = m.start + k
            if 0 <= f < n_frames:
                table[f].append((m.tid, boxes[k]))
    return table


def coverage(table: list[list[tuple[int, np.ndarray]]]) -> np.ndarray:
    """Number of blur boxes per frame."""
    return np.array([len(row) for row in table], np.int32)


def _decode_thread(cap, q: "queue.Queue", stop: threading.Event,
                   name: str = "render-decode") -> threading.Thread:
    def work() -> None:
        idx = 0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            q.put((idx, frame))
            idx += 1
        while not stop.is_set():
            try:
                q.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
    th = threading.Thread(target=work, name=name, daemon=True)
    th.start()
    return th


def render_video(
    video_path: str, out_path: str,
    table: list[list[tuple[int, np.ndarray]]],
    cfg: RenderConfig = RenderConfig(), *,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[threading.Event] = None,
    on_status: Optional[Callable[[str], None]] = None,
    preview: Optional[Callable[[int, np.ndarray], None]] = None,
) -> tuple[bool, str]:
    """Blur ``video_path`` per ``table`` into ``out_path``. Returns
    ``(ok, message)``."""
    import cv2

    from libs.utils import BlurPipeline, render_head_mask
    from libs.video_writer import make_video_writer, source_bitrate_kbps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"cannot open {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(table)
    bitrate = source_bitrate_kbps(cap)
    writer = make_video_writer(out_path, fw, fh, fps, bitrate_kbps=bitrate,
                               crf=cfg.crf, on_status=on_status,
                               audio_source=video_path)
    if writer is None:
        cap.release()
        return False, "no video writer available"
    blur = BlurPipeline()
    blur.reconfigure(cfg.blur_layers)
    q: "queue.Queue" = queue.Queue(maxsize=4)
    stop = threading.Event()
    th = _decode_thread(cap, q, stop)
    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            if cancel is not None and cancel.is_set():
                stop.set()
                return False, "cancelled"
            item = q.get()
            if item is None:
                break
            idx, frame = item
            boxes = [b for _tid, b in (table[idx] if idx < len(table) else [])]
            if boxes:
                mask = render_head_mask((fh, fw), boxes, pad=cfg.mask_pad,
                                        feather=cfg.mask_feather)
                blur.apply(frame, mask)
            writer.write(frame)
            n += 1
            if preview is not None and n % 15 == 0:
                preview(idx, frame)
            if progress is not None and (n % 10 == 0 or n == total):
                progress(idx + 1, total)
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        th.join(timeout=1.0)
        cap.release()
        writer.release()
    dt = time.perf_counter() - t0
    return True, f"wrote {n} frames in {dt:.0f}s ({n / max(dt, 1e-6):.1f} fps)"
