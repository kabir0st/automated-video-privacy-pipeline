"""Pass 1: decode → union-detect → track → record, on an analysis stride.

Owns nothing itself; the :class:`Models` registry holds the ONNX sessions
for the process lifetime (never rebuilt — session churn corrupts DirectML
device state and aborts the MIGraphX EP).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from .fuse import FrameCands, FuseConfig, fuse
from .geom import unrotate_boxes
from .record import TrackRecorder
from .tracker import Tracker
from .types import Tracklet

_ROT_CODES = {90: 0, 180: 1, 270: 2}   # cv2.ROTATE_* enum values


@dataclass(frozen=True)
class AnalysisConfig:
    stride: int = 1
    headdet_rots: tuple[int, ...] = (0, 90, 180, 270)
    wb_rots: tuple[int, ...] = (0, 90, 180, 270)
    scrfd_rots: tuple[int, ...] = (0, 180)
    headdet_size: int = 640
    scrfd_size: int = 640
    headdet_floor: float = 0.10
    scrfd_floor: float = 0.30
    use_wholebody: bool = True
    # NudeNet's non-face classes as a per-frame body-part veto source. Cheap
    # (320 px, one pass) and the only witness trained on what this footage's
    # head detectors mistake for heads.
    use_bodypart: bool = True
    bodypart_floor: float = 0.25
    spawn_conf: float = 0.35
    sustain_conf: float = 0.10
    min_hits: int = 2
    max_age_s: float = 2.5
    fuse: FuseConfig = field(default_factory=FuseConfig)

    # Fusion and tracker knobs are NOT part of the detection fingerprint:
    # per-source detections are cached and both stages re-run from them
    # (see :func:`retrack`), so changing them never costs a detection pass.
    TRACKING_FIELDS = ("spawn_conf", "sustain_conf", "min_hits", "max_age_s",
                       "fuse")

    def fingerprint(self) -> dict:
        """Everything that changes what detection records."""
        d = asdict(self)
        for k in self.TRACKING_FIELDS:
            d.pop(k, None)
        return d


class Models:
    """Lazy, process-lifetime model registry."""

    def __init__(self, on_status: Optional[Callable[[str], None]] = None,
                 headdet_size: int = 640, scrfd_size: int = 640) -> None:
        self._status = on_status
        self._headdet = None
        self._wb = None
        self._scrfd = None
        self._embed = None
        self._nudenet = None
        self._headdet_size = headdet_size
        self._scrfd_size = scrfd_size

    def headdet(self):
        if self._headdet is None:
            from libs.headdet import HeadDetectorV2
            self._headdet = HeadDetectorV2(size=self._headdet_size,
                                           on_status=self._status)
        return self._headdet

    def wb(self):
        if self._wb is None:
            from libs.detector import HeadDetector
            self._wb = HeadDetector(on_status=self._status)
        return self._wb

    def scrfd(self):
        if self._scrfd is None:
            from libs.scrfd import ScrfdDetector
            self._scrfd = ScrfdDetector(on_status=self._status)
            self._scrfd._in_hw = (self._scrfd_size, self._scrfd_size)
        return self._scrfd

    def embed(self):
        if self._embed is None:
            from libs.embed import FaceEmbedder
            self._embed = FaceEmbedder(on_status=self._status)
        return self._embed

    def nudenet(self):
        if self._nudenet is None:
            from libs.nudenet import NudeNetDetector
            self._nudenet = NudeNetDetector(on_status=self._status)
        return self._nudenet


def _rotated(frame: np.ndarray, rot: int) -> np.ndarray:
    import cv2
    return cv2.rotate(frame, _ROT_CODES[rot])


def detect_frame(models: Models, frame: np.ndarray, cfg: AnalysisConfig,
                 timings: Optional[dict] = None
                 ) -> tuple[FrameCands, dict[str, np.ndarray]]:
    """Run every configured source on one frame → ``(fused, sources)``.
    ``sources`` is what gets cached; fusion can be redone from it."""
    from libs.scrfd import kps_plausible

    fh, fw = frame.shape[:2]
    src: dict[str, np.ndarray] = {}

    def add(key: str, arr: np.ndarray) -> None:
        if len(arr) == 0:
            return
        src[key] = np.concatenate([src[key], arr]) if key in src else arr

    def tick(name: str, ms: float) -> None:
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + ms

    hd = models.headdet()
    for rot in cfg.headdet_rots:
        img = frame if rot == 0 else _rotated(frame, rot)
        b = hd.detect(img, rotations=(0,), floor=cfg.headdet_floor)
        tick("headdet", hd.last_ms)
        add("headdet" if rot == 0 else "headdet_rot",
            b if rot == 0 else unrotate_boxes(b, rot, fw, fh))

    if cfg.use_wholebody:
        wb = models.wb()
        for rot in cfg.wb_rots:
            img = frame if rot == 0 else _rotated(frame, rot)
            d = wb.detect(img, rotations=(0,))
            tick("wholebody", wb.last_ms)
            sfx = "" if rot == 0 else "_rot"
            add("wb_head" + sfx, d.heads if rot == 0
                else unrotate_boxes(d.heads, rot, fw, fh))
            add("wb_face" + sfx, d.faces if rot == 0
                else unrotate_boxes(d.faces, rot, fw, fh))

    if cfg.use_bodypart:
        nn = models.nudenet()
        rows = nn.detect_full(frame, floor=cfg.bodypart_floor)
        tick("nudenet", nn.last_ms)
        if len(rows):
            from libs.nudenet import _FACE_IDS
            body = rows[~np.isin(rows[:, 5].astype(int), _FACE_IDS)]
            add("bodypart", body[:, :5].astype(np.float32))

    sc = models.scrfd()
    for rot in cfg.scrfd_rots:
        img = frame if rot == 0 else _rotated(frame, rot)
        b, k = sc.detect_full(img, floor=cfg.scrfd_floor)
        tick("scrfd", sc.last_ms)
        if len(b):
            b = b[kps_plausible(k)]
        add("scrfd" if rot == 0 else "scrfd_rot",
            b if rot == 0 else unrotate_boxes(b, rot, fw, fh))

    return fuse(src, (fh, fw), cfg.fuse), src


def fuse_sources(sources: dict[str, np.ndarray], frame_hw: tuple[int, int],
                 cfg: FuseConfig) -> FrameCands:
    return fuse(sources, frame_hw, cfg)


@dataclass
class AnalysisResult:
    fps: float
    n_frames: int
    stride: int
    tracklets: list[Tracklet]                       # step space
    raw: dict[int, np.ndarray]                      # frame -> (N, 6) xyxy,score,flags
    raw_faces: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> (M, 5)
    raw_sources: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    frame_hw: tuple[int, int] = (0, 0)
    timings: dict = field(default_factory=dict)
    seconds: float = 0.0


def _decode_thread(cap, q: "queue.Queue", stop: threading.Event):
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
    th = threading.Thread(target=work, name="analyse-decode", daemon=True)
    th.start()
    return th


def analyse(
    video_path: str, cfg: AnalysisConfig, models: Models, *,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[threading.Event] = None,
    on_status: Optional[Callable[[str], None]] = None,
    preview: Optional[Callable[[int, np.ndarray, FrameCands, list], None]] = None,
    max_frames: int = 0,
) -> Optional[AnalysisResult]:
    """Pass 1 over the whole file. ``None`` when cancelled or unreadable."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if on_status:
            on_status(f"cannot open {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    if max_frames:
        total = min(total, max_frames)
    step = max(1, int(cfg.stride))
    tracker = Tracker(fps=fps / step, spawn_conf=cfg.spawn_conf,
                      sustain_conf=cfg.sustain_conf, min_hits=cfg.min_hits,
                      max_age_s=cfg.max_age_s)
    recorder = TrackRecorder()
    raw: dict[int, np.ndarray] = {}
    raw_faces: dict[int, np.ndarray] = {}
    raw_sources: dict[int, dict[str, np.ndarray]] = {}
    timings: dict = {}
    q: "queue.Queue" = queue.Queue(maxsize=4)
    stop = threading.Event()
    th = _decode_thread(cap, q, stop)
    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            if cancel is not None and cancel.is_set():
                stop.set()
                return None
            item = q.get()
            if item is None:
                break
            idx, frame = item
            if max_frames and idx >= max_frames:
                stop.set()
                break
            n = idx + 1
            if step > 1 and idx % step:
                if progress and idx % 10 == 0:
                    progress(idx + 1, total)
                continue
            tf = time.perf_counter()
            cands, sources = detect_frame(models, frame, cfg, timings)
            if sources:
                raw_sources[idx] = sources
            obs = tracker.update(cands.heads, cands.flags, frame.shape,
                                 faces=cands.faces)
            recorder.observe(idx // step, obs)
            timings["frame"] = timings.get("frame", 0.0) + (
                time.perf_counter() - tf) * 1e3
            timings["analysed"] = timings.get("analysed", 0) + 1
            if len(cands.heads):
                raw[idx] = np.concatenate(
                    [cands.heads, cands.flags[:, None].astype(np.float32)],
                    axis=1)
            if len(cands.faces):
                raw_faces[idx] = cands.faces
            if preview is not None:
                preview(idx, frame, cands, obs)
            if progress and (idx % 10 == 0 or idx + 1 == total):
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
    return AnalysisResult(fps=fps, n_frames=n, stride=step,
                          tracklets=recorder.finalize(), raw=raw,
                          raw_faces=raw_faces, raw_sources=raw_sources,
                          frame_hw=frame_hw, timings=timings,
                          seconds=time.perf_counter() - t0)


def retrack(raw_sources: dict[int, dict[str, np.ndarray]],
            raw: dict[int, np.ndarray], raw_faces: dict[int, np.ndarray],
            n_frames: int, stride: int, fps: float, frame_hw: tuple[int, int],
            cfg: AnalysisConfig,
            ) -> tuple[list[Tracklet], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Re-fuse (when per-source detections are cached) and re-track from the
    cache — pure numpy, seconds for a long clip — so fusion and tracker knobs
    can change without re-detecting. Returns ``(tracklets, fused raw,
    raw faces)``; the fused dicts replace the project's so alerts and manual
    propagation see the current fusion."""
    step = max(1, int(stride))
    tracker = Tracker(fps=fps / step, spawn_conf=cfg.spawn_conf,
                      sustain_conf=cfg.sustain_conf, min_hits=cfg.min_hits,
                      max_age_s=cfg.max_age_s)
    recorder = TrackRecorder()
    shape = (frame_hw[0] or 10 ** 6, frame_hw[1] or 10 ** 6, 3)
    out_raw: dict[int, np.ndarray] = {}
    out_faces: dict[int, np.ndarray] = {}
    refuse = bool(raw_sources) and frame_hw[0] > 0
    for idx in range(0, n_frames, step):
        if refuse:
            srcs = raw_sources.get(idx)
            c = fuse(srcs, frame_hw, cfg.fuse) if srcs else FrameCands.empty()
            heads, flags, faces = c.heads, c.flags, c.faces
            if len(heads):
                out_raw[idx] = np.concatenate(
                    [heads, flags[:, None].astype(np.float32)], axis=1)
            if len(faces):
                out_faces[idx] = faces
        else:
            r = raw.get(idx)
            if r is None or len(r) == 0:
                heads = np.empty((0, 5), np.float32)
                flags = np.empty(0, np.int64)
            else:
                r = np.asarray(r, np.float32).reshape(-1, 6)
                heads, flags = r[:, :5], r[:, 5].astype(np.int64)
            faces = raw_faces.get(idx)
        obs = tracker.update(heads, flags, shape, faces=faces)
        recorder.observe(idx // step, obs)
    if not refuse:
        out_raw, out_faces = raw, raw_faces
    return recorder.finalize(), out_raw, out_faces
