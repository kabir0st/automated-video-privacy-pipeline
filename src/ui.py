"""Modern inspector UI: open video → view Before | Tracking | After frames with live tunables.

Usage:
    uv run python src/ui.py
"""

from __future__ import annotations

import faulthandler
import sys
import tempfile
import time
import threading
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

# Crash log next to the temp dir so frozen (windowed) builds, which have no
# console, still leave a readable traceback behind.
_CRASH_LOG = Path(tempfile.gettempdir()) / "FaceBlurInspector-error.log"


def _log_exception(exc: BaseException) -> str:
    tb = "".join(traceback.format_exception(exc))
    try:
        with open(_CRASH_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{tb}")
    except OSError:
        pass
    return tb


# Native crashes (access violations in onnxruntime/Qt/OpenCV) kill the process
# before any Python except clause runs; faulthandler dumps the thread stacks to
# the crash log so they are diagnosable in the windowed build. The handle must
# stay open for the lifetime of the process.
try:
    _faulthandler_fh = open(_CRASH_LOG, "a", encoding="utf-8")
    faulthandler.enable(file=_faulthandler_fh)
except OSError:
    pass


def _excepthook(exc_type, exc, tb) -> None:
    _log_exception(exc)
    sys.__excepthook__(exc_type, exc, tb)


# PyQt6 aborts the process on unhandled exceptions in slots; in a windowed
# build the traceback would vanish into devnull without these hooks.
sys.excepthook = _excepthook
threading.excepthook = lambda args: _log_exception(args.exc_value)

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QGridLayout,
    QSizePolicy, QComboBox, QFrame, QSplashScreen,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

from libs.face_app import FaceApp
from libs.pose_head import PoseHeadEstimator, _VIS_THRESHOLD
from libs.tracker import KalmanFaceTracker
from libs.utils import (
    DEFAULT_BLUR_LAYERS,
    BlurPipeline,
    MaskBuilder,
    best_onnx_providers,
    crop_face_patch,
    unproject_landmark,
)
from libs.video_writer import make_video_writer, source_bitrate_kbps

CLOSE_UP_TARGET_SIZE = 1024
_TRACK_COLOURS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 200, 100),
    (200, 100, 0), (100, 0, 200),
]

# ── Design tokens — Nordic muted: polar-night surfaces, frost-blue accent ────
_BG         = "#232831"   # window base
_SURFACE    = "#2B313C"   # cards / group boxes (gradient bottom)
_SURFACE_HI = "#303744"   # cards gradient top
_FIELD      = "#333B48"   # inputs, buttons
_FIELD_HI   = "#3C4554"   # hovered field
_BORDER     = "#3E4654"
_BORDER_HI  = "#515C6E"
_TEXT       = "#ECEFF4"
_TEXT_MID   = "#AEB6C3"
_TEXT_DIM   = "#7A8494"
_ACCENT     = "#88C0D0"   # interactive elements only
_ACCENT_HI  = "#A3D3E0"
_ACCENT_BG  = "rgba(136, 192, 208, 0.16)"
_VALUE      = "#A3BE8C"   # live numeric readouts, sage
_PANEL_BG   = "#1B2028"   # video panel letterbox
_MONO       = '"Cascadia Mono", "JetBrains Mono", "Consolas", monospace'

STYLE = """
QWidget {
    background-color: transparent;
    color: @text;
    font-family: "Inter", "Segoe UI", "Ubuntu", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: @bg;
}
QGroupBox {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 @surfaceHi, stop:1 @surface);
    border: 1px solid @border;
    border-radius: 14px;
    margin-top: 18px;
    padding: 12px 12px 10px 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: @textDim;
    font-size: 10px;
    font-weight: bold;
}
QSlider::groove:horizontal {
    height: 6px;
    background: @border;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: @accent;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 2px solid @surface;
}
QSlider::handle:horizontal:hover { background: @accentHi; border-color: @borderHi; }
QSlider::handle:horizontal:disabled { background: @borderHi; }
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 @accent, stop:1 @accentHi);
    border-radius: 3px;
}
QSlider::sub-page:horizontal:disabled { background: @border; }
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 @fieldHi, stop:1 @field);
    border: 1px solid @border;
    border-radius: 9px;
    padding: 7px 16px;
    color: @text;
    min-width: 80px;
}
QPushButton:hover {
    background-color: @fieldHi;
    border-color: @borderHi;
}
QPushButton:pressed {
    background-color: @accentBg;
    border-color: @accent;
}
QPushButton:disabled { color: @textDim; background-color: @surface; }
QPushButton[accent="true"] {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 @accentHi, stop:1 @accent);
    color: @bg;
    font-weight: 600;
    border: 1px solid @accentHi;
}
QPushButton[accent="true"]:hover { background-color: @accentHi; }
QPushButton[accent="true"]:disabled {
    background-color: @field;
    color: @textDim;
    border-color: @border;
}
QPushButton[chip="true"] {
    background-color: transparent;
    border: 1px solid @border;
    border-radius: 16px;
    padding: 6px 16px;
    color: @textMid;
    min-width: 0px;
    font-weight: 500;
}
QPushButton[chip="true"]:hover { border-color: @borderHi; color: @text; }
QPushButton[chip="true"]:checked {
    background-color: @accentBg;
    border-color: @accent;
    color: @accentHi;
}
QPushButton[ghost="true"] {
    background-color: transparent;
    border: none;
    color: @textMid;
    min-width: 0px;
}
QPushButton[ghost="true"]:hover { color: @text; background-color: @field; }
QPushButton[ghost="true"]:checked { color: @accentHi; }
QComboBox {
    background-color: @field;
    border: 1px solid @border;
    border-radius: 9px;
    padding: 4px 10px;
    color: @text;
    min-width: 64px;
}
QComboBox:hover { border-color: @borderHi; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background-color: @field;
    selection-background-color: @accentBg;
    border: 1px solid @borderHi;
    color: @text;
}
QLineEdit {
    background-color: @field;
    border: 1px solid @border;
    border-radius: 8px;
    padding: 4px 8px;
    color: @text;
    selection-background-color: @accentBg;
}
QAbstractItemView { background-color: @field; color: @text; }
QHeaderView::section {
    background-color: @surface;
    color: @textMid;
    border: none;
    padding: 4px 6px;
}
QToolTip {
    background-color: @surfaceHi;
    color: @text;
    border: 1px solid @borderHi;
    border-radius: 6px;
    padding: 5px 9px;
}
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical {
    background: @borderHi;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: transparent; height: 10px; }
QScrollBar::handle:horizontal {
    background: @borderHi;
    border-radius: 5px;
    min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""

# Longer tokens first so e.g. @textMid is consumed before @text.
for _token, _colour in (
    ("@borderHi", _BORDER_HI), ("@border", _BORDER),
    ("@fieldHi", _FIELD_HI), ("@field", _FIELD),
    ("@textMid", _TEXT_MID), ("@textDim", _TEXT_DIM), ("@text", _TEXT),
    ("@accentHi", _ACCENT_HI), ("@accentBg", _ACCENT_BG), ("@accent", _ACCENT),
    ("@surfaceHi", _SURFACE_HI), ("@surface", _SURFACE), ("@bg", _BG),
):
    STYLE = STYLE.replace(_token, _colour)


def _letterspace(lbl: QLabel, px: float = 1.5) -> None:
    """Spread uppercase caption labels; QSS has no letter-spacing property."""
    font = lbl.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, px)
    lbl.setFont(font)


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class Params:
    target_size: int = 640
    det_score: float = 0.55
    face_aspect: float = 0.40
    close_up_ratio: float = 0.60
    blur_expand: float = 0.45
    blur_hair_extra: float = 0.90
    # Ordered blur stack: ("gaussian", kernel) / ("pixelate", block).
    blur_layers: tuple[tuple[str, int], ...] = DEFAULT_BLUR_LAYERS
    match_iou: float = 0.30
    hold_secs: float = 2.0      # keep blurring this long after last correction
    pose_assist: bool = True    # pose-estimated head boxes revive lost tracks


# ── Presets — curated Params bundles; Advanced exposes every value ───────────

PRESETS: dict[str, tuple[str, Params]] = {
    "Balanced": (
        "Sensible defaults for most footage",
        Params(),
    ),
    "Max Privacy": (
        "Catch every face and blur hard — favours coverage over speed",
        Params(target_size=1024, det_score=0.35, face_aspect=0.25,
               blur_expand=0.80, blur_hair_extra=1.50,
               blur_layers=(("gaussian", 99), ("pixelate", 16)),
               match_iou=0.20, hold_secs=4.0),
    ),
    "Crowded Scene": (
        "Many small faces — high-res detection, stricter ID matching",
        Params(target_size=1024, det_score=0.45, face_aspect=0.35,
               match_iou=0.45),
    ),
    "Fast Preview": (
        "Low-res detection for quick scrubbing on CPU",
        Params(target_size=320,
               blur_layers=(("gaussian", 41), ("pixelate", 12)),
               pose_assist=False),
    ),
}

# Overlay tag per tracking source, drawn after the track id.
_SOURCE_TAGS = {"face": "", "head": "·pose", "coast": "·hold"}

# Standard BlazePose 33-point skeleton connections, used to draw the pose
# overlay on the tracking panel so pose estimation is visibly alive.
_POSE_EDGES = (
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # Arms
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Legs
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)
# Distinct from the per-track palette so the skeleton reads as a separate layer.
_POSE_EDGE_COLOUR = (220, 220, 220)
_POSE_BOX_COLOUR = (0, 200, 255)


# ── Helper functions ──────────────────────────────────────────────────────────

def _track_colour(tid: int) -> tuple[int, int, int]:
    return _TRACK_COLOURS[tid % len(_TRACK_COLOURS)]


def _draw_outline(
    frame: np.ndarray,
    poly: Optional[np.ndarray],
    bbox: np.ndarray,
    tid: int,
    colour: tuple[int, int, int],
    tag: str = "",
) -> None:
    if poly is not None:
        cv2.polylines(frame, [poly], True, colour, 1, cv2.LINE_AA)
        x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
    else:
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
        x, y = x1, y1
    cv2.putText(frame, f"id:{tid}{tag}", (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def _draw_pose(
    frame: np.ndarray,
    pose: np.ndarray,
    head_box: Optional[np.ndarray] = None,
) -> None:
    """Draw one BlazePose skeleton (and its derived head box) onto frame.

    pose is an (33, 3) array of (x_px, y_px, visibility). Edges and keypoints
    are drawn only where visibility clears the same threshold the head-box
    estimator uses; keypoints are coloured by confidence (green = high,
    red = low) so a glance tells whether pose estimation is healthy.
    """
    vis = pose[:, 2]
    for a, b in _POSE_EDGES:
        if vis[a] > _VIS_THRESHOLD and vis[b] > _VIS_THRESHOLD:
            pa = (int(pose[a, 0]), int(pose[a, 1]))
            pb = (int(pose[b, 0]), int(pose[b, 1]))
            cv2.line(frame, pa, pb, _POSE_EDGE_COLOUR, 1, cv2.LINE_AA)
    for x, y, v in pose:
        if v <= _VIS_THRESHOLD:
            continue
        # Lerp red→green over [threshold, 1.0] so weak joints stand out.
        t = min(1.0, (v - _VIS_THRESHOLD) / max(1.0 - _VIS_THRESHOLD, 1e-6))
        kp_colour = (0, int(255 * t), int(255 * (1 - t)))
        cv2.circle(frame, (int(x), int(y)), 2, kp_colour, -1, cv2.LINE_AA)
    if head_box is not None:
        x1, y1, x2, y2 = (int(v) for v in head_box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), _POSE_BOX_COLOUR, 1)


def _ok_face(bbox: np.ndarray, score: float, p: Params) -> bool:
    x1, y1, x2, y2 = bbox[:4]
    bh = y2 - y1
    bw = x2 - x1
    return bh > 0 and (bw / bh) >= p.face_aspect and score >= p.det_score


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# ── Processing worker thread ──────────────────────────────────────────────────

class ProcessWorker(QThread):
    # Lightweight notification; the actual frames travel through a
    # latest-wins mailbox (take_preview). Emitting full frames per converted
    # frame would queue up faster than the GUI can build pixmaps, leaving the
    # panels stuck displaying stale frames.
    preview_ready = pyqtSignal()
    status = pyqtSignal(str)
    export_progress = pyqtSignal(int, int)
    export_finished = pyqtSignal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: Optional[tuple[np.ndarray, Params, int, float]] = None
        self._wake = threading.Event()
        self._running = True
        self._app: Optional[FaceApp] = None
        self._model_size = -1
        self._blur = BlurPipeline()
        # Lazy: no model load (or one-time download) until first head query.
        self._pose = PoseHeadEstimator(on_status=self.status.emit)
        # Preview tracking state persists across sequential frames (Play),
        # so detection-gap coasting is visible live; any scrub/jump resets it.
        self._pv_tracker: Optional[KalmanFaceTracker] = None
        self._pv_masks = MaskBuilder()
        self._pv_last_idx = -2
        self._preview_lock = threading.Lock()
        self._preview: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = None
        # Export state — latest params are re-read every frame so slider
        # changes apply live mid-conversion.
        self._params = Params()
        self._params_dirty = False
        self._export_request: Optional[tuple[str, str]] = None
        self._export_run = threading.Event()    # cleared = paused
        self._export_cancel = threading.Event()

    def submit(self, frame: np.ndarray, params: Params,
               frame_idx: int = -1, fps: float = 25.0) -> None:
        with self._lock:
            self._pending = (frame.copy(), replace(params), frame_idx, fps)
        self._wake.set()

    def update_params(self, params: Params) -> None:
        with self._lock:
            self._params = replace(params)
            self._params_dirty = True

    def start_export(self, input_path: str, output_path: str) -> None:
        with self._lock:
            self._export_request = (input_path, output_path)
        self._export_cancel.clear()
        self._export_run.set()
        self._wake.set()

    def pause_export(self) -> None:
        self._export_run.clear()

    def resume_export(self) -> None:
        self._export_run.set()

    def cancel_export(self) -> None:
        self._export_cancel.set()
        self._export_run.set()

    def _emit_preview(
        self, orig: np.ndarray, tracking: np.ndarray,
        blurred: np.ndarray, fps: float,
    ) -> None:
        """Latest-wins handoff: overwrite the mailbox, notify only when it was
        empty so at most one preview_ready is ever queued to the GUI thread."""
        with self._preview_lock:
            was_empty = self._preview is None
            self._preview = (orig, tracking, blurred, fps)
        if was_empty:
            self.preview_ready.emit()

    def take_preview(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        with self._preview_lock:
            preview = self._preview
            self._preview = None
        return preview

    def _latest_params(self) -> Params:
        with self._lock:
            return replace(self._params)

    def _take_params_dirty(self) -> bool:
        with self._lock:
            dirty = self._params_dirty
            self._params_dirty = False
            return dirty

    def stop(self) -> None:
        self._running = False
        self._export_cancel.set()
        self._export_run.set()
        self._wake.set()

    def run(self) -> None:
        while self._running:
            self._wake.wait()
            self._wake.clear()
            if not self._running:
                break
            with self._lock:
                export_job = self._export_request
                self._export_request = None
            if export_job is not None:
                try:
                    self._run_export(*export_job)
                except Exception as exc:  # noqa: BLE001
                    _log_exception(exc)
                    self.export_finished.emit(
                        False, f"Export failed: {exc!r}   (full trace: {_CRASH_LOG})")
                continue
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                continue
            frame, params, frame_idx, fps = job
            try:
                results = self._process(frame, params, frame_idx, fps)
                self._emit_preview(*results)
            except Exception as exc:  # noqa: BLE001
                tb = _log_exception(exc)
                last_frame = tb.strip().splitlines()[-3:-1]
                self.status.emit(
                    f"Error: {exc!r} @ {' '.join(l.strip() for l in last_frame)}"
                    f"   (full trace: {_CRASH_LOG})"
                )

    def _ensure_model(self, size: int) -> None:
        if size == self._model_size and self._app is not None:
            return
        self.status.emit(f"Loading model  det_size={size}×{size} …")
        # FaceApp loads only detection + 2D landmarks (the other buffalo_l
        # models would crash DirectML when their probe sessions are destroyed,
        # and the 3D model's meanshape_68.pkl lookup breaks in PyInstaller
        # bundles). On det-size changes the existing instance is re-prepared
        # in place — recreating it would destroy live DirectML sessions and
        # poison the provider.
        if self._app is None:
            self._app = FaceApp(providers=best_onnx_providers())
        self._app.prepare(ctx_id=0, det_size=(size, size))
        self._model_size = size
        self.status.emit("Ready")

    def _detect_refined(self, frame: np.ndarray, p: Params) -> list:
        """Detection + close-up refinement with live params."""
        assert self._app is not None
        fh, fw = frame.shape[:2]

        raw = self._app.get(frame)
        faces = [f for f in raw if _ok_face(f.bbox, float(f.det_score), p)]

        refined: list = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox[:4]
            if ((x2 - x1) * (y2 - y1)) / (fh * fw) > p.close_up_ratio:
                crop, (ox, oy, sc) = crop_face_patch(
                    frame, face.bbox[:4], target_size=CLOSE_UP_TARGET_SIZE)
                cfs = self._app.get(crop)
                if cfs:
                    cf = cfs[0]
                    if cf.landmark_2d_106 is not None:
                        cf.landmark_2d_106 = np.array(
                            [unproject_landmark(x, y, ox, oy, sc)
                             for x, y in cf.landmark_2d_106], dtype=np.float32)
                        cf.bbox[:4] = [
                            ox + cf.bbox[0] / sc, oy + cf.bbox[1] / sc,
                            ox + cf.bbox[2] / sc, oy + cf.bbox[3] / sc]
                    refined.append(cf)
                    continue
            refined.append(face)
        return refined

    def _head_provider(self, frame: np.ndarray, p: Params):
        """Lazy head-box source for the tracker; None disables pose assist."""
        if not p.pose_assist:
            return None
        return lambda: self._pose.head_boxes(frame)

    def _process(
        self, frame: np.ndarray, p: Params, frame_idx: int = -1,
        fps: float = 25.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        self._ensure_model(p.target_size)

        fh, fw = frame.shape[:2]
        orig = frame.copy()
        tracking = frame.copy()
        blurred = frame.copy()

        refined = self._detect_refined(frame, p)

        # Sequential frames (Play) keep the tracker so gap-coasting shows in
        # the live preview; scrubbing or single-frame inspection resets it.
        if self._pv_tracker is None or frame_idx < 0 \
                or frame_idx != self._pv_last_idx + 1:
            self._pv_tracker = KalmanFaceTracker(
                fps=fps, match_iou=p.match_iou, hold_secs=p.hold_secs)
            self._pv_masks.reset()
        else:
            self._pv_tracker.configure(match_iou=p.match_iou,
                                       hold_secs=p.hold_secs)
        self._pv_last_idx = frame_idx

        # Run pose every preview frame (even with a face present) so the
        # skeleton can be drawn live and the tracker can lean on it; the boxes
        # are precomputed here and handed to the tracker so pose runs once.
        head_boxes: list[tuple[np.ndarray, float]] = []
        poses: list[np.ndarray] = []
        if p.pose_assist:
            head_boxes, poses = self._pose.estimate(frame)
        tracked = self._pv_tracker.update(
            refined, frame.shape,
            (lambda: head_boxes) if p.pose_assist else None)

        # Build blur mask
        blur_mask = np.zeros((fh, fw), dtype=np.uint8)
        polys: dict[int, Optional[np.ndarray]] = {}
        for t in tracked:
            polys[t.track_id] = self._pv_masks.add(
                blur_mask, t, expand=p.blur_expand,
                hair_extra=p.blur_hair_extra)
        self._pv_masks.evict({t.track_id for t in tracked})

        # Apply blur with live params
        self._blur.reconfigure(p.blur_layers)
        self._blur.apply(blurred, blur_mask)

        # Draw tracking overlays
        for t in tracked:
            bbox = t.face.bbox if t.face is not None else t.bbox
            c = _track_colour(t.track_id)
            poly = polys.get(t.track_id)
            tag = _SOURCE_TAGS.get(t.source, "")
            _draw_outline(tracking, poly, bbox, t.track_id, c, tag)
            _draw_outline(blurred, poly, bbox, t.track_id, c, tag)

        # Pose overlay (tracking panel only): skeletons on top of the outlines,
        # plus the coarse head boxes the tracker is fed.
        for pose in poses:
            _draw_pose(tracking, pose)
        for hb, _score in head_boxes:
            cv2.rectangle(tracking, (int(hb[0]), int(hb[1])),
                          (int(hb[2]), int(hb[3])), _POSE_BOX_COLOUR, 1)

        elapsed = time.perf_counter() - t0
        return orig, tracking, blurred, 1.0 / max(elapsed, 1e-6)

    # ── Export (sequential conversion with live params) ────────────────────────

    def _export_frame(
        self,
        frame: np.ndarray,
        p: Params,
        tracker: KalmanFaceTracker,
        masks: MaskBuilder,
    ) -> tuple[np.ndarray, list, dict[int, Optional[np.ndarray]]]:
        """Process one frame with the persistent offline pipeline.

        Returns (clean blurred frame, tracked faces, polys for overlay drawing).
        """
        self._ensure_model(p.target_size)
        fh, fw = frame.shape[:2]

        refined = self._detect_refined(frame, p)
        tracker.configure(match_iou=p.match_iou, hold_secs=p.hold_secs)
        tracked = tracker.update(
            refined, frame.shape, self._head_provider(frame, p))

        blur_mask = np.zeros((fh, fw), dtype=np.uint8)
        polys: dict[int, Optional[np.ndarray]] = {}
        for t in tracked:
            polys[t.track_id] = masks.add(blur_mask, t,
                                          expand=p.blur_expand,
                                          hair_extra=p.blur_hair_extra)
        masks.evict({t.track_id for t in tracked})

        blurred = frame.copy()
        self._blur.reconfigure(p.blur_layers)
        self._blur.apply(blurred, blur_mask)
        return blurred, tracked, polys

    def _paused_preview(self, frame: np.ndarray) -> None:
        """Re-render the held frame while paused so slider changes show live.

        Uses the single-frame inspect path (fresh tracker) so the persistent
        export tracker/mask state is untouched; blur appearance matches
        what resume will write.
        """
        try:
            results = self._process(frame, self._latest_params())
            self._emit_preview(*results)
        except Exception as exc:  # noqa: BLE001
            _log_exception(exc)
            self.status.emit(f"Preview error: {exc!r}   (full trace: {_CRASH_LOG})")

    def _run_export(self, input_path: str, output_path: str) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.export_finished.emit(False, f"Cannot open: {input_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Match the source's bitrate so the export keeps its format/quality
        # without OpenCV's uncontrolled ~250 Mbps blow-up (the 4 GiB / 2:18
        # corruption). Encoded via FFmpeg → co64-safe even past 4 GiB.
        src_kbps = source_bitrate_kbps(cap)

        ret, frame = cap.read()
        if not ret:
            cap.release()
            self.export_finished.emit(False, "Cannot read first frame")
            return
        fh, fw = frame.shape[:2]

        writer = make_video_writer(
            output_path, fw, fh, fps,
            bitrate_kbps=src_kbps, on_status=self.status.emit,
        )
        if writer is None:
            cap.release()
            self.export_finished.emit(False, f"Cannot create output: {output_path}")
            return

        p0 = self._latest_params()
        tracker = KalmanFaceTracker(fps=fps, match_iou=p0.match_iou,
                                    hold_secs=p0.hold_secs)
        masks = MaskBuilder()
        idx = 0
        cancelled = False

        try:
            while True:
                # Paused: hold position, re-render on param changes.
                while not self._export_run.is_set():
                    if self._take_params_dirty():
                        self._paused_preview(frame)
                    self._export_run.wait(0.05)
                if self._export_cancel.is_set():
                    cancelled = True
                    break

                p = self._latest_params()
                t0 = time.perf_counter()
                blurred, tracked, polys = self._export_frame(
                    frame, p, tracker, masks)
                writer.write(blurred)

                # Preview panels: overlays only on copies, never in the file.
                tracking = frame.copy()
                preview = blurred.copy()
                for t in tracked:
                    bbox = t.face.bbox if t.face is not None else t.bbox
                    c = _track_colour(t.track_id)
                    tag = _SOURCE_TAGS.get(t.source, "")
                    _draw_outline(tracking, polys.get(t.track_id), bbox,
                                  t.track_id, c, tag)
                    _draw_outline(preview, polys.get(t.track_id), bbox,
                                  t.track_id, c, tag)
                elapsed = time.perf_counter() - t0
                self._emit_preview(
                    frame.copy(), tracking, preview, 1.0 / max(elapsed, 1e-6))
                self.export_progress.emit(idx, total)

                idx += 1
                ret, frame = cap.read()
                if not ret:
                    break
        finally:
            cap.release()
            writer.release()

        out_name = Path(output_path).name
        if cancelled:
            self.export_finished.emit(
                False, f"Stopped at frame {idx} — partial file kept: {out_name}")
        else:
            self.export_finished.emit(True, f"Exported {idx} frames → {out_name}")


# ── Reusable widgets ──────────────────────────────────────────────────────────

class VideoPanel(QWidget):
    """Titled panel displaying a single BGR frame, aspect-ratio correct."""

    def __init__(self, title: str) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        hdr = QLabel(title.upper())
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(
            f"font-weight: bold; font-size: 10px; color: {_TEXT_DIM};"
        )
        _letterspace(hdr)

        self._img = QLabel("No frame")
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(
            f"background-color: {_PANEL_BG}; border: 1px solid {_BORDER};"
            f"border-radius: 14px; color: {_TEXT_DIM};"
        )
        self._img.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img.setMinimumSize(240, 160)

        lay.addWidget(hdr)
        lay.addWidget(self._img)

    def set_frame(self, frame: np.ndarray) -> None:
        pix = bgr_to_qpixmap(frame)
        scaled = pix.scaled(
            self._img.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img.setPixmap(scaled)

    def clear(self) -> None:
        self._img.clear()
        self._img.setText("No frame")


class TunableSlider(QWidget):
    """Labeled horizontal slider with live numeric display."""
    changed = pyqtSignal(float)

    def __init__(self, label: str, lo: float, hi: float, default: float,
                 decimals: int = 2) -> None:
        super().__init__()
        self._scale = 10 ** decimals
        self._decimals = decimals

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 2)
        lay.setSpacing(1)

        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {_TEXT_MID}; font-size: 11px;")
        self._val = QLabel(self._fmt(default))
        self._val.setStyleSheet(
            f"color: {_VALUE}; font-family: {_MONO}; font-size: 11px;"
        )
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val.setFixedWidth(42)
        row.addWidget(lbl, stretch=1)
        row.addWidget(self._val)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(round(lo * self._scale))
        self._slider.setMaximum(round(hi * self._scale))
        self._slider.setValue(round(default * self._scale))
        self._slider.valueChanged.connect(self._emit)

        lay.addLayout(row)
        lay.addWidget(self._slider)

    def _fmt(self, v: float) -> str:
        return str(int(v)) if self._decimals == 0 else f"{v:.{self._decimals}f}"

    def _emit(self, raw: int) -> None:
        v = raw / self._scale
        self._val.setText(self._fmt(v))
        self.changed.emit(v)

    def value(self) -> float:
        return self._slider.value() / self._scale

    def int_value(self) -> int:
        return int(self.value())

    def set_value(self, v: float) -> None:
        """Programmatic move; emits changed exactly like a user drag."""
        self._slider.setValue(round(v * self._scale))


class BlurLayerRow(QWidget):
    """One layer of the blur stack: kind selector + strength + remove."""

    changed = pyqtSignal()
    remove_clicked = pyqtSignal(object)   # emits self

    _RANGES = {"gaussian": (3, 151, 71), "pixelate": (2, 40, 10)}

    def __init__(self, kind: str = "gaussian",
                 strength: Optional[int] = None) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._kind = QComboBox()
        self._kind.addItem("Gaussian", "gaussian")
        self._kind.addItem("Pixelate", "pixelate")
        self._kind.setCurrentIndex(0 if kind == "gaussian" else 1)

        lo, hi, default = self._RANGES[kind]
        self._sl = QSlider(Qt.Orientation.Horizontal)
        self._sl.setMinimum(lo)
        self._sl.setMaximum(hi)
        self._sl.setValue(strength if strength is not None else default)

        self._val = QLabel(str(self._sl.value()))
        self._val.setStyleSheet(
            f"color: {_VALUE}; font-family: {_MONO}; font-size: 11px;")
        self._val.setFixedWidth(28)
        self._val.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._rm = QPushButton("✕")
        self._rm.setProperty("ghost", "true")
        self._rm.setFixedWidth(24)
        self._rm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rm.setToolTip("Remove layer")

        self._kind.currentIndexChanged.connect(self._on_kind_change)
        self._sl.valueChanged.connect(self._on_strength_change)
        self._rm.clicked.connect(lambda: self.remove_clicked.emit(self))

        lay.addWidget(self._kind)
        lay.addWidget(self._sl, stretch=1)
        lay.addWidget(self._val)
        lay.addWidget(self._rm)

    def _on_kind_change(self) -> None:
        lo, hi, default = self._RANGES[self._kind.currentData()]
        self._sl.blockSignals(True)
        self._sl.setMinimum(lo)
        self._sl.setMaximum(hi)
        self._sl.setValue(default)
        self._sl.blockSignals(False)
        self._val.setText(str(default))
        self.changed.emit()

    def _on_strength_change(self, v: int) -> None:
        self._val.setText(str(v))
        self.changed.emit()

    def set_removable(self, removable: bool) -> None:
        self._rm.setEnabled(removable)

    def value(self) -> tuple[str, int]:
        return self._kind.currentData(), int(self._sl.value())


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Automated Video Privacy Pipeline")
        self.resize(1360, 860)

        self._params = Params()
        self._applying_preset = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: Optional[str] = None
        self._video_fps = 25.0
        self._total_frames = 0
        self._current_frame_idx = 0
        self._pending_frame: Optional[np.ndarray] = None
        self._exporting = False
        self._export_paused = False

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        # Debounce slider changes so we don't flood the worker
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._submit_current_frame)

        self._worker = ProcessWorker()
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.status.connect(self._on_status)
        self._worker.export_progress.connect(self._on_export_progress)
        self._worker.export_finished.connect(self._on_export_finished)
        self._worker.start()

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        lay.addWidget(self._build_toolbar())
        lay.addWidget(self._build_video_area(), stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {_BORDER}; background-color: {_BORDER};")
        lay.addWidget(sep)

        lay.addWidget(self._build_controls())

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(38)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._open_btn = QPushButton("  Open Video")
        self._open_btn.setProperty("accent", "true")
        self._open_btn.clicked.connect(self._open_video)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.clicked.connect(self._toggle_play)
        self._play_btn.setEnabled(False)

        self._export_btn = QPushButton("⇪  Export")
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._export_btn.setEnabled(False)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_scrub)
        self._frame_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._frame_lbl = QLabel("—  /  —")
        self._frame_lbl.setStyleSheet(
            f"color: {_TEXT_MID}; font-family: {_MONO}; font-size: 11px;")
        self._frame_lbl.setFixedWidth(84)

        self._fps_lbl = QLabel("— fps")
        self._fps_lbl.setStyleSheet(
            f"color: {_VALUE}; font-family: {_MONO}; font-size: 11px;")
        self._fps_lbl.setFixedWidth(60)

        self._status_lbl = QLabel("Open a video to begin")
        self._status_lbl.setStyleSheet(f"color: {_TEXT_MID}; font-size: 11px;")

        lay.addWidget(self._open_btn)
        lay.addWidget(self._play_btn)
        lay.addWidget(self._export_btn)
        lay.addWidget(self._frame_slider, stretch=1)
        lay.addWidget(self._frame_lbl)
        lay.addWidget(self._fps_lbl)
        lay.addWidget(self._status_lbl)
        return bar

    def _build_video_area(self) -> QWidget:
        area = QWidget()
        lay = QHBoxLayout(area)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._panel_orig  = VideoPanel("Before")
        self._panel_track = VideoPanel("Tracking")
        self._panel_blur  = VideoPanel("After  ·  Blurred")

        lay.addWidget(self._panel_orig,  stretch=1)
        lay.addWidget(self._panel_track, stretch=1)
        lay.addWidget(self._panel_blur,  stretch=1)
        return area

    def _build_controls(self) -> QWidget:
        ctrl = QWidget()
        lay = QVBoxLayout(ctrl)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lay.addWidget(self._build_presets_bar())

        # Full tunable control, revealed by the Advanced toggle.
        self._advanced = QWidget()
        adv = QHBoxLayout(self._advanced)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(8)
        adv.addWidget(self._build_detection_group(), stretch=5)
        adv.addWidget(self._build_blur_group(),      stretch=7)
        adv.addWidget(self._build_tracking_group(),  stretch=3)
        self._advanced.setVisible(False)
        lay.addWidget(self._advanced)
        return ctrl

    def _build_presets_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(6)

        cap = QLabel("PRESETS")
        cap.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; font-weight: bold;")
        _letterspace(cap)
        lay.addWidget(cap)
        lay.addSpacing(4)

        self._preset_btns: dict[str, QPushButton] = {}
        for name, (tip, _p) in PRESETS.items():
            btn = QPushButton(name)
            btn.setProperty("chip", "true")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _c, n=name: self._apply_preset(n))
            self._preset_btns[name] = btn
            lay.addWidget(btn)
        self._preset_btns["Balanced"].setChecked(True)

        # Lights up when sliders diverge from every preset.
        self._custom_lbl = QLabel("custom")
        self._custom_lbl.setStyleSheet(
            f"color: {_VALUE}; font-size: 10px; font-style: italic;")
        self._custom_lbl.setVisible(False)
        lay.addWidget(self._custom_lbl)

        lay.addStretch()

        self._adv_btn = QPushButton("Advanced  ▾")
        self._adv_btn.setProperty("ghost", "true")
        self._adv_btn.setCheckable(True)
        self._adv_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._adv_btn.toggled.connect(self._toggle_advanced)
        lay.addWidget(self._adv_btn)
        return bar

    def _build_detection_group(self) -> QGroupBox:
        grp = QGroupBox("DETECTION")
        lay = QGridLayout(grp)
        lay.setSpacing(6)

        size_lbl = QLabel("Target Size")
        size_lbl.setStyleSheet(f"color: {_TEXT_MID}; font-size: 11px;")
        self._size_combo = QComboBox()
        for sz in (320, 480, 640, 800, 1024):
            self._size_combo.addItem(f"{sz} px", sz)
        self._size_combo.setCurrentIndex(2)
        self._size_combo.currentIndexChanged.connect(self._on_target_size_change)

        self._det_score_sl  = TunableSlider("Det Score",     0.10, 1.00, 0.55)
        self._face_aspect_sl = TunableSlider("Face Aspect",  0.10, 1.50, 0.40)
        self._closeup_sl     = TunableSlider("Close-up Thr", 0.10, 1.00, 0.60)

        for sl in (self._det_score_sl, self._face_aspect_sl, self._closeup_sl):
            sl.changed.connect(self._on_param_change)

        lay.addWidget(size_lbl,              0, 0)
        lay.addWidget(self._size_combo,      0, 1)
        lay.addWidget(self._det_score_sl,    1, 0, 1, 2)
        lay.addWidget(self._face_aspect_sl,  2, 0, 1, 2)
        lay.addWidget(self._closeup_sl,      3, 0, 1, 2)
        return grp

    def _build_blur_group(self) -> QGroupBox:
        grp = QGroupBox("BLUR")
        lay = QHBoxLayout(grp)
        lay.setSpacing(12)

        self._expand_sl = TunableSlider("Hull Expand", 0.00, 2.00, 0.45)
        self._hair_sl   = TunableSlider("Hair Extra",  0.00, 3.00, 0.90)
        for sl in (self._expand_sl, self._hair_sl):
            sl.changed.connect(self._on_param_change)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self._expand_sl)
        left.addWidget(self._hair_sl)
        left.addStretch()

        # Stackable blur layers, applied top to bottom.
        cap = QLabel("LAYER STACK")
        cap.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; font-weight: bold;")
        _letterspace(cap)
        self._layers_box = QVBoxLayout()
        self._layers_box.setSpacing(2)
        self._add_layer_btn = QPushButton("+ Add Layer")
        self._add_layer_btn.setProperty("ghost", "true")
        self._add_layer_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_layer_btn.clicked.connect(self._on_add_layer)

        right = QVBoxLayout()
        right.setSpacing(4)
        right.addWidget(cap)
        right.addLayout(self._layers_box)
        right.addWidget(self._add_layer_btn,
                        alignment=Qt.AlignmentFlag.AlignLeft)
        right.addStretch()

        lay.addLayout(left, stretch=4)
        lay.addLayout(right, stretch=6)

        self._layer_rows: list[BlurLayerRow] = []
        self._set_blur_layers(self._params.blur_layers)
        return grp

    # ── blur layer stack management ────────────────────────────────────────────

    _MAX_LAYERS = 5

    def _new_layer_row(self, kind: str, strength: Optional[int]) -> BlurLayerRow:
        row = BlurLayerRow(kind, strength)
        row.changed.connect(self._on_param_change)
        row.remove_clicked.connect(self._on_remove_layer)
        self._layer_rows.append(row)
        self._layers_box.addWidget(row)
        return row

    def _refresh_layer_buttons(self) -> None:
        removable = len(self._layer_rows) > 1
        for row in self._layer_rows:
            row.set_removable(removable)
        self._add_layer_btn.setEnabled(len(self._layer_rows) < self._MAX_LAYERS)

    def _set_blur_layers(self, layers: tuple[tuple[str, int], ...]) -> None:
        """Rebuild the rows without emitting per-row change signals."""
        for row in self._layer_rows:
            self._layers_box.removeWidget(row)
            row.deleteLater()
        self._layer_rows = []
        for kind, strength in (layers or DEFAULT_BLUR_LAYERS):
            self._new_layer_row(kind, strength)
        self._refresh_layer_buttons()

    def _blur_layers_value(self) -> tuple[tuple[str, int], ...]:
        return tuple(row.value() for row in self._layer_rows)

    def _on_add_layer(self) -> None:
        if len(self._layer_rows) >= self._MAX_LAYERS:
            return
        self._new_layer_row("gaussian", None)
        self._refresh_layer_buttons()
        self._on_param_change()

    def _on_remove_layer(self, row: BlurLayerRow) -> None:
        if len(self._layer_rows) <= 1:
            return
        self._layer_rows.remove(row)
        self._layers_box.removeWidget(row)
        row.deleteLater()
        self._refresh_layer_buttons()
        self._on_param_change()

    def _build_tracking_group(self) -> QGroupBox:
        grp = QGroupBox("TRACKING")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        self._iou_sl = TunableSlider("Match IoU", 0.05, 0.95, 0.30)
        self._iou_sl.changed.connect(self._on_param_change)
        # How long a lost face keeps its blur, coasting on Kalman prediction.
        self._hold_sl = TunableSlider("Hold (s)", 0.0, 5.0, 2.0, decimals=1)
        self._hold_sl.changed.connect(self._on_param_change)

        self._pose_btn = QPushButton("Pose Assist")
        self._pose_btn.setProperty("chip", "true")
        self._pose_btn.setCheckable(True)
        self._pose_btn.setChecked(True)
        self._pose_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pose_btn.setToolTip(
            "Track the head via body pose when the face detector loses it\n"
            "(person turns away / looks down) so the blur stays put.")
        self._pose_btn.toggled.connect(lambda _c: self._on_param_change())

        lay.addWidget(self._iou_sl)
        lay.addWidget(self._hold_sl)
        lay.addWidget(self._pose_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        lay.addStretch()
        return grp

    # ── Video handling ────────────────────────────────────────────────────────

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            filter="Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if not path:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setText("▶  Play")
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._on_status(f"Cannot open: {path}")
            return
        self._video_path = path
        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_slider.setMaximum(max(0, self._total_frames - 1))
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(0)
        self._frame_slider.blockSignals(False)
        self._play_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._on_status(f"{Path(path).name}   {self._total_frames} frames")
        self._go_to_frame(0)

    def _go_to_frame(self, idx: int) -> None:
        if self._cap is None:
            return
        self._current_frame_idx = idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        self._pending_frame = frame
        self._frame_lbl.setText(f"{idx} / {self._total_frames - 1}")
        if self._play_timer.isActive():
            # Play ticks arrive faster than the debounce delay, which would
            # postpone rendering until pause. Submit directly — the worker's
            # single-slot queue keeps only the newest frame anyway.
            self._submit_current_frame()
        else:
            self._debounce.start(60)

    def _on_frame_scrub(self, val: int) -> None:
        if self._play_timer.isActive():
            return
        self._go_to_frame(val)

    def _toggle_play(self) -> None:
        if self._exporting:
            if self._export_paused:
                self._worker.resume_export()
                self._export_paused = False
                self._play_btn.setText("⏸  Pause")
                self._on_status("Exporting…")
            else:
                self._worker.pause_export()
                self._export_paused = True
                self._play_btn.setText("▶  Resume")
                self._on_status("Paused — tune sliders, then Resume")
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setText("▶  Play")
        else:
            fps = 25.0
            if self._cap:
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
            self._play_timer.start(max(1, int(1000 / fps)))
            self._play_btn.setText("⏸  Pause")

    def _play_tick(self) -> None:
        if self._cap is None:
            return
        nxt = self._current_frame_idx + 1
        if nxt >= self._total_frames:
            self._play_timer.stop()
            self._play_btn.setText("▶  Play")
            return
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(nxt)
        self._frame_slider.blockSignals(False)
        self._go_to_frame(nxt)

    # ── Export handling ───────────────────────────────────────────────────────

    def _on_export_clicked(self) -> None:
        if self._exporting:
            self._worker.cancel_export()
            return
        if not self._video_path:
            return
        src = Path(self._video_path)
        default = str(src.with_name(f"{src.stem}_blurred.mp4"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Blurred Video", default, "Videos (*.mp4)")
        if not path:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
        self._debounce.stop()

        self._exporting = True
        self._export_paused = False
        self._open_btn.setEnabled(False)
        self._frame_slider.setEnabled(False)
        self._export_btn.setText("■  Stop")
        self._play_btn.setText("⏸  Pause")
        self._play_btn.setEnabled(True)

        self._worker.update_params(self._params)
        self._worker.start_export(self._video_path, path)
        self._on_status("Exporting…")

    def _on_export_progress(self, current: int, total: int) -> None:
        self._frame_slider.blockSignals(True)
        self._frame_slider.setMaximum(max(0, total - 1))
        self._frame_slider.setValue(current)
        self._frame_slider.blockSignals(False)
        self._frame_lbl.setText(f"{current} / {max(0, total - 1)}")

    def _on_export_finished(self, ok: bool, msg: str) -> None:
        self._exporting = False
        self._export_paused = False
        self._export_btn.setText("⇪  Export")
        self._open_btn.setEnabled(True)
        self._frame_slider.setEnabled(True)
        self._play_btn.setText("▶  Play")
        self._play_btn.setEnabled(self._cap is not None)
        # Restore the scrub slider to the inspected frame and re-render it.
        self._frame_slider.blockSignals(True)
        self._frame_slider.setMaximum(max(0, self._total_frames - 1))
        self._frame_slider.setValue(self._current_frame_idx)
        self._frame_slider.blockSignals(False)
        self._on_status(msg)
        if self._cap is not None:
            self._go_to_frame(self._current_frame_idx)

    # ── Param handling ────────────────────────────────────────────────────────

    def _toggle_advanced(self, checked: bool) -> None:
        self._advanced.setVisible(checked)
        self._adv_btn.setText("Advanced  ▴" if checked else "Advanced  ▾")

    def _apply_preset(self, name: str) -> None:
        _tip, p = PRESETS[name]
        self._applying_preset = True
        try:
            idx = self._size_combo.findData(p.target_size)
            if idx >= 0:
                self._size_combo.setCurrentIndex(idx)
            self._det_score_sl.set_value(p.det_score)
            self._face_aspect_sl.set_value(p.face_aspect)
            self._closeup_sl.set_value(p.close_up_ratio)
            self._expand_sl.set_value(p.blur_expand)
            self._hair_sl.set_value(p.blur_hair_extra)
            self._set_blur_layers(p.blur_layers)
            self._iou_sl.set_value(p.match_iou)
            self._hold_sl.set_value(p.hold_secs)
            self._pose_btn.setChecked(p.pose_assist)
            # Re-submit even when no slider actually moved (e.g. re-click).
            self._on_param_change()
        finally:
            self._applying_preset = False
        for n, btn in self._preset_btns.items():
            btn.setChecked(n == name)
        self._custom_lbl.setVisible(False)

    def _mark_custom(self) -> None:
        for btn in self._preset_btns.values():
            btn.setChecked(False)
        self._custom_lbl.setVisible(True)

    def _on_target_size_change(self) -> None:
        self._params.target_size = self._size_combo.currentData()
        self._on_param_change()

    def _on_param_change(self, _val: float = 0.0) -> None:
        self._params.det_score      = self._det_score_sl.value()
        self._params.face_aspect    = self._face_aspect_sl.value()
        self._params.close_up_ratio = self._closeup_sl.value()
        self._params.blur_expand    = self._expand_sl.value()
        self._params.blur_hair_extra = self._hair_sl.value()
        self._params.blur_layers    = self._blur_layers_value()
        self._params.match_iou      = self._iou_sl.value()
        self._params.hold_secs      = self._hold_sl.value()
        self._params.pose_assist    = self._pose_btn.isChecked()
        self._worker.update_params(self._params)
        if not self._applying_preset:
            self._mark_custom()
        if self._exporting:
            return  # export loop re-reads params each frame; it owns the panels
        self._debounce.start(120)

    def _submit_current_frame(self) -> None:
        if self._exporting:
            return
        if self._pending_frame is not None:
            self._worker.submit(self._pending_frame, self._params,
                                self._current_frame_idx, self._video_fps)

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_preview_ready(self) -> None:
        preview = self._worker.take_preview()
        if preview is not None:
            self._on_result(*preview)

    def _on_result(
        self,
        orig: np.ndarray,
        tracking: np.ndarray,
        blurred: np.ndarray,
        fps: float,
    ) -> None:
        self._panel_orig.set_frame(orig)
        self._panel_track.set_frame(tracking)
        self._panel_blur.set_frame(blurred)
        self._fps_lbl.setText(f"{fps:.1f} fps")

    def _on_status(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._play_timer.stop()
        self._worker.cancel_export()
        self._worker.stop()
        self._worker.wait(3000)
        if self._cap:
            self._cap.release()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(splash: QSplashScreen | None = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    if splash is not None:
        splash.showMessage("Preparing interface…")
        app.processEvents()
    win = MainWindow()
    win.show()
    if splash is not None:
        splash.finish(win)
    # Bootloader splash from a PyInstaller --splash build, if present.
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
