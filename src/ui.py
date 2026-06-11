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
    QSizePolicy, QComboBox, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

from libs.face_app import FaceApp
from libs.smoother import LandmarkSmoother
from libs.tracker import ByteTrackWrapper
from libs.utils import (
    BlurPipeline,
    add_bbox_mask,
    add_face_mask,
    best_onnx_providers,
    crop_face_patch,
    unproject_landmark,
)

CLOSE_UP_TARGET_SIZE = 1024
_TRACK_COLOURS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 200, 100),
    (200, 100, 0), (100, 0, 200),
]

# ── Catppuccin-inspired dark theme ────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background-color: #11111b;
    color: #cdd6f4;
    font-family: "Inter", "Segoe UI", "Ubuntu", sans-serif;
    font-size: 11px;
}
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 16px;
    padding: 8px 6px 6px 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #cba6f7;
    font-size: 9px;
    font-weight: bold;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
QSlider::groove:horizontal {
    height: 3px;
    background: #313244;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #cba6f7;
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid #1e1e2e;
}
QSlider::sub-page:horizontal {
    background: #cba6f7;
    border-radius: 2px;
}
QPushButton {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 14px;
    color: #cdd6f4;
    min-width: 80px;
}
QPushButton:hover {
    background-color: #313244;
    border-color: #cba6f7;
}
QPushButton:pressed {
    background-color: #cba6f7;
    color: #1e1e2e;
}
QPushButton[accent="true"] {
    background-color: #cba6f7;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton[accent="true"]:hover {
    background-color: #d4b8ff;
}
QLabel {
    color: #cdd6f4;
}
QComboBox {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 3px 8px;
    color: #cdd6f4;
    min-width: 64px;
}
QComboBox:hover { border-color: #cba6f7; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #1e1e2e;
    selection-background-color: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
}
"""


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class Params:
    target_size: int = 640
    det_score: float = 0.55
    face_aspect: float = 0.40
    close_up_ratio: float = 0.60
    blur_expand: float = 0.45
    blur_hair_extra: float = 0.90
    blur_k: int = 71        # must remain odd
    blur_block: int = 10
    match_iou: float = 0.30


# ── Helper functions ──────────────────────────────────────────────────────────

def _track_colour(tid: int) -> tuple[int, int, int]:
    return _TRACK_COLOURS[tid % len(_TRACK_COLOURS)]


def _draw_outline(
    frame: np.ndarray,
    poly: Optional[np.ndarray],
    bbox: np.ndarray,
    tid: int,
    colour: tuple[int, int, int],
) -> None:
    if poly is not None:
        cv2.polylines(frame, [poly], True, colour, 1, cv2.LINE_AA)
        x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
    else:
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
        x, y = x1, y1
    cv2.putText(frame, f"id:{tid}", (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


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
        self._pending: Optional[tuple[np.ndarray, Params]] = None
        self._wake = threading.Event()
        self._running = True
        self._app: Optional[FaceApp] = None
        self._model_size = -1
        self._blur = BlurPipeline()
        self._preview_lock = threading.Lock()
        self._preview: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = None
        # Export state — latest params are re-read every frame so slider
        # changes apply live mid-conversion.
        self._params = Params()
        self._params_dirty = False
        self._export_request: Optional[tuple[str, str]] = None
        self._export_run = threading.Event()    # cleared = paused
        self._export_cancel = threading.Event()

    def submit(self, frame: np.ndarray, params: Params) -> None:
        with self._lock:
            self._pending = (frame.copy(), replace(params))
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
            frame, params = job
            try:
                results = self._process(frame, params)
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

    def _process(
        self, frame: np.ndarray, p: Params
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        self._ensure_model(p.target_size)

        fh, fw = frame.shape[:2]
        orig = frame.copy()
        tracking = frame.copy()
        blurred = frame.copy()

        refined = self._detect_refined(frame, p)

        # Tracking (fresh per-frame for stable single-frame inspection)
        tracked = ByteTrackWrapper(match_iou=p.match_iou).update(refined, frame)

        # Build blur mask
        blur_mask = np.zeros((fh, fw), dtype=np.uint8)
        polys: dict[int, Optional[np.ndarray]] = {}
        for tid, face, tbox in tracked:
            if face is not None and face.landmark_2d_106 is not None:
                poly = add_face_mask(blur_mask, face.landmark_2d_106,
                                     expand=p.blur_expand,
                                     hair_extra=p.blur_hair_extra)
                polys[tid] = poly
            else:
                add_bbox_mask(blur_mask, tbox)
                polys[tid] = None

        # Apply blur with live params
        self._blur.reconfigure(p.blur_k, p.blur_block)
        self._blur.apply(blurred, blur_mask)

        # Draw tracking overlays
        for tid, face, tbox in tracked:
            bbox = face.bbox if face is not None else tbox
            c = _track_colour(tid)
            poly = polys.get(tid)
            _draw_outline(tracking, poly, bbox, tid, c)
            _draw_outline(blurred, poly, bbox, tid, c)

        elapsed = time.perf_counter() - t0
        return orig, tracking, blurred, 1.0 / max(elapsed, 1e-6)

    # ── Export (sequential conversion with live params) ────────────────────────

    def _export_frame(
        self,
        frame: np.ndarray,
        p: Params,
        tracker: ByteTrackWrapper,
        smoother: LandmarkSmoother,
        last_landmarks: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, list, dict[int, Optional[np.ndarray]]]:
        """Process one frame with the persistent offline pipeline.

        Returns (clean blurred frame, tracked triples, polys for overlay drawing).
        """
        self._ensure_model(p.target_size)
        fh, fw = frame.shape[:2]

        refined = self._detect_refined(frame, p)
        tracker.set_match_iou(p.match_iou)
        tracked = tracker.update(refined, frame)

        blur_mask = np.zeros((fh, fw), dtype=np.uint8)
        polys: dict[int, Optional[np.ndarray]] = {}
        for tid, face, tbox in tracked:
            if face is not None and face.landmark_2d_106 is not None:
                smoothed = smoother.update(
                    tid, face.landmark_2d_106, float(face.det_score))
                last_landmarks[tid] = smoothed
                polys[tid] = add_face_mask(blur_mask, smoothed,
                                           expand=p.blur_expand,
                                           hair_extra=p.blur_hair_extra)
            elif tid in last_landmarks:
                # Ghost track: hold the last smoothed landmarks for continuity.
                polys[tid] = add_face_mask(blur_mask, last_landmarks[tid],
                                           expand=p.blur_expand,
                                           hair_extra=p.blur_hair_extra)
            else:
                add_bbox_mask(blur_mask, tbox)
                polys[tid] = None

        active = {t for t, _, _ in tracked}
        for sid in list(last_landmarks):
            if sid not in active:
                del last_landmarks[sid]

        blurred = frame.copy()
        self._blur.reconfigure(p.blur_k, p.blur_block)
        self._blur.apply(blurred, blur_mask)
        return blurred, tracked, polys

    def _paused_preview(self, frame: np.ndarray) -> None:
        """Re-render the held frame while paused so slider changes show live.

        Uses the single-frame inspect path (fresh tracker) so the persistent
        export tracker/smoother state is untouched; blur appearance matches
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

        ret, frame = cap.read()
        if not ret:
            cap.release()
            self.export_finished.emit(False, "Cannot read first frame")
            return
        fh, fw = frame.shape[:2]

        fourcc = cv2.VideoWriter.fourcc(*"avc1")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
        if not writer.isOpened():
            cap.release()
            self.export_finished.emit(False, f"Cannot create output: {output_path}")
            return

        tracker = ByteTrackWrapper(fps=fps, match_iou=self._latest_params().match_iou)
        smoother = LandmarkSmoother()
        last_landmarks: dict[int, np.ndarray] = {}
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
                    frame, p, tracker, smoother, last_landmarks)
                writer.write(blurred)

                # Preview panels: overlays only on copies, never in the file.
                tracking = frame.copy()
                preview = blurred.copy()
                for tid, face, tbox in tracked:
                    bbox = face.bbox if face is not None else tbox
                    c = _track_colour(tid)
                    _draw_outline(tracking, polys.get(tid), bbox, tid, c)
                    _draw_outline(preview, polys.get(tid), bbox, tid, c)
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

        hdr = QLabel(title)
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(
            "font-weight: bold; font-size: 10px; color: #cba6f7;"
            "letter-spacing: 1.5px; text-transform: uppercase;"
        )

        self._img = QLabel("No frame")
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(
            "background-color: #0d0d1a; border-radius: 8px; color: #45475a;"
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
        lbl.setStyleSheet("color: #a6adc8; font-size: 10px;")
        self._val = QLabel(self._fmt(default))
        self._val.setStyleSheet(
            "color: #a6e3a1; font-family: monospace; font-size: 10px;"
        )
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val.setFixedWidth(38)
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


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Face Blur Pipeline Inspector")
        self.resize(1360, 860)

        self._params = Params()
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: Optional[str] = None
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
        sep.setStyleSheet("color: #313244;")
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
        self._frame_lbl.setStyleSheet("color: #6c7086; font-size: 10px;")
        self._frame_lbl.setFixedWidth(84)

        self._fps_lbl = QLabel("— fps")
        self._fps_lbl.setStyleSheet("color: #6c7086; font-size: 10px;")
        self._fps_lbl.setFixedWidth(60)

        self._status_lbl = QLabel("Open a video to begin")
        self._status_lbl.setStyleSheet("color: #585b70; font-size: 10px;")

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
        lay = QHBoxLayout(ctrl)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lay.addWidget(self._build_detection_group(), stretch=5)
        lay.addWidget(self._build_blur_group(),      stretch=6)
        lay.addWidget(self._build_tracking_group(),  stretch=2)
        return ctrl

    def _build_detection_group(self) -> QGroupBox:
        grp = QGroupBox("Detection")
        lay = QGridLayout(grp)
        lay.setSpacing(6)

        size_lbl = QLabel("Target Size")
        size_lbl.setStyleSheet("color: #a6adc8; font-size: 10px;")
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
        grp = QGroupBox("Blur")
        lay = QGridLayout(grp)
        lay.setSpacing(6)
        lay.setColumnStretch(0, 1)
        lay.setColumnStretch(1, 1)

        self._expand_sl = TunableSlider("Hull Expand",    0.00, 2.00, 0.45)
        self._hair_sl   = TunableSlider("Hair Extra",     0.00, 3.00, 0.90)
        self._k_sl      = TunableSlider("Gaussian K",     3,    151,  71,  decimals=0)
        self._block_sl  = TunableSlider("Pixelate Block", 2,    40,   10,  decimals=0)

        for sl in (self._expand_sl, self._hair_sl, self._k_sl, self._block_sl):
            sl.changed.connect(self._on_param_change)

        lay.addWidget(self._expand_sl, 0, 0)
        lay.addWidget(self._hair_sl,   1, 0)
        lay.addWidget(self._k_sl,      0, 1)
        lay.addWidget(self._block_sl,  1, 1)
        return grp

    def _build_tracking_group(self) -> QGroupBox:
        grp = QGroupBox("Tracking")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        self._iou_sl = TunableSlider("Match IoU", 0.05, 0.95, 0.30)
        self._iou_sl.changed.connect(self._on_param_change)
        lay.addWidget(self._iou_sl)
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

    def _on_target_size_change(self) -> None:
        self._params.target_size = self._size_combo.currentData()
        self._on_param_change()

    def _on_param_change(self, _val: float = 0.0) -> None:
        self._params.det_score      = self._det_score_sl.value()
        self._params.face_aspect    = self._face_aspect_sl.value()
        self._params.close_up_ratio = self._closeup_sl.value()
        self._params.blur_expand    = self._expand_sl.value()
        self._params.blur_hair_extra = self._hair_sl.value()
        self._params.blur_k         = self._k_sl.int_value() | 1   # ensure odd
        self._params.blur_block     = max(2, self._block_sl.int_value())
        self._params.match_iou      = self._iou_sl.value()
        self._worker.update_params(self._params)
        if self._exporting:
            return  # export loop re-reads params each frame; it owns the panels
        self._debounce.start(120)

    def _submit_current_frame(self) -> None:
        if self._exporting:
            return
        if self._pending_frame is not None:
            self._worker.submit(self._pending_frame, self._params)

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

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
