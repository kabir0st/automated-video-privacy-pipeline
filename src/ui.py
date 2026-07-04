"""Modern inspector UI: open video → view Before | Tracking | After frames with live tunables.

Usage:
    uv run python src/ui.py
"""

from __future__ import annotations

import faulthandler
import queue
import sys
import tempfile
import time
import threading
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

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

from libs.detector import Detections, HeadDetector, NEG_HAND
from libs.evidence import (PROFILES, Ev, GateDebug, gate_face_candidates,
                           sustain_pool, weak_faces as gate_weak_faces)
from libs.head_tracker import HeadTracker, TrackObs
from libs.models import preflight as preflight_models
from libs.pose import PersonPose, PoseEstimator
from libs.scrfd import WITNESS_FLOOR, ScrfdDetector, kps_plausible
from libs.tracklets import (PostParams, TrackRecorder, build_table,
                            clean_tracklets, verify_tracklets)
from libs.utils import (
    DEFAULT_BLUR_LAYERS,
    BlurPipeline,
    bloom_face_box,
    debug_log,
    render_head_mask,
)
from libs.video_writer import make_video_writer, source_bitrate_kbps
from splash import show_splash, update as splash_update

# Preview panels render at this long edge — computing/drawing at 4K would be
# wasted on panels a third of the window wide. Detection is unaffected (the
# detector resizes to its own fixed input internally).
_DISPLAY_LONG_EDGE = 1280
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
    # Detection / track gating. det_conf is the one knob that matters: a
    # gated face candidate (libs.evidence.gate_face_candidates) at or above
    # it can spawn and drive a blur; the band between det_conf_low and
    # det_conf only *sustains* an existing blur through occlusion (BYTE
    # association) and can never start one. Coverage is bought with the
    # evidence profile and holds below, not by lowering this floor — a
    # weaker confidence bar is exactly what let skin/fabric misreads through
    # before this gate existed.
    det_conf: float = 0.50
    det_conf_low: float = 0.10
    # How much anatomical/consensus evidence a face candidate needs to clear
    # the gate and, offline, the evidence ledger — a key into
    # libs.evidence.PROFILES ("balanced" / "max" / "strict").
    evidence_profile: str = "balanced"
    # Consecutive hits before a new track may blur — kills 1-frame false
    # positives at the cost of a few frames of onset (repaired offline by the
    # export's end-extension).
    min_hits: int = 3
    # How long a lost track keeps coasting as a bridge *candidate* (it is not
    # blurred while coasting; the offline pass decides whether the gap joins).
    max_age_s: float = 1.75
    # Offline cleanup (export pass 1 → 2).
    bridge_gap_s: float = 1.5    # max detection gap to interpolate across
    min_track_s: float = 0.25    # tracklets with fewer hits are noise
    smooth_win_s: float = 0.5    # zero-phase SavGol window
    # How long a track's last matched face keeps steering the face-only blur
    # (live preview) or may be interpolated across (export's face-gap fill)
    # before falling back to the wider head box / being reported as a
    # genuine coverage gap.
    face_hold_s: float = 0.8
    face_gap_bridge_s: float = 1.0
    # Mask geometry.
    mask_pad: float = 0.18       # ellipse expansion per side of the head box
    mask_feather: float = 0.12   # edge feather as a fraction of head diagonal
    # Live preview only: how long a lost head keeps its blur on screen.
    preview_hold_s: float = 0.3
    # Also detect on ±90°-rotated frames and merge — recovers sideways heads
    # (lying down / bed angles) that upright-trained detectors miss. ~3× the
    # (single, small) detection cost.
    rot_assist: bool = True
    # Blur target: "face" = the matched face box bloomed to catch a bit of
    # hair — blurred only while face evidence is fresh/interpolated (a
    # turned-away head with no face evidence stays unblurred by design); or
    # "head" = the whole tracked head-scale anchor box (safest, biggest —
    # never loses coverage, at the cost of a much bigger blur).
    blur_region: str = "face"
    # Ordered blur stack: ("gaussian", kernel) / ("pixelate", block).
    blur_layers: tuple[tuple[str, int], ...] = DEFAULT_BLUR_LAYERS


# ── Presets — curated Params bundles; Advanced exposes every value ───────────

PRESETS: dict[str, tuple[str, Params]] = {
    "Balanced": (
        "Sensible defaults for most footage",
        Params(),
    ),
    "Max Privacy": (
        "Catch every head and blur hard — favours coverage over precision. "
        "Coverage comes from a looser evidence bar and longer holds, never "
        "from trusting the detector's raw confidence more.",
        Params(det_conf_low=0.05, evidence_profile="max", min_hits=2,
               max_age_s=2.5, bridge_gap_s=2.5, min_track_s=0.15,
               face_hold_s=1.5, face_gap_bridge_s=2.0,
               mask_pad=0.30, mask_feather=0.15,
               blur_layers=(("gaussian", 99), ("pixelate", 16))),
    ),
    "Strict": (
        "Fewer false blurs — higher confidence bar, more evidence required, "
        "shorter bridging",
        Params(det_conf=0.60, evidence_profile="strict", min_hits=5,
               bridge_gap_s=0.8, min_track_s=0.4, face_hold_s=0.5,
               mask_pad=0.15),
    ),
}

# TRACKING panel palette: layers distinct from the per-track colours.
_BODY_BOX_COLOUR = (180, 60, 180)      # faint magenta — context only
_HEAD_DET_COLOUR = (0, 220, 0)         # confident head-class detections
_HEAD_LOW_COLOUR = (140, 140, 140)     # low-score band (evidence only)
_FACE_DET_COLOUR = (255, 220, 0)       # face-class detections (cyan-ish in BGR)
_PART_COLOUR = (0, 255, 255)           # eye/nose/mouth part evidence (yellow dot)
_NEG_COLOUR = (0, 128, 255)            # hand/foot veto evidence (orange)
_SCRFD_FACE_COLOUR = (255, 0, 170)     # SCRFD witness faces (violet)
_SCRFD_REJ_COLOUR = (60, 60, 230)      # landmark-implausible faces (red)
_REJECTED_COLOUR = (0, 0, 160)         # gate-rejected face claims (dark red)
_POSE_SKELETON_COLOUR = (255, 255, 255)  # pose joints/bones (white)
_POSE_ANCHOR_COLOUR = (0, 255, 128)     # pose-derived head anchor box (spring green)


# ── Helper functions ──────────────────────────────────────────────────────────

def _track_colour(tid: int) -> tuple[int, int, int]:
    return _TRACK_COLOURS[tid % len(_TRACK_COLOURS)]


def _draw_box(
    frame: np.ndarray,
    box: np.ndarray,
    colour: tuple[int, int, int],
    thickness: int = 1,
    label: str = "",
) -> None:
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    if label:
        cv2.putText(frame, label, (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def _draw_mask_ellipse(
    frame: np.ndarray,
    box: np.ndarray,
    pad: float,
    colour: tuple[int, int, int],
    label: str = "",
) -> None:
    """Outline of the blur ellipse render_head_mask will paint for this box,
    so the AFTER panel shows exactly where the blur lands."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w, h = x2 - x1, y2 - y1
    cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
    ax = max(1, int(round(w / 2 * (1.0 + 2.0 * pad))))
    ay = max(1, int(round(h / 2 * (1.0 + 2.0 * pad))))
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, colour, 1, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (cx - ax, max(10, cy - ay - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def _obs_tag(o: TrackObs) -> str:
    if not o.confirmed:
        return "·tent"
    if not o.hit:
        return "·coast"
    return ""


def _blur_target_box(o: TrackObs, region: str, face_hold: int) -> np.ndarray:
    """Live-preview blur target for one track (approximate by design — the
    export instead records the raw face box + freshness and lets the
    offline pass in libs/tracklets.py interpolate short gaps precisely).
    In face mode: the bloomed face box while evidence is fresh, otherwise
    the whole head/anchor box, so a momentary detector flicker doesn't blank
    the preview blur outright. Always the head/anchor box in head mode."""
    if region == "face" and o.face_box is not None \
            and o.face_age <= face_hold:
        return bloom_face_box(o.face_box)
    return o.box


def _reject_reason(flags: int) -> str:
    """Short label for a gate-rejected face claim's dominant failure — see
    libs/evidence.py's ``Ev`` flags."""
    f = Ev(flags)
    if f & Ev.VETO_FOOT:
        return "rej:foot"
    if f & Ev.VETO_HAND:
        return "rej:hand"
    if f & Ev.VETO_AXIS:
        return "rej:axis"
    if not (f & (Ev.HEAD_ANCHOR | Ev.POSE_ANCHOR | Ev.CLOSEUP)):
        return "rej:anchor"
    return "rej:parts"


_POSE_BONES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # shoulders, arms
    (5, 11), (6, 12), (11, 12),                    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # legs
)


def _draw_poses(tracking: np.ndarray, poses: "Sequence[PersonPose]") -> None:
    """Skeleton + head-anchor box per pose, so the TRACKING panel shows the
    anatomical context the gate's HEAD_ANCHOR/POSE_ANCHOR/AXIS_OK flags are
    reasoning about (see libs/pose.py, libs/evidence.py)."""
    for p in poses:
        kpts = p.kpts
        for i, j in _POSE_BONES:
            if kpts[i, 2] <= 0.3 or kpts[j, 2] <= 0.3:
                continue
            pt1 = (int(round(kpts[i, 0])), int(round(kpts[i, 1])))
            pt2 = (int(round(kpts[j, 0])), int(round(kpts[j, 1])))
            cv2.line(tracking, pt1, pt2, _POSE_SKELETON_COLOUR, 1, cv2.LINE_AA)
        for x, y, s in kpts:
            if s > 0.3:
                cv2.circle(tracking, (int(round(x)), int(round(y))), 2,
                          _POSE_SKELETON_COLOUR, -1, cv2.LINE_AA)
        if p.anchor is not None:
            _draw_box(tracking, p.anchor, _POSE_ANCHOR_COLOUR, 1, "pose")


def _draw_tracking_overlay(
    tracking: np.ndarray,
    dets: Detections,
    obs: list[TrackObs],
    det_conf: float,
    scrfd_view: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    gate_debug: Optional[GateDebug] = None,
    poses: "Sequence[PersonPose]" = (),
) -> None:
    """Composite every detection layer onto the TRACKING panel copy.

    Bottom layer up: body boxes (faint, context), raw head/face-class
    detections (green/cyan when they clear det_conf, grey in the low band —
    evidence only, no longer a blur target by itself), eye/nose/mouth part
    hits (yellow dots) and hand/foot veto evidence (orange boxes), SCRFD
    witness faces with their 5-point landmarks (violet) plus the faces the
    landmark check rejected (red), pose skeletons with their head-anchor box
    (white/green — see libs/pose.py), every face claim libs/evidence.py
    rejected (dark red, tagged with the failing gate — this is where the
    sock/knee/shoulder claims from the old pipeline now show up instead of
    turning into a blur), and finally the Kalman tracks with id, state tag
    and each track's remembered face box. Drawn on a frame copy only — never
    on the written output.
    """
    for b in dets.bodies:
        _draw_box(tracking, b, _BODY_BOX_COLOUR, 1)
    for b in dets.heads:
        strong = b[4] >= det_conf
        _draw_box(tracking, b,
                  _HEAD_DET_COLOUR if strong else _HEAD_LOW_COLOUR,
                  1, f"{b[4]:.2f}")
    for b in dets.faces:
        _draw_box(tracking, b, _FACE_DET_COLOUR, 1, "face")
    for b in dets.parts:
        cx, cy = int(round((b[0] + b[2]) / 2)), int(round((b[1] + b[3]) / 2))
        cv2.circle(tracking, (cx, cy), 3, _PART_COLOUR, -1, cv2.LINE_AA)
    for b in dets.negatives:
        label = "hand" if int(b[5]) == NEG_HAND else "foot"
        _draw_box(tracking, b, _NEG_COLOUR, 1, label)
    if scrfd_view is not None:
        ok_boxes, rej_boxes, ok_kps = scrfd_view
        for b in rej_boxes:
            _draw_box(tracking, b, _SCRFD_REJ_COLOUR, 1, "rej")
        for b, kp in zip(ok_boxes, ok_kps):
            _draw_box(tracking, b, _SCRFD_FACE_COLOUR, 1, "scrfd")
            if not np.isnan(kp).any():
                for x, y in kp:
                    cv2.circle(tracking, (int(round(x)), int(round(y))),
                               2, _SCRFD_FACE_COLOUR, -1, cv2.LINE_AA)
    _draw_poses(tracking, poses)
    if gate_debug is not None:
        for box, flags in gate_debug.rejected:
            _draw_box(tracking, box, _REJECTED_COLOUR, 1,
                      _reject_reason(flags))
    for o in obs:
        colour = _track_colour(o.track_id)
        _draw_box(tracking, o.box, colour, 2 if o.confirmed else 1,
                  f"id:{o.track_id}{_obs_tag(o)}")
        # The face box the face-only blur would target, riding the track.
        if o.face_box is not None:
            _draw_box(tracking, o.face_box, colour, 1, "face")


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
    # (pass_no 1|2, current_frame, total_frames) — export runs two passes.
    export_progress = pyqtSignal(int, int, int)
    export_finished = pyqtSignal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: Optional[tuple[np.ndarray, Params, int, float]] = None
        self._wake = threading.Event()
        self._running = True
        # One detector session for the process lifetime — built lazily, never
        # rebuilt (destroying a DirectML session corrupts the provider).
        self._detector: Optional[HeadDetector] = None
        # Same lifecycle for the SCRFD close-up assist (fallback only: runs
        # when the primary sees nothing confident, so usually never loads).
        self._scrfd: Optional[ScrfdDetector] = None
        # Same lifecycle for the pose estimator (Phase 2): supplies the
        # gate's anatomical anchor + torso axis. Failure degrades to
        # head-anchor-only gating, never crashes.
        self._pose: Optional[PoseEstimator] = None
        self._blur = BlurPipeline()
        # Preview tracking state persists across sequential frames (Play),
        # so Kalman smoothing/hold is visible live; any scrub/jump resets it.
        self._pv_tracker: Optional[HeadTracker] = None
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
        # Per-stage wall-clock of the last processed frame, for the export
        # timing log (the detector sub-times are read off the model objects).
        self._last_detect_ms = 0.0
        self._last_blur_ms = 0.0

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

    def _ensure_detector(self) -> HeadDetector:
        if self._detector is None:
            self._detector = HeadDetector(on_status=self.status.emit)
        return self._detector

    def _ensure_scrfd(self) -> ScrfdDetector:
        if self._scrfd is None:
            self._scrfd = ScrfdDetector(on_status=self.status.emit)
        return self._scrfd

    def _ensure_pose(self) -> PoseEstimator:
        if self._pose is None:
            self._pose = PoseEstimator(on_status=self.status.emit)
        return self._pose

    @staticmethod
    def _display_copy(frame: np.ndarray) -> np.ndarray:
        """Downscale for the preview panels (long edge ≤ _DISPLAY_LONG_EDGE)."""
        fh, fw = frame.shape[:2]
        longest = max(fw, fh)
        if longest <= _DISPLAY_LONG_EDGE:
            return frame.copy()
        s = _DISPLAY_LONG_EDGE / float(longest)
        return cv2.resize(frame, (round(fw * s), round(fh * s)),
                          interpolation=cv2.INTER_AREA)

    def _compute(
        self, frame: np.ndarray, p: Params, tracker: HeadTracker,
    ) -> tuple[Detections, list, list[TrackObs], GateDebug,
               tuple[np.ndarray, np.ndarray, np.ndarray], list[PersonPose]]:
        """Detection → pose → evidence gate → tracking on ``frame``.

        Returns ``(dets, cands, obs, gate_debug, scrfd_view, poses)`` with
        every box in ``frame``'s coordinate space. ``cands`` is this frame's
        gated ``libs.evidence.FaceCandidate`` list — the only thing that may
        spawn or drive a blur track (see libs/evidence.py); ``gate_debug.
        rejected`` carries every claim the gate turned down, for the
        TRACKING overlay. ``scrfd_view`` is ``(ok_boxes, rejected_boxes,
        ok_kps)``: faces whose 5-point landmarks read as a real face, faces
        the landmark check rejected (fabric/skin misreads — display only),
        and the landmarks of the accepted ones. ``poses`` is up to two
        ``libs.pose.PersonPose`` (one per tracked body, Phase 2) — empty when
        no bodies are in frame or the pose model is unavailable, in which
        case the gate degrades to head-anchor-only. The detector resizes to
        its own fixed input internally, so this costs the same at any frame
        resolution.
        """
        detector = self._ensure_detector()
        rots = (0, 90, 270) if p.rot_assist else (0,)
        dets = detector.detect(frame, rotations=rots)

        # SCRFD now runs every frame — a first-class consensus/part-evidence
        # source for the gate, not a lazy close-up-only fallback. Landmark-
        # implausible faces are cut before the gate ever sees them.
        boxes, kps = self._ensure_scrfd().detect_full(frame, floor=WITNESS_FLOOR)
        ok = kps_plausible(kps)
        scrfd_view = (boxes[ok], boxes[~ok], kps[ok])

        poses = self._ensure_pose().estimate(frame, dets.bodies)

        thr = PROFILES.get(p.evidence_profile, PROFILES["balanced"])
        cands, gate_debug = gate_face_candidates(
            dets, poses, scrfd_view[0], frame.shape[:2], thr, p.det_conf)

        tracker.configure(det_conf=p.det_conf, det_conf_low=p.det_conf_low,
                          min_hits=p.min_hits, max_age_s=p.max_age_s)
        if cands:
            anchors = np.stack([c.anchor for c in cands])
            faces = np.stack([c.face for c in cands])
            flags = np.array([c.flags for c in cands], dtype=np.int64)
        else:
            anchors = np.empty((0, 5), np.float32)
            faces = np.empty((0, 5), np.float32)
            flags = np.empty(0, dtype=np.int64)
        obs = tracker.update(
            anchors, frame.shape, faces=faces, flags=flags,
            sustain=sustain_pool(dets, thr, p.det_conf_low, p.det_conf),
            weak_faces=gate_weak_faces(dets, scrfd_view[0]))
        return dets, cands, obs, gate_debug, scrfd_view, poses

    def _process(
        self, frame: np.ndarray, p: Params, frame_idx: int = -1,
        fps: float = 25.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()

        # The preview computes and renders at display resolution (the panels
        # are small); the export paths run on the full-res frame instead.
        proc = self._display_copy(frame)

        # Sequential frames (Play) keep the tracker so Kalman smoothing/hold
        # shows live; scrubbing or single-frame inspection resets it.
        if self._pv_tracker is None or frame_idx < 0 \
                or frame_idx != self._pv_last_idx + 1:
            self._pv_tracker = HeadTracker(
                fps=fps, det_conf=p.det_conf, det_conf_low=p.det_conf_low,
                min_hits=p.min_hits, max_age_s=p.max_age_s)
        self._pv_last_idx = frame_idx

        dets, cands, obs, gate_debug, scrfd_view, poses = self._compute(
            proc, p, self._pv_tracker)
        hold = max(0, int(round(p.preview_hold_s * fps)))
        face_hold = max(0, int(round(p.face_hold_s * fps)))
        rendered = [o for o in obs
                    if o.confirmed and o.coast_frames <= hold]
        boxes = [_blur_target_box(o, p.blur_region, face_hold)
                 for o in rendered]
        mask = render_head_mask(proc.shape[:2], boxes,
                                pad=p.mask_pad, feather=p.mask_feather)

        blurred = proc.copy()
        self._blur.reconfigure(p.blur_layers)
        self._blur.apply(blurred, mask)

        # AFTER panel outlines the exact blur ellipses so the target is legible.
        for o, b in zip(rendered, boxes):
            _draw_mask_ellipse(blurred, b, p.mask_pad,
                               _track_colour(o.track_id),
                               f"id:{o.track_id}")

        tracking = proc.copy()
        _draw_tracking_overlay(tracking, dets, obs, p.det_conf, scrfd_view,
                               gate_debug, poses)

        elapsed = time.perf_counter() - t0
        return proc, tracking, blurred, 1.0 / max(elapsed, 1e-6)

    # ── Export (two passes: analyse, then render) ──────────────────────────────

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

    def _log_export_timings(self, tag: str, idx: int, total_ms: float) -> None:
        """Append a per-stage timing line to the debug log (~every 30 frames).

        Surfaces which stage dominates on the user's GPU and whether the model
        silently fell back to CPU (which shows up as a huge detect time).
        Decode/encode overlap on their own threads so they don't appear here."""
        det_ms = self._detector.last_ms if self._detector else 0.0
        debug_log(
            f"[{tag} f{idx}] detect={det_ms:.0f}ms "
            f"blur={self._last_blur_ms:.0f}ms frame={total_ms:.0f}ms "
            f"→ {1000.0 / max(total_ms, 1e-6):.1f} fps")

    @staticmethod
    def _start_decode_thread(
        cap: "cv2.VideoCapture",
        decode_q: "queue.Queue",
        stop_io: threading.Event,
        first_frame: Optional[np.ndarray] = None,
        name: str = "export-decode",
    ) -> threading.Thread:
        """Read frames ahead into ``decode_q`` as ``(idx, frame)``; None at EOF.

        Decode overlaps inference/encode on its own thread so the GPU stays
        fed. ``first_frame`` seeds index 0 when the caller already read it."""
        def _decode() -> None:
            if first_frame is not None:
                local_idx, f = 0, first_frame
            else:
                ret, f = cap.read()
                if not ret:
                    decode_q.put(None)
                    return
                local_idx = 0
            while not stop_io.is_set():
                try:
                    decode_q.put((local_idx, f), timeout=0.2)
                except queue.Full:
                    continue
                ret2, nf = cap.read()
                local_idx += 1
                if not ret2:
                    break
                f = nf
            try:
                decode_q.put(None, timeout=0.2)   # sentinel (best-effort)
            except queue.Full:
                pass

        t = threading.Thread(target=_decode, name=name, daemon=True)
        t.start()
        return t

    def _wait_if_paused(self, frame: np.ndarray) -> bool:
        """Block while paused (re-rendering on slider changes); True = cancel."""
        while not self._export_run.is_set():
            if self._take_params_dirty():
                self._paused_preview(frame)
            self._export_run.wait(0.05)
        return self._export_cancel.is_set()

    def _make_verifier(self, input_path: str, p: Params):
        """Build the ``verify=`` hook for clean_tracklets: cropped
        re-inference. Seeks back into the source and re-runs the *same*
        evidence gate on a magnified crop around every tracklet that
        survived the prune (see tracklets.verify_tracklets) — a real face
        re-earns its anatomical/part evidence when magnified; a
        skin/fabric hallucination usually doesn't. Returns None —
        verification off, fail open — when the detector is down or the
        source can't be reopened. (Interim, single-model verifier — Phase 3
        adds an independent cross-model version.)
        """
        det = self._detector
        if det is None or not det.available:
            return None
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None
        rots = (0, 90, 270) if p.rot_assist else (0,)
        thr = PROFILES.get(p.evidence_profile, PROFILES["balanced"])

        def frame_at(idx: int) -> Optional[np.ndarray]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            return frame if ok else None

        def detect_fn(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            d = det.detect(crop, rotations=rots)
            sboxes, skps = self._ensure_scrfd().detect_full(
                crop, floor=WITNESS_FLOOR)
            scrfd_ok = sboxes[kps_plausible(skps)]
            poses = self._ensure_pose().estimate(crop, d.bodies)
            accepted, _dbg = gate_face_candidates(
                d, poses, scrfd_ok, crop.shape[:2], thr, p.det_conf)
            if not accepted:
                _empty = np.empty((0, 5), np.float32)
                return _empty, _empty
            return (np.stack([c.anchor for c in accepted]),
                    np.stack([c.face for c in accepted]))

        def verify(tracklets: list) -> list:
            try:
                if not tracklets:
                    return tracklets
                self.status.emit(
                    f"Verifying {len(tracklets)} tracklet(s)…")
                kept, dropped = verify_tracklets(
                    tracklets, frame_at, detect_fn)
                if dropped:
                    msg = (f"verify: rejected {len(dropped)}/"
                           f"{len(tracklets)} tracklet(s) as false "
                           f"positives (ids {[t.tid for t in dropped]})")
                    debug_log(msg)
                    self.status.emit(msg)
                return kept
            finally:
                cap.release()

        return verify

    def _analyse_pass(
        self,
        cap: "cv2.VideoCapture",
        frame0: np.ndarray,
        total: int,
        tracker: HeadTracker,
        recorder: TrackRecorder,
        fps: float,
    ) -> tuple[int, bool]:
        """Pass 1: detect + track every frame, record observations, blur
        nothing. Returns ``(frames_decoded, cancelled)``."""
        decode_q: queue.Queue = queue.Queue(maxsize=4)
        stop_io = threading.Event()
        dec_thread = self._start_decode_thread(
            cap, decode_q, stop_io, frame0, "export-analyse-decode")

        n = 0
        cancelled = False
        last_preview = 0.0
        frame = frame0
        try:
            while True:
                if self._wait_if_paused(frame):
                    cancelled = True
                    break
                item = decode_q.get()
                if item is None:
                    break
                idx, frame = item

                p = self._latest_params()
                t0 = time.perf_counter()
                dets, cands, obs, gate_debug, scrfd_view, poses = self._compute(
                    frame, p, tracker)
                # Record the raw face-target position (the track's own best
                # guess — not gated by a live hold) and whether it is a
                # literal match this exact frame; libs/tracklets.py's
                # offline face-gap fill decides how to bridge short misses,
                # knowledge a streaming recorder can't have. Head vs face is
                # chosen live in pass 2.
                recorder.observe(
                    idx, obs,
                    face_boxes=[o.face_box if o.face_box is not None
                               else o.box for o in obs],
                    face_valid=[o.face_age == 0 for o in obs],
                    ev_flags=[o.flags for o in obs])
                n = idx + 1
                total_ms = (time.perf_counter() - t0) * 1e3

                # Preview at ~10 fps wall-clock: the TRACKING panel shows the
                # raw evidence live; AFTER stays honest — no blur exists yet.
                now = time.perf_counter()
                if now - last_preview >= 0.1:
                    last_preview = now
                    overlay = frame.copy()
                    _draw_tracking_overlay(overlay, dets, obs, p.det_conf,
                                           scrfd_view, gate_debug, poses)
                    tracking = self._display_copy(overlay)
                    after = self._display_copy(frame)
                    cv2.putText(after, "analysing  ·  pass 1/2",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 170, 255), 2, cv2.LINE_AA)
                    self._emit_preview(self._display_copy(frame), tracking,
                                       after, 1000.0 / max(total_ms, 1e-6))
                self.export_progress.emit(1, idx, total)
                if n % 30 == 0:
                    self._log_export_timings("analyse", idx, total_ms)
        finally:
            stop_io.set()
            try:
                while True:
                    decode_q.get_nowait()
            except queue.Empty:
                pass
            dec_thread.join(timeout=1.0)
        return n, cancelled

    def _render_pass(
        self,
        input_path: str,
        output_path: str,
        table: list,
        total: int,
        fps: float,
        fw: int,
        fh: int,
        src_kbps: int,
    ) -> None:
        """Pass 2: re-decode, blur from the cleaned track table (no inference),
        encode. Emits export_finished."""
        # Fresh decoder rather than seeking the pass-1 capture back — a clean
        # open is the only way OpenCV guarantees the identical frame sequence.
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.export_finished.emit(False, f"Cannot reopen: {input_path}")
            return
        writer = make_video_writer(
            output_path, fw, fh, fps,
            bitrate_kbps=src_kbps, on_status=self.status.emit,
            audio_source=input_path,
        )
        if writer is None:
            cap.release()
            self.export_finished.emit(
                False, f"Cannot create output: {output_path}")
            return

        decode_q: queue.Queue = queue.Queue(maxsize=4)
        write_q: queue.Queue = queue.Queue(maxsize=4)
        stop_io = threading.Event()
        write_err: list[BaseException] = []

        def _write() -> None:
            while True:
                item = write_q.get()
                if item is None:
                    break
                if write_err:
                    continue                       # already failed → just drain
                try:
                    writer.write(item)
                except Exception as exc:            # noqa: BLE001
                    write_err.append(exc)

        dec_thread = self._start_decode_thread(
            cap, decode_q, stop_io, None, "export-render-decode")
        wr_thread = threading.Thread(target=_write, name="export-write",
                                     daemon=True)
        wr_thread.start()

        processed = 0
        cancelled = False
        last_preview = 0.0
        frame: Optional[np.ndarray] = None
        try:
            while True:
                if frame is not None and self._wait_if_paused(frame):
                    cancelled = True
                    break
                item = decode_q.get()
                if item is None:
                    break
                idx, frame = item

                # Appearance stays live-tunable in pass 2; the tracking data
                # is already frozen in the table (both head and face-target
                # channels, so the blur-region choice stays live too). In
                # face mode an entry with stale/never-seen face evidence
                # (face_ok False) is skipped outright rather than falling
                # back to the head box — see Params.blur_region.
                p = self._latest_params()
                entries = table[idx] if idx < len(table) else []
                face_mode = p.blur_region == "face"
                boxes = ([fb for _tid, _b, fb, ok in entries if ok]
                        if face_mode else [b for _tid, b, _fb, _ok in entries])

                t0 = time.perf_counter()
                do_preview = (t0 - last_preview >= 0.1)
                before = self._display_copy(frame) if do_preview else None

                mask = render_head_mask((fh, fw), boxes,
                                        pad=p.mask_pad, feather=p.mask_feather)
                self._blur.reconfigure(p.blur_layers)
                t_b = time.perf_counter()
                self._blur.apply(frame, mask)
                self._last_blur_ms = (time.perf_counter() - t_b) * 1e3
                write_q.put(frame)
                if write_err:
                    raise write_err[0]

                total_ms = (time.perf_counter() - t0) * 1e3
                if do_preview and before is not None:
                    last_preview = t0
                    after = self._display_copy(frame)
                    s = before.shape[1] / float(fw)
                    tracking = before.copy()
                    for tid, b, fb, ok in entries:
                        if face_mode and not ok:
                            continue
                        _draw_mask_ellipse(tracking,
                                           (fb if face_mode else b) * s,
                                           p.mask_pad,
                                           _track_colour(tid), f"id:{tid}")
                    self._emit_preview(before, tracking, after,
                                       1000.0 / max(total_ms, 1e-6))
                self.export_progress.emit(2, idx, total)

                processed += 1
                if processed % 30 == 0:
                    self._log_export_timings("render", idx, total_ms)
        finally:
            stop_io.set()
            try:
                while True:
                    decode_q.get_nowait()
            except queue.Empty:
                pass
            write_q.put(None)
            wr_thread.join(timeout=5.0)
            dec_thread.join(timeout=1.0)
            cap.release()
            writer.release()

        out_name = Path(output_path).name
        if write_err and not cancelled:
            self.export_finished.emit(
                False, f"Export write failed at frame {processed}: {write_err[0]!r}"
                f"   (full trace: {_CRASH_LOG})")
        elif cancelled:
            self.export_finished.emit(
                False, f"Stopped at frame {processed} — partial file kept: {out_name}")
        else:
            self.export_finished.emit(
                True, f"Exported {processed} frames → {out_name}")

    def _run_export(self, input_path: str, output_path: str) -> None:
        """Two-pass export: analyse (detect+track, no blur), clean the
        tracklets offline, then render from the cleaned table (no inference).

        The offline pass is what makes the output steady: false positives are
        pruned with hindsight, detection gaps are interpolated along the head's
        path instead of held or dropped, and zero-phase smoothing removes
        jitter without lag."""
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

        ret, frame0 = cap.read()
        if not ret:
            cap.release()
            self.export_finished.emit(False, "Cannot read first frame")
            return
        fh, fw = frame0.shape[:2]

        p0 = self._latest_params()
        tracker = HeadTracker(fps=fps, det_conf=p0.det_conf,
                              det_conf_low=p0.det_conf_low,
                              min_hits=p0.min_hits, max_age_s=p0.max_age_s)
        recorder = TrackRecorder()
        self.status.emit("Pass 1/2 — analysing…")
        n1, cancelled = self._analyse_pass(cap, frame0, total, tracker,
                                           recorder, fps)
        cap.release()
        if cancelled:
            self.export_finished.emit(
                False, "Cancelled during analysis — nothing written")
            return
        if n1 == 0:
            self.export_finished.emit(False, "No frames decoded")
            return

        # Offline tracklet cleanup between the passes (fast, pure numpy,
        # composite ledger-aware prune, plus the cropped re-inference
        # verification pass — see _make_verifier — which is what kills
        # persistent hallucinations).
        p = self._latest_params()
        raw = recorder.finalize()
        kept, rejected = clean_tracklets(
            raw, fps=fps, n_frames=n1,
            p=PostParams(det_conf=p.det_conf, min_hits=p.min_hits,
                         min_track_s=p.min_track_s,
                         bridge_gap_s=p.bridge_gap_s,
                         smooth_win_s=p.smooth_win_s,
                         face_gap_bridge_s=p.face_gap_bridge_s,
                         evidence_profile=p.evidence_profile),
            verify=self._make_verifier(input_path, p))
        table = build_table(kept, n1)
        covered = sum(1 for entries in table if entries)
        grades = {}
        for ct in kept:
            grades[ct.grade] = grades.get(ct.grade, 0) + 1
        msg = (f"Analysis: {len(raw)} raw tracklets → {len(kept)} kept "
               f"(grades {grades}), {len(rejected)} rejected — "
               f"blur on {covered}/{n1} frames — pass 2/2 rendering…")
        debug_log(msg)
        self.status.emit(msg)

        self._render_pass(input_path, output_path, table, n1, fps,
                          fw, fh, src_kbps)


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
        lay.addWidget(self._build_legend())

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

    def _build_legend(self) -> QWidget:
        """Colour key for the TRACKING / AFTER overlays: which box is what."""
        def swatch(colour_bgr: tuple[int, int, int]) -> QLabel:
            b, g, r = colour_bgr
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(
                f"background-color: #{r:02x}{g:02x}{b:02x};"
                "border-radius: 5px;")
            return dot

        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(6)

        cap = QLabel("LEGEND")
        cap.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; font-weight: bold;")
        _letterspace(cap)
        lay.addWidget(cap)
        lay.addSpacing(6)

        entries: list[tuple[tuple[int, int, int], str]] = [
            (_BODY_BOX_COLOUR, "Body"),
            (_HEAD_DET_COLOUR, "Head (confident)"),
            (_HEAD_LOW_COLOUR, "Head (low score)"),
            (_FACE_DET_COLOUR, "Face"),
            (_PART_COLOUR, "Eye/nose/mouth"),
            (_NEG_COLOUR, "Hand/foot (veto evidence)"),
            (_SCRFD_FACE_COLOUR, "Face + landmarks (SCRFD)"),
            (_SCRFD_REJ_COLOUR, "Face rejected (bad landmarks)"),
            (_POSE_SKELETON_COLOUR, "Pose skeleton"),
            (_POSE_ANCHOR_COLOUR, "Pose head anchor"),
            (_REJECTED_COLOUR, "Face rejected (gate)"),
        ]
        for colour, name in entries:
            lay.addWidget(swatch(colour))
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {_TEXT_MID}; font-size: 10px;")
            lay.addWidget(lbl)
            lay.addSpacing(8)

        # Tracks (and their blur ellipses in AFTER) are coloured per id.
        for tid in range(3):
            lay.addWidget(swatch(_track_colour(tid + 1)))
        lbl = QLabel("Track id + blur ellipse (colour per id)")
        lbl.setStyleSheet(f"color: {_TEXT_MID}; font-size: 10px;")
        lay.addWidget(lbl)

        lay.addStretch()
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

        self._det_conf_sl = TunableSlider("Confidence", 0.10, 0.95, 0.50)
        self._det_conf_sl.setToolTip(
            "A head detection at or above this can start and drive a blur.\n"
            "Lower = catch more heads (and more noise); the tracker's\n"
            "confirmation and the export's cleanup absorb most noise.")
        self._det_low_sl = TunableSlider("Sustain floor", 0.02, 0.50, 0.10)
        self._det_low_sl.setToolTip(
            "Detections between this and Confidence only *sustain* an\n"
            "existing blur through occlusion — they can never start one.")
        for sl in (self._det_conf_sl, self._det_low_sl):
            sl.changed.connect(self._on_param_change)

        evidence_row = QHBoxLayout()
        evidence_row.setSpacing(6)
        evidence_lbl = QLabel("Evidence")
        evidence_lbl.setStyleSheet(f"color: {_TEXT_MID}; font-size: 11px;")
        self._evidence_combo = QComboBox()
        self._evidence_combo.addItem("Balanced", "balanced")
        self._evidence_combo.addItem("Max Privacy", "max")
        self._evidence_combo.addItem("Strict", "strict")
        self._evidence_combo.setToolTip(
            "How much anatomical/consensus evidence a face candidate needs\n"
            "to be trusted (see libs/evidence.py) — this, not Confidence,\n"
            "is what tells a real face apart from skin/fabric misread as\n"
            "one. Max Privacy accepts weaker evidence (more coverage, some\n"
            "risk); Strict requires more (fewer false blurs, some risk of\n"
            "missing an odd-angle face).")
        self._evidence_combo.currentIndexChanged.connect(
            lambda _i: self._on_param_change())
        evidence_row.addWidget(evidence_lbl)
        evidence_row.addWidget(self._evidence_combo, stretch=1)

        self._rot_btn = QPushButton("Rotation Assist")
        self._rot_btn.setProperty("chip", "true")
        self._rot_btn.setCheckable(True)
        self._rot_btn.setChecked(True)
        self._rot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rot_btn.setToolTip(
            "Also detect on ±90°-rotated frames and merge the results —\n"
            "recovers sideways heads (lying down / bed angles) that\n"
            "upright-trained detectors miss. ~3× the detection cost.")
        self._rot_btn.toggled.connect(lambda _c: self._on_param_change())

        lay.addWidget(self._det_conf_sl, 0, 0, 1, 2)
        lay.addWidget(self._det_low_sl,  1, 0, 1, 2)
        lay.addLayout(evidence_row,      2, 0, 1, 2)
        lay.addWidget(self._rot_btn,     3, 0,
                      alignment=Qt.AlignmentFlag.AlignLeft)
        return grp

    def _build_blur_group(self) -> QGroupBox:
        grp = QGroupBox("BLUR")
        lay = QHBoxLayout(grp)
        lay.setSpacing(12)

        region_row = QHBoxLayout()
        region_row.setSpacing(6)
        region_lbl = QLabel("Region")
        region_lbl.setStyleSheet(f"color: {_TEXT_MID}; font-size: 11px;")
        self._region_combo = QComboBox()
        self._region_combo.addItem("Face only", "face")
        self._region_combo.addItem("Whole head", "head")
        self._region_combo.setToolTip(
            "What the blur covers.\n"
            "Face only — the detected face bloomed to include a bit of\n"
            "hair; blurred only while face evidence is fresh or bridged\n"
            "(a track with no face evidence at all stays unblurred — see\n"
            "Face Hold below and the Evidence setting above).\n"
            "Whole head — the tracked head/anchor box (safest, biggest,\n"
            "never loses coverage).")
        self._region_combo.currentIndexChanged.connect(
            lambda _i: self._on_param_change())
        region_row.addWidget(region_lbl)
        region_row.addWidget(self._region_combo, stretch=1)

        self._pad_sl = TunableSlider("Mask Pad", 0.00, 0.60, 0.18)
        self._pad_sl.setToolTip(
            "How far the blur ellipse extends beyond the blur target box,\n"
            "per side. Bigger = safer margin, less tight.")
        self._feather_sl = TunableSlider("Edge Feather", 0.00, 0.40, 0.12)
        self._feather_sl.setToolTip(
            "Soft fade at the mask edge, as a fraction of the head size.\n"
            "Hides residual jitter and reads less harsh than a hard edge.")
        self._face_hold_sl = TunableSlider("Face Hold (s)", 0.1, 3.0, 0.8,
                                           decimals=1)
        self._face_hold_sl.setToolTip(
            "Face-only mode: how long a track's last face sighting keeps\n"
            "steering the blur (preview) or may be bridged across (export)\n"
            "before it's treated as a genuine loss of face evidence.")
        for sl in (self._pad_sl, self._feather_sl, self._face_hold_sl):
            sl.changed.connect(self._on_param_change)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addLayout(region_row)
        left.addWidget(self._pad_sl)
        left.addWidget(self._feather_sl)
        left.addWidget(self._face_hold_sl)
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
        grp = QGroupBox("TRACKING · CLEANUP")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        self._min_hits_sl = TunableSlider("Confirm frames", 1, 10, 3,
                                          decimals=0)
        self._min_hits_sl.setToolTip(
            "Consecutive detections before a new head is blurred —\n"
            "suppresses one-frame false positives.")
        self._min_track_sl = TunableSlider("Min track (s)", 0.0, 2.0, 0.25,
                                           decimals=2)
        self._min_track_sl.setToolTip(
            "Export cleanup: tracklets shorter than this are treated as\n"
            "detector noise and dropped.")
        self._bridge_sl = TunableSlider("Bridge gap (s)", 0.0, 4.0, 1.5,
                                        decimals=1)
        self._bridge_sl.setToolTip(
            "Export cleanup: a head that vanishes and reappears within this\n"
            "window is re-joined and the gap blurred along its path.")
        self._max_age_sl = TunableSlider("Coast (s)", 0.5, 4.0, 1.75,
                                         decimals=2)
        self._max_age_sl.setToolTip(
            "How long a lost track stays alive as a re-acquire/bridge\n"
            "candidate (it is not blurred while coasting).")
        self._smooth_sl = TunableSlider("Smooth (s)", 0.1, 2.0, 0.5,
                                        decimals=1)
        self._smooth_sl.setToolTip(
            "Export cleanup: zero-phase smoothing window for the blur's\n"
            "position/size — steadier blur, no lag.")
        for sl in (self._min_hits_sl, self._min_track_sl, self._bridge_sl,
                   self._max_age_sl, self._smooth_sl):
            sl.changed.connect(self._on_param_change)

        lay.addWidget(self._min_hits_sl)
        lay.addWidget(self._min_track_sl)
        lay.addWidget(self._bridge_sl)
        lay.addWidget(self._max_age_sl)
        lay.addWidget(self._smooth_sl)
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

    def _on_export_progress(self, pass_no: int, current: int, total: int) -> None:
        # One slider spans both passes: [0, total) = analyse, [total, 2·total)
        # = render, so progress never appears to rewind between passes.
        span = max(1, total)
        self._frame_slider.blockSignals(True)
        self._frame_slider.setMaximum(2 * span - 1)
        self._frame_slider.setValue((pass_no - 1) * span + current)
        self._frame_slider.blockSignals(False)
        self._frame_lbl.setText(f"P{pass_no}/2 · {current} / {max(0, total - 1)}")

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
            self._det_conf_sl.set_value(p.det_conf)
            self._det_low_sl.set_value(p.det_conf_low)
            idx = self._evidence_combo.findData(p.evidence_profile)
            self._evidence_combo.setCurrentIndex(max(0, idx))
            self._rot_btn.setChecked(p.rot_assist)
            self._region_combo.setCurrentIndex(
                0 if p.blur_region == "face" else 1)
            self._pad_sl.set_value(p.mask_pad)
            self._feather_sl.set_value(p.mask_feather)
            self._face_hold_sl.set_value(p.face_hold_s)
            self._set_blur_layers(p.blur_layers)
            self._min_hits_sl.set_value(p.min_hits)
            self._min_track_sl.set_value(p.min_track_s)
            self._bridge_sl.set_value(p.bridge_gap_s)
            self._max_age_sl.set_value(p.max_age_s)
            self._smooth_sl.set_value(p.smooth_win_s)
            # No dedicated widget for face_gap_bridge_s yet — carried
            # straight from the preset.
            self._params.face_gap_bridge_s = p.face_gap_bridge_s
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

    def _on_param_change(self, _val: float = 0.0) -> None:
        self._params.det_conf     = self._det_conf_sl.value()
        self._params.det_conf_low = self._det_low_sl.value()
        self._params.evidence_profile = self._evidence_combo.currentData()
        self._params.rot_assist   = self._rot_btn.isChecked()
        self._params.blur_region  = self._region_combo.currentData()
        self._params.mask_pad     = self._pad_sl.value()
        self._params.mask_feather = self._feather_sl.value()
        self._params.face_hold_s  = self._face_hold_sl.value()
        self._params.blur_layers  = self._blur_layers_value()
        self._params.min_hits     = self._min_hits_sl.int_value()
        self._params.min_track_s  = self._min_track_sl.value()
        self._params.bridge_gap_s = self._bridge_sl.value()
        self._params.max_age_s    = self._max_age_sl.value()
        self._params.smooth_win_s = self._smooth_sl.value()
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
        debug_log(msg)
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

def _run_preflight(app: QApplication, splash: QSplashScreen | None) -> None:
    """Check/download all models with the splash showing live progress.

    The downloads block (hundreds of MB on first run), so they run on a worker
    thread while this (GUI) thread pumps the event loop and repaints the splash —
    a loading pop-up that updates instead of freezing into "Not Responding". The
    worker only stores the latest status string; every Qt call stays on this
    thread. preflight() mirrors each line to the debug log itself, so progress is
    recorded even in the windowed .exe where there is no console."""
    latest = {"msg": "Checking models…"}
    lock = threading.Lock()
    done = threading.Event()

    def on_status(msg: str) -> None:
        with lock:
            latest["msg"] = msg

    def work() -> None:
        try:
            preflight_models(on_status)
        except Exception as exc:  # noqa: BLE001 — never crash startup on preflight
            _log_exception(exc)
        finally:
            done.set()

    threading.Thread(target=work, name="model-preflight", daemon=True).start()
    shown: Optional[str] = None
    while not done.is_set():
        with lock:
            msg = latest["msg"]
        if msg != shown:
            shown = msg
            splash_update(splash, msg)   # repaints via processEvents
        else:
            app.processEvents()
        time.sleep(0.03)
    with lock:
        splash_update(splash, latest["msg"])


def main(splash: QSplashScreen | None = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    # Show a loading pop-up even when launched without main.py (e.g. ui.py run
    # directly, or a frozen build whose entry is this module).
    if splash is None:
        splash = show_splash()

    # Resolve/download models up front with the splash reporting progress, BEFORE
    # the app stylesheet is applied (its `QWidget{background:transparent}` rule
    # would otherwise blank the splash) and before MainWindow is built — so
    # opening the first video is no longer the first time anything downloads.
    _run_preflight(app, splash)

    app.setStyleSheet(STYLE)
    splash_update(splash, "Preparing interface…")
    win = MainWindow()
    win.show()
    if splash is not None:
        splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
