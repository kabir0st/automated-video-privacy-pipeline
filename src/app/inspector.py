"""Selected-track panel: what it is, why the pipeline thinks so, a few
thumbnails, and the actions review needs."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from pipeline.geom import rows_to_xyxy
from pipeline.types import ManualTrack, Track

from . import theme
from .frames import FrameSource

_THUMB = 108


def _pix(bgr: np.ndarray, size: int) -> QPixmap:
    import cv2
    h, w = bgr.shape[:2]
    s = size / max(h, w, 1)
    img = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))))
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
               QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class Inspector(QWidget):
    toggle_enabled = pyqtSignal(int, bool)
    delete_track = pyqtSignal(int)
    split_at = pyqtSignal(int, int)
    trim = pyqtSignal(int, int, str)
    jump = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self._tid: Optional[int] = None
        self._enabled = True
        self._start = 0
        self._playhead = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        self.title = QLabel("No track selected")
        self.title.setObjectName("h1")
        self.meta = QLabel("")
        self.meta.setObjectName("dim")
        self.meta.setWordWrap(True)
        self.susp = QLabel("")
        self.susp.setTextFormat(Qt.TextFormat.RichText)
        self.reasons = QLabel("")
        self.reasons.setTextFormat(Qt.TextFormat.RichText)
        self.reasons.setWordWrap(True)
        thumbs = QWidget()
        self._thumb_grid = QGridLayout(thumbs)
        self._thumb_grid.setContentsMargins(0, 0, 0, 0)
        self._thumb_grid.setSpacing(4)
        self._thumbs = []
        for i in range(4):
            lbl = QLabel()
            lbl.setFixedSize(_THUMB, _THUMB)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"background:{theme.PANEL_ALT}; border-radius:6px;")
            self._thumb_grid.addWidget(lbl, i // 2, i % 2)
            self._thumbs.append(lbl)
        self.btn_toggle = QPushButton("Disable blur")
        self.btn_toggle.setObjectName("primary")
        self.btn_toggle.clicked.connect(self._on_toggle)
        self.btn_jump = QPushButton("Jump to start")
        self.btn_jump.clicked.connect(lambda: self.jump.emit(self._start))
        self.btn_split = QPushButton("Split at playhead")
        self.btn_split.clicked.connect(
            lambda: self._tid is not None and self.split_at.emit(self._tid, self._playhead))
        self.btn_trim_before = QPushButton("Trim before playhead")
        self.btn_trim_before.clicked.connect(
            lambda: self._tid is not None and self.trim.emit(self._tid, self._playhead, "before"))
        self.btn_trim_after = QPushButton("Trim after playhead")
        self.btn_trim_after.clicked.connect(
            lambda: self._tid is not None and self.trim.emit(self._tid, self._playhead, "after"))
        self.btn_delete = QPushButton("Delete track")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(
            lambda: self._tid is not None and self.delete_track.emit(self._tid))
        row1 = QHBoxLayout()
        row1.addWidget(self.btn_toggle)
        row1.addWidget(self.btn_jump)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_trim_before)
        row2.addWidget(self.btn_trim_after)
        for w in (self.title, self.meta, self.susp, self.reasons, thumbs):
            lay.addWidget(w)
        lay.addLayout(row1)
        lay.addWidget(self.btn_split)
        lay.addLayout(row2)
        lay.addWidget(self.btn_delete)
        lay.addStretch(1)
        self.hint = QLabel("Click a box in the frame or a lane in the timeline.\n"
                           "N: draw a missed head   E: toggle blur   Space: play")
        self.hint.setObjectName("dim")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._set_buttons(False)

    def set_playhead(self, f: int) -> None:
        self._playhead = f

    def _set_buttons(self, on: bool) -> None:
        for b in (self.btn_toggle, self.btn_jump, self.btn_split,
                  self.btn_trim_before, self.btn_trim_after, self.btn_delete):
            b.setEnabled(on)

    def _on_toggle(self) -> None:
        if self._tid is None:
            return
        self._enabled = not self._enabled
        self.btn_toggle.setText("Disable blur" if self._enabled else "Enable blur")
        self.toggle_enabled.emit(self._tid, self._enabled)

    def clear(self) -> None:
        self._tid = None
        self.title.setText("No track selected")
        self.meta.setText("")
        self.susp.setText("")
        self.reasons.setText("")
        for t in self._thumbs:
            t.clear()
        self._set_buttons(False)

    def show_track(self, track: Optional[Track], manual: Optional[ManualTrack],
                   enabled: bool, frames: Optional[FrameSource], fps: float) -> None:
        if track is None and manual is None:
            self.clear()
            return
        self._enabled = enabled
        self.btn_toggle.setText("Disable blur" if enabled else "Enable blur")
        self._set_buttons(True)
        if manual is not None:
            self._tid = manual.tid
            self._start = manual.start
            n = len(manual.boxes)
            self.title.setText(f"Manual head M{-manual.tid - 999}")
            self.meta.setText(f"{self._ts(manual.start, fps)} → {self._ts(manual.end, fps)}"
                              f"  ·  {n / fps:.1f} s  ·  drawn by you")
            self.susp.setText("")
            self.reasons.setText("")
            boxes = manual.boxes
            picks = np.linspace(0, n - 1, min(4, n)).round().astype(int)
            self._thumbnails(frames, manual.start, picks, boxes)
            self.btn_delete.setText("Delete manual head")
            self.btn_delete.setEnabled(True)
            return
        t = track.t
        self._tid = t.tid
        self._start = t.start
        n_hit = int(t.hits.sum())
        self.title.setText(f"Head #{t.tid}")
        v = "" if track.verified is None else f"  ·  re-detected {track.verified:.0%}"
        self.meta.setText(f"{self._ts(t.start, fps)} → {self._ts(t.end, fps)}"
                          f"  ·  {len(t.boxes) / fps:.1f} s  ·  {n_hit} measurements{v}")
        s = track.suspicion
        col = theme.OK if s < 0.35 else (theme.WARN if s < 0.7 else theme.BAD)
        self.susp.setText(f"<span style='color:{col}; font-size:14px; font-weight:600'>"
                          f"suspicion {s:.2f}</span>"
                          f"<span style='color:{theme.TEXT_DIM}'> — "
                          f"{'pipeline kept it' if track.kept else 'pipeline set it aside'}</span>")
        rows = []
        why = track.reasons.get("why")
        if why:
            rows.append(f"<span style='color:{theme.WARN}'>{why}</span>")
        for k, val in sorted(((k, v) for k, v in track.reasons.items()
                              if isinstance(v, float) and k != "pruned"),
                             key=lambda kv: -kv[1])[:5]:
            bar = "█" * int(round(val * 8)) + "░" * (8 - int(round(val * 8)))
            rows.append(f"<span style='color:{theme.TEXT_DIM}'>{k:<11}</span> "
                        f"<span style='font-family:monospace'>{bar}</span> {val:.2f}")
        self.reasons.setText("<br>".join(rows))
        idx = np.nonzero(t.hits)[0]
        if len(idx) == 0:
            idx = np.arange(len(t.boxes))
        picks = idx[np.linspace(0, len(idx) - 1, min(4, len(idx))).round().astype(int)]
        self._thumbnails(frames, t.start, picks, rows_to_xyxy(t.boxes))
        self.btn_delete.setText("Delete track")
        self.btn_delete.setEnabled(True)

    def _thumbnails(self, frames: Optional[FrameSource], start: int,
                    picks: np.ndarray, boxes: np.ndarray) -> None:
        for lbl in self._thumbs:
            lbl.clear()
        if frames is None:
            return
        for lbl, k in zip(self._thumbs, picks):
            fr = frames.frame(start + int(k))
            if fr is None:
                continue
            b = boxes[int(k)]
            h, w = fr.shape[:2]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            side = max(b[2] - b[0], b[3] - b[1]) * 1.6
            x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            lbl.setPixmap(_pix(fr[y1:y2, x1:x2], _THUMB))
            lbl.setToolTip(f"frame {start + int(k)}")

    @staticmethod
    def _ts(f: int, fps: float) -> str:
        s = f / max(fps, 1e-6)
        m, s = divmod(int(s), 60)
        return f"{m}:{s:02d}"
