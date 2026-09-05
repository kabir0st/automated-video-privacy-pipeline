"""Frame view: paints the current frame fitted to the widget, draws the
track overlays on top, optionally previews the blur, and lets the user
click a box to select its track or drag out a new box."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from . import theme


@dataclass
class Overlay:
    tid: int
    xyxy: np.ndarray
    color: tuple[int, int, int]
    label: str = ""
    selected: bool = False
    dashed: bool = False
    thin: bool = False


class FrameCanvas(QWidget):
    box_drawn = pyqtSignal(int, object)      # frame idx, xyxy (frame coords)
    track_clicked = pyqtSignal(int)          # tid
    draw_mode_changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self._img: Optional[QImage] = None
        self._frame_idx = -1
        self._frame_hw = (1, 1)
        self._overlays: list[Overlay] = []
        self._draw_mode = False
        self._drag_start: Optional[QPointF] = None
        self._drag_cur: Optional[QPointF] = None
        self._blur = None
        self._message = "Open a video to begin"

    # ── state ────────────────────────────────────────────────────────────
    def set_message(self, msg: str) -> None:
        self._message = msg
        self.update()

    def set_frame(self, idx: int, frame_bgr: Optional[np.ndarray],
                  blur_boxes: Optional[list] = None, blur_cfg=None) -> None:
        self._frame_idx = idx
        if frame_bgr is None:
            self._img = None
            self.update()
            return
        if blur_boxes and blur_cfg is not None:
            frame_bgr = self._apply_blur(frame_bgr, blur_boxes, blur_cfg)
        h, w = frame_bgr.shape[:2]
        self._frame_hw = (h, w)
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def _apply_blur(self, frame: np.ndarray, boxes: list, cfg) -> np.ndarray:
        from libs.utils import BlurPipeline, render_head_mask
        if self._blur is None:
            self._blur = BlurPipeline()
        self._blur.reconfigure(cfg.blur_layers)
        out = frame.copy()
        mask = render_head_mask(frame.shape[:2], boxes, pad=cfg.mask_pad,
                                feather=cfg.mask_feather)
        self._blur.apply(out, mask)
        return out

    def set_overlays(self, items: list[Overlay]) -> None:
        self._overlays = items
        self.update()

    def set_draw_mode(self, on: bool) -> None:
        if on != self._draw_mode:
            self._draw_mode = on
            self.setCursor(Qt.CursorShape.CrossCursor if on
                           else Qt.CursorShape.ArrowCursor)
            self.draw_mode_changed.emit(on)
            self.update()

    @property
    def draw_mode(self) -> bool:
        return self._draw_mode

    # ── geometry ─────────────────────────────────────────────────────────
    def _target(self) -> QRectF:
        h, w = self._frame_hw
        W, H = self.width(), self.height()
        s = min(W / max(w, 1), H / max(h, 1))
        tw, th = w * s, h * s
        return QRectF((W - tw) / 2, (H - th) / 2, tw, th)

    def _to_widget(self, xyxy: np.ndarray) -> QRectF:
        t = self._target()
        h, w = self._frame_hw
        sx, sy = t.width() / max(w, 1), t.height() / max(h, 1)
        return QRectF(t.left() + xyxy[0] * sx, t.top() + xyxy[1] * sy,
                      (xyxy[2] - xyxy[0]) * sx, (xyxy[3] - xyxy[1]) * sy)

    def _to_frame(self, p: QPointF) -> tuple[float, float]:
        t = self._target()
        h, w = self._frame_hw
        sx, sy = t.width() / max(w, 1), t.height() / max(h, 1)
        x = (p.x() - t.left()) / max(sx, 1e-9)
        y = (p.y() - t.top()) / max(sy, 1e-9)
        return float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        if self._img is None:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            p.end()
            return
        t = self._target()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(t, self._img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for ov in self._overlays:
            r = self._to_widget(ov.xyxy)
            col = QColor(*ov.color)
            pen = QPen(col, 1 if ov.thin else (3 if ov.selected else 2))
            if ov.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(r)
            if ov.selected:
                pen2 = QPen(QColor(theme.SELECT), 1)
                p.setPen(pen2)
                p.drawRect(r.adjusted(-3, -3, 3, 3))
            if ov.label:
                p.setPen(QPen(col, 1))
                p.drawText(QPointF(r.left() + 3, max(r.top() - 4, 12)), ov.label)
        if self._drag_start is not None and self._drag_cur is not None:
            r = QRectF(self._drag_start, self._drag_cur).normalized()
            pen = QPen(QColor(theme.MANUAL), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(r)
        if self._draw_mode:
            p.setPen(QColor(theme.MANUAL))
            p.drawText(QPointF(t.left() + 10, t.top() + 20),
                       "Draw a box around the head to add it (Esc to cancel)")
        p.end()

    # ── mouse ────────────────────────────────────────────────────────────
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if self._img is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = ev.position()
        if self._draw_mode:
            self._drag_start = pos
            self._drag_cur = pos
            self.update()
            return
        hits = []
        for ov in self._overlays:
            if ov.thin:
                continue
            r = self._to_widget(ov.xyxy)
            if r.contains(pos):
                hits.append((r.width() * r.height(), ov.tid))
        if hits:
            hits.sort()
            self.track_clicked.emit(hits[0][1])

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_cur = ev.position()
            self.update()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._drag_start is None:
            return
        a, b = self._drag_start, ev.position()
        self._drag_start = self._drag_cur = None
        self.set_draw_mode(False)
        if abs(b.x() - a.x()) < 6 or abs(b.y() - a.y()) < 6:
            self.update()
            return
        x1, y1 = self._to_frame(a)
        x2, y2 = self._to_frame(b)
        box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                       np.float32)
        self.box_drawn.emit(self._frame_idx, box)
        self.update()

    def keyPressEvent(self, ev) -> None:  # noqa: N802
        if ev.key() == Qt.Key.Key_Escape and self._draw_mode:
            self._drag_start = self._drag_cur = None
            self.set_draw_mode(False)
        else:
            super().keyPressEvent(ev)
