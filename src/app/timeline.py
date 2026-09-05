"""Timeline: one lane per track (sorted by suspicion, most suspicious on
top), a ruler, an alert strip marking frames where a confident detection
has no blur, and a draggable playhead. Ctrl+wheel zooms, wheel scrolls
lanes, Shift+wheel pans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from . import theme


@dataclass
class Lane:
    tid: int
    start: int
    end: int
    color: tuple[int, int, int]
    label: str
    enabled: bool
    suspicion: float = 0.0
    manual: bool = False
    hits: Optional[np.ndarray] = None       # bool per frame from start


_GUTTER = 84
_RULER = 22
_ALERT = 10
_LANE = 18


class Timeline(QWidget):
    seek = pyqtSignal(int)
    lane_clicked = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self._n = 1
        self._fps = 25.0
        self._lanes: list[Lane] = []
        self._alerts: Optional[np.ndarray] = None
        self._coverage: Optional[np.ndarray] = None
        self._play = 0
        self._sel: Optional[int] = None
        self._v0 = 0.0
        self._v1 = 1.0
        self._scroll = 0
        self._dragging = False

    # ── state ────────────────────────────────────────────────────────────
    def set_video(self, n_frames: int, fps: float) -> None:
        self._n = max(1, n_frames)
        self._fps = fps or 25.0
        self._v0, self._v1 = 0.0, float(self._n)
        self.update()

    def set_lanes(self, lanes: list[Lane]) -> None:
        self._lanes = lanes
        self._scroll = 0
        self.update()

    def set_alerts(self, alerts: Optional[np.ndarray],
                   coverage: Optional[np.ndarray] = None) -> None:
        self._alerts = alerts
        self._coverage = coverage
        self.update()

    def set_playhead(self, f: int, follow: bool = True) -> None:
        self._play = int(f)
        if follow and not (self._v0 <= f < self._v1):
            span = self._v1 - self._v0
            self._v0 = float(np.clip(f - span * 0.2, 0, max(0, self._n - span)))
            self._v1 = self._v0 + span
        self.update()

    def set_selected(self, tid: Optional[int]) -> None:
        self._sel = tid
        self.update()

    def visible(self) -> tuple[int, int]:
        return int(self._v0), int(self._v1)

    # ── geometry ─────────────────────────────────────────────────────────
    def _x(self, f: float) -> float:
        span = max(self._v1 - self._v0, 1.0)
        return _GUTTER + (f - self._v0) / span * (self.width() - _GUTTER)

    def _f(self, x: float) -> int:
        span = max(self._v1 - self._v0, 1.0)
        f = self._v0 + (x - _GUTTER) / max(self.width() - _GUTTER, 1) * span
        return int(np.clip(round(f), 0, self._n - 1))

    def _lane_at(self, y: float) -> Optional[Lane]:
        top = _RULER + _ALERT
        if y < top:
            return None
        i = int((y - top) / _LANE) + self._scroll
        return self._lanes[i] if 0 <= i < len(self._lanes) else None

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        W, H = self.width(), self.height()
        span = max(self._v1 - self._v0, 1.0)
        # ruler
        p.fillRect(QRectF(_GUTTER, 0, W - _GUTTER, _RULER), QColor(theme.PANEL_ALT))
        secs = span / self._fps
        step_s = next(s for s in (0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800)
                      if secs / s <= 14)
        p.setPen(QColor(theme.TEXT_DIM))
        t = np.floor(self._v0 / self._fps / step_s) * step_s
        while t * self._fps <= self._v1:
            x = self._x(t * self._fps)
            if x >= _GUTTER:
                p.drawLine(int(x), _RULER - 6, int(x), _RULER)
                m, s = divmod(int(t), 60)
                p.drawText(int(x) + 3, _RULER - 8, f"{m}:{s:02d}")
            t += step_s
        # alert strip
        y0 = _RULER
        p.fillRect(QRectF(_GUTTER, y0, W - _GUTTER, _ALERT), QColor(theme.BG))
        if self._coverage is not None and len(self._coverage):
            self._strip(p, self._coverage > 0, y0, QColor(theme.OK), 0.35)
        if self._alerts is not None and len(self._alerts):
            self._strip(p, self._alerts, y0, QColor(theme.BAD), 1.0)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(4, _RULER - 8, "time")
        p.drawText(4, y0 + _ALERT - 1, "alerts")
        # lanes
        top = _RULER + _ALERT
        n_vis = max(1, (H - top) // _LANE)
        if len(self._lanes) > n_vis:
            n_vis = max(1, n_vis - 1)          # leave room for the footer
        for row, lane in enumerate(self._lanes[self._scroll:self._scroll + n_vis]):
            y = top + row * _LANE
            if lane.tid == self._sel:
                p.fillRect(QRectF(0, y, W, _LANE), QColor(theme.PANEL_ALT))
            col = QColor(*lane.color)
            if not lane.enabled:
                col.setAlpha(90)
            x1, x2 = self._x(lane.start), self._x(lane.end + 1)
            x1, x2 = max(x1, _GUTTER), min(x2, W)
            if x2 > x1:
                p.fillRect(QRectF(x1, y + 4, max(x2 - x1, 2), _LANE - 8), col)
                if lane.hits is not None and span < 2500:
                    hit_col = QColor(255, 255, 255, 110)
                    idx = np.nonzero(lane.hits)[0]
                    for k in idx:
                        hx = self._x(lane.start + k)
                        if _GUTTER <= hx <= W:
                            p.fillRect(QRectF(hx, y + 4, max(self._x(1) - self._x(0), 1),
                                              _LANE - 8), hit_col)
            p.setPen(QColor(theme.TEXT if lane.enabled else theme.TEXT_DIM))
            p.drawText(6, y + _LANE - 5, lane.label)
            if lane.tid == self._sel:
                p.setPen(QPen(QColor(theme.SELECT), 1))
                p.drawRect(QRectF(x1, y + 3, max(x2 - x1, 2), _LANE - 6))
        if len(self._lanes) > n_vis:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(6, H - 4, f"{self._scroll + 1}-{min(len(self._lanes), self._scroll + n_vis)}"
                                 f" of {len(self._lanes)}")
        # playhead
        x = self._x(self._play)
        if _GUTTER <= x <= W:
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.drawLine(int(x), 0, int(x), H)
        p.end()

    def _strip(self, p: QPainter, mask: np.ndarray, y0: int, col: QColor,
               alpha: float) -> None:
        """Draw runs of True in ``mask`` (per frame) onto the alert strip,
        binned to pixels so a long clip never paints per frame."""
        W = self.width()
        px = max(W - _GUTTER, 1)
        f0, f1 = int(max(0, self._v0)), int(min(len(mask), self._v1))
        if f1 <= f0:
            return
        seg = mask[f0:f1]
        bins = np.linspace(0, len(seg), px + 1).astype(int)
        col = QColor(col)
        col.setAlphaF(alpha)
        run_start = None
        for i in range(px):
            a, b = bins[i], bins[i + 1]
            on = bool(seg[a:max(b, a + 1)].any())
            if on and run_start is None:
                run_start = i
            if (not on or i == px - 1) and run_start is not None:
                end = i + 1 if on else i
                p.fillRect(QRectF(_GUTTER + run_start, y0 + 1, max(end - run_start, 1),
                                  _ALERT - 2), col)
                run_start = None

    # ── mouse / wheel ────────────────────────────────────────────────────
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        x, y = ev.position().x(), ev.position().y()
        if x < _GUTTER:
            lane = self._lane_at(y)
            if lane is not None:
                self.lane_clicked.emit(lane.tid)
            return
        lane = self._lane_at(y)
        f = self._f(x)
        if lane is not None:
            self.lane_clicked.emit(lane.tid)
        self._dragging = True
        self.seek.emit(f)

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._dragging:
            self.seek.emit(self._f(ev.position().x()))

    def mouseReleaseEvent(self, _ev) -> None:  # noqa: N802
        self._dragging = False

    def wheelEvent(self, ev) -> None:  # noqa: N802
        d = ev.angleDelta().y()
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            f = self._f(ev.position().x())
            factor = 0.8 if d > 0 else 1.25
            span = max(20.0, min(float(self._n), (self._v1 - self._v0) * factor))
            frac = (f - self._v0) / max(self._v1 - self._v0, 1)
            self._v0 = float(np.clip(f - frac * span, 0, max(0, self._n - span)))
            self._v1 = self._v0 + span
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            span = self._v1 - self._v0
            shift = -span * 0.1 * np.sign(d)
            self._v0 = float(np.clip(self._v0 + shift, 0, max(0, self._n - span)))
            self._v1 = self._v0 + span
        else:
            self._scroll = int(np.clip(self._scroll - int(np.sign(d)) * 3, 0,
                                       max(0, len(self._lanes) - 1)))
        self.update()
