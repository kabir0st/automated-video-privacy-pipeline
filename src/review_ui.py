"""Review dialog (Phase 4): the seam between analysis and render.

Between ``clean_tracklets`` and the render pass, the user gets to see
exactly what survived cleanup (and what didn't) and override it: disable a
track that's a false positive despite passing the gate/ledger/verify
pipeline, re-enable a track that was wrongly dropped, or draw a manual blur
region over something the pipeline missed entirely.

This dialog runs on the GUI thread — PyQt widgets can't be created or exec'd
from ``ProcessWorker``'s ``QThread``. The worker blocks on a
``threading.Event`` while the dialog is open, the same block-and-signal
pattern ``ProcessWorker._wait_if_paused`` already uses for pause/resume (see
``ui.py``'s ``review_ready``/``take_review_request``/
``submit_review_decision`` handshake). The dialog is application-modal
(``QDialog.exec()``'s default), so there is no separate "cancel while the
dialog is open" race to handle — the dialog's own Cancel button (or closing
its window) is the only way out, and both map to ``ReviewDecision(accepted=
False)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QPushButton, QSizePolicy, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QRubberBand,
)

from libs.sidecar import ManualRegion, ReviewDecisions

_THUMB_SIZE = 96


@dataclass
class ReviewTrack:
    """One row of the review table — everything the dialog needs is
    computed up front on the worker thread (see
    ``ui.ProcessWorker._build_review_tracks``), so the dialog itself does no
    inference or extra frame decoding beyond its own scrub preview."""
    tid: int
    start_frame: int
    end_frame: int
    grade: str                         # "A"/"B"/"C" — libs.evidence.grade
    kept: bool                         # survived clean_tracklets, vs. shown for possible re-enable
    thumbnail: Optional[np.ndarray]    # small BGR crop around the midpoint, or None


@dataclass
class ReviewRequest:
    video_path: str
    fps: float
    n_frames: int
    tracks: list[ReviewTrack] = field(default_factory=list)
    initial: ReviewDecisions = field(default_factory=ReviewDecisions)


@dataclass
class ReviewDecision:
    accepted: bool                      # False = user cancelled the export
    review: ReviewDecisions = field(default_factory=ReviewDecisions)


def _bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


def _fmt_time(frame: int, fps: float) -> str:
    s = frame / max(fps, 1e-6)
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


class _RegionPreview(QLabel):
    """Frame preview supporting rubber-band region drawing. Displays a BGR
    frame scaled (aspect-preserving, letterboxed) to the label's size; a
    mouse drag draws a rectangle that :meth:`take_last_rect` returns mapped
    back to source-frame pixel coordinates."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1d24; color: #888;")
        self.setText("No frame")
        self._frame: Optional[np.ndarray] = None
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin: Optional[QPoint] = None
        self._last_rect: Optional[tuple[float, float, float, float]] = None

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._frame is None:
            return
        pix = _bgr_to_qpixmap(self._frame)
        self.setPixmap(pix.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override
        if self._frame is None:
            return
        self._origin = event.pos()
        self._rubber.setGeometry(QRect(self._origin, QSize()))
        self._rubber.show()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt override
        if self._origin is None:
            return
        self._rubber.setGeometry(QRect(self._origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt override
        if self._origin is None or self._frame is None:
            return
        rect = QRect(self._origin, event.pos()).normalized()
        self._origin = None
        self._rubber.hide()
        self._last_rect = self._widget_rect_to_frame(rect)

    def _widget_rect_to_frame(
        self, rect: QRect,
    ) -> Optional[tuple[float, float, float, float]]:
        pix = self.pixmap()
        if pix is None or pix.isNull() or self._frame is None:
            return None
        fh, fw = self._frame.shape[:2]
        pw, ph = pix.width(), pix.height()
        if pw <= 0 or ph <= 0:
            return None
        ox, oy = (self.width() - pw) / 2.0, (self.height() - ph) / 2.0
        scale = fw / pw
        x1 = (rect.left() - ox) * scale
        y1 = (rect.top() - oy) * scale
        x2 = (rect.right() - ox) * scale
        y2 = (rect.bottom() - oy) * scale
        x1, x2 = sorted((max(0.0, min(x1, fw)), max(0.0, min(x2, fw))))
        y1, y2 = sorted((max(0.0, min(y1, fh)), max(0.0, min(y2, fh))))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)

    def take_last_rect(self) -> Optional[tuple[float, float, float, float]]:
        r, self._last_rect = self._last_rect, None
        return r


class ReviewDialog(QDialog):
    """Track table (enable/disable, worst-grade-first) + manual region
    drawing, seeded from ``request`` and returning a :class:`ReviewDecision`
    via :meth:`decision` once ``exec()`` returns."""

    def __init__(self, request: ReviewRequest,
                parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review tracks before render")
        self.resize(900, 600)
        self._request = request
        self._enabled: dict[int, bool] = dict(request.initial.enabled)
        self._manual_regions: list[ManualRegion] = list(
            request.initial.manual_regions)
        self._pending_start: Optional[tuple[int, tuple]] = None  # (frame, box)
        self._cap = cv2.VideoCapture(request.video_path)

        root = QVBoxLayout(self)
        root.addWidget(self._build_table(), stretch=2)
        root.addWidget(self._build_region_row(), stretch=3)
        root.addWidget(self._build_buttons())

    # ── track table ──────────────────────────────────────────────────────────

    def _build_table(self) -> QWidget:
        # Worst-first: rejected before kept, then grade "C" < "B" < "A".
        tracks = sorted(
            self._request.tracks,
            key=lambda t: (t.kept, {"C": 0, "B": 1, "A": 2}.get(t.grade, 1)))
        self._row_track: list[ReviewTrack] = tracks

        table = QTableWidget(len(tracks), 6)
        table.setHorizontalHeaderLabels(
            ["Blur", "Track", "Start", "End", "Grade", "Status"])
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        for row, t in enumerate(tracks):
            chk = QCheckBox()
            chk.setChecked(self._enabled.get(t.tid, t.kept))
            chk.stateChanged.connect(
                lambda _s, tid=t.tid, cb=chk: self._set_enabled(
                    tid, cb.isChecked()))
            table.setCellWidget(row, 0, chk)
            table.setItem(row, 1, QTableWidgetItem(f"id {t.tid}"))
            table.setItem(row, 2, QTableWidgetItem(
                _fmt_time(t.start_frame, self._request.fps)))
            table.setItem(row, 3, QTableWidgetItem(
                _fmt_time(t.end_frame, self._request.fps)))
            table.setItem(row, 4, QTableWidgetItem(t.grade))
            table.setItem(row, 5, QTableWidgetItem(
                "kept" if t.kept else "rejected"))

        table.itemSelectionChanged.connect(self._on_row_selected)
        self._table = table
        return table

    def _set_enabled(self, tid: int, enabled: bool) -> None:
        self._enabled[tid] = enabled

    def _on_row_selected(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        t = self._row_track[rows[0].row()]
        mid = (t.start_frame + t.end_frame) // 2
        self._region_frame_spin.setValue(mid)

    # ── manual region drawing ────────────────────────────────────────────────

    def _build_region_row(self) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)

        self._region_preview = _RegionPreview()
        self._region_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self._region_preview, stretch=2)

        side = QVBoxLayout()
        side.addWidget(QLabel("Manual blur regions"))

        self._region_frame_spin = QSpinBox()
        self._region_frame_spin.setRange(0, max(0, self._request.n_frames - 1))
        self._region_frame_spin.valueChanged.connect(self._seek_region_preview)
        side.addWidget(QLabel("Preview frame:"))
        side.addWidget(self._region_frame_spin)

        self._region_status = QLabel("Drag on the preview to draw a box.")
        self._region_status.setWordWrap(True)
        side.addWidget(self._region_status)

        start_btn = QPushButton("Mark region start")
        start_btn.clicked.connect(self._mark_region_start)
        side.addWidget(start_btn)

        end_btn = QPushButton("Mark region end && add")
        end_btn.clicked.connect(self._commit_region)
        side.addWidget(end_btn)

        self._regions_list = QListWidget()
        self._refresh_regions_list()
        side.addWidget(self._regions_list, stretch=1)

        remove_btn = QPushButton("Remove selected region")
        remove_btn.clicked.connect(self._remove_selected_region)
        side.addWidget(remove_btn)

        lay.addLayout(side, stretch=1)
        self._seek_region_preview(self._region_frame_spin.value())
        return box

    def _seek_region_preview(self, frame_idx: int) -> None:
        if not self._cap.isOpened():
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if ok:
            self._region_preview.set_frame(frame)

    def _mark_region_start(self) -> None:
        rect = self._region_preview.take_last_rect()
        if rect is None:
            self._region_status.setText(
                "Drag a box on the preview first, then mark the start.")
            return
        self._pending_start = (self._region_frame_spin.value(), rect)
        self._region_status.setText(
            f"Start set at frame {self._pending_start[0]} — "
            "scrub to the end frame, draw the box there, then "
            "'Mark region end'.")

    def _commit_region(self) -> None:
        if self._pending_start is None:
            self._region_status.setText(
                "Mark a region start before marking its end.")
            return
        rect = self._region_preview.take_last_rect()
        if rect is None:
            self._region_status.setText(
                "Drag a box on the preview for the end frame too.")
            return
        start_frame, start_box = self._pending_start
        end_frame = self._region_frame_spin.value()
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
            start_box, rect = rect, start_box
        self._manual_regions.append(ManualRegion(
            start=start_frame, end=end_frame, box0=start_box, box1=rect))
        self._pending_start = None
        self._region_status.setText(
            f"Region added: frames {start_frame}–{end_frame}.")
        self._refresh_regions_list()

    def _refresh_regions_list(self) -> None:
        self._regions_list.clear()
        for i, r in enumerate(self._manual_regions):
            self._regions_list.addItem(
                f"{i}: frames {r.start}–{r.end}")

    def _remove_selected_region(self) -> None:
        row = self._regions_list.currentRow()
        if 0 <= row < len(self._manual_regions):
            del self._manual_regions[row]
            self._refresh_regions_list()

    # ── buttons / result ─────────────────────────────────────────────────────

    def _build_buttons(self) -> QWidget:
        box = QDialogButtonBox()
        box.addButton("Render", QDialogButtonBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        return box

    def done(self, code: int) -> None:  # noqa: N802 — Qt override
        if self._cap.isOpened():
            self._cap.release()
        super().done(code)

    def decision(self) -> ReviewDecision:
        return ReviewDecision(
            accepted=(self.result() == QDialog.DialogCode.Accepted),
            review=ReviewDecisions(enabled=dict(self._enabled),
                                   manual_regions=list(self._manual_regions)))
