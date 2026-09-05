"""Main window: open a video, analyse, review on a timeline, export.

Layout: toolbar · [canvas over transport over timeline] · right panel
(inspector + settings). Every review action edits the project sidecar (saved
with a short debounce) and rebuilds the blur table, so the canvas, timeline
and alert strip always show exactly what export will do.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QProgressBar, QPushButton, QSpinBox,
                             QSplitter, QToolBar, QToolButton, QVBoxLayout,
                             QWidget, QCheckBox, QSizePolicy)

from pipeline import project as proj
from pipeline.geom import iou_matrix
from pipeline.presets import DEFAULT_PRESET, PRESETS, Preset
from pipeline.render import RenderConfig, build_table, coverage
from pipeline.types import ManualTrack, Track

from . import theme
from .canvas import FrameCanvas, Overlay
from .frames import FrameSource
from .inspector import Inspector
from .timeline import Lane, Timeline
from .worker import Job, PipelineWorker

BLUR_STYLES = {
    "Gaussian + pixelate": (("gaussian", 71), ("pixelate", 12)),
    "Heavy (privacy)": (("gaussian", 99), ("pixelate", 18)),
    "Gaussian only": (("gaussian", 81),),
    "Pixelate only": (("pixelate", 16),),
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Privacy Pipeline")
        self.resize(1480, 920)
        self.video: Optional[str] = None
        self.frames: Optional[FrameSource] = None
        self.project: Optional[proj.Project] = None
        self.tracks: list[Track] = []
        self.table: list = []
        self.alerts: Optional[np.ndarray] = None
        self.cur = 0
        self.selected: Optional[int] = None
        self.busy = False
        self._edits_counter = 0
        self.preset_name = DEFAULT_PRESET

        self.worker = PipelineWorker()
        self.worker.status.connect(self._on_status)
        self.worker.progress.connect(self._on_progress)
        self.worker.analysed.connect(self._on_analysed)
        self.worker.offline_done.connect(self._on_offline)
        self.worker.exported.connect(self._on_exported)
        self.worker.propagated.connect(self._on_propagated)
        self.worker.preview_ready.connect(self._on_preview)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

        self._build()
        self._shortcuts()
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._save_project)

    # ── UI construction ─────────────────────────────────────────────────
    def _build(self) -> None:
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_open = QAction("Open video…", self)
        self.act_open.triggered.connect(lambda: self.open_video())
        tb.addAction(self.act_open)
        tb.addSeparator()
        tb.addWidget(QLabel(" Preset "))
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS))
        self.cmb_preset.setCurrentText(self.preset_name)
        self.cmb_preset.currentTextChanged.connect(self._on_preset)
        tb.addWidget(self.cmb_preset)
        tb.addWidget(QLabel(" every "))
        self.spn_stride = QSpinBox()
        self.spn_stride.setRange(1, 10)
        self.spn_stride.setValue(PRESETS[self.preset_name].analysis.stride)
        self.spn_stride.setSuffix(" frame(s)")
        tb.addWidget(self.spn_stride)
        self.btn_analyse = QPushButton("Analyse")
        self.btn_analyse.setObjectName("primary")
        self.btn_analyse.clicked.connect(self.analyse)
        tb.addWidget(self.btn_analyse)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.worker.cancel)
        self.btn_cancel.setEnabled(False)
        tb.addWidget(self.btn_cancel)
        tb.addSeparator()
        self.btn_draw = QToolButton()
        self.btn_draw.setText("Add head (N)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(lambda on: self.canvas.set_draw_mode(on))
        tb.addWidget(self.btn_draw)
        self.chk_blur = QCheckBox("Preview blur")
        self.chk_blur.toggled.connect(lambda _: self.show_frame(self.cur))
        tb.addWidget(self.chk_blur)
        self.chk_dets = QCheckBox("Show detections")
        self.chk_dets.toggled.connect(lambda _: self.show_frame(self.cur))
        tb.addWidget(self.chk_dets)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self.btn_export = QPushButton("Export…")
        self.btn_export.setObjectName("primary")
        self.btn_export.clicked.connect(self.export)
        self.btn_export.setEnabled(False)
        tb.addWidget(self.btn_export)

        # centre
        self.canvas = FrameCanvas()
        self.canvas.track_clicked.connect(self.select)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.draw_mode_changed.connect(self.btn_draw.setChecked)
        transport = QWidget()
        tl = QHBoxLayout(transport)
        tl.setContentsMargins(6, 2, 6, 2)
        self.btn_prev_alert = QPushButton("◀ alert")
        self.btn_prev_alert.clicked.connect(lambda: self.jump_alert(-1))
        self.btn_step_back = QPushButton("◀")
        self.btn_step_back.clicked.connect(lambda: self.step(-1))
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_step_fwd = QPushButton("▶")
        self.btn_step_fwd.clicked.connect(lambda: self.step(1))
        self.btn_next_alert = QPushButton("alert ▶")
        self.btn_next_alert.clicked.connect(lambda: self.jump_alert(1))
        self.btn_next_susp = QPushButton("Next suspicious")
        self.btn_next_susp.clicked.connect(self.next_suspicious)
        self.lbl_time = QLabel("0:00.00 · frame 0")
        self.lbl_time.setObjectName("dim")
        for w in (self.btn_prev_alert, self.btn_step_back, self.btn_play,
                  self.btn_step_fwd, self.btn_next_alert, self.btn_next_susp):
            tl.addWidget(w)
        tl.addStretch(1)
        tl.addWidget(self.lbl_time)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        self.timeline.lane_clicked.connect(self._on_lane_clicked)
        centre = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(0)
        tv.addWidget(self.canvas, 1)
        tv.addWidget(transport, 0)
        centre.addWidget(top)
        centre.addWidget(self.timeline)
        centre.setStretchFactor(0, 4)
        centre.setStretchFactor(1, 1)

        # right panel
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.inspector = Inspector()
        self.inspector.toggle_enabled.connect(self.set_enabled)
        self.inspector.delete_track.connect(self.delete_track)
        self.inspector.split_at.connect(self.split_track)
        self.inspector.trim.connect(self.trim_track)
        self.inspector.jump.connect(self.seek)
        rv.addWidget(self.inspector, 1)
        rv.addWidget(self._settings_group(), 0)
        right.setMinimumWidth(300)
        right.setMaximumWidth(380)

        root = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(centre)
        root.addWidget(right)
        root.setStretchFactor(0, 1)
        root.setStretchFactor(1, 0)
        self.setCentralWidget(root)

        sb = self.statusBar()
        self.lbl_status = QLabel("Open a video to begin")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.progress)

    def _settings_group(self) -> QWidget:
        g = QGroupBox("Blur && review settings")
        v = QVBoxLayout(g)
        v.setSpacing(6)

        def row(label: str, w: QWidget) -> None:
            h = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setObjectName("dim")
            h.addWidget(lbl, 1)
            h.addWidget(w, 0)
            v.addLayout(h)

        p = PRESETS[self.preset_name]
        self.cmb_region = QComboBox()
        self.cmb_region.addItems(["head", "face"])
        self.cmb_region.currentTextChanged.connect(lambda _: self.rebuild())
        row("Blur region", self.cmb_region)
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(list(BLUR_STYLES))
        self.cmb_style.currentTextChanged.connect(lambda _: self.show_frame(self.cur))
        row("Blur style", self.cmb_style)
        self.spn_pad = QDoubleSpinBox()
        self.spn_pad.setRange(0.0, 0.6)
        self.spn_pad.setSingleStep(0.02)
        self.spn_pad.setValue(p.render.mask_pad)
        self.spn_pad.valueChanged.connect(lambda _: self.show_frame(self.cur))
        row("Head padding", self.spn_pad)
        self.spn_lead = QDoubleSpinBox()
        self.spn_lead.setRange(0.0, 2.0)
        self.spn_lead.setSingleStep(0.1)
        self.spn_lead.setValue(p.render.motion_lead)
        self.spn_lead.valueChanged.connect(lambda _: self.rebuild())
        row("Motion lead", self.spn_lead)
        self.spn_disable = QDoubleSpinBox()
        self.spn_disable.setRange(0.2, 1.01)
        self.spn_disable.setSingleStep(0.05)
        self.spn_disable.setValue(p.score.auto_disable_above)
        row("Auto-disable above suspicion", self.spn_disable)
        self.spn_spawn = QDoubleSpinBox()
        self.spn_spawn.setRange(0.1, 0.8)
        self.spn_spawn.setSingleStep(0.05)
        self.spn_spawn.setValue(p.analysis.spawn_conf)
        row("Spawn confidence", self.spn_spawn)
        self.spn_age = QDoubleSpinBox()
        self.spn_age.setRange(0.2, 6.0)
        self.spn_age.setSingleStep(0.25)
        self.spn_age.setValue(p.analysis.max_age_s)
        self.spn_age.setSuffix(" s")
        row("Keep lost head for", self.spn_age)
        self.spn_gap = QDoubleSpinBox()
        self.spn_gap.setRange(0.0, 8.0)
        self.spn_gap.setSingleStep(0.25)
        self.spn_gap.setValue(p.refine.bridge_gap_s)
        self.spn_gap.setSuffix(" s")
        row("Bridge gaps up to", self.spn_gap)
        self.spn_extend = QDoubleSpinBox()
        self.spn_extend.setRange(0.0, 2.0)
        self.spn_extend.setSingleStep(0.1)
        self.spn_extend.setValue(p.refine.extend_s)
        self.spn_extend.setSuffix(" s")
        row("Extend ends by", self.spn_extend)
        self.btn_apply = QPushButton("Re-run tracking && scoring")
        self.btn_apply.clicked.connect(self.rerun_offline)
        self.btn_apply.setEnabled(False)
        v.addWidget(self.btn_apply)
        return g

    def _shortcuts(self) -> None:
        def sc(key: str, fn) -> None:
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)
        sc("Space", self.toggle_play)
        sc("Right", lambda: self.step(1))
        sc("Left", lambda: self.step(-1))
        sc("Shift+Right", lambda: self.step(10))
        sc("Shift+Left", lambda: self.step(-10))
        sc("Home", lambda: self.seek(0))
        sc("End", lambda: self.seek((self.frames.n_frames - 1) if self.frames else 0))
        sc("N", lambda: self.btn_draw.setChecked(not self.btn_draw.isChecked()))
        sc("E", self._toggle_selected)
        sc("Delete", lambda: self.selected is not None and self.delete_track(self.selected))
        sc("Ctrl+E", self.export)
        sc("Ctrl+O", lambda: self.open_video())
        sc("]", lambda: self.jump_alert(1))
        sc("[", lambda: self.jump_alert(-1))
        sc("S", self.next_suspicious)
        sc("B", lambda: self.chk_blur.setChecked(not self.chk_blur.isChecked()))

    # ── preset / config ─────────────────────────────────────────────────
    def _on_preset(self, name: str) -> None:
        self.preset_name = name
        p = PRESETS[name]
        self.spn_stride.setValue(p.analysis.stride)
        self.spn_pad.setValue(p.render.mask_pad)
        self.spn_lead.setValue(p.render.motion_lead)
        self.spn_disable.setValue(p.score.auto_disable_above)
        self.spn_spawn.setValue(p.analysis.spawn_conf)
        self.spn_age.setValue(p.analysis.max_age_s)
        self.spn_gap.setValue(p.refine.bridge_gap_s)
        self.spn_extend.setValue(p.refine.extend_s)
        idx = list(BLUR_STYLES.values()).index(p.render.blur_layers) \
            if p.render.blur_layers in BLUR_STYLES.values() else 0
        self.cmb_style.setCurrentIndex(idx)

    def current_preset(self) -> Preset:
        p = PRESETS[self.preset_name]
        return replace(
            p,
            analysis=replace(p.analysis, stride=self.spn_stride.value(),
                             spawn_conf=self.spn_spawn.value(),
                             max_age_s=self.spn_age.value()),
            refine=replace(p.refine, bridge_gap_s=self.spn_gap.value(),
                           extend_s=self.spn_extend.value()),
            score=replace(p.score, auto_disable_above=self.spn_disable.value()),
            render=self.render_cfg())

    def render_cfg(self) -> RenderConfig:
        p = PRESETS[self.preset_name].render
        return replace(p, region=self.cmb_region.currentText(),
                       mask_pad=self.spn_pad.value(),
                       motion_lead=self.spn_lead.value(),
                       blur_layers=BLUR_STYLES[self.cmb_style.currentText()])

    # ── file / jobs ─────────────────────────────────────────────────────
    def open_video(self, path: Optional[str] = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open video", "",
                "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*)")
            if not path:
                return
        if self.frames is not None:
            self.frames.close()
        self.frames = FrameSource(path)
        if not self.frames.ok:
            QMessageBox.warning(self, "Cannot open", f"Could not open {path}")
            self.frames = None
            return
        self.video = path
        self.project = None
        self.tracks = []
        self.table = []
        self.alerts = None
        self.selected = None
        self.cur = 0
        self.timeline.set_video(self.frames.n_frames, self.frames.fps)
        self.timeline.set_lanes([])
        self.timeline.set_alerts(None, None)
        self.inspector.clear()
        self.setWindowTitle(f"Video Privacy Pipeline — {Path(path).name}")
        self.show_frame(0)
        existing = proj.load(path)
        if existing is not None:
            self.project = existing
            self._on_status(f"Loaded analysis for {Path(path).name}; scoring…")
            self._set_busy(True)
            self.worker.submit(Job("offline", {"project": existing,
                                               "preset": self.current_preset()}))
        else:
            self._on_status("No analysis yet — press Analyse")
            self.canvas.set_message("Press Analyse")
            self.btn_export.setEnabled(False)

    def analyse(self) -> None:
        if not self.video or self.busy:
            return
        self._set_busy(True)
        self.worker.submit(Job("analyse", {"video": self.video,
                                           "preset": self.current_preset()}))

    def rerun_offline(self) -> None:
        if self.project is None or self.busy:
            return
        self._set_busy(True)
        self.worker.submit(Job("offline", {"project": self.project,
                                           "preset": self.current_preset()}))

    def export(self) -> None:
        if self.project is None or self.busy:
            return
        default = str(Path(self.video).with_name(Path(self.video).stem + "_blurred.mp4"))
        out, _ = QFileDialog.getSaveFileName(self, "Export blurred video", default,
                                             "MP4 video (*.mp4)")
        if not out:
            return
        self._save_project()
        try:
            from libs.labels import record as record_labels
            record_labels(self.video, self.project.fps, self.tracks,
                          self.project.enabled)
        except Exception:  # noqa: BLE001 — never block an export on the corpus
            pass
        self._set_busy(True)
        self.worker.submit(Job("export", {"project": self.project,
                                          "tracks": self.tracks, "out_path": out,
                                          "render": self.render_cfg()}))

    def _set_busy(self, on: bool) -> None:
        self.busy = on
        self.btn_analyse.setEnabled(not on and self.video is not None)
        self.btn_export.setEnabled(not on and self.project is not None)
        self.btn_apply.setEnabled(not on and self.project is not None)
        self.btn_cancel.setEnabled(on)
        self.progress.setVisible(on)
        if not on:
            self.progress.setValue(0)

    # ── worker callbacks ────────────────────────────────────────────────
    def _on_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _on_progress(self, stage: str, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.lbl_status.setText(f"{'Analysing' if stage == 'analyse' else 'Exporting'}"
                                f" {done}/{total}")

    def _on_analysed(self, project, tracks) -> None:
        self.project = project
        self._edits_counter = 0
        self.tracks = self._apply_edits(tracks)
        self._set_busy(False)
        self.rebuild()
        self._on_status(f"{len([t for t in tracks if t.kept])} heads tracked, "
                        f"{len([t for t in tracks if not t.kept])} set aside — "
                        f"review the top lanes, then Export")

    def _on_offline(self, tracks) -> None:
        self._edits_counter = 0
        self.tracks = self._apply_edits(tracks)
        self._set_busy(False)
        self.rebuild()
        self._on_status(f"{len([t for t in tracks if t.kept])} heads tracked, "
                        f"{len([t for t in tracks if not t.kept])} set aside")

    def _on_exported(self, ok: bool, msg: str) -> None:
        self._set_busy(False)
        self._on_status(msg)
        self.show_frame(self.cur)
        if ok:
            QMessageBox.information(self, "Export finished", msg)
        elif msg != "cancelled":
            QMessageBox.warning(self, "Export failed", msg)

    def _on_propagated(self, mt: ManualTrack) -> None:
        if self.project is None:
            return
        self.project.manual.append(mt)
        self.project.enabled[mt.tid] = True
        self._set_busy(False)
        self._schedule_save()
        self.rebuild()
        self.select(mt.tid)
        self._on_status(f"Added manual head over {len(mt.boxes)} frames "
                        f"({mt.start}–{mt.end})")

    def _on_preview(self) -> None:
        item = self.worker.take_preview()
        if item is None or not self.busy:
            return
        idx, frame = item
        self.canvas.set_frame(idx, frame)
        self.timeline.set_playhead(idx)

    def _on_failed(self, msg: str) -> None:
        self._set_busy(False)
        self._on_status(msg.splitlines()[0])
        QMessageBox.warning(self, "Pipeline error", msg)

    # ── model ───────────────────────────────────────────────────────────
    def rebuild(self) -> None:
        if self.project is None:
            return
        n = self.project.n_frames
        self.table = build_table(self.tracks, self.project.manual, n,
                                 self.project.enabled, self.render_cfg())
        cov = coverage(self.table)
        self.alerts = self._compute_alerts(cov)
        self.timeline.set_lanes(self._lanes())
        self.timeline.set_alerts(self.alerts, cov)
        self.show_frame(self.cur)
        self._refresh_inspector()
        self.btn_export.setEnabled(not self.busy)

    def _enabled(self, tr: Track) -> bool:
        return self.project.enabled.get(tr.tid, tr.kept)

    def _review_order(self) -> list[Track]:
        """Enabled tracks first, most suspicious on top (is this blur a
        false positive?), then disabled ones with the *least* suspicious on
        top (did the pipeline set aside a real head?)."""
        on = sorted((t for t in self.tracks if self._enabled(t)),
                    key=lambda t: -t.suspicion)
        off = sorted((t for t in self.tracks if not self._enabled(t)),
                     key=lambda t: t.suspicion)
        return on + off

    def _lanes(self) -> list[Lane]:
        lanes = []
        for m in self.project.manual:
            on = self.project.enabled.get(m.tid, True)
            lanes.append(Lane(m.tid, m.start, m.end, theme.LANE_MANUAL,
                              f"M{-m.tid - 999}", on, 0.0, True))
        fa = float(max(self.project.width * self.project.height, 1))
        for tr in self._review_order():
            on = self._enabled(tr)
            frac = float(np.median(tr.t.boxes[:, 2] * tr.t.boxes[:, 3])) / fa
            size = f"  {frac:.0%}" if frac >= 0.15 else ""     # big boxes deserve a look
            lanes.append(Lane(tr.tid, tr.t.start, tr.t.end,
                              theme.lane_rgb(on, tr.suspicion),
                              f"#{tr.tid}  {tr.suspicion:.2f}{size}", on, tr.suspicion,
                              False, tr.t.hits))
        return lanes

    def _compute_alerts(self, cov: np.ndarray) -> np.ndarray:
        """Frames where a confident raw candidate has no blur box near it."""
        n = self.project.n_frames
        alerts = np.zeros(n, bool)
        thr = self.current_preset().analysis.spawn_conf
        for idx, r in self.project.raw.items():
            if idx >= n:
                continue
            r = np.asarray(r).reshape(-1, 6)
            strong = r[r[:, 4] >= thr]
            if len(strong) == 0:
                continue
            boxes = np.array([b for _t, b in self.table[idx]], np.float32).reshape(-1, 4)
            if len(boxes) == 0:
                alerts[idx] = True
                continue
            m = iou_matrix(strong[:, :4], boxes)
            if (m.max(axis=1) < 0.1).any():
                alerts[idx] = True
        return alerts

    def _overlays(self, idx: int) -> list[Overlay]:
        out = []
        if self.project is None:
            return out
        by_tid = {tr.tid: tr for tr in self.tracks}
        manual = {m.tid for m in self.project.manual}
        for tid, box in (self.table[idx] if idx < len(self.table) else []):
            if tid in manual:
                col, label = theme.LANE_MANUAL, f"M{-tid - 999}"
            else:
                tr = by_tid.get(tid)
                s = tr.suspicion if tr else 0.0
                col = theme.lane_rgb(True, s)
                label = f"#{tid} {s:.2f}"
            out.append(Overlay(tid, box, col, label, selected=(tid == self.selected)))
        # disabled tracks in this frame, dashed, so they can be clicked back on
        for tr in self.tracks:
            if self._enabled(tr) or not (tr.t.start <= idx <= tr.t.end):
                continue
            from pipeline.geom import to_xyxy
            box = to_xyxy(tr.t.boxes[idx - tr.t.start])
            out.append(Overlay(tr.tid, box, theme.LANE_OFF, f"#{tr.tid} off",
                               selected=(tr.tid == self.selected), dashed=True))
        if self.chk_dets.isChecked():
            r = self.project.raw.get(idx)
            if r is not None:
                for row in np.asarray(r).reshape(-1, 6):
                    out.append(Overlay(-1, row[:4], (150, 150, 170), f"{row[4]:.2f}",
                                       dashed=True, thin=True))
        return out

    def show_frame(self, idx: int) -> None:
        if self.frames is None:
            return
        idx = int(np.clip(idx, 0, max(self.frames.n_frames - 1, 0)))
        self.cur = idx
        frame = self.frames.frame(idx)
        blur_boxes = None
        cfg = None
        if self.chk_blur.isChecked() and self.table and idx < len(self.table):
            blur_boxes = [b for _t, b in self.table[idx]]
            cfg = self.render_cfg()
        self.canvas.set_frame(idx, frame, blur_boxes, cfg)
        self.canvas.set_overlays(self._overlays(idx))
        self.timeline.set_playhead(idx, follow=True)
        self.inspector.set_playhead(idx)
        t = idx / max(self.frames.fps, 1e-6)
        m, s = divmod(t, 60)
        self.lbl_time.setText(f"{int(m)}:{s:05.2f} · frame {idx}")

    # ── navigation ──────────────────────────────────────────────────────
    def seek(self, f: int) -> None:
        self.show_frame(f)

    def step(self, n: int) -> None:
        self.show_frame(self.cur + n)

    def toggle_play(self) -> None:
        if self.frames is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("Play")
        else:
            self.play_timer.start(int(1000 / max(self.frames.fps, 1)))
            self.btn_play.setText("Pause")

    def _tick(self) -> None:
        if self.frames is None or self.cur + 1 >= self.frames.n_frames:
            self.toggle_play()
            return
        self.show_frame(self.cur + 1)

    def jump_alert(self, direction: int) -> None:
        if self.alerts is None:
            return
        idx = np.nonzero(self.alerts)[0]
        if len(idx) == 0:
            self._on_status("No alerts: every confident detection is covered")
            return
        if direction > 0:
            nxt = idx[idx > self.cur]
            target = int(nxt[0]) if len(nxt) else int(idx[0])
        else:
            prv = idx[idx < self.cur]
            target = int(prv[-1]) if len(prv) else int(idx[-1])
        self.show_frame(target)

    def next_suspicious(self) -> None:
        if not self.tracks:
            return
        order = self._review_order()
        tids = [t.tid for t in order]
        i = (tids.index(self.selected) + 1) % len(tids) if self.selected in tids else 0
        tr = order[i]
        self.select(tr.tid)
        if not (tr.t.start <= self.cur <= tr.t.end):
            self.show_frame(tr.t.start + int(np.argmax(tr.t.hits)) if tr.t.hits.any()
                            else tr.t.start)

    # ── selection / editing ─────────────────────────────────────────────
    def _on_lane_clicked(self, tid: int) -> None:
        self.select(tid)

    def select(self, tid: Optional[int]) -> None:
        self.selected = tid
        self.timeline.set_selected(tid)
        self._refresh_inspector()
        self.canvas.set_overlays(self._overlays(self.cur))

    def _refresh_inspector(self) -> None:
        if self.project is None or self.selected is None:
            self.inspector.clear()
            return
        tid = self.selected
        manual = next((m for m in self.project.manual if m.tid == tid), None)
        track = next((t for t in self.tracks if t.tid == tid), None)
        if manual is None and track is None:
            self.selected = None
            self.inspector.clear()
            return
        on = self.project.enabled.get(tid, track.kept if track else True)
        self.inspector.show_track(track, manual, on, self.frames, self.project.fps)

    def _toggle_selected(self) -> None:
        if self.selected is None or self.project is None:
            return
        track = next((t for t in self.tracks if t.tid == self.selected), None)
        on = self.project.enabled.get(self.selected, track.kept if track else True)
        self.set_enabled(self.selected, not on)

    def set_enabled(self, tid: int, on: bool) -> None:
        if self.project is None:
            return
        self.project.enabled[tid] = bool(on)
        self._schedule_save()
        self.rebuild()

    def delete_track(self, tid: int) -> None:
        if self.project is None:
            return
        before = len(self.project.manual)
        self.project.manual = [m for m in self.project.manual if m.tid != tid]
        if len(self.project.manual) == before:
            self.project.enabled[tid] = False          # pipeline track: just disable
        else:
            self.project.enabled.pop(tid, None)
        if self.selected == tid:
            self.selected = None
        self._schedule_save()
        self.rebuild()

    def _record_edit(self, edit: dict) -> None:
        self.project.settings.setdefault("edits", []).append(edit)
        self._schedule_save()

    def _apply_edits(self, tracks: list[Track]) -> list[Track]:
        """Replay stored split/trim edits on freshly scored tracks so they
        survive a re-run of the offline stages."""
        if self.project is None:
            return tracks
        for e in self.project.settings.get("edits", []):
            tracks = self._edit(tracks, e)
        return tracks

    def _edit(self, tracks: list[Track], e: dict) -> list[Track]:
        tid, f = int(e["tid"]), int(e["frame"])
        for i, tr in enumerate(tracks):
            if tr.tid != tid:
                continue
            t = tr.t
            k = f - t.start
            if not (0 < k < len(t.boxes)):
                return tracks
            if e["op"] == "split":
                self._edits_counter += 1
                new_tid = 100000 + self._edits_counter
                a = replace(tr, t=t.slice(0, k))
                b = replace(tr, t=t.slice(k, len(t.boxes)))
                b.t.tid = new_tid
                if tid in self.project.enabled:
                    self.project.enabled[new_tid] = self.project.enabled[tid]
                return tracks[:i] + [a, b] + tracks[i + 1:]
            if e["op"] == "trim":
                sl = t.slice(k, len(t.boxes)) if e["side"] == "before" else t.slice(0, k)
                return tracks[:i] + [replace(tr, t=sl)] + tracks[i + 1:]
        return tracks

    def split_track(self, tid: int, frame: int) -> None:
        if self.project is None:
            return
        e = {"op": "split", "tid": tid, "frame": frame}
        self.tracks = self._edit(self.tracks, e)
        self._record_edit(e)
        self.rebuild()

    def trim_track(self, tid: int, frame: int, side: str) -> None:
        if self.project is None:
            return
        manual = next((m for m in self.project.manual if m.tid == tid), None)
        if manual is not None:
            k = frame - manual.start
            if 0 < k < len(manual.boxes):
                if side == "before":
                    manual.boxes = manual.boxes[k:]
                    manual.start = frame
                else:
                    manual.boxes = manual.boxes[:k]
            self._schedule_save()
            self.rebuild()
            return
        e = {"op": "trim", "tid": tid, "frame": frame, "side": side}
        self.tracks = self._edit(self.tracks, e)
        self._record_edit(e)
        self.rebuild()

    def _on_box_drawn(self, frame: int, box: np.ndarray) -> None:
        if self.project is None or self.video is None:
            self._on_status("Analyse first, then draw missed heads")
            return
        tid = self.project.next_manual_id()
        self._set_busy(True)
        self._on_status("Following the drawn head forward and backward…")
        self.worker.submit(Job("propagate", {"video": self.video, "frame": frame,
                                             "box": box, "raw": self.project.raw,
                                             "tid": tid, "both": True}))

    # ── persistence ─────────────────────────────────────────────────────
    def _schedule_save(self) -> None:
        self.save_timer.start(600)

    def _save_project(self) -> None:
        if self.project is None:
            return
        try:
            proj.save(self.project)
        except OSError as exc:
            self._on_status(f"Could not save decisions: {exc}")

    def closeEvent(self, ev) -> None:  # noqa: N802
        self._save_project()
        self.worker.stop()
        self.worker.wait(2000)
        if self.frames is not None:
            self.frames.close()
        super().closeEvent(ev)


def _run_preflight(app: QApplication, splash) -> None:
    from libs.models import preflight
    latest = {"msg": "Checking models…"}
    lock = threading.Lock()
    done = threading.Event()

    def on_status(msg: str) -> None:
        with lock:
            latest["msg"] = msg

    def work() -> None:
        try:
            preflight(on_status)
        except Exception:  # noqa: BLE001
            pass
        finally:
            done.set()

    threading.Thread(target=work, name="model-preflight", daemon=True).start()
    shown = None
    while not done.is_set():
        with lock:
            msg = latest["msg"]
        if msg != shown and splash is not None:
            shown = msg
            from splash import update as splash_update
            splash_update(splash, msg)
        else:
            app.processEvents()
        time.sleep(0.03)


def main(splash=None, argv: Optional[list[str]] = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    if os.environ.get("AVPP_SKIP_PREFLIGHT", "0") != "1":
        _run_preflight(app, splash)
    app.setStyleSheet(theme.STYLE)
    win = MainWindow()
    win.show()
    argv = argv if argv is not None else sys.argv[1:]
    if argv and Path(argv[0]).is_file():
        win.open_video(argv[0])
    if splash is not None:
        splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
