"""Background worker: owns the ONNX sessions for the process lifetime and
runs one job at a time — analyse, refine/score, export, propagate. Every
result travels back through Qt signals; frames for the live preview go
through a latest-wins mailbox so the GUI never queues stale images."""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from pipeline import project as proj
from pipeline.analysis import Models
from pipeline.presets import Preset
from pipeline.propagate import PropagateConfig, propagate
from pipeline.render import RenderConfig, build_table, render_video
from pipeline.session import ensure_analysed, run_offline
from pipeline.types import ManualTrack, Track


@dataclass
class Job:
    kind: str                       # analyse | offline | export | propagate
    args: dict = field(default_factory=dict)


class PipelineWorker(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)            # stage, done, total
    analysed = pyqtSignal(object, object)           # Project, list[Track]
    offline_done = pyqtSignal(object)               # list[Track]
    exported = pyqtSignal(bool, str)
    propagated = pyqtSignal(object)                 # ManualTrack
    preview_ready = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._models: Optional[Models] = None
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._cancel = threading.Event()
        self._running = True
        self._preview: Optional[tuple[int, np.ndarray]] = None

    # ── job API (GUI thread) ─────────────────────────────────────────────
    def submit(self, job: Job) -> None:
        with self._lock:
            self._jobs.append(job)
        self._wake.set()

    def cancel(self) -> None:
        self._cancel.set()

    def stop(self) -> None:
        self._running = False
        self._cancel.set()
        self._wake.set()

    def take_preview(self) -> Optional[tuple[int, np.ndarray]]:
        with self._lock:
            p, self._preview = self._preview, None
        return p

    def _emit_preview(self, idx: int, frame: np.ndarray) -> None:
        with self._lock:
            empty = self._preview is None
            self._preview = (idx, frame)
        if empty:
            self.preview_ready.emit()

    # ── thread body ──────────────────────────────────────────────────────
    def models(self) -> Models:
        if self._models is None:
            self._models = Models(on_status=self.status.emit)
        return self._models

    def run(self) -> None:
        while self._running:
            self._wake.wait()
            self._wake.clear()
            while self._running:
                with self._lock:
                    if not self._jobs:
                        break
                    job = self._jobs.pop(0)
                self._cancel.clear()
                try:
                    getattr(self, f"_do_{job.kind}")(**job.args)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(f"{job.kind} failed: {exc!r}\n"
                                     f"{traceback.format_exc()}")

    def _do_analyse(self, video: str, preset: Preset, use_cache: bool = True) -> None:
        def prog(done: int, total: int) -> None:
            self.progress.emit("analyse", done, total)

        def preview(idx, frame, cands, obs) -> None:
            self._emit_preview(idx, frame)

        p = ensure_analysed(video, preset, self.models(), use_cache=use_cache,
                            progress=prog, cancel=self._cancel,
                            on_status=self.status.emit, preview=preview)
        if p is None:
            self.status.emit("Analysis cancelled")
            return
        self.status.emit("Refining and scoring…")
        tracks = run_offline(p, preset, self.models(), on_status=self.status.emit)
        self.analysed.emit(p, tracks)

    def _do_offline(self, project: proj.Project, preset: Preset,
                    verify: bool = True) -> None:
        tracks = run_offline(project, preset, self.models(), verify=verify,
                             on_status=self.status.emit)
        self.offline_done.emit(tracks)

    def _do_export(self, project: proj.Project, tracks: list[Track],
                   out_path: str, render: RenderConfig) -> None:
        table = build_table(tracks, project.manual, project.n_frames,
                            project.enabled, render)

        def prog(done: int, total: int) -> None:
            self.progress.emit("export", done, total)

        ok, msg = render_video(project.video, out_path, table, render,
                               progress=prog, cancel=self._cancel,
                               on_status=self.status.emit,
                               preview=self._emit_preview)
        self.exported.emit(ok, msg)

    def _do_propagate(self, video: str, frame: int, box: np.ndarray,
                      raw: dict, tid: int, both: bool = True,
                      max_frames: int = 300) -> None:
        cfg = PropagateConfig(max_frames=max_frames)
        fwd = propagate(video, frame, box, direction=1, raw=raw, cfg=cfg,
                        cancel=self._cancel)
        back = (propagate(video, frame, box, direction=-1, raw=raw, cfg=cfg,
                          cancel=self._cancel) if both else [(frame, box)])
        seq = sorted({f: b for f, b in list(back) + list(fwd)}.items())
        start = seq[0][0]
        boxes = np.stack([b for _f, b in seq]).astype(np.float32)
        # Fill any hole with a linear interpolation so the track is dense.
        frames = np.array([f for f, _b in seq])
        dense = np.arange(start, seq[-1][0] + 1)
        if len(dense) != len(frames):
            boxes = np.stack([np.interp(dense, frames, boxes[:, c])
                              for c in range(4)], axis=1).astype(np.float32)
        self.propagated.emit(ManualTrack(tid, int(start), boxes))
