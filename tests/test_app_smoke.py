"""Headless smoke of the editor's model logic (offscreen Qt, no video, no
models): review ordering, alert computation, edit replay, table rebuild."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AVPP_SKIP_PREFLIGHT"] = "1"

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication            # noqa: E402

from pipeline import project as proj                # noqa: E402
from pipeline.types import Src, Track, Tracklet     # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _track(tid, start, n, susp, kept=True, x=100.0):
    boxes = np.stack([np.array([x + 2 * i, 200.0, 90.0, 110.0], np.float32)
                      for i in range(n)])
    t = Tracklet(tid, start, boxes, np.full(n, 0.7, np.float32), np.ones(n, bool),
                 boxes.copy(), np.full(n, int(Src.HEADDET), np.uint32), np.zeros(n, bool))
    return Track(t=t, suspicion=susp, reasons={"lonely": susp}, kept=kept)


@pytest.fixture
def win(app, tmp_path):
    from app.window import MainWindow
    w = MainWindow()
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 64)
    w.project = proj.Project(video=str(video), fps=25.0, n_frames=200, stride=1,
                             fingerprint="f", width=1280, height=720)
    # a confident raw candidate far from every track at frame 150 → alert
    w.project.raw = {150: np.array([[900, 500, 1000, 620, 0.8, 1]], np.float32),
                     10: np.array([[100, 145, 190, 255, 0.8, 1]], np.float32)}
    w.tracks = [_track(1, 0, 50, 0.10), _track(2, 60, 50, 0.70),
                _track(3, 120, 20, 0.95, kept=False), _track(4, 0, 5, 0.30, kept=False)]
    w.rebuild()
    yield w
    w.worker.stop()
    w.worker.wait(1000)
    w.close()


def test_review_order_enabled_first_then_plausible_rejects(win):
    order = [t.tid for t in win._review_order()]
    assert order == [2, 1, 4, 3]          # enabled by suspicion desc, then off by asc


def test_alerts_only_where_confident_detection_is_uncovered(win):
    assert win.alerts is not None
    assert win.alerts[150] and not win.alerts[10]


def test_toggle_updates_table_and_lanes(win):
    assert len(win.table[70]) == 1
    win.set_enabled(2, False)
    assert win.table[70] == []
    lane = next(l for l in win.timeline._lanes if l.tid == 2)
    assert not lane.enabled


def test_split_and_replay(win):
    win.split_track(1, 25)
    tids = sorted(t.tid for t in win.tracks)
    assert 100001 in tids and len(win.tracks) == 5
    a = next(t for t in win.tracks if t.tid == 1)
    b = next(t for t in win.tracks if t.tid == 100001)
    assert a.t.end == 24 and b.t.start == 25 and b.t.end == 49
    # replay on fresh tracks reproduces the split deterministically
    fresh = [_track(1, 0, 50, 0.10), _track(2, 60, 50, 0.70)]
    win._edits_counter = 0
    replayed = win._apply_edits(fresh)
    assert sorted(t.tid for t in replayed) == [1, 2, 100001]


def test_trim_before_and_after(win):
    win.trim_track(2, 80, "before")
    t = next(t for t in win.tracks if t.tid == 2).t
    assert t.start == 80 and t.end == 109
    win.trim_track(2, 100, "after")
    t = next(t for t in win.tracks if t.tid == 2).t
    assert t.start == 80 and t.end == 99


def test_manual_track_in_table_and_lanes(win):
    from pipeline.types import ManualTrack
    win._on_propagated(ManualTrack(-1000, 150, np.tile(
        np.array([900, 500, 1000, 620], np.float32), (10, 1))))
    assert any(tid == -1000 for tid, _ in win.table[155])
    assert not win.alerts[150]                    # the alert is now covered
    assert win.timeline._lanes[0].tid == -1000    # manual lanes first


def test_delete_manual_and_disable_pipeline(win):
    from pipeline.types import ManualTrack
    win._on_propagated(ManualTrack(-1000, 150, np.zeros((5, 4), np.float32)))
    win.delete_track(-1000)
    assert not win.project.manual
    win.delete_track(1)
    assert win.project.enabled[1] is False
