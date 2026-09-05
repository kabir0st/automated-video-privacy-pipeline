"""Blur table semantics and project sidecar round-trip."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import project as proj                     # noqa: E402
from pipeline.render import RenderConfig, build_table     # noqa: E402
from pipeline.types import ManualTrack, Src, Track, Tracklet  # noqa: E402


def track(tid, n=20, vx=0.0, kept=True, fvalid=False):
    boxes = np.stack([np.array([200.0 + vx * i, 200.0, 100.0, 120.0], np.float32)
                      for i in range(n)])
    fboxes = boxes.copy(); fboxes[:, 2:] *= 0.5
    t = Tracklet(tid, 5, boxes, np.full(n, 0.7, np.float32), np.ones(n, bool),
                 fboxes, np.full(n, int(Src.HEADDET), np.uint32),
                 np.full(n, fvalid, bool))
    return Track(t=t, suspicion=0.1, kept=kept)


class TestTable:
    def test_default_verdict_and_overrides(self):
        tab = build_table([track(1), track(2, kept=False)], [], 40)
        assert all(len(tab[f]) == 1 for f in range(5, 25))
        tab = build_table([track(1), track(2, kept=False)], [], 40, {1: False, 2: True})
        assert all([tid for tid, _ in tab[10]] == [2] for _ in [0])

    def test_outside_span_empty(self):
        tab = build_table([track(1)], [], 40)
        assert tab[0] == [] and tab[30] == []

    def test_motion_pad_grows_moving_only(self):
        static = build_table([track(1, vx=0.0)], [], 40, cfg=RenderConfig(motion_lead=1.0))
        moving = build_table([track(1, vx=10.0)], [], 40, cfg=RenderConfig(motion_lead=1.0))
        ws = static[10][0][1][2] - static[10][0][1][0]
        wm = moving[10][0][1][2] - moving[10][0][1][0]
        assert abs(ws - 100) < 1e-3 and wm > 115

    def test_face_region_falls_back_to_head(self):
        tab = build_table([track(1, fvalid=False)], [], 40, cfg=RenderConfig(region="face"))
        w = tab[10][0][1][2] - tab[10][0][1][0]
        assert abs(w - 100) < 1e-3
        tab = build_table([track(1, fvalid=True)], [], 40, cfg=RenderConfig(region="face"))
        w = tab[10][0][1][2] - tab[10][0][1][0]
        assert w < 100                           # bloomed face, tighter than head

    def test_manual_tracks_included(self):
        m = ManualTrack(-1000, 2, np.tile(np.array([0, 0, 50, 50], np.float32), (5, 1)))
        tab = build_table([], [m], 10)
        assert [tid for tid, _ in tab[3]] == [-1000]
        tab = build_table([], [m], 10, {-1000: False})
        assert tab[3] == []


class TestProject:
    def test_roundtrip_and_fingerprint(self, tmp_path):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 100)
        t = track(7).t
        p = proj.Project(video=str(video), fps=25.0, n_frames=100, stride=2,
                         fingerprint=proj.fingerprint(video, {"a": 1}),
                         tracklets=[t], raw={3: np.array([[1, 2, 3, 4, 0.5, 9]], np.float32)},
                         enabled={7: False}, manual=[ManualTrack(-1000, 1, np.zeros((3, 4), np.float32))],
                         settings={"k": "v"},
                         raw_sources={3: {"headdet": np.array([[1, 2, 3, 4, 0.5]], np.float32),
                                          "scrfd": np.empty((0, 5), np.float32)}})
        proj.save(p)
        q = proj.load(video, {"a": 1})
        assert q is not None and q.stride == 2 and q.enabled == {7: False}
        assert len(q.tracklets) == 1 and np.allclose(q.tracklets[0].boxes, t.boxes)
        assert 3 in q.raw and q.raw[3].shape == (1, 6)
        assert q.raw_sources[3]["headdet"].shape == (1, 5) and "scrfd" not in q.raw_sources[3]
        assert q.manual[0].tid == -1000 and q.manual[0].boxes.shape == (3, 4)
        assert proj.load(video, {"a": 2}) is None       # different analysis params
        assert proj.load(video) is not None             # no params → any fingerprint
        assert q.next_manual_id() == -1001
