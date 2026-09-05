"""Tracker + recorder policy: low spawn bar, long coast, face memory."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.record import TrackRecorder     # noqa: E402
from pipeline.tracker import Tracker           # noqa: E402
from pipeline.types import Src                 # noqa: E402

SHAPE = (720, 1280, 3)
E = np.empty((0, 5), np.float32)


def head(x, s=0.6):
    return np.array([[x, 200, x + 100, 320, s]], np.float32)


def run(seq, **kw):
    tr = Tracker(fps=25, **kw)
    rec = TrackRecorder()
    for f, (c, fl) in enumerate(seq):
        obs = tr.update(c, fl, SHAPE)
        rec.observe(f, obs)
    return rec.finalize()


class TestPolicy:
    def test_continuous_head_is_one_track(self):
        seq = [(head(100 + 3 * f), np.array([int(Src.HEADDET)])) for f in range(30)]
        tl = run(seq)
        assert len(tl) == 1 and tl[0].hits.sum() == 30

    def test_gap_below_max_age_keeps_id(self):
        seq = [(head(100 + 3 * f), np.array([1])) for f in range(10)]
        seq += [(E, np.empty(0, np.int64))] * 15
        seq += [(head(100 + 3 * f), np.array([1])) for f in range(25, 40)]
        tl = run(seq, max_age_s=1.0)   # 25 frames
        assert len(tl) == 1
        assert tl[0].hits.sum() == 25
        assert (~tl[0].hits).sum() == 15          # coasted, recorded as misses

    def test_gap_above_max_age_splits(self):
        seq = [(head(100), np.array([1]))] * 10
        seq += [(E, np.empty(0, np.int64))] * 40
        seq += [(head(100), np.array([1]))] * 10
        tl = run(seq, max_age_s=1.0)
        assert len(tl) == 2

    def test_weak_candidate_never_spawns_but_sustains(self):
        weak = [(head(100, 0.2), np.array([1]))] * 10
        assert run(weak, spawn_conf=0.35, sustain_conf=0.1) == []
        seq = [(head(100, 0.6), np.array([1]))] * 3 + [(head(100, 0.2), np.array([1]))] * 10
        tl = run(seq, spawn_conf=0.35, sustain_conf=0.1)
        assert len(tl) == 1 and tl[0].hits.sum() == 13
        assert (tl[0].src_arr()[3:] & int(Src.SUSTAIN)).all()

    def test_unconfirmed_track_dies_quickly(self):
        seq = [(head(100), np.array([1]))] + [(E, np.empty(0, np.int64))] * 5
        tl = run(seq, min_hits=2)
        # never confirmed: recorded but with a single hit
        assert all(t.hits.sum() <= 1 for t in tl)

    def test_face_memory_rides_head(self):
        tr = Tracker(fps=25)
        face = np.array([[130, 230, 170, 280, 0.9]], np.float32)
        obs = tr.update(head(100), np.array([1]), SHAPE, faces=face)
        assert obs[0].face_age == 0
        obs = tr.update(head(150), np.array([1]), SHAPE)        # head moved, no face
        assert obs[0].face_age == 1
        fb = obs[0].face_box
        assert fb is not None and 170 < fb[0] < 200               # followed the head


class TestSizeGates:
    def test_giant_weak_box_does_not_hijack_head_track(self):
        tr = Tracker(fps=25, spawn_conf=0.35, sustain_conf=0.1)
        for _ in range(3):
            tr.update(head(300, 0.7), np.array([1]), SHAPE)
        # a torso-sized weak box containing the head: IoU with the track ≈ 0.27
        torso = np.array([[250, 150, 550, 600, 0.3]], np.float32)
        obs = tr.update(torso, np.array([1]), SHAPE)
        b = obs[0].box
        assert (b[2] - b[0]) < 140 and not obs[0].hit          # coasted, not inflated

    def test_giant_confident_box_spawns_its_own_track(self):
        tr = Tracker(fps=25, spawn_conf=0.35, sustain_conf=0.1)
        for _ in range(3):
            tr.update(head(300, 0.7), np.array([1]), SHAPE)
        torso = np.array([[250, 150, 550, 600, 0.6]], np.float32)
        obs = tr.update(torso, np.array([1]), SHAPE)
        widths = sorted((o.box[2] - o.box[0]) for o in obs)
        assert len(obs) == 2 and widths[0] < 140 < widths[1]


class TestRecorder:
    def test_hole_is_padded_not_split(self):
        from pipeline.tracker import TrackObs
        rec = TrackRecorder()
        b = np.array([0, 0, 10, 10], np.float32)
        rec.observe(0, [TrackObs(1, b, 0.5, True, True, 0)])
        rec.observe(3, [TrackObs(1, b, 0.5, True, True, 0)])
        tl = rec.finalize()
        assert len(tl) == 1 and len(tl[0].boxes) == 4
        assert tl[0].hits.tolist() == [True, False, False, True]
