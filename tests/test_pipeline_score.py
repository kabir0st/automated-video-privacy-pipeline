"""Suspicion ranks, never deletes; verification only lowers suspicion."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.score import ScoreConfig, components, score_tracks, suspicion, verify_ratio  # noqa: E402
from pipeline.types import Src, Tracklet   # noqa: E402


def tl(tid, n=40, src=int(Src.HEADDET), score=0.7, vx=3.0):
    boxes = np.stack([np.array([100.0 + vx * i, 200.0, 90.0, 110.0], np.float32)
                      for i in range(n)])
    return Tracklet(tid, 0, boxes, np.full(n, score, np.float32),
                    np.ones(n, bool), boxes.copy(), np.full(n, src, np.uint32),
                    np.zeros(n, bool))


def test_lonely_scores_higher_than_agreeing():
    lonely = suspicion(components(tl(1), 25, None), ScoreConfig())
    agree = suspicion(components(tl(2, src=int(Src.HEADDET | Src.MULTI)), 25, None),
                      ScoreConfig())
    assert lonely > agree


def test_verification_lowers_suspicion():
    cfg = ScoreConfig()
    assert suspicion(components(tl(1), 25, 1.0), cfg) < \
        suspicion(components(tl(1), 25, 0.0), cfg)


def test_static_blob_more_suspicious_than_moving():
    cfg = ScoreConfig()
    assert suspicion(components(tl(1, vx=0.0), 25, None), cfg) > \
        suspicion(components(tl(2, vx=3.0), 25, None), cfg)


def test_rejected_tracks_default_off():
    tracks = score_tracks([tl(1)], [tl(2)], 25, ScoreConfig(),
                          reject_reasons={2: "sparse"})
    by = {t.tid: t for t in tracks}
    assert by[1].kept and not by[2].kept
    assert by[2].suspicion >= 0.9 and by[2].reasons["why"] == "sparse"


def test_auto_disable_threshold():
    t = tl(1, n=3, score=0.25)                # short, weak, lonely
    hi = score_tracks([t], [], 25, ScoreConfig(auto_disable_above=1.01))
    lo = score_tracks([t], [], 25, ScoreConfig(auto_disable_above=0.3))
    assert hi[0].kept and not lo[0].kept


class TestVerify:
    def test_redetection_counts(self):
        t = tl(1)
        frame = np.zeros((720, 1280, 3), np.uint8)
        calls = []

        def detect(crop):
            calls.append(crop.shape)
            h, w = crop.shape[:2]
            return np.array([[w * 0.3, h * 0.3, w * 0.7, h * 0.7, 0.9]], np.float32)
        r = verify_ratio(t, lambda i: frame, detect, ScoreConfig(verify_samples=4))
        assert r == 1.0 and len(calls) == 4

    def test_small_head_inside_giant_box_does_not_verify_it(self):
        boxes = np.tile(np.array([640.0, 360.0, 900.0, 600.0], np.float32), (20, 1))
        t = Tracklet(1, 0, boxes, np.full(20, 0.6, np.float32), np.ones(20, bool),
                     boxes.copy(), np.full(20, int(Src.WB_HEAD), np.uint32), np.zeros(20, bool))
        frame = np.zeros((720, 1280, 3), np.uint8)

        def detect(crop):                     # a real head-sized box at the centre
            h, w = crop.shape[:2]
            return np.array([[w * 0.45, h * 0.45, w * 0.55, h * 0.6, 0.9]], np.float32)
        assert verify_ratio(t, lambda i: frame, detect, ScoreConfig()) == 0.0

    def test_no_redetection_is_zero(self):
        t = tl(1)
        frame = np.zeros((720, 1280, 3), np.uint8)
        r = verify_ratio(t, lambda i: frame, lambda c: np.empty((0, 5), np.float32),
                         ScoreConfig())
        assert r == 0.0

    def test_unreadable_frames_is_none(self):
        assert verify_ratio(tl(1), lambda i: None, lambda c: None, ScoreConfig()) is None
