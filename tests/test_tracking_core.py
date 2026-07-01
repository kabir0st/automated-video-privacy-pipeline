"""Unit tests for the new detection→tracking→cleanup core.

Covers the failure modes the rewrite exists to fix:
  * flicker: 1-frame dropouts must not drop a track;
  * occlusion: low-score detections sustain (BYTE) but never spawn;
  * false positives: min_hits suppresses short-lived tracks, offline prune
    kills short tracklets;
  * ghosts: coasted tails are trimmed, far detections are not stolen;
  * bridging: clean gaps interpolate, corridors/ambiguity refuse bad joins.

All synthetic — no models, runs on CPU in WSL.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.detector import Detections, fuse_heads, _nms, _unrotate_boxes  # noqa: E402
from libs.head_tracker import HeadTracker, TrackObs  # noqa: E402
from libs.tracklets import (PostParams, TrackRecorder, Tracklet,  # noqa: E402
                            postprocess)

SHAPE = (720, 1280)


def det(x, y, w=100, h=120, score=0.9):
    return np.array([x, y, x + w, y + h, score], dtype=np.float32)


# ── detector post-processing ────────────────────────────────────────────────

class TestFuseHeads:
    def test_face_inside_head_is_not_duplicated(self):
        d = Detections(heads=np.array([det(100, 100)]),
                       faces=np.array([det(120, 130, 50, 60)]))
        assert len(fuse_heads(d)) == 1

    def test_orphan_face_grows_pseudo_head(self):
        d = Detections(heads=np.array([det(100, 100)]),
                       faces=np.array([det(800, 400, 50, 60)]))
        fused = fuse_heads(d)
        assert len(fused) == 2
        pseudo = fused[1]
        fw, fh = pseudo[2] - pseudo[0], pseudo[3] - pseudo[1]
        assert fw == pytest.approx(50 * 1.7, rel=0.01)
        assert fh == pytest.approx(60 * 1.9, rel=0.01)
        # centre shifted up relative to the face centre
        assert (pseudo[1] + pseudo[3]) / 2 < 400 + 30

    def test_no_faces_passthrough(self):
        d = Detections(heads=np.array([det(0, 0)]))
        assert len(fuse_heads(d)) == 1


class TestRotations:
    def test_unrotate_90_roundtrip(self):
        # A box at the top-left of a 90°CW-rotated frame came from the
        # bottom-left of the original.
        fw, fh = 1280, 720
        b = np.array([[10, 20, 110, 140, 0.9]], dtype=np.float32)
        out = _unrotate_boxes(b, 90, fw, fh)
        assert out[0, 0] == pytest.approx(20)
        assert out[0, 2] == pytest.approx(140)
        assert out[0, 1] == pytest.approx(fh - 1 - 110)
        assert out[0, 3] == pytest.approx(fh - 1 - 10)

    def test_unrotate_180(self):
        fw, fh = 1280, 720
        b = np.array([[0, 0, 100, 100, 0.9]], dtype=np.float32)
        out = _unrotate_boxes(b, 180, fw, fh)
        assert out[0, 2] == pytest.approx(fw - 1)
        assert out[0, 3] == pytest.approx(fh - 1)

    def test_nms_merges_duplicates(self):
        a = det(100, 100, score=0.9)
        b = det(104, 102, score=0.7)   # near-identical
        c = det(600, 300, score=0.8)
        out = _nms(np.stack([a, b, c]))
        assert len(out) == 2
        assert out[0, 4] == pytest.approx(0.9)


# ── tracker ─────────────────────────────────────────────────────────────────

def run_frames(tracker, frames):
    outs = []
    for dets in frames:
        arr = np.stack(dets) if dets else np.empty((0, 5), np.float32)
        outs.append(tracker.update(arr, SHAPE))
    return outs


class TestHeadTracker:
    def test_confirms_after_min_hits(self):
        tr = HeadTracker(min_hits=3)
        outs = run_frames(tr, [[det(100, 100)]] * 4)
        assert not outs[0][0].confirmed
        assert not outs[1][0].confirmed
        assert outs[2][0].confirmed

    def test_one_frame_dropout_keeps_track(self):
        tr = HeadTracker(min_hits=2, max_age_s=1.0, fps=30)
        frames = [[det(100, 100)]] * 5 + [[]] + [[det(102, 101)]] * 3
        outs = run_frames(tr, frames)
        ids = {o.track_id for frame in outs for o in frame}
        assert ids == {1}          # never re-spawned
        assert outs[5][0].coast_frames == 1
        assert outs[6][0].hit

    def test_low_score_sustains_but_never_spawns(self):
        tr = HeadTracker(min_hits=2, det_conf=0.5, det_conf_low=0.1)
        # only low-score dets: nothing must ever spawn
        outs = run_frames(tr, [[det(100, 100, score=0.3)]] * 5)
        assert all(len(f) == 0 for f in outs)
        # established track survives an occlusion dip to 0.2
        tr.reset()
        frames = ([[det(100, 100, score=0.9)]] * 3
                  + [[det(100, 100, score=0.2)]] * 10
                  + [[det(100, 100, score=0.9)]])
        outs = run_frames(tr, frames)
        assert len(outs[-1]) == 1
        assert all(len(f) == 1 for f in outs[3:])

    def test_min_hits_suppresses_flash_fp(self):
        tr = HeadTracker(min_hits=3)
        frames = [[det(500, 300)], [det(500, 300)], [], [], []]
        outs = run_frames(tr, frames)
        # never confirmed, and dead immediately after the first miss
        assert all(not o.confirmed for f in outs for o in f)
        assert len(outs[2]) == 0

    def test_two_close_heads_keep_ids(self):
        tr = HeadTracker(min_hits=1)
        a, b = det(100, 100), det(230, 100)
        outs = run_frames(tr, [[a, b]] * 2)
        id_a = next(o.track_id for o in outs[0]
                    if abs(o.box[0] - 100) < 20)
        # swap the detection order — ids must not swap
        outs2 = run_frames(tr, [[b, a]] * 3)
        for f in outs2:
            for o in f:
                if abs(o.box[0] - 100) < 30:
                    assert o.track_id == id_a

    def test_far_detection_not_stolen(self):
        """The old tracker's threshold-0.0 centre match dragged a stale track
        onto any detection within a diagonal; the gates must refuse this."""
        tr = HeadTracker(min_hits=1, max_age_s=2.0, fps=30)
        run_frames(tr, [[det(100, 100)]] * 3)
        # track coasts; a new far detection with a very different size shows up
        outs = run_frames(tr, [[det(400, 400, w=400, h=500)]] * 3)
        ids = {o.track_id for o in outs[-1]}
        assert 2 in ids            # the far det spawned its own track
        boxes1 = [o for o in outs[0] if o.track_id == 1]
        assert not boxes1 or not boxes1[0].hit   # old track was not dragged

    def test_coasting_follows_motion(self):
        tr = HeadTracker(min_hits=1, max_age_s=1.0, fps=30)
        frames = [[det(100 + 10 * i, 100)] for i in range(10)]
        outs = run_frames(tr, frames)
        x_last_hit = outs[-1][0].box[0]
        outs2 = run_frames(tr, [[]] * 3)
        # prediction keeps moving right instead of freezing
        assert outs2[-1][0].box[0] > x_last_hit + 10


# ── offline cleanup ─────────────────────────────────────────────────────────

def make_tracklet(tid, start, n, x0=100.0, vx=0.0, score=0.9, w=100.0,
                  h=120.0):
    boxes = np.stack([np.array([x0 + vx * i, 200.0, w, h], np.float32)
                      for i in range(n)])
    return Tracklet(tid, start, boxes,
                    np.full(n, score, np.float32), np.ones(n, bool))


PP = PostParams(det_conf=0.5, min_hits=3, min_track_s=0.25,
                bridge_gap_s=1.5, smooth_win_s=0.5)


class TestPostprocess:
    def test_short_tracklet_pruned(self):
        t = make_tracklet(1, 10, 3)     # 3 frames < 0.25s @ 30fps
        table = postprocess([t], fps=30, n_frames=100, p=PP)
        assert all(len(f) == 0 for f in table)

    def test_low_confidence_pruned(self):
        t = make_tracklet(1, 10, 30, score=0.3)
        table = postprocess([t], fps=30, n_frames=100, p=PP)
        assert all(len(f) == 0 for f in table)

    def test_coasted_tail_trimmed(self):
        t = make_tracklet(1, 10, 30)
        t.hits[-10:] = False            # coasted tail
        table = postprocess([t], fps=30, n_frames=100, p=PP)
        ext = round(0.12 * 30)
        last = max(i for i, f in enumerate(table) if f)
        assert last == 10 + 19 + ext    # last hit + extension, not the tail

    def test_clean_gap_bridged_and_interpolated(self):
        a = make_tracklet(1, 0, 30, x0=100, vx=2)
        b = make_tracklet(2, 50, 30, x0=100 + 2 * 50, vx=2)
        table = postprocess([a, b], fps=30, n_frames=120, p=PP)
        # gap frames are covered
        assert all(table[f] for f in range(30, 50))
        # and interpolation moves along the path, not held static
        x35 = (table[35][0][1][0] + table[35][0][1][2]) / 2
        x45 = (table[45][0][1][0] + table[45][0][1][2]) / 2
        assert x45 > x35 + 10

    def test_corridor_blocks_bridge_through_other_head(self):
        a = make_tracklet(1, 0, 30, x0=100)
        b = make_tracklet(2, 50, 30, x0=100)
        # third head sits exactly on the corridor during the gap
        c = make_tracklet(3, 30, 20, x0=100)
        table = postprocess([a, b, c], fps=30, n_frames=120, p=PP)
        ext = round(0.12 * 30)
        # frames between a's end(+ext) and c's start must stay unbridged
        for f in range(29 + ext + 1, 30):
            for tid, _ in table[f]:
                assert tid != 1

    def test_ambiguous_bridge_refused(self):
        a = make_tracklet(1, 0, 30, x0=100)
        b1 = make_tracklet(2, 40, 30, x0=105)
        b2 = make_tracklet(3, 40, 30, x0=110)
        table = postprocess([a, b1, b2], fps=30, n_frames=120, p=PP)
        # gap frames (with margin for the end-extension) stay empty
        ext = round(0.12 * 30)
        for f in range(30 + ext, 40 - ext):
            assert len(table[f]) == 0

    def test_smoothing_is_continuous_at_seam(self):
        a = make_tracklet(1, 0, 30, x0=100, vx=2)
        b = make_tracklet(2, 40, 30, x0=190, vx=2)
        table = postprocess([a, b], fps=30, n_frames=120, p=PP)
        xs = [(f[0][1][0] + f[0][1][2]) / 2 for f in table if f]
        jumps = np.abs(np.diff(xs))
        assert jumps.max() < 15        # no teleporting at the bridge seam

    def test_recorder_contiguity(self):
        rec = TrackRecorder()
        obs = [TrackObs(1, np.array([0, 0, 100, 100], np.float32),
                        0.9, True, True, 0)]
        rec.observe(0, obs)
        rec.observe(1, obs)
        rec.observe(3, obs)             # hole at frame 2
        ts = rec.finalize()
        assert len(ts) == 1
        assert len(ts[0].boxes) == 4    # padded to stay contiguous
