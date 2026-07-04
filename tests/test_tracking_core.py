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

from libs.detector import _nms, _unrotate_boxes  # noqa: E402
from libs.evidence import Ev  # noqa: E402
from libs.head_tracker import HeadTracker, TrackObs  # noqa: E402
from libs.sidecar import ManualRegion  # noqa: E402
from libs.tracklets import (PostParams, TrackRecorder, Tracklet,  # noqa: E402
                            apply_review, build_table, clean_tracklets)

SHAPE = (720, 1280)


def det(x, y, w=100, h=120, score=0.9):
    return np.array([x, y, x + w, y + h, score], dtype=np.float32)


# ── detector post-processing ────────────────────────────────────────────────
# (fuse_heads/filter_orphan_faces/suppress_shadow_heads and their pseudo-head
# machinery were removed with the evidence-gated rework; the scenarios below
# move into TestGateFaceCandidates in tests/test_evidence.py, which tests the
# replacement directly.)


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

    def test_rotated_giant_hallucination_suppressed(self):
        """A rotated pass can hallucinate a frame-spanning 'head' that
        contains the real upright-detected heads — it must be dropped, while
        a genuine sideways head (small, strong, not containing anything)
        survives on the lone-detection floor."""
        from libs.detector import HeadDetector
        upright = np.stack([det(880, 30, 230, 260, 0.92),
                            det(470, 190, 240, 230, 0.89)])
        rotated = np.stack([
            np.array([0, 0, 1275, 544, 0.79], np.float32),   # giant fake
            det(100, 500, 200, 150, 0.82),                   # real sideways
            det(300, 500, 200, 150, 0.40),                   # below rot floor
        ])
        out = HeadDetector._filter_rotated(
            rotated, upright, upright, 1280, 720)
        assert len(out) == 1
        assert out[0, 4] == pytest.approx(0.82)

    def test_rotated_giant_killed_without_upright_witness(self):
        """The empty-frame case: no upright head exists, so the containment
        gate is blind — the absolute area cap must still kill a
        frame-spanning rotated hallucination, while a strong plausible-sized
        sideways head survives."""
        from libs.detector import HeadDetector
        upright = np.empty((0, 5), np.float32)
        rotated = np.stack([
            np.array([40, 150, 1240, 700, 0.86], np.float32),  # giant fake
            det(100, 500, 200, 150, 0.82),                     # real sideways
        ])
        out = HeadDetector._filter_rotated(
            rotated, upright, upright, 1280, 720)
        assert len(out) == 1
        assert out[0, 4] == pytest.approx(0.82)

    def test_rotated_moderate_fake_without_witness_killed(self):
        """The bed-blur case: a mid-size hallucination (well under the giant
        area cap) on a frame the upright pass read as completely empty. With
        no upright corroboration it must clear the lone floor — a mediocre
        score dies, only a genuinely confident lone detection survives."""
        from libs.detector import HeadDetector
        empty = np.empty((0, 5), np.float32)
        fake = det(300, 200, 576, 252, 0.60)     # ~16 % of frame, sub-cap
        out = HeadDetector._filter_rotated(
            fake[None], empty, empty, 1280, 720)
        assert len(out) == 0
        strong = det(300, 200, 576, 252, 0.80)   # lone but very confident
        out = HeadDetector._filter_rotated(
            strong[None], empty, empty, 1280, 720)
        assert len(out) == 1

    def test_rotated_weak_kept_with_witness(self):
        """Corroborated rotated boxes keep the normal floor: a weak upright
        head at the same spot (the classic sideways case — upright pass sees
        it at 0.1, rotated pass at 0.6) or an upright body containing the
        centre both count. A speck of upright noise inside a big fake does
        not (IoU bar)."""
        from libs.detector import HeadDetector
        empty = np.empty((0, 5), np.float32)

        rot = det(110, 505, 200, 150, 0.55)
        weak_up = det(100, 500, 200, 150, 0.08)[None]   # parse-floor witness
        out = HeadDetector._filter_rotated(
            rot[None], empty, weak_up, 1280, 720)
        assert len(out) == 1

        rot = det(700, 150, 120, 140, 0.55)
        body = det(600, 100, 300, 500, 0.60)[None]      # centre inside body
        out = HeadDetector._filter_rotated(
            rot[None], empty, body, 1280, 720)
        assert len(out) == 1

        fake = det(300, 200, 576, 252, 0.60)
        speck = det(320, 220, 30, 30, 0.06)[None]       # noise it covers
        out = HeadDetector._filter_rotated(
            fake[None], empty, speck, 1280, 720)
        assert len(out) == 0


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

    def test_face_box_remembered_and_rides_head_motion(self):
        """The face-only blur anchor: a matched face is stored relative to
        the head box, so it translates with the track between face frames
        instead of freezing at its last absolute position."""
        tr = HeadTracker(min_hits=1)
        head = det(100, 100)                       # 100×120 head
        face = det(120, 130, 50, 60, 0.7)          # inside the head
        obs = tr.update(head[None], SHAPE, faces=face[None])
        o = obs[0]
        assert o.face_age == 0
        np.testing.assert_allclose(o.face_box, face[:4], atol=1.5)
        # head moves right, no face evidence this frame
        moved = det(140, 100)
        o2 = tr.update(moved[None], SHAPE, faces=None)[0]
        assert o2.face_age == 1
        # face box translated with the head (~+40 px, KF-smoothed)
        assert o2.face_box[0] > o.face_box[0] + 10

    def test_no_face_evidence_gives_no_face_box(self):
        tr = HeadTracker(min_hits=1)
        o = tr.update(det(100, 100)[None], SHAPE)[0]
        assert o.face_box is None
        # a face elsewhere in the frame must not attach to this track
        far_face = det(900, 500, 50, 60, 0.9)
        o = tr.update(det(100, 100)[None], SHAPE, faces=far_face[None])[0]
        assert o.face_box is None
        assert o.face_age > 0


# ── offline cleanup ─────────────────────────────────────────────────────────

# Default evidence for synthetic tracklets: strong enough to clear the
# ledger (grade "A"/"B") so these tests exercise length/score/bridge/smooth
# behavior in isolation from the evidence dimension (covered separately by
# test_evidence_free_tracklet_pruned below and tests/test_ledger.py).
_GOOD_EV = int(Ev.HEAD_ANCHOR | Ev.PART_EYE_HIT)


def make_tracklet(tid, start, n, x0=100.0, vx=0.0, score=0.9, w=100.0,
                  h=120.0, ev=_GOOD_EV, fvalid=True):
    boxes = np.stack([np.array([x0 + vx * i, 200.0, w, h], np.float32)
                      for i in range(n)])
    return Tracklet(tid, start, boxes,
                    np.full(n, score, np.float32), np.ones(n, bool),
                    ev=np.full(n, ev, np.uint32),
                    fvalid=np.full(n, fvalid, bool))


PP = PostParams(det_conf=0.5, min_hits=3, min_track_s=0.25,
                bridge_gap_s=1.5, smooth_win_s=0.5)


class TestPostprocess:
    def test_short_tracklet_pruned(self):
        t = make_tracklet(1, 10, 3)     # 3 frames < 0.25s @ 30fps
        kept, _ = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        table = build_table(kept, 100)
        assert all(len(f) == 0 for f in table)

    def test_low_confidence_pruned(self):
        t = make_tracklet(1, 10, 30, score=0.3)
        kept, _ = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        table = build_table(kept, 100)
        assert all(len(f) == 0 for f in table)

    def test_evidence_free_tracklet_pruned(self):
        """The core ledger promise: long, confident and reproducing every
        frame — the sock/skin-misread signature — is rejected once it never
        earns real anatomical/part evidence, even though it clears every
        length/score/hit-ratio gate a score-and-length-only prune would
        apply."""
        t = make_tracklet(1, 10, 30, ev=0)
        kept, rejected = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        assert len(kept) == 0
        assert len(rejected) == 1 and rejected[0].grade == "C"

    def test_coasted_tail_trimmed(self):
        t = make_tracklet(1, 10, 30)
        t.hits[-10:] = False            # coasted tail
        kept, _ = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        table = build_table(kept, 100)
        ext = round(0.12 * 30)
        last = max(i for i, f in enumerate(table) if f)
        assert last == 10 + 19 + ext    # last hit + extension, not the tail

    def test_clean_gap_bridged_and_interpolated(self):
        a = make_tracklet(1, 0, 30, x0=100, vx=2)
        b = make_tracklet(2, 50, 30, x0=100 + 2 * 50, vx=2)
        kept, _ = clean_tracklets([a, b], fps=30, n_frames=120, p=PP)
        table = build_table(kept, 120)
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
        kept, _ = clean_tracklets([a, b, c], fps=30, n_frames=120, p=PP)
        table = build_table(kept, 120)
        ext = round(0.12 * 30)
        # frames between a's end(+ext) and c's start must stay unbridged
        for f in range(29 + ext + 1, 30):
            for tid, *_ in table[f]:
                assert tid != 1

    def test_ambiguous_bridge_refused(self):
        a = make_tracklet(1, 0, 30, x0=100)
        b1 = make_tracklet(2, 40, 30, x0=105)
        b2 = make_tracklet(3, 40, 30, x0=110)
        kept, _ = clean_tracklets([a, b1, b2], fps=30, n_frames=120, p=PP)
        table = build_table(kept, 120)
        # gap frames (with margin for the end-extension) stay empty
        ext = round(0.12 * 30)
        for f in range(30 + ext, 40 - ext):
            assert len(table[f]) == 0

    def test_smoothing_is_continuous_at_seam(self):
        a = make_tracklet(1, 0, 30, x0=100, vx=2)
        b = make_tracklet(2, 40, 30, x0=190, vx=2)
        kept, _ = clean_tracklets([a, b], fps=30, n_frames=120, p=PP)
        table = build_table(kept, 120)
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
        assert len(ts[0].fboxes) == 4   # face channel padded in lockstep
        assert len(ts[0].ev) == 4       # evidence channel padded too
        assert len(ts[0].fvalid) == 4

    def test_face_channel_recorded_and_emitted(self):
        """The face-target channel travels recorder → clean_tracklets →
        build_table alongside the head channel and stays distinct from it."""
        rec = TrackRecorder()
        head = np.array([100, 100, 200, 220], np.float32)
        face = np.array([120, 130, 170, 190], np.float32)
        obs = [TrackObs(1, head, 0.9, True, True, 0)]
        for f in range(30):
            rec.observe(f, obs, face_boxes=[face], face_valid=[True],
                       ev_flags=[_GOOD_EV])
        kept, _ = clean_tracklets(rec.finalize(), fps=30, n_frames=60, p=PP)
        table = build_table(kept, 60)
        frames = [f for f in table if f]
        assert frames
        tid, hb, fb, ok = frames[len(frames) // 2][0]
        np.testing.assert_allclose(hb, head, atol=1.0)
        np.testing.assert_allclose(fb, face, atol=1.0)
        assert ok

    def test_face_channel_defaults_to_head(self):
        rec = TrackRecorder()
        head = np.array([100, 100, 200, 220], np.float32)
        obs = [TrackObs(1, head, 0.9, True, True, 0)]
        for f in range(30):
            rec.observe(f, obs, ev_flags=[_GOOD_EV])  # no face_boxes given
        kept, _ = clean_tracklets(rec.finalize(), fps=30, n_frames=60, p=PP)
        table = build_table(kept, 60)
        frames = [f for f in table if f]
        _tid, hb, fb, _ok = frames[len(frames) // 2][0]
        np.testing.assert_allclose(fb, hb, atol=1e-3)

    def test_verify_hook_runs_after_prune_and_is_final(self):
        good = make_tracklet(1, 10, 30)
        noise = make_tracklet(2, 50, 3)         # dies in the prune
        seen: list[int] = []

        def verify(ts):
            seen.extend(t.tid for t in ts)
            return []                            # reject everything

        kept, _ = clean_tracklets([good, noise], fps=30, n_frames=100, p=PP,
                                  verify=verify)
        table = build_table(kept, 100)
        assert seen == [1]                       # prune ran first
        assert all(len(f) == 0 for f in table)   # verify verdict is final


class TestReviewOverrides:
    """apply_review() + build_table()'s manual_regions — the Phase 4 review
    UI's non-GUI logic (ReviewDialog itself lives in src/review_ui.py and is
    smoke-tested separately, headlessly, via QTest)."""

    def test_kept_enabled_by_default(self):
        t = make_tracklet(1, 10, 30)
        kept, rejected = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        out = apply_review(kept, rejected, {})
        assert [ct.t.tid for ct in out] == [1]

    def test_kept_can_be_disabled(self):
        t = make_tracklet(1, 10, 30)
        kept, rejected = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        out = apply_review(kept, rejected, {1: False})
        assert out == []

    def test_rejected_disabled_by_default(self):
        t = make_tracklet(1, 10, 3)      # too short — pruned
        kept, rejected = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        assert len(rejected) == 1
        out = apply_review(kept, rejected, {})
        assert out == []

    def test_rejected_can_be_reenabled(self):
        t = make_tracklet(1, 10, 3)
        kept, rejected = clean_tracklets([t], fps=30, n_frames=100, p=PP)
        out = apply_review(kept, rejected, {1: True})
        assert [ct.t.tid for ct in out] == [1]

    def test_manual_region_renders_across_its_range(self):
        region = ManualRegion(start=5, end=9, box0=(0, 0, 20, 20),
                              box1=(0, 0, 20, 20))
        table = build_table([], 20, manual_regions=[region])
        for f in range(5, 10):
            assert len(table[f]) == 1
            tid, hb, fb, ok = table[f][0]
            assert tid < 0                # never collides with a real tid
            np.testing.assert_allclose(hb, [0, 0, 20, 20])
            np.testing.assert_allclose(fb, [0, 0, 20, 20])
            assert ok is True
        assert table[4] == [] and table[10] == []

    def test_manual_region_interpolates_between_keyframes(self):
        region = ManualRegion(start=0, end=10, box0=(0, 0, 10, 10),
                              box1=(10, 10, 20, 20))
        table = build_table([], 11, manual_regions=[region])
        _tid, hb, _fb, _ok = table[5][0]
        np.testing.assert_allclose(hb, [5, 5, 15, 15], atol=1e-3)

    def test_manual_regions_get_distinct_negative_tids(self):
        r1 = ManualRegion(0, 5, (0, 0, 10, 10), (0, 0, 10, 10))
        r2 = ManualRegion(0, 5, (50, 50, 60, 60), (50, 50, 60, 60))
        table = build_table([], 6, manual_regions=[r1, r2])
        tids = sorted(e[0] for e in table[0])
        assert len(tids) == 2 and len(set(tids)) == 2
        assert all(tid < 0 for tid in tids)

    def test_manual_region_coexists_with_real_tracks(self):
        t = make_tracklet(1, 0, 30)
        kept, _ = clean_tracklets([t], fps=30, n_frames=40, p=PP)
        region = ManualRegion(0, 39, (0, 0, 5, 5), (0, 0, 5, 5))
        table = build_table(kept, 40, manual_regions=[region])
        tids = sorted(e[0] for e in table[15])
        assert 1 in tids
        assert any(tid < 0 for tid in tids)


# (verify_tracklets/TestVerifyTracklets — the interim single-model verifier —
# were replaced by tracklets.verify_tracklets_xmodel with the Phase 3
# cross-model rework; its coverage moved to tests/test_verify_xmodel.py,
# which tests the replacement directly.)


# ── scrfd decode ─────────────────────────────────────────────────────────────

class TestScrfdDecode:
    @staticmethod
    def _zero_outs(hw=(640, 640), with_kps=False):
        from libs.scrfd import _ANCHORS_PER_CELL, _STRIDES
        h, w = hw
        scores, bboxes, kpss = [], [], []
        for s in _STRIDES:
            n = (h // s) * (w // s) * _ANCHORS_PER_CELL
            scores.append(np.zeros((n, 1), np.float32))
            bboxes.append(np.zeros((n, 4), np.float32))
            kpss.append(np.zeros((n, 10), np.float32))
        return (scores, bboxes, kpss) if with_kps else (scores, bboxes)

    def test_synthetic_single_face(self):
        from libs.scrfd import _ANCHORS_PER_CELL, _decode
        scores, bboxes = self._zero_outs()
        stride, row, col = 16, 10, 5
        flat = (row * (640 // stride) + col) * _ANCHORS_PER_CELL
        scores[1][flat] = 0.8
        bboxes[1][flat] = [2.0, 3.0, 4.0, 5.0]   # l, t, r, b in stride units
        out = _decode(scores + bboxes, (640, 640), 0.5)
        assert out.shape == (1, 5)
        cx, cy = col * stride, row * stride
        np.testing.assert_allclose(
            out[0, :4], [cx - 32, cy - 48, cx + 64, cy + 80])
        assert out[0, 4] == pytest.approx(0.8)

    def test_all_below_floor_is_empty(self):
        from libs.scrfd import _decode
        scores, bboxes = self._zero_outs()
        out = _decode(scores + bboxes, (640, 640), 0.5)
        assert out.shape == (0, 5)

    def test_kps_decoded_when_exported(self):
        from libs.scrfd import _ANCHORS_PER_CELL, _decode
        scores, bboxes, kpss = self._zero_outs(with_kps=True)
        stride, row, col = 16, 10, 5
        flat = (row * (640 // stride) + col) * _ANCHORS_PER_CELL
        scores[1][flat] = 0.8
        bboxes[1][flat] = [2.0, 3.0, 4.0, 5.0]
        # left eye offset (−1, −2) stride units from the anchor centre
        kpss[1][flat, 0:2] = [-1.0, -2.0]
        out = _decode(scores + bboxes + kpss, (640, 640), 0.5)
        assert out.shape == (1, 15)
        cx, cy = col * stride, row * stride
        assert out[0, 5] == pytest.approx(cx - 16)
        assert out[0, 6] == pytest.approx(cy - 32)


class TestKpsPlausible:
    """The landmark face-ness gate: real layouts pass at any rotation,
    degenerate/scattered ones (skin, socks, fabric folds) die."""

    # A plausible frontal face: eyes level, nose centred, mouth below.
    FACE = np.array([[30, 30], [70, 30], [50, 55], [35, 75], [65, 75]],
                    np.float32)

    @staticmethod
    def _rot90(kps):
        return np.stack([kps[:, 1], -kps[:, 0]], axis=1)

    def test_frontal_face_passes(self):
        from libs.scrfd import kps_plausible
        assert kps_plausible(self.FACE[None]).all()

    def test_rotation_invariant(self):
        # Lying-down and upside-down faces are the norm in this footage.
        from libs.scrfd import kps_plausible
        k = self.FACE
        for _ in range(3):
            k = self._rot90(k)
            assert kps_plausible(k[None]).all()

    def test_degenerate_cluster_rejected(self):
        # All five points collapsed — the classic misread signature.
        from libs.scrfd import kps_plausible
        k = np.full((1, 5, 2), 50.0, np.float32) \
            + np.random.default_rng(0).normal(0, 0.3, (1, 5, 2))
        assert not kps_plausible(k).any()

    def test_collinear_smear_rejected(self):
        # Eyes/nose/mouth on one line — a fold or edge, not a face.
        from libs.scrfd import kps_plausible
        k = np.array([[[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]],
                     np.float32)
        assert not kps_plausible(k).any()

    def test_nan_fails_open(self):
        # A model without landmark outputs must not veto anything.
        from libs.scrfd import kps_plausible
        k = np.full((1, 5, 2), np.nan, np.float32)
        assert kps_plausible(k).all()


class TestCloseupFilter:
    """The close-up assist only exists for faces big enough to defeat the
    primary; anything smaller is presumed to be SCRFD's known skin/texture
    false positive and must not become a blur candidate."""

    HW = (720, 1280)   # min side 720 → close-up threshold is 180 px

    def test_closeup_scale_face_kept(self):
        from libs.scrfd import closeup_filter
        faces = det(200, 100, 300, 360, 0.80)[None]
        assert len(closeup_filter(faces, self.HW)) == 1

    def test_small_face_dropped(self):
        # The bed-blur case: a confident but small SCRFD-only "face" on a
        # frame with no real face anywhere near close-up scale.
        from libs.scrfd import closeup_filter
        faces = det(500, 200, 120, 140, 0.90)[None]
        assert len(closeup_filter(faces, self.HW)) == 0

    def test_low_score_closeup_dropped(self):
        from libs.scrfd import closeup_filter
        faces = det(200, 100, 300, 360, 0.40)[None]   # below module floor
        assert len(closeup_filter(faces, self.HW)) == 0

    def test_min_score_raises_but_never_lowers_the_floor(self):
        from libs.scrfd import closeup_filter
        faces = det(200, 100, 300, 360, 0.50)[None]
        assert len(closeup_filter(faces, self.HW, min_score=0.60)) == 0
        # a permissive preset (det_conf 0.35) still can't dip below _FLOOR
        faces = det(200, 100, 300, 360, 0.44)[None]
        assert len(closeup_filter(faces, self.HW, min_score=0.35)) == 0
        faces = det(200, 100, 300, 360, 0.46)[None]
        assert len(closeup_filter(faces, self.HW, min_score=0.35)) == 1
