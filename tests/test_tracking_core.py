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

from libs.detector import (Detections, filter_orphan_faces,  # noqa: E402
                           fuse_heads, suppress_shadow_heads, _nms,
                           _unrotate_boxes)
from libs.head_tracker import HeadTracker, TrackObs  # noqa: E402
from libs.tracklets import (PostParams, TrackRecorder, Tracklet,  # noqa: E402
                            postprocess, verify_tracklets)

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


def no_witnesses():
    return np.empty((0, 5), np.float32)


def never_called():
    raise AssertionError("lazy witness fn consulted when it must not be")


class TestFilterOrphanFaces:
    """A primary face with no covering head grows a pseudo-head (= blur);
    on bare skin the face class misreads, so orphans need SCRFD agreement."""

    def test_covered_face_passes_without_witness_pass(self):
        heads = det(100, 100)[None]
        faces = det(120, 130, 50, 60, 0.6)[None]        # inside the head
        out = filter_orphan_faces(faces, heads, never_called)
        assert len(out) == 1

    def test_orphan_with_scrfd_agreement_kept(self):
        faces = det(500, 200, 80, 90, 0.6)[None]
        witness = det(505, 205, 70, 80, 0.35)[None]     # SCRFD sees it too
        out = filter_orphan_faces(faces, np.empty((0, 5), np.float32),
                                  lambda: witness)
        assert len(out) == 1

    def test_orphan_without_agreement_dies(self):
        # The chest-blur case, face-class flavour: skin misread as a face
        # with nothing face-like there for SCRFD at any angle.
        faces = det(500, 200, 80, 90, 0.7)[None]
        out = filter_orphan_faces(faces, np.empty((0, 5), np.float32),
                                  no_witnesses)
        assert len(out) == 0


class TestSuppressShadowHeads:
    """One head per body where a face pins it down — and never anywhere
    else, because two entangled people can merge into one body box."""

    BODY = det(100, 300, 700, 250, 0.8)[None]           # lying, wide box
    REAL = det(110, 320, 100, 120, 0.85)                # head at left end
    FACE = det(120, 330, 60, 70, 0.6)[None]             # inside REAL
    FAKE = det(400, 380, 120, 130, 0.55)                # chest, mid-body

    def test_chest_fake_dropped_next_to_face_backed_head(self):
        heads = np.stack([self.REAL, self.FAKE])
        out = suppress_shadow_heads(heads, self.BODY, lambda: self.FACE)
        assert len(out) == 1
        assert out[0, 4] == pytest.approx(0.85)

    def test_faceless_body_never_arbitrated(self):
        # Entangled couple in one body box: neither head shows a face —
        # the partner's back-of-head must survive.
        heads = np.stack([self.REAL, self.FAKE])
        out = suppress_shadow_heads(heads, self.BODY, no_witnesses)
        assert len(out) == 2

    def test_strong_rival_never_dropped(self):
        strong = self.FAKE.copy()
        strong[4] = 0.75                                # above the cap
        heads = np.stack([self.REAL, strong])
        out = suppress_shadow_heads(heads, self.BODY, lambda: self.FACE)
        assert len(out) == 2

    def test_rival_outside_every_body_untouched(self):
        outside = det(900, 100, 120, 130, 0.55)         # not in any body
        heads = np.stack([self.REAL, outside])
        out = suppress_shadow_heads(heads, self.BODY, lambda: self.FACE)
        assert len(out) == 2

    def test_single_claim_skips_the_face_pass(self):
        out = suppress_shadow_heads(self.REAL[None], self.BODY, never_called)
        assert len(out) == 1


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

    def test_verify_hook_runs_after_prune_and_is_final(self):
        good = make_tracklet(1, 10, 30)
        noise = make_tracklet(2, 50, 3)         # dies in the prune
        seen: list[int] = []

        def verify(ts):
            seen.extend(t.tid for t in ts)
            return []                            # reject everything

        table = postprocess([good, noise], fps=30, n_frames=100, p=PP,
                            verify=verify)
        assert seen == [1]                       # prune ran first
        assert all(len(f) == 0 for f in table)   # verify verdict is final


# ── tracklet verification (cropped re-inference) ────────────────────────────

FRAME = np.zeros((720, 1280, 3), np.uint8)


NO_BOXES = np.empty((0, 5), np.float32)


def centred_det(score, face_score=0.0):
    """detect_fn stub: one candidate in the middle of whatever crop it gets —
    crops are centred on the tracklet box, so this lands inside it. A
    non-zero ``face_score`` adds face evidence at the same spot."""
    def detect_fn(crop):
        h, w = crop.shape[:2]
        cand = np.array([[w / 2 - 40, h / 2 - 50, w / 2 + 40, h / 2 + 50,
                          score]], np.float32)
        faces = (np.array([[w / 2 - 20, h / 2 - 25, w / 2 + 20, h / 2 + 25,
                            face_score]], np.float32)
                 if face_score else NO_BOXES)
        return cand, faces
    return detect_fn


class TestVerifyTracklets:
    def test_reproducing_tracklet_kept(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets([t], lambda i: FRAME,
                                         centred_det(0.9))
        assert len(kept) == 1 and not dropped

    def test_silent_tracklet_dropped(self):
        """The screenshot bug's kill switch: a long, confident tracklet whose
        'head' does not re-detect on magnified crops is a hallucination."""
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets(
            [t], lambda i: FRAME, lambda crop: (NO_BOXES, NO_BOXES))
        assert not kept and len(dropped) == 1

    def test_low_score_redetection_dropped(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets([t], lambda i: FRAME,
                                         centred_det(0.3))
        assert not kept and len(dropped) == 1

    def test_off_target_redetection_dropped(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets(
            [t], lambda i: FRAME,
            lambda crop: (np.array([[0, 0, 10, 10, 0.9]], np.float32),
                          NO_BOXES))
        assert not kept and len(dropped) == 1

    def test_midscore_faceless_redetection_dropped(self):
        """The chest-blur kill switch: skin misread as a head *reproduces*
        under magnification, but only at modest confidence and never with
        face evidence — below the lone floor it dies."""
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets([t], lambda i: FRAME,
                                         centred_det(0.55))
        assert not kept and len(dropped) == 1

    def test_midscore_face_backed_kept(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets([t], lambda i: FRAME,
                                         centred_det(0.55, face_score=0.4))
        assert len(kept) == 1 and not dropped

    def test_unreadable_frames_fail_open(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets([t], lambda i: None,
                                         centred_det(0.9))
        assert len(kept) == 1 and not dropped

    def test_only_hit_frames_sampled(self):
        t = make_tracklet(1, 10, 30)
        t.hits[10:] = False              # only frames 10..19 are evidence
        asked: list[int] = []

        def frame_at(idx):
            asked.append(idx)
            return FRAME

        verify_tracklets([t], frame_at, centred_det(0.9))
        assert asked and all(10 <= i < 20 for i in asked)


# ── scrfd decode ─────────────────────────────────────────────────────────────

class TestScrfdDecode:
    @staticmethod
    def _zero_outs(hw=(640, 640)):
        from libs.scrfd import _ANCHORS_PER_CELL, _STRIDES
        h, w = hw
        scores, bboxes = [], []
        for s in _STRIDES:
            n = (h // s) * (w // s) * _ANCHORS_PER_CELL
            scores.append(np.zeros((n, 1), np.float32))
            bboxes.append(np.zeros((n, 4), np.float32))
        return scores, bboxes

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
