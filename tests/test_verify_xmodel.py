"""Unit tests for libs/tracklets.verify_tracklets_xmodel — the Phase 3
cross-model tracklet verifier that replaced the interim, single-model
verify_tracklets. Synthetic only, no models.

Covers the property the rework exists to enforce: only an INDEPENDENT
witness (never the primary detector) may confirm a tracklet, and when no
witness can run at all, the decision falls back to the tracklet's own
evidence grade (thr.verify_fail_open_grades) rather than blanket-trusting
everything, which is what the old single-model verifier did on failure.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.evidence import Ev, PROFILES  # noqa: E402
from libs.tracklets import Tracklet, verify_tracklets_xmodel  # noqa: E402

FRAME = np.zeros((720, 1280, 3), np.uint8)
NO_FACES = np.empty((0, 5), np.float32)

_GOOD_EV = int(Ev.HEAD_ANCHOR | Ev.PART_EYE_HIT)   # clears grade "A"
_WEAK_EV = int(Ev.CONSENSUS)                        # clears grade "B" only


def make_tracklet(tid, start, n, x0=100.0, w=100.0, h=120.0, ev=_GOOD_EV):
    boxes = np.stack([np.array([x0, 200.0, w, h], np.float32)
                      for _ in range(n)])
    return Tracklet(tid, start, boxes, np.full(n, 0.9, np.float32),
                    np.ones(n, bool), ev=np.full(n, ev, np.uint32),
                    fvalid=np.ones(n, bool))


def centred_witness(score):
    """witness_fn stub: one face in the middle of whatever crop it gets —
    crops are centred on the tracklet box, so this lands inside it."""
    def witness_fn(crop):
        h, w = crop.shape[:2]
        return np.array([[w / 2 - 20, h / 2 - 25, w / 2 + 20, h / 2 + 25,
                          score]], np.float32)
    return witness_fn


class TestVerifyTrackletsXmodel:
    def test_reproducing_tracklet_kept(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME, centred_witness(0.9), PROFILES["balanced"])
        assert len(kept) == 1 and not dropped

    def test_silent_tracklet_dropped(self):
        """The screenshot bug's kill switch: a long, confident tracklet whose
        face does not re-detect on magnified crops via an independent model
        is a hallucination — even at a strong evidence grade."""
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME, lambda crop: NO_FACES, PROFILES["balanced"])
        assert not kept and len(dropped) == 1

    def test_low_score_redetection_dropped(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME, centred_witness(0.2), PROFILES["balanced"])
        assert not kept and len(dropped) == 1

    def test_off_target_redetection_dropped(self):
        t = make_tracklet(1, 10, 30)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME,
            lambda crop: np.array([[0, 0, 10, 10, 0.9]], np.float32),
            PROFILES["balanced"])
        assert not kept and len(dropped) == 1

    def test_only_hit_frames_sampled(self):
        t = make_tracklet(1, 10, 30)
        t.hits[10:] = False              # only frames 10..19 are evidence
        asked: list[int] = []

        def frame_at(idx):
            asked.append(idx)
            return FRAME

        verify_tracklets_xmodel([t], frame_at, centred_witness(0.9),
                                PROFILES["balanced"])
        assert asked and all(10 <= i < 20 for i in asked)

    # ── fail-open-by-grade (verifier genuinely can't run) ───────────────────

    def test_no_witness_strong_grade_kept_under_balanced(self):
        """witness_fn=None means 'every independent model is unavailable' —
        under 'balanced' (fail_open_grades=('A',)), a strongly-evidenced
        tracklet still survives unverified."""
        t = make_tracklet(1, 10, 30, ev=_GOOD_EV)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME, None, PROFILES["balanced"])
        assert len(kept) == 1 and not dropped

    def test_no_witness_weak_grade_dropped_under_balanced(self):
        """A merely 'B'-graded tracklet is NOT trusted on faith when no
        independent model could check it — tighter than the old blanket
        fail-open verify_tracklets shipped with in Phase 1."""
        t = make_tracklet(1, 10, 30, ev=_WEAK_EV)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME, None, PROFILES["balanced"])
        assert not kept and len(dropped) == 1

    def test_no_witness_strict_profile_drops_everything(self):
        """'strict' sets verify_fail_open_grades=() — nothing survives
        unverified, regardless of how strong its evidence grade is."""
        t = make_tracklet(1, 10, 30, ev=_GOOD_EV)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME, None, PROFILES["strict"])
        assert not kept and len(dropped) == 1

    def test_unreadable_frames_fall_back_to_grade(self):
        """Every sampled frame comes back None (source unreadable): nothing
        was actually tested, so this is the same fail-open-by-grade path as
        witness_fn=None, not a blanket keep."""
        t = make_tracklet(1, 10, 30, ev=_GOOD_EV)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: None, centred_witness(0.9), PROFILES["balanced"])
        assert len(kept) == 1 and not dropped

        t2 = make_tracklet(2, 10, 30, ev=_WEAK_EV)
        kept2, dropped2 = verify_tracklets_xmodel(
            [t2], lambda i: None, centred_witness(0.9), PROFILES["balanced"])
        assert not kept2 and len(dropped2) == 1

    def test_max_profile_fail_opens_on_b_grade_too(self):
        t = make_tracklet(1, 10, 30, ev=_WEAK_EV)
        kept, dropped = verify_tracklets_xmodel(
            [t], lambda i: FRAME, None, PROFILES["max"])
        assert len(kept) == 1 and not dropped
