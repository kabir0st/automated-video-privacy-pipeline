"""Unit tests for the offline evidence ledger (libs/evidence.summarize and
libs/evidence.grade, consumed by libs/tracklets.clean_tracklets' composite
prune). Synthetic only, no models.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.evidence import Ev, PROFILES, grade, summarize  # noqa: E402


def hits(n):
    return np.ones(n, dtype=bool)


class TestSummarizeGrade:
    def test_no_hits_grade_c(self):
        s = summarize(np.zeros(0, np.uint32), np.zeros(0, bool))
        assert s.n_hits == 0
        assert grade(s, PROFILES["balanced"]) == "C"

    def test_strong_evidence_every_frame_grade_a(self):
        ev = np.full(20, int(Ev.HEAD_ANCHOR | Ev.PART_EYE_HIT), np.uint32)
        s = summarize(ev, hits(20))
        assert s.part_ratio == 1.0 and s.anchor_ratio == 1.0
        assert grade(s, PROFILES["balanced"]) == "A"

    def test_zero_evidence_grade_c_regardless_of_anchor(self):
        """The sock signature: anchored (a head box happens to cover it)
        every frame, but never any part/landmark corroboration — grade must
        stay C, not ride on the anchor ratio alone."""
        ev = np.full(20, int(Ev.HEAD_ANCHOR), np.uint32)
        s = summarize(ev, hits(20))
        assert s.anchor_ratio == 1.0 and s.part_ratio == 0.0
        assert grade(s, PROFILES["balanced"]) == "C"

    def test_partial_evidence_clears_balanced_min_ratio_grade_b(self):
        # 25% of frames carry a part hit; balanced's min_part_ratio is 0.20.
        n = 20
        ev = np.zeros(n, np.uint32)
        ev[:5] = int(Ev.HEAD_ANCHOR | Ev.PART_EYE_HIT)
        s = summarize(ev, hits(n))
        assert s.part_ratio == pytest.approx(0.25)
        assert grade(s, PROFILES["balanced"]) == "B"

    def test_partial_evidence_below_strict_min_ratio_grade_c(self):
        # Same 25% part-hit ratio, but strict requires >= 0.35.
        n = 20
        ev = np.zeros(n, np.uint32)
        ev[:5] = int(Ev.HEAD_ANCHOR | Ev.PART_EYE_HIT)
        s = summarize(ev, hits(n))
        assert grade(s, PROFILES["strict"]) == "C"

    def test_consensus_ratio_alone_can_clear_the_bar(self):
        n = 20
        ev = np.zeros(n, np.uint32)
        ev[:3] = int(Ev.CONSENSUS)   # 15% >= balanced's min_consensus_ratio .10
        s = summarize(ev, hits(n))
        assert grade(s, PROFILES["balanced"]) == "B"

    def test_coasted_frames_excluded_from_ratios(self):
        """hits=False frames (coast/interpolated) must not dilute the
        ratios — only real evidence frames count."""
        n = 10
        ev = np.full(n, int(Ev.HEAD_ANCHOR | Ev.PART_EYE_HIT), np.uint32)
        h = hits(n).copy()
        h[5:] = False   # half the tracklet is coasted, carries ev=0 anyway
        ev[5:] = 0
        s = summarize(ev, h)
        assert s.n_hits == 5
        assert s.part_ratio == 1.0   # ratio over the 5 real hits, not 10
