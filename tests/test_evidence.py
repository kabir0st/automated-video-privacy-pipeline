"""Unit tests for the face-candidate acceptance gate (libs/evidence.py) —
the pipeline's precision mechanism. Synthetic only, no models.

Covers the truth table the gate exists to enforce: a face-shaped claim with
no anatomical/part backing (the sock/knee/shoulder failure) is rejected even
at high confidence; genuine anatomical evidence (head anchor, parts) is
accepted; a real hand-over-face scene is not mistaken for the sock case.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.detector import (Detections, NEG_FOOT, NEG_HAND,  # noqa: E402
                           PART_EYE, PART_NOSE)
from libs.evidence import (Ev, PROFILES, gate_face_candidates)  # noqa: E402


def box(x, y, w, h, score=0.9):
    return np.array([x, y, x + w, y + h, score], dtype=np.float32)


def part(x, y, w, h, cls, score=0.4):
    return np.array([x, y, x + w, y + h, score, cls], dtype=np.float32)


FRAME_HW = (1080, 1920)
NO5 = np.empty((0, 5), np.float32)


class FakePose:
    """Duck-typed stand-in for libs.pose.PersonPose (Phase 2)."""

    def __init__(self, anchor, torso=None):
        self.anchor = anchor
        self.torso = torso


class TestGateFaceCandidates:
    def test_sock_no_anchor_no_parts_rejected(self):
        """A face-shaped claim on a sock: no head box nearby, no body, no
        parts, and it's substantially covered by a foot detection — the
        exact failure the rework exists to fix."""
        sock_face = box(500, 600, 80, 90, 0.7)
        dets = Detections(
            faces=sock_face[None],
            negatives=part(490, 590, 100, 110, NEG_FOOT, 0.8)[None],
        )
        scrfd = box(505, 605, 70, 80, 0.6)[None]   # agrees -> CONSENSUS
        accepted, dbg = gate_face_candidates(
            dets, [], scrfd, FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 0
        assert len(dbg.rejected) == 1

    def test_real_face_head_anchor_and_part_accepted(self):
        face = box(300, 200, 100, 120, 0.8)
        head = box(280, 170, 150, 180, 0.75)
        eye = part(320, 220, 15, 10, PART_EYE)
        nose = part(340, 250, 12, 12, PART_NOSE)
        dets = Detections(faces=face[None], heads=head[None],
                          parts=np.stack([eye, nose]))
        accepted, dbg = gate_face_candidates(
            dets, [], NO5, FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 1
        assert accepted[0].flags & Ev.HEAD_ANCHOR
        assert accepted[0].flags & Ev.PART_EYE_HIT

    def test_bare_head_with_no_face_is_never_a_candidate(self):
        """A head-class detection with no accompanying face claim can't
        become a blur candidate at all — candidates only ever come from
        face claims, never from bare heads."""
        head = box(280, 170, 150, 180, 0.95)
        dets = Detections(heads=head[None])
        accepted, dbg = gate_face_candidates(
            dets, [], NO5, FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 0
        assert len(dbg.rejected) == 0

    def test_hand_over_face_overridden_by_pose_axis(self):
        anchor = box(280, 170, 150, 180, 0.9)
        shoulder_mid = np.array([350.0, 350.0])
        hip_mid = np.array([350.0, 550.0])
        pose = FakePose(anchor, (shoulder_mid, hip_mid, 100.0))
        face = box(300, 200, 100, 120, 0.8)      # centre on the head side
        eye = part(320, 220, 15, 10, PART_EYE)
        dets = Detections(
            faces=face[None], parts=eye[None],
            negatives=part(290, 190, 120, 140, NEG_HAND, 0.9)[None])
        accepted, _ = gate_face_candidates(
            dets, [pose], NO5, FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 1
        assert accepted[0].flags & Ev.VETO_OVERRIDDEN
        assert accepted[0].flags & Ev.AXIS_OK

    def test_hand_over_face_single_part_no_pose_rejected(self):
        """Without axis confirmation, a single part hit is not enough to
        override a hand veto — a hand-shaped misread over an otherwise
        weakly-evidenced claim stays rejected."""
        face = box(300, 200, 100, 120, 0.8)
        eye = part(320, 220, 15, 10, PART_EYE)
        dets = Detections(
            faces=face[None], parts=eye[None],
            negatives=part(290, 190, 120, 140, NEG_HAND, 0.9)[None])
        accepted, _ = gate_face_candidates(
            dets, [], NO5, FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 0

    def test_hand_over_face_two_parts_overrides_without_pose(self):
        """Two independent part hits are strong enough evidence to override
        a hand veto even with no pose available."""
        face = box(300, 200, 100, 120, 0.8)
        head = box(280, 170, 150, 180, 0.9)
        eye = part(320, 220, 15, 10, PART_EYE)
        nose = part(340, 250, 12, 12, PART_NOSE)
        dets = Detections(
            faces=face[None], heads=head[None], parts=np.stack([eye, nose]),
            negatives=part(290, 190, 120, 140, NEG_HAND, 0.9)[None])
        accepted, _ = gate_face_candidates(
            dets, [], NO5, FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 1
        assert accepted[0].flags & Ev.VETO_OVERRIDDEN

    def test_closeup_yolo_scrfd_consensus_with_parts_accepted(self):
        """Extreme close-up: face fills the frame, no head/body in view —
        both the primary and SCRFD must agree, and part evidence backs it."""
        big_face = box(400, 300, 500, 600, 0.7)
        scrfd_face = box(410, 310, 480, 580, 0.6)
        eye = part(500, 350, 15, 10, PART_EYE)
        dets = Detections(faces=big_face[None], parts=eye[None])
        accepted, _ = gate_face_candidates(
            dets, [], scrfd_face[None], FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 1
        assert accepted[0].flags & Ev.CLOSEUP

    def test_closeup_scrfd_only_rejected(self):
        """The historical false positive this assist caused: SCRFD alone
        claiming a close-up-scale face with no primary corroboration must
        not be trusted, however confident."""
        scrfd_face = box(410, 310, 480, 580, 0.9)
        dets = Detections()
        accepted, _ = gate_face_candidates(
            dets, [], scrfd_face[None], FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 0

    def test_small_consensus_face_without_anchor_rejected(self):
        """Both sources agree, but the face is small (sub-close-up scale)
        and unanchored — a coincidental agreement on a small region is not
        enough without either anatomical context or close-up scale."""
        small = box(500, 400, 60, 70, 0.6)
        small_scrfd = box(505, 405, 55, 65, 0.55)
        dets = Detections(faces=small[None])
        accepted, _ = gate_face_candidates(
            dets, [], small_scrfd[None], FRAME_HW, PROFILES["balanced"], 0.5)
        assert len(accepted) == 0

    @pytest.mark.parametrize("profile_name,expect_accept", [
        ("balanced", False),   # 1 part hit only; balanced requires >=1 — check below
        ("strict", False),
    ])
    def test_profile_matrix_part_requirement(self, profile_name, expect_accept):
        """'max' relaxes the part requirement for pose-anchored candidates
        (Phase 2); 'balanced'/'strict' both still require >=1 part hit even
        with a head anchor present."""
        face = box(300, 200, 100, 120, 0.8)
        head = box(280, 170, 150, 180, 0.9)
        dets = Detections(faces=face[None], heads=head[None])  # no parts at all
        accepted, _ = gate_face_candidates(
            dets, [], NO5, FRAME_HW, PROFILES[profile_name], 0.5)
        assert (len(accepted) == 1) == expect_accept

    def test_max_profile_accepts_head_anchor_without_parts(self):
        face = box(300, 200, 100, 120, 0.8)
        head = box(280, 170, 150, 180, 0.9)
        dets = Detections(faces=face[None], heads=head[None])
        accepted, _ = gate_face_candidates(
            dets, [], NO5, FRAME_HW, PROFILES["max"], 0.5)
        assert len(accepted) == 1
