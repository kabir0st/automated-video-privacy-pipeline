"""Union fusion: any source may propose; agreement boosts; only the
frame-spanning rotated hallucination is dropped, and not when corroborated."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.fuse import FuseConfig, fuse          # noqa: E402
from pipeline.types import Src                       # noqa: E402

HW = (720, 1280)
E = np.empty((0, 5), np.float32)


def box(x1, y1, x2, y2, s):
    return np.array([[x1, y1, x2, y2, s]], np.float32)


class TestUnion:
    def test_single_head_source_is_enough(self):
        c = fuse({"headdet": box(100, 100, 200, 220, 0.4)}, HW)
        assert len(c.heads) == 1
        assert c.flags[0] & Src.HEADDET
        assert not (c.flags[0] & Src.MULTI)
        assert abs(c.heads[0, 4] - 0.4) < 1e-6

    def test_lone_face_becomes_head_sized_candidate(self):
        c = fuse({"scrfd": box(140, 140, 180, 190, 0.8)}, HW)
        assert len(c.heads) == 1
        w = c.heads[0, 2] - c.heads[0, 0]
        assert w > 40 * 1.2                      # grown to head scale
        assert abs(c.heads[0, 4] - 0.8 * FuseConfig().face_weight) < 1e-6
        assert c.flags[0] & Src.SCRFD
        assert len(c.faces) == 1                 # raw face pool kept

    def test_agreement_merges_and_boosts(self):
        c = fuse({"headdet": box(100, 100, 200, 220, 0.5),
                  "wb_head": box(104, 98, 204, 224, 0.6),
                  "scrfd": box(130, 130, 170, 180, 0.7)}, HW)
        assert len(c.heads) == 1
        assert c.flags[0] & Src.MULTI
        assert c.flags[0] & Src.FACE_IN_HEAD
        assert c.heads[0, 4] > 0.6

    def test_two_heads_stay_separate(self):
        c = fuse({"headdet": np.concatenate([box(100, 100, 200, 220, 0.5),
                                             box(600, 100, 700, 220, 0.5)])}, HW)
        assert len(c.heads) == 2

    def test_empty(self):
        c = fuse({}, HW)
        assert len(c.heads) == 0 and len(c.faces) == 0
        c = fuse({"headdet": E, "scrfd": E}, HW)
        assert len(c.heads) == 0


class TestSanity:
    def test_rotated_only_giant_dropped(self):
        c = fuse({"wb_head_rot": box(0, 0, 900, 700, 0.6)}, HW)
        assert len(c.heads) == 0

    def test_rotated_giant_kept_when_corroborated(self):
        c = fuse({"wb_head_rot": box(0, 0, 900, 700, 0.6),
                  "headdet": box(20, 10, 880, 690, 0.3)}, HW)
        assert len(c.heads) == 1

    def test_rotated_only_normal_size_kept_and_flagged(self):
        c = fuse({"headdet_rot": box(100, 100, 200, 220, 0.5)}, HW)
        assert len(c.heads) == 1
        assert c.flags[0] & Src.ROTATED

    def test_tiny_boxes_dropped(self):
        c = fuse({"headdet": box(10, 10, 14, 14, 0.9)}, HW)
        assert len(c.heads) == 0


class TestTrust:
    def test_oversized_single_source_is_downweighted(self):
        big = box(50, 50, 750, 600, 0.6)                 # 42 % of the frame
        c = fuse({"wb_head": big}, HW)
        assert len(c.heads) == 1 and c.heads[0, 4] <= 0.31

    def test_large_closeup_head_keeps_score(self):
        head = box(300, 100, 700, 600, 0.6)              # 22 % — a real close-up
        c = fuse({"headdet": head}, HW)
        assert abs(c.heads[0, 4] - 0.6) < 1e-6

    def test_single_source_wholebody_is_trusted_less(self):
        c = fuse({"wb_head": box(100, 100, 200, 220, 0.6)}, HW)
        assert abs(c.heads[0, 4] - 0.6 * 0.7) < 1e-6
        c = fuse({"wb_head": box(100, 100, 200, 220, 0.6),
                  "headdet": box(102, 98, 204, 222, 0.5)}, HW)
        assert c.heads[0, 4] >= 0.6                      # corroborated: full trust

    def test_face_inside_a_torso_box_does_not_validate_it(self):
        torso = box(200, 50, 900, 700, 0.6)              # 47 % of the frame
        face = box(500, 150, 560, 230, 0.9)              # small face inside it
        c = fuse({"wb_head": torso, "scrfd": face}, HW)
        # two candidates: the grown face (a head) and the down-weighted torso
        assert len(c.heads) == 2
        big = c.heads[np.argmax((c.heads[:, 2] - c.heads[:, 0]))]
        assert big[4] < 0.35
        assert not (c.flags[np.argmax(c.heads[:, 2] - c.heads[:, 0])] & Src.FACE_IN_HEAD)

    def test_torso_box_is_not_merged_into_head_cluster(self):
        head = box(400, 200, 500, 320, 0.7)
        torso = box(300, 150, 700, 650, 0.6)             # contains the head, 8× area
        c = fuse({"headdet": head, "wb_head": torso}, HW)
        assert len(c.heads) == 2
        w = c.heads[:, 2] - c.heads[:, 0]
        assert abs(w.min() - 100) < 1.0                  # head box not inflated

    def test_oversized_with_head_sized_face_inside_keeps_score(self):
        big = box(50, 50, 750, 600, 0.6)                 # 42 % of the frame
        face = box(150, 150, 650, 550, 0.8)              # a face nearly as big → close-up
        c = fuse({"wb_head": big, "scrfd": face}, HW)
        assert len(c.heads) == 1 and c.heads[0, 4] >= 0.6

    def test_odd_aspect_downweighted(self):
        c = fuse({"wb_head": box(100, 100, 400, 180, 0.6)}, HW)   # 3.75:1
        assert c.heads[0, 4] <= 0.31

    def test_rotated_single_downweighted_but_kept(self):
        c = fuse({"headdet_rot": box(100, 100, 200, 220, 0.6)}, HW)
        assert len(c.heads) == 1 and 0.35 < c.heads[0, 4] < 0.6


class TestFacesCorroborate:
    def test_face_inside_head_does_not_change_geometry(self):
        head = box(100, 100, 220, 240, 0.6)
        face = box(125, 130, 195, 220, 0.9)
        c = fuse({"headdet": head, "scrfd": face}, HW)
        assert len(c.heads) == 1
        assert np.allclose(c.heads[0, :4], head[0, :4], atol=1e-3)
        assert c.flags[0] & Src.FACE_IN_HEAD and c.flags[0] & Src.MULTI

    def test_closeup_faces_do_not_spawn_giant_duplicate(self):
        # every source agrees on a ~13 % head; faces cover most of it
        srcs = {"headdet": box(2, 307, 426, 600, 0.63),
                "wb_head_rot": box(0, 327, 419, 592, 0.61),
                "scrfd": box(13, 341, 387, 566, 0.62),
                "scrfd_rot": box(3, 337, 424, 582, 0.75),
                "wb_face_rot": box(4, 338, 406, 577, 0.54)}
        c = fuse(srcs, HW)
        assert len(c.heads) == 1
        area = (c.heads[0, 2] - c.heads[0, 0]) * (c.heads[0, 3] - c.heads[0, 1]) / (1280 * 720)
        assert area < 0.16

    def test_overboxed_head_is_pulled_toward_face(self):
        head = box(60, 100, 700, 620, 0.6)                # head + torso, 36 %
        face = box(250, 200, 450, 440, 0.67)              # 5.2 % — a real close-up face
        c = fuse({"headdet": head, "wb_head": head, "scrfd": face}, HW)
        assert len(c.heads) == 1
        w = c.heads[0, 2] - c.heads[0, 0]
        assert 270 < w < 600                              # pulled halfway toward the face

    def test_realistic_closeup_head_is_kept_as_is(self):
        head = box(78, 178, 575, 526, 0.6)                # ~2× the face area: a real head
        face = box(92, 279, 483, 505, 0.67)
        c = fuse({"headdet": head, "wb_head": head, "scrfd": face}, HW)
        assert np.allclose(c.heads[0, :4], head[0, :4], atol=1e-3)


class TestFamiliesAndBodyParts:
    def test_wholebody_head_and_face_are_one_family(self):
        c = fuse({"wb_head_rot": box(439, 579, 836, 719, 0.72),     # 2.85:1, rotated
                  "wb_face_rot": box(438, 591, 753, 719, 0.62)}, HW)
        assert len(c.heads) == 1
        assert not (c.flags[0] & Src.MULTI)
        assert not (c.flags[0] & Src.FACE_IN_HEAD)
        assert c.heads[0, 4] < 0.35                       # cannot spawn

    def test_cross_family_face_still_corroborates(self):
        c = fuse({"wb_head": box(100, 100, 220, 240, 0.6),
                  "scrfd": box(125, 130, 195, 220, 0.9)}, HW)
        assert c.flags[0] & Src.MULTI and c.flags[0] & Src.FACE_IN_HEAD

    def test_bodypart_veto_beats_two_agreeing_head_models(self):
        srcs = {"headdet": box(654, 0, 1279, 570, 0.73),
                "wb_head": box(538, 0, 1279, 587, 0.95),
                "bodypart": box(600, 40, 1279, 600, 0.8)}
        c = fuse(srcs, HW)                                  # default: flag only
        assert len(c.heads) == 1 and c.heads[0, 4] >= 0.9
        assert c.flags[0] & Src.BODYPART
        c = fuse(srcs, HW, FuseConfig(bodypart_weight=0.3))  # Strict: veto
        assert c.heads[0, 4] < 0.35

    def test_cross_family_face_exempts_from_veto(self):
        srcs = {"headdet": box(100, 100, 300, 340, 0.7),
                "scrfd": box(140, 150, 260, 300, 0.8),
                "bodypart": box(90, 90, 310, 350, 0.8)}          # e.g. an armpit box on top
        c = fuse(srcs, HW)
        assert c.heads[0, 4] >= 0.7 and not (c.flags[0] & Src.BODYPART)

    def test_bodypart_inside_candidate_vetoes(self):
        # a genitalia box lying inside a "head" the size of a torso
        srcs = {"headdet": box(369, 211, 958, 719, 0.85),
                "bodypart": box(733, 463, 950, 719, 0.5)}
        c = fuse(srcs, HW, FuseConfig(bodypart_weight=0.3))
        assert c.heads[0, 4] < 0.35 and c.flags[0] & Src.BODYPART

    def test_same_network_head_and_face_get_small_bonus_not_trust_penalty(self):
        c = fuse({"wb_head": box(100, 100, 220, 240, 0.5),
                  "wb_face": box(125, 130, 195, 220, 0.6)}, HW)
        base = max(0.5, 0.6 * FuseConfig().face_weight)           # best member
        assert abs(c.heads[0, 4] - (base + FuseConfig().intra_bonus)) < 1e-6
        assert not (c.flags[0] & Src.MULTI)

    def test_small_bodypart_inside_head_does_not_veto(self):
        c = fuse({"headdet": box(100, 100, 300, 340, 0.7),
                  "bodypart": box(180, 250, 220, 290, 0.9)}, HW)      # tiny, e.g. a nipple-sized box
        assert c.heads[0, 4] >= 0.7 and not (c.flags[0] & Src.BODYPART)
