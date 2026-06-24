"""Standalone unit tests for the segmentation-first detection core.

Pure numpy/cv2 logic only — no model loads — so it runs anywhere numpy+cv2 are
importable. Run with: ``python tests/test_pipeline_core.py`` (or via pytest).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from libs.pipeline import (  # noqa: E402
    Subject,
    _best_face_for_head,
    _head_fits_person,
    build_subjects,
    locate_head,
)
from libs.pose_rtmw import PoseFrame  # noqa: E402
from libs.tracker import SubjectTracker  # noqa: E402
from libs.utils import _clip_to_box, clip_to_body  # noqa: E402

FW, FH = 320, 320


def _box_centre(b):
    return (float(b[0] + b[2]) / 2, float(b[1] + b[3]) / 2)


def _tadpole_mask(rot: int = 0):
    """A body silhouette: a narrow head end (top) over a wide body (bottom).

    rot rotates by rot*90° so the head end moves; used to prove orientation is
    data-driven, not a hardcoded 'top'.
    """
    m = np.zeros((FH, FW), dtype=np.uint8)
    # Wide body: rows 130..260, cols 110..210 (width 100).
    m[130:260, 110:210] = 1
    # Narrow head: rows 60..130, cols 145..175 (width 30).
    m[60:130, 145:175] = 1
    for _ in range(rot % 4):
        m = np.rot90(m).copy()
    return np.ascontiguousarray(m)


def _pose_with_head(cx, cy):
    """A 133-kpt pose with confident head anchors at (cx, cy) and a valid torso
    below it (shoulders above hips), so _head_box_from_anchors accepts it."""
    pose = np.zeros((133, 3), dtype=np.float32)
    # nose, eyes, ears around the head centre
    for i, (dx, dy) in enumerate([(0, 0), (-6, -2), (6, -2), (-12, 0), (12, 0)]):
        pose[i] = (cx + dx, cy + dy, 0.9)
    sh_y, hp_y = cy + 50, cy + 110
    pose[5] = (cx - 18, sh_y, 0.9)   # L shoulder
    pose[6] = (cx + 18, sh_y, 0.9)   # R shoulder
    pose[11] = (cx - 16, hp_y, 0.9)  # L hip
    pose[12] = (cx + 16, hp_y, 0.9)  # R hip
    return pose


def test_locate_head_prefers_pose_on_body():
    mask = _tadpole_mask(0)
    pose = _pose_with_head(160, 95)  # head over the narrow top, on the body
    head = locate_head(mask, np.array([100, 40, 220, 270], np.float32), pose, FW, FH)
    assert head is not None
    cx, cy = _box_centre(head)
    assert abs(cx - 160) < 40 and abs(cy - 95) < 50, f"pose head expected ~ (160,95), got ({cx},{cy})"
    print("  ok: locate_head uses pose head box when on body")


def test_locate_head_rejects_offbody_pose():
    """A pose head box over empty background is dropped → no head (no guess)."""
    mask = _tadpole_mask(0)
    pose = _pose_with_head(20, 20)  # head anchors far off the body (background)
    head = locate_head(mask, np.array([100, 40, 220, 270], np.float32), pose, FW, FH)
    assert head is None, "off-body pose head must not be trusted, and the body shape is never guessed from"
    print("  ok: locate_head rejects off-body pose (no silhouette guess)")


def test_locate_head_none_without_pose():
    # No pose → no head, regardless of whether a mask is present.
    assert locate_head(None, np.array([0, 0, 10, 10], np.float32), None, FW, FH) is None
    assert locate_head(_tadpole_mask(0), np.array([100, 40, 220, 270], np.float32),
                       None, FW, FH) is None
    print("  ok: locate_head None without pose (mask alone never guesses a head)")


class _FakeFace:
    def __init__(self, bbox, n_lm, score):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.det_score = score
        self.landmark_2d_106 = None if n_lm == 0 else np.zeros((n_lm, 2), np.float32)


def test_best_face_for_head_picks_inside_and_richer():
    head = np.array([100, 100, 160, 160], np.float32)
    inside_scrfd = _FakeFace([110, 110, 150, 150], 106, 0.8)
    inside_pose = _FakeFace([108, 108, 152, 152], 68, 0.95)
    outside = _FakeFace([10, 10, 40, 40], 106, 0.99)
    chosen = _best_face_for_head(head, [outside, inside_pose, inside_scrfd])
    assert chosen is inside_scrfd, "should prefer the 106-pt face inside the head"
    assert _best_face_for_head(head, [outside]) is None, "a far face must not match"
    print("  ok: best_face_for_head picks inside + richer, drops outside")


def test_clip_to_body_drops_offbody():
    body = np.zeros((FH, FW), dtype=np.uint8)
    body[100:160, 100:160] = 1
    scratch = np.zeros((FH, FW), dtype=np.uint8)
    scratch[100:160, 100:160] = 255   # on body
    scratch[10:40, 10:40] = 255        # off body
    clip_to_body(scratch, body, np.array([100, 100, 160, 160], np.float32))
    assert scratch[120, 120] == 255, "on-body blur must survive"
    assert scratch[20, 20] == 0, "off-body blur must be removed"
    print("  ok: clip_to_body removes off-body blur")


def test_clip_to_body_noop_without_mask():
    scratch = np.zeros((FH, FW), dtype=np.uint8)
    scratch[10:40, 10:40] = 255
    clip_to_body(scratch, None, np.array([10, 10, 40, 40], np.float32))
    assert scratch[20, 20] == 255, "no mask → blur untouched (never erased)"
    print("  ok: clip_to_body no-op without mask")


def _subj(head, score=0.9, face=None, mask=None):
    h = np.asarray(head, dtype=np.float32)
    return Subject(h.copy(), mask, h.copy(), face, score)


def test_tracker_stable_ids_and_hold():
    tr = SubjectTracker(fps=10.0, match_iou=0.3, hold_secs=0.3)  # hold = 3 frames
    a = [100, 100, 140, 140]
    b = [200, 200, 240, 240]
    out1 = tr.update([_subj(a), _subj(b)], (FH, FW))
    assert len(out1) == 2
    ids1 = {round(t.bbox[0]): t.track_id for t in out1}
    # Frame 2: both move slightly → same ids.
    out2 = tr.update([_subj([104, 102, 144, 142]), _subj([198, 201, 238, 241])], (FH, FW))
    ids2 = {round(t.bbox[0] / 10) * 10: t.track_id for t in out2}
    assert {t.track_id for t in out1} == {t.track_id for t in out2}, "ids must persist"
    # Frame 3: B disappears → held (source 'hold'), still present.
    out3 = tr.update([_subj([104, 102, 144, 142])], (FH, FW))
    held = [t for t in out3 if t.source == "hold"]
    assert len(out3) == 2 and len(held) == 1, "missing subject should be held one frame"
    # Hold expires after hold_frames(=3) misses.
    tr.update([_subj([104, 102, 144, 142])], (FH, FW))
    tr.update([_subj([104, 102, 144, 142])], (FH, FW))
    out6 = tr.update([_subj([104, 102, 144, 142])], (FH, FW))
    assert len(out6) == 1, "held track must drop after hold window"
    print("  ok: tracker stable ids + hold-then-drop")


def test_tracker_source_face_vs_head():
    tr = SubjectTracker(fps=10.0)
    face = _FakeFace([105, 105, 135, 135], 106, 0.9)
    out = tr.update([_subj([100, 100, 140, 140], face=face)], (FH, FW))
    assert out[0].source == "face" and out[0].face is face
    out2 = tr.update([_subj([100, 100, 140, 140])], (FH, FW))
    assert out2[0].source == "head" and out2[0].face is None
    print("  ok: tracker source face vs head")


def test_build_subjects_end_to_end():
    mask = _tadpole_mask(0)
    pose = _pose_with_head(160, 95)
    pf = PoseFrame(
        person_boxes=[(np.array([100, 40, 220, 270], np.float32), 0.9)],
        person_masks=[mask],
        poses=[pose],
    )
    face_inside = _FakeFace([145, 75, 175, 115], 106, 0.85)
    subjects = build_subjects(pf, [face_inside], (FH, FW))
    assert len(subjects) == 1, "one person → one subject"
    s = subjects[0]
    assert s.mask is mask
    assert s.face is face_inside, "the on-head face should refine this subject"
    print("  ok: build_subjects end-to-end (person → head → face)")


def test_build_subjects_face_only_fallback():
    """No RF-DETR people but a detected face → still produce a blur subject."""
    pf = PoseFrame()
    face = _FakeFace([50, 50, 90, 90], 106, 0.8)
    subjects = build_subjects(pf, [face], (FH, FW))
    assert len(subjects) == 1 and subjects[0].mask is None
    print("  ok: build_subjects face-only fallback")


def test_build_subjects_skips_body_without_head():
    """Person detected but no pose and no face → not blurred (the bug fix).

    The top-down false positive: a body silhouette with no head evidence must
    no longer fabricate a head from its shape and blur a leg/torso.
    """
    pf = PoseFrame(
        person_boxes=[(np.array([100, 40, 220, 270], np.float32), 0.9)],
        person_masks=[_tadpole_mask(0)],
        poses=[],
    )
    subjects = build_subjects(pf, [], (FH, FW))
    assert subjects == [], "a body with no head/face evidence must not be blurred"
    print("  ok: build_subjects skips a body with no head evidence")


def test_build_subjects_face_without_pose():
    """No pose, but a face on the body → one subject, head = the face box."""
    pf = PoseFrame(
        person_boxes=[(np.array([100, 40, 220, 270], np.float32), 0.9)],
        person_masks=[_tadpole_mask(0)],
        poses=[],
    )
    face = _FakeFace([150, 80, 180, 120], 106, 0.85)  # centre (165,100) on the body
    subjects = build_subjects(pf, [face], (FH, FW))
    assert len(subjects) == 1, "a face on a body must be blurred even without pose"
    s = subjects[0]
    assert s.face is face
    assert np.allclose(s.head, face.bbox[:4]), "head region should be the face box"
    print("  ok: build_subjects uses the face box as head when pose is absent")


def _subj_pb(person_box, head, score=0.9, face=None, mask=None):
    """Subject with an explicit person box distinct from the head box."""
    return Subject(np.asarray(person_box, np.float32), mask,
                   np.asarray(head, np.float32), face, score)


def test_tracker_no_ghost_on_fast_head_move():
    """A head darting across the frame stays ONE track (the ghost-trail fix).

    Body box barely moves; the head jumps far enough that head-box IoU and the
    centre-distance fallback both fail. Old behaviour: old track coasts at the
    stale spot while a new track is born at the new head → two blurred heads on
    the motion path. New behaviour: person-box IoU keeps it a single track.
    """
    tr = SubjectTracker(fps=30.0, match_iou=0.3, hold_secs=0.5)
    body = [100, 40, 220, 270]
    out1 = tr.update([_subj_pb(body, [150, 50, 185, 95])], (FH, FW))
    assert len(out1) == 1
    tid = out1[0].track_id
    # Frame 2: body unchanged, head jumped to the far corner (IoU 0, far centre).
    out2 = tr.update(
        [_subj_pb([102, 42, 222, 272], [40, 220, 75, 265])], (FH, FW))
    assert len(out2) == 1, f"fast head move must not spawn a ghost track, got {len(out2)}"
    assert out2[0].track_id == tid, "the same person must keep its id across a fast head move"
    # The single track followed the head (snap-on-motion), not held at the old spot.
    cx, cy = _box_centre(out2[0].bbox)
    assert cx < 120 and cy > 180, f"head should have followed the move, got ({cx},{cy})"
    print("  ok: fast head move keeps one track (no ghost trail)")


def test_head_fits_person_rejects_oversized():
    person = np.array([100, 40, 220, 270], np.float32)   # 120 × 230
    small = np.array([150, 60, 185, 110], np.float32)     # a real head
    assert _head_fits_person(small, person)
    huge = np.array([100, 40, 220, 230], np.float32)       # ~69 % of the body
    assert not _head_fits_person(huge, person), "a head spanning most of the body is a misfit"
    print("  ok: _head_fits_person rejects an oversized head")


def test_build_subjects_rejects_oversized_pose_head():
    """A pose whose head box covers most of the body → no head → skipped (no face)."""
    mask = _tadpole_mask(0)
    pose = _pose_with_head(160, 95)
    # Blow the ears far apart so the anchor head box becomes huge.
    pose[3] = (40, 150, 0.9)    # L ear
    pose[4] = (280, 150, 0.9)   # R ear
    pf = PoseFrame(
        person_boxes=[(np.array([100, 40, 220, 270], np.float32), 0.9)],
        person_masks=[mask], poses=[pose])
    subjects = build_subjects(pf, [], (FH, FW))
    assert subjects == [], "an oversized pose head with no face must not blur the body"
    print("  ok: build_subjects rejects an oversized pose head")


def test_clip_to_box_bounds_maskless():
    """The maskless cap zeroes everything outside the grown head box."""
    scratch = np.full((FH, FW), 255, dtype=np.uint8)
    head = np.array([120, 120, 160, 160], np.float32)   # 40 × 40
    _clip_to_box(scratch, head, grow=0.35)
    # Inside the head box always survives.
    assert scratch[140, 140] == 255
    # Far outside (a frame corner) is always cleared — no full-frame smear.
    assert scratch[10, 10] == 0 and scratch[300, 300] == 0
    # The kept region is bounded: nothing past head ± grow (×2 up) remains.
    ys, xs = np.nonzero(scratch)
    assert xs.min() >= 120 - 0.35 * 40 - 1 and xs.max() <= 160 + 0.35 * 40 + 1
    assert ys.min() >= 120 - 0.7 * 40 - 1 and ys.max() <= 160 + 0.35 * 40 + 1
    print("  ok: _clip_to_box bounds a maskless blur")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} tests\n")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
