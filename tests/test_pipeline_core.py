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
    _head_from_mask,
    build_subjects,
    locate_head,
)
from libs.pose_rtmw import PoseFrame  # noqa: E402
from libs.tracker import SubjectTracker  # noqa: E402
from libs.utils import clip_to_body  # noqa: E402

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


def test_head_from_mask_upright():
    head = _head_from_mask(_tadpole_mask(0), np.array([100, 40, 220, 270], np.float32))
    assert head is not None, "should locate a head on a clear tadpole"
    cx, cy = _box_centre(head)
    # Head is the narrow top: centre should be in the upper part of the body.
    assert cy < 150, f"head should be near the top, got cy={cy}"
    assert 130 < cx < 190, f"head should be horizontally centred, got cx={cx}"
    print("  ok: head_from_mask upright")


def test_head_from_mask_orientation_tracks_rotation():
    """The head end must follow the silhouette when rotated, not stay at 'top'."""
    # 180°: head end now at the bottom.
    head = _head_from_mask(_tadpole_mask(2), np.array([100, 40, 220, 280], np.float32))
    assert head is not None
    _, cy = _box_centre(head)
    assert cy > 170, f"after 180° rotation head should be low, got cy={cy}"
    print("  ok: head_from_mask follows rotation")


def test_head_from_mask_rejects_tiny():
    m = np.zeros((FH, FW), dtype=np.uint8)
    m[10:14, 10:14] = 1  # 16 px < _MIN_MASK_PX
    assert _head_from_mask(m, np.array([0, 0, FW, FH], np.float32)) is None
    print("  ok: head_from_mask rejects tiny mask")


def test_locate_head_prefers_pose_on_body():
    mask = _tadpole_mask(0)
    pose = _pose_with_head(160, 95)  # head over the narrow top, on the body
    head = locate_head(mask, np.array([100, 40, 220, 270], np.float32), pose, FW, FH)
    assert head is not None
    cx, cy = _box_centre(head)
    assert abs(cx - 160) < 40 and abs(cy - 95) < 50, f"pose head expected ~ (160,95), got ({cx},{cy})"
    print("  ok: locate_head uses pose head box when on body")


def test_locate_head_rejects_offbody_pose():
    """A pose head box over empty background is dropped; the mask relocates it."""
    mask = _tadpole_mask(0)
    pose = _pose_with_head(20, 20)  # head anchors far off the body (background)
    head = locate_head(mask, np.array([100, 40, 220, 270], np.float32), pose, FW, FH)
    assert head is not None, "should fall back to the silhouette head"
    cx, cy = _box_centre(head)
    # Must be relocated onto the body, not left at the off-body (20,20).
    assert cy > 50 and 120 < cx < 200, f"off-body pose should relocate to body, got ({cx},{cy})"
    print("  ok: locate_head rejects off-body pose, relocates to silhouette")


def test_locate_head_none_without_mask_or_pose():
    assert locate_head(None, np.array([0, 0, 10, 10], np.float32), None, FW, FH) is None
    print("  ok: locate_head None when nothing available")


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
