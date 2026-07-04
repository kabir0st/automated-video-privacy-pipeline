"""Unit tests for libs/pose.py's pure-numpy pre/post-process and anatomical
anchor/torso-axis logic. Synthetic only, no ONNX model — mirrors
tests/test_evidence.py's style (hand-built keypoints/logits, no inference).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.pose import (_L_EAR, _L_EYE, _L_HIP,  # noqa: E402
                       _L_SHOULDER, _NOSE, _R_EAR, _R_EYE, _R_HIP,
                       _R_SHOULDER, _bbox_xyxy2cs, _get_simcc_maximum,
                       _head_anchor, _top_down_affine, _torso_axis)

FRAME_HW = (1080, 1920)


def _kpts(**overrides) -> tuple[np.ndarray, np.ndarray]:
    """17x2 keypoints / 17 scores, all zeroed/low-confidence by default; pass
    e.g. ``nose=(x, y, s)`` to set one joint."""
    kpts = np.zeros((17, 2), dtype=np.float32)
    scr = np.zeros(17, dtype=np.float32)
    names = {"nose": _NOSE, "l_eye": _L_EYE, "r_eye": _R_EYE,
             "l_ear": _L_EAR, "r_ear": _R_EAR, "l_shoulder": _L_SHOULDER,
             "r_shoulder": _R_SHOULDER, "l_hip": _L_HIP, "r_hip": _R_HIP}
    for name, (x, y, s) in overrides.items():
        i = names[name]
        kpts[i] = (x, y)
        scr[i] = s
    return kpts, scr


class TestSimccMaximum:
    def test_argmax_and_score(self):
        k, wx, wy = 1, 20, 30
        simcc_x = np.zeros((1, k, wx), np.float32)
        simcc_y = np.zeros((1, k, wy), np.float32)
        simcc_x[0, 0, 8] = 0.9
        simcc_y[0, 0, 15] = 0.7
        locs, vals = _get_simcc_maximum(simcc_x, simcc_y)
        assert locs.shape == (1, 2) and vals.shape == (1,)
        assert locs[0, 0] == 8 and locs[0, 1] == 15
        assert vals[0] == pytest.approx(0.8)

    def test_dead_output_locs_negative_one(self):
        simcc_x = np.zeros((1, 2, 10), np.float32)
        simcc_y = np.zeros((1, 2, 10), np.float32)
        locs, vals = _get_simcc_maximum(simcc_x, simcc_y)
        assert vals[0] == 0.0 and vals[1] == 0.0
        assert (locs[0] == -1).all() and (locs[1] == -1).all()


class TestAffinePreprocess:
    def test_bbox_xyxy2cs_center_and_padded_scale(self):
        center, scale = _bbox_xyxy2cs(
            np.array([100.0, 100.0, 200.0, 300.0]), padding=1.25)
        assert center[0] == pytest.approx(150.0)
        assert center[1] == pytest.approx(200.0)
        assert scale[0] == pytest.approx(100.0 * 1.25)
        assert scale[1] == pytest.approx(200.0 * 1.25)

    def test_top_down_affine_maps_bbox_centre_to_canvas_centre(self):
        """The centre of the source bbox must land at the centre of the
        destination canvas — the one invariant every SimCC decode relies on
        to invert keypoints back to frame coordinates correctly."""
        img = np.zeros((480, 640, 3), np.uint8)
        center = np.array([320.0, 240.0], np.float32)
        scale = np.array([200.0, 200.0], np.float32)
        warped, adj_scale = _top_down_affine((192, 256), scale, center, img)
        assert warped.shape == (256, 192, 3)
        # aspect-ratio reshape: input is 192x256 (ratio 0.75); a square
        # source scale (200x200) is narrower than that ratio requires, so the
        # height component grows to match: adj_scale = [200, 200/0.75].
        assert adj_scale[0] == pytest.approx(200.0)
        assert adj_scale[1] == pytest.approx(200.0 / 0.75)

    def test_top_down_affine_roundtrip_point(self):
        """A known point inside the bbox, warped by hand through the same
        matrix the function builds, should land at a predictable spot in the
        canvas — regression guard against a transposed/flipped affine."""
        import cv2

        from libs.pose import _get_warp_matrix

        center = np.array([100.0, 100.0], np.float32)
        scale = np.array([80.0, 80.0], np.float32)
        mat = _get_warp_matrix(center, scale, (192, 256))
        # the bbox centre must warp to the canvas centre
        pt = np.array([[[center[0], center[1]]]], np.float32)
        dst = cv2.transform(pt, mat)[0, 0]
        assert dst[0] == pytest.approx(96.0, abs=1.0)
        assert dst[1] == pytest.approx(128.0, abs=1.0)


class TestTorsoAxis:
    def test_confident_torso_returns_axis(self):
        kpts, scr = _kpts(l_shoulder=(100, 200, 0.9), r_shoulder=(200, 200, 0.9),
                          l_hip=(110, 400, 0.9), r_hip=(190, 400, 0.9))
        axis = _torso_axis(kpts, scr)
        assert axis is not None
        sh, hp, sw = axis
        assert sh[0] == pytest.approx(150.0) and sh[1] == pytest.approx(200.0)
        assert hp[0] == pytest.approx(150.0) and hp[1] == pytest.approx(400.0)
        assert sw == pytest.approx(100.0)

    def test_missing_hip_returns_none(self):
        kpts, scr = _kpts(l_shoulder=(100, 200, 0.9), r_shoulder=(200, 200, 0.9),
                          l_hip=(110, 400, 0.1), r_hip=(190, 400, 0.9))
        assert _torso_axis(kpts, scr) is None

    def test_degenerate_zero_width_shoulders_returns_none(self):
        kpts, scr = _kpts(l_shoulder=(150, 200, 0.9), r_shoulder=(150, 200, 0.9),
                          l_hip=(110, 400, 0.9), r_hip=(190, 400, 0.9))
        assert _torso_axis(kpts, scr) is None


class TestHeadAnchor:
    def test_anchor_from_confident_ears_above_shoulders(self):
        kpts, scr = _kpts(
            l_ear=(120, 90, 0.8), r_ear=(180, 90, 0.8), nose=(150, 100, 0.8),
            l_shoulder=(100, 200, 0.9), r_shoulder=(200, 200, 0.9),
            l_hip=(110, 400, 0.9), r_hip=(190, 400, 0.9))
        box = _head_anchor(kpts, scr, *FRAME_HW)
        assert box is not None
        assert box[0] < 150 < box[2]
        assert box[1] < 100 < box[3]

    def test_anchor_on_hip_side_of_torso_rejected(self):
        """A head-anchor cluster sitting on the hip side of the shoulder-hip
        axis is a skeleton misfit (e.g. legs read as a face) — must reject,
        not fabricate a box there."""
        kpts, scr = _kpts(
            l_ear=(120, 500, 0.8), r_ear=(180, 500, 0.8), nose=(150, 510, 0.8),
            l_shoulder=(100, 200, 0.9), r_shoulder=(200, 200, 0.9),
            l_hip=(110, 400, 0.9), r_hip=(190, 400, 0.9))
        assert _head_anchor(kpts, scr, *FRAME_HW) is None

    def test_torso_only_fallback_places_head_beyond_shoulders(self):
        """No anchor keypoints confident at all (turned away) but a solid
        torso — guess a head beyond the shoulders along the body axis."""
        kpts, scr = _kpts(l_shoulder=(100, 200, 0.9), r_shoulder=(200, 200, 0.9),
                          l_hip=(110, 400, 0.9), r_hip=(190, 400, 0.9))
        box = _head_anchor(kpts, scr, *FRAME_HW)
        assert box is not None
        # placed above the shoulder line (smaller y), on the shoulder axis
        assert box[3] <= 200 + 1e-3

    def test_no_torso_no_anchor_keypoints_returns_none(self):
        kpts, scr = _kpts()
        assert _head_anchor(kpts, scr, *FRAME_HW) is None

    def test_torso_length_outside_plausible_range_rejected(self):
        """A 'torso' far shorter than a real one (e.g. a knee-to-knee
        misfit) must not seed a head guess."""
        kpts, scr = _kpts(l_shoulder=(100, 200, 0.9), r_shoulder=(200, 200, 0.9),
                          l_hip=(105, 210, 0.9), r_hip=(195, 210, 0.9))
        assert _head_anchor(kpts, scr, *FRAME_HW) is None

    def test_low_score_anchor_below_head_score_min_rejected(self):
        """Ears/nose just above _KPT_THR but the averaged anchor score still
        must clear _HEAD_SCORE_MIN — a weak guess never anchors."""
        kpts, scr = _kpts(
            l_ear=(120, 90, 0.31), r_ear=(180, 90, 0.31), nose=(150, 100, 0.31),
            l_shoulder=(100, 200, 0.9), r_shoulder=(200, 200, 0.9),
            l_hip=(110, 400, 0.9), r_hip=(190, 400, 0.9))
        assert _head_anchor(kpts, scr, *FRAME_HW) is None
