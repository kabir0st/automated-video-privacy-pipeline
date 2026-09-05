"""Tests for render_head_mask + soft-mask blur compositing."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.utils import (BlurPipeline, bloom_face_box,  # noqa: E402
                        render_head_mask)

BOX = np.array([300, 200, 420, 340], dtype=np.float32)  # w=120 h=140


def test_core_fully_opaque():
    mask = render_head_mask((720, 1280), [BOX])
    cx, cy = 360, 270
    assert mask[cy, cx] == 255
    # the whole unpadded box centre region stays opaque despite feathering
    assert (mask[cy - 40:cy + 40, cx - 30:cx + 30] == 255).all()


def test_feather_band_is_gradual_and_bounded():
    mask = render_head_mask((720, 1280), [BOX], pad=0.15, feather=0.15)
    row = mask[270, :].astype(int)
    edge = np.nonzero(row)[0]
    assert len(edge)
    band = row[edge[0]:360]
    assert band.max() == 255
    assert ((row > 0) & (row < 255)).sum() > 0        # feather exists
    assert (np.diff(band) >= -2).all()                # ramps up, no ringing
    # blur never reaches absurdly far: the mask support stays within ~1.7x box
    w = BOX[2] - BOX[0]
    assert edge[0] > BOX[0] - 0.7 * w


def test_half_out_of_frame_still_masked():
    b = np.array([-60, 200, 60, 340], dtype=np.float32)   # centre near x=0
    mask = render_head_mask((720, 1280), [b])
    assert mask[270, 5] > 0


def test_empty_boxes():
    mask = render_head_mask((100, 100), [])
    assert mask.sum() == 0


def test_bloom_face_box_favours_hair():
    face = np.array([100, 100, 200, 220, 0.8], dtype=np.float32)  # w=100 h=120
    out = bloom_face_box(face)
    top_growth = 100 - out[1]
    bottom_growth = out[3] - 220
    side_growth = 100 - out[0]
    assert top_growth > bottom_growth          # hair, not chin
    assert top_growth > side_growth
    assert out[2] - 200 == side_growth         # symmetric sides
    assert out[4] == face[4]                   # score untouched
    # still much tighter than the head box a pseudo-head would grow (×1.9 h)
    assert (out[3] - out[1]) < 1.5 * 120
    # input is not mutated
    assert face[1] == 100


def test_soft_blur_composite_cpu():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    orig = frame.copy()
    mask = render_head_mask((200, 200), [np.array([60, 60, 140, 140])],
                            pad=0.1, feather=0.2)
    bp = BlurPipeline()
    bp.backend = "cpu"          # force deterministic path for the assert
    bp.reconfigure((("gaussian", 31),))
    bp.apply(frame, mask)
    # centre strongly changed, far corner untouched
    assert np.abs(frame[100, 100].astype(int) - orig[100, 100].astype(int)).sum() > 0
    assert (frame[5, 5] == orig[5, 5]).all()
    assert (frame[195, 195] == orig[195, 195]).all()
    # feather band is a mix: closer to original near the mask edge
    ys, xs = np.nonzero((mask > 0) & (mask < 128))
    i = len(xs) // 2
    y, x = ys[i], xs[i]
    d_edge = np.abs(frame[y, x].astype(int) - orig[y, x].astype(int)).mean()
    d_core = np.abs(frame[100, 100].astype(int) - orig[100, 100].astype(int)).mean()
    assert d_edge <= d_core


def test_binary_mask_hard_path_unchanged():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    frame[40:60, 40:60] = 10
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    bp = BlurPipeline()
    bp.backend = "cpu"
    bp.reconfigure((("pixelate", 8),))
    orig = frame.copy()
    bp.apply(frame, mask)
    assert (frame[0:29, 0:29] == orig[0:29, 0:29]).all()
    assert not (frame[30:70, 30:70] == orig[30:70, 30:70]).all()


class TestFeatherROIEquivalence:
    """render_head_mask feathers only the sub-rect the ellipses touch. That is
    a pure optimisation, so it has to be bit-identical to blurring the whole
    frame — including when a head hangs off an edge (cv2.ellipse clips, and
    the ROI must clip the same way) or fills the frame (ROI == frame)."""

    @staticmethod
    def _full_frame(shape_hw, boxes, pad=0.18, feather=0.12):
        mask = np.zeros(shape_hw, np.uint8)
        boxes = list(boxes)
        if not boxes:
            return mask
        diags = [float(np.hypot(b[2] - b[0], b[3] - b[1])) for b in boxes]
        k = (max(3, int(round(feather * (sum(diags) / len(diags)))) | 1)
             if feather > 0 else 0)
        r = k / 2.0
        for b in boxes:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax = max(1.0, w / 2.0 * (1.0 + 2.0 * pad) + r)
            ay = max(1.0, h / 2.0 * (1.0 + 2.0 * pad) + r)
            cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                        (int(round(ax)), int(round(ay))), 0, 0, 360, 255, -1)
        if k >= 3:
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask

    @pytest.mark.parametrize("feather", [0.0, 0.05, 0.12, 0.4])
    @pytest.mark.parametrize("boxes", [
        [np.array([300., 200., 460., 400.], np.float32)],
        [np.array([100., 100., 260., 300.], np.float32),
         np.array([700., 400., 900., 620.], np.float32)],
        [np.array([-80., -60., 120., 140.], np.float32)],       # off top-left
        [np.array([1180., 640., 1400., 900.], np.float32)],     # off bot-right
        [np.array([0., 0., 1280., 720.], np.float32)],          # fills frame
    ])
    def test_matches_full_frame_gaussian(self, boxes, feather):
        shape = (720, 1280)
        assert np.array_equal(
            self._full_frame(shape, boxes, feather=feather),
            render_head_mask(shape, boxes, feather=feather))
