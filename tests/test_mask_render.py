"""Tests for render_head_mask + soft-mask blur compositing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.utils import BlurPipeline, render_head_mask  # noqa: E402

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
