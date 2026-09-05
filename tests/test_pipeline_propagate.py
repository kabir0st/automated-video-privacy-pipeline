"""Manual-box propagation on a synthetic clip: a bright square drifting
over noise. Template matching alone must follow it; raw candidates snap."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.propagate import PropagateConfig, propagate   # noqa: E402

W, H, N = 320, 240, 60


def pos(f):
    return 40 + 2 * f, 60 + f


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("v") / "sq.mp4")
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    rng = np.random.default_rng(0)
    for f in range(N):
        img = rng.integers(0, 60, (H, W, 3), np.uint8)
        x, y = pos(f)
        cv2.rectangle(img, (x, y), (x + 40, y + 40), (230, 230, 230), -1)
        cv2.circle(img, (x + 20, y + 20), 8, (30, 30, 30), -1)   # distinctive texture
        wr.write(img)
    wr.release()
    cap = cv2.VideoCapture(path)
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= N - 1
    cap.release()
    return path


def _err(track, f):
    d = dict(track)
    x, y = pos(f)
    b = d[f]
    return abs((b[0] + b[2]) / 2 - (x + 20)) + abs((b[1] + b[3]) / 2 - (y + 20))


def test_forward_template_follow(clip):
    x, y = pos(5)
    out = propagate(clip, 5, np.array([x, y, x + 40, y + 40]), direction=1,
                    cfg=PropagateConfig(max_frames=30))
    assert out[0][0] == 5 and out[-1][0] >= 30
    assert _err(out, 30) < 8


def test_backward_follow(clip):
    x, y = pos(40)
    out = propagate(clip, 40, np.array([x, y, x + 40, y + 40]), direction=-1,
                    cfg=PropagateConfig(max_frames=30, chunk=8))
    frames = [f for f, _ in out]
    assert frames[0] == 40 and min(frames) <= 12
    assert _err(out, 15) < 8


def test_raw_candidates_snap_exactly(clip):
    raw = {}
    for f in range(N):
        x, y = pos(f)
        raw[f] = np.array([[x + 1, y + 1, x + 41, y + 41, 0.9, 1]], np.float32)
    x, y = pos(5)
    out = propagate(clip, 5, np.array([x, y, x + 40, y + 40]), direction=1, raw=raw,
                    cfg=PropagateConfig(max_frames=20))
    d = dict(out)
    assert np.allclose(d[15], raw[15][0, :4])
