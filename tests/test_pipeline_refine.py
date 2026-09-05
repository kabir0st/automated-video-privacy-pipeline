"""Offline refinement: soft prune, bridging, extension, upsampling."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.refine import (RefineConfig, bridge, extend, prune, refine,  # noqa: E402
                             trim, upsample)
from pipeline.types import Src, Tracklet                                   # noqa: E402

HEAD = int(Src.HEADDET)
FACE = int(Src.HEADDET | Src.FACE_IN_HEAD | Src.MULTI)


def tl(tid, start, n, x0=100.0, vx=3.0, hits=None, src=HEAD, score=0.7, w=90.0):
    boxes = np.stack([np.array([x0 + vx * i, 200.0, w, 110.0], np.float32)
                      for i in range(n)])
    hits = np.ones(n, bool) if hits is None else np.asarray(hits, bool)
    return Tracklet(tid, start, boxes, np.full(n, score, np.float32), hits,
                    boxes.copy(), np.full(n, src, np.uint32), np.zeros(n, bool))


class TestTrimPrune:
    def test_trim_cuts_coast(self):
        t = tl(1, 0, 10, hits=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        r = trim(t)
        assert r.start == 2 and len(r.boxes) == 4

    def test_all_coast_is_none(self):
        assert trim(tl(1, 0, 5, hits=[0] * 5)) is None

    def test_single_measurement_rejected(self):
        kept, rej, why = prune([tl(1, 0, 1)], RefineConfig(), 25)
        assert not kept and rej and "single" in why[1]

    def test_short_no_face_rejected_but_short_with_face_kept(self):
        cfg = RefineConfig(min_track_s=0.2, min_hits=3)
        kept, rej, _ = prune([tl(1, 0, 2, src=HEAD)], cfg, 25)
        assert not kept
        kept, rej, _ = prune([tl(2, 0, 2, src=FACE)], cfg, 25)
        assert kept and not rej

    def test_normal_kept(self):
        kept, rej, _ = prune([tl(1, 0, 30)], RefineConfig(), 25)
        assert len(kept) == 1 and not rej


class TestBridge:
    def test_joins_across_gap_with_interp_flags(self):
        a = tl(1, 0, 20)                       # ends x=157
        b = tl(2, 30, 20, x0=100 + 3 * 30)     # continues the line
        out = bridge([a, b], gap_max=25, cfg=RefineConfig())
        assert len(out) == 1
        t = out[0]
        assert t.start == 0 and t.end == 49
        assert (~t.hits[20:30]).all()
        assert (t.src_arr()[20:30] & int(Src.INTERP)).all()
        # linear path
        assert abs(t.boxes[25, 0] - (100 + 3 * 25)) < 6

    def test_refuses_when_gap_too_long(self):
        a, b = tl(1, 0, 20), tl(2, 60, 20, x0=100 + 3 * 60)
        assert len(bridge([a, b], 25, RefineConfig())) == 2

    def test_refuses_when_corridor_occupied(self):
        a = tl(1, 0, 20)
        b = tl(2, 30, 20, x0=100 + 3 * 30)
        blocker = tl(3, 15, 20, x0=100 + 3 * 22, vx=0.0)   # sits in the gap path
        out = bridge([a, b, blocker], 25, RefineConfig())
        assert len(out) == 3

    def test_identity_veto(self):
        a, b = tl(1, 0, 20), tl(2, 30, 20, x0=100 + 3 * 30)
        v1 = np.zeros(8, np.float32); v1[0] = 1
        v2 = np.zeros(8, np.float32); v2[1] = 1          # orthogonal → cos 0
        out = bridge([a, b], 25, RefineConfig(), identities={1: v1, 2: v2})
        assert len(out) == 2

    def test_size_ratio_gate(self):
        a, b = tl(1, 0, 20), tl(2, 30, 20, x0=100 + 3 * 30, w=400.0)
        assert len(bridge([a, b], 25, RefineConfig())) == 2


class TestExtendUpsample:
    def test_extend_holds_ends_within_bounds(self):
        t = extend(tl(1, 2, 10), ext=5, n_frames=20)
        assert t.start == 0 and t.end == 16
        assert (t.src_arr()[:2] & int(Src.INTERP)).all()
        assert not t.hits[:2].any()

    @pytest.mark.parametrize("stride", [2, 3, 4])
    def test_upsample_marks_only_measured_frames(self, stride):
        t = upsample(tl(1, 4, 10), stride, n_frames=1000)
        assert t.start == 4 * stride
        assert len(t.boxes) == 9 * stride + 1
        assert t.hits[::stride].all()
        assert not t.hits[1::stride].any()
        assert (t.src_arr()[1::stride] & int(Src.INTERP)).all()
        # linear: box at frame k matches source line
        k = 5
        assert abs(t.boxes[k, 0] - (100 + 3 * k / stride)) < 1e-3

    def test_refine_end_to_end_frame_space(self):
        raw = [tl(1, 0, 20), tl(2, 30, 20, x0=100 + 3 * 30), tl(3, 0, 1)]
        kept, rej, ids, why = refine(raw, fps=25 / 2, n_frames=60, stride=2,
                                     cfg=RefineConfig(bridge_gap_s=2.5))
        assert len(kept) == 1 and len(rej) == 1
        t = kept[0]
        assert t.start == 0 and t.end <= 119
        assert 3 in why
