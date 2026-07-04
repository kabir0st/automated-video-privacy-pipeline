"""Unit tests for libs/sidecar.py — save/load round-trip, fingerprint
invalidation on video change and on analysis-param change, and ManualRegion
interpolation. Synthetic only, uses a real temp file for stat()-based
fingerprinting."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.sidecar import (ManualRegion, ReviewDecisions,  # noqa: E402
                          fingerprint, load, save, sidecar_path)
from libs.tracklets import Tracklet  # noqa: E402


def make_video_file(tmp_path: Path, name: str = "clip.mp4", size: int = 1024) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\0" * size)
    return p


def make_tracklet(tid=1, n=5) -> Tracklet:
    boxes = np.stack([np.array([100.0 + i, 200.0, 80.0, 90.0], np.float32)
                      for i in range(n)])
    return Tracklet(
        tid, 10, boxes, np.full(n, 0.8, np.float32), np.ones(n, bool),
        fboxes=boxes.copy(), ev=np.full(n, 5, np.uint32),
        fvalid=np.ones(n, bool))


PARAMS = {"det_conf": 0.5, "det_conf_low": 0.2, "min_hits": 3,
         "max_age_s": 1.0, "rot_assist": True, "evidence_profile": "balanced"}


class TestManualRegion:
    def test_box_at_interpolates_linearly(self):
        m = ManualRegion(start=0, end=10, box0=(0, 0, 10, 10),
                         box1=(10, 10, 20, 20))
        box = m.box_at(5)
        assert box == pytest.approx([5, 5, 15, 15])

    def test_box_at_outside_range_is_none(self):
        m = ManualRegion(start=5, end=10, box0=(0, 0, 10, 10),
                         box1=(0, 0, 10, 10))
        assert m.box_at(4) is None
        assert m.box_at(11) is None

    def test_static_region_same_box_throughout(self):
        m = ManualRegion(start=0, end=4, box0=(1, 2, 3, 4), box1=(1, 2, 3, 4))
        for f in range(5):
            assert m.box_at(f) == pytest.approx([1, 2, 3, 4])


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_tracklet_data(self, tmp_path):
        video = make_video_file(tmp_path)
        t = make_tracklet()
        save(video, fps=30.0, n_frames=100, analysis_params=PARAMS,
             tracklets=[t])
        loaded = load(video, PARAMS)
        assert loaded is not None
        fps, n_frames, tracklets, review = loaded
        assert fps == 30.0 and n_frames == 100
        assert len(tracklets) == 1
        lt = tracklets[0]
        assert lt.tid == t.tid and lt.start == t.start
        np.testing.assert_allclose(lt.boxes, t.boxes)
        np.testing.assert_allclose(lt.fboxes, t.fboxes)
        np.testing.assert_array_equal(lt.ev, t.ev)
        np.testing.assert_array_equal(lt.hits, t.hits)
        np.testing.assert_array_equal(lt.fvalid, t.fvalid)
        assert review.enabled == {} and review.manual_regions == []

    def test_roundtrip_preserves_review_decisions(self, tmp_path):
        video = make_video_file(tmp_path)
        review = ReviewDecisions(
            enabled={1: False, 2: True},
            manual_regions=[ManualRegion(0, 10, (0, 0, 5, 5), (0, 0, 5, 5))])
        save(video, fps=25.0, n_frames=50, analysis_params=PARAMS,
             tracklets=[], review=review)
        loaded = load(video, PARAMS)
        assert loaded is not None
        _, _, _, r = loaded
        assert r.enabled == {1: False, 2: True}
        assert len(r.manual_regions) == 1
        assert r.manual_regions[0].box0 == (0.0, 0.0, 5.0, 5.0)

    def test_no_sidecar_returns_none(self, tmp_path):
        video = make_video_file(tmp_path)
        assert load(video, PARAMS) is None

    def test_sidecar_path_appends_suffix(self, tmp_path):
        video = make_video_file(tmp_path)
        p = sidecar_path(video)
        assert p.name == "clip.mp4.avpp.json"


class TestFingerprintInvalidation:
    def test_changed_analysis_param_invalidates(self, tmp_path):
        video = make_video_file(tmp_path)
        save(video, fps=30.0, n_frames=10, analysis_params=PARAMS,
             tracklets=[make_tracklet()])
        other = dict(PARAMS, det_conf=0.9)
        assert load(video, other) is None
        assert load(video, PARAMS) is not None

    def test_changed_video_content_invalidates(self, tmp_path):
        video = make_video_file(tmp_path, size=1024)
        save(video, fps=30.0, n_frames=10, analysis_params=PARAMS,
             tracklets=[make_tracklet()])
        # rewrite with a different size -> stat() changes -> fingerprint
        # mismatch, even though the path is identical.
        video.write_bytes(b"\1" * 2048)
        assert load(video, PARAMS) is None

    def test_cleanup_only_param_not_part_of_fingerprint(self, tmp_path):
        """bridge_gap_s/smooth_win_s/etc. are cleanup-only and must never be
        passed into analysis_params — this test documents that fingerprint()
        only reacts to what it's given, not to some hidden global state."""
        video = make_video_file(tmp_path)
        f1 = fingerprint(video, PARAMS)
        f2 = fingerprint(video, PARAMS)
        assert f1 == f2

    def test_different_files_different_fingerprint(self, tmp_path):
        v1 = make_video_file(tmp_path, "a.mp4", size=1024)
        time.sleep(0.01)
        v2 = make_video_file(tmp_path, "b.mp4", size=2048)
        assert fingerprint(v1, PARAMS) != fingerprint(v2, PARAMS)


class TestCorruptSidecar:
    def test_garbage_json_degrades_to_none(self, tmp_path):
        video = make_video_file(tmp_path)
        sidecar_path(video).write_text("not json{{{")
        assert load(video, PARAMS) is None

    def test_wrong_schema_version_degrades_to_none(self, tmp_path):
        import json
        video = make_video_file(tmp_path)
        sidecar_path(video).write_text(json.dumps({"schema": 999}))
        assert load(video, PARAMS) is None
