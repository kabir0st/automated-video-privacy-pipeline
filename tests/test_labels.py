"""Label-corpus behaviour, with the privacy properties pinned down as hard
assertions: no imagery, no paths, no filenames, and video ids that can't be
linked back to a file without a machine-local secret."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libs import labels                              # noqa: E402
from pipeline.types import Src, Track, Tracklet      # noqa: E402


def _tr(tid, n=30, vx=2.0, kept=True, verified=None, susp=0.2):
    boxes = np.stack([np.array([100.0 + vx * i, 200.0, 90.0, 110.0], np.float32)
                      for i in range(n)])
    t = Tracklet(tid, 0, boxes, np.full(n, 0.85, np.float32), np.ones(n, bool),
                 boxes.copy(), np.full(n, int(Src.HEADDET | Src.MULTI), np.uint32),
                 np.ones(n, bool))
    return Track(t=t, suspicion=susp, reasons={"lonely": 0.1, "weak": 0.2},
                 verified=verified, kept=kept)


@pytest.fixture
def corpus(tmp_path):
    return tmp_path / "labels.jsonl"


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class TestPrivacy:
    def test_no_path_or_filename_anywhere(self, tmp_path, corpus):
        video = tmp_path / "very_identifying_name.mp4"
        video.write_bytes(b"x" * 64)
        labels.record(video, 30.0, [_tr(1), _tr(2, kept=False)], corpus=corpus)
        blob = corpus.read_text()
        assert "very_identifying_name" not in blob
        assert ".mp4" not in blob
        assert str(tmp_path) not in blob

    def test_no_image_like_payload(self, tmp_path, corpus):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        labels.record(video, 30.0, [_tr(1)], corpus=corpus)
        for row in _rows(corpus):
            for k, v in row.items():
                assert not isinstance(v, (list, dict)), k
                if isinstance(v, str):
                    assert len(v) < 64, k

    def test_same_file_same_id_different_file_different_id(self, tmp_path):
        a = tmp_path / "a.mp4"; a.write_bytes(b"x" * 64)
        b = tmp_path / "b.mp4"; b.write_bytes(b"y" * 65)
        assert labels.video_id(a) == labels.video_id(a)
        assert labels.video_id(a) != labels.video_id(b)

    def test_id_is_not_a_bare_hash_of_the_fingerprint(self, tmp_path):
        import hashlib
        a = tmp_path / "a.mp4"; a.write_bytes(b"x" * 64)
        st = a.stat()
        stamp = f"{st.st_size}:{int(st.st_mtime)}".encode()
        bare = {hashlib.blake2b(stamp, digest_size=12).hexdigest(),
                hashlib.sha256(stamp).hexdigest()[:24]}
        assert labels.video_id(a) not in bare

    def test_salt_file_is_owner_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(labels, "_DIR", tmp_path)
        monkeypatch.setattr(labels, "_SALT_FILE", tmp_path / "labels.salt")
        labels._salt()
        assert oct((tmp_path / "labels.salt").stat().st_mode & 0o777) == "0o600"

    def test_corpus_file_is_owner_only(self, tmp_path, corpus):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        labels.record(video, 30.0, [_tr(1)], corpus=corpus)
        assert oct(corpus.stat().st_mode & 0o777) == "0o600"


class TestRecording:
    def test_one_row_per_track(self, tmp_path, corpus):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        n = labels.record(video, 30.0, [_tr(1), _tr(2), _tr(3, kept=False)], corpus=corpus)
        assert n == 3 and len(_rows(corpus)) == 3

    def test_appends_across_exports(self, tmp_path, corpus):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        labels.record(video, 30.0, [_tr(1)], corpus=corpus)
        labels.record(video, 30.0, [_tr(1)], corpus=corpus)
        assert len(_rows(corpus)) == 2

    def test_label_follows_the_users_verdict_not_the_pipelines(self, tmp_path, corpus):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        labels.record(video, 30.0, [_tr(1, kept=True), _tr(2, kept=False)],
                      enabled_ids={1: False, 2: True}, corpus=corpus)
        by = {r["tid"]: r for r in _rows(corpus)}
        assert by[1]["blur"] is False and by[1]["overridden"] is True
        assert by[2]["blur"] is True and by[2]["overridden"] is True

    def test_defaults_when_user_touched_nothing(self, tmp_path, corpus):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        labels.record(video, 30.0, [_tr(1, kept=True), _tr(2, kept=False)], corpus=corpus)
        by = {r["tid"]: r for r in _rows(corpus)}
        assert by[1]["blur"] and not by[1]["overridden"]
        assert not by[2]["blur"] and not by[2]["overridden"]

    def test_disabled_writes_nothing(self, tmp_path, corpus, monkeypatch):
        monkeypatch.setenv("AVPP_LABELS", "0")
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        assert labels.record(video, 30.0, [_tr(1)], corpus=corpus) == 0
        assert not corpus.exists()

    def test_unwritable_corpus_is_swallowed(self, tmp_path):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        bad = tmp_path / "nodir" / "x" / "labels.jsonl"
        os.makedirs(bad.parent.parent, exist_ok=True)
        (bad.parent.parent / "x").write_bytes(b"")     # a file where a dir must be
        assert labels.record(video, 30.0, [_tr(1)], corpus=bad) == 0

    def test_empty_input_writes_nothing(self, tmp_path, corpus):
        video = tmp_path / "v.mp4"; video.write_bytes(b"x" * 64)
        assert labels.record(video, 30.0, [], corpus=corpus) == 0


class TestFeatures:
    def test_static_and_moving_tracks_differ_in_motion(self):
        assert labels.features(_tr(1, vx=0.0), 30)["motion_med"] < \
            labels.features(_tr(2, vx=3.0), 30)["motion_med"]

    def test_components_carried_through(self):
        f = labels.features(_tr(1), 30)
        assert f["c_lonely"] == 0.1 and f["suspicion"] == 0.2

    def test_duration_uses_fps(self):
        assert labels.features(_tr(1, n=30), 30)["duration_s"] == 1.0

    def test_verify_absent_is_none_not_false(self):
        assert labels.features(_tr(1), 30)["verified"] is None
        assert labels.features(_tr(1, verified=0.8), 30)["verified"] == 0.8

    def test_single_frame_track_does_not_crash(self):
        labels.features(_tr(1, n=1), 30)
