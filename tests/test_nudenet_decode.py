"""Unit tests for libs/nudenet.py's pure-numpy YOLOv8 decode. Synthetic
only, no ONNX model — mirrors test_pose.py's style (hand-built logits)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from libs.detector import _nms  # noqa: E402
from libs.nudenet import (FACE_FEMALE, FACE_MALE, LABELS,  # noqa: E402
                          _decode_yolov8)


def _row(cx, cy, w, h, cls, score, n_classes=18):
    """One synthetic YOLOv8-export anchor row: [cx, cy, w, h, *class_scores]."""
    scores = np.zeros(n_classes, np.float32)
    scores[cls] = score
    return np.concatenate([[cx, cy, w, h], scores]).astype(np.float32)


class TestDecodeYolov8:
    def test_labels_face_indices(self):
        assert LABELS[FACE_FEMALE] == "FACE_FEMALE"
        assert LABELS[FACE_MALE] == "FACE_MALE"

    def test_single_detection_decodes_to_xyxy(self):
        rows = np.stack([_row(100, 100, 40, 60, FACE_FEMALE, 0.8)])
        out = np.stack([rows.T])   # (1, 22, 1)
        dec = _decode_yolov8(out, floor=0.25)
        assert dec.shape == (1, 6)
        x1, y1, x2, y2, score, cls = dec[0]
        assert x1 == pytest.approx(80.0)
        assert y1 == pytest.approx(70.0)
        assert x2 == pytest.approx(120.0)
        assert y2 == pytest.approx(130.0)
        assert score == pytest.approx(0.8)
        assert int(cls) == FACE_FEMALE

    def test_below_floor_dropped(self):
        rows = np.stack([_row(100, 100, 40, 60, FACE_FEMALE, 0.1)])
        out = np.stack([rows.T])
        dec = _decode_yolov8(out, floor=0.25)
        assert len(dec) == 0

    def test_multiple_anchors_multiple_classes(self):
        rows = np.stack([
            _row(50, 50, 20, 20, FACE_FEMALE, 0.9),
            _row(200, 200, 30, 30, FACE_MALE, 0.6),
            _row(10, 10, 5, 5, 3, 0.05),   # below floor, different class
        ])
        out = np.stack([rows.T])
        dec = _decode_yolov8(out, floor=0.25)
        assert len(dec) == 2
        classes = set(int(c) for c in dec[:, 5])
        assert classes == {FACE_FEMALE, FACE_MALE}

    def test_no_anchors_returns_empty(self):
        out = np.zeros((1, 22, 0), np.float32)
        dec = _decode_yolov8(out, floor=0.25)
        assert dec.shape == (0, 6)

    def test_2d_input_without_batch_dim(self):
        rows = np.stack([_row(100, 100, 40, 60, FACE_FEMALE, 0.8)])
        out = rows.T   # (22, 1), no leading batch dim
        dec = _decode_yolov8(out, floor=0.25)
        assert dec.shape == (1, 6)

    def test_nms_suppresses_overlapping_same_class(self):
        rows = np.stack([
            _row(100, 100, 40, 60, FACE_FEMALE, 0.9),
            _row(102, 101, 40, 60, FACE_FEMALE, 0.7),   # near-duplicate
            _row(300, 300, 40, 60, FACE_FEMALE, 0.8),   # far away, distinct
        ])
        out = np.stack([rows.T])
        dec = _decode_yolov8(out, floor=0.25)
        kept = _nms(dec, iou_thr=0.45)
        assert len(kept) == 2
        assert kept[0, 4] == pytest.approx(0.9)   # highest score kept first
