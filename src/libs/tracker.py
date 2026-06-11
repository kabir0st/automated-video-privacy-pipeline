from typing import Any

import numpy as np

from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack

# Minimum IoU to consider a tracked box matched to an InsightFace detection.
_MATCH_IOU = 0.3


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class ByteTrackWrapper:
    """Wraps boxmot ByteTrack, returning both matched and predicted-only tracks.

    Each call to update() returns a list of
        (track_id, face_or_None, predicted_bbox)
    where face_or_None is the InsightFace Face object when a detection was
    matched this frame, or None for tracks that are coasting on Kalman
    prediction (e.g. face partially out of frame).
    """

    def __init__(self, fps: float = 30.0, match_iou: float = _MATCH_IOU) -> None:
        self._tracker = ByteTrack(frame_rate=fps)
        self._match_iou = match_iou

    def set_match_iou(self, match_iou: float) -> None:
        """Update the detection-matching threshold without resetting track IDs."""
        self._match_iou = match_iou

    def update(
        self,
        faces: list[Any],
        frame: np.ndarray,
    ) -> list[tuple[int, Any | None, np.ndarray]]:
        if faces:
            dets = np.array(
                [[*f.bbox[:4], float(f.det_score), 0] for f in faces],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracked = self._tracker.update(dets, frame)
        if len(tracked) == 0:
            return []

        result: list[tuple[int, Any | None, np.ndarray]] = []
        for row in tracked:
            track_box = row[:4]
            track_id = int(row[4])

            best_face = None
            if faces:
                candidate = max(faces, key=lambda f, tb=track_box: _iou(f.bbox[:4], tb))
                if _iou(candidate.bbox[:4], track_box) >= self._match_iou:
                    best_face = candidate

            result.append((track_id, best_face, track_box))

        return result
