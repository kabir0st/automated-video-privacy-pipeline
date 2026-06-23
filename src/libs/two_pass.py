"""Offline two-pass processing: detect across the whole clip, then forward-
backward interpolate so each face is blurred over its entire on-screen span.

The single-pass live path can only coast a lost face *forward* for hold_secs, so
a face detected intermittently (occlusion, motion blur, a hard angle) leaks on
the frames between detections. Offline we have the whole timeline, so:

  Pass 1   run the detection ensemble (SCRFD ⊕ RTMW, RF-DETR person boxes) on
           every frame and store the detections.
  Link     associate detections across frames into per-face timelines, tolerating
           short gaps (the face was present but undetected).
  Fill     linearly interpolate the bbox + landmarks across each gap — this is
           the forward-*and*-backward fill a single online pass cannot do — and
           acausally smooth each timeline.
  Pass 2   re-read the video and blur every frame from the filled timelines.

Reuses pipeline.detect and the utils mask primitives; it does not use the online
KalmanFaceTracker (offline association replaces it). Degrades the same way the
rest of the pipeline does — RF-DETR/pose simply contribute nothing when absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .pipeline import detect
from .utils import add_bbox_mask, add_ellipse_mask, add_face_mask
from .video_writer import make_video_writer, source_bitrate_kbps

# Association: a detection links to a track whose last box overlaps it this much,
# and a track stays linkable across a gap of up to this many seconds.
_LINK_IOU = 0.2
_GAP_SECS = 1.0
# Acausal smoothing window (frames) and polynomial order, applied per run of
# equal-length landmark sets within a track.
_SG_WINDOW = 15
_SG_POLY = 2
_SG_MIN = 5


@dataclass
class _Det:
    bbox: np.ndarray              # xyxy float32
    landmarks: Optional[np.ndarray]  # (N, 2) float32 or None (anchor/box only)
    score: float
    is_anchor: bool = False


class _Track:
    __slots__ = ("dets", "last_frame", "last_bbox")

    def __init__(self, frame_idx: int, det: _Det) -> None:
        self.dets: dict[int, _Det] = {frame_idx: det}
        self.last_frame = frame_idx
        self.last_bbox = det.bbox


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def _associate(frames: list[list[_Det]], gap: int) -> list[_Track]:
    """Greedy IoU association across frames, tolerating gaps of up to ``gap``
    frames so a momentarily-undetected face stays one timeline."""
    tracks: list[_Track] = []
    for fi, dets in enumerate(frames):
        claimed: set[int] = set()
        for det in dets:
            best: tuple[float, _Track] | None = None
            for tr in tracks:
                if id(tr) in claimed or fi - tr.last_frame > gap:
                    continue
                s = _iou(det.bbox, tr.last_bbox)
                if s > _LINK_IOU and (best is None or s > best[0]):
                    best = (s, tr)
            if best is not None:
                tr = best[1]
                tr.dets[fi] = det
                tr.last_frame = fi
                tr.last_bbox = det.bbox
                claimed.add(id(tr))
            else:
                nt = _Track(fi, det)
                tracks.append(nt)
                claimed.add(id(nt))
    return tracks


def _fill_gaps(track: _Track) -> None:
    """Linearly interpolate bbox (and landmarks when both ends match) across the
    gaps between a track's observed frames — forward-backward by construction."""
    obs = sorted(track.dets)
    for a, b in zip(obs, obs[1:]):
        if b - a <= 1:
            continue
        da, db = track.dets[a], track.dets[b]
        lm_ok = (da.landmarks is not None and db.landmarks is not None
                 and da.landmarks.shape == db.landmarks.shape)
        for fi in range(a + 1, b):
            w = (fi - a) / (b - a)
            bbox = (1 - w) * da.bbox + w * db.bbox
            lm = ((1 - w) * da.landmarks + w * db.landmarks) if lm_ok else None
            track.dets[fi] = _Det(bbox.astype(np.float32),
                                  None if lm is None else lm.astype(np.float32),
                                  (1 - w) * da.score + w * db.score,
                                  is_anchor=da.is_anchor and db.is_anchor)


def _smooth(track: _Track) -> None:
    """Acausal Savitzky-Golay over each run of equal-length landmark sets.

    Faces can switch landmark source mid-life (SCRFD's 106 ⇄ RTMW's 68), so we
    only smooth contiguous runs of identical point count; mixed/short runs are
    left as-is (still gap-filled, just not smoothed)."""
    try:
        from scipy.signal import savgol_filter
    except Exception:  # noqa: BLE001 — smoothing is optional
        return
    obs = sorted(track.dets)
    run: list[int] = []
    n = -1
    for fi in obs + [None]:  # sentinel flushes the final run
        cur = track.dets[fi].landmarks if fi is not None else None
        cn = -1 if cur is None else cur.shape[0]
        if cn == n and fi is not None:
            run.append(fi)
            continue
        if n > 0 and len(run) >= _SG_MIN:
            arr = np.stack([track.dets[i].landmarks for i in run])  # (T, N, 2)
            wlen = min(_SG_WINDOW, len(run))
            if wlen % 2 == 0:
                wlen -= 1
            if wlen > _SG_POLY:
                sm = savgol_filter(arr, wlen, _SG_POLY, axis=0)
                for k, i in enumerate(run):
                    track.dets[i].landmarks = sm[k].astype(np.float32)
        run = [fi] if fi is not None else []
        n = cn


def run_two_pass(
    source: str,
    output: str,
    *,
    app,
    pose,
    blur,
    det_score: float,
    face_aspect: float,
    close_up_ratio: float,
    close_up_target: int,
    hold_secs: float,
    safety_net: bool,
    on_status: Callable[[str], None] = print,
) -> None:
    """Two-pass blur of ``source`` → ``output``. See module docstring."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open '{source}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_kbps = source_bitrate_kbps(cap)
    gap = max(1, int(round(max(hold_secs, _GAP_SECS) * fps)))

    # ── Pass 1: detect on every frame ────────────────────────────────────────
    frames: list[list[_Det]] = []
    fh = fw = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fh == 0:
            fh, fw = frame.shape[:2]
        faces, _head_boxes, anchors, _pf = detect(
            app, frame, pose, det_score=det_score, face_aspect=face_aspect,
            close_up_ratio=close_up_ratio, close_up_target=close_up_target)
        dets = [
            _Det(np.asarray(f.bbox[:4], dtype=np.float32),
                 None if f.landmark_2d_106 is None
                 else np.asarray(f.landmark_2d_106, dtype=np.float32),
                 float(f.det_score))
            for f in faces
        ]
        if safety_net:
            dets += [_Det(np.asarray(b, dtype=np.float32), None, float(s),
                          is_anchor=True) for b, s in anchors]
        frames.append(dets)
        if len(frames) % 200 == 0:
            on_status(f"[two-pass] pass 1: {len(frames)} frames detected")
    cap.release()

    if not frames:
        raise RuntimeError("no frames read from source")

    # ── Link + fill + smooth ─────────────────────────────────────────────────
    tracks = _associate(frames, gap)
    for tr in tracks:
        _fill_gaps(tr)
        _smooth(tr)
    on_status(f"[two-pass] {len(tracks)} face timelines over {len(frames)} frames")

    # Index filled detections by frame for the render pass.
    per_frame: list[list[_Det]] = [[] for _ in frames]
    for tr in tracks:
        for fi, det in tr.dets.items():
            per_frame[fi].append(det)

    # ── Pass 2: re-read and blur from the filled timelines ───────────────────
    writer = make_video_writer(output, fw, fh, fps, bitrate_kbps=src_kbps,
                               on_status=on_status)
    if writer is None:
        raise RuntimeError(f"cannot create output '{output}'")
    cap = cv2.VideoCapture(source)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = np.zeros((fh, fw), dtype=np.uint8)
        for det in per_frame[fi]:
            if det.landmarks is not None:
                add_face_mask(mask, det.landmarks)
            elif det.is_anchor:
                add_ellipse_mask(mask, det.bbox)
            else:
                box_area = (max(0.0, det.bbox[2] - det.bbox[0])
                            * max(0.0, det.bbox[3] - det.bbox[1]))
                if box_area <= 0.08 * fh * fw:
                    add_bbox_mask(mask, det.bbox)
        blur.apply(frame, mask)
        writer.write(frame)
        fi += 1
        if fi % 200 == 0:
            on_status(f"[two-pass] pass 2: {fi}/{len(frames)} frames blurred")
    cap.release()
    writer.release()
    on_status(f"[two-pass] done — {fi} frames written to {output}")
