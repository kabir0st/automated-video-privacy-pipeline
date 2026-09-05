"""Suspicion scoring: how likely is a refined tracklet *not* a head?

Nothing here deletes anything. The score orders tracks for review (most
suspicious first) and sets the pipeline's default verdict through one
threshold, ``auto_disable_above``, which the presets set high — a spurious
blur is cheap, a missed head is not.

Components (each 0..1, higher = more suspicious):
  lonely      no second source ever agreed (no face inside, no other head
              detector) — the single strongest tell for skin misreads
  unverified  magnified re-detection failed (when a verifier ran)
  weak        low detector confidence throughout
  short       very brief
  sparse      mostly coasting between few measurements
  static      does not move or change size at all (a fabric fold does not)
  rotated     only ever seen by rotated passes
  bodypart    coincided with a NudeNet body-part box (vetoed at fusion)
  oversized   larger than a quarter of the frame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .geom import clip_boxes, iou_matrix, rows_to_xyxy
from .types import FACE_SOURCES, Src, Track, Tracklet

_FACE_EV = int(FACE_SOURCES | Src.FACE_IN_HEAD)


@dataclass(frozen=True)
class ScoreConfig:
    auto_disable_above: float = 0.80
    verify_samples: int = 5
    verify_margin: float = 2.5       # crop side = max(w, h) × this
    verify_min_crop: int = 96
    verify_match_iou: float = 0.20
    weights: dict = field(default_factory=lambda: {
        "lonely": 0.25, "bodypart": 0.25, "unverified": 0.15, "oversized": 0.12,
        "weak": 0.08, "short": 0.05, "sparse": 0.05, "static": 0.03,
        "rotated": 0.02})


def components(t: Tracklet, fps: float, verified: Optional[float],
               frame_hw: Optional[tuple[int, int]] = None) -> dict[str, float]:
    hits = t.hits
    src = t.src_arr()
    n_hit = int(hits.sum())
    measured = ~((src & int(Src.INTERP)) != 0)
    n_meas = max(int(measured.sum()), 1)
    # Span, not hit count: at an analysis stride most frames are interpolated
    # and a 6 s track would otherwise read as 0.5 s of "hits".
    dur = len(t.boxes) / max(fps, 1e-6)
    c: dict[str, float] = {}
    c["short"] = float(np.clip((0.5 - dur) / 0.5, 0.0, 1.0))
    c["sparse"] = float(np.clip(1.0 - n_hit / n_meas, 0.0, 1.0))
    if n_hit:
        top = np.sort(t.scores[hits])[-5:]
        c["weak"] = float(np.clip(1.0 - top.mean(), 0.0, 1.0))
        agree = ((src[hits] & int(Src.MULTI | Src.FACE_IN_HEAD)) != 0)
        c["lonely"] = float(1.0 - agree.mean())
        rot = ((src[hits] & int(Src.ROTATED)) != 0) & ~agree
        c["rotated"] = float(rot.mean())
        c["bodypart"] = float(((src[hits] & int(Src.BODYPART)) != 0).mean())
    else:
        c["weak"] = 1.0
        c["lonely"] = 1.0
        c["rotated"] = 0.0
        c["bodypart"] = 0.0
    boxes = t.boxes
    if len(boxes) > 1:
        diag = float(np.median(np.hypot(boxes[:, 2], boxes[:, 3]))) or 1.0
        step = np.linalg.norm(np.diff(boxes[:, :2], axis=0), axis=1) / diag
        motion = float(np.median(step))
        size_cv = float(np.std(boxes[:, 2]) / max(float(np.mean(boxes[:, 2])),
                                                  1e-3))
        c["static"] = float(np.clip(1.0 - motion / 0.01, 0, 1)
                            * np.clip(1.0 - size_cv / 0.10, 0, 1))
    else:
        c["static"] = 1.0
    if verified is not None:
        c["unverified"] = float(np.clip(1.0 - verified, 0.0, 1.0))
    if frame_hw and frame_hw[0] > 0 and frame_hw[1] > 0 and len(boxes):
        frac = float(np.median(boxes[:, 2] * boxes[:, 3])) / float(
            frame_hw[0] * frame_hw[1])
        # Real heads reach 40 % of the frame in close-ups on this kind of
        # footage; only beyond a quarter of the frame does size start to
        # suggest a torso or a bed.
        c["oversized"] = float(np.clip((frac - 0.25) / 0.25, 0.0, 1.0))
    return c


def suspicion(c: dict[str, float], cfg: ScoreConfig) -> float:
    w = {k: v for k, v in cfg.weights.items() if k in c}
    tot = sum(w.values()) or 1.0
    return float(sum(c[k] * v for k, v in w.items()) / tot)


def verify_ratio(
    t: Tracklet,
    frame_at: Callable[[int], "np.ndarray | None"],
    detect_fn: Callable[[np.ndarray], np.ndarray],
    cfg: ScoreConfig,
) -> Optional[float]:
    """Fraction of sampled hit frames on which an independent detector
    re-finds this head in a magnified crop (the crop makes small heads big
    enough for any detector). ``None`` when nothing could be tested."""
    idx = np.nonzero(t.hits)[0]
    if len(idx) == 0:
        return None
    pick = np.unique(idx[np.linspace(0, len(idx) - 1,
                                     min(cfg.verify_samples, len(idx)))
                         .round().astype(int)])
    xyxy = rows_to_xyxy(t.boxes)
    tested = found = 0
    for k in pick:
        frame = frame_at(t.start + int(k))
        if frame is None:
            continue
        fh, fw = frame.shape[:2]
        b = xyxy[k]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        side = max(b[2] - b[0], b[3] - b[1]) * cfg.verify_margin
        side = max(side, cfg.verify_min_crop)
        x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x2, y2 = int(min(fw, cx + side / 2)), int(min(fh, cy + side / 2))
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        crop = frame[y1:y2, x1:x2]
        det = np.asarray(detect_fn(crop), np.float32).reshape(-1, 5)
        tested += 1
        if len(det) == 0:
            continue
        local = np.array([[b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1]],
                         np.float32)
        local = clip_boxes(local, x2 - x1, y2 - y1)
        ious = iou_matrix(local, det[:, :4])[0]
        dcx = (det[:, 0] + det[:, 2]) / 2
        dcy = (det[:, 1] + det[:, 3]) / 2
        inside = ((local[0, 0] <= dcx) & (dcx <= local[0, 2])
                  & (local[0, 1] <= dcy) & (dcy <= local[0, 3]))
        # A re-detection inside the box only counts if it is comparable in
        # size — a real head found inside a torso-sized box must not verify
        # the torso.
        larea = float((local[0, 2] - local[0, 0]) * (local[0, 3] - local[0, 1]))
        darea = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        if (ious >= cfg.verify_match_iou).any() or \
                (inside & (darea >= 0.3 * larea)).any():
            found += 1
    if tested == 0:
        return None
    return found / tested


def score_tracks(
    kept: list[Tracklet], rejected: list[Tracklet], fps: float,
    cfg: ScoreConfig = ScoreConfig(),
    verifier: Optional[Callable[[Tracklet], Optional[float]]] = None,
    identities: Optional[dict] = None,
    reject_reasons: Optional[dict] = None,
    frame_hw: Optional[tuple[int, int]] = None,
) -> list[Track]:
    """Wrap every refined tracklet as a :class:`Track` with its suspicion.
    Kept tracklets default to enabled unless their suspicion clears
    ``auto_disable_above``; rejected ones default to disabled and carry the
    prune reason."""
    out: list[Track] = []
    for t in kept:
        v = verifier(t) if verifier is not None else None
        c = components(t, fps, v, frame_hw)
        s = suspicion(c, cfg)
        out.append(Track(t=t, suspicion=s, reasons=c, verified=v,
                         identity=(identities or {}).get(t.tid),
                         kept=s < cfg.auto_disable_above))
    for t in rejected:
        c = components(t, fps, None, frame_hw)
        c["pruned"] = 1.0
        s = max(suspicion(c, cfg), 0.9)
        tr = Track(t=t, suspicion=s, reasons=c, verified=None, kept=False)
        if reject_reasons and t.tid in reject_reasons:
            tr.reasons["why"] = reject_reasons[t.tid]
        out.append(tr)
    return out
