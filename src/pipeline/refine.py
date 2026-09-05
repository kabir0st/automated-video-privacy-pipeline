"""Offline refinement: raw step-space tracklets → frame-space tracklets.

Runs after pass 1 with the whole timeline in hand, so it can do what an
online tracker cannot: look ahead. Stages, in order:

  1. TRIM      cut coasted heads/tails (no measurement there).
  2. PRUNE     *soft* — only tracklets with essentially no temporal support
               are set aside, and even those are returned (``rejected``) so
               review can re-enable them. Everything else is kept; the
               suspicion score (pipeline/score.py) ranks it for review.
  3. IDENTIFY  optional appearance vectors (ArcFace) for the survivors.
  4. BRIDGE    join tracklets across gaps up to ``bridge_gap_s`` when the
               geometry says "same head" (velocity-extrapolated landing
               point, size ratio, no other head in the corridor, no
               near-equal competing join) and appearance does not veto.
  5. FILL      interpolate the face channel across short face-less runs.
  6. EXTEND    hold the first/last box ``extend_s`` beyond the measured span
               (detector spin-up; blurring early/late is safe).
  7. SMOOTH    zero-phase Savitzky–Golay on every box channel.
  8. UPSAMPLE  step space → frame space (linear); interpolated frames are
               flagged ``Src.INTERP`` and never count as hits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.signal import savgol_filter

from .types import FACE_SOURCES, Src, Tracklet

FACE_EVIDENCE = int(FACE_SOURCES | Src.FACE_IN_HEAD)


@dataclass(frozen=True)
class RefineConfig:
    min_hits: int = 3              # below this AND no face evidence → rejected
    min_track_s: float = 0.20
    min_hit_ratio: float = 0.20    # measured / span after trim
    min_top_score: float = 0.20    # mean of the 3 best hit scores
    bridge_gap_s: float = 2.5
    bridge_size_ratio: tuple[float, float] = (0.5, 2.0)
    corridor_iou: float = 0.45
    ambiguity_margin: float = 0.10
    gap_pad: float = 0.20
    extend_s: float = 0.30
    smooth_win_s: float = 0.30
    face_fill_s: float = 2.0
    identity_samples: int = 7


# ── 1. trim ──────────────────────────────────────────────────────────────────

def trim(t: Tracklet) -> Optional[Tracklet]:
    idx = np.nonzero(t.hits)[0]
    if len(idx) == 0:
        return None
    return t.slice(int(idx[0]), int(idx[-1]) + 1)


# ── 2. prune (soft) ──────────────────────────────────────────────────────────

def has_face_evidence(t: Tracklet) -> bool:
    return bool((t.src_arr()[t.hits] & FACE_EVIDENCE).any())


def prune(tracklets: list[Tracklet], cfg: RefineConfig, fps: float,
          ) -> tuple[list[Tracklet], list[Tracklet], dict[int, str]]:
    """→ ``(kept, rejected, reason_by_tid)``. A tracklet is rejected only
    when it has essentially no temporal support."""
    kept, rejected, why = [], [], {}
    for t in tracklets:
        n_hit = int(t.hits.sum())
        dur = len(t.boxes) / max(fps, 1e-6)
        top = float(np.sort(t.scores[t.hits])[-3:].mean()) if n_hit else 0.0
        ratio = n_hit / max(len(t.boxes), 1)
        face = has_face_evidence(t)
        reason = None
        if n_hit < 2:
            reason = "single measurement"
        elif dur < cfg.min_track_s and n_hit < cfg.min_hits and not face:
            reason = f"too short ({dur:.2f}s, {n_hit} hits, no face)"
        elif ratio < cfg.min_hit_ratio:
            reason = f"sparse ({ratio:.0%} measured)"
        elif top < cfg.min_top_score:
            reason = f"weak (top score {top:.2f})"
        if reason is None:
            kept.append(t)
        else:
            rejected.append(t)
            why[t.tid] = reason
    return kept, rejected, why


# ── 3. identify ──────────────────────────────────────────────────────────────

def identity_vectors(
    tracklets: list[Tracklet],
    frame_at: Callable[[int], "np.ndarray | None"],
    embed_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    samples: int = 7,
) -> dict[int, np.ndarray]:
    """One L2-normalised mean embedding per tracklet from up to ``samples``
    evenly spaced *face-evidence* hit frames (a head seen from behind has no
    identity to embed). Tracklets with no usable crop are simply absent."""
    from .geom import rows_to_xyxy

    out: dict[int, np.ndarray] = {}
    for t in tracklets:
        src = t.src_arr()
        good = np.nonzero(t.hits & ((src & FACE_EVIDENCE) != 0))[0]
        if len(good) == 0:
            continue
        pick = good[np.linspace(0, len(good) - 1, min(samples, len(good)))
                    .round().astype(int)]
        fb = rows_to_xyxy(t.fb())
        acc = None
        n = 0
        for k in np.unique(pick):
            frame = frame_at(t.start + int(k))
            if frame is None:
                continue
            v = np.asarray(embed_fn(frame, fb[k:k + 1]), np.float32).reshape(-1)
            if v.size == 0 or not v.any():
                continue
            acc = v if acc is None else acc + v
            n += 1
        if acc is not None and n:
            norm = float(np.linalg.norm(acc))
            if norm > 0:
                out[t.tid] = acc / norm
    return out


# ── 4. bridge ────────────────────────────────────────────────────────────────

def _tail_velocity(t: Tracklet, n: int = 5) -> np.ndarray:
    k = min(n, len(t.boxes) - 1)
    if k < 1:
        return np.zeros(2, np.float32)
    return (t.boxes[-1, :2] - t.boxes[-1 - k, :2]) / k


def _iou_cxcywh_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    inter = iw * ih
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-9)


def _corridor_blocked(tracklets: list[Tracklet], i: int, j: int,
                      corridor_iou: float) -> bool:
    """Another surviving tracklet occupies the i→j gap path (two tangled
    heads must not be smeared into one blur path)."""
    a, b = tracklets[i], tracklets[j]
    g = b.start - a.end - 1
    if g <= 0:
        return False
    f0, f1 = a.end + 1, b.start - 1
    frames = np.arange(f0, f1 + 1, dtype=np.float32)
    w = ((frames - a.end) / (g + 1)).astype(np.float32)[:, None]
    path = (1.0 - w) * a.boxes[-1][None, :] + w * b.boxes[0][None, :]
    for k, c in enumerate(tracklets):
        if k in (i, j):
            continue
        lo, hi = max(c.start, f0), min(c.end, f1)
        if lo > hi:
            continue
        cb = c.boxes[lo - c.start:hi - c.start + 1]
        seg = path[lo - f0:hi - f0 + 1]
        if _iou_cxcywh_many(seg, cb).max() > corridor_iou:
            return True
    return False


def _identity_conflict(tracklets: list[Tracklet], i: int, j: int,
                       identities: Optional[dict]) -> bool:
    """Appearance veto only: low similarity kills a geometric join; high
    similarity never buys one. Missing vectors mean "no opinion"."""
    if not identities:
        return False
    a = identities.get(tracklets[i].tid)
    b = identities.get(tracklets[j].tid)
    if a is None or b is None:
        return False
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    if a.shape != b.shape or not a.any() or not b.any():
        return False
    from libs.embed import DIFF_ID_COS
    return float(a @ b) < DIFF_ID_COS


def _bridge_candidates(tracklets: list[Tracklet], gap_max: int,
                       cfg: RefineConfig, identities: Optional[dict],
                       ) -> list[tuple[float, int, int]]:
    n = len(tracklets)
    if n < 2:
        return []
    ends = np.array([t.end for t in tracklets], np.int64)
    starts = np.array([t.start for t in tracklets], np.int64)
    last = np.stack([t.boxes[-1] for t in tracklets]).astype(np.float32)
    first = np.stack([t.boxes[0] for t in tracklets]).astype(np.float32)
    vel = np.stack([_tail_velocity(t) for t in tracklets]).astype(np.float32)
    diag = np.hypot(last[:, 2], last[:, 3]).astype(np.float32)
    diag[diag == 0.0] = 1.0

    gap = starts[None, :] - ends[:, None] - 1
    ok = (gap >= 0) & (gap <= gap_max)
    np.fill_diagonal(ok, False)
    if not ok.any():
        return []
    steps = (gap + 1).astype(np.float32)
    pred = last[:, None, :2] + vel[:, None, :] * steps[:, :, None]
    dist = np.hypot(pred[..., 0] - first[None, :, 0],
                    pred[..., 1] - first[None, :, 1])
    # Allowance widens with the gap: uncertainty grows, and a head that
    # vanished for a second may have moved a long way.
    allow = diag[:, None] * (0.75 + 1.0 * gap / max(gap_max, 1))
    ok &= dist <= allow
    lo, hi = cfg.bridge_size_ratio
    rw = first[None, :, 2] / np.maximum(last[:, None, 2], 1e-3)
    rh = first[None, :, 3] / np.maximum(last[:, None, 3], 1e-3)
    ok &= (rw >= lo) & (rw <= hi) & (rh >= lo) & (rh <= hi)

    cands: list[tuple[float, int, int]] = []
    for i, j in zip(*np.nonzero(ok)):
        i, j = int(i), int(j)
        if _identity_conflict(tracklets, i, j, identities):
            continue
        if _corridor_blocked(tracklets, i, j, cfg.corridor_iou):
            continue
        cands.append((float(dist[i, j] / diag[i] + 0.05 * gap[i, j]), i, j))
    cands.sort()
    return cands


def _merge_pair(a: Tracklet, b: Tracklet, gap_pad: float) -> Tracklet:
    g = b.start - a.end - 1
    gap_boxes = np.empty((g, 4), np.float32)
    gap_fboxes = np.empty((g, 4), np.float32)
    fa, fb = a.fb(), b.fb()
    for f in range(g):
        w = (f + 1) / (g + 1)
        pad = 1.0 + gap_pad * np.sin(np.pi * (f + 1) / (g + 1))
        gap_boxes[f] = (1 - w) * a.boxes[-1] + w * b.boxes[0]
        gap_boxes[f, 2:] *= pad
        gap_fboxes[f] = (1 - w) * fa[-1] + w * fb[0]
        gap_fboxes[f, 2:] *= pad
    return Tracklet(
        a.tid, a.start,
        np.concatenate([a.boxes, gap_boxes, b.boxes]),
        np.concatenate([a.scores, np.zeros(g, np.float32), b.scores]),
        np.concatenate([a.hits, np.zeros(g, bool), b.hits]),
        np.concatenate([fa, gap_fboxes, fb]),
        np.concatenate([a.src_arr(), np.full(g, int(Src.INTERP), np.uint32),
                        b.src_arr()]),
        np.concatenate([a.fvalid_arr(), np.zeros(g, bool), b.fvalid_arr()]))


def bridge(tracklets: list[Tracklet], gap_max: int, cfg: RefineConfig,
           identities: Optional[dict] = None) -> list[Tracklet]:
    """Merge across gaps; chains join over repeated rounds."""
    tracklets = list(tracklets)
    while True:
        cands = _bridge_candidates(tracklets, gap_max, cfg, identities)
        if not cands:
            return tracklets
        merged: dict[int, Tracklet] = {}
        consumed: set[int] = set()
        used: set[int] = set()
        for cost, i, j in cands:
            if i in used or j in used:
                continue
            near = [c for c, x, y in cands
                    if (x, y) != (i, j) and (x == i or y == j)
                    and x not in used and y not in used
                    and c <= cost * (1 + cfg.ambiguity_margin)]
            if near:
                used.add(i)
                used.add(j)
                continue
            merged[i] = _merge_pair(tracklets[i], tracklets[j], cfg.gap_pad)
            consumed.add(j)
            used.add(i)
            used.add(j)
        if not merged:
            return tracklets
        tracklets = [merged.get(k, t) for k, t in enumerate(tracklets)
                     if k not in consumed]


# ── 5. face fill ─────────────────────────────────────────────────────────────

def fill_face_gaps(t: Tracklet, max_gap: int) -> Tracklet:
    fvalid = t.fvalid_arr().copy()
    fboxes = t.fb().copy()
    n = len(fvalid)
    i = 0
    while i < n:
        if fvalid[i]:
            i += 1
            continue
        j = i
        while j < n and not fvalid[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n:
            a, b = fboxes[i - 1], fboxes[j]
            for k in range(gap_len):
                w = (k + 1) / (gap_len + 1)
                fboxes[i + k] = (1 - w) * a + w * b
                fvalid[i + k] = True
        i = j
    return Tracklet(t.tid, t.start, t.boxes, t.scores, t.hits, fboxes,
                    t.src_arr(), fvalid)


# ── 6–8. extend, smooth, upsample ────────────────────────────────────────────

def extend(t: Tracklet, ext: int, n_frames: int) -> Tracklet:
    pre = min(ext, t.start)
    post = min(ext, max(0, n_frames - 1 - t.end))
    if pre == 0 and post == 0:
        return t
    fb, src, fv = t.fb(), t.src_arr(), t.fvalid_arr()
    rep = lambda a, k, edge: np.repeat(a[edge:edge + 1] if edge == 0
                                       else a[-1:], k, axis=0)
    return Tracklet(
        t.tid, t.start - pre,
        np.concatenate([rep(t.boxes, pre, 0), t.boxes, rep(t.boxes, post, -1)]),
        np.concatenate([np.zeros(pre, np.float32), t.scores,
                        np.zeros(post, np.float32)]),
        np.concatenate([np.zeros(pre, bool), t.hits, np.zeros(post, bool)]),
        np.concatenate([rep(fb, pre, 0), fb, rep(fb, post, -1)]),
        np.concatenate([np.full(pre, int(Src.INTERP), np.uint32), src,
                        np.full(post, int(Src.INTERP), np.uint32)]),
        np.concatenate([rep(fv, pre, 0), fv, rep(fv, post, -1)]))


def smooth(t: Tracklet, win: int) -> Tracklet:
    n = len(t.boxes)
    if n < 5:
        return t
    w = min(win | 1, n if n % 2 else n - 1)
    if w < 5:
        return t
    boxes = t.boxes.copy()
    fboxes = t.fb().copy()
    for ch in range(4):
        boxes[:, ch] = savgol_filter(boxes[:, ch], w, polyorder=2, mode="interp")
        fboxes[:, ch] = savgol_filter(fboxes[:, ch], w, polyorder=2,
                                      mode="interp")
    return Tracklet(t.tid, t.start, boxes, t.scores, t.hits, fboxes,
                    t.src_arr(), t.fvalid_arr())


def upsample(t: Tracklet, stride: int, n_frames: int) -> Tracklet:
    if stride <= 1:
        return t
    n = len(t.boxes)
    start = t.start * stride
    if n < 2:
        return Tracklet(t.tid, start, t.boxes, t.scores, t.hits, t.fb(),
                        t.src_arr(), t.fvalid_arr())
    src_idx = np.arange(n, dtype=np.float64) * stride
    span = min(int(src_idx[-1]), max(0, n_frames - 1 - start))
    dst = np.arange(span + 1, dtype=np.float64)
    fb, src, fv = t.fb(), t.src_arr(), t.fvalid_arr()
    boxes = np.stack([np.interp(dst, src_idx, t.boxes[:, c]) for c in range(4)],
                     axis=1).astype(np.float32)
    fboxes = np.stack([np.interp(dst, src_idx, fb[:, c]) for c in range(4)],
                      axis=1).astype(np.float32)
    scores = np.interp(dst, src_idx, t.scores).astype(np.float32)
    step = np.clip((dst / stride).astype(np.int64), 0, n - 1)
    nxt = np.clip(step + 1, 0, n - 1)
    exact = (dst % stride) == 0
    hits = np.where(exact, t.hits[step], False)
    fvalid = np.where(exact, fv[step], fv[step] & fv[nxt])
    src_out = np.where(exact, src[step], int(Src.INTERP)).astype(np.uint32)
    return Tracklet(t.tid, start, boxes, scores, hits, fboxes, src_out,
                    fvalid.astype(bool))


# ── driver ───────────────────────────────────────────────────────────────────

def refine(
    tracklets: list[Tracklet], *, fps: float, n_frames: int, stride: int = 1,
    cfg: RefineConfig = RefineConfig(),
    identify: Optional[Callable[[list[Tracklet]], dict]] = None,
) -> tuple[list[Tracklet], list[Tracklet], dict[int, np.ndarray], dict[int, str]]:
    """``(kept, rejected, identities, reject_reasons)`` — all tracklets in
    frame space. ``fps``/``n_frames`` are in step space (``fps/stride``,
    number of analysed steps), as recorded."""
    fps = max(fps, 1.0)
    trimmed = [t for t in (trim(t) for t in tracklets) if t is not None]
    kept, rejected, why = prune(trimmed, cfg, fps)
    identities = identify(kept) if (identify is not None and kept) else {}
    kept = bridge(kept, int(round(cfg.bridge_gap_s * fps)), cfg, identities)
    fill = max(0, int(round(cfg.face_fill_s * fps)))
    ext = max(0, int(round(cfg.extend_s * fps)))
    win = max(5, int(round(cfg.smooth_win_s * fps)) | 1)
    total_frames = n_frames * stride

    def finish(t: Tracklet) -> Tracklet:
        t = fill_face_gaps(t, fill)
        t = extend(t, ext, n_frames)
        t = smooth(t, win)
        return upsample(t, stride, total_frames)

    return ([finish(t) for t in kept], [finish(t) for t in rejected],
            identities, why)
