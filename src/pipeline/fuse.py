"""Per-frame union of every detector's output into head candidates.

The principle is the opposite of a consensus gate: any single source may
propose a head; agreement between sources *raises* the candidate's score
and disagreement never vetoes it. The only per-frame sanity rule is for the
one failure mode a rotated pass has that nothing downstream can fix — a
frame-spanning hallucinated "head" — and even that is waived when a second
source agrees. Everything else (skin misreads, one-frame flashes) is left to
the tracker's spawn bar and the offline temporal prune, which see time.

Sources (all ``(K, 5)`` xyxy+score in frame coords, already NMS'd within a
source):

* ``headdet``  the dedicated head detector, upright pass
* ``wb_head``  Wholebody17 head class, upright pass
* ``wb_face``  Wholebody17 face class, upright pass
* ``scrfd``    SCRFD faces that passed the landmark plausibility check
* ``*_rot``    the same source from rotated passes, boxes already unrotated

Heads define geometry; faces corroborate. A face whose centre lies inside a
head-sized head cluster attaches to it (source bit, ``FACE_IN_HEAD``,
agreement bonus) without changing its box — in close-ups a face box grown
to "head scale" is larger than the head itself and used to spawn
torso-sized duplicates. A face no head cluster covers becomes its own
head-sized claim via ``geom.grow_to_head``. All raw faces are also kept as
a flat pool for the tracker's face memory (face-tight blur mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import centres_inside, grow_to_head, iou_matrix
from .types import Src

_SOURCE_BITS = {"headdet": Src.HEADDET, "wb_head": Src.WB_HEAD,
                "wb_face": Src.WB_FACE, "scrfd": Src.SCRFD}
_BIT_NAME = {int(v): k for k, v in _SOURCE_BITS.items()}
# Independence is per *model*, not per class: Wholebody's head and face
# classes come from one network and one mistake, so they are one family.
_FAMILY = {"headdet": "headdet", "wb_head": "wb", "wb_face": "wb", "scrfd": "scrfd"}
_BIT_FAMILY = {int(_SOURCE_BITS[k]): v for k, v in _FAMILY.items()}
_FACE_SOURCES = ("wb_face", "scrfd")


@dataclass(frozen=True)
class FuseConfig:
    merge_iou: float = 0.45       # two claims describe the same head
    # A claim whose centre lies inside a higher-scored claim also merges when
    # their areas are within this ratio (a face-grown head inside a head).
    # Kept tight so a torso-sized box can never be absorbed into a head
    # cluster and inflate the fused box.
    contain_area_ratio: tuple[float, float] = (0.45, 1.8)
    # A face inside a box validates the box only while the box is head-sized
    # relative to the face (a head is ~3.2× a face). Beyond this ratio the
    # face says nothing about the box — a torso contains the face too.
    face_validates_ratio: float = 8.0
    # Trust multiplier for a *single-source* claim, by source. Measured on the
    # reference clip: Wholebody's unsupported boxes outnumber the head
    # detector's twelve to one.
    source_trust: dict = field(default_factory=lambda: {
        "headdet": 1.0, "wb_head": 0.7, "wb_face": 0.9, "scrfd": 1.0})
    # When head detectors over-box (head + shoulders) but a validating face
    # is attached, pull the geometry halfway toward the face-grown head once
    # the cluster exceeds this multiple of the grown face's area.
    face_anchor_ratio: float = 1.5
    # Body-part veto (source ``bodypart``: NudeNet's non-face classes). A
    # candidate that mostly coincides with an exposed/covered body part is
    # not a head, however many head detectors agree — on this footage all
    # three fire together on a harness-covered groin. Down-weighted below
    # the spawn bar rather than dropped, and flagged for the scorer.
    bodypart_iou: float = 0.45
    bodypart_cover: float = 0.70    # candidate area inside a body part
    bodypart_inside: float = 0.80   # body part (≥10 % of candidate) inside candidate
    # Multiplier for a candidate coinciding with a body part. 1.0 = flag only:
    # measured on the reference clip, suppressing spawn cost three points of
    # head coverage (ten real heads beside breasts, bellies and feet) for a
    # dozen fewer unsupported boxes — the wrong trade for a privacy tool. The
    # flag feeds the ``bodypart`` suspicion component for review; the Strict
    # preset sets 0.3 to actually veto.
    bodypart_weight: float = 1.0
    # A candidate corroborated by a face from another model family is a head
    # whatever NudeNet says nearby (heads sit beside breasts and armpits on
    # this footage); the veto applies to the uncorroborated ones.
    veto_exempt_face: bool = True
    # Same-network head+face agreement (Wholebody's two classes) is weaker
    # than a second model but not nothing: a small bonus, and no source-trust
    # penalty, without counting as an extra family.
    intra_bonus: float = 0.07
    agree_bonus: float = 0.15     # per additional independent source
    face_weight: float = 0.90     # a face-grown claim's score vs its face score
    # Rotated-only, single-source head bigger than this fraction of the frame
    # area is a hallucination (a rotated bed reads as one giant head).
    giant_frac: float = 0.30
    min_side_px: float = 8.0
    # Single-source claims (no second head model, no face inside) are
    # trusted less when they look unlike a head. Down-weighted, not dropped:
    # they may still sustain a track, just not start one. Measured on the
    # reference clip: real heads are *large* here (median 10 % of the frame,
    # 95th percentile 40 %), so area separates little; aspect ratio does
    # (unsupported boxes run to 3:1 and beyond, real heads stay under ~2.2),
    # and 85 % of unsupported boxes came from rotated passes.
    oversize_frac: float = 0.30
    oversize_weight: float = 0.5
    aspect_range: tuple[float, float] = (0.45, 2.2)
    aspect_weight: float = 0.5
    rotated_single_weight: float = 0.7


@dataclass
class FrameCands:
    heads: np.ndarray             # (N, 5) xyxy + fused score
    flags: np.ndarray             # (N,) int64 Src bits
    faces: np.ndarray             # (M, 5) xyxy + score — raw face pool

    @staticmethod
    def empty() -> "FrameCands":
        return FrameCands(np.empty((0, 5), np.float32),
                          np.empty(0, np.int64), np.empty((0, 5), np.float32))


_HEAD_SOURCES = ("headdet", "wb_head")


def _split(sources: dict[str, np.ndarray]):
    """→ head claims ``(boxes, bits, rot)`` and face claims
    ``(boxes, bits, rot)`` (faces un-grown), plus the flat raw face pool."""
    hb, hbit, hrot, fb, fbit, frot = [], [], [], [], [], []
    for name, arr in sources.items():
        arr = np.asarray(arr, np.float32).reshape(-1, 5)
        if len(arr) == 0:
            continue
        base = name[:-4] if name.endswith("_rot") else name
        bit = _SOURCE_BITS.get(base)
        if bit is None:
            continue
        is_rot = name.endswith("_rot")
        tgt = (hb, hbit, hrot) if base in _HEAD_SOURCES else (fb, fbit, frot)
        tgt[0].append(arr)
        tgt[1].append(np.full(len(arr), int(bit), np.int64))
        tgt[2].append(np.full(len(arr), is_rot, bool))
    e5 = np.empty((0, 5), np.float32)
    cat = lambda xs, dt, shape: (np.concatenate(xs) if xs else np.empty(shape, dt))  # noqa: E731
    return (cat(hb, np.float32, (0, 5)), cat(hbit, np.int64, (0,)), cat(hrot, bool, (0,)),
            cat(fb, np.float32, (0, 5)), cat(fbit, np.int64, (0,)), cat(frot, bool, (0,)))


def _cluster(boxes: np.ndarray, cfg: FuseConfig) -> list[np.ndarray]:
    """Greedy score-ordered clustering → list of member index arrays (into
    ``boxes`` sorted by score, which the caller must apply)."""
    if len(boxes) == 0:
        return []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ious = iou_matrix(boxes, boxes)
    inside = centres_inside(boxes, boxes)
    ratio = area[:, None] / np.maximum(area[None, :], 1e-6)
    lo, hi = cfg.contain_area_ratio
    same = (ious >= cfg.merge_iou) | ((inside | inside.T) & (ratio >= lo) & (ratio <= hi))
    used = np.zeros(len(boxes), bool)
    out = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        mem = np.nonzero(same[i] & ~used)[0]
        used[mem] = True
        out.append(mem)
    return out


def fuse(sources: dict[str, np.ndarray], frame_hw: tuple[int, int],
         cfg: FuseConfig = FuseConfig()) -> FrameCands:
    """Union-fuse one frame's detector outputs → :class:`FrameCands`.

    1. Head claims cluster greedily in score order (IoU ≥ ``merge_iou`` or
       centre containment within a tight area ratio); each cluster's box is
       the score-weighted mean of its members.
    2. Every face attaches to the head cluster that contains its centre and
       is at most ``face_validates_ratio`` × its area — corroboration only.
    3. Faces nothing covers become head-sized claims and cluster among
       themselves.
    4. Scores: best member (faces × ``face_weight``) plus ``agree_bonus`` per
       extra independent *model family*, capped below 1; a candidate that
       coincides with a NudeNet body-part box is scaled by
       ``bodypart_weight`` and flagged; single-family claims with no
       cross-family face are scaled by source trust and by how little they
       look like a head."""
    hbox, hbit, hrot, fbox, fbit, frot = _split(sources)
    fh, fw = frame_hw
    frame_area = float(fw * fh)
    faces_pool = fbox.copy()
    side_ok = lambda b: ((b[:, 2] - b[:, 0] >= cfg.min_side_px)  # noqa: E731
                         & (b[:, 3] - b[:, 1] >= cfg.min_side_px))
    if len(hbox):
        m = side_ok(hbox)
        hbox, hbit, hrot = hbox[m], hbit[m], hrot[m]
    if len(fbox):
        m = side_ok(fbox)
        fbox, fbit, frot = fbox[m], fbit[m], frot[m]

    clusters: list[dict] = []
    if len(hbox):
        order = np.argsort(-hbox[:, 4])
        hbox, hbit, hrot = hbox[order], hbit[order], hrot[order]
        for mem in _cluster(hbox, cfg):
            w = hbox[mem, 4]
            box = (hbox[mem, :4] * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
            clusters.append({"box": box, "score": float(w.max()),
                             "bits": {int(b) for b in hbit[mem]},
                             "fams": {_BIT_FAMILY[int(b)] for b in hbit[mem]},
                             "rot": bool(hrot[mem].all()), "faces": [],
                             "lead": int(hbit[mem][0])})

    # 2. attach faces to covering, head-sized clusters
    orphan_idx = []
    if len(fbox):
        cboxes = (np.stack([c["box"] for c in clusters]) if clusters
                  else np.empty((0, 4), np.float32))
        carea = ((cboxes[:, 2] - cboxes[:, 0]) * (cboxes[:, 3] - cboxes[:, 1])
                 if len(cboxes) else np.empty(0, np.float32))
        grown_all = np.stack([grow_to_head(f) for f in fbox])
        for k in range(len(fbox)):
            f = fbox[k]
            farea = (f[2] - f[0]) * (f[3] - f[1])
            if len(cboxes):
                cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
                inside = ((cboxes[:, 0] <= cx) & (cx <= cboxes[:, 2])
                          & (cboxes[:, 1] <= cy) & (cy <= cboxes[:, 3]))
                sized = carea <= cfg.face_validates_ratio * max(farea, 1.0)
                ok = np.nonzero(inside & sized)[0]
                if len(ok):
                    ious = iou_matrix(grown_all[k:k + 1, :4], cboxes[ok])[0]
                    c = clusters[int(ok[np.argmax(ious)])]
                    c["faces"].append((grown_all[k], float(f[4]), int(fbit[k]),
                                       bool(frot[k])))
                    continue
            orphan_idx.append(k)

    # 3. orphan faces → head-sized claims, clustered among themselves
    if orphan_idx:
        ob = np.stack([grown_all[k] for k in orphan_idx]).astype(np.float32)
        obit = fbit[orphan_idx]
        orot = frot[orphan_idx]
        order = np.argsort(-ob[:, 4])
        ob, obit, orot = ob[order], obit[order], orot[order]
        for mem in _cluster(ob, cfg):
            w = ob[mem, 4]
            box = (ob[mem, :4] * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
            clusters.append({"box": box, "score": float(w.max()) * cfg.face_weight,
                             "bits": {int(b) for b in obit[mem]},
                             "fams": {_BIT_FAMILY[int(b)] for b in obit[mem]},
                             "rot": bool(orot[mem].all()), "faces": [],
                             "lead": int(obit[mem][0]), "face_only": True})

    # body-part boxes (NudeNet non-face classes), if that source ran
    bp = np.asarray(sources.get("bodypart", np.empty((0, 5), np.float32)),
                    np.float32).reshape(-1, 5)

    # 4. finalise
    out_boxes, out_flags = [], []
    for c in clusters:
        box = c["box"].astype(np.float32)
        bits = set(c["bits"])
        fams = set(c["fams"])
        score = c["score"]
        all_rot = c["rot"]
        face_in = False
        same_fam_face = False
        if c["faces"]:
            head_fams = set(c["fams"])
            # a face corroborates only when it comes from another model
            face_in = any(_BIT_FAMILY[b] not in head_fams
                          for _g, _s, b, _r in c["faces"])
            same_fam_face = not face_in
            bits |= {b for _g, _s, b, _r in c["faces"]}
            fams |= {_BIT_FAMILY[b] for _g, _s, b, _r in c["faces"]}
            score = max(score, max(s for _g, s, _b, _r in c["faces"]) * cfg.face_weight)
            all_rot = all_rot and all(r for _g, _s, _b, r in c["faces"])
            # face-anchored shrink when head detectors over-box
            grown = np.stack([g[:4] for g, _s, _b, _r in c["faces"]]).mean(0)
            garea = (grown[2] - grown[0]) * (grown[3] - grown[1])
            barea = (box[2] - box[0]) * (box[3] - box[1])
            if barea > cfg.face_anchor_ratio * garea:
                box = (0.5 * (box + grown)).astype(np.float32)
        n_src = len(fams)
        score = min(0.99, score + cfg.agree_bonus * (n_src - 1)
                    + (cfg.intra_bonus if same_fam_face else 0.0))
        flags = 0
        for b in bits:
            flags |= b
        if n_src >= 2:
            flags |= int(Src.MULTI)
        if face_in or c.get("face_only"):
            flags |= int(Src.FACE_IN_HEAD)
        bw, bh = box[2] - box[0], box[3] - box[1]
        barea = bw * bh
        if all_rot:
            flags |= int(Src.ROTATED)
            if n_src < 2 and barea > cfg.giant_frac * frame_area:
                continue
        if len(bp) and not (cfg.veto_exempt_face and face_in):
            ious = iou_matrix(box[None, :], bp[:, :4])[0]
            ix1 = np.maximum(box[0], bp[:, 0]); iy1 = np.maximum(box[1], bp[:, 1])
            ix2 = np.minimum(box[2], bp[:, 2]); iy2 = np.minimum(box[3], bp[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            bparea = (bp[:, 2] - bp[:, 0]) * (bp[:, 3] - bp[:, 1])
            cover = inter / max(barea, 1e-6)              # candidate inside part
            inside = inter / np.maximum(bparea, 1e-6)     # part inside candidate
            inside_ok = (inside >= cfg.bodypart_inside) & (bparea >= 0.10 * barea) \
                & (bp[:, 4] >= 0.35)
            if ((ious >= cfg.bodypart_iou).any() or (cover >= cfg.bodypart_cover).any()
                    or inside_ok.any()):
                score *= cfg.bodypart_weight
                flags |= int(Src.BODYPART)
        if n_src < 2 and not face_in:
            if not same_fam_face:
                score *= cfg.source_trust.get(_BIT_NAME.get(c["lead"], ""), 1.0)
            if barea > cfg.oversize_frac * frame_area:
                score *= cfg.oversize_weight
            lo, hi = cfg.aspect_range
            asp = bw / max(bh, 1e-6)
            if asp < lo or asp > hi:
                score *= cfg.aspect_weight
            if all_rot:
                score *= cfg.rotated_single_weight
        out_boxes.append(np.concatenate([box, [score]]).astype(np.float32))
        out_flags.append(flags)
    if not out_boxes:
        return FrameCands(np.empty((0, 5), np.float32), np.empty(0, np.int64),
                          faces_pool)
    return FrameCands(np.stack(out_boxes), np.asarray(out_flags, np.int64),
                      faces_pool)
