"""Pass-1 sink: per-frame :class:`TrackObs` → contiguous :class:`Tracklet`s
(step space). Frames a track is absent from close it; a hole the tracker
skipped (recycled id) is padded with the previous box, ``hit=False``."""
from __future__ import annotations

import numpy as np

from .geom import to_cxcywh
from .types import Tracklet


class TrackRecorder:
    def __init__(self) -> None:
        self._open: dict[int, list] = {}
        self._done: list[Tracklet] = []

    def observe(self, frame_idx: int, obs: list) -> None:
        seen: set[int] = set()
        for o in obs:
            z = to_cxcywh(o.box)
            fz = to_cxcywh(o.face_box) if o.face_box is not None else z.copy()
            fv = bool(o.face_age == 0)
            rec = self._open.get(o.track_id)
            if rec is None:
                self._open[o.track_id] = [frame_idx, [z], [float(o.score)],
                                          [bool(o.hit)], [fz], [int(o.flags)],
                                          [fv]]
            else:
                expect = rec[0] + len(rec[1])
                while expect < frame_idx:
                    rec[1].append(rec[1][-1].copy()); rec[2].append(0.0)
                    rec[3].append(False); rec[4].append(rec[4][-1].copy())
                    rec[5].append(0); rec[6].append(False)
                    expect += 1
                rec[1].append(z); rec[2].append(float(o.score))
                rec[3].append(bool(o.hit)); rec[4].append(fz)
                rec[5].append(int(o.flags)); rec[6].append(fv)
            seen.add(o.track_id)
        for tid in list(self._open):
            if tid not in seen:
                self._close(tid)

    def _close(self, tid: int) -> None:
        start, boxes, scores, hits, fboxes, src, fvalid = self._open.pop(tid)
        self._done.append(Tracklet(
            tid, start,
            np.asarray(boxes, np.float32).reshape(-1, 4),
            np.asarray(scores, np.float32), np.asarray(hits, bool),
            np.asarray(fboxes, np.float32).reshape(-1, 4),
            np.asarray(src, np.uint32), np.asarray(fvalid, bool)))

    def finalize(self) -> list[Tracklet]:
        for tid in list(self._open):
            self._close(tid)
        return self._done
