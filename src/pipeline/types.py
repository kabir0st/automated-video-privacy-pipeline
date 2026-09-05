"""Data model shared by every stage.

Coordinates: ``Tracklet.boxes``/``fboxes`` are ``cx, cy, w, h`` (they get
filtered, interpolated and smoothed per channel); everything the detectors
and the renderer touch is ``xyxy``. ``pipeline.geom`` converts.

Step space vs frame space: pass 1 records at ``frame // stride``; ``refine``
runs in step space at ``fps / stride`` and its last step upsamples to frame
space, so a ``Track`` is always in frame space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Optional

import numpy as np


class Src(IntFlag):
    """Per-frame provenance bits for a candidate / a tracklet frame."""
    NONE = 0
    HEADDET = 1 << 0        # dedicated head detector (libs/headdet.py)
    WB_HEAD = 1 << 1        # Wholebody17 head class
    WB_FACE = 1 << 2        # Wholebody17 face class (grown to head scale)
    SCRFD = 1 << 3          # SCRFD face, landmark-plausible (grown to head)
    ROTATED = 1 << 4        # only a rotated pass saw it this frame
    MULTI = 1 << 5          # two or more independent sources agreed
    FACE_IN_HEAD = 1 << 6   # a raw face box sits inside this head box
    SUSTAIN = 1 << 7        # matched below the spawn bar (BYTE stage 2)
    INTERP = 1 << 8         # interpolated (bridge / upsample), not measured
    MANUAL = 1 << 9         # user-drawn
    BODYPART = 1 << 10      # overlapped a NudeNet body-part box (vetoed)


HEAD_SOURCES = Src.HEADDET | Src.WB_HEAD
FACE_SOURCES = Src.WB_FACE | Src.SCRFD


@dataclass
class Tracklet:
    """One contiguous run of a track id. Six parallel channels indexed from
    ``start``. In step space until ``refine`` upsamples."""
    tid: int
    start: int
    boxes: np.ndarray                    # (T, 4) cx, cy, w, h — head channel
    scores: np.ndarray                   # (T,)
    hits: np.ndarray                     # (T,) bool — a measurement landed
    fboxes: Optional[np.ndarray] = None  # (T, 4) cxcywh — face channel
    src: Optional[np.ndarray] = None     # (T,) uint32 Src bits
    fvalid: Optional[np.ndarray] = None  # (T,) bool — face measured this frame

    @property
    def end(self) -> int:
        return self.start + len(self.boxes) - 1

    @property
    def n(self) -> int:
        return len(self.boxes)

    def fb(self) -> np.ndarray:
        return self.fboxes if self.fboxes is not None else self.boxes

    def src_arr(self) -> np.ndarray:
        return (self.src if self.src is not None
                else np.zeros(len(self.boxes), np.uint32))

    def fvalid_arr(self) -> np.ndarray:
        return (self.fvalid if self.fvalid is not None
                else np.zeros(len(self.boxes), bool))

    def slice(self, a: int, b: int) -> "Tracklet":
        return Tracklet(self.tid, self.start + a, self.boxes[a:b],
                        self.scores[a:b], self.hits[a:b], self.fb()[a:b],
                        self.src_arr()[a:b], self.fvalid_arr()[a:b])


@dataclass
class Track:
    """A refined tracklet in frame space plus what review needs to know."""
    t: Tracklet
    suspicion: float = 0.0               # 0 = surely a head … 1 = surely not
    reasons: dict = field(default_factory=dict)   # component → 0..1
    verified: Optional[float] = None     # magnified re-detection ratio
    identity: Optional[np.ndarray] = None
    kept: bool = True                    # pipeline's default verdict

    @property
    def tid(self) -> int:
        return self.t.tid


@dataclass
class ManualTrack:
    """A user-drawn head, propagated frame by frame. Dense boxes (xyxy) from
    ``start``; ids are negative so they never collide with tracker ids."""
    tid: int
    start: int
    boxes: np.ndarray                    # (T, 4) xyxy

    @property
    def end(self) -> int:
        return self.start + len(self.boxes) - 1
