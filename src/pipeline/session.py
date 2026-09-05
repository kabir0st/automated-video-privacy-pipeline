"""Orchestration shared by the GUI and the CLI: run (or load) pass 1, then
the offline stages, with the frame-access closures the offline stages need.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from . import project as proj
from .analysis import Models, analyse, retrack
from .presets import Preset
from .refine import identity_vectors, refine
from .score import ScoreConfig, score_tracks, verify_ratio
from .types import Track, Tracklet


class FrameReader:
    """Random access by frame index over one ``VideoCapture`` (seek per
    call; fine for the few dozen crops the offline stages need)."""

    def __init__(self, video: str) -> None:
        import cv2
        self._cap = cv2.VideoCapture(video)
        self._ok = self._cap.isOpened()
        self._last = -2

    def __call__(self, idx: int) -> Optional[np.ndarray]:
        import cv2
        if not self._ok:
            return None
        if idx != self._last + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self._cap.read()
        self._last = idx if ok else -2
        return frame if ok else None

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:  # noqa: BLE001
            pass


def make_identify(models: Models, video: str, stride: int, samples: int):
    """``identify(tracklets) -> {tid: vec}`` for refine's bridge veto, or
    ``None`` when the appearance channel is off."""
    from libs.embed import enabled as embed_enabled
    if not embed_enabled():
        return None
    reader = FrameReader(video)
    embedder = models.embed()

    def identify(tracklets: list[Tracklet]) -> dict:
        try:
            return identity_vectors(
                tracklets, lambda k: reader(k * stride), embedder.embed,
                samples=samples)
        except Exception:  # noqa: BLE001 — best-effort channel
            return {}
    return identify


def make_verifier(models: Models, video: str, cfg: ScoreConfig):
    """``verifier(track) -> ratio | None`` using an independent witness
    pair on magnified crops: SCRFD faces and the head detector."""
    from libs.scrfd import WITNESS_FLOOR, kps_plausible
    reader = FrameReader(video)
    scrfd = models.scrfd()
    hd = models.headdet()

    def detect(crop: np.ndarray) -> np.ndarray:
        parts = []
        b, k = scrfd.detect_full(crop, floor=WITNESS_FLOOR)
        if len(b):
            parts.append(b[kps_plausible(k)])
        h = hd.detect(crop, rotations=(0,), floor=0.20)
        if len(h):
            parts.append(h)
        return (np.concatenate(parts) if parts
                else np.empty((0, 5), np.float32))

    def verifier(t: Tracklet) -> Optional[float]:
        try:
            return verify_ratio(t, reader, detect, cfg)
        except Exception:  # noqa: BLE001
            return None
    return verifier


def ensure_analysed(
    video: str, preset: Preset, models: Models, *, use_cache: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[threading.Event] = None,
    on_status: Optional[Callable[[str], None]] = None,
    preview=None, max_frames: int = 0,
) -> Optional[proj.Project]:
    """Load the project when its fingerprint matches, else run pass 1 and
    save it. ``None`` on cancel/unreadable input."""
    cfg = preset.analysis
    params = cfg.fingerprint()
    if max_frames:
        params = {**params, "max_frames": max_frames}
    if use_cache:
        p = proj.load(video, params)
        if p is not None:
            if on_status:
                on_status(f"Analysis cache hit ({len(p.tracklets)} tracklets)")
            return p
    res = analyse(video, cfg, models, progress=progress, cancel=cancel,
                  on_status=on_status, preview=preview, max_frames=max_frames)
    if res is None:
        return None
    p = proj.Project(video=video, fps=res.fps, n_frames=res.n_frames,
                     stride=res.stride,
                     fingerprint=proj.fingerprint(video, params),
                     tracklets=res.tracklets, raw=res.raw,
                     raw_faces=res.raw_faces, raw_sources=res.raw_sources,
                     height=res.frame_hw[0], width=res.frame_hw[1],
                     settings={"timings": res.timings, "seconds": res.seconds})
    proj.save(p)
    if on_status:
        a = max(res.timings.get("analysed", 1), 1)
        on_status(f"Analysed {res.n_frames} frames in {res.seconds:.0f}s "
                  f"({res.timings.get('frame', 0) / a:.0f} ms per analysed "
                  f"frame)")
    return p


def run_offline(
    p: proj.Project, preset: Preset, models: Optional[Models], *,
    verify: bool = True, identify: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
) -> list[Track]:
    """Refine + score the project's raw tracklets → review-ready tracks."""
    step = max(1, p.stride)
    n_steps = (p.n_frames + step - 1) // step
    # Fusion and tracking are cheap and their knobs live in the preset, so
    # always rebuild from the cached detections; the live pass-1 tracklets
    # are only there for the preview (and for sidecars written without raw).
    if p.raw or p.raw_sources:
        tracklets, p.raw, p.raw_faces = retrack(
            p.raw_sources, p.raw, p.raw_faces, p.n_frames, step, p.fps,
            p.frame_hw, preset.analysis)
    else:
        tracklets = p.tracklets
    ident = (make_identify(models, p.video, step, preset.refine.identity_samples)
             if (identify and models is not None) else None)
    kept, rejected, identities, why = refine(
        tracklets, fps=p.fps / step, n_frames=n_steps, stride=step,
        cfg=preset.refine, identify=ident)
    if on_status:
        on_status(f"Refined: {len(kept)} tracks, {len(rejected)} set aside")
    verifier = (make_verifier(models, p.video, preset.score)
                if (verify and models is not None) else None)
    tracks = score_tracks(kept, rejected, p.fps, preset.score,
                          verifier=verifier, identities=identities,
                          reject_reasons=why,
                          frame_hw=p.frame_hw if p.width else None)
    tracks.sort(key=lambda t: -t.suspicion)
    return tracks
