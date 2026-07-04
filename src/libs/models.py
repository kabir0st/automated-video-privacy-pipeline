"""Central model registry + startup preflight.

One module that knows where the model the pipeline needs lives on disk, checks
whether it is present, downloads it when missing, and reports through a status
callback. Run :func:`preflight` at startup so the windowed .exe surfaces model
status on the splash and in the debug log (:data:`libs.utils.DEBUG_LOG`)
instead of stalling silently the first time a video is opened.

Two models: the PINTO body/head/face detector (see libs/detector.py for the
spec, resolution order and env overrides) that owns the pipeline, and the
SCRFD face detector (libs/scrfd.py) that only assists on extreme close-ups.
The frozen build bundles both, so a normal first run downloads nothing; the
download paths exist for dev machines and for spec overrides.

Downloads are idempotent and best-effort: any failure is reported and skipped
so startup never dies — the pipeline degrades to "no blur + status line"
(HeadDetector.available == False). Set ``AVPP_SKIP_MODEL_DOWNLOAD=1`` to
check-and-report only.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from . import detector, scrfd
from .utils import debug_log

StatusCb = Callable[[str], None]


def _skip_downloads() -> bool:
    return os.environ.get("AVPP_SKIP_MODEL_DOWNLOAD", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _preflight_detector(status: StatusCb, download: bool) -> None:
    spec = detector.active_spec()
    path = detector.model_path(spec)
    if path is not None:
        status(f"{spec.name} detector: {path}  [found]")
        return
    status(f"{spec.name} detector: {detector.default_cache(spec)}  [MISSING]")
    if not download:
        return
    detector.download_model(on_status=status, spec=spec)
    status(f"{spec.name} detector: ready")


def _preflight_scrfd(status: StatusCb, download: bool) -> None:
    path = scrfd.model_path()
    if path is not None:
        status(f"scrfd close-up detector: {path}  [found]")
        return
    status(f"scrfd close-up detector: {scrfd.default_cache()}  [MISSING]")
    if not download:
        return
    scrfd.download_model(on_status=status)
    status("scrfd close-up detector: ready")


def preflight(status_cb: Optional[StatusCb] = None, *, download: bool = True) -> None:
    """Check the model's location, download it if missing, and report.

    ``status_cb`` receives one line per location / download step (the GUI
    routes it to the splash + debug log). Best-effort: a failure is logged and
    skipped so startup never dies."""
    def status(msg: str) -> None:
        debug_log(msg)
        if status_cb is not None:
            status_cb(msg)

    if download and _skip_downloads():
        download = False
        status("AVPP_SKIP_MODEL_DOWNLOAD set — checking model locations only")

    status("Checking models…")
    try:
        _preflight_detector(status, download)
    except Exception as exc:  # noqa: BLE001 — never block startup on a model
        status(f"detector: download/check failed, will degrade — {exc!r}")
    try:
        _preflight_scrfd(status, download)
    except Exception as exc:  # noqa: BLE001 — the assist degrades to off
        status(f"scrfd: download/check failed, will degrade — {exc!r}")
    status("Model check complete")
