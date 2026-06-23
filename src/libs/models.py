"""Central model registry + startup preflight.

One module that knows where every model the pipeline needs lives on disk, checks
whether it is present, downloads any that are missing, and reports each location
through a status callback. Run :func:`preflight` at startup so the windowed .exe
surfaces model status on the splash and in the debug log
(:data:`libs.utils.DEBUG_LOG`) instead of stalling silently the first time a
video is opened.

Locations are the per-library defaults (kept as-is, just centralised here):
  * InsightFace buffalo_l → ``~/.insightface/models/buffalo_l/``
  * RTMW pose + YOLOX     → ``~/.cache/rtmlib/hub/checkpoints/`` (rtmlib's cache)
  * RF-DETR person det    → ``~/.cache/avpp/rfdetr/rf-detr.onnx`` (or
    ``$AVPP_RFDETR_ONNX``)

Downloads are idempotent and best-effort: any failure is reported and skipped so
startup never dies and the pipeline keeps its existing graceful degradation
(SCRFD-only, rtmlib's YOLOX instead of RF-DETR, …). Set
``AVPP_SKIP_MODEL_DOWNLOAD=1`` to check-and-report only (skip the large
RTMW/RF-DETR pulls).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from . import person_detector
from .pose_rtmw import DEFAULT_MODE
from .utils import debug_log

StatusCb = Callable[[str], None]

INSIGHTFACE_ROOT = Path(os.path.expanduser("~/.insightface"))
BUFFALO_DIR = INSIGHTFACE_ROOT / "models" / "buffalo_l"
_BUFFALO_FILES = ("det_10g.onnx", "2d106det.onnx")


def _skip_downloads() -> bool:
    return os.environ.get("AVPP_SKIP_MODEL_DOWNLOAD", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _rtmlib_targets() -> list[tuple[str, str, Path]]:
    """``[(label, url, expected_onnx_path)]`` for the active RTMW mode.

    Mirrors rtmlib's own cache layout: ``download_checkpoint`` saves each model
    as ``<cache>/hub/checkpoints/<url-stem>.onnx`` (see rtmlib/tools/file.py), so
    we can report presence without constructing a ``Wholebody`` (which would
    build ONNX sessions — off-limits at startup per the DirectML/never-destroy
    rules)."""
    from rtmlib.tools.file import _get_rtmhub_dir
    from rtmlib.tools.solution.wholebody import Wholebody

    ckpt_dir = Path(_get_rtmhub_dir()) / "checkpoints"
    mode = Wholebody.MODE[DEFAULT_MODE]
    out: list[tuple[str, str, Path]] = []
    for label, url in (("YOLOX detector", mode["det"]),
                       ("RTMW pose", mode["pose"])):
        stem = os.path.basename(url).split(".")[0]
        out.append((label, url, ckpt_dir / f"{stem}.onnx"))
    return out


def _preflight_buffalo(status: StatusCb, download: bool) -> None:
    present = all((BUFFALO_DIR / f).exists() for f in _BUFFALO_FILES)
    status(f"InsightFace buffalo_l: {BUFFALO_DIR}  [{'found' if present else 'MISSING'}]")
    if present or not download:
        return
    from insightface.utils import ensure_available

    status("Downloading InsightFace buffalo_l (~300 MB)…")
    ensure_available("models", "buffalo_l", root="~/.insightface")
    status("InsightFace buffalo_l: ready")


def _preflight_rtmlib(status: StatusCb, download: bool) -> None:
    from rtmlib.tools.file import download_checkpoint

    for label, url, path in _rtmlib_targets():
        present = path.exists()
        status(f"{label}: {path}  [{'found' if present else 'MISSING'}]")
        if present or not download:
            continue
        status(f"Downloading {label} (~300 MB)…")
        download_checkpoint(url)
        status(f"{label}: ready")


def _preflight_rfdetr(status: StatusCb, download: bool) -> None:
    path = person_detector._model_path()
    if path is not None:
        status(f"RF-DETR person detector: {path}  [found]")
        return
    status(f"RF-DETR person detector: {person_detector._DEFAULT_CACHE}  [MISSING]")
    if not download:
        return
    person_detector.download_model(on_status=status)
    status("RF-DETR person detector: ready")


def preflight(status_cb: Optional[StatusCb] = None, *, download: bool = True) -> None:
    """Check every model's location, download any that are missing, and report.

    ``status_cb`` receives one line per model location / download step (the GUI
    routes it to the splash + debug log). Each model is handled independently and
    best-effort: a failure is logged and skipped so startup never dies."""
    def status(msg: str) -> None:
        debug_log(msg)
        if status_cb is not None:
            status_cb(msg)

    if download and _skip_downloads():
        download = False
        status("AVPP_SKIP_MODEL_DOWNLOAD set — checking model locations only")

    status("Checking models…")
    for name, fn in (("InsightFace buffalo_l", _preflight_buffalo),
                     ("RTMW/YOLOX", _preflight_rtmlib),
                     ("RF-DETR", _preflight_rfdetr)):
        try:
            fn(status, download)
        except Exception as exc:  # noqa: BLE001 — never block startup on one model
            status(f"{name}: download/check failed, will degrade — {exc!r}")
    status("Model check complete")
