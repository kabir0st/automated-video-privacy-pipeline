"""Robust video writer that encodes via an FFmpeg subprocess.

Why this exists
---------------
`cv2.VideoWriter` exposes no bitrate/quality control, so high-resolution
exports (e.g. 4K @ 30 fps) balloon to ~250 Mbps. Past the 4 GiB mark OpenCV's
MP4 muxer keeps using 32-bit ``stco`` chunk offsets, which overflow — every
frame written beyond 4 GiB gets an unreadable offset and players freeze at the
boundary (~2:18 for a 250 Mbps stream). See the cleanup_blurred.mp4 incident.

Piping raw BGR frames to FFmpeg fixes the whole class of problem:
  * real rate control (match the source bitrate, or CRF) keeps files small,
  * ``-movflags +faststart`` plus FFmpeg's automatic ``co64`` promotion means
    even a >4 GiB output stays valid and seekable,
  * a dead encoder raises instead of silently truncating the file.

The returned objects are drop-in compatible with the subset of the
``cv2.VideoWriter`` API the pipeline uses: ``write(frame)``, ``release()`` and
``isOpened()``.
"""

from __future__ import annotations

import collections
import os
import shutil
import subprocess
import sys
import threading
from typing import Callable, Optional

import cv2
import numpy as np

# Windows: keep the bundled ffmpeg.exe from flashing a console window in the
# --windowed frozen build.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary.

    Prefers the static binary shipped by ``imageio-ffmpeg`` (bundled into the
    frozen Windows build), falling back to a system ``ffmpeg`` on PATH.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:  # noqa: BLE001 - any import/lookup failure → try PATH
        pass
    return shutil.which("ffmpeg")


def source_bitrate_kbps(cap: "cv2.VideoCapture") -> int:
    """Best-effort source video bitrate in kbit/s (0 if unknown)."""
    try:
        br = int(cap.get(cv2.CAP_PROP_BITRATE))
    except Exception:  # noqa: BLE001
        return 0
    return br if br > 0 else 0


class FFmpegWriter:
    """Encode BGR frames to H.264/MP4 by piping rawvideo to FFmpeg's stdin."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        *,
        ffmpeg: str,
        bitrate_kbps: int = 0,
        crf: int = 18,
        preset: str = "medium",
    ) -> None:
        self._w, self._h = width, height
        self._frame_bytes = width * height * 3

        if bitrate_kbps > 0:
            # Match the source's average bitrate (≈ same size & quality), with a
            # VBV ceiling so a busy clip can never blow past ~2× and approach
            # the old 4 GiB failure mode again.
            rate_args = [
                "-b:v", f"{bitrate_kbps}k",
                "-maxrate", f"{int(bitrate_kbps * 1.5)}k",
                "-bufsize", f"{bitrate_kbps * 2}k",
            ]
        else:
            rate_args = ["-crf", str(crf)]

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
            "-an",  # video only — audio intentionally dropped
            "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p",
            *rate_args,
            "-movflags", "+faststart",
            path,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )
        # Drain stderr in the background so a chatty/failing ffmpeg can never
        # deadlock our stdin writes by filling its stderr pipe.
        self._err: "collections.deque[str]" = collections.deque(maxlen=50)
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._err_thread.start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            self._err.append(raw.decode("utf-8", "replace").rstrip())

    def error_tail(self) -> str:
        return "\n".join(self._err)

    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2.VideoWriter
        return self.proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"ffmpeg exited early (code {self.proc.returncode}):\n{self.error_tail()}"
            )
        buf = np.ascontiguousarray(frame, dtype=np.uint8)
        if buf.nbytes != self._frame_bytes:
            raise ValueError(
                f"frame size {buf.shape} != writer geometry {self._h}x{self._w}x3"
            )
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(buf.tobytes())
        except BrokenPipeError as exc:  # ffmpeg died mid-stream
            raise RuntimeError(f"ffmpeg pipe closed:\n{self.error_tail()}") from exc

    def release(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self.proc.wait()
        self._err_thread.join(timeout=1.0)
        if self.proc.returncode not in (0, None):
            raise RuntimeError(
                f"ffmpeg failed (code {self.proc.returncode}):\n{self.error_tail()}"
            )


class _Cv2Writer:
    """Fallback wrapper around cv2.VideoWriter (used only when ffmpeg is absent).

    Retains the legacy avc1→mp4v behaviour; carries the known 4 GiB / no-rate-
    control limitations, so it is a last resort, not the default path.
    """

    def __init__(self, path: str, width: int, height: int, fps: float) -> None:
        fourcc = cv2.VideoWriter.fourcc(*"avc1")  # type: ignore[attr-defined]
        self._w = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._w.isOpened():
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
            self._w = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def isOpened(self) -> bool:  # noqa: N802
        return self._w.isOpened()

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def release(self) -> None:
        self._w.release()


def make_video_writer(
    path: str,
    width: int,
    height: int,
    fps: float,
    *,
    bitrate_kbps: int = 0,
    crf: int = 18,
    on_status: Optional[Callable[[str], None]] = None,
):
    """Build the best available writer for ``path``.

    Returns an FFmpeg-backed writer when ffmpeg is available (the robust path),
    otherwise falls back to ``cv2.VideoWriter``. Returns ``None`` if no writer
    could be opened at all.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        try:
            w = FFmpegWriter(
                path, width, height, fps,
                ffmpeg=ffmpeg, bitrate_kbps=bitrate_kbps, crf=crf,
            )
            if w.isOpened():
                return w
        except Exception as exc:  # noqa: BLE001 - fall back on any spawn failure
            if on_status:
                on_status(f"FFmpeg unavailable ({exc!r}); falling back to OpenCV writer")

    if on_status and not ffmpeg:
        on_status("FFmpeg not found; falling back to OpenCV writer (4 GiB limit applies)")
    fallback = _Cv2Writer(path, width, height, fps)
    return fallback if fallback.isOpened() else None
