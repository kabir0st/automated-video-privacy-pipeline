"""Pipeline B — Highest Accuracy Offline face landmark tracking.

Usage:
    uv run python src/main.py                        # open GUI inspector
    uv run python src/main.py --input path/to/video.mp4
    uv run python src/main.py --input 0              # webcam
    uv run python src/main.py --input video.mp4 --output out.mp4
"""

import argparse
import sys
import time

import cv2
import numpy as np
from libs.face_app import FaceApp

from libs.pose_head import PoseHeadEstimator
from libs.tracker import KalmanFaceTracker
from libs.utils import (
    BlurPipeline,
    MaskBuilder,
    best_onnx_providers,
    crop_face_patch,
    is_likely_face,
    unproject_landmark,
)

CLOSE_UP_AREA_RATIO = 0.6
CLOSE_UP_TARGET_SIZE = 1024
NORMAL_TARGET_SIZE = 640

_TRACK_COLOURS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 200, 100),
    (200, 100, 0),
    (100, 0, 200),
]


def track_colour(track_id: int) -> tuple[int, int, int]:
    return _TRACK_COLOURS[track_id % len(_TRACK_COLOURS)]


def is_close_up(bbox: np.ndarray, frame_h: int, frame_w: int) -> bool:
    x1, y1, x2, y2 = bbox[:4]
    return ((x2 - x1) * (y2 - y1)) / (frame_h * frame_w) > CLOSE_UP_AREA_RATIO


def draw_track_outline(
    frame: np.ndarray,
    poly: np.ndarray | None,
    bbox: np.ndarray,
    track_id: int,
    colour: tuple[int, int, int],
) -> None:
    """Draw the expanded hull polygon (or bbox rectangle for ghost tracks)."""
    if poly is not None:
        cv2.polylines(frame, [poly], True, colour, 1, cv2.LINE_AA)
        x, y = poly[:, 0].min(), poly[:, 1].min()
    else:
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
        x, y = x1, y1
    cv2.putText(
        frame, f"id:{track_id}", (x, max(0, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline B: InsightFace + Kalman face tracking (+ pose assist) + GPU blur")
    p.add_argument("--input", default="0", help="Video path or webcam index (default: 0)")
    p.add_argument("--output", default="", help="Optional output video path")
    p.add_argument(
        "--target-size", type=int, default=NORMAL_TARGET_SIZE,
        help="Detection input size (default: 640; use 1024+ for offline accuracy)",
    )
    p.add_argument("--no-display", action="store_true", help="Suppress cv2.imshow")
    p.add_argument(
        "--no-pose", action="store_true",
        help="Disable pose-assisted head tracking for lost faces",
    )
    p.add_argument(
        "--hold-secs", type=float, default=2.0,
        help="Keep blurring a lost face this long on Kalman prediction (default: 2.0)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # No --input supplied → launch the GUI inspector instead of defaulting to webcam.
    if args.input == "0" and len(sys.argv) == 1:
        from ui import main as ui_main
        ui_main()
        return

    source = int(args.input) if args.input.isdigit() else args.input

    app = FaceApp(providers=best_onnx_providers())
    app.prepare(ctx_id=0, det_size=(args.target_size, args.target_size))

    blur = BlurPipeline()
    masks = MaskBuilder()
    pose = None if args.no_pose else PoseHeadEstimator(on_status=print)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source '{args.input}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = KalmanFaceTracker(fps=fps, hold_secs=args.hold_secs)
    # Use actual first-frame dimensions — more reliable than codec-reported values.
    ret0, frame0 = cap.read()
    if not ret0:
        print("ERROR: cannot read first frame", file=sys.stderr)
        sys.exit(1)
    frame_h, frame_w = frame0.shape[:2]

    writer: cv2.VideoWriter | None = None
    if args.output:
        _fourcc = cv2.VideoWriter.fourcc(*"avc1")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(args.output, _fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            _fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(args.output, _fourcc, fps, (frame_w, frame_h))

    frame_idx = 0
    t_start = time.perf_counter()
    pending = [frame0]  # seed with the pre-read first frame

    while True:
        if pending:
            frame = pending.pop(0)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # ── detection ────────────────────────────────────────────────────────
        raw_faces = app.get(frame)
        faces = [f for f in raw_faces if is_likely_face(f.bbox, float(f.det_score))]

        # Close-up: face fills > 60 % of frame → crop + upsample for precision.
        refined: list = []
        for face in faces:
            if is_close_up(face.bbox, frame_h, frame_w):
                crop, (ox, oy, scale) = crop_face_patch(
                    frame, face.bbox[:4], target_size=CLOSE_UP_TARGET_SIZE
                )
                crop_faces = app.get(crop)
                if crop_faces:
                    cf = crop_faces[0]
                    if cf.landmark_2d_106 is not None:
                        cf.landmark_2d_106 = np.array(
                            [unproject_landmark(x, y, ox, oy, scale) for x, y in cf.landmark_2d_106],
                            dtype=np.float32,
                        )
                        cf.bbox[:4] = [
                            ox + cf.bbox[0] / scale, oy + cf.bbox[1] / scale,
                            ox + cf.bbox[2] / scale, oy + cf.bbox[3] / scale,
                        ]
                    refined.append(cf)
                    continue
            refined.append(face)

        # ── tracking — Kalman predict + face correction; pose head boxes
        # revive tracks whose face the detector lost this frame ───────────────
        head_provider = (lambda f=frame: pose.head_boxes(f)) if pose else None
        tracked = tracker.update(refined, frame.shape, head_provider)

        # ── build combined blur mask (one pass for all faces) ─────────────────
        blur_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        polys: dict[int, np.ndarray | None] = {}  # track_id → hull polygon for drawing

        for t in tracked:
            polys[t.track_id] = masks.add(blur_mask, t)
        masks.evict({t.track_id for t in tracked})

        # ── apply stacked GPU/CPU blur in a single call ───────────────────────
        blur.apply(frame, blur_mask)

        # ── draw hull outlines + track IDs on top of blur ─────────────────────
        for t in tracked:
            bbox = t.face.bbox if t.face is not None else t.bbox
            draw_track_outline(frame, polys.get(t.track_id), bbox,
                               t.track_id, track_colour(t.track_id))

        # FPS overlay
        elapsed = time.perf_counter() - t_start
        cv2.putText(
            frame, f"fps:{((frame_idx + 1) / elapsed):.1f}",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

        if writer:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Pipeline B", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    total = time.perf_counter() - t_start
    print(f"Processed {frame_idx} frames — avg {frame_idx / total:.1f} fps")


if __name__ == "__main__":
    main()
