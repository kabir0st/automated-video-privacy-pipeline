"""Pipeline B CLI — Highest Accuracy Offline face landmark tracking.

Processes a video file only (no webcam/live capture).

Usage (via the main.py dispatcher):
    uv run python src/main.py --input path/to/video.mp4 --output out.mp4
    uv run python src/main.py --input video.mp4 --privacy-safety-net
    uv run python src/main.py --input video.mp4 --output out.mp4 --two-pass
"""

import argparse
import sys
import time

import cv2
import numpy as np
from libs.face_app import FaceApp

from libs.pipeline import detect, make_pose_backend
from libs.tracker import KalmanFaceTracker
from libs.video_writer import make_video_writer, source_bitrate_kbps
from libs.utils import BlurPipeline, MaskBuilder, best_onnx_providers

CLOSE_UP_AREA_RATIO = 0.6
CLOSE_UP_TARGET_SIZE = 1024
NORMAL_TARGET_SIZE = 640
DET_SCORE = 0.55          # SCRFD confidence floor
FACE_ASPECT = 0.40        # min width/height ratio for an SCRFD face

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
    p.add_argument("--input", required=True, help="Path to the video file to process")
    p.add_argument("--output", default="", help="Optional output video path")
    p.add_argument(
        "--target-size", type=int, default=NORMAL_TARGET_SIZE,
        help="Detection input size (default: 640; use 1024+ for offline accuracy)",
    )
    p.add_argument("--no-display", action="store_true", help="Suppress cv2.imshow")
    p.add_argument(
        "--pose-backend", default="rtmw", choices=["rtmw", "mediapipe", "none"],
        help="Head/pose source: rtmw (RTMDet+RTMW, robust at odd angles, "
             "default), mediapipe (legacy), or none",
    )
    p.add_argument(
        "--pose-mode", default="performance",
        choices=["performance", "balanced", "lightweight"],
        help="RTMW model: performance (RTMW-x, best), balanced, or lightweight "
             "(RTMW-l, fast)",
    )
    p.add_argument(
        "--no-pose", action="store_true",
        help="Alias for --pose-backend none (disable pose assist entirely)",
    )
    p.add_argument(
        "--hold-secs", type=float, default=0.6,
        help="Keep blurring a lost face this long on Kalman prediction (default: 0.6)",
    )
    p.add_argument(
        "--privacy-safety-net", action="store_true",
        help="Blur the head region of any RF-DETR-detected person even when no "
             "face/pose is found (fewer missed-face leaks; may over-blur). "
             "Requires --pose-backend rtmw and an exported RF-DETR ONNX model.",
    )
    p.add_argument(
        "--two-pass", action="store_true",
        help="Offline two-pass: collect detections over the whole video, then "
             "forward-backward interpolate to blur faces across their full "
             "on-screen span (requires --output; no live display).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    source = args.input

    providers = best_onnx_providers()
    print(f"[providers] ONNX inference: {providers[0]}  (available: {providers})")
    if providers[0] == "CPUExecutionProvider":
        print("[providers] WARNING: no GPU provider — on an AMD RX 6800 install "
              "onnxruntime-directml (Windows) for GPU inference.", file=sys.stderr)
    app = FaceApp(providers=providers)
    app.prepare(ctx_id=0, det_size=(args.target_size, args.target_size))

    blur = BlurPipeline()
    masks = MaskBuilder()
    backend = "none" if args.no_pose else args.pose_backend
    pose = make_pose_backend(backend, mode=args.pose_mode, on_status=print)

    if args.two_pass:
        if not args.output:
            print("ERROR: --two-pass requires --output", file=sys.stderr)
            sys.exit(1)
        from libs.two_pass import run_two_pass

        run_two_pass(
            source, args.output, app=app, pose=pose, blur=blur,
            det_score=DET_SCORE, face_aspect=FACE_ASPECT,
            close_up_ratio=CLOSE_UP_AREA_RATIO,
            close_up_target=CLOSE_UP_TARGET_SIZE,
            hold_secs=args.hold_secs, safety_net=args.privacy_safety_net,
            on_status=lambda m: print(m, file=sys.stderr),
        )
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source '{args.input}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_kbps = source_bitrate_kbps(cap)
    tracker = KalmanFaceTracker(fps=fps, hold_secs=args.hold_secs)
    # Use actual first-frame dimensions — more reliable than codec-reported values.
    ret0, frame0 = cap.read()
    if not ret0:
        print("ERROR: cannot read first frame", file=sys.stderr)
        sys.exit(1)
    frame_h, frame_w = frame0.shape[:2]

    writer = None
    if args.output:
        # FFmpeg-backed, source-bitrate-matched encode: avoids OpenCV's
        # uncontrolled bitrate that overflows MP4's 32-bit offsets past 4 GiB.
        writer = make_video_writer(
            args.output, frame_w, frame_h, fps,
            bitrate_kbps=src_kbps, on_status=lambda m: print(m, file=sys.stderr),
        )
        if writer is None:
            print(f"ERROR: cannot create output '{args.output}'", file=sys.stderr)
            sys.exit(1)

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

        # ── detection — SCRFD faces ⊕ RTMW pose faces (ensembled) ─────────────
        faces, head_boxes, anchors, _pf = detect(
            app, frame, pose,
            det_score=DET_SCORE, face_aspect=FACE_ASPECT,
            close_up_ratio=CLOSE_UP_AREA_RATIO,
            close_up_target=CLOSE_UP_TARGET_SIZE,
        )

        # ── tracking — Kalman predict + face correction; pose head boxes
        # revive tracks whose face the detector lost this frame; RF-DETR person
        # anchors (opt-in) spawn/sustain blur where no face was ever found ─────
        head_provider = (lambda: head_boxes) if pose is not None else None
        anchor_provider = (lambda: anchors) if args.privacy_safety_net else None
        tracked = tracker.update(faces, frame.shape, head_provider, anchor_provider)

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
