#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames for data/scene3/images.")
    parser.add_argument("--video", default="data/scene3/source_video.mp4")
    parser.add_argument("--output", default="data/scene3/images")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--resize_long_edge", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1 if source_fps <= 0 or args.fps <= 0 else max(1, round(source_fps / args.fps))

    frame_idx = 0
    saved = 0
    while saved < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        if args.resize_long_edge > 0:
            h, w = frame.shape[:2]
            scale = args.resize_long_edge / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

        saved += 1
        out_path = output / f"{saved:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        frame_idx += 1

    cap.release()
    if saved == 0:
        raise RuntimeError(f"No frames extracted from {video}")
    print(f"Extracted {saved} frames to {output}")
    print(f"Source FPS: {source_fps:.3f}, step: {step}")


if __name__ == "__main__":
    main()
