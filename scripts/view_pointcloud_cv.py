#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a PLY point cloud with OpenCV.")
    parser.add_argument("input", help="Input .ply point cloud.")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_points", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional screenshot path.")
    parser.add_argument(
        "--no_window",
        action="store_true",
        help="Render one frame and save --output without opening an OpenCV window.",
    )
    return parser.parse_args()


def load_cloud(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    if points.size == 0:
        raise RuntimeError(f"No points found in {path}")

    colors = None
    if hasattr(cloud, "colors") and cloud.colors is not None and len(cloud.colors) == len(points):
        colors = np.asarray(cloud.colors[:, :3], dtype=np.uint8)
    if colors is None:
        z = points[:, 2]
        z_norm = (z - z.min()) / max(float(z.max() - z.min()), 1e-8)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, 0, :]

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    points = points - points.mean(axis=0, keepdims=True)
    scale = float(np.ptp(points, axis=0).max())
    if scale > 0:
        points = points / scale
    return points, colors


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ rx @ ry


def render(
    points: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    zoom: float,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    rotated = points @ rotation_matrix(yaw, pitch, 0.0).T

    xy = rotated[:, :2] * (min(width, height) * zoom)
    px = np.round(xy[:, 0] + width / 2).astype(np.int32)
    py = np.round(-xy[:, 1] + height / 2).astype(np.int32)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)

    px = px[valid]
    py = py[valid]
    z = rotated[:, 2][valid]
    draw_colors = colors[valid]

    order = np.argsort(z)
    px = px[order]
    py = py[order]
    draw_colors = draw_colors[order]

    canvas[py, px] = draw_colors[:, ::-1]
    cv2.putText(
        canvas,
        "W/S pitch  A/D yaw  +/- zoom  R reset  P save  Q quit",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    points, colors = load_cloud(input_path, args.max_points, args.seed)

    yaw = np.deg2rad(-35.0)
    pitch = np.deg2rad(20.0)
    zoom = 0.85

    frame = render(points, colors, args.width, args.height, yaw, pitch, zoom)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"wrote: {output_path}")

    if args.no_window:
        return

    window = "pointcloud"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        frame = render(points, colors, args.width, args.height, yaw, pitch, zoom)
        cv2.imshow(window, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in {ord("q"), 27}:
            break
        if key == ord("a"):
            yaw -= np.deg2rad(5.0)
        elif key == ord("d"):
            yaw += np.deg2rad(5.0)
        elif key == ord("w"):
            pitch += np.deg2rad(5.0)
        elif key == ord("s"):
            pitch -= np.deg2rad(5.0)
        elif key in {ord("+"), ord("=")}:
            zoom *= 1.1
        elif key in {ord("-"), ord("_")}:
            zoom /= 1.1
        elif key == ord("r"):
            yaw = np.deg2rad(-35.0)
            pitch = np.deg2rad(20.0)
            zoom = 0.85
        elif key == ord("p"):
            out = Path(args.output or "outputs/pointcloud_cv_preview.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            print(f"wrote: {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
