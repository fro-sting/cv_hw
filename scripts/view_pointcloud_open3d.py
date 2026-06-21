#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a point cloud with Open3D.")
    parser.add_argument("input", help="Input point cloud, usually sparse/points.ply.")
    parser.add_argument("--voxel_size", type=float, default=0.0, help="Optional voxel downsample size.")
    parser.add_argument("--max_points", type=int, default=0, help="Optional random point limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--background", choices=["black", "white"], default="black")
    parser.add_argument("--screenshot", default=None, help="Optional screenshot path.")
    return parser.parse_args()


def limit_points(pcd: o3d.geometry.PointCloud, max_points: int, seed: int) -> o3d.geometry.PointCloud:
    if max_points <= 0 or len(pcd.points) <= max_points:
        return pcd
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(pcd.points), size=max_points, replace=False)
    return pcd.select_by_index(keep.tolist())


def load_point_cloud(path: Path, voxel_size: float, max_points: int, seed: int) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Open3D loaded an empty point cloud: {path}")
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd = limit_points(pcd, max_points=max_points, seed=seed)
    return pcd


def main() -> None:
    args = parse_args()
    pcd = load_point_cloud(Path(args.input), args.voxel_size, args.max_points, args.seed)

    print(f"points: {len(pcd.points)}")
    print("Controls: left-drag rotate, wheel zoom, right-drag/pan translate, Q/Esc close.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Point Cloud", width=1280, height=900)
    vis.add_geometry(pcd)

    render_options = vis.get_render_option()
    render_options.point_size = args.point_size
    if args.background == "white":
        render_options.background_color = np.asarray([1.0, 1.0, 1.0])
    else:
        render_options.background_color = np.asarray([0.0, 0.0, 0.0])

    vis.run()
    if args.screenshot:
        out = Path(args.screenshot)
        out.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(out), do_render=True)
        print(f"wrote: {out}")
    vis.destroy_window()


if __name__ == "__main__":
    main()
