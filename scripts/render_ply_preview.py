#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a lightweight PNG preview for a PLY point cloud.")
    parser.add_argument("input", help="Input .ply point cloud.")
    parser.add_argument("output", help="Output .png path.")
    parser.add_argument("--max_points", type=int, default=50000, help="Maximum points to draw.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elev", type=float, default=20.0, help="Matplotlib 3D elevation angle.")
    parser.add_argument("--azim", type=float, default=-65.0, help="Matplotlib 3D azimuth angle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = trimesh.load(input_path, process=False)
    points = np.asarray(cloud.vertices)
    if points.size == 0:
        raise RuntimeError(f"No points found in {input_path}")

    colors = None
    if hasattr(cloud, "colors") and cloud.colors is not None and len(cloud.colors) == len(points):
        colors = np.asarray(cloud.colors[:, :3], dtype=float) / 255.0

    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]

    center = points.mean(axis=0)
    points = points - center
    span = np.ptp(points, axis=0).max()
    if span > 0:
        points = points / span

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=0.4,
        c=colors if colors is not None else points[:, 2],
        cmap=None if colors is not None else "viridis",
        linewidths=0,
    )
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"points: {len(points)}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
