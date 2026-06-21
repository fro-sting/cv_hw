#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pycolmap
import trimesh


REQUIRED = ("cameras.bin", "images.bin", "points3D.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a COLMAP sparse reconstruction.")
    parser.add_argument("sparse_dir", help="Directory containing cameras.bin/images.bin/points3D.bin.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sparse_dir = Path(args.sparse_dir)
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")

    missing = [name for name in REQUIRED if not (sparse_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sparse files in {sparse_dir}: {', '.join(missing)}")

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    track_lengths = [len(point.track.elements) for point in reconstruction.points3D.values()]
    mean_track_length = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0

    ply_path = sparse_dir / "points.ply"
    ply_points = None
    if ply_path.exists():
        cloud = trimesh.load(ply_path, process=False)
        ply_points = len(cloud.vertices)

    print(f"sparse_dir: {sparse_dir}")
    print(f"cameras: {len(reconstruction.cameras)}")
    print(f"images: {len(reconstruction.images)}")
    print(f"points3D: {len(reconstruction.points3D)}")
    print(f"mean_track_length: {mean_track_length:.3f}")
    if ply_points is None:
        print("points.ply: missing")
    else:
        print(f"points.ply: {ply_points} points")
    status = "ok" if reconstruction.cameras and reconstruction.images and reconstruction.points3D else "empty"
    print(f"status: {status}")


if __name__ == "__main__":
    main()
