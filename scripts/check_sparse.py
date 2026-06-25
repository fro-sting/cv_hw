#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

import pycolmap
import trimesh


REQUIRED = ("cameras.bin", "images.bin", "points3D.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a COLMAP sparse reconstruction.")
    parser.add_argument("sparse_dir", help="Directory containing cameras.bin/images.bin/points3D.bin.")
    parser.add_argument("--csv", help="Optional CSV path for metrics output.")
    parser.add_argument("--append_csv", action="store_true", help="Append to --csv instead of replacing it.")
    return parser.parse_args()


def resolve_sparse_dir(sparse_dir: Path) -> Path:
    if (sparse_dir / "0").is_dir() and not (sparse_dir / "cameras.bin").exists():
        return sparse_dir / "0"
    return sparse_dir


def count_ply_points(ply_path: Path) -> int | None:
    if not ply_path.exists():
        return None
    cloud = trimesh.load(ply_path, process=False)
    return len(cloud.vertices)


def write_csv(path: Path, row: dict[str, object], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sparse_dir",
        "cameras",
        "images",
        "points3D",
        "mean_track_length",
        "median_track_length",
        "max_track_length",
        "track_ge_2",
        "track_ge_3",
        "track_ge_5",
        "ply_points",
    ]
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    sparse_dir = resolve_sparse_dir(Path(args.sparse_dir))
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")

    missing = [name for name in REQUIRED if not (sparse_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sparse files in {sparse_dir}: {', '.join(missing)}")

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    track_lengths = [len(point.track.elements) for point in reconstruction.points3D.values()]
    mean_track_length = mean(track_lengths) if track_lengths else 0.0
    median_track_length = median(track_lengths) if track_lengths else 0.0
    max_track_length = max(track_lengths) if track_lengths else 0
    track_ge_2 = sum(length >= 2 for length in track_lengths)
    track_ge_3 = sum(length >= 3 for length in track_lengths)
    track_ge_5 = sum(length >= 5 for length in track_lengths)

    ply_path = sparse_dir / "points.ply"
    ply_points = count_ply_points(ply_path)

    print(f"sparse_dir: {sparse_dir}")
    print(f"cameras: {len(reconstruction.cameras)}")
    print(f"images: {len(reconstruction.images)}")
    print(f"points3D: {len(reconstruction.points3D)}")
    print(f"mean_track_length: {mean_track_length:.3f}")
    print(f"median_track_length: {median_track_length:.3f}")
    print(f"max_track_length: {max_track_length}")
    print(f"track_ge_2: {track_ge_2}")
    print(f"track_ge_3: {track_ge_3}")
    print(f"track_ge_5: {track_ge_5}")
    if ply_points is None:
        print("points.ply: missing")
    else:
        print(f"points.ply: {ply_points} points")
    status = "ok" if reconstruction.cameras and reconstruction.images and reconstruction.points3D else "empty"
    print(f"status: {status}")

    if args.csv:
        write_csv(
            Path(args.csv),
            {
                "sparse_dir": str(sparse_dir),
                "cameras": len(reconstruction.cameras),
                "images": len(reconstruction.images),
                "points3D": len(reconstruction.points3D),
                "mean_track_length": f"{mean_track_length:.6f}",
                "median_track_length": f"{median_track_length:.6f}",
                "max_track_length": max_track_length,
                "track_ge_2": track_ge_2,
                "track_ge_3": track_ge_3,
                "track_ge_5": track_ge_5,
                "ply_points": "" if ply_points is None else ply_points,
            },
            append=args.append_csv,
        )


if __name__ == "__main__":
    main()
