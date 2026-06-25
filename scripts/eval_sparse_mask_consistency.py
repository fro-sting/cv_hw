#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pycolmap


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether sparse tracks land on foreground masks.")
    parser.add_argument("--sparse_dir", required=True, help="COLMAP sparse directory.")
    parser.add_argument("--image_dir", required=True, help="Image directory used by the sparse model.")
    parser.add_argument("--mask_dir", required=True, help="Mask directory.")
    parser.add_argument("--csv", required=True, help="CSV path for mask consistency metrics.")
    parser.add_argument("--append_csv", action="store_true", help="Append to CSV instead of replacing it.")
    parser.add_argument("--mask_dilate", type=int, default=0, help="Dilate foreground masks by this many pixels.")
    parser.add_argument("--mask_threshold", type=int, default=127, help="Foreground threshold for masks.")
    parser.add_argument(
        "--foreground_point_threshold",
        type=float,
        default=0.5,
        help="Point is foreground-dominated when this fraction of its observations are foreground.",
    )
    return parser.parse_args()


def resolve_sparse_dir(sparse_dir: Path) -> Path:
    if (sparse_dir / "0").is_dir() and not (sparse_dir / "cameras.bin").exists():
        return sparse_dir / "0"
    return sparse_dir


def index_masks(mask_dir: Path) -> dict[str, Path]:
    masks: dict[str, Path] = {}
    for path in sorted(mask_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        masks[path.name] = path
        masks[path.stem] = path
        if path.stem.startswith("msk_"):
            masks["rgb_" + path.stem[4:]] = path
        if path.stem.startswith("mask_"):
            masks["rgb_" + path.stem[5:]] = path
    return masks


def find_mask(image_name: str, masks: dict[str, Path]) -> Path | None:
    image_path = Path(image_name)
    return (
        masks.get(image_path.name)
        or masks.get(image_path.stem)
        or masks.get("msk_" + image_path.stem.removeprefix("rgb_"))
    )


def load_mask(mask_path: Path, shape: tuple[int, int] | None, threshold: int, dilate: int) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    if shape is not None and mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask >= threshold).astype(np.uint8)
    if dilate > 0:
        kernel_size = 2 * dilate + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def image_shape(image_dir: Path, image_name: str) -> tuple[int, int] | None:
    image_path = image_dir / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image.shape[:2]


def mask_value(mask: np.ndarray, x: float, y: float) -> int:
    col = int(round(x))
    row = int(round(y))
    if row < 0 or col < 0 or row >= mask.shape[0] or col >= mask.shape[1]:
        return 0
    return int(mask[row, col] > 0)


def point2d_xy(point2d: object) -> tuple[float, float]:
    if hasattr(point2d, "x") and hasattr(point2d, "y"):
        return float(point2d.x), float(point2d.y)
    xy = getattr(point2d, "xy")
    return float(xy[0]), float(xy[1])


def write_csv(path: Path, row: dict[str, object], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sparse_dir",
        "image_dir",
        "mask_dir",
        "points3D",
        "tracked_observations",
        "foreground_point_ratio",
        "foreground_observation_ratio",
        "background_dominated_points",
        "mask_dilate",
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
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    masks = index_masks(mask_dir)
    mask_cache: dict[int, np.ndarray] = {}

    foreground_points = 0
    background_dominated_points = 0
    foreground_observations = 0
    tracked_observations = 0
    points_with_observations = 0
    missing_masks: set[str] = set()

    for point in reconstruction.points3D.values():
        point_foreground = 0
        point_observations = 0
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            mask_path = find_mask(image.name, masks)
            if mask_path is None:
                missing_masks.add(image.name)
                continue
            if element.image_id not in mask_cache:
                shape = image_shape(image_dir, image.name)
                mask_cache[element.image_id] = load_mask(mask_path, shape, args.mask_threshold, args.mask_dilate)
            point2d = image.points2D[element.point2D_idx]
            x, y = point2d_xy(point2d)
            is_foreground = mask_value(mask_cache[element.image_id], x, y)
            point_foreground += is_foreground
            point_observations += 1

        if point_observations == 0:
            continue
        points_with_observations += 1
        foreground_observations += point_foreground
        tracked_observations += point_observations
        if point_foreground / point_observations >= args.foreground_point_threshold:
            foreground_points += 1
        else:
            background_dominated_points += 1

    foreground_point_ratio = foreground_points / points_with_observations if points_with_observations else 0.0
    foreground_observation_ratio = foreground_observations / tracked_observations if tracked_observations else 0.0

    print(f"sparse_dir: {sparse_dir}")
    print(f"points3D: {len(reconstruction.points3D)}")
    print(f"points_with_mask_observations: {points_with_observations}")
    print(f"tracked_observations: {tracked_observations}")
    print(f"foreground_point_ratio: {foreground_point_ratio:.6f}")
    print(f"foreground_observation_ratio: {foreground_observation_ratio:.6f}")
    print(f"background_dominated_points: {background_dominated_points}")
    if missing_masks:
        print(f"missing_masks: {len(missing_masks)} image names")

    write_csv(
        Path(args.csv),
        {
            "sparse_dir": str(sparse_dir),
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "points3D": len(reconstruction.points3D),
            "tracked_observations": tracked_observations,
            "foreground_point_ratio": f"{foreground_point_ratio:.6f}",
            "foreground_observation_ratio": f"{foreground_observation_ratio:.6f}",
            "background_dominated_points": background_dominated_points,
            "mask_dilate": args.mask_dilate,
        },
        append=args.append_csv,
    )


if __name__ == "__main__":
    main()
