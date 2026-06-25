#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a foreground-aware image scene from images and masks.")
    parser.add_argument("--image_dir", required=True, help="Directory containing RGB images.")
    parser.add_argument("--mask_dir", required=True, help="Directory containing masks.")
    parser.add_argument("--output_scene_dir", required=True, help="Output scene directory. Writes images/ inside it.")
    parser.add_argument("--mode", required=True, choices=["black", "white", "blur"], help="Background replacement mode.")
    parser.add_argument("--blur_kernel", type=int, default=51, help="Gaussian kernel size for --mode blur.")
    parser.add_argument("--mask_threshold", type=int, default=127, help="Foreground threshold for mask images.")
    return parser.parse_args()


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


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


def find_mask(image: Path, masks: dict[str, Path]) -> Path | None:
    return masks.get(image.name) or masks.get(image.stem) or masks.get("msk_" + image.stem.removeprefix("rgb_"))


def make_odd_kernel(value: int) -> int:
    value = max(3, value)
    return value if value % 2 == 1 else value + 1


def apply_mask(image: np.ndarray, mask: np.ndarray, mode: str, blur_kernel: int) -> np.ndarray:
    foreground = mask > 0
    if foreground.ndim == 2:
        foreground = foreground[:, :, None]

    if mode in {"black", "white"}:
        background_value = 0 if mode == "black" else 255
        background = np.full_like(image, background_value)
    else:
        kernel = make_odd_kernel(blur_kernel)
        background = cv2.GaussianBlur(image, (kernel, kernel), 0)
    return np.where(foreground, image, background)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_image_dir = Path(args.output_scene_dir) / "images"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    images = collect_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    masks = index_masks(mask_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    skipped = 0
    for image_path in images:
        mask_path = find_mask(image_path, masks)
        if mask_path is None:
            missing += 1
            print(f"missing mask: {image_path.name}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            skipped += 1
            print(f"skipped unreadable pair: {image_path.name}, {mask_path.name}")
            continue
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        binary_mask = np.where(mask >= args.mask_threshold, 255, 0).astype(np.uint8)
        output = apply_mask(image, binary_mask, args.mode, args.blur_kernel)
        if not cv2.imwrite(str(output_image_dir / image_path.name), output):
            raise RuntimeError(f"Failed to write {output_image_dir / image_path.name}")
        written += 1

    print("Masked scene summary")
    print(f"  image_dir: {image_dir}")
    print(f"  mask_dir: {mask_dir}")
    print(f"  output_images: {output_image_dir}")
    print(f"  mode: {args.mode}")
    print(f"  input_images: {len(images)}")
    print(f"  written: {written}")
    print(f"  missing_masks: {missing}")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
