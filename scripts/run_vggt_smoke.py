#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = ROOT / "vggt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight VGGT smoke test without downloading pretrained weights."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene_dir", help="Scene directory containing an images/ folder.")
    group.add_argument("--image_dir", help="Directory containing input images directly.")
    parser.add_argument("--max_images", type=int, default=4, help="Maximum images to inspect.")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Also run VGGT image preprocessing on the selected images.",
    )
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Instantiate VGGT with random weights. This checks model construction only.",
    )
    return parser.parse_args()


def collect_images(args: argparse.Namespace) -> list[Path]:
    image_dir = Path(args.image_dir) if args.image_dir else Path(args.scene_dir) / "images"
    if not image_dir.is_absolute():
        image_dir = ROOT / image_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No jpg/png images found in {image_dir}")
    return image_paths[: args.max_images]


def inspect_images(image_paths: list[Path]) -> None:
    print(f"images: {len(image_paths)} selected")
    for path in image_paths:
        with Image.open(path) as img:
            print(f"  {path.relative_to(ROOT)}  size={img.size}  mode={img.mode}")


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args)
    inspect_images(image_paths)

    sys.path.insert(0, str(VGGT_ROOT))

    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("device: cpu")

    from vggt.models.vggt import VGGT

    print("VGGT import: ok")

    if args.preprocess:
        from vggt.utils.load_fn import load_and_preprocess_images_square

        images, original_coords = load_and_preprocess_images_square(
            [str(path) for path in image_paths],
            target_size=518,
        )
        print(f"preprocess: images={tuple(images.shape)} original_coords={tuple(original_coords.shape)}")

    if args.init_model:
        model = VGGT()
        param_count = sum(param.numel() for param in model.parameters())
        print(f"VGGT init: ok ({param_count / 1_000_000:.1f}M parameters, random weights)")

    print("smoke test: ok")


if __name__ == "__main__":
    main()
