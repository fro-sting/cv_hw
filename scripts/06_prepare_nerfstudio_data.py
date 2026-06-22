#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pycolmap
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a COLMAP sparse scene for nerfstudio splatfacto.")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images.")
    parser.add_argument("--sparse_dir", required=True, help="COLMAP sparse directory containing cameras/images/points3D.")
    parser.add_argument("--output_dir", required=True, help="Output nerfstudio-style data directory.")
    parser.add_argument(
        "--mask_dir",
        default=None,
        help=(
            "Optional source mask directory. For rgb_0001.png images this script looks for "
            "rgb_0001.png first, then msk_0001.png, and writes masks with image-matching names."
        ),
    )
    parser.add_argument("--copy_images", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument(
        "--make_images_2",
        action="store_true",
        help="Also create images_2/ by downscaling images by 2 with Pillow, avoiding nerfstudio's ffmpeg step.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument(
        "--allow_missing_images",
        action="store_true",
        help="Do not fail if COLMAP references images missing from --image_dir.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value


def rel_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)


def copy_or_link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        rel_symlink(src, dst)


def check_sparse(sparse_dir: Path) -> pycolmap.Reconstruction:
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")
    missing = [name for name in SPARSE_FILES if not (sparse_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sparse files in {sparse_dir}: {', '.join(missing)}")
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    if not reconstruction.images:
        raise RuntimeError(f"No registered images found in {sparse_dir}")
    return reconstruction


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "sparse" / "0").mkdir(parents=True)


def link_images(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    output_dir: Path,
    copy_images: bool,
    allow_missing: bool,
) -> list[str]:
    copied: list[str] = []
    missing: list[str] = []
    names = sorted({image.name for image in reconstruction.images.values()})
    for name in names:
        src = image_dir / name
        if not src.exists():
            missing.append(name)
            continue
        copy_or_link(src, output_dir / "images" / name, copy_images)
        copied.append(name)

    if missing and not allow_missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} COLMAP image names were not found in {image_dir}: {preview}"
        )
    return copied


def find_mask(mask_dir: Path, image_name: str) -> Path | None:
    direct = mask_dir / image_name
    if direct.exists():
        return direct

    stem = Path(image_name).stem
    suffix = Path(image_name).suffix
    if stem.startswith("rgb_"):
        candidate = mask_dir / f"msk_{stem.split('_', 1)[1]}{suffix}"
        if candidate.exists():
            return candidate

    candidate = mask_dir / f"{stem}.png"
    if candidate.exists():
        return candidate
    return None


def copy_masks(mask_dir: Path, output_dir: Path, copied_images: list[str], copy_images: bool) -> list[str]:
    output_mask_dir = output_dir / "masks"
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    copied_masks: list[str] = []
    missing: list[str] = []
    for image_name in copied_images:
        src = find_mask(mask_dir, image_name)
        if src is None:
            missing.append(image_name)
            continue
        # Nerfstudio expects mask filenames to match image filenames.
        copy_or_link(src, output_mask_dir / image_name, copy_images)
        copied_masks.append(image_name)

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} masks were not found in {mask_dir}: {preview}")
    return copied_masks


def copy_sparse(sparse_dir: Path, output_dir: Path) -> None:
    target = output_dir / "sparse" / "0"
    for name in SPARSE_FILES:
        shutil.copy2(sparse_dir / name, target / name)
    ply = sparse_dir / "points.ply"
    if ply.exists():
        shutil.copy2(ply, target / "points.ply")


def make_downscaled_images(output_dir: Path, copied_images: list[str], factor: int = 2) -> None:
    source_dir = output_dir / "images"
    target_dir = output_dir / f"images_{factor}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in copied_images:
        src = source_dir / name
        dst = target_dir / name
        with Image.open(src) as image:
            width, height = image.size
            resized = image.resize(
                (max(1, width // factor), max(1, height // factor)),
                resample=Image.Resampling.LANCZOS,
            )
            if resized.mode not in {"RGB", "RGBA"}:
                resized = resized.convert("RGB")
            resized.save(dst)


def make_downscaled_masks(output_dir: Path, copied_masks: list[str], factor: int = 2) -> None:
    if not copied_masks:
        return
    source_dir = output_dir / "masks"
    target_dir = output_dir / f"masks_{factor}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in copied_masks:
        src = source_dir / name
        dst = target_dir / name
        with Image.open(src) as image:
            width, height = image.size
            mask = image.convert("L")
            resized = mask.resize(
                (max(1, width // factor), max(1, height // factor)),
                resample=Image.Resampling.NEAREST,
            )
            resized.save(dst)


def write_readme(
    output_dir: Path,
    image_dir: Path,
    sparse_dir: Path,
    copied_images: list[str],
    copied_masks: list[str],
) -> None:
    text = f"""# Nerfstudio / 3DGS Data

Source images: `{image_dir}`
Source sparse: `{sparse_dir}`

Layout:

```text
images/      # images referenced by COLMAP images.bin
sparse/0/    # cameras.bin, images.bin, points3D.bin
```

Prepared images: {len(copied_images)}
Prepared masks: {len(copied_masks)}

Train example:

```bash
bash scripts/07_train_splatfacto.sh \\
  --data {output_dir.relative_to(ROOT)} \\
  --output outputs/3dgs
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_dir = resolve(args.image_dir)
    sparse_dir = resolve(args.sparse_dir)
    output_dir = resolve(args.output_dir)
    mask_dir = resolve(args.mask_dir) if args.mask_dir else None

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if mask_dir is not None and not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    reconstruction = check_sparse(sparse_dir)
    prepare_output(output_dir, args.overwrite)
    copied_images = link_images(
        reconstruction,
        image_dir=image_dir,
        output_dir=output_dir,
        copy_images=args.copy_images,
        allow_missing=args.allow_missing_images,
    )
    copied_masks: list[str] = []
    if mask_dir is not None:
        copied_masks = copy_masks(mask_dir, output_dir, copied_images, args.copy_images)
    copy_sparse(sparse_dir, output_dir)
    if args.make_images_2:
        make_downscaled_images(output_dir, copied_images, factor=2)
        make_downscaled_masks(output_dir, copied_masks, factor=2)
    write_readme(output_dir, image_dir, sparse_dir, copied_images, copied_masks)

    track_lengths = [len(point.track.elements) for point in reconstruction.points3D.values()]
    mean_track = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
    print(f"output_dir: {output_dir.relative_to(ROOT)}")
    print(f"images: {len(copied_images)}")
    print(f"cameras: {len(reconstruction.cameras)}")
    print(f"registered_images: {len(reconstruction.images)}")
    print(f"masks: {len(copied_masks)}")
    print(f"points3D: {len(reconstruction.points3D)}")
    print(f"mean_track_length: {mean_track:.3f}")
    print(f"sparse: {output_dir.relative_to(ROOT) / 'sparse' / '0'}")
    if args.make_images_2:
        print(f"images_2: {output_dir.relative_to(ROOT) / 'images_2'}")


if __name__ == "__main__":
    main()
