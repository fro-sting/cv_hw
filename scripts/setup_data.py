#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rel_symlink(src: Path, dst: Path) -> None:
    """Create a relative symlink, replacing an existing symlink/file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)


def link_human_scene(source_dir: Path, target_dir: Path) -> None:
    images_dir = target_dir / "images"
    masks_dir = target_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(source_dir.glob("rgb_*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No rgb_*.png files found in {source_dir}")

    for rgb in rgb_files:
        idx = rgb.stem.split("_", 1)[1]
        mask = source_dir / f"msk_{idx}.png"
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask for {rgb.name}: {mask}")
        rel_symlink(rgb, images_dir / rgb.name)
        rel_symlink(mask, masks_dir / mask.name)


def write_readme(data_root: Path) -> None:
    text = """# Data Layout

This folder is a lightweight standard entrypoint for the provided datasets.
It uses symlinks to avoid copying the original files in `../大作业数据`.

```text
data/
├── human1/
│   ├── images/   # rgb_*.png
│   └── masks/    # msk_*.png
├── human2/
│   ├── images/   # rgb_*.png
│   └── masks/    # msk_*.png
└── scene3/
    └── source_video.mp4
```

For VGGT, use each `images/` directory for human data. For scene3, extract
frames from `source_video.mp4` into `data/scene3/images/` in the next stage.
"""
    (data_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a simple standard data layout using symlinks.")
    parser.add_argument("--source", default="大作业数据", help="Provided data folder.")
    parser.add_argument("--output", default="data", help="Standard data folder to create.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = (ROOT / args.source).resolve()
    output = ROOT / args.output
    if not source.exists():
        raise FileNotFoundError(f"Source data folder not found: {source}")

    link_human_scene(source / "数据1-人体", output / "human1")
    link_human_scene(source / "数据2-人体", output / "human2")

    scene_video = source / "数据3-场景.mp4"
    if not scene_video.exists():
        raise FileNotFoundError(f"Scene video not found: {scene_video}")
    rel_symlink(scene_video, output / "scene3" / "source_video.mp4")

    write_readme(output)
    print("Prepared simple data layout:")
    print(f"  {output / 'human1' / 'images'}")
    print(f"  {output / 'human1' / 'masks'}")
    print(f"  {output / 'human2' / 'images'}")
    print(f"  {output / 'human2' / 'masks'}")
    print(f"  {output / 'scene3' / 'source_video.mp4'}")


if __name__ == "__main__":
    main()
