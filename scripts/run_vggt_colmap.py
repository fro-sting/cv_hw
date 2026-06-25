#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = ROOT / "vggt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT's COLMAP export from the project scripts directory."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scene_dir",
        help="Scene directory containing an images/ folder. VGGT writes sparse/ under this directory unless --output_scene_dir is set.",
    )
    source.add_argument(
        "--image_dir",
        help="Directory containing input images directly. Requires --output_scene_dir.",
    )
    parser.add_argument(
        "--output_scene_dir",
        help="Optional output scene directory. Images are symlinked into output_scene_dir/images before running VGGT.",
    )
    parser.add_argument("--max_images", type=int, default=None, help="Use at most this many images.")
    parser.add_argument(
        "--sample_strategy",
        choices=["first", "uniform"],
        default="uniform",
        help="How to choose images when --max_images is set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_ba",
        action="store_true",
        help="Ask VGGT demo_colmap.py to run its built-in pycolmap BA. This is slower.",
    )
    parser.add_argument(
        "--skip_builtin_ba",
        action="store_true",
        help=(
            "With --use_ba, run VGGT track prediction and COLMAP construction, "
            "but skip demo_colmap.py's final pycolmap.bundle_adjustment call. "
            "Use this to create an initial sparse model for scripts/run_self_ba.py."
        ),
    )
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera", action="store_true")
    parser.add_argument("--camera_type", default="SIMPLE_PINHOLE")
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--no_fine_tracking", action="store_true")
    parser.add_argument("--conf_thres_value", type=float, default=5.0)
    parser.add_argument(
        "--overwrite_sparse",
        action="store_true",
        help="Remove an existing sparse/ output directory before running.",
    )
    parser.add_argument(
        "--run_self_ba",
        action="store_true",
        help="After VGGT export, run scripts/run_self_ba.py on the generated sparse model.",
    )
    parser.add_argument("--self_ba_output_name", default="sparse_self_ba", help="Output directory name under the scene.")
    parser.add_argument("--self_ba_max_points", type=int, default=5000)
    parser.add_argument("--self_ba_max_nfev", type=int, default=80)
    parser.add_argument("--self_ba_loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--self_ba_f_scale", type=float, default=2.0)
    parser.add_argument(
        "--self_ba_max_initial_reproj_error_px",
        type=float,
        default=64.0,
        help="Initial outlier filter passed to run_self_ba.py. Use <=0 to disable.",
    )
    parser.add_argument(
        "--self_ba_prune_reproj_error_px",
        type=float,
        default=12.0,
        help="Post-BA point prune threshold. Use <=0 to disable.",
    )
    parser.add_argument(
        "--self_ba_optimize_shared_intrinsics",
        action="store_true",
        help="Pass --optimize_shared_intrinsics to run_self_ba.py.",
    )
    parser.add_argument(
        "--self_ba_keep_unoptimized_points",
        action="store_true",
        help="Keep points outside the optimized self-BA subset.",
    )
    parser.add_argument(
        "--overwrite_self_ba_output",
        action="store_true",
        help="Remove an existing self-BA output directory before running.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No jpg/png images found in {image_dir}")
    return images


def select_images(images: list[Path], max_images: int | None, strategy: str) -> list[Path]:
    if max_images is None or max_images >= len(images):
        return images
    if max_images <= 0:
        raise ValueError("--max_images must be positive.")
    if strategy == "first":
        return images[:max_images]

    if max_images == 1:
        return [images[0]]
    indices = [round(i * (len(images) - 1) / (max_images - 1)) for i in range(max_images)]
    return [images[index] for index in indices]


def rel_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)


def prepare_scene(args: argparse.Namespace) -> Path:
    if args.image_dir and not args.output_scene_dir:
        raise ValueError("--image_dir requires --output_scene_dir so VGGT has a scene directory.")

    if args.scene_dir:
        input_scene = resolve_path(args.scene_dir)
        input_image_dir = input_scene / "images"
    else:
        input_scene = None
        input_image_dir = resolve_path(args.image_dir)

    selected = select_images(collect_images(input_image_dir), args.max_images, args.sample_strategy)

    if args.output_scene_dir:
        output_scene = resolve_path(args.output_scene_dir)
        output_image_dir = output_scene / "images"
        output_image_dir.mkdir(parents=True, exist_ok=True)

        existing = [path for path in output_image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        for path in existing:
            path.unlink()
        for image in selected:
            rel_symlink(image, output_image_dir / image.name)
        return output_scene

    if input_scene is None:
        raise ValueError("Internal error: input scene is missing.")
    if len(selected) != len(collect_images(input_image_dir)):
        raise ValueError("--max_images without --output_scene_dir would partially modify the input scene.")
    return input_scene


def remove_sparse_if_requested(scene_dir: Path, overwrite: bool) -> None:
    sparse_dir = scene_dir / "sparse"
    if not sparse_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"{sparse_dir} already exists. Use --overwrite_sparse to replace it, "
            "or choose a new --output_scene_dir."
        )
    shutil.rmtree(sparse_dir)


def run_vggt(scene_dir: Path, args: argparse.Namespace) -> None:
    sys.path.insert(0, str(VGGT_ROOT))
    from demo_colmap import demo_fn
    import pycolmap
    import torch

    vggt_args = argparse.Namespace(
        scene_dir=str(scene_dir),
        seed=args.seed,
        use_ba=args.use_ba,
        max_reproj_error=args.max_reproj_error,
        shared_camera=args.shared_camera,
        camera_type=args.camera_type,
        vis_thresh=args.vis_thresh,
        query_frame_num=args.query_frame_num,
        max_query_pts=args.max_query_pts,
        fine_tracking=not args.no_fine_tracking,
        conf_thres_value=args.conf_thres_value,
    )

    original_bundle_adjustment = pycolmap.bundle_adjustment
    if args.skip_builtin_ba:
        if not args.use_ba:
            raise ValueError("--skip_builtin_ba only makes sense together with --use_ba.")

        def skip_bundle_adjustment(reconstruction, options):
            print("Skipping VGGT demo_colmap.py built-in pycolmap.bundle_adjustment.")
            return None

        pycolmap.bundle_adjustment = skip_bundle_adjustment

    try:
        with torch.no_grad():
            demo_fn(vggt_args)
    finally:
        pycolmap.bundle_adjustment = original_bundle_adjustment


def check_outputs(scene_dir: Path) -> None:
    sparse_dir = scene_dir / "sparse"
    required = ["cameras.bin", "images.bin", "points3D.bin", "points.ply"]
    missing = [name for name in required if not (sparse_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"VGGT finished but sparse output is incomplete: {missing}")
    print("VGGT sparse output:")
    for name in required:
        path = sparse_dir / name
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size} bytes")

    try:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
        track_lengths = [len(point.track.elements) for point in reconstruction.points3D.values()]
        usable = sum(length >= 2 for length in track_lengths)
        mean_track = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
        print(f"  cameras: {len(reconstruction.cameras)}")
        print(f"  images: {len(reconstruction.images)}")
        print(f"  points3D: {len(reconstruction.points3D)}")
        print(f"  mean_track_length: {mean_track:.3f}")
        print(f"  points usable for BA (track_len>=2): {usable}")
    except Exception as exc:
        print(f"  sparse stats skipped: {exc}")


def run_self_ba(scene_dir: Path, args: argparse.Namespace) -> Path:
    output_sparse = scene_dir / args.self_ba_output_name
    if output_sparse.exists():
        if not args.overwrite_self_ba_output:
            raise FileExistsError(
                f"{output_sparse} already exists. Use --overwrite_self_ba_output to replace it, "
                "or choose a different --self_ba_output_name."
            )
        shutil.rmtree(output_sparse)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_self_ba.py"),
        "--input_sparse",
        str(scene_dir / "sparse"),
        "--output_sparse",
        str(output_sparse),
        "--max_points",
        str(args.self_ba_max_points),
        "--max_nfev",
        str(args.self_ba_max_nfev),
        "--loss",
        args.self_ba_loss,
        "--f_scale",
        str(args.self_ba_f_scale),
    ]
    if args.self_ba_max_initial_reproj_error_px > 0:
        cmd.extend(["--max_initial_reproj_error_px", str(args.self_ba_max_initial_reproj_error_px)])
    if args.self_ba_prune_reproj_error_px > 0:
        cmd.extend(["--prune_reproj_error_px", str(args.self_ba_prune_reproj_error_px)])
    if args.self_ba_optimize_shared_intrinsics:
        cmd.append("--optimize_shared_intrinsics")
    if args.self_ba_keep_unoptimized_points:
        cmd.append("--keep_unoptimized_points")

    print("Running self BA:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_sparse


def main() -> None:
    args = parse_args()
    if not VGGT_ROOT.exists():
        raise FileNotFoundError(f"VGGT repository not found: {VGGT_ROOT}")

    scene_dir = prepare_scene(args)
    remove_sparse_if_requested(scene_dir, args.overwrite_sparse)

    print(f"Running VGGT COLMAP export for: {scene_dir.relative_to(ROOT)}")
    print("Note: this loads the VGGT-1B checkpoint and is intended for a GPU environment.")
    run_vggt(scene_dir, args)
    check_outputs(scene_dir)
    if args.run_self_ba:
        output_sparse = run_self_ba(scene_dir, args)
        print(f"Self BA sparse output: {output_sparse.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
