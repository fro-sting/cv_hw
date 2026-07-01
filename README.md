# CV Final Project

VGGT + self bundle adjustment + 3D Gaussian Splatting pipeline for the computer vision final project.

The GitHub repository keeps only code, documentation, and the `vggt` submodule pointer. Course data, generated reconstructions, Nerfstudio outputs, logs, point clouds, and slides are local artifacts and are ignored by default.

## Environment

Python 3.10 is recommended.

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r vggt/requirements.txt
```

Optional 3DGS training/export requires Nerfstudio with a working CUDA `gsplat` build:

```bash
pip install nerfstudio
```

## Data

Put the provided course data in this layout:

```text
大作业数据/
├── 数据1-人体/
├── 数据2-人体/
└── 数据3-场景.mp4
```

Create the standard project data entry points:

```bash
python scripts/setup_data.py
```

This creates:

```text
data/human1/images
data/human1/masks
data/human2/images
data/human2/masks
data/scene3/source_video.mp4
```

## Main Pipeline

Run VGGT COLMAP export and the self-implemented BA pass:

```bash
python scripts/run_vggt_colmap.py \
  --scene_dir data/human1 \
  --output_scene_dir data/human1_vggt_self_ba \
  --max_images 16 \
  --use_ba \
  --skip_builtin_ba \
  --shared_camera \
  --run_self_ba \
  --self_ba_output_name sparse_self_ba \
  --self_ba_max_points 12000 \
  --self_ba_max_nfev 120 \
  --self_ba_max_initial_reproj_error_px 64 \
  --self_ba_prune_reproj_error_px 12 \
  --self_ba_optimize_shared_intrinsics
```

Check sparse reconstruction metrics:

```bash
python scripts/check_sparse.py data/human1_vggt_self_ba/sparse_self_ba
```

## Masked Human Experiment

Create foreground-aware images:

```bash
python scripts/make_masked_scene.py \
  --image_dir data/human1/images \
  --mask_dir data/human1/masks \
  --output_scene_dir data/human1_mask_black \
  --mode black
```

Then rerun the main pipeline with:

```bash
python scripts/run_vggt_colmap.py \
  --scene_dir data/human1_mask_black \
  --output_scene_dir data/human1_mask_black_vggt_self_ba \
  --max_images 16 \
  --use_ba \
  --skip_builtin_ba \
  --shared_camera \
  --run_self_ba \
  --self_ba_output_name sparse_self_ba \
  --self_ba_max_points 12000 \
  --self_ba_max_nfev 120 \
  --self_ba_max_initial_reproj_error_px 64 \
  --self_ba_prune_reproj_error_px 12 \
  --self_ba_optimize_shared_intrinsics
```

## Scene Video Experiment

Select quality-aware keyframes from the scene video:

```bash
python scripts/filter_views.py \
  --video data/scene3/source_video.mp4 \
  --output_dir data/scene3_quality_24 \
  --max_frames 24 \
  --strategy quality_segmented \
  --fps 2
```

Run the main pipeline on the selected frames:

```bash
python scripts/run_vggt_colmap.py \
  --scene_dir data/scene3_quality_24 \
  --max_images 24 \
  --use_ba \
  --skip_builtin_ba \
  --shared_camera \
  --run_self_ba \
  --self_ba_output_name sparse_self_ba_intrinsics_pruned \
  --self_ba_max_points 12000 \
  --self_ba_max_nfev 120 \
  --self_ba_max_initial_reproj_error_px 64 \
  --self_ba_prune_reproj_error_px 12 \
  --self_ba_optimize_shared_intrinsics
```

## Optional 3DGS

Prepare Nerfstudio data:

```bash
python scripts/prepare_nerfstudio_data.py \
  --image_dir data/human1/images \
  --mask_dir data/human1/masks \
  --sparse_dir data/human1_vggt_self_ba/sparse_self_ba \
  --output_dir data/human1_ns \
  --make_images_2 \
  --apply_masks_to_images \
  --background_color 0,0,0 \
  --overwrite
```

Train and export:

```bash
bash scripts/train_splatfacto.sh --data data/human1_ns --experiment-name human1 --max-num-iterations 30000
bash scripts/export_splat.sh --search-dir outputs/3dgs/human1 --output gs_batch/human1
```

## Repository Layout

```text
scripts/          Project pipeline scripts
vggt/             VGGT upstream submodule
requirements.txt  Python dependencies for project scripts
data/             Local/generated data, ignored except data/README.md
outputs/          Local experiment outputs, ignored
report/           Local report artifacts, ignored
gs_batch/         Local exported Gaussian splats, ignored
```
