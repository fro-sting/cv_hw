# CV Final Project

This repo is for the VGGT + Bundle Adjustment + 3D Gaussian Splatting final project.

Current stage: keep the dataset layout simple and debuggable.

## 1. Prepare Data Entrypoints

The original provided data stays in `大作业数据/`. Create a lightweight `data/`
folder with symlinks:

```bash
python scripts/setup_data.py  
```

Result:

```text
data/human1/images/        # rgb_*.png links
data/human1/masks/         # msk_*.png links
data/human2/images/        # rgb_*.png links
data/human2/masks/         # msk_*.png links
data/scene3/source_video.mp4
```

## 2. Dataset Roles

- `human1`: RGB multi-view human images plus masks.
- `human2`: RGB multi-view human images plus masks.
- `scene3`: scene video; extract frames later before running VGGT.

## 3. Lightweight VGGT Smoke Test

On a CPU-only or low-memory machine, first run the smoke test. It checks the
data layout, imports VGGT, optionally runs preprocessing, and can instantiate
the model with random weights without downloading the 1B checkpoint:

```bash
conda run -n vggt python scripts/run_vggt_smoke.py \
  --scene_dir data/human1_vggt_test \
  --preprocess \
  --init_model
```

The full VGGT COLMAP export is still much heavier because it downloads and runs
the pretrained 1B model:

```bash
conda run -n vggt python vggt/demo_colmap.py --scene_dir data/human1_vggt_test
```

## 4. VGGT + Self BA + 3DGS Prep

For better 3DGS initialization, export VGGT with tracks, skip VGGT's final
pycolmap BA, and run the project self BA as a cleanup pass:

```bash
conda run -n vggt python scripts/run_vggt_colmap.py \
  --scene_dir data/human1 \
  --output_scene_dir data/human1_vggt_self_ba \
  --max_images 16 \
  --use_ba \
  --skip_builtin_ba \
  --shared_camera \
  --run_self_ba \
  --self_ba_max_points 5000 \
  --self_ba_max_initial_reproj_error_px 64 \
  --self_ba_prune_reproj_error_px 12
```

This writes the VGGT initialization to `sparse/` and the optimized, pruned
self-BA model to `sparse_self_ba/`. By default the self-BA output keeps only the
3D points that were actually optimized, so `points.ply` is less likely to carry
unoptimized noisy points into later visualization or splat initialization. Use
`--self_ba_keep_unoptimized_points` if you need the old behavior.

Then prepare a nerfstudio-style data folder. For human scenes, masks can be
copied and optionally applied as a black background:

```bash
conda run -n vggt python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/human1_vggt_self_ba/images \
  --mask_dir data/human1/masks \
  --sparse_dir data/human1_vggt_self_ba/sparse_self_ba \
  --output_dir data/human1_ns_self_ba \
  --make_images_2 \
  --apply_masks_to_images \
  --background_color 0,0,0
```

## VGGT Improvement Experiments

This project now separates the VGGT improvement study into two tracks:

- **Human foreground-aware VGGT ablation**: the human subject already occupies
  most of the views, so the main variable is not cropping. Use the provided
  masks as foreground-aware inputs and compare VGGT parameters such as query
  frame count, fine tracking, and max query points.
- **Indoor video quality-aware keyframe selection**: `scene3` is a video
  reconstruction task, so the main variable is which frames enter VGGT. Compare
  fixed interval, uniform, and quality-aware keyframes before spending time on
  dense reconstruction.

Human mask input preparation:

```bash
python scripts/make_masked_scene.py \
  --image_dir data/human1/images \
  --mask_dir data/human1/masks \
  --output_scene_dir data/human1_mask_blur \
  --mode blur
```

Human VGGT ablation commands are printed by default:

```bash
bash scripts/run_human_vggt_ablation.sh human1
```

Run them for real with:

```bash
bash scripts/run_human_vggt_ablation.sh human1 --execute
```

The human ablation writes sparse metrics to
`outputs/tables/human_vggt_ablation.csv` and foreground consistency metrics to
`outputs/tables/human_vggt_mask_consistency.csv`.

Indoor keyframe selection for `scene3` is handled by `scripts/10_filter_views.py`
and wrapped by:

```bash
bash scripts/run_scene3_vggt_ablation.sh scene3
bash scripts/run_scene3_vggt_ablation.sh scene3 --execute
```

This compares `fixed_interval_24`, `uniform_24`, `quality_24`, `quality_48`,
and `quality_96`. The wrapper uses VGGT's BA track-building path, skips the
final pycolmap BA, then runs the project self-implemented BA and writes sparse
metrics from `sparse_self_ba_intrinsics_pruned/` to
`outputs/tables/scene3_vggt_ablation.csv`.

For both tracks, compare the sparse and BA-level metrics first:

```bash
python scripts/check_sparse.py data/human1_vggt_mask_blur_q16/sparse
python scripts/eval_sparse_mask_consistency.py \
  --sparse_dir data/human1_vggt_mask_blur_q16/sparse \
  --image_dir data/human1_vggt_mask_blur_q16/images \
  --mask_dir data/human1/masks \
  --csv outputs/tables/human_vggt_mask_consistency.csv \
  --append_csv
```

Only run 3DGS for the best sparse configuration from each track.
