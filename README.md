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
