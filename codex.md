# Codex Project Context

## Project

This repository is for the Computer Vision final project:

**VGGT + self-implemented Bundle Adjustment + 3D Gaussian Splatting**

The goal is to reconstruct a scene from uncalibrated multi-view images. VGGT provides initial camera parameters and point cloud, a custom Bundle Adjustment module refines camera poses and 3D points, and 3D Gaussian Splatting provides optimized Gaussian representation plus real-time interactive rendering.

Primary source document: `init.md`.

## Current Workspace State

Current root:

```text
/home/frosting/26spring/大作业
```

Existing important files and directories:

```text
init.md
大作业.pptx
vggt/
大作业数据/
```

Existing data:

```text
大作业数据/
├── 数据1-人体/
│   ├── rgb_0000.png ... rgb_0015.png
│   └── msk_0000.png ... msk_0015.png
├── 数据2-人体/
│   ├── rgb_0000.png ... rgb_0015.png
│   └── msk_0000.png ... msk_0015.png
└── 数据3-场景.mp4
```

Existing VGGT source is already present at:

```text
vggt/
```

The repository root is not currently a Git repository, although `vggt/` itself has its own `.git`.

Existing conda environment:

```text
vggt
```

The user has already created a conda environment named `vggt`. Prefer using this environment for the project. Install BA-related packages, data-processing packages, and later 3DGS/nerfstudio dependencies into this existing `vggt` environment when feasible. Do not create a separate `cvfinal`/`ns` environment unless the dependency conflict is concrete and cannot be resolved cleanly.

Current implementation style preference:

- Keep the repository simple and stage-based.
- Keep `大作业数据/` as the original data source.
- Use `data/` only as a lightweight standardized entrypoint, preferably via symlinks rather than copying large files.
- Do not pre-create complicated BA/3DGS code before the previous stage has been tested.

Current compute strategy:

- The local machine/server is relatively weak for full VGGT/3DGS runs.
- Use the local machine only for smoke tests, command validation, file-format checks, and small visual sanity checks.
- Full VGGT on all frames, full BA, and 3DGS training should be run later on a stronger server.
- Important long-running commands and visual checks must be run manually by the user. Codex should prepare commands/scripts and explain expected outputs, but should not automatically launch heavy VGGT/BA/3DGS jobs unless the user explicitly asks in that turn.
- Keep every stage debuggable: one command, one expected output folder, one manual check.

Report artifact convention:

- `outputs/` remains ignored by Git and can hold large temporary training results.
- Anything needed for the final report or for moving results between the server and laptop should be copied into `report/artifacts/<experiment_name>/`.
- When giving 3DGS training commands, prefer adding:

```bash
--archive-dir report/artifacts/<experiment_name>
```

to `scripts/07_train_splatfacto.sh`. This saves the training command, `train.log`, latest `config.yml`, and `dataparser_transforms.json`.

- When giving Gaussian export commands, prefer adding:

```bash
--archive-dir report/artifacts/<experiment_name>
```

to `scripts/08_export_splat.sh`. This saves the export command, `export.log`, export config, and exported `.ply` files.
- For BA/check_sparse/report evidence, use:

```bash
python scripts/archive_experiment.py \
  --name <experiment_name> \
  --sparse_dir <sparse_dir> \
  --metrics_csv <optional_metrics_csv> \
  --data_dir <optional_nerfstudio_data_dir> \
  --config <optional_config_yml> \
  --export_dir <optional_export_dir>
```

This keeps the final evidence outside ignored `outputs/`, in a folder that can be copied between machines.

Current local smoke-test status:

- `data/human1/images/` and `data/human2/images/` are standard RGB inputs.
- `data/human1/masks/` and `data/human2/masks/` are mask inputs for later mask experiments.
- `data/scene3/images/` has extracted frames from `data/scene3/source_video.mp4`.
- `data/human1_vggt_4/sparse/` exists and contains a successful small VGGT output (`cameras.bin`, `images.bin`, `points3D.bin`, `points.ply`).
- Full/large VGGT runs on the local RTX 4060 Laptop 8GB are slow; 8 or 16 images can take too long for interactive debugging.

## Course Requirement Mapping

The implementation should support the grading points from `init.md`:

1. Use VGGT to estimate camera parameters and initial point cloud.
2. Implement Bundle Adjustment to refine camera extrinsics and 3D points.
3. Implement or integrate 3D Gaussian Splatting optimization and show real-time interactive rendering.
4. Investigate and implement a VGGT improvement method.
5. Prepare PPT/demo materials with experiments and analysis.

The required experimental pipelines are:

```text
Pipeline A:
multi-view images -> VGGT -> 3DGS

Pipeline B:
multi-view images -> VGGT -> custom BA -> 3DGS

Pipeline C:
multi-view images -> input filtering / keyframe selection -> VGGT -> custom BA -> 3DGS
```

Pipeline A is the baseline. Pipeline B is used to analyze whether BA improves 3DGS. Pipeline C is the practical VGGT improvement method.

## Planned Repository Layout

Initialize toward this structure:

```text
.
├── README.md
├── PROJECT_INIT.md
├── codex.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── raw_videos/
│   ├── scene1/
│   │   ├── images/
│   │   ├── sparse_vggt/
│   │   ├── sparse_ba/
│   │   ├── sparse_filtered/
│   │   └── splits/
│   └── demo_scene/
│       └── images/
├── external/
│   ├── vggt/
│   ├── gaussian-splatting/
│   └── nerfstudio/
├── scripts/
│   ├── 00_extract_frames.py
│   ├── 01_run_vggt.sh
│   ├── 02_check_colmap_output.py
│   ├── 03_parse_colmap.py
│   ├── 04_bundle_adjustment.py
│   ├── 05_visualize_reconstruction.py
│   ├── 06_prepare_nerfstudio_data.py
│   ├── 07_train_splatfacto.sh
│   ├── 08_export_splat.sh
│   ├── 09_eval_rendering.py
│   ├── 10_filter_views.py
│   ├── 11_run_all_baseline.sh
│   ├── 12_run_all_ba.sh
│   └── 13_run_all_filtered.sh
├── src/
│   ├── cvfinal/
│   │   ├── __init__.py
│   │   ├── colmap_io.py
│   │   ├── geometry.py
│   │   ├── ba.py
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   ├── view_filter.py
│   │   └── utils.py
│   └── tests/
│       ├── test_geometry.py
│       ├── test_ba_residual.py
│       └── test_colmap_io.py
├── outputs/
│   ├── vggt_raw/
│   ├── ba_refined/
│   ├── 3dgs_raw/
│   ├── 3dgs_ba/
│   ├── 3dgs_filtered/
│   ├── figures/
│   ├── logs/
│   └── tables/
└── report/
    ├── figures/
    ├── final_ppt_outline.md
    └── final_report_notes.md
```

Because `vggt/` already exists in the root, either move or symlink it into `external/vggt/`, or adapt scripts to detect both paths:

```text
vggt/
external/vggt/
```

Prefer path detection over assuming only one layout.

## Implementation Stages

### Stage 1: Project Skeleton

Create:

- `README.md`
- `PROJECT_INIT.md`
- `requirements.txt`
- `environment.yml`
- `.gitignore`
- `data/`, `scripts/`, `src/cvfinal/`, `src/tests/`, `outputs/`, `report/`

Do not create empty placeholder-only code. Each script should have an argparse interface and clear failure messages.

### Stage 2: Data Preparation

Implement `scripts/00_extract_frames.py`.

Required command:

```bash
python scripts/00_extract_frames.py \
  --video data/raw_videos/scene1.mp4 \
  --output data/scene1/images \
  --fps 2 \
  --max_frames 120 \
  --resize_long_edge 1600
```

Features:

- Extract frames from video.
- Save as `000001.jpg`, `000002.jpg`, ...
- Support FPS sampling, max frame count, and optional resize.
- Create output directories automatically.

Use existing `大作业数据/数据3-场景.mp4` as the likely initial raw scene video.

### Stage 3: VGGT

Use VGGT to produce COLMAP-style sparse reconstruction.

Expected command shape:

```bash
cd vggt
python demo_colmap.py --scene_dir ../data/scene1
cd ..
```

or, if moved:

```bash
cd external/vggt
python demo_colmap.py --scene_dir ../../data/scene1
cd ../..
```

Expected outputs:

```text
data/scene1/sparse/cameras.bin
data/scene1/sparse/images.bin
data/scene1/sparse/points3D.bin
```

Then standardize to:

```text
data/scene1/sparse_vggt/
```

Scripts should detect common sparse layouts:

- `sparse/`
- `sparse/0/`
- `sparse_vggt/`

### Stage 4: COLMAP IO and Checks

Implement `src/cvfinal/colmap_io.py`.

Required APIs:

```python
read_cameras_binary(path)
read_images_binary(path)
read_points3D_binary(path)
write_cameras_binary(path, cameras)
write_images_binary(path, images)
write_points3D_binary(path, points3D)
```

If binary IO is too slow to implement at first, support text IO and use `pycolmap` or COLMAP CLI conversion where available. Fail explicitly with `NotImplementedError` when unsupported paths are reached.

Implement `scripts/02_check_colmap_output.py`.

Required output:

- camera count
- image count
- 3D point count
- mean track length
- missing image checks
- invalid camera checks
- initial mean reprojection error

### Stage 5: Geometry

Implement `src/cvfinal/geometry.py`.

Required functions:

```python
qvec_to_rotmat(qvec)
rotmat_to_qvec(R)
rodrigues_to_rotmat(rvec)
rotmat_to_rodrigues(R)
project_points(points3d, rvec, tvec, intrinsics)
```

Projection convention:

```text
X_c = R X_w + t
u = fx * X_c / Z_c + cx
v = fy * Y_c / Z_c + cy
```

Only points with `Z_c > 0` are valid.

Add tests in `src/tests/test_geometry.py`.

### Stage 6: Bundle Adjustment

Implement `src/cvfinal/ba.py` and `scripts/04_bundle_adjustment.py`.

BA requirements:

- Fix intrinsics.
- Fix the first camera pose to avoid gauge freedom.
- Optimize remaining camera extrinsics.
- Optimize selected 3D points.
- Use Rodrigues vectors for rotation parameters.
- Use robust loss, preferably Huber or soft-L1.
- Use `scipy.optimize.least_squares`.
- Support `min_track_len`, `max_points`, `loss`, `max_nfev`.

Required APIs:

```python
build_ba_problem(cameras, images, points3d, min_track_len=3, max_points=None)
ba_residuals(params, problem)
run_bundle_adjustment(problem, loss="huber", max_nfev=100)
```

Required command:

```bash
python scripts/04_bundle_adjustment.py \
  --input_sparse data/scene1/sparse_vggt \
  --image_dir data/scene1/images \
  --output_sparse data/scene1/sparse_ba \
  --min_track_len 3 \
  --max_points 30000 \
  --loss huber \
  --max_nfev 100
```

Required outputs:

```text
outputs/logs/ba_scene1.log
outputs/tables/ba_metrics_scene1.csv
data/scene1/sparse_ba/
```

CSV columns:

```csv
scene,method,num_images,num_points,num_observations,reproj_before,reproj_after,time_sec
```

### Stage 7: Visualization

Implement `scripts/05_visualize_reconstruction.py`.

Required command:

```bash
python scripts/05_visualize_reconstruction.py \
  --sparse_dir data/scene1/sparse_vggt \
  --output outputs/figures/vggt_reconstruction.png
```

Visualize:

- point cloud
- camera centers
- camera directions
- camera trajectory
- optionally reprojection error color

### Stage 8: Nerfstudio / 3DGS

Prefer nerfstudio `splatfacto` for the main 3DGS route.

Implement `scripts/06_prepare_nerfstudio_data.py`.

Required command:

```bash
python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/scene1/images \
  --sparse_dir data/scene1/sparse_ba \
  --output_dir data/scene1_ns_ba
```

Required output layout:

```text
data/scene1_ns_ba/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

Train command:

```bash
ns-train splatfacto \
  --data data/scene1_ns_ba \
  --output-dir outputs/3dgs_ba
```

Implement:

- `scripts/07_train_splatfacto.sh`
- `scripts/08_export_splat.sh`

Scripts should call `ns-train splatfacto --help` or fail clearly if nerfstudio is not installed.

### Stage 9: VGGT Improvement Method

Use input image filtering / keyframe selection as the improvement method.

Implement `src/cvfinal/view_filter.py` and `scripts/10_filter_views.py`.

Required command:

```bash
python scripts/10_filter_views.py \
  --input_dir data/scene1/images \
  --output_dir data/scene1_filtered/images \
  --sharpness_quantile 0.15 \
  --min_hist_diff 0.05 \
  --max_hist_diff 0.95 \
  --max_images 80
```

Features:

- sharpness score via variance of Laplacian
- histogram difference for redundancy
- anomalous frame marking
- CSV output
- score plot output

Required CSV:

```csv
image,sharpness,hist_diff,kept,reason
```

### Stage 10: Evaluation and Report Assets

Implement `src/cvfinal/metrics.py` and `scripts/09_eval_rendering.py`.

Minimum metrics:

- mean reprojection error
- median reprojection error
- p90 reprojection error
- PSNR if rendered/ground truth pairs are available
- SSIM if feasible

Expected experiment tables:

```text
outputs/tables/ba_metrics_scene1.csv
outputs/tables/render_metrics_raw.csv
outputs/tables/render_metrics_ba.csv
outputs/tables/filter_metrics_scene1.csv
```

## Dependencies

Base `requirements.txt` should include:

```text
numpy
scipy
opencv-python
pillow
matplotlib
tqdm
rich
pyyaml
imageio
imageio-ffmpeg
open3d
trimesh
scikit-image
scikit-learn
pandas
plotly
viser
tyro
pycolmap
pytest
```

Recommended conda environment:

```yaml
name: cvfinal
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - pandas
  - matplotlib
  - pillow
  - pyyaml
  - tqdm
  - pytest
  - pip:
      - opencv-python
      - imageio
      - imageio-ffmpeg
      - rich
      - open3d
      - trimesh
      - scikit-image
      - scikit-learn
      - plotly
      - viser
      - tyro
      - pycolmap
```

PyTorch should be installed separately according to the CUDA version.

## Minimum Viable Run

The minimum successful version should produce:

```text
data/scene1/images/
data/scene1/sparse_vggt/
data/scene1/sparse_ba/
outputs/tables/ba_metrics_scene1.csv
outputs/figures/vggt_reconstruction.png
outputs/figures/ba_reconstruction.png
outputs/3dgs_ba/
```

If full 3DGS training is too slow, still complete VGGT, BA, reprojection metrics, visualization, and prepare the nerfstudio data directory.

## Experiment Questions

The PPT and report should answer:

1. Does BA reduce reprojection error?
2. Does lower reprojection error improve 3DGS visual quality?
3. Does image filtering improve VGGT initialization quality or speed?
4. What are the tradeoffs between fewer frames, coverage, quality, and training time?
5. When does BA hurt instead of help?

Conclusions must follow the measured results. Do not assume BA or filtering always improves final rendering.

## Development Rules for Codex

- Prefer small, runnable stages.
- Use existing local patterns once code exists.
- Create output directories automatically.
- Add explicit errors for unsupported paths.
- Use deterministic seeds for random sampling.
- Avoid silently skipping missing files.
- Do not edit `vggt/` internals unless absolutely necessary.
- Keep scripts runnable from the repository root.
- Prefer `rg` for search.
- Use `apply_patch` for manual edits.
- Do not delete user data or generated outputs without explicit permission.
- Do not auto-run long VGGT/BA/3DGS commands. For important commands and visual checks, provide the exact command and let the user run it manually.
- When a command is expected to be slow, state the expected runtime/risk and define a small smoke-test alternative first.

## Immediate Next Step

The current goal is not to finish all experiments locally. The goal is to make sure every step of the pipeline has a small, inspectable example before moving full runs to a stronger server.

Current local TODO order:

1. Manually inspect data layout.

```bash
find data -maxdepth 3 -type d | sort
find data/human1/images -maxdepth 1 -type l | wc -l
find data/human2/images -maxdepth 1 -type l | wc -l
find data/scene3/images -maxdepth 1 -type f | wc -l
```

Expected:

```text
human1 images: 16
human2 images: 16
scene3 images: 60 or another chosen extracted-frame count
```

2. Manually run a lightweight VGGT import/preprocess smoke test, not full inference.

```bash
conda run -n vggt --no-capture-output python scripts/run_vggt_smoke.py \
  --scene_dir data/human1 \
  --max_images 4 \
  --preprocess
```

Expected:

```text
VGGT import: ok
preprocess: images=(4, 3, 518, 518)
smoke test: ok
```

3. Manually inspect the already-created small VGGT output.

```bash
find data/human1_vggt_4/sparse -maxdepth 1 -type f | sort
```

Expected:

```text
cameras.bin
images.bin
points3D.bin
points.ply
```

4. Manually visualize the small VGGT point cloud before writing BA.

Use one of these options:

```bash
conda run -n vggt --no-capture-output python -c "import trimesh; m=trimesh.load('data/human1_vggt_4/sparse/points.ply'); print(m)"
```

or open `data/human1_vggt_4/sparse/points.ply` in MeshLab / CloudCompare / Open3D viewer manually.

Visual check:

- Point cloud is non-empty.
- Shape roughly matches a human/object region.
- It is not a single collapsed plane or random explosion.

5. Next code stage: write only a tiny COLMAP sparse checker.

The checker should read `data/human1_vggt_4/sparse/` and print:

```text
num_cameras
num_images
num_points3D
has_points_ply
```

Do not implement BA yet.

6. Server stage later:

After local smoke tests pass, move to the stronger server and run full VGGT for:

```text
data/human1
data/human2
data/scene3
```

Run full commands manually on the server, for example:

```bash
cd vggt
conda run -n vggt --no-capture-output python demo_colmap.py --scene_dir ../data/human1
conda run -n vggt --no-capture-output python demo_colmap.py --scene_dir ../data/human2
conda run -n vggt --no-capture-output python demo_colmap.py --scene_dir ../data/scene3
```

Only after full VGGT outputs are checked should Codex implement custom BA.
