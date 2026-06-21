# CV Final Project Initialization: VGGT + Bundle Adjustment + 3D Gaussian Splatting

> 用途：本文件用于初始化《计算机视觉期末大作业》的代码仓库。  
> 项目目标：给定一个无相机标定参数的多视角图像场景，使用 VGGT 估计相机参数和初始点云，实现 Bundle Adjustment 优化相机外参与点云，再用 3D Gaussian Splatting 进行高斯点云优化与实时交互渲染，并完成实验对比与答辩材料。

---

## 0. 作业要求对应关系

课程要求如下：

1. 使用 VGGT 求相机参数和初步点云：3 分
2. 编程实现 Bundle Adjustment 优化相机外参和点云：4 分
3. 编程实现 3D Gaussian Splatting 优化，并展示实时交互渲染：4 分
4. 调研 VGGT 改进方法，提高重建精度或速度：3 分
5. PPT 制作与答辩：6 分  
   - 技术实现讲解与展示：2 分
   - 分析 BA 是否影响高斯泼溅效果：2 分
   - 对提出的改进方法进行实验分析：1 分
   - 未来可开展的研究方向：1 分

本项目要跑通三条 pipeline：

```text
Pipeline A:
多视角图像 -> VGGT -> 3DGS

Pipeline B:
多视角图像 -> VGGT -> 自实现 BA -> 3DGS

Pipeline C:
多视角图像 -> 输入筛选 / 关键帧选择 -> VGGT -> 自实现 BA -> 3DGS
```

其中：

- Pipeline A 用作 baseline；
- Pipeline B 用于分析 BA 对 3DGS 的影响；
- Pipeline C 用于体现 VGGT 改进方法，并做实验对比。

---

## 1. 推荐技术路线

### 1.1 VGGT

VGGT 用于从无标定多视角图像直接估计：

- 相机内参 `K`
- 相机外参 `R, t`
- 初始 3D 点云
- 深度图 / point map / tracks

本项目中，VGGT 输出作为后续 BA 和 3DGS 的初始化结果。

### 1.2 Bundle Adjustment

自实现 BA 模块，不直接把 COLMAP 或 VGGT 官方 BA 当成最终答案。

建议优化目标：

```math
\min_{\{R_i,t_i,X_j\}}
\sum_{(i,j)\in \mathcal{O}}
\rho
\left(
\left\|
\pi(K_i,R_i,t_i,X_j)-u_{ij}
\right\|_2^2
\right)
```

其中：

- 固定相机内参 `K_i`；
- 固定第一帧相机外参，避免 gauge freedom；
- 优化其余相机外参和 3D 点；
- 使用 Huber loss 或 soft-L1 loss 抑制错误匹配。

### 1.3 3D Gaussian Splatting

首选使用 nerfstudio 的 `splatfacto`，因为：

- 配置和可视化比官方 3DGS 更方便；
- 支持实时 viewer；
- 可导出 Gaussian `.ply`；
- 可以读取 COLMAP 风格数据。

备选：官方 GraphDeco 3DGS 实现。

---

## 2. 仓库目录结构

Codex 需要初始化如下结构：

```text
cv_final_vggt_3dgs/
├── README.md
├── PROJECT_INIT.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw_videos/
│   ├── scene1/
│   │   ├── images/
│   │   ├── sparse_vggt/
│   │   ├── sparse_ba/
│   │   ├── sparse_filtered/
│   │   └── splits/
│   │       ├── train.txt
│   │       └── test.txt
│   └── demo_scene/
│       └── images/
│
├── external/
│   ├── vggt/
│   ├── gaussian-splatting/
│   └── nerfstudio/
│
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
│
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
│
├── outputs/
│   ├── vggt_raw/
│   ├── ba_refined/
│   ├── 3dgs_raw/
│   ├── 3dgs_ba/
│   ├── 3dgs_filtered/
│   ├── figures/
│   ├── logs/
│   └── tables/
│
└── report/
    ├── figures/
    ├── final_ppt_outline.md
    └── final_report_notes.md
```

---

## 3. 环境配置

### 3.1 基础环境

建议创建一个主环境：

```bash
conda create -n cvfinal python=3.10 -y
conda activate cvfinal
```

安装 PyTorch。根据实际 CUDA 版本选择。若服务器支持 CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

若是 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

检查 GPU：

```bash
python - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
EOF
```

### 3.2 安装基础 Python 依赖

创建 `requirements.txt`：

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
```

安装：

```bash
pip install -r requirements.txt
```

---

## 4. 安装 VGGT

### 4.1 克隆 VGGT

```bash
mkdir -p external
cd external
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
cd ../..
```

如果 `requirements.txt` 与本环境冲突，优先保证 VGGT demo 能运行。

### 4.2 测试 VGGT demo

准备一个小测试目录：

```text
data/demo_scene/images/
├── 0001.jpg
├── 0002.jpg
├── 0003.jpg
└── ...
```

先用 5 到 10 张图测试：

```bash
cd external/vggt
python demo_viser.py --image_folder ../../data/demo_scene/images
cd ../..
```

验收标准：

- 程序能够启动；
- 能看到相机轨迹和点云；
- 没有 CUDA OOM；
- 输出结果不为空。

### 4.3 导出 COLMAP 格式

正式场景目录：

```text
data/scene1/
└── images/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

运行：

```bash
cd external/vggt
python demo_colmap.py --scene_dir ../../data/scene1
cd ../..
```

预期输出：

```text
data/scene1/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

将结果复制到项目内部标准目录：

```bash
mkdir -p data/scene1/sparse_vggt
cp data/scene1/sparse/* data/scene1/sparse_vggt/
```

如果 VGGT 输出目录结构不同，需要在脚本里自动检测：

- `data/scene1/sparse/`
- `data/scene1/sparse/0/`
- `data/scene1/sparse_vggt/`

---

## 5. 数据采集与预处理

### 5.1 拍摄建议

推荐拍摄静态场景：

- 桌面物体；
- 实验室角落；
- 书架；
- 纹理丰富的室内场景。

不建议：

- 白墙；
- 透明玻璃；
- 强反光金属；
- 大面积重复纹理；
- 有明显运动物体的场景。

拍摄要求：

- 使用手机拍摄 80 到 120 张图；
- 或拍摄视频后抽帧；
- 相邻图像保持较大重叠；
- 绕场景慢速移动；
- 曝光尽量稳定；
- 避免过度模糊。

### 5.2 从视频抽帧

`scripts/00_extract_frames.py` 需要实现：

输入：

```bash
python scripts/00_extract_frames.py \
  --video data/raw_videos/scene1.mp4 \
  --output data/scene1/images \
  --fps 2 \
  --max_frames 120
```

功能：

- 从视频抽帧；
- 自动重命名为 `000001.jpg` 格式；
- 支持限制最大帧数；
- 支持按间隔抽帧；
- 输出图像分辨率可选缩放。

建议参数：

```bash
python scripts/00_extract_frames.py \
  --video data/raw_videos/scene1.mp4 \
  --output data/scene1/images \
  --fps 2 \
  --max_frames 120 \
  --resize_long_edge 1600
```

---

## 6. COLMAP 格式读取与检查

### 6.1 `src/cvfinal/colmap_io.py`

需要实现或复用 COLMAP binary/text 读取逻辑：

- `read_cameras_binary(path)`
- `read_images_binary(path)`
- `read_points3D_binary(path)`
- `write_cameras_binary(path, cameras)`
- `write_images_binary(path, images)`
- `write_points3D_binary(path, points3D)`

如果直接实现二进制较复杂，可以先：

1. 用 `pycolmap` 读取；
2. 或调用 COLMAP 工具把 binary 转 text；
3. 自己实现 text 格式读写。

建议优先实现 text 版本，binary 版本可以调用 `pycolmap` 或外部转换。

### 6.2 检查脚本

`scripts/02_check_colmap_output.py` 需要检查：

```bash
python scripts/02_check_colmap_output.py \
  --sparse_dir data/scene1/sparse_vggt \
  --image_dir data/scene1/images
```

输出：

- 相机数量；
- 图像数量；
- 3D 点数量；
- 每张图平均观测点数；
- 是否存在缺图；
- 是否存在无效相机；
- 初始平均重投影误差。

输出示例：

```text
[COLMAP Check]
num_cameras: 1
num_images: 96
num_points3D: 18342
mean_track_length: 4.8
mean_reprojection_error_px: 2.73
status: OK
```

---

## 7. Bundle Adjustment 自实现要求

### 7.1 文件位置

核心代码：

```text
src/cvfinal/geometry.py
src/cvfinal/ba.py
scripts/04_bundle_adjustment.py
```

### 7.2 几何函数

`src/cvfinal/geometry.py` 需要实现：

```python
def qvec_to_rotmat(qvec):
    """COLMAP qvec -> 3x3 rotation matrix."""

def rotmat_to_qvec(R):
    """3x3 rotation matrix -> COLMAP qvec."""

def rodrigues_to_rotmat(rvec):
    """Rodrigues vector -> 3x3 rotation matrix."""

def rotmat_to_rodrigues(R):
    """3x3 rotation matrix -> Rodrigues vector."""

def project_points(points3d, rvec, tvec, intrinsics):
    """
    Project 3D points to 2D.

    Args:
        points3d: [N, 3]
        rvec: [3]
        tvec: [3]
        intrinsics: dict containing fx, fy, cx, cy

    Returns:
        pixels: [N, 2]
        valid_mask: [N]
    """
```

投影公式：

```math
X_c = R X_w + t
```

```math
u = f_x \frac{X_c}{Z_c} + c_x
```

```math
v = f_y \frac{Y_c}{Z_c} + c_y
```

只保留 `Z_c > 0` 的点。

### 7.3 BA 参数化

建议：

- 固定内参；
- 固定第一帧相机；
- 优化第 2 到第 N 帧相机外参；
- 优化被至少 3 个视角观测到的 3D 点；
- 对旋转使用 Rodrigues 向量；
- 对残差使用 Huber / soft-L1 loss。

优化变量排列：

```text
params =
[
  camera_1_rvec, camera_1_tvec,
  camera_2_rvec, camera_2_tvec,
  ...
  point_0_xyz,
  point_1_xyz,
  ...
]
```

其中 `camera_0` 固定不放入优化变量。

### 7.4 BA 残差函数

`src/cvfinal/ba.py` 需要实现：

```python
def build_ba_problem(cameras, images, points3d, min_track_len=3, max_points=None):
    """
    Build observations and parameter arrays for BA.

    Returns:
        camera_params
        points
        camera_indices
        point_indices
        points_2d
        fixed_camera
        metadata
    """

def ba_residuals(params, problem):
    """
    Compute reprojection residuals.

    Returns:
        residuals: [2 * num_observations]
    """

def run_bundle_adjustment(problem, loss="huber", max_nfev=100):
    """
    Run scipy.optimize.least_squares.
    """
```

使用：

```python
from scipy.optimize import least_squares

result = least_squares(
    fun=ba_residuals,
    x0=params0,
    args=(problem,),
    verbose=2,
    x_scale="jac",
    ftol=1e-4,
    xtol=1e-4,
    gtol=1e-4,
    loss="huber",
    f_scale=2.0,
    max_nfev=100,
)
```

### 7.5 BA 运行脚本

命令：

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

脚本要输出：

- 优化前平均重投影误差；
- 优化后平均重投影误差；
- 使用的相机数量；
- 使用的 3D 点数量；
- 使用的观测数量；
- 优化耗时；
- 是否收敛；
- 输出新的 COLMAP sparse 目录。

日志保存到：

```text
outputs/logs/ba_scene1.log
outputs/tables/ba_metrics_scene1.csv
```

CSV 格式：

```csv
scene,method,num_images,num_points,num_observations,reproj_before,reproj_after,time_sec
scene1,vggt_ba,96,30000,124532,2.73,1.81,231.4
```

---

## 8. 重建结果可视化

### 8.1 可视化脚本

`scripts/05_visualize_reconstruction.py` 需要实现：

```bash
python scripts/05_visualize_reconstruction.py \
  --sparse_dir data/scene1/sparse_vggt \
  --output outputs/figures/vggt_reconstruction.png
```

```bash
python scripts/05_visualize_reconstruction.py \
  --sparse_dir data/scene1/sparse_ba \
  --output outputs/figures/ba_reconstruction.png
```

可视化内容：

- 3D 点云；
- 相机中心；
- 相机朝向；
- 相机轨迹；
- 可选：按 reprojection error 给点云着色。

推荐使用：

- Open3D；
- matplotlib 3D；
- plotly。

输出：

```text
outputs/figures/vggt_camera_points.png
outputs/figures/ba_camera_points.png
outputs/figures/reprojection_error_before_after.png
```

---

## 9. 3D Gaussian Splatting：nerfstudio splatfacto 路线

### 9.1 安装 nerfstudio

推荐单独环境。如果共用环境出现依赖冲突，创建新环境：

```bash
conda create -n ns python=3.10 -y
conda activate ns
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio
ns-install-cli
```

检查：

```bash
ns-train --help
```

### 9.2 准备 nerfstudio 数据

nerfstudio 读取 COLMAP 数据时，一般要求目录包含：

```text
data/scene1_ns_raw/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

为 Pipeline A 准备数据：

```bash
python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/scene1/images \
  --sparse_dir data/scene1/sparse_vggt \
  --output_dir data/scene1_ns_raw
```

为 Pipeline B 准备数据：

```bash
python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/scene1/images \
  --sparse_dir data/scene1/sparse_ba \
  --output_dir data/scene1_ns_ba
```

脚本功能：

- 复制图像；
- 创建 `sparse/0/`；
- 复制 `cameras.bin/images.bin/points3D.bin`；
- 检查图片名与 COLMAP images 记录一致；
- 自动生成 train/test split，如果 nerfstudio 需要。

### 9.3 训练 splatfacto

Pipeline A：

```bash
ns-train splatfacto \
  --data data/scene1_ns_raw \
  --output-dir outputs/3dgs_raw
```

Pipeline B：

```bash
ns-train splatfacto \
  --data data/scene1_ns_ba \
  --output-dir outputs/3dgs_ba
```

Pipeline C：

```bash
ns-train splatfacto \
  --data data/scene1_ns_filtered \
  --output-dir outputs/3dgs_filtered
```

如果要限制训练步数：

```bash
ns-train splatfacto \
  --data data/scene1_ns_ba \
  --output-dir outputs/3dgs_ba \
  --max-num-iterations 30000
```

注意：不同 nerfstudio 版本的参数名可能变化。Codex 需要在脚本中优先调用：

```bash
ns-train splatfacto --help
```

并根据实际版本调整参数。

### 9.4 实时交互渲染

训练时 nerfstudio 会启动 viewer 或显示 viewer URL。答辩时需要录屏或现场展示：

- 可绕场景旋转；
- 可缩放；
- 可从未训练视角观察；
- 显示高斯渲染结果。

保存内容：

```text
outputs/figures/3dgs_raw_viewer.png
outputs/figures/3dgs_ba_viewer.png
outputs/figures/3dgs_filtered_viewer.png
outputs/videos/3dgs_demo.mp4
```

### 9.5 导出 Gaussian splat

训练完成后导出：

```bash
ns-export gaussian-splat \
  --load-config outputs/3dgs_ba/scene1/splatfacto/*/config.yml \
  --output-dir outputs/exports/scene1_ba_splat
```

注意：`config.yml` 的实际路径需要脚本自动查找最新目录。

---

## 10. 3D Gaussian Splatting：官方 GraphDeco 备选路线

如果 nerfstudio 不稳定，可以使用官方 3DGS。

### 10.1 克隆官方实现

```bash
cd external
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
pip install -r requirements.txt
cd ../..
```

### 10.2 数据结构

官方 3DGS 通常要求：

```text
data/scene1_3dgs/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

### 10.3 训练

```bash
cd external/gaussian-splatting
python train.py -s ../../data/scene1_3dgs -m ../../outputs/3dgs_official_ba
cd ../..
```

### 10.4 渲染

```bash
cd external/gaussian-splatting
python render.py -m ../../outputs/3dgs_official_ba
cd ../..
```

### 10.5 viewer

官方 3DGS 有 remote viewer 和 real-time viewer，但 SIBR viewer 配置可能比较复杂。除非时间充足，否则优先用 nerfstudio viewer 完成答辩展示。

---

## 11. VGGT 改进方法：输入筛选 / 关键帧选择

本项目选择一个容易落地的改进方法：

```text
改进方法：输入图像筛选 + 关键帧选择
目标：减少模糊图、低重叠图和异常视角，提高 VGGT 初始化质量或整体重建速度。
```

### 11.1 图像清晰度筛选

实现文件：

```text
src/cvfinal/view_filter.py
scripts/10_filter_views.py
```

清晰度指标：

```python
sharpness = variance_of_laplacian(gray_image)
```

剔除：

- sharpness 低于分位数阈值的图片；
- 或低于固定阈值的图片。

### 11.2 图像相似度 / 关键帧筛选

可以用简单策略：

1. 按时间顺序遍历图像；
2. 计算当前图与上一个保留图的颜色直方图差异；
3. 差异太小说明视角变化太小，跳过；
4. 差异太大可能是模糊、跳帧或无重叠，也标记为风险图。

也可以用 ORB 匹配数量：

- 相邻图匹配点太少，可能无重叠；
- 匹配点太多且图像差异很小，可能冗余。

### 11.3 筛选脚本

命令：

```bash
python scripts/10_filter_views.py \
  --input_dir data/scene1/images \
  --output_dir data/scene1_filtered/images \
  --sharpness_quantile 0.15 \
  --min_hist_diff 0.05 \
  --max_hist_diff 0.95 \
  --max_images 80
```

输出：

```text
data/scene1_filtered/
└── images/

outputs/tables/view_filter_scene1.csv
outputs/figures/view_filter_scores.png
```

CSV：

```csv
image,sharpness,hist_diff,kept,reason
000001.jpg,132.5,0.00,true,first_frame
000002.jpg,18.2,0.03,false,blurry
000003.jpg,145.7,0.02,false,redundant
```

### 11.4 筛选后重新跑 VGGT

```bash
cd external/vggt
python demo_colmap.py --scene_dir ../../data/scene1_filtered
cd ../..
```

复制输出：

```bash
mkdir -p data/scene1/sparse_filtered
cp data/scene1_filtered/sparse/* data/scene1/sparse_filtered/
```

然后对 filtered 结果跑 BA 和 3DGS：

```bash
python scripts/04_bundle_adjustment.py \
  --input_sparse data/scene1/sparse_filtered \
  --image_dir data/scene1_filtered/images \
  --output_sparse data/scene1_filtered/sparse_ba \
  --min_track_len 3 \
  --max_points 30000 \
  --loss huber \
  --max_nfev 100
```

```bash
python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/scene1_filtered/images \
  --sparse_dir data/scene1_filtered/sparse_ba \
  --output_dir data/scene1_ns_filtered
```

```bash
ns-train splatfacto \
  --data data/scene1_ns_filtered \
  --output-dir outputs/3dgs_filtered
```

---

## 12. 实验设计

### 12.1 实验一：BA 是否改善 3DGS

比较：

```text
A: VGGT -> 3DGS
B: VGGT -> BA -> 3DGS
```

需要输出表格：

```csv
scene,method,num_images,num_points,reproj_error,psnr,ssim,lpips,train_time_sec,notes
scene1,vggt_raw,96,18342,2.73,22.1,0.74,0.23,1800,baseline
scene1,vggt_ba,96,18102,1.81,23.0,0.77,0.21,1840,less floating artifacts
```

如果无法稳定计算 PSNR/SSIM/LPIPS，则至少输出：

- 重投影误差；
- 训练 loss 曲线截图；
- 同一视角渲染结果对比；
- viewer 截图；
- 是否有浮点、模糊、ghosting。

### 12.2 实验二：输入筛选是否改善 VGGT / 3DGS

比较：

```text
A: 原始图像 -> VGGT -> BA -> 3DGS
C: 筛选图像 -> VGGT -> BA -> 3DGS
```

表格：

```csv
scene,method,num_input_images,num_kept_images,vggt_time_sec,ba_reproj_error,3dgs_train_time_sec,psnr,ssim,notes
scene1,original,96,96,85.3,1.81,1840,23.0,0.77,baseline
scene1,filtered,96,74,61.2,1.65,1460,23.2,0.78,faster and cleaner
```

结论写法需要谨慎：

- 如果筛选后效果变好：说明剔除模糊图和异常视角有利于几何一致性；
- 如果筛选后效果变差：说明删图过多导致覆盖不足，3DGS 视角监督减少；
- 如果速度变快但质量略降：说明存在速度和质量折中。

---

## 13. 评价指标实现

### 13.1 Reprojection Error

必须实现：

```python
mean_error = mean(norm(projected_2d - observed_2d))
median_error = median(norm(projected_2d - observed_2d))
```

输出：

```text
mean_reprojection_error_px
median_reprojection_error_px
p90_reprojection_error_px
```

### 13.2 PSNR / SSIM / LPIPS

如果能划分测试图，则实现：

```bash
python scripts/09_eval_rendering.py \
  --renders outputs/3dgs_ba/renders \
  --gt data/scene1/images \
  --split data/scene1/splits/test.txt \
  --output outputs/tables/render_metrics_ba.csv
```

至少实现 PSNR 和 SSIM。LPIPS 可以选做。

### 13.3 训练耗时

所有主脚本都记录：

- 开始时间；
- 结束时间；
- 总耗时；
- GPU 型号；
- 输入图像数量。

---

## 14. 一键运行脚本

### 14.1 Baseline

`scripts/11_run_all_baseline.sh`

```bash
#!/usr/bin/env bash
set -e

SCENE=${1:-scene1}

python scripts/02_check_colmap_output.py \
  --sparse_dir data/${SCENE}/sparse_vggt \
  --image_dir data/${SCENE}/images

python scripts/05_visualize_reconstruction.py \
  --sparse_dir data/${SCENE}/sparse_vggt \
  --output outputs/figures/${SCENE}_vggt_reconstruction.png

python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/${SCENE}/images \
  --sparse_dir data/${SCENE}/sparse_vggt \
  --output_dir data/${SCENE}_ns_raw

ns-train splatfacto \
  --data data/${SCENE}_ns_raw \
  --output-dir outputs/3dgs_raw
```

### 14.2 BA Pipeline

`scripts/12_run_all_ba.sh`

```bash
#!/usr/bin/env bash
set -e

SCENE=${1:-scene1}

python scripts/04_bundle_adjustment.py \
  --input_sparse data/${SCENE}/sparse_vggt \
  --image_dir data/${SCENE}/images \
  --output_sparse data/${SCENE}/sparse_ba \
  --min_track_len 3 \
  --max_points 30000 \
  --loss huber \
  --max_nfev 100

python scripts/05_visualize_reconstruction.py \
  --sparse_dir data/${SCENE}/sparse_ba \
  --output outputs/figures/${SCENE}_ba_reconstruction.png

python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/${SCENE}/images \
  --sparse_dir data/${SCENE}/sparse_ba \
  --output_dir data/${SCENE}_ns_ba

ns-train splatfacto \
  --data data/${SCENE}_ns_ba \
  --output-dir outputs/3dgs_ba
```

### 14.3 Filtered Pipeline

`scripts/13_run_all_filtered.sh`

```bash
#!/usr/bin/env bash
set -e

SCENE=${1:-scene1}

python scripts/10_filter_views.py \
  --input_dir data/${SCENE}/images \
  --output_dir data/${SCENE}_filtered/images \
  --sharpness_quantile 0.15 \
  --min_hist_diff 0.05 \
  --max_hist_diff 0.95 \
  --max_images 80

cd external/vggt
python demo_colmap.py --scene_dir ../../data/${SCENE}_filtered
cd ../..

python scripts/04_bundle_adjustment.py \
  --input_sparse data/${SCENE}_filtered/sparse \
  --image_dir data/${SCENE}_filtered/images \
  --output_sparse data/${SCENE}_filtered/sparse_ba \
  --min_track_len 3 \
  --max_points 30000 \
  --loss huber \
  --max_nfev 100

python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/${SCENE}_filtered/images \
  --sparse_dir data/${SCENE}_filtered/sparse_ba \
  --output_dir data/${SCENE}_ns_filtered

ns-train splatfacto \
  --data data/${SCENE}_ns_filtered \
  --output-dir outputs/3dgs_filtered
```

---

## 15. README 需要包含的运行说明

Codex 需要创建 `README.md`，内容包括：

```markdown
# CV Final Project: VGGT + BA + 3D Gaussian Splatting

## 1. Setup

```bash
conda create -n cvfinal python=3.10 -y
conda activate cvfinal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Install VGGT

```bash
mkdir -p external
cd external
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
cd ../..
```

## 3. Run VGGT

```bash
cd external/vggt
python demo_colmap.py --scene_dir ../../data/scene1
cd ../..
```

## 4. Run BA

```bash
python scripts/04_bundle_adjustment.py \
  --input_sparse data/scene1/sparse_vggt \
  --image_dir data/scene1/images \
  --output_sparse data/scene1/sparse_ba
```

## 5. Train 3DGS

```bash
python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/scene1/images \
  --sparse_dir data/scene1/sparse_ba \
  --output_dir data/scene1_ns_ba

ns-train splatfacto --data data/scene1_ns_ba --output-dir outputs/3dgs_ba
```

## 6. Export Splat

```bash
ns-export gaussian-splat \
  --load-config PATH_TO_CONFIG_YML \
  --output-dir outputs/exports/scene1_ba_splat
```
```

---

## 16. Codex 具体开发任务清单

请 Codex 按以下顺序实现。

### Stage 1: 初始化项目

- [ ] 创建目录结构；
- [ ] 创建 `requirements.txt`；
- [ ] 创建 `environment.yml`；
- [ ] 创建 `.gitignore`；
- [ ] 创建 `README.md`；
- [ ] 创建本文件 `PROJECT_INIT.md`。

### Stage 2: 数据预处理

- [ ] 实现 `scripts/00_extract_frames.py`；
- [ ] 支持视频抽帧；
- [ ] 支持按 fps 抽帧；
- [ ] 支持最大帧数；
- [ ] 支持 resize；
- [ ] 输出统一命名格式。

### Stage 3: COLMAP IO

- [ ] 实现 `src/cvfinal/colmap_io.py`；
- [ ] 能读取 cameras/images/points3D；
- [ ] 优先支持 text 格式；
- [ ] 如果 binary 太复杂，允许通过 pycolmap 或 COLMAP CLI 转换；
- [ ] 实现 `scripts/02_check_colmap_output.py`。

### Stage 4: 几何模块

- [ ] 实现 `src/cvfinal/geometry.py`；
- [ ] 实现 qvec / rotmat / Rodrigues 转换；
- [ ] 实现投影函数；
- [ ] 实现 reprojection error 计算；
- [ ] 写 `tests/test_geometry.py`。

### Stage 5: BA

- [ ] 实现 `src/cvfinal/ba.py`；
- [ ] 构建 BA problem；
- [ ] 固定第一帧；
- [ ] 固定内参；
- [ ] 支持 max_points；
- [ ] 支持 min_track_len；
- [ ] 使用 scipy least_squares；
- [ ] 输出优化前后重投影误差；
- [ ] 写回 sparse 目录；
- [ ] 保存 metrics CSV；
- [ ] 写 `tests/test_ba_residual.py`。

### Stage 6: 可视化

- [ ] 实现 `scripts/05_visualize_reconstruction.py`；
- [ ] 可视化相机轨迹；
- [ ] 可视化点云；
- [ ] 输出 png；
- [ ] 支持 VGGT 和 BA 结果对比。

### Stage 7: nerfstudio 数据准备

- [ ] 实现 `scripts/06_prepare_nerfstudio_data.py`；
- [ ] 自动创建 `sparse/0/`；
- [ ] 自动复制图像；
- [ ] 检查图像名一致性；
- [ ] 输出 nerfstudio 可用目录。

### Stage 8: 训练与导出脚本

- [ ] 实现 `scripts/07_train_splatfacto.sh`；
- [ ] 实现 `scripts/08_export_splat.sh`；
- [ ] 自动查找最新 config.yml；
- [ ] 保存 viewer 截图路径说明；
- [ ] 保存导出 `.ply`。

### Stage 9: 图像筛选改进

- [ ] 实现 `src/cvfinal/view_filter.py`；
- [ ] 实现清晰度打分；
- [ ] 实现冗余帧筛选；
- [ ] 实现异常帧标记；
- [ ] 输出筛选后图片目录；
- [ ] 输出 CSV 和 score 曲线。

### Stage 10: 实验汇总

- [ ] 实现 `src/cvfinal/metrics.py`；
- [ ] 实现 `scripts/09_eval_rendering.py`；
- [ ] 实现 `outputs/tables` 汇总；
- [ ] 输出 BA 是否提升 3DGS 的表格；
- [ ] 输出输入筛选是否有效的表格。

---

## 17. 注意事项

### 17.1 不要一开始做太复杂

优先顺序：

1. 跑通 VGGT；
2. 读出 COLMAP 数据；
3. 简化 BA；
4. 跑通 3DGS；
5. 做对比实验；
6. 再优化细节。

不要一开始改 VGGT 网络结构。

### 17.2 BA 规模要控制

如果图像和点太多，scipy BA 会很慢。建议：

- 第一次 BA：20 张图、3000 个点；
- 第二次 BA：50 张图、10000 个点；
- 最终实验：80 到 120 张图、30000 个点以内。

### 17.3 BA 不一定提升 3DGS

实验结论不能预设。可能出现：

- 重投影误差下降，但 3DGS 视觉效果变化不大；
- BA 消除部分错位相机，3DGS 浮点减少；
- BA 被错误匹配影响，反而使效果变差；
- 筛图减少模糊图后，质量提升；
- 筛图过度后，视角覆盖不足，质量下降。

PPT 中要如实分析。

### 17.4 训练时间控制

建议最终至少保存：

- `VGGT raw` 的结果；
- `VGGT + BA` 的结果；
- `Filtered + VGGT + BA` 的结果。

如果时间不够：

1. 优先完成 Pipeline A 和 B；
2. Pipeline C 可以只完成 VGGT + BA 的误差对比，不一定完整训练 3DGS；
3. 但最好至少训练一个 filtered 版本，用于答辩展示。

---

## 18. 预期最终交付物

```text
交付物：
1. 代码仓库
2. 自采集多视角图像
3. VGGT 初始重建结果
4. 自实现 BA 代码
5. BA 前后重投影误差表格
6. 3DGS 实时 viewer 展示
7. BA 前后 3DGS 对比图
8. VGGT 改进方法调研与实验
9. PPT
10. 答辩 demo 视频或现场演示
```

---

## 19. PPT 大纲

```text
1. 任务背景与目标
2. 整体 Pipeline
3. 数据采集与预处理
4. VGGT 初始化：相机参数与初始点云
5. VGGT 输出结果展示
6. Bundle Adjustment 数学模型
7. 自实现 BA 的参数化与优化细节
8. BA 前后重投影误差对比
9. 3D Gaussian Splatting 原理简介
10. 3DGS 训练与实时渲染展示
11. 实验一：BA 是否改善 3DGS
12. VGGT 改进方法：输入筛选 / 关键帧选择
13. 实验二：改进方法是否提升速度或质量
14. 局限性分析
15. 未来研究方向
```

---

## 20. 参考命令总览

```bash
# 1. 创建环境
conda create -n cvfinal python=3.10 -y
conda activate cvfinal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. 安装 VGGT
mkdir -p external
cd external
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
cd ../..

# 3. 跑 VGGT
cd external/vggt
python demo_colmap.py --scene_dir ../../data/scene1
cd ../..

# 4. 检查 VGGT 输出
python scripts/02_check_colmap_output.py \
  --sparse_dir data/scene1/sparse_vggt \
  --image_dir data/scene1/images

# 5. 跑 BA
python scripts/04_bundle_adjustment.py \
  --input_sparse data/scene1/sparse_vggt \
  --image_dir data/scene1/images \
  --output_sparse data/scene1/sparse_ba \
  --min_track_len 3 \
  --max_points 30000 \
  --loss huber \
  --max_nfev 100

# 6. 准备 nerfstudio 数据
python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/scene1/images \
  --sparse_dir data/scene1/sparse_ba \
  --output_dir data/scene1_ns_ba

# 7. 训练 3DGS
ns-train splatfacto \
  --data data/scene1_ns_ba \
  --output-dir outputs/3dgs_ba

# 8. 导出 splat
ns-export gaussian-splat \
  --load-config PATH_TO_CONFIG_YML \
  --output-dir outputs/exports/scene1_ba_splat

# 9. 图像筛选
python scripts/10_filter_views.py \
  --input_dir data/scene1/images \
  --output_dir data/scene1_filtered/images \
  --sharpness_quantile 0.15 \
  --min_hist_diff 0.05 \
  --max_hist_diff 0.95 \
  --max_images 80
```

---

## 21. Codex 输出要求

Codex 初始化仓库时，不要只生成空文件。至少完成：

1. 所有脚本的 argparse 参数框架；
2. 所有核心函数的 docstring；
3. `geometry.py` 中的投影和旋转转换；
4. `ba.py` 中的 BA problem 数据结构和 residual 框架；
5. `view_filter.py` 中的清晰度筛选；
6. README 中写清楚运行顺序；
7. 如果某些功能暂时未实现，用 `NotImplementedError` 明确标出，不要静默跳过；
8. 所有输出目录自动创建；
9. 所有脚本失败时给出明确错误信息；
10. 所有随机操作设置 seed。

---

## 22. 最小可运行版本定义

如果时间有限，最小版本应当能完成：

```text
输入：data/scene1/images
输出：
1. VGGT sparse_vggt
2. BA sparse_ba
3. outputs/tables/ba_metrics_scene1.csv
4. outputs/figures/vggt_reconstruction.png
5. outputs/figures/ba_reconstruction.png
6. 一个 nerfstudio splatfacto 训练结果
```

只要以上内容跑通，就可以继续补实验分析和 PPT。

