# 报告与答辩 PPT 记忆文档

用途：持续记录报告/PPT 需要讲的内容、实验材料、待补图表和答辩表达。后续新增内容按下面四个部分归档。

## 0. 总体顺序

1. VGGT 相关
2. BA 相关
3. 3DGS / Gaussian Splatting 相关
4. VGGT 改进相关

## 1. VGGT 相关

### 1.1 人像蒙版实验：是否加入 mask 对 VGGT 点云与后续 Nerfstudio/3DGS 训练的影响

想讲的问题：

- 人像数据里，背景和人物前景同时出现在输入图像中。
- VGGT 会根据输入图像整体估计相机、深度和点云。如果背景纹理、遮挡边界或人物外区域参与建模，可能会让点云包含不需要的背景点或边缘噪声。
- 后续 3DGS/Nerfstudio 训练会继承 VGGT 的相机与稀疏点云初始化，因此前期点云质量会影响训练稳定性、前景聚焦程度和渲染效果。

准备对比：

- 不加 mask：直接使用原始 RGB 多视角人像图像跑 VGGT。
- 加 mask：使用提供的人像 mask，把背景处理掉或弱化，只保留/强调人体前景后跑 VGGT。
- 为了公平对比，除输入图像目录和输出目录外，其余 VGGT/self BA 参数应尽量保持一致。
- 对比 VGGT 输出点云：
  - 人体主体是否更集中。
  - 背景杂点是否减少。
  - 人物边界附近点云是否更干净。
  - 相机估计和点云是否出现明显漂移。
- 对比后续 NS/3DGS 训练：
  - 训练是否更容易聚焦人体前景。
  - 背景区域是否产生不必要的 Gaussian。
  - 渲染图里人物轮廓、空洞、漂浮噪声是否改善。

答辩表达主线：

1. VGGT 本身不只是在“找人体”，它会从整张图像中恢复几何。
2. 人像重建的目标主要是前景人体，所以 mask 可以作为一种输入先验，减少背景对点云和后续 3DGS 初始化的干扰。
3. 这个实验的重点不是证明 mask 一定总是提升所有指标，而是分析 mask 改变了 VGGT 可见内容以后，对稀疏点云和后续高斯训练有什么影响。

建议 PPT 图：

- 原始图像 vs mask 图像。
- 无 mask 的 VGGT 点云截图。
- 加 mask 的 VGGT 点云截图。
- 无 mask / 加 mask 的 3DGS 渲染对比图。
- 如果有指标，可以补一张小表：点数、重投影误差、训练最终 PSNR/SSIM/LPIPS 或主观现象。

需要补的材料：

- [ ] 不再把 VGGT 点云作为 mask 对比重点；mask 对比重点改为训练好的高斯点云/渲染结果。
- [ ] 找出对应 NS/3DGS 训练渲染图。
- [ ] 如果有日志，整理 sparse 点数、BA 后点数、训练质量指标。

已记录的 human1 加 mask 命令：

```bash
cd /root/cv_hw

python scripts/run_vggt_colmap.py \
  --scene_dir data/human1_mask_black \
  --output_scene_dir data/human1_vggt_mask_black_q16_self_ba \
  --max_images 16 \
  --sample_strategy uniform \
  --use_ba \
  --skip_builtin_ba \
  --shared_camera \
  --overwrite_sparse \
  --query_frame_num 16 \
  --run_self_ba \
  --self_ba_output_name sparse_self_ba \
  --self_ba_max_points 12000 \
  --self_ba_max_nfev 120 \
  --self_ba_max_initial_reproj_error_px 64 \
  --self_ba_prune_reproj_error_px 12 \
  --self_ba_optimize_shared_intrinsics \
  --overwrite_self_ba_output
```

建议补跑的 human1 不加 mask 对照命令：

```bash
cd /root/cv_hw

python scripts/run_vggt_colmap.py \
  --scene_dir data/human1 \
  --output_scene_dir data/human1_vggt_nomask_q16_self_ba \
  --max_images 16 \
  --sample_strategy uniform \
  --use_ba \
  --skip_builtin_ba \
  --shared_camera \
  --overwrite_sparse \
  --query_frame_num 16 \
  --run_self_ba \
  --self_ba_output_name sparse_self_ba \
  --self_ba_max_points 12000 \
  --self_ba_max_nfev 120 \
  --self_ba_max_initial_reproj_error_px 64 \
  --self_ba_prune_reproj_error_px 12 \
  --self_ba_optimize_shared_intrinsics \
  --overwrite_self_ba_output
```

这组对照的核心变量：

- mask 组输入：`data/human1_mask_black`
- no-mask 组输入：`data/human1`
- 其他参数保持一致：16 张图、uniform 采样、`query_frame_num=16`、跳过 VGGT 内置 BA、使用自实现 self BA、共享相机内参并优化 shared intrinsics。

human1 不加 mask 对照组运行结果：

- 输出目录：`data/human1_vggt_nomask_q16_self_ba`
- VGGT sparse 输出：
  - cameras: 1
  - images: 16
  - points3D: 61768
  - mean_track_length: 4.479
  - points usable for BA: 61768
- self BA 设置：
  - max_points: 12000
  - max_nfev: 120
  - loss: soft_l1
  - f_scale: 2.0
  - max_initial_reproj_error_px: 64
  - prune_reproj_error_px: 12
  - optimize_shared_intrinsics: True
- self BA 结果：
  - reproj_before_mean: 7.976725 px
  - reproj_before_median: 7.985362 px
  - reproj_before_p90: 13.993116 px
  - reproj_after_mean: 1.683995 px
  - reproj_after_median: 1.028669 px
  - reproj_after_p90: 3.866525 px
  - shared_intrinsics_before: `[2014.9943, 1024.0, 1024.0]`
  - shared_intrinsics_after: `[1847.7897, 1026.0167, 1020.5120]`
  - pruned_points: 293
  - removed_unoptimized_points: 49768
  - success: True
  - time_sec: 98.113
- 产物：
  - `data/human1_vggt_nomask_q16_self_ba/sparse/{cameras.bin, images.bin, points3D.bin, points.ply}`
  - `data/human1_vggt_nomask_q16_self_ba/sparse_self_ba/{cameras.bin, images.bin, points3D.bin, points.ply}`

可用于答辩的初步结论：

- no-mask 组本身可以成功跑通 VGGT + self BA。
- BA 对 no-mask 组的几何一致性有明显优化：平均重投影误差从约 7.98 px 降到约 1.68 px。
- 因为 `keep_unoptimized_points=False`，最终 BA 点云只保留参与优化且未被裁剪的点；这解释了 sparse 初始点数 61768 与 BA 后点云数量明显变少的现象。
- 后续需要和 mask black 组比较：初始点数、重投影误差下降幅度、点云中背景杂点数量、3DGS 渲染中的漂浮噪声和人体轮廓质量。

当前对比策略调整：

- mask 对 VGGT sparse 点云的影响可能不明显，因此不把 VGGT 点云截图作为主要 mask 对比证据。
- 更合适的比较对象是训练完成后的 3DGS/Gaussian 表示与渲染结果：
  - no-mask 图片训练：观察背景 Gaussian、漂浮噪声、人体边界。
  - mask-black 图片训练：观察模型是否更聚焦人体前景，背景噪声是否减少。

human1 no-mask 3DGS 训练命令：

```bash
cd /root/cv_hw

python scripts/06_prepare_nerfstudio_data.py \
  --image_dir data/human1/images \
  --sparse_dir data/human1_vggt_nomask_q16_self_ba/sparse_self_ba \
  --output_dir data/human1_nomask_ns_self_ba \
  --make_images_2 \
  --overwrite

bash scripts/07_train_splatfacto.sh \
  --data data/human1_nomask_ns_self_ba \
  --output outputs/3dgs \
  --experiment-name human1_nomask_self_ba \
  --max-num-iterations 30000 \
  --masks-path none \
  --background-color black \
  --archive-dir report/artifacts/human1_nomask_self_ba
```

注意：

- 这组训练使用 `data/human1/images` 原始无 mask 图片。
- NS/3DGS 初始化使用 `data/human1_vggt_nomask_q16_self_ba/sparse_self_ba`，也就是 VGGT + self BA 后的相机和点云。
- 不传 `--mask_dir`，并在训练时使用 `--masks-path none`，避免 Nerfstudio 使用 mask。
- 如果担心 BA 后点云过稀，可以补跑一个更密的 self BA 版本，例如 `--self_ba_max_points 30000`，再用对应的 `sparse_self_ba` 训练最终展示版。

### 1.2 视频采样：均匀采样与 VGGT 输入帧选择的影响

想讲的问题：

- 第三组数据是视频，不是已经筛好的多视角图片。
- 从视频中抽帧时，常见做法是均匀采样，优点是简单、覆盖时间轴均匀。
- 但对 VGGT 来说，输入帧不是越“平均”越好。帧之间的视角差、模糊程度、重复程度、遮挡、运动速度都会影响相机估计和点云质量。

先在 VGGT 部分简单引入：

- 均匀采样可以作为 baseline。
- 如果相邻帧变化太小，输入冗余高，几何约束不够强。
- 如果帧间变化太大，重叠区域不足，匹配/track 可能变差。
- 如果采到模糊帧或质量差的帧，点云会更噪。

注意：具体的改进方法放到最后“VGGT 改进相关”部分展开，这里只埋下问题。

建议 PPT 图：

- 视频帧均匀采样示意。
- 固定间隔/均匀采样得到的一组关键帧缩略图。
- 对应 VGGT 点云初始结果。

需要补的材料：

- [ ] 整理 scene3 均匀采样帧缩略图。
- [ ] 整理 scene3 均匀采样 VGGT 点云图。
- [ ] 后面和质量筛选/关键帧筛选实验形成呼应。

### 1.3 展示训练出来的点云

想讲的问题：

- VGGT 直接给出了相机和初始点云，是后续 BA 与 3DGS 的起点。
- 展示训练/重建得到的点云，可以让老师直观看到从输入图像到三维结构的结果。

建议 PPT 内容：

- 一张整体点云截图。
- 一张局部或不同视角截图。
- 如果有多组实验，可以按“人像 / 场景视频”分开展示。

需要补的材料：

- [ ] 人像点云最终截图。
- [ ] 场景点云最终截图。
- [ ] 如果展示 BA 前后差异，标清“VGGT 初始点云”和“BA 后点云”。

### 1.4 参数是否需要讲

当前判断：

- 主线答辩里不需要详细讲参数，除非某个参数直接影响实验结论。
- 参数更适合放在备份页或报告实验设置表里，用于说明公平对比和可复现性。
- PPT 主线应优先讲：
  - 为什么做这个实验。
  - 对比了什么。
  - 结果说明了什么。
  - 对后续 BA/3DGS 有什么影响。

可放到附录/备份页的参数：

- 输入图片数量。
- 是否使用 mask。
- VGGT 是否使用 track / BA 相关导出。
- self BA 的误差阈值、最大点数等。
- 3DGS/Nerfstudio 训练步数、主要配置。

## 2. BA 相关

### 2.1 这一部分要解决的问题

VGGT 可以直接输出相机和点云，但这些结果仍然是神经网络预测和 track/匹配结果的组合，可能存在相机位姿偏差、3D 点位置不准、异常观测等问题。BA 部分的目标是用经典几何优化进一步约束这些结果，使 3D 点投影回各个视角时尽量对齐原始观测点。

答辩里可以把 BA 定位为：

- VGGT 负责给出初始相机和点云。
- 自实现 BA 负责做几何一致性优化。
- 3DGS 使用 BA 后的相机和点云作为训练初始化。

### 2.2 需要讲清楚的实现内容

优化目标：

```text
min sum rho( || project(K_i, R_i, t_i, X_j) - u_ij ||^2 )
```

其中：

- `X_j`：3D 点坐标。
- `R_i, t_i`：每张图像的相机外参。
- `K_i`：相机内参；当前实现支持固定内参，也支持优化共享内参。
- `u_ij`：VGGT/COLMAP sparse 中 track 对应的 2D 观测点。
- `rho`：鲁棒损失，用于减小错误匹配/异常点影响。

当前代码实现点：

- 输入：VGGT 导出的 COLMAP sparse 模型，包含 `cameras.bin`、`images.bin`、`points3D.bin`。
- 选点：只选 track length >= 2 的点，并按 track length 排序，最多取 `max_points` 个点。
- 固定第一帧相机位姿，避免整体坐标系/gauge freedom 不确定。
- 优化变量：
  - 除第一帧外的相机外参，每帧 6 维旋转向量 + 平移。
  - 参与 BA 的 3D 点坐标，每点 3 维。
  - 可选共享相机内参，如 `SIMPLE_PINHOLE` 的 `[f, cx, cy]`。
- 使用 `scipy.optimize.least_squares` 做非线性最小二乘。
- 使用稀疏 Jacobian 结构，加快大规模观测优化。
- 使用 `soft_l1`/`huber` 等鲁棒 loss。
- BA 前可按初始重投影误差过滤异常观测。
- BA 后可按重投影误差裁剪坏点。
- 输出仍然是 COLMAP sparse 格式，可以无缝接到 Nerfstudio/3DGS。

更详细的实现流程：

1. 读取 VGGT sparse 模型

   - 输入目录中包含 `cameras.bin`、`images.bin`、`points3D.bin`。
   - 用 `pycolmap.Reconstruction` 读取相机、图像位姿、3D 点和 track。
   - 每个 3D 点的 track 记录了它在哪些图像中被观测到，以及对应的 2D 点坐标。

2. 构造 BA 优化问题

   - 相机位姿从 `cam_from_world` 转成 6 维参数：旋转向量 `rvec` + 平移 `tvec`。
   - 3D 点坐标直接使用 VGGT/COLMAP sparse 中的 `xyz`。
   - 只选择 track length >= 2 的点，因为只被一个视角看到的点不能提供有效多视角约束。
   - 按 track length 从大到小排序，优先优化被更多视角看到的稳定点。
   - 用 `max_points` 控制参与 BA 的点数，避免优化规模过大。

3. 选择优化变量

   当前实现把所有变量打包成一个一维向量：

   ```text
   params = [camera_poses, points_3d, optional_shared_intrinsics]
   ```

   - `camera_poses`：除第一帧外，每个相机 6 维。
   - `points_3d`：每个 3D 点 3 维。
   - `optional_shared_intrinsics`：如果开启 `--optimize_shared_intrinsics`，则优化共享内参。

   第一帧相机不优化，用来固定整体坐标系，避免 BA 出现整体平移/旋转自由度。

4. 计算重投影残差

   对每个观测关系 `(image_i, point_j, u_ij)`：

   - 先把 3D 点从世界坐标变换到相机坐标：

     ```text
     X_cam = R_i X_j + t_i
     ```

   - 再用 pinhole camera model 投影到像素平面：

     ```text
     u_hat = fx * X_cam / Z_cam + cx
     v_hat = fy * Y_cam / Z_cam + cy
     ```

   - 残差就是预测像素和原始观测像素的差：

     ```text
     r_ij = [u_hat - u_ij, v_hat - v_ij]
     ```

   BA 优化就是让所有观测残差尽量小。

5. 鲁棒优化和稀疏加速

   - 使用 `scipy.optimize.least_squares` 求解非线性最小二乘。
   - 使用 `soft_l1` 或 `huber` loss，降低错误匹配和异常观测的影响。
   - 构造 sparse Jacobian pattern，因为每个残差只和一个相机、一个 3D 点、可选内参有关。这样比直接 dense 优化更高效。

6. 优化前后过滤

   - 优化前：如果设置 `max_initial_reproj_error_px`，会先删除初始重投影误差过大的观测。
   - 优化后：如果设置 `prune_reproj_error_px`，会删除仍然有较大重投影误差的 3D 点。
   - 默认不保留未参与优化的点，所以 BA 后点云可能比 VGGT 初始点云稀疏。

7. 写回结果

   - 把优化后的相机位姿写回 `images.bin`。
   - 把优化后的 3D 点写回 `points3D.bin`。
   - 如果优化了共享内参，把新的 `f, cx, cy` 写回 `cameras.bin`。
   - 额外导出 `points.ply`，方便可视化。
   - 输出格式仍然是 COLMAP sparse，所以后续 `06_prepare_nerfstudio_data.py` 可以直接接到 3DGS。

### 2.3 PPT 建议结构

建议 BA 部分放 3-4 页：

1. 为什么需要 BA：
   - VGGT 给出初值，但还需要用多视角几何一致性细化。
   - BA 优化的是“3D 点投影回图像后与观测点的误差”。

2. 自实现 BA 方法：
   - 放一张公式或流程图。
   - 讲输入、优化变量、固定第一帧、鲁棒损失、异常点裁剪。

3. BA 前后指标对比：
   - 放表格：`reproj_before_mean/median/p90` vs `reproj_after_mean/median/p90`。
   - human1 no-mask 是最明显例子：mean 从约 `7.98 px` 降到约 `1.68 px`。
   - scene3 如果初始 already good，可以说 BA 后仍略微下降，但收益较小。

4. BA 对 3DGS 的影响：
   - 重点比较 BA 前后训练出来的高斯点云/渲染效果。
   - 观察点：漂浮 Gaussian、边缘噪声、几何错位、重建稳定性。

### 2.4 已有 BA 结果记录

human1 no-mask：    

- 输入：`data/human1_vggt_nomask_q16_self_ba/sparse`
- 输出：`data/human1_vggt_nomask_q16_self_ba/sparse_self_ba`
- images: 16
- VGGT points3D: 61768
- BA points: 12000
- observations: 92548
- reproj_before_mean: 7.976725 px
- reproj_before_median: 7.985362 px
- reproj_before_p90: 13.993116 px
- reproj_after_mean: 1.683995 px
- reproj_after_median: 1.028669 px
- reproj_after_p90: 3.866525 px
- success: True
- 可讲结论：这组 BA 改善非常明显，说明 VGGT 初始化虽然能给出点云，但几何一致性仍有优化空间。

human2：

- 标准 self BA 输入：`data/human2_std_self_ba/sparse`
- 标准 self BA 输出：`data/human2_std_self_ba/sparse_self_ba`
- images: 16
- VGGT points3D: 53032
- mean_track_length: 6.031
- BA points: 12000
- observations: 106905
- reproj_before_mean: 6.818656 px
- reproj_before_median: 6.279719 px
- reproj_before_p90: 12.928179 px
- reproj_after_mean: 1.878409 px
- reproj_after_median: 1.242167 px
- reproj_after_p90: 4.189113 px
- pruned_points: 326
- removed_unoptimized_points: 41032
- time_sec: 99.005
- success: True
- 可讲结论：在不使用 VGGT 官方 BA 的标准流程下，human2 的 self BA 也带来明显误差下降，mean 从约 6.82 px 降到约 1.88 px。

scene3 48 帧 quality 采样宽松构建：

- 标准 self BA 输入：`data/scene3_quality48_std_self_ba_loose/sparse`
- 标准 self BA 输出：`data/scene3_quality48_std_self_ba_loose/sparse_self_ba`
- images: 48
- VGGT points3D: 15378
- mean_track_length: 9.683
- BA points: 15378
- observations: 148786
- initial_removed_observations: 116
- initial_removed_points: 0
- reproj_before_mean: 12.213140 px
- reproj_before_median: 9.293802 px
- reproj_before_p90: 26.178774 px
- reproj_after_mean: 6.076538 px
- reproj_after_median: 2.737204 px
- reproj_after_p90: 14.213629 px
- pruned_points: 6646
- removed_unoptimized_points: 0
- time_sec: 62.968
- success: True
- 可讲结论：scene3 的视频输入更难，VGGT 初始重投影误差明显高于人像；self BA 可以显著降低误差，但优化后仍高于人像场景。这说明视频采样/关键帧质量会强烈影响 VGGT sparse 初始化质量，也呼应后面的 VGGT 改进部分。

标准 self BA 总表：

| Scene | Images | VGGT points | BA points | Observations | Mean before | Mean after | Median before | Median after | P90 before | P90 after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| human1 | 16 | 61770 | 12000 | 92557 | 7.979 | 1.681 | 7.995 | 1.028 | 13.986 | 3.869 |
| human2 | 16 | 53032 | 12000 | 106905 | 6.819 | 1.878 | 6.280 | 1.242 | 12.928 | 4.189 |
| scene3 quality48 loose | 48 | 15378 | 15378 | 148786 | 12.213 | 6.077 | 9.294 | 2.737 | 26.179 | 14.214 |

### 2.5 需要补的材料

- [ ] 整理 BA 前后指标表，至少包含 human1 no-mask、human2、scene3。
- [ ] 准备一张 BA 流程图：VGGT sparse -> 读取 tracks -> 最小化重投影误差 -> 输出 sparse_self_ba -> 3DGS。
- [ ] 准备 BA 前后点云截图，注意不要只用点数评价；BA 后点云可能因为 `max_points` 和裁剪变稀。
- [ ] 准备 BA 前后 3DGS 渲染对比图，这是回答“BA 是否影响高斯泼溅效果”的关键。
- [ ] 如果点云太稀，补跑更高 `self_ba_max_points` 的版本，例如 30000 或 50000。

### 2.6 答辩时可以说的话

> VGGT 输出的是一个很好的初始化，但它不是显式最小化多视角重投影误差得到的最终几何结果。因此我在 VGGT 输出的 COLMAP sparse 模型上实现了一个 Bundle Adjustment 模块，同时优化相机外参和 3D 点，并可选优化共享内参。优化目标是让 3D 点重新投影到各个视角时尽量接近原来的 2D track 观测。为了避免异常匹配影响优化，我使用鲁棒 loss，并在优化前后根据重投影误差做过滤和裁剪。最终输出仍然保持 COLMAP 格式，可以直接作为 Nerfstudio/3DGS 的初始化。

注意事项：

- BA 指标提升大小取决于初始结果。初始误差高的 human1 no-mask 提升明显；初始已经很好的 scene3 提升较小。
- BA 后点云减少不一定代表更差，因为当前实现默认只保留参与优化并通过裁剪的点；但是如果用于 3DGS 初始化，点太少可能影响覆盖，需要通过更大的 `max_points` 或 3DGS 结果对比来判断。

## 3. 3DGS / Gaussian Splatting 相关

### 3.1 这一部分要回答的问题

3DGS 部分不只是展示最终模型，而是用最终渲染效果验证前面 VGGT 和 BA 的初始化质量。核心问题是：

- VGGT 直接输出的 sparse 能不能训练出可用的 3DGS？
- self BA 优化后的相机和点云，是否让 3DGS 训练更稳定、渲染更干净？
- mask / no-mask 输入对最终 Gaussian 和渲染背景噪声有没有影响？
- 视频场景中，采样策略对最终重建质量有什么影响？

### 3.2 建议 PPT 结构

建议 3DGS 部分放 4-5 页：

1. 3DGS 在 pipeline 中的位置

   ```text
   images -> VGGT sparse -> optional self BA -> Nerfstudio data -> splatfacto -> Gaussian point cloud / rendering
   ```

   讲清楚：3DGS 使用图像、相机位姿和 sparse 点云作为初始化，训练出可实时渲染的 Gaussian 表示。

2. 展示最终生成模型

   - 放人像最终模型截图或旋转视角截图。
   - 放 scene3 最终模型截图。
   - 如果有 viewer/视频，可以答辩时展示交互旋转或渲染视频。

3. VGGT-only vs VGGT+selfBA 对比

   这是最重要的实验。

   对比对象：

   - 不使用 BA：`VGGT sparse -> 3DGS`
   - 使用 BA：`VGGT sparse -> self BA -> 3DGS`

   观察维度：

   - 人物/物体边缘是否更稳定。
   - 是否减少漂浮 Gaussian。
   - 背景或非目标区域是否更干净。
   - 新视角渲染是否有错位、重影、破碎。
   - 训练 loss 或 PSNR/SSIM/LPIPS 是否改善；如果没有指标，就用同视角渲染截图主观比较。

4. mask vs no-mask 最终效果对比

   不再强调 mask 对 VGGT 点云的影响，而是比较 3DGS 最终结果：

   - no-mask：可能会学习背景、产生背景高斯点或人体边缘漂浮点。
   - mask-black：更聚焦前景人体，背景更干净，但也可能因为黑背景边界导致边缘略硬。

   PPT 上放：

   - no-mask 渲染截图
   - mask-black 渲染截图
   - 对应导出的 Gaussian 点云截图

5. 视频场景结果与采样呼应

   - 展示 scene3 的 3DGS 结果。
   - 说明视频场景比人像更依赖关键帧选择，因为 VGGT sparse 初始化误差更高。
   - 和最后 VGGT 改进部分的 quality keyframe selection 呼应。

### 3.3 还可以补做的实验

优先级从高到低：

1. BA 前后 3DGS 对比

   必做。因为课程要求里明确有“分析 BA 是否影响高斯泼溅效果”。

   最少做一组：

   ```text
   human1 VGGT sparse -> 3DGS
   human1 self BA sparse -> 3DGS
   ```

   如果时间够，再做 human2 或 scene3。

2. mask / no-mask 3DGS 对比

   已经决定不重点比较 VGGT 点云，这里就用最终 Gaussian 和渲染结果比较。

   ```text
   human1 no-mask images + self BA sparse -> 3DGS
   human1 mask-black images + self BA sparse -> 3DGS
   ```

3. Gaussian 点云可视化

   导出 `ns-export gaussian-splat` 的 `.ply`，用 Open3D 或脚本查看 Gaussian center 分布。

   注意：3DGS `.ply` 不是普通 RGB 点云，颜色在 `f_dc_*` 球谐系数里；用项目脚本查看更合适。

4. 定量指标

   如果 Nerfstudio 有 eval 指标，整理 PSNR/SSIM/LPIPS。

   如果没有完整 test split，也可以只放训练日志中的 loss 曲线或最终渲染截图，不强求指标。

5. 训练稳定性/收敛速度

   如果日志里能看到训练 loss，比较 BA 前后前期 loss 是否下降更快。

   这不是必做，但能补充说明 BA 给 3DGS 提供更稳定初始化。

### 3.4 答辩时可以讲的结论

> 3DGS 使用 VGGT 或 BA 输出的相机位姿和稀疏点云作为初始化。实验中，我主要比较了 VGGT 直接初始化和 self BA 后初始化对最终 Gaussian Splatting 的影响。BA 的作用不是直接生成更漂亮的图，而是先降低相机和点云的重投影误差，让多视角几何更一致，从而为 3DGS 训练提供更稳定的初始结构。最终对比时主要观察渲染中的重影、漂浮 Gaussian、边缘破碎和背景噪声。

### 3.5 需要补的材料

- [ ] human1 不使用 BA 的 3DGS 结果。
- [ ] human1 使用 self BA 的 3DGS 结果。
- [ ] mask-black 与 no-mask 的 3DGS 渲染对比。
- [ ] 至少一张导出的 Gaussian `.ply` 点云截图。
- [ ] scene3 最终渲染或点云截图。
- [ ] 如果有 eval，整理 PSNR/SSIM/LPIPS；如果没有，整理同视角截图做主观对比。

## 4. VGGT 改进相关

待补充。

可先占位的问题：

- 人像方向：foreground-aware/mask-aware 输入是否改善前景重建。
- 视频方向：质量感知或视角覆盖感知的关键帧选择，替代简单均匀采样。
- 需要把第 1.2 节中提到的视频采样问题在这里正式展开为改进方法与实验分析。
