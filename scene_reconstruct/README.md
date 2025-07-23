# 场景重建工具与方法说明

## 1. blender_trans.py 用法

该脚本用于将 OBJ 格式的三维模型转换为 USD 格式，便于后续在仿真环境（如Isaac Sim）中加载。

**使用方法：**
1. 需在命令行中通过 Blender 的 Python 解释器运行该脚本。
2. 修改脚本中的 `input_path` 和 `output_path` 变量，分别指向你的 OBJ 文件和期望输出的 USD 文件路径。
3. 运行命令示例：
   ```bash
   blender --background --python blender_trans.py
   ```
   > 注意：需提前安装好 Blender，并确保 `bpy` 模块可用。

## 2. isaac-sim_test.py 用法

该脚本用于在 NVIDIA Isaac Sim 仿真环境中加载机器人和重建场景，并进行简单的运动控制和传感器数据采集。

**使用方法：**
1. 确保已正确安装并配置 Isaac Sim。
2. 修改脚本中的 `usd_path` 参数，指向你要加载的 USD 场景文件。
3. 直接用 Isaac Sim 的 Python 解释器运行：
   ```bash
   ./python.sh isaac-sim_test.py
   ```
4. 脚本会自动加载 Jetbot 机器人、相机、光源和地面，并导入指定的场景 USD 文件。
5. 机器人会按照预设轨迹运动，并采集相机数据。

> **注意：**
> - 导入的场景建议提前在 3D 软件（如 Blender）中校准好位置和比例，避免导入后地面不平整或比例异常。

## 3. 场景重建方法说明

### SuGaR 方法
- SuGaR（Surface-Aligned Gaussian Splatting）是一种高效的 3D 场景重建方法，能够从 3D Gaussian Splatting 表示中快速提取高质量可编辑的网格（mesh）。
- SuGaR 支持直接导出可用于仿真和动画的 mesh，且重建速度快、细节丰富。
- SuGaR 重建的 mesh 可直接用于仿真，但由于高斯点云与表面拟合方式，**碰撞面可能存在不平整或不连续的情况**，在物理仿真中可能导致碰撞检测不准确。

### pgsr 方法
- pgsr（Poisson Surface Reconstruction）是一种基于泊松重建的表面重建算法，适合对点云或高斯点云进行表面拟合。
- 使用 pgsr 进行 mesh 重建时，**建议根据实际场景调整重建参数**（如泊松深度、密度阈值等），以获得更平滑、连续的表面，提升仿真中的碰撞检测效果。
- 常见调参建议：
  - 若 mesh 表面有孔洞，可适当降低密度阈值（如 vertices_density_quantile）。
  - 若表面有明显的高斯点"包块"或不平整，可适当降低泊松重建深度（如 poisson_depth=7 或 6）。

## 4. 推荐流程
1. 使用 SuGaR 或 pgsr 方法进行场景 mesh 重建。
2. 若需进一步优化碰撞面，优先尝试 pgsr 并调参获得更平滑的 mesh。
3. 用 blender_trans.py 转换 mesh 为 USD 格式。
4. 用 isaac-sim_test.py 在 Isaac Sim 中加载并验证场景。

---

## 注：重建数据集也可从重建方法的git中获取下载，校准的colmap 数据集

**参考资料：**
- [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering (CVPR 2024)](https://github.com/Anttwo/SuGaR)
- [SuGaR 官方文档与调参建议](https://imagine.enpc.fr/~guedona/sugar/)
