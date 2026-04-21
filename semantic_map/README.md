# Semantic Map（语义地图）集成说明

本目录将 NPC 仓库中 `test_scene_construct.py` 所使用的 **`mapping.SemanticMap`**（GroundedSAM + `home_robot` 2D 语义栅格）接入到 **EmbodiedAI**，便于在 Isaac Sim 循环里更新语义地图。

## 依赖关系（重要）

`SemanticMap` 的实现仍在 **NPC 仓库**中（`NPC/mapping.py`），并依赖：

- `home_robot`（NPC 内 `home-robot/src`）
- `grounded_sam_perception.py`、GroundingDINO、SAM2 等（见 NPC 环境）

本包**不复制**上述大型依赖，仅通过 `sys.path` **桥接** NPC。请先克隆/配置好 NPC，并安装其文档要求的 Python 包与模型权重。

**离线权重一键下载**（BERT + GroundingDINO SwinB + SAM2.1 large，与 `grounded_sam_perception.py` 路径一致）：在可联网环境执行  
`python /path/to/NPC/scripts/download_semantic_map_models.py`；后台执行：`bash /path/to/NPC/scripts/download_hf_models_background.sh`。可选 `--sam2-all` 拉取全部 SAM2.1 变体。

### Isaac Sim + GroundingDINO：为何必须子进程分割（根因）

Isaac 使用 **`omni.isaac.ml_archive` 自带的 PyTorch**。在该进程里再跑 GroundingDINO/SAM2 时，CUDA 上下文常与 Omniverse 栈**不兼容**，表现为：

- `RuntimeError: CUDA error: an illegal memory access was encountered`
- 甚至在 **`model.to("cpu")`** 迁移权重时也会崩（GPU 侧参数已处于损坏状态）

**从根源解决**：在**独立操作系统进程**中运行 `grounded_sam_perception`（不 `import isaacsim`），通过 `npz` + `pickle` 与主进程交换观测与分割结果。实现见：

- `semantic_map/isolated_segmentation.py`（父进程客户端）
- `semantic_map/workers/grounded_sam_isolated_worker.py`（子进程 worker）

`examples/run_reference_path_semantic_map.py` **默认启用子进程分割**；仅调试时才使用 `--inprocess-grounded-sam`。

可选使用 **`EMBODIEDAI_SEG_PYTHON`** 指向**未捆绑 Isaac** 的 Python（独立 venv），进一步减少与 Kit 自带 torch 的耦合。

**注意**：在 Cursor/VSCode 用 **调试器（F5 / debugpy）** 启动主脚本时，子进程可能继承调试相关环境，导致 **stdout 首行不是 JSON**，出现 `Expecting value: line 1 column 1`。请改用 **「运行而不调试」**，或设置 **`EMBODIEDAI_SEG_PYTHON`** 为系统/conda 里的 `python`（脚本已对子进程环境做了部分清理，并会跳过空行/非 JSON 行）。

## 环境变量

| 变量 | 含义 |
|------|------|
| `NPC_REPO_ROOT` 或 `EMBODIEDAI_NPC_ROOT` | NPC 仓库根目录，默认 `/data2/linmin/NPC` |
| `NPC_GROUNDED_SAM_MODEL_ROOT` | GroundingDINO / SAM2 权重根目录，默认 `/data2/linmin/model`（见 NPC `grounded_sam_perception.py`） |
| `SEM_GPU_ID` / `MAP_GPU_ID` | 分割模型与地图模块使用的 GPU |
| `SEM_MAP_DUMP` | 可视化等输出根目录 |
| `SEM_MAP_EXP_NAME` | 实验名子目录 |
| `SEM_MAP_PRINT_IMAGES` | 是否保存建图可视化（`0`/`1`） |
| `SEM_MAP_LEGEND_PATH` | 图例 PNG 路径（可选） |
| `EMBODIEDAI_SEG_PYTHON` | 子进程分割使用的 Python 可执行文件（可选，见上文根因说明） |

## 示例脚本：`examples/run_reference_path_semantic_map.py`

与 `tests/test_scene/test_agent.py` 相同方式加载 `DatasetLoader` + `BaseEnv` + `ReferencePathAgent`，用前视相机 **RGB + depth** 与 **位姿** 驱动 NPC `SemanticMap`（若初始化失败则仅保存传感器帧）。

```bash
export PYTHONPATH=/data2/linmin/EmbodiedAI:$PYTHONPATH
export NPC_REPO_ROOT=/data2/linmin/NPC

python semantic_map/examples/run_reference_path_semantic_map.py 0 \
  --output-dir ./semantic_map/outputs/task_0 --headless
```

输出目录包含：

- `frames/`：`rgb_*.png`、`depth_*.png`、`pose_*.npz`
- `maps/`：`obstacle_*`、`explored_*`、`semantic_argmax_*`（语义模块可用时）
- `npc_dump/`：NPC 在 `print_images=1` 下的拼图（若 GroundedSAM 跑通）

仅调试传感器、不加载 NPC 时加 `--skip-semantic-map`。

**注意**：当前任务里机器人相机需在配置中启用 `depth`（与 `tests/test_configs/test.yaml` 中 `modals: ["rgb","depth"]` 一致），否则脚本会报错提示。

**分辨率**：NPC `Categorical2DSemanticMapModule` 的 `env_frame_height/width` 必须与 **`--camera-key` 对应传感器的 `SensorConfig.resolution`（宽×高）** 一致。`DatasetLoader` / `simulator/core/dataset.py` 中前视相机已与 NPC `stretch.py` 设为 **640×480**；`run_reference_path_semantic_map.py` 从 `BaseEnv` 读取该分辨率写入 `env_frame_*`。若你在其它任务里自定义分辨率，须与 `default_semantic_map_args` 的 `env_frame_*` 一致，否则可能在 `AvgPool2d` 等步骤出现维度不匹配。

**水平视场 hfov**：NPC 用 `hfov`（度）做深度→栅格投影。上述 Dataset 中 `SensorConfig.fov` 已与 NPC **`arguments.py`/co_test 的 60°** 对齐；脚本默认使用该相机的 `fov`，也可用 **`--semantic-hfov`** 覆盖。相机高度默认 **1.32 m**（与 NPC），可用 **`--semantic-camera-height`** 或环境变量 **`SEM_MAP_CAMERA_HEIGHT`**。

**位姿 / 重影**：`SemanticMap` 用相邻帧 `gps + compass` 的相对位姿 `pose_delta` 滚动地图。若首帧仍用「上一时刻 = 全零」与当前绝对坐标算增量，会把起始位置误当成一步特大位移，全局图错位并出现**同一障碍物拖影、两层语义**。NPC `mapping.py` 已对**每个 env 首帧**将 `pose_delta` 置零，仅融合观测；之后仍用真实帧间增量。

---

## 代码入口

```python
from semantic_map import (
    SemanticMapRunner,
    default_semantic_map_args,
    build_nav_observations_from_isaac_camera,
    configure_npc_imports,
)

configure_npc_imports("/path/to/NPC")  # 或使用环境变量

runner = SemanticMapRunner(
    {"chair", "table", "wall"},  # 与场景 USD 语义/任务一致的类别名
    npc_root="/path/to/NPC",
)
runner.reset()

# 每帧：从 Isaac 相机构造观测（与 NPC test_scene_construct 对齐）
obs = build_nav_observations_from_isaac_camera(
    rgb=rgb_uint8_hw3,
    depth_m=depth_from_isaac,  # distance_to_image_plane
    gps_xy=position_xy - scene_center_xy,
    yaw=yaw_rad,
    max_depth=args.max_depth,
)
state, map_features = runner.update(obs, step=t, update_global=True)
# state.get_obstacle_map(0), state.get_frontier_map(0), state.global_map, ...
```

### 从场景物体 id 收集类别名

若你使用与 NPC 相同的 `semantics_objects.csv`（EmbodiedAI 根目录也有 `semantics_objects.csv`）：

```python
from semantic_map.isaac_integration import categories_from_prim_ids

names = categories_from_prim_ids(["obj_xxx", "obj_yyy"], "semantics_objects.csv")
runner = SemanticMapRunner(names, npc_root="...")
```

## 与 `simulator.utils.semantic_utils` 的关系

- **USD 语义标注**：仍可用 `omni.isaac.core.utils.semantics` + `ItemLookup` 在 prim 上写 `semantic_label`（与 NPC `test_scene_construct.py` 一致）。
- **在线语义地图**：由本目录桥接的 `SemanticMap` 根据 **RGB-D + 位姿 + GroundedSAM** 更新栅格，二者互补。

## 故障排除

1. **`ModuleNotFoundError: mapping` / `home_robot`**  
   检查 `NPC_REPO_ROOT` 是否正确，且 NPC 下存在 `mapping.py` 与 `home-robot/src/home_robot`。

2. **GroundingDINO / SAM2 报错**  
   按 NPC 中 `grounded_sam_perception.py` 的配置准备权重与配置文件路径（常为机器相关路径，需自行改环境或 fork）。

3. **实例内存日志目录**  
   NPC 内 `mapping.py` 中 `InstanceMemory` 的 `log_dir` 仍可能写死为 NPC 路径；如需改到 EmbodiedAI 下，可在 NPC 侧改一行或使用符号链接。
