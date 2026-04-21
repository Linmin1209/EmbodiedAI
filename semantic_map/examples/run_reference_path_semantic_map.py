# -*- coding: utf-8 -*-
"""
与 `tests/test_scene/test_agent.py` 相同方式加载 DatasetLoader + BaseEnv + ReferencePathAgent，
用前视相机 RGB-D 与位姿驱动 NPC `SemanticMap`；默认在子进程中运行 GroundedSAM（与 README 一致）。

在 EmbodiedAI 仓库根目录执行::

    export PYTHONPATH=/data2/linmin/EmbodiedAI:$PYTHONPATH
    export NPC_REPO_ROOT=/data2/linmin/NPC

    python semantic_map/examples/run_reference_path_semantic_map.py 0 \\
        --output-dir ./semantic_map/outputs/task_0 --headless

相机与建图参数与 NPC ``Co-scene-construct/stretch.py`` + ``arguments.get_args()`` 对齐：
``DatasetLoader`` 中前视相机为 640×480、水平视场约 60°、15Hz、focal/aperture 在 ``VisionSensor.init`` 中与 stretch 一致；
SemanticMap 使用 ``env_frame`` 与该分辨率一致，``camera_height=1.32``、``min/max_depth=0.1/10``、``du_scale=1`` 等可由环境变量 ``SEM_MAP_*`` 覆盖。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# 路径：必须在 import simulator / isaac 之前保证 EmbodiedAI 在 PYTHONPATH 中
# -----------------------------------------------------------------------------
_EMBODIED_ROOT = Path(__file__).resolve().parents[2]
if str(_EMBODIED_ROOT) not in sys.path:
    sys.path.insert(0, str(_EMBODIED_ROOT))

_DATASET_DEFAULT = _EMBODIED_ROOT / "resource" / "datasets" / "all_task"
if not _DATASET_DEFAULT.is_dir():
    _DATASET_DEFAULT = _EMBODIED_ROOT / "resource" / "datasets" / "merge_task"

# 与 /NPC/arguments.py + Co-scene-construct/stretch.py 中相机/地图一致（作默认值与文档）
_NPC_ARGS_CAMERA_HEIGHT = 1.32
_NPC_ARGS_HFOV_DEG = 60.0
_NPC_ARGS_MIN_DEPTH = 0.1
_NPC_ARGS_MAX_DEPTH = 10.0
_NPC_ARGS_DU_SCALE = 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReferencePathAgent + NPC SemanticMap（EmbodiedAI / Isaac Sim）"
    )
    p.add_argument(
        "task_index",
        type=int,
        help="DatasetLoader 中的任务索引（与 test_agent.py 第一个参数相同）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./semantic_map/outputs/run",
        help="输出根目录：frames/、maps/、npc_dump/ 等",
    )
    p.add_argument("--headless", action="store_true", help="无头渲染（SimulationApp）")
    p.add_argument(
        "--skip-semantic-map",
        action="store_true",
        help="仅保存相机帧与位姿，不初始化 SemanticMap（调试用）",
    )
    p.add_argument(
        "--inprocess-grounded-sam",
        action="store_true",
        help="在同进程内加载 GroundedSAM（易与 Isaac 冲突；默认使用子进程分割）",
    )
    p.add_argument(
        "--camera-key",
        type=str,
        default="robot0_front_camera",
        help="NavigateTask 观测里相机 dict 的键（与 VisionSensor.name 一致）",
    )
    p.add_argument(
        "--semantic-hfov",
        type=float,
        default=None,
        help="覆盖语义地图水平视场（度）；默认使用相机 SensorConfig.fov（Dataset 已为 60，与 NPC arguments 一致）",
    )
    p.add_argument(
        "--semantic-camera-height",
        type=float,
        default=None,
        help="相机高度（米）；默认 1.32（与 NPC arguments.camera_height）",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=str(_DATASET_DEFAULT),
        help="processed_config.json 所在数据集根目录（默认同 test_agent：all_task，不存在则用 merge_task）",
    )
    p.add_argument(
        "--scene-path",
        type=str,
        default="/data2/linmin/NPC/hssd_scene_new",
        help="场景 USD 根目录（与 DatasetLoader / test_agent 一致）",
    )
    p.add_argument(
        "--robot-usd",
        type=str,
        default=str(_EMBODIED_ROOT / "resource" / "robots" / "stretch" / "stretch_pos.usd"),
        help="机器人 USD 路径",
    )
    p.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="SimulationApp device",
    )
    p.add_argument(
        "--categories",
        type=str,
        default="",
        help="逗号分隔的语义类别名（与 GroundedSAM 词表一致）；为空则使用 NPC HSSD-28 类",
    )
    return p.parse_args()


def _sensor_config_for_camera(cfg, camera_key: str):
    from simulator.core.config import SensorConfig

    for robot in cfg.task.robots or []:
        for s in robot.sensors or []:
            if isinstance(s, SensorConfig) and s.name == camera_key:
                return s
    return None


def _semantic_categories_from_args(args: argparse.Namespace) -> list[str]:
    raw = (args.categories or "").strip()
    if raw:
        return [c.strip() for c in raw.split(",") if c.strip()]
    # 需已 configure_npc_imports，使 NPC 在 sys.path 中
    from constant import hssd_28categories_indexes

    return list(hssd_28categories_indexes.values())


def main() -> None:
    args = _parse_args()
    out_root = Path(args.output_dir).resolve()
    frames_dir = out_root / "frames"
    maps_dir = out_root / "maps"
    frames_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    # 注册 Interactive_Scene（与 test_load / test_agent 一致）
    from simulator.scenes import Interactive_Scene  # noqa: F401
    from simulator.core.env import BaseEnv
    from simulator.core.dataset import DatasetLoader
    from simulator.agents import ReferencePathAgent
    from simulator.utils.scene_utils import compute_enclosing_square

    from semantic_map import (
        SemanticMapRunner,
        build_nav_observations_from_isaac_camera,
        configure_npc_imports,
        default_semantic_map_args,
    )
    from semantic_map.isolated_segmentation import (
        IsolatedGroundedSAMClient,
        npc_vocabulary_from_category_list,
    )

    npc_root = configure_npc_imports(os.environ.get("NPC_REPO_ROOT") or os.environ.get("EMBODIEDAI_NPC_ROOT"))

    loader = DatasetLoader(
        root_dir=args.dataset_root,
        scene_path=args.scene_path,
        robot_path=str(Path(args.robot_usd).resolve()),
        headless=args.headless,
        device_id=args.device_id,
    )
    if args.task_index < 0 or args.task_index >= len(loader):
        raise SystemExit(f"task_index 越界: {args.task_index}, len={len(loader)}")

    cfg = loader[args.task_index]
    cfg.sim.headless = bool(args.headless)
    cfg.sim.device = args.device_id

    sensor_cfg = _sensor_config_for_camera(cfg, args.camera_key)
    if sensor_cfg is None:
        raise SystemExit(
            f"未找到相机 sensor name={args.camera_key!r}；请检查 DatasetLoader 的 robot 传感器配置。"
        )
    if "depth" not in (sensor_cfg.modals or []):
        raise SystemExit(
            f"相机 {args.camera_key} 未启用 depth modal；请在 VisionSensor.modals 中加入 depth。"
        )

    res = sensor_cfg.resolution or (640, 480)
    env_w, env_h = int(res[0]), int(res[1])
    if args.semantic_hfov is not None:
        hfov = float(args.semantic_hfov)
    else:
        hfov = float(sensor_cfg.fov if sensor_cfg.fov is not None else _NPC_ARGS_HFOV_DEG)

    sem_categories = _semantic_categories_from_args(args)

    # dump / 实验名：输出到用户指定目录下
    dump_root = str(out_root)
    os.environ.setdefault("SEM_MAP_DUMP", dump_root)
    exp_name = os.environ.get("SEM_MAP_EXP_NAME", "reference_path_run")

    map_arg_overrides: dict = {
        "dump_location": dump_root,
        "exp_name": exp_name,
        "env_frame_width": env_w,
        "env_frame_height": env_h,
        "hfov": hfov,
        "sem_gpu_id": int(os.environ.get("SEM_GPU_ID", "0")),
        "map_gpu_id": int(os.environ.get("MAP_GPU_ID", "0")),
        "print_images": int(os.environ.get("SEM_MAP_PRINT_IMAGES", "1")),
        # 与 NPC arguments.py 中建图项一致（co_test_scene_construct / SemanticMap）
        "camera_height": float(
            args.semantic_camera_height
            if args.semantic_camera_height is not None
            else os.environ.get("SEM_MAP_CAMERA_HEIGHT", str(_NPC_ARGS_CAMERA_HEIGHT))
        ),
        "min_depth": float(os.environ.get("SEM_MAP_MIN_DEPTH", str(_NPC_ARGS_MIN_DEPTH))),
        "max_depth": float(os.environ.get("SEM_MAP_MAX_DEPTH", str(_NPC_ARGS_MAX_DEPTH))),
        "du_scale": int(os.environ.get("SEM_MAP_DU_SCALE", str(_NPC_ARGS_DU_SCALE))),
        "collision_threshold": float(os.environ.get("SEM_MAP_COLLISION_THRESHOLD", "0.10")),
        "coll_thre": float(os.environ.get("SEM_MAP_COLL_THRE", "0.10")),
    }

    seg_client = None
    if not args.skip_semantic_map:
        if args.inprocess_grounded_sam:
            map_arg_overrides["skip_inprocess_grounded_sam"] = False
            map_arg_overrides["segmentation_client"] = None
        else:
            map_arg_overrides["skip_inprocess_grounded_sam"] = True
            seg_client = IsolatedGroundedSAMClient(
                npc_root=npc_root,
                vocabulary=npc_vocabulary_from_category_list(sem_categories),
                sem_gpu_id=map_arg_overrides["sem_gpu_id"],
                similar_threshold=float(
                    os.environ.get("SEM_MAP_SIMILAR_THRESHOLD", "0.8")
                ),
            )
            map_arg_overrides["segmentation_client"] = seg_client

    sm_args = default_semantic_map_args(npc_root=npc_root, **map_arg_overrides)
    runner: SemanticMapRunner | None = None
    if not args.skip_semantic_map:
        runner = SemanticMapRunner(sem_categories, args=sm_args, npc_root=npc_root)
        runner.reset()

    env = BaseEnv(cfg)
    scene_prim = env.scenes[0].scene_prim_dict["Scene"]
    aabb = scene_prim.get_aabb()
    center_xy, _, _ = compute_enclosing_square(aabb)
    scene_center = np.array([center_xy[0], center_xy[1]], dtype=np.float64)

    agent = ReferencePathAgent(cfg)
    obs_list = env.reset()
    agent.reset()

    step_idx = 0
    done = False
    while env.is_running and not done:
        action = agent.act(obs_list[0])
        obs_list, reward, done, info = env.step([action])
        o0 = obs_list[0]
        cam = o0.get(args.camera_key)
        if cam is None:
            raise RuntimeError(f"观测中缺少 {args.camera_key!r}，可用键: {list(o0.keys())!r}")

        rgb = cam["rgb"]
        depth_m = cam.get("depth")
        pos = np.asarray(o0["position"][0]).reshape(-1)
        yaw = float(o0["yaw"][0])
        gps_xy = np.array([pos[0] - scene_center[0], pos[1] - scene_center[1]], dtype=np.float64)

        np.savez_compressed(
            frames_dir / f"pose_{step_idx:06d}.npz",
            position=pos,
            yaw=yaw,
            gps_xy=gps_xy,
            scene_center=scene_center,
        )
        try:
            from PIL import Image

            Image.fromarray(rgb).save(frames_dir / f"rgb_{step_idx:06d}.png")
        except Exception:
            pass

        if runner is not None:
            nav_obs = build_nav_observations_from_isaac_camera(
                rgb=rgb,
                depth_m=depth_m,
                gps_xy=gps_xy,
                yaw=yaw,
                max_depth=sm_args.max_depth,
            )
            semantic_state, map_features = runner.update(
                nav_obs, step=step_idx, update_global=True
            )
            try:
                ob = semantic_state.get_obstacle_map(0)
                np.save(maps_dir / f"obstacle_{step_idx:06d}.npy", ob)
            except Exception:
                pass

        step_idx += 1
        if env.task[0].steps > len(agent.path) + 5:
            done = True
        if isinstance(done, (list, tuple)):
            done = bool(done[0])
        else:
            done = bool(done)

    env.close()
    if seg_client is not None:
        seg_client.close()

    print(f"完成。输出目录: {out_root}")


if __name__ == "__main__":
    main()
