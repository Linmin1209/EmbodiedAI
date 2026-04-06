"""
与 NPC `arguments.get_args()` 中语义地图相关的默认参数，用于构造 `argparse.Namespace`
供 `SemanticMap(args, categories)` 使用。
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Any

from .npc_paths import get_embodiedai_root, resolve_npc_repo_root


def default_semantic_map_args(
    npc_root: str | Path | None = None,
    **overrides: Any,
) -> Namespace:
    """
    默认与 `/data1/linmin/NPC/arguments.py` 中训练/建图相关项对齐，
    可通过关键字或环境变量覆盖。

    环境变量示例：
    - SEM_GPU_ID, MAP_GPU_ID
    - SEM_MAP_DUMP, SEM_MAP_EXP_NAME
    - SEM_MAP_LEGEND_PATH（图例 PNG，可为 None）
    """
    npc = resolve_npc_repo_root(str(npc_root) if npc_root is not None else None)
    embodied_root = get_embodiedai_root()
    dump = os.environ.get("SEM_MAP_DUMP", str(embodied_root / "tmp" / "semantic_map"))
    legend_default = os.environ.get("SEM_MAP_LEGEND_PATH")
    if legend_default:
        lp = Path(legend_default).expanduser()
        legend_default = str(lp.resolve()) if lp.is_file() else None
    if legend_default is None:
        cand = npc / "hssd_legend.png"
        if cand.is_file():
            legend_default = str(cand)
        else:
            emb_cand = embodied_root / "tmp" / "hssd_legend.png"
            legend_default = str(emb_cand) if emb_cand.is_file() else None

    defaults: dict[str, Any] = {
        "sem_gpu_id": int(os.environ.get("SEM_GPU_ID", "0")),
        "map_gpu_id": int(os.environ.get("MAP_GPU_ID", "0")),
        "dump_location": dump,
        "exp_name": os.environ.get("SEM_MAP_EXP_NAME", "embodiedai_sem_map"),
        "num_envs": 1,
        "print_images": int(os.environ.get("SEM_MAP_PRINT_IMAGES", "0")),
        "legend_path": legend_default,
        "env_frame_width": 640,
        "env_frame_height": 480,
        "frame_width": 320,
        "frame_height": 180,
        "camera_height": 1.32,
        "hfov": 60.0,
        "min_depth": 0.1,
        "max_depth": 10.0,
        "similar_threshold": 0.8,
        "global_downscaling": 2,
        "vision_range": 100,
        "map_resolution": 5,
        "du_scale": 4,
        "map_size_cm": 4800,
        "cat_pred_threshold": 5.0,
        "map_pred_threshold": 1.0,
        "exp_pred_threshold": 1.0,
        # True 会启用 InstanceMemory 按步拼接 RGB；若各帧 (H,W) 不一致会 torch.cat 报错（如 1280 vs 720）
        "record_instance_ids": False,
        # 以下字段被 `SemanticMap` 间接使用或与其他 NPC 脚本共用，给默认值避免 AttributeError
        "scene_path": "",
        "results_path": "",
        "collision_threshold": 0.20,
        "max_steps": 500,
        "turn_angle": 30.0,
        "forward_cm": 20.0,
        "discrete_actions": True,
        "step_size": 10,
        "obs_dilation_selem_radius": 3.0,
        "goal_dilation_selem_radius": 10,
        "coll_thre": 0.20,
        "plan_print": 1,
        "plan_visualize": 0,
        "num_local_steps": 20,
        "exp_strategy": "seen_frontier",
        "num_sem_categories": 28,
        # NPC mapping.SemanticMap：为 True 时不在本进程创建 GroundedSAMPerception，改用 segmentation_client
        "skip_inprocess_grounded_sam": False,
        "segmentation_client": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)
