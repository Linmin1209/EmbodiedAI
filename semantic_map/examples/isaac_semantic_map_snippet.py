# -*- coding: utf-8 -*-
"""
示例片段：在 Isaac Sim 主循环中调用语义地图（需已配置 NPC 与 GPU 依赖）。

运行前（在 EmbodiedAI 仓库根目录）:
    export PYTHONPATH=/data1/linmin/EmbodiedAI:$PYTHONPATH
    export NPC_REPO_ROOT=/data1/linmin/NPC

本文件不直接启动 SimulationApp，仅展示与 NPC test_scene_construct 对齐的调用顺序。
"""

from __future__ import annotations

# 在创建 SimulationApp 并 import isaac 之后，再执行下列逻辑：

# from semantic_map import (
#     SemanticMapRunner,
#     build_nav_observations_from_isaac_camera,
#     default_semantic_map_args,
#     configure_npc_imports,
# )
#
# configure_npc_imports()  # 或传 npc_root=
#
# semantic_categories = {"chair", "table"}  # 与任务/GroundedSAM 词表一致
# args = default_semantic_map_args()
# runner = SemanticMapRunner(semantic_categories, args=args)
# runner.reset()
#
# # 每步（示意）:
# # rgb: (H,W,3) uint8
# # depth: camera.get_current_frame()["distance_to_image_plane"]
# # position: 机器人世界坐标 xy；scene_center: 场景包围盒中心 xy（与 NPC 一致）
# # yaw: 弧度
# obs = build_nav_observations_from_isaac_camera(
#     rgb=rgb,
#     depth_m=depth,
#     gps_xy=position[:2] - scene_center,
#     yaw=yaw,
#     max_depth=args.max_depth,
# )
# semantic_state, map_features = runner.update(obs, step=step_idx, update_global=True)
# obstacle = semantic_state.get_obstacle_map(0)
# frontier = semantic_state.get_frontier_map(0)
