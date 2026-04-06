"""
语义地图（Semantic Map）集成：桥接 NPC 仓库中的 `mapping.SemanticMap` 与 Isaac Sim 观测。

使用前请配置 NPC 仓库路径（见 `semantic_map/README.md`）。
"""

from .args import default_semantic_map_args
from .isaac_integration import build_nav_observations_from_isaac_camera
from .isolated_segmentation import IsolatedGroundedSAMClient, npc_vocabulary_from_category_list
from .mapper import SemanticMapRunner
from .npc_bridge import configure_npc_imports, get_semantic_map_class

__all__ = [
    "configure_npc_imports",
    "get_semantic_map_class",
    "default_semantic_map_args",
    "build_nav_observations_from_isaac_camera",
    "SemanticMapRunner",
    "IsolatedGroundedSAMClient",
    "npc_vocabulary_from_category_list",
]
