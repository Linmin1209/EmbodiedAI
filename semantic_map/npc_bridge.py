"""
将 NPC 仓库加入 sys.path，并加载其 `mapping.SemanticMap`。

`SemanticMap` 实现位于 NPC 的 `mapping.py`，依赖 `home_robot`、GroundingDINO、SAM2 等，
本包不负责安装这些依赖，仅提供统一入口。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Type

from .npc_paths import home_robot_src_path, resolve_npc_repo_root

_npc_configured: Path | None = None


def configure_npc_imports(npc_root: str | Path | None = None) -> Path:
    """
    把 NPC 根目录与 `home-robot/src` 插入 sys.path 最前，保证 `import mapping` 等解析到 NPC。

    重复调用时，若根路径相同则跳过；若不同则重新插入。
    """
    global _npc_configured
    root = resolve_npc_repo_root(str(npc_root) if npc_root is not None else None)
    hr_src = home_robot_src_path(root)

    if _npc_configured == root:
        return root

    for p in (str(hr_src), str(root)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    _npc_configured = root
    return root


def get_semantic_map_class(npc_root: str | Path | None = None) -> Type[Any]:
    """返回 NPC 的 `SemanticMap` 类。"""
    configure_npc_imports(npc_root)
    import mapping as npc_mapping  # noqa: WPS433 — 运行时依赖 NPC 侧模块名

    return npc_mapping.SemanticMap


def get_npc_mapping_module(npc_root: str | Path | None = None) -> ModuleType:
    """返回已加载的 NPC `mapping` 模块（高级用法）。"""
    configure_npc_imports(npc_root)
    import mapping as npc_mapping  # noqa: WPS433

    return npc_mapping


def preprocess_depth_like_npc(
    depth: Any,
    min_d: float,
    max_d: float,
) -> Any:
    """
    与 NPC `mapping.preprocess_depth` 行为一致，避免重复依赖时可直接调用本函数。

    depth: (H, W) 或 (H, W, 1)，单位为米（与 Isaac `distance_to_image_plane` 一致）。
    返回 (H, W, 1)，单位为厘米。
    """
    configure_npc_imports()
    import mapping as npc_mapping  # noqa: WPS433

    fn: Callable[..., Any] = npc_mapping.preprocess_depth
    return fn(depth, min_d, max_d)
