"""
封装 NPC `SemanticMap`：初始化、reset、单步更新。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .args import default_semantic_map_args
from .npc_bridge import get_semantic_map_class

CategoryInput = Union[str, Iterable[str]]


class SemanticMapRunner:
    """
    将 NPC 的语义建图接到 EmbodiedAI / Isaac 流程。

    Example
    -------
    >>> runner = SemanticMapRunner({"chair", "table"}, npc_root="/data1/linmin/NPC")
    >>> runner.reset()
    >>> state, features = runner.update(obs_list, step=0, update_global=True)
    """

    def __init__(
        self,
        semantic_categories: CategoryInput,
        *,
        npc_root: Optional[str | Path] = None,
        args: Optional[Any] = None,
        **arg_overrides: Any,
    ) -> None:
        cats: List[str]
        if isinstance(semantic_categories, str):
            cats = [semantic_categories]
        else:
            cats = list(semantic_categories)
        SemanticMap = get_semantic_map_class(npc_root)
        self._args = args if args is not None else default_semantic_map_args(
            npc_root=npc_root, **arg_overrides
        )
        self._mapper = SemanticMap(self._args, cats)
        self._SemanticMap_cls = SemanticMap

    @property
    def inner(self) -> Any:
        """底层 NPC `SemanticMap` 实例。"""
        return self._mapper

    def reset(self) -> None:
        self._mapper.reset()

    def reset_map(self, env_index: int) -> None:
        self._mapper.reset_map(env_index)

    def update(
        self,
        observations: List[Any],
        *,
        step: int,
        update_global: bool = True,
    ) -> Tuple[Any, Any]:
        """
        调用 NPC `SemanticMap.map()`。

        返回
        ----
        semantic_map_state :
            `Categorical2DSemanticMapState`，含 `global_map`、`get_obstacle_map` 等。
        map_features :
            CPU 上的序列特征张量（与 NPC 一致）。
        """
        return self._mapper.map(observations, step=step, update_global=update_global)
