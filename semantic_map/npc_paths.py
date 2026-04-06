"""解析 NPC（含 home_robot）仓库根路径。"""

from __future__ import annotations

import os
from pathlib import Path


def get_embodiedai_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_npc_repo_root(explicit: str | None = None) -> Path:
    """
    优先顺序：
    1. 参数 explicit
    2. 环境变量 NPC_REPO_ROOT
    3. 环境变量 EMBODIEDAI_NPC_ROOT
    4. 默认 /data1/linmin/NPC（可按本机修改）
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("NPC_REPO_ROOT") or os.environ.get("EMBODIEDAI_NPC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path("/data1/linmin/NPC").resolve()


def home_robot_src_path(npc_root: Path | None = None) -> Path:
    root = npc_root or resolve_npc_repo_root()
    return (root / "home-robot" / "src").resolve()
