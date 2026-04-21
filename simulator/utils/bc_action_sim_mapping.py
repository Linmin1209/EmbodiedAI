"""
将 **BC 训练标注**（``ImitationDataset.action_map``）与 **仿真 ``PositionController``**
对 ``command[0]`` 的约定对齐。

训练数据（``imitation_learning_project/dataset/dataset.py``）::

    0 → \"move forward\"
    1 → \"turn left\"
    2 → \"turn right\"

仿真 ``simulator/controllers/position_controller.py``::

    0 / 'w' → 前进
    1 / 's' → 后退
    2 / 'a' → 左转（yaw +）
    3 / 'd' → 右转（yaw -）

因此 **若直接把训练 argmax id 当作 ``command[0]``**，会出现：
``1`` 被当成 **后退**，``2`` 被当成 **左转**，与标注语义完全错位。

``env.step([action])`` 之前应对 ``[train_id, value]`` 做映射::

    train 0 → sim 0
    train 1 → sim 2
    train 2 → sim 3
"""
from __future__ import annotations

from typing import Any, List, Sequence

# train_action_id -> position_controller command[0]
TRAIN_TO_SIM_ACTION_ID: tuple[int, int, int] = (0, 2, 3)


def train_action_tuple_to_sim(action: Sequence[Any]) -> List[Any]:
    """
    ``[train_action_id, value]`` → ``[sim_action_id, value]``，供 ``BaseEnv.step`` 使用。
    value 原样传递。
    """
    if len(action) < 2:
        return list(action)
    aid = int(action[0])
    if 0 <= aid < len(TRAIN_TO_SIM_ACTION_ID):
        return [TRAIN_TO_SIM_ACTION_ID[aid], action[1]]
    return [action[0], action[1]]
