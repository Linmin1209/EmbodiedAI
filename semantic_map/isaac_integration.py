"""
从 Isaac Sim 相机帧构造 NPC / home_robot 风格的观测，供 `SemanticMap.map()` 使用。

逻辑对齐 `NPC/test_scene_construct.py` 中 Observations 的构建方式。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np


def _fill_depth_m(max_depth: float) -> float:
    """无效深度统一用上限（米）。"""
    return float(max_depth)


def _scalar_to_fill_m(value: Any, max_depth: float) -> float:
    """标量深度转米；None / 非数 / ≤0 时用 max_depth。"""
    if value is None:
        return _fill_depth_m(max_depth)
    try:
        v = float(np.asarray(value).item())
    except (TypeError, ValueError, OverflowError):
        return _fill_depth_m(max_depth)
    if not np.isfinite(v) or v <= 0:
        return _fill_depth_m(max_depth)
    return v


def _depth_to_hw1(
    depth_m: Any,
    rgb: np.ndarray,
    *,
    max_depth: float,
) -> np.ndarray:
    """
    将 Isaac `distance_to_image_plane` 转为 NPC `preprocess_depth` 所需的 (H, W, 1) float32。

    首帧或未渲染完成时，深度有时是 0 维标量，直接进 NPC 会触发
    ``IndexError: too many indices for array: array is 0-dimensional``.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb 期望 (H,W,3)，当前 shape={getattr(rgb, 'shape', None)}")
    H, W = int(rgb.shape[0]), int(rgb.shape[1])

    if depth_m is None:
        return np.full((H, W, 1), _fill_depth_m(max_depth), dtype=np.float32)

    d = np.asarray(depth_m)
    # Isaac 偶发返回含 None 的 object 数组，float() 会失败
    if d.dtype == object:
        try:
            d = d.astype(np.float64)
        except (TypeError, ValueError):
            return np.full((H, W, 1), _fill_depth_m(max_depth), dtype=np.float32)

    if d.size == 0:
        return np.full((H, W, 1), _fill_depth_m(max_depth), dtype=np.float32)

    if d.ndim == 0:
        fill = _scalar_to_fill_m(d.item(), max_depth)
        plane = np.full((H, W), fill, dtype=np.float32)
    elif d.ndim == 1:
        if d.size == H * W:
            plane = d.reshape(H, W).astype(np.float32)
        else:
            raise ValueError(
                f"depth 1D 长度 {d.size} 与 RGB H*W={H * W} 不一致，无法 reshape"
            )
    elif d.ndim == 2:
        if d.shape[0] == H and d.shape[1] == W:
            plane = d.astype(np.float32)
        elif d.size == 1:
            fill = _scalar_to_fill_m(d.reshape(-1)[0], max_depth)
            plane = np.full((H, W), fill, dtype=np.float32)
        else:
            raise ValueError(
                f"depth 2D shape {d.shape} 与 RGB ({H},{W}) 不一致"
            )
    elif d.ndim == 3:
        if d.shape[0] != H or d.shape[1] != W:
            raise ValueError(
                f"depth 3D shape {d.shape} 与 RGB ({H},{W},*) 不一致"
            )
        plane = d[:, :, 0].astype(np.float32)
    else:
        raise ValueError(f"不支持的 depth ndim={d.ndim}, shape={d.shape}")

    depth = np.expand_dims(plane, axis=2)
    inf_mask = np.isinf(depth)
    depth[inf_mask] = float(max_depth)
    nan_mask = np.isnan(depth)
    depth[nan_mask] = float(max_depth)
    return depth


def build_nav_observations_from_isaac_camera(
    rgb: np.ndarray,
    depth_m: Any,
    gps_xy: np.ndarray,
    yaw: float,
    *,
    max_depth: float,
    num_envs: int = 1,
) -> List[Any]:
    """
    构造与 `test_scene_construct.py` 一致的观测列表（长度为 num_envs）。

    参数
    ----
    rgb:
        (H, W, 3) uint8 RGB。
    depth_m:
        Isaac `distance_to_image_plane`，形状 (H, W) 或 (H, W, 1)，单位米；
        inf 会在内部替换为 max_depth。
    gps_xy:
        相对场景中心的平面坐标 [x, y]，与 NPC 脚本中 `position[:2] - scene_center` 一致。
    yaw:
        航向角（弧度），与 NPC 中 `compass` 一致。
    max_depth:
        深度上限（米），与 `args.max_depth` 一致。
    """
    rgb = np.asarray(rgb)
    depth = _depth_to_hw1(depth_m, rgb, max_depth=max_depth)

    obs_list: List[Any] = []
    for _ in range(num_envs):
        o = SimpleNamespace(
            rgb=rgb,
            depth=depth,
            gps=np.asarray(gps_xy, dtype=np.float64),
            compass=np.asarray([yaw], dtype=np.float64),
            camera_pose=None,
            task_observations={},
        )
        obs_list.append(o)
    return obs_list


def categories_from_prim_ids(
    prim_ids: Sequence[str],
    csv_path: str,
) -> List[str]:
    """
    使用 EmbodiedAI 自带的 `ItemLookup`（与 NPC 同源逻辑）从物体 id 得到类别名列表，
    供 `SemanticMap(..., categories)` 使用。

    返回去重后的有序列表（顺序不保证稳定，如需稳定可先 sorted）。
    """
    from simulator.utils.semantic_utils import ItemLookup

    lookup = ItemLookup(csv_path)
    seen: set[str] = set()
    out: List[str] = []
    for pid in prim_ids:
        name = lookup.get_item_name_by_id(str(pid))
        if name and len(str(name)) > 0 and name not in seen:
            seen.add(name)
            out.append(name)
    return out
