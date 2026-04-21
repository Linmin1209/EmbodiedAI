"""
动作幅值 ``value`` 的归一化 / 反归一化。

- 默认与 ``build_nav_dataset`` 一致：按 ``max_action_value`` 做 **scale** 归一化
  ``v_norm = v_raw / scale``，推理时 ``v_raw = v_norm * scale``。
- 可选 **minmax**：由 ``train.json``+``val.json`` 统计 min/max 后写入 ``value_norm.json``。

训练数据目录下会生成或读取 ``value_norm.json``，测试/部署时用同一文件反归一化。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

VALUE_NORM_FILENAME = "value_norm.json"


def _flatten_nav_samples(raw: Any) -> List[dict]:
    if not raw:
        return []
    if isinstance(raw, dict):
        raise ValueError("顶层 JSON 需为 list")
    first = raw[0]
    if isinstance(first, dict):
        return list(raw)
    out: List[dict] = []
    for demo in raw:
        if isinstance(demo, list):
            out.extend(demo)
        elif isinstance(demo, dict):
            out.append(demo)
    return out


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_value_norm(data_root: str) -> Dict[str, Any]:
    path = os.path.join(os.path.abspath(data_root), VALUE_NORM_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_value_norm(data_root: str, cfg: Dict[str, Any]) -> str:
    root = os.path.abspath(data_root)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, VALUE_NORM_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path


def default_scale_config(scale: float) -> Dict[str, Any]:
    s = float(scale)
    return {
        "version": 1,
        "mode": "scale",
        "scale": s,
        "formula_norm": "v_norm = v_raw / scale",
        "formula_denorm": "v_raw = v_norm * scale",
    }


def default_minmax_config(
    vmin: float, vmax: float, eps: float = 1e-8
) -> Dict[str, Any]:
    return {
        "version": 1,
        "mode": "minmax",
        "min": float(vmin),
        "max": float(vmax),
        "eps": float(eps),
        "formula_norm": "v_norm = (v_raw - min) / (max - min + eps)",
        "formula_denorm": "v_raw = v_norm * (max - min + eps) + min",
    }


def ensure_value_norm_file(
    data_root: str,
    *,
    default_scale_if_missing_meta: float = 0.2,
) -> Dict[str, Any]:
    """
    若 ``value_norm.json`` 不存在，则根据 ``dataset_meta.json`` 的 ``max_action_value``
    写默认 **scale** 配置；若无 meta，则用 ``default_scale_if_missing_meta``。
    """
    root = os.path.abspath(data_root)
    path = os.path.join(root, VALUE_NORM_FILENAME)
    if os.path.isfile(path):
        return load_value_norm(root)

    meta_path = os.path.join(root, "dataset_meta.json")
    scale = default_scale_if_missing_meta
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("max_action_value") is not None:
            scale = float(meta["max_action_value"])

    cfg = default_scale_config(scale)
    cfg["source"] = "ensure_value_norm_file_auto"
    save_value_norm(root, cfg)
    return cfg


def collect_values_from_split(data_root: str, json_names: List[str]) -> List[float]:
    vals: List[float] = []
    root = os.path.abspath(data_root)
    for name in json_names:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        raw = _load_json(p)
        for s in _flatten_nav_samples(raw):
            if isinstance(s.get("value"), (int, float)):
                vals.append(float(s["value"]))
    return vals


def compute_and_save_minmax(
    data_root: str,
    json_names: Optional[List[str]] = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """扫描 ``train.json`` / ``val.json`` 中所有 value，写 minmax ``value_norm.json``。"""
    if json_names is None:
        json_names = ["train.json", "val.json"]
    vals = collect_values_from_split(data_root, json_names)
    if not vals:
        raise RuntimeError(f"未在 {data_root} 的 {json_names} 中找到任何 value 字段")
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        raise RuntimeError(f"value 全为常数 {vmin}，无法用 minmax，请改用 scale 模式")
    cfg = default_minmax_config(vmin, vmax, eps=eps)
    cfg["num_samples_scanned"] = len(vals)
    cfg["source"] = "compute_and_save_minmax"
    save_value_norm(data_root, cfg)
    return cfg


def normalize_value(
    v: Union[float, "torch.Tensor"],
    cfg: Dict[str, Any],
) -> Union[float, "torch.Tensor"]:
    import torch

    mode = cfg.get("mode", "scale")
    if mode == "none":
        return v
    if mode == "scale":
        s = float(cfg["scale"])
        if isinstance(v, torch.Tensor):
            return v / s
        return float(v) / s
    if mode == "minmax":
        lo = float(cfg["min"])
        hi = float(cfg["max"])
        eps = float(cfg.get("eps", 1e-8))
        if isinstance(v, torch.Tensor):
            return (v - lo) / (hi - lo + eps)
        return (float(v) - lo) / (hi - lo + eps)
    raise ValueError(f"未知 value_norm mode: {mode}")


def denormalize_value(
    v: Union[float, "torch.Tensor"],
    cfg: Dict[str, Any],
) -> Union[float, "torch.Tensor"]:
    import torch

    mode = cfg.get("mode", "scale")
    if mode == "none":
        return v
    if mode == "scale":
        s = float(cfg["scale"])
        if isinstance(v, torch.Tensor):
            return v * s
        return float(v) * s
    if mode == "minmax":
        lo = float(cfg["min"])
        hi = float(cfg["max"])
        eps = float(cfg.get("eps", 1e-8))
        if isinstance(v, torch.Tensor):
            return v * (hi - lo + eps) + lo
        return float(v) * (hi - lo + eps) + lo
    raise ValueError(f"未知 value_norm mode: {mode}")


def quantize_value_physical(
    v: float,
    *,
    step: float,
    vmax: Optional[float] = None,
    vmin: float = 0.0,
) -> float:
    """
    将**物理空间**的 value 量化为 ``step`` 的整数倍（与 ``build_nav_dataset`` 的 ``step_value`` /
    ``ModelConfig.value_step_physical`` 对齐）。

    若给定 ``vmax``，先将 ``v`` 限制在 ``[vmin, vmax]``，再 ``round(v / step) * step``；
    最后按 ``step`` 的小数位数做 ``round``，减轻浮点误差。
    """
    if step <= 0:
        return float(v)
    x = float(v)
    if vmax is not None and vmax > 0:
        x = max(vmin, min(float(vmax), x))
    n = round(x / step)
    q = n * step
    step_s = f"{step:.12f}".rstrip("0").rstrip(".")
    if "." in step_s:
        nd = len(step_s.split(".")[1])
    else:
        nd = 0
    return float(round(q, nd))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="生成或覆盖 value_norm.json（value 归一化参数）")
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="含 train.json / val.json 的目录",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=("scale", "minmax"),
        default="scale",
        help="scale: 使用 max_action_value（dataset_meta）或 --scale；minmax: 扫描 JSON 统计",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="mode=scale 时覆盖；默认读 dataset_meta.json 的 max_action_value",
    )
    args = p.parse_args()
    root = os.path.abspath(args.data_root)
    if args.mode == "minmax":
        cfg = compute_and_save_minmax(root)
        print(f"已写入 minmax: {os.path.join(root, VALUE_NORM_FILENAME)}")
        print(cfg)
        return
    scale = args.scale
    if scale is None:
        meta_path = os.path.join(root, "dataset_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            scale = float(meta.get("max_action_value", 0.2))
        else:
            scale = 0.2
    cfg = default_scale_config(scale)
    cfg["source"] = "cli_scale"
    path = save_value_norm(root, cfg)
    print(f"已写入 scale: {path} scale={scale}")


if __name__ == "__main__":
    main()
