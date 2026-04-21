#!/usr/bin/env python3
"""
从多个导航数据根目录（如 extra_data / extra_data2 / gen_scene_data）扫描含 task.json 与
output.json 或 new_output.json 的 demo，统一帧格式后构建与 ``dataset/dataset.py`` 兼容的
``train.json`` / ``val.json``。

动作压缩（默认）：单帧步长 ``step_value=0.02``，每条样本 ``value`` 不超过 ``max_action_value=0.2``。
``train.json`` / ``val.json`` / ``dataset_meta.json`` 默认 ``indent=4`` 写入；需要单行紧凑可加 ``--compact-json``。

可选 ``--symlink_images``：在输出目录 ``image_symlinks/<demo_hash>/`` 下创建指向原 PNG 的软链接，
JSON 内路径改为链接路径（不占复制图像空间）。

帧格式兼容：
  - ``[action, pos, yaw, ...]``
  - ``[action, value, pos, yaw, ...]``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

try:
    from dataset.value_normalization import default_scale_config, save_value_norm
except ImportError:
    from value_normalization import default_scale_config, save_value_norm

# 与 ImitationDataset.action_map 一致；其它动作整段跳过
ALLOWED_ACTIONS = frozenset({"move forward", "turn left", "turn right"})


def normalize_frame_to_triple(frame: Any) -> Optional[Tuple[str, List[float], float]]:
    """展开为 (action, [x,y,z], yaw)。"""
    if not isinstance(frame, (list, tuple)) or len(frame) < 3:
        return None
    action = str(frame[0]).lower().strip()
    if isinstance(frame[1], (list, tuple)) and len(frame[1]) >= 3:
        pos = [float(frame[1][0]), float(frame[1][1]), float(frame[1][2])]
        yaw = float(frame[2])
        return action, pos, yaw
    if len(frame) >= 5 and isinstance(frame[2], (list, tuple)) and len(frame[2]) >= 3:
        pos = [float(frame[2][0]), float(frame[2][1]), float(frame[2][2])]
        yaw = float(frame[3])
        return action, pos, yaw
    return None


def load_output_json(demo_dir: Path) -> Dict[str, Any]:
    for name in ("new_output.json", "output.json"):
        p = demo_dir / name
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"未找到 new_output.json 或 output.json: {demo_dir}")


def _chunk_lengths_balanced(run_len: int, max_frames_per_chunk: int) -> List[int]:
    """
    将连续同动作总帧数拆成多段，每段帧数 ≤ max_frames_per_chunk。

    在必须多段时**均匀分配**（各段帧数最多相差 1），避免出现「先吃满 0.2 再剩极小尾差」。
    例如 run_len=12、max_frames=10：旧逻辑为 10+2 → 0.20+0.04；现逻辑为 6+6 → 0.12+0.12，
    更易满足「各段动作量尽量大一些（例如大于 0.1）」。
    """
    if run_len <= 0:
        return []
    K = max_frames_per_chunk
    if run_len <= K:
        return [run_len]
    num_chunks = (run_len + K - 1) // K  # ceil(run_len / K)
    base = run_len // num_chunks
    rem = run_len % num_chunks
    sizes: List[int] = []
    for i in range(num_chunks):
        sizes.append(base + (1 if i < rem else 0))
    m = max(sizes)
    if m > K:
        # 极罕见：退回贪心切段保证合法
        return _chunk_lengths_greedy(run_len, K)
    return sizes


def _chunk_lengths_greedy(run_len: int, max_frames_per_chunk: int) -> List[int]:
    """贪心：每段尽量取满 K，作兜底。"""
    if run_len <= 0:
        return []
    out: List[int] = []
    remaining = run_len
    while remaining > 0:
        take = min(max_frames_per_chunk, remaining)
        out.append(take)
        remaining -= take
    return out


def _end_pos_after_frames(
    triples: Sequence[Tuple[str, List[float], float]],
    chunk_start: int,
    chunk_frames: int,
) -> Tuple[List[float], float]:
    """chunk 覆盖下标 [chunk_start, chunk_start + chunk_frames - 1]；返回该段末位姿（与与原 merge 一致：取下一帧位姿若存在）。"""
    end_after = chunk_start + chunk_frames
    if end_after < len(triples):
        p = triples[end_after][1]
        y = triples[end_after][2]
    else:
        last = chunk_start + chunk_frames - 1
        p = triples[last][1]
        y = triples[last][2]
    return p, y


def merge_consecutive_actions(
    action_sequence: List[Tuple[str, List[float], float]],
    step_value: float = 0.02,
    max_action_value: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    连续相同动作合并；每条 ``value = step_value * frames``，且单条 ``value`` 不超过 ``max_action_value``。
    超长连续段按**均匀切段**（见 ``_chunk_lengths_balanced``），使各段幅度更接近、减少过小的尾段。
    """
    if not action_sequence:
        return []
    max_frames = max(1, int(math.floor(max_action_value / step_value + 1e-9)))

    merged: List[Dict[str, Any]] = []
    run_start = 0

    def emit_run(start_idx: int, end_inclusive: int) -> None:
        """[start_idx, end_inclusive] 为连续同一动作，再按 max_action_value 切段。"""
        act = action_sequence[start_idx][0]
        run_len = end_inclusive - start_idx + 1
        idx = start_idx
        for L in _chunk_lengths_balanced(run_len, max_frames):
            start_pos = action_sequence[idx][1]
            start_yaw = action_sequence[idx][2]
            end_p, end_y = _end_pos_after_frames(action_sequence, idx, L)
            merged.append(
                {
                    "action": act,
                    "value": round(step_value * L, 4),
                    "frames": L,
                    "start_pos": start_pos,
                    "start_yaw": start_yaw,
                    "end_pos": end_p,
                    "end_yaw": end_y,
                }
            )
            idx += L

    for i in range(1, len(action_sequence)):
        if action_sequence[i][0] != action_sequence[run_start][0]:
            emit_run(run_start, i - 1)
            run_start = i

    emit_run(run_start, len(action_sequence) - 1)
    return merged


def _demo_symlink_key(demo_dir: Path) -> str:
    return hashlib.sha256(str(demo_dir.resolve()).encode("utf-8")).hexdigest()[:16]


def _ensure_symlink(src: Path, dst: Path) -> Path:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        try:
            if dst.resolve() == src:
                return dst
        except OSError:
            pass
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    os.symlink(str(src), str(dst), target_is_directory=False)
    return dst


def resolve_camera_roots(demo_dir: Path, demo_name: str) -> Optional[Tuple[Path, Path, Path]]:
    """返回 left/center/right 图像目录（存在则选用）。"""
    sub = [
        (demo_dir / f"{demo_name}_l", demo_dir / f"{demo_name}_m", demo_dir / f"{demo_name}_r"),
        (demo_dir / "l", demo_dir / "m", demo_dir / "r"),
    ]
    for ldir, mdir, rdir in sub:
        if ldir.is_dir() and mdir.is_dir() and rdir.is_dir():
            return ldir, mdir, rdir
    return None


def load_instruction(task_path: Path) -> str:
    with open(task_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    ins = task_data.get("Task instruction", task_data.get("instruction", ""))
    if isinstance(ins, list):
        ins = ins[0] if ins else ""
    return str(ins).strip()


def iter_demo_dirs(roots: Sequence[Path]) -> Iterator[Path]:
    """递归查找含 task.json 且含输出轨迹的 demo 目录。"""
    seen: set[str] = set()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for task_path in root.rglob("task.json"):
            demo_dir = task_path.parent
            key = str(demo_dir.resolve())
            if key in seen:
                continue
            if any((demo_dir / n).is_file() for n in ("new_output.json", "output.json")):
                seen.add(key)
                yield demo_dir


def process_demo(
    demo_dir: Path,
    step_value: float = 0.02,
    max_action_value: float = 0.2,
    symlink_under: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    demo_name = demo_dir.name
    cams = resolve_camera_roots(demo_dir, demo_name)
    if cams is None:
        return []

    raw = load_output_json(demo_dir)
    frame_keys = sorted(raw.keys(), key=lambda x: int(re.findall(r"\d+", x)[-1]))
    triples: List[Tuple[str, List[float], float]] = []
    for frame_key in frame_keys:
        t = normalize_frame_to_triple(raw[frame_key])
        if t is None:
            continue
        triples.append(t)

    if not triples:
        return []

    merged_actions = merge_consecutive_actions(
        triples, step_value=step_value, max_action_value=max_action_value
    )
    task_path = demo_dir / "task.json"
    if not task_path.is_file():
        return []
    instruction = load_instruction(task_path)

    ldir, mdir, rdir = cams
    samples: List[Dict[str, Any]] = []
    current_frame = 1
    link_key = _demo_symlink_key(demo_dir) if symlink_under is not None else None

    for action_data in merged_actions:
        act = action_data["action"]
        if act not in ALLOWED_ACTIONS:
            current_frame += action_data["frames"]
            continue

        frame_idx = f"frame_{str(current_frame).zfill(4)}"
        left_img = ldir / f"{frame_idx}_l.png"
        center_img = mdir / f"{frame_idx}_m.png"
        right_img = rdir / f"{frame_idx}_r.png"
        if not (left_img.is_file() and center_img.is_file() and right_img.is_file()):
            current_frame += action_data["frames"]
            continue

        if symlink_under is not None and link_key is not None:
            sub = symlink_under / link_key
            left_p = _ensure_symlink(left_img, sub / f"{frame_idx}_l.png")
            center_p = _ensure_symlink(center_img, sub / f"{frame_idx}_m.png")
            right_p = _ensure_symlink(right_img, sub / f"{frame_idx}_r.png")
            left_s, center_s, right_s = str(left_p), str(center_p), str(right_p)
        else:
            left_s = str(left_img.resolve())
            center_s = str(center_img.resolve())
            right_s = str(right_img.resolve())

        sample = {
            "left": left_s,
            "center": center_s,
            "right": right_s,
            "instruction": instruction,
            "action": act,
            "value": action_data["value"],
            "start_pos": action_data["start_pos"],
            "end_pos": action_data["end_pos"],
            "start_yaw": action_data["start_yaw"],
            "end_yaw": action_data["end_yaw"],
        }
        samples.append(sample)
        current_frame += action_data["frames"]

    return samples


def build_nav_dataset(
    data_roots: Sequence[str],
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
    step_value: float = 0.02,
    max_action_value: float = 0.2,
    compact_json: bool = False,
    json_indent: int = 4,
    symlink_images: bool = False,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = [Path(p).expanduser().resolve() for p in data_roots]
    symlink_root = (out / "image_symlinks") if symlink_images else None
    if symlink_root is not None:
        symlink_root.mkdir(parents=True, exist_ok=True)

    demos = list(iter_demo_dirs(roots))
    rng = random.Random(seed)
    rng.shuffle(demos)
    split_idx = int(len(demos) * train_ratio)
    train_dirs = demos[:split_idx]
    val_dirs = demos[split_idx:]

    train_blocks: List[List[Dict[str, Any]]] = []
    val_blocks: List[List[Dict[str, Any]]] = []

    def run_one(demo: Path) -> List[Dict[str, Any]]:
        return process_demo(
            demo,
            step_value=step_value,
            max_action_value=max_action_value,
            symlink_under=symlink_root,
        )

    for d in tqdm(train_dirs, desc="train demos"):
        s = run_one(Path(d))
        if s:
            train_blocks.append(s)
    for d in tqdm(val_dirs, desc="val demos"):
        s = run_one(Path(d))
        if s:
            val_blocks.append(s)

    json_kw: Dict[str, Any] = {"ensure_ascii": False}
    if compact_json:
        json_kw["separators"] = (",", ":")
        json_kw["indent"] = None
    else:
        json_kw["indent"] = json_indent

    with open(out / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_blocks, f, **json_kw)
    with open(out / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_blocks, f, **json_kw)

    n_train = sum(len(b) for b in train_blocks)
    n_val = sum(len(b) for b in val_blocks)
    meta = {
        "data_roots": [str(p) for p in roots],
        "train_demos": len(train_dirs),
        "val_demos": len(val_dirs),
        "train_samples": n_train,
        "val_samples": n_val,
        "train_ratio": train_ratio,
        "seed": seed,
        "step_value": step_value,
        "max_action_value": max_action_value,
        "compact_json": compact_json,
        "json_indent": None if compact_json else json_indent,
        "symlink_images": symlink_images,
        "image_symlinks_dir": str(symlink_root.resolve()) if symlink_root else None,
    }
    with open(out / "dataset_meta.json", "w", encoding="utf-8") as f:
        if compact_json:
            json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(meta, f, ensure_ascii=False, indent=json_indent)

    vn_cfg = default_scale_config(float(max_action_value))
    vn_cfg["source"] = "build_nav_dataset"
    save_value_norm(str(out), vn_cfg)

    print(
        f"完成: demos 总数={len(demos)}, train={len(train_dirs)} val={len(val_dirs)}, "
        f"样本 train={n_train} val={n_val} -> {out}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="合并 extra_data / gen_scene 等导航原始目录，生成 train.json / val.json"
    )
    p.add_argument(
        "--data_roots",
        nargs="+",
        default=[
            "/data2/lfwj/extra_data",
            "/data2/lfwj/extra_data2",
            "/data2/lfwj/gen_scene_data",
        ],
        help="一个或多个数据根目录",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录（写入 train.json / val.json / dataset_meta.json）",
    )
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--step_value",
        type=float,
        default=0.02,
        help="单帧微观动作量（米/弧度），用于合成 value",
    )
    p.add_argument(
        "--max_action_value",
        type=float,
        default=0.2,
        help="单条训练样本 action value 上限；超长连续同动作会拆成多条",
    )
    p.add_argument(
        "--compact-json",
        action="store_true",
        help="train/val.json 与 dataset_meta.json compact 单行输出（默认 indent=4）",
    )
    p.add_argument(
        "--symlink_images",
        action="store_true",
        help="在 output_dir/image_symlinks 下为每帧三视角建立指向原图的软链接，JSON 中路径指向链接",
    )
    args = p.parse_args()
    build_nav_dataset(
        args.data_roots,
        args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        step_value=args.step_value,
        max_action_value=args.max_action_value,
        compact_json=args.compact_json,
        symlink_images=args.symlink_images,
    )


if __name__ == "__main__":
    main()
