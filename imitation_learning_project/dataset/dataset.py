import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Sampler

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer

from configs.model_config import DEFAULT_TEXT_ENCODER_NAME
from hf_compat import hf_from_pretrained_kwargs

try:
    from dataset.value_normalization import (
        VALUE_NORM_FILENAME,
        ensure_value_norm_file,
        load_value_norm,
        normalize_value,
    )
except ImportError:
    from value_normalization import (
        VALUE_NORM_FILENAME,
        ensure_value_norm_file,
        load_value_norm,
        normalize_value,
    )

# ``build_nav_dataset.py`` 默认输出目录（含 train.json / val.json / dataset_meta.json）
DEFAULT_NAV_DATA_ROOT = "/data2/linmin/EmbodiedAI/train_data"


def _load_samples_and_demo_ranges(raw: Any) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """
    扁平化样本并记录每条 demo 在扁平列表中的 ``[start, end)``，用于 obs_window 内不跨 demo 取历史帧。
    若顶层为扁平 dict 列表，则视为**单条连续轨迹**（整表一个 range）。
    """
    if not raw:
        return [], []
    if isinstance(raw, dict):
        raise ValueError("顶层 JSON 需为 list，不能为 dict")
    first = raw[0]
    if isinstance(first, dict):
        flat = list(raw)
        return flat, [(0, len(flat))]
    out: List[dict] = []
    ranges: List[Tuple[int, int]] = []
    for demo in raw:
        if isinstance(demo, list):
            lo = len(out)
            out.extend(demo)
            ranges.append((lo, len(out)))
        elif isinstance(demo, dict):
            lo = len(out)
            out.append(demo)
            ranges.append((lo, len(out)))
        else:
            raise ValueError(f"不支持的 demo 类型: {type(demo)}")
    return out, ranges


def _flatten_nav_samples(raw: Any) -> List[dict]:
    """
    支持两类 JSON 顶层结构：
    - ``build_nav_dataset``：``[[sample, ...], ...]`` 按 demo 分段；
    - 扁平列表：``[sample, ...]``。
    """
    if not raw:
        return []
    if isinstance(raw, dict):
        raise ValueError("顶层 JSON 需为 list，不能为 dict")
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


def _load_dataset_meta(data_root: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(data_root, "dataset_meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ImitationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        json_file: str,
        tokenizer_name: str = DEFAULT_TEXT_ENCODER_NAME,
        max_text_length: int = 128,
        image_size: int = 224,
        transform=None,
        load_dataset_meta: bool = True,
        value_normalize: bool = True,
        obs_window: int = 2,
    ):
        """
        :param data_root: 含 ``train.json`` / ``val.json`` 的目录；处理后的数据常为
            ``DEFAULT_NAV_DATA_ROOT``（与 ``image_symlinks/`` 同级）。
        :param json_file: 相对于 ``data_root`` 的列表文件，如 ``train.json``。
        :param load_dataset_meta: 若为 True 且存在 ``dataset_meta.json``，则读入 ``self.dataset_meta``。
        :param value_normalize: 为 True 时按 ``value_norm.json`` 归一化 ``value``；测试反归一化见
            ``dataset.value_normalization.denormalize_value``。
        :param obs_window: 视觉历史长度 T；输出 ``left_img`` 等为 ``(T, C, H, W)``，时间从旧到新，
            同一 demo 内取 ``idx-(T-1)..idx``，段首不足则重复该 demo 首帧。JSON 须为按 demo 嵌套的 list
            或单轨迹扁平 list（见 ``_load_samples_and_demo_ranges``）。
        """
        self.data_root = os.path.abspath(data_root)
        self.value_normalize = value_normalize
        if value_normalize:
            vn_path = os.path.join(self.data_root, VALUE_NORM_FILENAME)
            if os.path.isfile(vn_path):
                self.value_norm_cfg = load_value_norm(self.data_root)
            else:
                self.value_norm_cfg = ensure_value_norm_file(self.data_root)
        else:
            self.value_norm_cfg = {"mode": "none", "version": 1}
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            **hf_from_pretrained_kwargs(),
        )
        self.max_text_length = max_text_length
        self.dataset_meta: Optional[Dict[str, Any]] = None
        if load_dataset_meta:
            self.dataset_meta = _load_dataset_meta(self.data_root)

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        json_path = os.path.join(self.data_root, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples, self._demo_ranges = _load_samples_and_demo_ranges(raw)
        self.obs_window = max(1, int(obs_window))

        # 离散与训练 loss 一致；仿真 BaseEnv.step 的 PositionController 对 command[0] 另有约定
        #（0 前 1 后 2 左 3 右）。部署时勿直接当作 sim id，见 ``simulator/utils/bc_action_sim_mapping.py``。
        self.action_map = {
            "move forward": 0,
            "turn left": 1,
            "turn right": 2,
        }

    def _resolve_image_path(self, p: str) -> str:
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(self.data_root, p)

    def preprocess_instruction(self, instruction):
        if instruction is None:
            return ""
        if isinstance(instruction, list):
            instruction = instruction[0] if instruction else ""
        instruction = str(instruction)
        instruction = re.sub(r"-\d+", "", instruction)
        instruction = re.sub(r"_\d+", "", instruction)
        instruction = " ".join(instruction.split())
        return instruction

    def __len__(self):
        return len(self.samples)

    def _demo_range_for_index(self, idx: int) -> Tuple[int, int]:
        for a, b in self._demo_ranges:
            if a <= idx < b:
                return a, b
        return 0, len(self.samples)

    def _window_indices(self, idx: int) -> List[int]:
        """时间顺序：最旧 → 当前（最后一帧为 idx）。"""
        w = self.obs_window
        a, _b = self._demo_range_for_index(idx)
        out: List[int] = []
        for k in range(w):
            j = idx - (w - 1) + k
            if j < a:
                j = a
            out.append(j)
        return out

    def _load_one_frame_tensors(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lp = self._resolve_image_path(sample["left"])
        cp = self._resolve_image_path(sample["center"])
        rp = self._resolve_image_path(sample["right"])
        left_img = Image.open(lp).convert("RGB")
        center_img = Image.open(cp).convert("RGB")
        right_img = Image.open(rp).convert("RGB")
        if self.transform:
            left_img = self.transform(left_img)
            center_img = self.transform(center_img)
            right_img = self.transform(right_img)
        return left_img, center_img, right_img

    def __getitem__(self, idx):
        sample = self.samples[idx]

        left_stack: List[torch.Tensor] = []
        center_stack: List[torch.Tensor] = []
        right_stack: List[torch.Tensor] = []
        for j in self._window_indices(idx):
            lj, cj, rj = self._load_one_frame_tensors(self.samples[j])
            left_stack.append(lj)
            center_stack.append(cj)
            right_stack.append(rj)
        left_img = torch.stack(left_stack, dim=0)
        center_img = torch.stack(center_stack, dim=0)
        right_img = torch.stack(right_stack, dim=0)

        start_pos = torch.tensor(sample["start_pos"], dtype=torch.float32)
        end_pos = torch.tensor(sample["end_pos"], dtype=torch.float32)
        start_yaw = torch.tensor(sample["start_yaw"], dtype=torch.float32)
        end_yaw = torch.tensor(sample["end_yaw"], dtype=torch.float32)

        instruction = self.preprocess_instruction(sample.get("instruction", ""))

        encoded = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        action_name = str(sample["action"]).lower().strip()
        if action_name not in self.action_map:
            raise KeyError(f"未知动作: {action_name} @ {idx}")
        action = torch.tensor(self.action_map[action_name], dtype=torch.long)
        raw_v = float(sample["value"])
        if self.value_normalize:
            nv = normalize_value(raw_v, self.value_norm_cfg)
            value = torch.tensor(nv, dtype=torch.float32)
        else:
            value = torch.tensor(raw_v, dtype=torch.float32)

        return {
            "left_img": left_img,
            "center_img": center_img,
            "right_img": right_img,
            "instruction": instruction,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "action": action,
            "value": value,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "start_yaw": start_yaw,
            "end_yaw": end_yaw,
        }


def get_dataloader(
    data_root: Optional[str] = None,
    json_file: str = "train.json",
    batch_size=32,
    num_workers=4,
    shuffle=True,
    tokenizer_name=DEFAULT_TEXT_ENCODER_NAME,
    max_text_length=128,
    image_size=224,
    pin_memory=True,
    drop_last: bool = False,
    load_dataset_meta: bool = True,
    sampler: Optional[Sampler] = None,
    dataset: Optional[ImitationDataset] = None,
    value_normalize: bool = True,
    obs_window: int = 2,
):
    if dataset is None:
        root = data_root if data_root is not None else DEFAULT_NAV_DATA_ROOT
        dataset = ImitationDataset(
            root,
            json_file,
            tokenizer_name=tokenizer_name,
            max_text_length=max_text_length,
            image_size=image_size,
            load_dataset_meta=load_dataset_meta,
            value_normalize=value_normalize,
            obs_window=obs_window,
        )
    if sampler is not None and shuffle:
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
