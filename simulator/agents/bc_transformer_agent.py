"""
使用 ``imitation_learning_project`` 中训练的 ``BCTransformer`` 做仿真推理。

依赖：仓库根下存在 ``imitation_learning_project``；检查点由 ``train.py`` 保存（含 ``model_state_dict``）。

配置（传入 ``BaseAgent`` 的 ``config``）示例::

    config.checkpoint_path = "/data2/linmin/.../best.pt"
    # 可选：与 checkpoint 同目录的 ``config_resolved.yaml``，用于 obs_window / step_bins 等与训练一致
    config.model_config_path = None  # 默认尝试 ``dirname(checkpoint)/config_resolved.yaml``
    # 可选：训练数据目录，用于读取 ``value_norm.json`` 将归一化 value 还原为物理量
    config.data_root = "/path/to/train_data"

若在 **DDP** 下保存，权重键可能带 ``module.`` 前缀，加载时会自动剥除。
"""
from __future__ import annotations

import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional


def _prefer_conda_site_over_isaac_prebundle() -> None:
    """
    Isaac Kit 的 ``pip_prebundle`` 里带有与 conda 不兼容的旧版 scipy；
    ``transformers`` 会间接 import sklearn，sklearn 再 import scipy 时若误用 bundle 内 scipy，
    会出现 ``_ufuncs_cxx does not export _export_expit`` 乃至 **Segmentation fault**。

    在 import transformers 之前：优先插入 ``CONDA_PREFIX`` 的 site-packages，
    并移除路径中含 ``pip_prebundle`` 的 Kit 扩展目录。
    """
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_candidates: list[str] = []
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        site_candidates.append(os.path.join(conda, "lib", py_ver, "site-packages"))
    # 当前解释器前缀（非 conda 或未设 CONDA_PREFIX 时仍可用）
    site_candidates.append(os.path.join(sys.prefix, "lib", py_ver, "site-packages"))
    seen: set[str] = set()
    for sp in site_candidates:
        if sp in seen or not os.path.isdir(sp):
            continue
        seen.add(sp)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    kept: list[str] = []
    for p in sys.path:
        pl = p.replace("\\", "/").lower()
        # 任一带 pip_prebundle 的路径都可能提供错误 scipy；Kit 下 omni.data 同理
        if "pip_prebundle" in pl:
            continue
        if "omni/data/kit" in pl or "isaac-sim python" in pl:
            continue
        kept.append(p)
    sys.path[:] = kept


def _purge_kit_scipy_sklearn_from_sys_modules() -> None:
    """
    SimulationApp 运行后可能已从 ``.../pip_prebundle/scipy`` 装入 ``scipy``；
    仅删路径仍会继续用缓存模块。若检测到来自 Kit 的 scipy，则移除 scipy/sklearn 相关缓存，
    后续 import 将使用 conda site-packages 中的版本。
    """
    bad = False
    sc = sys.modules.get("scipy")
    if sc is not None:
        fp = (getattr(sc, "__file__", "") or "").replace("\\", "/").lower()
        if "pip_prebundle" in fp or "omni/data/kit" in fp:
            bad = True
    if not bad:
        return
    for k in list(sys.modules):
        if k == "scipy" or k.startswith("scipy."):
            del sys.modules[k]
        if k == "sklearn" or k.startswith("sklearn."):
            del sys.modules[k]


_prefer_conda_site_over_isaac_prebundle()
_purge_kit_scipy_sklearn_from_sys_modules()

import torch
import torchvision.transforms as transforms
import yaml
from gym import spaces
from PIL import Image
# 勿用 AutoTokenizer：会走 tokenization_auto → generation → sklearn → scipy，与 Kit 内 scipy 冲突。
from transformers.models.bert.tokenization_bert import BertTokenizer

from simulator.core.agent import BaseAgent

_IL_ROOT = Path(__file__).resolve().parents[2] / "imitation_learning_project"
if _IL_ROOT.is_dir():
    _ilp = str(_IL_ROOT)
    if _ilp not in sys.path:
        sys.path.insert(0, _ilp)
else:
    raise ImportError(
        f"未找到 {_IL_ROOT}。请从 EmbodiedAI 仓库根运行或把该路径加入 PYTHONPATH。"
    )

from configs.model_config import ModelConfig  # noqa: E402
from dataset.value_normalization import (  # noqa: E402
    denormalize_value,
    load_value_norm,
    quantize_value_physical,
)
from models.bc_transformer import BCTransformer  # noqa: E402


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _load_model_config_from_yaml(path: str) -> ModelConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    mc = ModelConfig()
    for k, v in raw.items():
        if hasattr(mc, k) and v is not None:
            setattr(mc, k, v)
    return mc


class BCTransformerAgent(BaseAgent):
    """加载训练好的 ``BCTransformer``，支持 ``obs_window`` 视觉历史与 value 反归一化。"""

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(
            getattr(config, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        ckpt_path = getattr(config, "checkpoint_path", None)
        if not ckpt_path:
            raise ValueError("请在 config 中设置 checkpoint_path（如 best.pt）")

        yaml_path = getattr(config, "model_config_path", None)
        if yaml_path is None:
            cand = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config_resolved.yaml")
            if os.path.isfile(cand):
                yaml_path = cand

        if yaml_path and os.path.isfile(yaml_path):
            self.model_config = _load_model_config_from_yaml(yaml_path)
        else:
            self.model_config = ModelConfig()

        self.model = BCTransformer(self.model_config).to(self.device)

        self._load_checkpoint(ckpt_path)

        self.model.eval()

        tok_name = self.model_config.text_encoder_name
        self.tokenizer = BertTokenizer.from_pretrained(tok_name)

        self.obs_window = int(
            getattr(self.model_config, "obs_window", getattr(self.model_config, "history_len", 1))
        )
        self._left_seq: List[torch.Tensor] = []
        self._center_seq: List[torch.Tensor] = []
        self._right_seq: List[torch.Tensor] = []

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.model_config.image_size, self.model_config.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        dr = getattr(config, "data_root", None)
        self._value_norm_cfg: Optional[Dict[str, Any]] = None
        if dr and getattr(self.model_config, "value_normalize", True):
            vn = os.path.join(os.path.abspath(dr), "value_norm.json")
            if os.path.isfile(vn):
                self._value_norm_cfg = load_value_norm(dr)

        na = int(self.model_config.num_actions)
        self.action_space = spaces.Tuple(
            (spaces.Discrete(na), spaces.Box(low=0.0, high=1.0, shape=(1,)))
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        sd = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(sd, dict) or not sd:
            raise KeyError("checkpoint 中需包含 model_state_dict 或可为纯 state_dict")
        sd = _strip_module_prefix(sd)
        try:
            self.model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            import warnings

            ret = self.model.load_state_dict(sd, strict=False)
            mk = getattr(ret, "missing_keys", ()) or ()
            uk = getattr(ret, "unexpected_keys", ()) or ()
            warnings.warn(
                f"strict 加载失败，已退回 strict=False: {e} | "
                f"missing_keys={len(mk)} unexpected_keys={len(uk)}"
            )

    def preprocess_instruction(self, instruction: Any) -> str:
        if instruction is None:
            return ""
        if isinstance(instruction, list):
            instruction = instruction[0] if instruction else ""
        instruction = str(instruction)
        instruction = re.sub(r"-\d+", "", instruction)
        instruction = re.sub(r"_\d+", "", instruction)
        instruction = " ".join(instruction.split())
        return instruction

    def _append_and_stack_obs_window(
        self, left: torch.Tensor, center: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """维护三视角最近帧序列，段首不足时用当前最早帧向左填充（与 Dataset 一致），输出 (1,T,C,H,W)。"""
        self._left_seq.append(left.detach().clone())
        self._center_seq.append(center.detach().clone())
        self._right_seq.append(right.detach().clone())
        if len(self._left_seq) > self.obs_window:
            self._left_seq = self._left_seq[-self.obs_window :]
            self._center_seq = self._center_seq[-self.obs_window :]
            self._right_seq = self._right_seq[-self.obs_window :]

        def _window_tensor(hist: List[torch.Tensor]) -> torch.Tensor:
            if len(hist) < self.obs_window:
                pad_count = self.obs_window - len(hist)
                chunk = [hist[0].clone() for _ in range(pad_count)] + hist
            else:
                chunk = hist[-self.obs_window :]
            return torch.stack(chunk, dim=0)

        left_b = _window_tensor(self._left_seq).unsqueeze(0)
        center_b = _window_tensor(self._center_seq).unsqueeze(0)
        right_b = _window_tensor(self._right_seq).unsqueeze(0)
        return left_b, center_b, right_b

    def _postprocess_value(self, value_pred: torch.Tensor) -> float:
        """模型输出为归一化 value 时（训练一致）反归一化到物理量；可选按 ``value_step_physical`` 量化。"""
        v = value_pred.squeeze(-1).squeeze(0)
        if self._value_norm_cfg is not None and getattr(self.model_config, "value_normalize", True):
            out = float(denormalize_value(v, self._value_norm_cfg))
        else:
            out = float(v.item())
        if os.environ.get("BC_VALUE_QUANTIZE", "1").strip().lower() in ("0", "false", "no"):
            return out
        step = os.environ.get("BC_VALUE_QUANTIZE_STEP")
        step_f = float(step) if step is not None else float(
            getattr(self.model_config, "value_step_physical", 0.02)
        )
        vmax_s = os.environ.get("BC_VALUE_QUANTIZE_MAX")
        vmax = float(vmax_s) if vmax_s is not None else float(
            getattr(self.model_config, "value_max_physical", 0.2)
        )
        if step_f <= 0:
            return out
        return quantize_value_physical(out, step=step_f, vmax=vmax)

    def act(self, observation: Dict[str, Any]) -> List[Any]:
        rgb_m = Image.fromarray(observation["robot0_front_camera"]["rgb"])
        rgb_l = Image.fromarray(observation["robot0_left_camera"]["rgb"])
        rgb_r = Image.fromarray(observation["robot0_right_camera"]["rgb"])
        left = self.transform(rgb_l)
        center = self.transform(rgb_m)
        right = self.transform(rgb_r)

        instruction = self.preprocess_instruction(observation.get("instruction", ""))
        encoded = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=self.model_config.max_text_length,
            return_tensors="pt",
        )

        left_b, center_b, right_b = self._append_and_stack_obs_window(left, center, right)

        batch = {
            "left_img": left_b.to(self.device),
            "center_img": center_b.to(self.device),
            "right_img": right_b.to(self.device),
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            outputs = self.model(batch)
            logits = outputs["action_logits"]
            act_id = int(logits.argmax(dim=-1).item())

            value_raw = outputs["value_pred"]
            value = self._postprocess_value(value_raw)

        return [act_id, value]

    def reset(self) -> None:
        self._left_seq.clear()
        self._center_seq.clear()
        self._right_seq.clear()
