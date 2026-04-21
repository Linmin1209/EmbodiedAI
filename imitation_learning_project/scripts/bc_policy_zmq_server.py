#!/usr/bin/env python3
"""
在 **独立进程**（无 Isaac / 无 Kit）中加载 ``BCTransformer``，通过 ZMQ REP 接收观测、返回动作。

避免 Isaac 进程内 ``transformers`` → ``sklearn`` → ``scipy`` 与 Kit ``pip_prebundle`` 冲突。

用法::

    pip install pyzmq  # 若尚未安装
    python bc_policy_zmq_server.py --checkpoint /path/to/best.pt --bind tcp://0.0.0.0:5556

仿真侧（仅 Isaac，不加载 torch/transformers）::

    export BC_POLICY_ZMQ=tcp://127.0.0.1:5556
    python tests/test_scene/test_bc_transformer.py 0

同网段机器可将 ``--bind`` 设为 ``tcp://0.0.0.0:5556``，客户端设置 ``BC_POLICY_ZMQ=tcp://<服务器IP>:5556``。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# 勿在模块顶层 import torch/transformers：会阻塞数秒～数分钟且 bind 之前无任何输出。
print("[bc_policy_zmq] process started (heavy deps load in worker thread only)", flush=True)

# 脚本位于 imitation_learning_project/scripts/
_IL_ROOT = Path(__file__).resolve().parents[1]
if str(_IL_ROOT) not in sys.path:
    sys.path.insert(0, str(_IL_ROOT))

from configs.model_config import ModelConfig  # noqa: E402


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


class PolicySession:
    """单连接顺序请求：维护 obs_window 与 tokenizer 状态。"""

    def __init__(
        self,
        model: Any,
        model_config: ModelConfig,
        device: Any,
        tokenizer: Any,
        transform: Any,
        value_norm_cfg: Optional[Dict[str, Any]],
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.device = device
        self.tokenizer = tokenizer
        self.transform = transform
        self._value_norm_cfg = value_norm_cfg
        self.obs_window = int(
            getattr(model_config, "obs_window", getattr(model_config, "history_len", 1))
        )
        self._left_seq: List[Any] = []
        self._center_seq: List[Any] = []
        self._right_seq: List[Any] = []

    def reset(self) -> None:
        self._left_seq.clear()
        self._center_seq.clear()
        self._right_seq.clear()

    def _append_and_stack(
        self, left: Any, center: Any, right: Any
    ) -> tuple[Any, Any, Any]:
        import torch

        self._left_seq.append(left.detach().clone())
        self._center_seq.append(center.detach().clone())
        self._right_seq.append(right.detach().clone())
        if len(self._left_seq) > self.obs_window:
            self._left_seq = self._left_seq[-self.obs_window :]
            self._center_seq = self._center_seq[-self.obs_window :]
            self._right_seq = self._right_seq[-self.obs_window :]

        def _window_tensor(hist: List[Any]) -> Any:
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

    def _postprocess_value(self, value_pred: Any) -> float:
        import os
        import torch
        from dataset.value_normalization import denormalize_value, quantize_value_physical

        v = value_pred.squeeze(-1).squeeze(0)
        if self._value_norm_cfg is not None and getattr(
            self.model_config, "value_normalize", True
        ):
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

    def infer(
        self,
        instruction: str,
        left_u8: Any,
        center_u8: Any,
        right_u8: Any,
        reset_history: bool,
    ) -> Dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        if reset_history:
            self.reset()

        rgb_m = Image.fromarray(np.ascontiguousarray(center_u8))
        rgb_l = Image.fromarray(np.ascontiguousarray(left_u8))
        rgb_r = Image.fromarray(np.ascontiguousarray(right_u8))
        left = self.transform(rgb_l)
        center = self.transform(rgb_m)
        right = self.transform(rgb_r)

        encoded = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=self.model_config.max_text_length,
            return_tensors="pt",
        )

        left_b, center_b, right_b = self._append_and_stack(left, center, right)
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
            value = self._postprocess_value(outputs["value_pred"])

        return {"action_id": act_id, "value": float(value), "error": None}


def _load_session_worker(
    ckpt: str,
    model_config: ModelConfig,
    data_root: Optional[str],
    session_holder: List[Optional[PolicySession]],
    load_error: List[Optional[str]],
    model_ready: threading.Event,
) -> None:
    """在后台线程中加载 checkpoint / BERT / PolicySession，主线程可先 bind ZMQ 并响应 ping。"""
    try:
        print("[bc_policy_zmq] worker: importing torch / transformers / BCTransformer ...", flush=True)
        import torch
        import torchvision.transforms as transforms
        from transformers.models.bert.tokenization_bert import BertTokenizer

        from dataset.value_normalization import load_value_norm
        from models.bc_transformer import BCTransformer

        print("[bc_policy_zmq] loading checkpoint & tokenizer (may take minutes)...", flush=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BCTransformer(model_config).to(device)
        checkpoint = torch.load(ckpt, map_location=device)
        sd = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(sd, dict) or not sd:
            raise ValueError("checkpoint 需含 model_state_dict 或可为纯 state_dict")
        model.load_state_dict(_strip_module_prefix(sd), strict=False)
        model.eval()

        tok_name = model_config.text_encoder_name
        tokenizer = BertTokenizer.from_pretrained(tok_name)

        transform = transforms.Compose(
            [
                transforms.Resize((model_config.image_size, model_config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        value_norm_cfg: Optional[Dict[str, Any]] = None
        if data_root and getattr(model_config, "value_normalize", True):
            dr = os.path.abspath(data_root)
            if os.path.isdir(dr):
                vn = os.path.join(dr, "value_norm.json")
                if os.path.isfile(vn):
                    value_norm_cfg = load_value_norm(dr)

        session_holder[0] = PolicySession(
            model, model_config, device, tokenizer, transform, value_norm_cfg
        )
        print("[bc_policy_zmq] model ready.", flush=True)
    except BaseException as e:
        load_error[0] = f"{type(e).__name__}: {e}"
        print(f"[bc_policy_zmq] load failed:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    finally:
        model_ready.set()


def main() -> None:
    try:
        import zmq
    except ImportError as e:
        print("需要安装 pyzmq: pip install pyzmq", file=sys.stderr)
        raise SystemExit(1) from e

    ap = argparse.ArgumentParser(description="BCTransformer ZMQ policy server (no Isaac)")
    ap.add_argument("--checkpoint", required=True, help="best.pt / epoch_*.pt")
    ap.add_argument(
        "--bind",
        default="tcp://0.0.0.0:5556",
        help="ZMQ REP bind address",
    )
    ap.add_argument(
        "--model-config",
        default=None,
        help="config_resolved.yaml；默认 checkpoint 同目录",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="训练数据根目录（含 value_norm.json）；也可设环境变量 BC_DATA_ROOT",
    )
    args = ap.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        print(f"Checkpoint 不存在: {ckpt}", file=sys.stderr)
        raise SystemExit(1)

    yaml_path = args.model_config
    if yaml_path is None:
        cand = os.path.join(os.path.dirname(ckpt), "config_resolved.yaml")
        if os.path.isfile(cand):
            yaml_path = cand

    if yaml_path and os.path.isfile(yaml_path):
        model_config = _load_model_config_from_yaml(yaml_path)
    else:
        model_config = ModelConfig()

    data_root = args.data_root or os.environ.get("BC_DATA_ROOT")

    session_holder: List[Optional[PolicySession]] = [None]
    load_error: List[Optional[str]] = [None]
    model_ready = threading.Event()

    # 先 bind，再启动加载线程：ping 可立刻成功，不等待 torch/transformers import
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(
        f"BCTransformer ZMQ server listening on {args.bind} (model loading in background)",
        flush=True,
    )

    loader = threading.Thread(
        target=_load_session_worker,
        args=(ckpt, model_config, data_root, session_holder, load_error, model_ready),
        daemon=True,
        name="bc_policy_load",
    )
    loader.start()

    obs_window_cfg = int(
        getattr(model_config, "obs_window", getattr(model_config, "history_len", 1))
    )

    while True:
        try:
            req = sock.recv_pyobj()
        except Exception as e:
            print(f"recv_pyobj failed: {e}", file=sys.stderr, flush=True)
            continue

        if not isinstance(req, dict):
            sock.send_pyobj({"error": "request must be a dict", "action_id": 0, "value": 0.0})
            continue

        if req.get("type") == "ping":
            err = load_error[0]
            loaded = (
                model_ready.is_set()
                and err is None
                and session_holder[0] is not None
            )
            sock.send_pyobj(
                {
                    "ok": True,
                    "obs_window": obs_window_cfg,
                    "model_loaded": loaded,
                    "load_error": err,
                }
            )
            continue

        if req.get("type") != "infer":
            sock.send_pyobj({"error": "unknown type", "action_id": 0, "value": 0.0})
            continue

        if not model_ready.is_set():
            if not model_ready.wait(timeout=7200.0):
                sock.send_pyobj(
                    {
                        "error": "timeout waiting for model load (7200s)",
                        "action_id": 0,
                        "value": 0.0,
                    }
                )
                continue
        if load_error[0] is not None or session_holder[0] is None:
            err = load_error[0] or "model not loaded"
            sock.send_pyobj({"error": err, "action_id": 0, "value": 0.0})
            continue

        session = session_holder[0]

        try:
            import numpy as np

            instruction = str(req.get("instruction", "") or "")
            reset_history = bool(req.get("reset_history", False))
            left_u8 = req["left"]
            center_u8 = req["center"]
            right_u8 = req["right"]
            if not all(isinstance(x, np.ndarray) for x in (left_u8, center_u8, right_u8)):
                raise ValueError("left/center/right must be numpy arrays")
            out = session.infer(
                instruction, left_u8, center_u8, right_u8, reset_history
            )
            sock.send_pyobj(out)
        except Exception as e:
            sock.send_pyobj(
                {"error": str(e), "action_id": 0, "value": 0.0}
            )


if __name__ == "__main__":
    main()
