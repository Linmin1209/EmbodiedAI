"""
Hugging Face Hub 与 transformers 的网络/离线行为，与 train / Dataset / 模型加载一致。

- **镜像**（无法直连 huggingface.co）：在启动前设置
  ``export HF_ENDPOINT=https://hf-mirror.com``
  （或其它可用镜像），``huggingface_hub`` 会自动使用。
- **离线**（仅从本机缓存或已下载目录加载）：设置
  ``export TRANSFORMERS_OFFLINE=1`` 或 ``export HF_HUB_OFFLINE=1``，
  并确保 ``~/.cache/huggingface`` 中已有对应模型，或将模型下载到本地后把
  ``text_encoder_name`` 指向该目录。
"""
from __future__ import annotations

import os
from typing import Any, Dict


def hf_local_files_only() -> bool:
    """是否强制不发起网络请求（仅本地文件）。"""
    for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        v = os.environ.get(key, "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False


def hf_from_pretrained_kwargs() -> Dict[str, Any]:
    """传给 ``AutoTokenizer.from_pretrained`` / ``AutoModel.from_pretrained`` 的附加参数。"""
    return {"local_files_only": hf_local_files_only()}
