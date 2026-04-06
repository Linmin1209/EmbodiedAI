"""
Isaac 进程外运行 GroundedSAM：独立子进程 + 干净 CUDA 上下文，从根源避免与 omni.isaac.ml_archive 的 PyTorch 冲突。
"""

from __future__ import annotations

import atexit
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np


def npc_vocabulary_from_category_list(categories: list[str]) -> str:
    """与 NPC `mapping.SemanticMap` 中 GroundedSAM 的 `custom_vocabulary` 一致。"""
    cats = ["other", *[str(c).replace("_", " ") for c in categories], "other"]
    return " . ".join(cats)


_WORKER_SCRIPT = Path(__file__).resolve().parent / "workers" / "grounded_sam_isolated_worker.py"


class IsolatedGroundedSAMClient:
    """
    实现与 `GroundedSAMPerception` 相同的 `predict(obs)` 接口；观测为 SimpleNamespace（rgb, depth, gps, compass…）。
    """

    def __init__(
        self,
        *,
        npc_root: str | Path,
        vocabulary: str,
        sem_gpu_id: int = 0,
        similar_threshold: float = 0.8,
        force_cpu: bool = False,
        python_exe: Optional[str] = None,
    ) -> None:
        self._closed = False
        self._python = python_exe or os.environ.get("EMBODIEDAI_SEG_PYTHON") or sys.executable
        self._npc_root = str(Path(npc_root).resolve())
        if not _WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"worker 脚本不存在: {_WORKER_SCRIPT}")

        # 子进程环境：避免 Cursor/VSCode debugpy 继承变量后在子进程 stdout 混入非协议内容
        _child_env = os.environ.copy()
        _child_env["PYTHONUNBUFFERED"] = "1"
        for _k in (
            "PYTHONBREAKPOINT",
            "DEBUGPY_LAUNCHER_PORT",
            "VSCODE_DEBUGPY_ADAPTER_ENDPOINTS",
            "PYDEVD_LOAD_VALUES_ASYNC",
        ):
            _child_env.pop(_k, None)

        self._proc = subprocess.Popen(
            [self._python, "-u", str(_WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._npc_root,
            env=_child_env,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        init_payload = {
            "cmd": "init",
            "npc_root": self._npc_root,
            "vocabulary": vocabulary,
            "sem_gpu_id": int(sem_gpu_id),
            "similar_threshold": float(similar_threshold),
            "force_cpu": bool(force_cpu),
        }
        self._send_json(init_payload)
        line = self._readline_json()
        if not (isinstance(line, dict) and line.get("ok")):
            self.close()
            raise RuntimeError(f"子进程 GroundedSAM init 失败: {line}")

        def _cleanup() -> None:
            if not self._closed:
                self.close()

        atexit.register(_cleanup)

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        for b in iter(self._proc.stderr.readline, b""):
            if b:
                try:
                    sys.stderr.write(f"[seg-worker] {b.decode(errors='replace')}")
                except Exception:
                    pass

    def _send_json(self, obj: dict[str, Any]) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("worker stdin 已关闭")
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _readline_json(self) -> dict[str, Any]:
        if self._proc.stdout is None:
            raise RuntimeError("worker stdout 已关闭")
        _max = 200
        for _ in range(_max):
            raw = self._proc.stdout.readline()
            if not raw:
                code = self._proc.poll()
                raise RuntimeError(
                    f"子进程 stdout 已结束（code={code}）。若用 Cursor/VSCode「调试」运行主脚本，"
                    "debugpy 可能干扰子进程；请改用「运行而不调试」，或设置 EMBODIEDAI_SEG_PYTHON 为独立 python。"
                )
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            if text.startswith("\ufeff"):
                text = text.lstrip("\ufeff")
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                print(
                    f"[seg-worker] (忽略 stdout 非 JSON 行): {text[:240]!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not isinstance(obj, dict):
                # 误匹配到 JSON 数组等（如 [1,2,3]），继续读下一行
                print(
                    f"[seg-worker] (忽略非 dict 的 JSON): {type(obj).__name__} {str(obj)[:200]!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            return obj
        raise RuntimeError(
            f"子进程 {_max} 行内未读到合法 JSON 对象（dict）。"
            "多为调试器/启动器污染 stdout；请不用调试启动，或 export EMBODIEDAI_SEG_PYTHON=/path/to/python。"
        )

    def predict(self, obs: Any) -> Any:
        rgb = np.asarray(obs.rgb)
        depth = np.asarray(obs.depth)
        gps = np.asarray(obs.gps)
        compass = np.asarray(obs.compass)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f_in:
            in_path = f_in.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_out:
            out_path = f_out.name

        try:
            np.savez_compressed(in_path, rgb=rgb, depth=depth, gps=gps, compass=compass)
            self._send_json({"cmd": "predict", "in": in_path, "out": out_path})
            line = self._readline_json()
            if not (isinstance(line, dict) and line.get("ok")):
                raise RuntimeError(f"子进程 predict 失败: {line}")

            with open(out_path, "rb") as f:
                payload = pickle.load(f)

            obs.semantic = payload["semantic"]
            obs.instance = payload["instance"]
            if obs.task_observations is None:
                obs.task_observations = {}
            to = payload.get("task_observations") or {}
            for k, v in to.items():
                if v is not None:
                    obs.task_observations[k] = v
            return obs
        finally:
            for p in (in_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None and self._proc.stdin:
                self._send_json({"cmd": "shutdown"})
                self._proc.wait(timeout=30)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
