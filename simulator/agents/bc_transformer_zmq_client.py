"""
与 ``bc_policy_zmq_server.py`` 配对：本模块 **不** 依赖 torch / transformers，仅 numpy + pyzmq。

在 Isaac 进程中设置 ``BC_POLICY_ZMQ=tcp://主机:端口`` 后，由 ``test_bc_transformer`` 选用本 Agent，
策略推理在独立 Python 进程中完成，避免 Kit 内 scipy 与 conda sklearn 冲突。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import numpy as np
from gym import spaces

from simulator.core.agent import BaseAgent

try:
    import zmq
except ImportError as e:  # pragma: no cover
    zmq = None  # type: ignore
    _ZMQ_IMPORT_ERROR = e
else:
    _ZMQ_IMPORT_ERROR = None


class BCTransformerZmqAgent(BaseAgent):
    def __init__(self, config) -> None:
        super().__init__(config)
        if zmq is None:
            raise ImportError(
                "需要安装 pyzmq: pip install pyzmq"
            ) from _ZMQ_IMPORT_ERROR

        self._endpoint = getattr(config, "zmq_endpoint", None) or os.environ.get(
            "BC_POLICY_ZMQ"
        )
        if not self._endpoint:
            raise ValueError("请设置 config.zmq_endpoint 或环境变量 BC_POLICY_ZMQ")

        self._timeout_ms = int(
            getattr(config, "zmq_timeout_ms", None)
            or os.environ.get("BC_POLICY_ZMQ_TIMEOUT_MS", "60000")
        )
        na = int(
            getattr(config, "num_actions", None)
            or os.environ.get("BC_NUM_ACTIONS", "3")
        )
        self.action_space = spaces.Tuple(
            (spaces.Discrete(na), spaces.Box(low=0.0, high=1.0, shape=(1,)))
        )

        self._pending_reset = False
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.connect(self._endpoint)

        self._ping()

    def _ping(self) -> None:
        self._sock.send_pyobj({"type": "ping"})
        rep = self._sock.recv_pyobj()
        if not isinstance(rep, dict) or not rep.get("ok"):
            raise RuntimeError(f"policy server ping failed: {rep!r}")

    @staticmethod
    def preprocess_instruction(instruction: Any) -> str:
        if instruction is None:
            return ""
        if isinstance(instruction, list):
            instruction = instruction[0] if instruction else ""
        instruction = str(instruction)
        instruction = re.sub(r"-\d+", "", instruction)
        instruction = re.sub(r"_\d+", "", instruction)
        instruction = " ".join(instruction.split())
        return instruction

    def act(self, observation: Dict[str, Any]) -> List[Any]:
        left = np.asarray(observation["robot0_left_camera"]["rgb"], dtype=np.uint8)
        center = np.asarray(observation["robot0_front_camera"]["rgb"], dtype=np.uint8)
        right = np.asarray(observation["robot0_right_camera"]["rgb"], dtype=np.uint8)
        instruction = self.preprocess_instruction(observation.get("instruction", ""))

        reset_history = self._pending_reset
        self._pending_reset = False

        req = {
            "type": "infer",
            "instruction": instruction,
            "reset_history": reset_history,
            "left": left,
            "center": center,
            "right": right,
        }
        self._sock.send_pyobj(req)
        rep = self._sock.recv_pyobj()
        if not isinstance(rep, dict):
            raise RuntimeError(f"bad reply from policy server: {rep!r}")
        err = rep.get("error")
        if err:
            raise RuntimeError(f"policy server error: {err}")
        return [int(rep["action_id"]), float(rep["value"])]

    def reset(self) -> None:
        self._pending_reset = True
