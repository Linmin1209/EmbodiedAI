#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立进程跑 NPC GroundedSAM（不 import isaacsim）。stdin/stdout 每行 JSON。"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace


def _configure_paths(npc_root: str) -> None:
    root = Path(npc_root).resolve()
    hr = root / "home-robot" / "src"
    for p in (str(hr), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(root)


def _serve() -> None:
    perception = None
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"json: {e}"}), flush=True)
            continue
        cmd = msg.get("cmd")
        try:
            if cmd == "init":
                _configure_paths(msg["npc_root"])
                if msg.get("force_cpu"):
                    os.environ["GROUNDED_SAM_DEVICE"] = "cpu"
                from grounded_sam_perception import GroundedSAMPerception

                perception = GroundedSAMPerception(
                    custom_vocabulary=msg["vocabulary"],
                    sem_gpu_id=msg.get("sem_gpu_id", 0),
                    similar_threshold=float(msg.get("similar_threshold", 0.8)),
                )
                print(json.dumps({"ok": True}), flush=True)
            elif cmd == "predict":
                if perception is None:
                    raise RuntimeError("predict before init")
                import numpy as np

                data = np.load(msg["in"], allow_pickle=True)
                o = SimpleNamespace(
                    rgb=np.asarray(data["rgb"]),
                    depth=np.asarray(data["depth"]),
                    gps=np.asarray(data["gps"]),
                    compass=np.asarray(data["compass"]),
                    camera_pose=None,
                    task_observations={},
                )
                out_obs = perception.predict(o)
                to = out_obs.task_observations or {}
                payload = {
                    "semantic": out_obs.semantic,
                    "instance": out_obs.instance,
                    "task_observations": {
                        "instance_map": to.get("instance_map"),
                        "instance_classes": to.get("instance_classes"),
                        "instance_scores": to.get("instance_scores"),
                        "semantic_frame": to.get("semantic_frame"),
                    },
                }
                with open(msg["out"], "wb") as f:
                    pickle.dump(payload, f, protocol=4)
                print(json.dumps({"ok": True, "out": msg["out"]}), flush=True)
            elif cmd == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                break
            else:
                print(json.dumps({"ok": False, "error": f"unknown cmd {cmd}"}), flush=True)
        except Exception as e:
            print(
                json.dumps({"ok": False, "error": str(e), "type": type(e).__name__}),
                flush=True,
            )


if __name__ == "__main__":
    _serve()
