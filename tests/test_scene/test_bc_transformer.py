from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator.core.dataset import DatasetLoader
from simulator.core.env import BaseEnv
from simulator.scenes import Interactive_Scene  # noqa: F401 — 注册 Scene 类型，BaseEnv 需要
from simulator.utils.bc_action_sim_mapping import train_action_tuple_to_sim
from simulator.utils.scene_utils import extract_target_ids


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_bct_agent_class():
    """
    单独加载 ``bc_transformer_agent`` 模块，避免 ``from simulator.agents import ...``
    触发 ``agents/__init__.py`` 里其它 Agent（会先 ``import torch``）。

    Isaac Sim 要求：**SimulationApp 启动后再导入 torch**，否则易告警或 Segmentation fault。
    """
    path = _REPO_ROOT / "simulator" / "agents" / "bc_transformer_agent.py"
    spec = importlib.util.spec_from_file_location(
        "bc_transformer_agent_isolated", path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.BCTransformerAgent


def _load_zmq_agent_class():
    """仅 numpy + zmq，无 torch；与独立进程 ``bc_policy_zmq_server`` 通信。"""
    path = _REPO_ROOT / "simulator" / "agents" / "bc_transformer_zmq_client.py"
    spec = importlib.util.spec_from_file_location(
        "bc_transformer_zmq_client_isolated", path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.BCTransformerZmqAgent

# 与 train 输出一致；可改为环境变量 BC_TRANSFORMER_CHECKPOINT
_DEFAULT_CKPT = (
    "/data2/linmin/EmbodiedAI/imitation_learning_project/checkpoints_ddp/epoch_42.pt"
)
# 可选：训练数据目录，用于 value_norm.json 反归一化
_DEFAULT_DATA_ROOT = "/data2/linmin/EmbodiedAI/train_data"

_REPO_LOGS = Path("/data2/linmin/EmbodiedAI/logs")

# 无参考 path 时按步数上限结束一轮；可用环境变量覆盖
_DEFAULT_MAX_STEPS = 250

_loader: DatasetLoader | None = None


def get_loader() -> DatasetLoader:
    global _loader
    if _loader is None:
        _loader = DatasetLoader(
            root_dir="/data2/linmin/EmbodiedAI/resource/datasets/all_task",
            scene_path="/data2/linmin/NPC/hssd_scene_new",
            robot_path="/data2/linmin/EmbodiedAI/resource/robots/stretch/stretch_pos.usd",
            headless=True,
        )
    return _loader


def _coerce_done_flag(done: object) -> bool:
    if isinstance(done, (list, tuple)) and len(done) > 0:
        return bool(done[0])
    return bool(done)


def _log_loop_exit_reason(
    log_file_path: Path,
    env: BaseEnv,
    done: object,
    max_steps: int,
) -> None:
    """在 while 正常结束时写入原因（未 break）。Kit 崩溃/segfault 时不会执行到此处。"""
    done_b = _coerce_done_flag(done)
    try:
        is_run = bool(env.is_running)
        steps = int(env.task[0].steps) if env.task else -1
    except Exception as e:
        is_run, steps = False, -1
        tail = f" (读取 env 状态时异常: {e!r})"
    else:
        tail = ""

    if not is_run:
        reason = (
            "env.is_running=False — SimulationApp/Kit 已停止；"
            "常见：GPU/显存、驱动、env.step 内 native 崩溃、进程被 kill；"
            "若上一行停在 before env.step 而无 after，多为本次 env.step 未返回即退出。"
        )
    elif done_b:
        if steps >= max_steps:
            reason = f"done=True（task.steps>={max_steps}，达到 BC_TRANSFORMER_MAX_STEPS）"
        else:
            reason = "done=True（任务/环境 task.step 返回 episode 结束）"
    else:
        reason = "unexpected（逻辑上不应出现：while 已退出但 done=False 且 is_running=True）"

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"=== loop_exit: {reason}{tail} | task.steps={steps} | "
            f"env.is_running={is_run} | done={done} (coerced={done_b}) ===\n\n"
        )


def _log_action_distribution(
    log_file_path: Path,
    count_train: Counter,
    count_sim: Counter,
    num_env_steps: int,
) -> None:
    """汇总本段仿真中模型输出的 train_action_id 与映射后 sim_command0 的频次。"""
    other_train = {k: v for k, v in count_train.items() if k not in (0, 1, 2)}
    other_sim = {k: v for k, v in count_sim.items() if k not in (0, 1, 2, 3)}
    lines = [
        "=== action distribution（本段实际执行 env.step 的步数） ===\n",
        f"env_steps={num_env_steps}\n",
        "train_action_id（与 dataset / 训练一致）: "
        f"0=forward={count_train[0]} | 1=turn_left={count_train[1]} | 2=turn_right={count_train[2]}",
    ]
    if other_train:
        lines.append(f" | other={other_train}")
    lines.append("\n")
    lines.append(
        "sim_command[0]（PositionController）: "
        f"0=fwd={count_sim[0]} | 1=back={count_sim[1]} | 2=left={count_sim[2]} | 3=right={count_sim[3]}"
    )
    if other_sim:
        lines.append(f" | other={other_sim}")
    lines.append("\n")
    total_t = sum(count_train.values())
    if num_env_steps:
        lines.append(
            "train 占比: "
            + " | ".join(
                f"id{k}={100.0 * count_train[k] / num_env_steps:.1f}%"
                for k in (0, 1, 2)
            )
            + "\n"
        )
        if num_env_steps and total_t != num_env_steps:
            lines.append(
                f"note: count_train_sum={total_t} vs env_steps={num_env_steps}（若不等请检查是否中途异常）\n"
            )
    lines.append("\n")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write("".join(lines))


def run_bc_transformer_scene(scene_id: int, log_file_path: str | Path) -> None:
    """
    在单个场景 id 上运行 BCTransformerAgent；动作与 step 信息追加写入 ``log_file_path``。
    """
    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    loader = get_loader()
    cfg = loader[scene_id]

    ckpt = os.environ.get("BC_TRANSFORMER_CHECKPOINT", _DEFAULT_CKPT)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"未找到 checkpoint: {ckpt}，请训练生成或设置 BC_TRANSFORMER_CHECKPOINT"
        )

    # 先启动 Isaac SimulationApp；若使用 ZMQ 策略进程，则本进程不加载 torch/transformers。
    env = BaseEnv(cfg)

    zmq_ep = os.environ.get("BC_POLICY_ZMQ", "").strip()
    if zmq_ep:
        BCTransformerZmqAgent = _load_zmq_agent_class()
        agent_cfg = SimpleNamespace(
            zmq_endpoint=zmq_ep,
            num_actions=int(os.environ.get("BC_NUM_ACTIONS", "3")),
        )
        agent = BCTransformerZmqAgent(agent_cfg)
    else:
        import torch

        BCTransformerAgent = _load_bct_agent_class()
        agent_cfg = SimpleNamespace(
            checkpoint_path=ckpt,
            model_config_path=None,
            data_root=_DEFAULT_DATA_ROOT if os.path.isdir(_DEFAULT_DATA_ROOT) else None,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        agent = BCTransformerAgent(agent_cfg)

    max_steps = int(os.environ.get("BC_TRANSFORMER_MAX_STEPS", str(_DEFAULT_MAX_STEPS)))

    count_train = Counter()
    count_sim = Counter()
    num_env_steps = 0
    obs = env.reset()
    agent.reset()
    done = False
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"=== scene_id={scene_id} | max_steps={max_steps} "
            f"(BC_TRANSFORMER_MAX_STEPS) ===\n"
            f"BC train_id→sim PositionController: 0→0(fwd), 1→2(left), 2→3(right) "
            f"(see simulator/utils/bc_action_sim_mapping.py; set BC_SKIP_TRAIN_SIM_ACTION_MAP=1 to disable)\n\n"
        )
    try:
        while env.is_running and not done:
            target_ids = extract_target_ids(cfg.task.task_path)
            objs_xformprim = env.sim.find_object_by_id(env.scenes[0], target_ids)
            _goal_pos1, _ = objs_xformprim[0].get_world_pose()
            steps_before = env.task[0].steps
            action = agent.act(obs[0])
            if os.environ.get("BC_SKIP_TRAIN_SIM_ACTION_MAP", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                sim_action = action
            else:
                sim_action = train_action_tuple_to_sim(action)
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"--- before env.step | task.steps={steps_before} ---\n"
                    f"action [train_action_id, value] (matches dataset): {action}\n"
                    f"action [sim_command0, value] (PositionController): {sim_action}\n"
                    f"observation position (x,y,z): {obs[0]['position']}\n"
                    f"observation yaw (rad): {obs[0]['yaw']}\n"
                )
            obs, reward, done, info = env.step([sim_action])
            num_env_steps += 1
            try:
                tid = int(action[0])
            except (TypeError, ValueError, IndexError):
                tid = -1
            count_train[tid] += 1
            try:
                sid = int(sim_action[0])
            except (TypeError, ValueError, IndexError):
                sid = -1
            count_sim[sid] += 1
            steps_after = env.task[0].steps
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"--- after env.step | task.steps={steps_after}/{max_steps} ---\n"
                    f"reward: {reward}\n"
                    f"done: {done}\n"
                    f"info: {info}\n\n"
                )
            if env.task[0].steps >= max_steps:
                done = True
            else:
                done = done[0]
        else:
            # while 未用 break 退出：要么 env.is_running 为假，要么 done 为真
            _log_loop_exit_reason(log_file_path, env, done, max_steps)
    except BaseException:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write("=== loop_exit: PYTHON EXCEPTION（见下方 traceback）===\n")
            log_file.write(traceback.format_exc())
            log_file.write("\n")
        raise
    finally:
        _log_action_distribution(log_file_path, count_train, count_sim, num_env_steps)
        env.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_bc_transformer.py <scene_id>")
        sys.exit(1)
    sid = int(sys.argv[1])
    _REPO_LOGS.mkdir(parents=True, exist_ok=True)
    out = _REPO_LOGS / f"bc_transformer_scene_{sid}.log"
    print("total len:", len(get_loader()))
    print(f"scene_id={sid} -> {out}")
    run_bc_transformer_scene(sid, out)
