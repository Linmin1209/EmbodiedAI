#!/usr/bin/env bash
# 逐个运行 tests/test_scene/test_bc_transformer.py，每个场景 id 对应
# /data2/linmin/EmbodiedAI/logs/bc_transformer_scene_<id>.log
#
# 用法（在任意目录）:
#   bash /data2/linmin/EmbodiedAI/scripts/run_bc_transformer_scenes.sh
#   bash .../run_bc_transformer_scenes.sh 0 99          # 仅 id 0～99（含）
#   CUDA_VISIBLE_DEVICES=2,3 BC_TRANSFORMER_CHECKPOINT=/path/to.pt bash .../run_bc_transformer_scenes.sh
#
# ZMQ 模式（策略在独立进程，Isaac 进程不加载 torch/transformers）::
#   bash .../run_bc_transformer_scenes.sh --zmq
#   bash .../run_bc_transformer_scenes.sh --zmq 0 99
#   USE_BC_POLICY_ZMQ=1 bash .../run_bc_transformer_scenes.sh   # 与 --zmq 等价
# 需设置 BC_TRANSFORMER_CHECKPOINT；本机启动服务后自动 export BC_POLICY_ZMQ=tcp://127.0.0.1:5556
# 可选: BC_POLICY_BIND（默认 tcp://127.0.0.1:5556） BC_POLICY_SERVER_PYTHON（默认与 PYTHON 相同）
#       BC_DATA_ROOT（传给策略服务的 --data-root）
#       BC_POLICY_WAIT_MODEL_READY=1 等权重加载完再跑场景；=0（默认）仅等端口可连
# 若你已在别处启动策略服务并自行 export BC_POLICY_ZMQ，则不要加 --zmq，脚本只跑仿真。
#
# Python 解释器默认用环境变量 PYTHON，未设置则为 python（Isaac 环境请改为 isaac 自带 python 全路径）。
#
# 停止批处理（本脚本、ZMQ 策略服务、残留的 test_bc_transformer）::
#   bash /data2/linmin/EmbodiedAI/scripts/stop_bc_transformer_scenes.sh

set -euo pipefail

ROOT="/data2/linmin/EmbodiedAI"
LOG_DIR="${ROOT}/logs"
TEST_PY="${ROOT}/tests/test_scene/test_bc_transformer.py"
SERVER_PY="${ROOT}/imitation_learning_project/scripts/bc_policy_zmq_server.py"
PYTHON="${PYTHON:-python}"

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ ! -f "$TEST_PY" ]]; then
  echo "未找到: $TEST_PY" >&2
  exit 1
fi

USE_ZMQ=0
if [[ "${USE_BC_POLICY_ZMQ:-0}" == "1" ]]; then
  USE_ZMQ=1
fi
if [[ "${1:-}" == "--zmq" ]]; then
  USE_ZMQ=1
  shift
fi

ZMQ_SERVER_PID=""
ZMQ_SERVER_PIDFILE="${LOG_DIR}/bc_policy_zmq_server.pid"
ZMQ_SERVER_LOG="${LOG_DIR}/bc_policy_zmq_server.log"
# 供 cleanup / trap 使用；在写入 run_bc_transformer_scenes.pid 之前可能仍为空
PIDFILE=""

kill_tree() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  pkill -TERM -P "$pid" 2>/dev/null || true
  sleep 1
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  sleep 1
  kill -KILL "$pid" 2>/dev/null || true
  return 0
}

_client_uri_from_bind() {
  local b="$1"
  # 本机仿真进程连接用 127.0.0.1，避免 bind 在 0.0.0.0 时客户端不知如何写地址
  echo "$b" | sed 's/0\.0\.0\.0/127.0.0.1/'
}

# 等待策略进程可连，且（默认）等权重与 BERT 加载完毕（见 bc_policy_zmq_server 先 bind 再后台加载）
_wait_policy_server() {
  local uri="$1"
  local py="${2:-python}"
  # 默认 0：只等 bind 成功（ping ok）；=1 时等 model_loaded（或 load_error 立即失败）
  local wait_model="${BC_POLICY_WAIT_MODEL_READY:-0}"
  "$py" -c "
import sys
import time
import zmq
ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.connect('${uri}')
s.setsockopt(zmq.RCVTIMEO, 5000)
wait_model = '${wait_model}' not in ('0', 'false', 'False', '')
for i in range(7200):
    try:
        s.send_pyobj({'type': 'ping'})
        r = s.recv_pyobj()
        if isinstance(r, dict) and r.get('ok'):
            err = r.get('load_error')
            if err:
                print('ZMQ policy load_error:', err, file=sys.stderr)
                sys.exit(1)
            if not wait_model or r.get('model_loaded'):
                sys.exit(0)
    except Exception as e:
        if i == 0 or i % 30 == 0:
            print('wait policy (retry):', repr(e), file=sys.stderr)
    time.sleep(1)
sys.exit(1)
" 2>&1
}

_start_zmq_policy_server() {
  if [[ ! -f "$SERVER_PY" ]]; then
    echo "未找到 ZMQ 策略脚本: $SERVER_PY" >&2
    exit 1
  fi
  local ckpt="${BC_TRANSFORMER_CHECKPOINT:-}"
  if [[ -z "$ckpt" ]]; then
    echo "ZMQ 模式需要设置 BC_TRANSFORMER_CHECKPOINT（checkpoint 路径）" >&2
    exit 1
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "Checkpoint 不存在: $ckpt" >&2
    exit 1
  fi

  local bind="${BC_POLICY_BIND:-tcp://127.0.0.1:5556}"
  local client_uri
  client_uri="$(_client_uri_from_bind "$bind")"
  if [[ -n "${BC_POLICY_ZMQ_CLIENT_URI:-}" ]]; then
    client_uri="${BC_POLICY_ZMQ_CLIENT_URI}"
  fi

  local pol_py="${BC_POLICY_SERVER_PYTHON:-$PYTHON}"
  : >"$ZMQ_SERVER_LOG"
  {
    echo "[$(date -Is)] shell: starting policy server pid will follow"
    echo "[$(date -Is)] cmd: $pol_py -u $SERVER_PY --checkpoint <ckpt> --bind $bind"
  } >>"$ZMQ_SERVER_LOG"
  export PYTHONUNBUFFERED=1
  # 避免 ( exec >>log; exec python ) 在部分环境下子进程无输出
  "$pol_py" -u "$SERVER_PY" \
    --checkpoint "$ckpt" \
    --bind "$bind" \
    ${BC_DATA_ROOT:+--data-root "$BC_DATA_ROOT"} >>"$ZMQ_SERVER_LOG" 2>&1 &
  ZMQ_SERVER_PID=$!
  echo "$ZMQ_SERVER_PID" >"$ZMQ_SERVER_PIDFILE"
  echo "[$(date -Is)] shell: policy server PID=$ZMQ_SERVER_PID" >>"$ZMQ_SERVER_LOG"

  if ! _wait_policy_server "$client_uri" "$pol_py"; then
    echo "ZMQ 策略服务在约 7200s 内未就绪。请查看 $ZMQ_SERVER_LOG；BC_POLICY_WAIT_MODEL_READY=0 仅等端口，=1 等权重加载完" >&2
    kill_tree "$ZMQ_SERVER_PID"
    rm -f "$ZMQ_SERVER_PIDFILE"
    exit 1
  fi
  export BC_POLICY_ZMQ="$client_uri"
  echo "[$(date -Is)] ZMQ 策略服务就绪 BC_POLICY_ZMQ=$BC_POLICY_ZMQ (PID=$ZMQ_SERVER_PID)"
}

cleanup() {
  rm -f "${PIDFILE:-}" 2>/dev/null || true
  if [[ -n "${ZMQ_SERVER_PID:-}" ]]; then
    echo "[$(date -Is)] 结束 ZMQ 策略服务 PID=$ZMQ_SERVER_PID"
    kill_tree "$ZMQ_SERVER_PID"
    rm -f "$ZMQ_SERVER_PIDFILE"
    ZMQ_SERVER_PID=""
  fi
}

if [[ "$USE_ZMQ" -eq 1 ]]; then
  _start_zmq_policy_server
fi
trap cleanup EXIT INT TERM

# 数据集长度（与 test_bc_transformer.get_loader 一致）
# 注意：Gym/Isaac 可能在 import 时往 stdout 打警告，不能整段当整数用；只取「纯数字行」
_N_OUT="$(
  PYTHONWARNINGS=ignore \
  GYM_DISABLE_WARNINGS=1 \
  "$PYTHON" -W ignore -c "
import sys
from pathlib import Path
import importlib.util
root = Path(r'${ROOT}')
sys.path.insert(0, str(root))
p = root / 'tests' / 'test_scene' / 'test_bc_transformer.py'
spec = importlib.util.spec_from_file_location('tbc', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(len(m.get_loader()))
" 2>/dev/null
)"
N="$(echo "$_N_OUT" | grep -E '^[0-9]+$' | tail -n 1)"
if [[ -z "$N" ]]; then
  echo "无法解析数据集长度。Python 输出（节选）：" >&2
  echo "$_N_OUT" >&2
  cleanup 2>/dev/null || true
  exit 1
fi

START="${1:-0}"
if [[ -n "${2:-}" ]]; then
  END_INCL="$2"
else
  END_INCL=$((N - 1))
fi

if (( START < 0 || END_INCL >= N || START > END_INCL )); then
  echo "无效区间: START=$START END=$END_INCL (数据集长度 N=$N)" >&2
  cleanup 2>/dev/null || true
  exit 1
fi

SUMMARY="${LOG_DIR}/bc_transformer_batch_summary.log"
{
  echo "--- $(date -Is) batch START=$START END_INCL=$END_INCL N=$N ---"
  if [[ "$USE_ZMQ" -eq 1 ]]; then
    echo "mode=zmq BC_POLICY_ZMQ=${BC_POLICY_ZMQ:-}"
  else
    echo "mode=in_process_bc_agent"
  fi
} >>"$SUMMARY"

# 供 stop_bc_transformer_scenes.sh 结束本脚本及子进程；正常退出时由 trap 删除
PIDFILE="${LOG_DIR}/run_bc_transformer_scenes.pid"
echo $$ >"$PIDFILE"

failed=0
for (( sid=START; sid<=END_INCL; sid++ )); do
  LOG="${LOG_DIR}/bc_transformer_scene_${sid}.log"
  echo "[$(date -Is)] scene_id=$sid / $((END_INCL - START + 1)) -> $LOG"
  : > "$LOG"
  if "$PYTHON" "$TEST_PY" "$sid" >>"$LOG" 2>&1; then
    echo "[$(date -Is)] OK scene_id=$sid" >> "$SUMMARY"
  else
    ec=$?
    {
      echo "[$(date -Is)] FAIL scene_id=$sid exit_code=$ec log=$LOG"
      echo "----- tail ${LOG} (last 50 lines) -----"
      tail -n 50 "$LOG" 2>/dev/null || true
      echo "----- end tail -----"
    } | tee -a "$SUMMARY"
    failed=$((failed + 1))
  fi
done

echo "--- $(date -Is) batch done fail=$failed ---" >> "$SUMMARY"
echo "完成: 失败 $failed 个，日志目录 $LOG_DIR"
if (( failed > 0 )); then exit 2; else exit 0; fi
