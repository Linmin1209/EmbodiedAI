#!/usr/bin/env bash
# 停止由 run_bc_transformer_scenes.sh 启动的批处理及其子进程，以及本机启动的 ZMQ 策略服务。
#
# 用法:
#   bash /data2/linmin/EmbodiedAI/scripts/stop_bc_transformer_scenes.sh
#
# 逻辑:
#   1) 若存在 logs/run_bc_transformer_scenes.pid，对其发 SIGTERM，再必要时 SIGKILL
#   2) 若存在 logs/bc_policy_zmq_server.pid，结束 ZMQ 策略服务进程树
#   3) 对仍匹配的 python .../test_bc_transformer.py、pkill
#   4) 对仍匹配的 bc_policy_zmq_server.py、pkill

set -uo pipefail

ROOT="/data2/linmin/EmbodiedAI"
LOG_DIR="${ROOT}/logs"
PIDFILE="${LOG_DIR}/run_bc_transformer_scenes.pid"
ZMQ_PIDFILE="${LOG_DIR}/bc_policy_zmq_server.pid"

kill_tree() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  # 先结束子进程（常见：python / Isaac）
  pkill -TERM -P "$pid" 2>/dev/null || true
  sleep 1
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  sleep 1
  kill -KILL "$pid" 2>/dev/null || true
  return 0
}

if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE" | tr -d '[:space:]')"
  if [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "结束批处理 shell PID=$pid (来自 $PIDFILE)"
    kill_tree "$pid"
  fi
  rm -f "$PIDFILE"
else
  echo "未找到 $PIDFILE（批处理可能已结束或未写 PID）"
fi

if [[ -f "$ZMQ_PIDFILE" ]]; then
  zpid="$(cat "$ZMQ_PIDFILE" | tr -d '[:space:]')"
  if [[ -n "$zpid" ]] && [[ "$zpid" =~ ^[0-9]+$ ]]; then
    echo "结束 ZMQ 策略服务 PID=$zpid (来自 $ZMQ_PIDFILE)"
    kill_tree "$zpid"
  fi
  rm -f "$ZMQ_PIDFILE"
fi

# 残留：bc_policy_zmq_server.py（路径特征）
if pgrep -f "bc_policy_zmq_server.py" >/dev/null 2>&1; then
  echo "结束残留的 bc_policy_zmq_server.py 进程..."
  pkill -TERM -f "bc_policy_zmq_server.py" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "bc_policy_zmq_server.py" 2>/dev/null || true
fi

# 残留：直接跑的 test_bc_transformer.py（路径特征）
if pgrep -f "test_bc_transformer.py" >/dev/null 2>&1; then
  echo "结束残留的 test_bc_transformer.py 进程..."
  pkill -TERM -f "test_bc_transformer.py" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "test_bc_transformer.py" 2>/dev/null || true
fi

echo "完成。"
