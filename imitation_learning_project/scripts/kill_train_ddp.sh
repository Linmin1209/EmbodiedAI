#!/usr/bin/env bash
# 结束由 scripts/train_ddp.sh 启动的训练：匹配本仓库 ``train.py`` 的进程（含 torchrun 子进程）。
#
#   ./scripts/kill_train_ddp.sh           # 先发 SIGTERM，2 秒后仍存活则 SIGKILL
#   ./scripts/kill_train_ddp.sh -n      # 仅打印将杀掉的进程，不执行
#   ./scripts/kill_train_ddp.sh -9      # 直接 SIGKILL
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MATCH="${PROJECT_ROOT}/train.py"

DRY_RUN=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n | --dry-run)
      DRY_RUN=1
      shift
      ;;
    -9 | --force)
      FORCE=1
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [-n|--dry-run] [-9|--force]"
      echo "  结束匹配 ${MATCH} 的进程（DDP 各 rank / torchrun 子进程）。"
      exit 0
      ;;
    *)
      echo "未知参数: $1 （见 -h）" >&2
      exit 1
      ;;
  esac
done

mapfile -t PID_LIST < <(pgrep -f "$MATCH" 2>/dev/null | sort -u || true)

if [[ ${#PID_LIST[@]} -eq 0 ]]; then
  echo "[kill_train_ddp] 未发现匹配进程: ${MATCH}"
  exit 0
fi

echo "[kill_train_ddp] 匹配 ${MATCH} 的 PID: ${PID_LIST[*]}"
for pid in "${PID_LIST[@]}"; do
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    printf "  "
    tr '\0' ' ' <"/proc/${pid}/cmdline" || true
    echo " (pid=${pid})"
  else
    ps -p "$pid" -o pid,cmd= 2>/dev/null | sed 's/^/  /' || true
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[kill_train_ddp] dry-run，未发送信号"
  exit 0
fi

kill_pids() {
  local sig=$1
  shift
  local p
  for p in "$@"; do
    kill "-${sig}" "$p" 2>/dev/null || true
  done
}

if [[ "$FORCE" -eq 1 ]]; then
  kill_pids KILL "${PID_LIST[@]}"
  echo "[kill_train_ddp] 已发送 SIGKILL"
  exit 0
fi

kill_pids TERM "${PID_LIST[@]}"
echo "[kill_train_ddp] 已发送 SIGTERM，等待 2s…"
sleep 2

mapfile -t LEFT < <(pgrep -f "$MATCH" 2>/dev/null | sort -u || true)
if [[ ${#LEFT[@]} -gt 0 ]]; then
  echo "[kill_train_ddp] 仍存活: ${LEFT[*]}，发送 SIGKILL"
  kill_pids KILL "${LEFT[@]}"
else
  echo "[kill_train_ddp] 进程已结束"
fi
