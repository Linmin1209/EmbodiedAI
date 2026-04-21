#!/usr/bin/env bash
# 单机多卡 DDP 启动（torchrun）。
#
# 默认（可直接 ./scripts/train_ddp.sh）：
#   CUDA_VISIBLE_DEVICES=0,1,4,5  NUM_GPUS=4
#   DATA_ROOT=/data2/linmin/EmbodiedAI/train_data
#   --batch_size 32
# wandb 随 ModelConfig 默认 online；关闭可附加 --wandb_mode disabled
#
# 默认物理卡 **0,1,4,5**（CUDA_VISIBLE_DEVICES）；与 NUM_GPUS=4 一致。
# 覆盖示例：
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_ddp.sh
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=6,7 ./scripts/train_ddp.sh
#   DATA_ROOT=/other/path ./scripts/train_ddp.sh --batch_size 32
#
# argparse 对重复选项取**最后一次**，故命令行写在后时会覆盖上面的默认 train 参数。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 仅使用物理 GPU 0、1、4、5；进程内映射为 cuda:0..3。改卡号请设 CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5}"

TRAIN_PY="${PROJECT_ROOT}/train.py"

NUM_GPUS="${NUM_GPUS:-4}"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=4
fi

SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/checkpoints_ddp}"
DATA_ROOT="${DATA_ROOT:-/data2/linmin/EmbodiedAI/train_data}"

TORCHRUN_ARGS=(
  --standalone
  --nproc_per_node="${NUM_GPUS}"
)

# 默认 train 参数；若命令行再传同名字段，通常以命令行为准（靠后覆盖）
TRAIN_ARGS=(
  --save_dir "${SAVE_DIR}"
  --data_root "${DATA_ROOT}"
  --batch_size 32
)

EXTRA=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  EXTRA=("$@")
fi

echo "[train_ddp] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[train_ddp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train_ddp] NUM_GPUS=${NUM_GPUS}  (export NUM_GPUS=N 覆盖，须与可见卡数量一致)"
echo "[train_ddp] SAVE_DIR=${SAVE_DIR}"
echo "[train_ddp] DATA_ROOT=${DATA_ROOT}  (export DATA_ROOT=... 覆盖)"
exec torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_PY}" "${TRAIN_ARGS[@]}" "${EXTRA[@]}"
