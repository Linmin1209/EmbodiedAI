#!/usr/bin/env python3
"""
遍历 ``DatasetLoader`` 中全部场景 id，逐个调用 ``test_bc_transformer.run_bc_transformer_scene``，
并将每个场景的 **标准输出/标准错误** 以及脚本内 ``print`` 写入独立日志文件::

    /data2/linmin/EmbodiedAI/logs/bc_transformer_scene_<id>.log

用法（在 EmbodiedAI 仓库根目录）::

    python scripts/run_bc_transformer_all_scenes.py
    python scripts/run_bc_transformer_all_scenes.py --start 0 --end 10
    BC_TRANSFORMER_CHECKPOINT=/path/to/best.pt python scripts/run_bc_transformer_all_scenes.py

指定可见物理 GPU（逗号分隔，等价于 ``CUDA_VISIBLE_DEVICES``），例如只用 2、3 号卡::

    python scripts/run_bc_transformer_all_scenes.py --gpu 2,3

单卡示例::

    python scripts/run_bc_transformer_all_scenes.py --gpu 23

可选：某场景失败时立即停止::

    python scripts/run_bc_transformer_all_scenes.py --fail-fast
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# 仓库根目录
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOG_DIR = _ROOT / "logs"


def _load_test_bc_module():
    path = _ROOT / "tests" / "test_scene" / "test_bc_transformer.py"
    spec = importlib.util.spec_from_file_location("test_bc_transformer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description="批量跑 BCTransformer 全场景并分别写日志")
    parser.add_argument("--start", type=int, default=0, help="起始场景 id（含）")
    parser.add_argument("--end", type=int, default=None, help="结束场景 id（不含）；默认到数据集长度")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="单场景异常时立即退出（默认跳过失败场景并继续）",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        metavar="IDS",
        help="可见物理 GPU，逗号分隔，如 2,3 表示仅 2、3 号卡（CUDA_VISIBLE_DEVICES=2,3）；"
        "在加载 PyTorch 前生效。多卡时依次为 cuda:0、cuda:1…，当前 BCTransformer 仍用 cuda:0。",
    )
    args = parser.parse_args()

    # 须在 import torch（经 test_bc_transformer）之前设置
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    tbt = _load_test_bc_module()
    get_loader = tbt.get_loader
    run_bc_transformer_scene = tbt.run_bc_transformer_scene

    loader = get_loader()
    n = len(loader)
    end = args.end if args.end is not None else n
    end = min(end, n)
    start = max(0, args.start)
    if start >= end:
        print(f"无效区间: start={start} end={end} (loader len={n})")
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / "bc_transformer_batch_summary.log"
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(f"--- batch start: range [{start}, {end}) total_loader={n} ---\n")

    print(f"DatasetLoader len={n}, will run scene_id in [{start}, {end})")
    print(f"Per-scene logs: {LOG_DIR}/bc_transformer_scene_<id>.log")

    failed: list[tuple[int, str]] = []
    for sid in range(start, end):
        log_path = LOG_DIR / f"bc_transformer_scene_{sid}.log"
        print(f"[{sid - start + 1}/{end - start}] scene_id={sid} -> {log_path}", flush=True)
        try:
            with open(log_path, "w", encoding="utf-8") as log_f:
                with redirect_stdout(log_f), redirect_stderr(log_f):
                    print(f"scene_id={sid} dataset_len={n}", flush=True)
                    run_bc_transformer_scene(sid, log_path)
                    print(f"scene_id={sid} finished OK", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write("\n--- EXCEPTION ---\n")
                log_f.write(err)
            failed.append((sid, err))
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(f"FAIL scene_id={sid}: {e!r}\n")
            print(f"FAIL scene_id={sid}: {e}", flush=True)
            if args.fail_fast:
                sys.exit(1)

    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(
            f"--- batch end: ok={end - start - len(failed)} fail={len(failed)} ---\n"
        )
    if failed:
        print(f"完成：失败 {len(failed)} 个，见各场景 log 尾部与 {summary_path}")
        sys.exit(2)
    print("全部场景运行完成。")


if __name__ == "__main__":
    main()
