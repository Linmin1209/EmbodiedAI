# Imitation Learning Project

导航模仿学习：三视角 RGB + 自然语言指令，预测离散动作与连续动作幅值。

## 依赖

Python 3.8+，建议：

```bash
pip install torch torchvision
pip install transformers tqdm pillow numpy pyyaml wandb
```

## 1. 构建导航数据集（多根目录）

支持同时扫描 `extra_data` / `extra_data2` / `gen_scene_data` 等目录：递归查找包含 `task.json` 与 `new_output.json` 或 `output.json` 的 demo，统一帧格式后生成 `train.json` / `val.json`。

```bash
python dataset/build_nav_dataset.py \
  --data_roots /data2/lfwj/extra_data /data2/lfwj/extra_data2 /data2/lfwj/gen_scene_data \
  --output_dir /data2/linmin/EmbodiedAI/resource/datasets/nav_mix \
  --train_ratio 0.9 \
  --seed 42
```

- **动作压缩**：默认单帧微观量 `--step_value 0.02`；连续同动作若超过 `--max_action_value 0.2`，按**均匀切段**（例如 12 帧 → 两段各 6 帧，即 `0.12`+`0.12`，避免 `0.20`+`0.04` 这种尾差过小）。
- **JSON**：默认 **`indent=4`** 写入 `train.json` / `val.json` / `dataset_meta.json`；需要单行紧凑可加 `--compact-json`。
- **软链接**（可选）：加 `--symlink_images` 会在 `output_dir/image_symlinks/<hash>/` 下为原图建立软链接，JSON 内路径指向链接（不复制像素）。
会额外写入 `dataset_meta.json`。仅保留动作 `move forward` / `turn left` / `turn right`。

### 旧版单目录构建

仍可使用 `dataset/build_dataset.py` 或 `build_hssd_dataset.py`（单根 `data_root`）。

## 2. 训练

```bash
python train.py \
  --data_root /data2/linmin/EmbodiedAI/resource/datasets/nav_mix \
  --save_dir checkpoints/run1 \
  --wandb_mode online \
  --wandb_project imitation_nav_bc \
  --wandb_run_name exp1
```

- **日志**：控制台 + `save_dir/logs/train_*.txt`
- **wandb**：`--wandb_mode` 可选 `online` / `offline` / `disabled`；离线日志在 `save_dir/wandb/`。
- **超参**：默认见 `configs/model_config.py`，也可用 `--learning_rate`、`--batch_size` 等覆盖，或 `--from_config custom.yaml`。

检查点：`best.pt`（按验证 loss）、`epoch_*.pt`。

### 说明

- 数据 JSON 中图像路径为**绝对路径**；若使用 `--symlink_images` 则为输出目录下**链接文件的路径**（训练时仍可正常 `open`）。
- 文本 tokenizer 与预训练编码器一致（默认同为 `bert-base-uncased`）。

## 3. 配置

编辑 `configs/model_config.py`：隐藏维度、层数、label smoothing、Huber 值损失、warmup 比例、梯度累积、冻结骨干 epoch 数、wandb 项目名等。
