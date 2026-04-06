# Training Process Response (Issue Reply)

Thanks for the question! Based on the current repository structure, here is a practical training workflow.

## 1) Dataset preparation

Please prepare your raw demos under a root directory where each demo folder contains:

- `task.json` (with either `"Task instruction"` or `"instruction"`)
- `output.json` (frame-wise action records)
- image folders:
  - `<demo_name>_l/frame_XXXX_l.png`
  - `<demo_name>_m/frame_XXXX_m.png`
  - `<demo_name>_r/frame_XXXX_r.png`

Then build training metadata (`train.json` / `val.json`) with:

```bash
cd imitation_learning_project
python dataset/build_dataset.py --data_root <raw_data_directory> --output_dir <dataset_output_directory>
```

> Note: `build_hssd_dataset.py` currently outputs a flat sample list, while the current dataloader expects a nested-per-demo structure.  
> For now, `build_dataset.py` is the safer default unless dataloader logic is updated.

## 2) Start training

Run training from `imitation_learning_project`:

```bash
cd imitation_learning_project
python train.py --data_root <dataset_output_directory> --save_dir <checkpoint_directory>
```

`--data_root` should point to the folder that contains `train.json` and `val.json`.

## 3) Config files and key parameters to adjust

Main config: `imitation_learning_project/configs/model_config.py`

Most important knobs:

- `batch_size`
- `learning_rate`
- `max_epochs`
- `weight_decay`
- Transformer size (`hidden_size`, `num_heads`, `num_layers`, `ff_dim`)
- `text_encoder_name` (to match your instruction language/domain)

Also, logs/checkpoints are saved to:

- training logs: `logs/`
- wandb offline logs: `wandb/`
- model checkpoints: your `--save_dir`

## 4) Current code caveats you may want to patch first

Before running large-scale training, there are two things to double-check:

1. `train.py` imports `from config.model_config import ModelConfig`, while the folder is `configs/`.  
   If this raises `ModuleNotFoundError`, change it to:
   ```python
   from configs.model_config import ModelConfig
   ```

2. In `build_dataset.py`, train/val split logic currently sets:
   - `train_dirs = demo_dirs`
   - `val_dirs = demo_dirs[split_idx:]`

   This means validation demos are also included in training (data leakage).  
   Recommended:
   - `train_dirs = demo_dirs[:split_idx]`
   - `val_dirs = demo_dirs[split_idx:]`

If useful, I can open a PR with these two fixes plus a short "Training Quickstart" section in the README.

Reference repository: [pzhren/EmbodiedAI](https://github.com/pzhren/EmbodiedAI)
