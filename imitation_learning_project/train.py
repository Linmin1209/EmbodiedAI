#!/usr/bin/env python3
"""
行为克隆训练脚本。支持梯度累积、warmup+cosine、混合精度、详细日志与 wandb。

**单机多卡 DDP**（``batch_size`` 为每卡 batch，全局 batch ≈ ``N * batch_size``）::

    torchrun --standalone --nproc_per_node=4 \\
      /data2/linmin/EmbodiedAI/imitation_learning_project/train.py \\
      --save_dir checkpoints_ddp

单机单卡直接 ``python train.py`` 即可（不设 ``WORLD_SIZE`` 时不走分布式）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # PyTorch < 2.0
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

from transformers import get_cosine_schedule_with_warmup

from models.bc_transformer import BCTransformer
from dataset.dataset import DEFAULT_NAV_DATA_ROOT, ImitationDataset, get_dataloader
from dataset.value_normalization import ensure_value_norm_file
from configs.model_config import ModelConfig
from hf_compat import hf_local_files_only


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BC transformer for navigation imitation.")
    p.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_NAV_DATA_ROOT,
        help="含 train.json / val.json 的目录（默认：build_nav_dataset 输出 train_data）",
    )
    p.add_argument("--save_dir", type=str, default="checkpoints", help="检查点与 run 输出目录")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb_mode", type=str, default=None, help="online | offline | disabled")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--from_config", type=str, default=None, help="从 yaml 加载部分超参（可选）")
    p.add_argument(
        "--ddp_no_find_unused",
        action="store_true",
        help="DDP 关闭 find_unused_parameters（略快；本模型默认需开启以兼容 reg/cls 子图差异）",
    )
    return p.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    """
    若由 ``torchrun`` / ``torch.distributed.launch`` 启动且 WORLD_SIZE>1，则初始化进程组。
    返回 (is_ddp, rank, world_size, local_rank)；单进程时为 (False, 0, 1, 0)。
    """
    if "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return False, 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("多卡 DDP 需要 CUDA。")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    return True, rank, world_size, local_rank


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def load_yaml_overrides(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_cli_to_config(config: ModelConfig, cli: argparse.Namespace) -> ModelConfig:
    if cli.batch_size is not None:
        config.batch_size = cli.batch_size
    if cli.max_epochs is not None:
        config.max_epochs = cli.max_epochs
    if cli.learning_rate is not None:
        config.learning_rate = cli.learning_rate
    if cli.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = cli.gradient_accumulation_steps
    if cli.num_workers is not None:
        config.num_workers = cli.num_workers
    if cli.seed is not None:
        config.seed = cli.seed
    if cli.wandb_mode is not None:
        config.wandb_mode = cli.wandb_mode
    if cli.wandb_project is not None:
        config.wandb_project = cli.wandb_project
    if cli.wandb_run_name is not None:
        config.wandb_run_name = cli.wandb_run_name
    if cli.from_config:
        for k, v in load_yaml_overrides(cli.from_config).items():
            if hasattr(config, k) and v is not None:
                setattr(config, k, v)
    return config


def setup_logging(save_dir: str, rank: int = 0) -> logging.Logger:
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")

    logger = logging.getLogger("imitation_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if rank == 0:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.info("Log file: %s", log_path)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def setup_wandb(config: ModelConfig, save_dir: str):
    mode = (config.wandb_mode or "online").lower()
    if mode == "disabled":
        return None

    import wandb

    run_name = config.wandb_run_name or datetime.now().strftime("nav_bc_%Y%m%d_%H%M%S")
    wdir = os.path.join(save_dir, "wandb")
    os.makedirs(wdir, exist_ok=True)

    init_kw = {
        "project": config.wandb_project,
        "name": run_name,
        "config": {k: (v if not isinstance(v, (list, dict)) else str(v)) for k, v in asdict(config).items()},
        "dir": wdir,
        "tags": config.wandb_tags or ["nav", "bc"],
    }
    if mode == "offline":
        init_kw["mode"] = "offline"

    run = wandb.init(**init_kw)
    cfg_path = os.path.join(save_dir, "config_resolved.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, allow_unicode=True, default_flow_style=False)
    return run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_decay_param_groups(
    model: nn.Module,
    lr: float,
    lr_mult_pretrained: float,
    head_lr_mult: float = 1.0,
):
    """
    参数分组：任务头（cls/reg/shared）可用更高 lr；图像与文本骨干用 ``lr * lr_mult_pretrained``。
    """
    keyed: dict[tuple[float, float], list] = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        full = name.lower()
        wd = 0.0 if (len(p.shape) == 1 or name.endswith(".bias")) else 0.01
        if any(
            h in full
            for h in (
                "cls_head",
                "reg_head",
                "film_mlp",
                "shared_feature_layer",
                "spatial_proj",
                "text_proj",
            )
        ):
            mult = float(head_lr_mult)
        elif "image_encoder" in full or "text_encoder" in full:
            mult = float(lr_mult_pretrained)
        else:
            mult = 1.0
        keyed[(mult, wd)].append(p)

    groups = []
    for (mult, wd), params in keyed.items():
        groups.append({"params": params, "lr": lr * mult, "weight_decay": wd})
    if not groups:
        return [{"params": model.parameters(), "lr": lr, "weight_decay": 0.01}]
    return groups


def save_checkpoint(model, optimizer, scheduler, epoch, path: str, extras: Optional[Dict] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if extras:
        ckpt.update(extras)
    torch.save(ckpt, path)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    config: ModelConfig,
    logger: logging.Logger,
    epoch: int,
    global_step: int,
    wandb_run,
    rank: int = 0,
) -> tuple[float, float, float, int]:
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    accum = max(1, config.gradient_accumulation_steps)
    log_iv = max(1, config.log_interval)

    running = {"total": 0.0, "cls": 0.0, "reg": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{config.max_epochs} train",
        dynamic_ncols=True,
        disable=(rank != 0),
    )
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        use_amp = device.type == "cuda"
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch)
            losses = unwrap_model(model).compute_loss(outputs, batch)
            loss = losses["total_loss"] / accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        step_now = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader)
        if step_now:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        lt = losses["total_loss"].item()
        l_cls = losses.get("action_cls_loss", losses["action_loss"]).item()
        l_reg = losses.get("action_reg_loss", losses["value_loss"]).item()
        running["total"] += lt
        running["cls"] += l_cls
        running["reg"] += l_reg
        running["n"] += 1

        cur_lrs = scheduler.get_last_lr()
        lr_show = max(cur_lrs) if cur_lrs else 0.0
        pbar.set_postfix(
            loss=f"{lt:.4f}",
            cls=f"{l_cls:.4f}",
            reg=f"{l_reg:.4f}",
            lr=f"{lr_show:.2e}",
        )

        if wandb_run is not None and rank == 0 and global_step % log_iv == 0:
            try:
                wandb_run.log(
                    {
                        "train/step": global_step,
                        "train/loss_total": lt,
                        "train/loss_cls": l_cls,
                        "train/loss_reg": l_reg,
                        "train/loss_action": l_cls,
                        "train/loss_value": l_reg,
                        "train/lr": max(scheduler.get_last_lr()),
                        "train/epoch": epoch,
                    }
                )
            except Exception:
                pass

        if rank == 0 and batch_idx % log_iv == 0:
            logger.info(
                "Epoch %d batch %d/%d | loss %.6f | cls %.6f | reg %.6f | lr %.2e | step %d",
                epoch + 1,
                batch_idx,
                len(train_loader),
                lt,
                l_cls,
                l_reg,
                max(scheduler.get_last_lr()),
                global_step,
            )

    n = max(1, running["n"])
    return running["total"] / n, running["cls"] / n, running["reg"] / n, global_step


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    config: ModelConfig,
    ddp: bool = False,
    rank: int = 0,
) -> Dict[str, float]:
    model.eval()
    sum_loss = sum_cls = sum_reg = 0.0
    n_samples = 0
    correct = 0
    total_cls = 0
    for batch in tqdm(val_loader, desc="valid", dynamic_ncols=True, disable=(rank != 0)):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bs = int(batch["action"].shape[0])
        use_amp = device.type == "cuda"
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch)
            losses = unwrap_model(model).compute_loss(outputs, batch)
        l_cls = losses.get("action_cls_loss", losses["action_loss"]).item()
        l_reg = losses.get("action_reg_loss", losses["value_loss"]).item()
        sum_loss += losses["total_loss"].item() * bs
        sum_cls += l_cls * bs
        sum_reg += l_reg * bs
        n_samples += bs
        logits = outputs["action_logits"]
        pred = logits.argmax(dim=-1)
        correct += (pred == batch["action"]).sum().item()
        total_cls += batch["action"].numel()

    if ddp:
        t = torch.tensor(
            [sum_loss, sum_cls, sum_reg, n_samples, correct, total_cls],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_loss, sum_cls, sum_reg, n_samples, correct, total_cls = t.tolist()

    denom = max(1.0, float(n_samples))
    m_loss = sum_loss / denom
    m_cls = sum_cls / denom
    m_reg = sum_reg / denom
    return {
        "loss": m_loss,
        "action_cls_loss": m_cls,
        "action_reg_loss": m_reg,
        "action_loss": m_cls,
        "value_loss": m_reg,
        "action_acc": correct / max(1.0, float(total_cls)),
    }


def set_image_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    m = unwrap_model(model)
    for p in m.image_encoder.parameters():
        p.requires_grad = trainable


def main_inner(
    config: ModelConfig,
    data_root: str,
    save_dir: str,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    find_unused_parameters: bool = True,
) -> None:
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    if ddp:
        dist.barrier()
    logger = setup_logging(save_dir, rank=rank)
    set_seed(config.seed)

    if ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info("Device: %s", device)
        logger.info(
            "Distributed: %s | rank %d / world_size %d",
            "on" if ddp else "off",
            rank,
            world_size,
        )
        logger.info("data_root=%s", os.path.abspath(data_root))
        logger.info("save_dir=%s", os.path.abspath(save_dir))
        if hf_local_files_only():
            logger.info(
                "Hugging Face: 离线模式已开启（TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE），仅从本地加载 BERT。"
            )
        else:
            logger.info(
                "Hugging Face: 将尝试联网拉取模型；若失败可设置 HF_ENDPOINT=https://hf-mirror.com "
                "或先缓存后设 TRANSFORMERS_OFFLINE=1"
            )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    wandb_run = setup_wandb(config, save_dir) if rank == 0 else None

    drop_last = config.gradient_accumulation_steps > 1
    root_abs = os.path.abspath(data_root)
    if getattr(config, "value_normalize", True) and rank == 0:
        vn = ensure_value_norm_file(root_abs)
        logger.info(
            "value 归一化: %s | %s",
            os.path.join(root_abs, "value_norm.json"),
            json.dumps(vn, ensure_ascii=False)[:500],
        )
    if ddp:
        dist.barrier()

    ow = int(getattr(config, "obs_window", getattr(config, "history_len", 2)))
    if ddp:
        train_ds = ImitationDataset(
            root_abs,
            "train.json",
            tokenizer_name=config.text_encoder_name,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            value_normalize=getattr(config, "value_normalize", True),
            obs_window=ow,
        )
        val_ds = ImitationDataset(
            root_abs,
            "val.json",
            tokenizer_name=config.text_encoder_name,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            value_normalize=getattr(config, "value_normalize", True),
            obs_window=ow,
        )
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=config.seed)
        val_sampler = DistributedSampler(val_ds, shuffle=False, seed=config.seed)
        train_loader = get_dataloader(
            dataset=train_ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            tokenizer_name=config.text_encoder_name,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            pin_memory=config.pin_memory and device.type == "cuda",
            drop_last=drop_last,
            sampler=train_sampler,
            value_normalize=getattr(config, "value_normalize", True),
            obs_window=ow,
        )
        val_loader = get_dataloader(
            dataset=val_ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            tokenizer_name=config.text_encoder_name,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            pin_memory=config.pin_memory and device.type == "cuda",
            drop_last=False,
            sampler=val_sampler,
            value_normalize=getattr(config, "value_normalize", True),
            obs_window=ow,
        )
    else:
        train_loader = get_dataloader(
            data_root,
            "train.json",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            tokenizer_name=config.text_encoder_name,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            pin_memory=config.pin_memory and device.type == "cuda",
            drop_last=drop_last,
            value_normalize=getattr(config, "value_normalize", True),
            obs_window=ow,
        )
        val_loader = get_dataloader(
            data_root,
            "val.json",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            tokenizer_name=config.text_encoder_name,
            max_text_length=config.max_text_length,
            image_size=config.image_size,
            pin_memory=config.pin_memory and device.type == "cuda",
            drop_last=False,
            value_normalize=getattr(config, "value_normalize", True),
            obs_window=ow,
        )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    if rank == 0:
        logger.info(
            "Data | train samples=%d val=%d | batches train=%d val=%d | per-GPU batch_size=%d | obs_window=%d use_film=%s",
            n_train,
            n_val,
            len(train_loader),
            len(val_loader),
            config.batch_size,
            ow,
            getattr(config, "use_film", True),
        )

    model = BCTransformer(config).to(device)
    if ddp:
        # 默认可 True：compute_loss 中 cls/reg 分支或 reg_mask 可能导致部分参数在某些 step 无梯度
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )

    accum = max(1, config.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_training_steps = max(1, steps_per_epoch * config.max_epochs)
    warmup_steps = max(1, int(total_training_steps * config.warmup_ratio))

    param_groups = _split_decay_param_groups(
        model,
        lr=config.learning_rate,
        lr_mult_pretrained=float(config.lr_pretrained_mult),
        head_lr_mult=float(getattr(config, "head_lr_mult", 1.0)),
    )
    beta2 = float(getattr(config, "adam_beta2", 0.999))
    optimizer = AdamW(param_groups, betas=(0.9, beta2), eps=1e-8)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    try:
        scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    except TypeError:
        scaler = GradScaler(enabled=(device.type == "cuda"))

    global_step = 0
    best_val = float("inf")

    for epoch in range(config.max_epochs):
        if epoch < config.freeze_image_encoder_epochs:
            set_image_backbone_trainable(model, False)
        else:
            set_image_backbone_trainable(model, True)

        avg_t, avg_cls, avg_reg, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            logger,
            epoch,
            global_step,
            wandb_run,
            rank=rank,
        )

        metrics = validate(model, val_loader, device, config, ddp=ddp, rank=rank)

        if rank == 0:
            logger.info(
                "Epoch %d/%d summary | train loss %.6f (cls %.6f reg %.6f) | val loss %.6f | val cls %.6f | val reg %.6f | val acc %.4f | step %d",
                epoch + 1,
                config.max_epochs,
                avg_t,
                avg_cls,
                avg_reg,
                metrics["loss"],
                metrics["action_cls_loss"],
                metrics["action_reg_loss"],
                metrics["action_acc"],
                global_step,
            )

        if wandb_run is not None and rank == 0:
            try:
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "val/loss": metrics["loss"],
                        "val/loss_cls": metrics["action_cls_loss"],
                        "val/loss_reg": metrics["action_reg_loss"],
                        "val/action_loss": metrics["action_loss"],
                        "val/value_loss": metrics["value_loss"],
                        "val/action_acc": metrics["action_acc"],
                        "train/epoch_avg_loss": avg_t,
                        "train/epoch_avg_cls": avg_cls,
                        "train/epoch_avg_reg": avg_reg,
                    }
                )
            except Exception:
                pass

        if rank == 0 and metrics["loss"] < best_val:
            best_val = metrics["loss"]
            path = os.path.join(save_dir, "best.pt")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path,
                extras={"best_val_loss": best_val, "val_action_acc": metrics["action_acc"]},
            )
            logger.info("Saved new best checkpoint -> %s (val_loss=%.6f)", path, best_val)

        if rank == 0 and (epoch + 1) % max(1, config.save_every_epochs) == 0:
            path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, path)

    if wandb_run is not None and rank == 0:
        try:
            wandb_run.finish()
        except Exception:
            pass


def main() -> None:
    cli = parse_args()
    ddp, rank, world_size, local_rank = setup_distributed()
    config = ModelConfig()
    apply_cli_to_config(config, cli)
    try:
        main_inner(
            config,
            cli.data_root,
            cli.save_dir,
            ddp=ddp,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            find_unused_parameters=not cli.ddp_no_find_unused,
        )
    finally:
        if ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
