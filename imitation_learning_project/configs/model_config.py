from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# 本机已下载的 BERT（tokenizer 与 ``text_encoder_name`` 共用）；可被 yaml / 环境覆盖
DEFAULT_TEXT_ENCODER_NAME = "/data2/linmin/model/hf_cache/bert-base-uncased"


@dataclass
class ModelConfig:
    # 模型基本配置（略减深度/FF 以加快收敛与迭代；768 维与 BERT 对齐）
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 3
    ff_dim: int = 2048
    dropout: float = 0.08
    attn_dropout: float = 0.08
    head_dropout: float = 0.05  # 分类/回归头 dropout，略低于 trunk 利于早期拟合

    # 编码器
    text_encoder_name: str = DEFAULT_TEXT_ENCODER_NAME
    freeze_image_backbone: bool = False
    freeze_text_encoder: bool = False
    # 前若干 epoch 冻结图像骨干（便于稳定文本分支）
    freeze_image_encoder_epochs: int = 5

    # 时空输入（图像序列）
    history_len: int = 2  # 兼容旧字段；模型优先读 obs_window
    obs_window: int = 2  # 视觉历史帧数 T（含当前帧）；Dataset 在同一 demo 内取 idx-(T-1)..idx，段首不足则重复首帧
    use_film: bool = True  # 用语义向量对 spatial 特征图做 FiLM（γ·x+β）；False 则与旧行为一致（仅 Transformer 融合文本）
    film_gamma_scale: float = 0.1  # FiLM 中 γ = 1 + film_gamma_scale * tanh(·)，避免初期破坏预训练视觉特征

    # 输出
    num_actions: int = 3

    # 损失（混合头：分类 + 回归）
    # continuous：对归一化 value 做 SmoothL1（量级常远小于 CE，需 reg_loss_weight 放大，默认 2.0）。
    # step_bins：value 支路为 CrossEntropy，量级与动作 CE 同阶（约 0.5～2）；请用 reg_loss_weight_step_bins（默认 1.0）。
    # 若仍用很大的 reg_loss_weight（如 5.0）会压过分类梯度。
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 2.0
    reg_loss_weight_step_bins: float = 1.0  # 仅 value_head_mode=step_bins 时乘在 value_bin CE 上
    label_smoothing: float = 0.02  # 略降，加快脱离随机基线
    # value 预测头：continuous=标量回归；step_bins=按 value_step 分档分类（需与 build_nav_dataset 的 step/max 一致）
    value_head_mode: str = "continuous"  # continuous | step_bins
    value_step_physical: float = 0.02
    value_max_physical: float = 0.2  # 单段上限，bin 数 = max/step（默认 10 档，对应 1～10 个 0.02）
    value_bin_fuzzy_sigma: float = 0.0  # >0 时在相邻 bin 上扩散软标签（以 bin 索引为单位的「模糊」宽度，如 0.5～1.0）
    huber_beta: float = 0.1  # 仅 continuous 时 SmoothL1 beta
    reg_loss_eps: float = 1e-6  # 回归 mask 平均时分母平滑，避免全 False 时数值问题
    # 这些 action 类别不计算回归损失（如「停止」「原地等待」）；需在 Dataset 中为对应类别 id
    reg_skip_action_ids: Tuple[int, ...] = field(default_factory=tuple)
    # 兼容旧字段（未设置时仍可读）
    action_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    value_loss_type: str = "huber"  # 已由 compute_loss 固定为 smooth_l1，保留仅兼容

    # 优化
    # 略提高默认 LR：纯 3e-5 + 长 warmup 时分类 CE 易长期卡在 ~ln(num_actions)（随机水平）
    learning_rate: float = 1.2e-4
    lr_pretrained_mult: float = 0.55  # 略提高，视觉需与任务头同步学习
    head_lr_mult: float = 4.0  # cls/reg/shared 相对 base lr
    adam_beta2: float = 0.95  # 略小于 0.999，常见于 Transformer 微调，加快有效更新
    weight_decay: float = 0.01
    max_grad_norm: float = 5.0
    # SharedGLU / SpatialSoftmax 等随机初始化较重：过短 warmup（如 0.01）易在前几十 step 出现 loss 尖峰；默认 0.05 更稳；若前期仍尖峰可略增，若收敛偏慢再试减小。
    warmup_ratio: float = 0.05
    max_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1  # 有效 batch 已较大时可 1，加快 step；显存不足改回 2

    # 数据（value 归一化参数见 data_root/value_norm.json，由 build_nav_dataset 或脚本生成）
    # step_bins 时 compute_loss 假定 batch["value"] 为归一化标量（如 v_raw/scale，scale≈max_action_value）；勿传未除 scale 的物理值。
    value_normalize: bool = True  # True：Dataset 对 value 做归一化；与 step_bins 的 bin 标号一致
    image_size: int = 224
    max_text_length: int = 128
    num_workers: int = 4
    pin_memory: bool = True

    # 训练杂项
    seed: int = 42
    log_interval: int = 20
    val_interval_batches: Optional[int] = None  # None 表示只在 epoch 末验证
    save_every_epochs: int = 1

    # wandb
    wandb_project: str = "imitation_nav_bc"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"  # online | offline | disabled
    wandb_tags: List[str] = field(default_factory=list)
