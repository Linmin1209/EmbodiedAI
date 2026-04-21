"""
细粒度导航 BCTransformer：多时间步三视角 RGB + 指令；空间特征 7×7（49 token/视角）+ 时间嵌入；
Transformer 融合后经 Shared GLU 再分流为离散动作分类与连续动作幅值回归。
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel
from torchvision.models import resnet50, ResNet50_Weights

from hf_compat import hf_from_pretrained_kwargs


def _value_norm_to_bin_indices(v_norm: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    归一化 value（v_raw / max_physical）与「k 个 step」一致时，v_norm ≈ k / num_bins。
    将标量映射到 bin 索引 0..num_bins-1（对应 k=1..num_bins）。
    """
    k = (v_norm * num_bins).round().long().clamp(1, num_bins)
    return k - 1


def _soft_bin_targets(
    indices: torch.Tensor,
    num_bins: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """在 bin 索引上按高斯模糊（sigma 以 bin 为单位），用于软标签 CE。"""
    if sigma <= 0:
        return F.one_hot(indices, num_bins).to(dtype)
    grid = torch.arange(num_bins, device=device, dtype=dtype)
    diff = indices.float().unsqueeze(-1) - grid.unsqueeze(0)
    w = torch.exp(-0.5 * (diff / sigma) ** 2)
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return w


_step_bins_value_warned = False


def _warn_if_step_bins_value_likely_raw(
    vt: torch.Tensor, value_normalize: bool
) -> None:
    """
    `_value_norm_to_bin_indices` 需要 v_norm≈v_raw/scale（通常 [0,1]）。
    若误传物理量（如 0.02），round(v_norm*num_bins) 会错档；在典型 batch 上 max≈0.2 时给出一次性提示。
    """
    global _step_bins_value_warned
    if _step_bins_value_warned:
        return
    mx = float(vt.detach().max().item())
    # 归一化后满量程约 1.0；物理量上界常≈0.2（与 max_action_value 一致）
    if mx > 0.35:
        return
    _step_bins_value_warned = True
    hints = (
        "Dataset 中开启 value_normalize，或按 value_norm.json 的 scale 自行除后再入 batch。"
        if value_normalize
        else "当前 value_normalize=False，请仍保证传入的是 v/scale 的归一化标量（非物理米/秒步长）。"
    )
    warnings.warn(
        "step_bins：batch[\"value\"] max=%.4f 可能仍是物理幅值而非 v/scale。"
        " _value_norm_to_bin_indices 假定 v_norm∈[0,1] 量级；%s"
        % (mx, hints),
        UserWarning,
        stacklevel=3,
    )


class SpatialSoftmaxCoord2d(nn.Module):
    """
    在 ``spatial_proj`` 后对 H×W 做空间 Softmax，得到软 Argmax 坐标 ``(x,y)``（归一化到 [-1,1]），
    再投影到 ``hidden`` 维并 **逐位置** 加回特征图，作为关键点空间约束。
    """

    def __init__(self, hid: int, spatial_hw: tuple[int, int] = (7, 7)):
        super().__init__()
        self.spatial_attn = nn.Conv2d(hid, 1, kernel_size=1, bias=True)
        self.coord_proj = nn.Linear(2, hid)
        self.temperature = nn.Parameter(torch.ones(1))
        H, W = spatial_hw
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, W, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xx.reshape(-1), persistent=False)
        self.register_buffer("grid_y", yy.reshape(-1), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, C, H, W)
        n = z.shape[0]
        logits = self.spatial_attn(z).view(n, -1)
        logits = logits / self.temperature.clamp(min=1e-4)
        attn = F.softmax(logits, dim=-1)
        gx = self.grid_x.unsqueeze(0).expand(n, -1)
        gy = self.grid_y.unsqueeze(0).expand(n, -1)
        ex = (attn * gx).sum(dim=-1, keepdim=True)
        ey = (attn * gy).sum(dim=-1, keepdim=True)
        coords = torch.cat([ex, ey], dim=-1)
        feat = self.coord_proj(coords).view(n, -1, 1, 1)
        return z + feat


class ClsHeadResidual(nn.Module):
    """
    分类头：一层 MLP（Linear + LayerNorm + GELU）后与共享表征 **残差相加**，再投影到动作类；
    比单层线性更稳，残差减轻表征扭曲。
    """

    def __init__(self, hid: int, num_actions: int, dropout: float):
        super().__init__()
        self.l1 = nn.Linear(hid, hid)
        self.ln = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l1(x)
        h = self.ln(h)
        h = F.gelu(h)
        h = self.drop(h)
        return self.fc(x + h)


class SharedGLU(nn.Module):
    """
    门控线性单元（GLU）：``a ⊗ σ(b)``，提升共享表征在分类 / 回归前的解耦与表达力。
    """

    def __init__(self, hid: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(hid)
        self.up = nn.Linear(hid, hid * 2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hid, hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        u = self.up(h)
        a, b = u.chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        h = self.dropout(h)
        return self.out_proj(h)


class BCTransformer(nn.Module):
    """
    - 输入图像：``left_img`` / ``center_img`` / ``right_img`` 可选形状
      ``(B, C, H, W)`` 或 ``(B, T, C, H, W)``，其中 ``T = obs_window``（默认与 ``ModelConfig.obs_window`` 一致）；缺省时间维时视为 ``T=1``。
    - 骨干：ResNet50 去掉 GAP 与 FC，保留 ``7×7`` 空间图；``Conv2d(1×1)`` 将 2048→hidden_size；
      经 **SpatialSoftmaxCoord2d** 注入软坐标特征；可选 **FiLM**（指令向量调制 spatial 特征图）后再展平为 token。
    - 可学习 ``temp_embed`` / ``view_embed``：紧凑广播相加以省显存。
    - 文本：BERT 输出 masked mean-pool 为 1 个 token，并与视觉 token 拼接进 Transformer。
    - ``SharedGLU`` 后经 ``cls_head`` / ``reg_head``。
    """

    _spatial_hw = (7, 7)  # 224×224 输入下 ResNet 末层空间尺寸

    def __init__(self, config):
        super().__init__()
        self.config = config
        hid = config.hidden_size
        self.history_len = int(
            getattr(config, "obs_window", getattr(config, "history_len", 1))
        )
        self.use_film = bool(getattr(config, "use_film", True))

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self._backbone_out_c = 2048

        if getattr(config, "freeze_image_backbone", False):
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.spatial_proj = nn.Conv2d(self._backbone_out_c, hid, kernel_size=1, bias=True)
        self.spatial_softmax = SpatialSoftmaxCoord2d(hid, self._spatial_hw)

        self.temp_embed = nn.Parameter(torch.zeros(1, self.history_len, 1, 1, hid))
        nn.init.trunc_normal_(self.temp_embed, std=0.02)

        self.view_embed = nn.Parameter(torch.zeros(1, 1, 3, 1, hid))
        nn.init.trunc_normal_(self.view_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hid))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.text_encoder = BertModel.from_pretrained(
            config.text_encoder_name,
            **hf_from_pretrained_kwargs(),
        )
        if getattr(config, "freeze_text_encoder", False):
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        tdim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(tdim, hid),
            nn.LayerNorm(hid),
        )

        if self.use_film:
            self.film_mlp = nn.Sequential(
                nn.Linear(hid, hid * 2),
                nn.LayerNorm(hid * 2),
                nn.GELU(),
                nn.Linear(hid * 2, hid * 2),
            )
        else:
            self.film_mlp = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hid,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.pre_ln = nn.LayerNorm(hid)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)

        self.shared_feature_layer = SharedGLU(hid, float(config.dropout))

        hd = float(getattr(config, "head_dropout", config.dropout))
        self.cls_head = ClsHeadResidual(hid, config.num_actions, hd)
        self.value_head_mode = getattr(config, "value_head_mode", "continuous")
        if self.value_head_mode == "step_bins":
            step = float(getattr(config, "value_step_physical", 0.02))
            vmax = float(getattr(config, "value_max_physical", 0.2))
            self._num_step_bins = max(1, int(round(vmax / step)))
            reg_out = self._num_step_bins
        else:
            self._num_step_bins = 0
            reg_out = 1
        self.reg_head = nn.Sequential(
            nn.Dropout(hd),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(hd),
            nn.Linear(hid, reg_out),
        )

        self._init_task_modules()

    def _init_task_modules(self) -> None:
        """对非 ImageNet/BERT 预训练部分做标准初始化。"""
        for name, m in self.named_modules():
            if "image_encoder" in name or "text_encoder" in name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _ensure_temporal(
        self, x: torch.Tensor, name: str
    ) -> torch.Tensor:
        if x.dim() == 4:
            return x.unsqueeze(1)
        if x.dim() == 5:
            return x
        raise ValueError(f"{name} 期望 4D 或 5D，得到 shape={tuple(x.shape)}")

    def _add_temp_view_embed_inplace(self, z: torch.Tensor, t: int) -> torch.Tensor:
        """
        将 ``temp_embed`` / ``view_embed`` 加到 ``z``（B,T,3,49,Hid）上。
        先合并为小 ``bias = te + ve``（形状 ``(1,t,3,1,hid)``），再一次性 ``add_``，
        避免 ``z + te + ve`` 的中间广播张量；在 contiguous 张量上原地加以节省峰值显存。
        """
        te = self.temp_embed[:, :t]
        ve = self.view_embed[:, :, :3]
        bias = te + ve
        z = z.contiguous()
        z.add_(bias)
        return z

    def _film_modulate_spatial(
        self,
        z: torch.Tensor,
        text_vec: torch.Tensor,
        b: int,
        t: int,
    ) -> torch.Tensor:
        """FiLM：``z * γ + β``，γ 初值贴近 1（经 tanh 与 film_gamma_scale）。"""
        assert self.film_mlp is not None
        gb = self.film_mlp(text_vec)
        g, beta = gb.chunk(2, dim=-1)
        scale = float(getattr(self.config, "film_gamma_scale", 0.1))
        g = 1.0 + scale * torch.tanh(g)
        n = z.shape[0]
        assert n == b * t * 3
        batch_idx = torch.arange(n, device=z.device, dtype=torch.long) // (t * 3)
        g = g[batch_idx]
        beta = beta[batch_idx]
        return z * g.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)

    def encode_spatial_tokens(
        self,
        left: torch.Tensor,
        center: torch.Tensor,
        right: torch.Tensor,
        text_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, c, h, w = left.shape
        assert (
            center.shape == left.shape and right.shape == left.shape
        ), "三视角形状须一致"
        assert t == self.history_len, (
            f"输入时间维 T={t} 须等于模型 obs_window={self.history_len}（请对齐 Dataset.obs_window）"
        )

        x = torch.stack([left, center, right], dim=2)
        x = x.reshape(b * t * 3, c, h, w)

        feat = self.image_encoder(x)
        z = self.spatial_proj(feat)
        z = self.spatial_softmax(z)
        if self.use_film and text_vec is not None:
            z = self._film_modulate_spatial(z, text_vec, b, t)
        _, hid, hh, ww = z.shape
        assert (hh, ww) == self._spatial_hw, f"期望空间 {self._spatial_hw}，得到 {(hh, ww)}"

        z = z.flatten(2).transpose(1, 2)
        z = z.reshape(b, t, 3, 49, hid)

        z = self._add_temp_view_embed_inplace(z, t)

        z = z.reshape(b, t * 3 * 49, hid)
        return z

    def forward(self, batch):
        left = self._ensure_temporal(batch["left_img"], "left_img")
        center = self._ensure_temporal(batch["center_img"], "center_img")
        right = self._ensure_temporal(batch["right_img"], "right_img")

        out = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = self.text_proj(out.last_hidden_state)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        text_vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        text_token = text_vec.unsqueeze(1)

        tv = text_vec if self.use_film else None
        img_tokens = self.encode_spatial_tokens(left, center, right, text_vec=tv)

        b = left.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, img_tokens, text_token], dim=1)

        seq = self.pre_ln(seq)
        seq = self.transformer(seq)

        pooled = seq[:, 0]
        shared = self.shared_feature_layer(pooled) + pooled

        action_logits = self.cls_head(shared)
        value_out = self.reg_head(shared)
        if self.value_head_mode == "step_bins":
            logits = value_out
            probs = F.softmax(logits, dim=-1)
            centers = torch.arange(
                1,
                self._num_step_bins + 1,
                device=logits.device,
                dtype=logits.dtype,
            ) / float(self._num_step_bins)
            value_pred = (probs * centers.unsqueeze(0)).sum(dim=-1, keepdim=True)
            return {
                "action_logits": action_logits,
                "value_pred": value_pred,
                "value_bin_logits": logits,
            }
        return {
            "action_logits": action_logits,
            "value_pred": value_out,
        }

    def compute_loss(self, outputs, batch):
        """
        ``total_loss = cls_loss_weight * CE + reg_loss_weight * SmoothL1``（continuous）
        或 ``+ reg_loss_weight_step_bins * value_bin CE``（step_bins）。
        """
        cfg = self.config
        ls = float(getattr(cfg, "label_smoothing", 0.0))
        action_cls_loss = F.cross_entropy(
            outputs["action_logits"],
            batch["action"],
            label_smoothing=ls,
        )

        mode = getattr(cfg, "value_head_mode", "continuous")
        if mode == "step_bins":
            logits = outputs["value_bin_logits"]
            vt = batch["value"].to(logits.dtype)
            num_bins = logits.shape[-1]
            _warn_if_step_bins_value_likely_raw(
                vt,
                bool(getattr(cfg, "value_normalize", True)),
            )
            tgt_idx = _value_norm_to_bin_indices(vt, num_bins)
            sig = float(getattr(cfg, "value_bin_fuzzy_sigma", 0.0))
            if sig > 0:
                soft = _soft_bin_targets(
                    tgt_idx, num_bins, sig, logits.device, logits.dtype
                )
                logp = F.log_softmax(logits, dim=-1)
                per_reg = -(soft * logp).sum(dim=-1)
            else:
                per_reg = F.cross_entropy(logits, tgt_idx, reduction="none")
        else:
            vp = outputs["value_pred"].squeeze(-1)
            vt = batch["value"].to(vp.dtype)
            beta = float(getattr(cfg, "huber_beta", 0.1))
            per_reg = F.smooth_l1_loss(vp, vt, beta=beta, reduction="none")
        per_reg = torch.nan_to_num(per_reg, nan=0.0, posinf=0.0, neginf=0.0)

        bsz = per_reg.shape[0]
        device = per_reg.device
        reg_mask = torch.ones(bsz, dtype=torch.bool, device=device)

        skip_ids = getattr(cfg, "reg_skip_action_ids", ()) or ()
        if len(skip_ids) > 0:
            skip_t = torch.tensor(
                list(skip_ids), device=device, dtype=batch["action"].dtype
            )
            reg_mask = reg_mask & (~torch.isin(batch["action"], skip_t))

        rm = batch.get("reg_mask")
        if rm is not None:
            reg_mask = reg_mask & rm.to(device).bool().view(-1)
        sr = batch.get("skip_reg")
        if sr is not None:
            reg_mask = reg_mask & (~sr.to(device).bool().view(-1))

        eps = float(getattr(cfg, "reg_loss_eps", 1e-6))
        mask_f = reg_mask.float()
        denom = mask_f.sum() + eps
        action_reg_loss = (per_reg * mask_f).sum() / denom

        w_cls = float(getattr(cfg, "cls_loss_weight", getattr(cfg, "action_loss_weight", 1.0)))
        if mode == "step_bins":
            w_reg = float(
                getattr(cfg, "reg_loss_weight_step_bins", 1.0)
            )
        else:
            w_reg = float(
                getattr(cfg, "reg_loss_weight", getattr(cfg, "value_loss_weight", 5.0))
            )

        total_loss = w_cls * action_cls_loss + w_reg * action_reg_loss
        z = action_cls_loss * 0.0

        return {
            "total_loss": total_loss,
            "action_cls_loss": action_cls_loss,
            "action_reg_loss": action_reg_loss,
            "action_loss": action_cls_loss,
            "value_loss": action_reg_loss,
            "cls_loss": action_cls_loss,
            "reg_loss": action_reg_loss,
            "pos_loss": z,
            "yaw_loss": z,
        }
