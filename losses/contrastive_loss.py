"""
对比学习损失 (Inner Product / InfoNCE 风格)
=============================================

相似度: s_ij = z_i^T z_j (L2 normalized → cosine similarity)

损失设计:
  对每个 anchor 样本 z_i:
    正样本 = 同情绪类别的其他样本
    负样本 = 不同情绪类别的样本

  L_con = - (1/|P|) Σ_{j∈P(i)} log( exp(s_ij / τ) / Σ_{k} exp(s_ik / τ) )

其中 τ 是温度参数 (temperature), 控制 softmax 的锐度:
  - τ 小 → 更关注 hard positives/negatives → 特征分布更紧凑
  - τ 大 → 更平滑 → 训练更稳定

为什么用对比学习增强 DANN?
  - GRL 做域对齐但不保证类别分离
  - 对比学习显式约束 latent space 的类别结构
  - 同类接近、异类远离 → 分类边界更清晰
  - 与 DANN 互补: DANN 消除域差异, Con 建立类别结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE 风格对比损失，使用内积相似度。

    只使用有标签的样本 (source + 高置信度 target pseudo-label)。
    """

    def __init__(self, temperature: float = 0.1):
        """
        参数:
            temperature: 温度参数 τ
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失。

        参数:
            z: L2归一化特征 (batch, d_model)
            labels: 标签 (batch,) — 包含 source 真实标签 + target 伪标签

        返回:
            loss: 对比损失 (标量)
        """
        batch_size = z.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # ---- 内积相似度矩阵 ----
        # s[i,j] = z_i^T z_j (已 L2 normalized, 等价于 cosine)
        similarity = torch.mm(z, z.t())  # (batch, batch)

        # 数值稳定: 除以温度并 clamp 防止 exp 溢出
        similarity = similarity / self.temperature
        similarity = torch.clamp(similarity, min=-50.0, max=50.0)

        # ---- 构造正负样本 mask ----
        # 正样本: 同类标签 (不包括自己)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()  # (batch, batch)
        # 排除对角线 (自己和自己)
        pos_mask = pos_mask - torch.eye(batch_size, device=z.device)

        # 如果某样本没有正样本对, 跳过
        n_pos = pos_mask.sum(dim=1)  # (batch,)
        valid = n_pos > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # ---- InfoNCE Loss ----
        # exp(s) (already divided by temperature and clamped above)
        exp_sim = torch.exp(similarity)

        # 分子: Σ_{j∈P(i)} exp(s_ij / τ)
        numerator = (exp_sim * pos_mask).sum(dim=1)  # (batch,)

        # 分母: Σ_{k≠i} exp(s_ik / τ)  (所有非自身的样本)
        # 排除对角线
        neg_mask = 1.0 - torch.eye(batch_size, device=z.device)
        denominator = (exp_sim * neg_mask).sum(dim=1)  # (batch,)

        # -log(分子/分母)
        loss_per_sample = -torch.log(numerator[valid] / (denominator[valid] + 1e-8) + 1e-8)

        return loss_per_sample.mean()
