"""
分支 Con: 对比学习头
====================

与 C 端、D 端并列，直接接收 Transformer 输出特征 z。

目标: 拉近同类样本、推远异类样本。
  - 同类: 情绪标签相同 (positive-positive, neutral-neutral)
  - 异类: 情绪标签不同 (positive-neutral)

相似度用内积:
    s_ij = z_i^T z_j

若对 z 做 L2 normalization, 内积等价于 cosine similarity。
同类 s_ij 应大, 异类 s_ij 应小。

为什么需要 Con 端?
  - DANN 对齐源域和目标域时, 可能把不同类别错误压到一起
  - GRL 只保证 domain-invariant, 不保证 class-separable
  - 对比学习在 latent space 中显式约束: 同类聚合、异类分离
  - 防止域的全局对齐导致类别混合

这个分支不引入额外参数, 直接对 z 计算对比损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """
    对比学习头 — 分支 Con。

    无额外参数。对 z 做 L2 normalization 后计算内积相似度。
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize 特征，输出归一化后的向量用于对比学习。

        参数:
            z: Transformer 输出 (batch, d_model)
        返回:
            z_norm: L2归一化特征 (batch, d_model)
        """
        return F.normalize(z, p=2, dim=1)
