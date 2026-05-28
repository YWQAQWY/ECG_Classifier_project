"""
分支 D: 域判别器 (GRL + D)
===========================

与 C 端并列，接收 GRL(z) 后的特征，判断样本来自 Source 还是 Target。

GRL (梯度反转层):
    前向传播: GRL(z) = z (恒等映射，不改变特征)
    反向传播: 梯度 × (-λ) (反转梯度)

D (域判别器):
    GRL(z) → Linear → ReLU → Linear → 2 (source/target)

对抗训练机制:
    D 学习区分 source / target → L_domain ↓
    Transformer F 通过 GRL 学习混淆 D → 间接使 L_domain ↑
    最终达到均衡: F 输出 domain-invariant representation

目标: 不是让 D 分类准确，而是通过 D 与 GRL 的对抗，
      使 F 学习到源域和目标域通用的特征表示。
"""

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    """
    梯度反转层 — 对抗训练核心。

    前向: output = input (恒等)
    反向: grad_output × (-lambda) (反转梯度)

    为什么需要 GRL?
    - D 判别器要区分源域和目标域
    - 但 Transformer F 要学习域不变特征
    - GRL 在前向不改变特征，反向时将 D 的梯度反转
    - 因此 F 的更新方向与 D 的判别目标相反
    - F 学习 → 增大 D 的 loss → 混淆 D → 域不变特征
    """
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """
    域判别器 D — 分支 D。

    结构: z → GRL → Linear → ReLU → Dropout → Linear → 2
    输出: source=0, target=1 的二分类 logits
    """

    def __init__(self, input_dim: int = 64, hidden_dims: list = None,
                 n_domains: int = 2, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev, hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hd
        layers.append(nn.Linear(prev, n_domains))

        self.discriminator = nn.Sequential(*layers)

    def forward(self, z, grl_alpha=1.0):
        """
        参数:
            z: Transformer 输出特征 (batch, d_model)
            grl_alpha: GRL 梯度反转系数

        返回:
            domain_logits: (batch, 2) — source/target 分类 logits
        """
        reversed_z = grad_reverse(z, grl_alpha)
        return self.discriminator(reversed_z)
