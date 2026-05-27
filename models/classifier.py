"""
分支 C: 情绪分类器
=================

接收 Transformer 输出特征 z，预测情绪类别 (neutral=0, positive=1)。

该分支只使用 Source 域的情绪标签计算监督分类损失:
    L_cls = CE(C(z_s), y_s)

作用: 保证 Transformer 输出特征中保留足够的情绪判别信息，
      避免模型只做域对齐而丢失情绪分类能力。

结构 (参考导师代码 classifier):
    z (d_model) → Linear → ReLU → Dropout → Linear → 2 classes
"""

import torch.nn as nn


class EmotionClassifier(nn.Module):
    """情绪分类头 — 分支 C"""

    def __init__(self, input_dim: int = 64, hidden_dims: list = None,
                 n_classes: int = 2, dropout: float = 0.3):
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
        layers.append(nn.Linear(prev, n_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, z):
        """
        参数:
            z: Transformer 输出特征 (batch, d_model)
        返回:
            logits: 类别 logits (batch, n_classes)
        """
        return self.classifier(z)
