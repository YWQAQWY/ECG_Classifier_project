"""
情绪分类器
========

线性分类器 head，将编码器提取的特征映射到情绪类别。

论文中的分类器:
    "The emotion classifier is a linear layer whose nodes correspond to
     the emotion categories."
"""

import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    """
    线性分类器 — 将特征映射到情绪类别概率。

    输入: 特征向量 (batch, feature_dim)
    输出: 类别logits (batch, n_classes)

    为什么使用简单的线性分类器?
    - 特征提取能力由编码器负责
    - 线性分类器参数少, 不易过拟合
    - 论文明确使用 "a linear layer"
    - 简单的分类器使域对齐效果更容易归因于特征层面
    """

    def __init__(self, feature_dim: int = 64, n_classes: int = 2,
                 dropout_rate: float = 0.3):
        """
        参数:
            feature_dim: 编码器输出的特征维度 (默认64)
            n_classes: 类别数 (2: neutral/positive)
            dropout_rate: Dropout比率
        """
        super(EmotionClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, n_classes)
        )

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            features: 特征向量 (batch, feature_dim) — 编码器输出

        返回:
            logits: 未归一化的类别分数 (batch, n_classes)
                    使用 CrossEntropyLoss 时不需要提前做 softmax
        """
        return self.classifier(features)
