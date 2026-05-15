"""
MLP 编码器 (备选方案，与论文原文一致)
=====================================

论文原文使用4层MLP作为特征提取器:
    hidden_dims: [512, 128, 128, 64]
    每层后: BatchNorm + ReLU + Dropout

适用于: SEED/SEED-IV 数据集的预提取DE特征 (310维)
本项目中也可用于 DE特征展平后的输入 (30×4=120维)
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    多层感知机编码器。

    论文 Section IV-B:
    "We use a four-layer network, where the numbers of hidden nodes are
     512, 128, 128 and 64 in each layer."
    """

    def __init__(self, input_dim: int = 120, hidden_dims: list = None,
                 feature_dim: int = 64, dropout_rate: float = 0.5):
        """
        参数:
            input_dim: 输入特征维度 (30 channels × 4 bands = 120)
            hidden_dims: 隐藏层维度列表
            feature_dim: 最终特征维度
            dropout_rate: Dropout比率
        """
        super(MLPEncoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 128, 128, 64]

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        layers = []
        prev_dim = input_dim

        for i, hd in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hd

        # 最终特征层
        layers.append(nn.Linear(prev_dim, feature_dim))
        layers.append(nn.BatchNorm1d(feature_dim))
        layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入特征 (batch, input_dim)
               注意: 需要先展平 channels×bands 维度

        返回:
            features: 特征向量 (batch, feature_dim)
        """
        batch_size = x.size(0)
        # 展平: (batch, 1, 30, 4) → (batch, 120)
        x = x.view(batch_size, -1)
        return self.encoder(x)
