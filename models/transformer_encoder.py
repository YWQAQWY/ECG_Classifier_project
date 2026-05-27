"""
共享 Transformer 编码器 F
=========================

这是整个网络的端到端主干。Source 和 Target 共享同一个编码器。

输入: (batch, 30 tokens, 4 dims)
  - 30 tokens = 30个EEG通道
  - 4 dims = 每个通道的4个频段DE值 (theta, alpha, beta, gamma)

架构:
  Input Projection (4 → d_model) + Positional Encoding
    → N × TransformerEncoderLayer (Self-Attention + FFN)
    → Global Pooling (mean)
    → output z: (batch, d_model)

为什么用 Transformer?
  - 自注意力可以捕获不同EEG通道之间的关系
  - 不受限于卷积的局部感受野限制
  - 30个通道 × 4个频段的排列更适合序列建模
"""

import torch
import torch.nn as nn
import math
import warnings
# Pre-LN Transformer 不支持 nested tensor, 禁用并抑制 warning
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')


class PositionalEncoding(nn.Module):
    """
    正弦位置编码 — 给30个通道添加位置信息。

    为什么需要位置编码?
    - Transformer 的自注意力是置换不变的 (permutation invariant)
    - EEG 通道有明确的空间拓扑关系 (相邻电极测量相关信号)
    - 位置编码让模型知道每个 token 对应哪个物理电极位置
    """

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, n_tokens, d_model)
        返回:
            x + positional_encoding
        """
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerEncoder(nn.Module):
    """
    共享 Transformer 编码器 F。

    输入输出:
        x: (batch, n_tokens=30, token_dim=4)
        → z: (batch, d_model=64)

    Source 和 Target 都经过同一个 F:
        z_s = F(x_s),  z_t = F(x_t)

    这个 z 将同时送入 C (分类), GRL+D (域对抗), Con (对比学习) 三个并列分支。
    """

    def __init__(self, token_dim: int = 4, n_tokens: int = 30,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        """
        参数:
            token_dim: 每个 token 的原始维度 (4个频段DE)
            n_tokens: token 数量 (30通道)
            d_model: Transformer 隐层维度
            n_heads: 多头注意力头数
            n_layers: Transformer 层数
            dim_feedforward: FFN 隐层维度
            dropout: dropout 比率
        """
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_tokens = n_tokens

        # 输入投影: 4 → d_model
        self.input_proj = nn.Linear(token_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_tokens + 5, dropout=dropout)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,       # 使用 (batch, seq, dim) 格式
            norm_first=True,        # Pre-LN (训练更稳定)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 全局平均池化: (batch, n_tokens, d_model) → (batch, d_model)
        # 将30个通道的隐向量聚合为一个全局特征 z
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, 30, 4) — 30个通道 × 4个频段的DE值

        返回:
            z: (batch, d_model) — 域不变 + 情绪判别 + 结构化 的特征表示

        数据流:
            x (batch, 30, 4)
              → input_proj → (batch, 30, d_model)
              → pos_encoder
              → transformer layers → (batch, 30, d_model)
              → global pool → (batch, d_model, 1)
              → squeeze → (batch, d_model)
        """
        # 输入投影
        x = self.input_proj(x)  # (batch, 30, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer 编码
        x = self.transformer(x)  # (batch, 30, d_model)

        # 全局池化 → 聚合30个通道的信息 → 得到一个全局特征 z
        x = x.transpose(1, 2)        # (batch, d_model, 30)
        x = self.global_pool(x)       # (batch, d_model, 1)
        z = x.squeeze(-1)             # (batch, d_model)

        return z
