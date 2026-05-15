"""
EEGNet 编码器模型
================

基于论文: "EEGNet: A Compact Convolutional Neural Network for EEG-based
           Brain-Computer Interfaces" (Lawhern et al., 2018)

模型结构 (适配DE特征输入):

输入: (batch, 1, n_channels=30, n_bands=4)
  ↓
Block 1: 时间卷积 (Conv2D)
  - Conv2D(1, F1, (1, kernel_temporal))  → 每个通道独立做时间卷积
  - 输出: (batch, F1, 30, 4)  注: 频段维度可能因padding略有变化
  ↓
Block 2: 深度卷积 (DepthwiseConv2D)
  - DepthwiseConv2D(F1, D*F1, (channels, 1))  → 学习空间滤波器
  - 输出: (batch, D*F1, 1, T')
  ↓
Block 3: 可分离卷积 (SeparableConv2D)
  - SeparableConv2D(D*F1, F2, (1, 16))  → 时空特征整合
  - 输出: (batch, F2, 1, T'')
  ↓
Flatten → Linear(F2*T'', feature_dim=64)
  ↓  ← 这里输出 feature vector (用于GDD/LSD对齐)
  ↓
Classifier: Linear(64, n_classes=2)
  ↓
输出: (batch, 2)  — emotion prediction

为什么选择EEGNet而不是论文中的MLP?
- EEGNet专为EEG设计，用深度/可分离卷积捕获时空EEG特征
- 论文的MLP是针对SEED数据集 (62通道×5频段=310维已提取DE)
- 我们的DE特征 (30通道×4频段×时间窗口) 更适合卷积处理
- EEGNet参数少、训练快、泛化好，适合小样本跨被试场景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet 编码器 — 从DE特征中提取域不变的特征表示。

    输入维度变化 (以 window_size=250 为例):
        输入:  (batch, 1, 30, 4)
        时间卷积后: (batch, F1, 30, 4)
        深度卷积后: (batch, D*F1, 1, 4)
        可分离卷积后: (batch, F2, 1, 4)
        Flatten: (batch, F2*4)
        Linear: (batch, feature_dim)

    注意: 当频段维度较小(只有4)时，时间维度的卷积操作空间有限。
    因此我们对标准EEGNet做了适配调整。
    """

    def __init__(self, n_channels: int = 30, n_bands: int = 4,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 feature_dim: int = 64, dropout_rate: float = 0.5):
        """
        参数:
            n_channels: EEG通道数 (30)
            n_bands: 频段数 (4: theta, alpha, beta, gamma)
            F1: 第一个卷积层的滤波器数 (时间滤波器)
            D: 深度乘子 (控制每个通道的空间滤波器数量)
            F2: 可分离卷积的输出通道数
            feature_dim: 最终特征向量维度 (用于GDD/LSD对齐)
            dropout_rate: Dropout比率
        """
        super(EEGNet, self).__init__()

        self.n_channels = n_channels
        self.n_bands = n_bands
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.feature_dim = feature_dim

        # ============================================================
        # Block 1: 时间卷积
        # 对每个通道的频段向量做一维卷积 (跨频段的时间模式)
        #
        # Conv2D: (1, F1, kernel=(1, n_bands))
        # 输入: (batch, 1, channels, bands)
        # 输出: (batch, F1, channels, bands)
        # 这里用Conv2D的(1, n_bands)核，等价于对频段做全连接变换
        # 因为bands维度小(4)，实际上是对每个通道做频段特征融合
        # ============================================================
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, n_bands), padding=0)
        self.bn1 = nn.BatchNorm2d(F1)

        # ============================================================
        # Block 2: 深度卷积 (Depthwise Convolution)
        #
        # 对每个特征图做深度卷积 (跨通道的空间滤波)
        # 输入: (batch, F1, channels, 1)  ← 频段维度已被Block1压缩为1
        # 输出: (batch, D*F1, 1, 1)      ← 通道维度被压缩为1
        #
        # 为什么需要深度卷积?
        # - EEG不同通道之间存在空间关系 (相邻电极测量相似信号)
        # - 深度卷积为每个时间滤波器学习专门的空间滤波器
        # - 相比普通卷积，深度卷积参数更少，更不容易过拟合
        #
        # 维度变化详解:
        #   输入 (batch, F1, 30, 1)
        #   分成 F1 个 group, 每组有 D 个滤波器
        #   每个滤波器大小: (channels, 1) = (30, 1)
        #   输出: (batch, F1*D, 1, 1)
        # ============================================================
        self.depthwise_conv = nn.Conv2d(
            F1, D * F1,
            kernel_size=(n_channels, 1),
            groups=F1,  # groups=F1 使每个输入通道独立卷积
            padding=0
        )
        self.bn_depthwise = nn.BatchNorm2d(D * F1)

        # ============================================================
        # Block 3: 可分离卷积 (Separable Convolution)
        #
        # 可分离卷积 = Depthwise Conv + Pointwise Conv
        # 用于整合特征并进一步减少参数
        #
        # 输入: (batch, D*F1, 1, 1)
        # 输出: (batch, F2, 1, 1)
        #
        # 首先depthwise: kernel=(1, 1) 作用在空间维度上
        # 然后pointwise: 用1×1卷积将 D*F1 通道映射到 F2 通道
        # ============================================================
        self.separable_conv = nn.Conv2d(
            D * F1, F2,
            kernel_size=(1, 1),
            padding=0
        )
        self.bn_separable = nn.BatchNorm2d(F2)

        # ============================================================
        # 计算展平后的特征维度
        # 经过Block1→Block2→Block3后，空间维度变为 (1, 1)
        # 所以展平后: F2 * 1 * 1 = F2
        # ============================================================
        self._flatten_dim = F2

        # ============================================================
        # FC层: 将卷积特征映射到低维特征空间
        #
        # 这个特征空间是GDD和LSD对齐的目标空间
        # 论文中在这个空间计算MMD来对齐源域和目标域分布
        # ============================================================
        self.fc = nn.Linear(self._flatten_dim, feature_dim)
        self.bn_fc = nn.BatchNorm1d(feature_dim)

        # Dropout — 防止过拟合源域
        self.dropout = nn.Dropout(dropout_rate)

        # 激活函数
        self.elu = nn.ELU(inplace=True)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier初始化 + BatchNorm初始化为1/0。

        为什么需要良好的初始化?
        - 跨被试场景下训练数据有限, 不好的初始化会导致收敛困难
        - Xavier初始化适用于ELU激活函数
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 — 提取EEG特征。

        参数:
            x: 输入DE特征 (batch, 1, n_channels, n_bands)
               例如: (64, 1, 30, 4)

        返回:
            features: 特征向量 (batch, feature_dim)
               例如: (64, 64)
               这个特征向量用于:
               ① 输入分类器预测情绪
               ② 计算GDD (源域和目标域之间的MMD)
               ③ 计算LSD (类级别的MMD对齐)
        """
        # ---- Block 1: 时间/频段卷积 ----
        # 输入: (batch, 1, 30, 4)
        x = self.conv1(x)  # → (batch, F1, 30, 1)
        # 这里(n_bands=4)的维度被压缩为1, 因为kernel_size=(1,4)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout(x)
        # 输出: (batch, F1, 30, 1)

        # ---- Block 2: 深度卷积 (空间滤波) ----
        # 输入: (batch, F1, 30, 1)
        x = self.depthwise_conv(x)  # → (batch, D*F1, 1, 1)
        # channels维度(30)被压缩为1, 因为kernel_size=(30, 1)
        x = self.bn_depthwise(x)
        x = self.elu(x)
        x = self.dropout(x)
        # 输出: (batch, D*F1, 1, 1)

        # ---- Block 3: 可分离卷积 (特征整合) ----
        # 输入: (batch, D*F1, 1, 1)
        x = self.separable_conv(x)  # → (batch, F2, 1, 1)
        x = self.bn_separable(x)
        x = self.elu(x)
        x = self.dropout(x)
        # 输出: (batch, F2, 1, 1)

        # ---- Flatten ----
        x = x.view(x.size(0), -1)  # → (batch, F2)

        # ---- FC特征层 (GDD/LSD对齐的目标空间) ----
        features = self.fc(x)  # → (batch, feature_dim)
        features = self.bn_fc(features)
        features = self.elu(features)
        # 注意: 这里不对features做dropout, 因为GDD/LSD需要稳定的特征表示

        return features
