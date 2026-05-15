"""
DDA 总损失模块 — 动态域适应 (Dynamic Domain Adaptation) 的完整损失函数
=====================================================================

论文公式 (9) (12):
    L_align = α * L_gdd + (1-α) * L_lsd      (公式9)
    L_total = L_ce + β * L_align              (公式12)

其中:
    α: 动态平衡因子 (公式10)
       α = exp(-epoch + 100) / (1 + exp(-epoch + 100))   (sigmoid衰减)
       或简单线性: α = 1 - epoch/N

    β: 对齐损失权重超参数

动态 α 机制 (论文灵魂):
========================

训练初期 (α≈1):
    L_align ≈ L_gdd
    此时主要优化全局域对齐。
    为什么? → 初始伪标签不可靠，用LSD会引入噪声。
    GDD不需要标签，可以安全地进行整体分布对齐。

训练后期 (α≈0):
    L_align ≈ L_lsd
    此时主要优化局部子域对齐。
    为什么? → 经过GDD对齐后，伪标签逐渐可靠。
    此时LSD能够进行精细的类别级对齐。

这种从"粗到细"的策略是DDA的核心创新:
    GDD: 粗对齐 → 减小域间整体差异
    LSD: 细对齐 → 在类别层面拉近同类、推开异类

关于 β 的调参建议:
    - β 太小: 域对齐效果弱, 模型退化为CE-only
    - β 太大: CE被忽视, 分类边界学不好
    - 建议: 0.05 ~ 0.5, 默认 0.1
    - 可根据L_gdd和L_lsd的scale动态调整

损失对参数 θ 的影响分析:
    - CE → 影响分类器和编码器 (学习类别边界)
    - GDD → 主要影响编码器 (使源/目标特征分布接近)
    - LSD → 主要影响编码器 (使同类特征跨域聚类, 异类特征跨域分离)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mmd import MultiKernelMMD
from .lsd import LocalSubdomainDiscrepancy


class DDALoss(nn.Module):
    """
    DDA 总损失模块。

    封装了 CE + GDD + LSD + Dynamic α 的完整计算。
    """

    def __init__(self, n_classes: int = 2, beta: float = 0.1,
                 bandwidths: list = None, alpha_schedule: str = "linear",
                 min_samples_per_class: int = 3,
                 confidence_threshold: float = 0.7):
        """
        参数:
            n_classes: 类别数
            beta: 对齐损失权重 (平衡CE和域对齐)
            bandwidths: MMD核带宽列表
            alpha_schedule: α衰减策略 ("linear" 或 "sigmoid")
            min_samples_per_class: LSD每类最少样本数
            confidence_threshold: LSD伪标签置信度阈值
        """
        super(DDALoss, self).__init__()
        self.n_classes = n_classes
        self.beta = beta
        self.alpha_schedule = alpha_schedule

        # GDD: 全局域差异 (基于MMD)
        self.gdd = MultiKernelMMD(bandwidths=bandwidths)

        # LSD: 局部子域差异 (类条件MMD)
        self.lsd = LocalSubdomainDiscrepancy(
            n_classes=n_classes,
            bandwidths=bandwidths,
            min_samples_per_class=min_samples_per_class,
            confidence_threshold=confidence_threshold
        )

        # CE: 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_alpha(self, epoch: int, max_epoch: int) -> float:
        """
        计算动态平衡因子 α。

        训练初期 α≈1 → 主要优化GDD
        训练后期 α≈0 → 主要优化LSD

        两种策略:
        1. Linear (推荐, 更稳定):
           α = max(0, 1 - epoch/max_epoch)
           - 从1线性衰减到0
           - 简单直观, 训练稳定

        2. Sigmoid (论文原始, 公式10):
           α = exp(-epoch + 100) / (1 + exp(-epoch + 100))
           - S型曲线衰减
           - 论文中在epoch=100附近快速过渡
           - 注意: 对于不同的max_epoch需要调整偏移量

        参数:
            epoch: 当前epoch (从0开始)
            max_epoch: 总epoch数

        返回:
            alpha: 动态平衡因子 (0~1)
        """
        if self.alpha_schedule == "linear":
            # 线性衰减: 从1到0
            alpha = max(0.0, 1.0 - epoch / max_epoch)
        elif self.alpha_schedule == "sigmoid":
            # 论文公式10: sigmoid衰减
            # 调整为通用形式, 在epoch=0时 ≈ 1, epoch=max_epoch时 ≈ 0
            # 映射: 0 → large_positive, max_epoch → large_negative
            center = max_epoch * 0.5  # 在50% epoch时 α=0.5
            shift = center - epoch  # 正→负
            # 缩放使两端更极端
            scaled_shift = shift / max_epoch * 100  # 归一化+放大
            alpha = torch.sigmoid(torch.tensor(scaled_shift)).item()
        else:
            raise ValueError(f"不支持的alpha_schedule: {self.alpha_schedule}")

        return alpha

    def forward(self, fs: torch.Tensor, ft: torch.Tensor,
                logits_s: torch.Tensor, logits_t: torch.Tensor,
                ys: torch.Tensor, epoch: int, max_epoch: int) -> dict:
        """
        计算 DDA 总损失。

        完整的训练流程 (每batch执行):

        Step 1: 输入 Xs, Xt
        Step 2: 编码器 → Fs, Ft
        Step 3: 分类器 → Ps(logits_s), Pt(logits_t)
        Step 4: 伪标签 Yt_hat = argmax(Pt)
        Step 5: 计算 L_ce (源域有标签分类)
        Step 6: 计算 L_gdd (MMD对齐整体分布)
        Step 7: 计算 L_lsd (类条件MMD, 使用伪标签)
        Step 8: 计算 α (动态权重)
        Step 9: L_total = L_ce + β * (α*L_gdd + (1-α)*L_lsd)

        参数:
            fs: 源域特征 (N_s, dim)
            ft: 目标域特征 (N_t, dim)
            logits_s: 源域分类logits (N_s, n_classes)
            logits_t: 目标域分类logits (N_t, n_classes)
            ys: 源域真实标签 (N_s,)
            epoch: 当前epoch
            max_epoch: 总epoch数

        返回:
            loss_dict: {
                'total': 总损失 (用于反向传播),
                'ce': CE损失,
                'gdd': GDD损失,
                'lsd': LSD损失,
                'alpha': 当前α值,
                'yt_hat': 目标域伪标签 (用于评估)
            }
        """
        # ---- Step 5: CE Loss (仅源域有真实标签) ----
        # CrossEntropyLoss 内置了 log_softmax + NLLLoss
        # 论文公式 (11):
        #   L_ce = -(1/n) Σᵢ yᵢ log P_θ(ŷᵢ|xᵢ)
        loss_ce = self.ce_loss(logits_s, ys)

        # ---- Step 6: GDD Loss (全局域分布对齐) ----
        # 论文公式 (8):
        #   L_gdd = MMD(Fs, Ft)
        loss_gdd = self.gdd(fs, ft)

        # ---- Step 4 (前置): 生成伪标签 ----
        # Yt_hat = argmax(Pt)
        # 伪标签不是优化变量! 它们随 θ 更新而变化
        # θ → features → predictions → pseudo labels
        probs_t = F.softmax(logits_t, dim=1)  # (N_t, n_classes)
        yt_hat = torch.argmax(probs_t, dim=1)  # (N_t,)

        # ---- Step 7: LSD Loss (局部子域对齐) ----
        # 论文公式 (7):
        #   L_lsd = mean(同类MMD) - mean(异类MMD)
        loss_lsd = self.lsd(
            fs, ft, ys, yt_hat,
            ps=F.softmax(logits_s, dim=1),
            pt=probs_t
        )

        # ---- Step 8: 动态 α ----
        alpha = self.compute_alpha(epoch, max_epoch)

        # ---- Step 9: 总 Loss ----
        # 论文公式 (9)(12):
        #   L_align = α * L_gdd + (1-α) * L_lsd
        #   L_total = L_ce + β * L_align
        loss_align = alpha * loss_gdd + (1.0 - alpha) * loss_lsd
        loss_total = loss_ce + self.beta * loss_align

        return {
            'total': loss_total,
            'ce': loss_ce,
            'gdd': loss_gdd,
            'lsd': loss_lsd,
            'align': loss_align,
            'alpha': alpha,
            'yt_hat': yt_hat,
        }
