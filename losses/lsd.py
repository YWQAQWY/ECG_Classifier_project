"""
局部子域差异 (Local Subdomain Discrepancy, LSD) 实现
=====================================================

理论核心 (论文 Section III-B):

LSD 测量跨域的类别级 (子域级) 分布差异。
与GDD (只看整体分布) 不同, LSD关注每个情绪类别内部的分布对齐。

核心思想:
    "同类拉近, 异类推远"
    - 同类 (within-class): 最小化源域类别c与目标域类别c之间的MMD
    - 异类 (cross-class): 最大化源域类别c与目标域类别c'之间的MMD

LSD公式 (论文公式7):
    L_lsd = (1/M) * Σ_c d_{c,c}  -  (1/(M(M-1))) * Σ_c Σ_{c'≠c} d_{c,c'}
          = mean(同类MMD) - mean(异类MMD)

其中 d_{c1,c2} 是两个类别之间的MMD距离 (论文公式2):
    d_{c1,c2} = d1 + d2 - 2*d3
    d1 = same-class MMD within source
    d2 = same-class MMD within target
    d3 = cross-domain same-class MMD

分类讨论:
    - 同类MMD (c1=c2): 对齐同类跨域分布 → 我们希望它小
    - 异类MMD (c1≠c2): 比较不同类的跨域分布 → 我们希望它大

为什么LSD能提升分类性能?
    1. GDD只保证整体分布对齐, 但可能导致类别混合
       (如源域neutral和目标域positive被错误地对齐到一起)
    2. LSD在类别层面约束: 同类必须对齐, 异类必须分离
    3. 这保留了特征的类别判别性 (discriminability)
    4. 好的特征应该: 同类聚类 (不论域) + 异类分离 (不论域)

为什么伪标签用于target?
    - target数据没有标签, 无法知道每个样本属于哪个类别
    - 使用模型当前的预测作为"伪标签"
    - 伪标签 ≠ 真实标签, 但在训练后期越来越准确
    - 这是EM算法的思想: 用当前模型估计隐变量, 再更新模型

类别缺失处理:
    - 某batch可能只有一种类别的样本 (特别是batch较小时)
    - 此时无法计算异类MMD, 自动跳过该类别对
    - 这也保证了LSD计算的数值稳定性
"""

import torch
import torch.nn as nn
from .mmd import MultiKernelMMD


class LocalSubdomainDiscrepancy(nn.Module):
    """
    LSD 损失模块。

    计算流程:
        Step 1: 按类别分组
            Fs_c = Fs[Ys == c]  (源域类别c的特征)
            Ft_c = Ft[Yt_hat == c]  (目标域类别c的伪标签特征)

        Step 2: 计算同类距离
            d_{c,c} = MMD(Fs_c, Ft_c)  ← 同类对齐, 希望小

        Step 3: 计算异类距离
            d_{c,c'} = MMD(Fs_c, Ft_c')  ← 异类分离, 希望大

        Step 4: 组合
            L_lsd = mean(同类MMD) - mean(异类MMD)
    """

    def __init__(self, n_classes: int = 2, bandwidths: list = None,
                 min_samples_per_class: int = 3,
                 confidence_threshold: float = 0.7):
        """
        参数:
            n_classes: 类别数 (2: neutral/positive)
            bandwidths: MMD核带宽列表
            min_samples_per_class: 每类最少样本数 (低于此数跳过)
            confidence_threshold: 伪标签置信度阈值 (低于此阈值的样本不用于LSD)
        """
        super(LocalSubdomainDiscrepancy, self).__init__()
        self.n_classes = n_classes
        self.min_samples_per_class = min_samples_per_class
        self.confidence_threshold = confidence_threshold
        self.mmd = MultiKernelMMD(bandwidths=bandwidths)

    def _filter_by_confidence(self, features: torch.Tensor,
                              labels: torch.Tensor,
                              probabilities: torch.Tensor) -> tuple:
        """
        根据预测置信度过滤不可靠的伪标签。

        为什么需要置信度过滤?
        - 早期训练的伪标签噪声很大
        - 使用错误的伪标签计算LSD会误导对齐方向
        - 只保留模型"有信心"的样本来计算LSD
        - 低置信度的样本通常靠近决策边界, 标签不可靠

        参数:
            features: 特征 (N, dim)
            labels: 伪标签 (N,)
            probabilities: 预测概率 (N, n_classes) — softmax后的值

        返回:
            filtered_features, filtered_labels
        """
        # 取预测概率的最大值作为置信度
        max_probs = probabilities.max(dim=1)[0]  # (N,)
        # 高于阈值的样本保留
        mask = max_probs >= self.confidence_threshold
        if mask.sum() < self.min_samples_per_class * self.n_classes:
            # 如果过滤后样本太少, 不过滤 (退化到全量)
            return features, labels
        return features[mask], labels[mask]

    def forward(self, fs: torch.Tensor, ft: torch.Tensor,
                ys: torch.Tensor, yt_hat: torch.Tensor,
                ps: torch.Tensor = None, pt: torch.Tensor = None) -> torch.Tensor:
        """
        计算 LSD 损失。

        参数:
            fs: 源域特征 (N_s, dim) — 编码器输出
            ft: 目标域特征 (N_t, dim) — 编码器输出
            ys: 源域真实标签 (N_s,) — 值为 0 或 1
            yt_hat: 目标域伪标签 (N_t,) — argmax(Pt) 的结果
            ps: 源域预测概率 (N_s, n_classes) — 用于置信度过滤 (可选)
            pt: 目标域预测概率 (N_t, n_classes) — 用于置信度过滤 (可选)

        返回:
            lsd_loss: LSD损失值 (标量)
                > 0: 同类MMD > 异类MMD (不理想, 需要继续优化)
                ≈ 0 or < 0: 同类MMD <= 异类MMD (理想的类感知对齐)

        注意:
            - 当某类别在源域或目标域中样本不足时, 跳过该类别
            - 这保证了LSD计算的鲁棒性
        """
        # ---- 可选: 置信度过滤 ----
        if pt is not None and self.confidence_threshold > 0:
            ft, yt_hat = self._filter_by_confidence(ft, yt_hat, pt)
            if ft.size(0) < self.min_samples_per_class:
                # 过滤后样本太少, 返回一个小常数避免训练崩溃
                return torch.tensor(0.0, device=fs.device, requires_grad=True)

        # ---- 同类MMD: d_{c,c} ----
        within_class_mmd = []

        for c in range(self.n_classes):
            # 提取源域中类别为c的特征
            fs_c = fs[ys == c]  # (n_s_c, dim)
            # 提取目标域中伪标签为c的特征
            ft_c = ft[yt_hat == c]  # (n_t_c, dim)

            # 检查样本数是否足够 (太少则跳过)
            if fs_c.size(0) < self.min_samples_per_class:
                continue
            if ft_c.size(0) < self.min_samples_per_class:
                continue

            # 计算同类MMD: MMD(Fs_c, Ft_c)
            # 论文公式 (2) 中 c1=c2 的情况
            d_cc = self.mmd(fs_c, ft_c)
            within_class_mmd.append(d_cc)

        # ---- 异类MMD: d_{c,c'} (c ≠ c') ----
        cross_class_mmd = []

        for c in range(self.n_classes):
            for c2 in range(self.n_classes):
                if c == c2:
                    continue  # 跳过同类 (已在上面计算)

                # 提取源域类别c的特征
                fs_c = fs[ys == c]
                # 提取目标域类别c'的伪标签特征
                ft_c2 = ft[yt_hat == c2]

                if fs_c.size(0) < self.min_samples_per_class:
                    continue
                if ft_c2.size(0) < self.min_samples_per_class:
                    continue

                # 计算异类MMD: MMD(Fs_c, Ft_c')
                # 论文公式 (2) 中 c1≠c2 的情况
                d_cc2 = self.mmd(fs_c, ft_c2)
                cross_class_mmd.append(d_cc2)

        # ---- 组合: L_lsd = mean(同类) - mean(异类) ----
        # 为什么减去异类MMD?
        #   我们希望同类MMD小 (对齐同类特征)
        #   我们希望异类MMD大 (分离异类特征)
        #   而总loss在最小化L_lsd
        #   所以: L_lsd = 同类 - 异类
        #   最小化L_lsd → 同类变小 + 异类变大
        if len(within_class_mmd) == 0:
            # 没有任何有效类别对, 返回0
            return torch.tensor(0.0, device=fs.device, requires_grad=True)

        mean_within = torch.stack(within_class_mmd).mean()
        if len(cross_class_mmd) == 0:
            mean_cross = torch.tensor(0.0, device=fs.device)
        else:
            mean_cross = torch.stack(cross_class_mmd).mean()

        lsd_loss = mean_within - mean_cross

        return lsd_loss


def lsd_loss(fs: torch.Tensor, ft: torch.Tensor,
             ys: torch.Tensor, yt_hat: torch.Tensor,
             n_classes: int = 2) -> torch.Tensor:
    """
    便捷函数: 计算LSD损失。

    用法:
        loss_lsd = lsd_loss(f_source, f_target, y_source, y_target_hat)

    参数:
        fs: 源域特征 (N_s, dim)
        ft: 目标域特征 (N_t, dim)
        ys: 源域真实标签 (N_s,)
        yt_hat: 目标域伪标签 (N_t,)
        n_classes: 类别数

    返回:
        loss: LSD损失
    """
    lsd_module = LocalSubdomainDiscrepancy(n_classes=n_classes)
    return lsd_module(fs, ft, ys, yt_hat)
