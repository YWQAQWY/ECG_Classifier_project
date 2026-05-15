"""
最大均值差异 (Maximum Mean Discrepancy, MMD) 实现
===================================================

理论背景:
    MMD 是一种衡量两个概率分布差异的非参数方法。
    在 RKHS (Reproducing Kernel Hilbert Space) 中,
    通过比较两个分布的核嵌入 (kernel embeddings) 来测量差异。

    对于源分布 P 和目标分布 Q:
    MMD²[P, Q] = E_{x,x'~P}[k(x,x')] + E_{y,y'~Q}[k(y,y')] - 2*E_{x~P, y~Q}[k(x,y)]

    其中 k(·,·) 是核函数 (如高斯RBF核)。

为什么使用MMD进行域对齐?
    1. MMD 是非参数方法，不假设数据分布形式
    2. MMD 在 RKHS 中度量分布距离，具有理论保证
    3. 当 MMD=0 时，P 和 Q 在 RKHS 中完全相同
    4. 通过最小化 MMD，可使源域和目标域特征分布趋同

为什么使用多核 (multi-kernel) MMD?
    1. 单一核带宽只能捕获一种尺度的分布差异
    2. 多核覆盖多个带宽，能捕获不同尺度的分布特征
    3. 论文中的 MK-MMD 使用多个 RBF 核的组合
    4. 不同带宽的核对应不同"分辨率"的分布比较

论文公式 (8) — GDD:
    L_gdd = (1/n²) Σᵢⱼ k(φ(x_sⁱ), φ(x_sʲ))
          + (1/m²) Σᵢⱼ k(φ(x_tⁱ), φ(x_tʲ))
          - (2/nm) Σᵢⱼ k(φ(x_sⁱ), φ(x_tʲ))
"""

import torch
import torch.nn as nn


class MultiKernelMMD(nn.Module):
    """
    多核RBF MMD 计算模块。

    使用多个不同带宽的高斯RBF核:
        k(x, y) = exp(-||x - y||² / (2 * σ²))

    带宽选择:
        使用多个 σ 值 (或等价地用多个 γ = 1/(2σ²))
        每个核分别计算 MMD，然后求和/平均

    数值稳定性处理:
        1. 使用 L2 距离而非直接计算内积
        2. 对距离进行截断防止exp溢出
        3. 归一化带宽以确保跨特征维度的稳定性
    """

    def __init__(self, bandwidths: list = None):
        """
        参数:
            bandwidths: 高斯核带宽列表 (σ值)
                       默认 [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
                       这些值会乘以特征的中位成对距离进行缩放
        """
        super(MultiKernelMMD, self).__init__()
        self.bandwidths = bandwidths if bandwidths is not None else \
            [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor,
                         sigma: float) -> torch.Tensor:
        """
        计算高斯RBF核矩阵的均值。

        k_matrix[i,j] = exp(-||x[i] - y[j]||² / (2 * σ²))

        返回: mean(k_matrix) — 即公式中的 (1/nm) Σᵢⱼ k(xⁱ, yʲ)

        数值稳定性技巧:
        - ||x-y||² / (2σ²) 可能非常大导致 exp 溢出
        - 使用 torch.clamp 限制最大值
        - 当 σ 很小时，距离项很大，exp→0
        - 当 σ 很大时，距离项很小，exp→1
        """
        # 成对 L2 距离: ||x - y||²
        # (batch_x, dim) @ (dim, batch_y) = (batch_x, batch_y)
        # 注意: 使用 expand方法: ||x||² - 2xy^T + ||y||²
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (N, 1)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)  # (M, 1)
        xy = torch.mm(x, y.t())  # (N, M)

        # ||x - y||² = ||x||² + ||y||² - 2xy
        distances = xx + yy.t() - 2 * xy  # (N, M)

        # 数值保护: 防止负值 (浮点误差)
        distances = torch.clamp(distances, min=0.0)

        # 高斯核: exp(-distance / (2 * sigma²))
        # 除以 (2 * sigma²) 核 bandwidth
        gamma = 1.0 / (2.0 * sigma ** 2)
        kernel_matrix = torch.exp(-gamma * distances)

        # 截断极小值，防止后续计算出现 nan
        kernel_matrix = torch.clamp(kernel_matrix, min=1e-12, max=1.0)

        return torch.mean(kernel_matrix)

    def _mmd_single_kernel(self, x_s: torch.Tensor, x_t: torch.Tensor,
                           sigma: float) -> torch.Tensor:
        """
        使用单个核带宽计算MMD²。

        公式 (8):
            MMD² = E[k(s,s)] + E[k(t,t)] - 2*E[k(s,t)]

        参数:
            x_s: 源域特征 (N, dim)
            x_t: 目标域特征 (M, dim)
            sigma: 核带宽

        返回:
            mmd2: 单核M²值 (标量)
        """
        # E[k(s,s)]: 源域内的核期望
        k_ss = self._gaussian_kernel(x_s, x_s, sigma)

        # E[k(t,t)]: 目标域内的核期望
        k_tt = self._gaussian_kernel(x_t, x_t, sigma)

        # E[k(s,t)]: 源域到目标域的核期望
        k_st = self._gaussian_kernel(x_s, x_t, sigma)

        # MMD² = k_ss + k_tt - 2*k_st
        mmd2 = k_ss + k_tt - 2.0 * k_st

        return mmd2

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor,
                return_scaled: bool = True) -> torch.Tensor:
        """
        计算多核MMD。

        首先用中位成对距离缩放带宽 (自适应归一化),
        然后对每个核计算MMD并求均值。

        为什么需要自适应缩放?
        - 不同epoch/不同batch的特征范数不同
        - 固定sigma可能在一种尺度上工作而在另一种失效
        - 用中位距离缩放使MMD对特征scale不敏感

        参数:
            x_s: 源域特征 (N, dim) — 编码器输出
            x_t: 目标域特征 (M, dim) — 编码器输出
            return_scaled: 是否用中位距离缩放带宽

        返回:
            mmd: 多核MMD值 (标量) — 越接近0表示分布越相似
        """
        # 计算中位成对距离 (用于自适应带宽缩放)
        if return_scaled:
            # 合并源和目标计算中位距离
            x_all = torch.cat([x_s, x_t], dim=0)  # (N+M, dim)
            # 避免计算全部成对距离 (O(n²)), 对大数据采样
            if x_all.size(0) > 200:
                indices = torch.randperm(x_all.size(0))[:200]
                x_sample = x_all[indices]
            else:
                x_sample = x_all

            xx = torch.sum(x_sample ** 2, dim=1, keepdim=True)
            xy = torch.mm(x_sample, x_sample.t())
            dists = xx + xx.t() - 2 * xy
            dists = torch.clamp(dists, min=0.0)

            # 中位距离
            median_dist = torch.median(dists[dists > 0]) if (dists > 0).any() \
                else torch.tensor(1.0, device=x_s.device)
            median_dist = torch.clamp(median_dist, min=1e-6)
        else:
            median_dist = 1.0

        # 对每个带宽计算MMD并求均值
        mmd_values = []
        for sigma in self.bandwidths:
            # 用中位距离缩放带宽
            scaled_sigma = sigma * torch.sqrt(median_dist)
            mmd_val = self._mmd_single_kernel(x_s, x_t, scaled_sigma)
            mmd_values.append(mmd_val)

        # 多核MMD = 各核MMD的均值
        multi_mmd = torch.stack(mmd_values).mean()

        return multi_mmd


def mmd_rbf(x_s: torch.Tensor, x_t: torch.Tensor,
            bandwidths: list = None) -> torch.Tensor:
    """
    便捷函数: 计算多核RBF MMD。

    用法:
        loss_gdd = mmd_rbf(f_source, f_target)

    参数:
        x_s: 源域特征 (N, dim)
        x_t: 目标域特征 (M, dim)
        bandwidths: 带宽列表

    返回:
        mmd: MMD值
    """
    mmd_module = MultiKernelMMD(bandwidths=bandwidths)
    return mmd_module(x_s, x_t)
