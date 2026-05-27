"""
DANN 总损失模块
===============

组合三个并列分支的损失:

    L_total = λ_cls * L_cls + λ_domain * L_domain + λ_con * L_con

其中:
    L_cls    (分支C): CrossEntropy on source labeled data
    L_domain (分支D): CrossEntropy on domain classification (source vs target)
    L_con    (分支Con): 对比学习损失 (类结构约束)

参考导师代码: total_loss = loss_cls + 0.1 * loss_domain + constraint_loss
"""

import torch
import torch.nn as nn
from .contrastive_loss import ContrastiveLoss


class DANNTotalLoss(nn.Module):
    """
    DANN 总损失。

    组合三个分支:
      C:  情绪分类 CE loss (source only)
      D:  域对抗 CE loss (source + target, 通过 GRL 反转)
      Con: 对比学习 loss (source + high-conf target pseudo)

    Source 和 Target 使用说明:
      - L_cls: 只使用 source 域 (有真实标签)
      - L_domain: 使用 source (domain=0) + target (domain=1)
      - L_con: 使用 source (真实标签) + target pseudo (高置信伪标签)
    """

    def __init__(self, cls_weight: float = 1.0, domain_weight: float = 0.1,
                 contrastive_weight: float = 0.05, temperature: float = 0.1):
        super().__init__()
        self.cls_weight = cls_weight
        self.domain_weight = domain_weight
        self.contrastive_weight = contrastive_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.con_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, cls_logits_s: torch.Tensor, y_s: torch.Tensor,
                domain_logits_s: torch.Tensor, domain_logits_t: torch.Tensor,
                z_s_norm: torch.Tensor, z_t_norm: torch.Tensor = None,
                y_t_pseudo: torch.Tensor = None) -> dict:
        """
        计算总损失。

        参数:
            cls_logits_s: 源域分类 logits (N_s, n_classes)
            y_s: 源域真实标签 (N_s,)
            domain_logits_s: 源域域判别 logits (N_s, 2)
            domain_logits_t: 目标域域判别 logits (N_t, 2)
            z_s_norm: 源域 L2 归一化特征 (N_s, d_model)
            z_t_norm: 目标域 L2 归一化特征 (N_t, d_model) — 可选
            y_t_pseudo: 目标域伪标签 (N_t,) — 可选, 用于对比学习

        返回:
            {'total': ..., 'cls': ..., 'domain': ..., 'contrastive': ...}
        """
        # ---- L_cls: 情绪分类 (仅 source 真实标签) ----
        loss_cls = self.ce_loss(cls_logits_s, y_s)

        # ---- L_domain: 域对抗 ----
        # Source domain label = 0, Target domain label = 1
        domain_labels_s = torch.zeros(domain_logits_s.size(0), dtype=torch.long,
                                       device=domain_logits_s.device)
        domain_labels_t = torch.ones(domain_logits_t.size(0), dtype=torch.long,
                                      device=domain_logits_t.device)

        loss_domain_s = self.ce_loss(domain_logits_s, domain_labels_s)
        loss_domain_t = self.ce_loss(domain_logits_t, domain_labels_t)
        loss_domain = (loss_domain_s + loss_domain_t) / 2.0

        # ---- L_con: 对比学习 ----
        loss_con = torch.tensor(0.0, device=cls_logits_s.device, requires_grad=True)

        # Source 域内部对比 (用真实标签)
        if z_s_norm.size(0) >= 2:
            loss_con = self.con_loss(z_s_norm, y_s)

        # Source + Target 联合对比 (如果有伪标签)
        if z_t_norm is not None and y_t_pseudo is not None and z_t_norm.size(0) >= 2:
            z_all = torch.cat([z_s_norm, z_t_norm], dim=0)
            y_all = torch.cat([y_s, y_t_pseudo], dim=0)
            loss_con_cross = self.con_loss(z_all, y_all)
            loss_con = (loss_con + loss_con_cross) / 2.0

        # ---- 总损失 ----
        loss_total = (self.cls_weight * loss_cls +
                      self.domain_weight * loss_domain +
                      self.contrastive_weight * loss_con)

        return {
            'total': loss_total,
            'cls': loss_cls,
            'domain': loss_domain,
            'contrastive': loss_con,
        }
