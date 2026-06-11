"""
DANN + Transformer 训练器
=========================

训练流程:

每 batch:
  Source B → Transformer F → z_s
  Target B → Transformer F → z_t  (共享同一 F!)

  z_s → C → L_cls             (情绪分类, source only)
  z_s → GRL → D → L_dom_s     (域对抗)
  z_t → GRL → D → L_dom_t     (域对抗)
  z_s + z_t → Con → L_con     (对比学习, 类结构约束)

  L_total = λ1*L_cls + λ2*L_domain + λ3*L_con

注意: 集成伪标签已在数据预处理阶段完成 (data/dataset.py)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader

from models.transformer_encoder import TransformerEncoder
from models.classifier import EmotionClassifier
from models.domain_discriminator import DomainDiscriminator
from models.contrastive_head import ContrastiveHead
from losses.dann_loss import DANNTotalLoss
from utils.logger import Logger


class DANNTrainer:
    """
    DANN + Transformer 训练器。

    架构总结:
        ┌─────────────────────┐
        │ Shared Transformer F│  ← Source + Target 共享
        └────────┬────────────┘
                 │ z
        ┌────────┼────────┐
        ▼        ▼        ▼
       C()   GRL+D()    Con()
       CE    Domain    Contrastive
    """

    def __init__(self, encoder: TransformerEncoder, classifier: EmotionClassifier,
                 domain_disc: DomainDiscriminator, contrastive_head: ContrastiveHead,
                 config: dict, logger: Logger):
        self.encoder = encoder
        self.classifier = classifier
        self.domain_disc = domain_disc
        self.contrastive_head = contrastive_head
        self.config = config
        self.logger = logger

        # 设备
        device_str = config.get('training', {}).get('device', 'cuda')
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"设备: {self.device}")

        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.domain_disc.to(self.device)
        self.contrastive_head.to(self.device)

        # 优化器
        train_cfg = config.get('training', {})
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.classifier.parameters()) +
            list(self.domain_disc.parameters()),
            lr=train_cfg.get('learning_rate', 0.0005),
            weight_decay=train_cfg.get('weight_decay', 0.0001),
        )

        # 学习率调度
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_cfg.get('lr_step_size', 10),
            gamma=train_cfg.get('lr_gamma', 0.99),
        )

        # DANN 损失
        lw = config.get('loss_weights', {})
        self.criterion = DANNTotalLoss(
            cls_weight=lw.get('cls', 1.0),
            domain_weight=lw.get('domain', 0.1),
            contrastive_weight=lw.get('contrastive', 0.05),
            temperature=config.get('contrastive', {}).get('temperature', 0.1),
        )

        # 训练参数
        self.epochs = train_cfg.get('epochs', 100)
        self.eval_interval = config.get('experiment', {}).get('eval_interval', 1)
        self.save_dir = config.get('experiment', {}).get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        # 训练历史
        self.history = {'epoch': [], 'loss_cls': [], 'loss_domain': [],
                        'loss_con': [], 'val_acc': []}

    @torch.no_grad()
    def _evaluate_target_acc(self, tgt_loader: DataLoader) -> float:
        """评估目标域准确率"""
        self.encoder.eval()
        self.classifier.eval()
        correct, total = 0, 0
        for x, y, _, _ in tgt_loader:
            x, y = x.to(self.device), y.to(self.device)
            z = self.encoder(x)
            logits = self.classifier(z)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        self.encoder.train()
        self.classifier.train()
        return correct / total if total > 0 else 0.0

    def train_epoch(self, src_loader: DataLoader, tgt_loader: DataLoader,
                    epoch: int) -> dict:
        """训练一个 epoch"""
        self.encoder.train()
        self.classifier.train()
        self.domain_disc.train()

        ep_cls, ep_dom, ep_con, ep_total = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        tgt_iter = iter(tgt_loader)

        for src_batch in src_loader:
            # ---- 获取数据 ----
            x_s, y_s, _, _ = src_batch
            x_s, y_s = x_s.to(self.device), y_s.to(self.device)

            try:
                x_t, y_t, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_t, y_t, _, _ = next(tgt_iter)
            x_t = x_t.to(self.device)

            # ---- Forward: 共享 Transformer F ----
            z_s = self.encoder(x_s)  # (N_s, d_model)
            z_t = self.encoder(x_t)  # (N_t, d_model)

            # ---- 分支 C: 情绪分类 ----
            cls_logits_s = self.classifier(z_s)

            # ---- 分支 D: 域判别 (GRL + D) ----
            dom_logits_s = self.domain_disc(z_s, grl_alpha=1.0)
            dom_logits_t = self.domain_disc(z_t, grl_alpha=1.0)

            # ---- 分支 Con: 对比学习 ----
            z_s_norm = self.contrastive_head(z_s)
            z_t_norm = self.contrastive_head(z_t)

            # 生成 target 伪标签 (仅用于对比学习)
            with torch.no_grad():
                tgt_probs = torch.softmax(self.classifier(z_t), dim=1)
                y_t_pseudo = torch.argmax(tgt_probs, dim=1)

            # ---- 总损失 ----
            loss_dict = self.criterion(
                cls_logits_s, y_s,
                dom_logits_s, dom_logits_t,
                z_s_norm, z_t_norm, y_t_pseudo,
            )

            # ---- Backward ----
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0)
            self.optimizer.step()

            ep_cls += loss_dict['cls'].item()
            ep_dom += loss_dict['domain'].item()
            ep_con += loss_dict['contrastive'].item()
            ep_total += loss_dict['total'].item()
            n_batches += 1

        self.scheduler.step()

        return {
            'loss_cls': ep_cls / max(n_batches, 1),
            'loss_domain': ep_dom / max(n_batches, 1),
            'loss_con': ep_con / max(n_batches, 1),
            'loss_total': ep_total / max(n_batches, 1),
        }

    def train_fold(self, src_loader: DataLoader, tgt_loader: DataLoader,
                   target_subject_id: str, fold_idx: int = 0,
                   depression_label: int = -1) -> dict:
        """训练一个 fold

        参数:
            depression_label: 验证被试的抑郁症标签 (0=healthy, 1=depressed, -1=unknown)
        """
        best_acc = 0.0
        best_state = None
        self.history = {'epoch': [], 'loss_cls': [], 'loss_domain': [],
                        'loss_con': [], 'val_acc': []}

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Fold {fold_idx}: Target = {target_subject_id}")
        self.logger.info(f"{'='*60}")

        for epoch in range(self.epochs):
            # 训练
            stats = self.train_epoch(src_loader, tgt_loader, epoch)

            # 验证
            if (epoch + 1) % self.eval_interval == 0:
                val_acc = self._evaluate_target_acc(tgt_loader)

                self.history['epoch'].append(epoch + 1)
                self.history['loss_cls'].append(stats['loss_cls'])
                self.history['loss_domain'].append(stats['loss_domain'])
                self.history['loss_con'].append(stats['loss_con'])
                self.history['val_acc'].append(val_acc)

                self.logger.info(
                    f"Epoch [{epoch+1:3d}/{self.epochs}] "
                    f"Cls={stats['loss_cls']:.4f} "
                    f"Dom={stats['loss_domain']:.4f} "
                    f"Con={stats['loss_con']:.4f} "
                    f"ValAcc={val_acc:.4f} "
                    f"Best={best_acc:.4f}"
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {
                        'encoder': self.encoder.state_dict(),
                        'classifier': self.classifier.state_dict(),
                        'domain_disc': self.domain_disc.state_dict(),
                        'epoch': epoch + 1,
                        'val_accuracy': val_acc,
                    }

        # 保存最佳模型
        if best_state:
            torch.save(best_state, os.path.join(
                self.save_dir, f"best_dann_fold{fold_idx}_{target_subject_id}.pth"))

        self.logger.info(f"Fold {fold_idx} 完成 | Best Acc={best_acc:.4f}")

        return {
            'target_subject': target_subject_id,
            'best_accuracy': best_acc,
            'depression_label': depression_label,
        }
