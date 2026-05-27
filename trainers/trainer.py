"""
DANN + Transformer 训练器 (v2)
==============================

训练流程 (参考 DEEP_DANN_SEED.py):

每 batch:
  Source B → Transformer F → z_s
  Target B → Transformer F → z_t  (共享同一 F!)

  z_s → C → L_cls             (情绪分类, source only)
  z_s → GRL → D → L_dom_s     (域对抗)
  z_t → GRL → D → L_dom_t     (域对抗)
  z_s + z_t → Con → L_con     (对比学习, 类结构约束)

  L_total = λ1*L_cls + λ2*L_domain + λ3*L_con

集成伪标签 (周期性):
  训练基础分类器 → 目标域投票 → 全票通过 → 加入源域
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from models.transformer_encoder import TransformerEncoder
from models.classifier import EmotionClassifier
from models.domain_discriminator import DomainDiscriminator
from models.contrastive_head import ContrastiveHead
from losses.dann_loss import DANNTotalLoss
from ensemble.pseudo_labeler import EnsemblePseudoLabeler
from utils.logger import Logger
from utils.visualization import plot_training_curves


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

        # 优化器 (参考导师: Adam, lr=5e-4, wd=1e-4)
        train_cfg = config.get('training', {})
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.classifier.parameters()) +
            list(self.domain_disc.parameters()),
            lr=train_cfg.get('learning_rate', 5e-4),
            weight_decay=train_cfg.get('weight_decay', 1e-4),
        )

        # 学习率调度 (参考导师: StepLR, step=10, gamma=0.99)
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

        # 集成伪标签器
        self.ensemble = None
        ensemble_cfg = config.get('ensemble', {})
        if ensemble_cfg.get('enabled', True):
            self.ensemble = EnsemblePseudoLabeler(
                confidence_threshold=ensemble_cfg.get('confidence_threshold', 0.8),
            )
            self.ensemble_update_interval = ensemble_cfg.get('update_interval', 10)
            self.ensemble_max_ratio = ensemble_cfg.get('add_to_source_percent', 0.3)

        # 训练参数
        self.epochs = train_cfg.get('epochs', 100)
        self.log_interval = config.get('experiment', {}).get('log_interval', 10)
        self.eval_interval = config.get('experiment', {}).get('eval_interval', 1)
        self.save_dir = config.get('experiment', {}).get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        # 历史
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

        for batch_idx, src_batch in enumerate(src_loader):
            # ---- 获取数据 ----
            x_s, y_s, _, _ = src_batch  # domain=0
            x_s, y_s = x_s.to(self.device), y_s.to(self.device)

            try:
                x_t, y_t, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_t, y_t, _, _ = next(tgt_iter)
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)  # 不参与训练, 仅用于评估

            # ---- Forward: 共享 Transformer F ----
            z_s = self.encoder(x_s)  # (N_s, d_model)
            z_t = self.encoder(x_t)  # (N_t, d_model)

            # ---- 分支 C: 情绪分类 ----
            cls_logits_s = self.classifier(z_s)  # (N_s, 2)

            # ---- 分支 D: 域判别 (GRL + D) ----
            dom_logits_s = self.domain_disc(z_s, grl_alpha=1.0)  # → 0 (source)
            dom_logits_t = self.domain_disc(z_t, grl_alpha=1.0)  # → 1 (target)

            # ---- 分支 Con: 对比学习 ----
            z_s_norm = self.contrastive_head(z_s)
            z_t_norm = self.contrastive_head(z_t)

            # 生成 target 伪标签 (用于对比学习)
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
            'loss_cls': ep_cls / n_batches,
            'loss_domain': ep_dom / n_batches,
            'loss_con': ep_con / n_batches,
            'loss_total': ep_total / n_batches,
        }

    def run_ensemble_update(self, src_loader: DataLoader,
                            tgt_loader: DataLoader) -> Optional[DataLoader]:
        """
        执行集成伪标签更新: 训练基础分类器 → 投票 → 全票通过的加入源域。
        """
        if self.ensemble is None:
            return src_loader

        # 提取展平特征
        self.encoder.eval()
        X_src_list, y_src_list = [], []
        for x, y, _, _ in src_loader:
            X_src_list.append(x.view(x.size(0), -1).cpu().numpy())
            y_src_list.append(y.cpu().numpy())
        X_src = np.concatenate(X_src_list)
        y_src = np.concatenate(y_src_list)

        X_tgt_list = []
        z_tgt_list = []
        for x, _, _, _ in tgt_loader:
            X_tgt_list.append(x.view(x.size(0), -1).cpu().numpy())
            z = self.encoder(x.to(self.device))
            z_tgt_list.append(z.detach().cpu().numpy())
        X_tgt = np.concatenate(X_tgt_list)
        z_tgt = np.concatenate(z_tgt_list[:len(X_tgt_list)])  # align

        # 提取 source z
        z_src_list = []
        for x, _, _, _ in src_loader:
            z = self.encoder(x.to(self.device))
            z_src_list.append(z.detach().cpu().numpy())
        z_src = np.concatenate(z_src_list)

        # 训练集成模型 & 过滤伪标签
        self.ensemble.fit(X_src, y_src)
        good_idx, good_labels = self.ensemble.predict_and_filter(X_tgt)

        if len(good_idx) == 0:
            self.encoder.train()
            return src_loader

        # 限制数量
        max_add = int(len(X_src) * self.ensemble_max_ratio)
        if len(good_idx) > max_add:
            keep = np.random.choice(len(good_idx), max_add, replace=False)
            good_idx = np.array(good_idx)[keep]
            good_labels = np.array(good_labels)[keep]

        self.logger.info(f"  [Ensemble] 加入 {len(good_idx)} 个伪标签到源域")

        # 合并到源域
        # 展平特征 + z 都扩展
        augmented_X = np.concatenate([X_src, X_tgt[good_idx]])
        augmented_y = np.concatenate([y_src, good_labels])

        # 重建 DataLoader
        from torch.utils.data import TensorDataset
        aug_dataset = TensorDataset(
            torch.FloatTensor(augmented_X).view(-1, 30, 4),
            torch.LongTensor(augmented_y),
        )
        aug_loader = DataLoader(aug_dataset,
                                batch_size=src_loader.batch_size,
                                shuffle=True, drop_last=True)

        self.encoder.train()
        return aug_loader

    def train_fold(self, src_loader: DataLoader, tgt_loader: DataLoader,
                   target_subject_id: str, fold_idx: int = 0) -> dict:
        """训练一个 fold"""
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

            # 周期性集成伪标签更新
            if (self.ensemble is not None and
                (epoch + 1) % self.ensemble_update_interval == 0 and epoch > 0):
                new_src_loader = self.run_ensemble_update(src_loader, tgt_loader)
                if new_src_loader is not None:
                    src_loader = new_src_loader

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
                    best_epoch = epoch + 1
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

        return {'target_subject': target_subject_id, 'best_accuracy': best_acc}
