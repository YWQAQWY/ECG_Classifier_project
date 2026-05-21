"""
DDA 训练器 — 跨被试EEG情绪识别的核心训练循环
==============================================

实现论文 Algorithm 1 (UDDA) / Algorithm 2 (SDDA) 的训练流程。

每个epoch的训练流程 (论文 Algorithm 1):
    Step 1: 从源域采样 {(x_s^i, y_s^i)}  → 有标签
    Step 2: 从目标域采样 {x_t^j}        → 无标签 (训练时)
    Step 3: 编码器 → Fs, Ft  (特征提取)
    Step 4: 分类器 → Ps, Pt  (情绪预测)
    Step 5: 计算 L_ce      (源域分类损失, 使用真实标签)
    Step 6: 计算 L_gdd     (全局域差异, MMD(Fs, Ft))
    Step 7: 生成伪标签 Yt_hat = argmax(Pt)
    Step 8: 计算 L_lsd     (局部子域差异, 使用伪标签)
    Step 9: 计算 α         (动态平衡因子)
    Step 10: L_total = L_ce + β*(α*L_gdd + (1-α)*L_lsd)
    Step 11: loss.backward() + optimizer.step()

跨被试验证 (leave-one-subject-out):
    每个fold:
    - 训练: N-1个被试作为source domain
    - 验证: 1个被试作为target domain
    - 目标: 最大化target domain上的分类准确率

训练策略核心理解:
    1. 源域有标签 → CE监督学习分类边界
    2. 目标域无标签 → GDD无监督对齐整体分布
    3. 伪标签 → LSD半监督对齐类别分布
    4. 动态α → 从粗到细的渐进对齐
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from typing import Optional, Dict

try:
    from ..losses.dda_loss import DDALoss
    from ..utils.logger import Logger
    from ..utils.visualization import plot_training_curves, plot_tsne
    from .evaluator import Evaluator
except ImportError:
    # Fallback for direct import (e.g. from scripts that add project root to sys.path)
    from losses.dda_loss import DDALoss
    from utils.logger import Logger
    from utils.visualization import plot_training_curves, plot_tsne
    from trainers.evaluator import Evaluator


class DDATrainer:
    """
    DDA 训练器。

    管理完整的训练生命周期: 训练、验证、保存、可视化。
    """

    def __init__(self, encoder: nn.Module, classifier: nn.Module,
                 config: dict, logger: Logger):
        """
        参数:
            encoder: EEGNet 编码器
            classifier: 情绪分类器
            config: 配置字典 (从YAML加载)
            logger: 日志管理器
        """
        self.encoder = encoder
        self.classifier = classifier
        self.config = config
        self.logger = logger

        # 设备
        device_str = config.get('training', {}).get('device', 'cpu')
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")

        # 移动模型到设备
        self.encoder.to(self.device)
        self.classifier.to(self.device)

        # 优化器
        train_cfg = config.get('training', {})
        lr = train_cfg.get('learning_rate', 0.001)
        wd = train_cfg.get('weight_decay', 0.0001)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=lr, weight_decay=wd
        )
        self.logger.info(f"优化器: Adam (lr={lr}, weight_decay={wd})")

        # 学习率调度器
        self.scheduler = None
        lr_sched = train_cfg.get('lr_scheduler', 'none')
        if lr_sched == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg.get('epochs', 200)
            )
        elif lr_sched == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_cfg.get('lr_step_size', 50),
                gamma=train_cfg.get('lr_gamma', 0.5)
            )

        # DDA 损失模块
        dda_cfg = config.get('dda', {})
        self.dda_loss = DDALoss(
            n_classes=config.get('model', {}).get('classifier', {}).get('n_classes', 2),
            beta=dda_cfg.get('beta', 0.1),
            bandwidths=dda_cfg.get('mmd', {}).get('bandwidths', None),
            alpha_schedule=dda_cfg.get('alpha_schedule', 'linear'),
            min_samples_per_class=dda_cfg.get('lsd', {}).get('min_samples_per_class', 3),
            confidence_threshold=dda_cfg.get('lsd', {}).get('confidence_threshold', 0.7),
        )

        # 评估器
        self.evaluator = Evaluator(device=str(self.device))

        # 训练参数
        self.epochs = train_cfg.get('epochs', 200)
        self.log_interval = config.get('experiment', {}).get('log_interval', 10)
        self.eval_interval = config.get('experiment', {}).get('eval_interval', 1)

        # 实验模式
        self.mode = config.get('experiment', {}).get('mode', 'ce_gdd_lsd')
        self.logger.info(f"实验模式: {self.mode}")

        # 保存路径
        self.save_dir = config.get('experiment', {}).get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        # 训练历史
        self.history = {
            'epoch': [], 'loss_ce': [], 'loss_gdd': [], 'loss_lsd': [],
            'loss_total': [], 'alpha': [], 'train_acc': [], 'val_acc': [],
        }

    def _get_source_accuracy(self, logits_s: torch.Tensor, ys: torch.Tensor) -> float:
        """计算源域batch的训练准确率"""
        preds = torch.argmax(logits_s, dim=1)
        correct = (preds == ys).float().sum()
        return (correct / ys.size(0)).item()

    def train_epoch(self, src_loader: DataLoader, tgt_loader: DataLoader,
                    epoch: int) -> dict:
        """
        训练一个epoch。

        这是DDA训练的核心: 每个batch同时使用源域和目标域数据，
        计算CE+GDD+LSD三个损失，联合优化编码器和分类器。

        参数:
            src_loader: 源域DataLoader (有标签)
            tgt_loader: 目标域DataLoader (无标签训练用)
            epoch: 当前epoch编号 (0-indexed)

        返回:
            epoch_stats: 该epoch的平均统计信息
        """
        self.encoder.train()
        self.classifier.train()

        epoch_loss_ce = 0.0
        epoch_loss_gdd = 0.0
        epoch_loss_lsd = 0.0
        epoch_loss_total = 0.0
        epoch_alpha = 0.0
        epoch_src_acc = 0.0
        n_batches = 0

        # 将目标域loader转为无限迭代器，保证与源域同步训练
        # 如果目标域batch比源域少，循环使用
        tgt_iter = iter(tgt_loader)

        for batch_idx, src_batch in enumerate(src_loader):
            # ---- Step 1: 获取源域数据 (有标签) ----
            x_s, y_s, _ = src_batch
            x_s = x_s.to(self.device)  # (batch, 1, 30, 4)
            y_s = y_s.to(self.device)  # (batch,)

            # ---- Step 2: 获取目标域数据 (无标签) ----
            try:
                x_t, _, _ = next(tgt_iter)
            except StopIteration:
                # 目标域数据不够一轮, 重新开始迭代
                tgt_iter = iter(tgt_loader)
                x_t, _, _ = next(tgt_iter)
            x_t = x_t.to(self.device)

            # ---- Step 3: 编码器提取特征 ----
            # Fs: 源域特征, Ft: 目标域特征
            # 特征空间是GDD/LSD对齐的目标空间
            fs = self.encoder(x_s)  # (batch_s, feature_dim)
            ft = self.encoder(x_t)  # (batch_t, feature_dim)

            # ---- Step 4: 分类器预测 ----
            # Ps: 源域分类logits, Pt: 目标域分类logits
            logits_s = self.classifier(fs)  # (batch_s, n_classes)
            logits_t = self.classifier(ft)  # (batch_t, n_classes)

            # ---- Step 5-9: 计算 DDA 总损失 ----
            # 内部完成:
            #   L_ce (Step 5), L_gdd (Step 6),
            #   伪标签生成 (Step 7), L_lsd (Step 8),
            #   动态α (Step 9)
            loss_dict = self.dda_loss(
                fs, ft, logits_s, logits_t, y_s,
                epoch=epoch, max_epoch=self.epochs
            )

            # 根据实验模式调整损失:
            # - ce_only: 只使用CE
            # - ce_gdd: 使用CE+GDD (α固定为1)
            # - ce_gdd_lsd: 完整DDA (默认)
            if self.mode == "ce_only":
                loss = loss_dict['ce']
            elif self.mode == "ce_gdd":
                loss = loss_dict['ce'] + self.dda_loss.beta * loss_dict['gdd']
            else:  # ce_gdd_lsd (完整DDA)
                loss = loss_dict['total']

            # ---- Step 10: 反向传播 ----
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪 — 防止MMD计算可能产生的梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.classifier.parameters()),
                max_norm=10.0
            )
            self.optimizer.step()

            # ---- 累计统计 ----
            epoch_loss_ce += loss_dict['ce'].item()
            epoch_loss_gdd += loss_dict['gdd'].item()
            epoch_loss_lsd += loss_dict['lsd'].item()
            epoch_loss_total += loss.item()
            epoch_alpha += loss_dict['alpha']
            epoch_src_acc += self._get_source_accuracy(logits_s, y_s)
            n_batches += 1

            # 日志输出
            if batch_idx % self.log_interval == 0:
                self.logger.debug(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Batch [{batch_idx}/{len(src_loader)}] "
                    f"CE={loss_dict['ce'].item():.4f} "
                    f"GDD={loss_dict['gdd'].item():.4f} "
                    f"LSD={loss_dict['lsd'].item():.4f} "
                    f"α={loss_dict['alpha']:.3f} "
                    f"Total={loss.item():.4f}"
                )

        # 计算epoch均值
        return {
            'loss_ce': epoch_loss_ce / max(n_batches, 1),
            'loss_gdd': epoch_loss_gdd / max(n_batches, 1),
            'loss_lsd': epoch_loss_lsd / max(n_batches, 1),
            'loss_total': epoch_loss_total / max(n_batches, 1),
            'alpha': epoch_alpha / max(n_batches, 1),
            'src_accuracy': epoch_src_acc / max(n_batches, 1),
        }

    def train_fold(self, src_loader: DataLoader, tgt_loader: DataLoader,
                   target_subject_id: str, fold_idx: int = 0) -> dict:
        """
        训练一个 fold (一个target subject)。

        完整的训练+验证流程。

        参数:
            src_loader: 源域DataLoader
            tgt_loader: 目标域DataLoader (含标签用于验证)
            target_subject_id: 目标被试ID
            fold_idx: fold编号

        返回:
            best_result: 最佳验证结果
        """
        best_acc = 0.0
        best_epoch = 0
        best_state = None

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Fold {fold_idx}: Target Subject = {target_subject_id}")
        self.logger.info(f"{'='*60}")

        for epoch in range(self.epochs):
            # ---- 训练一个epoch ----
            train_stats = self.train_epoch(src_loader, tgt_loader, epoch)

            # ---- 学习率调度 ----
            if self.scheduler is not None:
                self.scheduler.step()

            # ---- 验证 ----
            if (epoch + 1) % self.eval_interval == 0:
                eval_results = self.evaluator.evaluate(
                    self.encoder, self.classifier, tgt_loader
                )
                val_acc = eval_results['accuracy']
                val_f1 = eval_results['f1']

                # 记录历史
                self.history['epoch'].append(epoch + 1)
                self.history['loss_ce'].append(train_stats['loss_ce'])
                self.history['loss_gdd'].append(train_stats['loss_gdd'])
                self.history['loss_lsd'].append(train_stats['loss_lsd'])
                self.history['loss_total'].append(train_stats['loss_total'])
                self.history['alpha'].append(train_stats['alpha'])
                self.history['train_acc'].append(train_stats['src_accuracy'])
                self.history['val_acc'].append(val_acc)

                # TensorBoard 记录
                self.logger.log_train(
                    epoch + 1,
                    train_stats['loss_ce'],
                    train_stats['loss_gdd'],
                    train_stats['loss_lsd'],
                    train_stats['loss_total'],
                    train_stats['alpha'],
                    train_stats['src_accuracy']
                )
                self.logger.log_val(epoch + 1, val_acc, val_f1)

                # 日志
                self.logger.info(
                    f"Epoch [{epoch+1:3d}/{self.epochs}] "
                    f"CE={train_stats['loss_ce']:.4f} "
                    f"GDD={train_stats['loss_gdd']:.4f} "
                    f"LSD={train_stats['loss_lsd']:.4f} "
                    f"α={train_stats['alpha']:.3f} "
                    f"SrcAcc={train_stats['src_accuracy']:.3f} "
                    f"TgtAcc={val_acc:.4f} "
                    f"F1={val_f1:.4f}"
                )

                # 保存最佳模型 (基于目标域准确率!)
                # 这才是我们真正关心的指标
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    best_state = {
                        'encoder': self.encoder.state_dict(),
                        'classifier': self.classifier.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'val_accuracy': val_acc,
                        'history': self.history,
                    }

        # ---- Fold 训练完成 ----
        self.logger.info(f"\nFold {fold_idx} 完成!")
        self.logger.info(f"  最佳目标域准确率: {best_acc:.4f} (Epoch {best_epoch})")

        # 保存最佳模型
        if best_state is not None:
            save_path = os.path.join(
                self.save_dir,
                f"best_model_fold{fold_idx}_{target_subject_id}.pth"
            )
            torch.save(best_state, save_path)
            self.logger.info(f"  模型已保存至: {save_path}")

        # 绘制训练曲线
        if len(self.history['epoch']) > 0:
            curve_path = os.path.join(
                self.logger.log_dir,
                f"training_curves_fold{fold_idx}.png"
            )
            plot_training_curves(self.history, curve_path)

        return {
            'target_subject': target_subject_id,
            'best_accuracy': best_acc,
            'best_epoch': best_epoch,
        }

    def visualize_features(self, src_loader: DataLoader,
                           tgt_loader: DataLoader,
                           save_name: str = "tsne_features"):
        """
        t-SNE 可视化编码器输出特征。

        可观察:
        - 源域和目标域特征是否对齐 (域差异)
        - 同类特征是否跨域聚类 (类感知对齐)
        - 与论文 Fig.4 对应
        """
        self.encoder.eval()

        # 提取源域特征
        src_features = []
        src_labels = []
        for x_s, y_s, _ in src_loader:
            x_s = x_s.to(self.device)
            f_s = self.encoder(x_s)
            src_features.append(f_s.detach().cpu().numpy())
            src_labels.append(y_s.numpy())
            if len(np.concatenate(src_features)) >= 300:
                break

        src_features = np.concatenate(src_features)[:300]
        src_labels = np.concatenate(src_labels)[:300]

        # 提取目标域特征
        tgt_features = []
        tgt_labels = []
        for x_t, y_t, _ in tgt_loader:
            x_t = x_t.to(self.device)
            f_t = self.encoder(x_t)
            tgt_features.append(f_t.detach().cpu().numpy())
            tgt_labels.append(y_t.numpy())
            if len(np.concatenate(tgt_features)) >= 300:
                break

        tgt_features = np.concatenate(tgt_features)[:300]
        tgt_labels = np.concatenate(tgt_labels)[:300]

        # 合并
        all_features = np.concatenate([src_features, tgt_features], axis=0)
        all_labels = np.concatenate([src_labels, tgt_labels], axis=0)
        domain_labels = np.concatenate([
            np.zeros(len(src_features)),
            np.ones(len(tgt_features))
        ]).astype(int)

        save_path = os.path.join(self.logger.log_dir, f"{save_name}.png")
        plot_tsne(all_features, all_labels, domain_labels, save_path,
                  title="DDA Feature Visualization (t-SNE)")

        self.encoder.train()
