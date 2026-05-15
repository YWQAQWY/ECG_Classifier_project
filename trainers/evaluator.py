"""
模型评估器 — 评估目标域上的情绪识别性能
========================================

在跨被试场景中，我们真正关心的是:
    模型在"未见过的新被试"上的分类准确率
    而不是源域 (训练集被试) 上的准确率

因此评估器专门设计用于评估目标域性能。
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


class Evaluator:
    """
    目标域评估器。

    在每个epoch末尾，用当前模型对目标被试的数据做预测，
    计算accuracy、F1等指标。
    注意: 评估时使用目标域的真实标签，但这些标签不参与训练!
    """

    def __init__(self, device: str = "cpu"):
        """
        参数:
            device: 评估设备
        """
        self.device = device

    @torch.no_grad()
    def evaluate(self, encoder: torch.nn.Module,
                 classifier: torch.nn.Module,
                 target_loader: DataLoader) -> dict:
        """
        在目标域上评估模型性能。

        评估流程:
            1. 遍历目标域DataLoader
            2. 编码器提取特征 → 分类器预测
            3. 收集所有预测和真实标签
            4. 计算 accuracy, F1, confusion matrix

        注意: 不使用梯度计算 (torch.no_grad)
              真实标签仅用于评估, 不参与训练

        参数:
            encoder: EEGNet编码器
            classifier: 情绪分类器
            target_loader: 目标域数据加载器 (包含标签用于评估)

        返回:
            metrics: {
                'accuracy': float,
                'f1': float,
                'y_true': np.ndarray,
                'y_pred': np.ndarray,
                'confusion_matrix': np.ndarray,
                'features': np.ndarray (可选, 用于t-SNE)
            }
        """
        encoder.eval()
        classifier.eval()

        all_preds = []
        all_labels = []
        all_features = []

        for batch_data in target_loader:
            # 解析batch (features, labels, subject_id)
            x, y_true, _ = batch_data
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            # 前向传播
            features = encoder(x)
            logits = classifier(features)

            # 预测: argmax
            probs = F.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1)

            # 收集结果
            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y_true.cpu().numpy())
            all_features.append(features.cpu().numpy())

        # 合并所有batch
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        features = np.concatenate(all_features)

        # 计算指标
        accuracy = (y_pred == y_true).mean()
        from sklearn.metrics import f1_score, confusion_matrix
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)

        encoder.train()
        classifier.train()

        return {
            'accuracy': accuracy,
            'f1': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'confusion_matrix': cm,
            'features': features,
        }

    @torch.no_grad()
    def predict(self, encoder: torch.nn.Module,
                classifier: torch.nn.Module,
                loader: DataLoader) -> np.ndarray:
        """
        对无标签数据进行预测 (用于测试集推理)。

        参数:
            encoder: 编码器
            classifier: 分类器
            loader: 数据加载器 (无标签)

        返回:
            predictions: 预测标签 (n_samples,)
        """
        encoder.eval()
        classifier.eval()

        all_preds = []

        for batch_data in loader:
            # 测试集: (features, subject_id) — 无标签
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data
            x = x.to(self.device)

            features = encoder(x)
            logits = classifier(features)
            probs = F.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1)

            all_preds.append(y_pred.cpu().numpy())

        encoder.train()
        classifier.train()

        return np.concatenate(all_preds)

    @torch.no_grad()
    def extract_features(self, encoder: torch.nn.Module,
                         loader: DataLoader,
                         max_samples: int = 500) -> tuple:
        """
        提取特征用于可视化 (t-SNE)。

        参数:
            encoder: 编码器
            loader: 数据加载器
            max_samples: 最大采样数

        返回:
            features: 特征向量 (n, dim)
            labels: 标签 (n,)
            domain_labels: 域标签 (n,) — 0=source, 1=target (需额外处理)
        """
        encoder.eval()

        all_features = []
        all_labels = []

        for batch_data in loader:
            x, y, _ = batch_data
            x = x.to(self.device)

            features = encoder(x)

            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())

            if len(np.concatenate(all_features)) >= max_samples:
                break

        encoder.train()

        return np.concatenate(all_features)[:max_samples], \
            np.concatenate(all_labels)[:max_samples]
