"""
集成伪标签生成器
================

利用集成学习找到"好的"伪标签目标域数据，加入源域训练。

核心逻辑:
  1. 训练多个基础分类器 (SVM, MLP, RandomForest 等) 在源域上
  2. 用这些分类器对目标域数据预测
  3. 如果所有分类器给出的标签一致 (全票通过) → "好"伪标签
  4. 高置信度伪标签样本加入到源域训练集

为什么需要集成?
  - 单一模型可能给出噪声伪标签
  - 多模型一致性 = 高可靠性
  - "容易分类"的目标域样本 → 加入源域 → 增强域适应
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class EnsemblePseudoLabeler:
    """
    集成伪标签器。

    使用多个分类器投票, 全票通过的伪标签被认为是"好"的。
    """

    def __init__(self, confidence_threshold: float = 1.0):
        """
        参数:
            confidence_threshold: 全票比例阈值 (1.0=全票, 0.75=75%一致)
        """
        self.confidence_threshold = confidence_threshold
        self.models = []

    def _get_base_models(self, input_dim: int):
        """构建基础分类器集合"""
        return [
            ('svm_rbf', SVC(kernel='rbf', probability=True, random_state=42)),
            ('svm_linear', SVC(kernel='linear', probability=True, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                  random_state=42, early_stopping=True)),
        ]

    def fit(self, X_src: np.ndarray, y_src: np.ndarray):
        """
        在源域数据上训练所有基础分类器。

        参数:
            X_src: 源域特征, 展平的 (n_src, 30*4)
            y_src: 源域标签 (n_src,)
        """
        input_dim = X_src.shape[1]
        self.models = self._get_base_models(input_dim)

        for name, model in self.models:
            try:
                model.fit(X_src, y_src)
            except Exception as e:
                print(f"  [Ensemble] {name} 训练失败: {e}")

    def predict_and_filter(self, X_tgt: np.ndarray, z_tgt: np.ndarray = None) -> tuple:
        """
        预测目标域数据并过滤出高置信伪标签。

        参数:
            X_tgt: 目标域特征, 展平的 (n_tgt, 120)
            z_tgt: 目标域 Transformer 特征 (n_tgt, d_model) — 可选

        返回:
            good_indices: 高置信样本在目标域中的索引
            good_labels: 对应的伪标签
        """
        if len(self.models) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # 收集所有模型的预测
        all_preds = []
        for name, model in self.models:
            try:
                preds = model.predict(X_tgt)
                all_preds.append(preds)
            except Exception:
                continue

        if len(all_preds) < 2:
            return np.array([], dtype=int), np.array([], dtype=int)

        all_preds = np.array(all_preds)  # (n_models, n_tgt)

        # 计算每个样本的全票比例
        # 对于二分类: 多数模型预测0 or 预测1
        agreement = []
        final_labels = []
        for i in range(len(X_tgt)):
            sample_preds = all_preds[:, i]  # (n_models,)
            # 出现最多的标签
            counts = np.bincount(sample_preds, minlength=2)
            max_count = counts.max()
            best_label = counts.argmax()
            agree_ratio = max_count / len(sample_preds)

            agreement.append(agree_ratio)
            final_labels.append(best_label)

        agreement = np.array(agreement)
        final_labels = np.array(final_labels)

        # 过滤: 全票比例 >= 阈值
        good_mask = agreement >= self.confidence_threshold
        good_indices = np.where(good_mask)[0]
        good_labels = final_labels[good_mask]

        return good_indices, good_labels

    def augment_source(self, X_src: np.ndarray, y_src: np.ndarray,
                       z_src: np.ndarray, X_tgt: np.ndarray,
                       z_tgt: np.ndarray, max_ratio: float = 0.3) -> tuple:
        """
        将高置信伪标签目标域样本加入源域。

        参数:
            X_src: 源域展平特征 (n_src, 120)
            y_src: 源域标签 (n_src,)
            z_src: 源域 Transformer 特征 (n_src, d_model)
            X_tgt: 目标域展平特征 (n_tgt, 120)
            z_tgt: 目标域 Transformer 特征 (n_tgt, d_model)
            max_ratio: 最多加入源域样本数的比例

        返回:
            augmented_X, augmented_y, augmented_z
        """
        good_indices, good_labels = self.predict_and_filter(X_tgt, z_tgt)

        if len(good_indices) == 0:
            print(f"  [Ensemble] 未找到全票通过的伪标签样本")
            return X_src, y_src, z_src

        # 限制加入数量
        max_add = int(len(X_src) * max_ratio)
        if len(good_indices) > max_add:
            keep = np.random.choice(len(good_indices), max_add, replace=False)
            good_indices = good_indices[keep]
            good_labels = good_labels[keep]

        print(f"  [Ensemble] 加入 {len(good_indices)} 个高置信伪标签到源域 "
              f"(全票通过, max={max_add})")

        augmented_X = np.concatenate([X_src, X_tgt[good_indices]], axis=0)
        augmented_y = np.concatenate([y_src, good_labels], axis=0)
        augmented_z = np.concatenate([z_src, z_tgt[good_indices]], axis=0)

        return augmented_X, augmented_y, augmented_z
