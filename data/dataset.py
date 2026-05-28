"""
PyTorch Dataset (v2 — DANN + Transformer 版本)
================================================

预处理管线 (按顺序):
1. 降采样: 每个被试随机保留 downsample_ratio 比例的窗口 (减少冗余)
2. Z-score 归一化: **每个被试独立归一化** (防止数据泄漏)
3. 集成伪标签: 训练基础分类器 → 目标域投票 → 全票通过的加入源域

输出格式适配 Transformer: (30 tokens × 4 dims)
每个样本附带 domain label: source=0, target=1
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os, glob
from typing import List, Tuple, Optional
from .preprocess import process_subject_train, process_subject_test
from ensemble.pseudo_labeler import EnsemblePseudoLabeler


class EEGDataset(Dataset):
    """
    单个被试的EEG数据集。
    输出: (features, emotion_label, domain_label, subject_id)
    features: (30, 4) — 30 tokens × 4 dims
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 domain_label: int, subject_id: str):
        self.features = torch.FloatTensor(features)         # (n, 30, 4)
        self.labels = torch.LongTensor(labels.astype(int))  # (n,)
        self.domain_label = domain_label
        self.subject_id = subject_id

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx],
                self.domain_label, self.subject_id)


class CrossSubjectDataLoader:
    """
    跨被试数据加载器 (含集成伪标签预处理)。

    预处理管线 (每个被试):
      .mat → Trial切分 → 降采样 → 滑动窗口 → DE特征 → 独立zscore归一化

    LOSO每折:
      1. 划分 Source (N-1人) / Target (1人)
      2. 集成伪标签: 在源域训练基础分类器 → 目标域投票 → 全票通过的加入源域
      3. 返回增强后的 source_loader + target_loader
    """

    def __init__(self, train_root: str, window_size: int = 250,
                 stride: int = 125, fs: float = 250.0,
                 batch_size: int = 128, n_folds: int = 5,
                 downsample_ratio: float = 0.5,
                 ensemble_enabled: bool = True,
                 ensemble_threshold: float = 0.8,
                 ensemble_max_ratio: float = 0.3):
        self.train_root = train_root
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.downsample_ratio = downsample_ratio
        self.ensemble_enabled = ensemble_enabled
        self.ensemble_threshold = ensemble_threshold
        self.ensemble_max_ratio = ensemble_max_ratio

        # 收集所有 .mat 文件
        self.file_paths = []
        for subdir in ["正常人", "抑郁症患者"]:
            d = os.path.join(train_root, subdir)
            if os.path.exists(d):
                self.file_paths.extend(sorted(glob.glob(os.path.join(d, "*.mat"))))

        print(f"[DataLoader] 找到 {len(self.file_paths)} 个训练文件")

        # 预加载所有被试数据 (含降采样 + 独立zscore归一化)
        self.all_subjects = {}
        self._load_all_subjects()

    def _extract_subject_id(self, filepath: str) -> str:
        return os.path.basename(filepath).replace("timedata.mat", "")

    def _load_all_subjects(self):
        n_total = len(self.file_paths)
        for i, fp in enumerate(self.file_paths):
            sid = self._extract_subject_id(fp)
            print(f"\r[DataLoader] 加载中... [{i+1}/{n_total}] {sid}", end="", flush=True)
            features, labels = process_subject_train(
                fp, trial_length=12500,
                window_size=self.window_size, stride=self.stride,
                fs=self.fs, downsample_ratio=self.downsample_ratio
            )
            if features is not None and len(features) > 0:
                self.all_subjects[sid] = (features, labels)

        print(f"\r[DataLoader] 成功加载 {len(self.all_subjects)}/{n_total} 个被试 "
              f"(降采样={self.downsample_ratio}, 每人独立zscore)")

    def get_subject_ids(self) -> List[str]:
        return sorted(self.all_subjects.keys())

    def _run_ensemble_pseudo_labeling(self, src_features, src_labels,
                                       tgt_features):
        """
        集成伪标签: 训练基础分类器 → 目标域全票通过 → 加入源域。

        在预处理阶段执行 (不是训练阶段), 使用展平的DE特征 (120维)。
        """
        ensemble = EnsemblePseudoLabeler(
            confidence_threshold=self.ensemble_threshold
        )

        # 展平: (n, 30, 4) → (n, 120)
        X_src = src_features.reshape(len(src_features), -1)
        X_tgt = tgt_features.reshape(len(tgt_features), -1)

        print(f"  [集成伪标签] 训练 {5} 个基础分类器...")

        ensemble.fit(X_src, src_labels)
        good_idx, good_labels = ensemble.predict_and_filter(X_tgt)

        if len(good_idx) == 0:
            print(f"  [集成伪标签] 未找到全票通过的伪标签样本")
            return src_features, src_labels

        # 限制加入数量
        max_add = int(len(src_features) * self.ensemble_max_ratio)
        if len(good_idx) > max_add:
            keep = np.random.choice(len(good_idx), max_add, replace=False)
            good_idx = good_idx[keep]
            good_labels = good_labels[keep]

        print(f"  [集成伪标签] 全票通过 {len(good_idx)} 个目标域样本 → 加入源域 "
              f"(neutral={(good_labels==0).sum()}, positive={(good_labels==1).sum()})")

        augmented_features = np.concatenate(
            [src_features, tgt_features[good_idx]], axis=0)
        augmented_labels = np.concatenate(
            [src_labels, good_labels], axis=0)

        return augmented_features, augmented_labels

    def folds(self):
        """
        留一法 (LOSO) / K折交叉验证生成器。

        每折预处理管线:
          1. 划分 source / target
          2. 集成伪标签 (全票通过的 target → source)
          3. 构建 DataLoader
        """
        subject_ids = np.array(self.get_subject_ids())
        n_subjects = len(subject_ids)
        rng = np.random.RandomState(42)
        shuffled = rng.permutation(subject_ids)

        if self.n_folds <= 0 or self.n_folds >= n_subjects:
            target_list = [[sid] for sid in subject_ids]
        else:
            fold_size = n_subjects // self.n_folds
            target_list = []
            for fold in range(self.n_folds):
                start = fold * fold_size
                end = start + fold_size if fold < self.n_folds - 1 else n_subjects
                target_list.append([shuffled[start]])

        for fold, val_ids in enumerate(target_list):
            target_id = val_ids[0]
            train_ids = [s for s in subject_ids if s not in val_ids]

            # ---- 构建 Source (合并多个被试) ----
            src_features, src_labels = [], []
            for sid in train_ids:
                f, l = self.all_subjects[sid]
                src_features.append(f)
                src_labels.append(l)
            src_features = np.concatenate(src_features)
            src_labels = np.concatenate(src_labels)

            # ---- 构建 Target ----
            tgt_features, tgt_labels = self.all_subjects[target_id]

            # ---- 集成伪标签: 高置信目标域样本 → 源域 (预处理阶段!) ----
            if self.ensemble_enabled:
                src_features, src_labels = self._run_ensemble_pseudo_labeling(
                    src_features, src_labels, tgt_features
                )

            # ---- 构建 DataLoader ----
            src_dataset = EEGDataset(src_features, src_labels,
                                     domain_label=0, subject_id="source")
            src_loader = DataLoader(src_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=True)

            tgt_dataset = EEGDataset(tgt_features, tgt_labels,
                                     domain_label=1, subject_id=target_id)
            tgt_loader = DataLoader(tgt_dataset,
                                    batch_size=min(self.batch_size, len(tgt_dataset)),
                                    shuffle=True, drop_last=False)

            print(f"[Fold {fold}] Source={len(train_ids)} subjects "
                  f"({len(src_dataset)} windows), Target={target_id} "
                  f"({len(tgt_dataset)} windows)")

            yield src_loader, tgt_loader, target_id


class TestDataLoader:
    """测试集加载器"""

    def __init__(self, test_root: str, window_size: int = 250,
                 stride: int = 125, fs: float = 250.0, batch_size: int = 128):
        self.test_root = test_root
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.batch_size = batch_size

        self.file_paths = sorted(glob.glob(os.path.join(test_root, "*.mat")))
        self.all_features = {}

        for fp in self.file_paths:
            sid = os.path.basename(fp).replace(".mat", "")
            f, _ = process_subject_test(fp, trial_length=2500,
                                         window_size=window_size,
                                         stride=stride, fs=fs)
            if f is not None:
                self.all_features[sid] = f

        print(f"[TestLoader] 加载 {len(self.all_features)} 个测试被试")

    def get_subject_ids(self):
        return sorted(self.all_features.keys())

    def get_loader(self, subject_id: str):
        f = self.all_features[subject_id]
        dataset = EEGDataset(f, np.zeros(len(f), dtype=int),
                             domain_label=1, subject_id=subject_id)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
