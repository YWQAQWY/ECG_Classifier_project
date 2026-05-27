"""
PyTorch Dataset (v2 — DANN + Transformer 版本)
================================================

核心变更:
- 5折交叉验证 (按被试分层)
- 每个样本附带 domain label: source=0, target=1
- 数据格式适配 Transformer: (30 tokens × 4 dims)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os, glob
from typing import List, Tuple, Optional
from .preprocess import process_subject_train, process_subject_test


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
    跨被试 5折交叉验证 + DANN 格式

    每折:
    - Source: N-1个被试 (domain=0, 有情绪标签)
    - Target: 1个被试   (domain=1, 标签仅验证用)
    """

    def __init__(self, train_root: str, window_size: int = 250,
                 stride: int = 125, fs: float = 250.0,
                 batch_size: int = 128, n_folds: int = 5,
                 downsample_ratio: float = 0.5):
        self.train_root = train_root
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.downsample_ratio = downsample_ratio

        # 收集所有 .mat 文件
        self.file_paths = []
        for subdir in ["正常人", "抑郁症患者"]:
            d = os.path.join(train_root, subdir)
            if os.path.exists(d):
                self.file_paths.extend(sorted(glob.glob(os.path.join(d, "*.mat"))))

        print(f"[DataLoader] 找到 {len(self.file_paths)} 个训练文件")

        # 预加载所有被试数据
        self.all_subjects = {}
        self._load_all_subjects()

    def _extract_subject_id(self, filepath: str) -> str:
        return os.path.basename(filepath).replace("timedata.mat", "")

    def _load_all_subjects(self):
        for fp in self.file_paths:
            sid = self._extract_subject_id(fp)
            features, labels = process_subject_train(
                fp, trial_length=12500,
                window_size=self.window_size, stride=self.stride,
                fs=self.fs, downsample_ratio=self.downsample_ratio
            )
            if features is not None and len(features) > 0:
                self.all_subjects[sid] = (features, labels)

        print(f"[DataLoader] 成功加载 {len(self.all_subjects)} 个被试")

    def get_subject_ids(self) -> List[str]:
        return sorted(self.all_subjects.keys())

    def folds(self):
        """
        留一法 (LOSO) / K折交叉验证生成器。

        n_folds=0 或 n_folds>=被试数 → 真正留一法: 每个被试轮流做 target
        否则 → K折: 每折一组被试做 target (只取第一个)

        每折: source_loader (domain=0, 有标签) + target_loader (domain=1, 有标签仅验证)
        """
        subject_ids = np.array(self.get_subject_ids())
        n_subjects = len(subject_ids)
        rng = np.random.RandomState(42)
        shuffled = rng.permutation(subject_ids)

        # 决定是 LOSO 还是 K-fold
        if self.n_folds <= 0 or self.n_folds >= n_subjects:
            # ---- 真正留一法: 每个被试做一次 target ----
            target_list = [[sid] for sid in subject_ids]
        else:
            # ---- K折: 每组被试取第一个做 target ----
            fold_size = n_subjects // self.n_folds
            target_list = []
            for fold in range(self.n_folds):
                start = fold * fold_size
                end = start + fold_size if fold < self.n_folds - 1 else n_subjects
                target_list.append([shuffled[start]])

        for fold, val_ids in enumerate(target_list):
            target_id = val_ids[0]
            train_ids = [s for s in subject_ids if s not in val_ids]

            # ---- 构建 Source (domain=0) ----
            src_features, src_labels = [], []
            for sid in train_ids:
                f, l = self.all_subjects[sid]
                src_features.append(f)
                src_labels.append(l)
            src_features = np.concatenate(src_features)
            src_labels = np.concatenate(src_labels)
            src_dataset = EEGDataset(src_features, src_labels, domain_label=0, subject_id="source")
            src_loader = DataLoader(src_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=True)

            # ---- 构建 Target (domain=1) ----
            tgt_features, tgt_labels = self.all_subjects[target_id]
            tgt_dataset = EEGDataset(tgt_features, tgt_labels, domain_label=1, subject_id=target_id)
            tgt_loader = DataLoader(tgt_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=True)

            print(f"[Fold {fold}] Source={len(train_ids)} subjects ({len(src_dataset)} windows), "
                  f"Target={target_id} ({len(tgt_dataset)} windows)")

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
        # 测试集用 dummy label=0, domain=1
        dataset = EEGDataset(f, np.zeros(len(f), dtype=int),
                             domain_label=1, subject_id=subject_id)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
