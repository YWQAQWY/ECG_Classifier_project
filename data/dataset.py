"""
PyTorch Dataset (v2 — DANN + Transformer 版本)
================================================

预处理管线 (每个被试独立):
  1. 降采样 (可选): downsample.enabled=true → 随机保留 ratio
  2. Z-score 归一化: 每个被试独立 (防数据泄漏)

源域扩充 (CV之前一次性):
  3. 集成学习: 全部60人训练 SVM+RF+MLP+LR → 测试集投票 → 全票通过加入源域
  4. 扩充后降采样 (可选): 再次降采样减少冗余
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os, glob
from typing import List, Tuple, Optional
from .preprocess import process_subject_train, process_subject_test
from ensemble.pseudo_labeler import EnsemblePseudoLabeler


class EEGDataset(Dataset):
    """输出: (features, emotion_label, domain_label, subject_id)
    features: (30, 4) — 30 tokens × 4 dims"""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 domain_label: int, subject_id: str):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels.astype(int))
        self.domain_label = domain_label
        self.subject_id = subject_id

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx],
                self.domain_label, self.subject_id)


class CrossSubjectDataLoader:
    """跨被试数据加载器 (纯 K 折划分, 不含 ensemble)。"""

    def __init__(self, all_subjects: dict, batch_size: int = 128,
                 n_folds: int = 5):
        self.all_subjects = all_subjects  # {sid: (features, labels)}
        self.batch_size = batch_size
        self.n_folds = n_folds

    def get_subject_ids(self) -> List[str]:
        return sorted(self.all_subjects.keys())

    def folds(self):
        """K折 / LOSO 交叉验证生成器。"""
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
                s = fold * fold_size
                e = s + fold_size if fold < self.n_folds - 1 else n_subjects
                target_list.append([shuffled[s]])

        for fold, val_ids in enumerate(target_list):
            target_id = val_ids[0]
            train_ids = [s for s in subject_ids if s not in val_ids]

            src_features, src_labels = [], []
            for sid in train_ids:
                f, l = self.all_subjects[sid]
                src_features.append(f)
                src_labels.append(l)
            src_features = np.concatenate(src_features)
            src_labels = np.concatenate(src_labels)

            tgt_features, tgt_labels = self.all_subjects[target_id]

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


# ============================================================
# 预处理函数: 加载 + 扩充源域
# ============================================================

def load_all_train_subjects(train_root: str, window_size: int = 250,
                            stride: int = 125, fs: float = 250.0,
                            downsample_enabled: bool = True,
                            downsample_ratio: float = 0.5) -> tuple:
    """
    加载全部训练被试, 每人独立降采样 + zscore 归一化。

    返回:
        all_subjects: {subject_id: (features, emotion_labels)}
        depression_labels: {subject_id: 0=healthy, 1=depressed}
    """
    file_paths = []
    for subdir in ["正常人", "抑郁症患者"]:
        d = os.path.join(train_root, subdir)
        if os.path.exists(d):
            for fp in sorted(glob.glob(os.path.join(d, "*.mat"))):
                file_paths.append((fp, subdir))

    print(f"[加载] 找到 {len(file_paths)} 个训练文件")

    all_subjects = {}
    depression_labels = {}
    for i, (fp, subdir) in enumerate(file_paths):
        sid = os.path.basename(fp).replace("timedata.mat", "")
        print(f"\r[加载] [{i+1}/{len(file_paths)}] {sid}", end="", flush=True)

        ratio = downsample_ratio if downsample_enabled else 1.0
        features, labels = process_subject_train(
            fp, trial_length=12500, window_size=window_size,
            stride=stride, fs=fs, downsample_ratio=ratio,
        )
        if features is not None and len(features) > 0:
            all_subjects[sid] = (features, labels)
            depression_labels[sid] = 0 if subdir == "正常人" else 1

    n_healthy = sum(1 for v in depression_labels.values() if v == 0)
    n_depressed = sum(1 for v in depression_labels.values() if v == 1)
    ds_status = f"降采样={downsample_ratio}" if downsample_enabled else "无降采样"
    print(f"\r[加载] 完成 {len(all_subjects)} 个被试 "
          f"(健康={n_healthy}, 抑郁={n_depressed}, {ds_status}, 每人独立zscore)")
    return all_subjects, depression_labels


def expand_source_with_ensemble(all_subjects: dict, test_root: str,
                                 window_size: int = 250, stride: int = 125,
                                 fs: float = 250.0,
                                 confidence_threshold: float = 0.8,
                                 post_downsample_enabled: bool = True,
                                 post_downsample_ratio: float = 0.5,
                                 seed: int = 42) -> dict:
    """
    集成学习扩充源域。

    1. 全部训练被试 → 训练 SVM×2+RF+LR+MLP
    2. 对测试集投票 → 全票通过 → 加入源域
    3. (可选) 扩充后降采样

    返回: 更新后的 all_subjects (含扩充的伪标签数据)
    """
    print(f"\n{'='*60}")
    print("源域扩充: 集成学习伪标签")
    print(f"{'='*60}")

    # 合并全部源域数据
    all_feats, all_labels = [], []
    for sid in sorted(all_subjects.keys()):
        f, l = all_subjects[sid]
        all_feats.append(f)
        all_labels.append(l)
    X_src = np.concatenate(all_feats).reshape(-1, 30 * 4)
    y_src = np.concatenate(all_labels)
    print(f"源域: {len(X_src)} windows ({len(all_subjects)} 被试)")

    # 加载测试集
    test_files = sorted(glob.glob(os.path.join(test_root, "*.mat")))
    test_data = {}
    for fp in test_files:
        sid = os.path.basename(fp).replace(".mat", "")
        f, _ = process_subject_test(fp, trial_length=2500,
                                     window_size=window_size,
                                     stride=stride, fs=fs)
        if f is not None:
            test_data[sid] = f
    print(f"测试集: {len(test_data)} 被试")

    # 训练集成分类器
    ensemble = EnsemblePseudoLabeler(confidence_threshold=confidence_threshold)
    ensemble.fit(X_src, y_src)

    # 对每个测试被试投票
    total_added = 0
    for sid in sorted(test_data.keys()):
        X_tgt = test_data[sid]
        X_flat = X_tgt.reshape(-1, 30 * 4)
        good_idx, good_labels = ensemble.predict_and_filter(X_flat)

        if len(good_idx) > 0:
            # 将全票通过的测试样本加入源域 (作为新的"虚拟被试")
            pseudo_features = X_tgt[good_idx]
            pseudo_sid = f"pseudo_{sid}"
            all_subjects[pseudo_sid] = (pseudo_features, good_labels.astype(float))
            total_added += len(good_idx)
            print(f"  {sid}: 全票通过 {len(good_idx)}/{len(X_tgt)} 窗口 "
                  f"(N={(good_labels==0).sum()}, P={(good_labels==1).sum()})")
        else:
            print(f"  {sid}: 无全票通过样本")

    # 扩充后降采样
    if post_downsample_enabled and total_added > 0:
        rng = np.random.RandomState(seed)
        for sid in list(all_subjects.keys()):
            f, l = all_subjects[sid]
            n_keep = max(10, int(len(f) * post_downsample_ratio))
            if len(f) > n_keep:
                idx = rng.choice(len(f), n_keep, replace=False)
                all_subjects[sid] = (f[idx], l[idx])

    n_total = sum(len(v[0]) for v in all_subjects.values())
    ds_status = "降采样" if post_downsample_enabled else "无降采样"
    print(f"扩充后源域: {len(all_subjects)} subjects, {n_total} windows ({ds_status})")
    print(f"{'='*60}")

    return all_subjects
