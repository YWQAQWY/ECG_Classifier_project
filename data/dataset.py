"""
PyTorch Dataset 模块
====================

实现跨被试EEG情绪识别的数据集加载和划分。

核心概念:
- Cross-Subject Split: 按subject(被试)划分训练/验证集
  - 不能随机打散所有样本再划分! 那会造成数据泄漏
  - 必须保证验证集的subject不出现在训练集中
  - 这样才能真正评估模型对"未见过被试"的泛化能力

- 训练模式: 多源域→单目标域
  - Source Domain: N-1个被试 (有标签)
  - Target Domain: 1个被试 (训练时无标签, 仅验证时用标签)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from typing import List, Tuple, Dict, Optional
from .preprocess import (
    process_train_file, process_test_file,
    compute_window_count, print_data_stats
)


class EEGSubjectDataset(Dataset):
    """
    单个被试的EEG数据集。

    加载一个 .mat 文件中的所有窗口DE特征，返回 (feature, label)。

    feature 形状: (30, 4, 1) — 适合EEGNet的2D卷积输入
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 subject_id: str = None):
        """
        参数:
            features: DE特征 (n_windows, 30, 4)
            labels: 标签 (n_windows,)
            subject_id: 被试ID (用于追踪)
        """
        self.subject_id = subject_id

        # 转换为 PyTorch tensor
        # EEGNet期望输入: (batch, 1, channels, time_or_freq)
        # 这里: (batch, 1, 30, 4) — channels=30, freq_bands=4
        self.features = torch.FloatTensor(features)
        # 增加 channel 维度: (n, 30, 4) → (n, 1, 30, 4)
        self.features = self.features.unsqueeze(1)

        self.labels = torch.LongTensor(labels.astype(np.int64))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.subject_id


class TestSubjectDataset(Dataset):
    """
    测试集被试的EEG数据集 (无真实标签)。
    """

    def __init__(self, features: np.ndarray, subject_id: str = None):
        """
        参数:
            features: DE特征 (n_windows, 30, 4)
            subject_id: 被试ID
        """
        self.subject_id = subject_id
        self.features = torch.FloatTensor(features).unsqueeze(1)  # (n, 1, 30, 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.subject_id


class CrossSubjectDataLoader:
    """
    跨被试数据加载器。

    核心功能:
    1. 扫描所有训练 .mat 文件
    2. 按subject划分source/target
    3. 为每个fold创建 DataLoader

    使用方法:
        loader = CrossSubjectDataLoader(data_root, config)
        for fold_idx, (src_loader, tgt_loader, tgt_subject) in enumerate(loader.folds()):
            # 训练: src_loader (有标签) + tgt_loader (无标签)
            # 验证: tgt_loader (有标签, 但训练时不使用)
            ...
    """

    def __init__(self, train_root: str, window_size: int = 250,
                 stride: int = 125, fs: float = 250.0,
                 batch_size: int = 64, n_folds: int = 5,
                 force_recompute: bool = False):
        """
        参数:
            train_root: 训练集根目录 (包含 '正常人/' 和 '抑郁症患者/')
            window_size: 窗口大小
            stride: 步长
            fs: 采样率
            batch_size: 批大小
            n_folds: 交叉验证折数 (0=使用全部数据训练, 不划分验证集)
            force_recompute: 是否强制重新计算特征
        """
        self.train_root = train_root
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.batch_size = batch_size
        self.n_folds = n_folds

        # 收集所有 .mat 文件路径
        self.file_paths = []
        healthy_dir = os.path.join(train_root, "正常人")
        depression_dir = os.path.join(train_root, "抑郁症患者")

        for d in [healthy_dir, depression_dir]:
            if os.path.exists(d):
                mat_files = sorted(glob.glob(os.path.join(d, "*.mat")))
                self.file_paths.extend(mat_files)

        print(f"[DataLoader] 找到 {len(self.file_paths)} 个训练文件")
        print(f"  正常人: {len(glob.glob(os.path.join(healthy_dir, '*.mat')))} 个")
        print(f"  抑郁症: {len(glob.glob(os.path.join(depression_dir, '*.mat')))} 个")

        # 预加载所有被试数据 (避免每次fold重复处理)
        self.all_subjects = {}  # {subject_id: (features, labels)}
        self._load_all_subjects()

    def _extract_subject_id(self, filepath: str) -> str:
        """从文件名提取被试ID, 如 'HC1005timedata.mat' → 'HC1005'"""
        basename = os.path.basename(filepath)
        # 移除 'timedata.mat' 后缀
        subject_id = basename.replace("timedata.mat", "")
        return subject_id

    def _load_all_subjects(self):
        """加载所有被试的数据到内存"""
        print("[DataLoader] 正在加载所有被试数据...")
        for fp in self.file_paths:
            subject_id = self._extract_subject_id(fp)
            print(f"  加载 {subject_id}...", end=" ")
            features, labels = process_train_file(
                fp,
                trial_length=12500,  # 训练集: 12500采样点/trial
                window_size=self.window_size,
                stride=self.stride,
                fs=self.fs
            )
            if features is not None:
                self.all_subjects[subject_id] = (features, labels)
                print(f"{features.shape[0]} 个窗口")
            else:
                print("跳过 (无数据)")
        print(f"[DataLoader] 成功加载 {len(self.all_subjects)} 个被试")

    def get_subject_ids(self) -> List[str]:
        """获取所有被试ID列表"""
        return sorted(self.all_subjects.keys())

    def folds(self):
        """
        生成器: 逐折返回 (source_loader, target_loader, target_subject_id)。

        每折: 留出一个被试作为target，其余作为source。

        n_folds=0 时: 返回全部数据作为一个fold (不划分验证集)
        """
        subject_ids = self.get_subject_ids()
        n_subjects = len(subject_ids)

        if self.n_folds <= 0:
            # 使用全部数据训练 (用于最终模型或测试集预测)
            all_features = []
            all_labels = []
            for sid in subject_ids:
                f, l = self.all_subjects[sid]
                all_features.append(f)
                all_labels.append(l)
            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            dataset = EEGSubjectDataset(all_features, all_labels, subject_id="all")
            loader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=0, drop_last=False)
            yield loader, None, "all"
            return

        # 随机排列被试顺序
        rng = np.random.RandomState(42)
        shuffled_ids = rng.permutation(subject_ids)

        # 划分fold
        fold_size = n_subjects // self.n_folds

        for fold_idx in range(self.n_folds):
            # 确定验证集被试
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < self.n_folds - 1 else n_subjects
            val_ids = list(shuffled_ids[start:end])
            train_ids = [sid for sid in subject_ids if sid not in val_ids]

            # 对于DDA，我们取验证集的第一个被试作为target
            # (简化: 每个fold只用1个被试验证)
            target_id = val_ids[0]
            source_ids = train_ids

            # ---- 构建source dataset (多源域, 有标签) ----
            src_features = []
            src_labels = []
            for sid in source_ids:
                f, l = self.all_subjects[sid]
                src_features.append(f)
                src_labels.append(l)
            src_features = np.concatenate(src_features, axis=0)
            src_labels = np.concatenate(src_labels, axis=0)

            src_dataset = EEGSubjectDataset(src_features, src_labels,
                                            subject_id="|".join(source_ids))
            src_loader = DataLoader(src_dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)

            # ---- 构建target dataset (单目标域, 标签仅用于验证) ----
            tgt_features, tgt_labels = self.all_subjects[target_id]
            tgt_dataset = EEGSubjectDataset(tgt_features, tgt_labels,
                                            subject_id=target_id)
            tgt_loader = DataLoader(tgt_dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)

            print(f"\n[Fold {fold_idx}] Source: {len(source_ids)} subjects, "
                  f"Target: {target_id}")
            print(f"  Source样本: {len(src_dataset)}")
            print(f"  Target样本: {len(tgt_dataset)}")

            yield src_loader, tgt_loader, target_id


class TestDataLoader:
    """
    测试集数据加载器。

    测试集特点:
    - 10个被试 (5健康 + 5抑郁)
    - 每个被试 30通道 × 20000采样点
    - trial顺序随机打乱
    - 无标签
    """

    def __init__(self, test_root: str, window_size: int = 250,
                 stride: int = 125, fs: float = 250.0,
                 batch_size: int = 64):
        """
        参数:
            test_root: 测试集根目录
            window_size: 窗口大小
            stride: 步长
            fs: 采样率
            batch_size: 批大小
        """
        self.test_root = test_root
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.batch_size = batch_size

        self.file_paths = sorted(glob.glob(os.path.join(test_root, "*.mat")))
        print(f"[TestDataLoader] 找到 {len(self.file_paths)} 个测试文件")

        # 预加载
        self.all_subjects = {}  # {subject_id: features}
        self._load_all_subjects()

    def _extract_subject_id(self, filepath: str) -> str:
        """从文件名提取被试ID, 如 'P_test1.mat' → 'P_test1'"""
        basename = os.path.basename(filepath)
        return basename.replace(".mat", "")

    def _load_all_subjects(self):
        """加载所有测试被试数据"""
        print("[TestDataLoader] 正在加载测试数据...")
        for fp in self.file_paths:
            subject_id = self._extract_subject_id(fp)
            print(f"  加载 {subject_id}...", end=" ")
            features, _ = process_test_file(
                fp,
                trial_length=2500,  # 测试集: 2500采样点/trial
                window_size=self.window_size,
                stride=self.stride,
                fs=self.fs
            )
            if features is not None:
                self.all_subjects[subject_id] = features
                print(f"{features.shape[0]} 个窗口")
            else:
                print("跳过 (无数据)")
        print(f"[TestDataLoader] 成功加载 {len(self.all_subjects)} 个测试被试")

    def get_loader(self, subject_id: str) -> DataLoader:
        """获取指定被试的 DataLoader"""
        if subject_id not in self.all_subjects:
            raise ValueError(f"被试 {subject_id} 不存在")
        features = self.all_subjects[subject_id]
        dataset = TestSubjectDataset(features, subject_id=subject_id)
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=0, drop_last=False)

    def get_subject_ids(self) -> List[str]:
        return sorted(self.all_subjects.keys())
