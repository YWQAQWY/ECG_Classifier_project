"""
EEG 数据预处理模块 (v2 — DANN + Transformer 版本)
===================================================

处理流程:
1. Trial 切分: 训练集每12500采样点一个trial, 测试集每2500采样点一个trial
2. 降采样: 每个被试随机保留 downsample_ratio 比例的窗口 (减少冗余)
3. 滑动窗口: window=250 (1秒), stride=125 (0.5秒)
4. Z-score 归一化: **每个被试独立归一化** (防止数据泄漏)
5. DE 特征提取: theta(4-8Hz), alpha(8-14Hz), beta(14-31Hz), gamma(31-45Hz)

输出: (n_windows, 30 channels, 4 bands) → Transformer 输入: 30 tokens × 4 dims
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat
import h5py
import os
from typing import Tuple, List, Dict, Optional

CHANNEL_NAMES = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'OZ', 'O2'
]

FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 45),
}


# ============================================================
# .mat 文件加载 (自动识别 v5 / v7.3 HDF5 格式)
# ============================================================
def _load_mat_file(filepath: str) -> dict:
    with open(filepath, 'rb') as f:
        header = f.read(20)
    if b'MATLAB 5.0 MAT-file' in header:
        raw = loadmat(filepath)
        return {k: v for k, v in raw.items() if not k.startswith('__')}
    else:
        data_dict = {}
        with h5py.File(filepath, 'r') as f:
            for k in f.keys():
                arr = f[k][:]
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T  # (samples, channels) → (channels, samples)
                data_dict[k] = arr
        return data_dict


# ============================================================
# 信号处理工具
# ============================================================
def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                    fs: float = 250.0, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data) if data.ndim == 1 else signal.filtfilt(b, a, data, axis=-1)


def compute_de(data: np.ndarray) -> float:
    """DE = 0.5 * log(2πeσ²) — 假设EEG服从高斯分布"""
    variance = max(np.var(data), 1e-12)
    return float(0.5 * np.log(2 * np.pi * np.e * variance))


def extract_de_features(eeg_segment: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    从EEG片段提取DE特征。
    输入: (30, n_samples)
    输出: (30, 4) — 每个通道4个频段的DE值
    """
    n_channels = eeg_segment.shape[0]
    de_features = np.zeros((n_channels, len(FREQ_BANDS)))
    for ch in range(n_channels):
        for band_idx, (_, (low, high)) in enumerate(FREQ_BANDS.items()):
            filtered = bandpass_filter(eeg_segment[ch, :], low, high, fs=fs)
            de_features[ch, band_idx] = compute_de(filtered)
    return de_features


def split_into_trials(eeg_data: np.ndarray, trial_length: int) -> np.ndarray:
    """切分trial: (30, N) → (n_trials, 30, trial_length)"""
    n_channels, n_samples = eeg_data.shape
    n_trials = n_samples // trial_length
    usable = n_trials * trial_length
    trials = eeg_data[:, :usable].reshape(n_channels, n_trials, trial_length)
    return np.transpose(trials, (1, 0, 2))


def apply_sliding_window(trial: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """滑动窗口: (30, trial_length) → (n_windows, 30, window_size)"""
    _, trial_length = trial.shape
    n_windows = max(0, (trial_length - window_size) // stride + 1)
    windows = np.zeros((n_windows, trial.shape[0], window_size))
    for i in range(n_windows):
        windows[i] = trial[:, i * stride : i * stride + window_size]
    return windows


def normalize_per_subject(features: np.ndarray) -> np.ndarray:
    """
    每个被试独立 Z-score 归一化。

    为什么独立归一化?
    - 不同被试的EEG幅值差异巨大
    - 如果混合归一化, 源域信息会泄漏到目标域
    - 独立归一化保证了跨被试评估的公平性

    参数:
        features: (n_windows, 30, 4) — 单个被试的所有窗口

    返回:
        normalized: 同形状
    """
    mean = features.mean(axis=0, keepdims=True)  # (1, 30, 4)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


# ============================================================
# 主处理函数
# ============================================================
def process_subject_train(filepath: str, trial_length: int = 12500,
                          window_size: int = 250, stride: int = 125,
                          fs: float = 250.0, downsample_ratio: float = 0.5,
                          seed: int = 42) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    处理单个训练被试的 .mat 文件。

    完整管线:
        .mat → trial切分 → 降采样 → 滑动窗口 → DE特征 → 归一化

    返回:
        features: (n_windows, 30, 4)
        labels:   (n_windows,)  0=neutral, 1=positive
    """
    data = _load_mat_file(filepath)

    all_features, all_labels = [], []

    for label, key in [(0, "EEG_data_neu"), (1, "EEG_data_pos")]:
        if key not in data:
            continue
        eeg = data[key]  # (30, N)

        trials = split_into_trials(eeg, trial_length)
        for trial in trials:
            windows = apply_sliding_window(trial, window_size, stride)
            if windows.shape[0] == 0:
                continue
            de_feats = np.array([extract_de_features(w, fs) for w in windows])
            all_features.append(de_feats)
            all_labels.append(np.full(de_feats.shape[0], label))

    if not all_features:
        return None, None

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # ---- 降采样 (减少冗余) ----
    if downsample_ratio < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = max(10, int(len(features) * downsample_ratio))
        indices = rng.choice(len(features), n_keep, replace=False)
        features = features[indices]
        labels = labels[indices]

    # ---- 每个被试独立归一化 ----
    features = normalize_per_subject(features)

    return features, labels


def process_subject_test(filepath: str, trial_length: int = 2500,
                         window_size: int = 250, stride: int = 125,
                         fs: float = 250.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """处理单个测试被试"""
    data = _load_mat_file(filepath)

    # 自动检测数据键
    eeg = None
    for k in data.keys():
        if isinstance(data[k], np.ndarray) and data[k].ndim == 2:
            eeg = data[k]
            break
    if eeg is None:
        return None, None

    trials = split_into_trials(eeg, trial_length)
    all_features = []
    for trial in trials:
        windows = apply_sliding_window(trial, window_size, stride)
        if windows.shape[0] > 0:
            de_feats = np.array([extract_de_features(w, fs) for w in windows])
            all_features.append(de_feats)

    if not all_features:
        return None, None

    features = np.concatenate(all_features, axis=0)
    features = normalize_per_subject(features)
    return features, np.full(len(features), -1)
