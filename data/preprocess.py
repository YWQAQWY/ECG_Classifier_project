"""
EEG 数据预处理模块
===================

处理流程:
1. Trial 切分: 训练集每12500采样点一个trial, 测试集每2500采样点一个trial
2. 滑动窗口: window=250 (1秒), stride=125 (0.5秒), 生成多个窗口
3. Z-score 归一化: 每个通道每个trial独立归一化
4. DE (微分熵) 特征提取: theta(4-8Hz), alpha(8-14Hz), beta(14-31Hz), gamma(31-45Hz)

DE 计算公式 (论文公式1):
    DE = 0.5 * log(2 * pi * e * sigma^2)
    其中 sigma^2 是带通滤波后信号的方差
    假设EEG信号服从高斯分布 N(mu, sigma^2)

参考:
- 论文 Section III-A: "Preprocessing and Feature Extraction"
- Duan et al., "Differential entropy feature for EEG-based emotion classification"
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat
import h5py
import os
from typing import Tuple, List, Dict, Optional


def _load_mat_file(filepath: str) -> dict:
    """
    加载 .mat 文件，自动检测格式。

    MATLAB .mat 文件有两种常见格式:
    - v5/v7 (<=R2006a): 使用 scipy.io.loadmat 读取
    - v7.3 (>=R2006b): 基于 HDF5, 使用 h5py 读取

    h5py 读取时的特殊处理:
    - MATLAB 以列优先存储, HDF5以行优先存储
    - h5py 读取的数组 shape = (n_samples, n_channels), 是转置后的
    - 需要显式转置回 (n_channels, n_samples)
    - 数据集需用 [:] 读取到内存 (h5py 默认返回引用)

    参数:
        filepath: .mat 文件路径

    返回:
        data_dict: {变量名: numpy数组} 字典
    """
    # 检测文件头
    with open(filepath, 'rb') as f:
        header = f.read(20)

    if b'MATLAB 5.0 MAT-file' in header:
        # ---- v5/v7 格式 (scipy) ----
        raw = loadmat(filepath)
        data_dict = {}
        for k, v in raw.items():
            if not k.startswith('__'):
                data_dict[k] = v
        return data_dict
    else:
        # ---- v7.3 格式 (HDF5) ----
        data_dict = {}
        with h5py.File(filepath, 'r') as f:
            for k in f.keys():
                arr = f[k][:]
                # h5py 读取的数组形状是 (n_samples, n_channels)
                # 转置为 (n_channels, n_samples) 以保持统一接口
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T  # (samples, channels) → (channels, samples)
                data_dict[k] = arr
        return data_dict


# ============================================================
# EEG 通道名称 (30通道, 10-20系统)
# ============================================================
CHANNEL_NAMES = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'OZ', 'O2'
]


# ============================================================
# 频段定义
# 论文使用: delta(1-3), theta(4-7), alpha(8-13), beta(14-30), gamma(31-35)
# 本数据集采样率为250Hz, 实际可用的频段:
#   theta: 4-8Hz, alpha: 8-14Hz, beta: 14-31Hz, gamma: 31-45Hz
# ============================================================
FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 45),
}


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                    fs: float = 250.0, order: int = 4) -> np.ndarray:
    """
    巴特沃斯带通滤波器。

    为什么需要带通滤波?
    - 提取特定频段的EEG信号 (theta/alpha/beta/gamma)
    - 不同频段对应不同的认知/情绪状态
    - 这是DE特征提取的前置步骤

    参数:
        data: 原始EEG数据, 支持 1D (n_samples,) 或 2D (n_channels, n_samples)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        fs: 采样率 (Hz), 默认250
        order: 滤波器阶数, 默认4

    返回:
        filtered: 滤波后数据, 维度与输入一致
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')

    # 处理1D或2D输入
    if data.ndim == 1:
        filtered = signal.filtfilt(b, a, data)
    else:
        filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered


def compute_de(data: np.ndarray) -> float:
    """
    计算微分熵 (Differential Entropy, DE) 特征。

    理论推导:
        假设EEG信号服从高斯分布 N(mu, sigma^2), 则:
        DE = -∫ f(x) log(f(x)) dx
           = 0.5 * log(2πeσ²)

    物理意义:
        DE 反映了信号的不确定性/复杂度。
        不同情绪状态下, 不同频段的DE值会发生变化。
        DE 能有效区分低频和高频EEG活动, 因此论文选择它作为特征。

    参数:
        data: 信号数据 (n_samples,)

    返回:
        de: DE特征值 (标量)
    """
    # 计算方差 σ²
    variance = np.var(data)
    # 防止方差为0 (log(0) = -inf) 或极小值导致数值不稳定
    variance = max(variance, 1e-12)
    # DE = 0.5 * log(2 * pi * e * σ²)
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return float(de)


def extract_de_features(eeg_segment: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    从EEG片段中提取DE特征。

    处理流程:
        对于每个通道 × 每个频段:
        1. 带通滤波 → 提取该频段信号
        2. 计算DE → 得到该频段的DE值
    最终输出: (n_channels, n_bands) = (30, 4)

    为什么对每个窗口分别提取DE?
    - EEG是非平稳信号, 不同时间段的统计特征不同
    - 滑动窗口 + DE = 时频表征, 比单一DE更能捕捉情绪动态

    参数:
        eeg_segment: EEG片段 (n_channels, n_samples) — 如 (30, 250)
        fs: 采样率

    返回:
        de_features: DE特征 (n_channels, n_bands) — 如 (30, 4)
    """
    n_channels = eeg_segment.shape[0]
    n_bands = len(FREQ_BANDS)
    de_features = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        for band_idx, (band_name, (low, high)) in enumerate(FREQ_BANDS.items()):
            # Step 1: 带通滤波 — 提取该频段的EEG信号
            ch_data = eeg_segment[ch, :]  # (n_samples,)
            filtered = bandpass_filter(ch_data, low, high, fs=fs)
            # Step 2: 计算DE — 得到该频段在该通道的DE值 (标量)
            de_features[ch, band_idx] = compute_de(filtered)

    return de_features


def split_into_trials(eeg_data: np.ndarray, trial_length: int) -> np.ndarray:
    """
    将连续EEG数据切分为trial。

    训练集:
        EEG_data_neu: (30, 50000) → 4个trial × 12500 samples
        EEG_data_pos: (30, 50000) → 4个trial × 12500 samples
    测试集:
        测试数据: (30, 20000) → 8个trial × 2500 samples

    参数:
        eeg_data: EEG数据 (n_channels, n_samples)
        trial_length: 每个trial的采样点数

    返回:
        trials: (n_trials, n_channels, trial_length)
    """
    n_channels, n_samples = eeg_data.shape
    n_trials = n_samples // trial_length

    # 只使用完整的trial, 丢弃末尾不足的部分
    usable_samples = n_trials * trial_length
    eeg_data = eeg_data[:, :usable_samples]

    trials = eeg_data.reshape(n_channels, n_trials, trial_length)
    trials = np.transpose(trials, (1, 0, 2))  # → (n_trials, n_channels, trial_length)
    return trials


def apply_sliding_window(trial: np.ndarray, window_size: int,
                         stride: int) -> np.ndarray:
    """
    对单个trial应用滑动窗口，生成多个窗口片段。

    为什么需要滑动窗口?
    - 单个trial很长 (训练:12500点=50秒, 测试:2500点=10秒)
    - 直接对整个trial提取DE会丢失时间动态信息
    - 滑动窗口将长trial分解为多个短片段，捕获情绪的时变特性
    - 窗口大小1秒 (250点) 是EEG分析的常用选择

    参数:
        trial: 单个trial (n_channels, trial_length)
        window_size: 窗口大小 (采样点数)
        stride: 滑动步长 (采样点数)

    返回:
        windows: (n_windows, n_channels, window_size)
    """
    n_channels, trial_length = trial.shape
    n_windows = max(0, (trial_length - window_size) // stride + 1)

    windows = np.zeros((n_windows, n_channels, window_size))
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = trial[:, start:end]

    return windows


def normalize_trial(trial: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    对trial进行归一化。

    为什么需要归一化?
    - 不同被试的EEG幅值范围差异很大 (个体差异)
    - 不同通道的幅值范围也不同
    - Z-score归一化使每个通道具有零均值和单位方差
    - 这有助于: ① 稳定训练 ② 减少个体差异 ③ MMD计算更稳定

    Z-score: x_norm = (x - mean) / std

    参数:
        trial: 单个trial (n_channels, n_samples)
        method: 归一化方法 ("zscore")

    返回:
        normalized: 归一化后的trial
    """
    if method == "zscore":
        # 每个通道独立归一化: 计算该通道在整个trial上的均值和标准差
        mean = np.mean(trial, axis=-1, keepdims=True)  # (n_channels, 1)
        std = np.std(trial, axis=-1, keepdims=True) + 1e-8  # 避免除零
        return (trial - mean) / std
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def process_mat_file(filepath: str, trial_length: int,
                     window_size: int = 250, stride: int = 125,
                     fs: float = 250.0, normalize: bool = True,
                     label: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理单个 .mat 文件，提取所有窗口的DE特征和标签。

    这是数据预处理的核心函数。完整流程:

    原始EEG (30, N)
      ↓ trial切分
    trials (n_trials, 30, trial_length)
      ↓ 归一化 (每个trial)
    trials_normalized
      ↓ 滑动窗口
    windows (n_windows_total, 30, window_size)
      ↓ DE特征提取 (每个窗口 → 每通道每频段)
    features (n_windows_total, 30, 4)  ← 最终特征

    参数:
        filepath: .mat 文件路径
        trial_length: trial长度 (训练=12500, 测试=2500)
        window_size: 窗口大小
        stride: 步长
        fs: 采样率
        normalize: 是否归一化
        label: 如果指定，所有trial使用此标签 (用于训练数据)

    返回:
        features: DE特征 (n_windows, n_channels, n_bands) = (n_windows, 30, 4)
        labels: 标签数组 (n_windows,)
    """
    data = _load_mat_file(filepath)

    # ---- 解析数据字段 ----
    if label is not None:
        # 训练数据: 单一情绪类型
        # 键名为 'EEG_data_neu' 或 'EEG_data_pos'
        eeg_key = "EEG_data_neu" if label == 0 else "EEG_data_pos"
        if eeg_key not in data:
            print(f"  [警告] 文件 {filepath} 中找不到键 {eeg_key}, 可用键: {list(data.keys())}")
            return None, None
        eeg = data[eeg_key]  # (30, N) — _load_mat_file已统一转置
    else:
        # 测试数据: 自动检测数据键
        if 'data' in data:
            eeg = data['data']
        else:
            # 尝试自动检测: 找第一个二维浮点数组
            for k in data.keys():
                if isinstance(data[k], np.ndarray) and data[k].ndim == 2:
                    eeg = data[k]
                    break
            else:
                raise ValueError(f"无法从文件 {filepath} 中读取EEG数据。可用键: {list(data.keys())}")

    # ---- Step 1: Trial 切分 ----
    trials = split_into_trials(eeg, trial_length)  # (n_trials, 30, trial_length)

    n_trials = trials.shape[0]
    all_features = []
    all_labels = []

    for t in range(n_trials):
        trial = trials[t]  # (30, trial_length)

        # ---- Step 2: Z-score 归一化 (每个trial独立) ----
        if normalize:
            trial = normalize_trial(trial, method="zscore")

        # ---- Step 3: 滑动窗口 ----
        windows = apply_sliding_window(trial, window_size, stride)
        # (n_windows_per_trial, 30, window_size)

        n_windows = windows.shape[0]
        if n_windows == 0:
            continue

        # ---- Step 4: DE特征提取 (每个窗口) ----
        de_features_window = np.zeros((n_windows, len(CHANNEL_NAMES), len(FREQ_BANDS)))
        for w in range(n_windows):
            de_features_window[w] = extract_de_features(windows[w], fs=fs)

        all_features.append(de_features_window)

        # 标签: 训练数据使用已知标签，测试数据暂用-1占位
        if label is not None:
            all_labels.append(np.full(n_windows, label))
        else:
            all_labels.append(np.full(n_windows, -1))

    if len(all_features) == 0:
        return None, None

    features = np.concatenate(all_features, axis=0)  # (total_windows, 30, 4)
    labels = np.concatenate(all_labels, axis=0)  # (total_windows,)

    return features, labels


def process_test_file(filepath: str, trial_length: int = 2500,
                      window_size: int = 250, stride: int = 125,
                      fs: float = 250.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理单个测试集 .mat 文件。

    测试集特殊处理:
    - 没有标签 (labels=-1)
    - trial_length = 2500 (10秒)
    - trial 顺序已被随机打乱 (⚠️ 竞赛设定)

    参数:
        filepath: .mat 文件路径
        trial_length: trial长度
        window_size: 窗口大小
        stride: 步长
        fs: 采样率

    返回:
        features: DE特征 (n_windows, 30, 4)
        labels: 全-1 (n_windows,)
    """
    return process_mat_file(
        filepath,
        trial_length=trial_length,
        window_size=window_size,
        stride=stride,
        fs=fs,
        normalize=True,
        label=None
    )


def process_train_file(filepath: str, trial_length: int = 12500,
                       window_size: int = 250, stride: int = 125,
                       fs: float = 250.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理单个训练集 .mat 文件。

    训练集每个被试包含两种情绪:
    - EEG_data_neu: neutral (label=0), 4个trial
    - EEG_data_pos: positive (label=1), 4个trial

    分别处理两种情绪数据后合并。

    参数:
        filepath: .mat 文件路径
        trial_length: trial长度
        window_size: 窗口大小
        stride: 步长
        fs: 采样率

    返回:
        features: DE特征 (n_windows_total, 30, 4)
        labels: 标签 (n_windows_total,)
    """
    # 处理 neutral 数据
    features_neu, labels_neu = process_mat_file(
        filepath, trial_length, window_size, stride, fs,
        normalize=True, label=0
    )

    # 处理 positive 数据
    features_pos, labels_pos = process_mat_file(
        filepath, trial_length, window_size, stride, fs,
        normalize=True, label=1
    )

    if features_neu is None and features_pos is None:
        return None, None
    if features_neu is None:
        return features_pos, labels_pos
    if features_pos is None:
        return features_neu, labels_neu

    # 合并两种情绪数据
    features = np.concatenate([features_neu, features_pos], axis=0)
    labels = np.concatenate([labels_neu, labels_pos], axis=0)

    return features, labels


def compute_window_count(trial_length: int, window_size: int, stride: int) -> int:
    """
    计算一个trial可以产生的窗口数量。

    参数:
        trial_length: trial的采样点数
        window_size: 窗口大小
        stride: 步长

    返回:
        窗口数量
    """
    return max(0, (trial_length - window_size) // stride + 1)


# ---- 调试工具 ----

def print_data_stats(features: np.ndarray, labels: np.ndarray, name: str = "Data"):
    """打印数据统计信息，用于调试和验证"""
    if features is None:
        print(f"[{name}] 无数据")
        return
    print(f"\n[{name}]")
    print(f"  特征形状: {features.shape}  (n_samples, n_channels, n_bands)")
    print(f"  标签形状: {labels.shape}")
    print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  特征均值: {features.mean():.4f}")
    print(f"  特征标准差: {features.std():.4f}")
    if len(labels) > 0 and labels[0] >= 0:
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            name_map = {0: "Neutral", 1: "Positive"}
            print(f"  类别 {u} ({name_map.get(u, '?')}): {c} 样本 ({c/len(labels)*100:.1f}%)")
