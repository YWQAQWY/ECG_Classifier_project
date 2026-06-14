import os
import h5py
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class EEGFeatureExtractor:
    """针对赛题四的EEG特征提取器"""

    def __init__(self, fs=250, window_length=4, overlap=0.5):
        self.fs = fs
        # 五个频带
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 49)
        }
        self.window_length = window_length  # 4秒窗口
        self.overlap = overlap
        self.window_size = int(window_length * fs)  # 1000个采样点
        self.step_size = int(self.window_size * (1 - overlap))  # 步长

    def differential_entropy(self, x):
        """计算微分熵 DE = 1/2 * log(2πeσ²)"""
        if len(x) < 2:
            return 0
        sigma = np.std(x)
        if sigma < 1e-10:
            return 0
        de = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
        return de

    def extract_features_from_trail(self, trail_data):
        """
        从一个trail（50秒数据）中提取DE特征（滑动窗口）

        参数:
        - trail_data: (12500, 30) 50秒数据

        返回:
        - features: (n_windows, 30*5=150) 每个窗口的特征
        """
        n_samples, n_channels = trail_data.shape
        # 计算滑动窗口数量
        n_windows = ((n_samples - self.window_size) // self.step_size) + 1

        # 存储特征 (n_windows, n_channels, n_bands)
        features_3d = np.zeros((n_windows, n_channels, len(self.freq_bands)))

        for band_idx, (band_name, (low_freq, high_freq)) in enumerate(self.freq_bands.items()):
            nyquist = self.fs / 2
            b, a = butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band')

            for ch in range(n_channels):
                channel_data = trail_data[:, ch]

                for w in range(n_windows):
                    start = w * self.step_size
                    end = start + self.window_size
                    window_data = channel_data[start:end]

                    try:
                        filtered = filtfilt(b, a, window_data)
                    except:
                        filtered = window_data

                    de = self.differential_entropy(filtered)
                    features_3d[w, ch, band_idx] = de

        # 展平
        features = features_3d.reshape(n_windows, -1)
        return features

    def apply_lds_smoothing(self, features):
        """LDS平滑"""
        alpha = 0.8
        smoothed = np.zeros_like(features)
        smoothed[0] = features[0]
        for t in range(1, len(features)):
            smoothed[t] = alpha * features[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed


def extract_all_features(data_path, output_path):
    """
    提取所有被试的特征

    每个被试输出:
    - 8个trail
    - 每个trail有 n_windows 个窗口
    - 每个窗口150维特征
    """

    extractor = EEGFeatureExtractor(fs=250, window_length=4, overlap=0.5)

    # 计算每个trail的窗口数
    trail_samples = 12500  # 50秒 * 250Hz
    n_windows_per_trail = ((trail_samples - extractor.window_size) // extractor.step_size) + 1
    print(f"特征提取参数:")
    print(f"  采样率: {extractor.fs} Hz")
    print(f"  窗口长度: {extractor.window_length} 秒")
    print(f"  重叠率: {extractor.overlap * 100}%")
    print(f"  每段视频窗口数: {n_windows_per_trail}")
    print(f"  每个窗口特征维度: 30通道 × 5频带 = 150维")
    print(f"  每个被试总窗口数: 8段 × {n_windows_per_trail} = {8 * n_windows_per_trail}")

    # 存储所有数据
    all_features = []  # 每个窗口的特征
    all_labels = []  # 每个窗口的标签 (0=中性, 1=积极)
    all_subjects = []  # 每个窗口对应的被试ID
    all_trail_ids = []  # 每个窗口对应的trail ID (1-8)
    all_types = []  # 正常人/抑郁症患者

    # 处理正常人和抑郁症患者
    for group_name in ['正常人', '抑郁症患者']:
        folder_path = os.path.join(data_path, group_name)
        if not os.path.exists(folder_path):
            print(f"警告: {folder_path} 不存在")
            continue

        mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        print(f"\n处理 {group_name}: 找到 {len(mat_files)} 个被试")

        for file_idx, file_name in enumerate(mat_files):
            file_path = os.path.join(folder_path, file_name)
            subject_id = file_name.replace('timedata.mat', '')

            print(f"  处理被试 {file_idx + 1}/{len(mat_files)}: {subject_id}")

            with h5py.File(file_path, 'r') as f:
                # ===== 处理中性情绪 (4段视频，标签=0) =====
                if 'EEG_data_neu' in f:
                    neu_data = f['EEG_data_neu'][:]  # (50000, 30)
                    # 分成4个trail，每个12500采样点
                    trail_samples = 12500
                    for trail_idx in range(4):
                        start = trail_idx * trail_samples
                        end = start + trail_samples
                        trail_data = neu_data[start:end, :]  # (12500, 30)

                        # 提取特征
                        features = extractor.extract_features_from_trail(trail_data)
                        features_smoothed = extractor.apply_lds_smoothing(features)

                        n_windows = features_smoothed.shape[0]
                        for w_idx in range(n_windows):
                            all_features.append(features_smoothed[w_idx])
                            all_labels.append(0)  # 中性
                            all_subjects.append(subject_id)
                            all_trail_ids.append(trail_idx + 1)  # trail 1-4
                            all_types.append(group_name)

                        print(f"    中性视频{trail_idx + 1}: {n_windows}个窗口")

                # ===== 处理积极情绪 (4段视频，标签=1) =====
                if 'EEG_data_pos' in f:
                    pos_data = f['EEG_data_pos'][:]  # (50000, 30)
                    trail_samples = 12500
                    for trail_idx in range(4):
                        start = trail_idx * trail_samples
                        end = start + trail_samples
                        trail_data = pos_data[start:end, :]  # (12500, 30)

                        # 提取特征
                        features = extractor.extract_features_from_trail(trail_data)
                        features_smoothed = extractor.apply_lds_smoothing(features)

                        n_windows = features_smoothed.shape[0]
                        for w_idx in range(n_windows):
                            all_features.append(features_smoothed[w_idx])
                            all_labels.append(1)  # 积极
                            all_subjects.append(subject_id)
                            all_trail_ids.append(trail_idx + 5)  # trail 5-8
                            all_types.append(group_name)

                        print(f"    积极视频{trail_idx + 1}: {n_windows}个窗口")

    # 转换为numpy数组
    X = np.array(all_features)
    y = np.array(all_labels)
    subjects = np.array(all_subjects)
    trail_ids = np.array(all_trail_ids)
    types = np.array(all_types)

    print(f"\n{'=' * 60}")
    print(f"特征提取完成!")
    print(f"{'=' * 60}")
    print(f"总窗口样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"积极情绪(1): {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"中性情绪(0): {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"总被试数: {len(np.unique(subjects))}")
    print(f"总trail数: {len(np.unique(trail_ids))}")

    # 按类型统计
    print(f"\n按类型统计:")
    for type_name in ['正常人', '抑郁症患者']:
        mask = types == type_name
        print(f"  {type_name}: {np.sum(mask)} 个窗口样本")
        print(f"    积极: {np.sum((y == 1) & mask)}")
        print(f"    中性: {np.sum((y == 0) & mask)}")

    # 按trail统计标签分布
    print(f"\n按trail统计 (每个被试):")
    for trail_id in range(1, 9):
        if trail_id <= 4:
            expected_label = 0
            label_name = "中性"
        else:
            expected_label = 1
            label_name = "积极"

        # 检查每个被试的该trail是否标签正确
        correct_count = 0
        total_count = 0
        for subj in np.unique(subjects):
            mask = (subjects == subj) & (trail_ids == trail_id)
            if np.any(mask):
                total_count += 1
                trail_label = y[mask][0]
                if trail_label == expected_label:
                    correct_count += 1

        print(f"  Trail {trail_id} ({label_name}): {correct_count}/{total_count} 被试标签正确")

    # 标准化特征（全局标准化，因为每个被试已经单独归一化过？不对，这里先不做，留到后续）
    # 标准化
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # 保存特征
    np.savez(output_path,
             features=X_normalized,
             labels=y,
             subjects=subjects,
             trail_ids=trail_ids,
             types=types,
             n_windows_per_trail=n_windows_per_trail)

    print(f"\n特征已保存到: {output_path}")

    return X_normalized, y, subjects, trail_ids, types


if __name__ == "__main__":
    # 设置路径
    base_path = os.path.join(os.path.dirname(__file__), '训练集')
    output_path = os.path.join(os.path.dirname(__file__), 'eeg_features_trail_based.npz')

    X, y, subjects, trail_ids, types = extract_all_features(base_path, output_path)

    print(f"\n数据形状验证:")
    print(f"  每个被试应有 {8 * 25} = 200个窗口 (当overlap=0.75时)")
    print(f"  实际每个被试平均窗口数: {len(X) / len(np.unique(subjects)):.1f}")