"""
测试集推理脚本
==============

对竞赛测试集进行情绪预测。

测试集特点:
  - 10个被试 (5健康 + 5抑郁)
  - 每个被试 30通道 × 20000采样点
  - 8个trial, 每个2500采样点
  - trial顺序被随机打乱!
  - 无真实标签

输出格式:
  按照竞赛要求输出预测结果。
  每个被试的8个trial预测为 neutral(0) 或 positive(1)。

用法:
  cd /home/yanwq/ECG
  python -m scripts.test --checkpoint checkpoints/final_model.pth
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from data.dataset import TestDataLoader
from models.eegnet import EEGNet
from models.classifier import EmotionClassifier


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    """构建模型 (与训练时一致)"""
    model_cfg = config.get('model', {})
    n_channels = config.get('data', {}).get('n_channels', 30)
    n_bands = len(config.get('data', {}).get('frequency_bands', {}))
    feature_dim = model_cfg.get('classifier', {}).get('feature_dim', 64)
    n_classes = model_cfg.get('classifier', {}).get('n_classes', 2)

    eegnet_cfg = model_cfg.get('eegnet', {})
    encoder = EEGNet(
        n_channels=n_channels,
        n_bands=n_bands,
        F1=eegnet_cfg.get('F1', 8),
        D=eegnet_cfg.get('D', 2),
        F2=eegnet_cfg.get('F2', 16),
        feature_dim=feature_dim,
        dropout_rate=eegnet_cfg.get('dropout_rate', 0.5),
    )
    classifier = EmotionClassifier(
        feature_dim=feature_dim,
        n_classes=n_classes,
    )
    return encoder, classifier


def aggregate_window_predictions(window_preds: np.ndarray,
                                 windows_per_trial: int,
                                 n_trials: int = 8) -> np.ndarray:
    """
    将窗口级预测聚合为trial级预测。

    测试集: 每个trial = 2500采样点, 窗口大小=250, stride=125
    windows_per_trial = (2500 - 250) // 125 + 1 = 19

    聚合策略: 多数投票 (majority voting)
    - 每个窗口预测一个标签
    - 该trial内所有窗口中出现次数最多的标签 = trial预测

    参数:
        window_preds: 窗口级预测 (n_windows_total,)
        windows_per_trial: 每个trial的窗口数
        n_trials: trial总数

    返回:
        trial_preds: trial级预测 (n_trials,)
    """
    trial_preds = []
    for t in range(n_trials):
        start = t * windows_per_trial
        end = start + windows_per_trial
        if end > len(window_preds):
            end = len(window_preds)
        trial_windows = window_preds[start:end]
        if len(trial_windows) == 0:
            # 随机预测
            trial_preds.append(np.random.randint(0, 2))
            continue
        # 多数投票
        counts = np.bincount(trial_windows, minlength=2)
        trial_preds.append(np.argmax(counts))
    return np.array(trial_preds)


def main():
    parser = argparse.ArgumentParser(description="DDA 测试集推理")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='预测结果输出路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    set_seed(config.get('experiment', {}).get('seed', 42))

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 构建模型
    encoder, classifier = build_model(config)
    encoder.to(device)
    classifier.to(device)

    # 加载权重
    print(f"加载模型权重: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    print(f"  模型来自 epoch {checkpoint.get('epoch', 'unknown')}")

    encoder.eval()
    classifier.eval()

    # 构建测试数据加载器
    data_cfg = config.get('data', {})
    test_loader = TestDataLoader(
        test_root=data_cfg.get('test_root', './赛题四数据集及说明文档/公开测试集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=config.get('training', {}).get('batch_size', 64),
    )

    # 计算每个trial的窗口数
    trial_length = data_cfg.get('test_trial_length', 2500)
    window_size = data_cfg.get('window_size', 250)
    stride = data_cfg.get('stride', 125)
    windows_per_trial = max(0, (trial_length - window_size) // stride + 1)
    print(f"每个trial窗口数: {windows_per_trial}")

    # 推理
    all_predictions = {}  # {subject_id: trial_predictions}

    for subject_id in test_loader.get_subject_ids():
        print(f"\n推理 {subject_id}...")
        loader = test_loader.get_loader(subject_id)

        window_preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                features = encoder(x)
                logits = classifier(features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                window_preds.append(preds.cpu().numpy())

        window_preds = np.concatenate(window_preds)

        # 聚合为trial预测
        n_trials = 8  # 测试集每个被试8个trial
        trial_preds = aggregate_window_predictions(
            window_preds, windows_per_trial, n_trials
        )

        all_predictions[subject_id] = trial_preds
        print(f"  预测: {trial_preds} (0=Neutral, 1=Positive)")
        print(f"  Neutral: {(trial_preds == 0).sum()}, Positive: {(trial_preds == 1).sum()}")

    # 保存结果
    print(f"\n保存预测结果至: {args.output}")
    with open(args.output, 'w') as f:
        f.write("Subject,Trial1,Trial2,Trial3,Trial4,Trial5,Trial6,Trial7,Trial8\n")
        for subject_id in sorted(all_predictions.keys()):
            preds = all_predictions[subject_id]
            f.write(f"{subject_id}," + ",".join(map(str, preds)) + "\n")

    print("推理完成!")

    # 打印汇总
    print("\n" + "=" * 60)
    print("预测汇总:")
    print("=" * 60)
    for subject_id in sorted(all_predictions.keys()):
        preds = all_predictions[subject_id]
        n_pos = (preds == 1).sum()
        n_neu = (preds == 0).sum()
        print(f"  {subject_id}: {n_neu} Neutral, {n_pos} Positive")


if __name__ == "__main__":
    main()
