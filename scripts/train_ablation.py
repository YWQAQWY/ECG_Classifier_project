"""
消融实验脚本 (Ablation Study)
==============================

对比三种配置:
  1. CE Only (Baseline)         — 无域适应, 仅源域监督学习
  2. CE + GDD                   — 全局域分布对齐 (传统域适应)
  3. CE + GDD + LSD (完整DDA)   — 论文提出的动态域适应

消融实验目的:
  - 验证GDD的有效性 (相比纯CE)
  - 验证LSD的增益 (相比仅GDD)
  - 验证DDA的完整效果

评估指标:
  - Target Domain Accuracy (最核心!)
  - Confusion Matrix (各类别分类情况)
  - t-SNE Feature Visualization (特征对齐可视化)

论文 Table III 对应:
  - "Baseline" = CE Only
  - "w/o. LSD" = CE + GDD
  - "w/o. GDD" = CE + LSD (我们也可以测试)
  - "Full DDA" = CE + GDD + LSD
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logger import Logger
from utils.metrics import compute_all_metrics, print_metrics
from data.dataset import CrossSubjectDataLoader
from models.eegnet import EEGNet
from models.classifier import EmotionClassifier
from trainers.trainer import DDATrainer


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_experiment(config: dict, mode: str, data_loader: CrossSubjectDataLoader,
                   logger: Logger) -> list:
    """
    运行单个实验模式。

    参数:
        config: 配置
        mode: 实验模式 ('ce_only', 'ce_gdd', 'ce_gdd_lsd')
        data_loader: 数据加载器
        logger: 日志器

    返回:
        results: 各fold的结果列表
    """
    # 覆盖实验模式
    config = config.copy()
    config['experiment'] = config.get('experiment', {}).copy()
    config['experiment']['mode'] = mode

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})

    n_channels = config.get('data', {}).get('n_channels', 30)
    n_bands = len(config.get('data', {}).get('frequency_bands', {}))
    feature_dim = model_cfg.get('classifier', {}).get('feature_dim', 64)
    n_classes = model_cfg.get('classifier', {}).get('n_classes', 2)

    all_results = []

    for fold_idx, (src_loader, tgt_loader, target_id) in enumerate(data_loader.folds()):
        if tgt_loader is None:
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"[{mode}] Fold {fold_idx}: Target={target_id}")
        logger.info(f"{'='*60}")

        # 构建新模型 (每fold独立初始化)
        encoder = EEGNet(
            n_channels=n_channels, n_bands=n_bands,
            F1=model_cfg.get('eegnet', {}).get('F1', 8),
            D=model_cfg.get('eegnet', {}).get('D', 2),
            F2=model_cfg.get('eegnet', {}).get('F2', 16),
            feature_dim=feature_dim,
            dropout_rate=model_cfg.get('eegnet', {}).get('dropout_rate', 0.5),
        )
        classifier = EmotionClassifier(
            feature_dim=feature_dim, n_classes=n_classes
        )

        trainer = DDATrainer(encoder, classifier, config, logger)
        result = trainer.train_fold(src_loader, tgt_loader, target_id, fold_idx)
        all_results.append(result)

        # 有限折数 (消融实验不需全跑)
        if fold_idx >= 2:  # 跑3折即可
            break

    return all_results


def main():
    parser = argparse.ArgumentParser(description="DDA 消融实验")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.debug:
        config['training']['epochs'] = 30
        config['training']['n_folds'] = 2

    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    # 日志
    exp_name = config.get('experiment', {}).get('name', 'ablation')
    logger = Logger(log_dir='./logs', experiment_name=f'ablation_{exp_name}')

    logger.info("=" * 60)
    logger.info("DDA 消融实验 (Ablation Study)")
    logger.info("=" * 60)

    # 构建数据加载器
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})

    data_loader = CrossSubjectDataLoader(
        train_root=data_cfg.get('train_root', './赛题四数据集及说明文档/训练集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=train_cfg.get('batch_size', 64),
        n_folds=train_cfg.get('n_folds', 5),
    )

    # ---- 运行三种消融实验 ----
    modes = ['ce_only', 'ce_gdd', 'ce_gdd_lsd']
    all_experiment_results = {}

    for mode in modes:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 实验: {mode}")
        logger.info(f"{'#'*60}")

        results = run_experiment(config, mode, data_loader, logger)
        all_experiment_results[mode] = results

        # 打印该模式的汇总
        accuracies = [r['best_accuracy'] for r in results]
        logger.info(f"\n[{mode}] 汇总:")
        logger.info(f"  各折准确率: {[f'{a:.4f}' for a in accuracies]}")
        logger.info(f"  平均: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

    # ---- 最终对比 ----
    logger.info("\n" + "=" * 60)
    logger.info("消融实验最终对比 (Ablation Study Final Results)")
    logger.info("=" * 60)

    # 表格头
    logger.info(f"{'模式':<25} {'平均准确率':<15} {'标准差':<10}")
    logger.info("-" * 50)

    for mode in modes:
        accuracies = [r['best_accuracy'] for r in all_experiment_results[mode]]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        logger.info(f"{mode:<25} {mean_acc:.4f}          {std_acc:.4f}")

    logger.info("-" * 50)

    # 计算提升
    baseline_acc = np.mean([r['best_accuracy'] for r in all_experiment_results['ce_only']])
    gdd_acc = np.mean([r['best_accuracy'] for r in all_experiment_results['ce_gdd']])
    dda_acc = np.mean([r['best_accuracy'] for r in all_experiment_results['ce_gdd_lsd']])

    logger.info(f"\nGDD 相对 Baseline 提升: {gdd_acc - baseline_acc:+.4f} ({(gdd_acc - baseline_acc) / baseline_acc * 100:+.1f}%)")
    logger.info(f"DDA 相对 Baseline 提升: {dda_acc - baseline_acc:+.4f} ({(dda_acc - baseline_acc) / baseline_acc * 100:+.1f}%)")
    logger.info(f"DDA 相对 GDD 提升:     {dda_acc - gdd_acc:+.4f} ({(dda_acc - gdd_acc) / gdd_acc * 100:+.1f}%)")

    logger.info("\n消融实验完成!")
    logger.close()


if __name__ == "__main__":
    main()
