"""
DDA 训练入口脚本
================

基于 Dynamic Domain Adaptation (DDA) 的跨被试EEG情绪识别训练。

用法:
    cd /home/yanwq/ECG
    python -m scripts.train

    # 或指定配置文件:
    python -m scripts.train --config configs/config.yaml

    # 调试模式 (小数据, 少epoch):
    python -m scripts.train --debug

训练流程:
    1. 加载配置
    2. 固定随机种子 (保证可复现)
    3. 构建数据加载器 (Cross-Subject Split)
    4. 构建模型 (EEGNet + Classifier)
    5. 逐折训练+验证 (Leave-One-Subject-Out)
    6. 汇总结果
    7. 保存最终模型

核心概念重申:
    - Source Domain: N-1个被试 (有标签) → 用于CE监督训练
    - Target Domain: 1个被试 (标签仅用于评估) → 评估泛化能力
    - GDD: 对齐源域和目标域的整体特征分布 (无监督)
    - LSD: 在类别层面拉近同类、推远异类 (伪标签半监督)
    - Dynamic α: 从粗到细的渐进对齐策略
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logger import Logger
from data.dataset import CrossSubjectDataLoader
from models.eegnet import EEGNet
from models.encoder import MLPEncoder
from models.classifier import EmotionClassifier
from trainers.trainer import DDATrainer


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict):
    """
    根据配置构建编码器和分类器。

    返回:
        encoder, classifier
    """
    model_cfg = config.get('model', {})
    model_name = model_cfg.get('name', 'eegnet')

    n_channels = config.get('data', {}).get('n_channels', 30)
    n_bands = len(config.get('data', {}).get('frequency_bands', {}))
    feature_dim = model_cfg.get('classifier', {}).get('feature_dim', 64)
    n_classes = model_cfg.get('classifier', {}).get('n_classes', 2)

    if model_name == 'eegnet':
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
    elif model_name == 'mlp':
        mlp_cfg = model_cfg.get('mlp', {})
        encoder = MLPEncoder(
            input_dim=n_channels * n_bands,  # 30*4=120
            hidden_dims=mlp_cfg.get('hidden_dims', [512, 128, 128, 64]),
            feature_dim=feature_dim,
            dropout_rate=mlp_cfg.get('dropout_rate', 0.5),
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    classifier = EmotionClassifier(
        feature_dim=feature_dim,
        n_classes=n_classes,
    )

    return encoder, classifier


def main():
    parser = argparse.ArgumentParser(description="DDA 跨被试EEG情绪识别训练")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式 (少被试, 少epoch)')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['ce_only', 'ce_gdd', 'ce_gdd_lsd'],
                        help='实验模式 (覆盖配置文件)')
    parser.add_argument('--folds', type=int, default=None,
                        help='交叉验证折数 (覆盖配置文件)')
    args = parser.parse_args()

    # ---- 加载配置 ----
    config = load_config(args.config)

    # 调试模式覆盖
    if args.debug:
        print("=" * 60)
        print("调试模式: 使用简化参数")
        print("=" * 60)
        config['training']['epochs'] = 20
        config['training']['n_folds'] = 2
        config['experiment']['eval_interval'] = 1
        config['experiment']['log_interval'] = 1

    if args.mode is not None:
        config['experiment']['mode'] = args.mode

    if args.folds is not None:
        config['training']['n_folds'] = args.folds

    # ---- 固定随机种子 (保证可复现!) ----
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    # ---- 初始化日志 ----
    exp_cfg = config.get('experiment', {})
    logger = Logger(
        log_dir=exp_cfg.get('log_dir', './logs'),
        experiment_name=exp_cfg.get('name', 'dda_eeg')
    )
    logger.info("=" * 60)
    logger.info("DDA 跨被试EEG情绪识别训练")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"实验模式: {config.get('experiment', {}).get('mode', 'ce_gdd_lsd')}")
    logger.info(f"随机种子: {seed}")
    logger.info(f"模型类型: {config.get('model', {}).get('name', 'eegnet')}")

    # ---- 构建数据加载器 ----
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})

    logger.info("\n" + "-" * 40)
    logger.info("构建数据加载器...")
    logger.info("-" * 40)

    data_loader = CrossSubjectDataLoader(
        train_root=data_cfg.get('train_root', './赛题四数据集及说明文档/训练集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=train_cfg.get('batch_size', 64),
        n_folds=train_cfg.get('n_folds', 5),
        force_recompute=data_cfg.get('force_recompute', False),
    )

    # ---- 构建模型 ----
    logger.info("\n" + "-" * 40)
    logger.info("构建模型...")
    logger.info("-" * 40)

    encoder, classifier = build_model(config)

    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in classifier.parameters())
    logger.info(f"编码器参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"分类器参数量: {sum(p.numel() for p in classifier.parameters()):,}")
    logger.info(f"总参数量: {total_params:,}")

    # ---- 训练 (Leave-One-Subject-Out) ----
    logger.info("\n" + "=" * 60)
    logger.info("开始跨被试验证训练")
    logger.info("=" * 60)

    all_results = []

    for fold_idx, (src_loader, tgt_loader, target_id) in enumerate(data_loader.folds()):
        if tgt_loader is None:
            # 无验证集 (n_folds=0), 直接训练
            logger.info("训练模式: 全部数据 (无验证集划分)")
            # 重新构建模型 (避免之前fold的参数污染)
            encoder, classifier = build_model(config)
            trainer = DDATrainer(encoder, classifier, config, logger)
            # 这里只做训练不做验证 (用于最终测试集预测)
            for epoch in range(train_cfg.get('epochs', 200)):
                stats = trainer.train_epoch(src_loader, src_loader, epoch)
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: CE={stats['loss_ce']:.4f}, "
                                f"GDD={stats['loss_gdd']:.4f}, LSD={stats['loss_lsd']:.4f}")
            # 保存最终模型
            final_path = os.path.join(
                exp_cfg.get('save_dir', './checkpoints'),
                'final_model.pth'
            )
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
            }, final_path)
            logger.info(f"最终模型已保存至: {final_path}")
            break

        # 重新初始化模型 (每折从头训练)
        encoder, classifier = build_model(config)

        # 构建训练器
        trainer = DDATrainer(encoder, classifier, config, logger)

        # 训练该fold
        result = trainer.train_fold(src_loader, tgt_loader, target_id, fold_idx)

        # 可视化 (仅对最后一折做t-SNE)
        if fold_idx == data_loader.n_folds - 1 or fold_idx == 0:
            trainer.visualize_features(src_loader, tgt_loader,
                                       save_name=f"tsne_fold{fold_idx}")

        all_results.append(result)

    # ---- 汇总结果 ----
    if len(all_results) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("跨被试验证结果汇总")
        logger.info("=" * 60)
        accuracies = [r['best_accuracy'] for r in all_results]
        logger.info(f"各折目标域准确率: {[f'{a:.4f}' for a in accuracies]}")
        logger.info(f"平均准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        logger.info(f"最高准确率: {np.max(accuracies):.4f}")
        logger.info(f"最低准确率: {np.min(accuracies):.4f}")
        logger.info(f"各折最佳epoch: {[r['best_epoch'] for r in all_results]}")
        logger.info(f"目标被试: {[r['target_subject'] for r in all_results]}")

    logger.info("\n训练完成!")
    logger.close()


if __name__ == "__main__":
    main()
