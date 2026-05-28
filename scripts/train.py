"""
DANN + Transformer 训练入口
============================

用法:
    cd /home/yanwq/ECG
    source venv/bin/activate
    python -m scripts.train --config configs/config.yaml
    python -m scripts.train --debug
"""

import sys, os, argparse, yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logger import Logger
from data.dataset import CrossSubjectDataLoader
from models.transformer_encoder import TransformerEncoder
from models.classifier import EmotionClassifier
from models.domain_discriminator import DomainDiscriminator
from models.contrastive_head import ContrastiveHead
from trainers.trainer import DANNTrainer


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_models(config: dict):
    """构建 Transformer F + 三个并列分支"""
    tf_cfg = config.get('transformer', {})
    encoder = TransformerEncoder(
        token_dim=tf_cfg.get('token_dim', 4),
        n_tokens=tf_cfg.get('n_tokens', 30),
        d_model=tf_cfg.get('d_model', 64),
        n_heads=tf_cfg.get('n_heads', 4),
        n_layers=tf_cfg.get('n_layers', 3),
        dim_feedforward=tf_cfg.get('dim_feedforward', 256),
        dropout=tf_cfg.get('dropout', 0.1),
    )

    cls_cfg = config.get('classifier', {})
    classifier = EmotionClassifier(
        input_dim=tf_cfg.get('d_model', 64),
        hidden_dims=cls_cfg.get('hidden_dims', [64, 32]),
        n_classes=cls_cfg.get('n_classes', 2),
        dropout=cls_cfg.get('dropout', 0.3),
    )

    dom_cfg = config.get('domain_discriminator', {})
    domain_disc = DomainDiscriminator(
        input_dim=tf_cfg.get('d_model', 64),
        hidden_dims=dom_cfg.get('hidden_dims', [64, 32]),
        n_domains=dom_cfg.get('n_domains', 2),
        dropout=dom_cfg.get('dropout', 0.3),
    )

    contrastive_head = ContrastiveHead()

    return encoder, classifier, domain_disc, contrastive_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--loso', action='store_true', help='使用留一法 (60折, 替代5折CV)')
    parser.add_argument('--folds', type=int, default=None, help='自定义折数 (0=LOSO)')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.loso:
        config['data']['n_folds'] = 0
    elif args.folds is not None:
        config['data']['n_folds'] = args.folds

    if args.debug:
        config['training']['epochs'] = 20
        config['data']['n_folds'] = 3      # 调试模式只跑3折
        config['data']['downsample_ratio'] = 0.3

    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    exp_cfg = config.get('experiment', {})
    logger = Logger(log_dir=exp_cfg.get('log_dir', './logs'),
                    experiment_name=exp_cfg.get('name', 'dann_transformer'))
    logger.info("=" * 60)
    logger.info("DANN + Transformer 跨被试EEG情绪识别")
    logger.info("=" * 60)

    # 数据
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})

    ensemble_cfg = config.get('ensemble', {})
    data_loader = CrossSubjectDataLoader(
        train_root=data_cfg.get('train_root', './赛题四数据集及说明文档/训练集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=train_cfg.get('batch_size', 128),
        n_folds=data_cfg.get('n_folds', 5),
        downsample_ratio=data_cfg.get('downsample_ratio', 0.5),
        ensemble_enabled=ensemble_cfg.get('enabled', True),
        ensemble_threshold=ensemble_cfg.get('confidence_threshold', 0.8),
        ensemble_max_ratio=ensemble_cfg.get('add_to_source_percent', 0.3),
    )

    # 留一法交叉验证
    all_results = []
    for fold_idx, (src_loader, tgt_loader, target_id) in enumerate(data_loader.folds()):
        if tgt_loader is None:
            break

        encoder, classifier, domain_disc, contrastive_head = build_models(config)
        trainer = DANNTrainer(encoder, classifier, domain_disc, contrastive_head,
                              config, logger)
        result = trainer.train_fold(src_loader, tgt_loader, target_id, fold_idx)
        all_results.append(result)

    # 汇总
    if all_results:
        accs = [r['best_accuracy'] for r in all_results]
        subjects = [r['target_subject'] for r in all_results]

        logger.info(f"\n{'='*60}")
        logger.info(f"留一法结果汇总 ({len(all_results)} 被试)")
        logger.info(f"{'='*60}")
        for subj, acc in zip(subjects, accs):
            logger.info(f"  {subj}: {acc:.4f}")
        logger.info(f"{'='*60}")
        logger.info(f"Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        logger.info(f"Max:  {np.max(accs):.4f}")
        logger.info(f"Min:  {np.min(accs):.4f}")

    logger.close()


if __name__ == "__main__":
    main()
