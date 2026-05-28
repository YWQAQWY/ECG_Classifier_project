"""
DANN + Transformer 训练入口
============================

流程:
  Step 1 (预处理): 集成扩充源域 (CV之前一次性完成)
  Step 2 (K折框架): 每折 训练 + 评估 + 测试推理 → predictions.csv

用法:
    python -m scripts.train --config configs/config.yaml
    python -m scripts.train --debug
    python -m scripts.train --loso
"""

import sys, os, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logger import Logger
from data.dataset import (
    load_all_train_subjects, expand_source_with_ensemble,
    CrossSubjectDataLoader, EEGDataset, TestDataLoader,
)
from models.transformer_encoder import TransformerEncoder
from models.classifier import EmotionClassifier
from models.domain_discriminator import DomainDiscriminator
from models.contrastive_head import ContrastiveHead
from trainers.trainer import DANNTrainer


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_models(config):
    tf_cfg = config.get('transformer', {})
    encoder = TransformerEncoder(
        token_dim=tf_cfg.get('token_dim', 4), n_tokens=tf_cfg.get('n_tokens', 30),
        d_model=tf_cfg.get('d_model', 64), n_heads=tf_cfg.get('n_heads', 4),
        n_layers=tf_cfg.get('n_layers', 3),
        dim_feedforward=tf_cfg.get('dim_feedforward', 256),
        dropout=tf_cfg.get('dropout', 0.1),
    )
    cls_cfg = config.get('classifier', {})
    classifier = EmotionClassifier(
        input_dim=64, hidden_dims=cls_cfg.get('hidden_dims', [64, 32]),
        n_classes=2, dropout=cls_cfg.get('dropout', 0.3),
    )
    dom_cfg = config.get('domain_discriminator', {})
    domain_disc = DomainDiscriminator(
        input_dim=64, hidden_dims=dom_cfg.get('hidden_dims', [64, 32]),
        n_domains=2, dropout=dom_cfg.get('dropout', 0.3),
    )
    contrastive_head = ContrastiveHead()
    return encoder, classifier, domain_disc, contrastive_head


def predict_test_set(encoder, classifier, test_loader, device):
    """对测试集推理, 返回 {sid: [trial_preds]}"""
    encoder.eval(); classifier.eval()
    all_results = {}
    windows_per_trial = max(0, (2500 - 250) // 125 + 1)

    for sid in test_loader.get_subject_ids():
        loader = test_loader.get_loader(sid)
        window_preds = []
        with torch.no_grad():
            for x, _, _, _ in loader:
                x = x.to(device)
                logits = classifier(encoder(x))
                preds = torch.argmax(logits, dim=1)
                window_preds.append(preds.cpu().numpy())
        window_preds = np.concatenate(window_preds)

        trial_preds = []
        for t in range(8):
            s, e = t * windows_per_trial, (t + 1) * windows_per_trial
            tw = window_preds[s:min(e, len(window_preds))]
            trial_preds.append(int(np.argmax(np.bincount(tw, minlength=2))) if len(tw) > 0 else 0)
        all_results[sid] = trial_preds

    encoder.train(); classifier.train()
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--loso', action='store_true', help='留一法 LOSO')
    parser.add_argument('--folds', type=int, default=None, help='自定义折数')
    parser.add_argument('--skip-ensemble', action='store_true', help='跳过集成扩充')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.loso:
        config['data']['n_folds'] = 0
    elif args.folds is not None:
        config['data']['n_folds'] = args.folds

    if args.debug:
        config['training']['epochs'] = 20
        config['data']['n_folds'] = 3

    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    exp_cfg = config.get('experiment', {})
    logger = Logger(log_dir=exp_cfg.get('log_dir', './logs'),
                    experiment_name=exp_cfg.get('name', 'dann_transformer'))
    logger.info("=" * 60)
    logger.info("DANN + Transformer 跨被试EEG情绪识别")
    logger.info("=" * 60)

    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    ensemble_cfg = config.get('ensemble', {})
    ds_cfg = data_cfg.get('downsample', {})

    # ================================================================
    # Step 1: 加载 + 集成扩充源域 (预处理, 一次性)
    # ================================================================
    all_subjects = load_all_train_subjects(
        train_root=data_cfg.get('train_root', './赛题四数据集及说明文档/训练集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        downsample_enabled=ds_cfg.get('enabled', True),
        downsample_ratio=ds_cfg.get('ratio', 0.5),
    )

    if not args.skip_ensemble and ensemble_cfg.get('enabled', True):
        all_subjects = expand_source_with_ensemble(
            all_subjects,
            test_root=data_cfg.get('test_root', './赛题四数据集及说明文档/公开测试集'),
            window_size=data_cfg.get('window_size', 250),
            stride=data_cfg.get('stride', 125),
            fs=data_cfg.get('sampling_rate', 250),
            confidence_threshold=ensemble_cfg.get('confidence_threshold', 0.8),
            post_downsample_enabled=ds_cfg.get('enabled', True),
            post_downsample_ratio=ds_cfg.get('ratio', 0.5),
            seed=seed,
        )

    # ================================================================
    # Step 2: K 折 CV (训练 + 评估 + 测试推理)
    # ================================================================
    cv_loader = CrossSubjectDataLoader(
        all_subjects,
        batch_size=train_cfg.get('batch_size', 128),
        n_folds=data_cfg.get('n_folds', 5),
    )

    # 构建测试集 loader (所有折共用)
    test_loader = TestDataLoader(
        test_root=data_cfg.get('test_root', './赛题四数据集及说明文档/公开测试集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=train_cfg.get('batch_size', 128),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_val_results = []         # 各折 val accuracy
    all_test_predictions = []    # 各折测试集预测

    for fold_idx, (src_loader, tgt_loader, target_id) in enumerate(cv_loader.folds()):
        if tgt_loader is None:
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx}: Target = {target_id}")
        logger.info(f"{'='*60}")

        encoder, classifier, domain_disc, contrastive_head = build_models(config)
        trainer = DANNTrainer(encoder, classifier, domain_disc, contrastive_head,
                              config, logger)
        result = trainer.train_fold(src_loader, tgt_loader, target_id, fold_idx)
        all_val_results.append(result)

        # 该折模型对测试集推理
        fold_preds = predict_test_set(encoder, classifier, test_loader, device)
        all_test_predictions.append(fold_preds)
        logger.info(f"Fold {fold_idx} 测试集预测完成")

    # ================================================================
    # 汇总
    # ================================================================
    if all_val_results:
        accs = [r['best_accuracy'] for r in all_val_results]
        subjects = [r['target_subject'] for r in all_val_results]
        logger.info(f"\n{'='*60}")
        logger.info(f"K折CV 结果汇总 ({len(all_val_results)} 折)")
        logger.info(f"{'='*60}")
        for subj, acc in zip(subjects, accs):
            logger.info(f"  {subj}: {acc:.4f}")
        logger.info(f"{'='*60}")
        logger.info(f"Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # K 折投票 → 最终测试集预测
    if all_test_predictions:
        output_path = "predictions.csv"
        all_sids = sorted(all_test_predictions[0].keys())
        with open(output_path, 'w') as f:
            f.write("Subject," + ",".join(f"Trial{i+1}" for i in range(8)) + "\n")
            for sid in all_sids:
                # 收集各折对该被试的预测 → 多数投票
                fold_votes = []
                for fold_preds in all_test_predictions:
                    if sid in fold_preds:
                        fold_votes.append(fold_preds[sid])
                if fold_votes:
                    final = []
                    for t in range(8):
                        tv = [fv[t] for fv in fold_votes]
                        final.append(int(np.argmax(np.bincount(tv, minlength=2))))
                else:
                    final = [0] * 8
                f.write(f"{sid}," + ",".join(map(str, final)) + "\n")

        logger.info(f"\n测试集分类结果 (K折投票) 已保存至: {output_path}")
        for sid in all_sids:
            fold_votes = [fp.get(sid, [0]*8) for fp in all_test_predictions]
            final = []
            for t in range(8):
                tv = [fv[t] for fv in fold_votes]
                final.append(int(np.argmax(np.bincount(tv, minlength=2))))
            n_pos = sum(final)
            logger.info(f"  {sid}: {len(final)-n_pos} Neutral, {n_pos} Positive")

    logger.info("\n训练完成!")
    logger.close()


if __name__ == "__main__":
    main()
