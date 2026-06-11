"""
DANN + Transformer 训练入口 — 两级层次分类
============================================

Stage 1: 抑郁症分类 (healthy=0, depressed=1)
  → 对测试被试预测健康/抑郁 → 路由到对应 Stage 2

Stage 2-A: 正常人情绪分类 (Neutral=0, Positive=1) — 训练于 40 健康人
Stage 2-B: 抑郁症情绪分类 (Neutral=0, Positive=1) — 训练于 20 抑郁患者

最终: 每个测试被试先经过 Stage 1 → Stage 2-A 或 2-B

用法:
    python -m scripts.train --config configs/config.yaml
    python -m scripts.train --debug
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


def _get_device(config):
    """获取可用设备，CUDA 不可用时回退 CPU"""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        t = torch.zeros(1).to('cuda')
        del t
        return torch.device('cuda')
    except Exception:
        print('[WARN] CUDA driver 版本不兼容，回退到 CPU')
        config['training']['device'] = 'cpu'
        return torch.device('cpu')


def build_models(config, n_classes):
    """构建 Transformer F + C + D + Con。n_classes 区分 Stage1(2) vs Stage2(2)"""
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
        n_classes=n_classes, dropout=cls_cfg.get('dropout', 0.3),
    )
    dom_cfg = config.get('domain_discriminator', {})
    domain_disc = DomainDiscriminator(
        input_dim=64, hidden_dims=dom_cfg.get('hidden_dims', [64, 32]),
        n_domains=2, dropout=dom_cfg.get('dropout', 0.3),
    )
    contrastive_head = ContrastiveHead()
    return encoder, classifier, domain_disc, contrastive_head


def predict_test_subject_level(encoder, classifier, test_loader, device):
    """
    对测试集逐被试推理, 返回被试级预测。

    每个被试: 所有窗口投票 → 被试级标签。
    返回: {sid: subject_level_prediction}
    """
    encoder.eval(); classifier.eval()
    results = {}
    for sid in test_loader.get_subject_ids():
        loader = test_loader.get_loader(sid)
        window_preds = []
        with torch.no_grad():
            for x, _, _, _ in loader:
                logits = classifier(encoder(x.to(device)))
                window_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        all_preds = np.concatenate(window_preds)
        # 被试级: 所有窗口多数投票
        results[sid] = int(np.argmax(np.bincount(all_preds, minlength=2)))

    encoder.train(); classifier.train()
    return results


def predict_test_trial_level(encoder, classifier, test_loader, device):
    """对测试集逐被试推理, 返回 trial 级预测 (8 trial/被试)"""
    encoder.eval(); classifier.eval()
    all_results = {}
    windows_per_trial = max(0, (2500 - 250) // 125 + 1)

    for sid in test_loader.get_subject_ids():
        loader = test_loader.get_loader(sid)
        window_preds = []
        with torch.no_grad():
            for x, _, _, _ in loader:
                logits = classifier(encoder(x.to(device)))
                window_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        window_preds = np.concatenate(window_preds)

        trial_preds = []
        for t in range(8):
            s, e = t * windows_per_trial, (t + 1) * windows_per_trial
            tw = window_preds[s:min(e, len(window_preds))]
            trial_preds.append(int(np.argmax(np.bincount(tw, minlength=2))) if len(tw) > 0 else 0)
        all_results[sid] = trial_preds

    encoder.train(); classifier.train()
    return all_results


# ================================================================
# Stage 1: 抑郁症分类 (K折CV)
# ================================================================
def train_stage1(all_subjects, depression_labels, test_loader, config, logger):
    """
    Stage 1: 对训练集做 K折CV 训练抑郁症分类器, 每折对测试集推理。

    标签: 0=healthy, 1=depressed

    返回: test_depression_preds — {sid: 0=healthy, 1=depressed}
    """
    logger.info(f"\n{'='*60}")
    logger.info("Stage 1: 抑郁症分类 (healthy vs depressed)")
    logger.info(f"{'='*60}")

    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    device = _get_device(config)

    # 将 emotion_labels 替换为 depression_labels (跳过伪标签被试)
    stage1_subjects = {}
    for sid in sorted(all_subjects.keys()):
        if sid.startswith('pseudo_'):
            continue  # 伪标签被试没有 depression 标签
        if sid not in depression_labels:
            continue
        features, _ = all_subjects[sid]
        dep_label = depression_labels[sid]
        # 每个被试的所有窗口共享同一个 depression label
        stage1_subjects[sid] = (features, np.full(len(features), dep_label))

    cv_loader = CrossSubjectDataLoader(
        stage1_subjects,
        batch_size=train_cfg.get('batch_size', 128),
        n_folds=data_cfg.get('n_folds', 5),
    )

    all_depression_preds = []
    stage1_val_results = []  # 收集验证结果用于报告

    for fold_idx, (src_loader, tgt_loader, target_id) in enumerate(cv_loader.folds()):
        if tgt_loader is None:
            break

        logger.info(f"\nStage1 Fold {fold_idx}: Target={target_id}")

        encoder, classifier, domain_disc, contrastive_head = build_models(config, n_classes=2)
        s1_cfg = config.get('stage1', {})
        trainer = DANNTrainer(encoder, classifier, domain_disc, contrastive_head,
                              config, logger,
                              stage_config={
                                  'is_stage1': True,
                                  'epochs': s1_cfg.get('epochs',
                                                       train_cfg.get('stage1_epochs', 100)),
                                  'domain_weight': s1_cfg.get('domain_weight', 0.3),
                                  'class_weight': s1_cfg.get('class_weight', [1.0, 2.0]),
                              })
        tgt_dep_label = depression_labels.get(target_id, -1)
        result = trainer.train_fold(src_loader, tgt_loader, target_id, fold_idx,
                                    depression_label=tgt_dep_label)
        logger.info(f"Stage1 Fold {fold_idx}: Val Acc={result['best_accuracy']:.4f}")
        stage1_val_results.append(result)

        # 该折模型对测试集预测 (被试级)
        fold_preds = predict_test_subject_level(encoder, classifier, test_loader, device)
        all_depression_preds.append(fold_preds)

    # K 折投票 → 最终抑郁症预测
    final_depression = {}
    all_sids = sorted(all_depression_preds[0].keys()) if all_depression_preds else []
    for sid in all_sids:
        votes = [fp[sid] for fp in all_depression_preds if sid in fp]
        final_depression[sid] = int(np.argmax(np.bincount(votes, minlength=2)))

    logger.info(f"\nStage 1 抑郁症预测结果:")
    n_h, n_d = 0, 0
    for sid in sorted(final_depression.keys()):
        label = final_depression[sid]
        if label == 0:
            n_h += 1
        else:
            n_d += 1
        logger.info(f"  {sid}: {'Healthy' if label == 0 else 'Depressed'}")
    logger.info(f"  → {n_h} Healthy, {n_d} Depressed")

    # 按预测分组
    healthy_tests = [sid for sid, v in final_depression.items() if v == 0]
    depressed_tests = [sid for sid, v in final_depression.items() if v == 1]

    # Stage 1 报告: 按组准确率
    if stage1_val_results:
        healthy_vals = [r for r in stage1_val_results if r['depression_label'] == 0]
        depressed_vals = [r for r in stage1_val_results if r['depression_label'] == 1]
        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 1 验证集报告")
        logger.info(f"{'='*60}")
        if healthy_vals:
            h_accs = [r['best_accuracy'] for r in healthy_vals]
            logger.info(f"  Healthy (n={len(healthy_vals)}): Mean={np.mean(h_accs):.4f} ± {np.std(h_accs):.4f}, "
                         f"Max={np.max(h_accs):.4f}, Min={np.min(h_accs):.4f}")
        if depressed_vals:
            d_accs = [r['best_accuracy'] for r in depressed_vals]
            logger.info(f"  Depressed (n={len(depressed_vals)}): Mean={np.mean(d_accs):.4f} ± {np.std(d_accs):.4f}, "
                         f"Max={np.max(d_accs):.4f}, Min={np.min(d_accs):.4f}")
        logger.info(f"{'='*60}")

    return final_depression, healthy_tests, depressed_tests


# ================================================================
# Stage 2: 情绪分类 (按组 K折CV)
# ================================================================
def train_stage2(group_subjects, group_name, test_sids, test_loader, config, logger):
    """
    Stage 2: 对指定组的被试做 K折CV 情绪分类, 每折对测试集推理。

    参数:
        group_subjects: {sid: (features, emotion_labels)}
        group_name: "Healthy" / "Depressed"
        test_sids: 要用这个组预测的测试被试 ID 列表

    返回:
        test_emotion_preds: [{sid: [trial_preds]}] 各折预测
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 2-{group_name}: 情绪分类 ({len(group_subjects)} 被试)")
    logger.info(f"{'='*60}")

    if len(group_subjects) < 3:
        logger.info(f"被试太少, 跳过 K折CV ({len(group_subjects)} subjects)")
        return []

    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    device = _get_device(config)

    # K 折数自适应
    n_folds_stage2 = min(data_cfg.get('n_folds', 5), len(group_subjects))
    cv_loader = CrossSubjectDataLoader(
        group_subjects,
        batch_size=train_cfg.get('batch_size', 128),
        n_folds=n_folds_stage2,
    )

    all_val_results = []
    all_test_preds = []

    for fold_idx, (src_loader, tgt_loader, target_id) in enumerate(cv_loader.folds()):
        if tgt_loader is None:
            break

        logger.info(f"\nStage2-{group_name} Fold {fold_idx}: Target={target_id}")

        encoder, classifier, domain_disc, contrastive_head = build_models(config, n_classes=2)
        trainer = DANNTrainer(encoder, classifier, domain_disc, contrastive_head,
                              config, logger)
        # Stage2 val subjects: 0=healthy, 1=depressed
        tgt_dep_label = 0 if group_name == "Healthy" else 1
        result = trainer.train_fold(src_loader, tgt_loader, target_id, fold_idx,
                                    depression_label=tgt_dep_label)
        all_val_results.append(result)

        # 只对属于该组的测试被试推理
        fold_test_preds = {}
        for sid in test_sids:
            preds = predict_test_trial_level(encoder, classifier, test_loader, device)
            fold_test_preds[sid] = preds.get(sid, [0] * 8)
        all_test_preds.append(fold_test_preds)
        logger.info(f"Stage2-{group_name} Fold {fold_idx}: Val={result['best_accuracy']:.4f}, "
                     f"Test推理={len(test_sids)} 被试")

    # Stage 2 per-group report
    if all_val_results:
        accs = [r['best_accuracy'] for r in all_val_results]
        logger.info(f"\nStage 2-{group_name} 验证集报告 ({len(all_val_results)} 折)")
        logger.info(f"  Mean={np.mean(accs):.4f} ± {np.std(accs):.4f}, "
                     f"Max={np.max(accs):.4f}, Min={np.min(accs):.4f}")

    return all_test_preds, all_val_results


# ================================================================
# Main
# ================================================================
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
        config['training']['stage1_epochs'] = 10
        config['data']['n_folds'] = 3

    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    exp_cfg = config.get('experiment', {})
    logger = Logger(log_dir=exp_cfg.get('log_dir', './logs'),
                    experiment_name=exp_cfg.get('name', 'dann_transformer'))
    logger.info("=" * 60)
    logger.info("DANN + Transformer 两级层次情绪分类")
    logger.info("=" * 60)

    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    ensemble_cfg = config.get('ensemble', {})
    ds_cfg = data_cfg.get('downsample', {})

    # ================================================================
    # Step 1: 加载训练数据 + 抑郁症标签
    # ================================================================
    all_subjects, depression_labels = load_all_train_subjects(
        train_root=data_cfg.get('train_root', './赛题四数据集及说明文档/训练集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        downsample_enabled=ds_cfg.get('enabled', True),
        downsample_ratio=ds_cfg.get('ratio', 0.5),
        match_test_duration=data_cfg.get('match_test_duration', True),
        test_segment_length=data_cfg.get('test_segment_length', 2500),
    )

    # ================================================================
    # Step 2: 集成扩充源域
    # ================================================================
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
    # Step 3: 构建测试集 loader
    # ================================================================
    test_loader = TestDataLoader(
        test_root=data_cfg.get('test_root', './赛题四数据集及说明文档/公开测试集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=train_cfg.get('batch_size', 128),
    )

    # ================================================================
    # Step 4: Stage 1 — 抑郁症分类 + 路由
    # ================================================================
    final_depression, healthy_tests, depressed_tests = train_stage1(
        all_subjects, depression_labels, test_loader, config, logger,
    )

    # ================================================================
    # Step 5: Stage 2 — 按组情绪分类
    # ================================================================
    # 按抑郁症标签拆分训练数据 (只使用原始 60 人, 不含 pseudo)
    healthy_train = {sid: val for sid, val in all_subjects.items()
                     if not sid.startswith('pseudo_') and depression_labels.get(sid, -1) == 0}
    depressed_train = {sid: val for sid, val in all_subjects.items()
                       if not sid.startswith('pseudo_') and depression_labels.get(sid, -1) == 1}

    logger.info(f"\nStage 2 分组: Healthy={len(healthy_train)} subjects, "
                 f"Depressed={len(depressed_train)} subjects")
    logger.info(f"测试路由: Healthy→{len(healthy_tests)} subjects, "
                 f"Depressed→{len(depressed_tests)} subjects")

    all_test_preds_stage2 = []
    stage2_all_val_results = []  # 用于最终报告

    # Stage 2-A: 正常人情绪分类
    if healthy_train and healthy_tests:
        preds_a, val_results_a = train_stage2(healthy_train, "Healthy", healthy_tests,
                                                test_loader, config, logger)
        all_test_preds_stage2.extend(preds_a)
        stage2_all_val_results.extend(val_results_a)

    # Stage 2-B: 抑郁症情绪分类
    if depressed_train and depressed_tests:
        preds_b, val_results_b = train_stage2(depressed_train, "Depressed", depressed_tests,
                                                test_loader, config, logger)
        all_test_preds_stage2.extend(preds_b)
        stage2_all_val_results.extend(val_results_b)

    # ================================================================
    # Step 6: 最终总结报告
    # ================================================================
    logger.info(f"\n{'#'*60}")
    logger.info(f"#  训练总结报告")
    logger.info(f"{'#'*60}")

    # Stage 2 情绪分类按组评估
    if stage2_all_val_results:
        healthy_stage2 = [r for r in stage2_all_val_results if r['depression_label'] == 0]
        depressed_stage2 = [r for r in stage2_all_val_results if r['depression_label'] == 1]

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 2 情绪分类 — 验证集准确率")
        logger.info(f"{'='*60}")

        if healthy_stage2:
            h_accs = [r['best_accuracy'] for r in healthy_stage2]
            logger.info(f"  Healthy 组 (n={len(healthy_stage2)} folds):")
            logger.info(f"    Mean = {np.mean(h_accs):.4f} ± {np.std(h_accs):.4f}")
            logger.info(f"    Max  = {np.max(h_accs):.4f}, Min = {np.min(h_accs):.4f}")

        if depressed_stage2:
            d_accs = [r['best_accuracy'] for r in depressed_stage2]
            logger.info(f"  Depressed 组 (n={len(depressed_stage2)} folds):")
            logger.info(f"    Mean = {np.mean(d_accs):.4f} ± {np.std(d_accs):.4f}")
            logger.info(f"    Max  = {np.max(d_accs):.4f}, Min = {np.min(d_accs):.4f}")

        logger.info(f"{'='*60}")

    # ================================================================
    # Step 7: K 折投票 → predictions.csv
    # ================================================================
    if all_test_preds_stage2:
        all_sids = sorted(test_loader.get_subject_ids())
        output_path = "predictions.csv"
        with open(output_path, 'w') as f:
            f.write("Subject," + ",".join(f"Trial{i+1}" for i in range(8)) + "\n")
            for sid in all_sids:
                fold_votes = []
                for fp in all_test_preds_stage2:
                    if sid in fp:
                        fold_votes.append(fp[sid])
                if fold_votes:
                    final = []
                    for t in range(8):
                        tv = [fv[t] for fv in fold_votes]
                        final.append(int(np.argmax(np.bincount(tv, minlength=2))))
                else:
                    final = [0] * 8
                f.write(f"{sid}," + ",".join(map(str, final)) + "\n")

        logger.info(f"\n测试集分类结果 → {output_path}")
        for sid in all_sids:
            fold_votes = [fp.get(sid, [0]*8) for fp in all_test_preds_stage2]
            final = []
            for t in range(8):
                tv = [fv[t] for fv in fold_votes]
                final.append(int(np.argmax(np.bincount(tv, minlength=2))))
            n_pos = sum(final)
            group = "Healthy" if sid in healthy_tests else "Depressed"
            logger.info(f"  {sid} [{group}]: {len(final)-n_pos} Neutral, {n_pos} Positive")

    logger.info(f"\n{'#'*60}")
    logger.info(f"#  训练完成!")
    logger.info(f"{'#'*60}")
    logger.close()


if __name__ == "__main__":
    main()
