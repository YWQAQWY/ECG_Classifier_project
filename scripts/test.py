"""
测试集 LOSO 评估 + 推理
========================

对每个测试被试单独做 DANN 域适应:
  Source = 训练集 60 人 (有标签)
  Target = 1 个测试被试 (无标签, 用于域对抗)

然后预测该测试被试的 8 个 trial 情绪标签。

用法:
    python -m scripts.test --config configs/config.yaml
"""

import sys, os, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from data.dataset import EEGDataset, CrossSubjectDataLoader, TestDataLoader
from data.preprocess import process_subject_test
from models.transformer_encoder import TransformerEncoder
from models.classifier import EmotionClassifier
from models.domain_discriminator import DomainDiscriminator
from models.contrastive_head import ContrastiveHead
from losses.dann_loss import DANNTotalLoss
import torch.optim as optim


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_models(config, device):
    tf_cfg = config.get('transformer', {})
    encoder = TransformerEncoder(
        token_dim=tf_cfg.get('token_dim', 4),
        n_tokens=tf_cfg.get('n_tokens', 30),
        d_model=tf_cfg.get('d_model', 64),
        n_heads=tf_cfg.get('n_heads', 4),
        n_layers=tf_cfg.get('n_layers', 3),
        dim_feedforward=tf_cfg.get('dim_feedforward', 256),
        dropout=tf_cfg.get('dropout', 0.1),
    ).to(device)

    cls_cfg = config.get('classifier', {})
    classifier = EmotionClassifier(
        input_dim=tf_cfg.get('d_model', 64),
        hidden_dims=cls_cfg.get('hidden_dims', [64, 32]),
        n_classes=cls_cfg.get('n_classes', 2),
        dropout=cls_cfg.get('dropout', 0.3),
    ).to(device)

    dom_cfg = config.get('domain_discriminator', {})
    domain_disc = DomainDiscriminator(
        input_dim=tf_cfg.get('d_model', 64),
        hidden_dims=dom_cfg.get('hidden_dims', [64, 32]),
        n_domains=dom_cfg.get('n_domains', 2),
        dropout=dom_cfg.get('dropout', 0.3),
    ).to(device)

    contrastive_head = ContrastiveHead().to(device)

    return encoder, classifier, domain_disc, contrastive_head


def adapt_and_predict(config, src_loader, tgt_features, tgt_subject_id, device):
    """
    LOSO 域适应: 对单个测试被试做 DANN 训练 + 推理。

    参数:
        src_loader: 训练集 60 人 DataLoader
        tgt_features: 测试被试 DE 特征 (n, 30, 4)
        tgt_subject_id: 测试被试 ID
        device: 设备

    返回:
        trial_preds: 8 个 trial 的预测 (list of int)
    """
    encoder, classifier, domain_disc, contrastive_head = build_models(config, device)

    train_cfg = config.get('training', {})
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()) +
        list(domain_disc.parameters()),
        lr=train_cfg.get('learning_rate', 0.0005),
        weight_decay=train_cfg.get('weight_decay', 0.0001),
    )

    lw = config.get('loss_weights', {})
    criterion = DANNTotalLoss(
        cls_weight=lw.get('cls', 1.0),
        domain_weight=lw.get('domain', 0.1),
        contrastive_weight=lw.get('contrastive', 0.05),
        temperature=config.get('contrastive', {}).get('temperature', 0.1),
    )

    # 构建 target DataLoader (无标签, domain=1)
    tgt_dataset = EEGDataset(tgt_features, np.zeros(len(tgt_features), dtype=int),
                             domain_label=1, subject_id=tgt_subject_id)
    tgt_loader = DataLoader(tgt_dataset,
                            batch_size=min(train_cfg.get('batch_size', 128), len(tgt_dataset)),
                            shuffle=True, drop_last=False)

    # 测试集域适应用较少 epoch (快速适应)
    adapt_epochs = train_cfg.get('epochs', 100) // 3  # ~33 epochs

    tgt_iter = iter(tgt_loader)
    for epoch in range(adapt_epochs):
        encoder.train(); classifier.train(); domain_disc.train()
        for src_batch in src_loader:
            x_s, y_s, _, _ = src_batch
            x_s, y_s = x_s.to(device), y_s.to(device)

            try:
                x_t, _, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_t, _, _, _ = next(tgt_iter)
            x_t = x_t.to(device)

            # Forward
            z_s = encoder(x_s); z_t = encoder(x_t)
            cls_logits_s = classifier(z_s)
            dom_logits_s = domain_disc(z_s)
            dom_logits_t = domain_disc(z_t)
            z_s_n = contrastive_head(z_s); z_t_n = contrastive_head(z_t)

            with torch.no_grad():
                y_t_pseudo = torch.argmax(torch.softmax(classifier(z_t), dim=1), dim=1)

            loss_dict = criterion(cls_logits_s, y_s, dom_logits_s, dom_logits_t,
                                  z_s_n, z_t_n, y_t_pseudo)
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

    # ---- 推理 ----
    encoder.eval(); classifier.eval()
    infer_loader = DataLoader(tgt_dataset, batch_size=train_cfg.get('batch_size', 128),
                              shuffle=False)
    window_preds = []
    with torch.no_grad():
        for x, _, _, _ in infer_loader:
            x = x.to(device)
            z = encoder(x)
            logits = classifier(z)
            preds = torch.argmax(logits, dim=1)
            window_preds.append(preds.cpu().numpy())
    window_preds = np.concatenate(window_preds)

    # 多数投票 → trial 级预测
    windows_per_trial = max(0, (2500 - 250) // 125 + 1)
    trial_preds = []
    for t in range(8):
        start, end = t * windows_per_trial, (t + 1) * windows_per_trial
        tw = window_preds[start:min(end, len(window_preds))]
        if len(tw) > 0:
            trial_preds.append(int(np.argmax(np.bincount(tw, minlength=2))))
        else:
            trial_preds.append(0)

    return trial_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get('experiment', {}).get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- 加载训练集 (60 人, 有标签) ----
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})

    print("加载训练集...")
    train_loader = CrossSubjectDataLoader(
        train_root=data_cfg.get('train_root', './赛题四数据集及说明文档/训练集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=train_cfg.get('batch_size', 128),
        n_folds=0,        # 全部被试作为 source
        downsample_ratio=data_cfg.get('downsample_ratio', 0.5),
        ensemble_enabled=False,  # 测试时不做集成伪标签
    )

    # 获取 "全量 source" loader (n_folds=0 时 folds() 返回 60 折 LOSO)
    # 我们需要 1 个包含所有 60 人的 source loader
    # 用 folds() 的第一折的 src_loader 即可 (它包含 59 人)
    # 更好的方式: 直接从 all_subjects 构建
    all_features, all_labels = [], []
    for sid in train_loader.get_subject_ids():
        f, l = train_loader.all_subjects[sid]
        all_features.append(f)
        all_labels.append(l)
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    src_dataset = EEGDataset(all_features, all_labels, domain_label=0, subject_id="source")
    src_loader = DataLoader(src_dataset, batch_size=train_cfg.get('batch_size', 128),
                            shuffle=True, drop_last=True)
    print(f"训练集: {len(src_dataset)} windows ({len(train_loader.get_subject_ids())} 被试)")

    # ---- LOSO: 逐个测试被试域适应 + 推理 ----
    print("\n加载测试集...")
    test_files = sorted(train_loader.file_paths)  # use the test files from test_root
    import glob
    test_root = data_cfg.get('test_root', './赛题四数据集及说明文档/公开测试集')
    test_files = sorted(glob.glob(os.path.join(test_root, "*.mat")))
    print(f"测试集: {len(test_files)} 个被试")

    all_results = {}
    for i, fp in enumerate(test_files):
        sid = os.path.basename(fp).replace(".mat", "")
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(test_files)}] LOSO 域适应: Target = {sid}")
        print(f"{'='*50}")

        # 加载测试被试特征
        tgt_features, _ = process_subject_test(fp, trial_length=2500,
                                                window_size=data_cfg.get('window_size', 250),
                                                stride=data_cfg.get('stride', 125),
                                                fs=data_cfg.get('sampling_rate', 250))
        if tgt_features is None:
            print(f"  跳过 {sid} (无数据)")
            continue

        print(f"  Target windows: {len(tgt_features)}")

        # DANN 域适应 + 预测
        trial_preds = adapt_and_predict(config, src_loader, tgt_features, sid, device)
        all_results[sid] = trial_preds
        print(f"  预测: {trial_preds} (0=Neutral, 1=Positive)")

    # ---- 保存 ----
    with open(args.output, 'w') as f:
        f.write("Subject," + ",".join(f"Trial{i+1}" for i in range(8)) + "\n")
        for sid in sorted(all_results.keys()):
            f.write(f"{sid}," + ",".join(map(str, all_results[sid])) + "\n")
    print(f"\nPredictions saved to {args.output}")

    # 汇总
    print(f"\n{'='*50}")
    print("预测汇总:")
    for sid in sorted(all_results.keys()):
        preds = all_results[sid]
        n_pos = sum(preds)
        n_neu = len(preds) - n_pos
        print(f"  {sid}: {n_neu} Neutral, {n_pos} Positive")


if __name__ == "__main__":
    main()
