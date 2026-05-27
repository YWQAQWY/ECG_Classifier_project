"""
测试集推理脚本
==============

用法:
    python -m scripts.test --checkpoint checkpoints/best_dann_fold0.pth
"""

import sys, os, argparse, yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from data.dataset import TestDataLoader
from models.transformer_encoder import TransformerEncoder
from models.classifier import EmotionClassifier


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get('experiment', {}).get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
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
    classifier = EmotionClassifier(input_dim=64, n_classes=2)
    encoder.to(device)
    classifier.to(device)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    encoder.eval()
    classifier.eval()

    # 推理
    data_cfg = config.get('data', {})
    test_loader = TestDataLoader(
        test_root=data_cfg.get('test_root', './赛题四数据集及说明文档/公开测试集'),
        window_size=data_cfg.get('window_size', 250),
        stride=data_cfg.get('stride', 125),
        fs=data_cfg.get('sampling_rate', 250),
        batch_size=config.get('training', {}).get('batch_size', 128),
    )

    all_results = {}
    windows_per_trial = max(0, (2500 - 250) // 125 + 1)

    for sid in test_loader.get_subject_ids():
        loader = test_loader.get_loader(sid)
        window_preds = []
        with torch.no_grad():
            for x, _, _, _ in loader:
                x = x.to(device)
                z = encoder(x)
                logits = classifier(z)
                preds = torch.argmax(logits, dim=1)
                window_preds.append(preds.cpu().numpy())
        window_preds = np.concatenate(window_preds)

        # 多数投票 → trial 级预测
        trial_preds = []
        for t in range(8):
            start, end = t * windows_per_trial, (t + 1) * windows_per_trial
            tw = window_preds[start:min(end, len(window_preds))]
            if len(tw) > 0:
                trial_preds.append(np.argmax(np.bincount(tw, minlength=2)))
            else:
                trial_preds.append(0)
        all_results[sid] = trial_preds

    # 保存
    with open(args.output, 'w') as f:
        f.write("Subject," + ",".join(f"Trial{i+1}" for i in range(8)) + "\n")
        for sid in sorted(all_results.keys()):
            f.write(f"{sid}," + ",".join(map(str, all_results[sid])) + "\n")
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
