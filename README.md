# DANN + Transformer 跨被试 EEG 情绪识别系统

## 方法概述

基于 **Domain Adversarial Neural Network (DANN)** + **Transformer** + **对比学习** 的跨被试 EEG 情绪识别。

**核心思想**: 用域对抗训练 (GRL + Domain Discriminator) 消除不同被试间的 domain shift，同时用对比学习保持 latent space 的类别结构。

## 项目结构

```
ECG/
├── configs/config.yaml                     # 全局配置
├── data/
│   ├── preprocess.py                       # Trial切分/降采样/DE特征/zscore归一化
│   └── dataset.py                          # DataLoader + K折划分 + 集成扩充源域
├── models/
│   ├── transformer_encoder.py              # 共享 Transformer 编码器 F
│   ├── classifier.py                       # 分支 C: 情绪分类器
│   ├── domain_discriminator.py             # 分支 D: GRL + 域判别器
│   └── contrastive_head.py                 # 分支 Con: 对比学习头
├── losses/
│   ├── contrastive_loss.py                 # InfoNCE 对比损失 (内积相似度)
│   └── dann_loss.py                        # 组合损失: CE + λd·Domain + λc·Con
├── ensemble/
│   └── pseudo_labeler.py                   # 集成伪标签器 (SVM×2+RF+LR+MLP)
├── trainers/
│   └── trainer.py                          # DANN 训练循环
├── scripts/
│   ├── train.py                            # 训练入口 (全流程)
│   └── test.py                             # 独立测试推理
└── utils/                                  # seed / logger / metrics / visualization
```

## 数据流

```
Step 1 (预处理, 一次性):
  加载 60 人训练数据
    → 降采样 (可选, config 开关)
    → 每人独立 Z-score 归一化
    → 集成学习 (SVM×2+RF+LR+MLP) 对测试集投票
    → 全票通过的测试样本加入源域
    → 扩充后降采样 (可选)

Step 2 (K 折框架):
  For each fold:
    在扩充源域上划分 train/val
    → DANN+Transformer 训练
    → val accuracy
    → 对测试集推理
  → K 个 val acc → Mean ± Std (性能评估)
  → K 折投票 → predictions.csv (测试集分类结果)
```

## 网络架构

```
                         ┌─────────────────────────┐
  Source EEG (30,4) ────►│  Shared Transformer F   │
  Target EEG (30,4) ────►│  3 layers, 4 heads      │
                         │  Pre-LN, d_model=64     │
                         └───────────┬─────────────┘
                                     │ z (batch, 64)
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
              ┌──────────┐  ┌──────────────┐  ┌──────────┐
              │ Branch C │  │  Branch D    │  │ Branch Con│
              │Classifier│  │  GRL + Disc  │  │ L2 Norm  │
              │64→32→2  │  │  64→32→2     │  │ (no param)│
              └────┬─────┘  └──────┬───────┘  └────┬─────┘
                   │               │               │
              L_cls (CE)   L_domain (CE)    L_con (InfoNCE)
              source only   source+target    source+target
```

## 使用方法

```bash
cd /home/yanwq/ECG && source venv/bin/activate

# 快速调试 (3折, 20 epoch)
python -m scripts.train --config configs/config.yaml --debug

# 5折交叉验证 (默认)
python -m scripts.train --config configs/config.yaml

# 留一法 LOSO
python -m scripts.train --config configs/config.yaml --loso

# 自定义折数
python -m scripts.train --config configs/config.yaml --folds 10

# 跳过集成扩充源域
python -m scripts.train --config configs/config.yaml --skip-ensemble
```

## 关键配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.downsample.enabled` | true | 降采样开关 |
| `data.downsample.ratio` | 0.5 | 降采样保留比例 |
| `data.n_folds` | 5 | K 折数 (0=LOSO) |
| `ensemble.enabled` | true | 集成扩充源域开关 |
| `ensemble.confidence_threshold` | 0.8 | 全票通过阈值 |
| `transformer.d_model` | 64 | Transformer 隐层维度 |
| `transformer.n_layers` | 3 | Transformer 层数 |
| `transformer.n_heads` | 4 | 注意力头数 |
| `training.epochs` | 100 | 训练轮数 |
| `training.learning_rate` | 0.0005 | 学习率 |
| `loss_weights.cls` | 1.0 | 分类损失权重 |
| `loss_weights.domain` | 0.1 | 域对抗损失权重 |
| `loss_weights.contrastive` | 0.05 | 对比学习损失权重 |
