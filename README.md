# DANN + Transformer 跨被试 EEG 情绪识别系统

## 方法概述

基于 **Domain Adversarial Neural Network (DANN)** + **Transformer** 的跨被试 EEG 情绪识别。

**核心思想**: 用域对抗训练 (GRL + Domain Discriminator) 消除不同被试间的 domain shift，同时用对比学习保持 latent space 的类别结构。

参考: 导师代码 `DEEP_DANN_SEED.py`，将其 DANN 思路适配到竞赛 EEG 数据集。

## 项目结构

```
ECG/
├── configs/
│   └── config.yaml                        # 全局配置
│
├── data/
│   ├── preprocess.py                      # EEG 预处理管线
│   │   ├── Trial 切分 + 降采样 + 滑动窗口
│   │   ├── 每被试独立 Z-score 归一化 (防泄漏)
│   │   └── DE 特征提取 (θ/α/β/γ 4频段)
│   └── dataset.py                         # PyTorch Dataset + LOSO 划分
│       ├── EEGDataset                     # 含 domain label (0=source, 1=target)
│       ├── CrossSubjectDataLoader.folds() # 留一法 / K折交叉验证
│       └── TestDataLoader                 # 测试集加载
│
├── models/
│   ├── transformer_encoder.py             # 共享 Transformer 编码器 F
│   │   └── 输入(30 tokens×4 dims) → PosEnc → 3×Self-Attn → z(64-dim)
│   ├── classifier.py                      # 分支 C: 情绪分类器 (CE loss)
│   ├── domain_discriminator.py            # 分支 D: GRL + 域判别器 (对抗)
│   │   └── GradReverse (前向恒等, 反向梯度×(-λ))
│   └── contrastive_head.py                # 分支 Con: 对比学习头 (L2 norm)
│
├── losses/
│   ├── contrastive_loss.py                # InfoNCE 对比损失 (内积相似度)
│   └── dann_loss.py                       # 组合损失: CE + λd·Domain + λc·Contrastive
│
├── ensemble/
│   └── pseudo_labeler.py                  # 集成伪标签 (SVM+RF+MLP+LR 全票通过)
│
├── trainers/
│   └── trainer.py                         # DANN 训练循环 + 周期集成更新
│
├── scripts/
│   ├── train.py                           # LOSO 训练入口
│   └── test.py                            # 测试集推理
│
└── utils/                                 # seed / logger / metrics / visualization
```

## 网络架构

```
                         ┌─────────────────────────┐
  Source EEG (30,4) ────►│                         │
                         │  Shared Transformer F   │
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

**三个并列分支共同约束 Transformer:**
- **C**: emotion-discriminative (情绪可判别)
- **D**: domain-invariant (域不变，对抗训练)
- **Con**: contrastively-structured (类结构约束，同类聚/异类分)

## 数据管线

```
文件: data/preprocess.py
────────────────────────

.mat 文件 → Trial 切分 (train:12500/test:2500)
  → 降采样 (downsample_ratio=0.5, 每被试随机保留)
  → 滑动窗口 (window=250, stride=125)
  → DE 特征 (bandpass + DE=0.5·log(2πeσ²))
  → 每被试独立 Z-score 归一化 (防数据泄漏)
  → 输出 (n_windows, 30 channels, 4 bands)
```

## 训练流程

```
文件: trainers/trainer.py  losses/dann_loss.py
────────────────────────────────────────────────

每 batch:
  Source B ──► Transformer F ──► z_s
                                     ├──► C ──► L_cls (CE, source only)
                                     ├──► GRL+D ──► L_domain (source vs target)
                                     └──► Con ──► L_con (同类近/异类远)

  Target B ──► Transformer F ──► z_t  (共享F, 同上三个分支)

  L_total = λ_cls*L_cls + λ_domain*L_domain + λ_con*L_con
  (λ_cls=1.0, λ_domain=0.1, λ_con=0.05)

周期集成伪标签:
  每 10 epoch → SVM+RF+MLP+LR 训练 → 目标域投票
  → 全票通过 → 加入源域训练集
```

## 评估方法: 留一法 (LOSO)

### 哪些模块使用 LOSO?

| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `data/dataset.py:94` | `CrossSubjectDataLoader.folds()` | 生成 LOSO 划分 |
| `scripts/train.py:98` | `main()` 训练循环 | 遍历每折, 训练+评估 |
| `configs/config.yaml` | `data.n_folds: 0` | 0 = 真正留一法 |

### LOSO 流程

```
n_folds=0 → 真正留一法:

  Subject 1 做 Target, Subject 2~60 做 Source → 训练 → 评估 Acc₁
  Subject 2 做 Target, Subject 1,3~60 做 Source → 训练 → 评估 Acc₂
  ...
  Subject 60 做 Target, Subject 1~59 做 Source → 训练 → 评估 Acc₆₀

最终: Mean ± Std (60个被试的目标域准确率)
```

### K-Fold 模式

```
n_folds=5 → 5折交叉验证 (快速调试用):

  60个被试随机分为5组, 每组取第1个做 Target
  跑5折, 每折1个 target subject
```

## 使用方法

```bash
cd /home/yanwq/ECG
source venv/bin/activate

# 调试模式 (3折, 20 epoch, ~5分钟)
python -m scripts.train --config configs/config.yaml --debug

# 完整 LOSO (60折, 100 epoch, 耗时长)
python -m scripts.train --config configs/config.yaml

# 测试集推理
python -m scripts.test --checkpoint checkpoints/best_dann_fold0.pth --output predictions.csv
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.downsample_ratio` | 0.5 | 降采样比例 |
| `data.n_folds` | 0 | 0=LOSO, N=K折 |
| `transformer.d_model` | 64 | Transformer 隐层维度 |
| `transformer.n_layers` | 3 | Transformer 层数 |
| `transformer.n_heads` | 4 | 注意力头数 |
| `loss_weights.cls` | 1.0 | CE 分类损失权重 |
| `loss_weights.domain` | 0.1 | 域对抗损失权重 |
| `loss_weights.contrastive` | 0.05 | 对比学习损失权重 |
| `contrastive.temperature` | 0.1 | 对比学习温度 |
| `ensemble.enabled` | true | 是否启用集成伪标签 |
| `training.learning_rate` | 5e-4 | 学习率 |
| `training.batch_size` | 128 | 批大小 |

## 与导师代码的对应关系

| 导师代码 DEEP_DANN_SEED.py | 本项目 |
|---|---|
| `GradReverse` | `models/domain_discriminator.py:GradReverse` |
| `SingleModalityFeatureExtractor` (MLP) | `models/transformer_encoder.py:TransformerEncoder` |
| `classifier` (128→64→2) | `models/classifier.py:EmotionClassifier` (64→32→2) |
| `domain_discriminator` (128→64→N) | `models/domain_discriminator.py:DomainDiscriminator` (64→32→2) |
| `total_loss = cls + 0.1*domain + constraint` | `losses/dann_loss.py` (CE + 0.1*Domain + 0.05*Con) |
| `ModalityMaskEnhancement` | 未采用 (用降采样+对比学习替代) |
| SEED 多模态 (EEG+EYE) | 竞赛 EEG 单模态 |
| Leave-One-Subject-Out (for loop) | `CrossSubjectDataLoader.folds()` LOSO |
