# DANN + Transformer 跨被试 EEG 情绪识别系统

## 方法概述

基于 **Domain Adversarial Neural Network (DANN)** + **Transformer** + **对比学习** 的跨被试 EEG 情绪识别。

**核心思想**: 用域对抗训练 (GRL + Domain Discriminator) 消除不同被试间的 domain shift，同时用对比学习保持 latent space 的类别结构。


## 项目结构

```
ECG/
├── configs/
│   └── config.yaml                        # 全局配置
│
├── data/
│   ├── preprocess.py                      # EEG 预处理管线
│   │   ├── Trial 切分 + 滑动窗口
│   │   ├── 降采样 (每被试随机保留)
│   │   ├── 每被试独立 Z-score 归一化 (防泄漏)
│   │   └── DE 特征提取 (θ/α/β/γ 4频段)
│   └── dataset.py                         # PyTorch Dataset + 预处理集成
│       ├── EEGDataset                     # 含 domain label (0=source, 1=target)
│       ├── CrossSubjectDataLoader         # LOSO / K折划分
│       │   └── folds() 中完成集成伪标签   # 预处理阶段: SVM+RF+MLP+LR 全票通过
│       └── TestDataLoader                 # 测试集加载
│
├── models/
│   ├── transformer_encoder.py             # 共享 Transformer 编码器 F
│   │   └── 输入(30 tokens×4 dims) → PosEnc → 3×Self-Attn → z(64-dim)
│   ├── classifier.py                      # 分支 C: 情绪分类器 (CE loss)
│   ├── domain_discriminator.py            # 分支 D: GRL + 域判别器 (对抗)
│   │   └── GradReverse (前向恒等, 反向梯度×(-λ))
│   └── contrastive_head.py                # 分支 Con: 对比学习头 (L2 norm, 无参数)
│
├── losses/
│   ├── contrastive_loss.py                # InfoNCE 对比损失 (内积相似度)
│   └── dann_loss.py                       # 组合损失: CE + λd·Domain + λc·Contrastive
│
├── ensemble/
│   └── pseudo_labeler.py                  # 集成伪标签器 (SVM×2 + RF + LR + MLP)
│
├── trainers/
│   └── trainer.py                         # DANN 训练循环 (纯训练, 无预处理逻辑)
│
├── scripts/
│   ├── train.py                           # 训练入口 (5折CV / LOSO)
│   └── test.py                            # 测试集 LOSO 域适应 + 推理
│
└── utils/                                 # seed / logger / metrics / visualization
```

## 预处理管线 (按顺序执行)

```
文件: data/preprocess.py  data/dataset.py  ensemble/pseudo_labeler.py
─────────────────────────────────────────────────────────────────────

对每个被试独立执行:

  Step 1. 降采样
    → 每个被试随机保留 downsample_ratio 比例的窗口 (减少冗余)
    → config: data.downsample_ratio = 0.5

  Step 2. 每被试独立 Z-score 归一化
    → normalize_per_subject(): 每个被试各自算 mean/std (防数据泄漏)
    → x_norm = (x - μ_subject) / σ_subject

对每折 (fold) 执行:

  Step 3. 集成伪标签 (预处理阶段, 非训练阶段!)
    → 5 个基础分类器 (SVM×2 + RF + LR + MLP) 在源域训练
    → 对目标域投票 → 全票通过 → 加入源域
    → config: ensemble.enabled, ensemble.confidence_threshold = 0.8
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
- **C**: emotion-discriminative (情绪可判别, 仅 source 真实标签)
- **D**: domain-invariant (域不变, GRL 梯度反转对抗训练)
- **Con**: contrastively-structured (类结构约束, 同类内积大/异类内积小)

## 训练流程

```
文件: trainers/trainer.py  losses/dann_loss.py
────────────────────────────────────────────────

每 batch:
  Source B ──► Transformer F ──► z_s
                                     ├──► C ──► L_cls (CE, source only)
                                     ├──► GRL+D ──► L_domain (source=0 vs target=1)
                                     └──► Con ──► L_con (同类近/异类远)

  Target B ──► Transformer F ──► z_t  (共享 F, 同上三个分支)

  L_total = λ_cls*L_cls + λ_domain*L_domain + λ_con*L_con
           = 1.0 * CE  + 0.1 * Domain  + 0.05 * Contrastive
```

## 评估方法

### 训练阶段: 5 折 CV 或 LOSO

| 命令 | 行为 | 折数 |
|------|------|------|
| `--folds 5` (默认) | 5 折交叉验证 | 5 |
| `--loso` | 留一法 (每个被试轮流做 target) | 60 |
| `--folds N` | 自定义 N 折 | N |

```
LOSO 原理:
  Round 1: 训练 subj 2~60 → 测试 subj 1 → Acc₁
  Round 2: 训练 subj 1,3~60 → 测试 subj 2 → Acc₂
  ...
  Round 60: 训练 subj 1~59 → 测试 subj 60 → Acc₆₀
  最终: Mean Acc ± Std

5 折 CV:
  60 人分 5 组, 每组取第 1 个做 target, 跑 5 轮
```

### 测试阶段: LOSO 域适应 + 推理

```
文件: scripts/test.py
─────────────────────

对每个测试被试 (10 人), 单独做一次 DANN 域适应:
  Source = 训练集 60 人 (有标签)
  Target = 1 个测试被试 (无标签, 用于 GRL+D 域对抗)
  → 训练 ~33 epoch → 推理 8 个 trial → 多数投票

输出: predictions.csv → 提交竞赛平台
```

注意: 测试集无标签, **无法本地评估准确度**, 只能提交看分数。

## 使用方法

```bash
cd /home/yanwq/ECG && source venv/bin/activate

# 5折交叉验证 (默认)
python -m scripts.train --config configs/config.yaml

# 留一法 LOSO (60折)
python -m scripts.train --config configs/config.yaml --loso

# 自定义折数
python -m scripts.train --config configs/config.yaml --folds 10

# 调试模式 (3折, 20 epoch)
python -m scripts.train --config configs/config.yaml --debug

# 测试集推理 (LOSO 域适应)
python -m scripts.test --config configs/config.yaml --output predictions.csv
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.downsample_ratio` | 0.5 | 降采样比例 (每被试) |
| `data.n_folds` | 5 | 折数 (5=5折CV, 0=LOSO) |
| `data.normalization` | zscore | 每被试独立归一化 |
| `transformer.d_model` | 64 | Transformer 隐层维度 |
| `transformer.n_layers` | 3 | Transformer 层数 |
| `transformer.n_heads` | 4 | 注意力头数 |
| `transformer.dropout` | 0.1 | Transformer dropout |
| `classifier.dropout` | 0.3 | 分类器 dropout |
| `ensemble.enabled` | true | 预处理中启用集成伪标签 |
| `ensemble.confidence_threshold` | 0.8 | 全票通过阈值 |
| `loss_weights.cls` | 1.0 | CE 分类损失权重 |
| `loss_weights.domain` | 0.1 | 域对抗损失权重 |
| `loss_weights.contrastive` | 0.05 | 对比学习损失权重 |
| `contrastive.temperature` | 0.1 | 对比学习温度 |
| `training.learning_rate` | 0.0005 | 学习率 |
| `training.batch_size` | 128 | 批大小 |


## 数据流总结

```
原始 .mat (60+10 subjects)
  │
  ▼
preprocess.py:
  Trial切分 → 滑动窗口 → DE特征 (30ch×4bands)
  │
  ▼
dataset.py (每被试独立):
  降采样 → Z-score归一化 → 集成伪标签(预处理)
  │
  ▼
trainer.py (5折CV / LOSO):
  Transformer F → C + GRL+D + Con → 训练
  │
  ▼
test.py (测试集LOSO域适应):
  每测试被试: 源域60人 + 目标域该被试 → DANN适应 → 推理
```
