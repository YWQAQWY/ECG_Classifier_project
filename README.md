# DANN + Transformer 跨被试 EEG 情绪识别

## 方法概述

**两级层次分类架构** — 先判健康/抑郁，再分情绪。

```
                    测试被试 EEG (30ch × 4bands)
                         │
                ┌────────▼────────┐
                │  Stage 1        │
                │  抑郁症分类器     │
                │  0 = Healthy    │
                │  1 = Depressed  │
                └───┬─────────┬───┘
                    │         │
              0=Healthy    1=Depressed
                    │         │
          ┌─────────▼──┐  ┌──▼─────────┐
          │ Stage 2-A  │  │ Stage 2-B   │
          │ 正常人情绪  │  │ 抑郁患者情绪 │
          │ 训练于40人  │  │ 训练于20人  │
          └──────┬─────┘  └──────┬──────┘
                 │               │
                 ▼               ▼
         Neutral(0) / Positive(1) per trial
```

**动机**: 抑郁症患者 EEG 与正常人存在本质差异，混合训练会使情绪分类被"是否抑郁"混淆。分两阶段后，每个 Stage 2 模型只需处理组内情绪差异。

**单级网络 (Stage 1 & Stage 2 共用架构)**:

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

  L_total = 1.0 * CE + 0.1 * Domain + 0.05 * Contrastive
```

**三个并列分支:**
- **C**: emotion-discriminative — 保证情绪分类能力
- **D**: domain-invariant — GRL 梯度反转, 消除被试间 domain shift
- **Con**: contrastively-structured — 同类拉近、异类推远, 防止域对齐导致类别混合

## 项目结构

```
ECG/
├── configs/config.yaml                     # 全局配置 (数据/模型/训练超参)
├── data/
│   ├── preprocess.py                       # 预处理 (Trial切分/降采样/DE特征/zscore)
│   └── dataset.py                          # DataLoader + K折划分 + 集成扩充源域
├── models/
│   ├── transformer_encoder.py              # 共享 Transformer 编码器 F
│   ├── classifier.py                       # 分支 C: 情绪分类器
│   ├── domain_discriminator.py             # 分支 D: GRL + 域判别器
│   └── contrastive_head.py                 # 分支 Con: 对比学习头 (L2 norm)
├── losses/
│   ├── contrastive_loss.py                 # InfoNCE 对比损失 (内积相似度)
│   └── dann_loss.py                        # 组合损失: CE + λd·Domain + λc·Con
├── ensemble/
│   └── pseudo_labeler.py                   # 集成伪标签 (SVM×2+RF+LR+MLP 全票通过)
├── trainers/
│   └── trainer.py                          # DANN 训练循环 (每 fold 独立)
├── scripts/
│   ├── train.py                            # 主入口 (两级训练全流程)
│   └── test.py                             # 独立测试推理
└── utils/                                  # seed / logger / metrics / visualization
```

## 完整数据流

```
Step 1: 加载训练数据 + 抑郁症标签
  ├── 60 人 .mat 文件 → Trial切分 → 降采样(可选) → 滑动窗口 → DE 特征
  └── 每人独立 Z-score 归一化 + 标记抑郁症标签 (0=正常, 1=抑郁)

Step 2: 集成扩充源域 (可选, --skip-ensemble 跳过)
  └── SVM×2 + RF + LR + MLP → 对测试集投票 → 全票通过加入源域 → 再降采样

Step 3: Stage 1 — 抑郁症分类 (K折CV)
  ├── 标签: 0=Healthy, 1=Depressed
  ├── 每折 DANN 训练 → val 评估 → 对测试集推理 (被试级)
  └── K折投票 → 每测试被试路由到 Stage 2-A 或 2-B

Step 4: Stage 2 — 按组情绪分类 (K折CV)
  ├── Stage 2-A: 40 正常人 → K折CV → 对 Healthy 测试被试推理
  ├── Stage 2-B: 20 抑郁患者 → K折CV → 对 Depressed 测试被试推理
  └── 每折 DANN 训练 → val 评估 → 测试推理 (trial 级)

Step 5: K折投票 → predictions.csv
  └── 每被试 8 trial × 各折模型预测 → 多数投票 → 最终标签
```

## 使用方法

```bash
cd /home/yanwq/ECG && source venv/bin/activate

# 快速调试 (3折, 20 epoch)
python -m scripts.train --config configs/config.yaml --debug

# 完整训练 (5折CV)
python -m scripts.train --config configs/config.yaml

# 留一法 LOSO (60折)
python -m scripts.train --config configs/config.yaml --loso

# 自定义折数
python -m scripts.train --config configs/config.yaml --folds 10

# 跳过集成扩充源域
python -m scripts.train --config configs/config.yaml --skip-ensemble
```

## 数据集概览

| | 训练集 | 测试集 |
|------|--------|--------|
| 总数 | 60 人 | 10 人 |
| 正常人 | 40 | 5 |
| 抑郁症患者 | 20 | 5 |
| 通道 | 30 | 30 |
| 采样率 | 250 Hz | 250 Hz |
| 视频段 | 8 (4积极+4中性) × 50秒 | 8 × 10秒 (顺序打乱) |
| 标签 | EEG_data_neu/pos | 无 |

详见 [DATASET.md](DATASET.md)。

## 关键配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.downsample.enabled` | true | 降采样开关 |
| `data.downsample.ratio` | 0.5 | 降采样保留比例 |
| `data.n_folds` | 5 | K 折数 (0=LOSO) |
| `ensemble.enabled` | true | 集成扩充源域 |
| `transformer.d_model` | 64 | Transformer 隐层维度 |
| `transformer.n_layers` | 3 | Transformer 层数 |
| `transformer.n_heads` | 4 | 注意力头数 |
| `training.epochs` | 100 | Stage 2 训练轮数 |
| `training.stage1_epochs` | 50 | Stage 1 训练轮数 |
| `training.learning_rate` | 0.0005 | 学习率 |
| `training.batch_size` | 512 | 批大小 |
| `loss_weights.cls` | 1.0 | 分类损失权重 |
| `loss_weights.domain` | 0.1 | 域对抗损失权重 |
| `loss_weights.contrastive` | 0.05 | 对比学习损失权重 |
| `contrastive.temperature` | 0.1 | 对比学习温度 |
