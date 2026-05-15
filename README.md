# DDA 跨被试 EEG 情绪识别系统

基于论文 *"Dynamic Domain Adaptation for Class-aware Cross-subject and Cross-session EEG Emotion Recognition"* (Li et al., IEEE JBHI 2022) 的 PyTorch 实现。

## 项目概述

本项目实现了一个基于动态域适应 (Dynamic Domain Adaptation, DDA) 的跨被试 EEG 情绪识别系统，用于竞赛数据集的情绪分类 (Neutral=0, Positive=1)。

**核心问题**: 不同被试之间存在严重的 domain shift，需要实现跨被试泛化。

## 项目结构

```
ECG/
├── configs/
│   └── config.yaml                 # 全局超参数配置
│
├── data/
│   ├── __init__.py
│   ├── preprocess.py               # EEG 预处理管线
│   │   ├── Trial 切分              # 训练:12500点/trial, 测试:2500点/trial
│   │   ├── 滑动窗口                # window=250(1s), stride=125(0.5s)
│   │   ├── Z-score 归一化          # 每通道每trial独立归一化
│   │   └── DE 特征提取             # θ(4-8), α(8-14), β(14-31), γ(31-45) Hz
│   └── dataset.py                  # PyTorch Dataset + 跨被试数据划分
│       ├── EEGSubjectDataset       # 单被试数据集
│       ├── TestSubjectDataset      # 测试集数据集 (无标签)
│       ├── CrossSubjectDataLoader  # 跨被试 Leave-One-Subject-Out 划分
│       └── TestDataLoader          # 测试集加载器
│
├── models/
│   ├── __init__.py
│   ├── eegnet.py                   # EEGNet 编码器 (特征提取)
│   │   └── Conv2D → DepthwiseConv2D → SeparableConv2D → FC → 64-dim feature
│   ├── encoder.py                  # MLP 编码器 (备选, 与论文一致)
│   │   └── 4层MLP: [512, 128, 128, 64] → 64-dim feature
│   └── classifier.py               # 线性分类器 head
│       └── Linear(64, 2) → emotion logits
│
├── losses/
│   ├── __init__.py
│   ├── mmd.py                      # 多核 RBF MMD (最大均值差异)
│   │   └── MultiKernelMMD: 多带宽高斯核 + 自适应中位距离缩放
│   ├── lsd.py                      # LSD (局部子域差异)
│   │   └── LocalSubdomainDiscrepancy: 类条件MMD (同类拉近, 异类推远)
│   └── dda_loss.py                 # DDA 总损失 (CE + GDD + LSD + Dynamic α)
│       └── DDALoss: 集成所有损失 + 动态α调度 + 伪标签生成
│
├── trainers/
│   ├── __init__.py
│   ├── trainer.py                  # DDA 训练器 (核心训练循环)
│   │   ├── train_epoch()           # 单epoch训练: CE+GDD+LSD+α 联合优化
│   │   ├── train_fold()            # 单fold训练+验证+保存
│   │   └── visualize_features()    # t-SNE 特征可视化
│   └── evaluator.py                # 模型评估器
│       ├── evaluate()              # 目标域 accuracy/F1/confusion matrix
│       ├── predict()               # 测试集推理
│       └── extract_features()      # 特征提取 (用于t-SNE)
│
├── utils/
│   ├── __init__.py
│   ├── seed.py                     # 随机种子固定 (保证可复现)
│   ├── logger.py                   # TensorBoard + 控制台日志
│   ├── metrics.py                  # Accuracy, F1, Confusion Matrix
│   └── visualization.py            # t-SNE 特征可视化 + 训练曲线 + 混淆矩阵
│
├── scripts/
│   ├── __init__.py
│   ├── train.py                    # 训练入口 (Leave-One-Subject-Out CV)
│   ├── train_ablation.py           # 消融实验 (CE-only / CE+GDD / CE+GDD+LSD)
│   └── test.py                     # 测试集推理
│
├── checkpoints/                    # 模型保存目录
├── logs/                           # TensorBoard 日志目录
├── venv/                           # Python 虚拟环境
└── README.md                       # 本文件
```

## 数据管线 (Data Pipeline)

```
原始 EEG (.mat 文件)
    │  30 channels × N samples, 250Hz
    ▼
┌──────────────────────────────┐
│ Step 1: Trial 切分           │
│  训练: 12500点/trial × 8     │
│  测试:  2500点/trial × 8     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Step 2: Z-score 归一化       │
│  每通道每trial独立归一化      │
│  x_norm = (x - μ) / σ        │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Step 3: 滑动窗口             │
│  window=250(1s), stride=125  │
│  训练: ~99窗口/trial          │
│  测试: ~19窗口/trial          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Step 4: DE 特征提取          │
│  4频段带通滤波 + DE计算       │
│  DE = 0.5·log(2πeσ²)        │
│  输出: (n_windows, 30, 4)    │
└──────────────┬───────────────┘
               ▼
        PyTorch Dataset
        (n, 1, 30, 4) tensor
```

## 模型架构 (Model Architecture)

```
输入: (batch, 1, 30 channels, 4 bands)
         │
    ┌────▼───────────────────────────────────┐
    │ EEGNet Encoder (特征提取器)             │
    │                                        │
    │ Block 1: Conv2D(1→F1, kernel=1×4)     │
    │          + BN + ELU + Dropout          │
    │          输出: (batch, F1, 30, 1)      │
    │                                        │
    │ Block 2: DepthwiseConv2D(F1→D·F1,     │
    │          kernel=30×1, groups=F1)       │
    │          + BN + ELU + Dropout          │
    │          输出: (batch, D·F1, 1, 1)     │
    │                                        │
    │ Block 3: SeparableConv2D(D·F1→F2,     │
    │          kernel=1×1)                   │
    │          + BN + ELU + Dropout          │
    │          输出: (batch, F2, 1, 1)       │
    │                                        │
    │ Flatten → Linear(F2, 64)               │
    │          输出: (batch, 64) ← GDD/LSD  │
    │                对齐的目标特征空间       │
    └────┬───────────────────────────────────┘
         │
    ┌────▼───────────────────────────────────┐
    │ Emotion Classifier (分类器)            │
    │ Dropout → Linear(64, 2)               │
    │ 输出: (batch, 2) ← emotion logits      │
    └────────────────────────────────────────┘
```

## 训练流程 (Training Pipeline)

每个 epoch 的完整训练流程 (论文 Algorithm 1):

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 输入数据                                        │
│   Xs, Ys ← Source Domain (有标签)                       │
│   Xt     ← Target Domain (无标签, 仅训练时)              │
├─────────────────────────────────────────────────────────┤
│ Step 2: 编码器前向传播                                  │
│   Fs = encoder(Xs)  → (batch_s, 64)                    │
│   Ft = encoder(Xt)  → (batch_t, 64)                    │
├─────────────────────────────────────────────────────────┤
│ Step 3: 分类器前向传播                                  │
│   Ps = classifier(Fs)  → (batch_s, 2)                  │
│   Pt = classifier(Ft)  → (batch_t, 2)                  │
├─────────────────────────────────────────────────────────┤
│ Step 4: 生成伪标签                                      │
│   Yt_hat = argmax(Pt)  → (batch_t,)                    │
│   ⚠️ 伪标签不是优化变量, 随θ更新而变化                  │
├─────────────────────────────────────────────────────────┤
│ Step 5: 计算 CE Loss                                   │
│   L_ce = CrossEntropy(Ps, Ys)                          │
│   仅使用源域真实标签                                    │
├─────────────────────────────────────────────────────────┤
│ Step 6: 计算 GDD Loss                                  │
│   L_gdd = MMD(Fs, Ft)                                  │
│   多核RBF MMD, 自适应带宽缩放                           │
│   目标: 对齐源域和目标域的整体特征分布                    │
├─────────────────────────────────────────────────────────┤
│ Step 7: 计算 LSD Loss                                  │
│   L_lsd = mean(同类MMD) - mean(异类MMD)               │
│   同类: MMD(Fs[c], Ft[c])         ← 希望小             │
│   异类: MMD(Fs[c], Ft[c'])        ← 希望大             │
│   使用目标域伪标签 Yt_hat 分组                          │
├─────────────────────────────────────────────────────────┤
│ Step 8: 动态 α 计算                                    │
│   Linear: α = 1 - epoch/N    (从1线性衰减到0)          │
│   Sigmoid: α = σ(-epoch+100)  (论文原始S型曲线)        │
│   初期 α≈1 → 主要优化 GDD                              │
│   后期 α≈0 → 主要优化 LSD                              │
├─────────────────────────────────────────────────────────┤
│ Step 9: 总 Loss                                        │
│   L = L_ce + β·(α·L_gdd + (1-α)·L_lsd)               │
├─────────────────────────────────────────────────────────┤
│ Step 10: 反向传播                                      │
│   loss.backward()                                      │
│   optimizer.step()                                     │
└─────────────────────────────────────────────────────────┘
```

## DDA 核心理论

### 1. CE (Cross Entropy)
- **作用**: 学习情绪分类边界
- **数据**: 仅使用源域真实标签
- **公式**: `L_ce = -(1/n) Σ y_i·log(P_θ(ŷ_i|x_i))`

### 2. GDD (Global Domain Distribution Alignment)
- **作用**: 对齐源域和目标域的整体特征分布
- **实现**: 多核 RBF MMD
- **公式**: `L_gdd = MMD(Fs, Ft)`
- **局限**: 仅对齐整体分布可能导致类别混合

### 3. LSD (Local Subdomain Alignment)
- **作用**: 类别级精细对齐 (同类拉近, 异类推远)
- **实现**: 类条件 MMD + 伪标签
- **公式**: `L_lsd = mean(同类MMD) - mean(异类MMD)`

### 4. Dynamic α Schedule (论文灵魂)
- **初期** (α≈1): 伪标签不可靠 → 主要优化 GDD
- **后期** (α≈0): 伪标签变可靠 → 主要优化 LSD
- **粗到细**: GDD (域级粗对齐) → LSD (类别级细对齐)

### 5. Pseudo Label Update (EM 思想)
- 每轮动态更新伪标签 Yt_hat = argmax(Pt)
- θ → features → predictions → pseudo labels (自然更新)
- EM: E步估计隐变量(伪标签) → M步更新参数(θ) → 交替迭代 → 收敛

## 使用方法

### 环境配置

```bash
cd /home/yanwq/ECG
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy torch scikit-learn matplotlib pyyaml tensorboard
```

### 训练

```bash
# 完整训练 (Leave-One-Subject-Out 交叉验证)
python -m scripts.train --config configs/config.yaml

# 调试模式 (少被试, 少epoch)
python -m scripts.train --config configs/config.yaml --debug

# 指定实验模式
python -m scripts.train --mode ce_gdd_lsd
```

### 消融实验

```bash
# 对比 CE-only / CE+GDD / CE+GDD+LSD
python -m scripts.train_ablation --config configs/config.yaml
```

### 测试集推理

```bash
python -m scripts.test \
    --config configs/config.yaml \
    --checkpoint checkpoints/final_model.pth \
    --output predictions.csv
```

### 查看 TensorBoard

```bash
tensorboard --logdir logs/
```

## 数据集

- **训练集**: 40名健康 + 20名抑郁症被试
- **测试集**: 5名健康 + 5名抑郁症被试
- **EEG**: 30通道, 250Hz
- **标签**: Neutral=0, Positive=1
- **数据路径**: `赛题四数据集及说明文档/`

## 参考文献

Li, Z., Zhu, E., Jin, M., Fan, C., He, H., Cai, T., & Li, J. (2022). Dynamic Domain Adaptation for Class-aware Cross-subject and Cross-session EEG Emotion Recognition. *IEEE Journal of Biomedical and Health Informatics*, 26(12), 5964-5974.

## 配置说明

主要超参数 (见 `configs/config.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dda.beta` | 0.1 | 对齐损失权重 |
| `dda.alpha_schedule` | linear | α衰减策略 (linear/sigmoid) |
| `dda.mmd.bandwidths` | [0.5,1,2,4,8,16] | MMD多核带宽 |
| `dda.lsd.confidence_threshold` | 0.7 | LSD伪标签置信度阈值 |
| `training.epochs` | 200 | 训练轮数 |
| `training.batch_size` | 64 | 批大小 |
| `training.learning_rate` | 0.001 | 学习率 |
| `model.eegnet.F1` | 8 | EEGNet时间滤波器数 |
| `model.eegnet.D` | 2 | 深度乘子 |
| `data.window_size` | 250 | 滑动窗口大小 (1秒) |
| `data.stride` | 125 | 滑动步长 (0.5秒) |
# ECG_Classifier_project
