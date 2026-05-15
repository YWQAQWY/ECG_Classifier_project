"""
可视化工具 — t-SNE 特征可视化 + 训练曲线 + 混淆矩阵热力图
用于直观理解 DDA 的对齐效果

论文 Fig.4 展示了: 原始数据 → 中间对齐 → 最终类感知对齐 的 t-SNE 变化
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import os


def plot_tsne(features: np.ndarray, labels: np.ndarray, domain_labels: np.ndarray,
              save_path: str, title: str = "t-SNE Feature Visualization",
              class_names: list = None, domain_names: list = None):
    """
    使用 t-SNE 可视化特征分布。

    为什么使用 t-SNE?
    - 高维特征 (如64维) 无法直接可视化
    - t-SNE 保持局部邻域结构，能展示类间/域间关系
    - 论文 Fig.4 展示了不同训练阶段的对齐效果:
      (a) 原始数据 → 源域和目标域分离
      (b) 中间状态 → 域间靠近但类别混合
      (c) 最终结果 → 同类跨域聚类 (类感知对齐)

    参数:
        features: 特征向量 (n_samples, feature_dim)
        labels: 类别标签 (n_samples,) — 如 [0,0,1,1,...]
        domain_labels: 域标签 (n_samples,) — 0=source, 1=target
        save_path: 图像保存路径
        title: 图像标题
        class_names: 类别名称列表，如 ['Neutral', 'Positive']
        domain_names: 域名称列表，如 ['Source', 'Target']
    """
    if class_names is None:
        class_names = ["Neutral", "Positive"]
    if domain_names is None:
        domain_names = ["Source", "Target"]

    # 降采样 (避免 t-SNE 过慢)
    n_samples = features.shape[0]
    if n_samples > 500:
        indices = np.random.choice(n_samples, 500, replace=False)
        features = features[indices]
        labels = labels[indices]
        domain_labels = domain_labels[indices]

    # t-SNE 降维到 2D
    print(f"[t-SNE] 正在降维 {features.shape[0]} 个样本...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # ---- 子图1: 按域着色 (显示域间分布差异) ----
    for d, name, marker in zip([0, 1], domain_names, ['o', '^']):
        mask = domain_labels == d
        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                        marker=marker, alpha=0.6, label=name, s=30)
    axes[0].set_title("(a) Domain Distribution", fontsize=13)
    axes[0].legend()
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")

    # ---- 子图2: 按类别着色 (显示类别分布) ----
    colors = ['#1f77b4', '#ff7f0e']
    for c, name, color in zip(range(len(class_names)), class_names, colors):
        mask = labels == c
        axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=color, alpha=0.6, label=name, s=30)
    axes[1].set_title("(b) Class Distribution", fontsize=13)
    axes[1].legend()
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")

    # ---- 子图3: 域+类别联合着色 (显示类感知对齐效果) ----
    # marker=域, color=类别
    markers = ['o', '^']
    for d, dname in enumerate(domain_names):
        for c, (cname, color) in enumerate(zip(class_names, colors)):
            mask = (domain_labels == d) & (labels == c)
            axes[2].scatter(features_2d[mask, 0], features_2d[mask, 1],
                            marker=markers[d], c=color, alpha=0.6,
                            label=f"{dname}-{cname}", s=30)
    axes[2].set_title("(c) Class-aware Domain Alignment", fontsize=13)
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("t-SNE dim 1")
    axes[2].set_ylabel("t-SNE dim 2")

    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[t-SNE] 图像已保存至: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str,
                          class_names: list = None):
    """
    绘制混淆矩阵热力图。

    参数:
        cm: 混淆矩阵 (n_classes × n_classes)
        save_path: 图像保存路径
        class_names: 类别名称
    """
    if class_names is None:
        class_names = ["Neutral", "Positive"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    # 在每个格子中标注数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=18,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ConfusionMatrix] 图像已保存至: {save_path}")


def plot_training_curves(history: dict, save_path: str):
    """
    绘制训练曲线。

    参数:
        history: 训练历史字典，包含:
            - 'epoch': epoch列表
            - 'loss_ce': CE loss
            - 'loss_gdd': GDD loss
            - 'loss_lsd': LSD loss
            - 'loss_total': 总loss
            - 'alpha': 动态α
            - 'train_acc': 源域准确率
            - 'val_acc': 目标域准确率
        save_path: 图像保存路径
    """
    epochs = history['epoch']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss 曲线
    axes[0, 0].plot(epochs, history['loss_ce'], label='CE Loss', color='blue')
    axes[0, 0].set_title("Cross-Entropy Loss (Source)", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['loss_gdd'], label='GDD Loss', color='green')
    axes[0, 1].set_title("Global Domain Discrepancy", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MMD")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['loss_lsd'], label='LSD Loss', color='red')
    axes[0, 2].set_title("Local Subdomain Discrepancy", fontsize=12)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("LSD")
    axes[0, 2].grid(True, alpha=0.3)

    # 总Loss + α + Accuracy
    axes[1, 0].plot(epochs, history['loss_total'], label='Total Loss', color='purple')
    axes[1, 0].set_title("Total Loss", fontsize=12)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['alpha'], label='α (Dynamic)', color='orange')
    axes[1, 1].set_title("Dynamic α Schedule", fontsize=12)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("α")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, history['train_acc'], label='Source Acc', color='blue', linestyle='--')
    axes[1, 2].plot(epochs, history['val_acc'], label='Target Acc', color='red')
    axes[1, 2].set_title("Accuracy", fontsize=12)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Accuracy")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("DDA Training Curves", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[TrainingCurves] 图像已保存至: {save_path}")
