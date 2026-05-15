"""
评估指标工具 — accuracy, confusion matrix, F1-score
用于跨被试EEG情绪识别任务的模型评估
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算分类准确率。

    参数:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)

    返回:
        accuracy: 准确率 (0~1)
    """
    return accuracy_score(y_true, y_pred)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list = None):
    """
    计算混淆矩阵。

    为什么需要混淆矩阵?
    - 查看模型在各类别上的分类表现
    - 判断是否存在类别偏向 (如总是预测positive)
    - 在跨被试场景中，domain shift 常导致某类准确率极低

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签列表，如 [0, 1] 对应 [neutral, positive]

    返回:
        cm: 混淆矩阵 (n_classes × n_classes)
    """
    if labels is None:
        labels = [0, 1]
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """
    计算F1分数。

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('macro'=各类别平均, 'micro'=全局, 'binary'=二分类)

    返回:
        f1: F1分数
    """
    return f1_score(y_true, y_pred, average=average)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算所有评估指标。

    参数:
        y_true: 真实标签
        y_pred: 预测标签

    返回:
        metrics: 包含 accuracy, cm, f1_macro, f1_binary, report 的字典
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
        "f1_macro": compute_f1(y_true, y_pred, average="macro"),
        "f1_weighted": compute_f1(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["neutral", "positive"],
            digits=4
        ),
    }


def print_metrics(metrics: dict):
    """打印评估指标到控制台"""
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"\n混淆矩阵:")
    cm = metrics["confusion_matrix"]
    print(f"              预测Neutral  预测Positive")
    print(f"  真实Neutral     {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"  真实Positive    {cm[1,0]:5d}        {cm[1,1]:5d}")
    print(f"\n详细报告:")
    print(metrics["classification_report"])
    print("=" * 60)
