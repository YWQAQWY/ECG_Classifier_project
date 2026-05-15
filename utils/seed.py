"""
随机种子固定工具 — 保证实验可复现 (reproducibility)
论文要求: 固定随机种子以获取可复现结果
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    固定所有随机种子，确保实验可复现。

    为什么需要固定种子?
    - 神经网络训练包含大量随机性 (权重初始化, dropout, 数据shuffle等)
    - 不固定种子会导致每次运行结果不同，无法比较消融实验
    - 论文明确要求 "experimental results are reported on a random seed"

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况

    # cuDNN 确定性设置 — 保证卷积操作可复现
    # 注意: 这会略微降低训练速度，但对实验一致性至关重要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash 确定性 (用于DataLoader的多进程)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed] 随机种子已固定为: {seed}")
    print(f"[Seed] cuDNN deterministic = True (卷积操作可复现)")
