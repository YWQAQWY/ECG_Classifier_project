"""
日志工具 — TensorBoard + 控制台日志
用于记录训练过程中的各项loss和指标变化
"""

import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    训练日志管理器。

    功能:
    - 控制台日志: 使用Python logging模块，同时输出到文件和终端
    - TensorBoard: 记录loss曲线、accuracy、alpha变化等，支持可视化
    """

    def __init__(self, log_dir: str = "./logs", experiment_name: str = "dda_eeg"):
        """
        参数:
            log_dir: 日志根目录
            experiment_name: 实验名称 (用于子目录命名)
        """
        # 创建带时间戳的日志目录，避免覆盖之前的实验
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化 TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 初始化 Python logging
        self._setup_logging()

        print(f"[Logger] 日志目录: {self.log_dir}")

    def _setup_logging(self):
        """配置 Python logging 模块"""
        self.logger = logging.getLogger("DDA_EEG")
        self.logger.setLevel(logging.DEBUG)

        # 文件 handler — 保存完整日志
        log_file = os.path.join(self.log_dir, "training.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # 控制台 handler — 只输出INFO及以上
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式化
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, msg: str):
        """记录INFO级别日志"""
        self.logger.info(msg)

    def debug(self, msg: str):
        """记录DEBUG级别日志"""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """记录WARNING级别日志"""
        self.logger.warning(msg)

    # ---- TensorBoard 标量记录 ----

    def log_scalar(self, tag: str, value: float, step: int):
        """
        记录标量到 TensorBoard。

        参数:
            tag: 标签名 (如 'train/loss_ce', 'val/accuracy')
            value: 标量值
            step: 全局步数 (通常为epoch)
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_value_dict: dict, step: int):
        """
        记录多个标量到同一图表。

        参数:
            main_tag: 主标签 (如 'train/losses')
            tag_value_dict: {tag: value} 字典
            step: 全局步数
        """
        self.writer.add_scalars(main_tag, tag_value_dict, step)

    # ---- 训练专用便捷方法 ----

    def log_train(self, epoch: int, loss_ce: float, loss_gdd: float,
                  loss_lsd: float, loss_total: float, alpha: float,
                  source_acc: float):
        """
        记录训练阶段的各项指标。

        参数:
            epoch: 当前epoch
            loss_ce: 交叉熵损失
            loss_gdd: 全局域分布对齐损失
            loss_lsd: 局部子域对齐损失
            loss_total: 总损失
            alpha: 动态平衡因子 (控制GDD和LSD的相对权重)
            source_acc: 源域训练准确率
        """
        self.log_scalars("train/loss_components", {
            "ce": loss_ce,
            "gdd": loss_gdd,
            "lsd": loss_lsd,
        }, epoch)
        self.log_scalar("train/loss_total", loss_total, epoch)
        self.log_scalar("train/alpha", alpha, epoch)
        self.log_scalar("train/source_accuracy", source_acc, epoch)

    def log_val(self, epoch: int, target_acc: float, target_f1: float,
                val_loss_ce: float = None):
        """
        记录验证阶段的各项指标。

        参数:
            epoch: 当前epoch
            target_acc: 目标域 (target subject) 准确率
            target_f1: 目标域 F1 分数
            val_loss_ce: 目标域交叉熵 (如果有标签)
        """
        self.log_scalar("val/target_accuracy", target_acc, epoch)
        self.log_scalar("val/target_f1", target_f1, epoch)
        if val_loss_ce is not None:
            self.log_scalar("val/loss_ce", val_loss_ce, epoch)

    def close(self):
        """关闭日志写入器"""
        self.writer.close()
        self.logger.info("[Logger] 日志记录已关闭")
