"""训练曲线绘制与 ultralytics 出图补丁。

ultralytics 8.3.x 的 plot_results 依赖 polars，而本机环境里的 polars
一 import 就会让进程被系统杀掉（包体损坏，重装无效）。这里用 pandas
重新实现同一功能，并在训练开始前替换掉 ultralytics 的实现。

同时把图表样式统一：中文标题用系统字体，线宽与配色按报告排版需要设定。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 中文显示：Windows 自带字体，逐级回退
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 指标列 -> 中文标题
METRIC_LABELS = {
    "metrics/precision(B)": "精确率 Precision",
    "metrics/recall(B)": "召回率 Recall",
    "metrics/mAP50(B)": "mAP@0.5",
    "metrics/mAP50-95(B)": "mAP@0.5:0.95",
}

# 损失列形如 train/box_loss、val/cls_loss、val/dfl_loss
LOSS_PARTS = {"box": "定位损失", "cls": "分类损失", "dfl": "分布损失"}
SPLIT_PARTS = {"train": "训练", "val": "验证"}


def _column_title(name: str) -> str:
    """把 results.csv 的列名翻成中文标题。"""
    if name in METRIC_LABELS:
        return METRIC_LABELS[name]
    if "loss" in name:
        split, _, kind = name.partition("/")
        kind = kind.replace("_loss", "").replace("loss", "").strip("_")
        prefix = SPLIT_PARTS.get(split, split)
        return f"{prefix}{LOSS_PARTS.get(kind, kind)}"
    return name


def _smooth(values: np.ndarray, sigma: int = 3) -> np.ndarray:
    """高斯滑动平均，与 ultralytics 的平滑曲线一致。"""
    if len(values) < 3:
        return values
    radius = max(1, sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_training_results(file: str = "", dir: str = "",
                          on_plot: Optional[Callable] = None,
                          out_dir: Optional[Path] = None) -> Optional[Path]:
    """读取 results.csv 绘制损失与指标曲线，保存为 results.png。

    ``out_dir`` 指定图片输出目录。默认不传时写回 results.csv 所在目录
    （ultralytics 训练结束时画图，图属于训练结果目录本身，这是预期行为）。
    导出报告时传 out_dir，避免把图写进用户选中的权重目录污染其数据。
    """
    csv_path = Path(file) if file else None
    save_dir = csv_path.parent if csv_path else Path(dir)
    files = sorted(save_dir.glob("results*.csv"))
    if not files:
        return None

    frame = pd.read_csv(files[0])
    frame.columns = [c.strip() for c in frame.columns]
    loss_cols = [c for c in frame.columns if "loss" in c]
    metric_cols = [c for c in frame.columns if "metrics/" in c]
    if not loss_cols and not metric_cols:
        return None

    epoch = frame.iloc[:, 0].to_numpy()
    panels = loss_cols + metric_cols
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), tight_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, panels):
        y = frame[name].to_numpy(dtype=float)
        ax.plot(epoch, y, color="#2E86DE", linewidth=1.4, marker=".",
                markersize=3, label="实测")
        if len(y) >= 5:
            ax.plot(epoch, _smooth(y), linestyle="--", color="#C0392B",
                    linewidth=1.2, label="平滑")
        ax.set_title(_column_title(name), fontsize=10)
        ax.grid(alpha=0.25, linestyle=":")
        ax.tick_params(labelsize=8)
        if name in metric_cols:
            ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, frameon=False)

    for ax in axes[len(panels):]:
        ax.axis("off")

    write_dir = Path(out_dir) if out_dir else save_dir
    out = write_dir / "results.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if on_plot:
        on_plot(str(out))
    return out


def install_plot_patch() -> None:
    """替换 ultralytics 中依赖 polars 的两个函数。

    - plot_results      出图（utils.plotting 与 trainer 各持一份引用）
    - read_results_csv  保存 checkpoint 时把训练曲线塞进权重文件

    本机 polars 包已损坏，import 会让进程被系统终止，因此统一改用 pandas。
    """
    import ultralytics.utils.plotting as plotting
    import ultralytics.engine.trainer as trainer

    plotting.plot_results = plot_training_results
    trainer.plot_results = plot_training_results
    # 实际定义在 BaseTrainer 上，Trainer 只是模块级懒加载别名
    trainer.BaseTrainer.read_results_csv = _read_results_csv


def _read_results_csv(self) -> dict:
    """pandas 版 read_results_csv，输出结构与 polars 版一致（列名 -> 列表）。"""
    try:
        frame = pd.read_csv(self.csv)
        frame.columns = [c.strip() for c in frame.columns]
        return {c: frame[c].tolist() for c in frame.columns}
    except Exception:
        return {}
