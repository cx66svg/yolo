"""嵌入式图表画布。

训练过程中的曲线直接画在界面里，避免让用户去翻 results.png。
matplotlib 后端选用 QtAgg，与 PyQt6 共用一套事件循环。
"""
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from . import style  # noqa: E402

# 中文显示
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class MetricChart(FigureCanvasQTAgg):
    """多曲线指标图。"""

    def __init__(self, title: str = "指标曲线", parent=None):
        self._fig = Figure(figsize=(5.2, 3.0), dpi=100, tight_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)
        self._ax = self._fig.add_subplot(111)
        self._title = title
        self._reset()

    def _reset(self) -> None:
        self._ax.clear()
        self._ax.set_facecolor("#FFFFFF")
        self._fig.patch.set_facecolor("#FFFFFF")
        self._ax.set_title(self._title, fontsize=10, color="#111827", pad=8)
        self._ax.set_xlabel("训练轮次", fontsize=9, color="#6B7280")
        self._ax.set_ylabel("数值", fontsize=9, color="#6B7280")
        # 去掉上下右边框，只留左侧与底部，配合浅灰虚线网格
        for side in ("top", "right"):
            self._ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            self._ax.spines[side].set_color("#E5E9F2")
        self._ax.grid(alpha=1.0, linestyle="--", linewidth=0.6, color="#EEF2F8")
        self._ax.tick_params(labelsize=8, colors="#6B7280", length=0)
        self._ax.set_axisbelow(True)
        self._empty_hint = self._ax.text(
            0.5, 0.5, "训练开始后显示曲线",
            ha="center", va="center", fontsize=9, color="#9CA3AF",
            transform=self._ax.transAxes,
        )
        self._lines: dict[str, tuple] = {}
        self.draw_idle()

    def update_series(self, series: dict[str, list[float]]) -> None:
        """用完整数据重画，series 形如 {名称: [值...]}。"""
        if not series:
            return
        self._ax.clear()
        self._ax.set_title(self._title, fontsize=10)
        self._ax.set_xlabel("训练轮次", fontsize=9)
        self._ax.grid(alpha=0.25, linestyle=":")
        self._ax.tick_params(labelsize=8)

        colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]
        for idx, (name, values) in enumerate(series.items()):
            if not values:
                continue
            xs = list(range(1, len(values) + 1))
            self._ax.plot(xs, values, linewidth=1.6,
                          color=colors[idx % len(colors)], label=name)
        if getattr(self, "_empty_hint", None) is not None:
            self._empty_hint.set_visible(False)
        self._ax.legend(fontsize=7, frameon=False, loc="lower right")
        self._ax.set_ylim(0, 1.02)
        # 轮次是整数，避免出现 1.25、2.75 这种刻度
        self._ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.draw_idle()


class BarChart(FigureCanvasQTAgg):
    """横向条形图，用于类别分布。"""

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(5.0, 2.4), dpi=100, tight_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)
        self._ax = self._fig.add_subplot(111)
        self.clear()

    def clear(self) -> None:
        self._ax.clear()
        self._ax.text(0.5, 0.5, "暂无数据", ha="center", va="center",
                      fontsize=10, color="#9CA3AF", transform=self._ax.transAxes)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_visible(False)
        self.draw_idle()

    def plot(self, labels: list[str], values: list[float],
             colors: Optional[list[str]] = None) -> None:
        self._ax.clear()
        if not values:
            self.clear()
            return
        palette = colors or ["#2563EB"] * len(values)
        bars = self._ax.barh(labels[::-1], values[::-1],
                             color=palette[::-1], height=0.6)
        total = sum(values) or 1
        for bar, value in zip(bars, values[::-1]):
            self._ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                          f"{int(value)} ({value / total * 100:.1f}%)",
                          va="center", fontsize=8, color=style.TEXT_MUTED)
        self._ax.set_xlim(0, max(values) * 1.28)
        self._ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            self._ax.spines[spine].set_visible(False)
        self.draw_idle()
