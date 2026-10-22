"""通用界面组件。

把重复出现的卡片、指标块、表单行抽出来，避免各页面各写一套样式。
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from . import style


class Card(QFrame):
    """带标题的白底卡片容器。"""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 16, 18, 18)
        self._layout.setSpacing(10)
        if title:
            label = QLabel(title)
            label.setObjectName("CardTitle")
            self._layout.addWidget(label)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)


class StatCard(QFrame):
    """单指标展示块：大号数值 + 小号说明。"""

    def __init__(self, label: str, value: str = "—", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setMinimumHeight(80)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        self._value = QLabel(value)
        self._value.setObjectName("StatValue")
        self._value.setProperty("placeholder", value.strip() in ("", "—", "-"))
        self._label = QLabel(label)
        self._label.setObjectName("StatLabel")
        layout.addWidget(self._value)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def set_value(self, text: str) -> None:
        self._value.setText(text)
        # 未出数时用弱色小字号，避免一个长横线看起来像空条
        self._value.setProperty("placeholder", text.strip() in ("", "—", "-"))
        self._value.style().unpolish(self._value)
        self._value.style().polish(self._value)


class StatRow(QWidget):
    """一排指标块。"""

    def __init__(self, labels: list[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.cards: dict[str, StatCard] = {}
        for name in labels:
            card = StatCard(name)
            layout.addWidget(card)
            self.cards[name] = card

    def set(self, name: str, value: str) -> None:
        if name in self.cards:
            self.cards[name].set_value(value)


class PageHeader(QWidget):
    """页面标题 + 说明。"""

    def __init__(self, title: str, desc: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(5)
        t = QLabel(title)
        t.setObjectName("PageTitle")
        layout.addWidget(t)
        if desc:
            d = QLabel(desc)
            d.setObjectName("PageDesc")
            d.setWordWrap(True)
            layout.addWidget(d)


class FormGrid(QWidget):
    """两列表单：左标签右控件。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(8)
        self._grid.setColumnStretch(1, 1)
        self._row = 0

    def add_row(self, label: str, widget: QWidget, hint: str = "") -> None:
        # 统一控件宽度：否则数值框只有 80px、下拉框却有 250px，列看起来参差
        widget.setMinimumWidth(150)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        text = QLabel(label)
        text.setObjectName("Hint")
        text.setMinimumWidth(84)
        self._grid.addWidget(text, self._row, 0, Qt.AlignmentFlag.AlignTop)
        self._grid.addWidget(widget, self._row, 1)
        self._row += 1
        if hint:
            h = QLabel(hint)
            h.setObjectName("Hint")
            h.setWordWrap(True)
            self._grid.addWidget(h, self._row, 1)
            self._row += 1


class Separator(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {style.BORDER};")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def list_run_weights() -> list[tuple[str, str]]:
    """列出所有训练产出的 best.pt，返回 [(下拉显示文本, 路径)]，按时间倒序。

    显示文本带训练时间，用户才能分清「第几次训练」；只显示实验名时，
    多次训练在列表里长得一样，没法选。
    """
    from datetime import datetime

    from .. import config

    items: list[tuple[float, str, str]] = []
    if not config.RUNS_DIR.exists():
        return []
    for best in config.RUNS_DIR.glob("*/weights/best.pt"):
        try:
            mtime = best.stat().st_mtime
        except OSError:
            continue
        stamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        items.append((mtime, f"{best.parent.parent.name}　·　{stamp}", str(best)))
    items.sort(key=lambda item: item[0], reverse=True)
    return [(text, path) for _, text, path in items]


def pick_weights_dialog(parent: Optional[QWidget]) -> str | None:
    """选择权重：支持训练结果文件夹自动定位 best.pt，或手动选 .pt 文件。

    返回 best.pt 的绝对路径；取消或失败返回 None。
    """
    from pathlib import Path

    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    msg = QMessageBox(parent)
    msg.setWindowTitle("选择权重")
    msg.setText("你想怎么选择权重？")
    msg.setInformativeText(
        "· 选择「训练结果文件夹」：程序自动找到里面的 weights/best.pt\n"
        "· 选择「.pt 文件」：手动定位权重文件"
    )
    btn_folder = msg.addButton("训练结果文件夹", QMessageBox.ButtonRole.YesRole)
    btn_file = msg.addButton("手动选 .pt 文件", QMessageBox.ButtonRole.NoRole)
    btn_cancel = msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
    msg.exec()

    if msg.clickedButton() == btn_cancel:
        return None

    if msg.clickedButton() == btn_folder:
        folder = QFileDialog.getExistingDirectory(
            parent,
            "选择训练结果文件夹（含 weights/best.pt）",
            str(Path.home() / "Desktop"),
        )
        if not folder:
            return None
        candidates = [Path(folder) / "weights" / "best.pt", Path(folder) / "best.pt"]
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        QMessageBox.information(
            parent,
            "未找到权重",
            f"在「{folder}」里没有找到 weights/best.pt，\n"
            "请确认选中的是训练结果文件夹（里面应有 weights/ 目录）。",
        )
        return None

    path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择权重文件",
        str(Path.home() / "Desktop"),
        "PyTorch 权重 (*.pt)",
    )
    return path if path else None
