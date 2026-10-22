"""主窗口：侧边导航 + 四个功能页面 + 状态栏。"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .. import __app_name__, __version__, config
from . import style
from .gpu import GpuBadge
from .pages import DatasetPage, EvalPage, PredictPage, ProjectPage, TrainPage
from .state import AppState
from .workers import running_scripts, stop_all_scripts

NAV_ITEMS = [
    ("项目管理", "建档与数据集导入"),
    ("模型训练", "改进结构训练"),
    ("模型评估", "mAP 与每类 AP"),
    ("检测推理", "图像与视频检测"),
    ("数据集管理", "类别方案与派生"),
]


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.setObjectName("Root")
        self.setWindowTitle(f"{__app_name__}  v{__version__}")
        self.resize(1280, 820)
        self.setStyleSheet(style.stylesheet())
        self._build()
        self._bootstrap()

    # ------------------------------------------------------------ 界面
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._sidebar())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.project_page = ProjectPage(self.state)
        self.pages = [
            self.project_page,
            TrainPage(self.state),
            EvalPage(self.state),
            PredictPage(self.state),
            DatasetPage(self.state),
        ]
        for page in self.pages:
            self.stack.addWidget(page)
        right_layout.addWidget(self.stack, 1)

        # 底栏：左侧状态文字，右侧实时 GPU 资源
        bottom = QWidget()
        bottom.setStyleSheet(
            f"background:{style.CARD}; border-top:1px solid {style.BORDER};"
        )
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(14, 6, 14, 6)
        bottom_layout.setSpacing(8)

        self.status = QLabel("就绪")
        self.status.setStyleSheet(f"color:{style.TEXT_MUTED};")
        bottom_layout.addWidget(self.status, 1)

        self.gpu_badge = GpuBadge()
        bottom_layout.addWidget(self.gpu_badge, 0)

        right_layout.addWidget(bottom)
        self.state.status_message.connect(self.status.setText)
        self.gpu_badge.start()

        root.addWidget(right, 1)

    def _sidebar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(214)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        brand = QLabel("目标检测")
        brand.setObjectName("Brand")
        sub = QLabel(f"训练与推理平台 v{__version__}")
        sub.setObjectName("BrandSub")
        layout.addWidget(brand)
        layout.addWidget(sub)

        divider = QFrame()
        divider.setObjectName("BrandDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        # 导航按钮放进带左右留白的容器，才能做出胶囊式选中态
        nav_wrap = QWidget()
        nav_layout = QVBoxLayout(nav_wrap)
        nav_layout.setContentsMargins(10, 4, 10, 4)
        nav_layout.setSpacing(4)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for idx, (title, tip) in enumerate(NAV_ITEMS):
            btn = QPushButton(title)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, i=idx: self._switch(i))
            self.group.addButton(btn, idx)
            nav_layout.addWidget(btn)

        layout.addWidget(nav_wrap)
        layout.addStretch(1)
        footer = QLabel("本地推理 · 无需联网")
        footer.setObjectName("SidebarFooter")
        layout.addWidget(footer)
        return bar

    # ------------------------------------------------------------ 逻辑
    def _switch(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        # 同步按钮选中态，否则程序化切页时导航高亮不跟随
        button = self.group.button(index)
        if button is not None:
            button.setChecked(True)
        self.state.notify(f"{NAV_ITEMS[index][0]} — {NAV_ITEMS[index][1]}")

    def _bootstrap(self) -> None:
        """启动时自动挂上已有的数据集与最新权重，省去每次手工选择。

        数据集配置必须校验其 path 真实存在：随程序分发的模板配置写的是
        打包机器的绝对路径，换一台电脑后那个目录并不存在，直接挂上会让
        训练页显示一个假的数据集，点开始才报错。
        """
        yaml_path = config.dataset_yaml_path()
        if yaml_path.exists() and self._yaml_is_valid(yaml_path):
            self.state.dataset_yaml = yaml_path
        weights = self.state.latest_weights()
        if weights:
            self.state.weights = weights
            self.state.notify(f"已加载最新权重：{weights.parent.parent.name}")

    @staticmethod
    def _yaml_is_valid(yaml_path) -> bool:
        import yaml

        try:
            cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return False
        root = Path(cfg.get("path", ""))
        if not root.exists():
            return False
        train = cfg.get("train", "train/images")
        return (root / train).exists()

    # ------------------------------------------------------------ 退出
    def closeEvent(self, event) -> None:
        """关闭窗口前先确认并终止训练。

        训练跑在独立子进程里，直接关窗口会留下孤儿进程继续占显存，
        下次启动再训练就会 OOM，而用户以为已经关掉了。
        """
        active = running_scripts()
        if active:
            reply = QMessageBox.question(
                self, "训练仍在进行",
                f"当前有 {len(active)} 个任务在运行，退出会立即终止它们。\n\n"
                "未完成的训练不会保留最优权重。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            stopped = stop_all_scripts()
            self.state.notify(f"已终止 {stopped} 个任务")

        self.gpu_badge.stop()
        event.accept()
