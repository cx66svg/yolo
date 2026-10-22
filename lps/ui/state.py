"""界面共享状态。

各页面不直接互相引用，统一通过 AppState 读写当前数据集、类别方案与权重，
并靠信号做联动刷新。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from .. import config


class AppState(QObject):
    """全局可变状态。"""

    dataset_changed = pyqtSignal(object)   # Path | None
    scheme_changed = pyqtSignal(str)
    weights_changed = pyqtSignal(object)   # Path | None
    project_changed = pyqtSignal(object)   # Project | None
    status_message = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._dataset_yaml: Optional[Path] = None
        self._scheme_key: str = config.ACTIVE_SCHEME_KEY
        self._weights: Optional[Path] = None
        self._source_root: Optional[Path] = config.DATA_ROOT
        self._project = None

        # 项目建档的持久化存储，所有页面共用同一份实例
        from ..core.project import ProjectStore

        self.store = ProjectStore()

    # ---------------------------------------------------------- 项目
    @property
    def project(self):
        return self._project

    def open_project(self, project) -> None:
        """切换当前项目，并把它的数据集与权重同步到共享状态。"""
        self._project = project
        if project is None:
            self.project_changed.emit(None)
            return
        self._dataset_yaml = Path(project.data_yaml)
        self.dataset_changed.emit(self._dataset_yaml)
        if project.best_weights and Path(project.best_weights).exists():
            self._weights = Path(project.best_weights)
            self.weights_changed.emit(self._weights)
        self.project_changed.emit(project)

    # ---------------------------------------------------------- 数据集
    @property
    def dataset_yaml(self) -> Optional[Path]:
        return self._dataset_yaml

    @dataset_yaml.setter
    def dataset_yaml(self, value: Optional[Path]) -> None:
        self._dataset_yaml = Path(value) if value else None
        self.dataset_changed.emit(self._dataset_yaml)

    @property
    def source_root(self) -> Optional[Path]:
        return self._source_root

    @source_root.setter
    def source_root(self, value: Optional[Path]) -> None:
        self._source_root = Path(value) if value else None

    # ---------------------------------------------------------- 类别方案
    @property
    def scheme(self):
        """当前预设方案；导入任意数据集时可能没有预设，返回 None。"""
        return config.CLASS_SCHEMES.get(self._scheme_key)

    @property
    def scheme_key(self) -> str:
        return self._scheme_key

    def set_scheme(self, key: str) -> None:
        if key == self._scheme_key or key not in config.CLASS_SCHEMES:
            return
        self._scheme_key = key
        config.set_scheme(key)
        self.scheme_changed.emit(key)

    # ---------------------------------------------------------- 权重
    @property
    def weights(self) -> Optional[Path]:
        return self._weights

    @weights.setter
    def weights(self, value: Optional[Path]) -> None:
        self._weights = Path(value) if value else None
        self.weights_changed.emit(self._weights)

    # ---------------------------------------------------------- 工具
    def notify(self, message: str) -> None:
        self.status_message.emit(message)

    def latest_weights(self) -> Optional[Path]:
        """扫描输出目录，取最新的 best.pt。"""
        candidates = sorted(
            config.RUNS_DIR.glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        return candidates[-1] if candidates else None
