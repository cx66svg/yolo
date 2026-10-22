"""GPU 资源指示器。

单次 ``nvidia-smi`` 查询约 170ms，放在界面线程里每 2 秒卡一下，鼠标都会顿。
所以查询放到独立线程，结果通过信号回到界面。
"""
from __future__ import annotations

import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QLabel, QWidget

from ..services.gpu import GpuStats, query_gpus
from . import style


class GpuMonitor(QThread):
    """后台轮询显卡状态。"""

    updated = pyqtSignal(list)          # list[GpuStats]

    def __init__(self, interval_ms: int = 2000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._interval = interval_ms

    def run(self) -> None:
        # Windows 子线程里调 subprocess 可能触发 0x8001010d COM 异常，
        # 先初始化 COM 公寓模式即可避免。
        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.ole32.CoInitializeEx(None, 0)
            except Exception:
                pass
        try:
            while not self.isInterruptionRequested():
                self.updated.emit(query_gpus())
                # 用可中断的等待，退出时不必等满一个周期
                for _ in range(self._interval // 100):
                    if self.isInterruptionRequested():
                        return
                    self.msleep(100)
        finally:
            if sys.platform == "win32":
                try:
                    import ctypes

                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(1500)


class GpuBadge(QLabel):
    """状态栏右侧的 GPU 状态标签。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("GPU 检测中…", parent)
        self.setObjectName("GpuBadge")
        self.setStyleSheet(f"color:{style.TEXT_MUTED}; padding:0 10px;")
        self._monitor = GpuMonitor(parent=self)
        self._monitor.updated.connect(self._on_update)

    def start(self) -> None:
        self._monitor.start()

    def stop(self) -> None:
        self._monitor.stop()

    def _on_update(self, gpus: list) -> None:
        if not gpus:
            self.setText("未检测到 NVIDIA 显卡")
            self.setStyleSheet(f"color:{style.TEXT_MUTED}; padding:0 10px;")
            return

        gpu: GpuStats = gpus[0]
        self.setText(gpu.summary())
        if gpu.mem_percent >= 92 or gpu.temperature >= 85:
            self.setToolTip("显存或温度接近上限，建议调小批大小或输入尺寸")
            self.setStyleSheet("color:#C0392B; font-weight:600; padding:0 10px;")
        elif gpu.utilization >= 30:
            self.setStyleSheet("color:#1E8449; padding:0 10px;")
        else:
            self.setStyleSheet(f"color:{style.TEXT_MUTED}; padding:0 10px;")
