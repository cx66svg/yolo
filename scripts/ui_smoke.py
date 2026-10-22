# -*- coding: utf-8 -*-
"""界面自检：离屏启动主窗口并逐页截图。

不依赖显示器，可在命令行环境下生成真实的界面截图，
用于确认排版、控件尺寸与中文字体渲染是否正常。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from lps import config  # noqa: E402
from lps.ui import MainWindow  # noqa: E402
from lps.ui.style import ensure_fonts  # noqa: E402


def settle(ms: int = 400) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def main() -> None:
    config.ensure_dirs()
    out = config.OUTPUT_DIR / "ui_shots"
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    print("已注册字体:", ensure_fonts()[:5])
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    settle(600)

    names = ["01_project", "02_train", "03_eval", "04_predict", "05_dataset"]
    for idx, name in enumerate(names):
        window._switch(idx)  # noqa: SLF001
        settle(300)
        pix = window.grab()
        path = out / f"{name}.png"
        pix.save(str(path))
        print(f"已保存 {path}  ({pix.width()}x{pix.height()})")

    print("界面自检完成")


if __name__ == "__main__":
    main()
