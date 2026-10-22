# -*- coding: utf-8 -*-
"""目标检测训练与推理平台 — 桌面端入口。

运行::

    python app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# 启动日志：无控制台模式下排查启动卡在哪里的唯一手段。
# 直接写到当前工作目录（exe 双击时即为 exe 所在目录）。
_startup_log = None


def _log(msg: str) -> None:
    global _startup_log
    if _startup_log is None:
        try:
            _startup_log = open(Path.cwd() / "startup.log", "w", encoding="utf-8", errors="replace")
        except Exception:
            return
    try:
        import datetime
        _startup_log.write(f"{datetime.datetime.now():%H:%M:%S.%f} {msg}\n")
        _startup_log.flush()
    except Exception:
        pass


_log("app.py loaded")

# 捕获段错误/C 层崩溃并写进启动日志
import faulthandler  # noqa: E402
faulthandler.enable(_startup_log if _startup_log else sys.stderr)
_log("faulthandler enabled")


# 所有 import 都放在 _main() 内部，这样打包后即使缺失依赖也能把 traceback
# 写进 crash.log；否则模块级导入异常会直接让无控制台程序静默退出。
def _try_single_instance(lock_path: Path) -> tuple[bool, Optional[object]]:
    """使用 Windows 文件锁实现单实例。

    QLockFile 在打包后不能可靠地跨进程锁定，改用 msvcrt.locking
    对 .app.lock 加排他锁；锁在文件句柄关闭时自动释放。
    """
    import msvcrt

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(lock_path, "wb")
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        return True, f
    except OSError:
        try:
            f.close()
        except Exception:
            pass
        return False, None


def _main() -> int:
    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))

    _log("importing PyQt6")
    from PyQt6.QtCore import Qt  # noqa: E402
    from PyQt6.QtGui import QFont  # noqa: E402
    from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

    _log("importing lps")
    from lps import __app_name__, __version__, config  # noqa: E402
    _log("imported lps.config")
    from lps.ui import MainWindow, style  # noqa: E402
    _log("imports done")

    _log("ensure_dirs")
    config.ensure_dirs()

    _log("creating QApplication")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    _log("ensure_fonts")
    style.ensure_fonts()          # 显式注册中文字体，避免渲染成方框
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setFont(QFont(style.resolve_family(), 9))

    # 单实例锁：同时开两个窗口会让两份训练抢同一块显存，
    # 结果就是第一次参数更新时 CUDA out of memory（8GB 卡上很容易踩到）。
    _log("single-instance lock")
    lock_path = Path.cwd() / ".app.lock"
    locked, _lock_handle = _try_single_instance(lock_path)
    if not locked:
        QMessageBox.warning(
            None, "已经有一个窗口在运行",
            "检测到本程序已在运行。\n\n"
            "同时打开两个窗口会导致两个训练同时抢显存，直接报显存不足。\n"
            "请先关闭已打开的窗口，或在任务栏里切换到它。",
        )
        return 1

    _log("creating MainWindow")
    window = MainWindow()
    _log("showing MainWindow")
    window.show()
    _log("entering event loop")
    code = app.exec()
    if _lock_handle is not None:
        try:
            _lock_handle.close()
        except Exception:
            pass
    return code


def main() -> int:
    """入口包装：把启动期的异常写入 crash.log，方便在打包后排查。"""
    try:
        return _main()
    except Exception as exc:
        try:
            # 即使 config 没导入，也要尽量把错误写到当前目录
            crash = Path.cwd() / "crash.log"
            with open(crash, "w", encoding="utf-8", errors="replace") as f:
                f.write(f"启动失败：{type(exc).__name__}: {exc}\n\n")
                import traceback
                traceback.print_exc(file=f)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    sys.exit(main())
