"""后台任务封装。

训练与验证有两种运行方式，由是否处于打包环境决定：

  1. 源码运行：用 QProcess 起 **独立进程**。ultralytics 训练无法优雅中断，
     进程可以直接 terminate；且不占用界面线程。
  2. 打包成 exe 后：磁盘上没有 python.exe，scripts/*.py 也不是独立文件，
     子进程方式失效，改为在 **QThread 内直接调用脚本的 main()**，
     通过重定向 stdout 把日志推回界面。

外部只需要用 ScriptProcess，两种模式对调用方没有差别。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, pyqtSignal

from ..utils.text import (PROGRESS_PREFIX, clean_log_line,
                          looks_like_progress_bar)


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包环境里。"""
    return bool(getattr(sys, "frozen", False))


# 所有在跑的任务。关闭窗口时要靠它确认「真的把训练停了」，
# 否则子进程会变成孤儿继续占显存，用户以为关了其实还在跑。
_ACTIVE_SCRIPTS: List["ScriptProcess"] = []


def running_scripts() -> List["ScriptProcess"]:
    """当前仍在运行的任务。"""
    return [proc for proc in _ACTIVE_SCRIPTS if proc.running]


def stop_all_scripts() -> int:
    """终止全部任务，返回被终止的数量。"""
    stopped = 0
    for proc in list(_ACTIVE_SCRIPTS):
        if proc.running:
            proc.stop()
            stopped += 1
    return stopped


def is_progress_bar(segment: str) -> bool:
    """判断是否是 tqdm 进度条的碎片。

    ultralytics 用 ``\\r`` 原地刷新进度条（``1/120 2.87G ... 45% ━━━ 72/160 1.9it/s``），
    这些片段对日志框没有信息价值，而且一秒钟能刷十几次，必须丢掉。
    """
    # 带了进度标记的行必须先放行给界面解析，否则会被当成进度条碎片丢掉
    if PROGRESS_PREFIX in segment:
        return False
    return looks_like_progress_bar(segment)


def emit_stream(buffer: str, emit) -> str:
    """按 ``\\n`` 与 ``\\r`` 切分缓冲，逐段回调，返回未完成的尾部。

    只按 ``\\n`` 切会导致 tqdm 进度条永远凑不成完整行（它不换行），
    整段输出卡在缓冲区里，界面看起来就像训练卡死了。
    """
    remaining = buffer
    while True:
        position = -1
        for index, char in enumerate(remaining):
            if char in "\r\n":
                position = index
                break
        if position < 0:
            return remaining
        segment, remaining = remaining[:position], remaining[position + 1:]
        if segment.strip():
            emit(segment)


class _SignalWriter:
    """把写入重定向成信号（打包模式用）。"""

    def __init__(self, emit) -> None:
        self._emit = emit
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        self._buf = emit_stream(self._buf, self._emit_line)
        return len(text)

    def _emit_line(self, segment: str) -> None:
        segment = clean_log_line(segment)
        if not segment.strip() or is_progress_bar(segment):
            return
        self._emit(segment)

    def flush(self) -> None:
        if self._buf.strip():
            self._emit_line(self._buf)
            self._buf = ""


class _InProcessWorker(QThread):
    """在界面进程内跑脚本，供打包环境使用。"""

    log = pyqtSignal(str)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, script: Path, args: List[str], parent: Optional[QObject] = None):
        super().__init__(parent)
        self._script = Path(script)
        self._args = list(args)
        print("[提示] 打包模式下训练在本进程内执行，停止按钮可能延迟生效")

    def run(self) -> None:
        old_stdout, old_argv = sys.stdout, sys.argv
        writer = _SignalWriter(self.log.emit)
        sys.stdout = writer
        sys.stderr = writer
        sys.argv = [str(self._script), *self._args]
        code = 0
        try:
            spec = importlib.util.spec_from_file_location(
                f"_lps_task_{self._script.stem}", str(self._script)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载脚本 {self._script}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.main()
        except SystemExit as exc:                      # argparse 正常退出
            code = int(exc.code or 0)
        except Exception as exc:                       # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")
            self.log.emit(traceback.format_exc())
            code = 1
        finally:
            writer.flush()
            sys.stdout, sys.argv = old_stdout, old_argv
        self.finished.emit(code)

    def stop(self) -> None:
        # 线程无法安全强杀，只能请求中断；训练会在下一轮回调处响应
        self.requestInterruption()


class ScriptProcess(QObject):
    """运行一个 python 脚本并把输出按行转成信号。"""

    log = pyqtSignal(str)
    finished = pyqtSignal(int)          # 退出码
    error = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._drain)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)
        self._buffer = ""
        self._thread: Optional[_InProcessWorker] = None
        _ACTIVE_SCRIPTS.append(self)

    # ------------------------------------------------------------ 控制
    def start(self, script: Path, args: List[str], workdir: Path) -> None:
        if is_frozen():
            self._start_in_thread(script, args)
            return

        self._buffer = ""
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self._proc.setProcessEnvironment(env)
        self._proc.setWorkingDirectory(str(workdir))
        self._proc.start(sys.executable, ["-u", str(script), *args])
        if not self._proc.waitForStarted(5000):
            self.error.emit("进程启动失败，请检查 Python 环境")

    def _start_in_thread(self, script: Path, args: List[str]) -> None:
        self._thread = _InProcessWorker(script, args, self)
        self._thread.log.connect(self.log)
        self._thread.finished.connect(self.finished)
        self._thread.error.connect(self.error)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop()
            return
        if self._proc.state() != QProcess.ProcessState.NotRunning:
            # Windows 下 terminate 对无窗口的控制台进程常常无效，1.5 秒后强杀。
            # 训练进程必须真的死掉：否则它继续吃显存，界面却显示已停止。
            self._proc.terminate()
            if not self._proc.waitForFinished(1500):
                self._proc.kill()
                self._proc.waitForFinished(1500)

    @property
    def process_id(self) -> int:
        """子进程 PID；线程模式或未启动时为 0。"""
        try:
            return int(self._proc.processId())
        except (RuntimeError, TypeError, ValueError):
            return 0

    @property
    def running(self) -> bool:
        if self._thread is not None:
            return self._thread.isRunning()
        return self._proc.state() != QProcess.ProcessState.NotRunning

    # ------------------------------------------------------------ 内部
    def _drain(self) -> None:
        chunk = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buffer += chunk
        self._buffer = emit_stream(self._buffer, self._emit_line)

    def _emit_line(self, segment: str) -> None:
        segment = clean_log_line(segment)
        if not segment.strip() or is_progress_bar(segment):
            return
        self.log.emit(segment)

    def _on_finished(self, code: int, _status) -> None:
        if self._buffer.strip():
            self.log.emit(clean_log_line(self._buffer))
            self._buffer = ""
        if self in _ACTIVE_SCRIPTS:
            _ACTIVE_SCRIPTS.remove(self)
        self.finished.emit(code)

    def _on_error(self, err) -> None:
        self.error.emit(f"进程错误: {err}")


class MetricsWatcher(QObject):
    """轮询 results.csv，把训练曲线数据推给界面。

    训练日志里虽然有每轮指标，但格式随版本变化，读 CSV 更稳。
    """

    updated = pyqtSignal(dict)          # {列名: [值...]}
    epoch_done = pyqtSignal(int, dict)  # 当前轮次, 本轮关键指标

    KEYS = ("epoch", "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss")

    def __init__(self, interval_ms: int = 2000, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)
        self._csv: Optional[Path] = None
        self._run_dir: Optional[Path] = None
        self._last_rows = -1
        self._started_at = 0.0

    def watch(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._csv = self._run_dir / "results.csv"
        self._last_rows = -1
        self._started_at = time.time()
        self._timer.start()

    @property
    def csv_path(self) -> Optional[Path]:
        """实际被监听的 results.csv，可能在训练开始后发生变化。"""
        return self._csv

    @property
    def run_dir(self) -> Optional[Path]:
        if self._csv is not None:
            return self._csv.parent
        return self._run_dir

    def stop(self) -> None:
        self._timer.stop()

    def _newest_results(self) -> Optional[Path]:
        """在输出根目录下找**本次训练产生**的 results.csv。

        训练脚本会把实验名收敛成 ASCII，与界面输入的名称可能不同
        （例如中文名会被净化），此时按目录名监听会一直等不到文件。

        这里必须限定「文件时间晚于本次开始监听的时刻」，否则会认领目录里
        别的历史实验：曾出现界面显示 2/120、mAP 0.4272，实际是上一次
        1 轮实验的数字，而当前训练连第一轮验证都没做完。
        """
        parent = self._run_dir.parent if self._run_dir else None
        if not parent or not parent.exists():
            return None
        cutoff = self._started_at - 5.0          # 容忍 5 秒的时钟误差
        candidates = [
            p for p in parent.glob("*/results.csv")
            if p.stat().st_mtime >= cutoff
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _poll(self) -> None:
        if self._csv is None or not self._csv.exists():
            found = self._newest_results()
            if found is None:
                return
            self._csv = found
            self._last_rows = -1
        try:
            with self._csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error):
            return
        if not rows or len(rows) == self._last_rows:
            return
        self._last_rows = len(rows)

        history: Dict[str, List[float]] = {}
        for key in self.KEYS:
            if key not in rows[0]:
                continue
            values: List[float] = []
            for r in rows:
                try:
                    values.append(float(r[key]))
                except (TypeError, ValueError):
                    values.append(0.0)
            history[key] = values

        self.updated.emit(history)
        last = rows[-1]
        snapshot = {}
        for key in ("metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in last:
                try:
                    snapshot[key] = float(last[key])
                except (TypeError, ValueError):
                    pass
        try:
            epoch = int(float(last.get("epoch", 0)))
        except (TypeError, ValueError):
            epoch = 0
        self.epoch_done.emit(epoch, snapshot)
