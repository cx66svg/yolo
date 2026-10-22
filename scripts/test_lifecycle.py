"""验证三件事：日志能实时流出、关闭窗口会真的杀掉训练进程、GPU 状态可读。"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from lps.services.gpu import query_gpus  # noqa: E402
from lps.ui.workers import ScriptProcess, running_scripts, stop_all_scripts  # noqa: E402

FAILED = False


def check(label: str, ok: bool, detail: str = "") -> None:
    global FAILED
    print(f"  [{'通过' if ok else '失败'}] {label}  {detail}")
    if not ok:
        FAILED = True


def settle(ms: int) -> None:
    end = time.time() + ms / 1000.0
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def pid_alive(pid: int) -> bool:
    """PID 是否仍存在（用于确认子进程真的死了，而不是变成孤儿）。"""
    if pid <= 0:
        return False
    try:
        # tasklist 输出是 GBK，按字节比对即可（PID 本身是 ASCII）
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return str(pid).encode() in (result.stdout or b"")


app = QApplication(sys.argv)

print("① 子进程输出能逐行流到界面（不再卡在缓冲区）")
received = []
proc = ScriptProcess()
proc.log.connect(received.append)
proc.start(ROOT / "scripts" / "_dummy_train.py", ["20"], ROOT)
settle(1500)
check("已注册为运行中任务", len(running_scripts()) == 1, f"{len(running_scripts())} 个")
check("收到日志行", len(received) >= 3, f"{len(received)} 行")
check("日志内容正确", any("模拟训练中" in line for line in received),
      received[0] if received else "(空)")

print("② 停止后进程真的死了（不留孤儿占显存）")
pid = proc.process_id
check("拿到子进程 PID", pid > 0, f"pid={pid}")
check("停止前进程存活", pid_alive(pid), f"pid={pid}")

stopped = stop_all_scripts()
settle(2500)
check("stop_all 报告已终止", stopped == 1, f"stopped={stopped}")
check("该 PID 已消失", not pid_alive(pid), f"pid={pid}")
check("注册表已清空", len(running_scripts()) == 0)

print("③ GPU 状态可读")
gpus = query_gpus()
check("读到显卡", len(gpus) >= 1, gpus[0].summary() if gpus else "无显卡")
if gpus:
    g = gpus[0]
    check("显存数值合理", 0 < g.mem_used <= g.mem_total, f"{g.mem_used}/{g.mem_total} MB")
    check("温度可读", 0 < g.temperature < 120, f"{g.temperature}°C")
    check("摘要格式正确", "显存" in g.summary() and "%" in g.summary(), g.summary())

print()
print("总体：", "全部通过" if not FAILED else "存在失败项")
sys.exit(1 if FAILED else 0)
