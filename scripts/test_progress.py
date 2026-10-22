"""验证实时进度链路：训练脚本输出 -> 子进程切行 -> 界面进度条。

不跑真实训练，用假 trainer 驱动回调，避免占用 GPU（用户正在跑 120 轮）。
"""
import io
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

import train as train_script  # noqa: E402
from lps.ui.pages.train_page import TrainPage  # noqa: E402
from lps.ui.state import AppState  # noqa: E402
from lps.ui.workers import emit_stream, is_progress_bar  # noqa: E402
from lps.utils.text import parse_progress_line  # noqa: E402

FAILED = False


def check(label: str, ok: bool, detail: str = "") -> None:
    global FAILED
    print(f"  [{'通过' if ok else '失败'}] {label}  {detail}")
    if not ok:
        FAILED = True


class FakeTrainer:
    def __init__(self, epoch: int = 0, epochs: int = 120, batches: int = 160) -> None:
        self.epoch = epoch
        self.epochs = epochs
        self.train_loader = list(range(batches))


print("① 训练脚本按节流输出 [progress] 行（速率用滑动窗口）")
train_script._train_started_at = time.time() - 100
train_script._batch_state.update(
    {"epoch": -1, "iter": 0, "last_done": 0, "last_time": 0.0, "last_emit": 0.0}
)
trainer = FakeTrainer()
buffer = io.StringIO()
with redirect_stdout(buffer):
    for _ in range(8):                     # 前 8 批间隔极短，窗口不足 0.5s 不报
        train_script.report_batch_progress(trainer)
    time.sleep(0.7)                        # 模拟真实训练耗时
    for _ in range(8):
        train_script.report_batch_progress(trainer)

lines = [ln for ln in buffer.getvalue().splitlines() if ln.startswith("[progress]")]
check("攒够窗口后开始上报", len(lines) >= 1, f"实际 {len(lines)} 行")
check("格式含轮次/批次/速度/剩余",
      bool(lines) and "epoch=1/120" in lines[0] and "iter=16/160" in lines[0]
      and "rate=" in lines[0] and "remain=" in lines[0],
      lines[0] if lines else "(无输出)")

info = parse_progress_line(lines[0]) if lines else None
check("可被解析", info is not None, str(info))
remain = float(info["remain"]) if info and "remain" in info else 1e9
check("剩余时间是合理量级（非 500 小时级）", remain < 3600 * 8,
      f"remain={remain:.0f}s ≈ {remain / 3600:.1f}h")

print("② 界面进度条按批次推进，且不写进日志框")
app = QApplication(sys.argv)
page = TrainPage(AppState())
page.epochs.setValue(120)
page._train_t0 = time.time() - 100
page.log.clear()
page._append_log(lines[0])
check("进度条已推进", page.progress.value() > 0, f"value={page.progress.value()}/10000")
check("进度文字含剩余时间", "剩余" in page.progress.text(), page.progress.text())
check("状态行同步显示", "剩余" in page.hint.text(), page.hint.text()[:60])
check("[progress] 未进日志框", "[progress]" not in page.log.toPlainText())

print("③ 真实粘连行：tqdm 进度条 + 我们的进度行在同一段里")

# 这是从真实训练日志里截下来的原文（tqdm 不换行，我们的一行被粘在尾部）
glued = ("        1/1     0.576G      4.708      4.585      4.315        107        320: "
         "7% ╸─────────── 7/100 2.6it/s 17.9s<35.1s"
         "[progress] epoch=1/1 iter=8/100 rate=3.23 elapsed=42 remain=29")
info2 = parse_progress_line(glued)
check("粘连行也能解析", info2 is not None, str(info2))
check("不被当成进度条丢弃", not is_progress_bar(glued), f"is_progress_bar={is_progress_bar(glued)}")

page2 = TrainPage(AppState())
page2.epochs.setValue(100)
page2._train_t0 = time.time() - 42
page2.log.clear()
page2._append_log(glued)
check("界面进度条被推进", page2.progress.value() == int(8 / 100 / 1 * 10000),
      f"value={page2.progress.value()}/10000 text={page2.progress.text()}")
check("粘连行没污染日志框", page2.log.toPlainText() == "", repr(page2.log.toPlainText()[:30]))

print("④ tqdm 进度条碎片被丢弃、\\r 能正确切行")
raw = ("\rtrain: Scanning ...1594 images\n"
       "      1/120      2.87G      2.811  960: 45% ━━━━━━ 72/160 1.9it/s 2:10<1:30"
       "\r      1/120      2.87G      2.755  960: 46% ━━━━━━ 73/160 1.9it/s 2:11<1:28")
collected = []
emit_stream(raw + "\n", collected.append)
kept = [seg for seg in collected if not is_progress_bar(seg)]
check("扫描日志保留", any("Scanning" in seg for seg in kept), str(kept))
check("进度条碎片全部丢弃", not any("━" in seg for seg in kept),
      f"丢弃 {len(collected) - len(kept)} 段")

print()
print("总体：", "全部通过" if not FAILED else "存在失败项")
sys.exit(1 if FAILED else 0)
