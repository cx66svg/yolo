"""离屏驱动真实界面完成一次训练，用于找出软件层面的问题。

模拟用户操作路径：打开项目 → 设 20 轮 → 点开始训练 → 等结束 → 校验产出。
不直接调 train.py，这样 ScriptProcess、进度解析、记录刷新、导出解锁全部被真实走一遍。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from lps import config  # noqa: E402
from lps.ui.main_window import MainWindow  # noqa: E402

RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "gui20"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TIMEOUT_S = int(sys.argv[3]) if len(sys.argv) > 3 else 1800
PROJECT = sys.argv[4] if len(sys.argv) > 4 else "GOAT"


def pump(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.05)


app = QApplication(sys.argv)
win = MainWindow()
win.show()

store = win.state.store
project = store.by_name(PROJECT)
if project is None:
    print(f"[失败] 找不到项目 {PROJECT}")
    sys.exit(1)

win.state.open_project(project)
page = win.pages[1]
pump(0.5)

print(f"项目：{project.name} | {project.num_classes} 类 | 配置 {Path(project.data_yaml).name}")
print(f"界面默认：尺寸={page.imgsz.currentText()} 线程={page.workers.currentData()} "
      f"batch={page.batch.value()} 设备={page.device.currentData()}")

page.epochs.setValue(EPOCHS)
page.batch.setValue(10)
page.run_name.setText(RUN_NAME)
run_dir = config.RUNS_DIR / RUN_NAME
if run_dir.exists():
    print(f"[失败] 实验名 {RUN_NAME} 已存在")
    sys.exit(1)

print(f"开始训练：{RUN_NAME}，{EPOCHS} 轮")
started = time.time()
page._start()

max_progress = 0
progress_samples = []
hint_samples = []
log_seen = False
last_report = 0.0
finished = False

while time.time() - started < TIMEOUT_S:
    QApplication.processEvents()
    pump(0.2)

    value = page.progress.value()
    if value > max_progress:
        max_progress = value
    text = page.progress.text()
    if text and (not progress_samples or progress_samples[-1] != text):
        progress_samples.append(text)
    hint = page.hint.text()
    if hint and (not hint_samples or hint_samples[-1] != hint):
        hint_samples.append(hint)
    if page.log.toPlainText().strip():
        log_seen = True

    elapsed = time.time() - started
    if elapsed - last_report >= 60:
        last_report = elapsed
        print(f"  [{elapsed / 60:.1f}m] 进度 {value}/10000  {text}")

    if not page.proc.running and value >= 0:
        finished = True
        break

elapsed = time.time() - started
print(f"\n训练结束，用时 {elapsed / 60:.1f} 分钟，退出状态：{'完成' if finished else '超时'}")

# ---------------------------------------------------------------- 自检
problems = []

csv_path = run_dir / "results.csv"
rows = 0
if csv_path.exists():
    import csv as _csv

    with csv_path.open(encoding="utf-8") as f:
        rows = len(list(_csv.DictReader(f)))
else:
    problems.append("没有生成 results.csv")

best = run_dir / "weights" / "best.pt"
if not best.exists():
    problems.append("没有生成 best.pt")

if max_progress <= 0:
    problems.append("进度条全程没有推进")
if not any("剩余" in t for t in progress_samples):
    problems.append("进度文字里没出现预计剩余时间")
if not log_seen:
    problems.append("日志区没有内容")
if not page.btn_export.isEnabled():
    problems.append("训练完成后「导出交付包」按钮仍不可点")

print(f"\n产出检查：results.csv {rows} 行 | best.pt {'有' if best.exists() else '无'} "
      f"| 进度峰值 {max_progress}/10000 | 进度文字样本 {len(progress_samples)} 条")
for text in progress_samples[:3]:
    print("   进度：", text)
for text in hint_samples[:2]:
    print("   状态：", text)

registered = [r for r in store.all() if r.pid == project.pid]
if registered:
    runs = registered[0].runs
    print(f"项目登记的训练记录：{len(runs)} 条 {runs[-2:] if len(runs) > 2 else runs}")
    if not any(RUN_NAME in str(item) for item in runs):
        problems.append("项目里没有登记这次训练")

pending = [p for p in config.RUNS_DIR.glob(f"*{RUN_NAME}*") if p.is_dir()]
print(f"输出目录：{[p.name for p in pending]}")

print()
if problems:
    print("发现问题：")
    for item in problems:
        print("  ✗", item)
else:
    print("检查项全部通过")

win.close()
sys.exit(1 if problems else 0)
