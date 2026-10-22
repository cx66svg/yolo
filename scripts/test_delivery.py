"""交付前全量自检：逐页创建、关键控件、数据一致性。

任何一处抛异常都会打印堆栈并计入失败，避免"看起来能开"但点进去就崩。
"""
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(r"C:\Users\Administrator\Desktop\livestock_posture")
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

FAILED = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}  {detail}")
    if not ok:
        FAILED.append(label)


app = QApplication(sys.argv)

print("① 主窗口与五个页面")
from lps.ui.main_window import MainWindow  # noqa: E402

win = MainWindow()
check("主窗口创建", win is not None)
check("页面数量为 5", len(win.pages) == 5, f"{len(win.pages)}")

names = ["项目管理", "模型训练", "模型评估", "检测推理", "数据集管理"]
for idx, page in enumerate(win.pages):
    try:
        win.stack.setCurrentIndex(idx)
        app.processEvents()
        check(f"{names[idx]} 可切换", True, type(page).__name__)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        check(f"{names[idx]} 可切换", False, str(exc))

print("\n② 训练页关键控件")
train = win.pages[1]
check("轮次/批大小/尺寸/线程存在",
      all(hasattr(train, k) for k in ("epochs", "batch", "imgsz", "workers")))
check("指标块 5 个", len(train.stats.cards) == 5, str(list(train.stats.cards)))
check("尺寸默认 640", train.imgsz.currentText().startswith("640"), train.imgsz.currentText())
check("线程默认 2", train.workers.currentData() == 2, str(train.workers.currentData()))
check("显存预检可用（空闲充足时不弹窗）", train._check_vram(640, 10) is True)

print("\n③ 评估页：权重列表与划分")
ev = win.pages[2]
texts = [ev.weights.itemText(i) for i in range(ev.weights.count())]
check("权重列表非空", auto_ok := len([t for t in texts if "·" in t]) > 0, str(texts[:2]))
check("权重显示带时间", any("·" in t for t in texts))
splits = [ev.split.itemText(i) for i in range(ev.split.count())]
check("划分来自真实数据集", len(splits) >= 1, str(splits))

print("\n④ 推理页")
pd = win.pages[3]
check("权重列表存在", pd.weights.count() >= 1, f"{pd.weights.count()} 项")

print("\n⑤ 项目数据一致性")
from lps.core import ProjectStore  # noqa: E402

store = ProjectStore()
for proj in store.all():
    ok, msg = store.validate(proj)
    check(f"项目 {proj.name} 可用", ok, msg)
    check(f"项目 {proj.name} 有训练记录", bool(proj.runs), str(proj.runs))

print("\n⑥ 模块导入（脚本入口可运行）")
for name, target in (("train", ROOT / "scripts" / "train.py"),
                     ("val", ROOT / "scripts" / "val.py"),
                     ("predict", ROOT / "scripts" / "predict.py"),
                     ("check_dataset", ROOT / "check_dataset.py")):
    try:
        sys.argv = [name, "--help"]
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"_chk_{name}", target)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            check(f"scripts/{name}.py 可加载", True)
        except SystemExit:
            check(f"scripts/{name}.py 可加载", True, "argparse 正常退出")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        check(f"scripts/{name}.py 可加载", False, str(exc))

print("\n⑦ 资源清理")
lock = ROOT / ".app.lock"
check("锁文件不残留（应用未运行时）", not lock.exists() or lock.stat().st_size == 0,
      "存在但为空，下次启动会自动接管")

win.close()
print()
if FAILED:
    print("存在问题：", FAILED)
else:
    print("全部自检通过")
sys.exit(1 if FAILED else 0)
