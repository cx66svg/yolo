# -*- coding: utf-8 -*-
"""界面功能验证（离屏运行）。

逐项验证本轮修复的功能点，避免只靠肉眼观察：
    1. 数据集三个划分是否被正确识别并写入配置
    2. 项目列表是否支持多选与批量删除
    3. 输入尺寸是否按数据集分辨率自动建议
    4. 运算设备是否探测到真实显卡
    5. 训练记录能否列出并点击载入历史日志
    6. 日志是否已去除 ANSI 控制符
"""
from __future__ import annotations

import faulthandler
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
faulthandler.enable()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QEventLoop, QItemSelectionModel, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from lps import config  # noqa: E402
from lps.core import ProjectStore  # noqa: E402
from lps.data import load_profile  # noqa: E402
from lps.ui.main_window import MainWindow  # noqa: E402
from lps.ui.style import ensure_fonts  # noqa: E402
from lps.utils.text import strip_ansi  # noqa: E402

SAMPLE = config.OUTPUT_DIR / "datasets" / "behavior4"
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")


def settle(ms: int = 300) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_splits() -> None:
    profile = load_profile(SAMPLE)
    names = list(profile.splits)
    ok = {"train", "val", "test"} <= set(names)
    detail = "、".join(f"{k}={v.images}图" for k, v in profile.splits.items())
    record("① 划分识别", ok, f"识别到 {detail}")

    yaml_path = Path(tempfile.gettempdir()) / "lps_split_check.yaml"
    profile.to_yaml(yaml_path)
    text = yaml_path.read_text(encoding="utf-8")
    has_all = all(key in text for key in ("train:", "val:", "test:"))
    record("① 划分写入配置", has_all, "配置同时包含 train / val / test 三项")
    yaml_path.unlink(missing_ok=True)


def test_multi_delete(window: MainWindow, store: ProjectStore) -> None:
    # 清理上次可能残留的测试项目，保证脚本可重复运行
    for stale in [p for p in store.all() if p.name.startswith("测试_批量")]:
        store.delete(stale.pid)

    created = []
    for i in range(3):
        project = store.create(f"测试_批量{i}", SAMPLE, "批量删除验证")
        created.append(project.pid)
    page = window.project_page
    page.refresh()

    # 只选中刚建的三个项目
    target_rows = []
    for row in range(page.table.rowCount()):
        cell = page.table.item(row, 0)
        if cell and cell.data(Qt.ItemDataRole.UserRole) in created:
            target_rows.append(row)
    # 正确做法：往选择模型里 Select，selectRow 会清掉之前的选择
    selection = page.table.selectionModel()
    selection.clearSelection()
    for row in target_rows:
        index = page.table.model().index(row, 0)
        selection.select(
            index,
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    selected = page._selected_pids()  # noqa: SLF001
    record("② 多选生效", len(selected) == 3, f"一次选中 {len(selected)} 个项目")

    original = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    try:
        page._delete_selected()  # noqa: SLF001
    finally:
        QMessageBox.question = original

    remaining = {p.pid for p in store.all()}
    ok = not (set(created) & remaining)
    record("② 批量删除", ok, f"删除 3 个后剩余 {len(remaining)} 个项目")


def test_imgsz_and_device(window: MainWindow, store: ProjectStore) -> None:
    project = store.by_name("山羊") or (store.all()[0] if store.all() else None)
    if project is None:
        record("③ 输入尺寸建议", False, "没有可用项目，跳过")
        return
    window.state.open_project(project)
    window._switch(1)  # noqa: SLF001
    settle(400)

    page = window.pages[1]
    value = page._imgsz_value()  # noqa: SLF001
    editable = page.imgsz.isEditable()
    record("③ 输入尺寸建议", value > 0 and editable,
           f"自动建议为 {value}，允许手动修改")

    text = page.device.currentText()
    has_gpu = "GPU" in text
    record("④ 运算设备识别", has_gpu, f"下拉框显示：{text}")


def test_run_history(window: MainWindow) -> None:
    page = window.pages[1]
    page._refresh_runs()  # noqa: SLF001
    count = page.run_list.count()
    record("⑤ 训练记录列表", count > 0, f"列出 {count} 次历史训练")

    if count == 0:
        return
    # 直接调用槽函数并在 Python 层捕获异常：
    # PyQt 的槽里若抛出未捕获异常会直接 abort 进程，看不到堆栈
    item = page.run_list.item(0)
    page._on_run_changed(item, None)  # noqa: SLF001
    settle(200)
    text = page.log.toPlainText()
    ok = bool(text.strip())
    record("⑤ 点击载入历史日志", ok, f"载入 {len(text)} 个字符")


def test_ansi() -> None:
    raw = "\x1b[34m\x1b[1mengine\\trainer: \x1b[0mepochs=29"
    cleaned = strip_ansi(raw)
    ok = "\x1b" not in cleaned and "engine" in cleaned
    record("⑥ 日志去控制符", ok, f"清理结果：{cleaned}")

    # 真实训练日志里若还残留 ESC 则说明写文件时没清理
    logs = list((config.OUTPUT_DIR / "runs").glob("*/train.log"))
    dirty = [p for p in logs if "\x1b" in p.read_text(encoding="utf-8", errors="ignore")]
    record("⑥ 日志文件洁净度", not dirty,
           f"检查 {len(logs)} 个日志文件，残留控制符 {len(dirty)} 个")


def main() -> int:
    config.ensure_dirs()
    app = QApplication(sys.argv)
    ensure_fonts()
    window = MainWindow()
    window.resize(1280, 900)
    window.show()
    settle(600)

    store = window.state.store
    for name, func in (
        ("① 划分", test_splits),
        ("② 批量删除", lambda: test_multi_delete(window, store)),
        ("③④ 尺寸与设备", lambda: test_imgsz_and_device(window, store)),
        ("⑤ 训练记录", lambda: test_run_history(window)),
        ("⑥ 日志清理", test_ansi),
    ):
        try:
            func()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            record(name, False, f"{type(exc).__name__}: {exc}")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n通过 {passed}/{len(results)} 项")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
