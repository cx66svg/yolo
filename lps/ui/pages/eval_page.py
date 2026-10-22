"""评估页面：选择权重与划分，输出指标表与日志。"""
from __future__ import annotations

import csv
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ... import config
from ...data import load_profile
from ..state import AppState
from ..widgets import Card, FormGrid, PageHeader, StatRow, list_run_weights, pick_weights_dialog
from ..workers import ScriptProcess

ROOT = Path(__file__).resolve().parent.parent.parent

# 评估页指标标题：英文缩写后面补中文，避免不知道含义
STAT_MAP50 = "mAP@0.5 平均精度"
STAT_MAP5095 = "mAP@0.5:0.95 综合精度"
STAT_MAP75 = "mAP@0.75 严格精度"
STAT_WEAKEST = "最弱类别"


class EvalPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.proc = ScriptProcess(self)
        self.proc.log.connect(self._append_log)
        self.proc.finished.connect(self._on_finished)
        self.proc.error.connect(lambda m: self._append_log(f"[错误] {m}"))
        self._build()
        self.state.dataset_changed.connect(lambda _path: self._reload_splits())
        # 切换项目后权重要重新列（默认选中该项目的权重）
        self.state.dataset_changed.connect(lambda _path: self._reload_weights())
        # 训练完成时会更新 state.weights，这里同步刷新，否则新训练不会出现在下拉框
        self.state.weights_changed.connect(lambda _path: self._reload_weights())
        self._reload_splits()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(PageHeader(
            "模型评估",
            "在验证集或测试集上计算 mAP 与每类 AP。测试集全程不参与训练与调参，用于检验泛化能力。",
        ))

        self.stats = StatRow([STAT_MAP50, STAT_MAP5095, STAT_MAP75, STAT_WEAKEST])
        root.addWidget(self.stats)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left_box = QWidget()
        left_box.setMinimumWidth(470)
        left_box.setMaximumWidth(560)
        left = QVBoxLayout(left_box)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        body.addWidget(left_box, 0)
        left.addWidget(self._config_card())
        left.addWidget(self._class_card())
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, 1)
        right.addWidget(self._log_card())

    def _config_card(self) -> Card:
        card = Card("评估设置")
        form = FormGrid()

        self.weights = QComboBox()
        self.weights.setMinimumWidth(300)
        self._reload_weights()

        pick = QPushButton("…")
        pick.setToolTip("从磁盘选择其他权重文件")
        pick.setFixedWidth(44)
        pick.clicked.connect(self._pick_weights)

        weight_row = QHBoxLayout()
        weight_row.setContentsMargins(0, 0, 0, 0)
        weight_row.addWidget(self.weights, 1)
        weight_row.addWidget(pick)
        holder = QWidget()
        holder.setLayout(weight_row)
        form.add_row("模型权重", holder,
                     "自动列出所有训练过的权重（实验名 · 完成时间），默认选最新一次")

        self.split = QComboBox()
        form.add_row("数据划分", self.split, "只列出数据集里真实存在的划分")
        card.add(form)

        row = QHBoxLayout()
        self.btn_run = QPushButton("开始评估")
        self.btn_run.setObjectName("Primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_export = QPushButton("导出报告")
        self.btn_export.clicked.connect(self._export)
        self.btn_export.setEnabled(False)
        self.hint = QLabel("就绪")
        self.hint.setObjectName("Hint")
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_export)
        row.addWidget(self.hint, 1)
        card.add_layout(row)
        return card

    def _class_card(self) -> Card:
        card = Card("每类表现")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["类别", "AP@0.5:0.95", "说明"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(150)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        card.add(self.table)

        # 空态提示：表格没有数据时不要把卡片留成一片空白
        self.class_hint = QLabel("评估完成后，这里按类别列出 AP，便于定位弱类")
        self.class_hint.setObjectName("Hint")
        card.add(self.class_hint)
        return card

    def _log_card(self) -> Card:
        card = Card("评估日志")
        self.log = QPlainTextEdit()
        self.log.setPlaceholderText("点击「开始评估」后，这里显示推理过程与逐类结果")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        card.add(self.log)
        return card

    # ------------------------------------------------------------ 交互
    def _reload_weights(self) -> None:
        """列出 outputs/runs 下所有训练产出的权重，按时间倒序（最新在最前）。"""
        self.weights.clear()
        items = list_run_weights()
        for index, (text, path) in enumerate(items):
            self.weights.addItem(text, path)
            self.weights.setItemData(index, path, Qt.ItemDataRole.ToolTipRole)
        if not items:
            self.weights.addItem("（暂无已训练权重，请先训练）", "")
        # 默认选中「当前项目的最新权重」；没有项目时退回全局最新
        preferred = ""
        project = self.state.project
        if project and project.best_weights:
            preferred = str(project.best_weights)
        elif self.state.weights:
            preferred = str(self.state.weights)
        if preferred:
            idx = self.weights.findData(preferred)
            if idx >= 0:
                self.weights.setCurrentIndex(idx)

    def _pick_weights(self) -> None:
        path = pick_weights_dialog(self)
        if path:
            self.weights.addItem(Path(path).name, path)
            self.weights.setCurrentIndex(self.weights.count() - 1)

    def _reload_splits(self) -> None:
        """按当前数据集的真实划分填充下拉框。

        YOLO 格式本身只要求 images/labels 成对，train/val/test 是 data.yaml
        里约定的，**不是所有数据集都有 test**（很多只有 train/val）。
        写死「测试集」会让这类数据集一选就报错，这里改为只列真实存在的。
        """
        self.split.clear()
        available: list[str] = []
        yaml_path = self.state.dataset_yaml
        if yaml_path and Path(yaml_path).exists():
            try:
                profile = load_profile(Path(yaml_path))
                available = [key for key in ("val", "test") if key in profile.splits]
            except (OSError, ValueError, KeyError):
                available = []

        labels = {"val": "验证集 val", "test": "测试集 test"}
        for key in ("val", "test"):
            if key in available:
                self.split.addItem(labels[key], key)
        if not available:
            self.split.addItem("（数据集中没有 val/test 划分）", "")
        self.btn_run.setEnabled(bool(available))

    def _run(self) -> None:
        if self.proc.running:
            return
        weights = self.weights.currentData()
        if not weights:
            QMessageBox.information(self, "提示", "请先训练或选择一个权重文件")
            return
        if not self.state.dataset_yaml:
            QMessageBox.information(self, "提示", "请先在「数据集管理」导入数据集")
            return

        split = self.split.currentData()
        if not split:
            QMessageBox.information(
                self, "提示",
                "当前数据集没有 val 或 test 划分，无法评估。\n"
                "YOLO 数据集里这三个划分是可选的，只有 train 时也能训练，但不能做划分评估。",
            )
            return

        # 模型类别数必须与数据集一致：否则 ultralytics 在混淆矩阵里会抛
        # "IndexError: index 2 is out of bounds for axis 1 with size 2"，
        # 报错信息完全看不出原因，所以这里提前拦下来说清楚。
        try:
            from ...core import read_model_names

            model_names = read_model_names(weights)
        except (OSError, RuntimeError, ValueError, ImportError):
            model_names = {}
        if model_names:
            profile = load_profile(Path(self.state.dataset_yaml))
            if len(model_names) != profile.num_classes:
                model_desc = "、".join(str(v) for v in model_names.values())[:60]
                QMessageBox.warning(
                    self, "类别数不匹配",
                    f"这个权重是 {len(model_names)} 类模型（{model_desc}），"
                    f"而当前数据集是 {profile.num_classes} 类。\n\n"
                    "评估会直接失败。请换成用当前数据集训练出来的权重，"
                    "或在「项目管理」里切回与权重匹配的项目。",
                )
                return

        args = [
            "--weights", weights,
            "--data", str(self.state.dataset_yaml),
            "--split", split,
        ]
        if self.state.project is None:
            args += ["--scheme", self.state.scheme_key]
        self.log.clear()
        self._append_log(f"评估权重：{weights}")
        self._append_log(f"数据划分：{split}")
        self.hint.setText("评估中…")
        self.btn_run.setEnabled(False)
        self.proc.start(config.scripts_dir() / "val.py", args, config.APP_ROOT)
        self._pending_split = split

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_finished(self, code: int) -> None:
        self.btn_run.setEnabled(True)
        self.hint.setText("评估完成" if code == 0 else f"评估失败（退出码 {code}）")
        if code == 0:
            self.btn_export.setEnabled(True)
        self._load_report(getattr(self, "_pending_split", "val"))

    def _export(self) -> None:
        """导出评估与训练结果，不重新推理。"""
        from ...services import export_all

        weights = self.weights.currentData()
        run_dir = Path(weights).parent.parent if weights else None
        try:
            result = export_all(run_dir=run_dir, split=self.split.currentData())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", f"{type(exc).__name__}: {exc}")
            return
        QMessageBox.information(self, "导出完成",
                                "\n".join(str(p) for p in result.files))
        self.state.notify(f"已导出 {len(result.files)} 个文件")
        self.btn_export.setEnabled(True)

    def _load_report(self, split: str) -> None:
        path = config.OUTPUT_DIR / "reports" / f"per_class_ap_{split}.csv"
        if not path.exists():
            return
        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
        self.table.setRowCount(0)
        overall: dict[str, str] = {}
        per_class = []
        for row in rows:
            if not row:
                continue
            if row[0] in ("overall",):
                overall[row[1]] = row[3]
                continue
            if row[0] == "class_id":
                continue
            per_class.append(row)

        for row in per_class:
            r = self.table.rowCount()
            self.table.insertRow(r)
            name = f"{row[1]}（{row[2]}）" if len(row) > 2 and row[2] else row[1]
            ap = float(row[3]) if len(row) > 3 else 0.0
            note = "偏低：该类样本少，建议单独处理" if ap < 0.45 else "正常"
            for col, value in enumerate([name, f"{ap:.4f}", note]):
                item = QTableWidgetItem(str(value))
                if col == 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, col, item)

        if overall:
            self.stats.set(STAT_MAP50, overall.get("mAP50", "—"))
            self.stats.set(STAT_MAP5095, overall.get("mAP50-95", "—"))
            self.stats.set(STAT_MAP75, overall.get("mAP75", "—"))
        if per_class:
            weakest = min(per_class, key=lambda x: float(x[3]))
            self.stats.set(STAT_WEAKEST, f"{weakest[1]} {float(weakest[3]):.3f}")
