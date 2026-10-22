"""推理页面：图片/目录/视频/摄像头检测，结果预览与统计。"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ... import config
from ..charts import BarChart
from ..state import AppState
from ..widgets import Card, FormGrid, PageHeader, StatRow, list_run_weights, pick_weights_dialog
from ..workers import ScriptProcess

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_NAME = "gui"


class PredictPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.proc = ScriptProcess(self)
        self.proc.log.connect(self._append_log)
        self.proc.finished.connect(self._on_finished)
        self.proc.error.connect(lambda m: self._append_log(f"[错误] {m}"))
        self._build()
        self.state.dataset_changed.connect(lambda _path: self._reload_weights())
        # 训练完成时会更新 state.weights，这里同步刷新，否则新训练不会出现在下拉框
        self.state.weights_changed.connect(lambda _path: self._reload_weights())

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(PageHeader(
            "检测推理",
            "对图片、视频或摄像头画面执行姿态识别，输出标注画面、类别分布与目标属性统计。",
        ))

        self.stats = StatRow(["处理帧数", "目标总数", "主要类别", "重度遮挡"])
        root.addWidget(self.stats)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        body.addLayout(left, 0)
        left.addWidget(self._config_card())
        left.addWidget(self._stat_card())
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, 1)
        right.addWidget(self._preview_card(), 3)
        right.addWidget(self._log_card(), 2)

    def _config_card(self) -> Card:
        card = Card("检测设置")
        form = FormGrid()

        self.weights = QComboBox()
        self._reload_weights()

        pick = QPushButton("…")
        pick.setToolTip("从磁盘选择其他权重文件或训练结果文件夹")
        pick.setFixedWidth(44)
        pick.clicked.connect(self._pick_weights)

        weight_row = QHBoxLayout()
        weight_row.setContentsMargins(0, 0, 0, 0)
        weight_row.addWidget(self.weights, 1)
        weight_row.addWidget(pick)
        holder = QWidget()
        holder.setLayout(weight_row)
        form.add_row("模型权重", holder,
                     "只识别该权重训练时定义的类别；换成别的数据集训练的权重，类别名会完全不同")

        self.src_type = QComboBox()
        for label, key in (("单张图片", "image"), ("图片目录", "dir"),
                           ("视频文件", "video"), ("摄像头", "camera")):
            self.src_type.addItem(label, key)
        self.src_type.currentIndexChanged.connect(self._on_type_changed)
        form.add_row("输入类型", self.src_type)

        row = QHBoxLayout()
        self.src_path = QComboBox()
        self.src_path.setEditable(True)
        self.src_path.lineEdit().setPlaceholderText("路径或摄像头编号 0")
        self.btn_browse = QPushButton("浏览")
        self.btn_browse.clicked.connect(self._browse)
        row.addWidget(self.src_path, 1)
        row.addWidget(self.btn_browse)
        src_label = QLabel("输入来源")
        src_label.setObjectName("Hint")
        card.add(src_label)
        card.add_layout(row)
        card.add(form)

        params = QHBoxLayout()
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.1, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.50)
        params.setSpacing(8)
        params.addWidget(QLabel("置信度"))
        params.addWidget(self.conf, 1)
        params.addSpacing(6)
        params.addWidget(QLabel("IoU"))
        params.addWidget(self.iou, 1)
        card.add_layout(params)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("开始检测")
        self.btn_run.setObjectName("Primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_export = QPushButton("导出报告")
        self.btn_export.clicked.connect(self._export)
        self.btn_export.setEnabled(False)
        self.hint = QLabel("就绪")
        self.hint.setObjectName("Hint")
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.hint, 1)
        card.add_layout(btn_row)
        return card

    def _export(self) -> None:
        """把本次推理的统计结果导出为 Markdown 报告。"""
        from ...services import export_detection_report

        path = export_detection_report(OUT_NAME)
        if path is None:
            QMessageBox.information(self, "提示", "尚未产生检测结果，请先执行检测")
            return
        QMessageBox.information(self, "导出完成", str(path))
        self.state.notify(f"检测报告已导出：{path.name}")

    def _stat_card(self) -> Card:
        card = Card("类别分布")
        self.chart = BarChart()
        self.chart.setMinimumHeight(150)
        card.add(self.chart)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["属性", "分布"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(150)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.setVisible(False)          # 没结果时收起，避免一张空表
        card.add(self.table)

        self.dist_hint = QLabel("检测完成后，这里列出每个类别的数量与目标属性分布")
        self.dist_hint.setObjectName("Hint")
        card.add(self.dist_hint)
        return card

    def _preview_card(self) -> Card:
        card = Card("检测画面")
        self.preview = QLabel("尚未执行检测")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet(
            "background:#F8FAFD; color:#9CA3AF; border:1px dashed #D6DDEA;"
            "border-radius:10px; font-size:12px;"
        )
        card.add(self.preview)
        return card

    def _log_card(self) -> Card:
        card = Card("运行日志")
        self.log = QPlainTextEdit()
        self.log.setPlaceholderText("点击「开始检测」后，这里显示推理过程与统计信息")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        card.add(self.log)
        return card

    # ------------------------------------------------------------ 交互
    def _reload_weights(self) -> None:
        """列出所有训练产出的权重，按时间倒序。"""
        self.weights.clear()
        items = list_run_weights()
        for text, path in items:
            self.weights.addItem(text, path)
        if not items:
            self.weights.addItem("（暂无权重，请先训练）", "")
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

    def _on_type_changed(self) -> None:
        is_camera = self.src_type.currentData() == "camera"
        self.btn_browse.setEnabled(not is_camera)
        self.src_path.setEnabled(True)
        self.src_path.lineEdit().setPlaceholderText(
            "摄像头编号，通常为 0" if is_camera else "路径或摄像头编号 0")

    def _browse(self) -> None:
        kind = self.src_type.currentData()
        if kind == "image":
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        elif kind == "video":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if path:
            self.src_path.setEditText(path)

    def _run(self) -> None:
        if self.proc.running:
            return
        weights = self.weights.currentData()
        if not weights:
            QMessageBox.information(self, "提示", "请先训练模型或选择一个权重")
            return
        source = self.src_path.currentText().strip() or (
            "0" if self.src_type.currentData() == "camera" else "")
        if not source:
            QMessageBox.information(self, "提示", "请指定输入来源")
            return

        # 权重类别与当前项目不一致时提醒：例如用单类 goat 模型去检测 4 类行为数据集，
        # 结果只会全是 goat，用户会以为软件出错。
        try:
            from ...core import read_model_names

            model_names = read_model_names(weights)
        except (OSError, RuntimeError, ValueError, ImportError):
            model_names = {}
        if model_names:
            project = self.state.project
            expected = set(project.class_names.values()) if project else set()
            if expected and set(str(v) for v in model_names.values()) != expected:
                listed = "、".join(str(v) for v in model_names.values())[:60]
                QMessageBox.warning(
                    self, "权重与项目不匹配",
                    f"当前权重只认识 {len(model_names)} 类：{listed}\n"
                    f"而项目「{project.name}」是 {project.num_classes} 类。\n\n"
                    "检测结果只会出现该权重认识的类别。"
                    "要看本项目的行为识别效果，请选本项目训练出来的权重。",
                )
                return

        args = [
            "--weights", weights,
            "--source", source,
            "--conf", f"{self.conf.value():.2f}",
            "--iou", f"{self.iou.value():.2f}",
            "--name", OUT_NAME,
            "--max-frames", "300",
        ]
        if self.state.project is None:
            args += ["--scheme", self.state.scheme_key]

        # 开始新一次检测前，清空上一次的预览与汇总，避免 0 结果时界面还显示旧内容
        self._clear_result()

        self.log.clear()
        self._append_log(f"权重：{weights}")
        self._append_log(f"来源：{source}")
        self.hint.setText("检测中…")
        self.btn_run.setEnabled(False)
        self.proc.start(config.scripts_dir() / "predict.py", args, config.APP_ROOT)

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_finished(self, code: int) -> None:
        self.btn_run.setEnabled(True)
        self.hint.setText("检测完成" if code == 0 else f"检测失败（退出码 {code}）")
        if code == 0:
            self.btn_export.setEnabled(True)
            self._load_result()

    def _load_result(self) -> None:
        out_dir = config.OUTPUT_DIR / "predict" / OUT_NAME
        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            return
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        counts = data.get("class_counts", {})
        self.stats.set("处理帧数", str(data.get("frames", 0)))
        self.stats.set("目标总数", str(data.get("total_objects", 0)))
        if counts:
            top = max(counts.items(), key=lambda kv: kv[1])
            total = sum(counts.values()) or 1
            self.stats.set("主要类别", f"{top[0]} {top[1] / total * 100:.0f}%")
        else:
            self.stats.set("主要类别", "—")
        self.chart.plot(list(counts.keys()), list(counts.values()))
        self.table.setVisible(True)
        self.dist_hint.setVisible(False)

        per_frame = data.get("per_frame", [])
        agg: dict[str, dict[str, int]] = {}
        for rec in per_frame:
            for kind in ("zone", "scale", "occlusion", "crowd"):
                bucket = agg.setdefault(kind, {})
                for k, v in rec.get(kind, {}).items():
                    bucket[k] = bucket.get(k, 0) + v
        heavy = agg.get("occlusion", {}).get("heavy", 0)
        self.stats.set("重度遮挡", str(heavy))

        self.table.setRowCount(0)
        labels = {"scale": "目标尺度", "occlusion": "遮挡程度", "crowd": "群体密度"}
        for kind, label in labels.items():
            if kind not in agg:
                continue
            text = "，".join(f"{k} {v}" for k, v in sorted(agg[kind].items()))
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(label))
            self.table.setItem(r, 1, QTableWidgetItem(text))

        self._show_preview(out_dir)

    def _clear_result(self) -> None:
        """开始新检测前重置界面，避免旧结果残留。"""
        self.preview.setPixmap(QPixmap())
        self.preview.setText("尚未执行检测")
        self.chart.plot([], [])
        self.table.setVisible(False)
        self.dist_hint.setVisible(True)
        for name in ("处理帧数", "目标总数", "主要类别", "重度遮挡"):
            self.stats.set(name, "—")
        self.btn_export.setEnabled(False)

    def _show_preview(self, out_dir: Path) -> None:
        img_dir = out_dir / "annotated"
        images = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
        video = out_dir / "annotated.mp4"
        if images:
            pix = QPixmap(str(images[0]))
            if not pix.isNull():
                self.preview.setPixmap(pix.scaled(
                    self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                self.preview.setToolTip(f"共 {len(images)} 帧，显示第 1 帧")
                return
        if video.exists():
            self.preview.setText(f"视频结果已保存：\n{video}")
            return
        self.preview.setPixmap(QPixmap())
        self.preview.setText("没有可预览的画面")
