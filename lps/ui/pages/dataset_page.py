"""数据集页面：导入、校验、类别方案、分布统计。"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ... import config
from ...data import DatasetChecker, DatasetConverter, DatasetStats
from ...data import import_tools as imp
from ..charts import BarChart
from ..state import AppState
from ..widgets import Card, PageHeader, StatRow


class DatasetPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._layout_info: imp.DatasetLayout | None = None
        self._build()

    # ------------------------------------------------------------ 界面
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(PageHeader(
            "数据集管理",
            "导入 YOLO 格式数据集后自动校验完整性与类别分布，并生成可按类别方案切换的训练视图。",
        ))

        self.stats = StatRow(["图像总数", "目标总数", "训练/验证/测试", "类别数"])
        root.addWidget(self.stats)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        body.addLayout(left, 1)

        left.addWidget(self._import_card())
        left.addWidget(self._scheme_card())
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, 1)

        right.addWidget(self._detail_card(), 1)

    def _import_card(self) -> Card:
        card = Card("数据来源")
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择包含 train/valid/test 的目录，或 YOLO 压缩包")
        btn_dir = QPushButton("选择目录")
        btn_dir.clicked.connect(self._pick_dir)
        btn_zip = QPushButton("选择压缩包")
        btn_zip.clicked.connect(self._pick_zip)
        row.addWidget(self.path_edit, 1)
        row.addWidget(btn_dir)
        row.addWidget(btn_zip)
        card.add_layout(row)

        actions = QHBoxLayout()
        self.btn_check = QPushButton("校验并统计")
        self.btn_check.setObjectName("Primary")
        self.btn_check.clicked.connect(self._run_check)
        self.status_label = QLabel("尚未导入数据集")
        self.status_label.setObjectName("Hint")
        actions.addWidget(self.btn_check)
        actions.addWidget(self.status_label, 1)
        card.add_layout(actions)
        return card

    def _scheme_card(self) -> Card:
        card = Card("类别方案")
        hint = QLabel("切换方案会用同一批图像重新生成标签视图，不修改原始数据。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        card.add(hint)

        row = QHBoxLayout()
        self.scheme_box = QComboBox()
        for key, scheme in config.CLASS_SCHEMES.items():
            self.scheme_box.addItem(f"{key}（{scheme.num_classes} 类）", key)
        idx = self.scheme_box.findData(self.state.scheme_key)
        if idx >= 0:
            self.scheme_box.setCurrentIndex(idx)
        self.scheme_box.currentIndexChanged.connect(self._on_scheme_preview)

        self.btn_apply = QPushButton("生成训练视图")
        self.btn_apply.clicked.connect(self._apply_scheme)
        row.addWidget(self.scheme_box, 1)
        row.addWidget(self.btn_apply)
        card.add_layout(row)

        self.scheme_desc = QLabel("")
        self.scheme_desc.setObjectName("Hint")
        self.scheme_desc.setWordWrap(True)
        card.add(self.scheme_desc)
        self._on_scheme_preview()
        return card

    def _detail_card(self) -> Card:
        card = Card("划分明细")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["划分", "图像", "标签", "目标数", "每图均值"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        card.add(self.table)

        self.chart = BarChart()
        self.chart.setMinimumHeight(180)
        card.add(self.chart)
        return card

    # ------------------------------------------------------------ 交互
    def _pick_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.path_edit.setText(path)
            self._run_check()

    def _pick_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择压缩包", "", "压缩包 (*.zip)")
        if path:
            self.path_edit.setText(path)
            self._run_check()

    def _prepare_root(self, raw: str) -> Path | None:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".zip":
            target = config.OUTPUT_DIR / "datasets" / "imported" / p.stem
            self.state.notify("正在解压压缩包…")
            return imp.extract_archive(p, target)
        if p.is_dir():
            return p
        return None

    def _run_check(self) -> None:
        raw = self.path_edit.text().strip()
        if not raw:
            QMessageBox.information(self, "提示", "请先选择数据集目录或压缩包")
            return
        root = self._prepare_root(raw)
        if root is None:
            QMessageBox.warning(self, "路径无效", "所选路径不存在")
            return

        layout = imp.inspect(root)
        if not layout.splits:
            QMessageBox.warning(
                self, "结构不符",
                "没有找到 images 目录。\n期望结构：\n  train/images\n  valid/images\n  test/images",
            )
            return
        self._layout_info = layout

        yaml_path = layout.yaml_path
        if yaml_path is None or "path" not in (yaml_path.read_text(encoding="utf-8") or ""):
            yaml_path = imp.ensure_yaml(layout, config.USER_CONFIG_DIR / "data_imported.yaml")
        self.state.dataset_yaml = yaml_path
        self.state.source_root = layout.root

        self._fill_table(layout)
        self._fill_chart(layout)
        self._fill_stats(layout, yaml_path)

    def _fill_stats(self, layout: imp.DatasetLayout, yaml_path: Path) -> None:
        boxes = imp.count_boxes(layout)
        total_boxes = sum(boxes.values())
        per_split = "/".join(str(boxes.get(k, 0)) for k in ("train", "val", "test"))
        self.stats.set("图像总数", str(layout.total_images))
        self.stats.set("目标总数", str(total_boxes))
        self.stats.set("训练/验证/测试", per_split)
        self.stats.set("类别数", str(len(layout.classes)))

        report = DatasetChecker(yaml_path).run()
        if report.passed:
            self.status_label.setText(f"校验通过（{len(report.warnings)} 条提示）")
            self.status_label.setStyleSheet("color: #16A34A;")
        else:
            self.status_label.setText(f"发现 {len(report.errors)} 处错误")
            self.status_label.setStyleSheet("color: #DC2626;")
        self.state.notify(f"数据集已载入：{layout.root.name}")

    def _fill_table(self, layout: imp.DatasetLayout) -> None:
        boxes = imp.count_boxes(layout)
        self.table.setRowCount(0)
        for split, img_dir in layout.splits.items():
            images = sum(1 for p in img_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            lab_dir = img_dir.parent / "labels"
            labels = len(list(lab_dir.glob("*.txt"))) if lab_dir.exists() else 0
            n_boxes = boxes.get(split, 0)
            mean = f"{n_boxes / images:.2f}" if images else "0"
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, value in enumerate([split, images, labels, n_boxes, mean]):
                item = QTableWidgetItem(str(value))
                if col:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)

    def _fill_chart(self, layout: imp.DatasetLayout) -> None:
        dist = imp.class_distribution(layout)
        names = [layout.classes.get(cid, str(cid)) for cid in sorted(dist)]
        values = [dist[cid] for cid in sorted(dist)]
        self.chart.plot(names, values)

    def _on_scheme_preview(self) -> None:
        key = self.scheme_box.currentData()
        scheme = config.CLASS_SCHEMES.get(key)
        if not scheme:
            return
        mapping = "、".join(f"{src}→{scheme.zh_names.get(dst, dst)}"
                            for src, dst in scheme.source_map.items())
        self.scheme_desc.setText(f"映射关系：{mapping}")

    def _apply_scheme(self) -> None:
        key = self.scheme_box.currentData()
        source = self.state.source_root
        if not source:
            QMessageBox.information(self, "提示", "请先导入并校验数据集")
            return
        scheme = config.CLASS_SCHEMES[key]
        try:
            if self._needs_remap(source):
                conv = DatasetConverter(source)
                result = conv.convert(scheme)
                yaml_path = result.yaml_path
            else:
                # 类别定义本就一致，直接复用原数据，避免多存一份
                yaml_path = config.USER_CONFIG_DIR / f"data_{key}.yaml"
                yaml_path.write_text(
                    f"path: {str(source).replace(chr(92), '/')}\n"
                    f"train: train/images\nval: valid/images\n"
                    f"names: {scheme.names}\n",
                    encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "生成失败", str(exc))
            return

        self.state.set_scheme(key)
        self.state.dataset_yaml = yaml_path
        self.state.notify(f"已生成 {key} 训练视图：{yaml_path.name}")
        QMessageBox.information(
            self, "完成",
            f"类别方案 {key} 已生效。\n配置文件：\n{yaml_path}",
        )

    def _needs_remap(self, source: Path) -> bool:
        """原始类名与当前方案是否一致，一致则无需派生。"""
        if not self._layout_info:
            return False
        names = {v.lower() for v in self._layout_info.classes.values()}
        scheme = config.CLASS_SCHEMES[self.scheme_box.currentData()]
        return names != {v.lower() for v in scheme.names.values()}
