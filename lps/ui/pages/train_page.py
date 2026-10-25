"""训练页面：参数设置、后台训练、实时曲线与日志。"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from ... import config
from ...models.registry import MODELS
from ...utils.text import (
    clean_log_text, is_noise_line, parse_epoch_progress, parse_progress_line,
    strip_ansi,
)
from ..charts import MetricChart
from ..state import AppState
from ..widgets import Card, FormGrid, PageHeader, StatRow
from ..workers import MetricsWatcher, ScriptProcess

ROOT = Path(__file__).resolve().parent.parent.parent

# 指标块标题：带中文说明，避免只看英文缩写不知道含义
STAT_EPOCH = "当前轮次"
STAT_MAP50 = "mAP@0.5 平均精度"
STAT_MAP5095 = "mAP@0.5:0.95 综合精度"
STAT_PRECISION = "Precision 精确率"
STAT_RECALL = "Recall 召回率"


def _dir_size_text(root: Path, limit_files: int = 20000) -> str:
    """估算目录体积，用于在导出前告知用户包会多大。"""
    total = 0
    count = 0
    try:
        for path in Path(root).rglob("*"):
            if not path.is_file():
                continue
            total += path.stat().st_size
            count += 1
            if count >= limit_files:
                break
    except OSError:
        return "未知大小"
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.0f} {unit}" if unit == "B" else f"{total:.1f} {unit}"
        total /= 1024
    return "未知大小"


def _fmt_duration(seconds: float) -> str:
    """把秒数格式化成 12s / 5m30s / 1h05m。"""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class TrainPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.proc = ScriptProcess(self)
        self.proc.log.connect(self._append_log)
        self.proc.finished.connect(self._on_finished)
        self.proc.error.connect(lambda m: self._append_log(f"[错误] {m}"))

        self.watcher = MetricsWatcher(parent=self)
        self.watcher.updated.connect(self._on_metrics)
        self.watcher.epoch_done.connect(self._on_epoch)

        self._run_dir: Path | None = None
        self._train_t0: float = 0.0
        # 曲线数据由逐轮日志即时累积，比起等 results.csv 轮询更跟手
        self._series: dict[str, list[float]] = {}
        self._build()
        self._refresh_runs()

    # ------------------------------------------------------------ 界面
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(PageHeader(
            "模型训练",
            "支持多模型选择（YOLOv3/v5/v6/v8/v9/v10/v11/v12 及本项目 CBAM 改进版），训练日志与指标实时回填，无需等待结束。",
        ))

        self.stats = StatRow([STAT_EPOCH, STAT_MAP50, STAT_MAP5095, STAT_PRECISION, STAT_RECALL])
        root.addWidget(self.stats)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left_box = QWidget()
        left_box.setFixedWidth(392)
        left = QVBoxLayout(left_box)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        body.addWidget(left_box, 0)
        left.addWidget(self._param_card())
        left.addWidget(self._action_card())
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, 1)
        right.addWidget(self._chart_card(), 2)
        right.addWidget(self._log_card(), 3)

        # 参数控件建好后再初始化指标块，否则 self.epochs 还不存在
        self.stats.set(STAT_EPOCH, f"0 / {self.epochs.value()}")
        self.epochs.valueChanged.connect(self._on_epochs_changed)

    def _param_card(self) -> Card:
        card = Card("训练参数")
        form = FormGrid()

        # 模型选择：从注册表读取，按分组用分隔线隔开；无预训练权重的项额外标注
        self.model = QComboBox()
        self.model.setMinimumWidth(240)
        seen_groups: list[str] = []
        for spec in MODELS:
            if spec.group not in seen_groups:
                if self.model.count() > 0:
                    self.model.insertSeparator(self.model.count())
                seen_groups.append(spec.group)
            label = spec.label if spec.pretrained else f"{spec.label}（无预训练）"
            self.model.addItem(label, spec.key)
            idx = self.model.count() - 1
            self.model.setItemData(idx, f"加载方式：{spec.asset}",
                                   Qt.ItemDataRole.ToolTipRole)
        # 默认选中「YOLOv8n · CBAM 改进版」（本项目当前模型）
        default_idx = self.model.findData("v8n-cbam")
        if default_idx >= 0:
            self.model.setCurrentIndex(default_idx)
        form.add_row("模型", self.model,
                     "CBAM 改进版为本项目模型；官方模型首次使用会联网下载预训练权重，"
                     "标「无预训练」的版本从零训练")

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(config.TrainConfig.epochs)
        form.add_row("训练轮数", self.epochs)

        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(config.TrainConfig.batch)
        form.add_row("批大小", self.batch, "显存不足时降到 4 或 8")

        # 输入尺寸：默认按数据集实际分辨率自动填充，同时允许手动改
        # YOLO 的 imgsz 是单个整数：训练时图像会被 letterbox 成 imgsz×imgsz 的正方形。
        # 界面上显示成「N × N」更直观，但底层仍传一个数。
        self.imgsz = QComboBox()
        self.imgsz.setEditable(True)
        for size in (416, 512, 640, 768, 960, 1280):
            self.imgsz.addItem(f"{size} × {size}", size)
        self.imgsz.setCurrentText("640 × 640")
        form.add_row("输入尺寸", self.imgsz,
                     "推荐 640（与原始框架一致）；960 对小目标更友好，但耗时约为 640 的 2.2 倍")

        # 数据加载线程：为 0 时 CPU 串行读图，GPU 会空转等待
        self.workers = QComboBox()
        for value, label in ((0, "0（最稳，Windows 兼容性最好）"),
                             (2, "2（推荐）"), (4, "4"), (8, "8（与原始框架一致）")):
            self.workers.addItem(label, value)
        self.workers.setCurrentIndex(1)          # 默认 2
        form.add_row("数据加载线程", self.workers,
                     "调大后 GPU 不容易空转；若训练启动即报错，调回 0")

        # 运算设备：启动时探测显卡，显示真实型号而非写死的「GPU 0」
        self.device = QComboBox()
        self._fill_devices()
        form.add_row("运算设备", self.device)

        self.run_name = QLineEdit()
        self.run_name.setPlaceholderText("留空则按方案自动命名")
        form.add_row("实验名称", self.run_name)

        card.add(form)
        return card

    def _on_epochs_changed(self, value: int) -> None:
        """未开始训练时，改轮数就同步指标块的「0 / N」。"""
        if not self.proc.running:
            self.stats.set(STAT_EPOCH, f"0 / {value}")

    def _fill_devices(self) -> None:
        """探测可用运算设备并填入下拉框。

        有显卡时显示型号与显存，让用户确认确实在用 GPU；
        没有显卡时只给 CPU 并说明原因，避免误以为在跑 GPU。
        """
        self.device.clear()
        try:
            import torch

            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                for index in range(count):
                    name = torch.cuda.get_device_name(index)
                    memory = torch.cuda.get_device_properties(index).total_memory / 1024**3
                    short = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
                    self.device.addItem(f"GPU {index} · {short} · {memory:.0f}GB", str(index))
                    self.device.setItemData(index, f"{name}（{memory:.1f} GB）",
                                            Qt.ItemDataRole.ToolTipRole)
                self.device.setCurrentIndex(0)
            else:
                self.device.addItem("CPU（未检测到可用显卡）", "cpu")
        except (ImportError, RuntimeError, AssertionError):
            self.device.addItem("CPU", "cpu")

    def _suggest_imgsz(self, data_yaml: Path | None) -> None:
        """按训练集图像的实际分辨率给出输入尺寸建议。

        取若干张图的中位数长边，向上取到 32 的倍数。数据集本身分辨率远高于
        640 时用 640 会损失细节；分辨率本来就是小图时用大尺寸只是白费显存。
        """
        if not data_yaml or not Path(data_yaml).exists():
            return
        try:
            import yaml

            cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
            base = Path(str(cfg.get("path", ""))) or Path(data_yaml).parent
            train_rel = str(cfg.get("train", "train/images"))
            img_dir = Path(train_rel) if Path(train_rel).is_absolute() else base / train_rel
            if not img_dir.exists():
                return
        except (OSError, yaml.YAMLError):
            return

        import cv2
        import numpy as np

        sizes: list[int] = []
        for path in sorted(img_dir.iterdir())[:16]:
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            try:
                buf = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except OSError:
                continue
            if image is not None:
                sizes.append(max(image.shape[:2]))
        if not sizes:
            return
        median = int(np.median(sizes))
        # 推荐值固定 640：原始框架就是 640，已验证的模型也在 640 上取得 mAP50 0.9185。
        # 数据集长边通常 1280×720，按原分辨率训练耗时按边长平方增长（960 约 2.2 倍），
        # 收益却有限，所以只把数据集尺寸作为提示信息，不自动顶到 960。
        suggested = 640
        if median < 480:
            suggested = 512
        self.imgsz.setCurrentText(f"{suggested} × {suggested}")
        self.hint.setText(
            f"数据集图像长边约 {median}px，推荐 {suggested}（与原始框架一致）；"
            "小目标多可改 960，耗时约 2.2 倍"
        )

    def _action_card(self) -> Card:
        card = Card("执行")
        self.dataset_label = QLabel("项目：未选择")
        self.dataset_label.setObjectName("Hint")
        self.dataset_label.setWordWrap(True)
        card.add(self.dataset_label)
        self.state.dataset_changed.connect(self._sync_dataset)
        self.state.project_changed.connect(lambda _: self._sync_project())

        self.btn_start = QPushButton("开始训练")
        self.btn_start.setObjectName("Primary")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_export_run = QPushButton("导出训练结果")
        self.btn_export_run.setToolTip(
            "把本次训练的全部产物（权重、参数、逐轮指标、曲线、混淆矩阵）"
            "导出成一个带时间戳的文件夹，与原始框架 runs/train 的内容一致"
        )
        self.btn_export_run.setEnabled(False)
        self.btn_export_run.clicked.connect(self._export_run_folder)

        self.btn_export = QPushButton("导出交付包")
        self.btn_export.setEnabled(False)
        self.btn_export_run.setEnabled(False)
        self.btn_export.clicked.connect(self._export_model)

        # 四个按钮分两行，避免挤成一条看不清文字
        for button in (self.btn_start, self.btn_stop, self.btn_export_run, self.btn_export):
            button.setMinimumWidth(112)
        action = QHBoxLayout()
        action.setSpacing(8)
        action.addWidget(self.btn_start)
        action.addWidget(self.btn_stop)
        action.addStretch(1)
        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        export_row.addWidget(self.btn_export_run)
        export_row.addWidget(self.btn_export)
        export_row.addStretch(1)

        card.add_layout(action)
        card.add_layout(export_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        card.add(self.progress)

        self.hint = QLabel("就绪")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        card.add(self.hint)
        return card

    def _chart_card(self) -> Card:
        card = Card("指标曲线")
        self.chart = MetricChart("mAP 与损失")
        self.chart.setMinimumHeight(240)
        card.add(self.chart)
        return card

    def _log_card(self) -> Card:
        """训练记录列表 + 日志详情。

        左侧按时间倒序列出每次训练，点一条即可查看那一次的日志与曲线；
        右侧在训练过程中实时滚动当前这次。
        """
        card = Card("训练记录 / 日志")

        header = QHBoxLayout()
        self.run_count = QLabel("共 0 次训练")
        self.run_count.setObjectName("Hint")
        btn_reload = QPushButton("刷新记录")
        btn_reload.clicked.connect(self._refresh_runs)
        header.addWidget(self.run_count)
        header.addStretch(1)
        header.addWidget(btn_reload)
        card.add_layout(header)

        body = QHBoxLayout()
        self.run_list = QListWidget()
        self.run_list.setMaximumWidth(226)
        self.run_list.setMinimumWidth(196)
        self.run_list.currentItemChanged.connect(self._on_run_changed)
        body.addWidget(self.run_list)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        body.addWidget(self.log, 1)
        card.add_layout(body)
        return card

    # ------------------------------------------------------------ 训练记录
    def _refresh_runs(self) -> None:
        """扫描输出目录，列出历次训练。"""
        if self.proc.running:
            return
        # 只列「训练」产物：验证脚本会写 runs/val_xxx，那些目录没有权重，
        # 混进来会让导出按钮一直是灰的，用户以为功能坏了。
        runs = [
            p for p in config.RUNS_DIR.iterdir()
            if p.is_dir()
            and not p.name.startswith("val_")
            and ((p / "results.csv").exists() or (p / "weights" / "best.pt").exists())
        ] if config.RUNS_DIR.exists() else []
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        current = self.run_list.currentItem()
        current_path = current.data(Qt.ItemDataRole.UserRole) if current else None

        self.run_list.blockSignals(True)
        self.run_list.clear()
        for run_dir in runs:
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            item = QListWidgetItem(f"{run_dir.name}\n{mtime}")
            item.setData(Qt.ItemDataRole.UserRole, str(run_dir))
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)

        self.run_count.setText(f"共 {len(runs)} 次训练")
        if current_path:
            for row in range(self.run_list.count()):
                if self.run_list.item(row).data(Qt.ItemDataRole.UserRole) == current_path:
                    self.run_list.setCurrentRow(row)
                    break

    def _on_run_changed(self, current, _previous) -> None:
        """点击某次训练：载入它的日志与曲线。

        整体包在 try 里：这个函数由 Qt 信号触发，一旦抛异常 PyQt 会直接
        终止整个进程（实测退出码 127），表现成「点一下就闪退」。
        """
        if current is None or self.proc.running:
            return
        try:
            self._load_run(current)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # 不静默吞掉：把错误写进日志框，用户能看见也能反馈
            self.log.setPlainText(f"载入这次训练的结果失败：{type(exc).__name__}: {exc}")
            self.hint.setStyleSheet("color:#C0392B; font-weight:600;")
            self.hint.setText("载入历史记录失败，请查看上方日志")

    def _load_run(self, current) -> None:
        """真正载入某次训练产物（日志、曲线、记录）。"""
        run_dir = Path(current.data(Qt.ItemDataRole.UserRole))
        self._run_dir = run_dir

        # 选中历史记录也能直接导出，不必是刚训练完的那一次
        has_weights = (run_dir / "weights" / "best.pt").exists()
        self.btn_export.setEnabled(has_weights)
        self.btn_export_run.setEnabled(has_weights)

        log_path = run_dir / "train.log"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            # 日志里混着 tqdm 的 \r 刷新碎片，直接显示会折叠成一坨
            self.log.setPlainText(clean_log_text(text))
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        else:
            self.log.setPlainText(f"（该次训练没有日志文件：{run_dir.name}）")

        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            history: dict[str, list[float]] = {}
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            for key in self.watcher.KEYS:
                rows_key = key if key != "epoch" else "epoch"
                if rows and rows_key in rows[0]:
                    values = []
                    for row in rows:
                        try:
                            values.append(float(row[rows_key]))
                        except (TypeError, ValueError):
                            values.append(0.0)
                    history[key] = values
            self._on_metrics(history)
            self.hint.setText(f"已载入历史记录：{run_dir.name}")

    # ------------------------------------------------------------ 事件
    def _sync_dataset(self, path) -> None:
        if path:
            self.btn_start.setEnabled(True)
            if self.state.project is None:
                # 没有项目时（例如启动自动挂载了数据集）也要按数据给尺寸建议
                self._suggest_imgsz(path)
            self._sync_project()
        else:
            self.dataset_label.setText("项目：未选择（请先在「项目管理」新建并打开项目）")
            self.btn_start.setEnabled(False)

    def _sync_project(self) -> None:
        """把当前项目的名称、类别数与数据量显示出来，避免用户选错数据集。"""
        project = self.state.project
        if project:
            self.dataset_label.setText(
                f"项目：{project.name}\n"
                f"类别：{project.num_classes} 类（{project.class_summary()}）"
            )
            self._suggest_imgsz(self.state.dataset_yaml)
            self._refresh_runs()
            return
        yaml_path = self.state.dataset_yaml
        if yaml_path:
            self.dataset_label.setText(f"数据集：{Path(yaml_path).name}")
        else:
            self.dataset_label.setText("项目：未选择")

    def _append_log(self, line: str) -> None:
        """追加一行日志。

        大部分训练输出（torch 警告、pypi 升级提示）与训练进度无关，
        过滤掉才不会把真正有用的逐轮指标挤出可视区。
        """
        if not line:
            return
        info = parse_progress_line(line)
        if info is not None:
            self._on_progress_info(info)
            return
        if is_noise_line(line):
            return
        self.log.appendPlainText(line)
        if "out of memory" in line and not getattr(self, "_oom_reported", False):
            self._oom_reported = True
            self.hint.setStyleSheet("color:#C0392B; font-weight:600;")
            self.hint.setText(
                "显存不足：请把批大小降到 4，或关闭其他占用显卡的程序后重试"
            )
        self._consume_epoch(line)

    def _set_progress(self, fraction: float, text: str) -> None:
        """把 0~1 的完成度写进进度条。

        内部分辨率取 0~10000：120 轮训练时，第一轮的进度只占 0.8%，
        按 0~100 会让进度条几十秒纹丝不动，看着像卡死。
        """
        fraction = max(0.0, min(1.0, fraction))
        self.progress.setRange(0, 10000)
        self.progress.setValue(int(fraction * 10000))
        self.progress.setFormat(text)
        # 卡片底部的状态行同步显示，进度条本身很短时也能看到实时数字
        self.hint.setText(text)

    def _on_progress_info(self, info: dict) -> None:
        """消费训练脚本的批次进度，驱动进度条与剩余时间。

        对应 ultralytics 终端里那条 tqdm 进度条——界面拿不到它（它用 \\r 原地刷新），
        所以由训练脚本按固定节流自己报数。
        """
        epoch_now = int(info.get("epoch_current", 0) or 0)
        epoch_total = int(info.get("epoch_total", 0) or 0)
        iter_now = int(info.get("iter_current", 0) or 0)
        iter_total = int(info.get("iter_total", 0) or 0)
        if epoch_total <= 0 or iter_total <= 0 or epoch_now <= 0:
            return

        fraction = ((epoch_now - 1) + iter_now / iter_total) / epoch_total
        rate = float(info.get("rate", 0.0) or 0.0)
        remain = float(info.get("remain", 0.0) or 0.0)
        self._set_progress(
            fraction,
            f"{fraction * 100:.1f}%　第 {epoch_now}/{epoch_total} 轮　"
            f"第 {iter_now}/{iter_total} 批　{rate:.2f} 批/秒　"
            f"剩余 {_fmt_duration(remain)}",
        )
        self.stats.set(STAT_EPOCH, f"{epoch_now} / {epoch_total}")

    def _consume_epoch(self, line: str) -> None:
        """从日志行里读轮次，补上批次进度来不及覆盖的边界。"""
        parsed = parse_epoch_progress(line)
        if parsed is None:
            return
        current, total = parsed
        self.stats.set(STAT_EPOCH, f"{current} / {total}")
        self._push_metrics_from_line(line)

        if self._train_t0 and current > 0:
            elapsed = time.time() - self._train_t0
            per_epoch = elapsed / current
            remain = per_epoch * max(0, total - current)
            self._set_progress(
                current / total,
                f"{current / total * 100:.1f}%　第 {current}/{total} 轮　"
                f"每轮约 {_fmt_duration(per_epoch)}　剩余 {_fmt_duration(remain)}",
            )

    def _start(self) -> None:
        if self.proc.running:
            return
        data_yaml = self.state.dataset_yaml
        if not data_yaml:
            QMessageBox.information(self, "提示",
                                    "请先在「项目管理」新建项目并导入数据集")
            return

        name = self.run_name.text().strip() or self._default_run_name()
        self._run_dir = config.RUNS_DIR / name
        if self._run_dir.exists():
            QMessageBox.information(self, "提示", f"实验名 {name} 已存在，请换一个名称")
            return

        # 有项目时以项目的数据集为准，类别数由数据决定，不再传 --scheme
        args = ["--data", str(data_yaml)]
        if self.state.project is None:
            args += ["--scheme", self.state.scheme_key]
        args += [
            "--epochs", str(self.epochs.value()),
            "--batch", str(self.batch.value()),
            "--imgsz", str(self._imgsz_value()),
            "--device", self.device.currentData(),
            "--workers", str(self.workers.currentData()),
            "--model", str(self.model.currentData()),
            "--name", name,
            "--skip-check",
        ]

        # 训练前检查空闲显存：0 号卡常被其他程序占用，直接开训会在
        # 第一次参数更新时报 CUDA out of memory，报错信息还很难懂。
        if not self._check_vram(self._imgsz_value(), self.batch.value()):
            return

        self.log.clear()
        project = self.state.project
        if project:
            self._append_log(f"项目：{project.name}（{project.num_classes} 类）")
        self._append_log(f"启动训练：{name}")
        self._append_log(f"模型：{self.model.currentText()}")
        self._append_log(f"参数：epochs={self.epochs.value()} batch={self.batch.value()} "
                         f"imgsz={self._imgsz_value()} device={self.device.currentData()} "
                         f"workers={self.workers.currentData()}")
        self._series = {}
        self._oom_reported = False
        self.chart.update_series({})
        self._train_t0 = time.time()
        total_epochs = max(1, self.epochs.value())
        self._set_progress(0.0, "初始化中：正在扫描数据、构建模型…")
        # 清掉上一次实验残留的指标，避免把旧数字误认为当前进度
        for label in (STAT_MAP50, STAT_MAP5095, STAT_PRECISION, STAT_RECALL):
            self.stats.set(label, "—")
        self.stats.set(STAT_EPOCH, f"0 / {total_epochs}")
        self.hint.setStyleSheet("")
        self.hint.setText("训练中…")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_export_run.setEnabled(False)

        self.proc.start(config.scripts_dir() / "train.py", args, config.APP_ROOT)
        QTimer.singleShot(4000, lambda: self.watcher.watch(self._run_dir))

    def _check_vram(self, imgsz: int, batch: int) -> bool:
        """估算显存需求并提示；不足且用户选择取消时返回 False。

        注意不能在此抛异常：本函数由按钮的槽函数调用，
        槽里抛出异常会被 PyQt 直接 abort 进程（之前踩过）。
        """
        from ...services.gpu import free_memory_mb

        if self.device.currentData() == "cpu":
            return True
        free = free_memory_mb(0)
        if free < 0:
            return True
        # 640 尺寸实测约 2.6GB（batch 10 + AMP）；按边长平方与批大小线性放大
        need = 2.6 * (imgsz / 640) ** 2 * (batch / 10)
        if free >= need * 1.15:
            return True
        reply = QMessageBox.question(
            self, "显存可能不足",
            f"当前显卡空闲显存约 {free:.0f} MB，本次配置估计需要 {need * 1024:.0f} MB。\n\n"
            "常见原因：还有另一个训练在跑，或浏览器/游戏占用了显存。\n"
            "建议先把「批大小」降到 4，或关掉占显存的程序后重试。\n\n"
            "仍要继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _imgsz_value(self) -> int:
        """读取输入尺寸。

        显示形如「960 × 960」，可手动改成任意数字；这里只取第一个数字，
        否则把整串数字拼起来会得到 960960 这种非法值。
        """
        import re

        match = re.search(r"\d+", self.imgsz.currentText())
        value = int(match.group()) if match else 640
        value = max(64, min(2048, value))
        return int(round(value / 32)) * 32

    def _default_run_name(self) -> str:
        """默认实验名必须是纯 ASCII。

        中文名经子进程传给 ultralytics 时会被截断（实测「山羊_run」→「_run」），
        输出目录与界面监听的目录就对不上了。项目名保留在界面显示，
        目录名统一用项目 id。
        """
        project = self.state.project
        if project:
            # 带上完成时间，才能在权重下拉里分清「第几次训练」；保持纯 ASCII
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            return f"proj_{project.pid}_{stamp}"
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}"

    def _export_model(self) -> None:
        """把本次训练的最优权重连同模型卡片导出到用户选定目录。"""
        from ...services import export_model

        if not self._run_dir:
            QMessageBox.information(self, "提示", "还没有可导出的训练结果")
            return
        best = self._run_dir / "weights" / "best.pt"
        if not best.exists():
            QMessageBox.information(self, "提示", "未找到 best.pt，训练可能未完成")
            return
        include_dataset = False
        project = self.state.project
        if project:
            size_text = _dir_size_text(Path(project.dataset_root))
            reply = QMessageBox.question(
                self, "是否附带训练数据",
                f"是否把训练用的图像与标签一起打包？\n\n"
                f"· 是：包内会多一个 dataset/ 目录（约 {size_text}），"
                "对方不用找原始数据集就能直接重训\n"
                "· 否：只带权重与脚本，数据需对方自备",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            include_dataset = reply == QMessageBox.StandardButton.Yes

        default = str(Path.home() / "Desktop")
        target = QFileDialog.getExistingDirectory(
            self, "选择交付包要保存到哪个文件夹", default,
        )
        if not target:
            return
        try:
            result = export_model(best, Path(target), project=self.state.project,
                                  include_dataset=include_dataset)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.state.notify(f"交付包已导出到 {target}")
        pkg = Path(target)
        QMessageBox.information(
            self, "导出完成",
            f"已生成可脱离本软件使用的交付包：\n{pkg}\n\n"
            "包内包含权重、训练/验证/推理脚本、源码、依赖清单与使用说明。\n"
            "对方电脑装好 Python 与依赖后，即可用命令行训练与推理。",
        )

    def _export_run_folder(self) -> None:
        """导出训练结果文件夹（与 runs/train 同构）。"""
        from ...services import export_run_folder

        if not self._run_dir or not self._run_dir.is_dir():
            QMessageBox.information(self, "提示", "还没有可导出的训练结果")
            return

        default = str(Path.home() / "Desktop")
        target = QFileDialog.getExistingDirectory(
            self, "选择新文件夹 train_日期_时间 要保存到哪个位置", default,
        )
        if not target:
            return
        try:
            path = export_run_folder(self._run_dir, Path(target))
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return

        files = list(path.rglob("*"))
        total = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
        self.state.notify(f"训练结果已导出到 {path.name}")
        QMessageBox.information(
            self, "导出完成",
            f"已生成文件夹：\n{path}\n\n"
            f"包含 weights（best/last/分轮存档）、args.yaml（本次参数）、"
            f"results.csv（逐轮指标）、results.png、混淆矩阵、PR/F1/P/R 曲线、"
            f"训练与验证批次图、train.log。\n\n"
            f"共 {len([f for f in files if f.is_file()])} 个文件，约 {total:.1f} MB。",
        )

    def _stop(self) -> None:
        if self.proc.running:
            self.hint.setText("正在停止…")
            self.proc.stop()

    def _push_metrics_from_line(self, line: str) -> None:
        """从逐轮日志里取指标，立刻重画曲线。

        日志形如：
        ``epoch 3/20  已用 1m  mAP50=0.8930  mAP50-95=0.6965  P=0.8581  R=0.8446``
        这样每轮验证一结束曲线就动，不用等 results.csv 被轮询到。
        """
        import re

        pairs = dict(re.findall(r"([A-Za-z0-9@.:\-]+)=([\d.]+)", line))
        mapping = {
            "mAP@0.5": pairs.get("mAP50"),
            "mAP@0.5:0.95": pairs.get("mAP50-95"),
            "Precision": pairs.get("P"),
            "Recall": pairs.get("R"),
        }
        if not any(v is not None for v in mapping.values()):
            return
        for label, value in mapping.items():
            if value is None:
                continue
            self._series.setdefault(label, []).append(float(value))
        self.chart.update_series(self._series)

    def _on_metrics(self, history: dict) -> None:
        series = {}
        for key, label in (
            ("metrics/mAP50(B)", "mAP@0.5"),
            ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
        ):
            if key in history and history[key]:
                series[label] = history[key]
        if series:
            self._series = {k: list(v) for k, v in series.items()}
        self.chart.update_series(self._series or series)

        epochs = history.get("epoch")
        if epochs:
            total = max(self.epochs.value(), 1)
            done = min(total, len(epochs))
            # 只在批次进度尚未推进到该位置时补一下，避免把实时文字覆盖掉
            label = (f"{done / total * 100:.1f}%　已完成 {done}/{total} 轮"
                     if self.proc.running
                     else f"{done / total * 100:.1f}%　历史记录共 {done} 轮")
            if self.progress.value() < int(done / total * 10000):
                self._set_progress(done / total, label)

    def _on_epoch(self, epoch: int, snapshot: dict) -> None:
        # results.csv 里的 epoch 已经是 1 起，再加 1 会出现「4 / 3」
        self.stats.set(STAT_EPOCH, f"{epoch} / {self.epochs.value()}")
        for key, label in (
            ("metrics/mAP50(B)", STAT_MAP50),
            ("metrics/mAP50-95(B)", STAT_MAP5095),
            ("metrics/precision(B)", STAT_PRECISION),
            ("metrics/recall(B)", STAT_RECALL),
        ):
            if key in snapshot:
                self.stats.set(label, f"{snapshot[key]:.4f}")

    def _on_finished(self, code: int) -> None:
        self.watcher.stop()
        self._refresh_runs()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self.progress.setValue(self.progress.maximum())
            self.hint.setText("训练完成")
            if self.watcher.run_dir:
                self._run_dir = self.watcher.run_dir
            best = self._run_dir / "weights" / "best.pt" if self._run_dir else None
            if best and best.exists():
                self.state.weights = best
                # 训练成功即解锁交付包导出，并把这次实验登记到项目里，
                # 否则「训练记录」里翻不到，导出按钮也一直是灰的。
                self.btn_export.setEnabled(True)
                self.btn_export_run.setEnabled(True)
                project = self.state.project
                if project and self._run_dir:
                    self.state.store.register_run(project, self._run_dir.name, best)
                    self.state.notify(
                        f"训练完成，已登记到项目「{project.name}」：{self._run_dir.name}"
                    )
                else:
                    self.state.notify(f"训练完成，最优权重：{best.name}")
            else:
                self.state.notify("训练进程已结束，但没有找到 best.pt，请查看日志")
            self._append_log("========== 训练完成 ==========")
        else:
            self.hint.setStyleSheet("color:#C0392B; font-weight:600;")
            self.hint.setText(f"训练异常结束（退出码 {code}），请查看上方日志")
            self._append_log(f"进程退出，退出码 {code}")
