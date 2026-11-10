"""训练详情对话框：展示某次训练的模型信息与关键图表。"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

# 按展示顺序排列的关键图表（训练目录里可能不全有，按需展示）
_CHART_FILES = [
    ("results.png", "训练指标综合曲线"),
    ("confusion_matrix_normalized.png", "混淆矩阵（归一化）"),
    ("confusion_matrix.png", "混淆矩阵"),
    ("BoxPR_curve.png", "PR 曲线"),
    ("BoxF1_curve.png", "F1-置信度曲线"),
    ("BoxP_curve.png", "精确率-置信度曲线"),
    ("BoxR_curve.png", "召回率-置信度曲线"),
    ("val_batch0_pred.jpg", "验证集预测（批次 0）"),
    ("val_batch1_pred.jpg", "验证集预测（批次 1）"),
    ("val_batch2_pred.jpg", "验证集预测（批次 2）"),
]


class TrainDetailDialog(QDialog):
    """双击训练记录弹出的详情页：模型信息 + 关键图表。"""

    def __init__(self, run_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"训练详情 · {run_dir.name}")
        self.resize(720, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # 模型信息栏
        info_label = QLabel(self._build_info(run_dir))
        info_label.setWordWrap(True)
        info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        info_label.setStyleSheet(
            "background:#F4F6FB; border:1px solid #DDE3EE; border-radius:8px; "
            "padding:12px; font-size:12px; line-height:1.6;")
        root.addWidget(info_label)

        # 图表滚动区
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;}")
        container = QWidget()
        col = QVBoxLayout(container)
        col.setSpacing(10)

        shown = 0
        for filename, caption in _CHART_FILES:
            path = run_dir / filename
            if not path.exists():
                continue
            pix = QPixmap(str(path))
            if pix.isNull():
                continue
            caption_label = QLabel(caption)
            caption_label.setStyleSheet("font-weight:600; color:#2C3E50;")
            col.addWidget(caption_label)
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            img_label.setPixmap(
                pix.scaledToWidth(680, Qt.TransformationMode.SmoothTransformation))
            col.addWidget(img_label)
            shown += 1

        if shown == 0:
            col.addWidget(QLabel("（本次训练没有生成图表文件）"))
        col.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

    def _build_info(self, run_dir: Path) -> str:
        """构造模型信息文本：优先 model_info.json（新训练），回退 args.yaml（旧训练）。"""
        lines: list[str] = []
        mj = run_dir / "model_info.json"
        info: dict = {}
        if mj.exists():
            try:
                info = json.loads(mj.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                info = {}

        if info:
            lines.append("【模型信息】")
            lines.append(f"模型：{info.get('model_label') or info.get('model_key') or '未知'}")
            lines.append(f"类别数：{info.get('classes', '—')}")
            lines.append(f"训练轮数：{info.get('epochs', '—')}")
            lines.append(f"批大小：{info.get('batch', '—')}")
            if info.get("imgsz"):
                lines.append(f"输入尺寸：{info['imgsz']} × {info['imgsz']}")
            if info.get("data"):
                lines.append(f"数据集：{Path(str(info['data'])).name}")
        else:
            ay = run_dir / "args.yaml"
            if ay.exists():
                try:
                    import yaml

                    with ay.open("r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                    lines.append("【模型信息】（来自 args.yaml，无模型名记录）")
                    model_raw = str(cfg.get("model", "—"))
                    lines.append(f"模型：{Path(model_raw).name}")
                    lines.append(f"训练轮数：{cfg.get('epochs', '—')}")
                    lines.append(f"批大小：{cfg.get('batch', '—')}")
                    if cfg.get("imgsz"):
                        lines.append(f"输入尺寸：{cfg['imgsz']} × {cfg['imgsz']}")
                except (OSError, ImportError, ValueError):
                    lines.append("【模型信息】无法读取 args.yaml")
            else:
                lines.append("【模型信息】本次训练没有记录模型信息（旧版本训练）")
        return "\n".join(lines)
