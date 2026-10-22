"""检测引擎封装。

把 ultralytics 的推理结果转换成项目内部的数据结构，并顺带算出
每个目标的派生属性（区域、尺度、遮挡、密度），供界面统计与报告导出使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data import attributes as attr


@dataclass
class Detection:
    cls_id: int
    name: str
    zh: str
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2（像素）
    zone: int = 0
    scale: str = ""
    occlusion: str = "none"
    crowd: str = ""

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class FrameResult:
    detections: List[Detection] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)

    def counts(self) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for d in self.detections:
            result[d.zh] = result.get(d.zh, 0) + 1
        return result

    def attribute_summary(self, attr_name: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for d in self.detections:
            value = str(getattr(d, attr_name, ""))
            result[value] = result.get(value, 0) + 1
        return result


def read_model_names(weights: Path | str) -> dict:
    """读取权重文件里记录的类别表（不推理）。

    用于在评估/推理前校验「模型类别数」与「数据集类别数」是否一致——
    不一致时 ultralytics 会在混淆矩阵里抛 IndexError，报错信息极难懂。
    """
    from ultralytics import YOLO

    from ..models.attention import inject_modules

    inject_modules()
    model = YOLO(str(weights))
    return dict(getattr(model, "names", {}) or {})


class Detector:
    """加载一次模型，可重复推理多张图。"""

    def __init__(self, weights: Path | str, scheme=None, device: str = "0",
                 conf: float = 0.25, iou: float = 0.5, imgsz: int = 640):
        from ultralytics import YOLO

        from ..models.attention import inject_modules

        inject_modules()
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        # 中文对照只在「预设方案类数 == 模型类数」时启用。
        # 否则换成自定义数据集后编号含义不同，会把 goat 显示成「采食」。
        model_names = getattr(self.model, "names", {}) or {}
        if scheme is not None and len(scheme.names) != len(model_names):
            scheme = None
        self.scheme = scheme

    def _name_of(self, cls_id: int) -> tuple[str, str]:
        names = getattr(self.model, "names", {}) or {}
        en = str(names.get(cls_id, cls_id))
        zh = self.scheme.zh(cls_id) if self.scheme else en
        return en, zh

    def predict(self, source) -> tuple[FrameResult, np.ndarray]:
        """对单张图片推理，返回结构与标注后的图像。

        agnostic_nms=True：跨类别合并重叠框。默认的分类别 NMS 只在同类之间
        去重，同一只羊会同时留下「rest」和「eat」两个叠着的框；开启后不同
        类别的框也参与抑制，一个目标只保留置信度最高的那个框。
        """
        results = self.model.predict(
            source=source, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False, agnostic_nms=True,
        )
        if not results:
            return FrameResult(), np.zeros((1, 1, 3), np.uint8)

        r = results[0]
        h, w = r.orig_shape
        frame = FrameResult(image_shape=(h, w))
        annotated = r.plot(line_width=2)

        raw = []
        for box in r.boxes:
            cls_id = int(box.cls.item())
            en, zh = self._name_of(cls_id)
            xyxy = box.xyxy[0].tolist()
            raw.append(Detection(
                cls_id=cls_id, name=en, zh=zh,
                conf=float(box.conf.item()),
                box=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            ))

        # 派生属性：归一化坐标交给共用逻辑，避免两处判定标准不一致
        norm_boxes = [
            (d.center[0] / w, d.center[1] / h, d.width / w, d.height / h)
            for d in raw
        ]
        crowd = attr.crowd_of(len(raw))
        for i, (d, nb) in enumerate(zip(raw, norm_boxes)):
            others = [b for j, b in enumerate(norm_boxes) if j != i]
            overlap = max((attr._overlap_ratio(nb, o) for o in others), default=0.0)  # noqa: SLF001
            d.zone = attr.zone_of(nb[0], nb[1])
            d.scale = attr.scale_of(nb[2] * nb[3])
            d.occlusion = attr.occlusion_of(overlap)
            d.crowd = crowd
        frame.detections = raw
        return frame, annotated
