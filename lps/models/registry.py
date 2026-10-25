"""模型注册表：训练页「模型」下拉框与训练脚本的模型分派都从这里读。

以后要「上线」一个新模型（例如在 v8n-CBAM 基础上再改一个变体），只需在
MODELS 列表里加一行，下拉框和训练逻辑会自动识别，无需改界面或脚本。

字段说明：
  key        传给训练脚本 --model 的唯一标识
  label      下拉框显示名
  group      下拉框分组
  kind       custom = 本项目自定义结构（走 builder.build，含 CBAM）
             official = ultralytics 官方模型，带官方预训练权重（.pt）
             official-no-pretrain = ultralytics 官方结构，但无官方预训练权重（.yaml 随机初始化）
  asset      custom 时是预训练权重文件名；official 时是 .pt 文件名；no-pretrain 时是 .yaml 文件名
  pretrained 是否有预训练权重（仅用于界面标注）
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    group: str
    kind: str
    asset: str
    pretrained: bool


def _official(version: str, prefix: str, scales: str) -> list[ModelSpec]:
    """批量生成某版本 n/s/m/l/x 的官方预训练条目。"""
    return [
        ModelSpec(f"{prefix}{s}", f"YOLO{version}{s}", f"YOLO{version}",
                  "official", f"{prefix}{s}.pt", True)
        for s in scales
    ]


MODELS: list[ModelSpec] = [
    # ---- 本项目改进版 ----
    ModelSpec("v8n-cbam", "YOLOv8n · CBAM 改进版", "本项目改进",
              "custom", "yolov8n.pt", True),

    # ---- 官方版本（带预训练权重）----
    *_official("v8", "yolov8", "nsmlx"),
    *_official("v11", "yolo11", "nsmlx"),
    *_official("v12", "yolo12", "nsmlx"),
    *_official("v10", "yolov10", "nsmlx"),
    *_official("v5", "yolov5", "nsmlx"),

    # YOLOv3 无 n/s 尺度体系，就三个变体
    ModelSpec("yolov3", "YOLOv3", "YOLOv3", "official", "yolov3.pt", True),
    ModelSpec("yolov3-tiny", "YOLOv3-tiny", "YOLOv3", "official", "yolov3-tiny.pt", True),
    ModelSpec("yolov3-spp", "YOLOv3-spp", "YOLOv3", "official", "yolov3-spp.pt", True),

    # ---- 官方结构但无预训练权重（随机初始化，从零训练）----
    # YOLOv9 尺度命名是 t/s/m/c/e，各有独立 yaml
    *[ModelSpec(f"yolov9{s}", f"YOLOv9{s}", "YOLOv9",
                "official-no-pretrain", f"yolov9{s}.yaml", False)
      for s in "tsmce"],
    # YOLOv6 在 ultralytics 里只有默认结构（无独立尺度文件、无预训练）
    ModelSpec("yolov6", "YOLOv6", "YOLOv6",
              "official-no-pretrain", "yolov6.yaml", False),
]


def find(key: str) -> ModelSpec:
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(f"未知模型标识: {key!r}")


def groups() -> list[str]:
    """按出现顺序返回分组名（去重）。"""
    return list(dict.fromkeys(m.group for m in MODELS))
