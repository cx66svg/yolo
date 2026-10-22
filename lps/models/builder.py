"""模型构建。

负责三件事：
  1. 把类别数写进结构文件（yaml 里的 nc 若与实际数据不符，Detect 头会重建）
  2. 注入自定义注意力模块
  3. 从 COCO 预训练权重做**部分加载**：只加载形状匹配的层，
     新增的注意力层保持随机初始化，这是迁移学习的正确姿势
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import config
from .attention import inject_modules

DEFAULT_CFG = Path(__file__).parent / "yolov8n-posture.yaml"


@dataclass
class BuildReport:
    cfg_path: Path
    weights: Optional[Path]
    loaded_layers: int
    total_layers: int

    def summary(self) -> str:
        if self.weights is None:
            return f"结构 {self.cfg_path.name}：随机初始化，共 {self.total_layers} 层"
        return (
            f"结构 {self.cfg_path.name}：从 {self.weights.name} 加载 "
            f"{self.loaded_layers}/{self.total_layers} 层参数，"
            f"其余 {self.total_layers - self.loaded_layers} 层（注意力）随机初始化"
        )


def write_cfg(cfg_path: Path, nc: int, out_dir: Optional[Path] = None) -> Path:
    """把 nc 写进结构文件，生成一份临时副本。

    直接改原 yaml 会污染版本库，每次构建都从模板派生更安全。
    """
    text = Path(cfg_path).read_text(encoding="utf-8")
    text = re.sub(r"^nc:\s*\d+", f"nc: {nc}", text, count=1, flags=re.MULTILINE)
    target_dir = Path(out_dir or (config.OUTPUT_DIR / "models"))
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{Path(cfg_path).stem}_nc{nc}.yaml"
    target.write_text(text, encoding="utf-8")
    return target


def build(nc: int, cfg_path: Path | str = DEFAULT_CFG,
          pretrained: Optional[Path | str] = None, verbose: bool = True):
    """构建模型，返回 (YOLO 对象, BuildReport)。"""
    from ultralytics import YOLO

    inject_modules()
    cfg = write_cfg(Path(cfg_path), nc)

    model = YOLO(str(cfg))

    weights = None
    loaded = 0
    total = len(model.model.state_dict())
    if pretrained:
        weights = Path(pretrained)
        if weights.exists():
            loaded = _count_matched_keys(model.model, weights)
            model.load(str(weights))          # ultralytics 内部按名称+形状取交集
        else:
            weights = None

    report = BuildReport(cfg_path=cfg, weights=weights,
                         loaded_layers=loaded, total_layers=total)
    if verbose:
        print(report.summary())
        print(f"注意力模块数量: {_count_attention_modules(model.model)}")
    return model, report


def _count_matched_keys(model, weights: Path) -> int:
    """统计预训练权重里能与当前结构对上的参数张量个数。

    不能直接用 state_dict 的总数，因为新增的注意力层在预训练权重里
    根本不存在，Detect 头的类别数也变了（80 类 -> 4 类）。
    """
    import torch

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)
    try:
        src_sd = src.state_dict()
    except AttributeError:
        return 0
    dst_sd = model.state_dict()
    return sum(
        1 for k, v in src_sd.items()
        if k in dst_sd and hasattr(v, "shape") and dst_sd[k].shape == v.shape
    )


def _count_attention_modules(model) -> int:
    from .attention import CBAM, CoordAtt, ECA

    kinds = (CBAM, CoordAtt, ECA)
    return sum(1 for m in model.modules() if isinstance(m, kinds))


def param_summary(model) -> str:
    """参数量与计算量摘要，用于对比原始结构与改进结构。"""
    try:
        from ultralytics.utils.torch_utils import get_flops

        n_params = sum(p.numel() for p in model.model.parameters())
        flops = get_flops(model.model, 640)
        return f"参数量 {n_params/1e6:.3f} M，计算量 {flops:.1f} GFLOPs"
    except Exception as exc:  # pragma: no cover
        return f"参数量统计失败: {exc}"
