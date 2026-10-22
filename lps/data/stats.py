"""数据集统计。

输出三个层面的事实：
  1. 规模：各划分的图像数、目标数、每图平均目标数
  2. 类别：每类框数与占比、类别不平衡比（最大类 / 最小类）
  3. 几何：框面积与长宽比分布，判断是否存在大量小目标

统计结果用于训练前判断是否需要做类别加权或调整输入尺寸。
"""
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .checker import IMAGE_EXTENSIONS


@dataclass
class SplitStats:
    name: str
    images: int = 0
    objects: int = 0
    empty_labels: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)
    areas: List[float] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)
    objects_per_image: List[int] = field(default_factory=list)

    @property
    def mean_objects(self) -> float:
        return self.objects / self.images if self.images else 0.0


@dataclass
class DatasetStats:
    splits: Dict[str, SplitStats] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------ 计算
    @classmethod
    def compute(cls, data_root: Path, class_names: Optional[Dict[int, str]] = None) -> "DatasetStats":
        """统计一个 YOLO 数据集。

        类名优先级：显式传入 > 数据集自带 data.yaml > 空。
        同一批标签在不同类别方案下的编号可能不同，因此统计原始数据时
        必须用原始 data.yaml 的类名，否则会出现类别对调。
        """
        if class_names is None:
            class_names = _read_names(Path(data_root))
        stats = cls(class_names=class_names)
        for dir_name, split_name in (("train", "train"), ("valid", "val"), ("test", "test")):
            base = Path(data_root) / dir_name
            if not base.exists():
                continue
            stats.splits[split_name] = _scan_split(split_name, base)
        return stats

    # ------------------------------------------------------------ 输出
    def imbalance_ratio(self, split: str = "train") -> float:
        s = self.splits.get(split)
        if not s or len(s.class_counts) < 2:
            return 1.0
        values = [v for v in s.class_counts.values() if v > 0]
        return max(values) / min(values) if min(values) else float("inf")

    def class_weights(self, split: str = "train", power: float = 1.0) -> Dict[int, float]:
        """按逆频率给出类别权重，用于加权损失或采样。power 控制强度。"""
        s = self.splits.get(split)
        if not s or not s.class_counts:
            return {}
        total = sum(s.class_counts.values())
        n_cls = max(1, len(s.class_counts))
        weights = {}
        for cid, cnt in s.class_counts.items():
            freq = cnt / total
            weights[cid] = round((1.0 / max(freq, 1e-9) / n_cls) ** power, 4)
        return weights

    def summary(self) -> str:
        lines = []
        for name, s in self.splits.items():
            head = (f"[{name}] 图像 {s.images}  目标 {s.objects}  "
                    f"每图均值 {s.mean_objects:.2f}  空标签 {s.empty_labels}")
            lines.append(head)
            if s.class_counts:
                total = sum(s.class_counts.values())
                detail = ", ".join(
                    f"{self.class_names.get(cid, cid)}={cnt}({cnt/total*100:.1f}%)"
                    for cid, cnt in sorted(s.class_counts.items())
                )
                lines.append(f"    类别: {detail}")
            if s.areas:
                lines.append(
                    f"    框面积占比 中位数 {statistics.median(s.areas):.4f}  "
                    f"均值 {statistics.fmean(s.areas):.4f}  "
                    f"最大 {max(s.areas):.4f}"
                )
            if s.aspect_ratios:
                lines.append(
                    f"    长宽比 中位数 {statistics.median(s.aspect_ratios):.2f}  "
                    f"范围 {min(s.aspect_ratios):.2f}~{max(s.aspect_ratios):.2f}"
                )
        lines.append(f"训练集类别不平衡比（最大/最小）: {self.imbalance_ratio():.2f}")
        return "\n".join(lines)

    def export_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "class_id", "class_name", "boxes", "share_percent"])
            for name, s in self.splits.items():
                total = sum(s.class_counts.values()) or 1
                for cid, cnt in sorted(s.class_counts.items()):
                    writer.writerow([
                        name, cid, self.class_names.get(cid, cid),
                        cnt, round(cnt / total * 100, 2),
                    ])
        return path


def _read_names(data_root: Path) -> Dict[int, str]:
    """从数据集根目录的 data.yaml 读取类名。"""
    import yaml

    for candidate in (data_root / "data.yaml", data_root.parent / "data.yaml"):
        if candidate.exists():
            try:
                cfg = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            names = cfg.get("names") or {}
            if isinstance(names, list):
                names = dict(enumerate(names))
            return {int(k): str(v) for k, v in names.items()}
    return {}


def _scan_split(name: str, base: Path) -> SplitStats:
    st = SplitStats(name=name)
    img_dir = base / "images"
    lab_dir = base / "labels"
    if img_dir.exists():
        st.images = sum(1 for p in img_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not lab_dir.exists():
        return st

    for txt in lab_dir.glob("*.txt"):
        rows = [ln.split() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        valid = [r for r in rows if len(r) == 5]
        st.objects_per_image.append(len(valid))
        if not valid:
            st.empty_labels += 1
            continue
        for r in valid:
            try:
                cid = int(float(r[0]))
                w, h = float(r[3]), float(r[4])
            except ValueError:
                continue
            st.class_counts[cid] = st.class_counts.get(cid, 0) + 1
            st.objects += 1
            st.areas.append(w * h)
            if h > 0:
                st.aspect_ratios.append(w / h)
    if not st.images:
        st.images = len(st.objects_per_image)
    return st
