"""目标属性派生。

不新增人工标注，仅用已有的检测框几何信息派生四类可靠属性：

    zone        画面九宫格位置（1-9，左上为 1）
    scale       目标尺度分档（small / medium / large，按框面积占全图比例）
    occlusion   遮挡程度（与同图其他框的重叠比例，0~1）
    crowd       群体密度（该图目标数分档：sparse / normal / dense）

这些属性的共同点是可以从标注文件精确计算，不存在判定误差，
因此可以直接作为训练标签或统计分析维度使用。

毛色属性经原型验证不可靠（背景土壤与羊毛色接近，自动判定偏向白色），
故不在此实现，需要时应改为人工标注。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .checker import IMAGE_EXTENSIONS

Box = Tuple[float, float, float, float]  # x_center, y_center, w, h（归一化）


@dataclass
class ObjectRecord:
    split: str
    image: str
    cls_id: int
    box: Box
    zone: int
    area_ratio: float
    scale: str
    overlap: float
    occlusion: str
    crowd: str

    def as_row(self) -> List:
        x, y, w, h = self.box
        return [
            self.split, self.image, self.cls_id,
            round(x, 6), round(y, 6), round(w, 6), round(h, 6),
            self.zone, round(self.area_ratio, 6), self.scale,
            round(self.overlap, 4), self.occlusion, self.crowd,
        ]


CSV_HEADER = [
    "split", "image", "cls_id",
    "x_center", "y_center", "width", "height",
    "zone", "area_ratio", "scale", "overlap", "occlusion", "crowd",
]


def zone_of(x: float, y: float) -> int:
    """九宫格编号：按中心点落位，左上为 1，右下为 9。"""
    col = min(2, int(x * 3))
    row = min(2, int(y * 3))
    return row * 3 + col + 1


def scale_of(area_ratio: float) -> str:
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.10:
        return "medium"
    return "large"


def occlusion_of(overlap: float) -> str:
    if overlap < 0.10:
        return "none"
    if overlap < 0.35:
        return "partial"
    return "heavy"


def crowd_of(count: int) -> str:
    if count <= 2:
        return "sparse"
    if count <= 5:
        return "normal"
    return "dense"


def _xyxy(box: Box) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def _overlap_ratio(a: Box, b: Box) -> float:
    """a 被 b 覆盖的面积占 a 自身面积的比例。

    只与尺度相当（面积比 0.5~2 倍）的框比较。否则近距离大羊的框会把
    整只远处小羊包住，覆盖率算成 100%，但那并不是视觉遮挡，会把
    遮挡率虚高到 37%（实测修正前 37.2%，修正后 14.4%）。
    """
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    ratio = area_a / area_b
    if not (0.5 <= ratio <= 2.0):
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return min(1.0, (iw * ih) / area_a)


def derive_for_split(split: str, label_dir: Path) -> List[ObjectRecord]:
    records: List[ObjectRecord] = []
    if not label_dir.exists():
        return records

    for txt in sorted(label_dir.glob("*.txt")):
        items: List[Tuple[int, Box]] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
            items.append((cls_id, (x, y, w, h)))
        if not items:
            continue

        crowd = crowd_of(len(items))
        for idx, (cls_id, box) in enumerate(items):
            others = [b for j, (_, b) in enumerate(items) if j != idx]
            overlap = max((_overlap_ratio(box, o) for o in others), default=0.0)
            area = box[2] * box[3]
            records.append(ObjectRecord(
                split=split,
                image=txt.stem,
                cls_id=cls_id,
                box=box,
                zone=zone_of(box[0], box[1]),
                area_ratio=area,
                scale=scale_of(area),
                overlap=overlap,
                occlusion=occlusion_of(overlap),
                crowd=crowd,
            ))
    return records


def derive_all(data_root: Path) -> List[ObjectRecord]:
    """遍历 train / valid / test 三个划分。"""
    records: List[ObjectRecord] = []
    for split_name, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        label_dir = Path(data_root) / split_name / "labels"
        records.extend(derive_for_split(split, label_dir))
    return records


def export_csv(records: Sequence[ObjectRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in records:
            writer.writerow(r.as_row())
    return path


def summarize(records: Sequence[ObjectRecord],
              cls_names: Dict[int, str] | None = None) -> str:
    """生成属性分布摘要，供终端输出与报告使用。"""
    from collections import Counter

    names = cls_names or {}
    total = len(records)
    if not total:
        return "没有可统计的目标"

    lines = [f"目标总数 {total}"]

    def block(title: str, counter: Counter, order: Iterable | None = None) -> None:
        keys = order if order is not None else sorted(counter)
        parts = [f"{k}={counter.get(k, 0)}({counter.get(k,0)/total*100:.1f}%)" for k in keys]
        lines.append(f"  {title:8s} " + "  ".join(parts))

    block("姿态", Counter(names.get(r.cls_id, str(r.cls_id)) for r in records))
    block("区域", Counter(r.zone for r in records), order=range(1, 10))
    block("尺度", Counter(r.scale for r in records), order=["small", "medium", "large"])
    block("遮挡", Counter(r.occlusion for r in records), order=["none", "partial", "heavy"])
    block("密度", Counter(r.crowd for r in records), order=["sparse", "normal", "dense"])

    heavy = sum(1 for r in records if r.occlusion != "none")
    lines.append(f"  存在遮挡的目标 {heavy} 个，占 {heavy/total*100:.1f}%")
    return "\n".join(lines)


def cross_table(records: Sequence[ObjectRecord], attr: str,
                cls_names: Dict[int, str] | None = None) -> Dict[Tuple[str, str], int]:
    """姿态 × 属性 的交叉计数，用于软件里的交叉分析表。"""
    names = cls_names or {}
    table: Dict[Tuple[str, str], int] = {}
    for r in records:
        key = (names.get(r.cls_id, str(r.cls_id)), str(getattr(r, attr)))
        table[key] = table.get(key, 0) + 1
    return table
