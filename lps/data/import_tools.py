"""数据集导入工具。

用户可能给的是：一个 YOLO 目录、一个 Roboflow 导出的压缩包、或者只有 images
没有 data.yaml 的半成品。这里统一处理成项目内部可用的形态。
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .checker import IMAGE_EXTENSIONS

SPLIT_NAMES = ("train", "valid", "val", "test")


@dataclass
class DatasetLayout:
    root: Path
    splits: Dict[str, Path] = field(default_factory=dict)   # split -> images 目录
    yaml_path: Optional[Path] = None
    classes: Dict[int, str] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(
            sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            for d in self.splits.values()
        )

    def summary(self) -> str:
        parts = [f"{k}: {sum(1 for p in v.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)} 张"
                 for k, v in self.splits.items()]
        return " | ".join(parts)


def extract_archive(archive: Path, dest_root: Path) -> Path:
    """解压压缩包到目标目录，返回解压后的根目录。"""
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest_root)
    return dest_root


def _find_yaml(root: Path) -> Optional[Path]:
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        for candidate in root.rglob(name):
            return candidate
    return None


def inspect(root: Path) -> DatasetLayout:
    """识别目录结构，找出各划分的 images 目录与类别定义。"""
    root = Path(root)
    layout = DatasetLayout(root=root)

    for split in SPLIT_NAMES:
        img_dir = root / split / "images"
        if img_dir.exists():
            key = "val" if split == "valid" and "valid" not in layout.splits else split
            layout.splits[key] = img_dir
            continue
        alt = root / "images" if split == "train" else None
        if alt and alt.exists() and "train" not in layout.splits:
            layout.splits["train"] = alt

    yaml_path = _find_yaml(root)
    if yaml_path:
        layout.yaml_path = yaml_path
        try:
            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            names = cfg.get("names") or {}
            if isinstance(names, list):
                names = dict(enumerate(names))
            layout.classes = {int(k): str(v) for k, v in names.items()}
        except yaml.YAMLError:
            pass

    if not layout.classes:
        layout.classes = infer_classes(layout)
    return layout


def infer_classes(layout: DatasetLayout) -> Dict[int, str]:
    """没有 data.yaml 时，从标签文件里的编号推断类别数。"""
    seen = set()
    for img_dir in layout.splits.values():
        lab_dir = img_dir.parent / "labels"
        if not lab_dir.exists():
            continue
        for txt in lab_dir.glob("*.txt"):
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if parts:
                    try:
                        seen.add(int(float(parts[0])))
                    except ValueError:
                        continue
    return {i: f"class_{i}" for i in sorted(seen)}


def ensure_yaml(layout: DatasetLayout, out_path: Optional[Path] = None) -> Path:
    """生成或规范化 data.yaml，路径一律写绝对路径。"""
    root = layout.root
    payload = {
        "path": str(root).replace("\\", "/"),
        "train": "train/images",
        "val": ("valid/images" if "valid" in layout.splits else "val/images"),
        "names": layout.classes,
    }
    if "test" in layout.splits:
        payload["test"] = "test/images"

    target = Path(out_path) if out_path else (root / "data.yaml")
    target.write_text(
        "# 由系统导入时生成，路径为绝对路径\n"
        + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def count_boxes(layout: DatasetLayout) -> Dict[str, int]:
    """统计各划分的目标数。"""
    result: Dict[str, int] = {}
    for split, img_dir in layout.splits.items():
        lab_dir = img_dir.parent / "labels"
        boxes = 0
        if lab_dir.exists():
            for txt in lab_dir.glob("*.txt"):
                boxes += sum(1 for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines()
                             if len(line.split()) == 5)
        result[split] = boxes
    return result


def class_distribution(layout: DatasetLayout) -> Dict[int, int]:
    """全数据集类别分布。"""
    counter: Dict[int, int] = {}
    for img_dir in layout.splits.values():
        lab_dir = img_dir.parent / "labels"
        if not lab_dir.exists():
            continue
        for txt in lab_dir.glob("*.txt"):
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                counter[cid] = counter.get(cid, 0) + 1
    return counter
