"""数据集划分补全：给缺划分的数据集自动补出验证集 / 测试集。

`ultralytics` 对 data.yaml 的校验是硬性的（源码 data/utils.py）：

    "'train' and 'val' are required in all data YAMLs."

所以只有 train 的数据集原样导入会直接训练失败。而 test 是可选的，
但用户往往也需要一个独立测试集来看泛化效果。

这里从 train 里按比例切出缺少的划分，图像与标签用**硬链接**，
同分区下不额外占用磁盘，也不修改原始数据集。

需要说清楚的局限：切出来的验证/测试集与训练集同源，
只能反映"同分布留出集"上的表现，不等于真实场景的跨场景泛化。
"""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .profile import DatasetProfile, load_profile

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 三个划分的默认比例（只在对应划分缺失时才切）
VAL_RATIO = 0.15
TEST_RATIO = 0.10
MIN_IMAGES = 20


@dataclass
class SplitResult:
    root: Path
    counts: Dict[str, int] = field(default_factory=dict)
    labels_matched: int = 0
    created: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = " / ".join(f"{k} {v} 张" for k, v in self.counts.items())
        what = "、".join({"val": "验证集", "test": "测试集"}[k] for k in self.created)
        return f"已自动生成{what}（从训练集切分）：{parts}（标签配对 {self.labels_matched} 个）"


def link_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接（零额外空间），跨分区时退化为复制。"""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _pair_label(image: Path, labels_dir: Optional[Path]) -> Optional[Path]:
    if labels_dir is None:
        return None
    candidate = labels_dir / f"{image.stem}.txt"
    return candidate if candidate.exists() else None


def _place(image: Path, label: Optional[Path], dst_root: Path, split: str) -> bool:
    """把一张图（及标签）放进目标划分，返回是否有真实标签。"""
    link_or_copy(image, dst_root / split / "images" / image.name)
    if label is not None:
        link_or_copy(label, dst_root / split / "labels" / label.name)
        return True
    # 没有标签说明是负样本，仍要写一个空标签文件，否则 ultralytics 会告警缺标签
    (dst_root / split / "labels").mkdir(parents=True, exist_ok=True)
    (dst_root / split / "labels" / f"{image.stem}.txt").write_text("", encoding="utf-8")
    return False


def ensure_splits(source: DatasetProfile, out_root: Path,
                  val_ratio: float = VAL_RATIO,
                  test_ratio: float = TEST_RATIO,
                  seed: int = 42) -> Optional[SplitResult]:
    """补齐缺失的 val / test 划分，写到 out_root。

    - 已有 val 且已有 test：不做任何事，返回 None
    - 缺哪个就从 train 切哪个，没缺的不动
    - 没有任何 train：返回 None，由上层报错
    """
    if "train" not in source.splits:
        return None

    created = [name for name in ("val", "test") if name not in source.splits]
    if not created:
        return None

    train = source.splits["train"]
    images: List[Path] = sorted(
        path for path in train.images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < MIN_IMAGES:
        raise ValueError(
            f"训练集只有 {len(images)} 张图，不足以自动切分（至少需要 {MIN_IMAGES} 张）。"
            "请补充数据，或自行提供 valid 目录。"
        )

    random.Random(seed).shuffle(images)

    n_test = int(round(len(images) * test_ratio)) if "test" in created else 0
    n_val = int(round(len(images) * val_ratio)) if "val" in created else 0
    n_test = max(n_test, 1) if "test" in created else 0
    n_val = max(n_val, 1) if "val" in created else 0
    # 训练集必须留得下：至少保留一半
    while n_test + n_val > len(images) // 2 and (n_test + n_val) > 1:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    out_root = Path(out_root)
    cursor = 0
    split_images: Dict[str, List[Path]] = {}
    if n_test:
        split_images["test"] = images[cursor:cursor + n_test]
        cursor += n_test
    if n_val:
        split_images["val"] = images[cursor:cursor + n_val]
        cursor += n_val
    split_images["train"] = images[cursor:]

    matched = 0
    for split, items in split_images.items():
        for image in items:
            if _place(image, _pair_label(image, train.labels_dir), out_root, split):
                matched += 1

    # 原有的 val / test 若已存在则整体带过来，保证派生数据集自包含。
    # 漏了这一步会出现「有 train+val 的数据集补 test 后 val 消失」的问题。
    for existing in ("val", "test"):
        if existing not in source.splits:
            continue
        info = source.splits[existing]
        for image in sorted(info.images_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            _place(image, _pair_label(image, info.labels_dir), out_root, existing)

    derived = load_profile(out_root)
    # 派生目录里没有 data.yaml，推断出的类名会是 class_0/1…，
    # 必须沿用原数据集的类名，否则界面与日志里全是无意义的编号名。
    if source.names:
        derived.names = dict(source.names)
    derived.to_yaml(out_root / "data.yaml")

    # 计数以派生数据集实际扫描结果为准（含带过来的原始划分）
    counts = {name: info.images for name, info in derived.splits.items()}
    ordered = {k: counts[k] for k in ("train", "val", "test") if k in counts}

    return SplitResult(
        root=out_root,
        counts=ordered,
        labels_matched=matched,
        created=created,
    )


# 兼容旧调用名
ensure_val_split = ensure_splits

def remap_classes(source: DatasetProfile, out_root: Path) -> Optional[Dict[int, int]]:
    """把不连续的类别编号压成 0..n-1，返回「旧编号 -> 新编号」。

    有些数据集删过类别，标签里剩下 0、2、5 这种跳号。若直接拿来训练，
    `nc` 会按「不同编号的个数」算成 3，而标签里出现 5 → 类别索引越界。
    这里重建一份派生数据集：图像硬链接，标签只改第一列。
    编号已经连续时返回 None，不做任何事。
    """
    ids = sorted(cid for cid, count in source.distribution().items() if count > 0)
    if not ids or ids == list(range(len(ids))):
        return None

    mapping = {old: new for new, old in enumerate(ids)}
    out_root = Path(out_root)
    for split, info in source.splits.items():
        for image in sorted(info.images_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            link_or_copy(image, out_root / split / "images" / image.name)
            label = _pair_label(image, info.labels_dir)
            (out_root / split / "labels").mkdir(parents=True, exist_ok=True)
            target = out_root / split / "labels" / f"{image.stem}.txt"
            if label is None:
                target.write_text("", encoding="utf-8")
                continue
            lines = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    old = int(float(parts[0]))
                except ValueError:
                    continue
                new = mapping.get(old, old)
                lines.append(" ".join([str(new), *parts[1:]]))
            target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    derived = load_profile(out_root)
    # 自动生成的类名（class_2 这种）要把编号一起改，否则新编号 1 却叫 class_2 会误导；
    # 用户数据集里带的真实类名（如 goat）原样保留。
    renamed = {}
    for old, name in source.names.items():
        if old not in mapping:
            continue
        new = mapping[old]
        renamed[new] = f"class_{new}" if name == f"class_{old}" else name
    derived.names = renamed
    derived.to_yaml(out_root / "data.yaml")
    return mapping
