"""数据集画像。

项目要支持**任意符合 YOLO v8 格式的数据集**，类别数量不限。因此不能再用
预设的类别方案来决定网络输出维度，必须以数据集自身的 data.yaml 为准。

本模块负责：
    1. 从 data.yaml 或目录解析出真实的划分、类别名与类别数
    2. 统计每个划分的图像/标签/目标数
    3. 生成一份路径全部绝对化、可直接交给 ultralytics 的配置

不依赖项目内的任何预设方案，导入什么数据集就识别出什么类别。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .checker import IMAGE_EXTENSIONS

# 划分目录可能出现的名字，按优先级
SPLIT_CANDIDATES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}


@dataclass
class SplitInfo:
    name: str
    images_dir: Path
    labels_dir: Optional[Path] = None
    images: int = 0
    labels: int = 0
    boxes: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def mean_boxes(self) -> float:
        return self.boxes / self.images if self.images else 0.0


@dataclass
class DatasetProfile:
    """一个 YOLO 数据集的完整描述，类别完全来自数据本身。"""

    root: Path
    names: Dict[int, str]
    splits: Dict[str, SplitInfo]
    yaml_path: Optional[Path] = None

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def total_images(self) -> int:
        return sum(s.images for s in self.splits.values())

    @property
    def total_boxes(self) -> int:
        return sum(s.boxes for s in self.splits.values())

    def distribution(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for split in self.splits.values():
            for cid, n in split.class_counts.items():
                out[cid] = out.get(cid, 0) + n
        return out

    def imbalance_ratio(self) -> float:
        values = [v for v in self.distribution().values() if v > 0]
        return max(values) / min(values) if len(values) > 1 and min(values) else 0.0

    def summary(self) -> str:
        parts = [f"{k}={v.images} 图/{v.boxes} 框" for k, v in self.splits.items()]
        return (f"{self.num_classes} 类，{self.total_images} 图，"
                f"{self.total_boxes} 框  ({', '.join(parts)})")

    # ------------------------------------------------------------ 解析
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DatasetProfile":
        yaml_path = Path(yaml_path).resolve()
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        base = _resolve_base(cfg, yaml_path)

        names = cfg.get("names") or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        names = {int(k): str(v) for k, v in names.items()}

        splits: Dict[str, SplitInfo] = {}
        for split, keys in SPLIT_CANDIDATES.items():
            rel = next((cfg[k] for k in keys if k in cfg and cfg[k]), None)
            if not rel:
                continue
            img_dir = _resolve_dir(rel, base, yaml_path)
            if (img_dir is None or not img_dir.exists()) and not Path(str(rel)).is_absolute():
                # 配置里写了但按字面解析找不到时，退回约定目录名再试一次
                img_dir = _conventional_split_dir(split, base, yaml_path)
            if img_dir is None or not img_dir.exists():
                continue
            splits[split] = _scan_split(split, img_dir)

        # 数据集根目录要取「实际存在的划分目录的上级」，不能直接用 cfg['path']。
        # 原始 data.yaml 里写的是 path: data，而配置文件本身就在 data/ 目录下，
        # 直接拼接会得到 data/data 这种不存在的路径。
        root = base
        if splits:
            first = next(iter(splits.values()))
            root = first.images_dir.parent.parent
        return cls(root=root, names=names, splits=splits, yaml_path=yaml_path)

    @classmethod
    def from_dir(cls, root: Path) -> "DatasetProfile":
        """没有 data.yaml 时，按目录结构与标签内容推断。"""
        root = Path(root).resolve()
        splits: Dict[str, SplitInfo] = {}
        for split, keys in SPLIT_CANDIDATES.items():
            for key in keys:
                img_dir = root / key / "images"
                if img_dir.exists():
                    splits[split] = _scan_split(split, img_dir)
                    break
        names = infer_names(splits)
        return cls(root=root, names=names, splits=splits, yaml_path=None)

    def to_yaml(self, out_path: Path) -> Path:
        """写出路径绝对化的 data.yaml。

        ultralytics 对相对路径的解释依赖工作目录，换个地方跑就找不到数据，
        所以这里一律写绝对路径。
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {"path": str(self.root).replace("\\", "/")}
        for split, info in self.splits.items():
            rel = info.images_dir.relative_to(self.root) if _is_relative(info.images_dir, self.root) else info.images_dir
            key = "val" if split == "val" else split
            payload[key] = str(rel).replace("\\", "/")
        payload["names"] = self.names
        header = "# 由系统根据实际数据集生成，路径为绝对路径，请勿手工修改\n"
        out_path.write_text(
            header + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.yaml_path = out_path
        return out_path


def _is_relative(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_base(cfg: dict, yaml_path: Path) -> Path:
    raw = cfg.get("path")
    if not raw:
        return yaml_path.parent.resolve()
    base = Path(str(raw))
    return base.resolve() if base.is_absolute() else (yaml_path.parent / base).resolve()


def _strip_parent_refs(path: Path) -> Optional[Path]:
    """去掉路径里的 ``..`` 与 ``.``，用于兜底。

    Roboflow 导出的 data.yaml 常写 ``train: ../train/images``，
    但实际目录就在压缩包根下，按字面解析会跑到上一级去找。
    """
    parts = [part for part in path.parts if part not in ("..", ".", "")]
    if not parts or len(parts) == len(path.parts):
        return None
    return Path(*parts)


def _resolve_dir(rel: str, base: Path, yaml_path: Path) -> Optional[Path]:
    p = Path(str(rel))
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([base / p, yaml_path.parent / p])
        cleaned = _strip_parent_refs(p)
        if cleaned is not None:
            candidates.extend([yaml_path.parent / cleaned, base / cleaned])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _conventional_split_dir(split: str, base: Path, yaml_path: Path) -> Optional[Path]:
    """按约定目录名兜底：<root>/train/images、<root>/valid/images 或 val/images。"""
    for key in SPLIT_CANDIDATES.get(split, (split,)):
        for anchor in (base, yaml_path.parent):
            candidate = anchor / key / "images"
            if candidate.exists():
                return candidate.resolve()
    return None


def _scan_split(name: str, img_dir: Path) -> SplitInfo:
    """统计一个划分的图像、标签与目标数。"""
    info = SplitInfo(name=name, images_dir=img_dir)
    lab_dir = img_dir.parent / "labels"
    if lab_dir.exists():
        info.labels_dir = lab_dir

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    info.images = len(images)

    if info.labels_dir:
        for txt in info.labels_dir.glob("*.txt"):
            info.labels += 1
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                info.boxes += 1
                info.class_counts[cid] = info.class_counts.get(cid, 0) + 1
    return info


def infer_names(splits: Dict[str, SplitInfo]) -> Dict[int, str]:
    """从标签里出现过的类别编号推断类别表。

    没有 data.yaml 时只能拿到编号，占位成 class_N，之后可在建档界面改名。
    """
    seen: set[int] = set()
    for info in splits.values():
        seen.update(info.class_counts)
    return {cid: f"class_{cid}" for cid in sorted(seen)}


def load_profile(source: Path) -> DatasetProfile:
    """统一入口：给目录或 data.yaml 都能得到画像。"""
    source = Path(source).resolve()
    if source.is_file():
        return DatasetProfile.from_yaml(source)
    for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
        p = source / candidate
        if p.exists():
            return DatasetProfile.from_yaml(p)
    # 没有配置文件时看看放它的上一级（Roboflow 压缩包常这样）
    parent_yaml = source.parent / "data.yaml"
    if parent_yaml.exists():
        return DatasetProfile.from_yaml(parent_yaml)
    return DatasetProfile.from_dir(source)
