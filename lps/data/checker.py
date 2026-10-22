"""数据集自检。

原始框架的 README 承诺了一个 check_dataset.py，但压缩包里并不存在该文件。
这里按 YOLO 数据集的实际约束重新实现，覆盖六类常见问题：

1. 图像与标签没有一一配对
2. 标签行字段数不对、坐标越界、宽高非正
3. 类别编号超出 data.yaml 定义的范围
4. 空标签文件（有图无框）
5. 图像文件损坏无法读取
6. 数据集划分缺失或为空

检查过程只读，不修改任何文件。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 问题等级
ERROR = "error"
WARN = "warn"


@dataclass
class Issue:
    level: str
    split: str
    path: str
    message: str


@dataclass
class CheckReport:
    data_yaml: str
    splits: Dict[str, Dict[str, int]] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    class_counts: Dict[str, Dict[int, int]] = field(default_factory=dict)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.level == WARN]

    @property
    def passed(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"数据集: {self.data_yaml}"]
        for split, info in self.splits.items():
            counts = self.class_counts.get(split, {})
            lines.append(
                f"  {split:6s} 图像 {info['images']:5d}  标签 {info['labels']:5d}  "
                f"框 {info['boxes']:6d}  各类 {dict(sorted(counts.items()))}"
            )
        lines.append(f"  错误 {len(self.errors)} 项，警告 {len(self.warnings)} 项")
        return "\n".join(lines)


class DatasetChecker:
    """按 data.yaml 校验数据集完整性。"""

    def __init__(self, data_yaml: Path | str, max_issues_per_type: int = 20):
        self.data_yaml = Path(data_yaml)
        self.max_issues_per_type = max_issues_per_type
        self._issue_counter: Dict[str, int] = {}

    # ------------------------------------------------------------ 内部
    def _add(self, level: str, split: str, path: str, message: str) -> None:
        key = f"{level}:{message.split(':')[0]}"
        self._issue_counter[key] = self._issue_counter.get(key, 0) + 1
        if self._issue_counter[key] <= self.max_issues_per_type:
            self.report.issues.append(Issue(level, split, path, message))

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _resolve_root(cfg: dict) -> Path:
        root = cfg.get("path")
        if root:
            return Path(root)
        return Path(cfg.get("_yaml_dir", "."))

    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # ------------------------------------------------------------ 主流程
    def run(self) -> CheckReport:
        cfg = self._load_yaml(self.data_yaml)
        cfg["_yaml_dir"] = str(self.data_yaml.parent)
        self.report = CheckReport(data_yaml=str(self.data_yaml))

        names = cfg.get("names") or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        names = {int(k): str(v) for k, v in names.items()}
        if not names:
            self._add(ERROR, "-", str(self.data_yaml), "类别定义缺失: names 字段为空")
            return self.report

        root = self._resolve_root(cfg)

        for split in ("train", "val", "test"):
            rel = cfg.get(split)
            if not rel:
                if split == "train":
                    self._add(ERROR, split, str(self.data_yaml), "缺少 train 划分")
                continue
            img_dir = (root / rel) if not Path(rel).is_absolute() else Path(rel)
            # YOLO 约定：labels 目录与 images 同级。直接取父目录拼接，
            # 比做字符串替换可靠（目录名里出现 images 字样时不会误伤）。
            lab_dir = (img_dir.parent / "labels") if img_dir.name == "images" else img_dir

            self._check_split(split, img_dir, lab_dir, names)

        return self.report

    def _check_split(
        self, split: str, img_dir: Path, lab_dir: Path, names: Dict[int, str]
    ) -> None:
        images = self._list_images(img_dir)
        if not images:
            self._add(ERROR, split, str(img_dir), "划分内没有图像: 目录为空或不存在")
            self.report.splits[split] = {"images": 0, "labels": 0, "boxes": 0}
            self.report.class_counts[split] = {}
            return

        label_files = {p.stem for p in lab_dir.glob("*.txt")} if lab_dir.exists() else set()
        class_counts: Dict[int, int] = {}
        boxes = 0
        matched = 0

        for img in images:
            if img.stem not in label_files:
                self._add(WARN, split, str(img), "缺少标签: 图像无对应 txt")
                continue
            matched += 1
            txt = lab_dir / f"{img.stem}.txt"
            boxes += self._check_label_file(split, txt, names, class_counts)

        image_stems = {p.stem for p in images}
        for stem in sorted(label_files - image_stems):
            self._add(WARN, split, str(lab_dir / f"{stem}.txt"), "孤立标签: 无对应图像")

        self.report.splits[split] = {
            "images": len(images),
            "labels": len(label_files),
            "boxes": boxes,
        }
        self.report.class_counts[split] = class_counts
        if matched and len(label_files) and abs(matched - len(label_files)) > 0:
            self._add(
                WARN, split, str(lab_dir),
                f"配对不一致: 图像 {len(images)} 张，标签 {len(label_files)} 个",
            )

    def _check_label_file(
        self, split: str, txt: Path, names: Dict[int, str], class_counts: Dict[int, int]
    ) -> int:
        try:
            content = txt.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            self._add(ERROR, split, str(txt), "编码错误: 标签文件不是 UTF-8")
            return 0
        if not content:
            self._add(WARN, split, str(txt), "空标签: 文件存在但没有标注行")
            return 0

        count = 0
        for lineno, line in enumerate(content.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                self._add(ERROR, split, str(txt),
                          f"字段数错误: 第 {lineno} 行有 {len(parts)} 列，应为 5 列")
                continue
            try:
                cls_id = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                self._add(ERROR, split, str(txt), f"数值解析失败: 第 {lineno} 行")
                continue
            if cls_id not in names:
                self._add(ERROR, split, str(txt),
                          f"类别越界: 第 {lineno} 行 cls={cls_id}，允许 0~{max(names)}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                self._add(ERROR, split, str(txt),
                          f"坐标越界: 第 {lineno} 行未归一化到 [0,1]")
                continue
            if w <= 0 or h <= 0:
                self._add(ERROR, split, str(txt), f"宽高非法: 第 {lineno} 行 w={w} h={h}")
                continue
            if w >= 0.99 and h >= 0.99:
                self._add(WARN, split, str(txt),
                          f"框过大: 第 {lineno} 行几乎覆盖整幅图，疑似标注错误")
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            count += 1
        return count


def quick_verify_images(dirs: Sequence[Path], sample: int = 0) -> List[Issue]:
    """抽查图像能否被 OpenCV 正常解码。sample=0 表示不抽样。"""
    import cv2

    issues: List[Issue] = []
    for d in dirs:
        if not d.exists():
            continue
        images = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if sample:
            step = max(1, len(images) // sample)
            images = images[::step][:sample]
        for img in images:
            if cv2.imread(str(img)) is None:
                issues.append(Issue(ERROR, "verify", str(img), "图像损坏: 无法解码"))
    return issues


def file_digest(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """计算文件 MD5，用于发现跨划分重复样本。"""
    h = hashlib.md5()
    try:
        with path.open("rb") as f:
            while block := f.read(chunk):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def find_duplicates(image_dirs: Sequence[Path]) -> Dict[str, List[str]]:
    """跨划分查找重复图像，返回 {md5: [路径...]} 中数量大于 1 的项。"""
    seen: Dict[str, List[str]] = {}
    for d in image_dirs:
        if not d.exists():
            continue
        for img in sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            digest = file_digest(img)
            if digest:
                seen.setdefault(digest, []).append(str(img))
    return {k: v for k, v in seen.items() if len(v) > 1}
