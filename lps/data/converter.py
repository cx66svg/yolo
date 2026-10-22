"""类别重映射与数据集派生。

原始数据集的标签是 eat/rest/run/walk 四类。本模块按 ``ClassScheme``
生成一份新的数据集视图：图像用硬链接（同一分区下不占额外空间），
标签按映射表重写，data.yaml 一并生成。

原始文件全程只读，不会因为换方案被破坏。
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .. import config
from ..config import ClassScheme

SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}

# 原始数据集有中英两套类名（框架根目录用 eat/rest/run/walk，
# 数据目录内的 data.yaml 用 采食/休息/奔跑/行走），统一归一化到英文规范名。
NAME_ALIASES = {
    "eat": "eat", "eating": "eat", "采食": "eat", "进食": "eat",
    "rest": "rest", "resting": "rest", "休息": "rest", "卧息": "rest",
    "run": "run", "running": "run", "奔跑": "run", "跑": "run",
    "walk": "walk", "walking": "walk", "行走": "walk",
}


def canonical_name(name: str) -> str:
    key = str(name).strip().lower()
    return NAME_ALIASES.get(key, key)


@dataclass
class ConvertResult:
    scheme: str
    yaml_path: Path
    root: Path
    images: int = 0
    images_dropped: int = 0
    boxes_written: int = 0
    boxes_dropped: int = 0
    per_class: Dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_class is None:
            self.per_class = {}

    def summary(self) -> str:
        dropped = ""
        if self.images_dropped:
            dropped = f", 跳过 {self.images_dropped} 张（上下文不明）"
        return (
            f"方案 {self.scheme}: {self.images} 张图{dropped}, "
            f"写入 {self.boxes_written} 个框, 丢弃 {self.boxes_dropped} 个框, "
            f"类别分布 {dict(sorted(self.per_class.items()))}\n"
            f"配置: {self.yaml_path}"
        )


class DatasetConverter:
    """把源数据集转换成指定类别方案的派生数据集。"""

    def __init__(self, source_root: Path | str | None = None,
                 source_yaml: Optional[Path] = None):
        self.source_root = Path(source_root or config.DATA_ROOT)
        self.source_yaml = source_yaml or (self.source_root / "data.yaml")
        if not self.source_yaml.exists():
            # 原始框架的 data.yaml 与本项目不同名，兜底找同级 yaml
            candidates = sorted(self.source_root.glob("*.yaml")) if self.source_root.exists() else []
            if candidates:
                self.source_yaml = candidates[0]
        self._old_names: Dict[int, str] = {}

    # ------------------------------------------------------------ 读取
    def load_source_names(self) -> Dict[int, str]:
        if not self.source_yaml.exists():
            raise FileNotFoundError(f"找不到源数据配置: {self.source_yaml}")
        cfg = yaml.safe_load(self.source_yaml.read_text(encoding="utf-8")) or {}
        names = cfg.get("names") or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        self._old_names = {
            int(k): canonical_name(v) for k, v in names.items()
        }
        return self._old_names

    def _build_lut(self, scheme: ClassScheme) -> Dict[int, int]:
        """原始编号 -> 新编号。类名在 map 里找不到时抛错，避免静默丢数据。

        启用图像级上下文的方案不走这个表，改由 scheme.resolve 逐图计算。
        """
        if not self._old_names:
            self.load_source_names()
        lut: Dict[int, int] = {}
        for old_id, name in self._old_names.items():
            if scheme.uses_context:
                if name not in scheme.combine and name not in scheme.source_map:
                    raise KeyError(
                        f"类别方案 {scheme.key} 未定义源类名 {name!r} 的组合规则，"
                        f"请检查 configs/class_scheme.yaml 的 combine 字段"
                    )
                continue
            if name not in scheme.source_map:
                raise KeyError(
                    f"类别方案 {scheme.key} 未定义源类名 {name!r} 的映射，"
                    f"请检查 configs/class_scheme.yaml"
                )
            lut[old_id] = scheme.source_map[name]
        return lut

    @staticmethod
    def context_index(stem: str, scheme: ClassScheme) -> Optional[int]:
        """从文件名解析图像级上下文序号（如拍摄视角）。

        原始文件名形如 022-sit-front-_MOV-0000_jpg.rf.<hash>，
        视角信息由数据集作者写入，直接读取即可，无需推断。
        """
        if not scheme.uses_context:
            return None
        lowered = stem.lower()
        for token, index in scheme.context_tokens.items():
            if token.lower() in lowered:
                return index
        return None

    # ------------------------------------------------------------ 主流程
    def convert(self, scheme: ClassScheme, output_root: Optional[Path] = None,
                overwrite: bool = True) -> ConvertResult:
        self.load_source_names()
        lut = self._build_lut(scheme)

        root = Path(output_root or (config.OUTPUT_DIR / "datasets" / scheme.key))
        if root.exists() and overwrite:
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

        result = ConvertResult(scheme=scheme.key, yaml_path=Path(""), root=root)

        for split_dir in sorted(p for p in self.source_root.iterdir() if p.is_dir()):
            split = SPLIT_ALIASES.get(split_dir.name)
            if split is None:
                continue
            img_src = split_dir / "images"
            lab_src = split_dir / "labels"
            if not img_src.exists():
                continue
            self._convert_split(split, img_src, lab_src, root, lut, result, scheme)

        result.yaml_path = self._write_yaml(root, scheme, result)
        return result

    def _convert_split(self, split: str, img_src: Path, lab_src: Path,
                       root: Path, lut: Dict[int, int], result: ConvertResult,
                       scheme: ClassScheme) -> None:
        img_dst = root / split / "images"
        lab_dst = root / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lab_dst.mkdir(parents=True, exist_ok=True)

        for img in sorted(img_src.iterdir()):
            if not img.is_file():
                continue

            ctx = self.context_index(img.stem, scheme)
            # 启用上下文的方案遇到无法确定上下文的图像时整张跳过，
            # 避免把视角不明的样本混进某一类污染标签
            if scheme.uses_context and ctx is None and scheme.drop_unknown:
                result.images_dropped += 1
                continue

            dst_img = img_dst / img.name
            if not dst_img.exists():
                try:
                    os.link(img, dst_img)          # 同分区硬链接，零拷贝
                except OSError:
                    shutil.copy2(img, dst_img)     # 跨分区时退化为复制
            result.images += 1

            lab_src_file = lab_src / f"{img.stem}.txt"
            if not lab_src_file.exists():
                (lab_dst / f"{img.stem}.txt").write_text("", encoding="utf-8")
                continue

            out_lines: List[str] = []
            for line in lab_src_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                old_id = int(float(parts[0]))
                if scheme.uses_context:
                    name = self._old_names.get(old_id)
                    new_id = scheme.resolve(name, ctx) if name else None
                else:
                    new_id = lut.get(old_id)
                if new_id is None:
                    result.boxes_dropped += 1
                    continue
                out_lines.append(f"{new_id} " + " ".join(parts[1:]))
                result.boxes_written += 1
                result.per_class[new_id] = result.per_class.get(new_id, 0) + 1
            (lab_dst / f"{img.stem}.txt").write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
            )

    def _write_yaml(self, root: Path, scheme: ClassScheme,
                    result: ConvertResult) -> Path:
        config.USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        yaml_path = config.USER_CONFIG_DIR / f"data_{scheme.key}.yaml"

        splits = {
            s: f"{s}/images" for s in ("train", "val", "test")
            if (root / s / "images").exists() and any((root / s / "images").iterdir())
        }
        payload = {
            "path": str(root).replace("\\", "/"),
            "train": splits.get("train", "train/images"),
            "val": splits.get("val", splits.get("train", "train/images")),
            "names": scheme.names,
            "_scheme": scheme.key,
        }
        if "test" in splits:
            payload["test"] = splits["test"]

        header = (
            f"# 由 lps.data.converter 自动生成，请勿手工修改\n"
            f"# 类别方案: {scheme.key}（{scheme.num_classes} 类）\n"
        )
        yaml_path.write_text(
            header + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return yaml_path
