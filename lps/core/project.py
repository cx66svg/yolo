"""项目建档与持久化。

用户的工作方式从「一次训练」变成「一个项目」：
    新建项目 → 选/拖入数据集 → 保存 → 训练 → 导出模型

一个项目记录自己的数据集、类别表、历次训练与最优权重，彼此隔离。
元数据存成单个 JSON，写入用「临时文件 + 原子替换」，避免中途断电把
已有项目列表写坏。
"""
from __future__ import annotations

import json
import os
import shutil
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from ..data.profile import DatasetProfile, load_profile
from ..data.splitter import ensure_splits, remap_classes

STORE_PATH = config.APP_ROOT / "projects.json"


@dataclass
class Project:
    pid: str
    name: str
    dataset_root: str
    data_yaml: str
    class_names: Dict[int, str]
    created_at: str
    note: str = ""
    runs: List[str] = field(default_factory=list)
    best_weights: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def dataset_path(self) -> Path:
        return Path(self.dataset_root)

    def class_summary(self, limit: int = 6) -> str:
        items = [self.class_names[k] for k in sorted(self.class_names)][:limit]
        text = "、".join(items)
        if self.num_classes > limit:
            text += f" 等 {self.num_classes} 类"
        return text

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "name": self.name,
            "dataset_root": self.dataset_root,
            "data_yaml": self.data_yaml,
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "created_at": self.created_at,
            "note": self.note,
            "runs": list(self.runs),
            "best_weights": self.best_weights,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(
            pid=str(data["pid"]),
            name=str(data.get("name", "")),
            dataset_root=str(data.get("dataset_root", "")),
            data_yaml=str(data.get("data_yaml", "")),
            class_names={int(k): str(v) for k, v in (data.get("class_names") or {}).items()},
            created_at=str(data.get("created_at", "")),
            note=str(data.get("note", "")),
            runs=list(data.get("runs") or []),
            best_weights=str(data.get("best_weights", "")),
        )


class ProjectStore:
    """项目列表的读写。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or STORE_PATH)
        self._projects: Dict[str, Project] = {}
        self.load()

    # ------------------------------------------------------------ 持久化
    def load(self) -> None:
        self._projects = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 文件损坏时不覆盖用户数据，改成备份后从空开始
            backup = self.path.with_suffix(f".broken.{int(time.time())}.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return
        for item in raw.get("projects", []):
            try:
                project = Project.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
            self._projects[project.pid] = project
        self._repair_paths()
        if self._prune_runs():
            self.save()

    def _repair_paths(self) -> None:
        """修复历史项目里失效的数据集根路径。

        早期版本解析 data.yaml 时直接把 `path` 拼到配置文件所在目录，
        遇到 `path: data` 而配置文件本身就在 data/ 下的情况，会得到
        `data/data` 这种不存在的路径，项目详情里就显示"数据集目录不存在"。
        这里按"训练集目录能否取到"反推真实根目录并就地修正。
        """
        changed = False
        for project in self._projects.values():
            if Path(project.dataset_root).exists():
                continue
            fixed = self._derive_root(project)
            if fixed is None:
                continue
            project.dataset_root = str(fixed)
            # 配置里的 path 同样是错的，顺手按正确根目录重写一遍
            try:
                profile = load_profile(fixed)
                if profile.names:
                    project.class_names = profile.names
                profile.to_yaml(Path(project.data_yaml))
            except (OSError, ValueError):
                pass
            changed = True
        if changed:
            self.save()

    @staticmethod
    def _derive_root(project: "Project") -> Optional[Path]:
        """从数据配置反推数据集根目录。"""
        yaml_path = Path(project.data_yaml)
        if not yaml_path.exists():
            return None
        import yaml as _yaml

        try:
            cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except _yaml.YAMLError:
            return None

        train_rel = str(cfg.get("train", "train/images"))
        bases = [Path(str(cfg.get("path", ""))) if cfg.get("path") else None,
                 Path(project.dataset_root),
                 yaml_path.parent]
        for base in bases:
            if base is None:
                continue
            target = Path(train_rel) if Path(train_rel).is_absolute() else base / train_rel
            if target.exists():
                # target 形如 <root>/train/images，往上两级即根目录
                return target.parent.parent
        return None

    def _prune_runs(self) -> bool:
        """丢掉指向已不存在实验目录的训练记录，返回是否有改动。

        手动删除 outputs/runs 下的目录后，项目里的记录会指向空路径，
        界面上就会出现「点开却是空的」条目。
        """
        changed = False
        for project in self._projects.values():
            kept = [name for name in project.runs if (config.RUNS_DIR / name).is_dir()]
            if len(kept) != len(project.runs):
                project.runs = kept
                changed = True
        return changed

    def save(self) -> None:
        payload = {"version": 1, "projects": [p.to_dict() for p in self.all()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)   # 同盘原子替换，避免写一半损坏

    # ------------------------------------------------------------ 查询
    def all(self) -> List[Project]:
        return sorted(self._projects.values(), key=lambda p: p.created_at)

    def get(self, pid: str) -> Optional[Project]:
        return self._projects.get(pid)

    def by_name(self, name: str) -> Optional[Project]:
        for project in self._projects.values():
            if project.name == name:
                return project
        return None

    # ------------------------------------------------------------ 变更
    @staticmethod
    def _extract(archive: Path) -> Path:
        """解压到工作目录并定位真正的数据集根（有些包外面多套一层目录）。"""
        target = config.OUTPUT_DIR / "datasets" / "imported" / archive.stem
        if not target.exists() or not any(target.iterdir()):
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        return ProjectStore._locate_root(target)

    @staticmethod
    def _locate_root(root: Path) -> Path:
        from ..data.profile import SPLIT_CANDIDATES

        def has_split(path: Path) -> bool:
            return any((path / key / "images").exists()
                       for keys in SPLIT_CANDIDATES.values() for key in keys)

        if has_split(root):
            return root
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            if has_split(child):
                return child
            for grand in sorted(p for p in child.iterdir() if p.is_dir()):
                if has_split(grand):
                    return grand
        return root

    def create(self, name: str, dataset: Path, note: str = "") -> Project:
        """建档：解析数据集，写出该项目的 data.yaml。"""
        name = name.strip()
        if not name:
            raise ValueError("项目名称不能为空")
        if self.by_name(name):
            raise ValueError(f"项目「{name}」已存在，请换一个名称")

        # 用户常常直接选压缩包：先解压再解析，否则会把 .zip 当文本读而抛 UnicodeDecodeError
        dataset = Path(dataset)
        if dataset.is_file():
            if not zipfile.is_zipfile(dataset):
                raise ValueError(f"不支持的文件类型：{dataset.name}，请选择数据集文件夹或 .zip")
            dataset = self._extract(dataset)

        profile = load_profile(dataset)
        if not profile.splits:
            raise ValueError(
                "数据集结构不符合 YOLO 格式，未找到 images 目录。\n"
                "期望结构：train/images、valid/images、test/images"
            )
        if not profile.names:
            raise ValueError("未能识别出任何类别，请检查 labels 目录中的标签文件")

        pid = uuid.uuid4().hex[:10]

        # 类别编号不连续（例如删过类别后剩下 0/2/5）时，nc 会按个数算成 3，
        # 而标签里仍出现 5 → 训练时类别索引越界。先把编号压成 0..n-1。
        remap_note = ""
        used_ids = sorted(cid for cid, count in profile.distribution().items() if count > 0)
        if used_ids and used_ids != list(range(len(used_ids))):
            remap_root = config.OUTPUT_DIR / "datasets" / f"derived_{pid}_remap"
            mapping = remap_classes(profile, remap_root)
            if mapping:
                profile = load_profile(remap_root)
                pairs = "、".join(f"{old}→{new}" for old, new in sorted(mapping.items()))
                remap_note = f"类别编号已重排为连续（{pairs}）"

        # ultralytics 硬性要求 train 与 val 同时存在；test 虽可选，但缺了就没法做
        # 独立评估。缺哪个就从 train 切哪个，生成一份自包含的派生数据集。
        split_note = ""
        if ("val" not in profile.splits) or ("test" not in profile.splits):
            derived_root = config.OUTPUT_DIR / "datasets" / f"derived_{pid}"
            try:
                result = ensure_splits(profile, derived_root)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            if result is not None:
                profile = load_profile(derived_root)
                split_note = result.summary()

        yaml_path = config.USER_CONFIG_DIR / f"project_{pid}.yaml"
        profile.to_yaml(yaml_path)

        project = Project(
            pid=pid,
            name=name,
            dataset_root=str(profile.root),
            data_yaml=str(yaml_path),
            class_names=profile.names,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            note="；".join(x for x in (note, remap_note, split_note) if x),
        )
        self._projects[pid] = project
        self.save()
        return project

    def update(self, project: Project) -> None:
        self._projects[project.pid] = project
        self.save()

    def delete(self, pid: str) -> None:
        """删除建档记录，并清掉本系统自动生成的派生数据集与项目配置。

        删项目时不清派生数据集会留下几百 MB 的孤儿目录；但**只删 derived_* 这种
        自己生成的目录**，用户提供的原始数据集绝对不动。
        """
        project = self._projects.pop(pid, None)
        if project is not None:
            datasets_root = config.OUTPUT_DIR / "datasets"
            # 一个项目可能产生多个派生目录（重排编号 + 补充划分），按 pid 一并清掉
            for derived in datasets_root.glob(f"derived_{pid}*"):
                shutil.rmtree(derived, ignore_errors=True)
            cfg = Path(project.data_yaml)
            if cfg.name.startswith("project_") and cfg.exists():
                cfg.unlink(missing_ok=True)
        self.save()

    def refresh(self, project: Project) -> DatasetProfile:
        """重新解析项目的数据集（数据集可能被移动或补充过）。

        无论类别是否变化都要重写配置：数据集换位置后，配置里的绝对路径会失效，
        只按"类别是否变化"判断会漏掉这种情况。
        """
        profile = load_profile(Path(project.dataset_root))
        if profile.names:
            project.class_names = profile.names
        project.dataset_root = str(profile.root)
        profile.to_yaml(Path(project.data_yaml))
        self.update(project)
        return profile

    def register_run(self, project: Project, run_name: str,
                     best_weights: Optional[Path] = None) -> None:
        if run_name not in project.runs:
            project.runs.append(run_name)
        if best_weights and Path(best_weights).exists():
            project.best_weights = str(best_weights)
        self.update(project)

    # ------------------------------------------------------------ 校验
    def validate(self, project: Project) -> tuple[bool, str]:
        """检查项目的数据集是否仍可用。

        不要求配置里的 path 单独存在：原始 data.yaml 常写 path: data 而自身就在
        data/ 目录下，正确的判断依据是「训练集目录能否取到」。
        """
        if not Path(project.dataset_root).exists():
            return False, f"数据集目录不存在：{project.dataset_root}"
        yaml_path = Path(project.data_yaml)
        if not yaml_path.exists():
            return False, f"数据配置不存在：{yaml_path}"
        import yaml as _yaml

        try:
            cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except _yaml.YAMLError as exc:
            return False, f"数据配置损坏：{exc}"

        train_rel = str(cfg.get("train", "train/images"))
        candidates = [Path(str(cfg.get("path", ""))), Path(project.dataset_root),
                      yaml_path.parent]
        for base in candidates:
            if not str(base):
                continue
            target = Path(train_rel) if Path(train_rel).is_absolute() else base / train_rel
            if target.exists():
                return True, "就绪"
        return False, f"训练集目录不存在：{cfg.get('train')}"
