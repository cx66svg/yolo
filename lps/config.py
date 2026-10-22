"""全局配置。

所有路径、类别方案、训练超参数集中在此，避免散落在各个脚本里。
路径支持环境变量覆盖，便于换机器或换数据集时不改代码。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


# ------------------------------------------------------------------ 路径
# 打包成 exe 后，代码与配置被解包到临时目录（只读），而输出必须写到
# exe 所在目录（可写且不会被清理）。因此把"资源根"和"可写根"分开。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT)) if _is_frozen() else PROJECT_ROOT
APP_ROOT = Path(sys.executable).resolve().parent if _is_frozen() else PROJECT_ROOT

CONFIG_DIR = RESOURCE_ROOT / "configs"          # 只读：随程序分发的模板
OUTPUT_DIR = APP_ROOT / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
EXPORT_DIR = OUTPUT_DIR / "exports"
WEIGHTS_DIR = APP_ROOT / "weights"

# 可写：用户在界面上导入数据集后生成的配置放这里。
# 不能写进 CONFIG_DIR —— 打包后那是临时的解包目录，程序退出即丢失，
# 且里面预置的 data_*.yaml 携带的是打包机器的绝对路径，换机器后必然失效。
USER_CONFIG_DIR = APP_ROOT / "configs"

# 原始数据集（老师给的框架所用数据）。
# 实际使用的位置由下面的 _resolve_data_root() 探测决定：数据集目录被移动、
# 改名或删除后，程序应能自动找到仍在的那一份，而不是直接报路径不存在。
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Administrator\Desktop\山羊检测\sheep_behavior_yolov8(1)\sheep_behavior_yolov8\data"
)

# 按顺序探测，第一个含 train/images 的即为有效
DATA_ROOT_CANDIDATES: List[Path] = [
    Path(os.environ.get("LPS_DATA_ROOT", "")) if os.environ.get("LPS_DATA_ROOT") else None,
    DEFAULT_DATA_ROOT,
    Path(r"C:\Users\Administrator\Desktop\sheep_behavior_yolov8(1)\sheep_behavior_yolov8\data"),
    Path(r"C:\Users\Administrator\Desktop\sheep_behavior_yolov8_原始备份\sheep_behavior_yolov8\data"),
    APP_ROOT / "datasets" / "sheep_behavior",
]


def _resolve_data_root() -> Path:
    """找出可用的原始数据集根目录。"""
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate and (candidate / "train" / "images").exists():
            return candidate
    return DEFAULT_DATA_ROOT


DATA_ROOT = _resolve_data_root()

# 预训练权重搜索路径：项目内 → 原始框架目录
PRETRAINED_SEARCH_DIRS: List[Path] = [
    WEIGHTS_DIR,                                          # 工程内 weights/，交付时自带
    APP_ROOT / "weights",
    APP_ROOT,
    DEFAULT_DATA_ROOT.parent,                             # 原始框架根目录
    Path(r"C:\Users\Administrator\Desktop\sheep_behavior_yolov8_原始备份\sheep_behavior_yolov8"),
    Path(r"C:\Users\Administrator\Desktop\sheep_behavior_yolov8(1)\sheep_behavior_yolov8"),
]


def scripts_dir() -> Path:
    """训练/验证/推理脚本所在目录，打包后同样可用。"""
    return RESOURCE_ROOT / "scripts"

# ------------------------------------------------------------------ 类别


@dataclass
class ClassScheme:
    """一套类别定义方案。

    source_map 把原始标签里的类名映射到新编号，重映射时只依赖它，
    因此换方案不需要改动任何标签文件。

    当方案叠加了「图像级上下文」时（例如从文件名解析出的拍摄视角），
    最终类别由 行为 × 视角 组合而成：
        combine[行为] = [正面编号, 侧面编号]
    这样同一个框在新方案下的编号取决于它所属图像的视角。
    """

    key: str
    names: Dict[int, str]          # 英文类名，写进 data.yaml
    zh_names: Dict[int, str]       # 中文名，用于界面显示
    source_map: Dict[str, int]     # 原始类名 -> 新编号
    colors: Dict[int, str] = field(default_factory=dict)
    # 图像级上下文：token -> 序号，序号对应 combine 里的位置
    context_tokens: Dict[str, int] = field(default_factory=dict)
    combine: Dict[str, List[int]] = field(default_factory=dict)
    drop_unknown: bool = True      # 无法确定上下文时是否丢弃该目标的框

    @property
    def uses_context(self) -> bool:
        return bool(self.context_tokens and self.combine)

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def zh(self, cls_id: int) -> str:
        return self.zh_names.get(cls_id, self.names.get(cls_id, str(cls_id)))

    def color(self, cls_id: int) -> str:
        return self.colors.get(cls_id, "#2E86DE")

    def resolve(self, source_name: str, context_index: int | None) -> int | None:
        """给出原始类名与该图的上下文序号，返回新类别编号。

        未启用上下文时退化为普通 source_map 查询；
        启用上下文但序号缺失时返回 None（由调用方决定丢弃）。
        """
        if not self.uses_context:
            return self.source_map.get(source_name)
        targets = self.combine.get(source_name)
        if targets is None:
            return self.source_map.get(source_name)
        if context_index is None or context_index >= len(targets):
            return None
        return targets[context_index]


# 默认配色：采食-橙、卧息-蓝、行走-绿、奔跑-红
_DEFAULT_COLORS = {0: "#E67E22", 1: "#2E86DE", 2: "#27AE60", 3: "#C0392B"}


def _load_class_schemes() -> tuple[str, Dict[str, ClassScheme]]:
    """加载预设类别方案。

    这套方案只服务于项目自带的羊行为数据集。用户导入任意 YOLO 数据集时，
    类别由数据本身决定，并不需要这些预设。因此配置文件缺失时**不能抛异常**，
    否则独立交付的模型包（不含 configs/）会一导入就崩。
    """
    path = CONFIG_DIR / "class_scheme.yaml"
    if not path.exists():
        return "", {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return "", {}
    schemes: Dict[str, ClassScheme] = {}
    for key, item in (raw.get("schemes") or {}).items():
        names = {int(k): v for k, v in item["names"].items()}
        zh = {int(k): v for k, v in item.get("zh", {}).items()}
        colors = {int(k): v for k, v in item.get("colors", {}).items()}
        context = item.get("context") or {}
        schemes[key] = ClassScheme(
            key=key,
            names=names,
            zh_names=zh,
            source_map={str(k): int(v) for k, v in item.get("source_map", {}).items()},
            colors=colors or {i: _DEFAULT_COLORS.get(i, "#7F8C8D") for i in names},
            context_tokens={str(k): int(v) for k, v in (context.get("tokens") or {}).items()},
            combine={str(k): [int(x) for x in v] for k, v in (item.get("combine") or {}).items()},
            drop_unknown=bool(context.get("drop_unknown", True)),
        )
    if not schemes:
        return "", {}
    active = os.environ.get("LPS_CLASS_SCHEME") or raw.get("active") or next(iter(schemes))
    if active not in schemes:
        active = next(iter(schemes))
    return active, schemes


ACTIVE_SCHEME_KEY, CLASS_SCHEMES = _load_class_schemes()
# 没有预设方案时给一个空壳，避免各处访问 None 崩掉；
# 实际训练/推理走 --data，类别来自数据集或权重。
ACTIVE_SCHEME: Optional[ClassScheme] = CLASS_SCHEMES.get(ACTIVE_SCHEME_KEY)


def set_scheme(key: str) -> ClassScheme:
    """运行时切换类别方案（界面下拉框会调用）。"""
    global ACTIVE_SCHEME_KEY, ACTIVE_SCHEME
    if key not in CLASS_SCHEMES:
        raise KeyError(f"未知类别方案 {key!r}，可选: {list(CLASS_SCHEMES) or '（未加载预设方案）'}")
        raise KeyError(f"未知类别方案 {key!r}")
    ACTIVE_SCHEME_KEY, ACTIVE_SCHEME = key, CLASS_SCHEMES[key]
    return ACTIVE_SCHEME


# ------------------------------------------------------------------ 训练


@dataclass
class TrainConfig:
    """训练超参数。字段与 ultralytics 的 train() 参数同名，便于直接透传。"""

    model: str = "yolov8n-posture.yaml"
    epochs: int = 120
    batch: int = 10
    imgsz: int = 640
    device: str = "0"
    # 0 表示主进程串行读图，GPU 会空转等待；原始框架用 8。
    # 本机取 2：实测 640 下每轮约 22 秒已接近上限，再加进程收益不大，
    # 且 Windows 下进程越多越容易因共享内存不足报错。
    workers: int = 2
    patience: int = 30
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    cos_lr: bool = True
    warmup_epochs: float = 3.0
    close_mosaic: int = 15
    # 数据增强
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 8.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    # 保存
    save_period: int = 10
    seed: int = 42

    def to_kwargs(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if k not in {"model"}
        }


# ------------------------------------------------------------------ 工具


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, RUNS_DIR, EXPORT_DIR, USER_CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def find_pretrained(filename: str = "yolov8n.pt") -> Optional[Path]:
    """在预设目录里找预训练权重，找不到返回 None（交给 ultralytics 自动下载）。"""
    for base in PRETRAINED_SEARCH_DIRS:
        candidate = base / filename
        if candidate.exists() and candidate.stat().st_size > 3_000_000:
            return candidate
    return None


def dataset_yaml_path() -> Path:
    """当前类别方案对应的 data.yaml。

    优先取可写目录里用户生成的那份（路径指向本机数据），
    没有时退回随程序分发的模板。
    """
    if not ACTIVE_SCHEME_KEY:
        return USER_CONFIG_DIR / "data_default.yaml"
    name = f"data_{ACTIVE_SCHEME_KEY}.yaml"
    user_copy = USER_CONFIG_DIR / name
    if user_copy.exists():
        return user_copy
    return CONFIG_DIR / name
