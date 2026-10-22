"""字体自愈。

ultralytics 在绘制标签图与训练批次图时需要一个覆盖中文字形的字体，缺失时会
从 ultralytics.com 下载 `Arial.Unicode.ttf`（22 MB）。这带来两个现实问题：

    1. 网络慢时下载几分钟到几十分钟，训练迟迟不开始；
    2. 中途中断会留下**半截文件**，下次启动时 ultralytics 只检查文件存在就使用，
       FreeType 加载失败直接让进程崩溃（表现为 QProcess 报 Crashed、退出码异常）。

因此训练前主动检查该字体，损坏或缺失时用系统自带的中文字体顶替，
既不联网也不会崩。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

# 系统里常见的中文字体，按优先级尝试；必须是单体 TTF，ttc 集合部分库加载不了
_CJK_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\simfang.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

FONT_NAME = "Arial.Unicode.ttf"


def _font_dir() -> Path:
    """ultralytics 存放字体的配置目录。"""
    custom = os.environ.get("YOLO_CONFIG_DIR")
    if custom:
        return Path(custom)
    try:
        from ultralytics.utils import USER_CONFIG_DIR

        return Path(USER_CONFIG_DIR)
    except ImportError:
        return Path.home() / "AppData" / "Roaming" / "Ultralytics"


def _is_loadable(path: Path) -> bool:
    """判断字体能否被 FreeType 正常加载。

    只看文件大小不够：完整字体 22 MB，但半截文件也可能有几百 KB。
    """
    if not path.exists() or path.stat().st_size < 4096:
        return False
    try:
        from matplotlib.ft2font import FT2Font

        font = FT2Font(str(path))
        return font.num_glyphs > 0
    except (RuntimeError, OSError, ImportError):
        # 没有 matplotlib 时退化为大小判断
        return path.stat().st_size > 2_000_000


def ensure_ultralytics_font(verbose: bool = True) -> Optional[Path]:
    """确保 ultralytics 的中文字体可用，返回可用字体路径。

    返回 None 表示既没有现成可用字体、也没有系统字体可顶替，
    调用方应关闭绘图以避免崩溃。
    """
    target = _font_dir() / FONT_NAME
    if _is_loadable(target):
        return target

    # 损坏或缺失：先清掉半截文件，避免下次又被误用
    if target.exists():
        try:
            target.unlink()
            if verbose:
                print(f"[字体] 发现损坏的 {FONT_NAME}，已删除")
        except OSError:
            pass

    for candidate in _CJK_FONT_CANDIDATES:
        src = Path(candidate)
        if not src.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        except OSError:
            continue
        if _is_loadable(target):
            if verbose:
                print(f"[字体] 已用系统字体 {src.name} 顶替 {FONT_NAME}，无需联网下载")
            return target
        try:
            target.unlink()
        except OSError:
            pass

    if verbose:
        print("[字体] 未找到可用的中文字体，本次将关闭绘图以保证训练不中断")
    return None
