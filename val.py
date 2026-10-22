# -*- coding: utf-8 -*-
"""验证入口（与老师框架同名）。

实际实现位于 scripts/val.py。

用法::

    python val.py                                  # 默认取最新 best.pt
    python val.py --split test --scheme pose_view8
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    target = ROOT / "scripts" / "val.py"
    if not target.exists():
        raise SystemExit(f"找不到验证脚本：{target}")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
