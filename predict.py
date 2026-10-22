# -*- coding: utf-8 -*-
"""推理入口（与老师框架同名）。

实际实现位于 scripts/predict.py。

用法::

    python predict.py --source path/to/image.jpg
    python predict.py --source path/to/video.mp4 --conf 0.3
    python predict.py --source 0 --max-frames 300
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    target = ROOT / "scripts" / "predict.py"
    if not target.exists():
        raise SystemExit(f"找不到推理脚本：{target}")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
