# -*- coding: utf-8 -*-
"""训练入口（与老师框架同名，命令行参数保持兼容）。

实际实现位于 scripts/train.py，这里只做转发，避免同一套逻辑维护两份。

用法与原始框架一致::

    python train.py --epochs 100 --batch 16 --device 0
    python train.py --scheme pose_view8 --epochs 120

相对原框架新增的参数：
    --scheme   类别方案（behavior4 / activity3 / pose_view8）
    --data     自定义数据配置，默认按方案自动选择
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    target = ROOT / "scripts" / "train.py"
    if not target.exists():
        raise SystemExit(f"找不到训练脚本：{target}")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
