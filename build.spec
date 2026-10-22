# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

要点：
  1. 采用 onedir（文件夹）模式而非 onefile。onefile 每次启动都要把
     两三个 GB 解压到临时目录，冷启动几十秒，实际不可用。
  2. ultralytics 大量使用动态导入与运行时读取 yaml，必须整包收集，
     否则会出现"训练能跑、保存模型时报缺模块"这类隐蔽故障。
  3. scripts/ 不是 Python 包，是被 importlib 按文件路径加载的，
     必须作为数据文件一起打进去，否则打包后训练按钮失效。
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# ultralytics 与 lps 全量收集（含 yaml、字体等非代码资源）
for pkg in ("ultralytics", "lps"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 配置文件与任务脚本（运行时按路径读取）
datas += [("configs", "configs"), ("scripts", "scripts")]

# 预训练权重如果存在就带上
weights = Path("weights")
if weights.exists():
    datas += [(str(weights), "weights")]

hiddenimports += collect_submodules("matplotlib.backends")
hiddenimports += [
    "ultralytics.models.yolo.detect",
    "ultralytics.models.yolo.detect.train",
    "ultralytics.models.yolo.detect.val",
    "ultralytics.models.yolo.detect.predict",
    "ultralytics.nn.modules",
    "ultralytics.nn.tasks",
    "ultralytics.utils.plotting",
    "yaml",
    "cv2",
]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "IPython", "notebook", "jupyter", "pytest",
        "PyQt5", "PySide2", "PySide6", "torch.test", "tensorboard",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DetectStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX 压缩常导致 torch 的 dll 加载失败，关闭
    console=False,          # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DetectStudio",
)
