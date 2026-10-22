# -*- mode: python ; coding: utf-8 -*-
"""调试用 spec：保留控制台窗口，用于定位打包后启动失败的原因。"""
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).resolve()


def collect_dir(src: Path, dest: str):
    src = Path(src)
    if not src.exists():
        return []
    return [(str(src), dest)]


datas = []
datas += collect_dir(ROOT / "lps", "lps")
datas += collect_dir(ROOT / "scripts", "scripts")
datas += collect_dir(ROOT / "configs", "configs")
if (ROOT / "使用说明.md").exists():
    datas.append((str(ROOT / "使用说明.md"), "."))

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PyQt6.sip",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.figure",
        "ultralytics",
        "ultralytics.nn.modules",
        "torch",
        "torchvision",
        "cv2",
        "yaml",
        "PIL",
        "PIL.Image",
        "numpy",
        "pandas",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "unittest", "pdb", "tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="livestock_posture",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,                 # <-- 调试：保留控制台
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="livestock_posture",
)
