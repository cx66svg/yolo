# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for livestock_posture GUI application.

Single-directory mode (onedir) is used because:
- Startup is much faster than one-file mode.
- Missing dependencies are easier to diagnose.
- The bundled scripts/ and configs/ are visible beside the executable.
"""
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH).resolve()


def collect_dir(src: Path, dest: str):
    """Return a PyInstaller datas tuple for a directory tree."""
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
        # PyQt6
        "PyQt6.sip",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        # Matplotlib Qt backends used by the live charts
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.figure",
        # Core inference / training stack
        "ultralytics",
        "ultralytics.nn.modules",
        "ultralytics.yolo",
        "torch",
        "torchvision",
        "cv2",
        # Common data libs
        "yaml",
        "PIL",
        "PIL.Image",
        "numpy",
        "pandas",
        # torch.distributed 动态依赖（PyInstaller 静态分析检测不到）
        "pdb",
        "unittest",
        "unittest.mock",
        "inspect",
        "linecache",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 仅排除明显不需要的 GUI 工具包；pdb/unittest 被 torch 等库隐式依赖，
        # 排除后会导致启动期 ModuleNotFoundError，且错误信息在无控制台程序里不可见。
        "tkinter",
    ],
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
    console=False,
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
