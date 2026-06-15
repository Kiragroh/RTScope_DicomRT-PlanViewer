# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()

datas = [
    (str(project_root / "logo.png"), "."),
    (
        str(project_root / "planeval_viewer" / "refdb" / "offline_examples" / "stereotaxie_tables.json"),
        "planeval_viewer/refdb/offline_examples",
    ),
]

hiddenimports = [
    "OpenGL.GL",
    "OpenGL.GLU",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "pyqtgraph.opengl",
]

excludes = [
    "IPython",
    "matplotlib",
    "PyQt5",
    "PyQt6",
    "llvmlite",
    "numba",
    "openpyxl",
    "pandas",
    "pyarrow",
    "pytest",
    "tkinter",
]

a = Analysis(
    ["app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RTScopePlanEvalViewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "resources" / "rtscope.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RTScopePlanEvalViewer",
)
