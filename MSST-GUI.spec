# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the MSST GUI.
#
# Produces dist/MSST-GUI/ — a onedir bundle whose exe covers the GUI, while
# separation and training jobs run under the bundled Python runtime (runtime_pristine is
# copied to the writable app dir on first use and the GPU-appropriate
# PyTorch build is installed into it — see backend/runtime_setup.py).
#
# Build:  pyinstaller MSST-GUI.spec --noconfirm

import os

block_cipher = None
ROOT = os.path.abspath(SPECPATH)

CODE_EXTS = {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".sh"}


def walk_datas(src_root, dest_root, exts=None):
    """Collect files under src_root. exts=None takes every file; otherwise a
    whitelist of extensions (used for code trees so checkpoints and other
    heavy user artifacts are never bundled)."""
    pairs = []
    for root, _dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        dest = dest_root if rel == "." else os.path.join(dest_root, rel)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if exts is None or ext in exts:
                pairs.append((os.path.join(root, f), dest))
    return pairs


datas = []
for f in ("inference.py", "ensemble.py", "train.py", "valid.py",
          "train_accelerate.py", "valid_ddp.py",
          "requirements-runtime.txt"):
    datas.append((os.path.join(ROOT, f), "."))

datas += walk_datas(os.path.join(ROOT, "resources"), "resources")            # all: fonts, icons, qss
datas += walk_datas(os.path.join(ROOT, "configs"), "configs")                # all: model yamls
datas += walk_datas(os.path.join(ROOT, "utils"), "utils", CODE_EXTS)
datas += walk_datas(os.path.join(ROOT, "models"), "models", CODE_EXTS)       # code only — never checkpoints
# engine-side helpers imported by inference.py (runs from data files, not the PYZ)
datas.append((os.path.join(ROOT, "backend", "__init__.py"), "backend"))
datas.append((os.path.join(ROOT, "backend", "audio_names.py"), "backend"))
datas += walk_datas(os.path.join(ROOT, "build", "runtime_pristine"),
                    "runtime_pristine")                                      # all: embedded python

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "torchaudio",   # never in the GUI process
        "tkinter", "pytest", "IPython", "jupyter",
        "PyQt5", "PyQt6", "PySide2",            # the app uses PySide6 only
        # urllib3 does `from backports import zstd`; PyInstaller can pull a
        # broken `backports.zstd`/`zstd` stub (a lone `_zstd.pyd`) into the
        # code root. Excluding them forces the ImportError -> HAS_ZSTD=False
        # path (matching a dev venv, which has no backports), so the GUI never
        # crashes on urllib3's `zstd.ZstdError`. The inference subprocess
        # cleans these strays itself (backend/runtime_setup.py).
        "backports", "zstd",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MSST-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    # Windowed build: double-clicking the exe opens no terminal window.
    # Nothing is lost — main.py tees stdout/stderr into msst-gui.log (in the
    # app-data dir), installs a Qt message handler that logs Qt warnings and
    # qFatal aborts, and dumps all-thread stacks via faulthandler before a
    # hard crash. For live console output run:  MSST-GUI.exe --console
    # (re-attaches to the terminal the exe was launched from).
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(ROOT, "resources", "app_icon.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MSST-GUI",
)
