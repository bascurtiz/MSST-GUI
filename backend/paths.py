"""
backend/paths.py
-----------------
Single source of truth for the repo root path.

Layout on disk:
  <root>/msst_gui/backend/paths.py    ← this file
  <root>/msst_gui/                     ← the Python package
  <root>/inference.py                  ← inference entry point
  <root>/msst_settings.json            ← user settings

Frozen (PyInstaller) layout — two roots:
  <exe dir>/                 ← writable app dir: settings, downloads, runtime
  <exe dir>/_internal/       ← read-only code root: inference.py, models/, configs/
"""
import os
import sys


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def get_app_dir():
    """Writable directory for user data (settings, downloads, runtime).

    * Normal: the repo root (same as code root).
    * Frozen: the directory containing the GUI executable.
    """
    if is_frozen():
        return os.path.dirname(sys.executable)
    return get_repo_root()


def get_repo_root():
    """Return the absolute path to the repository root.

    * Normal: two directories up from this file.
    * Frozen: the read-only code/data root PyInstaller unpacked to
      (``sys._MEIPASS``), where inference.py, models/ and configs/ live.
    """
    if is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_runtime_dir():
    """Directory of the bundled Python runtime used for inference jobs."""
    return os.path.join(get_app_dir(), "runtime")


def get_runtime_python():
    """Path to the runtime interpreter (does not imply it exists yet)."""
    return os.path.join(get_runtime_dir(), "python.exe")


def get_python_exe():
    """Return the Python interpreter to use for subprocess calls.

    Preference order:
      1. The bundled runtime interpreter (<app dir>/runtime/python.exe),
         which carries the GPU-appropriate PyTorch build installed on
         first use.
      2. When frozen without a runtime, ``"python"`` from PATH (legacy).
      3. Dev: the interpreter running the GUI.
    """
    if is_frozen():
        runtime = get_runtime_python()
        if os.path.isfile(runtime):
            return runtime
        return "python"
    return sys.executable


REPO_ROOT = get_repo_root()
APP_DIR = get_app_dir()

# Derived paths
INFERENCE_SCRIPT = os.path.join(REPO_ROOT, "inference.py")
ENSEMBLE_SCRIPT = os.path.join(REPO_ROOT, "ensemble.py")
SETTINGS_PATH = os.path.join(APP_DIR, "msst_settings.json")
