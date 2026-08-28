"""
MSST GUI — entry point.
Run this from the root of the Music-Source-Separation-Training repo:
    python msst_gui/main.py

Debugging the frozen exe:
    MSST-GUI.exe --console     shows prints in the launching terminal
    msst-gui.log (next to the exe) always captures output and tracebacks
"""
import sys
import os

if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS
    sys.path.insert(0, BASE)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if BASE not in sys.path:
        sys.path.insert(0, BASE)

# --- console + logging bootstrap (before any heavy imports) ---------------
_DEBUG_FLAGS = {"--console", "--debug"}
_LOG_MAX_BYTES = 2 * 1024 * 1024


def _wants_console():
    return any(a in _DEBUG_FLAGS for a in sys.argv[1:])


def _strip_flags(argv):
    return [a for a in argv if a not in _DEBUG_FLAGS]


def _attach_parent_console():
    """Print to the terminal the exe was launched from, when there is one."""
    if sys.platform != "win32":
        return
    import ctypes
    k32 = ctypes.windll.kernel32
    if k32.GetConsoleWindow():
        return  # already attached (pythonw builds never are)
    try:
        if not k32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            return
    except Exception:
        return
    try:
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except Exception:
        pass


class _Tee:
    """Fan stdout/stderr out to several streams (None entries ignored)."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _open_log_file(app_dir):
    """Append-mode log file with simple rotation (msst-gui.log -> .old)."""
    log_path = os.path.join(app_dir, "msst-gui.log")
    try:
        if os.path.isfile(log_path) and \
                os.path.getsize(log_path) > _LOG_MAX_BYTES:
            old = log_path + ".old"
            if os.path.isfile(old):
                os.remove(old)
            os.replace(log_path, old)
        return open(log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return None


from backend.paths import get_app_dir  # noqa: E402  (light: os/sys only)

_log_file = _open_log_file(get_app_dir())
if _wants_console():
    _attach_parent_console()

# Windowed PyInstaller builds start with stdout/stderr = None; tee whatever
# exists together with the log file so prints and tracebacks are never lost.
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
sys.argv = _strip_flags(sys.argv)


def _excepthook(tp, val, tb):
    import traceback
    traceback.print_exception(tp, val, tb)


sys.excepthook = _excepthook

from PySide6.QtCore import qInstallMessageHandler, QtMsgType  # noqa: E402


def _qt_message_handler(mode, _context, message):
    print(f"[Qt] {message}")


qInstallMessageHandler(_qt_message_handler)
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import backend.settings as settings_store
import backend.runtime_setup  # noqa: F401  (sets PYTHONNOUSERSITE early when frozen)
from ui.main_window import MainWindow
from ui.theme import theme_manager, apply_palette


def main():
    # Named mutex so installers/updaters (Inno Setup AppMutex) can detect a
    # running instance and ask the user to close it before upgrading.
    try:
        import ctypes
        ctypes.windll.kernel32.CreateMutexW(None, False, "MSST-GUI-Mutex")
    except Exception:
        pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("MSST")
    app.setOrganizationName("ZFTurbo")

    # Qt's ICO reader crashes on the bundled favicon.ico (OS/2-style DIB),
    # so use the pre-decoded PNG copy for the window/taskbar icon.
    icon_path = os.path.join(BASE, "resources", "app_icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    apply_palette(app)

    saved_theme = settings_store.load().get("ui", {}).get("theme", "dark")
    theme_manager.set_mode(saved_theme, persist=False)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
