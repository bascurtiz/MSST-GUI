"""
MSST Modern GUI — entry point.
Run this from the root of the Music-Source-Separation-Training repo:
    python msst_gui/main.py
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

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import backend.settings as settings_store
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
