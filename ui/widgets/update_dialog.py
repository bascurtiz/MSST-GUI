"""
ui/widgets/update_dialog.py
---------------------------
Update checking for the frozen app, mirroring the STEM-organizer flow:
a background thread queries GitHub releases and, when a newer version is
available, a themed dialog offers to open the Releases page.

- run_startup_check(window): automatic silent check after launch (exe only).
- check_now(parent, on_status): manual check with a status callback
  (used by the Settings page button).
"""
import sys
import webbrowser

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

from backend import update_checker as uc
from ui.theme import theme_manager
from ui.widgets.common import run_blurred_dialog


class _CheckThread(QThread):
    done = Signal(bool, object)  # newer_available, latest tag (or None)

    def run(self):
        newer, tag = uc.check_for_update()
        self.done.emit(newer, tag)


class _UpdateDialog(QDialog):
    """Themed 'new version available' prompt."""

    def __init__(self, new_tag, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Available")
        self.setModal(True)
        self.setMinimumWidth(420)
        t = theme_manager.theme
        self.setStyleSheet(
            f"QDialog{{background:{t.bg};}}"
            f"QLabel{{color:{t.text};background:transparent;}}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)

        title = QLabel("Update Available")
        title.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:14px;font-weight:bold;")
        root.addWidget(title)

        msg = QLabel(
            f"A new version (<b style='color:{theme_manager.accent};'>{new_tag}</b>) "
            f"is available.<br>"
            f"Visit the Releases page on GitHub to download.")
        msg.setWordWrap(True)
        msg.setTextFormat(Qt.RichText)
        root.addWidget(msg)

        btns = QHBoxLayout()
        btns.addStretch()
        later = QPushButton("Later")
        later.setStyleSheet(
            f"QPushButton{{background:{t.surface};color:{t.text_dim};"
            f"border:1px solid {t.border_dim};border-radius:6px;"
            f"font-weight:600;padding:0 18px;min-height:32px;}}")
        later.clicked.connect(self.reject)
        btns.addWidget(later)
        dl = QPushButton("Download Update")
        dl.setStyleSheet(
            f"QPushButton{{background:{theme_manager.accent};color:#FFFFFF;"
            f"border:none;border-radius:6px;padding:0 18px;min-height:32px;"
            f"font-weight:600;}}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}")
        dl.clicked.connect(self._open_releases)
        btns.addWidget(dl)
        root.addLayout(btns)

    def _open_releases(self):
        try:
            webbrowser.open(uc.RELEASES_PAGE_URL, new=2)
        except Exception as exc:
            print(f"[Update Check] Failed to open browser: {exc}")
        self.accept()


def _show_dialog(new_tag, parent):
    dlg = _UpdateDialog(new_tag, parent)
    run_blurred_dialog(dlg)


def run_startup_check(window):
    """Automatic silent check ~2.5s after launch. Exe builds only — source
    checkouts are updated via git pull. Silent when up-to-date/offline."""
    if not getattr(sys, "frozen", False) and not uc.force_update_dialog():
        return

    def on_done(newer, tag):
        if newer and tag:
            print(f"[Update Check] New version found: {tag}")
            _show_dialog(tag, window)
        else:
            print("[Update Check] Application is up-to-date.")

    thread = _CheckThread(window)
    thread.done.connect(on_done)
    window._update_thread = thread  # keep a ref while it runs
    QTimer.singleShot(2500, thread.start)


def check_now(parent, on_status):
    """Manual check (Settings button). on_status(str) receives a short
    human-readable result; the update dialog is shown when newer."""
    on_status("Checking…")

    def on_done(newer, tag):
        if tag is None:
            on_status("Update check failed (offline?)")
            return
        if newer:
            on_status(f"New version {tag} available")
            _show_dialog(tag, parent)
        else:
            on_status("Up to date")

    thread = _CheckThread(parent)
    thread.done.connect(on_done)
    parent._update_check_thread = thread  # keep a ref while it runs
    thread.start()
