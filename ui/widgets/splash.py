"""
ui/widgets/splash.py
--------------------
Startup splash screen: the mvsep-logo.png logo with the "MSST GUI" title below it,
a status line naming the current startup phase, and a thin progress bar that
tracks it (interface build, settings load, runtime probe).

Driven from main(): MainWindow forwards its startup progress callbacks here.
`set_stage()` repaints the splash in place — it must not pump the event loop,
or combo popups and other tiny native windows appear beside it.
"""

from __future__ import annotations

import os
import sys
import ctypes
from ctypes import wintypes

from PySide6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QRectF,
)
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar,
)

from ui.theme import theme_manager

# Hide extra native windows that belong to this process while the splash is
# the only UI that should be on screen. System helper classes stay put.
_SWEEP_STRAYS = False
_SKIP_NATIVE_CLASSES = {
    "IME", "MSCTFIME UI", "GDI+ Hook Window Class",
    "CicMarshalWndClass", "OleMainThreadWndClass", "OleDdeWndClass",
    "Message", "STATIC", "ConsoleWindowClass",
}


def set_stray_sweep(active: bool):
    global _SWEEP_STRAYS
    _SWEEP_STRAYS = bool(active)


def hide_startup_strays(*keep_widgets):
    """SW_HIDE every visible top-level HWND of this PID except keep/console."""
    if not _SWEEP_STRAYS or sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    pid = os.getpid()
    keep = set()
    named = []
    try:
        app = QApplication.instance()
        if app is not None:
            named = [
                w for w in app.topLevelWidgets()
                if w.objectName() in {
                    "startupSplash", "mainWindow", "themeSwitchOverlay",
                }
            ]
    except Exception:
        named = []
    for keep_widget in list(keep_widgets) + named:
        if keep_widget is None:
            continue
        try:
            hwnd = int(keep_widget.internalWinId() or 0)
            if not hwnd:
                hwnd = int(keep_widget.winId())
            if hwnd:
                keep.add(hwnd)
        except Exception:
            pass
    try:
        console = int(kernel32.GetConsoleWindow() or 0)
        if console:
            keep.add(console)
    except Exception:
        pass

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lp):
        proc = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc))
        if proc.value != pid or int(hwnd) in keep:
            return True
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        if buf.value in _SKIP_NATIVE_CLASSES:
            return True
        user32.ShowWindow(int(hwnd), 0)
        return True

    try:
        user32.EnumWindows(_enum, 0)
    except Exception:
        pass



class SplashPanel(QWidget):
    """Frameless, always-on-top startup card."""

    _FLAGS = (
        Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        | Qt.NoDropShadowWindowHint | Qt.CustomizeWindowHint
    )

    def __init__(self, base_dir: str, version: str = "", parent=None):
        # No window flags in QWidget() — that creates and maps a tiny HWND
        # before WA_DontShowOnScreen can be set. Hide first, then apply flags.
        super().__init__(parent)
        self.setObjectName("startupSplash")
        self._base_dir = base_dir
        self._closing = False
        self.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowFlags(self._FLAGS)
        self.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(430, 430)

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 64, 40, 40)
        root.setSpacing(0)

        root.addStretch(1)

        self._logo = QLabel()
        self._logo.setAlignment(Qt.AlignCenter)
        self._logo.setStyleSheet("background:transparent;border:none;")
        logo_path = os.path.join(base_dir, "resources", "mvsep-logo.png")
        if os.path.isfile(logo_path):
            pm = QPixmap(logo_path)
            if not pm.isNull():
                # 1044x305 source -> display width, kept crisp.
                self._logo.setPixmap(pm.scaledToWidth(
                    260, Qt.SmoothTransformation))
        root.addWidget(self._logo)
        root.addSpacing(18)

        title = QLabel("MSST GUI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:27px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;border:none;"
            "letter-spacing:2px;"
        )
        root.addWidget(title)
        root.addSpacing(6)

        if version:
            ver = QLabel(f"v{version}")
            ver.setAlignment(Qt.AlignCenter)
            ver.setStyleSheet(
                "font-family:'Montserrat',sans-serif;font-size:10px;"
                f"color:{theme_manager.theme.text_muted};"
                "background:transparent;border:none;letter-spacing:1px;"
            )
            root.addWidget(ver)
        root.addSpacing(34)

        self._status = QLabel("Starting application...")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;"
            f"color:{theme_manager.theme.text_dim};"
            "background:transparent;border:none;"
        )
        root.addWidget(self._status)
        root.addSpacing(14)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(5)
        t = theme_manager.theme
        self._bar.setStyleSheet(
            f"QProgressBar {{ background: {t.border_visible}; border: none; "
            f"border-radius: 2px; max-height: 5px; }}"
            f"QProgressBar::chunk {{ background: {theme_manager.accent}; "
            f"border-radius: 2px; }}"
        )
        root.addWidget(self._bar)

        root.addStretch(2)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(420)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
        self._fade.finished.connect(self._on_fade_done)

        self._center_on_screen()

    def _native_hwnd(self):
        try:
            return int(self.internalWinId() or 0) or int(self.winId())
        except Exception:
            return 0

    def _apply_win32_frame(self):
        hwnd = self._native_hwnd()
        if not hwnd:
            return
        try:
            from backend.win_startup import keep_hwnd, strip_native_chrome
            keep_hwnd(hwnd)
            strip_native_chrome(hwnd)
        except Exception:
            pass

    def show(self):
        self._apply_win32_frame()
        self.setAttribute(Qt.WA_DontShowOnScreen, False)
        super().show()
        self._apply_win32_frame()

    # ── placement ────────────────────────────────────────────────────────────
    def _center_on_screen(self):
        screen = None
        try:
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
        except Exception:
            screen = None
        if screen is None:
            return
        avail = screen.availableGeometry()
        self.move(
            avail.x() + (avail.width() - self.width()) // 2,
            avail.y() + (avail.height() - self.height()) // 2,
        )

    # ── progress API ─────────────────────────────────────────────────────────
    def set_stage(self, message: str, percent: int):
        """Advance the splash: update status text + bar and paint NOW.

        Does not pump the event loop — processEvents() during page
        construction maps combo popups and other tiny native windows.
        """
        if self._closing:
            return
        pct = max(0, min(100, int(percent)))
        self._status.setText(message)
        self._bar.setValue(pct)
        if self.isVisible():
            self.repaint()
        self._apply_win32_frame()
        hide_startup_strays(self)

    def finish(self, message: str = "Ready"):
        """Mark 100% and drop the overlay immediately.

        A delayed fade needs the event loop. The first exec() pass is often
        busy attaching mvsep scores to the Model Library, so a 650 ms timer
        can sit behind tens of seconds of UI work while this card still says
        Ready on top of a finished window.
        """
        if self._closing or not self.isVisible():
            return
        self._status.setText(message)
        self._bar.setValue(100)
        self.repaint()
        self.close_now()

    def close_now(self):
        """Immediate teardown (used if window construction fails)."""
        self._closing = True
        self._fade.stop()
        self.close()
        self.deleteLater()

    def _on_fade_done(self):
        self.close()
        self.deleteLater()

    # ── painting ─────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        with QPainter(self) as p:
            p.setRenderHint(QPainter.Antialiasing, True)
            rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
            path = _rounded_path(rect, 18.0)
            p.fillPath(path, QColor(theme_manager.theme.bg))
            p.setPen(QPen(QColor(theme_manager.theme.border_dim), 1.0))
            p.drawPath(path)
        super().paintEvent(event)


def _rounded_path(rect: QRectF, radius: float):
    from PySide6.QtGui import QPainterPath
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path
