"""
ui/widgets/splash.py
--------------------
Startup splash screen: the mvsep-logo.png logo with the "MSST GUI" title below it,
a status line naming the current startup phase, and a thin progress bar that
tracks it (interface build, settings load, runtime probe).

Driven from main(): MainWindow forwards its startup progress callbacks here.
`set_stage()` pumps the event loop so each step actually repaints while the
UI thread is busy constructing the pages synchronously.
"""

from __future__ import annotations

import os

from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRectF,
)
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QApplication,
)

from ui.theme import theme_manager


class SplashPanel(QWidget):
    """Frameless, always-on-top startup card."""

    def __init__(self, base_dir: str, version: str = "", parent=None):
        super().__init__(parent)
        self._base_dir = base_dir
        self._closing = False

        self.setFixedSize(430, 430)
        self.setWindowFlags(
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

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
        self._bar.setStyleSheet(
            f"QProgressBar{{background:{theme_manager.theme.border};"
            "border:none;border-radius:2px;}"
            f"QProgressBar::chunk{{background:{theme_manager.accent};"
            "border-radius:2px;}}"
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
        """Advance the splash: update status text + bar and repaint NOW.

        processEvents() here is what keeps the bar live while startup work
        runs synchronously on the UI thread (page construction, settings
        load). Clamps to 0-100 and is ignored once closing has begun.
        """
        if self._closing or not self.isVisible():
            return
        pct = max(0, min(100, int(percent)))
        self._status.setText(message)
        self._bar.setValue(pct)
        QApplication.processEvents()

    def finish(self, message: str = "Ready"):
        """Show 100% briefly, then fade out and release the widget."""
        if self._closing or not self.isVisible():
            return
        self._closing = True
        self._status.setText(message)
        self._bar.setValue(100)
        QApplication.processEvents()
        QTimer.singleShot(650, self._fade.start)

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
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        path = _rounded_path(rect, 18.0)
        p.fillPath(path, QColor(theme_manager.theme.bg))
        p.setPen(QPen(QColor(theme_manager.theme.border_dim), 1.0))
        p.drawPath(path)


def _rounded_path(rect: QRectF, radius: float):
    from PySide6.QtGui import QPainterPath
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path
