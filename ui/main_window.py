"""
ui/main_window.py
Main window with native frame, header bar, animated nav tabs, and page stack.
All colors sourced from the active theme (ui.theme.theme_manager).
"""
import math
import os
import time
import ctypes
from ctypes import wintypes

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QSizePolicy, QStackedWidget, QGraphicsBlurEffect,
)
from PySide6.QtCore import Qt, QSize, QTimer, Signal, QVariantAnimation, QRectF, QPoint, QPointF, QEasingCurve, QCoreApplication, QEvent
from PySide6.QtGui import QColor, QPalette, QPainter, QBrush, QFont, QFontMetrics, QPen, QLinearGradient, QRadialGradient, QPixmap, QPainterPath

import backend.settings as settings_store
from ui.pages.inference_page import InferencePage
from ui.pages.ensemble_page import EnsembleLandingPage
from ui.pages.auto_ensemble_page import AutoEnsemblePage
from ui.pages.manual_ensemble_page import ManualEnsemblePage
from ui.pages.iterative_ensemble_page import IterativeEnsemblePage
from ui.pages.console_page import ConsolePage
from ui.pages.settings_page import SettingsPage, orphan_fetch_threads
from ui.widgets.ckpt_settings_dialog import CkptSettingsDialog
from ui.widgets.iterative_warning_dialog import IterativeWarningDialog
from ui.widgets.model_installer_dialog import _ModelInstallerDialog
from backend.model_installer import check_models, REQUIRED_MODELS, ModelInstaller
from ui.theme import theme_manager

_LOGO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "mvsep.png",
)


class _NavTab(QFrame):
    clicked = Signal(int)

    IDLE = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3

    def __init__(self, text, index, parent=None):
        super().__init__(parent)
        self._index = index
        self._active = False
        self._hovered = False
        self._attention_state = self.IDLE
        self._glow_intensity = 0.0
        self._sweep_pos = -0.3
        self._dot_pulse = 0.0
        self._fade_out = 0.0
        self._fading_out = False

        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background:transparent;border:none;")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 0, 20, 0)
        root.setSpacing(0)

        self._label = QLabel(text)
        self._label.setAlignment(Qt.AlignCenter)
        root.addWidget(self._label, 1)
        root.addSpacing(6)

        self._anim_timer = QTimer(self)
        self._anim_timer.setTimerType(Qt.PreciseTimer)
        self._anim_timer.timeout.connect(self._tick)

        self._apply_style()

    @property
    def indicator_width(self):
        fm = self._label.fontMetrics()
        n = len(self._label.text())
        letter_extra = 3 * (n - 1) if n > 1 else 0
        return fm.horizontalAdvance(self._label.text()) + letter_extra + 12

    def set_attention_state(self, state):
        if self._attention_state == state:
            return
        if self._attention_state == self.RUNNING and state == self.IDLE:
            self._fading_out = True
            self._fade_out = 1.0
            return
        self._fading_out = False
        self._attention_state = state
        if state == self.RUNNING:
            self._glow_intensity = 0.0
            self._sweep_pos = -0.3
            self._dot_pulse = 0.0
            self._anim_timer.start(16)
        elif state in (self.COMPLETED, self.FAILED):
            self._anim_timer.stop()
        self.update()

    def _tick(self):
        if self._fading_out:
            self._fade_out -= 0.04
            if self._fade_out <= 0:
                self._fade_out = 0
                self._fading_out = False
                self._attention_state = self.IDLE
                self._anim_timer.stop()
            self.update()
            return
        self._glow_intensity = 0.3 + 0.2 * math.sin(self._dot_pulse * 3.0)
        self._sweep_pos += 0.015
        if self._sweep_pos > 1.3:
            self._sweep_pos = -0.3
        self._dot_pulse += 0.05
        self.update()

    def paintEvent(self, event):
        if self._attention_state != self.IDLE or self._fading_out:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            intensity = self._fade_out if self._fading_out else 1.0
            if self._attention_state == self.FAILED:
                clr = QColor(theme_manager.theme.error)
            else:
                clr = QColor(theme_manager.accent)

            glow = self._glow_intensity if self._attention_state == self.RUNNING else 1.0
            center = self.rect().center()
            radius = max(self.width(), self.height()) * 0.55
            rg = QRadialGradient(center, radius)
            c0 = QColor(clr)
            c0.setAlpha(int(35 * glow * intensity))
            c1 = QColor(clr)
            c1.setAlpha(0)
            rg.setColorAt(0.0, c0)
            rg.setColorAt(0.5, c0)
            rg.setColorAt(1.0, c1)
            painter.fillRect(self.rect(), QBrush(rg))

            if self._attention_state == self.RUNNING:
                sweep_w = self.width() * 0.35
                sweep_cx = self._sweep_pos * self.width()
                lg = QLinearGradient(sweep_cx - sweep_w * 0.5, 0, sweep_cx + sweep_w * 0.5, 0)
                lg0 = QColor(clr)
                lg0.setAlpha(0)
                lg1 = QColor(clr)
                lg1.setAlpha(int(10 * intensity))
                lg.setColorAt(0.0, lg0)
                lg.setColorAt(0.5, lg1)
                lg.setColorAt(1.0, lg0)
                painter.fillRect(self.rect(), QBrush(lg))

                dot_r = 4
                dot_x = self.width() - 14
                dot_y = 8
                dc = QColor(theme_manager.accent)
                dc.setAlpha(int(160 + 95 * math.sin(self._dot_pulse)))
                painter.setBrush(dc)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)

            painter.end()
        super().paintEvent(event)

    def _apply_style(self):
        t = theme_manager.theme
        if self._active:
            color = t.text
        elif self._hovered:
            color = t.text_sec
        else:
            color = t.text_dim
        self._label.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:700;"
            f"letter-spacing:3px;color:{color};background:transparent;border:none;"
            "padding:0 2px;"
        )

    def enterEvent(self, event):
        self._hovered = True
        if not self._active:
            self._apply_style()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        if not self._active:
            self._apply_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def set_active(self, active):
        self._active = active
        self._apply_style()


class _NavIndicator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(2)
        self._update_color()
        self._anim = QVariantAnimation()
        self._anim.valueChanged.connect(self._on_step)
        self._anim.setDuration(400)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._from_cx = 0
        self._from_w = 0
        self._to_cx = 0
        self._to_w = 0
        self._y = 0

    def _update_color(self):
        self.setStyleSheet(f"background:{theme_manager.accent};border:none;border-radius:1px;")

    def move_to(self, tab):
        header = self.parent()
        pos = tab.mapTo(header, QPoint(0, 0))
        tw = tab.indicator_width
        self._y = header.height() - 6
        current = self.geometry()
        self._from_cx = current.center().x() if current.width() > 0 else pos.x() + tab.width() // 2
        self._from_w = current.width() if current.width() > 0 else tw
        self._to_cx = pos.x() + tab.width() // 2
        self._to_w = tw
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def snap_to(self, tab):
        header = self.parent()
        pos = tab.mapTo(header, QPoint(0, 0))
        tw = tab.indicator_width
        cx = pos.x() + tab.width() // 2
        y = header.height() - 6
        self.setGeometry(int(cx - tw // 2), y, int(tw), 2)
        self._from_cx = cx
        self._from_w = tw
        self._to_cx = cx
        self._to_w = tw

    def _on_step(self, t):
        cx = self._from_cx + (self._to_cx - self._from_cx) * t
        w = self._from_w + (self._to_w - self._from_w) * t
        stretch = min(self._from_w, self._to_w) * 0.15 * math.sin(t * math.pi)
        w += stretch
        self.setGeometry(int(cx - w // 2), self._y, int(w), 2)


class _WindowButton(QWidget):
    clicked = Signal()

    def __init__(self, btn_type, parent=None):
        super().__init__(parent)
        self._type = btn_type
        self._hovered = False
        self._pressed = False
        self._maximized = False
        self.setFixedSize(44, 32)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._pressed = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        was_pressed = self._pressed
        self._pressed = False
        self.update()
        if event.button() == Qt.LeftButton and was_pressed:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = theme_manager.theme
        dark = theme_manager.mode == "dark"

        # Windows-style hover / pressed fill
        if self._pressed:
            if self._type == "close":
                bg = QColor("#A0180D" if dark else "#C42B1C")
            else:
                bg = QColor("#2D2D2D" if dark else QColor(0, 0, 0, 35))
            p.fillRect(self.rect(), bg)
        elif self._hovered:
            if self._type == "close":
                bg = QColor("#C42B1C" if dark else "#E81123")
            else:
                bg = QColor("#3C3C3C" if dark else QColor(0, 0, 0, 22))
            p.fillRect(self.rect(), bg)

        if self._type == "close" and (self._hovered or self._pressed):
            pen_color = QColor("#FFFFFF")
        else:
            pen_color = QColor(t.text)
        p.setPen(QPen(pen_color, 1.2))
        p.setBrush(Qt.NoBrush)

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        if self._type == "minimize":
            p.drawLine(QPointF(cx - 5.5, cy + 1.5), QPointF(cx + 5.5, cy + 1.5))
        elif self._type == "maximize":
            if self._maximized:
                # restore: two overlapping squares
                p.drawRect(QRectF(cx - 6.0, cy - 4.0, 7.5, 7.5))
                p.drawRect(QRectF(cx - 2.5, cy - 3.5, 7.5, 7.5))
            else:
                p.drawRect(QRectF(cx - 5.5, cy - 5.5, 11, 11))
        else:  # close
            p.drawLine(QPointF(cx - 4.5, cy - 4.5), QPointF(cx + 4.5, cy + 4.5))
            p.drawLine(QPointF(cx - 4.5, cy + 4.5), QPointF(cx + 4.5, cy - 4.5))
        p.end()


class _WindowButtons(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._maximized = False
        self._saved_geometry = None
        self.setStyleSheet("background:transparent;border:none;")
        self._setup()

    def _setup(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._min_btn = _WindowButton("minimize")
        self._min_btn.clicked.connect(lambda: self.window().showMinimized())
        self._max_btn = _WindowButton("maximize")
        self._max_btn.clicked.connect(self._toggle_maximize)
        self._close_btn = _WindowButton("close")
        self._close_btn.clicked.connect(lambda: self.window().close())

        layout.addWidget(self._min_btn)
        layout.addWidget(self._max_btn)
        layout.addWidget(self._close_btn)

    def _toggle_maximize(self):
        w = self.window()
        if self._maximized:
            if self._saved_geometry:
                w.setGeometry(self._saved_geometry)
            self._maximized = False
            self._max_btn._maximized = False
            if hasattr(w, '_set_dwm_corners'):
                w._set_dwm_corners(True)
        else:
            self._saved_geometry = w.geometry()
            screen = w.screen()
            if screen:
                w.setGeometry(screen.availableGeometry())
            self._maximized = True
            self._max_btn._maximized = True
            if hasattr(w, '_set_dwm_corners'):
                w._set_dwm_corners(False)
        self._max_btn.update()


class _ThemeToggle(QFrame):
    """Segmented sun/moon theme switch: left half = light, right half = dark.
    The selected half is filled with a soft blue; clicking a half switches mode."""

    _SUN = "sun"
    _MOON = "moon"
    _REST_OPACITY = 0.6   # dimmed when idle; full opacity returns on hover

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(64, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Switch light / dark theme")
        self._hovered = None
        self._pressed = None
        self._hov = 0.0   # 0 = idle dimmed, 1 = hovered (full opacity)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(180)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
        self._fade.valueChanged.connect(self._on_fade)
        self._sel_t = 0.0 if theme_manager.mode == "light" else 1.0
        theme_manager.theme_changed.connect(self._slide)

    def _slide(self):
        target = 0.0 if theme_manager.mode == "light" else 1.0
        self._anim.stop()
        self._anim.setStartValue(self._sel_t)
        self._anim.setEndValue(target)
        self._anim.start()

    def _on_anim(self, value):
        self._sel_t = float(value)
        self.update()

    def _on_fade(self, value):
        self._hov = float(value)
        self.update()

    def _set_hover(self, on):
        self._fade.stop()
        self._fade.setStartValue(self._hov)
        self._fade.setEndValue(1.0 if on else 0.0)
        self._fade.start()

    def _half_at(self, pos):
        return self._SUN if pos.x() < self.width() / 2.0 else self._MOON

    def enterEvent(self, event):
        self._hovered = self._half_at(event.position())
        self._set_hover(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = None
        self._pressed = None
        self._set_hover(False)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        self._hovered = self._half_at(event.position())
        self._set_hover(True)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pressed = self._half_at(event.position())
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._pressed:
            half = self._half_at(event.position())
            if half == self._pressed:
                target = "light" if half == self._SUN else "dark"
                if theme_manager.mode != target:
                    theme_manager.set_mode(target)
        self._pressed = None
        self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = theme_manager.theme
        dark = theme_manager.mode == "dark"

        # Dim the whole switch when idle; ease back to full opacity on hover.
        p.setOpacity(self._REST_OPACITY
                     + (1.0 - self._REST_OPACITY) * self._hov)

        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = r.height() / 2.0
        hw = self.width() / 2.0

        # Clip to the pill so fills and icons stay inside the rounded shape
        pill = QPainterPath()
        pill.addRoundedRect(r, radius, radius)
        p.save()
        p.setClipPath(pill)

        # Hover tint on the hovered half
        if self._hovered:
            hx = 0.0 if self._hovered == self._SUN else hw
            p.fillRect(QRectF(hx, 0, hw, self.height()),
                       QColor(255, 255, 255, 40) if not dark else QColor(255, 255, 255, 24))

        # Selected fill, sliding between the two halves
        sel = QColor("#D6E7FF" if not dark else "#2B4259")
        p.fillRect(QRectF(self._sel_t * hw, 0, hw, self.height()), sel)
        p.restore()

        # Border
        p.setPen(QPen(QColor("#C7CDD6" if not dark else "#3A4350"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r, radius, radius)

        # Faint divider between the halves
        p.setPen(QPen(QColor(0, 0, 0, 26) if not dark else QColor(255, 255, 255, 26), 1))
        p.drawLine(QPointF(hw, 4), QPointF(hw, self.height() - 4))

        icon = QColor("#1F3047" if not dark else "#DDE3EA")
        cy = self.height() / 2.0
        self._draw_sun(p, QPointF(hw / 2.0, cy), icon)
        self._draw_moon(p, QPointF(hw + hw / 2.0, cy), icon)
        p.end()

    def _draw_sun(self, p, c, color):
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(c.x(), c.y()), 3.1, 3.1)
        p.setPen(QPen(color, 1.3))
        for a in range(0, 360, 45):
            rad = math.radians(a)
            p.drawLine(
                QPointF(c.x() + math.cos(rad) * 5.0, c.y() + math.sin(rad) * 5.0),
                QPointF(c.x() + math.cos(rad) * 6.5, c.y() + math.sin(rad) * 6.5),
            )

    def _draw_moon(self, p, c, color):
        outer = QPainterPath()
        outer.addEllipse(QPointF(c.x(), c.y()), 4.2, 4.2)
        inner = QPainterPath()
        inner.addEllipse(QPointF(c.x() + 2.3, c.y() - 1.3), 3.8, 3.8)
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawPath(outer.subtracted(inner))


class _ThemeSwitchOverlay(QWidget):
    """Opaque full-window cover shown while pages are rebuilt for a theme
    switch. Painted in the NEW theme so the app reads as already switched.
    Shows a progress bar: the rebuild runs in chunks, so a determinate bar
    (advancing per completed chunk) reads smooth, unlike a spinner that can
    only repaint between chunks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0

    def set_progress(self, fraction):
        self._progress = max(0.0, min(1.0, float(fraction)))
        self.update()

    def paintEvent(self, event):
        t = theme_manager.theme
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(t.bg))
        # Centered pill: caption + progress bar, all from the new palette.
        font = QFont(self.font())
        font.setFamily("Montserrat")
        font.setPointSizeF(8.5)
        font.setBold(True)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
        p.setFont(font)
        text = "SWITCHING THEME…"
        metrics = p.fontMetrics()
        text_w = metrics.horizontalAdvance(text)
        pill_w = text_w + 48
        pill_h = 62
        cx = self.width() / 2
        cy = self.height() / 2
        pill = QRectF(cx - pill_w / 2, cy - pill_h / 2, pill_w, pill_h)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(t.border_visible), 1))
        p.setBrush(QColor(t.surface))
        p.drawRoundedRect(pill, 12, 12)
        p.setPen(QPen(QColor(t.text), 1))
        text_rect = pill.adjusted(24, -10, -24, 12)
        p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)
        # Thin determinate progress bar along the bottom of the pill.
        track = QRectF(pill.left() + 16, pill.bottom() - 20,
                       pill.width() - 32, 3)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.border))
        p.drawRoundedRect(track, 1.5, 1.5)
        if self._progress > 0:
            fill = QRectF(track.left(), track.top(),
                          max(track.width() * self._progress, 6),
                          track.height())
            p.setBrush(QColor(theme_manager.accent))
            p.drawRoundedRect(fill, 1.5, 1.5)
        p.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


class _HeaderBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self._dragging = False
        self._drag_pos = None
        self._active_tab = None
        self._update_bg()

    def _update_bg(self):
        bg = theme_manager.theme.header_bg
        self.setStyleSheet(f"QFrame {{ background: {bg}; border: none; }}")

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme_manager.theme.header_bg))
        p.end()
        super().paintEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._active_tab:
            indicator = self.findChild(_NavIndicator)
            if indicator:
                indicator.snap_to(self._active_tab)

    def _is_maximized(self):
        w = self.window()
        btn = getattr(w, '_window_buttons', None)
        return btn is not None and btn._maximized

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._is_maximized():
            child = self.childAt(event.pos())
            if isinstance(child, QLabel) or child is None:
                self._dragging = True
                self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and not self._is_maximized():
            w = self.window()
            w.move(w.pos() + event.globalPosition().toPoint() - self._drag_pos)
            self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MUSIC SOURCE SEPARATION — MODERN GUI")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint)
        self.setMinimumSize(QSize(1100, 700))

        self._central = QWidget()
        cv = QVBoxLayout(self._central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        self._header = self._build_header()
        cv.addWidget(self._header)

        self._center_on_screen()

        self._processing = False
        # Set when the theme switched mid-run; honoured once the run ends.
        self._theme_rebuild_pending = False
        # Theme-switch orchestration: pages are stripped before the global
        # restyle, then rebuilt in chunks behind the switch overlay.
        self._switching = False
        self._switch_queued = False
        self._stripping = False
        self._pages_stripped = False
        self._switch_overlay = None
        self._stack = QStackedWidget()
        self.ensemble_stack = QStackedWidget()
        self._create_pages()

        self._stack.addWidget(self.inference_page)       # 0
        self._stack.addWidget(self.ensemble_stack)       # 1
        self._stack.addWidget(self.console_page)         # 2
        self._stack.addWidget(self.settings_page)        # 3
        cv.addWidget(self._stack, 1)
        self.setCentralWidget(self._central)
        self._apply_theme_bgs()

        # The blur effect is attached to the central widget ONLY while a
        # blur animation is running. Leaving it attached at radius 0 forces
        # every repaint through the effect pixmap path and floods the
        # console with 'Painter not active' warnings during inference.
        self._blur_effect = None
        self._blur_timer = QTimer()
        self._blur_timer.setSingleShot(True)
        self._blur_timer.timeout.connect(self._blur_tick)
        self._blur_target = 0
        self._blur_current = 0
        self._blur_step = 0
        self._blur_callback = None

        self._load_all()
        theme_manager.theme_changed.connect(self._on_theme_changed)
        theme_manager.theme_about_to_change.connect(self._on_theme_about_to_change)
        if self._NATIVE_RESIZE:
            self._enable_native_resize()
        # Update check for exe builds, ~2.5s after launch (silent if current).
        from ui.widgets.update_dialog import run_startup_check
        QTimer.singleShot(2500, lambda: run_startup_check(self))

    # —— Theme switching ————————————————————————————————————————————————
    def _create_pages(self):
        self.inference_page = InferencePage()
        self.ensemble_landing = EnsembleLandingPage()
        self.auto_ensemble = AutoEnsemblePage()
        self.manual_ensemble = ManualEnsemblePage()
        self.iterative_ensemble = IterativeEnsemblePage()
        self.console_page = ConsolePage()
        self.settings_page = SettingsPage()

        self.ensemble_stack.addWidget(self.ensemble_landing)
        self.ensemble_stack.addWidget(self.auto_ensemble)
        self.ensemble_stack.addWidget(self.manual_ensemble)
        self.ensemble_stack.addWidget(self.iterative_ensemble)
        self._wire_pages()

    def _wire_pages(self):
        self.settings_page.model_registered.connect(
            self.inference_page.on_model_registered
        )
        self.settings_page.model_registered.connect(
            self.auto_ensemble.on_model_registered
        )
        self.settings_page.model_registered.connect(
            self.iterative_ensemble.on_model_registered
        )
        self.settings_page.model_removed.connect(
            self.inference_page.on_model_removed
        )
        # Console Stop kills the active inference job (same path as the
        # INFERENCE page's own stop button).
        self.console_page.stop_requested.connect(
            self.inference_page._stop
        )
        self.settings_page.model_removed.connect(
            self.auto_ensemble.on_model_removed
        )
        self.settings_page.model_removed.connect(
            self.iterative_ensemble.on_model_removed
        )
        self.inference_page.log_output.connect(
            self.console_page.append_log
        )
        self.inference_page.input_files_submitted.connect(
            self.console_page.set_input_files
        )
        self.inference_page.model_selected.connect(
            self.console_page.set_current_model
        )
        self.manual_ensemble.log_output.connect(
            self.console_page.append_log
        )
        self.auto_ensemble.log_output.connect(
            self.console_page.append_log
        )
        self.iterative_ensemble.log_output.connect(
            self.console_page.append_log
        )
        self.iterative_ensemble.input_files_submitted.connect(
            self.console_page.set_input_files
        )
        self.settings_page.settings_changed.connect(
            self._save_all
        )
        self.inference_page.ckpt_settings_requested.connect(
            self._show_ckpt_settings
        )
        self.inference_page.add_model_requested.connect(
            lambda: self._switch(3)
        )
        self.ensemble_landing.auto_selected.connect(
            lambda: self.ensemble_stack.setCurrentWidget(self.auto_ensemble)
        )
        self.ensemble_landing.manual_selected.connect(
            lambda: self.ensemble_stack.setCurrentWidget(self.manual_ensemble)
        )
        self.auto_ensemble.navigate_back.connect(
            lambda: self.ensemble_stack.setCurrentWidget(self.ensemble_landing)
        )
        self.manual_ensemble.navigate_back.connect(
            lambda: self.ensemble_stack.setCurrentWidget(self.ensemble_landing)
        )
        self.ensemble_landing.iterative_selected.connect(
            self._on_iterative_selected
        )
        self.iterative_ensemble.navigate_back.connect(
            lambda: self.ensemble_stack.setCurrentWidget(self.ensemble_landing)
        )
        self.iterative_ensemble.log_output.connect(
            self.console_page.append_log
        )
        self.inference_page.process_running.connect(
            self._on_process_state
        )
        self.auto_ensemble.process_running.connect(
            self._on_process_state
        )
        self.iterative_ensemble.process_running.connect(
            self._on_process_state
        )
        # Console needs the same state: mid-run error lines must not flip
        # song cards to FAILED while the job is still going.
        self.inference_page.process_running.connect(
            self.console_page.set_job_active
        )
        self.auto_ensemble.process_running.connect(
            self.console_page.set_job_active
        )
        self.manual_ensemble.process_running.connect(
            self.console_page.set_job_active
        )
        self.iterative_ensemble.process_running.connect(
            self.console_page.set_job_active
        )

    def _apply_theme_bgs(self):
        bg = theme_manager.theme.bg
        # _central only: the stacks are transparent children, so restyling
        # them as well would re-polish the whole page tree two extra times —
        # a visible freeze on every theme switch.
        # Object-name scoped: a bare `background:` declaration would cascade
        # into child dialogs (QMessageBox etc.) and wash out their buttons.
        self._central.setObjectName("appCentral")
        self._central.setStyleSheet(f"#appCentral{{background:{bg};}}")

    def _on_theme_changed(self):
        self._apply_theme_bgs()
        self._header._update_bg()
        for tab in self._nav_tabs:
            tab._apply_style()
        self._indicator._update_color()
        self._theme_toggle._knob_t = 1.0 if theme_manager.mode == "dark" else 0.0
        self._theme_toggle.update()
        if self._processing:
            self._theme_rebuild_pending = True
            return
        if self._switching or self._stripping:
            self._switch_queued = True  # applied when the rebuild finishes
            return
        self._switch_queued = False  # a fresh rebuild applies the latest mode
        self._begin_theme_rebuild()

    def _on_theme_about_to_change(self):
        """Runs before the global stylesheet is re-applied. Removing the pages
        now means Qt re-polishes only the window chrome instead of the whole
        (about-to-be-discarded) page tree — the single biggest switch cost."""
        if self._processing or self._switching or self._stripping:
            return  # pages stay; the restyle has to walk the live tree
        self._strip_pages()

    # —— Theme rebuild pipeline ————————————————————————————————————————
    def _strip_pages(self):
        """Detach and delete the pages. Never blocks on the network: an
        in-flight settings fetch is orphaned (results go nowhere) and its
        page is retired automatically once the fetch finishes."""
        from ui.pages.settings_page import _thread_alive
        fm = getattr(self.settings_page, "_model_mgr", None)
        fetch = orphan_fetch_threads(fm) if fm is not None else None
        self._saved_stack_idx = self._stack.currentIndex()
        self._saved_ens_idx = self.ensemble_stack.currentIndex()
        self._stripping = True
        overlay = self._show_switch_overlay()
        if overlay is not None:
            overlay.set_progress(0.03)
        QApplication.processEvents()  # paint the overlay before tearing down

        pages = (self.inference_page, self.ensemble_landing, self.auto_ensemble,
                 self.manual_ensemble, self.iterative_ensemble, self.console_page,
                 self.settings_page)
        for w in pages:
            self._stack.removeWidget(w)
            self.ensemble_stack.removeWidget(w)
        # Destroy page by page: a plain deleteLater would only run when the
        # outer event loop resumes (leaving ~2000 dead widgets in the tree),
        # and flushing them all at once blocks for most of a second. One
        # flush + event-loop turn per page keeps the overlay spinner moving.
        n = len(pages)
        for i, w in enumerate(pages):
            if w is self.settings_page and fetch is not None \
                    and _thread_alive(fetch) and fetch.isRunning():
                # Its fetch thread is still running; retire the page when the
                # fetch finishes instead of destroying a live thread.
                fetch.finished.connect(w.deleteLater)
                continue
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            if overlay is not None:
                overlay.set_progress(0.03 + 0.07 * (i + 1) / n)
            QApplication.processEvents()
        self._stripping = False
        self._pages_stripped = True
        return True

    def _show_switch_overlay(self):
        if self._switch_overlay is None:
            self._switch_overlay = _ThemeSwitchOverlay(self)
        self._switch_overlay.setGeometry(self.rect())
        self._switch_overlay.show()
        self._switch_overlay.raise_()
        return self._switch_overlay

    def _begin_theme_rebuild(self):
        if self._switching:
            self._switch_queued = True
            return
        if not self._pages_stripped:
            if self._processing:
                self._theme_rebuild_pending = True
                return
            self._strip_pages()
            if self._switch_overlay is None or not self._switch_overlay.isVisible():
                self._show_switch_overlay()
            QApplication.processEvents()
        self._pages_stripped = False
        self._switching = True
        self._rebuild_steps = [
            lambda: setattr(self, "inference_page", InferencePage()),
            lambda: self._add_ensemble_page("ensemble_landing", EnsembleLandingPage()),
            lambda: self._add_ensemble_page("auto_ensemble", AutoEnsemblePage()),
            lambda: self._add_ensemble_page("manual_ensemble", ManualEnsemblePage()),
            lambda: self._add_ensemble_page("iterative_ensemble", IterativeEnsemblePage()),
            lambda: setattr(self, "console_page", ConsolePage()),
            lambda: setattr(self, "settings_page", SettingsPage()),
            self._wire_pages,
            self._apply_theme_bgs,
            self._attach_pages,
            self._load_all,
            self._restore_page_selection,
            self._finish_theme_rebuild,
        ]
        self._rebuild_step_i = 0
        QTimer.singleShot(0, self._rebuild_step)

    def _add_ensemble_page(self, attr, page):
        setattr(self, attr, page)
        self.ensemble_stack.addWidget(page)

    def _attach_pages(self):
        self._stack.addWidget(self.inference_page)       # 0
        self._stack.addWidget(self.ensemble_stack)       # 1
        self._stack.addWidget(self.console_page)         # 2
        self._stack.addWidget(self.settings_page)        # 3

    def _rebuild_step(self):
        if not self._switching:
            return  # a queued rebuild took over
        if self._rebuild_step_i >= len(self._rebuild_steps):
            return
        self._rebuild_steps[self._rebuild_step_i]()
        self._rebuild_step_i += 1
        if self._switch_overlay is not None:
            frac = 0.10 + 0.88 * (self._rebuild_step_i / len(self._rebuild_steps))
            self._switch_overlay.set_progress(frac)
        if self._rebuild_step_i < len(self._rebuild_steps):
            QTimer.singleShot(0, self._rebuild_step)

    def _restore_page_selection(self):
        self._stack.setCurrentIndex(getattr(self, "_saved_stack_idx", 0))
        self.ensemble_stack.setCurrentIndex(getattr(self, "_saved_ens_idx", 0))

    def _finish_theme_rebuild(self):
        if self._switch_overlay is not None:
            self._switch_overlay.hide()
            self._switch_overlay.deleteLater()
            self._switch_overlay = None
        self._switching = False
        if self._switch_queued:
            self._switch_queued = False
            # The mode changed again while rebuilding; apply the latest one.
            QTimer.singleShot(0, self._begin_theme_rebuild)

    def _rebuild_pages(self):
        self._begin_theme_rebuild()

    # —— Blur Animation ————————————————————————————————————————————————
    def _apply_blur_radius(self, r):
        """Attach a blur effect only while r > 0; detach (and dispose) at 0."""
        if r <= 0:
            if self._central.graphicsEffect() is not None:
                self._central.setGraphicsEffect(None)  # deletes the effect
            self._blur_effect = None
        else:
            if self._blur_effect is None:
                self._blur_effect = QGraphicsBlurEffect()
                self._central.setGraphicsEffect(self._blur_effect)
            self._blur_effect.setBlurRadius(r)

    def _animate_blur(self, target_radius, duration=300, callback=None):
        self._blur_target = target_radius
        self._blur_callback = callback
        frames = int(duration / 16)
        if frames <= 0:
            self._apply_blur_radius(target_radius)
            if callback:
                callback()
            return
        self._blur_step = (target_radius - self._blur_current) / frames
        self._blur_timer.start(16)

    def _blur_tick(self):
        self._blur_current += self._blur_step
        diff = self._blur_target - self._blur_current
        if abs(diff) < 0.5:
            self._blur_current = self._blur_target
            self._apply_blur_radius(self._blur_current)
            self._blur_timer.stop()
            if self._blur_callback:
                self._blur_callback()
                self._blur_callback = None
        else:
            self._apply_blur_radius(self._blur_current)
            self._blur_timer.start(16)

    # —— Ckpt Settings Dialog ——————————————————————————————————————————
    def _show_ckpt_settings(self, name, ckpt, yaml_path, arch):
        existing = settings_store.load_ckpt_settings().get(name, {})
        dialog = CkptSettingsDialog(name, yaml_path, arch, existing, self)
        dialog.settings_saved.connect(self._on_ckpt_settings_saved)

        def _open_dialog():
            dialog.exec()
            self._animate_blur(0, duration=300)

        self._animate_blur(8, duration=300, callback=_open_dialog)

    def _on_ckpt_settings_saved(self, ckpt_name, settings):
        settings_store.save_ckpt_settings(ckpt_name, settings)

    # —— Header ————————————————————————————————————————————————————————
    def _build_header(self):
        hdr = _HeaderBar()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(28, 0, 16, 0)
        hl.setSpacing(0)

        brand = QLabel(hdr)
        self._brand_label = brand
        self._update_brand()
        brand.setStyleSheet("background:transparent;border:none;")
        hl.addWidget(brand)

        hl.addStretch(1)

        self._nav_tabs = []
        tab_names = ["INFERENCE", "ENSEMBLE", "CONSOLE", "SETTINGS"]
        for i, name in enumerate(tab_names):
            tab = _NavTab(name, i)
            tab.clicked.connect(self._switch)
            self._nav_tabs.append(tab)
            hl.addWidget(tab)

        hl.addStretch(1)

        self._theme_toggle = _ThemeToggle()
        hl.addWidget(self._theme_toggle)
        hl.addSpacing(12)

        self._window_buttons = _WindowButtons()
        hl.addWidget(self._window_buttons)

        self._indicator = _NavIndicator(hdr)
        self._indicator.hide()

        self._nav_tabs[0].set_active(True)

        QTimer.singleShot(0, self._center_nav)
        QTimer.singleShot(0, self._init_indicator)

        return hdr

    def _center_nav(self):
        """True-center the nav tabs on the window. The right controls (theme
        toggle + window buttons) are much wider than the left logo, so with
        equal stretches the tabs would sit left of center. Insert a fixed
        spacer = controls + right_margin - left_margin - brand (the tab-group
        width and split-point shift cancel out, so this is exact and
        width-independent)."""
        hl = self._header.layout()
        if hl is None or not isinstance(hl, QHBoxLayout):
            return
        brand_w = self._brand_label.width()
        toggle_w = self._theme_toggle.width()
        buttons_w = self._window_buttons.width()
        if brand_w <= 0 or toggle_w <= 0 or buttons_w <= 0:
            # Not laid out yet; retry on the next event-loop pass.
            QTimer.singleShot(0, self._center_nav)
            return
        delta = (toggle_w + 12 + buttons_w) + 16 - 28 - brand_w
        if delta > 0:
            # index 2 = right after the left stretch (brand, stretch, ...)
            hl.insertSpacing(2, delta)
            self._nav_center_delta = delta
            if self._header._active_tab:
                self._indicator.snap_to(self._header._active_tab)

    def _init_indicator(self):
        self._header._active_tab = self._nav_tabs[0]
        self._indicator.snap_to(self._nav_tabs[0])
        self._indicator.show()
        # Re-snap once the nav-centering spacer has settled the layout.
        QTimer.singleShot(0, self._resnap_indicator)

    def _resnap_indicator(self):
        tab = self._header._active_tab or self._nav_tabs[0]
        self._indicator.snap_to(tab)

    def _update_brand(self):
        pm = QPixmap(_LOGO_PATH)
        if not pm.isNull():
            pm = pm.scaledToHeight(32, Qt.SmoothTransformation)
            self._brand_label.setPixmap(pm)
        self._brand_label.setAlignment(Qt.AlignVCenter)

    def _set_dwm_corners(self, round_corners=True):
        try:
            hwnd = int(self.winId())
            pref = 1 if not round_corners else 2
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                33,
                ctypes.byref(ctypes.c_int(pref)),
                ctypes.sizeof(ctypes.c_int)
            )
        except Exception:
            pass

    # —— Native edge resize ——————————————————————————————————————————————
    # The window is frameless, so Windows shows no resize borders. Two pieces
    # make resizing work: the native window style gains WS_THICKFRAME so the
    # OS is willing to run its sizing loop, and WM_NCCALCSIZE claims the whole
    # window as client area so that frame stays invisible. WM_NCHITTEST then
    # returns the standard HT* codes at the edges, which gives native resize
    # behaviour (correct cursors, smooth dragging) without touching any of
    # the custom chrome.
    _HT = {  # (left, top, right, bottom) combos → hit codes
        (True, True, False, False): 13,    # HTTOPLEFT
        (False, True, True, False): 14,    # HTTOPRIGHT
        (True, False, False, True): 16,    # HTBOTTOMLEFT
        (False, False, True, True): 17,    # HTBOTTOMRIGHT
        (True, False, False, False): 10,   # HTLEFT
        (False, False, True, False): 11,   # HTRIGHT
        (False, True, False, False): 12,   # HTTOP
        (False, False, False, True): 15,   # HTBOTTOM
    }
    _RESIZE_BORDER = 6
    # Resizing is disabled: with WS_THICKFRAME present, Windows still paints
    # its legacy inactive frame around the frameless window on some systems
    # (the "ugly border"). Fixed size + min/max/close only. Flip to True to
    # re-enable native edge resizing (requires the thick-frame styles below).
    _NATIVE_RESIZE = False

    def _enable_native_resize(self):
        if os.name != "nt" or not self._NATIVE_RESIZE:
            return
        try:
            hwnd = wintypes.HWND(int(self.winId()))
            user32 = ctypes.windll.user32
            get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            GWL_STYLE = -16
            WS_THICKFRAME = 0x00040000   # allows the OS sizing loop
            WS_MAXIMIZEBOX = 0x00010000  # allows aero-snap sizing states
            style = get_style(hwnd, GWL_STYLE)
            set_style(hwnd, GWL_STYLE, style | WS_THICKFRAME | WS_MAXIMIZEBOX)
            SWP_NOSIZE, SWP_NOMOVE = 0x0001, 0x0002
            SWP_NOZORDER, SWP_FRAMECHANGED = 0x0004, 0x0020
            user32.SetWindowPos(
                hwnd, None, 0, 0, 0, 0,
                SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED)
        except Exception:
            pass

    def _is_maximized(self):
        btn = getattr(self, "_window_buttons", None)
        return btn is not None and btn._maximized

    def nativeEvent(self, eventType, message):
        if eventType == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0086:  # WM_NCACTIVATE
                # Default processing makes DWM paint a light legacy border
                # around the frameless window whenever it loses focus.
                # Forwarding with lParam = -1 applies the activation change
                # without the non-client repaint, keeping the chrome clean.
                result = ctypes.windll.user32.DefWindowProcW(
                    wintypes.HWND(int(self.winId())), 0x0086,
                    msg.wParam, -1)
                return True, result
            if msg.message == 0x0083 and msg.wParam:  # WM_NCCALCSIZE
                return True, 0  # the added thick frame occupies no space
            if (msg.message == 0x0084 and self._NATIVE_RESIZE and
                    self.isVisible() and
                    not self._is_maximized()):  # WM_NCHITTEST
                # lParam packs the screen cursor position as two shorts.
                x = ctypes.c_short(msg.lParam & 0xFFFF).value
                y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(
                        wintypes.HWND(int(self.winId())),
                        ctypes.byref(rect)):
                    border = int(self._RESIZE_BORDER * self.devicePixelRatioF())
                    on_l = (x - rect.left) < border
                    on_r = (rect.right - x) <= border
                    on_t = (y - rect.top) < border
                    on_b = (rect.bottom - y) <= border
                    hit = self._HT.get((on_l, on_t, on_r, on_b), 0)
                    if hit:
                        return True, hit
        return super().nativeEvent(eventType, message)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            btn = getattr(self, '_window_buttons', None)
            if btn:
                btn._toggle_maximize()
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        if self._switch_overlay is not None and self._switch_overlay.isVisible():
            self._switch_overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def _center_on_screen(self):
        """Size the window to a comfortable default and center it on the screen."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.resize(min(1280, geo.width()), min(800, geo.height()))
        r = self.frameGeometry()
        self.move(geo.center().x() - r.width() // 2,
                  geo.center().y() - r.height() // 2)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
        super().keyPressEvent(event)

    def _switch(self, idx):
        self._stack.setCurrentIndex(idx)
        for i, tab in enumerate(self._nav_tabs):
            tab.set_active(i == idx)
        self._header._active_tab = self._nav_tabs[idx]
        self._indicator.move_to(self._nav_tabs[idx])

    def _on_iterative_selected(self):
        from PySide6.QtWidgets import QDialog
        ie_settings = settings_store.load().get("iterative_ensemble", {})
        if not ie_settings.get("warning_dismissed", False):
            dialog = IterativeWarningDialog(self)
            result = dialog.exec()
            if result == QDialog.Rejected:
                return
            if dialog.dont_show_again:
                data = settings_store.load()
                if "iterative_ensemble" not in data:
                    data["iterative_ensemble"] = {}
                data["iterative_ensemble"]["warning_dismissed"] = True
                settings_store.save(data)

        registered = self.settings_page.save_settings()
        results = check_models(registered)
        missing = [(m, s, p) for m, installed, s, p in results if not installed]

        if missing:
            installer_dialog = _ModelInstallerDialog(missing, registered, self)
            installer_dialog.installation_complete.connect(self._on_models_installed)
            installer_dialog.exec()
        else:
            self.ensemble_stack.setCurrentWidget(self.iterative_ensemble)

    def _on_models_installed(self, new_models):
        for model in new_models:
            name = model.get('name', '')
            if any(m.get('name') == name for m in self.settings_page._registered):
                continue
            self.settings_page._registered.append(model)
            self.settings_page._add_item_widget(model)
            self.settings_page.model_registered.emit(model)
            self.settings_page.settings_changed.emit()
        self.ensemble_stack.setCurrentWidget(self.iterative_ensemble)

    def _load_all(self):
        data = settings_store.load()
        self.inference_page.load_settings(data.get("inference", {}))
        models = data.get("registered_models", [])
        self.auto_ensemble.load_models(models)
        self.iterative_ensemble.load_models(models)
        self.settings_page.load_settings(models)

    def _on_process_state(self, running):
        self._processing = running
        tab = self._nav_tabs[2]
        if running:
            tab.set_attention_state(_NavTab.RUNNING)
            # Jump to CONSOLE so the running job is immediately visible
            if self._stack.currentIndex() != 2:
                self._switch(2)
        else:
            tab.set_attention_state(_NavTab.IDLE)
            if self._theme_rebuild_pending:
                self._theme_rebuild_pending = False
                self._rebuild_pages()

    def _save_all(self):
        data = settings_store.load()
        data.update({
            "inference":         self.inference_page.save_settings(),
            "registered_models": self.settings_page.save_settings(),
        })
        settings_store.save(data)

    def closeEvent(self, event):
        self._save_all()
        event.accept()
