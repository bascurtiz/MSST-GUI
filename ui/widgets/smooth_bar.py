"""Thin animated progress bar used in console cards and download dialogs."""
from PySide6.QtCore import Qt, QTimer, Property, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QWidget

from ui.theme import theme_manager


class SmoothBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(5)
        self._current = 0.0
        self._target = 0.0
        self._anim = QPropertyAnimation(self, b"animated_value", self)
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._indeterminate = False
        self._sweep_pos = 0.0
        self._sweep_dir = 1.0
        self._sweep_timer = QTimer(self)
        self._sweep_timer.setInterval(33)
        self._sweep_timer.timeout.connect(self._sweep_tick)

    def _get_animated(self):
        return self._current

    def _set_animated(self, v):
        self._current = v
        self.update()

    animated_value = Property(float, _get_animated, _set_animated)

    def setValue(self, pct):
        v = max(0.0, min(100.0, float(pct)))
        self._target = v
        self._anim.stop()
        self._anim.setStartValue(self._current)
        self._anim.setEndValue(v)
        self._anim.start()

    def setValueImmediate(self, pct):
        v = max(0.0, min(100.0, float(pct)))
        self._anim.stop()
        self._current = v
        self._target = v
        self.update()

    def set_indeterminate(self, on):
        self._indeterminate = bool(on)
        if self._indeterminate:
            self._anim.stop()
            self._current = 0.0
            if not self._sweep_timer.isActive():
                self._sweep_timer.start()
        else:
            self._sweep_timer.stop()
            self._sweep_pos = 0.0
            self._sweep_dir = 1.0
        self.update()

    def _sweep_tick(self):
        self._sweep_pos += 0.045 * self._sweep_dir
        if self._sweep_pos >= 1.0:
            self._sweep_pos = 1.0
            self._sweep_dir = -1.0
        elif self._sweep_pos <= 0.0:
            self._sweep_pos = 0.0
            self._sweep_dir = 1.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        bg = (QColor("#262B33") if theme_manager.mode == "dark"
              else QColor("#DFE5EC"))
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(r, 3, 3)
        if self._indeterminate:
            band = max(0.16, min(0.45, r.width() / 240.0))
            x0 = int((r.width() - r.width() * band) * self._sweep_pos)
            grad = QColor(theme_manager.accent)
            grad.setAlpha(170)
            p.setBrush(grad)
            p.drawRoundedRect(x0, 0, int(r.width() * band), r.height(), 3, 3)
        elif self._current > 0:
            fw = int(r.width() * self._current / 100.0)
            if fw > 0:
                p.setBrush(QColor(theme_manager.accent))
                p.drawRoundedRect(0, 0, fw, r.height(), 3, 3)
        p.end()
