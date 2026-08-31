"""
ui/pages/auto_ensemble_page.py
Auto Ensemble page — modern card-based redesign.
"""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QFileDialog, QScrollArea, QSizePolicy, QGridLayout, QButtonGroup,
    QLayout, QWidgetItem,
)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics

from backend.yaml_analyzer import get_stems_for_type
from backend.auto_ensemble_runner import AutoEnsembleRunner
from ui.theme import theme_manager, UIConstants
from ui.widgets.common import PageHeader, _type_badge_ss, _custom_badge_ss, _type_title

def _accent():
    return theme_manager.accent


def _bg():
    return theme_manager.theme.bg


def _card():
    return theme_manager.theme.card

STEM_TYPES = [
    "vocals",
    "instrumental",
    "dereverb / deecho",
    "denoise",
    "phantom centre",
    "karaoke",
    "dual target (instrumental & vocals)",
    "multi stems",
    "super resolution",
    "drums",
    "bass",
    "piano",
    "guitar",
    "wind",
    "strings",
    "percussion",
    "keys",
]

def _parse_hex(hex_str):
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def _rgba_str(hex_str, alpha):
    r, g, b = _parse_hex(hex_str)
    return f"rgba({r},{g},{b},{alpha})"


def _elide_text(text, max_width, font=None):
    if not text:
        return ""
    fm = QFontMetrics(font or QFont())
    return fm.elidedText(text, Qt.ElideRight, max_width)


# ── Icon Widget ──────────────────────────────────────────────────────────────

class _IconWidget(QWidget):
    WAVEFORM, FOLDER, SLIDERS, TARGET = range(4)

    def __init__(self, icon_type, size=20):
        super().__init__()
        self._type = icon_type
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        pen = QPen(QColor(_accent()), 1.5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        c = QColor(_accent())
        c.setAlpha(180)
        p.setPen(QPen(c, 1.5))
        w, h = self.width(), self.height()

        if self._type == self.WAVEFORM:
            pts = []
            for xf, yf in [
                (0.15, 0.70), (0.25, 0.30), (0.35, 0.55), (0.45, 0.15),
                (0.55, 0.60), (0.65, 0.25), (0.75, 0.50), (0.85, 0.35),
            ]:
                pts.append(QPointF(xf * w, yf * h))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])
            p.drawLine(QPointF(pts[0].x() - 2, pts[0].y()), pts[0])
            p.drawLine(pts[-1], QPointF(pts[-1].x() + 2, pts[-1].y()))

        elif self._type == self.FOLDER:
            p.drawRoundedRect(QRectF(w * 0.15, h * 0.35, w * 0.70, h * 0.50), 1.5, 1.5)
            p.drawLine(QPointF(w * 0.15, h * 0.50), QPointF(w * 0.85, h * 0.50))

        elif self._type == self.SLIDERS:
            for yf, lx in [(0.25, 0.60), (0.50, 0.75), (0.75, 0.50)]:
                yp = yf * h
                p.drawLine(QPointF(w * 0.15, yp), QPointF(w * lx, yp))
                p.drawEllipse(QPointF(w * lx, yp), 2.5, 2.5)

        elif self._type == self.TARGET:
            cx, cy = w / 2.0, h / 2.0
            p.drawEllipse(QPointF(cx, cy), w * 0.40, h * 0.40)
            p.drawEllipse(QPointF(cx, cy), w * 0.15, h * 0.15)
            p.drawLine(QPointF(cx, 0), QPointF(cx, h))
            p.drawLine(QPointF(0, cy), QPointF(w, cy))
        p.end()

# ── Card Container ───────────────────────────────────────────────────────────

class _CardContainer(QFrame):
    def __init__(self, title="", subtitle="", icon_type=None, right_widget=None, parent=None):
        super().__init__(parent)
        self.setObjectName("cardContainer")
        self.setStyleSheet(
            f"QFrame#cardContainer{{background:transparent;border:none;}}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 20)
        root.setSpacing(12)

        if title or subtitle or icon_type is not None:
            hdr = QHBoxLayout()
            hdr.setContentsMargins(0, 0, 0, 0)
            hdr.setSpacing(12)

            if icon_type is not None:
                hdr.addWidget(_IconWidget(icon_type))

            txt_col = QVBoxLayout()
            txt_col.setContentsMargins(0, 0, 0, 0)
            txt_col.setSpacing(3)
            if title:
                tl = QLabel(title)
                tl.setWordWrap(True)
                tl.setStyleSheet(
                    "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;"
                    f"color:{theme_manager.theme.text};background:transparent;letter-spacing:1.5px;"
                )
                txt_col.addWidget(tl)
            if subtitle:
                sl = QLabel(subtitle)
                sl.setWordWrap(True)
                sl.setStyleSheet(
                    "font-family:'Montserrat';font-size:11px;"
                    f"color:{theme_manager.theme.text_dim};background:transparent;"
                )
                txt_col.addWidget(sl)

            hdr.addLayout(txt_col, 1)

            if right_widget:
                hdr.addWidget(right_widget)

            root.addLayout(hdr)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{theme_manager.theme.border};border:none;")
        root.addWidget(sep)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(14)
        root.addLayout(self.content_layout)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        p.setBrush(QColor(_card()))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(r, UIConstants.CARD_RADIUS_PAINT, UIConstants.CARD_RADIUS_PAINT)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(theme_manager.theme.border_visible), 1))
        p.drawRoundedRect(r, UIConstants.CARD_RADIUS_PAINT, UIConstants.CARD_RADIUS_PAINT)

        p.end()
        super().paintEvent(event)

# ── Stem Type Button ─────────────────────────────────────────────────────────

class _StemTypeButton(QPushButton):
    def __init__(self, label, type_name, count=0, selected=False):
        super().__init__()
        self._type_name = type_name
        self._selected = selected
        self._count = count
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.setMinimumHeight(36)
        self._build(label)

    def _build(self, label):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        self._name_lbl = QLabel(label.upper())
        self._name_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._name_lbl)

        self._count_lbl = QLabel(f"({self._count})")
        self._count_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._count_lbl)

        self._update_style()

    def set_count(self, count):
        self._count = count
        self._count_lbl.setText(f"({count})")

    def set_selected(self, state):
        self._selected = state
        self._update_style()

    def _update_style(self):
        if self._selected:
            self.setStyleSheet(
                f"QPushButton{{background:{_accent()};color:{theme_manager._accent_text};border:none;border-radius:10px;}}"
                f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
            )
            self._name_lbl.setStyleSheet(
                f"font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;color:{theme_manager._accent_text};background:transparent;"
            )
            _at = QColor(theme_manager._accent_text)
            self._count_lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:9px;font-weight:600;"
                f"color:rgba({_at.red()},{_at.green()},{_at.blue()},0.80);background:transparent;"
            )
        else:
            self.setStyleSheet(
                f"QPushButton{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.border_visible};border-radius:10px;}}"
                f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};border-color:{theme_manager.theme.border_dim};}}"
            )
            self._name_lbl.setStyleSheet(
                f"font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;"
                f"color:{theme_manager.theme.text_sec};background:transparent;"
            )
            self._count_lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:9px;font-weight:600;"
                f"color:{theme_manager.theme.text_muted};background:transparent;"
            )

    @property
    def type_name(self):
        return self._type_name

    def reapply_theme(self):
        self._update_style()

# ── Toggle Circle ────────────────────────────────────────────────────────────

class _ToggleCheck(QPushButton):
    toggled = Signal(bool)

    def __init__(self, checked=False, enabled=True):
        super().__init__()
        self._checked = checked
        self._enabled = enabled
        self._hovered = False
        self.setFixedSize(24, 24)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setEnabled(enabled)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._on_click)

    def _on_click(self):
        self._checked = not self._checked
        self.update()
        self.toggled.emit(self._checked)

    def isChecked(self):
        return self._checked

    def setChecked(self, state):
        self._checked = state
        self.update()

    def setEnabled(self, state):
        self._enabled = state
        super().setEnabled(state)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = 12.0, 12.0

        if not self._enabled:
            p.setPen(QPen(QColor(theme_manager.theme.disabled_bg), 2))
            p.setBrush(QColor(theme_manager.theme.surface_alt))
            p.drawEllipse(QPointF(cx, cy), 10.0, 10.0)

        elif self._checked:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(_accent()))
            p.drawEllipse(QPointF(cx, cy), 10.0, 10.0)
            pen = QPen(QColor(theme_manager._accent_text), 2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(cx - 4, cy), QPointF(cx - 1, cy + 4))
            p.drawLine(QPointF(cx - 1, cy + 4), QPointF(cx + 5, cy - 3))

        else:
            alpha = 100 if self._hovered else 60
            pen = QPen(_rgba_color(theme_manager.theme.text, alpha), 2)
            p.setPen(pen)
            fill_a = 20 if self._hovered else 10
            p.setBrush(_rgba_color(theme_manager.theme.text, fill_a))
            p.drawEllipse(QPointF(cx, cy), 10.0, 10.0)
        p.end()

    def enterEvent(self, e):
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def reapply_theme(self):
        self.update()

def _rgba_color(hex_str, alpha):
    r, g, b = _parse_hex(hex_str)
    c = QColor(r, g, b)
    c.setAlpha(alpha)
    return c

# ── Model Card ───────────────────────────────────────────────────────────────

class _ModelCard(QFrame):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("modelRow")
        self.setFixedHeight(50)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 0, 10, 0)
        root.setSpacing(10)

        self._cb = _ToggleCheck(checked=False, enabled=True)
        self._cb.toggled.connect(self._on_toggle)
        root.addWidget(self._cb, 0, Qt.AlignCenter)

        info = QVBoxLayout()
        info.setContentsMargins(0, 0, 0, 0)
        info.setSpacing(2)

        self._name_label = QLabel(model.get("name", "Unknown"))
        self._name_label.setWordWrap(False)
        self._name_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._name_label.setMinimumWidth(0)
        info.addWidget(self._name_label)

        meta = QHBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(8)

        self._arch_label = QLabel(_elide_text(model.get("arch", ""), 160))
        self._arch_label.setWordWrap(False)
        meta.addWidget(self._arch_label)

        model_type = model.get("type", "unknown")
        self._type = model_type
        self._type_badge = QLabel(_type_title(model_type))
        self._type_badge.setToolTip(model_type)
        self._type_badge.setFixedHeight(18)
        meta.addWidget(self._type_badge)

        custom = model.get("custom_backend_enabled", False)
        self._official_badge = QLabel("CUSTOM" if custom else "OFFICIAL")
        self._official_badge.setFixedHeight(18)
        meta.addWidget(self._official_badge)

        meta.addStretch()
        info.addLayout(meta)
        root.addLayout(info, 1)

        self._dots_btn = QPushButton("\u00b7\u00b7\u00b7")
        self._dots_btn.setFixedSize(26, 26)
        self._dots_btn.setCursor(Qt.PointingHandCursor)
        root.addWidget(self._dots_btn, 0, Qt.AlignCenter)

        self._toggle_callbacks = []
        self._update_style()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide_name()

    def showEvent(self, event):
        super().showEvent(event)
        self._elide_name()

    def _elide_name(self):
        w = self._name_label.width() - 2
        if w > 0:
            fm = QFontMetrics(self._name_label.font())
            elided = fm.elidedText(self._model.get("name", "Unknown"), Qt.ElideRight, w)
            self._name_label.setText(elided)

    def _on_toggle(self, state):
        for cb in self._toggle_callbacks:
            cb(state)

    def mouseReleaseEvent(self, e):
        self._cb._on_click()
        super().mouseReleaseEvent(e)

    def _apply_badge_styles(self):
        self._type_badge.setStyleSheet(_type_badge_ss(self._type))
        is_custom = self._official_badge.text() == "CUSTOM"
        if is_custom:
            self._official_badge.setStyleSheet(_custom_badge_ss())
            return
        self._official_badge.setStyleSheet(
            f"font-family:'Montserrat';font-size:8px;font-weight:700;color:{theme_manager.theme.text_dim};background:transparent;"
            f"border:1px solid {theme_manager.theme.border_dim};border-radius:3px;padding:1px 7px;"
        )

    def _update_style(self):
        self.setStyleSheet(
            f"QFrame#modelRow{{background:{theme_manager.theme.surface};"
            f"border:1px solid {theme_manager.theme.border_visible};border-radius:8px;}}"
            f"QFrame#modelRow:hover{{background:{theme_manager.theme.surface_alt};border-color:{theme_manager.theme.border_dim};}}"
        )
        self._name_label.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:13px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;"
        )
        self._arch_label.setStyleSheet(
            "font-family:'Montserrat';font-size:10px;"
            f"color:{theme_manager.theme.text_dim};background:transparent;"
        )
        self._dots_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{theme_manager.theme.text_dim};"
            f"border:none;font-size:14px;font-weight:600;border-radius:4px;}}"
            f"QPushButton:hover{{color:{_accent()};background:{_rgba_str(_accent(), 0.12)};}}"
        )
        self._apply_badge_styles()

    @property
    def checked(self):
        return self._cb.isChecked()

    @property
    def model(self):
        return self._model

    def set_skipped(self):
        self._cb.setEnabled(False)
        self._name_label.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:13px;font-weight:bold;"
            f"color:{theme_manager.theme.disabled_text};background:transparent;"
        )

    def reapply_theme(self):
        self._update_style()
        self._cb.reapply_theme()

# ── Target Card ──────────────────────────────────────────────────────────────

class _TargetCard(QPushButton):
    def __init__(self, label, selected=False):
        super().__init__(label.capitalize())
        self._label = label
        self._selected = selected
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(32)
        self._update_style()

    def _update_style(self):
        if self._selected:
            self.setStyleSheet(
                f"QPushButton{{background:{_accent()};color:{theme_manager._accent_text};border:none;"
                "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:600;"
                "border-radius:8px;padding:0 20px;}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.border_visible};"
                "border-radius:10px;font-family:'Montserrat',sans-serif;font-size:12px;"
                f"font-weight:600;color:{theme_manager.theme.text_sec};padding:0 24px;}}"
                f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};border-color:{theme_manager.theme.border_dim};color:{theme_manager.theme.text};}}"
            )

    def set_selected(self, state):
        self._selected = state
        self._update_style()

    @property
    def selected(self):
        return self._selected

    @property
    def label(self):
        return self._label

    def reapply_theme(self):
        self._update_style()

# ── Ensemble Type Card ───────────────────────────────────────────────────────

class _EnsembleTypeCard(QPushButton):
    def __init__(self, label, selected=False):
        super().__init__(label)
        self._label = label
        self._selected = selected
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(32)
        self._update_style()

    def _update_style(self):
        if self._selected:
            self.setStyleSheet(
                f"QPushButton{{background:{_accent()};color:{theme_manager._accent_text};border:none;"
                "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:600;"
                "border-radius:8px;padding:0 16px;}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.border_visible};"
                "border-radius:8px;font-family:'Montserrat',sans-serif;font-size:10px;"
                f"font-weight:600;color:{theme_manager.theme.text_sec};padding:0 16px;}}"
                f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};border-color:{theme_manager.theme.border_dim};color:{theme_manager.theme.text};}}"
            )

    def set_selected(self, state):
        self._selected = state
        self._update_style()

    @property
    def selected(self):
        return self._selected

    @property
    def label(self):
        return self._label

    def reapply_theme(self):
        self._update_style()

# ── Browse Button ────────────────────────────────────────────────────────────

class _BrowseButton(QPushButton):
    def __init__(self, text="Browse"):
        super().__init__(text)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(80)
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(
            f"QPushButton{{background:transparent;color:{theme_manager.theme.text_dim};"
            f"border:1px solid {theme_manager.theme.border};"
            f"font-family:'Montserrat',sans-serif;font-weight:600;font-size:{UIConstants.BTN_FONT_SIZE}px;"
            f"border-radius:{UIConstants.BTN_RADIUS}px;padding:{UIConstants.BTN_FONT_SIZE - 4}px {UIConstants.BTN_PADDING_H + 4}px;}}"
            f"QPushButton:hover{{border-color:{theme_manager.theme.border_dim};color:{theme_manager.theme.text};}}"
        )

    def reapply_theme(self):
        self._update_style()


# ── TTA Button ───────────────────────────────────────────────────────────────

def _tta_btn_style(active):
    if active:
        return (
            f"QPushButton{{background:{_accent()};color:{theme_manager._accent_text};border:none;"
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.BTN_FONT_SIZE}px;font-weight:600;"
            f"border-radius:{UIConstants.BTN_RADIUS}px;padding:{UIConstants.BTN_FONT_SIZE - 4}px {UIConstants.BTN_PADDING_H + 4}px;}}"
        )
    else:
        return (
            f"QPushButton{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.border_visible};"
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.BTN_FONT_SIZE}px;font-weight:600;"
            f"color:{theme_manager.theme.text_sec};border-radius:{UIConstants.BTN_RADIUS}px;padding:{UIConstants.BTN_FONT_SIZE - 4}px {UIConstants.BTN_PADDING_H + 4}px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};border-color:{theme_manager.theme.border_dim};color:{theme_manager.theme.text};}}"
        )

# ── Quality Button ───────────────────────────────────────────────────────────

class _QualityButton(QPushButton):
    def __init__(self, label, selected=False):
        super().__init__(label)
        self._selected = selected
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._update_style()

    def set_selected(self, state):
        self._selected = state
        self._update_style()

    def _update_style(self):
        if self._selected:
            self.setStyleSheet(
                f"QPushButton{{background:{_accent()};color:{theme_manager._accent_text};border:none;"
                f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.BTN_FONT_SIZE}px;font-weight:600;"
                f"border-radius:{UIConstants.BTN_RADIUS}px;padding:{UIConstants.BTN_FONT_SIZE - 4}px {UIConstants.BTN_PADDING_H + 4}px;}}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.border_visible};"
                f"border-radius:{UIConstants.BTN_RADIUS}px;font-family:'Montserrat',sans-serif;font-size:{UIConstants.BTN_FONT_SIZE}px;"
                f"font-weight:600;color:{theme_manager.theme.text_sec};padding:{UIConstants.BTN_FONT_SIZE - 4}px {UIConstants.BTN_PADDING_H + 4}px;}}"
                f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};border-color:{theme_manager.theme.border_dim};color:{theme_manager.theme.text};}}"
            )

    def reapply_theme(self):
        self._update_style()

# ── Selected Badge ───────────────────────────────────────────────────────────

class _SelectedBadge(QLabel):
    def __init__(self, count=0):
        super().__init__()
        self._count = count
        self.setFixedHeight(22)
        self._update(count)

    def _update(self, count):
        self._count = count
        self.setText(f"{count} SELECTED")
        self.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:600;"
            f"color:{_accent()};background:transparent;"
            f"border:1px solid {_rgba_str(_accent(), 0.35)};border-radius:4px;padding:2px 10px;"
        )

    def set_count(self, count):
        self._update(count)

# ── Info Row (for input/output sections) ─────────────────────────────────────

class _InfoRow(QWidget):
    def __init__(self, label, value="", right_widget=None):
        super().__init__()
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )
        root.addWidget(self._lbl)

        self._val = QLabel(value)
        self._val.setStyleSheet(
            "font-family:'Montserrat';font-size:11px;"
            f"color:{theme_manager.theme.text_dim};background:transparent;"
        )
        root.addWidget(self._val, 1)

        if right_widget:
            root.addWidget(right_widget)

    def set_value(self, text):
        self._val.setText(text)

# ── Auto Ensemble Page ───────────────────────────────────────────────────────

class AutoEnsemblePage(QWidget):
    navigate_back = Signal()
    ensemble_finished = Signal(bool, str, str)
    log_output = Signal(str)
    process_running = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("autoEnsemblePage")
        # Object-name scoped so the background doesn't cascade into child
        # dialogs (QMessageBox etc.) and overwrite their button styles.
        self.setStyleSheet(f"#autoEnsemblePage{{background:{_bg()};}}")
        self._models = []
        self._stem_type = ""
        self._target_output = ""
        self._ensemble_type = "Average"
        self._input_audio = ""
        self._output_dir = ""
        self._runner = None
        self._needs_model_refresh = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._page_scroll = QScrollArea()
        self._page_scroll.setWidgetResizable(True)
        self._page_scroll.setFrameShape(QFrame.NoFrame)
        self._page_scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
            f"QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};"
            f"border-radius:2px;min-height:30px;}}"
            f"QScrollBar::handle:vertical:hover{{background:{theme_manager.theme.scrollbar_handle};}}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
        )
        self._page_scroll.viewport().setStyleSheet("background:transparent;border:none;")

        content = QWidget()
        content.setStyleSheet("background:transparent;")
        root = QVBoxLayout(content)
        root.setContentsMargins(32, 16, 32, 12)
        root.setSpacing(24)

        self._build_header(root)
        self._build_row1(root)
        self._build_row2(root)
        self._build_row3(root)

        self._page_scroll.setWidget(content)
        outer.addWidget(self._page_scroll, 1)

        self._build_action_bar(outer)

        self._update_start_button()

    def _build_header(self, root):
        hdr = PageHeader(
            "AUTO ENSEMBLE",
            "AUTOMATICALLY COMBINE COMPATIBLE MODELS",
            highlight="COMPATIBLE MODELS",
            back=True,
        )
        self._back_btn = hdr.back_btn
        self._back_btn.clicked.connect(self.navigate_back.emit)
        root.addWidget(hdr, 0)

    def _build_row1(self, root):
        row_w = QWidget()
        row_w.setStyleSheet("background:transparent;")
        row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(24)

        # ── STEM TYPE card ──
        stem_card = _CardContainer("STEM TYPE", "Select the stem you want to ensemble",
                                    icon_type=_IconWidget.WAVEFORM)
        self._stem_grid = QGridLayout()
        self._stem_grid.setContentsMargins(0, 0, 0, 0)
        self._stem_grid.setSpacing(8)
        self._stem_buttons = []
        stem_scroll = QScrollArea()
        stem_scroll.setWidgetResizable(True)
        stem_scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}"
                                  "QScrollBar:vertical{width:4px;background:transparent;}"
                                  f"QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};border-radius:2px;}}")
        stem_scroll.setMaximumHeight(240)
        stem_content = QWidget()
        stem_content.setStyleSheet("background:transparent;")
        stem_content.setLayout(self._stem_grid)
        stem_scroll.setWidget(stem_content)
        stem_card.content_layout.addWidget(stem_scroll)
        stem_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stem_card.layout().setContentsMargins(20, 12, 20, 14)
        stem_card.layout().setSpacing(6)
        row.addWidget(stem_card)

        # ── INPUT/OUTPUT card ──
        io_card = _CardContainer("INPUT / OUTPUT", "Configure input and output settings",
                                  icon_type=_IconWidget.FOLDER)
        io_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        io_card.setMaximumHeight(245)
        io_card.layout().setContentsMargins(20, 12, 20, 14)
        io_card.layout().setSpacing(6)
        self._io_layout = io_card.content_layout
        self._io_layout.setSpacing(10)

        # Input
        self._input_row = _InfoRow("INPUT AUDIO", "No file selected",
                                    right_widget=_BrowseButton("Browse"))
        self._input_row.findChild(_BrowseButton).clicked.connect(self._browse_input)
        self._io_layout.addWidget(self._input_row)

        sep1 = QFrame()
        sep1.setFixedHeight(1)
        sep1.setStyleSheet(f"background:{theme_manager.theme.border};border:none;")
        self._io_layout.addWidget(sep1)

        # Output
        self._output_row = _InfoRow("OUTPUT DIRECTORY", "./ensemble/",
                                     right_widget=_BrowseButton("Browse"))
        self._output_row.findChild(_BrowseButton).clicked.connect(self._browse_output)
        self._io_layout.addWidget(self._output_row)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background:{theme_manager.theme.border};border:none;")
        self._io_layout.addWidget(sep2)

        # TTA
        tta_row = QHBoxLayout()
        tta_row.setContentsMargins(0, 0, 0, 0)
        tta_row.setSpacing(10)
        tta_lbl = QLabel("TTA (Test Time Augmentation)")
        tta_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )
        tta_row.addWidget(tta_lbl)
        tta_row.addStretch()
        self._tta_group = QButtonGroup(self)
        self._tta_off = QPushButton("Off")
        self._tta_off.setCheckable(True)
        self._tta_off.setChecked(True)
        self._tta_off.setCursor(Qt.PointingHandCursor)
        self._tta_on = QPushButton("On")
        self._tta_on.setCheckable(True)
        self._tta_on.setChecked(False)
        self._tta_on.setCursor(Qt.PointingHandCursor)
        self._tta_group.addButton(self._tta_off)
        self._tta_group.addButton(self._tta_on)
        self._tta_group.buttonClicked.connect(self._on_tta_changed)
        self._update_tta_style()
        tta_row.addWidget(self._tta_off)
        tta_row.addWidget(self._tta_on)
        self._io_layout.addLayout(tta_row)

        sep3 = QFrame()
        sep3.setFixedHeight(1)
        sep3.setStyleSheet(f"background:{theme_manager.theme.border};border:none;")
        self._io_layout.addWidget(sep3)

        # Quality
        quality_row = QHBoxLayout()
        quality_row.setContentsMargins(0, 0, 0, 0)
        quality_row.setSpacing(8)
        qlbl = QLabel("OUTPUT QUALITY")
        qlbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )
        quality_row.addWidget(qlbl)
        quality_row.addStretch()
        self._quality_btns = []
        self._quality_group = QButtonGroup(self)
        for i, q in enumerate(["MP3", "FLAC", "WAV"]):
            btn = _QualityButton(q, selected=(q == "WAV"))
            btn.clicked.connect(lambda checked=False, b=btn: self._on_quality(b))
            self._quality_group.addButton(btn)
            self._quality_btns.append(btn)
            quality_row.addWidget(btn)
        self._io_layout.addLayout(quality_row)

        row.addWidget(io_card)
        root.addWidget(row_w, 1)

    def _build_row2(self, root):
        self._selected_badge = _SelectedBadge(0)
        model_card = _CardContainer("MODEL SELECTION",
                                     "Select the models you want to include in the ensemble",
                                     icon_type=_IconWidget.SLIDERS,
                                     right_widget=self._selected_badge)
        model_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._model_scroll = QScrollArea()
        self._model_scroll.setWidgetResizable(True)
        self._model_scroll.setFrameShape(QFrame.NoFrame)
        self._model_scroll.setMinimumHeight(160)
        self._model_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._model_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._model_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._model_scroll.setStyleSheet(
            f"QScrollArea{{background:transparent;border:none;}}"
            f"QScrollBar:vertical{{width:4px;background:transparent;margin:0;}}"
            f"QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            f"QScrollBar::handle:vertical:hover{{background:{theme_manager.theme.scrollbar_handle};}}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
            f"QScrollBar:horizontal{{height:4px;background:transparent;margin:0;}}"
            f"QScrollBar::handle:horizontal{{background:{theme_manager.theme.scrollbar_handle};"
            "border-radius:2px;min-width:30px;}"
            f"QScrollBar::handle:horizontal:hover{{background:{theme_manager.theme.scrollbar_handle};}}"
            "QScrollBar::add-line:horizontal{width:0;}"
            "QScrollBar::sub-line:horizontal{width:0;}"
            "QScrollBar::add-page:horizontal,QScrollBar::sub-page:horizontal{background:transparent;}"
        )

        self._model_container = QWidget()
        self._model_container.setStyleSheet("background:transparent;")
        self._model_layout = QGridLayout(self._model_container)
        self._model_layout.setContentsMargins(0, 0, 4, 0)
        self._model_layout.setSpacing(8)

        self._model_scroll.setWidget(self._model_container)
        model_card.content_layout.addWidget(self._model_scroll)

        self._no_models_label = QLabel("No models registered. Go to Settings to add models.")
        self._no_models_label.setStyleSheet(
            "font-family:'Montserrat';font-size:11px;"
            f"color:{theme_manager.theme.text_dim};background:transparent;padding:12px 0;"
        )
        self._no_models_label.setVisible(False)
        model_card.content_layout.addWidget(self._no_models_label)

        root.addWidget(model_card, 5)

    def _build_row3(self, root):
        row_w = QWidget()
        row_w.setStyleSheet("background:transparent;")
        row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(24)

        # ── ENSEMBLE TYPE card ──
        et_card = _CardContainer("ENSEMBLE TYPE", "How should the ensemble be created?",
                                  icon_type=_IconWidget.SLIDERS)
        et_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        et_card.setMaximumHeight(140)
        et_card.layout().setContentsMargins(20, 12, 20, 14)
        et_card.layout().setSpacing(8)
        self._type_layout = QHBoxLayout()
        self._type_layout.setContentsMargins(0, 0, 0, 0)
        self._type_layout.setSpacing(8)
        self._type_cards = []
        for t in ["Average", "Median", "Max Spec", "Min Spec"]:
            card = _EnsembleTypeCard(t, selected=(t == self._ensemble_type))
            card.clicked.connect(lambda checked=False, c=card: self._on_type(c))
            self._type_layout.addWidget(card)
            self._type_cards.append(card)
        self._type_layout.addStretch()
        et_card.content_layout.addLayout(self._type_layout)
        row.addWidget(et_card)

        # ── TARGET OUTPUT card ──
        to_card = _CardContainer("TARGET OUTPUT", "Select the output type",
                                  icon_type=_IconWidget.TARGET)
        to_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        to_card.setMaximumHeight(140)
        to_card.layout().setContentsMargins(20, 12, 20, 14)
        to_card.layout().setSpacing(8)
        self._target_layout = QHBoxLayout()
        self._target_layout.setContentsMargins(0, 0, 0, 0)
        self._target_layout.setSpacing(8)
        self._target_cards = []
        to_card.content_layout.addLayout(self._target_layout)
        row.addWidget(to_card)

        root.addWidget(row_w, 1)

    def _build_action_bar(self, outer):
        bar_container = QWidget()
        bar_container.setObjectName("actionBar")
        bar_container.setStyleSheet(
            f"QWidget#actionBar{{background:{_card()};border:none;}}"
        )

        bar = QHBoxLayout(bar_container)
        bar.setContentsMargins(UIConstants.ACTION_BAR_MARGIN_LR, UIConstants.ACTION_BAR_MARGIN_TOP, UIConstants.ACTION_BAR_MARGIN_LR, UIConstants.ACTION_BAR_MARGIN_BOTTOM)
        bar.setSpacing(UIConstants.ACTION_BAR_SPACING)

        self._start_btn = QPushButton("Start Ensemble")
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setMinimumWidth(200)
        self._start_btn.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self._start_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._start_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
            f"font-family:'Montserrat',sans-serif;font-weight:600;font-size:{UIConstants.ACTION_FONT_SIZE}px;"
            f"border-radius:{UIConstants.ACTION_RADIUS}px;}}"
            f"QPushButton:hover{{background:{theme_manager.accent};}}"
            f"QPushButton:pressed{{background:{theme_manager.accent};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
        )
        self._start_btn.clicked.connect(self._start)
        bar.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self._stop_btn.setMinimumWidth(90)
        self._stop_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._stop_btn.setCursor(Qt.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        self._update_stop_style()
        bar.addWidget(self._stop_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self._pause_btn.setMinimumWidth(90)
        self._pause_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._pause_btn.setCursor(Qt.PointingHandCursor)
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._update_pause_style()
        bar.addWidget(self._pause_btn)

        self._open_btn = QPushButton("Open Output")
        self._open_btn.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self._open_btn.setMinimumWidth(110)
        self._open_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._open_btn.setCursor(Qt.PointingHandCursor)
        self._open_btn.clicked.connect(self._open_output)
        self._update_open_style()
        bar.addWidget(self._open_btn)

        outer.addWidget(bar_container, 0)

    # ── Action Bar helpers ────────────────────────────────────────────────

    def _update_stop_style(self):
        self._stop_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.error};color:{theme_manager.theme.text};border:none;"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;"
            "border-radius:8px;}"
            f"QPushButton:hover{{background:{theme_manager.theme.error};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
        )

    def _update_pause_style(self):
        self._pause_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};"
            f"border:1px solid {theme_manager.theme.border};"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;"
            "border-radius:8px;}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
        )

    def _update_open_style(self):
        self._open_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};"
            f"border:1px solid {theme_manager.theme.border};"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:10px;"
            "border-radius:8px;}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
        )

    # ── Slots ─────────────────────────────────────────────────────────────

    def _update_tta_style(self):
        self._tta_off.setStyleSheet(_tta_btn_style(self._tta_off.isChecked()))
        self._tta_on.setStyleSheet(_tta_btn_style(self._tta_on.isChecked()))

    def _on_tta_changed(self, btn):
        self._update_tta_style()

    def _on_quality(self, btn):
        for b in self._quality_btns:
            b.set_selected(b is btn)

    def showEvent(self, event):
        super().showEvent(event)
        if self._needs_model_refresh:
            self._needs_model_refresh = False
            self._on_stem_type(self._stem_type)

    def _on_stem_type(self, type_name):
        self._stem_type = type_name
        for btn in self._stem_buttons:
            btn.set_selected(btn.type_name == type_name)

        stems = get_stems_for_type(type_name)
        self._target_output = stems[0] if stems else ""
        self._refresh_target_cards()
        self._refresh_models()
        self._update_start_button()

    def _refresh_target_cards(self):
        while self._target_layout.count():
            item = self._target_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._target_cards = []

        for t in get_stems_for_type(self._stem_type):
            card = _TargetCard(t, selected=(t == self._target_output))
            card.clicked.connect(lambda checked=False, c=card: self._on_target(c))
            self._target_layout.addWidget(card)
            self._target_cards.append(card)
        self._target_layout.invalidate()

    def _on_target(self, card):
        for c in self._target_cards:
            c.set_selected(c == card)
        self._target_output = card.label
        self._update_start_button()

    def _on_type(self, card):
        for c in self._type_cards:
            c.set_selected(c == card)
        self._ensemble_type = card.label

    def _browse_input(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio files (*.wav *.flac *.mp3 *.ogg *.aiff *.m4a *.opus *.wv);;All files (*.*)"
        )
        if files:
            self._input_audio = files
            label = f"{len(files)} file(s)" if len(files) > 1 else os.path.basename(files[0])
            self._input_row.set_value(label)
            self._update_start_button()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", "./ensemble/")
        if path:
            self._output_dir = path
            self._output_row.set_value(path)
        else:
            self._output_dir = os.path.join(os.getcwd(), "ensemble")
            self._output_row.set_value("./ensemble/")

    def _update_start_button(self):
        has_models = any(c.checked for c in self._model_cards) if hasattr(self, "_model_cards") else False
        has_input = bool(self._input_audio)
        has_target = bool(self._target_output)
        enabled = has_models and has_input and has_target and len(self._get_selected_models()) >= 2
        self._start_btn.setEnabled(enabled)

        count = len(self._get_selected_models())
        if hasattr(self, "_selected_badge"):
            self._selected_badge.set_count(count)

    def _get_selected_models(self):
        return [c.model for c in getattr(self, "_model_cards", []) if c.checked]

    def load_models(self, models):
        self._models = models
        self._refresh_stem_buttons()

    def on_model_registered(self, model):
        for i, m in enumerate(self._models):
            if m.get("name") == model.get("name"):
                # Re-registration (e.g. a type reconciliation): replace in
                # place so the corrected type is used.
                self._models[i] = model
                self._refresh_stem_buttons()
                self._refresh_models()
                return
        self._models.append(model)
        self._refresh_stem_buttons()
        self._refresh_models()

    def on_model_removed(self, name):
        self._models = [m for m in self._models if m.get("name") != name]
        self._refresh_stem_buttons()
        self._refresh_models()

    def _refresh_models(self):
        # Clear grid
        while self._model_layout.count():
            item = self._model_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        # Reset residual column stretches from previous stem type
        for c in range(self._model_layout.columnCount()):
            self._model_layout.setColumnStretch(c, 0)

        self._model_cards = []

        if not self._stem_type or not self._models:
            self._no_models_label.setVisible(True)
            self._model_scroll.setVisible(False)
            return

        filtered = [m for m in self._models if m.get("type", "").lower() == self._stem_type.lower()]

        if not filtered:
            self._no_models_label.setVisible(True)
            self._model_scroll.setVisible(False)
            return

        self._no_models_label.setVisible(False)
        self._model_scroll.setVisible(True)

        MAX_PER_COL = 3
        CARD_W = 400
        for i, model in enumerate(filtered):
            card = _ModelCard(model)
            card._cb.toggled.connect(self._update_start_button)
            self._model_cards.append(card)
            card.setFixedWidth(CARD_W)
            row = i % MAX_PER_COL
            col = i // MAX_PER_COL
            self._model_layout.addWidget(card, row, col, Qt.AlignLeft | Qt.AlignTop)

        cols = (len(filtered) + MAX_PER_COL - 1) // MAX_PER_COL
        # Stretch only the column AFTER the last content column —
        # this absorbs extra space so content columns stay compact.
        self._model_layout.setColumnStretch(cols, 1)
        self._model_layout.setRowStretch(MAX_PER_COL, 1)

        spacing = self._model_layout.spacing()
        m = self._model_layout.contentsMargins()
        min_w = cols * CARD_W + (cols - 1) * spacing + m.left() + m.right()
        self._model_container.setMinimumWidth(min_w)

        self._update_start_button()
        self._model_layout.activate()
        self._model_container.updateGeometry()
        self._model_container.update()
        self._model_scroll.widget().update()

    def _refresh_stem_buttons(self):
        for btn in self._stem_buttons:
            self._stem_grid.removeWidget(btn)
            btn.deleteLater()
        self._stem_buttons = []

        if not self._models:
            self._no_models_label.setVisible(True)
            self._model_scroll.setVisible(False)
            self._stem_type = ""
            return

        if self._stem_type not in STEM_TYPES:
            self._stem_type = ""

        for i, t in enumerate(STEM_TYPES):
            label = t.capitalize() if t != "dual target (instrumental & vocals)" else "Vocals / Inst"
            count = sum(1 for m in self._models if m.get("type", "").lower() == t.lower())
            btn = _StemTypeButton(label, t, count=count, selected=(t == self._stem_type))
            btn.clicked.connect(lambda checked, k=t: self._on_stem_type(k))
            row = i // 3
            col = i % 3
            self._stem_grid.addWidget(btn, row, col)
            self._stem_buttons.append(btn)

        if not self._stem_type:
            self._stem_type = STEM_TYPES[0]
            if self.isVisible():
                self._on_stem_type(self._stem_type)
            else:
                self._needs_model_refresh = True

    def _start(self):
        from ui.widgets.runtime_dialog import ensure_runtime
        if not ensure_runtime(self):
            return
        selected = self._get_selected_models()
        if len(selected) < 2:
            return
        if not self._input_audio:
            return

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        if not self._output_dir:
            self._output_dir = os.path.join(os.getcwd(), "ensemble")

        self._runner = AutoEnsembleRunner(
            selected, self._input_audio, self._target_output,
            self._ensemble_type, self._output_dir,
        )
        self._runner.stage_changed.connect(self._on_stage_changed)
        self._runner.model_progress.connect(self._on_model_progress)
        self._runner.ensemble_progress.connect(self._on_ensemble_progress)
        self._runner.log_line.connect(self.log_output.emit)
        self._runner.finished.connect(self._on_finished)
        self._runner.error.connect(self._on_error)
        self._runner.model_skipped.connect(self._on_model_skipped)
        self.process_running.emit(True)
        self._runner.start()

    def _stop(self):
        if self._runner:
            self.log_output.emit("Stopping...")
            self._runner.cancel()
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self.process_running.emit(False)

    def _toggle_pause(self):
        pass

    def _open_output(self):
        path = self._output_dir or os.path.join(os.getcwd(), "ensemble")
        if os.path.isdir(path):
            os.startfile(path)

    def _on_stage_changed(self, stage, current, total):
        pass

    def _on_model_progress(self, model_name, percentage):
        pass

    def _on_ensemble_progress(self, percentage):
        pass

    def _on_finished(self, success, message, output_path):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.process_running.emit(False)
        self.ensemble_finished.emit(success, message, output_path)

    def _on_error(self, message):
        self.log_output.emit(f"ERROR: {message}")
        self.process_running.emit(False)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_model_skipped(self, model_name, reason):
        self.log_output.emit(f"SKIPPED: {model_name} - {reason}")
        for card in getattr(self, "_model_cards", []):
            if card.model.get("name") == model_name:
                card.set_skipped()
                break

    def reapply_theme(self):
        for btn in self._stem_buttons:
            btn.reapply_theme()
        for card in getattr(self, "_model_cards", []):
            card.reapply_theme()
        for card in self._target_cards:
            card.reapply_theme()
        for card in self._type_cards:
            card.reapply_theme()
        for btn in self._quality_btns:
            btn.reapply_theme()
        self._input_row.findChild(_BrowseButton).reapply_theme()
        if hasattr(self, "_start_btn"):
            self._update_stop_style()
            self._update_pause_style()
            self._update_open_style()
        self._output_row.findChild(_BrowseButton).reapply_theme()
