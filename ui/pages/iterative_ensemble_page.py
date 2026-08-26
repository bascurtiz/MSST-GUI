"""ui/pages/iterative_ensemble_page.py
Iterative Ensemble page - cinematic premium interface.
"""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QFileDialog, QScrollArea, QSizePolicy, QComboBox, QSlider,
    QLineEdit, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QRectF, QTimer
from PySide6.QtGui import QPainter, QPainterPath, QLinearGradient, QRadialGradient, QColor, QPen, QBrush

from backend.iterative_ensemble.runner import IterativeEnsembleRunner
from ui.theme import theme_manager, UIConstants
from ui.widgets.common import PageHeader


def _parse_hex(hex_str):
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba_str(hex_str, alpha):
    r, g, b = _parse_hex(hex_str)
    return f"rgba({r},{g},{b},{alpha})"


class _WorkflowStep(QFrame):
    PENDING = 0
    ACTIVE = 1
    COMPLETED = 2
    ERROR = 3

    def __init__(self, icon, name, desc, parent=None):
        super().__init__(parent)
        self._icon = icon
        self._name = name
        self._desc = desc
        self._state = self.PENDING
        self._glow_pos = -0.3

        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        lo = QHBoxLayout(self)
        lo.setContentsMargins(12, 6, 12, 6)
        lo.setSpacing(10)

        self._icon_label = QLabel(icon)
        lo.addWidget(self._icon_label)

        tx = QVBoxLayout()
        tx.setSpacing(1)
        self._name_label = QLabel(name)
        tx.addWidget(self._name_label)
        self._desc_label = QLabel(desc)
        tx.addWidget(self._desc_label)
        lo.addLayout(tx, 1)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)

        self._apply_style()

    def _apply_style(self):
        t = theme_manager.theme
        accent = theme_manager.accent

        if self._state == self.PENDING:
            self.setStyleSheet(
                "QFrame{background:" + t.surface_alt + ";border:none;border-radius:8px;}"
            )
            self._icon_label.setStyleSheet(
                "font-size:12px;color:" + t.text_dim + ";background:transparent;border:none;"
            )
            self._name_label.setStyleSheet(
                "font-family:'Montserrat';font-size:12px;font-weight:700;color:" + t.text_dim + ";background:transparent;border:none;"
            )
            self._desc_label.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;color:" + t.text_dim + ";background:transparent;border:none;"
            )
        elif self._state == self.ACTIVE:
            self.setStyleSheet(
                "QFrame{background:" + t.surface_alt + ";border:none;border-radius:8px;}"
            )
            self._icon_label.setStyleSheet(
                "font-size:12px;color:" + accent + ";background:transparent;border:none;"
            )
            self._name_label.setStyleSheet(
                "font-family:'Montserrat';font-size:12px;font-weight:700;color:" + t.text + ";background:transparent;border:none;"
            )
            self._desc_label.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;color:" + t.text_sec + ";background:transparent;border:none;"
            )
        elif self._state == self.COMPLETED:
            self.setStyleSheet(
                "QFrame{background:" + t.surface_alt + ";border:1px solid " + _rgba_str(accent, 0.15) + ";border-radius:8px;}"
            )
            self._icon_label.setStyleSheet(
                "font-size:12px;color:" + accent + ";background:transparent;border:none;"
            )
            self._name_label.setStyleSheet(
                "font-family:'Montserrat';font-size:12px;font-weight:700;color:" + t.text + ";background:transparent;border:none;"
            )
            self._desc_label.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;color:" + t.text_sec + ";background:transparent;border:none;"
            )
        elif self._state == self.ERROR:
            self.setStyleSheet(
                "QFrame{background:" + t.surface_alt + ";border:1px solid " + _rgba_str(t.error, 0.25) + ";border-radius:8px;}"
            )
            self._icon_label.setStyleSheet(
                "font-size:12px;color:" + t.error + ";background:transparent;border:none;"
            )
            self._name_label.setStyleSheet(
                "font-family:'Montserrat';font-size:12px;font-weight:700;color:" + t.error + ";background:transparent;border:none;"
            )
            self._desc_label.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;color:" + t.text_sec + ";background:transparent;border:none;"
            )

    def set_state(self, state):
        if state == self._state:
            return
        if self._state == self.ACTIVE:
            self._timer.stop()
        self._state = state
        if state == self.ACTIVE:
            self._glow_pos = -0.3
            self._timer.start(16)
        self._apply_style()
        self.update()

    def _tick(self):
        self._glow_pos += 0.025
        if self._glow_pos > 1.3:
            self._glow_pos = -0.3
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._state != self.ACTIVE:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        if w < 1:
            painter.end()
            return
        band_w = w * 0.45
        cx = self._glow_pos * w
        grad = QLinearGradient(cx - band_w * 0.5, 0, cx + band_w * 0.5, 0)
        c0 = QColor(theme_manager.accent)
        c0.setAlpha(0)
        c1 = QColor(theme_manager.accent)
        c1.setAlpha(12)
        grad.setColorAt(0.0, c0)
        grad.setColorAt(0.5, c1)
        grad.setColorAt(1.0, c0)
        path = QPainterPath()
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path.addRoundedRect(rect, 8, 8)
        painter.setClipPath(path)
        painter.fillRect(self.rect(), QBrush(grad))
        painter.end()

    def reset(self):
        if self._state == self.ACTIVE:
            self._timer.stop()
        self._state = self.PENDING
        self._apply_style()
        self.update()

    def reapply_theme(self):
        self._apply_style()
        if self._state == self.ACTIVE:
            self._timer.start(16)


class _AtmosphericBackground(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme_manager.theme.bg))
        painter.end()


class _SectionLabel(QLabel):
    def __init__(self, text):
        super().__init__(text.upper())
        self.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;"
        )


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
        from PySide6.QtCore import QPointF
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = 12.0, 12.0

        if not self._enabled:
            c = QColor(theme_manager.theme.disabled_bg)
            p.setPen(QPen(c, 2))
            c2 = QColor(theme_manager.theme.surface_alt)
            p.setBrush(c2)
            p.drawEllipse(QPointF(cx, cy), 10.0, 10.0)
        elif self._checked:
            ba = 210 if self._hovered else 170
            c = QColor(theme_manager.accent)
            c.setAlpha(ba)
            p.setPen(QPen(c, 1.5))
            c2 = QColor(theme_manager.accent)
            c2.setAlpha(10)
            p.setBrush(c2)
            p.drawEllipse(QPointF(cx, cy), 10.0, 10.0)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme_manager.accent))
            p.drawEllipse(QPointF(cx, cy), 5.0, 5.0)
        else:
            ba = 70 if self._hovered else 50
            c = QColor(theme_manager.theme.text)
            c.setAlpha(ba)
            p.setPen(QPen(c, 1.5))
            fa = 14 if self._hovered else 8
            c2 = QColor(theme_manager.theme.text)
            c2.setAlpha(fa)
            p.setBrush(c2)
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


class _BrowseButton(QPushButton):
    def __init__(self, text="Browse"):
        super().__init__(text)
        self.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self.setMinimumWidth(UIConstants.BTN_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};border:1px solid {theme_manager.theme.border};"
            f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:{UIConstants.BTN_FONT_SIZE}px;"
            f"letter-spacing:1px;border-radius:{UIConstants.BTN_RADIUS}px;padding:0 {UIConstants.BTN_PADDING_H + 4}px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};border-color:{theme_manager.theme.border_dim};}}"
            f"QPushButton:pressed{{background:{theme_manager.theme.surface_alt};}}"
        )


class _StartButton(QPushButton):
    def __init__(self):
        super().__init__("START ITERATIVE ENSEMBLE")
        self.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
            f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:{UIConstants.ACTION_FONT_SIZE}px;"
            f"letter-spacing:2px;border-radius:{UIConstants.ACTION_RADIUS}px;}}"
            f"QPushButton:hover{{background:{theme_manager.accent};}}"
            f"QPushButton:pressed{{background:{theme_manager.accent};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
        )


class _StopButton(QPushButton):
    def __init__(self):
        super().__init__("STOP")
        self.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.error};color:{theme_manager.theme.text};border:none;"
            f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:{UIConstants.ACTION_FONT_SIZE}px;"
            f"letter-spacing:2px;border-radius:{UIConstants.ACTION_RADIUS}px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.error};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
        )


class _PauseButton(QPushButton):
    def __init__(self):
        super().__init__("PAUSE")
        self.setMinimumHeight(UIConstants.BTN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};border:1px solid {theme_manager.theme.border};"
            f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:{UIConstants.ACTION_FONT_SIZE}px;"
            f"letter-spacing:2px;border-radius:{UIConstants.ACTION_RADIUS}px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
        )


class _SliderWithLabel(QWidget):
    value_changed = Signal(int)

    def __init__(self, min_val, max_val, default, step=1):
        super().__init__()
        self._min = min_val
        self._max = max_val
        self._step = step

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(min_val)
        self._slider.setMaximum(max_val)
        self._slider.setValue(default)
        self._slider.setSingleStep(step)
        self._slider.setPageStep(step)
        self._slider.setStyleSheet(
            f"QSlider::groove:horizontal{{height:4px;background:{theme_manager.theme.surface};border-radius:2px;}}"
            f"QSlider::handle:horizontal{{width:16px;height:16px;margin:-6px 0;"
            f"background:{theme_manager.accent};border-radius:8px;}}"
            f"QSlider::sub-page:horizontal{{background:{theme_manager.accent};border-radius:2px;}}"
        )
        self._slider.valueChanged.connect(self._on_change)
        root.addWidget(self._slider, 1)

        self._label = QLabel(str(default))
        self._label.setStyleSheet(
            f"font-family:'Courier New',monospace;font-size:13px;font-weight:700;"
            f"color:{theme_manager.accent};background:transparent;border:none;min-width:24px;"
        )
        root.addWidget(self._label)

    def _on_change(self, value):
        self._label.setText(str(value))
        self.value_changed.emit(value)

    def value(self):
        return self._slider.value()


class _ModelCheckbox(QWidget):
    toggled = Signal(bool)

    def __init__(self, label, checked=False):
        super().__init__()
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self._cb = _ToggleCheck(checked=checked)
        self._cb.toggled.connect(self.toggled.emit)
        root.addWidget(self._cb)

        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"font-family:'Montserrat';font-size:12px;color:{theme_manager.theme.text};background:transparent;"
        )
        root.addWidget(self._label, 1)

    def isChecked(self):
        return self._cb.isChecked()

    def setChecked(self, state):
        self._cb.setChecked(state)


class _EyeToggle(QPushButton):
    def __init__(self):
        super().__init__()
        self._open = False
        self.setMinimumSize(40, 40)
        self.setStyleSheet(f"QPushButton{{background:transparent;border:1px solid {theme_manager.theme.disabled_bg};border-radius:8px;}}"
                           f"QPushButton:hover{{border-color:{theme_manager.theme.text_label};}}")

    def set_open(self, open_):
        self._open = open_
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)

        pen = QPen(QColor(theme_manager.theme.text_label))
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if self._open:
            path = QPainterPath()
            path.moveTo(-8, 0)
            path.cubicTo(-8, -5, 8, -5, 8, 0)
            path.cubicTo(8, 5, -8, 5, -8, 0)
            painter.drawPath(path)
            painter.setBrush(QColor(theme_manager.theme.text_label))
            painter.drawEllipse(-2, -2, 4, 4)
            painter.setBrush(Qt.NoBrush)
        else:
            path = QPainterPath()
            path.moveTo(-8, 0)
            path.cubicTo(-4, 2, 4, 2, 8, 0)
            painter.drawPath(path)

        painter.end()


class _PasswordRow(QWidget):
    def __init__(self):
        super().__init__()
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._input = QLineEdit()
        self._input.setEchoMode(QLineEdit.Password)
        self._input.setPlaceholderText("Enter MVSep API key...")
        self._input.setStyleSheet(
            f"QLineEdit{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text};border:1px solid {theme_manager.theme.border};"
            f"border-radius:6px;padding:8px 12px;font-family:'Courier New',monospace;font-size:12px;}}"
            f"QLineEdit:focus{{border-color:{theme_manager.accent};}}"
        )
        root.addWidget(self._input, 1)

        self._toggle = _EyeToggle()
        self._toggle.clicked.connect(self._toggle_visibility)
        root.addWidget(self._toggle)

    def _toggle_visibility(self):
        if self._input.echoMode() == QLineEdit.Password:
            self._input.setEchoMode(QLineEdit.Normal)
            self._toggle.set_open(True)
        else:
            self._input.setEchoMode(QLineEdit.Password)
            self._toggle.set_open(False)

    def text(self):
        return self._input.text()

    def setText(self, text):
        self._input.setText(text)


class _ExportDropdown(QComboBox):
    def __init__(self):
        super().__init__()
        self.addItems(["wav FLOAT", "flac PCM_16", "flac PCM_24"])
        self.setStyleSheet(
            f"QComboBox{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text};border:1px solid {theme_manager.theme.border};"
            f"border-radius:6px;padding:8px 12px;font-size:12px;}}"
            f"QComboBox::drop-down{{border:none;width:24px;}}"
            f"QComboBox QAbstractItemView{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text};"
            f"selection-background-color:{theme_manager.accent};selection-color:{theme_manager._accent_text};border:1px solid {theme_manager.theme.border};}}"
        )


class IterativeEnsemblePage(QWidget):
    navigate_back = Signal()
    log_output = Signal(str)
    input_files_submitted = Signal(list)
    process_running = Signal(bool)

    MODEL_MAP = {
        "inst_v1e.ckpt": "mel_v1e",
        "BS-Roformer-Resurrection-Inst.ckpt": "bs_resurrect",
        "BS-Roformer_LargeV1.ckpt": "bs_largev1",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_files = []
        self._output_dir = ""
        self._runner = None
        self._running = False
        self._registered_models = []
        self._workflow_steps = []
        self._active_workflow_step = -1
        self._build_ui()

    def _get_registered_model(self, ckpt_filename):
        for m in self._registered_models:
            if m.get("name") == ckpt_filename:
                return m
        return None

    def load_models(self, models):
        self._registered_models = list(models)

    def on_model_registered(self, model):
        for i, m in enumerate(self._registered_models):
            if m.get("name") == model.get("name"):
                self._registered_models[i] = model
                return
        self._registered_models.append(model)

    def on_model_removed(self, name):
        self._registered_models = [m for m in self._registered_models if m.get("name") != name]

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bg = _AtmosphericBackground()
        bg_layout = QVBoxLayout(bg)
        bg_layout.setContentsMargins(32, 32, 32, 32)
        bg_layout.setSpacing(16)

        # ── Header ──────────────────────────────────────────────────────
        self._header = PageHeader(
            "ITERATIVE ENSEMBLE",
            "SEQUENTIAL REFINEMENT THROUGH MULTIPLE MODEL STAGES",
            highlight="MULTIPLE MODEL STAGES",
            back=True,
        )
        self._back_btn = self._header.back_btn
        self._back_btn.clicked.connect(self.navigate_back.emit)

        # MVSep Connection Badge
        self._mvsep_badge = QFrame()
        self._mvsep_badge.setMinimumHeight(28)
        badge_layout = QHBoxLayout(self._mvsep_badge)
        badge_layout.setContentsMargins(12, 0, 12, 0)
        badge_layout.setSpacing(6)
        self._badge_dot = QLabel("\u25cf")
        self._badge_dot.setStyleSheet(f"font-size:10px;color:{theme_manager.theme.text_muted};background:transparent;border:none;")
        badge_layout.addWidget(self._badge_dot)
        self._badge_text = QLabel("Disconnected")
        self._badge_text.setStyleSheet(f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_muted};background:transparent;border:none;")
        badge_layout.addWidget(self._badge_text)
        self._mvsep_badge.setStyleSheet(f"QFrame{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.disabled_bg};border-radius:14px;}}")
        self._header.add_extra(self._mvsep_badge)

        bg_layout.addWidget(self._header)

        # ── Main Content: Two Columns ───────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(32)

        # ── Left Column — Scrollable Settings ──────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
            f"QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            f"QScrollBar::handle:vertical:hover{{background:{theme_manager.theme.border_dim};}}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,"
            "QScrollBar::sub-page:vertical{background:transparent;}"
        )
        left_scroll.viewport().setStyleSheet("background:transparent;border:none;")
        left_content = QWidget()
        left_content.setStyleSheet("background:transparent;")
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(12)

        # ── MAIN SETTINGS Card ─────────────────────────────────────────
        ms_card, ms_layout = self._make_card("Main Settings")

        input_row = QHBoxLayout()
        input_row.setSpacing(14)
        input_lbl = QLabel("INPUT AUDIO")
        input_lbl.setFixedWidth(95)
        input_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;font-weight:600;color:{theme_manager.theme.text_label};background:transparent;")
        input_row.addWidget(input_lbl)
        self._input_display = QLabel("No files selected")
        self._input_display.setStyleSheet(f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_dim};background:transparent;")
        input_row.addWidget(self._input_display, 1)
        input_btn = _BrowseButton("Browse")
        input_btn.clicked.connect(self._browse_input)
        input_row.addWidget(input_btn)
        ms_layout.addLayout(input_row)

        output_row = QHBoxLayout()
        output_row.setSpacing(14)
        out_lbl = QLabel("OUTPUT DIR")
        out_lbl.setFixedWidth(95)
        out_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;font-weight:600;color:{theme_manager.theme.text_label};background:transparent;")
        output_row.addWidget(out_lbl)
        self._output_display = QLabel("./iterative_output/")
        self._output_display.setStyleSheet(f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_dim};background:transparent;")
        output_row.addWidget(self._output_display, 1)
        output_btn = _BrowseButton("Browse")
        output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(output_btn)
        ms_layout.addLayout(output_row)

        format_row = QHBoxLayout()
        format_row.setSpacing(14)
        fmt_lbl = QLabel("FORMAT")
        fmt_lbl.setFixedWidth(95)
        fmt_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;font-weight:600;color:{theme_manager.theme.text_label};background:transparent;")
        format_row.addWidget(fmt_lbl)
        self._export_combo = _ExportDropdown()
        format_row.addWidget(self._export_combo, 1)
        ms_layout.addLayout(format_row)

        overlap_row = QHBoxLayout()
        overlap_row.setSpacing(14)
        ov_lbl = QLabel("OVERLAP")
        ov_lbl.setFixedWidth(95)
        ov_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;font-weight:600;color:{theme_manager.theme.text_label};background:transparent;")
        overlap_row.addWidget(ov_lbl)
        self._overlap_slider = _SliderWithLabel(1, 8, 2)
        overlap_row.addWidget(self._overlap_slider, 1)
        ms_layout.addLayout(overlap_row)

        left_layout.addWidget(ms_card)

        # ── MVSEP SETTINGS Card ────────────────────────────────────────
        mv_card, mv_layout = self._make_card("MVSep Settings")

        api_row = QHBoxLayout()
        api_row.setSpacing(14)
        api_lbl = QLabel("API KEY")
        api_lbl.setFixedWidth(95)
        api_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;font-weight:600;color:{theme_manager.theme.text_label};background:transparent;")
        api_row.addWidget(api_lbl)
        self._api_key_row = _PasswordRow()
        self._api_key_row._input.textChanged.connect(self._update_start_button)
        self._api_key_row._input.textChanged.connect(self._update_mvsep_badge)
        api_row.addWidget(self._api_key_row, 1)
        mv_layout.addLayout(api_row)
        self._update_mvsep_badge()

        self._api_no_credits = _ModelCheckbox("API No Credits (free account limits)", checked=True)
        mv_layout.addWidget(self._api_no_credits)

        helper = QLabel("You can get your API key from console.mvsep.com")
        helper.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;border:none;padding:0;"
        )
        mv_layout.addWidget(helper)

        left_layout.addWidget(mv_card)

        # ── ITERATIVE CONTROLS Card ─────────────────────────────────────
        ic_card, ic_layout = self._make_card("Iterative Controls")

        s1 = QLabel("ITERATIVE STAGE")
        s1.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        ic_layout.addWidget(s1)
        self._restore_side = _ModelCheckbox("Restore Side Iterative", checked=True)
        ic_layout.addWidget(self._restore_side)
        self._amplify_masked = _ModelCheckbox("Amplify Masked Details", checked=True)
        ic_layout.addWidget(self._amplify_masked)
        self._auto_trim = _ModelCheckbox("Auto Trim Normalization", checked=True)
        ic_layout.addWidget(self._auto_trim)
        self._auto_trim_model = _ModelCheckbox("Auto Trim Model Specific", checked=False)
        ic_layout.addWidget(self._auto_trim_model)

        sep1 = QFrame()
        sep1.setFixedHeight(1)
        sep1.setStyleSheet(f"QFrame{{background:{theme_manager.theme.border_visible};border:none;}}")
        ic_layout.addWidget(sep1)

        s2 = QLabel("LOCAL MODELS")
        s2.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        ic_layout.addWidget(s2)
        self._mel_v1e = _ModelCheckbox("MelBand RoFormer v1e", checked=True)
        ic_layout.addWidget(self._mel_v1e)
        self._bs_resurrect = _ModelCheckbox("BS RoFormer Resurrect", checked=True)
        ic_layout.addWidget(self._bs_resurrect)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"QFrame{{background:{theme_manager.theme.border_visible};border:none;}}")
        ic_layout.addWidget(sep2)

        s3 = QLabel("MVSEP API MODELS")
        s3.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        ic_layout.addWidget(s3)
        self._api_mvsep = _ModelCheckbox("MVSep BS RoFormer 2025.07", checked=False)
        ic_layout.addWidget(self._api_mvsep)
        self._api_scnet = _ModelCheckbox("SCNet XL IHF by becruily", checked=False)
        ic_layout.addWidget(self._api_scnet)

        left_layout.addWidget(ic_card)

        # ── POST-PROCESSING Card ────────────────────────────────────────
        pp_card, pp_layout = self._make_card("Post-Processing")

        s4 = QLabel("POST SEPARATION")
        s4.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        pp_layout.addWidget(s4)
        self._post_bs = _ModelCheckbox("Post Separate BS Resurrect", checked=True)
        pp_layout.addWidget(self._post_bs)
        self._post_scnet = _ModelCheckbox("Post Separate SCNet", checked=True)
        pp_layout.addWidget(self._post_scnet)

        sep3 = QFrame()
        sep3.setFixedHeight(1)
        sep3.setStyleSheet(f"QFrame{{background:{theme_manager.theme.border_visible};border:none;}}")
        pp_layout.addWidget(sep3)

        s5 = QLabel("2X SLOWDOWN")
        s5.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        pp_layout.addWidget(s5)
        self._sd_mel = _ModelCheckbox("2x Slowdown MelBand", checked=False)
        pp_layout.addWidget(self._sd_mel)
        self._sd_bs = _ModelCheckbox("2x Slowdown BS RoFormer", checked=False)
        pp_layout.addWidget(self._sd_bs)
        self._sd_mvsep = _ModelCheckbox("2x Slowdown MVSep", checked=False)
        pp_layout.addWidget(self._sd_mvsep)
        self._sd_scnet = _ModelCheckbox("2x Slowdown SCNet", checked=False)
        pp_layout.addWidget(self._sd_scnet)

        sep4 = QFrame()
        sep4.setFixedHeight(1)
        sep4.setStyleSheet(f"QFrame{{background:{theme_manager.theme.border_visible};border:none;}}")
        pp_layout.addWidget(sep4)

        s6 = QLabel("FINISHER VARIANTS")
        s6.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        pp_layout.addWidget(s6)
        self._fv_restore = _ModelCheckbox("Restore Side Variant", checked=True)
        pp_layout.addWidget(self._fv_restore)
        self._fv_mvsep_only = _ModelCheckbox("Variant MVSep Only", checked=True)
        pp_layout.addWidget(self._fv_mvsep_only)
        self._fv_mvsep_resurrect = _ModelCheckbox("Variant MVSep + Resurrect", checked=False)
        pp_layout.addWidget(self._fv_mvsep_resurrect)
        self._fv_lp_hp = _ModelCheckbox("Variant LP MVSep + LP Resurrect + HP V1EP", checked=False)
        pp_layout.addWidget(self._fv_lp_hp)
        self._fv_mvsep_resurrect_hp = _ModelCheckbox("Variant MVSep + Resurrect + HP V1EP", checked=False)
        pp_layout.addWidget(self._fv_mvsep_resurrect_hp)

        left_layout.addWidget(pp_card)

        # ── ADVANCED Card ───────────────────────────────────────────────
        adv_card, adv_layout = self._make_card("Advanced")

        iter_row = QHBoxLayout()
        iter_row.setSpacing(14)
        iter_lbl = QLabel("ITERATIONS")
        iter_lbl.setFixedWidth(95)
        iter_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;font-weight:600;color:{theme_manager.theme.text_label};background:transparent;")
        iter_row.addWidget(iter_lbl)
        self._iter_slider = _SliderWithLabel(1, 5, 4)
        iter_row.addWidget(self._iter_slider, 1)
        adv_layout.addLayout(iter_row)

        worker_row = QHBoxLayout()
        worker_row.setSpacing(14)
        wrk_lbl = QLabel("WORKERS")
        wrk_lbl.setFixedWidth(95)
        wrk_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;font-weight:600;color:{theme_manager.theme.text_label};background:transparent;")
        worker_row.addWidget(wrk_lbl)
        self._worker_slider = _SliderWithLabel(1, 8, 3)
        worker_row.addWidget(self._worker_slider, 1)
        adv_layout.addLayout(worker_row)

        self._delete_prev = _ModelCheckbox("Delete Previous Pass Folder", checked=True)
        adv_layout.addWidget(self._delete_prev)

        left_layout.addWidget(adv_card)

        # ── Info Cards Row ──────────────────────────────────────────────
        info_row = QHBoxLayout()
        info_row.setSpacing(16)

        about_card = QFrame()
        about_card.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.surface};border:none;border-radius:10px;}}"
        )
        about_lo = QVBoxLayout(about_card)
        about_lo.setContentsMargins(16, 12, 16, 12)
        about_lo.setSpacing(6)
        at = QLabel("ABOUT ITERATIVE ENSEMBLE")
        at.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        about_lo.addWidget(at)
        at2 = QLabel("Multi-pass processing with iterative refinement. Combines local models and MVSep API for enhanced separation quality.")
        at2.setWordWrap(True)
        at2.setStyleSheet(f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;")
        about_lo.addWidget(at2)

        notes_card = QFrame()
        notes_card.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.surface};border:none;border-radius:10px;}}"
        )
        notes_lo = QVBoxLayout(notes_card)
        notes_lo.setContentsMargins(16, 12, 16, 12)
        notes_lo.setSpacing(6)
        nt = QLabel("NOTES")
        nt.setStyleSheet(f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;")
        notes_lo.addWidget(nt)
        nt2 = QLabel("\u2022 Higher overlap may improve quality\n\u2022 Internet connection required\n\u2022 Outputs saved automatically")
        nt2.setStyleSheet(f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;")
        notes_lo.addWidget(nt2)

        info_row.addWidget(about_card, 1)
        info_row.addWidget(notes_card, 1)
        left_layout.addLayout(info_row)

        left_layout.addStretch()
        left_scroll.setWidget(left_content)
        content_row.addWidget(left_scroll, 5)

        # ── Right Column — Workflow Preview ─────────────────────────────
        right_panel = self._build_workflow_preview()
        right_panel.setMinimumWidth(240)
        content_row.addWidget(right_panel, 2)

        bg_layout.addLayout(content_row, 1)

        # ── Bottom Action Bar ───────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(12)
        self._start_btn = _StartButton()
        self._start_btn.setMinimumWidth(220)
        self._start_btn.clicked.connect(self._start)
        ctrl.addWidget(self._start_btn)

        self._stop_btn = _StopButton()
        self._stop_btn.setMinimumWidth(100)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self._stop_btn)

        self._pause_btn = _PauseButton()
        self._pause_btn.setMinimumWidth(100)
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        ctrl.addWidget(self._pause_btn)

        open_btn = QPushButton("OPEN OUTPUT")
        open_btn.setMinimumHeight(UIConstants.BTN_HEIGHT)
        open_btn.setMinimumWidth(120)
        open_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        open_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};border:1px solid {theme_manager.theme.border};"
            f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:{UIConstants.BTN_FONT_SIZE}px;"
            f"letter-spacing:1px;border-radius:{UIConstants.ACTION_RADIUS}px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
        )
        open_btn.clicked.connect(self._open_output)
        ctrl.addWidget(open_btn)

        bg_layout.addLayout(ctrl)

        root.addWidget(bg, 1)
        self._update_start_button()

    def reapply_theme(self):
        # ── 1. Re-set page background ──
        self.setStyleSheet(f"background:{theme_manager.theme.bg};")

        # ── 2. Trigger atmospheric background repaint ──
        bg = self.findChild(_AtmosphericBackground)
        if bg:
            bg.update()

        # ── 3. Re-style ALL QComboBox ──
        for cb in self.findChildren(QComboBox):
            cb.setStyleSheet(
                f"QComboBox{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text};"
                f"border:1px solid {theme_manager.theme.border};border-radius:6px;padding:8px 12px;font-size:12px;}}"
                f"QComboBox::drop-down{{border:none;width:24px;}}"
                f"QComboBox QAbstractItemView{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text};"
                f"selection-background-color:{theme_manager.accent};selection-color:{theme_manager._accent_text};"
                f"border:1px solid {theme_manager.theme.border};}}"
            )

        # ── 4. Re-style ALL QPushButton ──
        for btn in self.findChildren(QPushButton):
            if isinstance(btn, _BrowseButton):
                btn.setStyleSheet(
                    f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};"
                    f"border:1px solid {theme_manager.theme.border};"
                    f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:10px;"
                    f"letter-spacing:1px;border-radius:6px;padding:0 20px;}}"
                    f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};"
                    f"border-color:{theme_manager.theme.border_dim};}}"
                    f"QPushButton:pressed{{background:{theme_manager.theme.surface_alt};}}"
                )
            elif isinstance(btn, _StartButton):
                btn.setStyleSheet(
                    f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
                    f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:11px;"
                    f"letter-spacing:2px;border-radius:8px;}}"
                    f"QPushButton:hover{{background:{theme_manager.accent};}}"
                    f"QPushButton:pressed{{background:{theme_manager.accent};}}"
                    f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
                )
            elif isinstance(btn, _StopButton):
                btn.setStyleSheet(
                    f"QPushButton{{background:{theme_manager.theme.error};color:{theme_manager.theme.text};border:none;"
                    f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:11px;"
                    f"letter-spacing:2px;border-radius:8px;}}"
                    f"QPushButton:hover{{background:{theme_manager.theme.error};}}"
                    f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
                )
            elif isinstance(btn, _PauseButton):
                btn.setStyleSheet(
                    f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};"
                    f"border:1px solid {theme_manager.theme.border};"
                    f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:11px;"
                    f"letter-spacing:2px;border-radius:8px;}}"
                    f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
                    f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.disabled_text};}}"
                )
            elif isinstance(btn, _EyeToggle):
                btn.setStyleSheet(
                    f"QPushButton{{background:transparent;border:1px solid {theme_manager.theme.disabled_bg};border-radius:8px;}}"
                    f"QPushButton:hover{{border-color:{theme_manager.theme.text_label};}}"
                )
            elif isinstance(btn, _ToggleCheck):
                btn.update()
            elif "\u2190" in btn.text():
                btn.setStyleSheet(
                    f"QPushButton{{background:transparent;color:{theme_manager.theme.text_muted};"
                    f"border:1px solid {theme_manager.theme.border};"
                    f"font-family:'Montserrat';font-size:11px;border-radius:6px;padding:0 16px;}}"
                    f"QPushButton:hover{{background:{theme_manager.theme.border};color:{theme_manager.theme.text};}}"
                )
            elif "OPEN" in btn.text():
                btn.setStyleSheet(
                    f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};"
                    f"border:1px solid {theme_manager.theme.border};"
                    f"font-family:'Montserrat',sans-serif;font-weight:900;font-size:10px;"
                    f"letter-spacing:1px;border-radius:8px;}}"
                    f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
                )

        # ── 5. Re-style ALL QLabel ──
        for lbl in self.findChildren(QLabel):
            if isinstance(lbl, _SectionLabel):
                lbl.setStyleSheet(
                    f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
                    f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;"
                )
                continue
            # Skip labels managed by _WorkflowStep
            if isinstance(lbl.parent(), _WorkflowStep):
                continue
            txt = lbl.text().strip()
            ss = lbl.styleSheet()

            # Badge labels — handled by _update_mvsep_badge
            if txt == "\u25cf" or txt in ("Disconnected", "MVSep Connected"):
                continue

            # Slider labels (Courier New / 13px)
            if "font-size:13px" in ss:
                lbl.setStyleSheet(
                    f"font-family:'Courier New',monospace;font-size:13px;font-weight:700;"
                    f"color:{theme_manager.accent};background:transparent;border:none;min-width:24px;"
                )
                continue

            # Model checkbox labels (12px, parent is _ModelCheckbox)
            if isinstance(lbl.parent(), _ModelCheckbox):
                lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:12px;"
                    f"color:{theme_manager.theme.text};background:transparent;"
                )
                continue

            # Workflow icons
            if txt in ("\u25b6", "\u21ba", "\u2713"):
                lbl.setStyleSheet(
                    f"font-size:12px;color:{theme_manager.accent};background:transparent;border:none;"
                )
                continue

            # Main title "ITERATIVE ENSEMBLE" (28px, 900)
            if "font-size:28px" in ss:
                lbl.setStyleSheet(
                    f"font-family:'Montserrat',sans-serif;font-size:28px;font-weight:900;"
                    f"color:{theme_manager.theme.text};background:transparent;letter-spacing:-0.5px;"
                )
                continue

            # Card titles & workflow title (accent, 900, letter-spacing:1)
            if "font-weight:900" in ss:
                lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:11px;font-weight:900;"
                    f"color:{theme_manager.accent};background:transparent;letter-spacing:1px;border:none;"
                )
                continue

            # Section headers / info titles (9px, 700, letter-spacing:2)
            if "font-size:9px" in ss and "letter-spacing:2px" in ss:
                lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:9px;font-weight:700;"
                    f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:2px;"
                )
                continue

            # Field labels (11px, 600)
            if "font-size:11px" in ss and "font-weight:600" in ss:
                lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:11px;font-weight:600;"
                    f"color:{theme_manager.theme.text_label};background:transparent;"
                )
                continue

            # Workflow step names (12px, 700)
            if "font-size:12px" in ss and "font-weight:700" in ss:
                lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:12px;font-weight:700;"
                    f"color:{theme_manager.theme.text};background:transparent;border:none;"
                )
                continue

            # Workflow arrows (16px, centred)
            if txt == "\u2193":
                lbl.setStyleSheet(
                    f"font-size:16px;color:{theme_manager.theme.text_muted};"
                    f"background:transparent;border:none;padding:3px 0;"
                )
                lbl.setAlignment(Qt.AlignCenter)
                continue

            # Helper text, info content, workflow descriptions (10px)
            if "font-size:10px" in ss:
                if "padding:0" in ss:
                    lbl.setStyleSheet(
                        f"font-family:'Montserrat';font-size:10px;"
                        f"color:{theme_manager.theme.text_dim};background:transparent;border:none;padding:0;"
                    )
                elif txt.startswith("Multi-pass") or txt.startswith("\u2022"):
                    lbl.setStyleSheet(
                        f"font-family:'Montserrat';font-size:10px;"
                        f"color:{theme_manager.theme.text_dim};background:transparent;"
                    )
                else:
                    lbl.setStyleSheet(
                        f"font-family:'Montserrat';font-size:10px;"
                        f"color:{theme_manager.theme.text_dim};background:transparent;border:none;"
                    )
                continue

            # Display labels (11px fallback)
            if "font-size:11px" in ss:
                lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:11px;"
                    f"color:{theme_manager.theme.text_dim};background:transparent;"
                )
                continue

        # ── 6. Re-style ALL QFrame (separators, cards, workflow steps) ──
        for frm in self.findChildren(QFrame):
            ss = frm.styleSheet()
            if isinstance(frm, (_AtmosphericBackground, _WorkflowStep)) or not ss:
                continue
            # Badge frame — handled by _update_mvsep_badge
            if "border:1px solid" in ss:
                continue
            # Workflow step (border-radius:8px)
            if "border-radius:8px" in ss:
                frm.setStyleSheet(
                    f"QFrame{{background:{theme_manager.theme.surface_alt};border:none;border-radius:8px;}}"
                )
            # Card / panel (border-radius:10px)
            elif "border-radius:10px" in ss or "border-radius" in ss:
                frm.setStyleSheet(
                    f"QFrame{{background:{theme_manager.theme.surface};border:none;border-radius:10px;}}"
                )
            # Separator (border:none, no border-radius)
            elif "border:none" in ss:
                frm.setStyleSheet(
                    f"QFrame{{background:{theme_manager.theme.border_visible};border:none;}}"
                )

        # ── 7. Re-style QScrollArea + scrollbars ──
        for sa in self.findChildren(QScrollArea):
            old = sa.styleSheet()
            if "width:0px" in old:
                sa.setStyleSheet(
                    "QScrollArea{background:transparent;border:none;}"
                    "QScrollBar:vertical{width:0px;background:transparent;}"
                    "QScrollBar::handle:vertical{background:transparent;}"
                    "QScrollBar::add-line:vertical{height:0px;}"
                    "QScrollBar::sub-line:vertical{height:0px;}"
                )
            else:
                sa.setStyleSheet(
                    "QScrollArea{background:transparent;border:none;}"
                    "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
                    f"QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};"
                    "border-radius:2px;min-height:30px;}"
                    f"QScrollBar::handle:vertical:hover{{background:{theme_manager.theme.border_dim};}}"
                    "QScrollBar::add-line:vertical{height:0;}"
                    "QScrollBar::sub-line:vertical{height:0;}"
                    "QScrollBar::add-page:vertical,"
                    "QScrollBar::sub-page:vertical{background:transparent;}"
                )
            sa.viewport().setStyleSheet("background:transparent;border:none;")

        # ── 8. Re-style QLineEdit (API key input) ──
        for le in self.findChildren(QLineEdit):
            le.setStyleSheet(
                f"QLineEdit{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text};"
                f"border:1px solid {theme_manager.theme.border};border-radius:6px;padding:8px 12px;"
                f"font-family:'Courier New',monospace;font-size:12px;}}"
                f"QLineEdit:focus{{border-color:{theme_manager.accent};}}"
            )

        # ── 9. Re-style QSlider ──
        for sl in self.findChildren(QSlider):
            sl.setStyleSheet(
                f"QSlider::groove:horizontal{{height:4px;background:{theme_manager.theme.surface};border-radius:2px;}}"
                f"QSlider::handle:horizontal{{width:16px;height:16px;margin:-6px 0;"
                f"background:{theme_manager.accent};border-radius:8px;}}"
                f"QSlider::sub-page:horizontal{{background:{theme_manager.accent};border-radius:2px;}}"
            )

        # ── 10. Badge ──
        self._update_mvsep_badge()

        self.update()

    def _update_mvsep_badge(self):
        token = self._api_key_row.text().strip()
        if token:
            c = QColor(theme_manager.theme.success)
            self._badge_dot.setStyleSheet(f"font-size:10px;color:{theme_manager.theme.success};background:transparent;border:none;")
            self._badge_text.setText("MVSep Connected")
            self._badge_text.setStyleSheet(f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.success};background:transparent;border:none;")
            self._mvsep_badge.setStyleSheet(
                f"QFrame{{background:rgba({c.red()},{c.green()},{c.blue()},0.08);border:1px solid rgba({c.red()},{c.green()},{c.blue()},0.3);border-radius:14px;}}"
            )
        else:
            self._badge_dot.setStyleSheet(f"font-size:10px;color:{theme_manager.theme.text_muted};background:transparent;border:none;")
            self._badge_text.setText("Disconnected")
            self._badge_text.setStyleSheet(f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_muted};background:transparent;border:none;")
            self._mvsep_badge.setStyleSheet(
                f"QFrame{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.disabled_bg};border-radius:14px;}}"
            )

    def _make_card(self, title):
        card = QFrame()
        card.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.surface};border:none;border-radius:10px;}}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 12, 20, 14)
        layout.setSpacing(10)
        if title:
            lbl = QLabel(title.upper())
            lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:11px;font-weight:900;"
                f"color:{theme_manager.accent};background:transparent;letter-spacing:1px;border:none;"
            )
            layout.addWidget(lbl)
        return card, layout

    def _build_workflow_preview(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:0px;background:transparent;}"
            "QScrollBar::handle:vertical{background:transparent;}"
            "QScrollBar::add-line:vertical{height:0px;}"
            "QScrollBar::sub-line:vertical{height:0px;}"
        )
        scroll.viewport().setStyleSheet("background:transparent;border:none;")

        panel = QFrame()
        panel.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.surface};border:none;border-radius:10px;}}"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(0)

        wf_title = QLabel("WORKFLOW PREVIEW")
        wf_title.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;font-weight:900;"
            f"color:{theme_manager.accent};background:transparent;letter-spacing:1px;border:none;"
        )
        layout.addWidget(wf_title)
        layout.addSpacing(16)

        steps_data = [
            ("\u25b6", "Local Inference", "Run selected models \u2192 instrumental stems"),
            ("\u25b6", "MVSep API Processing", "Upload \u2192 process \u2192 download stem"),
            ("\u25b6", "Ensemble & Attenuation", "Merge stems, reduce vocals 50%"),
            ("\u21ba", "Repeat", "Loop for N\u20131 iterations"),
            ("\u25b6", "Finisher Pass", "MVSep API final cleanup"),
            ("\u2713", "Output Saved", "Separated stems written to disk"),
        ]

        self._workflow_steps = []
        for icon, name, desc in steps_data:
            step = _WorkflowStep(icon, name, desc)
            self._workflow_steps.append(step)
            layout.addWidget(step)
            arr = QLabel("\u2193")
            arr.setAlignment(Qt.AlignCenter)
            arr.setStyleSheet(f"font-size:16px;color:{theme_manager.theme.text_muted};background:transparent;border:none;padding:3px 0;")
            layout.addWidget(arr)
        # remove the trailing arrow
        trailing_arr = layout.takeAt(layout.count() - 1)
        if trailing_arr and trailing_arr.widget():
            trailing_arr.widget().deleteLater()

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _browse_input(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio files (*.wav *.flac *.mp3 *.ogg *.aiff *.m4a *.opus *.wv);;All files (*.*)"
        )
        if paths:
            self._input_files = paths
            names = [os.path.basename(p) for p in paths]
            if len(names) <= 3:
                self._input_display.setText(", ".join(names))
            else:
                self._input_display.setText(f"{len(names)} files selected")
            self._update_start_button()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", "./iterative_output/")
        if path:
            self._output_dir = path
            self._output_display.setText(path)
        else:
            self._output_dir = os.path.join(os.getcwd(), "iterative_output")
            self._output_display.setText("./iterative_output/")

    def _update_start_button(self):
        ready = len(self._input_files) >= 1
        self._start_btn.setEnabled(ready and not self._running)

    def _get_config(self):
        def build_model_entry(checkbox, ckpt_filename):
            reg = self._get_registered_model(ckpt_filename)
            if reg:
                return {
                    "enabled": checkbox.isChecked(),
                    "name": reg.get("name", ckpt_filename),
                    "ckpt": reg.get("ckpt", ""),
                    "yaml": reg.get("yaml", ""),
                    "arch": reg.get("arch", "BS Roformer Architecture"),
                    "stem_type": reg.get("type", "instrumental"),
                }
            return {
                "enabled": checkbox.isChecked(),
                "name": ckpt_filename,
                "ckpt": "",
                "yaml": "",
                "arch": "BS Roformer Architecture",
                "stem_type": "instrumental",
            }

        return {
            "input_files": self._input_files,
            "output_dir": self._output_dir or os.path.join(os.getcwd(), "iterative_output"),
            "export_format": self._export_combo.currentText(),
            "overlap": self._overlap_slider.value(),
            "mvsep_token": self._api_key_row.text(),
            "api_no_credits": self._api_no_credits.isChecked(),
            "restore_side": self._restore_side.isChecked(),
            "amplify_masked": self._amplify_masked.isChecked(),
            "auto_trim": self._auto_trim.isChecked(),
            "auto_trim_model": self._auto_trim_model.isChecked(),
            "models_local": {
                "mel_v1e": build_model_entry(self._mel_v1e, "inst_v1e.ckpt"),
                "bs_resurrect": build_model_entry(self._bs_resurrect, "BS-Roformer-Resurrection-Inst.ckpt"),
            },
            "models_api": {
                "mvsep_2025_07": {"enabled": self._api_mvsep.isChecked()},
                "scnet_becruily": {"enabled": self._api_scnet.isChecked()},
            },
            "post_separate": {
                "bs_resurrect": self._post_bs.isChecked(),
                "scnet": self._post_scnet.isChecked(),
            },
            "use_slowdown": {
                "mel_v1e": self._sd_mel.isChecked(),
                "bs_resurrect": self._sd_bs.isChecked(),
                "mvsep": self._sd_mvsep.isChecked(),
                "scnet": self._sd_scnet.isChecked(),
            },
            "finisher_variants": [v for v, cb in [
                ("mvsep_only", self._fv_mvsep_only),
                ("mvsep_plus_resurrect", self._fv_mvsep_resurrect),
                ("lp_mvsep_lp_resurrect_hp_v1ep", self._fv_lp_hp),
                ("mvsep_resurrect_hp_v1ep", self._fv_mvsep_resurrect_hp),
            ] if cb.isChecked()],
            "iterations": self._iter_slider.value(),
            "worker_count": self._worker_slider.value(),
            "delete_prev_pass": self._delete_prev.isChecked(),
        }

    def _start(self):
        from ui.widgets.runtime_dialog import ensure_runtime
        if not ensure_runtime(self):
            return
        if not self._input_files:
            self.log_output.emit("ERROR: No input files selected.")
            return

        self.input_files_submitted.emit(self._input_files)

        config = self._get_config()
        has_api_models = any(v.get("enabled") for v in config.get("models_api", {}).values())
        if has_api_models and not config.get("mvsep_token"):
            self.log_output.emit("ERROR: MVSep models selected but no API key provided.")
            return

        self.reset_workflow()
        self._running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._pause_btn.setEnabled(True)
        self.log_output.emit("Starting iterative ensemble...")

        self._log_debug_config(config)

        self._runner = IterativeEnsembleRunner(config)
        self._runner.stage_changed.connect(self._on_stage)
        self._runner.file_progress.connect(self._on_file_progress)
        self._runner.iteration_progress.connect(self._on_iteration_progress)
        self._runner.model_progress.connect(self._on_model_progress)
        self._runner.log_line.connect(self.log_output.emit)
        self._runner.finished.connect(self._on_finished)
        self._runner.error.connect(self._on_error)
        self.process_running.emit(True)
        self._runner.start()

    def _log_debug_config(self, config):
        self.log_output.emit("=== ITERATIVE ENSEMBLE CONFIG ===")
        self.log_output.emit(f"Input files: {len(config.get('input_files', []))}")
        self.log_output.emit(f"Output dir: {config.get('output_dir', '')}")
        self.log_output.emit(f"Iterations: {config.get('iterations', 4)}")
        self.log_output.emit(f"MVSEP token: {'set' if config.get('mvsep_token') else 'empty'}")
        local = config.get('models_local', {})
        for key, model in local.items():
            self.log_output.emit(f"  Local model {key}: enabled={model.get('enabled')}, ckpt={model.get('ckpt', 'MISSING')[:50]}")
        api = config.get('models_api', {})
        for key, val in api.items():
            self.log_output.emit(f"  API model {key}: enabled={val.get('enabled')}")
        self.log_output.emit("=================================")

    def _stop(self):
        if self._runner:
            self._runner.cancel()
        self.process_running.emit(False)
        self.log_output.emit("Stopping...")

    def _toggle_pause(self):
        if not self._runner:
            return
        if self._runner._paused:
            self._runner.resume()
            self._pause_btn.setText("PAUSE")
            self.log_output.emit("Resumed")
        else:
            self._runner.pause()
            self._pause_btn.setText("RESUME")
            self.log_output.emit("Paused")

    def reset_workflow(self):
        self._active_workflow_step = -1
        for step in self._workflow_steps:
            step.reset()

    def _on_stage(self, stage, current, total):
        self.log_output.emit(f"Stage: {stage} ({current}/{total})")
        idx = -1
        for i, step in enumerate(self._workflow_steps):
            if step._name == stage:
                idx = i
                break
        if idx < 0:
            return
        if self._active_workflow_step >= 0:
            prev = self._active_workflow_step
            if prev < len(self._workflow_steps):
                self._workflow_steps[prev].set_state(_WorkflowStep.COMPLETED)
        self._active_workflow_step = idx
        self._workflow_steps[idx].set_state(_WorkflowStep.ACTIVE)

    def _on_file_progress(self, song_name, pct):
        self.log_output.emit(f"Processing: {song_name} - {pct}%")

    def _on_iteration_progress(self, current, total):
        self.log_output.emit(f"Iteration {current}/{total}")

    def _on_model_progress(self, model_name, stage, pct):
        self.log_output.emit(f"{model_name} ({stage}): {pct}%")

    def _on_finished(self, success, message, output_path):
        self._running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("PAUSE")
        self.process_running.emit(False)
        if success:
            for step in self._workflow_steps:
                step.set_state(_WorkflowStep.COMPLETED)
        elif self._active_workflow_step >= 0:
            self._workflow_steps[self._active_workflow_step].set_state(_WorkflowStep.ERROR)
        self.log_output.emit(message)

    def _on_error(self, message):
        self.log_output.emit(f"ERROR: {message}")

    def _open_output(self):
        path = self._output_dir or os.path.join(os.getcwd(), "iterative_output")
        if os.path.isdir(path):
            os.startfile(path)

    def load_settings(self, settings):
        if settings.get("mvsep_token"):
            self._api_key_row.setText(settings["mvsep_token"])
            self._update_mvsep_badge()
        if settings.get("output_dir"):
            self._output_dir = settings["output_dir"]
            self._output_display.setText(settings["output_dir"])

    def save_settings(self):
        return {
            "mvsep_token": self._api_key_row.text(),
            "output_dir": self._output_dir,
        }
