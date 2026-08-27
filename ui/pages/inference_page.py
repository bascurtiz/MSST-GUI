"""
ui/pages/inference_page.py
Premium cinematic dark UI — 2 column layout.
"""
import os, sys

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QLineEdit, QFileDialog,
    QScrollArea, QSizePolicy, QSpacerItem, QDialog,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal, Property, QEasingCurve, QSize, QPoint, QPropertyAnimation, QVariantAnimation, QEvent
from PySide6.QtGui import QFont, QPainter, QPen, QColor

from backend.runner import ProcessRunner
from backend.paths import REPO_ROOT, get_python_exe
from backend.gpu_utils import list_gpus, device_ids_from_selection
from backend import settings as settings_store
from backend.yaml_analyzer import classify_model_type, get_stems_for_type
from ui.theme import theme_manager, UIConstants, FONT_STACK
from ui.widgets.common import (
    ConsoleLog, SpectrogramPanel, WaveformPanel, ProcessingStatusPanel, PageHeader,
)

AUDIO_FILTER = ("Audio files (*.wav *.flac *.mp3 *.ogg *.aiff *.m4a *.opus *.wv);;"
                "All files (*.*)")
ARCH_TYPES = [
    "MDX Architecture", "Demucs Architecture",
    "BS Roformer Architecture", "Melband Roformer Architecture",
    "SCNet Architecture", "Apollo Architecture", "Bandit Architecture",
]
ARCH_TO_MODEL_TYPE = {
    "MDX Architecture": "mdx23c",
    "Demucs Architecture": "htdemucs",
    "BS Roformer Architecture": "bs_roformer",
    "Melband Roformer Architecture": "mel_band_roformer",
    "SCNet Architecture": "scnet",
    "Apollo Architecture": "apollo",
    "Bandit Architecture": "bandit",
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _sec_hdr(text):
    w = QLabel(text.upper())
    w.setStyleSheet(
        "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
        f"color:{theme_manager.theme.text};background:transparent;padding-left:8px;"
        f"border-left:3px solid {theme_manager.accent};letter-spacing:1.5px;"
    )
    w.setFixedHeight(18)
    return w

def _hdiv(opacity=20):
    f = QFrame(); f.setFixedHeight(1)
    t = theme_manager.theme
    if opacity == 6:
        color = t.border
    elif opacity >= 12:
        color = t.border_dim
    else:
        c = QColor(t.text)
        c.setAlpha(int(opacity * 2.55))
        color = f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha()})"
    f.setStyleSheet(f"background:{color};border:none;")
    return f


# ── Search Icon ───────────────────────────────────────────────────────────────

class _SearchIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(120)
        p.setPen(QPen(_c, 1.8))
        p.drawEllipse(2, 2, 11, 11)
        p.drawLine(11, 11, 17, 17)
        p.end()


# ── Search Bar ─────────────────────────────────────────────────────────────────

class _SearchBar(QFrame):
    textChanged = Signal(str)

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("QFrame{background:transparent;border:none;}")

        # The QLineEdit IS the field (same recipe as the SETTINGS search): it
        # carries the background, border, hover and focus styles. Icon and
        # clear button are overlaid on it so they sit inside the rounded
        # field; Qt ignores `:focus-within`, so the input must be the widget
        # that actually receives focus for its :focus border to show.
        self._input = QLineEdit(self)
        self._input.setPlaceholderText(placeholder)
        self._input.setStyleSheet(self._input_ss())
        self._input.textChanged.connect(self.textChanged)

        self._icon = _SearchIcon(self._input)
        self._icon.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._clear_btn = QPushButton("\u2715", self._input)
        self._clear_btn.setFixedSize(22, 22)
        self._clear_btn.setStyleSheet(self._clear_btn_ss())
        self._clear_btn.clicked.connect(self._input.clear)
        self._clear_btn.setVisible(False)
        self._input.textChanged.connect(
            lambda t: self._clear_btn.setVisible(bool(t))
        )

        self._layout_children()

    def sizeHint(self):
        # No layout anymore (children are overlaid on the input), so the
        # frame would otherwise collapse to zero width in its parent layout.
        return QSize(155, 40)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_input"):
            self._layout_children()

    def _layout_children(self):
        self._input.setGeometry(0, 0, self.width(), self.height())
        cy = self.height() // 2
        self._icon.move(10, cy - 10)
        self._clear_btn.move(self.width() - 28, cy - 11)

    def _input_ss(self):
        t = theme_manager.theme
        return (
            f"QLineEdit{{background:{t.surface_alt};"
            f"border:1px solid {t.border};border-radius:8px;"
            "font-family:'Montserrat';font-size:11px;"
            f"color:{t.text_dim};padding:0 32px 0 34px;"
            f"selection-background-color:{theme_manager.accent};"
            f"selection-color:{theme_manager._accent_text};}}"
            f"QLineEdit:hover{{background:{t.input_hover};}}"
            f"QLineEdit:focus{{border:1px solid {theme_manager.accent};}}"
            f"QLineEdit::placeholder{{color:{t.text_muted};}}"
        )

    def _clear_btn_ss(self):
        t = theme_manager.theme
        return (
            "QPushButton{background:transparent;border:none;"
            f"color:{t.text_muted};font-size:10px;border-radius:3px;}}"
            f"QPushButton:hover{{color:{t.text};background:{t.border};}}"
        )

    def reapply_theme(self):
        self._input.setStyleSheet(self._input_ss())
        self._clear_btn.setStyleSheet(self._clear_btn_ss())

    def text(self):
        return self._input.text()

    def clear(self):
        self._input.clear()

    def setFocus(self):
        self._input.setFocus()


# ── Filter Button ──────────────────────────────────────────────────────────────

class _FilterButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.setToolTip("Filter models")
        self.setCursor(Qt.PointingHandCursor)
        self._apply_styles()

    def _apply_styles(self):
        t = theme_manager.theme
        self.setStyleSheet(
            "QPushButton{background:transparent;"
            f"border:1px solid {t.border};"
            "border-radius:6px;}"
            "QPushButton:hover{"
            f"background:{t.surface_alt};"
            f"border:1px solid {t.border_dim};}}"
            f"QPushButton:pressed{{background:{t.border_dim};border-color:{t.border_dim};}}"
        )

    def reapply_theme(self):
        self._apply_styles()

    def paintEvent(self, event):
        super().paintEvent(event)
        from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(120)
        p.setPen(QPen(_c, 1.5))

        path = QPainterPath()
        path.moveTo(12, 11)
        path.lineTo(28, 11)
        path.lineTo(22, 19)
        path.lineTo(22, 27)
        path.lineTo(18, 27)
        path.lineTo(18, 19)
        path.closeSubpath()
        p.drawPath(path)
        p.end()


# ── Icon widgets ──────────────────────────────────────────────────────────────

class _ArrowIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self._angle = 0
        self._anim = None

    def set_down(self, down):
        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        target = 90 if down else 0
        if self._anim:
            self._anim.stop()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setStartValue(self._angle)
        self._anim.setEndValue(target)
        self._anim.valueChanged.connect(lambda v: (setattr(self, '_angle', v), self.update()))
        self._anim.start()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.translate(10, 10)
        p.rotate(self._angle)
        p.translate(-10, -10)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(120)
        p.setPen(QPen(_c, 1.5))
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(6, 5)
        path.lineTo(14, 10)
        path.lineTo(6, 15)
        p.drawPath(path)
        p.end()


class _ComboBox(QComboBox):
    popupOpened = Signal()
    popupClosed = Signal()

    def showPopup(self):
        super().showPopup()
        self.popupOpened.emit()

    def hidePopup(self):
        super().hidePopup()
        self.popupClosed.emit()


# ── Animated icon button ────────────────────────────────────────────────────────
# QPushButton whose leading glyph icon animates on hover. The icon lives in a
# transparent-for-mouse QLabel child, so the button's stylesheet keeps drawing
# the text as usual while the icon moves/pulses independently.

# ── Custom QPainter icons ───────────────────────────────────────────────────────

class _UploadIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(115)
        pen = QPen(_c, 1.2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(12, 18, 12, 6)
        p.drawLine(12, 6, 7, 11)
        p.drawLine(12, 6, 17, 11)
        p.end()


class _FolderIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(115)
        pen = QPen(_c, 1.2)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(5, 9)
        path.lineTo(10, 9)
        path.lineTo(12, 12)
        path.lineTo(20, 12)
        path.lineTo(20, 19)
        path.lineTo(4, 19)
        path.lineTo(4, 10)
        path.closeSubpath()
        p.drawPath(path)
        p.end()


class _WaveIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(115)
        pen = QPen(_c, 1.2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(3, 12)
        path.cubicTo(5, 9, 7, 15, 9, 12)
        path.cubicTo(11, 9, 13, 15, 15, 12)
        path.cubicTo(17, 9, 19, 15, 21, 12)
        p.drawPath(path)
        p.end()


class _DiamondIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(115)
        pen = QPen(_c, 1.2)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(12, 8)
        path.lineTo(16, 12)
        path.lineTo(12, 16)
        path.lineTo(8, 12)
        path.closeSubpath()
        p.drawPath(path)
        p.end()


class _ChipIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(115)
        pen = QPen(_c, 1.2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(6, 6, 12, 12, 3, 3)
        p.drawRoundedRect(9, 9, 6, 6, 2, 2)
        for x in (9, 15):
            p.drawLine(x, 6, x, 4)
            p.drawLine(x, 18, x, 20)
        for y in (9, 15):
            p.drawLine(6, y, 4, y)
            p.drawLine(18, y, 20, y)
        p.end()


class _TargetIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(115)
        pen = QPen(_c, 1.4)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(5, 5, 14, 14)
        p.drawEllipse(8, 8, 8, 8)
        p.drawEllipse(10, 10, 4, 4)
        p.end()


# ── Config rows ────────────────────────────────────────────────────────────────

def _row_ss():
    c = QColor(theme_manager.accent)
    t = theme_manager.theme
    return (
        "QFrame#cfgRow{"
        "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        f"stop:0 {t.surface},stop:1 {t.bg});"
        f"border:1px solid {t.border_visible};"
        f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px;}}"
        "QFrame#cfgRow:hover{"
        f"border:1px solid {theme_manager.accent};"
        f"background:{theme_manager._accent_soft};}}"
    )

def _icon_ss():
    t = theme_manager.theme
    return (
        "font-size:13px;" + f"color:{t.text_sec};"
        "background:transparent;border:none;"
    )

def _lbl_ss():
    t = theme_manager.theme
    return (
        f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_HDR_FONT_SIZE}px;font-weight:700;"
        f"color:{t.text_sec};background:transparent;letter-spacing:1.5px;"
    )

def _combo_ss():
    t = theme_manager.theme
    c = QColor(theme_manager.accent)
    return (
        f"QComboBox{{background:transparent;border:none;"
        f"font-family:'Montserrat';font-size:11px;color:{t.text};padding:0 4px;}}"
        f"QComboBox::drop-down{{width:0;border:none;}}"
        f"QComboBox::down-arrow{{width:0;height:0;border:none;}}"
        f"QComboBox QAbstractItemView{{"
        f"background:{t.surface_alt};"
        f"border:1px solid {t.border_dim};"
        f"color:{t.text};selection-background-color:{theme_manager.accent};"
        f"selection-color:{theme_manager._accent_text};outline:none;}}"
        f"QComboBox QAbstractItemView::item{{padding:6px 12px;min-height:26px;}}"
        f"QComboBox QAbstractItemView::item:hover{{background:{theme_manager._accent_soft};color:{t.text};}}"
    )
ROW_H = 46


class _EllipsisButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 34)
        self.setCursor(Qt.PointingHandCursor)
        self._hovered = False

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        from PySide6.QtCore import QPointF
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        alpha = 170 if self._hovered else 110
        dot_r = 1.6
        gap = 5
        total_w = 3 * (dot_r * 2) + 2 * gap
        ox = (34 - total_w) / 2
        oy = (34 - dot_r * 2) / 2
        p.setPen(Qt.NoPen)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(alpha)
        p.setBrush(_c)
        for i in range(3):
            cx = ox + dot_r + i * (dot_r * 2 + gap)
            cy = oy + dot_r
            p.drawEllipse(QPointF(cx, cy), dot_r, dot_r)
        p.end()

    def enterEvent(self, e):
        self._hovered = True; self.update(); super().enterEvent(e)
    def leaveEvent(self, e):
        self._hovered = False; self.update(); super().leaveEvent(e)


class _BrowseRow(QFrame):
    def __init__(self, icon, label, placeholder, mode="file",
                 file_filter="All (*.*)", parent=None):
        super().__init__(parent)
        self._mode   = mode
        self._filter = file_filter
        self._files  = []
        self._drag_over = False
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setAcceptDrops(True)
        self.setStyleSheet(_row_ss())

        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 0, 4, 0)
        hl.setSpacing(0)

        if icon is not None:
            icon.setFixedWidth(24)
            hl.addWidget(icon)
            hl.addSpacing(6)

        lb = QLabel(label.upper())
        lb.setStyleSheet(_lbl_ss())
        lb.setFixedWidth(80)
        lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hl.addWidget(lb)

        self._edit = QLineEdit()
        self._edit.setReadOnly(True)
        self._edit.setAcceptDrops(False)  # let drops reach the row itself
        self._edit.setPlaceholderText(placeholder)
        self._edit.setStyleSheet(
            "QLineEdit{background:transparent;border:none;"
            "font-family:'Montserrat';font-size:11px;"
            f"color:{theme_manager.theme.text};padding:0;}}"
            f"QLineEdit::placeholder{{color:{theme_manager.theme.text_muted};}}"
        )
        self._edit.setFixedHeight(ROW_H)
        hl.addWidget(self._edit, 1)

        btn = _EllipsisButton()
        btn.clicked.connect(self._browse)
        hl.addWidget(btn)

    def _browse(self):
        if self._mode == "folder":
            path = QFileDialog.getExistingDirectory(
                self, "Select folder", self._edit.text() or "")
            if path:
                self._files = []
                self._edit.setText(path)
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select audio file(s)",
                os.path.dirname(self._files[0]) if self._files else "",
                self._filter)
            if files:
                self._files = files
                self._edit.setText(
                    os.path.basename(files[0]) if len(files) == 1
                    else f"{len(files)} files selected")

    def value(self):
        return self._files if self._mode != "folder" else self._edit.text()

    # ── Drag & drop ──

    def _drag_ss(self):
        t = theme_manager.theme
        return (
            "QFrame#cfgRow{"
            f"background:{theme_manager._accent_soft};"
            f"border:1px solid {theme_manager.accent};"
            f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px;}}"
        )

    def _set_drag_over(self, over):
        if self._drag_over != over:
            self._drag_over = over
            self.setStyleSheet(self._drag_ss() if over else _row_ss())

    def _local_paths(self, mime):
        return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]

    def _accepts(self, mime):
        paths = self._local_paths(mime)
        if not paths:
            return False
        if self._mode == "folder":
            return True  # a dropped file selects its containing folder
        return any(os.path.isfile(p) for p in paths)

    def dragEnterEvent(self, e):
        if self._accepts(e.mimeData()):
            self._set_drag_over(True)
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if self._accepts(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self._set_drag_over(False)
        super().dragLeaveEvent(e)

    def dropEvent(self, e):
        self._set_drag_over(False)
        paths = self._local_paths(e.mimeData())
        if not paths:
            e.ignore()
            return
        if self._mode == "folder":
            p = paths[0]
            folder = p if os.path.isdir(p) else os.path.dirname(p)
            if folder:
                self._files = []
                self._edit.setText(folder)
            e.acceptProposedAction()
        else:
            files = [p for p in paths if os.path.isfile(p)]
            if files:
                self._files = files
                self._edit.setText(
                    os.path.basename(files[0]) if len(files) == 1
                    else f"{len(files)} files selected")
            e.acceptProposedAction()

    def set_value(self, v):
        if isinstance(v, list):
            self._files = v
            if v:
                self._edit.setText(
                    os.path.basename(v[0]) if len(v) == 1
                    else f"{len(v)} files selected")
        elif isinstance(v, str) and v:
            self._edit.setText(v)


class _ComboRow(QFrame):
    def __init__(self, icon, label, items):
        super().__init__()
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        self.setCursor(Qt.PointingHandCursor)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 0, 10, 0)
        hl.setSpacing(0)

        if icon is not None:
            icon.setFixedWidth(24)
            hl.addWidget(icon)
            hl.addSpacing(6)

        lb = QLabel(label.upper())
        lb.setStyleSheet(_lbl_ss())
        lb.setFixedWidth(80)
        lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hl.addWidget(lb)

        self.combo = _ComboBox()
        self.combo.addItems(items)
        self.combo.setStyleSheet(_combo_ss())
        hl.addWidget(self.combo, 1)

        self._arrow = _ArrowIcon()
        hl.addWidget(self._arrow)
        self.combo.popupOpened.connect(lambda: self._arrow.set_down(True))
        self.combo.popupClosed.connect(lambda: self._arrow.set_down(False))

    def mousePressEvent(self, e):
        self.combo.showPopup()
        super().mousePressEvent(e)


# ── Stem Selection Dialog ───────────────────────────────────────────────────

class _StemTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self._drag_pos = None
        # QWidget subclasses skip stylesheet backgrounds unless this is set —
        # without it the bar renders unpainted (black) with dark text on top.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "QWidget{background:" + theme_manager.theme.bg + ";"
            "border-bottom:1px solid " + theme_manager.theme.border + ";}"
        )
        hl = QHBoxLayout(self)
        hl.setContentsMargins(20, 0, 12, 0)
        hl.setSpacing(0)
        title_lbl = QLabel("OUTPUT STEMS")
        title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
            "color:" + theme_manager.theme.text + ";"
            "background:transparent;letter-spacing:1px;"
        )
        hl.addWidget(title_lbl)
        hl.addStretch()
        _err = QColor(theme_manager.theme.error)
        self._close_btn = QPushButton("✕")
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.setStyleSheet(
            "QPushButton{background:transparent;"
            "color:" + theme_manager.theme.text_muted + ";"
            "border:none;font-size:14px;border-radius:4px;}"
            "QPushButton:hover{"
            "background:rgba(" + str(_err.red()) + "," + str(_err.green()) + ","
            + str(_err.blue()) + ",0.20);"
            "color:" + theme_manager.theme.error + ";}"
        )
        self._close_btn.clicked.connect(parent.reject)
        hl.addWidget(self._close_btn)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.parent().frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() == Qt.LeftButton:
            self.parent().move(e.globalPos() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_pos = None


class _StemSelectionDialog(QDialog):
    def __init__(self, all_stems, selected_stems, save_rest, primary_target=None, parent=None):
        super().__init__(parent)
        self._all_stems = all_stems[:]
        self._primary_target = primary_target.lower() if primary_target else None
        self._orig_selected = set(selected_stems)
        self._orig_save_rest = save_rest
        self._selected = set(selected_stems)
        self._save_rest = save_rest
        self._search_text = ""

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        # ensure the QDialog{background;...} rule actually paints
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumSize(420, 320)
        self.setStyleSheet(
            "QDialog{background:" + theme_manager.theme.bg + ";"
            "border:1px solid " + theme_manager.theme.border_dim + ";"
            "border-radius:8px;}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._title_bar = _StemTitleBar(self)
        root.addWidget(self._title_bar)

        vl = QVBoxLayout()
        vl.setContentsMargins(16, 10, 16, 12)
        vl.setSpacing(8)

        if len(all_stems) > 8:
            self._search_input = QLineEdit()
            self._search_input.setPlaceholderText("Search stems\u2026")
            self._search_input.setStyleSheet(
                "QLineEdit{"
                f"background:{theme_manager.theme.surface_alt};"
                f"border:1px solid {theme_manager.theme.border};"
                "border-radius:6px;padding:6px 10px;"
                f"color:{theme_manager.theme.text};"
                "font-family:'Montserrat';font-size:11px;}"
                f"QLineEdit::placeholder{{color:{theme_manager.theme.text_muted};}}"
            )
            self._search_input.textChanged.connect(self._on_search)
            vl.addWidget(self._search_input)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setMaximumHeight(240)
        self._scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
            "QScrollBar::handle:vertical{"
            f"background:{theme_manager.theme.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
        )

        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet("background:transparent;")
        self._grid = QVBoxLayout(self._scroll_content)
        self._grid.setContentsMargins(0, 4, 0, 4)
        self._grid.setSpacing(4)
        self._scroll.setWidget(self._scroll_content)
        vl.addWidget(self._scroll, 1)

        self._checkboxes = []
        self._rebuild_checkboxes()

        rest_row = QHBoxLayout()
        rest_row.setContentsMargins(0, 4, 0, 0)
        rest_row.setSpacing(6)
        self._rest_cb = _CircleCheck()
        self._rest_cb.set_checked(self._save_rest)
        self._rest_cb.toggled.connect(self._on_rest_toggled)
        rest_row.addWidget(self._rest_cb)
        rest_lbl = QLabel("Save rest (mix \u2212 selected stems)")
        rest_lbl.setStyleSheet(
            "font-family:'Montserrat';font-size:11px;"
            f"color:{theme_manager.theme.text_dim};background:transparent;"
        )
        rest_row.addWidget(rest_lbl)
        rest_row.addStretch()
        vl.addLayout(rest_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._sel_all_btn = QPushButton("Select All")
        self._sel_all_btn.setStyleSheet(self._action_btn_ss())
        self._sel_all_btn.clicked.connect(self._select_all)
        btn_row.addWidget(self._sel_all_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setStyleSheet(self._action_btn_ss())
        self._clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        vl.addLayout(btn_row)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{theme_manager.theme.border};border:none;")
        vl.addWidget(sep)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.setStyleSheet(
            "QPushButton{"
            f"background:{theme_manager.theme.surface_alt};"
            f"color:{theme_manager.theme.text};"
            "border:none;border-radius:4px;padding:6px 18px;"
            "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:600;}"
            "QPushButton:hover{"
            f"background:{theme_manager.theme.border};}}"
        )
        vl.addWidget(bb)
        root.addLayout(vl)

    def _action_btn_ss(self):
        t = theme_manager.theme
        return (
            "QPushButton{"
            f"background:{t.surface_alt};"
            f"color:{t.text_dim};"
            "border:none;border-radius:4px;padding:4px 12px;"
            "font-family:'Montserrat';font-size:10px;font-weight:600;}"
            "QPushButton:hover{"
            f"background:{t.border};color:{t.text};}}"
        )

    def _rebuild_checkboxes(self):
        for cb in self._checkboxes:
            self._grid.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes = []

        text = self._search_text.lower().strip()
        filtered = [s for s in self._all_stems
                    if not text or text in s.lower()]

        if not filtered:
            empty = QLabel("No matching stems.")
            empty.setStyleSheet(
                f"color:{theme_manager.theme.text_muted};"
                "font-family:'Montserrat';font-size:10px;padding:8px 0;"
            )
            self._grid.addWidget(empty)
            return

        # Sort: primary target first, then alphabetically
        if self._primary_target:
            def sort_key(s):
                return (0 if s.lower() == self._primary_target else 1, s.lower())
            filtered.sort(key=sort_key)

        per_row = 4
        for i in range(0, len(filtered), per_row):
            chunk = filtered[i:i + per_row]
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for stem in chunk:
                cb = _CircleCheck()
                cb.set_checked(stem.lower() in self._selected)
                cb.stem_name = stem
                cb.toggled.connect(lambda checked, s=stem: self._on_stem_toggled(s, checked))
                row.addWidget(cb)
                lbl = QLabel(stem.capitalize())
                lbl.setStyleSheet(
                    "font-family:'Montserrat';font-size:11px;"
                    f"color:{theme_manager.theme.text};background:transparent;"
                )
                row.addWidget(lbl)
                if self._primary_target and stem.lower() == self._primary_target:
                    tag = QLabel("target")
                    _ac = QColor(theme_manager.accent)
                    tag.setStyleSheet(
                        "font-family:'Montserrat';font-size:8px;font-weight:700;"
                        f"color:rgba({_ac.red()},{_ac.green()},{_ac.blue()},0.80);"
                        f"background:rgba({_ac.red()},{_ac.green()},{_ac.blue()},0.10);"
                        "padding:1px 5px;border-radius:3px;"
                    )
                    tag.setFixedHeight(16)
                    row.addWidget(tag)
                row.addSpacing(4)
                self._checkboxes.append(cb)
            row.addStretch()
            self._grid.addLayout(row)

        self._grid.addStretch()

    def _on_stem_toggled(self, stem, checked):
        if checked:
            self._selected.add(stem.lower())
        else:
            self._selected.discard(stem.lower())

    def _on_rest_toggled(self, checked):
        self._save_rest = checked

    def _on_search(self, text):
        self._search_text = text
        self._rebuild_checkboxes()

    def _select_all(self):
        self._selected = set(s.lower() for s in self._all_stems)
        self._sync_checkboxes()

    def _clear_all(self):
        self._selected = set()
        self._sync_checkboxes()

    def _sync_checkboxes(self):
        for cb in self._checkboxes:
            cb.set_checked(cb.stem_name.lower() in self._selected)

    def get_selected_stems(self):
        return sorted(self._selected)

    def get_save_rest(self):
        return self._save_rest

    def was_modified(self):
        return (self._selected != self._orig_selected or
                self._save_rest != self._orig_save_rest)


# ── Output Stems Row (replaces _TargetRow) ─────────────────────────────────

class _OutputStemsRow(QFrame):
    def __init__(self, icon, label, parent=None):
        super().__init__(parent)
        self._all_stems = []
        self._selected_stems = set()
        self._save_rest = False
        self._primary_target = None
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        self.setCursor(Qt.PointingHandCursor)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 0, 10, 0)
        hl.setSpacing(0)

        if icon is not None:
            icon.setFixedWidth(24)
            hl.addWidget(icon)
            hl.addSpacing(6)

        lb = QLabel(label.upper())
        lb.setStyleSheet(_lbl_ss())
        lb.setFixedWidth(80)
        lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hl.addWidget(lb)

        self._summary = QLabel("Select stems\u2026")
        self._summary.setStyleSheet(
            "font-family:'Montserrat';font-size:11px;"
            f"color:{theme_manager.theme.text_dim};background:transparent;"
            "padding:0;"
        )
        hl.addWidget(self._summary, 1)

        self._arrow = _ArrowIcon()
        hl.addWidget(self._arrow)

    def set_stems(self, all_stems, primary_target=None):
        self._all_stems = all_stems[:] if all_stems else []
        self._primary_target = primary_target.lower() if primary_target else None
        self._selected_stems = set()
        self._save_rest = False
        self._update_summary()

    def _update_summary(self):
        n = len(self._selected_stems)
        total = len(self._all_stems)
        if n == 0 and not self._save_rest:
            self._summary.setText("All stems")
        elif n == 0 and self._save_rest:
            self._summary.setText("Rest (all stems inverted)")
        elif n == 1:
            stem = list(self._selected_stems)[0].capitalize()
            suffix = " (+ rest)" if self._save_rest else ""
            self._summary.setText(f"{stem}{suffix}")
        elif n <= 3:
            stems = [s.capitalize() for s in sorted(self._selected_stems)]
            suffix = " (+ rest)" if self._save_rest else ""
            self._summary.setText(f"{', '.join(stems)}{suffix}")
        elif n > 3:
            stems = [s.capitalize() for s in sorted(self._selected_stems)]
            suffix = " (+ rest)" if self._save_rest else ""
            self._summary.setText(f"{stems[0]} +{n-1} more{suffix}")
        else:
            self._summary.setText("All stems")

    def mousePressEvent(self, e):
        self._open_dialog()
        super().mousePressEvent(e)

    def _open_dialog(self):
        if not self._all_stems:
            return
        dlg = _StemSelectionDialog(
            self._all_stems,
            list(self._selected_stems),
            self._save_rest,
            primary_target=self._primary_target,
            parent=self
        )
        if dlg.exec() == QDialog.Accepted:
            self._selected_stems = dlg.get_selected_stems()
            self._save_rest = dlg.get_save_rest()
            self._update_summary()

    def get_selected_stems(self):
        if not self._selected_stems and not self._save_rest:
            return self._all_stems[:]
        return sorted(self._selected_stems) if self._selected_stems else self._all_stems[:]

    def get_save_rest(self):
        return self._save_rest


# ── Circle Check Selector ─────────────────────────────────────────────────────

class _CircleCheck(QFrame):
    toggled = Signal(bool)
    _unsel_border   = 38
    _unsel_border_h = 70
    _sel_border     = 170
    _sel_border_h   = 210
    _fill_rest      = 8
    _fill_hover     = 14
    stem_name       = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self._hovered = False
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)

    def is_checked(self):
        return self._checked

    def set_checked(self, value):
        if self._checked == value:
            return
        self._checked = value
        self.update()
        self.toggled.emit(self._checked)

    def toggle(self):
        self.set_checked(not self._checked)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor, QBrush
        from PySide6.QtCore import QPointF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = 9.0, 9.0
        rr = 7.5
        _ac = QColor(theme_manager.accent)

        if self._checked:
            ba = self._sel_border_h if self._hovered else self._sel_border
            _ac_ba = QColor(_ac)
            _ac_ba.setAlpha(ba)
            p.setPen(QPen(_ac_ba, 1.5))
            _ac_fill = QColor(_ac)
            _ac_fill.setAlpha(10)
            p.setBrush(_ac_fill)
            p.drawEllipse(QPointF(cx, cy), rr, rr)
            p.setPen(Qt.NoPen)
            p.setBrush(_ac)
            p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)
        else:
            ba = self._unsel_border_h if self._hovered else self._unsel_border
            _cb = QColor(theme_manager.theme.text)
            _cb.setAlpha(ba)
            p.setPen(QPen(_cb, 1.5))
            fa = self._fill_hover if self._hovered else self._fill_rest
            _cf = QColor(theme_manager.theme.text)
            _cf.setAlpha(fa)
            p.setBrush(_cf)
            p.drawEllipse(QPointF(cx, cy), rr, rr)
        p.end()

    def enterEvent(self, e):
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self.toggle()
        super().mousePressEvent(e)


# ── Model Item ────────────────────────────────────────────────────────────────

class _ModelItem(QFrame):
    selected = Signal(str, str, str, str, str, bool)
    settings_requested = Signal(str, str, str, str)

    def sizeHint(self):
        return QSize(0, 38)

    def __init__(self, name, ckpt="", yaml_path="", arch="", model_type="",
                 backend_module="", custom_backend_enabled=False, parent=None):
        super().__init__(parent)
        self._name = name
        self._ckpt = ckpt
        self._yaml = yaml_path
        self._arch = arch
        self._type = model_type
        self._backend_module = backend_module
        self._custom = custom_backend_enabled
        self._is_selected = False
        self.setFixedHeight(38)
        self.setStyleSheet("QFrame{background:transparent;border:none;}")
        self.setCursor(Qt.PointingHandCursor)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(12, 0, 8, 0)
        hl.setSpacing(6)

        self._circle = _CircleCheck()
        self._circle.toggled.connect(self._on_circle_toggled)
        hl.addWidget(self._circle)

        self._lbl = QLabel(name)
        self._lbl.setObjectName("modelItemLabel")
        self._lbl.setStyleSheet(
            f"font-family:{FONT_STACK};font-size:12px;"
            "font-weight:600;letter-spacing:0.5px;"
            f"color:{theme_manager.theme.text_sec};background:transparent;"
        )
        hl.addWidget(self._lbl, 1)

        if custom_backend_enabled and backend_module:
            ctag = QLabel("CUSTOM")
            _ac = QColor(theme_manager.accent)
            _ac_bg = QColor(theme_manager.accent)
            _ac_bg.setAlpha(20)
            _ac_border = QColor(theme_manager.accent)
            _ac_border.setAlpha(38)
            ctag.setStyleSheet(
                "font-family:'Montserrat';font-size:8px;font-weight:700;"
                f"color:rgba({_ac.red()},{_ac.green()},{_ac.blue()},0.70);"
                f"background:rgba({_ac_bg.red()},{_ac_bg.green()},{_ac_bg.blue()},{_ac_bg.alpha()});"
                "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
                f"border:1px solid rgba({_ac_border.red()},{_ac_border.green()},{_ac_border.blue()},{_ac_border.alpha()});"
            )
            ctag.setFixedHeight(17)
            hl.addWidget(ctag)

        if model_type:
            display = "INSTRUMENTAL" if model_type == "instrumental" else model_type.upper()[:10]
            tag = QLabel(display)
            t = theme_manager.theme
            tag.setStyleSheet(
                "font-family:'Montserrat';font-size:8px;font-weight:700;"
                f"color:{t.text_label};background:{t.surface_alt};"
                "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
            )
            tag.setFixedHeight(17)
            hl.addWidget(tag)

        self._dots = QPushButton("\u00b7\u00b7\u00b7")
        self._dots.setFixedSize(26, 26)
        self._dots.setStyleSheet(
            "QPushButton{background:transparent;"
            f"color:{theme_manager.theme.text_muted};"
            "border:none;font-size:11px;font-weight:600;border-radius:4px;}"
            f"QPushButton:hover{{color:{theme_manager.accent};"
            f"background:{theme_manager._accent_soft};}}"
        )
        self._dots.clicked.connect(self._on_dots_clicked)
        hl.addWidget(self._dots)

        outer.addWidget(row, 1)

        self._divider = QFrame()
        self._divider.setFixedHeight(1)
        self._divider.setStyleSheet(
            f"background:{theme_manager.theme.border};border:none;")
        outer.addWidget(self._divider)

    def _on_circle_toggled(self, checked):
        self._is_selected = checked
        self._update_style()
        if checked:
            self.selected.emit(self._name, self._ckpt, self._yaml, self._arch,
                               self._backend_module, self._custom)

    def _on_dots_clicked(self):
        self.settings_requested.emit(self._name, self._ckpt, self._yaml, self._arch)

    def set_selected(self, value):
        if self._is_selected == value:
            return
        self._is_selected = value
        self._circle.set_checked(value)
        self._update_style()

    def _update_style(self):
        t = theme_manager.theme
        if self._is_selected:
            self.setStyleSheet(
                f"QFrame{{background:{t.border};border:none;}}"
            )
            self._divider.setStyleSheet(
                f"background:{t.border};border:none;")
            self._lbl.setStyleSheet(
                f"font-family:{FONT_STACK};font-size:12px;"
                "font-weight:600;letter-spacing:0.5px;"
                f"color:{t.text};background:transparent;"
            )
        else:
            self.setStyleSheet(
                "QFrame{background:transparent;border:none;}"
            )
            self._divider.setStyleSheet(
                f"background:{t.border};border:none;")
            self._lbl.setStyleSheet(
                f"font-family:{FONT_STACK};font-size:12px;"
                "font-weight:600;letter-spacing:0.5px;"
                f"color:{t.text_sec};background:transparent;"
            )

    def enterEvent(self, e):
        if not self._is_selected:
            self.setStyleSheet(
                f"QFrame{{background:{theme_manager.theme.border};border:none;}}"
            )
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._update_style()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._circle.toggle()
        super().mousePressEvent(e)


class _ExpandArrow(QWidget):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0.0
        self._hovered = False
        self.setFixedSize(24, 24)
        self.setCursor(Qt.PointingHandCursor)

    def get_angle(self):
        return self._angle

    def set_angle(self, val):
        self._angle = val
        self.update()

    angle = Property(float, get_angle, set_angle)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.translate(12, 12)
        p.rotate(self._angle)
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(51)
        pen = QPen(QColor(theme_manager.accent) if self._hovered else _c, 2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(-5, -6, 0, 0)
        p.drawLine(0, 0, -5, 6)
        p.end()

    def enterEvent(self, e):
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        e.accept()

    def mouseReleaseEvent(self, e):
        e.accept()


# ── Architecture Card ─────────────────────────────────────────────────────────

class _ArchCard(QFrame):
    model_selected = Signal(str, str, str, str, str, bool)
    ckpt_settings_requested = Signal(str, str, str, str)

    def __init__(self, arch_name, parent=None):
        super().__init__(parent)
        self._arch = arch_name
        self._expanded = False
        self._items = []
        self.setObjectName("archCard")
        self.setStyleSheet(
            "#archCard{"
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {theme_manager.theme.surface},stop:1 {theme_manager.theme.bg});"
            f"border:1px solid {theme_manager.theme.border_visible};"
            f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px;}}"
            "#archCard:hover{"
            f"border:1px solid {theme_manager.accent};"
            f"background:{theme_manager._accent_soft};}}"
        )

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # Card header
        self._hdr = QFrame()
        self._hdr.setObjectName("cardHdr")
        self._hdr.setFixedHeight(42)
        self._hdr.setStyleSheet(
            "QFrame#cardHdr{"
            "background:transparent;"
            "border:none;"
            f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px {UIConstants.CARD_RADIUS_STYLESHEET}px 0 0;}}"
        )
        hh = QHBoxLayout(self._hdr)
        hh.setContentsMargins(12, 0, 8, 0)
        hh.setSpacing(6)

        dot_key = f"arch_dot_{arch_name.lower().split()[0]}"
        dot_clr = getattr(theme_manager.theme, dot_key, theme_manager.theme.text_label)
        dot = QFrame()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{dot_clr};border:none;border-radius:4px;")
        hh.addWidget(dot)

        title_lbl = QLabel(arch_name.replace(" Architecture", "").upper())
        title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:700;"
            f"color:{theme_manager.theme.text};background:transparent;"
        )
        hh.addWidget(title_lbl, 1)

        self._count_lbl = QLabel("0")
        self._count_lbl.setAlignment(Qt.AlignCenter)
        self._count_lbl.setFixedWidth(24)
        self._count_lbl.setStyleSheet(
            f"background:{theme_manager.theme.border};color:{theme_manager.theme.text};"
            f"font-family:'Montserrat',sans-serif;font-size:8px;font-weight:600;"
            "border-radius:3px;"
        )
        self._count_lbl.setFixedHeight(18)
        hh.addWidget(self._count_lbl)

        self._toggle_btn = _ExpandArrow()
        self._toggle_btn.clicked.connect(lambda: self._toggle_expand(animated=True))
        hh.addWidget(self._toggle_btn)

        self._hdr.installEventFilter(self)
        for child in self._hdr.findChildren(QWidget):
            child.installEventFilter(self)

        vl.addWidget(self._hdr)

        # Model list
        self._list_w = QWidget()
        self._list_w.setStyleSheet("background:transparent;")
        self._list_vl = QVBoxLayout(self._list_w)
        self._list_vl.setContentsMargins(0, 0, 0, 0)
        self._list_vl.setSpacing(0)

        self._has_models = False
        self._empty_lbl = QLabel("No models registered")
        self._empty_lbl.setStyleSheet(
            f"color:{theme_manager.theme.text_muted};font-size:9px;font-style:italic;"
            "background:transparent;padding:6px 12px;")
        self._list_vl.addWidget(self._empty_lbl)

        self._content = QWidget()
        self._content.setStyleSheet("background:transparent;")
        self._content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        cl = QVBoxLayout(self._content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self._list_w)
        self._content.setVisible(False)
        self._content.setMaximumHeight(0)
        vl.addWidget(self._content)

        # ── Animation (smooth 120Hz via QPropertyAnimation) ──
        self._height_anim = QPropertyAnimation(self, b"anim_height", self)
        self._height_anim.setDuration(350)
        self._height_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._height_anim.finished.connect(self._on_anim_finished)

        self._rotate_anim = QPropertyAnimation(self._toggle_btn, b"angle", self)
        self._rotate_anim.setDuration(350)
        self._rotate_anim.setEasingCurve(QEasingCurve.OutCubic)

        self._cached_height = 0
        self._saved_expanded = None

    def _get_anim_height(self):
        return self.height()

    def _set_anim_height(self, h):
        self.setFixedHeight(int(h))

    anim_height = Property(int, _get_anim_height, _set_anim_height)

    # ── Events ──

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.MouseButtonRelease
                and event.button() == Qt.LeftButton
                and obj is not self._toggle_btn):
            self._toggle_expand(animated=True)
            return True
        return super().eventFilter(obj, event)

    # ── Animation helpers ──

    def _rebuild_cache(self):
        h = 0
        for i in range(self._list_vl.count()):
            w = self._list_vl.itemAt(i).widget()
            if isinstance(w, _ModelItem) and w.isVisibleTo(self._content):
                h += w.sizeHint().height()
        m = self._list_vl.contentsMargins()
        self._cached_height = h + m.top() + m.bottom()

    def _on_anim_finished(self):
        if self._expanded:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self._content.setMinimumHeight(0)
            self._content.setMaximumHeight(16777215)
            self._content.adjustSize()
        else:
            self._content.setVisible(False)
            self._content.setMaximumHeight(0)
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)

    def _stop_anim(self):
        self._height_anim.stop()
        self._rotate_anim.stop()

    def _toggle_expand(self, animated=True):
        self._stop_anim()
        self._expanded = not self._expanded

        if self._expanded:
            self._rebuild_cache()
            target_h = self._cached_height
            if target_h <= 0:
                self._toggle_btn.set_angle(90.0)
                return
            self._content.setMaximumHeight(target_h)
            self._content.setVisible(True)
            start_h = self.height()
            if animated:
                self._height_anim.setStartValue(float(start_h))
                self._height_anim.setEndValue(float(42 + target_h))
                self._rotate_anim.setStartValue(self._toggle_btn.get_angle())
                self._rotate_anim.setEndValue(90.0)
                self._height_anim.start()
                self._rotate_anim.start()
            else:
                self.setFixedHeight(42 + target_h)
                self._toggle_btn.set_angle(90.0)
                self._on_anim_finished()
        else:
            current_h = self._content.height()
            if current_h <= 0:
                self._content.setVisible(False)
                self._toggle_btn.set_angle(0.0)
                return
            start_h = self.height()
            if animated:
                self._height_anim.setStartValue(float(start_h))
                self._height_anim.setEndValue(42.0)
                self._rotate_anim.setStartValue(self._toggle_btn.get_angle())
                self._rotate_anim.setEndValue(0.0)
                self._height_anim.start()
                self._rotate_anim.start()
            else:
                self._content.setVisible(False)
                self._content.setMaximumHeight(0)
                self._toggle_btn.set_angle(0.0)

    def _update_expanded_height(self):
        if not self._expanded:
            return
        self._rebuild_cache()
        self.setFixedHeight(42 + self._cached_height)
        self._content.setMaximumHeight(self._cached_height)

    def _deselect_all_models(self):
        for i in range(self._list_vl.count()):
            w = self._list_vl.itemAt(i).widget()
            if isinstance(w, _ModelItem):
                w.set_selected(False)

    def add_model(self, name, ckpt="", yaml_path="", arch="", model_type="",
                  backend_module="", custom_backend_enabled=False):
        if name in self._items:
            return
        if not self._has_models:
            self._empty_lbl.setVisible(False)
            self._has_models = True
        item = _ModelItem(name, ckpt, yaml_path, arch, model_type,
                          backend_module, custom_backend_enabled)
        item.selected.connect(lambda n, ck, y, a, bm, cb:
                              self.model_selected.emit(n, ck, y, a, bm, cb))
        item.settings_requested.connect(self.ckpt_settings_requested)
        self._list_vl.addWidget(item)
        self._items.append(name)
        self._count_lbl.setText(str(len(self._items)))
        self._update_expanded_height()

    def remove_model(self, name):
        for i in range(self._list_vl.count()):
            w = self._list_vl.itemAt(i).widget()
            if isinstance(w, _ModelItem) and w._name == name:
                w.deleteLater(); break
        self._items = [n for n in self._items if n != name]
        self._count_lbl.setText(str(len(self._items)) if self._items else "0")
        if not self._items:
            self._has_models = False
            self._empty_lbl.setVisible(True)
        self._update_expanded_height()

    def deselect_all_models(self):
        self._deselect_all_models()

    def filter_models(self, text):
        text = text.lower().strip()
        if not text:
            self.setVisible(True)
            for i in range(self._list_vl.count()):
                w = self._list_vl.itemAt(i).widget()
                if isinstance(w, _ModelItem):
                    w.setVisible(True)
            if self._has_models:
                self._count_lbl.setText(str(len(self._items)))
            if self._saved_expanded is not None:
                target = self._saved_expanded
                self._saved_expanded = None
                if self._expanded != target:
                    self._toggle_expand(animated=False)
            self._rebuild_cache()
            return

        if self._saved_expanded is None:
            self._saved_expanded = self._expanded

        arch_match = text in self._arch.lower()
        has_ckpt_match = False
        for i in range(self._list_vl.count()):
            w = self._list_vl.itemAt(i).widget()
            if isinstance(w, _ModelItem):
                match = arch_match or text in w._name.lower() or text in w._arch.lower()
                w.setVisible(match)
                if match:
                    has_ckpt_match = True

        card_visible = arch_match or has_ckpt_match or not self._has_models
        self.setVisible(card_visible)

        total = len(self._items)
        visible = sum(1 for w in self._list_vl.findChildren(_ModelItem)
                      if w.isVisibleTo(self._content))
        self._count_lbl.setText(
            str(visible) if visible == total else f"{visible}/{total}")

        self._rebuild_cache()
        self._update_expanded_height()

        if has_ckpt_match and not self._expanded:
            self._toggle_expand(animated=False)
        elif not has_ckpt_match and self._expanded and not arch_match:
            self._toggle_expand(animated=False)


# ── InferencePage ─────────────────────────────────────────────────────────────

class InferencePage(QWidget):
    log_output = Signal(str)
    input_files_submitted = Signal(list)
    model_selected = Signal(str)
    ckpt_settings_requested = Signal(str, str, str, str)
    process_running = Signal(bool)
    add_model_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("inferencePage")
        # Object-name scoped so the background doesn't cascade into child
        # dialogs (QMessageBox etc.) and overwrite their button styles.
        self.setStyleSheet(f"#inferencePage{{background:{theme_manager.theme.bg};}}")
        self._runner         = None
        self._tmp_input      = None
        self._tmp_yaml       = None
        self._selected_model = None
        self._arch_cards     = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── PAGE HEADER (full width) ───────────────────────────────────
        header_w = QWidget()
        header_w.setStyleSheet("background:transparent;")
        hh = QHBoxLayout(header_w)
        hh.setContentsMargins(32, 32, 32, 0)
        hh.addWidget(PageHeader(
            "INFERENCE",
            "AI POWERED MUSIC STEMS EXTRACTION",
            highlight="MUSIC STEMS",
        ))
        root.addWidget(header_w)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ── LEFT COLUMN ────────────────────────────────────────────────
        left = QWidget()
        left.setStyleSheet(f"background:{theme_manager.theme.bg};")
        left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(32, 32, 10, 64)
        ll.setSpacing(0)

        t = theme_manager.theme

        # Configuration (pushed down so it aligns with the search-bar row
        # on the right column, where the MODEL LIBRARY header is centered)
        ll.addSpacing(11)
        ll.addWidget(_sec_hdr("Configuration"))
        ll.addSpacing(23)  # aligns the first INPUT row with the first model card

        cfg = QVBoxLayout()
        cfg.setSpacing(6)
        cfg.setContentsMargins(0, 0, 0, 0)

        self._input_row = _BrowseRow(
            None, "Input", "Select or drop audio file(s)\u2026",
            mode="file", file_filter=AUDIO_FILTER)
        cfg.addWidget(self._input_row)

        self._output_row = _BrowseRow(
            None, "Output", "Select or drop output folder\u2026", mode="folder")
        cfg.addWidget(self._output_row)

        self._fmt_row = _ComboRow(
            None, "Quality",
            ["WAV (PCM 32-bit)", "WAV (PCM 16-bit)", "FLAC", "MP3 320kbps", "MP3 128kbps"])
        self._fmt_combo = self._fmt_row.combo
        cfg.addWidget(self._fmt_row)

        self._output_stems_row = _OutputStemsRow(
            None, "Stems")
        cfg.addWidget(self._output_stems_row)

        self._tta_row = _ComboRow(
            None, "TTA", ["Disabled", "Enabled"])
        self._tta_combo = self._tta_row.combo
        cfg.addWidget(self._tta_row)

        self._dev_row = _ComboRow(
            None, "Device", list_gpus())
        self._device_combo = self._dev_row.combo
        cfg.addWidget(self._dev_row)

        ll.addLayout(cfg)

        ll.addSpacing(36)
        ll.addStretch()  # anchor the Run Inference block to the bottom so it
                         # aligns horizontally with + ADD in the right column

        # Run Inference
        ll.addWidget(_sec_hdr("Run Inference"))
        ll.addSpacing(28)  # roomier than CONFIGURATION: buttons sit close otherwise

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.btn_run = QPushButton("\u25B6  Separate")
        self.btn_run.setFixedSize(170, 44)
        self.btn_run.setStyleSheet(
            "QPushButton{"
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {theme_manager._accent_hover},stop:1 {theme_manager.accent});"
            f"color:{theme_manager._accent_text};border:none;border-radius:6px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:17px;}"
            "QPushButton:hover{"
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {theme_manager._accent_hover},stop:1 {theme_manager.accent});}}"
            f"QPushButton:pressed{{background:{theme_manager._accent_hover};}}"
            f"QPushButton:disabled{{background:{t.disabled_bg};color:{t.disabled_text};}}"
        )
        self.btn_run.clicked.connect(self._run)

        self.btn_stop = QPushButton("\u25A0  Stop")
        self.btn_stop.setFixedSize(100, 44)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton{"
            f"background:{t.surface};color:{t.text_muted};"
            f"border:1px solid {t.border};border-radius:6px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:12px;}"
            "QPushButton:enabled{"
            f"color:{t.error};border:1px solid {t.error};}}"
            f"QPushButton:hover:enabled{{background:{t.surface_alt};}}"
            f"QPushButton:disabled{{color:{t.text_muted};}}"
        )
        self.btn_stop.clicked.connect(self._stop)

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        ll.addLayout(btn_row)

        # ── RIGHT COLUMN ───────────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet(f"background:{theme_manager.theme.bg};")
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 32, 32, 64)
        rl.setSpacing(0)

        # Model Library header, vertically centered against the search box
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 14, 0)
        hdr_row.setSpacing(10)
        hdr_row.addWidget(_sec_hdr("Model Library"))
        hdr_row.addStretch()

        self._search_bar = _SearchBar("Search models\u2026")
        self._search_bar.textChanged.connect(self._filter_models)
        self._search_bar.setMaximumWidth(155)
        hdr_row.addWidget(self._search_bar)

        rl.addLayout(hdr_row)
        rl.addSpacing(12)

        # Model cards scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
            f"QScrollBar::handle:vertical{{background:{t.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            f"QScrollBar::handle:vertical:hover{{background:{t.border_dim};}}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,"
            "QScrollBar::sub-page:vertical{background:transparent;}"
        )
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        model_content = QWidget()
        model_content.setStyleSheet("background:transparent;")
        self._model_layout = QVBoxLayout(model_content)
        self._model_layout.setContentsMargins(0, 0, 14, 0)
        self._model_layout.setSpacing(6)

        for arch in ARCH_TYPES:
            card = _ArchCard(arch)
            card.model_selected.connect(self._on_model_selected)
            card.ckpt_settings_requested.connect(self._on_ckpt_settings_requested)
            self._arch_cards[arch] = card
            self._model_layout.addWidget(card)

        self._model_layout.addStretch()

        scroll.setWidget(model_content)
        rl.addWidget(scroll, 1)
        rl.addSpacing(10)

        # + ADD (bottom of the library) — jumps to Settings where models
        # are added/registered. Styled like the LOG button in CONSOLE;
        # lifted 7px so its centre aligns with the 44px RUN/STOP row.
        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 14, 7)
        add_row.addStretch()
        self._add_btn = QPushButton("+  Add")
        self._add_btn.setCursor(Qt.PointingHandCursor)
        self._add_btn.setFixedSize(70, 30)
        self._add_btn.setStyleSheet(
            f"QPushButton{{"
            f"background:{t.surface};color:{t.text_dim};"
            f"border:1px solid {t.border_dim};border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:9px;padding:0 8px;}"
            f"QPushButton:hover{{background:{theme_manager._accent_soft};"
            f"color:{t.text};border:1px solid {theme_manager.accent};}}")
        self._add_btn.clicked.connect(self.add_model_requested.emit)
        add_row.addWidget(self._add_btn)
        rl.addLayout(add_row)

        # Assemble
        body.addWidget(left, 48)
        body.addWidget(right, 52)
        root.addLayout(body, 1)

    # ── Theme re-apply ────────────────────────────────────────────────

    def reapply_theme(self):
        t = theme_manager.theme
        self.setStyleSheet(f"#inferencePage{{background:{t.bg};}}")

        # Update search bar
        self._search_bar.reapply_theme()

        # Update filter buttons
        for fb in self.findChildren(_FilterButton):
            fb.reapply_theme()

        # Update arch cards
        for arch_name, card in self._arch_cards.items():
            card.setStyleSheet(
                "#archCard{"
                "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                f"stop:0 {t.surface},stop:1 {t.bg});"
                f"border:1px solid {t.border_visible};"
                f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px;}}"
                "#archCard:hover{"
                f"border:1px solid {theme_manager.accent};"
                f"background:{theme_manager._accent_soft};}}"
            )
            dot_key = f"arch_dot_{arch_name.lower().split()[0]}"
            dot_clr = getattr(t, dot_key, t.text_label)
            for child in card.findChildren(QFrame):
                if child.objectName() == "cardHdr":
                    for c in child.findChildren(QFrame):
                        if c.parent() is child and c.objectName() == "":
                            c.setStyleSheet(f"background:{dot_clr};border:none;border-radius:4px;")
                            break
                    break
            # Card title
            for lbl in card._hdr.findChildren(QLabel):
                if lbl is not card._count_lbl:
                    lbl.setStyleSheet(
                        "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:700;"
                        f"color:{t.text};background:transparent;letter-spacing:1px;"
                    )
                    break
            # Count label
            card._count_lbl.setStyleSheet(
                f"background:{t.border};color:{t.text};"
                "font-family:'Montserrat',sans-serif;font-size:8px;font-weight:600;"
                "border-radius:3px;"
            )
            # Empty label
            card._empty_lbl.setStyleSheet(
                f"color:{t.text_muted};font-size:9px;font-style:italic;"
                "background:transparent;padding:6px 12px;")
            # Model items
            for item in card.findChildren(_ModelItem):
                item._lbl.setStyleSheet(
                    f"font-family:{FONT_STACK};font-size:12px;"
                    "font-weight:600;letter-spacing:0.5px;"
                    f"color:{t.text_sec};background:transparent;"
                )
                item._dots.setStyleSheet(
                    "QPushButton{background:transparent;"
                    f"color:{t.text_muted};"
                    "border:none;font-size:11px;font-weight:600;border-radius:4px;}"
                    f"QPushButton:hover{{color:{theme_manager.accent};"
                    f"background:{theme_manager._accent_soft};}}"
                )
                item._divider.setStyleSheet(f"background:{t.border};border:none;")
                if item._is_selected:
                    item.setStyleSheet(f"QFrame{{background:{t.border};border:none;}}")
                    item._lbl.setStyleSheet(
                        f"font-family:{FONT_STACK};font-size:12px;"
                        "font-weight:600;letter-spacing:0.5px;"
                        f"color:{t.text};background:transparent;"
                    )
                else:
                    item.setStyleSheet("QFrame{background:transparent;border:none;}")
                # Update type/CUSTOM tags
                for tag_lbl in item.findChildren(QLabel):
                    if tag_lbl is item._lbl:
                        continue
                    txt = tag_lbl.text()
                    if txt == "CUSTOM":
                        _ac = QColor(theme_manager.accent)
                        _ac_bg = QColor(theme_manager.accent)
                        _ac_bg.setAlpha(20)
                        _ac_border = QColor(theme_manager.accent)
                        _ac_border.setAlpha(38)
                        tag_lbl.setStyleSheet(
                            "font-family:'Montserrat';font-size:8px;font-weight:700;"
                            f"color:rgba({_ac.red()},{_ac.green()},{_ac.blue()},0.70);"
                            f"background:rgba({_ac_bg.red()},{_ac_bg.green()},{_ac_bg.blue()},{_ac_bg.alpha()});"
                            "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
                            f"border:1px solid rgba({_ac_border.red()},{_ac_border.green()},{_ac_border.blue()},{_ac_border.alpha()});"
                        )
                    else:
                        tag_lbl.setStyleSheet(
                            "font-family:'Montserrat';font-size:8px;font-weight:700;"
                            f"color:{t.text_label};background:{t.surface_alt};"
                            "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
                        )

        # Re-apply config row styles
        for row in self.findChildren(_BrowseRow):
            row.setStyleSheet(_row_ss())
            for edit in row.findChildren(QLineEdit):
                edit.setStyleSheet(
                    "QLineEdit{background:transparent;border:none;"
                    "font-family:'Montserrat';font-size:11px;"
                    f"color:{t.text};padding:0;}}"
                    f"QLineEdit::placeholder{{color:{t.text_muted};}}"
                )
        for row in self.findChildren(_ComboRow):
            row.setStyleSheet(_row_ss())
            row.combo.setStyleSheet(_combo_ss())
            for lbl in row.findChildren(QLabel):
                lbl.setStyleSheet(_lbl_ss())
        for row in self.findChildren(_OutputStemsRow):
            row.setStyleSheet(_row_ss())
            for lbl in row.findChildren(QLabel):
                pass  # summary label uses its own style

        # Update run/stop buttons
        self.btn_run.setStyleSheet(
            "QPushButton{"
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {theme_manager._accent_hover},stop:1 {theme_manager.accent});"
            f"color:{theme_manager._accent_text};border:none;border-radius:6px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:17px;}"
            "QPushButton:hover{"
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {theme_manager._accent_hover},stop:1 {theme_manager.accent});}}"
            f"QPushButton:pressed{{background:{theme_manager._accent_hover};}}"
            f"QPushButton:disabled{{background:{t.disabled_bg};color:{t.disabled_text};}}"
        )
        self.btn_stop.setStyleSheet(
            "QPushButton{"
            f"background:{t.surface};color:{t.text_muted};"
            f"border:1px solid {t.border};border-radius:6px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:12px;}"
            "QPushButton:enabled{"
            f"color:{t.error};border:1px solid {t.error};}}"
            f"QPushButton:hover:enabled{{background:{t.surface_alt};}}"
            f"QPushButton:disabled{{color:{t.text_muted};}}"
        )

    # ── Search ────────────────────────────────────────────────────────

    def _filter_models(self, text):
        text = text.lower().strip()
        for arch_name, card in self._arch_cards.items():
            if not text:
                card.filter_models("")
            else:
                card.filter_models(text)

    # ── Slots ─────────────────────────────────────────────────────────

    def on_model_registered(self, model: dict):
        arch = model.get("arch", "")
        if arch in self._arch_cards:
            self._arch_cards[arch].add_model(
                model["name"], model.get("ckpt", ""),
                model.get("yaml", ""), arch,
                model.get("type", ""),
                model.get("backend_module", ""),
                model.get("custom_backend_enabled", False))

    def on_model_removed(self, name: str):
        for card in self._arch_cards.values():
            card.remove_model(name)

    def _on_model_selected(self, name, ckpt, yaml_path, arch,
                           backend_module="", custom_backend_enabled=False):
        if self._selected_model:
            old_name = self._selected_model.get("name")
            old_arch = self._selected_model.get("arch")
            if old_name == name and old_arch == arch:
                card = self._arch_cards.get(arch)
                if card:
                    for i in range(card._list_vl.count()):
                        w = card._list_vl.itemAt(i).widget()
                        if isinstance(w, _ModelItem) and w._name == name:
                            w.set_selected(False)
                            break
                self._selected_model = None
                self._output_stems_row.set_stems([])
                return

            old_card = self._arch_cards.get(old_arch)
            if old_card:
                if old_arch == arch:
                    for i in range(old_card._list_vl.count()):
                        w = old_card._list_vl.itemAt(i).widget()
                        if isinstance(w, _ModelItem) and w._name == old_name:
                            w.set_selected(False)
                            break
                else:
                    old_card.deselect_all_models()

        self._selected_model = {
            "name": name, "ckpt": ckpt, "yaml": yaml_path, "arch": arch,
            "backend_module": backend_module,
            "custom_backend_enabled": custom_backend_enabled,
        }

        yaml_path = self._selected_model.get("yaml", "")
        instruments = []
        target_inst = None
        if yaml_path and os.path.isfile(yaml_path):
            try:
                import yaml as _yaml
                with open(yaml_path, encoding="utf-8") as _f:
                    _cfg = _yaml.load(_f, Loader=_yaml.FullLoader)
                instruments = _cfg.get("training", {}).get("instruments", []) or []
                target_inst = _cfg.get("training", {}).get("target_instrument", None)
            except Exception:
                pass
        self._output_stems_row.set_stems(instruments, primary_target=target_inst)

    def _on_ckpt_settings_requested(self, name, ckpt, yaml_path, arch):
        self.ckpt_settings_requested.emit(name, ckpt, yaml_path, arch)

    # ── Settings ──────────────────────────────────────────────────────

    def save_settings(self):
        return {
            "output_folder": self._output_row.value()
                             if isinstance(self._output_row.value(), str) else "",
            "output_format": self._fmt_combo.currentText(),
            "tta":           self._tta_combo.currentText(),
            "device":        self._device_combo.currentText(),
        }

    def load_settings(self, d):
        out = d.get("output_folder", "")
        if isinstance(out, str) and out:
            self._output_row.set_value(out)
        fmt = d.get("output_format", "WAV (PCM 32-bit)")
        if not isinstance(fmt, str): fmt = "WAV (PCM 32-bit)"
        idx = self._fmt_combo.findText(fmt)
        if idx >= 0: self._fmt_combo.setCurrentIndex(idx)
        tta = d.get("tta", "Disabled")
        if not isinstance(tta, str): tta = "Disabled"
        idx = self._tta_combo.findText(tta)
        if idx >= 0: self._tta_combo.setCurrentIndex(idx)
        dev = d.get("device", "")
        if not isinstance(dev, str): dev = ""
        idx = self._device_combo.findText(dev)
        if idx >= 0: self._device_combo.setCurrentIndex(idx)

    # ── Runner ────────────────────────────────────────────────────────

    def _validate(self):
        if not self._input_row.value():
            return "Please select at least one audio file."
        if not self._output_row.value():
            return "Please select an output folder."
        if not self._selected_model:
            return "Please select a model from the model library."
        return None

    def _run(self):
        from PySide6.QtWidgets import QMessageBox
        from ui.widgets.runtime_dialog import ensure_runtime
        if not ensure_runtime(self):
            return
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Missing input", err); return

        import tempfile, shutil, yaml
        files = self._input_row.value()
        self._tmp_input = tempfile.mkdtemp(prefix="msst_input_")
        for f in (files if isinstance(files, list) else []):
            shutil.copy2(f, self._tmp_input)

        ckpt = self._selected_model.get("ckpt", "")
        yaml_path = self._selected_model.get("yaml", "")
        arch = self._selected_model.get("arch", "")
        model_type = ARCH_TO_MODEL_TYPE.get(arch, "bs_roformer")
        device_ids = device_ids_from_selection(self._device_combo.currentText())
        force_cpu  = device_ids is None
        use_tta    = self._tta_combo.currentText() == "Enabled"

        # Apply per-ckpt settings and stem selection to YAML
        self._tmp_yaml = None
        ckpt_name = self._selected_model.get("name", "")
        ckpt_settings = settings_store.load_ckpt_settings().get(ckpt_name, {})

        selected_stems = self._output_stems_row.get_selected_stems()
        save_rest = self._output_stems_row.get_save_rest()
        all_stems = self._output_stems_row._all_stems
        custom_selection = (
            sorted(s.lower() for s in selected_stems) != sorted(s.lower() for s in all_stems)
        ) if all_stems else False

        # Single-target (generic) models only output the target instrument directly;
        # any other stem (e.g. "other") is only obtainable as "rest" (mix - target).
        # Auto-enable --save_rest when such a complement stem is selected.
        target_inst = ""
        if yaml_path and os.path.isfile(yaml_path):
            try:
                import yaml as _yaml
                with open(yaml_path, "r", encoding="utf-8") as _f:
                    _yc = _yaml.load(_f, Loader=_yaml.FullLoader)
                target_inst = (_yc.get("training", {}).get("target_instrument") or "")
                target_inst = target_inst.lower()
            except Exception:
                pass
        sel_lower = set(s.lower() for s in selected_stems)
        effective_save_rest = save_rest
        if (not effective_save_rest and target_inst and len(all_stems) > 1
                and any(s != target_inst for s in sel_lower)):
            effective_save_rest = True

        needs_mod = bool(ckpt_settings) or custom_selection or save_rest

        if needs_mod and yaml_path and os.path.isfile(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                if "inference" not in config:
                    config["inference"] = {}
                if "chunk_size" in ckpt_settings:
                    raw_cs = ckpt_settings["chunk_size"]
                    hop = config.get("audio", {}).get("hop_length", 1024)
                    num_scales = config.get("model", {}).get("num_scales", 5)
                    align = 2 ** num_scales
                    T = raw_cs // hop + 1
                    aligned_T = (T // align) * align
                    if aligned_T < align:
                        aligned_T = align
                    config["inference"]["chunk_size"] = (aligned_T - 1) * hop
                if "overlap" in ckpt_settings:
                    config["inference"]["num_overlap"] = ckpt_settings["overlap"]
                if "batch_size" in ckpt_settings:
                    config["inference"]["batch_size"] = ckpt_settings["batch_size"]

                if not custom_selection and not effective_save_rest:
                    pass
                elif len(selected_stems) == 1 and not effective_save_rest:
                    if "training" not in config:
                        config["training"] = {}
                    config["training"]["target_instrument"] = selected_stems[0]
                else:
                    if "training" in config:
                        config["training"]["target_instrument"] = None

                self._tmp_yaml = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False)
                yaml.dump(config, self._tmp_yaml, default_flow_style=False)
                self._tmp_yaml.close()
                yaml_path = self._tmp_yaml.name
            except Exception:
                pass

        cmd = [
            get_python_exe(), os.path.join(REPO_ROOT, "inference.py"),
            "--model_type",        model_type,
            "--config_path",       yaml_path,
            "--start_check_point", ckpt,
            "--input_folder",      self._tmp_input,
            "--store_dir",
            self._output_row.value()
            if isinstance(self._output_row.value(), str) else "",
        ]
        if force_cpu: cmd.append("--force_cpu")
        else: cmd += ["--device_ids"] + [str(d) for d in device_ids]

        # Map the Quality selection to the output format the CLI should write.
        fmt = self._fmt_combo.currentText()
        if fmt == "WAV (PCM 16-bit)":
            cmd += ["--output_format", "wav", "--pcm_type", "PCM_16"]
        elif fmt == "FLAC":
            cmd += ["--output_format", "flac", "--pcm_type", "PCM_16"]
        elif fmt == "MP3 320kbps":
            cmd += ["--output_format", "mp3", "--mp3_bitrate", "320"]
        elif fmt == "MP3 128kbps":
            cmd += ["--output_format", "mp3", "--mp3_bitrate", "128"]
        else:  # "WAV (PCM 32-bit)"
            cmd += ["--output_format", "wav", "--pcm_type", "FLOAT"]

        if use_tta: cmd.append("--use_tta")
        if self._selected_model.get("custom_backend_enabled"):
            bm = self._selected_model.get("backend_module", "")
            if bm:
                cmd += ["--custom_backend",
                        os.path.join(REPO_ROOT, "models", "custom", bm)]

        if custom_selection:
            cmd += ["--save_stems", ",".join(s.lower() for s in selected_stems)]
        if effective_save_rest:
            cmd.append("--save_rest")

        out = self._output_row.value()
        if isinstance(out, str) and out.strip():
            self.log_output.emit(f"Output directory: {out}")

        self.input_files_submitted.emit(
            [f for f in files if isinstance(f, str)] if isinstance(files, list) else []
        )
        self.model_selected.emit(self._selected_model.get("name", "") if self._selected_model else "")

        for i, f in enumerate(files if isinstance(files, list) else []):
            if i == 0:
                self.log_output.emit(f"Processing: {os.path.basename(f)}")
            else:
                self.log_output.emit(f"Queued: {os.path.basename(f)}")

        self._runner = ProcessRunner(cmd, cwd=REPO_ROOT)
        self._runner.log_line.connect(self.log_output.emit)
        self._runner.finished.connect(self._on_finished)
        self.process_running.emit(True)
        self._runner.start()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self):
        if self._runner:
            self._runner.stop()
        self.process_running.emit(False)

    def _on_finished(self, code):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.process_running.emit(False)
        if code == 0:
            self.log_output.emit("Completed: processing")
        else:
            self.log_output.emit("ERROR: processing failed")
        if self._tmp_input and os.path.isdir(self._tmp_input):
            import shutil
            try: shutil.rmtree(self._tmp_input)
            except OSError: pass
            self._tmp_input = None
        if self._tmp_yaml and os.path.isfile(self._tmp_yaml.name):
            try: os.remove(self._tmp_yaml.name)
            except OSError: pass
            self._tmp_yaml = None


# ── ModelRow compat ───────────────────────────────────────────────────────────
class ModelRow(QFrame):
    def __init__(self, label, widget, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(f"QFrame{{background:{theme_manager.theme.surface};border:none;}}")
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(0)
        lf = QFrame(); lf.setFixedWidth(100)
        lf.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.surface_alt};border:none;border-right:1px solid {theme_manager.theme.border_visible};}}")
        ll2 = QHBoxLayout(lf); ll2.setContentsMargins(8, 0, 8, 0)
        lb = QLabel(label.upper())
        lb.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:8px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;")
        ll2.addWidget(lb)
        hl.addWidget(lf)
        widget.setStyleSheet("border:none;background:transparent;padding:0 8px;")
        hl.addWidget(widget, 1)
