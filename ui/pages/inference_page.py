"""
ui/pages/inference_page.py
Premium cinematic dark UI — 2 column layout.
"""
import os, sys, time, threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QLineEdit, QFileDialog,
    QScrollArea, QSizePolicy, QSpacerItem, QDialog,
    QDialogButtonBox, QMenu, QMessageBox,
)
from PySide6.QtGui import QCursor
from PySide6.QtCore import (Qt, Signal, Property, QObject, QEasingCurve, QSize,
                            QPoint, QRectF, QPointF, QPropertyAnimation,
                            QVariantAnimation, QEvent, QUrl, QTimer)
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QDesktopServices


from backend.runner import ProcessRunner, describe_exit_code
from backend.model_manager import fetch_model_index
from backend.paths import REPO_ROOT, APP_DIR, get_python_exe
from backend.audio_names import INFERENCE_FILENAME_TEMPLATE
from backend.gpu_utils import list_gpus, device_ids_from_selection
from backend import settings as settings_store
from backend import msst_catalog as model_catalog
from backend.msst_catalog import (
    engine_effective_type,
    engine_unsupported_models,
    has_custom_sidecar,
    resolve_engine_model_type,
)
from backend.yaml_analyzer import classify_model_type, get_stems_for_type
from utils.stem_planning import (
    KEEP, complement_stem_name, is_single_output, plan_output_stems,
    resolve_target, rest_needed,
)
from ui.theme import theme_manager, UIConstants, FONT_STACK
from ui.widgets.common import (
    ConsoleLog, SpectrogramPanel, WaveformPanel, ProcessingStatusPanel, PageHeader,
    outline_button_ss, solid_button_ss, paint_chevron, EllipsisButton,
    GlyphButton, dark_menu_qss, run_blurred_dialog,
    _outline_icon_color, _solid_icon_color, _stop_icon_color, _add_icon_color,
    _type_badge_ss, _custom_badge_ss, _blocked_badge_ss,
    _type_badge_color, _type_title,
)

AUDIO_FILTER = ("Audio files (*.wav *.flac *.mp3 *.ogg *.aiff *.m4a *.opus *.wv);;"
                "All files (*.*)")
ARCH_TYPES = [
    "Apollo Architecture", "Bandit Architecture", "BS Roformer Architecture",
    "BSMamba2 Architecture", "Conformer Architecture", "Demucs Architecture",
    "DTTNet Architecture", "MDX23c Architecture", "MDX-Net Architecture",
    "Medley Vox Architecture", "Melband Roformer Architecture",
    "SCNet Architecture", "Swin Upernet Architecture", "TorchSeg Architecture",
    "VR Architecture", "VitLarge23 Architecture",
]
ARCH_TO_MODEL_TYPE = {
    "MDX Architecture": "mdx23c",  # legacy label from before the MDX split
    "MDX23c Architecture": "mdx23c",
    "MDX-Net Architecture": "mdxnet",
    "VR Architecture": "vr",
    "Demucs Architecture": "htdemucs",
    "BS Roformer Architecture": "bs_roformer",
    "Melband Roformer Architecture": "mel_band_roformer",
    "Medley Vox Architecture": "medley_vox",
    "SCNet Architecture": "scnet",
    "Apollo Architecture": "apollo",
    "Bandit Architecture": "bandit",
    "BSMamba2 Architecture": "bs_mamba2",
    "Conformer Architecture": "conformer",
    # Engine dispatch branch is spelled 'dttnet' (upstream also uses
    # 'dtt_net' in places); the registry stores the buildable spelling.
    "DTTNet Architecture": "dttnet",
    "Swin Upernet Architecture": "swin_upernet",
    "TorchSeg Architecture": "torchseg",
    "VitLarge23 Architecture": "segm_models",
}


# Display overrides for the MODEL LIBRARY card titles (the arch label itself
# stays unchanged — it's used as a key for grouping / model type mapping).
ARCH_DISPLAY_NAMES = {
    "Apollo Architecture": "Apollo",
    "Bandit Architecture": "Bandit",
    "BS Roformer Architecture": "Band-Split RoFormer",
    "Demucs Architecture": "Demucs",
    "MDX23c Architecture": "MDX23c",
    "MDX-Net Architecture": "MDX-Net",
    "Medley Vox Architecture": "Medley-Vox",
    "Melband Roformer Architecture": "Mel-Band RoFormer",
    "SCNet Architecture": "SCNet",
    "VR Architecture": "VR",
    "BSMamba2 Architecture": "BSMamba2",
    "Conformer Architecture": "Conformer",
    "DTTNet Architecture": "DTTNet",
    "Swin Upernet Architecture": "Swin Upernet",
    "TorchSeg Architecture": "TorchSeg",
    "VitLarge23 Architecture": "VitLarge23",
}

# Paper / project links shown behind the small "i" button on library cards.
ARCH_INFO_LINKS = {
    "Apollo Architecture": "https://mvsep.com/algorithms/52",
    "Bandit Architecture": "https://mvsep.com/algorithms/27",
    "BS Roformer Architecture": "https://mvsep.com/algorithms/34",
    "BSMamba2 Architecture": "https://arxiv.org/pdf/2508.14556",
    "Conformer Architecture": "https://arxiv.org/pdf/2005.08100",
    "Demucs Architecture": "https://mvsep.com/algorithms/3",
    "DTTNet Architecture": "https://arxiv.org/pdf/2309.08684",
    "MDX23c Architecture": "https://mvsep.com/algorithms/7",
    "MDX-Net Architecture": "https://mvsep.com/algorithms/12",
    "Medley Vox Architecture": "https://mvsep.com/algorithms/60",
    "Melband Roformer Architecture": "https://mvsep.com/algorithms/49",
    "SCNet Architecture": "https://mvsep.com/algorithms/51",
    "Swin Upernet Architecture": "https://arxiv.org/pdf/2103.14030",
    "TorchSeg Architecture": "https://github.com/qubvel-org/segmentation_models.pytorch",
    "VR Architecture": "https://mvsep.com/algorithms/68",
    "VitLarge23 Architecture": "https://mvsep.com/algorithms/21",
}


def _arch_dot_token(arch_name):
    """First word of an arch label, normalised onto the arch_dot_* tokens
    (MDX-Net shares the classic MDX color; MDX23c has its own)."""
    word = arch_name.lower().split()[0]
    return {"mdx-net": "mdx"}.get(word, word)

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
        self.setFixedHeight(32)
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
        return QSize(155, 32)

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


class _InfoDot(QWidget):
    """Small circled-'i' in front of a CONFIGURATION row label; the row's
    explanation tooltip shows only when hovering this dot, not the row.
    Hand-drawn (circle outline + i) so it renders identically everywhere —
    the U+24D8 character depends on font fallback and looks off. Turns
    accent blue on hover so it reads as interactive.

    W (14) and SLOT_W (the x where the dot lands inside _lbl_with_info's
    80px label column) are shared constants: the TRAINING page reuses them
    so its ⓘ dots align on the same column logic.
    """

    W = 14
    SLOT_W = 19

    def __init__(self, tooltip):
        super().__init__()
        self._hovered = False
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(14, 14)

    def _color(self):
        from ui.widgets.common import css_color
        return css_color(theme_manager.accent if self._hovered
                         else theme_manager.theme.text_muted)

    def enterEvent(self, e):
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = self._color()
        cx = cy = self.width() / 2.0
        r = 5.6
        pen = QPen(c, 1.2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)
        # the 'i': stem + dot
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(cx - 0.55, cy - 0.8, 1.1, 4.6), 0.55, 0.55)
        p.drawEllipse(QPointF(cx, cy - 2.9), 0.8, 0.8)


def _info_dot(tooltip):
    """Small circled-'i' in front of a CONFIGURATION row label; the row's
    explanation tooltip shows only when hovering this dot, not the row."""
    dot = _InfoDot(tooltip)
    return dot


def _label_with_info(label, tooltip=""):
    """The row's small-caps label, optionally followed by the ⓘ dot when a
    tooltip is given (the 80px label column is shared between them)."""
    if not tooltip:
        lb = QLabel(label.upper())
        lb.setStyleSheet(_lbl_ss())
        lb.setFixedWidth(80)
        lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        return lb
    wrap = QWidget()
    wrap.setStyleSheet("background:transparent;")
    hl = QHBoxLayout(wrap)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(4)
    lb = QLabel(label.upper())
    lb.setStyleSheet(_lbl_ss())
    # Fixed dot slot: all rows line their ⓘ up at the same x (right after
    # the widest label, QUALITY/DEVICE), with the value column unchanged at
    # the 80px mark.
    lb.setFixedWidth(58)
    lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    hl.addWidget(lb)
    hl.addWidget(_info_dot(tooltip))
    hl.addStretch()
    wrap.setFixedWidth(80)
    return wrap

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


def _test_all_ss():
    """Outline style for the TEST ALL MODELS batch button — accent outline
    that dims to neutral when disabled (during a batch or normal run)."""
    t = theme_manager.theme
    return (
        "QPushButton{background:transparent;"
        f"color:{theme_manager.accent};"
        f"border:1px solid {theme_manager.accent};border-radius:6px;"
        "font-family:'Montserrat',sans-serif;font-weight:600;font-size:12px;}"
        f"QPushButton:hover{{background:{theme_manager._accent_soft};}}"
        f"QPushButton:disabled{{color:{t.text_muted};border-color:{t.border};}}"
    )


def _audio_files_in_folder(folder, extensions):
    """All supported audio files under `folder`, walked recursively."""
    found = []
    for root, _dirs, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in extensions:
                found.append(os.path.join(root, n))
    found.sort()
    return found


class _BrowseRow(QFrame):
    def __init__(self, icon, label, placeholder, mode="file",
                 file_filter="All (*.*)", tooltip="", parent=None):
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
        # right margin matches the chevron rows so the '...' dots and the '>'
        # chevrons stack centered below each other (like SETTINGS LOCAL FILES)
        hl.setContentsMargins(12, 0, 14, 0)
        hl.setSpacing(0)

        if icon is not None:
            icon.setFixedWidth(24)
            hl.addWidget(icon)
            hl.addSpacing(6)

        hl.addWidget(_label_with_info(label, tooltip))

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

        btn = EllipsisButton()
        btn.clicked.connect(self._browse)
        hl.addWidget(btn)

    def _browse(self):
        if self._mode == "folder":
            path = QFileDialog.getExistingDirectory(
                self, "Select folder", self._edit.text() or "")
            if path:
                self._files = []
                self._edit.setText(path)
            return
        # File mode: pick individual files or a whole folder (the folder is
        # walked recursively for supported audio files).
        menu = QMenu(self)
        # Re-assert the dark menu look: the page column's bare `background:`
        # stylesheet cascades into popups and overrides the app-level dark
        # QMenu rule in light mode (white menu, near-white text).
        menu.setStyleSheet(dark_menu_qss())
        act_files = menu.addAction("Select audio file(s)\u2026")
        act_folder = menu.addAction("Select a folder\u2026")
        chosen = menu.exec(QCursor.pos())
        if chosen is act_files:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select audio file(s)",
                os.path.dirname(self._files[0]) if self._files else "",
                self._filter)
            self._set_files(files)
        elif chosen is act_folder:
            folder = QFileDialog.getExistingDirectory(
                self, "Select folder", self._start_dir())
            if folder:
                self._set_files(_audio_files_in_folder(folder, self._extensions()))

    def _start_dir(self):
        for f in self._files:
            if os.path.isfile(f):
                return os.path.dirname(f)
        return ""

    def _extensions(self):
        """Supported audio extensions parsed from the file filter, e.g.
        {'.wav', '.flac', ...}."""
        import re as _re
        return {m.lower() for m in _re.findall(r"\*(\.\w+)", self._filter or "")}

    def _set_files(self, files):
        if not files:
            return
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
            # Files are taken as-is; folders are walked recursively for
            # supported audio files.
            files = [p for p in paths if os.path.isfile(p)]
            for p in paths:
                if os.path.isdir(p):
                    files.extend(_audio_files_in_folder(p, self._extensions()))
            if files:
                self._set_files(files)
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
    def __init__(self, icon, label, items, tooltip=""):
        super().__init__()
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        self.setCursor(Qt.PointingHandCursor)

        hl = QHBoxLayout(self)
        # right margin matches the dots rows so the '>' chevrons and the '...'
        # dots stack centered below each other (like SETTINGS LOCAL FILES)
        hl.setContentsMargins(12, 0, 14, 0)
        hl.setSpacing(0)

        if icon is not None:
            icon.setFixedWidth(24)
            hl.addWidget(icon)
            hl.addSpacing(6)

        hl.addWidget(_label_with_info(label, tooltip))

        self.combo = _ComboBox()
        self.combo.addItems(items)
        self.combo.setStyleSheet(_combo_ss())
        hl.addWidget(self.combo, 1)

        self._arrow = _ExpandArrow()
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
    def __init__(self, all_stems, selected_stems, save_rest, primary_target=None,
                 single_output=False, parent=None):
        super().__init__(parent)
        self._all_stems = all_stems[:]
        self._primary_target = primary_target.lower() if primary_target else None
        self._single_output = single_output
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
                elif self._single_output and self._primary_target:
                    # Single-output models can only separate their trained
                    # target; every other stem is auto-derived as the
                    # complement (mix minus the separated stems).
                    hint = QLabel("(mix \u2212 selected)")
                    hint.setStyleSheet(
                        "font-family:'Montserrat';font-size:8px;font-weight:600;"
                        f"color:{theme_manager.theme.text_muted};"
                        f"background:{theme_manager.theme.surface_alt};"
                        "padding:1px 5px;border-radius:3px;"
                    )
                    hint.setFixedHeight(16)
                    row.addWidget(hint)
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
    def __init__(self, icon, label, tooltip="", parent=None):
        super().__init__(parent)
        self._all_stems = []
        self._selected_stems = set()
        self._save_rest = False
        self._primary_target = None
        self._single_output = False
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        self.setCursor(Qt.PointingHandCursor)

        hl = QHBoxLayout(self)
        # right margin matches the dots/chevron rows so everything stacks
        # centered below each other (like SETTINGS LOCAL FILES)
        hl.setContentsMargins(12, 0, 14, 0)
        hl.setSpacing(0)

        if icon is not None:
            icon.setFixedWidth(24)
            hl.addWidget(icon)
            hl.addSpacing(6)

        hl.addWidget(_label_with_info(label, tooltip))

        self._summary = QLabel("Select stems\u2026")
        self._summary.setStyleSheet(
            "font-family:'Montserrat';font-size:11px;"
            f"color:{theme_manager.theme.text_dim};background:transparent;"
            "padding:0;"
        )
        hl.addWidget(self._summary, 1)

        self._arrow = _ExpandArrow()
        hl.addWidget(self._arrow)

    def set_stems(self, all_stems, primary_target=None, single_output=False):
        self._all_stems = all_stems[:] if all_stems else []
        self._primary_target = primary_target.lower() if primary_target else None
        self._single_output = bool(single_output)
        self._selected_stems = set()
        self._save_rest = False
        self._update_summary()

    def restore_selection(self, selected_stems, save_rest):
        """Re-apply a previously saved stem selection, keeping only stems
        this model actually produces (models differ in their stem sets)."""
        valid = {s.lower() for s in self._all_stems}
        self._selected_stems = {s.lower() for s in (selected_stems or [])
                                if s.lower() in valid}
        self._save_rest = bool(save_rest)
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
            single_output=self._single_output,
            parent=self
        )
        if run_blurred_dialog(dlg) == QDialog.Accepted:
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

# Module-level registry: the fetch worker is referenced here while it runs so
# the Python wrapper can never be garbage-collected mid-run (the QThread crash
# class — a wrapper destroyed while its thread is winding down corrupts Qt's
# thread state and surfaces later as a native access violation).
_NAMES_FETCHERS = set()
_NAMES_FETCHERS_LOCK = threading.Lock()


class _NamesFetchThread(QObject):
    """Fetches the zoo index once so ckpt filenames can be shown with their
    friendly full names in the MODEL LIBRARY. Plain QObject + daemon thread
    (never a QThread): signals are emitted from the worker and delivered
    queued to the GUI thread."""
    done = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        with _NAMES_FETCHERS_LOCK:
            _NAMES_FETCHERS.add(self)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self.done.emit(fetch_model_index())
        except Exception:
            self.done.emit([])
        finally:
            with _NAMES_FETCHERS_LOCK:
                _NAMES_FETCHERS.discard(self)


class _ModelItem(QFrame):
    selected = Signal(str, str, str, str, str, str, bool)
    settings_requested = Signal(str, str, str, str)

    def sizeHint(self):
        return QSize(0, 38)

    def __init__(self, name, ckpt="", yaml_path="", arch="", model_type="",
                 engine_type="", backend_module="", custom_backend_enabled=False,
                 display="", runnable=True, blocked_reason="", parent=None):
        super().__init__(parent)
        self._name = name
        self._ckpt = ckpt
        self._yaml = yaml_path
        self._arch = arch
        self._type = model_type
        self._engine_type = engine_type
        self._backend_module = backend_module
        self._custom = custom_backend_enabled
        self._runnable = runnable
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

        self._display = display or name
        self._lbl = QLabel(self._display)
        self._lbl.setObjectName("modelItemLabel")
        if self._display != name:
            self._lbl.setToolTip(name)
        # The label must be allowed to shrink (and elide) below its full
        # text width: rows whose name is a long ckpt filename and/or carry
        # extra badges (CUSTOM + type) would otherwise push the fixed
        # right-side cluster — the ··· menu — past the card's visible
        # edge in narrow panes, hiding it (the "3-dot missing in arch view"
        # bug). Shrinkable + elided keeps the dots on-screen at any width.
        self._lbl.setMinimumWidth(0)
        self._lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._lbl.setStyleSheet(
            f"font-family:{FONT_STACK};font-size:12px;"
            "font-weight:600;letter-spacing:0.5px;"
            f"color:{theme_manager.theme.text_sec};background:transparent;"
        )
        hl.addWidget(self._lbl, 1)

        if custom_backend_enabled and backend_module:
            ctag = QLabel("CUSTOM")
            ctag.setStyleSheet(_custom_badge_ss())
            ctag.setFixedHeight(17)
            hl.addWidget(ctag)

        if model_type:
            display = _type_title(model_type)
            tag = QLabel(display)
            tag.setToolTip(model_type)
            tag.setStyleSheet(_type_badge_ss(model_type))
            tag.setFixedHeight(17)
            hl.addWidget(tag)

        if not runnable:
            # The type this card would launch with has no branch in the
            # engine — surface it here so it's visible before clicking Run
            # (the pre-run validation still blocks the run as a backstop).
            btag = QLabel("NOT RUNNABLE")
            btag.setToolTip(
                "Not runnable in this build — " + (blocked_reason or "")
                + " Select a different model or update the app.")
            btag.setStyleSheet(_blocked_badge_ss())
            btag.setFixedHeight(17)
            hl.addWidget(btag)

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

    def _elide_label(self):
        """Clip the label to its allotted width (it is free to shrink because
        its size policy is Ignored), keeping the full name in `self._display`
        for search. The full text is exposed as a tooltip when elided."""
        lbl = self._lbl
        full = self._display or ""
        w = lbl.width()
        text = full if w <= 0 else lbl.fontMetrics().elidedText(
            full, Qt.ElideRight, w)
        if text != lbl.text():
            lbl.setText(text)
            if text != full and not lbl.toolTip():
                lbl.setToolTip(full)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_lbl", None) is not None:
            self._elide_label()

    def set_display(self, text):
        """Show a friendlier label (e.g. the zoo full name); the ckpt
        filename stays in the tooltip and is still matched by search."""
        if not text or text == self._display:
            return
        self._display = text
        self._lbl.setToolTip(self._name if text != self._name else "")
        self._elide_label()

    def _on_circle_toggled(self, checked):
        self._is_selected = checked
        self._update_style()
        if checked:
            self.selected.emit(self._name, self._ckpt, self._yaml, self._arch,
                               self._engine_type, self._backend_module, self._custom)

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

    def set_down(self, down):
        """Combo-arrow helper: point down (`v`) while the popup is open."""
        self.set_angle(90.0 if down else 0.0)

    def paintEvent(self, event):
        p = QPainter(self)
        paint_chevron(p, 12, 12, self._angle, self._hovered)
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


def _qcolor(value, fallback=None):
    """QColor from a theme token. rgba()/rgb() tokens are stylesheet strings
    QColor can't parse — handle that form manually or the pen falls back to
    black."""
    c = QColor(str(value))
    if c.isValid():
        return c
    s = str(value).strip().replace(" ", "")
    if s.startswith("rgb") and "(" in s and ")" in s:
        try:
            parts = [p for p in s[s.index("(") + 1:s.index(")")].split(",") if p]
            if len(parts) >= 3:
                alpha = int(float(parts[3]) * 255) if len(parts) > 3 else 255
                c = QColor(int(parts[0]), int(parts[1]), int(parts[2]), alpha)
                if c.isValid():
                    return c
        except (ValueError, IndexError):
            pass
    return QColor(fallback) if fallback else QColor(Qt.white)


class _LinkBadge(QWidget):
    """Small rounded chip with a minimal external-link glyph (box + arrow) —
    same scheme as the + ADD button: surface fill, dim border and muted glyph
    at rest; soft accent fill, accent border and bright glyph on hover. Click
    opens the architecture's paper / project page."""
    clicked = Signal()

    def __init__(self, tooltip="", parent=None):
        super().__init__(parent)
        self._hovered = False
        self.setFixedSize(16, 16)
        self.setCursor(Qt.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)

    def paintEvent(self, event):
        from PySide6.QtCore import QPointF
        t = theme_manager.theme
        hovered = self._hovered
        fill = _qcolor(theme_manager._accent_soft if hovered else t.surface)
        ring = _qcolor(theme_manager.accent if hovered else t.border_dim)
        glyph = _qcolor(t.text if hovered else t.text_dim)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # No resting border: the link shows as a bare glyph. Only on hover
        # draw the accent chip so the clickable affordance reads clearly.
        if hovered:
            p.setPen(QPen(ring, 1.0))
            p.setBrush(fill)
            p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1.0,
                                     self.height() - 1.0), 4, 4)
        # Minimal external-link: page outline with an open top-right corner,
        # plus a thin diagonal arrow escaping through it (feather geometry).
        p.setBrush(Qt.NoBrush)
        p.translate(self.width() / 2.0, self.height() / 2.0)
        p.scale(0.85, 0.85)
        pen = QPen(glyph, 1.05)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline([QPointF(2.7, 0.45), QPointF(2.7, 4.05),
                        QPointF(-4.05, 4.05), QPointF(-4.05, -2.7),
                        QPointF(-0.45, -2.7)])
        p.drawPolyline([QPointF(1.35, -4.05), QPointF(4.05, -4.05),
                        QPointF(4.05, -1.35)])
        p.drawLine(QPointF(-0.9, 0.9), QPointF(4.05, -4.05))
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
    model_selected = Signal(str, str, str, str, str, str, bool)
    ckpt_settings_requested = Signal(str, str, str, str)

    def __init__(self, arch_name, parent=None, dot_color=None,
                 title_display=None, info_url=None):
        super().__init__(parent)
        self._arch = arch_name
        self._expanded = False
        self._items = []
        self._custom_dot = dot_color
        self._title_display = title_display
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

        if dot_color is not None:
            dot_clr = dot_color
        else:
            dot_key = f"arch_dot_{_arch_dot_token(arch_name)}"
            dot_clr = getattr(theme_manager.theme, dot_key, theme_manager.theme.text_label)
        dot = QFrame()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{dot_clr};border:none;border-radius:4px;")
        hh.addWidget(dot)

        if title_display is not None:
            display = title_display
        else:
            display = ARCH_DISPLAY_NAMES.get(arch_name, arch_name.replace(" Architecture", ""))
        title_lbl = QLabel(display.upper())
        title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:700;"
            f"color:{theme_manager.theme.text};background:transparent;"
        )

        # Title + link icon hug each other on the left; the stretch pushes
        # the count badge and chevron to the right edge.
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        title_row.addWidget(title_lbl)

        # link chip — opens the architecture's paper / project page. Pass
        # info_url="" to suppress (target-mode cards have no link).
        if info_url is None:
            info_url = ARCH_INFO_LINKS.get(arch_name)
        self._info_lbl = None
        if info_url and info_url != "":
            info = _LinkBadge(f"About {display}")
            info.clicked.connect(lambda u=info_url: QDesktopServices.openUrl(QUrl(u)))
            title_row.addWidget(info)
            self._info_lbl = info

        title_row.addStretch()
        hh.addLayout(title_row, 1)

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
        # Cards for archs absent from the mvsepless zoo stay hidden until a
        # model is registered under them (see _apply_library_visibility).
        self._default_visible = True
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
                and obj is not self._toggle_btn
                and obj is not self._info_lbl):
            self._toggle_expand(animated=True)
            return True
        return super().eventFilter(obj, event)

    def _style_info_lbl(self, lbl):
        # The link glyph resolves its colors at paint time, so a theme
        # switch only needs a repaint.
        lbl.update()

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
                  engine_type="", backend_module="", custom_backend_enabled=False,
                  display=""):
        if name in self._items:
            return
        if not self._has_models:
            self._empty_lbl.setVisible(False)
            self._has_models = True
        # Runnable = the effective engine type (stored precise type or arch
        # fallback, yaml override/variant sniffs, side-car exemption) has a
        # branch in this build. Same rule the pre-run validation applies, so
        # the badge can never disagree with the actual run.
        mini = {"name": name, "ckpt": ckpt, "yaml": yaml_path,
                "arch": arch, "model_type": engine_type,
                "backend_module": backend_module,
                "custom_backend_enabled": custom_backend_enabled}
        blocked = engine_unsupported_models([mini], ARCH_TO_MODEL_TYPE)
        runnable = not blocked
        blocked_reason = (f"model type '{engine_effective_type(mini, ARCH_TO_MODEL_TYPE)}' "
                          "has no branch in the inference engine." if blocked else "")
        item = _ModelItem(name, ckpt, yaml_path, arch, model_type, engine_type,
                          backend_module, custom_backend_enabled, display=display,
                          runnable=runnable, blocked_reason=blocked_reason)
        item.selected.connect(lambda n, ck, y, a, et, bm, cb:
                              self.model_selected.emit(n, ck, y, a, et, bm, cb))
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
            self.setVisible(self._default_visible)
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
                match = (arch_match or text in w._name.lower()
                         or text in w._display.lower() or text in w._arch.lower())
                w.setVisible(match)
                if match:
                    has_ckpt_match = True

        card_visible = (arch_match or has_ckpt_match
                        or (not self._has_models and self._default_visible))
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


# ── Sort toggle ───────────────────────────────────────────────────────────

class _SortToggle(QWidget):
    """Compact architecture/target sort switch for the Model Library header:
    two small labels around a painted pill switch. `changed` emits True when
    the user switches to "sort by target", False for architecture."""
    changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on = False
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(5)
        self._arch_lbl = QLabel("sort by architecture")
        self._target_lbl = QLabel("sort by target")
        for lbl in (self._arch_lbl, self._target_lbl):
            lbl.setStyleSheet(
                "font-family:'Montserrat';font-size:8px;font-weight:600;"
            )
        self._sw = _MiniSwitch()
        self._sw.toggled.connect(self._set_on)
        hl.addWidget(self._arch_lbl)
        hl.addWidget(self._sw)
        hl.addWidget(self._target_lbl)
        self._relabel()
        self.setCursor(Qt.PointingHandCursor)

    def _set_on(self, on):
        self._on = on
        self._relabel()
        self.changed.emit(on)

    def _relabel(self):
        accent = theme_manager.accent
        dim = theme_manager.theme.text_muted
        self._arch_lbl.setStyleSheet(
            "font-family:'Montserrat';font-size:8px;font-weight:600;"
            f"color:{dim if self._on else accent};"
        )
        self._target_lbl.setStyleSheet(
            "font-family:'Montserrat';font-size:8px;font-weight:600;"
            f"color:{accent if self._on else dim};"
        )

    def is_target(self):
        return self._on

    def reapply_theme(self):
        self._sw.update()
        self._relabel()


class _MiniSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self._on = checked
        self.setFixedSize(30, 16)
        self.setCursor(Qt.PointingHandCursor)

    def is_checked(self):
        return self._on

    def set_checked(self, on):
        on = bool(on)
        if on != self._on:
            self._on = on
            self.toggled.emit(on)
            self.update()

    def mousePressEvent(self, e):
        self.set_checked(not self._on)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Off-state track: fixed near-black in dark theme (#101318), a soft
        # gray on the bright theme (#C4CAD2 — low-contrast against the light
        # background); the on-state turns accent.
        # (Don't feed it the theme's rgba(...) string — QColor can't parse
        # float-alpha and would fall back to full black.)
        if self._on:
            track = QColor(theme_manager.accent)
        else:
            track = QColor("#101318" if theme_manager.mode == "dark" else "#C4CAD2")
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(self.rect(), 8, 8)
        k, m = 12, 2
        x = self.width() - k - m if self._on else m
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(int(x), (self.height() - k) // 2, k, k)


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
        self._target_cards   = {}  # model_type -> _ArchCard (sort-by-target mode)
        self._sort_by_target = False
        self._friendly_names = {}  # ckpt filename -> zoo full name
        self._stems_by_model = {}  # model name -> {"stems": [...], "save_rest": bool}
        self._loaded_stems_by_model = {}  # stems loaded from settings, applied per-model
        self._mvsepless_archs = None  # archs listed in the mvsepless zoo (index fetch)
        self._library_finalized = False  # trailing stretch added to _model_layout?
        # Coalesced library re-render: model_registered arrives ~60x during a
        # cold settings load; rendering per signal made startup and theme
        # switches take ~5s. One flush per event-loop pass instead.
        self._vis_pending = False
        self._vis_timer = QTimer(self)
        self._vis_timer.setSingleShot(True)
        self._vis_timer.setInterval(0)
        self._vis_timer.timeout.connect(self._flush_library_visibility)
        self._build_ui()
        self._names_thread = _NamesFetchThread()
        self._names_thread.done.connect(self._on_friendly_names)
        self._names_thread.start()

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
        ll.setContentsMargins(32, 32, 10, 32)
        ll.setSpacing(0)

        t = theme_manager.theme

        # Configuration (pushed down so it aligns with the MODEL LIBRARY
        # header on the right column, which is centered against its 32px
        # search bar: column content starts at 168 + (32-18)/2 = 175)
        ll.addSpacing(7)
        ll.addWidget(_sec_hdr("Configuration"))
        ll.addSpacing(19)  # aligns the first INPUT row with the first model card

        cfg = QVBoxLayout()
        cfg.setSpacing(6)
        cfg.setContentsMargins(0, 0, 0, 0)

        self._input_row = _BrowseRow(
            None, "Input", "Select or drop audio file(s)\u2026",
            mode="file", file_filter=AUDIO_FILTER,
            tooltip="Audio file(s) to separate.\n"
                    "Click to browse or drop files here.")
        cfg.addWidget(self._input_row)

        self._output_row = _BrowseRow(
            None, "Output", "Select or drop output folder\u2026", mode="folder",
            tooltip="Folder where the stems are written.\n"
                    "Each model gets its own subfolder per checkpoint.")
        cfg.addWidget(self._output_row)

        self._fmt_row = _ComboRow(
            None, "Quality",
            ["FLAC (16-bit)", "FLAC (24-bit)", "WAV (32-bit float)"],
            tooltip="Output format applied to all stems (lossless).\n"
                    "FLAC stems that would clip are peak-normalized to stay FLAC.")
        self._fmt_combo = self._fmt_row.combo
        cfg.addWidget(self._fmt_row)

        self._output_stems_row = _OutputStemsRow(
            None, "Stems",
            tooltip="Which stems are saved to disk.\n"
                    "All of the model's stems are saved by default.")
        cfg.addWidget(self._output_stems_row)

        self._tta_row = _ComboRow(
            None, "TTA", ["Disabled", "Enabled"],
            tooltip="Extra pass on flipped audio, averaged in.\n"
                    "Cleaner stems, but roughly doubles the time.")
        self._tta_combo = self._tta_row.combo
        cfg.addWidget(self._tta_row)

        self._dev_row = _ComboRow(
            None, "Device", list_gpus(),
            tooltip="GPU (CUDA) is by far the fastest.\n"
                    "Use CPU only as a fallback.")
        self._device_combo = self._dev_row.combo
        if self._device_combo.count() > 1:
            # Default to the first GPU when one is available (index 0 is CPU).
            self._device_combo.setCurrentIndex(1)
        cfg.addWidget(self._dev_row)

        ll.addLayout(cfg)

        ll.addSpacing(36)
        ll.addStretch()  # anchor the Run Inference block to the bottom so it
                         # aligns horizontally with + ADD in the right column

        # Run Inference
        ll.addWidget(_sec_hdr("Run Inference"))
        ll.addSpacing(19)  # same header-to-content rhythm as CONFIGURATION

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.btn_run = GlyphButton("Separate", "\u25B6", _solid_icon_color,
                                   glyph_size=18, text_size=12)
        self.btn_run.setFixedSize(170, 44)
        self.btn_run.setStyleSheet(solid_button_ss())
        self.btn_run.clicked.connect(self._run)

        self.btn_stop = GlyphButton("Stop", "\u25A0", _stop_icon_color,
                                    glyph_size=16, text_size=12)
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

        self.btn_test_all = QPushButton("Test All Models")
        self.btn_test_all.setFixedSize(168, 44)
        self.btn_test_all.setCursor(Qt.PointingHandCursor)
        self.btn_test_all.setToolTip(
            "Runs every installed model once on the selected input, so you "
            "can see at a glance which models work in this build.")
        self.btn_test_all.setStyleSheet(_test_all_ss())
        self.btn_test_all.clicked.connect(self._test_all_models)

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_test_all)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        ll.addLayout(btn_row)

        # ── RIGHT COLUMN ───────────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet(f"background:{theme_manager.theme.bg};")
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 32, 32, 32)
        rl.setSpacing(0)

        # Model Library header, vertically centered against the search box
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 14, 0)
        hdr_row.setSpacing(10)
        hdr_row.addWidget(_sec_hdr("Model Library"))
        hdr_row.addStretch()

        self._sort_toggle = _SortToggle()
        self._sort_toggle.changed.connect(self._on_sort_changed)
        hdr_row.addWidget(self._sort_toggle)
        hdr_row.addSpacing(8)

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
            self._create_arch_card(arch)

        self._model_layout.addStretch()
        self._library_finalized = True

        scroll.setWidget(model_content)
        rl.addWidget(scroll, 1)
        rl.addSpacing(10)

        # + ADD (bottom of the library) — jumps to Settings where models
        # are added/registered. Styled like the LOG button in CONSOLE;
        # lifted 7px so its centre aligns with the 44px RUN/STOP row.
        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 14, 7)
        add_row.addStretch()
        self._add_btn = GlyphButton("Add", "+", _add_icon_color,
                                    glyph_size=18, text_size=9)
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
        self._sort_toggle.reapply_theme()

        # Update filter buttons
        for fb in self.findChildren(_FilterButton):
            fb.reapply_theme()

        # Update arch cards (architecture + target groupings)
        all_cards = [(arch, card, False) for arch, card in self._arch_cards.items()]
        all_cards += [(tk, card, True) for tk, card in self._target_cards.items()]
        for arch_name, card, is_target in all_cards:
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
            if is_target:
                dot_clr = _type_badge_color(arch_name) or "#9A9FB3"
            else:
                dot_key = f"arch_dot_{_arch_dot_token(arch_name)}"
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
            # Link info button
            if card._info_lbl is not None:
                card._style_info_lbl(card._info_lbl)
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
                        tag_lbl.setStyleSheet(_custom_badge_ss())
                    else:
                        tag_lbl.setStyleSheet(_type_badge_ss(item._type))

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
        self.btn_run.setStyleSheet(solid_button_ss())
        self.btn_test_all.setStyleSheet(_test_all_ss())
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

    def _create_arch_card(self, arch):
        """Build a library card for an architecture. Also used at runtime for
        archs the user registers that aren't part of the static ARCH_TYPES."""
        card = _ArchCard(arch)
        card.model_selected.connect(self._on_model_selected)
        card.ckpt_settings_requested.connect(self._on_ckpt_settings_requested)
        self._arch_cards[arch] = card
        if self._library_finalized:
            # Insert before the trailing stretch so dynamically added archs
            # keep a stable position at the end of the library.
            self._model_layout.insertWidget(max(self._model_layout.count() - 1, 0), card)
        else:
            self._model_layout.addWidget(card)
        return card

    def _create_target_card(self, type_key):
        """Build a grouping card keyed by model type for "sort by target" mode.
        Only categories that actually have models are shown (see
        _apply_library_visibility). The colored dot is the model type's color."""
        card = _ArchCard(
            type_key,
            dot_color=_type_badge_color(type_key) or "#9A9FB3",
            title_display=_type_title(type_key),
            info_url="",
        )
        card.model_selected.connect(self._on_model_selected)
        card.ckpt_settings_requested.connect(self._on_ckpt_settings_requested)
        self._target_cards[type_key] = card
        if self._library_finalized:
            self._model_layout.insertWidget(max(self._model_layout.count() - 1, 0), card)
        else:
            self._model_layout.addWidget(card)
        return card

    def _apply_library_visibility(self):
        """Default the arch grouping to the architectures listed in the
        mvsepless zoo (archs outside it stay hidden unless registered); the
        target grouping only shows categories that have models. The active
        grouping's cards are shown, the inactive grouping's are hidden."""
        for arch, card in self._arch_cards.items():
            card._default_visible = (
                self._mvsepless_archs is None
                or arch in self._mvsepless_archs
                or bool(card._items))
        for card in self._target_cards.values():
            card._default_visible = bool(card._items)
        self._order_target_cards()
        try:
            txt = self._search_bar.text()
        except RuntimeError:
            txt = ""
        self._filter_models(txt)

    def _order_target_cards(self):
        """Sort the by-target grouping cards alphabetically by title (so, e.g.,
        VOCALS sits at the end rather than first). Cards are moved into a
        contiguous block just before the trailing stretch; in architecture mode
        they are hidden anyway, so their layout position is invisible there."""
        if not self._target_cards:
            return
        lay = self._model_layout
        in_layout = set(id(w) for i in range(lay.count())
                        if (w := lay.itemAt(i).widget()) is not None)
        tcards = [w for w in self._target_cards.values() if id(w) in in_layout]
        if len(tcards) < 2:
            return
        for w in tcards:
            lay.removeWidget(w)
        tcards.sort(key=lambda w: _type_title(w._arch).lower())
        idx = max(lay.count() - 1, 0)
        for w in tcards:
            lay.insertWidget(idx, w)
            idx += 1

    def _filter_models(self, text):
        show_target = self._sort_by_target
        for arch_name, card in self._arch_cards.items():
            card.filter_models(text)
            if show_target:
                card.setVisible(False)
        for card in self._target_cards.values():
            card.filter_models(text)
            if not show_target:
                card.setVisible(False)

    def _on_sort_changed(self, target_mode):
        self._sort_by_target = bool(target_mode)
        # Re-derive visibility: the active grouping's cards show, the inactive
        # grouping's are hidden. Search matching applies to both.
        self._apply_library_visibility()

    # ── Slots ─────────────────────────────────────────────────────────

    def on_model_registered(self, model: dict):
        arch = model.get("arch", "")
        if arch and arch not in self._arch_cards:
            # Arch label outside the static list — give it a card dynamically.
            self._create_arch_card(arch)
        name = model["name"]
        display = self._friendly_names.get(os.path.basename(name).lower(), "")
        if arch in self._arch_cards:
            card = self._arch_cards[arch]
            # Re-registration (e.g. a type reconciliation) refreshes the item
            # in place so its badge shows the corrected type.
            card.remove_model(name)
            card.add_model(
                name, model.get("ckpt", ""),
                model.get("yaml", ""), arch,
                model.get("type", ""),
                model.get("model_type", ""),
                model.get("backend_module", ""),
                model.get("custom_backend_enabled", False),
                display=display)
        # Mirror into the by-target grouping (a re-registration re-homes the
        # model across categories if its type changed).
        type_key = model.get("type", "") or ""
        if type_key:
            for card in self._target_cards.values():
                card.remove_model(name)
            tcard = self._target_cards.get(type_key) or self._create_target_card(type_key)
            tcard.add_model(
                name, model.get("ckpt", ""), model.get("yaml", ""),
                arch, type_key,
                model.get("model_type", ""),
                model.get("backend_module", ""),
                model.get("custom_backend_enabled", False),
                display=display)
        self._schedule_library_visibility()

    def _schedule_library_visibility(self):
        """Coalesce model_registered fan-outs into one visibility pass."""
        if self._vis_pending:
            return
        self._vis_pending = True
        self._vis_timer.start()

    def _flush_library_visibility(self):
        self._vis_pending = False
        self._apply_library_visibility()

    def _on_friendly_names(self, models):
        """Index arrived: cache ckpt -> full name and refresh existing rows."""
        if models:
            self._mvsepless_archs = {getattr(m, "arch", "") for m in models} - {""}
            self._apply_library_visibility()
        for m in models:
            ck = os.path.basename(getattr(m, "checkpoint_url", "").split("?")[0]).lower()
            full = getattr(m, "full_name", "")
            if ck and full:
                self._friendly_names.setdefault(ck, full)
        for card in list(self._arch_cards.values()) + list(self._target_cards.values()):
            for i in range(card._list_vl.count()):
                w = card._list_vl.itemAt(i).widget()
                if isinstance(w, _ModelItem):
                    w.set_display(self._friendly_names.get(
                        os.path.basename(w._name.lower()), ""))
        # Re-apply an active search so rows now match by their new label
        try:
            txt = self._search_bar.text()
        except RuntimeError:
            txt = ""
        if txt and txt.strip():
            self._filter_models(txt)

    def on_model_removed(self, name: str):
        self._stems_by_model.pop(name, None)
        self._loaded_stems_by_model.pop(name, None)
        for card in self._arch_cards.values():
            card.remove_model(name)
        for card in self._target_cards.values():
            card.remove_model(name)
        # Re-hide unlisted archs whose last model was just removed.
        self._apply_library_visibility()

    def _deselect_item_everywhere(self, name):
        for card in list(self._arch_cards.values()) + list(self._target_cards.values()):
            for i in range(card._list_vl.count()):
                w = card._list_vl.itemAt(i).widget()
                if isinstance(w, _ModelItem) and w._name == name:
                    w.set_selected(False)
                    break

    def _on_model_selected(self, name, ckpt, yaml_path, arch, engine_type="",
                           backend_module="", custom_backend_enabled=False):
        if self._selected_model:
            old_name = self._selected_model.get("name")
            old_arch = self._selected_model.get("arch")
            if old_name == name and old_arch == arch:
                self._snapshot_model_stems(old_name)
                self._deselect_item_everywhere(name)
                self._selected_model = None
                self._output_stems_row.set_stems([])
                return

            old_card = self._arch_cards.get(old_arch)
            if old_card and old_arch != arch:
                old_card.deselect_all_models()
            self._deselect_item_everywhere(old_name)
            # Remember the outgoing model's stem choice so it can be restored
            # when the user switches back to it.
            if old_name:
                self._snapshot_model_stems(old_name)

        self._selected_model = {
            "name": name, "ckpt": ckpt, "yaml": yaml_path, "arch": arch,
            "model_type": engine_type,
            "backend_module": backend_module,
            "custom_backend_enabled": custom_backend_enabled,
        }
        self._refresh_output_stems_row()

    def _refresh_output_stems_row(self):
        """(Re)apply the current _selected_model's stem context to the output
        row: list the model's config stems, flag single-output models, and
        default to "all stems" unless the user customized THIS model's
        selection. Shared by a real card click and the TEST ALL batch — the
        batch previously reused whatever stem state the row had from the last
        manual click (or none), so a model whose config pins a single target
        (e.g. bs_inst_large2_unwa: num_stems 1, target_instrument
        "instrument", instruments [vocals, instrument]) silently emitted only
        that one stem instead of its full output."""
        model = self._selected_model or {}
        yaml_path = model.get("yaml", "")
        instruments = []
        target_inst = None
        single_output = False
        if yaml_path and os.path.isfile(yaml_path):
            try:
                import yaml as _yaml
                with open(yaml_path, encoding="utf-8") as _f:
                    _cfg = _yaml.load(_f, Loader=_yaml.FullLoader)
                instruments = _cfg.get("training", {}).get("instruments", []) or []
                target_inst = _cfg.get("training", {}).get("target_instrument", None)
                # A single-output model (model.num_stems == 1) can only
                # separate its trained target; every other stem is auto-
                # derived as mix minus the separated stems, so flag those
                # in the stem dialog. (Shared helper: utils/stem_planning.py)
                single_output = is_single_output(_cfg)
            except Exception:
                pass
        self._output_stems_row.set_stems(
            instruments, primary_target=target_inst, single_output=single_output)
        # Default: all stems. Only restore a selection the user explicitly
        # made for THIS model (never carry one model's choice to another).
        name = model.get("name", "")
        saved = (self._loaded_stems_by_model.pop(name, None)
                 or self._stems_by_model.get(name))
        if saved is not None:
            stems = saved.get("stems", [])
            rest = saved.get("save_rest", False)
            if stems or rest:  # non-default choice only; empty == "all stems"
                self._output_stems_row.restore_selection(stems, rest)

    def _on_ckpt_settings_requested(self, name, ckpt, yaml_path, arch):
        self.ckpt_settings_requested.emit(name, ckpt, yaml_path, arch)

    # ── Settings ──────────────────────────────────────────────────────

    def _snapshot_model_stems(self, model_name):
        """Store a model's current stem choice, collapsing the default
        "all stems" state to an empty marker so only genuine user
        customizations are persisted."""
        row = self._output_stems_row
        stems = row.get_selected_stems()
        rest = row.get_save_rest()
        all_lower = {s.lower() for s in row._all_stems}
        if all_lower:
            is_default = (not rest and {s.lower() for s in stems} == all_lower)
        else:
            is_default = not rest
        self._stems_by_model[model_name] = {
            "stems": [] if is_default else stems,
            "save_rest": rest,
        }

    def save_settings(self):
        # Snapshot the current model's stem choice so it survives relaunches.
        if self._selected_model and self._selected_model.get("name"):
            self._snapshot_model_stems(self._selected_model["name"])
        return {
            "input_files":   self._input_row.value()
                             if isinstance(self._input_row.value(), list) else [],
            "output_folder": self._output_row.value()
                             if isinstance(self._output_row.value(), str) else "",
            "output_format": self._fmt_combo.currentText(),
            "tta":           self._tta_combo.currentText(),
            "device":        self._device_combo.currentText(),
            "stems":         self._output_stems_row.get_selected_stems(),
            "save_rest":     self._output_stems_row.get_save_rest(),
            "stems_by_model": dict(self._stems_by_model),
        }

    def load_settings(self, d):
        files = d.get("input_files", [])
        if isinstance(files, list) and files:
            self._input_row.set_value(files)
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
        sbm = d.get("stems_by_model", None)
        if isinstance(sbm, dict):
            clean = {}
            for mname, sv in sbm.items():
                if (isinstance(mname, str) and mname and isinstance(sv, dict)
                        and isinstance(sv.get("stems", []), list)):
                    clean[mname] = {
                        "stems": [s for s in sv["stems"] if isinstance(s, str)],
                        "save_rest": bool(sv.get("save_rest", False)),
                    }
            self._loaded_stems_by_model = clean
        # NOTE: legacy global "stems" is intentionally ignored — a stem choice
        # must never be carried to a different model; models default to all
        # stems unless the user customized them for that specific model.

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
        # Any error below would otherwise die silently inside this Qt slot
        # (PySide prints to stderr, invisible in the GUI) and leave the user
        # with a run that "does nothing". Surface it to the console instead.
        try:
            self._run_inner()
        except Exception as exc:
            import traceback as _tb
            self.log_output.emit(f"ERROR: {exc}")
            for ln in _tb.format_exc().splitlines():
                self.log_output.emit(ln)
            self.process_running.emit(False)
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def _run_inner(self):
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
        # Prefer the precise engine type recorded at install time (e.g.
        # 'scnet_tran', 'bandit_v2' — the arch label collapses variants).
        # Fall back to the arch→type map for manually registered models;
        # the yaml sniffs below then refine coarse labels as a safety net.
        model_type = self._selected_model.get("model_type") or \
            ARCH_TO_MODEL_TYPE.get(arch, "bs_roformer")
        # "Bandit Architecture" covers both bandit generations but only v2
        # configs nest hyper-parameters under `kwargs:` (v1 uses `model:`).
        # Refine the type from the yaml so the engine builds the right class;
        # the engine also sniffs this as a safety net (utils/settings.py).
        if model_type == "bandit" and yaml_path and os.path.isfile(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as _f:
                    _bcfg = yaml.load(_f, Loader=yaml.FullLoader)
                if _bcfg and "model" not in _bcfg and "kwargs" in _bcfg:
                    model_type = "bandit_v2"
            except Exception:
                pass
        # Same collapse as bandit: "SCNet Architecture" maps every SCNet
        # variant back to plain 'scnet', but tran configs carry tran_* keys
        # the plain class rejects (and needs the tran architecture). Refine
        # from the yaml so the engine builds SCNet_Tran.
        if model_type == "scnet" and yaml_path and os.path.isfile(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as _f:
                    _scfg = yaml.load(_f, Loader=yaml.FullLoader)
                _sm = (_scfg or {}).get("model", {}) or {}
                if isinstance(_sm, dict) and any(
                        str(k).startswith("tran_") for k in _sm):
                    model_type = "scnet_tran"
            except Exception:
                pass

        # Pre-run validation: the model's files must still exist on disk.
        # A missing checkpoint here means the install lives in the app's
        # code folder, which app updates wipe (older builds); re-registering
        # stores the model under the writable data folder instead.
        missing = [p for p in (ckpt, yaml_path) if p and not os.path.isfile(p)]
        if missing:
            _nm = self._selected_model.get("name") or ""
            self.log_output.emit(
                f"ERROR: model '{_nm}' is missing its files on disk:")
            for p in missing:
                self.log_output.emit("  " + p)
            self.log_output.emit(
                "The files were probably removed by an app update (older "
                "builds stored them in the app's code folder). Re-add the "
                "model from SETTINGS to store it in the persistent data "
                "folder, or reinstall it from the model library.")
            if self._tmp_input and os.path.isdir(self._tmp_input):
                try:
                    shutil.rmtree(self._tmp_input)
                except OSError:
                    pass
                self._tmp_input = None
            return

        # Pre-run validation: the model type must have an inference-engine
        # branch. Otherwise inference.py dies mid-launch with a raw
        # "Unknown model type" traceback; catch it here so the console shows
        # a clear error and nothing is spawned. Accepted types are parsed out
        # of get_model_from_config (backend.msst_catalog — the same source
        # the TRAINING tab uses) plus the ONNX 'mdxnet' special case handled
        # directly in inference.py. The yaml may override the type exactly as
        # the engine does, and models shipping an author side-car backend
        # bypass the built-in branches entirely.
        known_engine_types = model_catalog.model_types()
        if known_engine_types:
            engine_types = set(known_engine_types) | {"mdxnet"}
            # mdxnet is special-cased in inference.py *before* the config is
            # consulted, so its yaml never overrides the type.
            effective_type = (model_type if model_type == "mdxnet" else
                              resolve_engine_model_type(model_type, yaml_path))
            # Fork models with an author side-car bypass the built-ins.
            if not has_custom_sidecar(self._selected_model) and \
                    effective_type not in engine_types:
                _nm = (self._selected_model.get("name") or ""
                       or os.path.basename(ckpt or ""))
                self.log_output.emit(
                    f"ERROR: model '{_nm}' cannot be run: model type "
                    f"'{effective_type}' has no branch in this build's "
                    "inference engine.")
                self.log_output.emit(
                    "Supported model types: " + ", ".join(sorted(engine_types))
                    + ".")
                self.log_output.emit(
                    "Pick another model, or update the app if this model "
                    "needs a newer engine.")
                if self._tmp_input and os.path.isdir(self._tmp_input):
                    try:
                        shutil.rmtree(self._tmp_input)
                    except OSError:
                        pass
                    self._tmp_input = None
                return

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

                resolved = resolve_target(
                    config, selected_stems, effective_save_rest, custom_selection)
                if resolved is not KEEP:
                    if "training" not in config:
                        config["training"] = {}
                    config["training"]["target_instrument"] = resolved

                self._tmp_yaml = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False)
                yaml.dump(config, self._tmp_yaml, default_flow_style=False)
                self._tmp_yaml.close()
                yaml_path = self._tmp_yaml.name
            except Exception:
                pass

        # Nest single-inference outputs under a per-checkpoint subfolder
        # ("<output>/<ckpt>/<song> (<stem>)"), matching the per-ckpt layout the
        # CONSOLE cards expect — they open that subfolder on "Open folder".
        # The shared flat template is kept so the ensemble runners (which pass
        # a per-model store_dir) don't nest the ckpt name a second time.
        out_root = self._output_row.value()
        store_dir = out_root if isinstance(out_root, str) else ""
        if store_dir and ckpt:
            store_dir = os.path.join(
                store_dir, os.path.splitext(os.path.basename(ckpt))[0])
        # Remember the per-model output folder so the completion scan reports
        # only THIS run's files — scanning the whole base folder picked up
        # the previous model's stems when runs were fast back-to-back (their
        # mtimes fell inside the scan window), leaking them onto this card.
        self._last_store_dir = store_dir

        cmd = [
            get_python_exe(), os.path.join(REPO_ROOT, "inference.py"),
            "--model_type",        model_type,
            "--config_path",       yaml_path,
            "--start_check_point", ckpt,
            "--input_folder",      self._tmp_input,
            "--store_dir",         store_dir,
        ]
        if force_cpu: cmd.append("--force_cpu")
        else: cmd += ["--device_ids"] + [str(d) for d in device_ids]

        # Map the Quality selection to upstream's --pcm_type: PCM_16/24 are
        # written as FLAC for every stem (hot stems are peak-normalized so
        # the integer codec doesn't clip), FLOAT as 32-bit WAV.
        fmt = self._fmt_combo.currentText()
        if "24" in fmt:
            cmd += ["--pcm_type", "PCM_24"]
        elif "FLAC" in fmt:
            cmd += ["--pcm_type", "PCM_16"]
        else:
            cmd += ["--pcm_type", "FLOAT"]
        # Flat "<song> (<stem>)" files instead of upstream's per-song folders.
        cmd += ["--filename_template", INFERENCE_FILENAME_TEMPLATE]

        if use_tta: cmd.append("--use_tta")
        # Fork architectures ship their own backend .py (downloaded to
        # models/custom/<module> at install time, under the writable APP_DIR);
        # pass the folder so the engine loads the model class from the
        # author's file.
        if self._selected_model.get("custom_backend_enabled"):
            bm = self._selected_model.get("backend_module", "")
            if bm:
                cmd += ["--custom_backend",
                        os.path.join(APP_DIR, "models", "custom", bm)]
        # The stem *selection* is applied through the temp config above
        # (single stem -> target_instrument); "rest" maps onto upstream's
        # --extract_instrumental (mix minus the vocals / first stem).
        if effective_save_rest:
            cmd.append("--extract_instrumental")

        if isinstance(store_dir, str) and store_dir.strip():
            self.log_output.emit(f"Output directory: {store_dir}")

        # Show the resolved stem names (target + mix-complement) up front so
        # naming issues are visible before any file is written. Shared
        # planner: utils/stem_planning.py (same logic the engine applies).
        try:
            plan_cfg_path = self._selected_model.get("yaml", "")
            if plan_cfg_path and os.path.isfile(plan_cfg_path):
                with open(plan_cfg_path, "r", encoding="utf-8") as _f:
                    plan_cfg = yaml.load(_f, Loader=yaml.FullLoader)
                plan = plan_output_stems(
                    plan_cfg, selected_stems, save_rest, all_stems)
                if plan:
                    labels = [str(s) for s in plan]
                    if effective_save_rest and len(plan) >= 2:
                        labels[-1] = f"{plan[-1]} (mix \u2212 selected)"
                        if len(plan) == 2:
                            labels[0] = f"{plan[0]} (target)"
                    elif len(plan) == 1:
                        labels[0] = f"{plan[0]} (target)"
                    self.log_output.emit("Stems: " + ", ".join(labels))
        except Exception:
            pass

        self.input_files_submitted.emit(
            [f for f in files if isinstance(f, str)] if isinstance(files, list) else []
        )
        self.model_selected.emit(self._selected_model.get("name", "") if self._selected_model else "")

        for i, f in enumerate(files if isinstance(files, list) else []):
            if i == 0:
                self.log_output.emit(f"Processing: {os.path.basename(f)}")
            else:
                self.log_output.emit(f"Queued: {os.path.basename(f)}")

        self._run_started = time.time()
        self._runner = ProcessRunner(cmd, cwd=REPO_ROOT)
        self._runner.log_line.connect(self.log_output.emit)
        self._runner.finished.connect(self._on_finished)
        self.process_running.emit(True)
        self._runner.start()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _report_written_files(self):
        """Upstream inference.py writes the stems silently; list every audio
        file that appeared under the output folder since the run started so
        the CONSOLE cards pick them up ("Wrote file: ..." lines).

        The scan is scoped to the current model's own per-checkpoint
        subfolder: scanning the whole base output folder could re-report the
        previous run's files when runs are fast back-to-back (their mtimes
        still fall inside the `since` window) and leak their stems onto this
        model's card."""
        out = getattr(self, "_last_store_dir", None) or self._output_row.value()
        if not isinstance(out, str) or not os.path.isdir(out):
            return
        since = getattr(self, "_run_started", 0) - 2
        found = []
        for root, _dirs, names in os.walk(out):
            for n in names:
                if os.path.splitext(n)[1].lower() in (".wav", ".flac", ".mp3", ".ogg"):
                    p = os.path.join(root, n)
                    try:
                        if os.path.getmtime(p) >= since:
                            found.append(p)
                    except OSError:
                        pass
        for p in sorted(found, key=os.path.getmtime):
            self.log_output.emit(f"Wrote file: {p}")

    # ── Batch smoke test: run every installed model on the same input ─────

    def _test_all_models(self):
        """Test every registered model once on the selected input and report
        per-model PASS/FAIL, continuing past failures so a single broken
        model never blocks the rest. Useful to find which installed models
        fail (missing files, unsupported type, engine crash) in this build.

        The whole body is guarded so an unexpected error is logged to the
        console instead of dying silently inside the Qt slot (PySide only
        prints slot exceptions to stderr, invisible in the packaged app)."""
        try:
            from PySide6.QtCore import QEventLoop
            from ui.widgets.runtime_dialog import ensure_runtime
            if not ensure_runtime(self):
                return
            # The batch iterates every installed model, so no library model
            # needs to be pre-selected — only the shared input/output are
            # required (and a missing selection must show a real dialog, not
            # a swallowed NameError: QMessageBox is imported at module level).
            if not self._input_row.value():
                QMessageBox.warning(self, "Missing input",
                                    "Please select at least one audio file.")
                return
            if not self._output_row.value():
                QMessageBox.warning(self, "Missing input",
                                    "Please select an output folder.")
                return

            models = [m for m in settings_store.load()
                      .get("registered_models", [])
                      if m.get("name")]
            if not models:
                self.log_output.emit("No installed models to test.")
                return
            models = sorted(models,
                            key=lambda m: (m.get("name") or "").lower())

            original = self._selected_model
            self._batch_testing = True
            self._batch_cancel = False
            results = []
            try:
                self.btn_run.setEnabled(False)
                self.btn_test_all.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.process_running.emit(True)
                self.log_output.emit(
                    f"\u2500\u2500 TESTING ALL INSTALLED MODELS ({len(models)})"
                    " \u2014 same input, one run per model \u2500\u2500")
                for i, model in enumerate(models, 1):
                    if self._batch_cancel:
                        self.log_output.emit("Batch aborted by user.")
                        break
                    name = model.get("name", "?")
                    self.log_output.emit(f"[{i}/{len(models)}] {name}")
                    # Mirror the batch markers to stdout so msst-gui.log
                    # (tee) records which model was in flight when the app
                    # dies natively overnight — the console page alone
                    # disappears with the process.
                    print(f"[TEST-ALL {i}/{len(models)}] {name}",
                          flush=True)
                    self._selected_model = model
                    # Each model runs with ITS OWN stem context (default:
                    # all stems from its config). Without this the row kept
                    # the last manually-clicked model's state, so configs
                    # pinning a single target emitted only that one stem.
                    self._refresh_output_stems_row()
                    self._run_inner()

                    runner = self._runner
                    if runner is None:
                        # A pre-run guard (missing files / unsupported type)
                        # already logged the details; summarize here.
                        reason = self._batch_block_reason(model)
                        self.log_output.emit(f"  TEST FAIL: {reason}")
                        print(f"[TEST-ALL {i}/{len(models)}] {name}: "
                              f"FAIL ({reason})", flush=True)
                        results.append((name, False, reason))
                        continue

                    code_box = {}
                    loop = QEventLoop()
                    runner.finished.connect(
                        lambda c: (code_box.update(code=c), loop.quit()))
                    loop.exec()
                    code = code_box.get("code", -1)
                    self._runner = None
                    if code == 0:
                        self.log_output.emit("  TEST OK")
                        print(f"[TEST-ALL {i}/{len(models)}] {name}: OK",
                              flush=True)
                        results.append((name, True, ""))
                    else:
                        code_str = describe_exit_code(code)
                        self.log_output.emit(
                            f"  TEST FAIL (exit code {code_str})")
                        print(f"[TEST-ALL {i}/{len(models)}] {name}: "
                              f"FAIL ({code_str})", flush=True)
                        results.append(
                            (name, False, f"exit code {code_str}"))
                    # _on_finished re-enables the run buttons after each
                    # model; re-lock them so the batch keeps control.
                    self.btn_run.setEnabled(False)
                    self.btn_test_all.setEnabled(False)
                if self._batch_cancel:
                    results.append(("<aborted>", False, "cancelled"))
            finally:
                self._batch_testing = False
                self._batch_cancel = False
                self._selected_model = original
                # Put the stem row back on the model the user had selected
                # before the batch (each batch run swapped it to its own).
                if original:
                    self._refresh_output_stems_row()
                else:
                    self._output_stems_row.set_stems([])
                passed = [n for n, ok, _r in results if ok]
                failed = [(n, r) for n, ok, r in results if not ok]
                self.log_output.emit("\u2500\u2500 TEST SUMMARY \u2500\u2500")
                summary = (
                    f"{len(passed)}/{len(results)} models passed"
                    + (f" \u2014 FAILED: {', '.join(n for n, _ in failed)}"
                       if failed else ""))
                self.log_output.emit(summary)
                print(f"[TEST-ALL SUMMARY] {summary}", flush=True)
                self.btn_run.setEnabled(True)
                self.btn_test_all.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.process_running.emit(False)
        except Exception as exc:
            import traceback as _tb
            self.log_output.emit(f"ERROR: Test All Models failed: {exc}")
            for ln in _tb.format_exc().splitlines():
                self.log_output.emit(ln)
            self._batch_testing = False
            self._batch_cancel = False
            self.btn_run.setEnabled(True)
            self.btn_test_all.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.process_running.emit(False)

    def _batch_block_reason(self, model):
        """Why a model was blocked before launch (mirrors _run_inner's
        pre-run guards) for the batch summary line."""
        miss = [p for p in (model.get("ckpt", ""), model.get("yaml", ""))
                if p and not os.path.isfile(p)]
        if miss:
            return "files missing on disk (see log above)"
        if engine_unsupported_models([model], ARCH_TO_MODEL_TYPE):
            eff = engine_effective_type(model, ARCH_TO_MODEL_TYPE)
            return f"model type '{eff}' has no engine branch in this build"
        return "blocked before launch (see log above)"

    def _stop(self):
        # Only announce a stop when an inference job is actually ours to stop
        # — the console Stop button can reach this while an ensemble job (a
        # different runner) is active, and a false process_running(False)
        # would clear the console's job state mid-run.
        if getattr(self, "_batch_testing", False):
            self._batch_cancel = True
        if self._runner:
            self._runner.stop()
            self.process_running.emit(False)

    def _on_finished(self, code):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self._report_written_files()
        self.process_running.emit(False)
        if code == 0:
            self.log_output.emit("Completed: processing")
        else:
            # A native child crash (e.g. exit code 0xC0000005) is readable
            # here, not a bare negative integer.
            code_str = describe_exit_code(code)
            self.log_output.emit(
                f"ERROR: processing failed (exit code {code_str})")
            print(f"[RUN FAILED] exit code {code_str}", flush=True)
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
