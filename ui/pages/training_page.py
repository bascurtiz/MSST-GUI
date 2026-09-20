"""
ui/pages/training_page.py
TRAINING tab — set up and run model training with a live monitor.

The job itself is train.py (borrowed from ZFTurbo's
Music-Source-Separation-Training) spawned under the app runtime, exactly
like inference; this page only builds the command line, streams the
process output and renders progress / validation metrics / the log.

Layout: CONFIGURATION | TRAINING SETTINGS | TRAINING MONITOR with the
RUN TRAINING action bar along the bottom. Every colour comes from the
active theme and the rows re-use the INFERENCE page's row widgets.
"""
from dataclasses import replace as _dc_replace
import os
import re
import glob
import html
import time
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QFileDialog, QScrollArea, QSizePolicy,
    QDialog, QProgressBar, QSpinBox, QDoubleSpinBox, QMessageBox, QPlainTextEdit, QComboBox,
    QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QRect, QUrl, QSize, QObject, QTimer
from PySide6.QtGui import (
    QPainter, QPen, QColor, QFont, QFontMetrics, QDesktopServices,
    QIntValidator, QPainterPath, QPalette, QPixmap,
)

from backend.runner import ProcessRunner
from backend.paths import REPO_ROOT, get_python_exe
from backend.gpu_utils import (
    list_gpus, selected_gpu_memory_gb, selected_gpu_snapshot, query_gpu_memory,
    HEAVY_CONFORMER_TYPES,
)
from backend.train_advisor import (
    advise_from_selection, advise_from_probe, supports_torch_checkpoint,
)
from backend import settings as settings_store
from backend.train_cmd import (
    DEFAULT_RUN_OPTS, accelerate_available, build_train_command,
    detect_custom_backend, inject_lora_defaults, redact_train_cmd,
    resolve_launcher, subprocess_env,
)
from ui.strings import T_EXPORT_WEIGHTS, T_TRAIN_SETUP, T_OPEN_WANDB, HELP_OPTIONAL_CAPTION
from ui.theme import theme_manager, UIConstants
from ui.widgets.common import (
    PageHeader, PageHelpDialog, solid_button_ss, outline_button_ss,
    accent_outline_btn_ss, EllipsisButton, GlyphButton, css_color,
    _outline_icon_color, _solid_icon_color, _stop_icon_color, run_blurred_dialog,
    OptionalFold, _page_edge_scroll_ss, _help_section, _HELP_COL_GUTTER,
    CHIP_GLYPH, BOLT_GLYPH,
)
from ui.pages.inference_page import (
    _sec_hdr, _row_ss, _lbl_ss, _combo_ss, _ComboBox, _ExpandArrow,
    _InfoDot, _CircleCheck, _MiniSwitch, _SwitchRow, ROW_H,
    CONFIG_DOT_SLOT, CONFIG_VALUE_GAP,
)
from ui.widgets.ckpt_settings_dialog import _TitleBar
from ui.widgets.pretrained_models_dialog import PretrainedModelsDialog
from ui.widgets.smooth_bar import SmoothBar


# ── Catalogues ────────────────────────────────────────────────────────────────
# Model types, optimizers, losses and metrics are NOT listed here: they are
# read from the upstream code itself (utils/settings.py, utils/model_utils.py,
# utils/losses.py) by backend/msst_catalog.py, so a new architecture / loss /
# optimizer shows up in this tab after a plain file sync. Only the dataset
# types (documented in docs/dataset_types.md, not enumerated in code) stay
# spelled out.
from backend import msst_catalog as catalog

DATASET_TYPES = [
    ("1", "Type 1 — MUSDB: one folder per song with <stem>.wav files"),
    ("2", "Type 2 — Stems: one folder per stem, any number of files"),
    ("3", "Type 3 — CSV list: 'instrum,path' rows"),
    ("4", "Type 4 — MUSDB aligned: all stems cut from the same position"),
    ("5", "Type 5 — Precomputed chunks (MUSDB layout, 50% overlap)"),
    ("6", "Type 6 — Aligned + explicit mixture.wav per song"),
    ("7", "Type 7 — Class-balanced aligned (rare instruments boosted)"),
]

WANDB_HOME = "https://wandb.ai/"
WANDB_AUTHORIZE = "https://wandb.ai/authorize"
_WANDB_ICON_PATH = os.path.join(REPO_ROOT, "resources", "wandb-favicon.png")

LABEL_W = 160
MONO = "'Courier New','Consolas',monospace"


# ── Small shared styles ──────────────────────────────────────────────────────

def _value_ss(muted=False):
    t = theme_manager.theme
    return (
        "font-family:'Montserrat';font-size:11px;"
        f"color:{t.text_muted if muted else t.text};"
        "background:transparent;border:none;"
    )


def _edit_ss():
    t = theme_manager.theme
    return (
        "QLineEdit{background:transparent;border:none;"
        "font-family:'Montserrat';font-size:11px;"
        f"color:{t.text};padding:0;}}"
        f"QLineEdit::placeholder{{color:{t.text_muted};}}"
    )


def _spin_ss():
    t = theme_manager.theme
    box = (
        "font-family:'Courier New',monospace;font-size:13px;font-weight:bold;"
        f"color:{theme_manager.accent};background:{t.input_bg};"
        f"border:1px solid {t.border_dim};border-radius:4px;padding:2px 6px;}}"
    )
    return (
        "QSpinBox{" + box +
        "QSpinBox::up-button, QSpinBox::down-button{width:0px;border:none;}"
        "QDoubleSpinBox{" + box +
        "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button{width:0px;border:none;}"
    )


def _card_ss():
    t = theme_manager.theme
    return (
        "QFrame#monCard{"
        "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        f"stop:0 {t.surface},stop:1 {t.bg});"
        f"border:1px solid {t.border_visible};"
        f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px;}}"
    )


def _card_title_ss():
    t = theme_manager.theme
    return (
        "font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
        f"color:{t.text};background:transparent;letter-spacing:1.5px;"
    )


def _mono_box_ss():
    t = theme_manager.theme
    return (
        f"background:{t.console_bg};color:{t.console_text};"
        f"border:1px solid {t.border};border-radius:6px;"
        f"font-family:{MONO};font-size:10px;padding:8px 10px;"
    )


def _chip_ss(active):
    t = theme_manager.theme
    if active:
        color, bg, bd = theme_manager.accent, theme_manager._accent_soft, theme_manager.accent
    else:
        color, bg, bd = t.text_label, t.surface_alt, t.border
    return (
        "QPushButton{font-family:'Montserrat';font-size:8px;font-weight:700;"
        f"color:{color};background:{bg};border:1px solid {bd};"
        "padding:1px 7px;border-radius:3px;letter-spacing:0.5px;}"
        f"QPushButton:hover{{color:{theme_manager.accent};border:1px solid {theme_manager.accent};}}"
    )


def _small_button_ss():
    """The '+ Add' / LOG button look: surface fill, dim border, accent hover."""
    t = theme_manager.theme
    return (
        "QPushButton{"
        f"background:{t.surface};color:{t.text_dim};"
        f"border:1px solid {t.border_dim};border-radius:6px;"
        "font-family:'Montserrat',sans-serif;font-weight:600;"
        "font-size:10px;padding:0 14px;}"
        f"QPushButton:hover{{background:{theme_manager._accent_soft};"
        f"color:{t.text};border:1px solid {theme_manager.accent};}}"
    )


def _cancel_btn_ss(font_size=10):
    t = theme_manager.theme
    e = QColor(t.error)
    return (
        "QPushButton{"
        f"background:{t.surface};color:{t.text_muted};"
        f"border:1px solid {t.border_dim};border-radius:4px;"
        "font-family:'Montserrat',sans-serif;font-weight:600;"
        f"font-size:{int(font_size)}px;}}"
        f"QPushButton:hover{{color:{t.error};"
        f"border:1px solid rgba({e.red()},{e.green()},{e.blue()},0.40);}}"
    )


def _primary_btn_ss():
    return (
        "QPushButton{"
        f"background:{theme_manager.accent};color:{theme_manager._accent_text};"
        "border:none;border-radius:4px;"
        "font-family:'Montserrat',sans-serif;font-weight:600;font-size:10px;}"
        f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
        f"QPushButton:pressed{{background:{theme_manager.accent};}}"
    )


def _label_font():
    """The row-label font as the stylesheet renders it (bold 9px, 1.5px
    letter spacing) — for measuring whether a label fits on one line."""
    f = QFont("Montserrat")
    f.setPixelSize(UIConstants.SEC_HDR_FONT_SIZE)
    f.setBold(True)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
    return f


def _lbl_with_info(label, tooltip=""):
    """Row label column (wider than the INFERENCE page's: training labels
    like RESUME need the room). The ⓘ slot matches Inference CONFIGURATION:
    CONFIG_DOT_SLOT in front of the glyph, CONFIG_VALUE_GAP after it (added
    by the row). A label that cannot fit on one line wraps inside the row
    (the width is set explicitly: a word-wrapped QLabel's own sizeHint uses
    a narrow aspect-ratio heuristic and would wrap MODEL TYPE too)."""
    wrap = QWidget()
    wrap.setStyleSheet("background:transparent;")
    wrap.setFixedWidth(LABEL_W)
    hl = QHBoxLayout(wrap)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(5)
    lb = QLabel(label.upper())
    lb.setStyleSheet(_lbl_ss())
    lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    if tooltip:
        # Same slot as Inference: label text | 10px | 14px ⓘ, every dot on
        # one x so CONFIG_VALUE_GAP after the wrap is the air before the value.
        lb.setFixedWidth(LABEL_W - CONFIG_DOT_SLOT)
        natural = QFontMetrics(_label_font()).horizontalAdvance(label.upper()) + 2
        lb.setWordWrap(natural > LABEL_W - CONFIG_DOT_SLOT)
        hl.addWidget(lb, 0, Qt.AlignVCenter)
        hl.addSpacing(CONFIG_DOT_SLOT - _InfoDot.W - hl.spacing())
        hl.addWidget(_InfoDot(tooltip), 0, Qt.AlignVCenter)
    else:
        natural = QFontMetrics(_label_font()).horizontalAdvance(label.upper()) + 2
        if natural <= LABEL_W:
            lb.setFixedWidth(natural)
        else:
            lb.setWordWrap(True)
            lb.setFixedWidth(LABEL_W)
        hl.addWidget(lb, 0, Qt.AlignVCenter)
        hl.addStretch()
    return wrap


def _fmt_num(v):
    """Compact number text for the settings fields (9.0e-05 -> 9e-05)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ── Painted bits ─────────────────────────────────────────────────────────────

class _WrapLabel(QWidget):
    """Path value that wraps *anywhere* onto at most two lines. Paths have no
    spaces, so QLabel's word wrap would either overflow or break at random;
    this paints the text itself and elides the head of a longer path so the
    file name stays visible."""
    MAX_LINES = 2

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self._text = ""
        self._placeholder = placeholder
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(60)

    @staticmethod
    def _font():
        f = QFont("Montserrat")
        f.setPixelSize(11)
        return f

    def set_text(self, t):
        self._text = t or ""
        self.setToolTip(self._text if len(self._text) > 40 else "")
        self.update()

    def text(self):
        return self._text

    def _flags(self):
        # Real paths have no spaces — wrap anywhere and elide the head.
        # Placeholders are a single elided line so a 42px card never clips
        # "Checkpoint to start from…" into three half-visible rows.
        if self._text:
            return Qt.TextWrapAnywhere | Qt.AlignLeft
        return Qt.AlignLeft | Qt.TextSingleLine

    def _fitted(self, fm, w):
        s = self._text
        if not s:
            return fm.elidedText(self._placeholder, Qt.ElideRight, max(w, 10))
        avail = self.MAX_LINES * fm.height() + 1
        flags = self._flags()
        while len(s) > 4:
            r = fm.boundingRect(QRect(0, 0, max(w, 10), 100000), flags, s)
            if r.height() <= avail:
                return s
            core = s[1:] if s.startswith("…") else s
            s = "…" + core[max(1, len(core) // 12):]
        return s

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        f = self._font()
        p.setFont(f)
        t = theme_manager.theme
        p.setPen(css_color(t.text if self._text else t.text_muted))
        fm = QFontMetrics(f)
        s = self._fitted(fm, self.width())
        p.drawText(self.rect(), self._flags() | Qt.AlignVCenter, s)
        p.end()


class _SquareCheck(QWidget):
    """Square check box (accent fill + white tick) for the augmentation row."""
    toggled = Signal(bool)

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self._checked = bool(checked)
        self._hovered = False
        self.setFixedSize(16, 16)
        self.setCursor(Qt.PointingHandCursor)

    def is_checked(self):
        return self._checked

    def set_checked(self, on):
        on = bool(on)
        if on != self._checked:
            self._checked = on
            self.update()
            self.toggled.emit(on)

    def toggle(self):
        self.set_checked(not self._checked)

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
            self.toggle()
        e.accept()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        t = theme_manager.theme
        if self._checked:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme_manager.accent))
            p.drawRoundedRect(r, 3, 3)
            pen = QPen(QColor("#FFFFFF"), 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            path = QPainterPath()
            path.moveTo(4.5, 8.3)
            path.lineTo(7.0, 10.8)
            path.lineTo(11.8, 5.2)
            p.drawPath(path)
        else:
            border = QColor(theme_manager.accent) if self._hovered else css_color(t.border_dim)
            p.setPen(QPen(border, 1.5))
            p.setBrush(css_color(t.surface_alt))
            p.drawRoundedRect(r, 3, 3)
        p.end()


class _PencilIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(css_color(theme_manager.theme.text_sec), 1.5)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(2.5, 13.5)
        path.lineTo(2.5, 10.5)
        path.lineTo(10.5, 2.5)
        path.lineTo(13.5, 5.5)
        path.lineTo(5.5, 13.5)
        path.closeSubpath()
        p.drawPath(path)
        p.drawLine(QPointF(8.5, 4.5), QPointF(11.5, 7.5))
        p.end()


class _ButtonIcon(QWidget):
    """Folder / document outline icon for the bottom-right buttons."""

    _wandb_pm = None

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._color = QColor("#FFFFFF")
        self.setFixedSize(16, 16)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_color(self, c):
        self._color = css_color(c)
        self.update()

    @classmethod
    def _wandb_pixmap(cls):
        if cls._wandb_pm is None:
            pm = QPixmap(_WANDB_ICON_PATH)
            cls._wandb_pm = pm if not pm.isNull() else QPixmap()
        return cls._wandb_pm

    def _paint_wandb(self, p):
        """Official wandb mark, tinted to the button gray (not the yellow brand)."""
        src = self._wandb_pixmap()
        if src.isNull():
            return False
        scaled = src.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        tinted = QPixmap(self.size())
        tinted.fill(Qt.transparent)
        qp = QPainter(tinted)
        qp.setRenderHint(QPainter.SmoothPixmapTransform)
        qp.drawPixmap(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled)
        qp.setCompositionMode(QPainter.CompositionMode_SourceIn)
        qp.fillRect(tinted.rect(), self._color)
        qp.end()
        p.drawPixmap(0, 0, tinted)
        return True

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._kind == "wandb" and self._paint_wandb(p):
            p.end()
            return
        pen = QPen(self._color, 1.5)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        if self._kind == "export":
            # tray with an arrow rising out of it
            path.moveTo(2.5, 9.5)
            path.lineTo(2.5, 14.0)
            path.lineTo(13.5, 14.0)
            path.lineTo(13.5, 9.5)
            p.drawPath(path)
            p.drawLine(QPointF(8.0, 10.5), QPointF(8.0, 2.0))
            p.drawLine(QPointF(4.8, 5.2), QPointF(8.0, 2.0))
            p.drawLine(QPointF(11.2, 5.2), QPointF(8.0, 2.0))
        elif self._kind == "wandb":
            path.moveTo(8.0, 1.8)
            path.lineTo(13.2, 4.8)
            path.lineTo(13.2, 11.2)
            path.lineTo(8.0, 14.2)
            path.lineTo(2.8, 11.2)
            path.lineTo(2.8, 4.8)
            path.closeSubpath()
            p.drawPath(path)
        elif self._kind == "folder":
            path.moveTo(1.5, 4.0)
            path.lineTo(6.0, 4.0)
            path.lineTo(8.0, 6.0)
            path.lineTo(14.5, 6.0)
            path.lineTo(14.5, 13.0)
            path.lineTo(1.5, 13.0)
            path.closeSubpath()
            p.drawPath(path)
        else:
            path.moveTo(3.0, 1.5)
            path.lineTo(10.0, 1.5)
            path.lineTo(13.0, 4.5)
            path.lineTo(13.0, 14.5)
            path.lineTo(3.0, 14.5)
            path.closeSubpath()
            p.drawPath(path)
            p.drawLine(QPointF(10.0, 1.5), QPointF(10.0, 4.5))
            p.drawLine(QPointF(10.0, 4.5), QPointF(13.0, 4.5))
            p.drawLine(QPointF(5.5, 8.0), QPointF(10.5, 8.0))
            p.drawLine(QPointF(5.5, 11.0), QPointF(10.5, 11.0))
        p.end()


class _IconTextButton(QPushButton):
    """Small outline button with a painted leading icon (Open Output /
    View Logs) — same look as the CONSOLE page's LOG button."""

    def __init__(self, text, kind, parent=None):
        super().__init__("", parent)
        self._hovered = False
        self.setStyleSheet(_small_button_ss())
        self.setCursor(Qt.PointingHandCursor)
        self._content = QWidget(self)
        self._content.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._content.setStyleSheet("background:transparent;")
        h = QHBoxLayout(self._content)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        self._icon = _ButtonIcon(kind)
        self._lbl = QLabel(text)
        h.addWidget(self._icon, 0, Qt.AlignVCenter)
        h.addWidget(self._lbl, 0, Qt.AlignVCenter)
        self._refresh()

    def fit_contents(self, height=40, h_pad=28):
        """Lock height; width hugs the icon + label plus side padding."""
        self._content.adjustSize()
        self.setFixedSize(self._content.width() + h_pad, height)
        self._refresh()

    def _refresh(self):
        t = theme_manager.theme
        c = t.text if self._hovered else t.text_dim
        self._icon.set_color(c)
        self._lbl.setStyleSheet(
            "background:transparent;border:none;"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:10px;"
            f"color:{c};"
        )
        self._content.adjustSize()
        self._content.move((self.width() - self._content.width()) // 2,
                           (self.height() - self._content.height()) // 2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def enterEvent(self, e):
        self._hovered = True
        self._refresh()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self._refresh()
        super().leaveEvent(e)


# ── Rows ─────────────────────────────────────────────────────────────────────

def _same_paths(a, b):
    """True when two path lists are the same after slash/dot normalization."""
    def norm(p):
        return os.path.normpath(p).replace("\\", "/")
    return [norm(p) for p in (a or []) if p] == [norm(p) for p in (b or []) if p]


class _PathRow(QFrame):
    """Label + path value + '···' browse button. `mode` is "file" or
    "folder"; folder rows accept several folders via drag & drop (train.py
    takes lists for --data_path / --valid_path)."""
    changed = Signal()

    def __init__(self, label, placeholder, mode="file", file_filter="All files (*.*)",
                 tooltip="", multi=False, start_dir="", parent=None):
        super().__init__(parent)
        self._mode = mode
        self._filter = file_filter
        self._multi = multi
        self._start_dir = start_dir
        self._paths = []
        self._drag_over = False
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setAcceptDrops(True)
        self.setStyleSheet(_row_ss())

        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 0, 14, 0)
        hl.setSpacing(0)
        hl.addWidget(_lbl_with_info(label, tooltip))
        hl.addSpacing(CONFIG_VALUE_GAP)
        self._val = _WrapLabel(placeholder)
        hl.addWidget(self._val, 1)
        self._btn = EllipsisButton()
        self._btn.clicked.connect(self._browse)
        hl.addWidget(self._btn)

    def add_extra(self, w):
        """Dock a widget (e.g. the LATEST chip) between the value and '···'."""
        hl = self.layout()
        i = hl.count() - 1
        hl.insertSpacing(i, 8)
        hl.insertWidget(i + 1, w, 0, Qt.AlignVCenter)
        hl.insertSpacing(i + 2, 4)

    # — values —
    def paths(self):
        return list(self._paths)

    def value(self):
        return self._paths[0] if self._paths else ""

    def set_paths(self, paths):
        new = [p for p in (paths or []) if p]
        if _same_paths(new, self._paths):
            return
        self._paths = new
        if len(self._paths) <= 1:
            self._val.set_text(self.value())
        else:
            names = ", ".join(os.path.basename(p.rstrip("\\/")) or p for p in self._paths)
            self._val.set_text(f"{len(self._paths)} folders: {names}")
        self.changed.emit()

    def set_value(self, v):
        self.set_paths([v] if v else [])

    def _dir_hint(self):
        for p in self._paths:
            if os.path.isdir(p):
                return p
            if os.path.isfile(p):
                return os.path.dirname(p)
        return self._start_dir or ""

    def _browse(self):
        if self._mode == "folder":
            path = QFileDialog.getExistingDirectory(self, "Select folder", self._dir_hint())
            if path:
                self.set_paths([path])
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select file", self._dir_hint(), self._filter)
        if path:
            self.set_paths([path])

    # — drag & drop —
    def _drag_ss(self):
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

    @staticmethod
    def _local_paths(mime):
        return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]

    def dragEnterEvent(self, e):
        if self._local_paths(e.mimeData()):
            self._set_drag_over(True)
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

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
            folders = []
            for p in paths:
                d = p if os.path.isdir(p) else os.path.dirname(p)
                if d and d not in folders:
                    folders.append(d)
            if folders:
                self.set_paths(folders if self._multi else folders[:1])
        else:
            files = [p for p in paths if os.path.isfile(p)]
            if files:
                self.set_paths(files[:1])
        e.acceptProposedAction()


class _ClickablePathRow(_PathRow):
    """_PathRow whose row body is clickable (like the other rows): clicking
    anywhere on the row — except the '···' browse button — emits
    pick_requested. With picker=True a small chevron sits next to '···'
    advertising the extra action (lit while a picker dialog is open)."""
    pick_requested = Signal()

    def __init__(self, *args, picker=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._arrow = None
        if picker:
            self._arrow = _ExpandArrow()
            self._arrow.clicked.connect(self.pick_requested.emit)
            self.add_extra(self._arrow)
        self.setCursor(Qt.PointingHandCursor)

    def set_picker_active(self, on):
        """Rotate the chevron down while the picker dialog is open (the same
        active cue the combo rows use while their popup shows)."""
        if self._arrow is not None:
            self._arrow.set_down(bool(on))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.pick_requested.emit()
        super().mousePressEvent(e)


class _EditRow(QFrame):
    """Label + free-text value (numbers) — batch size, learning rate, …"""

    def __init__(self, label, tooltip="", placeholder="", int_only=False, parent=None):
        super().__init__(parent)
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 0, 14, 0)
        hl.setSpacing(0)
        hl.addWidget(_lbl_with_info(label, tooltip))
        hl.addSpacing(CONFIG_VALUE_GAP)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setStyleSheet(_edit_ss())
        self.edit.setFixedHeight(ROW_H - 2)
        if int_only:
            self.edit.setValidator(QIntValidator(0, 1_000_000_000, self.edit))
        hl.addWidget(self.edit, 1)

    def value(self):
        return self.edit.text().strip()

    def set_value(self, v):
        self.edit.setText(_fmt_num(v))

    def mousePressEvent(self, e):
        self.edit.setFocus()
        super().mousePressEvent(e)


class _ComboRowT(QFrame):
    """Label + combo with the app's chevron (same as INFERENCE's rows, with
    the wider training label column). Items are (key, display) pairs."""

    def __init__(self, label, items, tooltip="", parent=None):
        super().__init__(parent)
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        self.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 0, 14, 0)
        hl.setSpacing(0)
        hl.addWidget(_lbl_with_info(label, tooltip))
        hl.addSpacing(CONFIG_VALUE_GAP)
        self.combo = _ComboBox()
        for key, disp in items:
            self.combo.addItem(disp, key)
        self.combo.setStyleSheet(_combo_ss())
        # Let the combo shrink below its widest item (the popup still shows
        # the full names): three label columns must fit the 1100px minimum
        # window width.
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(6)
        hl.addWidget(self.combo, 1)
        self._arrow = _ExpandArrow()
        hl.addWidget(self._arrow)
        self.combo.popupOpened.connect(lambda: self._arrow.set_down(True))
        self.combo.popupClosed.connect(lambda: self._arrow.set_down(False))

    def key(self):
        return self.combo.currentData()

    def set_key(self, key):
        idx = self.combo.findData(key)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def set_items(self, items, keep=None):
        self.combo.blockSignals(True)
        self.combo.clear()
        for key, disp in items:
            self.combo.addItem(disp, key)
        self.combo.blockSignals(False)
        if keep is not None:
            self.set_key(keep)

    def mousePressEvent(self, e):
        self.combo.showPopup()
        super().mousePressEvent(e)


class _ChevronRow(QFrame):
    """Clickable row with a wrapping value text and the '>' chevron; opens a
    dialog. `label=""` gives the icon + text variant (Edit Configuration)."""
    clicked = Signal()

    def __init__(self, label, tooltip="", placeholder="", icon=None, parent=None):
        super().__init__(parent)
        self._placeholder = placeholder
        self.setObjectName("cfgRow")
        self.setMinimumHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        self.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 8, 14, 8)
        hl.setSpacing(0)
        if icon is not None:
            hl.addWidget(icon, 0, Qt.AlignVCenter)
            hl.addSpacing(10)
        if label:
            hl.addWidget(_lbl_with_info(label, tooltip))
            hl.addSpacing(CONFIG_VALUE_GAP)
        self._val = QLabel(placeholder)
        self._val.setWordWrap(True)
        self._val.setStyleSheet(_value_ss(muted=False))
        self._val.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hl.addWidget(self._val, 1)
        self._arrow = _ExpandArrow()
        self._arrow.clicked.connect(self.clicked.emit)
        hl.addWidget(self._arrow, 0, Qt.AlignVCenter)

    def set_text(self, text):
        if text:
            self._val.setText(text)
            self._val.setStyleSheet(_value_ss(muted=False))
        else:
            self._val.setText(self._placeholder)
            self._val.setStyleSheet(_value_ss(muted=True))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


class _CheckRow(QFrame):
    """Square check + small-caps label (+ ⓘ). Clicking the row toggles."""
    toggled = Signal(bool)

    def __init__(self, label, tooltip="", checked=True, parent=None):
        super().__init__(parent)
        self.setObjectName("cfgRow")
        self.setFixedHeight(ROW_H)
        self.setStyleSheet(_row_ss())
        self.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.setSpacing(0)
        self._check = _SquareCheck(checked)
        self._check.toggled.connect(self.toggled.emit)
        hl.addWidget(self._check, 0, Qt.AlignVCenter)
        hl.addSpacing(12)
        lb = QLabel(label.upper())
        lb.setStyleSheet(_lbl_ss())
        hl.addWidget(lb, 0, Qt.AlignVCenter)
        if tooltip:
            hl.addSpacing(CONFIG_DOT_SLOT - _InfoDot.W)
            hl.addWidget(_InfoDot(tooltip), 0, Qt.AlignVCenter)
        hl.addStretch()

    def is_checked(self):
        return self._check.is_checked()

    def set_checked(self, on):
        self._check.set_checked(on)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._check.toggle()
        super().mousePressEvent(e)


# ── Monitor widgets ───────────────────────────────────────────────────────────

class _Card(QFrame):
    """Bordered monitor card with a small-caps title row."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("monCard")
        self.setStyleSheet(_card_ss())
        self._vl = QVBoxLayout(self)
        self._vl.setContentsMargins(16, 12, 16, 14)
        self._vl.setSpacing(10)
        self._title_row = QHBoxLayout()
        self._title_row.setContentsMargins(0, 0, 0, 0)
        self._title_row.setSpacing(8)
        self._title = QLabel(title.upper())
        self._title.setStyleSheet(_card_title_ss())
        self._title_row.addWidget(self._title)
        self._title_row.addStretch()
        self._vl.addLayout(self._title_row)

    def add_title_widget(self, w):
        self._title_row.addWidget(w, 0, Qt.AlignVCenter)

    def add(self, w, stretch=0):
        self._vl.addWidget(w, stretch)

    def add_layout(self, l, stretch=0):
        self._vl.addLayout(l, stretch)


class _TileGrid(QWidget):
    """Metric tiles in a grid whose column count follows the width (5 on a
    wide window, fewer when the monitor column is squeezed)."""
    TILE_W = 78
    GAP = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(self.GAP)
        self._grid.setVerticalSpacing(self.GAP)
        self._tiles = []
        self._cols = 5

    def set_tiles(self, tiles):
        while self._grid.count():
            self._grid.takeAt(0)
        self._tiles = list(tiles)
        self._relayout()

    def _wanted_cols(self):
        w = max(self.width(), 1)
        return max(2, min(5, (w + self.GAP) // (self.TILE_W + self.GAP)))

    def _relayout(self):
        cols = self._wanted_cols()
        self._cols = cols
        while self._grid.count():
            self._grid.takeAt(0)
        for c in range(6):
            self._grid.setColumnStretch(c, 0)
        for i, tile in enumerate(self._tiles):
            self._grid.addWidget(tile, i // cols, i % cols)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._tiles and self._wanted_cols() != self._cols:
            self._relayout()


class _MetricTile(QFrame):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.setObjectName("metricTile")
        t = theme_manager.theme
        self.setStyleSheet(
            f"QFrame#metricTile{{background:{t.surface};border:1px solid {t.border_visible};"
            "border-radius:6px;}"
        )
        self.setFixedHeight(54)
        self.setMinimumWidth(60)
        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(2)
        self._name = QLabel(name)
        self._name.setAlignment(Qt.AlignCenter)
        self._name.setStyleSheet(
            f"font-family:{MONO};font-size:9px;color:{t.text_muted};background:transparent;border:none;"
        )
        self._value = QLabel("—")
        self._value.setAlignment(Qt.AlignCenter)
        self._value.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:14px;font-weight:600;"
            f"color:{t.text};background:transparent;border:none;"
        )
        vl.addWidget(self._name)
        vl.addWidget(self._value)

    def set_value(self, v, tooltip=""):
        if v is None:
            self._value.setText("—")
        else:
            self._value.setText(f"{v:.2f}")
        self.setToolTip(tooltip)


# ── Dialogs ───────────────────────────────────────────────────────────────────

class _DialogBase(QDialog):
    """Frameless dialog with the app's dark title strip (shared with the
    CHECKPOINT SETTINGS dialog) and a Cancel / primary button row."""

    def __init__(self, title, parent=None, size=(460, 420)):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setModal(True)
        self.resize(*size)
        self.setMinimumSize(360, 240)
        t = theme_manager.theme
        self.setStyleSheet(
            f"QDialog{{background:{t.bg};border:1px solid {t.border_dim};border-radius:8px;}}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._title_bar = _TitleBar(title, self)
        root.addWidget(self._title_bar)
        self._content = QWidget()
        self._content.setObjectName("dlgContent")
        self._content.setStyleSheet(f"#dlgContent{{background:{t.bg};}}")
        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(24, 16, 24, 18)
        self.body.setSpacing(10)
        root.addWidget(self._content, 1)

    def add_buttons(self, primary_text, on_primary):
        row = QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(0, 6, 0, 0)
        cancel = QPushButton("Cancel")
        cancel.setFixedSize(110, 34)
        cancel.setStyleSheet(_cancel_btn_ss())
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        row.addStretch()
        ok = QPushButton(primary_text)
        ok.setFixedSize(140, 34)
        ok.setStyleSheet(_primary_btn_ss())
        ok.clicked.connect(on_primary)
        row.addWidget(ok)
        self.body.addLayout(row)

    def hint(self, text):
        lb = QLabel(text)
        lb.setWordWrap(True)
        lb.setStyleSheet(
            f"font-family:'Montserrat';font-size:9px;color:{theme_manager.theme.text_dim};"
            "background:transparent;"
        )
        self.body.addWidget(lb)
        return lb

    @staticmethod
    def _scroll():
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        t = theme_manager.theme
        sc.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
            f"QScrollBar::handle:vertical{{background:{t.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
        )
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return sc


def _link_button(text, fn):
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    b.setFlat(True)
    b.setStyleSheet(
        "QPushButton{background:transparent;border:none;"
        "font-family:'Montserrat';font-size:13px;"
        f"color:{theme_manager.accent};text-align:left;padding:0;}}"
        f"QPushButton:hover{{color:{theme_manager._accent_hover};}}"
    )
    b.clicked.connect(fn)
    return b


_YAML_QUICK_TIPS = (
    ("training.target_instrument",
     "Stem the model focuses on (e.g. vocals, other). "
     "Must match your dataset and checkpoint."),
    ("training.other_fix",
     "Type-2 stem-folder datasets: checks that other is actually instrumental."),
    ("model.use_torch_checkpoint",
     "Gradient checkpointing — much less VRAM, slower steps. "
     "Enable when batch size OOMs."),
    ("inference.num_overlap",
     "Chunk overlap during validation / inference. "
     "Use 1 while training (faster validation); raise for cleaner separation."),
)


def _yaml_tips_panel():
    t = theme_manager.theme
    frame = QFrame()
    frame.setObjectName("yamlTipsPanel")
    frame.setStyleSheet(
        f"QFrame#yamlTipsPanel{{background:{t.surface_alt};"
        f"border:1px solid {t.border_dim};border-radius:6px;}}"
    )
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(12, 12, 12, 12)
    outer.setSpacing(10)
    head = QLabel("YAML QUICK TIPS")
    head.setStyleSheet(
        f"font-family:'Montserrat';font-size:10px;font-weight:bold;"
        f"color:{theme_manager.accent};letter-spacing:1px;background:transparent;"
    )
    outer.addWidget(head)
    for key, tip in _YAML_QUICK_TIPS:
        block = QWidget()
        block.setStyleSheet("background:transparent;")
        bl = QVBoxLayout(block)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)
        key_lb = QLabel(key)
        key_lb.setWordWrap(True)
        key_lb.setStyleSheet(
            f"font-family:{MONO};font-size:10px;color:{t.text};background:transparent;"
        )
        tip_lb = QLabel(tip)
        tip_lb.setWordWrap(True)
        tip_lb.setStyleSheet(
            f"font-family:'Montserrat';font-size:9px;color:{t.text_sec};"
            "background:transparent;"
        )
        bl.addWidget(key_lb)
        bl.addWidget(tip_lb)
        outer.addWidget(block)
    outer.addStretch(1)
    return frame


def _option_row(check, title, desc="", *, help_style=False):
    """check widget + title (+ dim description) for the dialogs' lists."""
    t = theme_manager.theme
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    hl = QHBoxLayout(w)
    hl.setContentsMargins(0, 4, 0, 4)
    hl.setSpacing(12)
    hl.addWidget(check, 0, Qt.AlignTop if desc else Qt.AlignVCenter)
    col = QVBoxLayout()
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(1)
    title_px = 13 if help_style else 11
    desc_px = 13 if help_style else 9
    title_c = t.text_sec if help_style else t.text
    tl = QLabel(title)
    tl.setStyleSheet(
        f"font-family:'Montserrat';font-size:{title_px}px;color:{title_c};"
        "background:transparent;"
    )
    col.addWidget(tl)
    if desc:
        dl = QLabel(desc)
        dl.setWordWrap(True)
        dl.setStyleSheet(
            f"font-family:'Montserrat';font-size:{desc_px}px;color:{t.text_muted};"
            "background:transparent;"
        )
        col.addWidget(dl)
    hl.addLayout(col, 1)
    w.setCursor(Qt.PointingHandCursor)
    w.mousePressEvent = lambda e, c=check: c.toggle() if hasattr(c, "toggle") else c.set_checked(not c.is_checked())
    return w


# Loss picker columns — classic / spectral on the left, perceptual + SNR
# family on the right; penalties stack last, bleedless then fullness.
_LOSS_COLUMNS = (
    (
        "masked_loss", "mse_loss", "l1_loss", "multistft_loss",
        "spec_masked_loss", "spec_rmse_loss",
    ),
    (
        "log_wmse_loss", "l1_snr_loss", "l1_snr_db_loss",
        "stft_l1_snr_db_loss", "multi_l1_snr_db_loss",
        "bleedless_penalty_loss", "fullness_penalty_loss",
    ),
)


def _split_loss_columns(options):
    """Bucket loss choices into the two display columns. Unknown future
    keys land on the shorter side so the grid stays balanced."""
    by_key = {item[0]: item for item in options}
    used = set()
    cols = []
    for keys in _LOSS_COLUMNS:
        col = []
        for key in keys:
            item = by_key.get(key)
            if item is not None:
                col.append(item)
                used.add(key)
        cols.append(col)
    for item in options:
        if item[0] in used:
            continue
        min(cols, key=len).append(item)
    return cols


def _pill_switch(off_text, on_text, checked=False):
    """Architecture-vs-Target pill: off_text | switch | on_text."""
    wrap = QWidget()
    wrap.setStyleSheet("background:transparent;")
    hl = QHBoxLayout(wrap)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(5)
    off_lbl = QLabel(off_text)
    on_lbl = QLabel(on_text)
    sw = _MiniSwitch(checked, wrap)

    def relabel(_on=None):
        on = sw.is_checked()
        accent = theme_manager.accent
        dim = theme_manager.theme.text_muted
        base = ("font-family:'Montserrat';font-size:8px;font-weight:600;"
                "margin-top:1px;background:transparent;")
        off_lbl.setStyleSheet(base + f"color:{dim if on else accent};")
        on_lbl.setStyleSheet(base + f"color:{accent if on else dim};")

    sw.toggled.connect(relabel)
    off_lbl.setCursor(Qt.PointingHandCursor)
    on_lbl.setCursor(Qt.PointingHandCursor)
    off_lbl.mousePressEvent = lambda e: sw.set_checked(False)
    on_lbl.mousePressEvent = lambda e: sw.set_checked(True)
    hl.addWidget(off_lbl, 0, Qt.AlignVCenter)
    hl.addWidget(sw, 0, Qt.AlignVCenter)
    hl.addWidget(on_lbl, 0, Qt.AlignVCenter)
    wrap._sw = sw
    relabel()
    return wrap


class _MultiSelectDialog(PageHelpDialog):
    """Loss / metric picker — same help-sheet chrome as GPUs / Workers / Seed."""

    def __init__(self, title, options, selected, hint="", toggle=None,
                 caption="", section="", parent=None):
        self._options = options
        self._checks = {}
        self._toggle_sw = None
        cap = caption or "The job uses this selection on Train."
        if not section:
            section = "LOSSES" if "LOSS" in title.upper() else "METRICS"
        columns = 2 if section in ("LOSSES", "METRICS") else 1
        metrics = section == "METRICS"
        losses = section == "LOSSES"
        self._col_width = 340 if metrics else (440 if losses else 300)
        left = self._build_list(section, options, selected, cap, columns=columns)
        below = self._build_toggle(toggle) if toggle is not None else None
        data = {
            "title": "TRAINING",
            "heading": f"{title}  ·  TRAINING",
            "intro": hint,
            "required": [
                hint or "Pick one or more entries.",
                "Select all or clear the list.",
            ],
            "required_caption": cap,
            "primary_text": "APPLY",
            "min_width": (
                740 if metrics else (940 if losses else (720 if columns == 2 else 680))),
            "max_width": 780 if metrics else (980 if losses else 0),
        }
        if losses:
            data["hide_scrollbar"] = True
        if toggle is not None:
            data["optional"] = [toggle[0]]
            data["optional_below"] = True
            data["optional_caption"] = HELP_OPTIONAL_CAPTION
        super().__init__(data, parent, left=left, below=below)
        inner = getattr(self, "_inner", None)
        lay = inner.layout() if inner is not None else None
        if lay is not None:
            lay.addSpacing(28)
            self._refit()

    def _copy(self, muted=False):
        t = theme_manager.theme
        color = t.text_muted if muted else t.text_sec
        return (
            f"font-family:'Montserrat';font-size:13px;color:{color};"
            "background:transparent;"
        )

    def _col_body(self, top=8):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, top, 0, 0)
        vl.setSpacing(12)
        return w, vl

    def _text_action(self, text, fn):
        return _link_button(text, fn)

    def _apply_preset(self, keys):
        pick = set(keys)
        for key, cb in self._checks.items():
            cb.set_checked(key in pick)

    def _build_list(self, section, options, selected, caption, columns=1):
        body, vl = self._col_body()
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(16)
        actions.addWidget(self._text_action(
            "Select all", lambda: self._set_all(True)))
        actions.addWidget(self._text_action(
            "Clear", lambda: self._set_all(False)))
        if section == "METRICS":
            actions.addWidget(self._text_action(
                "Vocal tuning",
                lambda: self._apply_preset(catalog.VOCAL_TUNING_METRICS)))
        actions.addStretch(1)
        vl.addLayout(actions)
        sel = set(selected or [])

        def _row(item):
            key, label, desc = item
            cb = _CircleCheck()
            cb.set_square(True)
            cb.set_checked(key in sel)
            self._checks[key] = cb
            return _option_row(cb, label, desc, help_style=True)

        cols = max(1, int(columns))
        if cols <= 1:
            for item in options:
                vl.addWidget(_row(item))
        else:
            # Hug each column so leftover sheet width does not open a
            # half-and-half canyon between the lists.
            if section == "LOSSES":
                buckets = _split_loss_columns(options)
            else:
                n = (len(options) + cols - 1) // cols
                buckets = [options[i:i + n] for i in range(0, len(options), n)]
            pair = QHBoxLayout()
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(16)
            pair.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            for bucket in buckets:
                col = QWidget()
                col.setStyleSheet("background:transparent;")
                col.setFixedWidth(getattr(self, "_col_width", 300))
                col.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
                cv = QVBoxLayout(col)
                cv.setContentsMargins(0, 0, 0, 0)
                cv.setSpacing(12)
                for item in bucket:
                    row = _row(item)
                    row.setMaximumWidth(self._col_width)
                    cv.addWidget(row)
                cv.addStretch(1)
                pair.addWidget(col, 0, Qt.AlignTop)
            pair.addStretch(1)
            vl.addLayout(pair)
        vl.addStretch(1)
        return _help_section("required", [], caption, extra=body, title=section)

    def _build_toggle(self, toggle):
        label, checked, tip = toggle
        body, vl = self._col_body()
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.setSpacing(12)
        self._toggle_sw = _MiniSwitch(bool(checked))
        hl.addWidget(self._toggle_sw, 0, Qt.AlignVCenter)
        tl = QLabel(label)
        tl.setToolTip(tip)
        tl.setWordWrap(True)
        tl.setStyleSheet(self._copy())
        tl.setCursor(Qt.PointingHandCursor)
        tl.mousePressEvent = (
            lambda e, s=self._toggle_sw: s.set_checked(not s.is_checked()))
        hl.addWidget(tl, 1)
        vl.addWidget(row)
        vl.addStretch(1)
        return _help_section(
            "optional", [], HELP_OPTIONAL_CAPTION, extra=body,
            title="STANDARD LOSS")

    def _set_all(self, on):
        for cb in self._checks.values():
            cb.set_checked(on)

    def selected(self):
        return [k for k, _l, _d in self._options if self._checks[k].is_checked()]

    def toggle_value(self):
        return self._toggle_sw.is_checked() if self._toggle_sw is not None else False


_LAUNCHER_ITEMS = [
    ("standard", "Standard — train.py (DataParallel if several GPUs)"),
    ("ddp", "DDP — train_ddp.py (one process per selected GPU)"),
    ("accelerate", "Accelerate — older train_accelerate.py loop"),
]


class _RunOptionsDialog(PageHelpDialog):
    """GPUs / workers / seed — devices stacked over DataLoader; flags on the right."""

    def __init__(self, state, parent=None):
        self._gpu_checks = {}
        self._switches = {}
        gpus = self._build_gpus(state)
        workers = self._build_workers(state)
        optional = self._build_optional(state)
        left = QWidget()
        left.setStyleSheet("background:transparent;")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(16)
        lv.setAlignment(Qt.AlignTop)
        lv.addWidget(gpus)
        lv.addWidget(workers)
        super().__init__({
            "title": "TRAINING",
            "heading": "GPUS / WORKERS / SEED  ·  TRAINING",
            "intro": (
                "Devices and launcher on the left, workers and seed under them.\n"
                "Training flags on the right stay at their defaults until you change them."
            ),
            # Width only — left/right widgets replace these lists.
            "required": [
                "Standard — train.py (DataParallel if several GPUs)",
                "Accelerate falls back to Standard when LoRA, freeze layers, "
                "or a custom backend is on.",
            ],
            "optional": [
                "--save_weights_every_epoch: keep a checkpoint per epoch.",
                "Prefixes, space-separated (e.g. layer1 attn)",
            ],
            "required_caption": "The job uses these devices and this launcher on Start.",
            "optional_caption": HELP_OPTIONAL_CAPTION,
            "right_stretch": 2,
            "min_width": 1140,
            "max_width": 1160,
            "expand_height": True,
            "hide_scrollbar": True,
            "pad_bottom": 20,
            "content_bottom_gap": 32,
            "primary_text": "APPLY",
        }, parent, left=left, right=optional)
        self._refit()

    def _update_inner_height(self):
        """Resize scroll content only — keep the dialog shell fixed."""
        sc, inner = self._sc, self._inner
        inner_w = inner.width() if inner.width() > 0 else sc.width()
        inner.setMinimumSize(0, 0)
        inner.setMaximumSize(16777215, 16777215)
        inner.setFixedWidth(inner_w)
        lay = inner.layout()
        if lay is not None:
            lay.invalidate()
            lay.activate()
        inner.adjustSize()
        if inner.hasHeightForWidth():
            hfw = inner.heightForWidth(inner_w)
            content_h = hfw if hfw > 0 else inner.sizeHint().height()
        else:
            content_h = max(
                inner.sizeHint().height(), inner.minimumSizeHint().height())
        inner.setFixedSize(inner_w, content_h)

    def _copy(self, muted=False):
        t = theme_manager.theme
        color = t.text_muted if muted else t.text_sec
        return (
            f"font-family:'Montserrat';font-size:13px;color:{color};"
            "background:transparent;"
        )

    def _group(self, text, wrap=True, tracking=1.5, uppercase=True):
        t = theme_manager.theme
        lb = QLabel(text.upper() if uppercase else text)
        lb.setWordWrap(wrap)
        lb.setSizePolicy(
            QSizePolicy.Ignored if wrap else QSizePolicy.Preferred,
            QSizePolicy.Preferred)
        lb.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;font-weight:bold;"
            f"color:{t.text};letter-spacing:{tracking}px;background:transparent;"
        )
        return lb

    def _col_body(self, top=8):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, top, 0, 0)
        vl.setSpacing(10)
        return w, vl

    def _linedit(self, text, placeholder, password=False):
        t = theme_manager.theme
        edit = QLineEdit(text or "")
        edit.setPlaceholderText(placeholder)
        edit.setFixedHeight(34)
        if password:
            edit.setEchoMode(QLineEdit.Password)
        edit.setStyleSheet(
            f"QLineEdit{{font-family:'Montserrat';font-size:13px;color:{t.text};"
            f"background:{t.input_bg};border:1px solid {t.border_visible};"
            f"border-radius:6px;padding:0 8px;}}"
            f"QLineEdit::placeholder{{color:{t.text_muted};}}"
        )
        return edit

    def _switch_row(self, state, key, label, tip, default=False):
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.setSpacing(12)
        sw = _MiniSwitch(bool(state.get(key, default)))
        self._switches[key] = sw
        hl.addWidget(sw, 0, Qt.AlignVCenter)
        lb = QLabel(label)
        lb.setToolTip(tip)
        lb.setWordWrap(True)
        lb.setStyleSheet(self._copy())
        lb.setCursor(Qt.PointingHandCursor)
        lb.mousePressEvent = lambda e, s=sw: s.set_checked(not s.is_checked())
        hl.addWidget(lb, 1)
        return row

    def _build_gpus(self, state):
        t = theme_manager.theme
        body, vl = self._col_body()

        vl.addWidget(self._group("Devices"))
        gpus = [g for g in list_gpus() if not g.startswith("CPU")]
        if not gpus:
            hint = QLabel("No CUDA GPU detected — training runs on the CPU (very slow).")
            hint.setWordWrap(True)
            hint.setStyleSheet(
                f"font-family:'Montserrat';font-size:13px;color:{t.warning};"
                "background:transparent;"
            )
            vl.addWidget(hint)
        selected_ids = set(state.get("device_ids") or [0])
        for label in gpus:
            try:
                idx = int(label.split("GPU")[1].split(":")[0].strip())
            except Exception:
                continue
            cb = _CircleCheck()
            cb.set_checked(idx in selected_ids)
            self._gpu_checks[idx] = cb
            vl.addWidget(_option_row(
                cb, label,
                "DataParallel across every checked GPU" if len(gpus) > 1 else "",
                help_style=True))
        if gpus:
            cpu = _CircleCheck()
            cpu.set_checked(state.get("force_cpu", False))
            self._gpu_checks["cpu"] = cpu
            vl.addWidget(_option_row(
                cpu, "CPU only",
                "Hide the GPUs from the job (CUDA_VISIBLE_DEVICES=\"\").",
                help_style=True))

        vl.addWidget(self._group("Launcher"))
        self._launcher = _ComboBox()
        for key, label in _LAUNCHER_ITEMS:
            self._launcher.addItem(label, key)
        self._launcher.setFixedHeight(34)
        self._launcher.fit_popup_to_items = True
        self._launcher.setStyleSheet(
            f"QComboBox{{font-family:'Montserrat';font-size:13px;color:{t.text};"
            f"background:{t.input_bg};border:1px solid {t.border_visible};"
            f"border-radius:6px;padding:0 8px;}}"
            "QComboBox::drop-down{border:none;width:22px;}"
            f"QComboBox QAbstractItemView{{background:{t.surface_alt};"
            f"border:1px solid {t.border_dim};color:{t.text};"
            f"selection-background-color:{theme_manager.accent};"
            f"selection-color:{theme_manager._accent_text};outline:none;}}"
            "QComboBox QAbstractItemView::item{padding:6px 12px;min-height:26px;}"
        )
        self._launcher.setToolTip(
            "standard: train.py (current). ddp: one process per GPU via "
            "CUDA_VISIBLE_DEVICES. accelerate: older train_accelerate.py "
            "(no LoRA / freeze / custom backends).")
        idx = self._launcher.findData(state.get("launcher") or "standard")
        if idx >= 0:
            self._launcher.setCurrentIndex(idx)
        vl.addWidget(self._launcher)
        acc_note = QLabel(
            "Accelerate falls back to Standard when LoRA, freeze layers, or a "
            "custom backend is on, or when the accelerate package is missing.")
        acc_note.setWordWrap(True)
        acc_note.setStyleSheet(self._copy(muted=True))
        vl.addWidget(acc_note)
        return _help_section(
            "required", [],
            "The job uses these devices and this launcher on Start.",
            extra=body, title="GPUS")

    def _build_workers(self, state):
        body, vl = self._col_body()
        vl.addWidget(self._group("Data loading"))
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        def spin(lo, hi, val):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(int(val))
            s.setFixedSize(110, 34)
            s.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            s.setStyleSheet(_spin_ss())
            return s

        def grid_label(text, tip):
            lb = QLabel(text)
            lb.setStyleSheet(self._copy())
            lb.setToolTip(tip)
            return lb

        self._workers = spin(0, 64, state.get("num_workers", 4))
        self._seed = spin(0, 999_999, state.get("seed", 0))
        grid.addWidget(grid_label("Workers", "DataLoader worker processes (--num_workers)."), 0, 0)
        grid.addWidget(self._workers, 0, 1)
        grid.addWidget(grid_label("Seed", "Random seed (--seed)."), 1, 0)
        grid.addWidget(self._seed, 1, 1)
        grid.setColumnStretch(2, 1)
        vl.addLayout(grid)
        vl.addWidget(self._switch_row(
            state, "pin_memory", "Pin memory",
            "--pin_memory: faster host→GPU copies."))
        vl.addWidget(self._switch_row(
            state, "persistent_workers", "Persistent workers",
            "--persistent_workers: keep DataLoader workers alive between epochs."))
        return _help_section(
            "required", [],
            "Workers, seed, and host→GPU copies for the DataLoader.",
            extra=body, title="WORKERS / SEED")

    def _two_col(self, left_w, right_w):
        t = theme_manager.theme
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(_HELP_COL_GUTTER)
        hl.addWidget(left_w, 5, Qt.AlignTop)
        vdiv = QFrame()
        vdiv.setFixedWidth(1)
        vdiv.setStyleSheet(f"background:{t.border_dim};border:none;")
        hl.addWidget(vdiv)
        hl.addWidget(right_w, 4, Qt.AlignTop)
        return row

    def _build_optional(self, state):
        body, vl = self._col_body()
        vl.setSpacing(16)

        run, rvl = self._col_body(top=0)
        rvl.addWidget(self._group("Run options"))
        rvl.addWidget(self._switch_row(
            state, "pre_valid", "Validate before training",
            "--pre_valid: run validation once before the first epoch."))
        rvl.addWidget(self._switch_row(
            state, "save_every_epoch", "Save weights every epoch",
            "--save_weights_every_epoch: keep a checkpoint per epoch (with all metrics in the name)."))
        rvl.addWidget(self._switch_row(
            state, "each_metrics_in_name", "Per-stem metrics in checkpoint names",
            "--each_metrics_in_name."))
        rvl.addWidget(self._switch_row(
            state, "safe_mode", "Safe mode",
            "--safe_mode: ignore forward errors and keep training."))

        resume, svl = self._col_body(top=0)
        svl.addWidget(self._group(
            "When resuming from a checkpoint", wrap=True, tracking=0.5))
        svl.addWidget(self._switch_row(
            state, "load_optimizer", "Load optimizer state", "--load_optimizer"))
        svl.addWidget(self._switch_row(
            state, "load_scheduler", "Load scheduler state", "--load_scheduler"))
        svl.addWidget(self._switch_row(
            state, "load_epoch", "Continue epoch numbering", "--load_epoch"))
        svl.addWidget(self._switch_row(
            state, "load_best_metric", "Load best metric so far", "--load_best_metric"))
        svl.addWidget(self._switch_row(
            state, "load_all_metrics", "Load all metrics", "--load_all_metrics"))
        svl.addWidget(self._switch_row(
            state, "load_all_losses", "Load all losses", "--load_all_losses"))

        lora, lvl = self._col_body(top=0)
        lvl.addWidget(self._group("LoRA"))
        mode = (state.get("lora_mode") or "off")
        lora_on = mode not in ("off", "", None)
        lora_head = QWidget()
        lora_head.setStyleSheet("background:transparent;")
        lora_hl = QHBoxLayout(lora_head)
        lora_hl.setContentsMargins(0, 4, 0, 4)
        lora_hl.setSpacing(12)
        lora_lb = QLabel("LoRA")
        lora_lb.setToolTip(
            "PEFT or loralib fine-tuning. A lora: block is written into the "
            "run config when missing.")
        lora_lb.setStyleSheet(self._copy())
        lora_hl.addWidget(lora_lb, 1)
        self._lora_kind_host = QWidget()
        self._lora_kind_host.setStyleSheet("background:transparent;")
        kind_hl = QHBoxLayout(self._lora_kind_host)
        kind_hl.setContentsMargins(0, 0, 0, 0)
        kind_hl.setSpacing(8)
        self._lora_kind_wrap = _pill_switch("PEFT", "loralib", mode == "loralib")
        self._lora_kind_sw = self._lora_kind_wrap._sw
        self._lora_kind_wrap.setToolTip(
            "PEFT (--train_lora_peft) or loralib (--train_lora_loralib).")
        kind_hl.addWidget(self._lora_kind_wrap, 0, Qt.AlignVCenter)
        self._lora_on_wrap = _pill_switch("Off", "On", lora_on)
        self._lora_on_sw = self._lora_on_wrap._sw
        lora_hl.addWidget(self._lora_kind_host, 0, Qt.AlignVCenter)
        lora_hl.addWidget(self._lora_on_wrap, 0, Qt.AlignVCenter)
        lvl.addWidget(lora_head)

        self._lora_ckpt_host = QWidget()
        self._lora_ckpt_host.setStyleSheet("background:transparent;")
        lora_row = QHBoxLayout(self._lora_ckpt_host)
        lora_row.setContentsMargins(0, 0, 0, 0)
        lora_row.setSpacing(6)
        self._lora_ckpt = self._linedit(
            state.get("lora_checkpoint") or "", "Optional LoRA adapter checkpoint…")
        lora_row.addWidget(self._lora_ckpt, 1)
        lora_btn = EllipsisButton()
        lora_btn.clicked.connect(self._browse_lora)
        lora_row.addWidget(lora_btn)
        lvl.addWidget(self._lora_ckpt_host)

        def _sync_lora(_on=None):
            on = self._lora_on_sw.is_checked()
            self._lora_kind_host.setVisible(on)
            self._lora_ckpt_host.setVisible(on)
            if getattr(self, "_inner", None) is not None:
                self._update_inner_height()
        self._lora_on_sw.toggled.connect(_sync_lora)
        _sync_lora()

        class _LoRAModeView:
            def currentData(inner_self):
                if not self._lora_on_sw.is_checked():
                    return "off"
                return "loralib" if self._lora_kind_sw.is_checked() else "peft"
        self._lora_mode = _LoRAModeView()

        freeze, fvl = self._col_body(top=0)
        fvl.addWidget(self._group("Freeze layers"))
        self._freeze = self._linedit(
            state.get("freeze_layers") or "",
            "Prefixes, space-separated (e.g. layer1 attn)")
        self._freeze.setToolTip(
            "--freeze_layers: freeze parameters whose names start with these prefixes.\n"
            "Stem-shift preset freezes the RoFormer backbone and trains only "
            "the mask head (mask_estimators).")
        fvl.addWidget(self._freeze)
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(8)
        stem_btn = _link_button(
            "Stem-shift preset (RoFormer)",
            lambda: self._freeze.setText(catalog.STEM_SHIFT_FREEZE_LAYERS))
        stem_btn.setToolTip(
            "Sets freeze layers to “layers band_split” — freezes the transformer "
            "backbone and band split, leaving mask_estimators trainable. "
            "For switching target stem on a RoFormer checkpoint.")
        preset_row.addWidget(stem_btn, 0, Qt.AlignLeft)
        preset_row.addStretch(1)
        fvl.addLayout(preset_row)

        wandb, wvl = self._col_body(top=0)
        wvl.addWidget(self._group("WEIGHTS & BIASES (wandb)", uppercase=False))
        self._wandb_key = self._linedit(
            state.get("wandb_key") or "", "API key (empty = wandb disabled)",
            password=True)
        self._wandb_key.setToolTip(
            "Injected into the job as WANDB_API_KEY so it does not appear in "
            "the spawned command or train_log.txt. If a key was ever printed "
            "in a log, revoke it at https://wandb.ai/settings and paste a new one.")
        wvl.addWidget(self._wandb_key)
        wvl.addWidget(self._switch_row(
            state, "wandb_offline", "Offline wandb",
            "--wandb_offline: log locally without uploading."))

        flags = QWidget()
        flags.setStyleSheet("background:transparent;")
        flags.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        fvl_all = QVBoxLayout(flags)
        fvl_all.setContentsMargins(0, 0, 0, 0)
        fvl_all.setSpacing(16)
        fvl_all.setAlignment(Qt.AlignTop)
        fvl_all.addWidget(run)
        fvl_all.addWidget(lora)
        fvl_all.addWidget(freeze)
        fvl_all.addWidget(wandb)

        resume.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        vl.addWidget(self._two_col(flags, resume))
        return _help_section("optional", [], HELP_OPTIONAL_CAPTION, extra=body)

    def _browse_lora(self):
        start = self._lora_ckpt.text() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "LoRA adapter checkpoint", start,
            "Checkpoints (*.ckpt *.pth *.pt *.bin *.safetensors);;All files (*.*)")
        if path:
            self._lora_ckpt.setText(path)

    def values(self):
        ids = sorted(i for i, cb in self._gpu_checks.items() if i != "cpu" and cb.is_checked())
        cpu = self._gpu_checks.get("cpu")
        launcher = self._launcher.currentData()
        lora_mode = self._lora_mode.currentData()
        out = {
            "device_ids": ids or [0],
            "force_cpu": bool(cpu.is_checked()) if cpu is not None else False,
            "num_workers": self._workers.value(),
            "seed": self._seed.value(),
            "launcher": launcher if launcher in ("standard", "ddp", "accelerate") else "standard",
            "lora_mode": lora_mode if lora_mode in ("off", "peft", "loralib") else "off",
            "lora_checkpoint": self._lora_ckpt.text().strip(),
            "freeze_layers": self._freeze.text().strip(),
            "wandb_key": self._wandb_key.text().strip(),
        }
        for k, sw in self._switches.items():
            out[k] = sw.is_checked()
        return out


class _FitProbeBridge(QObject):
    done = Signal(object)
    progress = Signal(object)


_PROBE_LOAD_HINT = 25.0
_PROBE_TRIAL_HINT = 15.0
_PROBE_ETA_MIN_REMAIN = 1.0


def _fmt_probe_eta(seconds):
    """Compact remaining time for the Fit GPU probe bar."""
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _probe_eta_remain(elapsed, step=0, total=0, phase="", load_elapsed=0.0,
                      since_step=0.0):
    """Seconds remaining for a Fit GPU probe.

    ``step`` is completed work (0=load, 1=loaded, then one per finished
    trial). ``since_step`` is time in the current in-flight step. Rate uses
    *finished* trials only so the label counts down during a trial instead
    of climbing as in-flight time inflates the average.
    """
    elapsed = max(0.0, float(elapsed or 0))
    since_step = max(0.0, float(since_step or 0))
    step = max(0, int(step or 0))
    total = max(0, int(total or 0))
    remaining = max(0, total - step) if total else 0
    if remaining <= 0 and step > 0 and total > 0:
        return 0.0

    n_trials = max(1, total - 1) if total else 4
    phase = (phase or "").strip().lower()

    if phase in ("", "load") or step <= 0:
        expected_load = max(_PROBE_LOAD_HINT, elapsed / 0.85)
        load_remain = max(0.0, expected_load - elapsed)
        return load_remain + n_trials * _PROBE_TRIAL_HINT

    load_elapsed = max(0.0, min(float(load_elapsed or 0), elapsed))
    trials_left = remaining if total else n_trials
    guessed = max(_PROBE_TRIAL_HINT, load_elapsed * 0.5)
    if phase == "loaded" or step <= 1:
        avg = guessed
    else:
        trials_done = max(1, step - 1)
        finished = max(0.0, elapsed - load_elapsed - since_step)
        avg = finished / float(trials_done) if finished > 0 else guessed
        if avg <= 0:
            avg = guessed
    remain = avg * max(0, trials_left - 1) + max(0.0, avg - since_step)
    if trials_left > 0:
        remain = max(remain, _PROBE_ETA_MIN_REMAIN)
    return remain


_FIT_HEURISTIC_LINE = (
    "These are best-practice heuristics, not a live VRAM probe."
)
_FIT_OPT_CAPTION = (
    "Free VRAM is live. Peak and weights fill after a train step."
)


def _fit_probe_will_run(probe_spec):
    spec = probe_spec or {}
    return bool(spec.get("config_path") and not spec.get("force_cpu"))


def _sanitize_fit_notes(notes):
    """Drop the lookup-table disclaimer so the sheet does not contradict itself."""
    out = []
    for n in notes or ():
        text = (n or "").replace(_FIT_HEURISTIC_LINE, "").strip(" \n")
        if text:
            out.append(text)
    return out


def _fit_gpu_help_data(advice, current, probe_spec=None):
    """Heading / intro / width copy. Knobs are widgets, not these lists."""
    spec = probe_spec or {}
    if advice.risk == "cpu" or spec.get("force_cpu"):
        status = "No CUDA GPU is selected — training on CPU is not practical."
    elif _fit_probe_will_run(spec):
        status = "Allocating a train step on this GPU. Knobs update when it finishes."
    else:
        status = (
            "Free VRAM is live. Select a config YAML to measure a train step; "
            "knobs start from the usual fit."
        )
    model = current.get("model_type") or "—"
    intro = f"{advice.gpu_label}. {advice.family_label} ({model}). {status}"
    return {
        "title": "TRAINING",
        "heading": "FIT GPU  ·  TRAINING",
        "intro": intro,
        "required": [
            "Batch size, grad accum, chunk size, checkpointing, pin memory.",
            "Live peak GB from a dummy train step on this GPU.",
        ],
        "optional": [
            "Free VRAM, peak allocated, weights, and why the knobs moved.",
            "Learning rate and epochs stay as they are.",
        ],
        "required_caption": "Based on this GPU. Apply writes to Training settings.",
        "optional_caption": _FIT_OPT_CAPTION,
        "required_title": "RECOMMENDED",
        "optional_title": "LIVE",
        "primary_text": "APPLY",
    }


class _FitGpuDialog(PageHelpDialog):
    """Help-sheet Fit GPU: blue measured fields + a live CUDA train-step probe."""

    def __init__(self, advice, current, parent=None, *, probe_spec=None):
        self.advice = advice
        self._current = current or {}
        self._probe_spec = probe_spec
        self._notes = _sanitize_fit_notes(advice.notes)
        self._bridge = None
        self._apply_btn = None
        self._probe_busy = False
        self._verifying = False
        left = self._build_required(advice)
        right = self._build_optional(advice)
        below = self._build_notes()
        super().__init__(
            _fit_gpu_help_data(advice, current, probe_spec), parent,
            left=left, right=right, below=below)
        self._apply_btn = self.findChild(QPushButton, "helpPrimaryBtn")
        self._fill_snapshot()
        self._probe_t0 = 0.0
        self._probe_step = 0
        self._probe_total = 0
        self._probe_phase = ""
        self._probe_loaded_t = 0.0
        self._probe_step_t = 0.0
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._tick_probe_eta)
        if _fit_probe_will_run(probe_spec):
            self._set_probing(True, "Allocating a train step on this GPU…")
            QTimer.singleShot(0, self._kick_probe)
        else:
            self._set_notes(self._notes)

    def _copy(self, muted=False):
        t = theme_manager.theme
        color = t.text_muted if muted else t.text_sec
        return (
            f"font-family:'Montserrat';font-size:13px;color:{color};"
            "background:transparent;"
        )

    def _group(self, text):
        t = theme_manager.theme
        lb = QLabel(text.upper())
        lb.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;font-weight:bold;"
            f"color:{t.text};letter-spacing:1.5px;background:transparent;"
        )
        return lb

    def _col_body(self):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(12)
        return w, vl

    def _blue_spin(self, lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(val))
        s.setFixedSize(110, 34)
        s.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s.setStyleSheet(_spin_ss())
        return s

    def _blue_secs(self, val):
        s = QDoubleSpinBox()
        s.setRange(0.5, 60.0)
        s.setDecimals(1)
        s.setSingleStep(0.5)
        s.setSuffix(" s")
        s.setValue(float(val))
        s.setFixedSize(110, 34)
        s.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s.setStyleSheet(_spin_ss())
        return s

    def _blue_field(self, text="—"):
        t = theme_manager.theme
        edit = QLineEdit(text)
        edit.setReadOnly(True)
        edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        edit.setFixedSize(110, 34)
        edit.setStyleSheet(
            f"QLineEdit{{font-family:'Courier New',monospace;font-size:13px;"
            f"font-weight:bold;color:{theme_manager.accent};"
            f"background:{t.input_bg};border:1px solid {t.border_dim};"
            f"border-radius:4px;padding:2px 8px;}}"
        )
        return edit

    def _labeled(self, vl, text, widget, tip=""):
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        lb = QLabel(text)
        lb.setStyleSheet(self._copy())
        hl.addWidget(lb, 0, Qt.AlignVCenter)
        dot = None
        if tip:
            dot = _InfoDot(tip)
            hl.addWidget(dot, 0, Qt.AlignVCenter)
        hl.addStretch(1)
        hl.addWidget(widget, 0, Qt.AlignVCenter)
        vl.addWidget(row)
        return dot

    def _pill_row(self, vl, text, wrap, tip=""):
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.setSpacing(8)
        lb = QLabel(text)
        lb.setStyleSheet(self._copy())
        hl.addWidget(lb, 0, Qt.AlignVCenter)
        if tip:
            hl.addWidget(_InfoDot(tip), 0, Qt.AlignVCenter)
        hl.addStretch(1)
        hl.addWidget(wrap, 0, Qt.AlignVCenter)
        vl.addWidget(row)
        return row

    def _build_required(self, advice):
        body, vl = self._col_body()
        vl.addWidget(self._group("Knobs"))
        self._batch = self._blue_spin(1, 64, advice.batch_size)
        self._accum = self._blue_spin(1, 64, advice.grad_accum)
        self._eff = self._blue_field(str(advice.effective_batch))
        self._batch.valueChanged.connect(self._sync_eff)
        self._accum.valueChanged.connect(self._sync_eff)
        self._labeled(vl, "Batch size", self._batch,
                      "training.batch_size per GPU — measured, then editable.")
        self._labeled(vl, "Grad accum steps", self._accum,
                      "training.gradient_accumulation_steps.")
        self._labeled(vl, "Effective batch", self._eff,
                      "batch × accum. Kept close to the value already on the tab.")

        chunk = advice.chunk_size if advice.chunk_size else 131072
        sr = int(getattr(advice, "sample_rate", None) or 44100) or 44100
        self._chunk_samples = int(chunk)
        self._chunk = self._blue_secs(int(chunk) / float(sr))
        self._chunk_dot = self._labeled(
            vl, "Chunk size", self._chunk, self._chunk_tip(chunk, advice))
        self._chunk.valueChanged.connect(self._on_chunk_secs)

        ckpt_on = bool(advice.use_torch_checkpoint)
        self._ckpt_wrap = _pill_switch("Off", "On", ckpt_on)
        self._ckpt_sw = self._ckpt_wrap._sw
        self._ckpt_host = self._pill_row(
            vl, "Activation checkpointing", self._ckpt_wrap,
            "model.use_torch_checkpoint — trades a slower step for less VRAM.")
        self._ckpt_host.setVisible(advice.use_torch_checkpoint is not None)

        self._pin_wrap = _pill_switch("Off", "On", bool(advice.pin_memory))
        self._pin = self._pin_wrap._sw
        self._pill_row(
            vl, "Pin memory", self._pin_wrap,
            "--pin_memory: faster host→GPU copies.")

        self._opt_field = self._blue_field(advice.optimizer or "—")
        self._opt_host = QWidget()
        self._opt_host.setStyleSheet("background:transparent;")
        oh = QVBoxLayout(self._opt_host)
        oh.setContentsMargins(0, 0, 0, 0)
        oh.setSpacing(0)
        self._labeled(
            oh, "Optimizer", self._opt_field,
            "training.optimizer — suggested when the card is tight.")
        vl.addWidget(self._opt_host)
        self._opt_host.setVisible(bool(advice.optimizer))
        vl.addStretch(1)
        return _help_section(
            "required", [],
            "Based on this GPU. Apply writes to Training settings.",
            extra=body, title="RECOMMENDED")

    def _build_optional(self, advice):
        body, vl = self._col_body()
        self._free = self._blue_field("—")
        self._peak = self._blue_field("—")
        self._weights = self._blue_field("—")
        self._total = self._blue_field(
            "—" if advice.gpu_mem_gb is None else f"{advice.gpu_mem_gb:.0f} GB")
        self._labeled(vl, "Free VRAM", self._free, "nvidia-smi memory.free right now.")
        self._labeled(vl, "Peak this step", self._peak,
                      "torch.cuda.max_memory_allocated during the dummy train step.")
        self._labeled(vl, "Weights", self._weights, "Model on GPU before the train step.")
        self._labeled(vl, "Card total", self._total,
                      "nvidia-smi memory.total for this GPU.")
        vl.addStretch(1)
        return _help_section("optional", [], _FIT_OPT_CAPTION, extra=body, title="LIVE")

    def _build_notes(self):
        body = QWidget()
        body.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(body)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(12)
        vl.addWidget(self._group("Notes"))
        self._status = QLabel("Waiting to probe this GPU…")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(self._copy(muted=True))
        vl.addWidget(self._status)
        self._probe_row = QWidget()
        self._probe_row.setStyleSheet("background:transparent;")
        ph = QHBoxLayout(self._probe_row)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(10)
        self._probe_bar = SmoothBar()
        self._probe_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._probe_eta = QLabel("ETA —")
        self._probe_eta.setObjectName("fitProbeEta")
        self._probe_eta.setStyleSheet(self._copy(muted=True))
        ph.addWidget(self._probe_bar, 1, Qt.AlignVCenter)
        ph.addWidget(self._probe_eta, 0, Qt.AlignVCenter)
        self._probe_row.setVisible(False)
        vl.addWidget(self._probe_row)
        self._notes_lbl = QLabel("")
        self._notes_lbl.setWordWrap(True)
        self._notes_lbl.setStyleSheet(self._copy())
        vl.addWidget(self._notes_lbl)
        return body

    def _set_notes(self, notes):
        self._notes = _sanitize_fit_notes(notes)
        text = "\n".join(self._notes) if self._notes else ""
        self._notes_lbl.setText(text)
        self._notes_lbl.setVisible(bool(text))

    def _set_probing(self, on, status=""):
        if getattr(self, "_apply_btn", None) is not None:
            self._apply_btn.setEnabled(not on)
        if on:
            self._probe_row.setVisible(True)
            self._probe_bar.setVisible(True)
            self._probe_bar.set_indeterminate(False)
            self._probe_bar.setValueImmediate(0)
            self._probe_eta.setVisible(True)
            self._probe_eta.setText("ETA —")
            self._probe_t0 = time.monotonic()
            self._probe_step = 0
            self._probe_total = 0
            self._probe_phase = ""
            self._probe_loaded_t = 0.0
            self._probe_step_t = self._probe_t0
            if getattr(self, "_eta_timer", None) is not None:
                self._eta_timer.start()
            self._refresh_probe_eta()
            self._notes_lbl.clear()
            self._notes_lbl.setVisible(False)
            if status:
                self._status.setText(status)
        else:
            if getattr(self, "_eta_timer", None) is not None:
                self._eta_timer.stop()
            self._probe_bar.set_indeterminate(False)
            self._probe_bar.setVisible(False)
            self._probe_eta.setVisible(False)
            self._probe_row.setVisible(False)

    def _tick_probe_eta(self):
        self._refresh_probe_eta()

    def _on_probe_progress(self, data):
        if not isinstance(data, dict):
            return
        try:
            self._probe_step = int(data.get("step") or 0)
            self._probe_total = int(data.get("total") or 0)
        except (TypeError, ValueError):
            return
        self._probe_phase = str(data.get("phase") or "")
        self._probe_step_t = time.monotonic()
        if self._probe_phase == "loaded" and not self._probe_loaded_t:
            self._probe_loaded_t = self._probe_step_t
        self._refresh_probe_eta()

    def _refresh_probe_eta(self):
        if not getattr(self, "_probe_eta", None):
            return
        now = time.monotonic()
        t0 = self._probe_t0 or now
        elapsed = max(0.0, now - t0)
        load_elapsed = 0.0
        if self._probe_loaded_t:
            load_elapsed = max(0.0, self._probe_loaded_t - t0)
        step_t = self._probe_step_t or t0
        since_step = max(0.0, now - step_t)
        remain = _probe_eta_remain(
            elapsed, self._probe_step, self._probe_total,
            phase=getattr(self, "_probe_phase", "") or "",
            load_elapsed=load_elapsed, since_step=since_step)
        self._probe_eta.setText(f"ETA {_fmt_probe_eta(remain)}")
        est = elapsed + remain
        pct = 0.0 if est <= 0 else min(99.0, 100.0 * elapsed / est)
        if getattr(self, "_probe_bar", None) is not None:
            self._probe_bar.setValue(pct)

    def _set_intro_status(self, status):
        model = (self._current or {}).get("model_type") or "—"
        self._intro.setText(
            f"{self.advice.gpu_label}. {self.advice.family_label} ({model}). {status}"
        )

    def _chunk_sr(self, advice=None):
        advice = advice or self.advice
        try:
            return int(getattr(advice, "sample_rate", None) or 44100) or 44100
        except (TypeError, ValueError):
            return 44100

    def _secs_to_samples(self, secs):
        return max(1024, int(round(float(secs) * self._chunk_sr())))

    def _set_chunk_samples(self, samples):
        samples = int(samples)
        self._chunk_samples = samples
        self._chunk.blockSignals(True)
        self._chunk.setValue(samples / float(self._chunk_sr()))
        self._chunk.blockSignals(False)
        if getattr(self, "_chunk_dot", None) is not None:
            self._chunk_dot.setToolTip(self._chunk_tip(samples))

    def _on_chunk_secs(self, secs):
        self._chunk_samples = self._secs_to_samples(secs)
        if getattr(self, "_chunk_dot", None) is not None:
            self._chunk_dot.setToolTip(self._chunk_tip(self._chunk_samples))

    def _chunk_tip(self, samples, advice=None):
        sr = self._chunk_sr(advice)
        try:
            n = int(samples)
        except (TypeError, ValueError):
            n = 0
        return (
            f"Entered in seconds. Apply writes audio.chunk_size as {n} samples "
            f"at {sr} Hz."
        )

    def _sync_eff(self, *_):
        self._eff.setText(str(int(self._batch.value()) * int(self._accum.value())))

    def _gb(self, val):
        if val is None:
            return "—"
        try:
            return f"{float(val):.1f} GB"
        except (TypeError, ValueError):
            return "—"

    def _fill_snapshot(self):
        spec = self._probe_spec or {}
        if spec.get("force_cpu"):
            self._status.setText("CPU is selected — nothing to allocate.")
            return
        device_id = int(spec.get("device_id") or 0)
        snap = selected_gpu_snapshot([device_id], query_gpu_memory(refresh=True))
        if snap:
            self._free.setText(self._gb(snap.get("free_gb")))
            self._total.setText(f"{float(snap.get('total_gb') or 0):.0f} GB")
        if spec.get("config_path"):
            self._status.setText("Allocating a train step on this GPU…")
        else:
            self._status.setText(
                "Pick a config YAML to measure a real train step. "
                "Knobs on the left are the usual fit until then.")

    def _knob_fingerprint(self, advice=None):
        a = advice if advice is not None else self.values()
        ckpt = a.use_torch_checkpoint
        return (
            int(a.batch_size),
            int(a.chunk_size or 0),
            None if ckpt is None else bool(ckpt),
        )

    def _measured_fp(self):
        a = self.advice
        if a.source != "probe" or a.peak_gb is None:
            return None
        return self._knob_fingerprint(a)

    def _needs_verify(self):
        if not _fit_probe_will_run(self._probe_spec):
            return False
        measured = self._measured_fp()
        if measured is None:
            return True
        return self._knob_fingerprint() != measured

    def _verify_spec(self):
        spec = dict(self._probe_spec or {})
        vals = self.values()
        spec["chunk_size"] = int(vals.chunk_size or 0)
        spec["verify_batch"] = int(vals.batch_size)
        spec["current_batch"] = int(vals.batch_size)
        spec["current_accum"] = int(vals.grad_accum)
        spec["checkpoint"] = (
            -1 if vals.use_torch_checkpoint is None
            else (1 if vals.use_torch_checkpoint else 0)
        )
        return spec

    def accept(self):
        if self._probe_busy:
            return
        if not self._needs_verify():
            super().accept()
            return
        self._kick_probe(self._verify_spec(), verifying=True)

    def _kick_probe(self, spec=None, *, verifying=False):
        spec = spec if spec is not None else self._probe_spec
        if self._probe_busy:
            return
        if not spec or spec.get("force_cpu") or not spec.get("config_path"):
            self._set_probing(False)
            self._set_notes(self._notes)
            if verifying:
                super().accept()
            return
        self._verifying = verifying
        self._probe_busy = True
        status = (
            "Checking these knobs with a train step…"
            if verifying else "Allocating a train step on this GPU…"
        )
        self._set_probing(True, status)
        if verifying:
            self._set_intro_status(
                "Checking these knobs with a train step. Apply waits until it fits.")
        self._bridge = _FitProbeBridge(self)
        self._bridge.done.connect(self._on_probe)
        self._bridge.progress.connect(self._on_probe_progress)

        def work():
            from backend.vram_probe import run_probe_process
            payload = dict(spec)
            payload["on_progress"] = self._bridge.progress.emit
            self._bridge.done.emit(run_probe_process(payload))

        threading.Thread(target=work, daemon=True).start()

    def _on_probe(self, result):
        verifying = self._verifying
        self._verifying = False
        self._probe_busy = False
        self.advice = advise_from_probe(result or {}, fallback=self.advice)
        self._batch.setValue(self.advice.batch_size)
        self._accum.setValue(self.advice.grad_accum)
        self._sync_eff()
        if self.advice.use_torch_checkpoint is not None:
            self._ckpt_host.setVisible(True)
            self._ckpt_sw.set_checked(bool(self.advice.use_torch_checkpoint))
        self._pin.set_checked(bool(self.advice.pin_memory))
        if self.advice.optimizer:
            self._opt_host.setVisible(True)
            self._opt_field.setText(self.advice.optimizer)
        if self.advice.chunk_size:
            self._set_chunk_samples(int(self.advice.chunk_size))
        self._peak.setText(self._gb(self.advice.peak_gb))
        self._weights.setText(self._gb(self.advice.weights_gb))
        if self.advice.free_gb is not None:
            self._free.setText(self._gb(self.advice.free_gb))
        if self.advice.gpu_mem_gb is not None:
            self._total.setText(f"{float(self.advice.gpu_mem_gb):.0f} GB")
        self._set_probing(False)
        self._set_notes(self.advice.notes)
        fitted = self.advice.source == "probe" and self.advice.peak_gb is not None
        if verifying:
            if fitted:
                self._status.setText(
                    f"Measured {self._gb(self.advice.peak_gb)} peak on this GPU.")
                super().accept()
                return
            if not (result or {}).get("ok"):
                self._status.setText("Could not re-check these knobs on this GPU.")
                self._set_intro_status(
                    "Could not re-check these knobs. Sheet stayed open — "
                    "nothing was written.")
            else:
                self._status.setText(
                    "Those knobs still CUDA OOM'd. Nothing was written.")
                self._set_intro_status(
                    "Those knobs still CUDA OOM'd. Chunk/batch on the left "
                    "were updated — APPLY again to re-check.")
            if getattr(self, "_inner", None) is not None:
                self._refit()
            return
        if fitted:
            peak = self._gb(self.advice.peak_gb)
            self._status.setText(f"Measured {peak} peak on this GPU.")
            self._set_intro_status(
                f"Measured {peak} peak. Apply re-checks if the knobs moved, "
                "then writes them.")
        elif self.advice.source == "probe":
            self._status.setText("Batch 1 CUDA OOM'd. Nothing was written.")
            self._set_intro_status(
                "Batch 1 CUDA OOM'd. Chunk on the left was halved — "
                "APPLY re-checks that window before writing.")
        else:
            self._status.setText("Could not run a train-step on this GPU.")
            self._set_intro_status(
                "Could not run a train-step on this GPU. Knobs on the left "
                "are the usual fit.")
        if getattr(self, "_inner", None) is not None:
            self._refit()

    def values(self):
        ckpt = self.advice.use_torch_checkpoint
        if ckpt is not None:
            ckpt = self._ckpt_sw.is_checked()
        return _dc_replace(
            self.advice,
            batch_size=int(self._batch.value()),
            grad_accum=int(self._accum.value()),
            use_torch_checkpoint=ckpt,
            pin_memory=self._pin.is_checked(),
            chunk_size=int(self._chunk_samples),
        )

    def closeEvent(self, e):
        from backend.vram_probe import abort_probe
        abort_probe()
        self._verifying = False
        self._probe_busy = False
        if hasattr(self, "_probe_bar"):
            self._set_probing(False)
        super().closeEvent(e)


def _starter_yamls(model_type=""):
    """Premade YAMLs under configs/, preferring ones that match `model_type`."""
    root = os.path.join(REPO_ROOT, "configs")
    all_paths = []
    if os.path.isdir(root):
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in files:
                if fn.lower().endswith((".yaml", ".yml")):
                    all_paths.append(os.path.join(dirpath, fn))
    all_paths.sort(key=lambda p: os.path.basename(p).lower())
    if not model_type:
        return all_paths
    matched = [
        p for p in all_paths
        if catalog.guess_model_type_from_name(os.path.basename(p)) == model_type
    ]
    return matched or all_paths


class _SetupChoice(QFrame):
    """One exclusive answer on a Setup step — title + why, accent when picked."""
    picked = Signal(str)

    def __init__(self, key, title, body, parent=None):
        super().__init__(parent)
        self._key = key
        self._on = False
        self._hover = False
        self._pressed = False
        self.setObjectName("setupChoice")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        vl = QVBoxLayout(self)
        vl.setContentsMargins(16, 14, 16, 14)
        vl.setSpacing(6)
        self._title = QLabel(title.upper())
        self._title.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._body = QLabel(body)
        self._body.setWordWrap(True)
        self._body.setAttribute(Qt.WA_TransparentForMouseEvents)
        vl.addWidget(self._title)
        vl.addWidget(self._body)
        self._paint()

    def key(self):
        return self._key

    def is_on(self):
        return self._on

    def set_on(self, on):
        self._on = bool(on)
        self._paint()

    def _paint(self):
        t = theme_manager.theme
        if self._pressed:
            bg, border = t.surface_alt, theme_manager.accent
        elif self._on and self._hover:
            bg, border = theme_manager._accent_soft, theme_manager._accent_glow
        elif self._on or self._hover:
            bg, border = theme_manager._accent_soft, theme_manager.accent
        else:
            bg, border = t.surface, t.border_visible
        self.setStyleSheet(
            f"QFrame#setupChoice{{background:{bg};"
            "border-width:1px;border-style:solid;"
            f"border-color:{border};"
            f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px;}}"
        )
        title_c = theme_manager.accent if self._on else t.text
        body_c = t.text if (self._on or self._hover) else t.text_sec
        self._title.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;"
            f"color:{title_c};letter-spacing:1.5px;background:transparent;"
        )
        self._body.setStyleSheet(
            "font-family:'Montserrat';font-size:13px;"
            f"color:{body_c};background:transparent;"
        )

    def enterEvent(self, e):
        self._hover = True
        self._paint()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self._pressed = False
        self._paint()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._pressed = True
            self._paint()
            self.picked.emit(self._key)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self._pressed = False
        self._paint()
        super().mouseReleaseEvent(e)


class _SetupStepTicks(QWidget):
    """Six pills beside the step badge — current / past / future. Not clickable."""

    def __init__(self, count, parent=None):
        super().__init__(parent)
        self.setObjectName("setupStepTicks")
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(10, 0, 0, 0)
        hl.setSpacing(4)
        self._pills = []
        for _ in range(count):
            pill = QFrame()
            pill.setFixedSize(8, 8)
            pill.setAttribute(Qt.WA_StyledBackground, True)
            self._pills.append(pill)
            hl.addWidget(pill, 0, Qt.AlignVCenter)
        self.set_index(0)

    def set_index(self, i):
        self._index = i
        t = theme_manager.theme
        for n, pill in enumerate(self._pills):
            if n == i:
                bg = theme_manager.accent
            elif n < i:
                bg = theme_manager._accent_soft
            else:
                bg = t.border_visible
            pill.setStyleSheet(
                f"QFrame{{background:{bg};border:none;border-radius:4px;}}"
            )


class _WandbPreviewChart(QWidget):
    """Two-series line chart — the kind wandb draws for loss / metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("wandbPreviewChart")
        self.setFixedSize(148, 96)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = theme_manager.theme
        axis = css_color(t.text)
        axis.setAlpha(200)
        left, top = 12.0, 8.0
        right, bottom = float(self.width() - 8), float(self.height() - 10)
        pen = QPen(axis, 1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(QPointF(left, top), QPointF(left, bottom))
        p.drawLine(QPointF(left, bottom), QPointF(right, bottom))

        span_x = right - left
        span_y = bottom - top

        def pts(fracs):
            return [
                QPointF(left + fx * span_x, bottom - fy * span_y)
                for fx, fy in fracs
            ]

        def draw_series(fracs, color, width, radius):
            points = pts(fracs)
            line = QPen(color, width)
            line.setCapStyle(Qt.RoundCap)
            line.setJoinStyle(Qt.RoundJoin)
            p.setPen(line)
            p.setBrush(Qt.NoBrush)
            for a, b in zip(points, points[1:]):
                p.drawLine(a, b)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            for pt in points:
                p.drawEllipse(pt, radius, radius)

        # Lower series first so the accent line sits on top, matching the sketch.
        draw_series(
            ((0.00, 0.06), (0.24, 0.24), (0.50, 0.40),
             (0.70, 0.56), (0.94, 0.76)),
            css_color(t.purple), 2.0, 3.2)
        draw_series(
            ((0.00, 0.06), (0.18, 0.30), (0.40, 0.52),
             (0.58, 0.64), (0.78, 0.88)),
            css_color(theme_manager.accent), 2.2, 3.6)
        p.end()


class _ReviewRow(QFrame):
    """One review fact — OK/NEED plus hover fill like the start cards."""

    _MIN_H = 48

    def __init__(self, ok, label, value, parent=None):
        super().__init__(parent)
        self._hover = False
        self.setObjectName("setupReviewRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumHeight(self._MIN_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(14, 12, 14, 12)
        hl.setSpacing(8)
        t = theme_manager.theme
        mark = QLabel("OK" if ok else "NEED")
        mark.setAttribute(Qt.WA_TransparentForMouseEvents)
        mark.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;"
            f"color:{t.success if ok else t.warning};background:transparent;"
            "letter-spacing:1px;"
        )
        name = QLabel(label)
        name.setAttribute(Qt.WA_TransparentForMouseEvents)
        name.setStyleSheet(
            "font-family:'Montserrat';font-size:13px;"
            f"color:{t.text_sec};background:transparent;"
        )
        dash = QLabel("—")
        dash.setAttribute(Qt.WA_TransparentForMouseEvents)
        dash.setStyleSheet(
            "font-family:'Montserrat';font-size:13px;"
            f"color:{t.text_muted};background:transparent;"
        )
        val = QLabel(value or "—")
        val.setAttribute(Qt.WA_TransparentForMouseEvents)
        val.setWordWrap(True)
        val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        val.setStyleSheet(
            "font-family:'Montserrat';font-size:13px;"
            f"color:{t.text};background:transparent;"
        )
        self._mark = mark
        self._name = name
        self._dash = dash
        self._val = val
        hl.addWidget(mark, 0, Qt.AlignVCenter)
        hl.addWidget(name, 0, Qt.AlignVCenter)
        hl.addWidget(dash, 0, Qt.AlignVCenter)
        hl.addWidget(val, 1, Qt.AlignVCenter)
        self._paint()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._content_height(w)

    def sizeHint(self):
        w = self.width() if self.width() > 1 else 400
        return QSize(super().sizeHint().width(), self._content_height(w))

    def minimumSizeHint(self):
        sh = self.sizeHint()
        return QSize(sh.width(), max(self._MIN_H, sh.height()))

    def _content_height(self, w):
        lay = self.layout()
        m = lay.contentsMargins() if lay is not None else self.contentsMargins()
        spacing = lay.spacing() if lay is not None else 8
        inner_w = max(40, int(w) - m.left() - m.right())
        for child in (self._mark, self._name, self._dash):
            inner_w -= max(child.sizeHint().width(), child.minimumSizeHint().width())
        inner_w -= spacing * 3
        inner_w = max(40, inner_w)
        fm = QFontMetrics(self._val.font())
        flags = Qt.TextWordWrap | Qt.TextWrapAnywhere
        r = fm.boundingRect(QRect(0, 0, inner_w, 100000), flags, self._val.text())
        text_h = max(fm.height(), r.height())
        return max(self._MIN_H, text_h + m.top() + m.bottom())

    def enterEvent(self, e):
        self._hover = True
        self._paint()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self._paint()
        super().leaveEvent(e)

    def _paint(self):
        t = theme_manager.theme
        if self._hover:
            bg, border = theme_manager._accent_soft, theme_manager.accent
        else:
            bg, border = t.surface, t.border_visible
        self.setStyleSheet(
            f"QFrame#setupReviewRow{{background:{bg};"
            "border-width:1px;border-style:solid;"
            f"border-color:{border};"
            f"border-radius:{UIConstants.CARD_RADIUS_STYLESHEET}px;}}"
        )


_SETUP_STEPS = (
    ("01  START", "From scratch or from a checkpoint."),
    ("02  MODEL", "Architecture and the YAML that defines it."),
    ("03  DATA", "How the audio is laid out, and where it lives."),
    ("04  WANDB", "Optional logging. Empty key means wandb stays off."),
    ("05  GPU", "Measure a train step so batch size fits this card."),
    ("06  REVIEW", "Everything Train needs. Then start."),
)


class _TrainSetupDialog(PageHelpDialog):
    """Linear Wizard sheet: start → model → data → wandb → Fit GPU → review / Train."""

    def __init__(self, page, parent=None):
        self._page = page
        self.start_train = False
        self._fitted = False
        self._mode = "resume" if (page._ckpt_row.value() or "").strip() else "scratch"
        self._step = 0
        stack = QStackedWidget()
        stack.setStyleSheet("background:transparent;")
        self._stack = stack
        stack.addWidget(self._build_start())
        stack.addWidget(self._build_model())
        stack.addWidget(self._build_data())
        stack.addWidget(self._build_wandb())
        stack.addWidget(self._build_gpu())
        stack.addWidget(self._build_review())
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        hv = QVBoxLayout(host)
        hv.setContentsMargins(0, 8, 0, 0)
        hv.setSpacing(12)
        hv.addWidget(stack)
        self._gate = QLabel("")
        self._gate.setWordWrap(True)
        self._gate.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{theme_manager.theme.warning};"
            "background:transparent;"
        )
        self._gate.setVisible(False)
        hv.addWidget(self._gate)
        title, caption = _SETUP_STEPS[0]
        left = _help_section(
            "required", [], caption, extra=host, title=title)
        super().__init__({
            "title": "TRAINING",
            "heading": "WIZARD  ·  TRAINING",
            "intro": (
                "Six questions. Next writes each answer onto the Training tab "
                "and saves it. The last step starts Train if nothing is missing."
            ),
            "required": [
                "From scratch or fine-tune from a checkpoint.",
                "Architecture, YAML config, and optional resume weights.",
                "Dataset layout, training folder, validation folder, results.",
                "Optional wandb API key from wandb.ai/authorize.",
                "Fit GPU measures a train step so batch size does not OOM.",
            ],
            "optional": [],
            "required_caption": caption,
            "primary_text": "NEXT",
            "min_width": 640,
            "content_bottom_gap": 24,
        }, parent or page, left=left)
        self._step_badge = self.findChild(QLabel, "helpRequiredBadge")
        self._step_ticks = _SetupStepTicks(len(_SETUP_STEPS))
        if self._step_badge is not None:
            sec = self._step_badge.parentWidget()
            lay = sec.layout() if sec is not None else None
            if lay is not None:
                for i in range(lay.count()):
                    row = lay.itemAt(i).layout()
                    if row is not None and row.indexOf(self._step_badge) >= 0:
                        row.insertWidget(
                            row.indexOf(self._step_badge) + 1,
                            self._step_ticks, 0, Qt.AlignVCenter)
                        break
        self._next_btn = self.findChild(QPushButton, "helpPrimaryBtn")
        if self._next_btn is not None:
            try:
                self._next_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._next_btn.setFixedSize(78, 30)
            self._next_btn.clicked.connect(self._on_next)
        self._back_btn = QPushButton("BACK")
        self._back_btn.setObjectName("helpBackBtn")
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.setFixedSize(70, 30)
        self._back_btn.setStyleSheet(_cancel_btn_ss(font_size=9))
        self._back_btn.clicked.connect(self._on_back)
        if getattr(self, "_hdr", None) is not None and self._next_btn is not None:
            self._hdr.insertWidget(
                self._hdr.indexOf(self._next_btn), self._back_btn, 0, Qt.AlignVCenter)
        self._fill_from_page()
        self._set_mode(self._mode)
        self._show_step(0)

    def _copy(self, muted=False):
        t = theme_manager.theme
        color = t.text_muted if muted else t.text_sec
        return (
            f"font-family:'Montserrat';font-size:13px;color:{color};"
            "background:transparent;"
        )

    def _page_body(self):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(10)
        return w, vl

    def _build_start(self):
        w, vl = self._page_body()
        self._choice_scratch = _SetupChoice(
            "scratch", "From scratch",
            "Random weights. You pick an architecture and a YAML. "
            "No resume checkpoint is sent.")
        self._choice_resume = _SetupChoice(
            "resume", "Fine-tune / resume",
            "Start from a pretrained catalog model or a checkpoint on disk.")
        self._choice_scratch.picked.connect(self._set_mode)
        self._choice_resume.picked.connect(self._set_mode)
        vl.addWidget(self._choice_scratch)
        vl.addWidget(self._choice_resume)
        vl.addStretch(1)
        return w

    def _build_model(self):
        w, vl = self._page_body()
        self._model = _ComboRowT(
            "Architecture", catalog.model_type_choices(),
            tooltip="--model_type. The YAML must match this architecture.")
        self._model.combo.currentIndexChanged.connect(self._on_arch)
        vl.addWidget(self._model)
        self._starter = _ComboRowT(
            "Starter YAML", [("", "Matching files under configs/")],
            tooltip="Premade configs for this architecture. "
                    "Browse below for a file outside that folder.")
        self._starter.combo.currentIndexChanged.connect(self._on_starter)
        vl.addWidget(self._starter)
        self._config = _PathRow(
            "Config path", "Select a model YAML config…",
            mode="file", file_filter="YAML config (*.yaml *.yml);;All files (*.*)",
            tooltip="Model / training YAML (--config_path).",
            start_dir=os.path.join(REPO_ROOT, "configs"))
        vl.addWidget(self._config)
        self._resume_host = QWidget()
        self._resume_host.setStyleSheet("background:transparent;")
        rv = QVBoxLayout(self._resume_host)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)
        catalog_btn = QPushButton("Pretrained catalog")
        catalog_btn.setCursor(Qt.PointingHandCursor)
        catalog_btn.setFixedHeight(40)
        catalog_btn.setStyleSheet(outline_button_ss())
        catalog_btn.clicked.connect(self._open_catalog)
        rv.addWidget(catalog_btn)
        self._ckpt = _PathRow(
            "Checkpoint", "Checkpoint to start from…",
            mode="file",
            file_filter="Checkpoints (*.ckpt *.pth *.pt *.bin *.th *.chpt);;All files (*.*)",
            tooltip="--start_check_point. Required for fine-tune / resume.")
        rv.addWidget(self._ckpt)
        vl.addWidget(self._resume_host)
        vl.addStretch(1)
        return w

    def _build_data(self):
        w, vl = self._page_body()
        self._dataset = _ComboRowT(
            "Dataset type", list(DATASET_TYPES),
            tooltip="--dataset_type. Validation is always type 1 "
                    "(one folder per song, mixture.wav + stems).")
        vl.addWidget(self._dataset)
        hint = QLabel(
            "Validation must be type 1 regardless of the training layout: "
            "one folder per song, each with mixture.wav and every stem. "
            "Use 16-bit WAV for validation. Training data can be WAV or FLAC "
            "(not MP3). A few hundred tracks is a practical minimum.")
        hint.setWordWrap(True)
        hint.setStyleSheet(self._copy(muted=True))
        vl.addWidget(hint)
        self._data = _PathRow(
            "Data path", "Select the training dataset folder(s)…",
            mode="folder",
            tooltip="Training dataset root(s) (--data_path).",
            multi=True)
        vl.addWidget(self._data)
        self._valid = _PathRow(
            "Valid path", "Select the validation folder(s)…",
            mode="folder",
            tooltip="Validation set (--valid_path).",
            multi=True)
        vl.addWidget(self._valid)
        self._results = _PathRow(
            "Results path", "Select the results folder…", mode="folder",
            tooltip="Checkpoints, metadata cache, and the training log.")
        vl.addWidget(self._results)
        vl.addStretch(1)
        return w

    def _build_wandb(self):
        w, vl = self._page_body()
        head = QWidget()
        head.setStyleSheet("background:transparent;")
        hh = QHBoxLayout(head)
        hh.setContentsMargins(0, 0, 0, 0)
        hh.setSpacing(16)
        why = QLabel(
            "Weights & Biases logs loss and metrics in the browser. "
            "Create a free account at wandb.ai, then copy the API key from "
            "wandb.ai/authorize. Leave the key empty to keep logging off. "
            "The key is saved with Training settings and injected as "
            "WANDB_API_KEY — it never appears on the command line.")
        why.setWordWrap(True)
        why.setStyleSheet(self._copy())
        hh.addWidget(why, 1)
        self._wandb_chart = _WandbPreviewChart()
        hh.addWidget(self._wandb_chart, 0, Qt.AlignTop)
        vl.addWidget(head)
        accent = theme_manager.accent
        t = theme_manager.theme
        links = QLabel(
            f'<a href="{WANDB_HOME}" style="color:{accent};text-decoration:none;">'
            "Open wandb.ai</a>"
            f'<span style="color:{t.text_muted};">&nbsp;&nbsp;·&nbsp;&nbsp;</span>'
            f'<a href="{WANDB_AUTHORIZE}" style="color:{accent};text-decoration:none;">'
            "Get an API key</a>"
        )
        links.setTextFormat(Qt.RichText)
        links.setOpenExternalLinks(True)
        pal = links.palette()
        pal.setColor(QPalette.ColorRole.Link, css_color(accent))
        pal.setColor(QPalette.ColorRole.LinkVisited, css_color(accent))
        links.setPalette(pal)
        links.setStyleSheet(
            "font-family:'Montserrat';font-size:13px;background:transparent;"
        )
        self._wandb_links = links
        vl.addWidget(links)
        self._wandb = QLineEdit()
        self._wandb.setEchoMode(QLineEdit.Password)
        self._wandb.setPlaceholderText("API key (empty = wandb disabled)")
        self._wandb.setFixedHeight(34)
        t = theme_manager.theme
        self._wandb.setStyleSheet(
            f"QLineEdit{{font-family:'Montserrat';font-size:13px;color:{t.text};"
            f"background:{t.input_bg};border:1px solid {t.border_visible};"
            f"border-radius:6px;padding:0 8px;}}"
            f"QLineEdit::placeholder{{color:{t.text_muted};}}"
        )
        vl.addWidget(self._wandb)
        off_wrap = _pill_switch("Online", "Offline", False)
        self._wandb_offline = off_wrap._sw
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        self._wandb_mode_lbl = QLabel("")
        self._wandb_mode_lbl.setStyleSheet(self._copy())
        hl.addWidget(self._wandb_mode_lbl, 0, Qt.AlignVCenter)
        hl.addStretch(1)
        hl.addWidget(off_wrap, 0, Qt.AlignVCenter)
        vl.addWidget(row)
        self._wandb_offline.toggled.connect(self._refresh_wandb_mode)
        self._refresh_wandb_mode()
        vl.addStretch(1)
        return w

    def _refresh_wandb_mode(self, *_):
        offline = bool(self._wandb_offline.is_checked())
        self._wandb_mode_lbl.setText(
            "Log offline (no upload)" if offline
            else "Log online (upload to wandb.ai)"
        )

    def _build_gpu(self):
        w, vl = self._page_body()
        why = QLabel(
            "A dummy train step on this GPU is the only way to know the batch "
            "size will not CUDA OOM. Probe, then Apply on that sheet — if the "
            "step fits, the wizard continues. Skip only if you already trust "
            "the knobs on the Training tab.")
        why.setWordWrap(True)
        why.setStyleSheet(self._copy())
        vl.addWidget(why)
        probe = QPushButton("Probe this GPU")
        probe.setCursor(Qt.PointingHandCursor)
        probe.setFixedHeight(40)
        probe.setStyleSheet(
            outline_button_ss() + "QPushButton{padding:0 18px;}")
        probe.clicked.connect(self._probe)
        self._probe_btn = probe
        vl.addWidget(probe, 0, Qt.AlignLeft)
        self._fit_status = QLabel("Not probed yet.")
        self._fit_status.setWordWrap(True)
        self._fit_status.setStyleSheet(self._copy(muted=True))
        vl.addWidget(self._fit_status)
        vl.addStretch(1)
        return w

    def _build_review(self):
        w, vl = self._page_body()
        self._review = QWidget()
        self._review.setObjectName("setupReview")
        self._review.setStyleSheet("background:transparent;")
        self._review_layout = QVBoxLayout(self._review)
        self._review_layout.setContentsMargins(0, 0, 0, 0)
        self._review_layout.setSpacing(6)
        vl.addWidget(self._review)
        vl.addStretch(1)
        return w

    def _fill_from_page(self):
        p = self._page
        if p._model_row.key():
            self._model.set_key(p._model_row.key())
        self._refresh_starters(keep=p._config_row.value())
        if p._config_row.value():
            self._config.set_value(p._config_row.value())
        if p._ckpt_row.value():
            self._ckpt.set_value(p._ckpt_row.value())
        if p._dataset_row.key():
            self._dataset.set_key(p._dataset_row.key())
        if p._data_row.paths():
            self._data.set_paths(p._data_row.paths())
        if p._valid_row.paths():
            self._valid.set_paths(p._valid_row.paths())
        if p._results_row.value():
            self._results.set_value(p._results_row.value())
        self._wandb.setText(p._run_opts.get("wandb_key") or "")
        self._wandb_offline.set_checked(bool(p._run_opts.get("wandb_offline")))

    def _set_mode(self, mode):
        self._mode = "resume" if mode == "resume" else "scratch"
        self._choice_scratch.set_on(self._mode == "scratch")
        self._choice_resume.set_on(self._mode == "resume")
        if getattr(self, "_resume_host", None) is not None:
            self._resume_host.setVisible(self._mode == "resume")
        self._clear_gate()

    def _on_arch(self, *_):
        self._refresh_starters()

    def _refresh_starters(self, keep=None):
        keep = keep if keep is not None else self._config.value()
        paths = _starter_yamls(self._model.key() or "")
        items = [("", "Matching files under configs/")]
        items += [(p, os.path.basename(p)) for p in paths[:40]]
        keys = {p for p, _ in items if p}
        if keep and os.path.isfile(keep):
            pick = keep
        elif paths:
            pick = paths[0]
        else:
            pick = ""
        self._starter.set_items(items, keep=pick if pick in keys else "")
        if pick:
            self._config.set_value(pick)

    def _on_starter(self, *_):
        path = self._starter.key()
        if path:
            self._config.set_value(path)

    def _open_catalog(self):
        dlg = PretrainedModelsDialog(self, architecture=self._model.key())
        dlg.use_for_training.connect(self._on_pretrained)
        dlg.exec()

    def _on_pretrained(self, config_path, checkpoint_path):
        if config_path:
            self._config.set_value(config_path)
            guessed = catalog.guess_model_type_from_name(config_path)
            locked = self._model.key()
            if guessed and (not locked or guessed == locked):
                self._model.set_key(guessed)
        if checkpoint_path:
            self._ckpt.set_value(checkpoint_path)
            self._set_mode("resume")

    def _probe(self):
        self._sync_to_page()
        if self._page._open_fit_gpu(blur=False):
            self._fitted = True
            self._on_next()
            return
        self._fit_status.setText("Probe closed without applying.")

    def _sync_to_page(self):
        p = self._page
        if self._mode == "scratch":
            p._ckpt_row.set_value("")
        elif self._ckpt.value():
            p._ckpt_row.set_value(self._ckpt.value())
        if self._model.key():
            p._model_row.set_key(self._model.key())
        if self._config.value():
            new = self._config.value()
            # Re-setting the same YAML must not reload training.batch_size
            # (and friends) from disk — Fit GPU already wrote those onto the
            # tab. A *new* path still emits so _on_config_changed can fill.
            if _same_paths([p._config_row.value()], [new]):
                p._config_row.blockSignals(True)
                p._config_row.set_value(new)
                p._config_row.blockSignals(False)
            else:
                p._config_row.set_value(new)
        if self._dataset.key():
            p._dataset_row.set_key(self._dataset.key())
        if self._data.paths():
            p._data_row.set_paths(self._data.paths())
        if self._valid.paths():
            p._valid_row.set_paths(self._valid.paths())
        if self._results.value():
            p._results_row.set_value(self._results.value())
        p._run_opts["wandb_key"] = self._wandb.text().strip()
        p._run_opts["wandb_offline"] = bool(self._wandb_offline.is_checked())
        p._persist()

    def _step_error(self):
        if self._step == 0:
            if self._mode not in ("scratch", "resume"):
                return "Pick from scratch or fine-tune to continue."
            return None
        if self._step == 1:
            if not self._model.key():
                return "Pick an architecture."
            cfg = self._config.value()
            if not cfg or not os.path.isfile(cfg):
                return "Select a config YAML that exists on disk."
            if self._mode == "resume":
                ckpt = self._ckpt.value()
                if not ckpt or not os.path.isfile(ckpt):
                    return "Fine-tune needs a checkpoint — catalog or a file on disk."
            return None
        if self._step == 2:
            if not self._data.paths():
                return "Select the training data folder."
            if not self._valid.paths():
                return "Select the validation folder."
            if not self._results.value():
                return "Select a results folder."
            return None
        if self._step == 5:
            self._sync_to_page()
            return self._page._validate()
        return None

    def _clear_gate(self):
        self._gate.clear()
        self._gate.setVisible(False)

    def _show_gate(self, text):
        self._gate.setText(text or "")
        self._gate.setVisible(bool(text))

    def _show_step(self, i):
        self._step = max(0, min(i, len(_SETUP_STEPS) - 1))
        self._stack.setCurrentIndex(self._step)
        title, caption = _SETUP_STEPS[self._step]
        if self._step_badge is not None:
            self._step_badge.setText(title)
        if getattr(self, "_step_ticks", None) is not None:
            self._step_ticks.set_index(self._step)
        sec = self._step_badge.parentWidget() if self._step_badge else None
        if sec is not None:
            for child in sec.findChildren(QLabel):
                if child is not self._step_badge and child.parent() is sec:
                    child.setText(caption)
                    break
        last = self._step >= len(_SETUP_STEPS) - 1
        if self._back_btn is not None:
            self._back_btn.setVisible(self._step > 0)
        if self._next_btn is not None:
            self._next_btn.setText("TRAIN" if last else "NEXT")
        if last:
            self._refresh_review()
        self._clear_gate()
        if getattr(self, "_inner", None) is not None:
            self._refit()

    def _review_line(self, ok, label, value):
        return _ReviewRow(ok, label, value)

    def _review_text(self):
        labels = self._review.findChildren(QLabel)
        return " ".join(l.text() for l in labels)

    def _clear_review(self):
        while self._review_layout.count():
            item = self._review_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _refresh_review(self):
        self._sync_to_page()
        cfg = self._config.value()
        ckpt = self._ckpt.value() if self._mode == "resume" else ""
        data = ", ".join(os.path.basename(p.rstrip("\\/")) or p
                         for p in self._data.paths()) or ""
        valid = ", ".join(os.path.basename(p.rstrip("\\/")) or p
                          for p in self._valid.paths()) or ""
        lines = [
            self._review_line(bool(self._model.key()), "Architecture",
                              self._model.key() or ""),
            self._review_line(bool(cfg and os.path.isfile(cfg)), "Config",
                              os.path.basename(cfg) if cfg else ""),
            self._review_line(
                self._mode == "scratch" or bool(ckpt and os.path.isfile(ckpt)),
                "Checkpoint",
                "from scratch" if self._mode == "scratch"
                else (os.path.basename(ckpt) if ckpt else "")),
            self._review_line(bool(self._data.paths()), "Training data", data),
            self._review_line(bool(self._valid.paths()), "Validation", valid),
            self._review_line(bool(self._results.value()), "Results",
                              self._results.value()),
            self._review_line(True, "wandb",
                              "key saved" if (self._wandb.text() or "").strip()
                              else "off — no API key"),
            self._review_line(
                self._fitted, "Fit GPU",
                (f"applied · batch {self._page._batch_row.value() or '—'}"
                 if self._fitted else "not probed — recommended")),
        ]
        err = self._page._validate()
        if err:
            extra = (
                f'<p style="color:{theme_manager.theme.warning};">'
                f'{html.escape(err)}</p>'
            )
        else:
            extra = (
                f'<p style="color:{theme_manager.theme.text_muted};">'
                "Train starts the job. × keeps these choices on the tab "
                "without starting.</p>"
            )
        self._clear_review()
        for row in lines:
            self._review_layout.addWidget(row)
        self._review_layout.addSpacing(12)
        foot = QLabel(extra)
        foot.setWordWrap(True)
        foot.setTextFormat(Qt.RichText)
        foot.setStyleSheet(self._copy())
        self._review_layout.addWidget(foot)
        QTimer.singleShot(0, self._refit)

    def _on_back(self):
        self._show_step(self._step - 1)

    def _on_next(self):
        err = self._step_error()
        if err:
            self._show_gate(err)
            if getattr(self, "_inner", None) is not None:
                self._refit()
            return
        self._sync_to_page()
        if self._step >= len(_SETUP_STEPS) - 1:
            self.start_train = True
            super().accept()
            return
        self._show_step(self._step + 1)

    def accept(self):
        self._on_next()


class _ConfigEditorDialog(_DialogBase):
    """Raw YAML editor for the selected config (validated before saving)."""
    saved = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__("EDIT CONFIGURATION", parent, size=(980, 620))
        self._path = path
        self.hint(f"Editing <span style=\"color:{theme_manager.accent};font-weight:bold;\">"
                  f"{html.escape(path)}</span> — the file is validated as YAML before it is written.")
        self._edit = QPlainTextEdit()
        self._edit.setStyleSheet(
            "QPlainTextEdit{" + _mono_box_ss() + "font-size:11px;}"
        )
        self._edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._edit.setPlainText(f.read())
        except Exception as exc:
            self._edit.setPlainText(f"# could not read file: {exc}")
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(self._edit, 1)
        tips = _yaml_tips_panel()
        tips.setFixedWidth(272)
        row.addWidget(tips, 0, Qt.AlignTop)
        self.body.addLayout(row, 1)
        self.add_buttons("Save Changes", self._save)

    def _save(self):
        import yaml
        text = self._edit.toPlainText()
        try:
            yaml.load(text, Loader=yaml.FullLoader)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid YAML", f"The configuration is not valid YAML:\n\n{exc}")
            return
        try:
            with open(self._path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.saved.emit(self._path)
        self.accept()


# ── Output parsing ────────────────────────────────────────────────────────────

_TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[([^\]]*)\]")
_LOSS_RE = re.compile(r"\bloss=([-+\d.eE]+)")
_AVG_RE = re.compile(r"avg_loss=([-+\d.eE]+)")
_EPOCH_RE = re.compile(r"Train epoch:\s*(\d+)\s+Learning rate:\s*([-+\d.eE]+)")
_INSTR_RE = re.compile(r"^Instr\s+(\S+)\s+(\S+):\s*([-+\d.eEinfa]+)")
_AVG_METRIC_RE = re.compile(r"^Metric avg\s+(\S+)\s*:\s*([-+\d.eEinfa]+)")
_METRIC_VAL_RE = re.compile(r"^Metric\s+(\S+)\s+value:\s*([-+\d.eEinfa]+)")
_TRAIN_FOR_RE = re.compile(r"Train for:\s*(\d+)\s*epochs")
_SUMMARY_KEYS = ("Model:", "Total parameters:", "Trainable parameters:",
                 "Model size:", "Number of layers:")


def _is_progress_line(line):
    return bool(_TQDM_RE.search(line)) and ("it/s" in line or "s/it" in line or "%|" in line)


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ── Page ──────────────────────────────────────────────────────────────────────

class _ExportWeightsDialog(_DialogBase):
    """Pick the checkpoint to strip, where to write the weights-only file,
    and whether to store them as float16 (half the size)."""

    def __init__(self, checkpoint, results_dir, parent=None):
        super().__init__("EXPORT WEIGHTS", parent, size=(560, 330))
        self._results_dir = results_dir or ""
        self.hint("Keeps only the model weights (model_state_dict) — optimizer, scheduler "
                  "and metric history are dropped. This is the file to register for inference.")

        self._src = self._path_field("CHECKPOINT", checkpoint, self._browse_src)
        self._dst = self._path_field("OUTPUT FILE", self._default_output(checkpoint), self._browse_dst)

        row = QHBoxLayout()
        row.setContentsMargins(0, 6, 0, 0)
        row.setSpacing(10)
        self._fp16 = _MiniSwitch(False, self)
        row.addWidget(self._fp16)
        lb = QLabel("Store weights as float16 (half the file size, same quality for inference)")
        lb.setStyleSheet(_value_ss())
        row.addWidget(lb, 1)
        self.body.addLayout(row)
        self.body.addStretch()
        self.add_buttons("Export", self._on_export)

    def _path_field(self, label, value, browse):
        t = theme_manager.theme
        cap = QLabel(label)
        cap.setStyleSheet(_lbl_ss())
        self.body.addWidget(cap)
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame{{background:{t.input_bg};border:1px solid {t.border_visible};border-radius:6px;}}")
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(12, 0, 8, 0)
        hl.setSpacing(6)
        edit = QLineEdit(value or "")
        edit.setStyleSheet(
            "QLineEdit{background:transparent;border:none;font-family:'Montserrat';"
            f"font-size:11px;color:{t.text};padding:0;}}"
            f"QLineEdit::placeholder{{color:{t.text_muted};}}")
        edit.setFixedHeight(38)
        hl.addWidget(edit, 1)
        btn = EllipsisButton()
        btn.clicked.connect(browse)
        hl.addWidget(btn)
        self.body.addWidget(frame)
        return edit

    @staticmethod
    def _default_output(checkpoint):
        if not checkpoint:
            return ""
        base, ext = os.path.splitext(checkpoint)
        return f"{base}_weights{ext or '.ckpt'}"

    def _browse_src(self):
        start = self._src.text() or self._results_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Select training checkpoint", start,
            "Checkpoints (*.ckpt *.pth *.pt *.bin *.th);;All files (*.*)")
        if path:
            self._src.setText(path)
            if not self._dst.text():
                self._dst.setText(self._default_output(path))

    def _browse_dst(self):
        start = self._dst.text() or self._default_output(self._src.text()) or self._results_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Save weights as", start, "Checkpoint (*.ckpt);;All files (*.*)")
        if path:
            self._dst.setText(path)

    def _on_export(self):
        src, dst = self._src.text().strip(), self._dst.text().strip()
        if not src or not os.path.isfile(src):
            QMessageBox.warning(self, "Export weights", "Select an existing checkpoint file.")
            return
        if not dst:
            QMessageBox.warning(self, "Export weights", "Choose where to write the weights file.")
            return
        if os.path.abspath(src) == os.path.abspath(dst):
            QMessageBox.warning(self, "Export weights",
                                "The output must be a different file than the checkpoint.")
            return
        self.accept()

    def values(self):
        return self._src.text().strip(), self._dst.text().strip(), self._fp16.is_checked()


class TrainingPage(QWidget):
    process_running = Signal(bool)
    log_output = Signal(str)

    # Log lines survive the page rebuild a theme switch performs.
    _LOG_HISTORY = []
    _LOG_HISTORY_MAX = 4000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingPage")
        # Object-name scoped so the background never cascades into dialogs.
        self.setStyleSheet(f"#trainingPage{{background:{theme_manager.theme.bg};}}")
        self._runner = None
        self._stop_requested = False
        self._log_file = None
        self._run_opts = dict(DEFAULT_RUN_OPTS)
        self._checkpoint_override = None
        self._chunk_override = None
        self._losses = catalog.default_losses()
        self._use_standard_loss = True
        self._metrics = catalog.default_metrics()
        self._metric_vals = {}      # metric -> {instrument: value}
        self._tiles = {}
        self._phase = "idle"
        self._epoch = None
        self._epochs_total = None
        self._lr = None
        self._summary = {}
        self._build_ui()
        self._restore_log()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        t = theme_manager.theme
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header_w = QWidget()
        header_w.setStyleSheet("background:transparent;")
        hh = QHBoxLayout(header_w)
        hh.setContentsMargins(32, 32, 32, 0)
        hh.addWidget(PageHeader(
            "TRAINING",
            "TRAIN & FINE-TUNE SEPARATION MODELS",
            highlight="SEPARATION MODELS",
            help_key="training",
        ))
        root.addWidget(header_w)
        root.addSpacing(UIConstants.HEADER_CONTENT_GAP)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(_page_edge_scroll_ss())
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setStyleSheet("background:transparent;")
        body = QHBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        body.addWidget(self._build_config_column(), 1)
        body.addWidget(self._build_settings_column(), 1)
        body.addWidget(self._build_monitor_column(), 1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        root.addWidget(self._build_action_bar())

    def _column(self, margins):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        vl = QVBoxLayout(w)
        vl.setContentsMargins(*margins)
        vl.setSpacing(0)
        return w, vl

    def _build_config_column(self):
        w, ll = self._column((32, 16, 10, 16))
        # Same header rhythm as the INFERENCE page: the 7px push-down makes
        # CONFIGURATION start at the same height as the INFERENCE tab's
        # section headers, so switching tabs doesn't jump.
        ll.addSpacing(7)
        ll.addWidget(_sec_hdr("Configuration"))
        ll.addSpacing(19)
        cfg = QVBoxLayout()
        cfg.setSpacing(6)
        cfg.setContentsMargins(0, 0, 0, 0)

        self._model_row = _ComboRowT(
            "Model type", catalog.model_type_choices(),
            tooltip="Architecture to train (--model_type).\n"
                    "Picked automatically from the config file name when possible.")
        cfg.addWidget(self._model_row)

        self._config_row = _PathRow(
            "Config path", "Select a model YAML config…",
            mode="file", file_filter="YAML config (*.yaml *.yml);;All files (*.*)",
            tooltip="Model / training YAML (--config_path).\n"
                    "The TRAINING SETTINGS column is filled from it.",
            start_dir=os.path.join(REPO_ROOT, "configs"))
        self._config_row.changed.connect(self._on_config_changed)
        cfg.addWidget(self._config_row)

        self._results_row = _PathRow(
            "Results path", "Select the results folder…", mode="folder",
            tooltip="Where checkpoints, metadata cache and the training log go\n(--results_path).")
        self._results_row.changed.connect(self._refresh_latest_chip)
        cfg.addWidget(self._results_row)

        self._data_row = _PathRow(
            "Data path", "Select the training dataset folder(s)…", mode="folder",
            tooltip="Training dataset root(s) (--data_path).\n"
                    "Use WAV or FLAC — not MP3. Drop several folders to train on all.\n"
                    "Type 1: one folder per song, all stems same length.\n"
                    "Type 2: one folder per stem; lengths can differ (random chunks).\n"
                    "A few hundred tracks is a practical minimum; more is better.",
            multi=True)
        cfg.addWidget(self._data_row)

        self._valid_row = _PathRow(
            "Valid path", "Select the validation folder(s)…", mode="folder",
            tooltip="Validation set (--valid_path): type 1 layout — one folder per song\n"
                    "with mixture.wav and every stem. Prefer 16-bit WAV (FLAC can fail).\n"
                    "All stems in a song must be the same length.\n"
                    "30–60 s clips validate faster; full tracks score higher.",
            multi=True)
        cfg.addWidget(self._valid_row)

        self._dataset_row = _ComboRowT(
            "Dataset type", [(k, k) for k, _d in DATASET_TYPES],
            tooltip="Dataset layout (--dataset_type):\n"
                    + "\n".join(d for _k, d in DATASET_TYPES)
                    + "\n\nDetails: github.com/ZFTurbo/Music-Source-Separation-Training/"
                    "blob/main/docs/dataset_types.md")
        cfg.addWidget(self._dataset_row)

        self._gpu_row = _ChevronRow(
            "GPUs / Workers / Seed",
            tooltip="Devices, DataLoader workers, random seed and the\n"
                    "run flags (pre-validation, resume options, …).",
            placeholder="Choose devices and run options…")
        self._gpu_row.clicked.connect(self._open_run_options)
        cfg.addWidget(self._gpu_row)

        self._ckpt_row = _ClickablePathRow(
            "Resume", "Checkpoint to start from…",
            mode="file", file_filter="Checkpoints (*.ckpt *.pth *.pt *.bin *.th *.chpt);;All files (*.*)",
            tooltip=("Initial weights (--start_check_point).\n"
                     "Leave empty to train from scratch.\n"
                     "LAST picks the newest checkpoint in the results folder;\n"
                     "the > arrow browses the pre-trained catalog."),
            picker=True)
        self._latest_chip = QPushButton("Last")
        self._latest_chip.setFixedHeight(18)
        self._latest_chip.setCursor(Qt.PointingHandCursor)
        self._latest_chip.setToolTip("Use the newest checkpoint from the results folder")
        self._latest_chip.setStyleSheet(_chip_ss(False))
        self._latest_chip.clicked.connect(self._pick_latest_ckpt)
        self._ckpt_row.add_extra(self._latest_chip)
        self._clear_ckpt_btn = QPushButton("\u2715")
        self._clear_ckpt_btn.setFixedSize(18, 18)
        self._clear_ckpt_btn.setCursor(Qt.PointingHandCursor)
        self._clear_ckpt_btn.setToolTip(
            "Remove the resume checkpoint so the next run trains from scratch")
        self._clear_ckpt_btn.setStyleSheet(
            "QPushButton{font-family:'Montserrat';font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:{theme_manager.theme.surface_alt};"
            f"border:1px solid {theme_manager.theme.border};border-radius:9px;padding:0;"
            "}"
            f"QPushButton:hover{{color:{theme_manager.theme.error};"
            f"border:1px solid {theme_manager.theme.error};}}"
        )
        self._clear_ckpt_btn.clicked.connect(self._clear_ckpt)
        self._clear_ckpt_btn.setVisible(False)
        self._ckpt_row.add_extra(self._clear_ckpt_btn)
        self._ckpt_row.changed.connect(self._on_ckpt_changed)
        self._ckpt_row.pick_requested.connect(self._open_pretrained)

        opt = OptionalFold(fill_leftover=False)
        opt.addWidget(self._ckpt_row)

        self._augment_row = _SwitchRow(
            "Augmentation", "Disabled", "Enabled",
            tooltip="Sets augmentations.enable in the run config.\n"
                    "Pitch/time/noise options live in the YAML under augmentations.\n"
                    "Useful for scratch training; often off when fine-tuning.",
            checked=True, make_label=_lbl_with_info)
        opt.addWidget(self._augment_row)

        self._edit_row = _ChevronRow("", placeholder="Edit Configuration", icon=_PencilIcon())
        self._edit_row.clicked.connect(self._open_config_editor)
        opt.addWidget(self._edit_row)

        ll.addLayout(cfg)
        ll.addWidget(opt, 1)
        return w

    def _build_settings_column(self):
        w, ml = self._column((10, 16, 10, 16))
        # 7px push-down + 19px gap: TRAINING SETTINGS starts at exactly the
        # same height as CONFIGURATION / TRAINING MONITOR (and the INFERENCE
        # tab's headers) — the three columns share one baseline.
        ml.addSpacing(7)
        ml.addWidget(_sec_hdr("Training Settings"))
        ml.addSpacing(19)
        st = QVBoxLayout()
        st.setSpacing(6)
        st.setContentsMargins(0, 0, 0, 0)

        self._batch_row = _EditRow("Batch size", "training.batch_size per GPU.", "from config", int_only=True)
        st.addWidget(self._batch_row)
        self._lr_row = _EditRow(
            "Learning rate",
            "training.lr — common fine-tune values: 1e-5 or 5e-6.",
            "from config")
        st.addWidget(self._lr_row)
        self._accum_row = _EditRow("Grad accum steps",
                                   "training.gradient_accumulation_steps: batches summed\nbefore an optimizer step.",
                                   "from config", int_only=True)
        st.addWidget(self._accum_row)
        self._epochs_row = _EditRow("Max epochs", "training.num_epochs.", "from config", int_only=True)
        st.addWidget(self._epochs_row)
        self._optim_row = _ComboRowT("Optimizer", [(o, o) for o in catalog.optimizers()],
                                     tooltip="training.optimizer. muon / adago need an 'optimizer:'\n"
                                             "section with muon_group / adam_group in the YAML.")
        st.addWidget(self._optim_row)
        self._optim_warn = QLabel("")
        self._optim_warn.setWordWrap(True)
        self._optim_warn.setStyleSheet(
            f"font-family:'Montserrat';font-size:9px;color:{theme_manager.theme.warning};"
            "background:transparent;"
        )
        self._optim_warn.setVisible(False)
        st.addWidget(self._optim_warn)
        self._optim_row.combo.currentIndexChanged.connect(
            lambda *_: self._refresh_gpu_warnings())
        self._model_row.combo.currentIndexChanged.connect(
            lambda *_: self._refresh_gpu_warnings())
        self._batch_row.edit.textChanged.connect(
            lambda *_: self._refresh_gpu_warnings())
        self._refresh_gpu_warnings()
        self._loss_row = _ChevronRow(
            "Loss", tooltip="Loss functions summed for training (--loss).\n"
                            "RoFormer / Conformer models use their internal loss unless\n"
                            "'use standard loss' is switched on in the dialog.",
            placeholder="Choose loss functions…")
        self._loss_row.clicked.connect(self._open_loss_dialog)
        st.addWidget(self._loss_row)
        self._metrics_row = _ChevronRow(
            "Metrics", tooltip="Validation metrics computed each epoch (--metrics).\n"
                               "More metrics = slower validation.\n"
                               "Bleedless and fullness often move in opposite directions.\n"
                               "Listen to outputs — numbers guide tuning, not taste.",
            placeholder="Choose validation metrics…")
        self._metrics_row.clicked.connect(self._open_metrics_dialog)
        st.addWidget(self._metrics_row)
        self._sched_row = _ComboRowT("Metric for scheduler", [(m, m) for m in self._metrics],
                                     tooltip="Metric that drives ReduceLROnPlateau and decides which\n"
                                             "epochs count as 'best' (--metric_for_scheduler).")
        st.addWidget(self._sched_row)
        self._patience_row = _EditRow("Patience", "training.patience: epochs without improvement\nbefore the LR is reduced.", "from config", int_only=True)
        st.addWidget(self._patience_row)
        self._reduce_row = _EditRow("Reduce factor", "training.reduce_factor: LR multiplier on plateau.", "from config")
        st.addWidget(self._reduce_row)
        ml.addLayout(st)

        ml.addSpacing(6)
        self._summary_card = _Card("Model Summary")
        self._summary_lbl = QLabel()
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setTextFormat(Qt.RichText)
        self._summary_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_sec};"
            "background:transparent;border:none;line-height:1.5;"
        )
        self._summary_card.add(self._summary_lbl)
        ml.addWidget(self._summary_card)
        self._render_summary()
        ml.addStretch()
        self._refresh_loss_text()
        self._refresh_metrics_text()
        return w

    def _build_monitor_column(self):
        t = theme_manager.theme
        w, rl = self._column((10, 16, 32, 16))
        # Same 7px push-down + 19px gap as the other two columns so all
        # three section headers and their first cards line up.
        rl.addSpacing(7)
        rl.addWidget(_sec_hdr("Training Monitor"))
        rl.addSpacing(19)

        # Progress
        self._prog_card = _Card("Training Progress")
        self._prog_lbl = QLabel("0 / 0 (0.00%)")
        self._prog_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:12px;color:{t.text_sec};background:transparent;"
        )
        self._prog_card.add_title_widget(self._prog_lbl)
        self._bar = QProgressBar()
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setStyleSheet(
            f"QProgressBar{{background:{t.border};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{theme_manager.accent};border-radius:3px;}}"
        )
        self._prog_card.add(self._bar)
        self._stat_box = QLabel()
        self._stat_box.setTextFormat(Qt.RichText)
        self._stat_box.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._stat_box.setStyleSheet(_mono_box_ss() + "font-size:12px;")
        self._stat_box.setMinimumHeight(64)
        self._stat_box.setWordWrap(True)
        self._prog_card.add(self._stat_box)
        rl.addWidget(self._prog_card)
        rl.addSpacing(10)

        # Metrics
        self._metrics_card = _Card("Metrics (validation)")
        self._tiles_grid = _TileGrid()
        self._metrics_card.add(self._tiles_grid)
        rl.addWidget(self._metrics_card)
        rl.addSpacing(10)

        # Log
        self._log_card = _Card("Training Log")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(self._LOG_HISTORY_MAX)
        self._log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._log.setStyleSheet(
            "QPlainTextEdit{" + _mono_box_ss() + "font-size:12px;}"
            "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
            f"QScrollBar::handle:vertical{{background:{t.scrollbar_handle};border-radius:2px;min-height:30px;}}"
            "QScrollBar::add-line:vertical{height:0;}QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar:horizontal{height:4px;background:transparent;margin:0;}"
            f"QScrollBar::handle:horizontal{{background:{t.scrollbar_handle};border-radius:2px;}}"
            "QScrollBar::add-line:horizontal{width:0;}QScrollBar::sub-line:horizontal{width:0;}"
        )
        self._log.setMinimumHeight(220)
        self._log_card.add(self._log, 1)
        rl.addWidget(self._log_card, 1)

        self._reset_monitor()
        self._rebuild_tiles()
        return w

    def _build_action_bar(self):
        t = theme_manager.theme
        bar = QWidget()
        bar.setStyleSheet("background:transparent;")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(32, 14, 32, 32)
        hb.setSpacing(0)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        left.addWidget(_sec_hdr("Run Training"))
        left.addSpacing(19)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_run = GlyphButton("Train", "▶", _solid_icon_color,
                                   glyph_size=18, text_size=12, parent=self)
        self.btn_run.setStyleSheet(solid_button_ss())
        self.btn_run.fit_contents()
        self.btn_run.clicked.connect(self._run)
        self.btn_setup = GlyphButton("Wizard", BOLT_GLYPH, _outline_icon_color,
                                     glyph_size=16, text_size=12, parent=self)
        self.btn_setup.setStyleSheet(outline_button_ss())
        self.btn_setup.fit_contents()
        self.btn_setup.setToolTip(T_TRAIN_SETUP)
        self.btn_setup.clicked.connect(self._open_setup)
        self.btn_fit = GlyphButton("Fit GPU", CHIP_GLYPH, _outline_icon_color,
                                   glyph_size=16, text_size=12, parent=self)
        self.btn_fit.setStyleSheet(outline_button_ss())
        self.btn_fit.fit_contents()
        self.btn_fit.clicked.connect(self._open_fit_gpu)
        self.btn_stop = GlyphButton("Stop", "■", _stop_icon_color,
                                    glyph_size=16, text_size=12, parent=self)
        self.btn_stop.fit_contents()
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
        btn_row.addWidget(self.btn_setup)
        btn_row.addWidget(self.btn_fit)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        left.addLayout(btn_row)
        self._refresh_fit_btn()
        hb.addLayout(left, 1)

        right = QHBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        self.btn_results = _IconTextButton("Open Output", "folder")
        self.btn_results.fit_contents()
        self.btn_results.clicked.connect(self._open_results)
        self.btn_logs = _IconTextButton("View Logs", "doc")
        self.btn_logs.fit_contents()
        self.btn_logs.clicked.connect(self._open_logs)
        self.btn_export = _IconTextButton("Export Weights", "export")
        self.btn_export.fit_contents()
        self.btn_export.setToolTip(T_EXPORT_WEIGHTS)
        self.btn_export.clicked.connect(self._export_weights)
        self.btn_wandb = _IconTextButton("Open wandb", "wandb")
        self.btn_wandb.fit_contents()
        self.btn_wandb.setToolTip(T_OPEN_WANDB)
        self.btn_wandb.clicked.connect(self._open_wandb)
        right.addWidget(self.btn_logs)
        right.addWidget(self.btn_wandb)
        right.addWidget(self.btn_export)
        right.addWidget(self.btn_results)
        hb.addLayout(right, 0)
        hb.setAlignment(right, Qt.AlignBottom)
        return bar

    def reapply_theme(self):
        self.setStyleSheet(f"#trainingPage{{background:{theme_manager.theme.bg};}}")
        for row in self.findChildren(_SwitchRow):
            row.reapply_theme()

    # ── Config / settings plumbing ───────────────────────────────────────
    def _on_config_changed(self):
        path = self._config_row.value()
        if not path or not os.path.isfile(path):
            return
        self._guess_model_type(path)
        self._load_fields_from_yaml(path)

    def _guess_model_type(self, path):
        cfg = self._read_yaml(path) or {}
        mt = (cfg.get("training") or {}).get("model_type") if isinstance(cfg.get("training"), dict) else None
        if not mt:
            mt = catalog.guess_model_type_from_name(path)
        if mt and self._model_row.combo.findData(mt) >= 0:
            self._model_row.set_key(mt)

    @staticmethod
    def _read_yaml(path):
        try:
            import yaml
            with open(path, "rb") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)
        except Exception as exc:
            print(f"[training] could not read {path}: {exc}")
            return None

    def _load_fields_from_yaml(self, path):
        cfg = self._read_yaml(path)
        if not isinstance(cfg, dict):
            return
        tr = cfg.get("training") or {}
        if not isinstance(tr, dict):
            tr = {}
        self._batch_row.set_value(tr.get("batch_size"))
        self._lr_row.set_value(tr.get("lr"))
        self._accum_row.set_value(tr.get("gradient_accumulation_steps", 1))
        self._epochs_row.set_value(tr.get("num_epochs"))
        opt = tr.get("optimizer")
        if opt and self._optim_row.combo.findData(opt) < 0:
            self._optim_row.combo.addItem(opt, opt)
        if opt:
            self._optim_row.set_key(opt)
        self._patience_row.set_value(tr.get("patience"))
        self._reduce_row.set_value(tr.get("reduce_factor"))
        aug = cfg.get("augmentations")
        if isinstance(aug, dict) and "enable" in aug:
            self._augment_row.set_on(bool(aug.get("enable")))
        self._refresh_gpu_warnings()

    def _refresh_loss_text(self):
        names = [k.replace("_", " ") for k in self._losses]
        text = " + ".join(names)
        if self._use_standard_loss:
            text += "  (forced for RoFormer models)"
        self._loss_row.set_text(text)

    def _refresh_metrics_text(self):
        self._metrics_row.set_text(", ".join(self._metrics))
        keep = self._sched_row.key() if hasattr(self, "_sched_row") else None
        if hasattr(self, "_sched_row"):
            self._sched_row.set_items([(m, m) for m in self._metrics],
                                      keep=keep if keep in self._metrics else (self._metrics[0] if self._metrics else None))
        if hasattr(self, "_tiles_grid"):
            self._rebuild_tiles()

    def _refresh_run_text(self):
        o = self._run_opts
        gpus = [g for g in list_gpus() if not g.startswith("CPU")]
        names = {}
        for g in gpus:
            try:
                idx = int(g.split("GPU")[1].split(":")[0].strip())
                names[idx] = g.split(":", 1)[1].strip().rsplit("(", 1)[0].strip()
            except Exception:
                pass
        if o.get("force_cpu") or not gpus:
            dev = "CPU"
        else:
            dev = ", ".join(names.get(i, f"GPU {i}") for i in o["device_ids"]) or "GPU 0"
        n = 1 if (o.get("force_cpu") or not gpus) else len(o["device_ids"])
        launcher = o.get("launcher") or "standard"
        extra = f" · {launcher}"
        if (o.get("lora_mode") or "off") != "off":
            extra += f" · lora {o['lora_mode']}"
        self._gpu_row.set_text(f"{dev}\n{n} / {o['num_workers']} / {o['seed']}{extra}")

    def _open_pretrained(self):
        self._ckpt_row.set_picker_active(True)   # keep the > chevron lit underneath
        dlg = PretrainedModelsDialog(self)
        dlg.use_for_training.connect(self._apply_pretrained)
        run_blurred_dialog(dlg)
        self._ckpt_row.set_picker_active(False)

    def _apply_pretrained(self, config_path, checkpoint_path):
        """Fill CONFIGURATION rows from a chosen pre-trained download."""
        self._config_row.set_value(config_path)
        self._ckpt_row.set_value(checkpoint_path)

    def _cuda_blockers(self):
        """Reasons the current TRAINING selection can't run without CUDA.

        Each GPU-only option surfaced in this tab (adamw8bit optimizer; add
        8-bit model loading or VRAM caps here as they gain UI) is checked
        live — the warning under the Optimizer row — and again when Start is
        pressed, so a CPU-only runtime fails with a readable message before a
        job spawns instead of mid-run.
        """
        o = getattr(self, "_run_opts", {})
        has_cuda = any(not g.startswith("CPU") for g in list_gpus())
        gpu_active = has_cuda and not o.get("force_cpu")
        reasons = []
        if self._optim_row.key() == "adamw8bit" and not gpu_active:
            why = "CPU-only is selected" if o.get("force_cpu") \
                else "no CUDA GPU is available"
            reasons.append(
                f"'adamw8bit' needs a CUDA GPU, but {why} — the run will be "
                "blocked at start. Choose 'adamw' or 'adam', or enable a GPU "
                "in GPUs / Workers / Seed."
            )
        return reasons

    def _refresh_gpu_warnings(self):
        reasons = self._cuda_blockers()
        self._optim_warn.setText("\n".join(reasons))
        self._optim_warn.setVisible(bool(reasons))
        if hasattr(self, "btn_fit"):
            self._refresh_fit_btn()

    def _parse_int_field(self, text):
        text = (text or "").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _current_advice(self):
        o = getattr(self, "_run_opts", {})
        cfg = self._read_yaml(self._config_row.value()) if self._config_row.value() else None
        cfg = cfg if isinstance(cfg, dict) else {}
        audio = cfg.get("audio") if isinstance(cfg.get("audio"), dict) else {}
        tr = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
        model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        bs = self._parse_int_field(self._batch_row.value())
        if bs is None:
            bs = self._parse_int_field(str(tr.get("batch_size") or ""))
        acc = None
        if hasattr(self, "_accum_row"):
            acc = self._parse_int_field(self._accum_row.value())
        if acc is None:
            acc = self._parse_int_field(str(tr.get("gradient_accumulation_steps") or ""))
        sr = audio.get("sample_rate") or 44100
        try:
            sr = int(sr)
        except (TypeError, ValueError):
            sr = 44100
        chunk = audio.get("chunk_size")
        if self._chunk_override is not None:
            chunk = self._chunk_override
        try:
            chunk = int(chunk) if chunk is not None else None
        except (TypeError, ValueError):
            chunk = None
        return advise_from_selection(
            model_type=self._model_row.key() or "",
            device_ids=o.get("device_ids") or [0],
            force_cpu=bool(o.get("force_cpu")),
            chunk_size=chunk,
            sample_rate=sr,
            current_batch=bs,
            current_accum=acc,
        ), model

    def _refresh_fit_btn(self):
        if not hasattr(self, "btn_fit"):
            return
        advice, _model = self._current_advice()
        o = getattr(self, "_run_opts", {})
        snap = None if o.get("force_cpu") else selected_gpu_snapshot(
            o.get("device_ids") or [0])
        if snap and snap.get("free_gb") is not None:
            lead = f"{advice.gpu_short} · {snap['free_gb']:.0f} GB free"
        else:
            lead = advice.gpu_short
        bits = [f"batch {advice.batch_size}", f"accum {advice.grad_accum}"]
        if advice.use_torch_checkpoint is True:
            bits.append("checkpoint on")
        elif advice.use_torch_checkpoint is False:
            bits.append("checkpoint off")
        self.btn_fit.setToolTip(
            "Live allocation probe: dummy train step on this GPU, "
            "then batch / accum / checkpoint from measured VRAM.\n"
            f"{lead} → {' · '.join(bits)}"
        )

    def _open_fit_gpu(self, *, blur=True):
        list_gpus(refresh=True)
        query_gpu_memory(refresh=True)
        advice, model = self._current_advice()
        cur_ckpt = None
        if "use_torch_checkpoint" in model:
            cur_ckpt = bool(model.get("use_torch_checkpoint"))
        elif self._checkpoint_override is not None:
            cur_ckpt = bool(self._checkpoint_override)
        o = getattr(self, "_run_opts", {})
        cfg_path = (self._config_row.value() or "").strip()
        snap = selected_gpu_snapshot(o.get("device_ids") or [0])
        device_id = int(snap["index"]) if snap else (o.get("device_ids") or [0])[0]
        probe_spec = {
            "model_type": self._model_row.key() or "",
            "config_path": cfg_path if cfg_path and os.path.isfile(cfg_path) else "",
            "device_id": device_id,
            "force_cpu": bool(o.get("force_cpu")),
            "current_batch": self._parse_int_field(self._batch_row.value()),
            "current_accum": self._parse_int_field(self._accum_row.value()),
            "custom_backend": detect_custom_backend(cfg_path) if cfg_path else "",
        }
        dlg = _FitGpuDialog(advice, {
            "model_type": self._model_row.key() or "",
            "batch": self._batch_row.value() or None,
            "accum": self._accum_row.value() or None,
            "checkpoint": cur_ckpt,
            "optimizer": self._optim_row.key(),
        }, self, probe_spec=probe_spec)
        result = run_blurred_dialog(dlg) if blur else dlg.exec()
        if result == QDialog.Accepted:
            self._apply_advice(dlg.values())
            return True
        return False

    def _open_setup(self):
        dlg = _TrainSetupDialog(self)
        if run_blurred_dialog(dlg) == QDialog.Accepted and dlg.start_train:
            self._run()

    def _apply_advice(self, advice):
        self._batch_row.set_value(advice.batch_size)
        self._accum_row.set_value(advice.grad_accum)
        if advice.use_torch_checkpoint is not None:
            self._checkpoint_override = bool(advice.use_torch_checkpoint)
        if advice.pin_memory:
            self._run_opts["pin_memory"] = True
        if advice.optimizer:
            if self._optim_row.combo.findData(advice.optimizer) < 0:
                self._optim_row.combo.addItem(advice.optimizer, advice.optimizer)
            self._optim_row.set_key(advice.optimizer)
        if advice.chunk_size:
            self._apply_chunk_size(int(advice.chunk_size))
        self._refresh_run_text()
        self._refresh_gpu_warnings()
        self._persist()

    def _apply_chunk_size(self, chunk):
        try:
            chunk = int(chunk)
        except (TypeError, ValueError):
            return
        if chunk <= 0:
            return
        self._chunk_override = chunk
        path = (self._config_row.value() or "").strip()
        if not path or not os.path.isfile(path):
            return
        cfg = self._read_yaml(path)
        if not isinstance(cfg, dict):
            return
        audio = cfg.get("audio")
        if not isinstance(audio, dict):
            audio = {}
            cfg["audio"] = audio
        if audio.get("chunk_size") == chunk:
            return
        audio["chunk_size"] = chunk
        try:
            import yaml
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)
        except OSError:
            pass

    def _open_run_options(self):
        dlg = _RunOptionsDialog(dict(self._run_opts), self)
        if run_blurred_dialog(dlg) == QDialog.Accepted:
            self._run_opts.update(dlg.values())
            self._refresh_run_text()
            self._refresh_gpu_warnings()
            self._persist()

    def _open_loss_dialog(self):
        dlg = _MultiSelectDialog(
            "LOSS FUNCTIONS", catalog.loss_choices(), self._losses,
            hint="Default: multi l1 snr db + fullness penalty — a solid vocal-remover start.\n"
                 "Selected losses are summed (each with its default coefficient).\n"
                 "The fullness / bleedless penalties are meant to accompany a primary loss.\n"
                 "RoFormer fullness-only training (multistft) is a special case — see upstream docs.",
            caption="The job uses these losses on Train.",
            toggle=("Use these losses for RoFormer / Conformer models too (--use_standard_loss)",
                    self._use_standard_loss,
                    "Without this, bs_roformer / mel_band_roformer / conformer models train with "
                    "their built-in loss and the selection above is ignored."),
            parent=self)
        if run_blurred_dialog(dlg) == QDialog.Accepted:
            sel = dlg.selected()
            self._losses = sel or catalog.default_losses()
            self._use_standard_loss = dlg.toggle_value()
            self._refresh_loss_text()

    def _open_metrics_dialog(self):
        dlg = _MultiSelectDialog(
            "VALIDATION METRICS", catalog.metric_choices(), self._metrics,
            hint="Computed on the validation set after every epoch.\n"
                 "The scheduler metric is added automatically if it is not selected.\n"
                 "Metrics are for quick comparison — always listen to the separated audio.\n"
                 "Bleedless and fullness often trade off (one up, the other down).",
            caption="The job reports these metrics after every epoch.",
            parent=self)
        if run_blurred_dialog(dlg) == QDialog.Accepted:
            sel = dlg.selected()
            self._metrics = sel or catalog.default_metrics()
            self._refresh_metrics_text()

    def _open_config_editor(self):
        path = self._config_row.value()
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "No config", "Select a config file first.")
            return
        dlg = _ConfigEditorDialog(path, self)
        dlg.saved.connect(lambda p: self._load_fields_from_yaml(p))
        run_blurred_dialog(dlg)

    def _newest_ckpt(self):
        results = self._results_row.value()
        if not results or not os.path.isdir(results):
            return None
        files = []
        for ext in ("*.ckpt", "*.pth", "*.pt", "*.bin", "*.th"):
            files.extend(glob.glob(os.path.join(results, ext)))
        if not files:
            return None
        return max(files, key=lambda p: os.path.getmtime(p))

    def _pick_latest_ckpt(self):
        newest = self._newest_ckpt()
        if newest is None:
            QMessageBox.information(self, "No checkpoint",
                                    "No checkpoint found in the results folder yet.")
            return
        self._ckpt_row.set_value(newest)

    def _on_ckpt_changed(self):
        """Picking a checkpoint arms the resume-continuity run flags
        (load optimizer/scheduler/epoch/best-metric), so a resume run
        continues from the saved state instead of restarting at epoch 0.
        Each flag is only consumed when the checkpoint actually carries that
        piece, so state-only pre-trained checkpoints are unaffected; uncheck
        any switch in Run Options to opt out for a specific run."""
        if self._ckpt_row.value():
            for key in ("load_optimizer", "load_scheduler",
                        "load_epoch", "load_best_metric"):
                self._run_opts[key] = True
        self._refresh_latest_chip()
        self._refresh_ckpt_clear()

    def _clear_ckpt(self):
        """Drop the resume checkpoint so the next run trains from scratch
        (the run flags armed for resuming are left as-is — they are
        harmless no-ops without --start_check_point, and unchecking them is
        one switch in Run Options)."""
        if not self._ckpt_row.value():
            return
        self._ckpt_row.set_paths([])  # emits changed -> _on_ckpt_changed

    def _refresh_ckpt_clear(self):
        """Show the ✕ clear chip only while a checkpoint is actually set."""
        self._clear_ckpt_btn.setVisible(bool(self._ckpt_row.value()))

    def _refresh_latest_chip(self):
        newest = self._newest_ckpt()
        cur = self._ckpt_row.value()
        active = bool(newest) and bool(cur) and os.path.normcase(os.path.abspath(newest)) == \
            os.path.normcase(os.path.abspath(cur))
        self._latest_chip.setStyleSheet(_chip_ss(active))

    # ── Persistence ──────────────────────────────────────────────────────
    def save_settings(self):
        return {
            "model_type": self._model_row.key(),
            "config_path": self._config_row.value(),
            "results_path": self._results_row.value(),
            "data_paths": self._data_row.paths(),
            "valid_paths": self._valid_row.paths(),
            "dataset_type": self._dataset_row.key(),
            "augment": self._augment_row.is_on(),
            "run_options": dict(self._run_opts),
            "checkpoint": self._ckpt_row.value(),
            "batch_size": self._batch_row.value(),
            "lr": self._lr_row.value(),
            "grad_accum": self._accum_row.value(),
            "epochs": self._epochs_row.value(),
            "optimizer": self._optim_row.key(),
            "loss": list(self._losses),
            "use_standard_loss": self._use_standard_loss,
            "metrics": list(self._metrics),
            "metric_for_scheduler": self._sched_row.key(),
            "patience": self._patience_row.value(),
            "reduce_factor": self._reduce_row.value(),
            "checkpoint_override": self._checkpoint_override,
            "chunk_override": self._chunk_override,
        }

    def load_settings(self, d):
        d = d or {}
        if d.get("model_type"):
            self._model_row.set_key(d["model_type"])
        # Config path first, without re-reading the YAML: the saved field
        # values below win over the file (they may be edited overrides).
        self._config_row.blockSignals(True)
        self._config_row.set_value(d.get("config_path", ""))
        self._config_row.blockSignals(False)
        self._results_row.set_value(d.get("results_path", ""))
        self._data_row.set_paths(d.get("data_paths") or [])
        self._valid_row.set_paths(d.get("valid_paths") or [])
        if d.get("dataset_type"):
            self._dataset_row.set_key(str(d["dataset_type"]))
        self._augment_row.set_on(d.get("augment", True))
        ro = d.get("run_options") or {}
        if isinstance(ro, dict):
            self._run_opts.update({k: v for k, v in ro.items() if k in self._run_opts})
        self._ckpt_row.set_value(d.get("checkpoint", ""))
        self._refresh_ckpt_clear()
        has_fields = any(d.get(k) for k in ("batch_size", "lr", "epochs"))
        if has_fields:
            for key, row in (("batch_size", self._batch_row), ("lr", self._lr_row),
                             ("grad_accum", self._accum_row), ("epochs", self._epochs_row),
                             ("patience", self._patience_row), ("reduce_factor", self._reduce_row)):
                row.set_value(d.get(key, ""))
            if d.get("optimizer"):
                if self._optim_row.combo.findData(d["optimizer"]) < 0:
                    self._optim_row.combo.addItem(d["optimizer"], d["optimizer"])
                self._optim_row.set_key(d["optimizer"])
        elif self._config_row.value() and os.path.isfile(self._config_row.value()):
            self._load_fields_from_yaml(self._config_row.value())
        if d.get("loss"):
            self._losses = [l for l in d["loss"] if l in catalog.losses()] or catalog.default_losses()
        self._use_standard_loss = bool(d.get("use_standard_loss", True))
        if d.get("metrics"):
            self._metrics = [m for m in d["metrics"] if m in catalog.metrics()] or catalog.default_metrics()
        self._refresh_loss_text()
        self._refresh_metrics_text()
        if d.get("metric_for_scheduler"):
            self._sched_row.set_key(d["metric_for_scheduler"])
        if "checkpoint_override" in d:
            self._checkpoint_override = d.get("checkpoint_override")
        if "chunk_override" in d:
            self._chunk_override = d.get("chunk_override")
        self._refresh_run_text()
        self._refresh_latest_chip()
        self._refresh_gpu_warnings()

    def _persist(self):
        try:
            data = settings_store.load()
            data["training"] = self.save_settings()
            settings_store.save(data)
        except Exception as exc:
            print(f"[training] could not persist settings: {exc}")

    # ── Monitor state ────────────────────────────────────────────────────
    def _reset_monitor(self):
        self._phase = "idle"
        self._epoch = None
        self._lr = None
        self._epochs_total = None
        self._metric_vals = {}
        self._summary = {}
        self._bar.setValue(0)
        self._prog_lbl.setText("0 / 0 (0.00%)")
        self._render_stats(None, None, None, None)
        for tile in self._tiles.values():
            tile.set_value(None)
        if hasattr(self, "_summary_lbl"):
            self._render_summary()

    def _rebuild_tiles(self):
        for tile in self._tiles.values():
            tile.setParent(None)
            tile.deleteLater()
        self._tiles = {}
        names = list(self._metrics)
        if "sdr" in names and "k_sdr" not in names:
            names.insert(names.index("sdr"), "k_sdr")  # valid.py reports both
        for name in list(self._metric_vals.keys()):
            if name not in names:
                names.append(name)
        for name in names:
            self._tiles[name] = _MetricTile(name)
        self._tiles_grid.set_tiles(self._tiles.values())
        self._push_metric_values()

    def _push_metric_values(self):
        for name, per_instr in self._metric_vals.items():
            tile = self._tiles.get(name)
            if tile is None:
                continue
            vals = [v for v in per_instr.values() if v is not None]
            if not vals:
                tile.set_value(None)
                continue
            avg = per_instr.get("__avg__")
            if avg is None:
                avg = sum(vals) / len(vals)
            tip = "\n".join(f"{k}: {v:.4f}" for k, v in per_instr.items() if k != "__avg__")
            tile.set_value(avg, tip)

    def _render_stats(self, bar_text, epoch, lr, extra=None):
        t = theme_manager.theme
        esc = html.escape
        lines = []
        if bar_text:
            # loss=… / avg_loss=… picked out in the warning tint, like the
            # learning-rate line below
            txt = esc(bar_text)
            txt = re.sub(r"(loss=)([-+\d.eE]+)",
                         lambda m: f'{m.group(1)}<span style="color:{t.warning};">{m.group(2)}</span>',
                         txt)
            lines.append(txt)
        else:
            lines.append(f'<span style="color:{t.text_muted};">waiting for the first step…</span>')
        ep = "—" if epoch is None else str(epoch)
        tot = f" / {self._epochs_total}" if self._epochs_total else ""
        lines.append(f'Train epoch: <span style="color:{t.success};">{esc(ep)}{esc(tot)}</span>')
        lines.append(f'Learning rate: <span style="color:{t.warning};">{esc("—" if lr is None else lr)}</span>')
        if extra:
            lines.append(esc(extra))
        self._stat_box.setText("<br>".join(lines))

    def _render_summary(self):
        s = self._summary
        t = theme_manager.theme

        def row(label, key):
            v = s.get(key)
            if v is None:
                return f'{html.escape(label)}: <span style="color:{t.text_muted};">—</span>'
            return f'{html.escape(label)}: {html.escape(str(v))}'

        self._summary_lbl.setText("<br>".join([
            row("Model", "Model"),
            row("Total parameters", "Total parameters"),
            row("Trainable parameters", "Trainable parameters"),
            row("Model size", "Model size"),
            row("Number of layers", "Number of layers"),
        ]))

    # ── Log ──────────────────────────────────────────────────────────────
    def _restore_log(self):
        if self._LOG_HISTORY:
            self._log.setPlainText("\n".join(self._LOG_HISTORY))
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _append_log(self, line, to_file=True):
        self._LOG_HISTORY.append(line)
        if len(self._LOG_HISTORY) > self._LOG_HISTORY_MAX:
            del self._LOG_HISTORY[:len(self._LOG_HISTORY) - self._LOG_HISTORY_MAX]
        self._log.appendPlainText(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
        if to_file and self._log_file:
            try:
                self._log_file.write(line + "\n")
            except Exception:
                pass

    def _open_log_file(self, results):
        try:
            self._log_file = open(os.path.join(results, "train_log.txt"), "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            self._log_file = None
            self._append_log(f"[WARN] could not open the log file: {exc}", to_file=False)

    def _close_log_file(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    # ── Output handling ──────────────────────────────────────────────────
    def _on_line(self, line):
        if _is_progress_line(line):
            self._update_progress(line)
            return
        self._append_log(line)
        self.log_output.emit(line)

        m = _EPOCH_RE.search(line)
        if m:
            self._phase = "train"
            self._epoch = int(m.group(1))
            self._lr = m.group(2)
            self._render_stats(None, self._epoch, self._lr)
            return
        m = _TRAIN_FOR_RE.search(line)
        if m:
            self._epochs_total = int(m.group(1))
            self._render_stats(None, self._epoch, self._lr)
            return
        if line.startswith("Training loss:"):
            self._phase = "valid"
            self._render_stats(None, self._epoch, self._lr, "Validating…")
            return
        m = _INSTR_RE.match(line)
        if m:
            instr, metric, val = m.group(1), m.group(2), _to_float(m.group(3))
            d = self._metric_vals.setdefault(metric, {})
            d[instr] = val
            d.pop("__avg__", None)
            if metric not in self._tiles:
                self._rebuild_tiles()
            else:
                self._push_metric_values()
            return
        m = _AVG_METRIC_RE.match(line) or _METRIC_VAL_RE.match(line)
        if m:
            metric, val = m.group(1), _to_float(m.group(2))
            d = self._metric_vals.setdefault(metric, {})
            d["__avg__"] = val
            if not any(k != "__avg__" for k in d):
                d["avg"] = val
            if metric not in self._tiles:
                self._rebuild_tiles()
            else:
                self._push_metric_values()
            return
        for key in _SUMMARY_KEYS:
            if line.startswith(key):
                self._summary[key.rstrip(":")] = line[len(key):].strip()
                self._render_summary()
                return

    def _update_progress(self, line):
        m = _TQDM_RE.search(line)
        if not m:
            return
        step, total = int(m.group(1)), int(m.group(2))
        inner = m.group(3)
        pct = (100.0 * step / total) if total else 0.0
        self._bar.setValue(int(pct))
        label = f"{step} / {total} ({pct:.2f}%)"
        if self._phase == "valid":
            label = "validating  " + label
        self._prog_lbl.setText(label)
        bar_text = f"[{step}/{total}] [{inner}]"
        self._render_stats(bar_text, self._epoch, self._lr)

    # ── Run / stop ───────────────────────────────────────────────────────
    def _validate(self):
        cfg = self._config_row.value()
        if not cfg or not os.path.isfile(cfg):
            return "Please select a model config (YAML)."
        if not self._results_row.value():
            return "Please select a results folder."
        if not self._data_row.paths():
            return "Please select the training data folder."
        if not self._valid_row.paths():
            return "Please select the validation folder."
        if not self._model_row.key():
            return "Please pick a model type."
        return None

    def _build_config(self):
        """The YAML actually used for the run: the selected config with the
        TRAINING SETTINGS fields applied on top."""
        cfg = self._read_yaml(self._config_row.value())
        if not isinstance(cfg, dict):
            raise ValueError("The config file is not a YAML mapping.")
        tr = cfg.setdefault("training", {})
        if not isinstance(tr, dict):
            raise ValueError("config.training is not a mapping.")

        def put(key, text, conv):
            text = (text or "").strip()
            if not text:
                return
            try:
                tr[key] = conv(text)
            except ValueError:
                raise ValueError(f"'{text}' is not a valid value for {key}.")

        put("batch_size", self._batch_row.value(), int)
        put("lr", self._lr_row.value(), float)
        put("gradient_accumulation_steps", self._accum_row.value(), int)
        put("num_epochs", self._epochs_row.value(), int)
        put("patience", self._patience_row.value(), int)
        put("reduce_factor", self._reduce_row.value(), float)
        if self._optim_row.key():
            tr["optimizer"] = self._optim_row.key()
        aug = cfg.get("augmentations")
        if isinstance(aug, dict):
            aug["enable"] = bool(self._augment_row.is_on())
        elif self._augment_row.is_on() is False:
            cfg["augmentations"] = {"enable": False}
        if (self._run_opts.get("lora_mode") or "off") != "off":
            inject_lora_defaults(cfg)
        ckpt = self._checkpoint_override
        model_type = self._model_row.key()
        if ckpt is None and model_type in HEAVY_CONFORMER_TYPES:
            mem = selected_gpu_memory_gb(self._run_opts.get("device_ids") or [0])
            if mem is not None and mem < 40:
                ckpt = True
        if ckpt is not None and supports_torch_checkpoint(model_type):
            model = cfg.setdefault("model", {})
            if isinstance(model, dict):
                model["use_torch_checkpoint"] = bool(ckpt)
        chunk = self._chunk_override
        if chunk is not None:
            audio = cfg.setdefault("audio", {})
            if isinstance(audio, dict):
                try:
                    audio["chunk_size"] = int(chunk)
                except (TypeError, ValueError):
                    pass
        return cfg

    def _run(self):
        try:
            self._run_inner()
        except Exception as exc:
            import traceback as _tb
            self._append_log(f"ERROR: {exc}")
            for ln in _tb.format_exc().splitlines():
                self._append_log(ln)
            self.process_running.emit(False)
            self.btn_run.setEnabled(True)
            self.btn_setup.setEnabled(True)
            self.btn_fit.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def _run_inner(self):
        from ui.widgets.runtime_dialog import ensure_runtime
        if not ensure_runtime(self):
            return
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Missing input", err)
            return
        import yaml
        try:
            cfg = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid setting", str(exc))
            return

        model_type = self._model_row.key()
        results = self._results_path()
        os.makedirs(results, exist_ok=True)
        # The effective config (YAML + the tab's overrides) goes into the
        # results folder under the config's own file name — one config per
        # run, next to the checkpoints it produced. When the config already
        # lives there it is simply updated in place.
        cfg_path = os.path.join(results, os.path.basename(self._config_row.value()))
        with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        o = self._run_opts
        blockers = self._cuda_blockers()
        if blockers:
            QMessageBox.warning(
                self, "GPU requirements not met",
                "This run can't start:\n\n• " + "\n• ".join(blockers))
            return
        metrics = list(self._metrics)
        sched = self._sched_row.key() or metrics[0]
        if sched not in metrics:
            metrics.append(sched)
        custom_backend = detect_custom_backend(self._config_row.value())
        launcher, warn = resolve_launcher(
            o, has_custom_backend=bool(custom_backend),
            accelerate_ok=accelerate_available(get_python_exe()))
        if warn:
            self._append_log(f"[LAUNCHER] {warn}")
            QMessageBox.information(self, "Launcher", warn)
        cmd = build_train_command(
            python_exe=get_python_exe(),
            opts=o,
            model_type=model_type,
            config_path=cfg_path,
            results_path=results,
            data_paths=self._data_row.paths(),
            valid_paths=self._valid_row.paths(),
            dataset_type=self._dataset_row.key() or "1",
            metrics=metrics,
            metric_for_scheduler=sched,
            losses=self._losses,
            checkpoint=self._ckpt_row.value(),
            custom_backend=custom_backend,
            use_standard_loss=self._use_standard_loss,
            launcher=launcher,
        )
        env = subprocess_env(o, launcher)

        self._persist()
        self._reset_monitor()
        self._open_log_file(results)
        self._append_log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting training")
        shown = redact_train_cmd(cmd)
        self._append_log("> " + " ".join(f'"{c}"' if " " in c else c for c in shown))
        self._append_log(f"Config written to: {cfg_path}")

        self._stop_requested = False
        self._runner = ProcessRunner(cmd, cwd=REPO_ROOT, env=env)
        self._runner.log_line.connect(self._on_line)
        self._runner.finished.connect(self._on_finished)
        self.process_running.emit(True)
        self._runner.start()
        self.btn_run.setEnabled(False)
        self.btn_setup.setEnabled(False)
        self.btn_fit.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self):
        if self._runner:
            self._stop_requested = True
            self._append_log("[STOP] Stopping training…")
            self.btn_stop.setEnabled(False)
            self._runner.stop()

    def _on_finished(self, code):
        self.btn_run.setEnabled(True)
        self.btn_setup.setEnabled(True)
        self.btn_fit.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._stop_requested:
            self._append_log("Training stopped")
        elif code == 0:
            self._append_log("Completed: training finished")
        else:
            self._append_log(f"ERROR: training exited with code {code}")
        self._stop_requested = False
        self._phase = "idle"
        self._close_log_file()
        self._runner = None
        self._refresh_latest_chip()
        self.process_running.emit(False)

    # ── Bottom-right buttons ─────────────────────────────────────────────
    def _results_path(self):
        return self._results_row.value()

    def _export_weights(self):
        """Run upstream's scripts/prepare_weights_for_inference.py on a
        checkpoint: keeps only model_state_dict (optionally as float16), so
        the file registered for inference is a fraction of the training
        checkpoint (which also carries optimizer / scheduler state and the
        full metrics + loss history)."""
        from ui.widgets.runtime_dialog import ensure_runtime
        if getattr(self, "_export_runner", None) is not None:
            QMessageBox.information(self, "Export running", "An export is already running.")
            return
        if not ensure_runtime(self):
            return
        default = self._ckpt_row.value() or self._newest_ckpt() or ""
        dlg = _ExportWeightsDialog(default, self._results_path(), self)
        if run_blurred_dialog(dlg) != QDialog.Accepted:
            return
        src, dst, fp16 = dlg.values()
        script = os.path.join(REPO_ROOT, "scripts", "prepare_weights_for_inference.py")
        cmd = [get_python_exe(), script, "--checkpoint", src, "--output_file", dst]
        if fp16:
            cmd.append("--float16")
        self._append_log(f"[EXPORT] {os.path.basename(src)} -> {dst}"
                         + (" (float16)" if fp16 else ""), to_file=False)
        self._export_dst = dst
        self._export_src = src
        self._export_runner = ProcessRunner(cmd, cwd=REPO_ROOT)
        self._export_runner.log_line.connect(
            lambda ln: self._append_log("[EXPORT] " + ln, to_file=False))
        self._export_runner.finished.connect(self._on_export_finished)
        self.btn_export.setEnabled(False)
        self._export_runner.start()

    def _on_export_finished(self, code):
        self.btn_export.setEnabled(True)
        self._export_runner = None
        dst = getattr(self, "_export_dst", "")
        if code == 0 and dst and os.path.isfile(dst):
            try:
                before = os.path.getsize(self._export_src) / (1024 * 1024)
                after = os.path.getsize(dst) / (1024 * 1024)
                self._append_log(f"[EXPORT] Done: {dst}  ({before:.0f} MB -> {after:.0f} MB)",
                                 to_file=False)
            except OSError:
                self._append_log(f"[EXPORT] Done: {dst}", to_file=False)
            self._refresh_latest_chip()
        else:
            self._append_log(f"[EXPORT] ERROR: export exited with code {code}", to_file=False)

    def _open_wandb(self, which="home"):
        if which not in ("home", "key"):
            which = "home"
        url = WANDB_AUTHORIZE if which == "key" else WANDB_HOME
        QDesktopServices.openUrl(QUrl(url))

    def _open_results(self):
        results = self._results_row.value()
        if not results or not os.path.isdir(results):
            QMessageBox.information(self, "No results folder",
                                    "Select a results folder first (it is created when training starts).")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(results))

    def _open_logs(self):
        results = self._results_row.value()
        log_path = os.path.join(results, "train_log.txt") if results else ""
        if log_path and os.path.isfile(log_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(log_path))
        elif results and os.path.isdir(results):
            QDesktopServices.openUrl(QUrl.fromLocalFile(results))
        else:
            QMessageBox.information(self, "No log yet",
                                    "The training log is written to train_log.txt in the results folder.")
