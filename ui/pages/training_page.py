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
import os
import re
import glob
import html
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QFileDialog, QScrollArea, QSizePolicy,
    QDialog, QProgressBar, QSpinBox, QMessageBox, QPlainTextEdit, QComboBox,
)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QRect, QUrl, QSize
from PySide6.QtGui import (
    QPainter, QPen, QColor, QFont, QFontMetrics, QDesktopServices,
    QIntValidator, QPainterPath,
)

from backend.runner import ProcessRunner
from backend.paths import REPO_ROOT, TRAIN_SCRIPT, get_python_exe
from backend.gpu_utils import list_gpus
from backend import settings as settings_store
from ui.theme import theme_manager, UIConstants
from ui.widgets.common import (
    PageHeader, solid_button_ss, EllipsisButton, GlyphButton, css_color,
    _solid_icon_color, _stop_icon_color,
)
from ui.pages.inference_page import (
    _sec_hdr, _row_ss, _lbl_ss, _combo_ss, _ComboBox, _ExpandArrow,
    _InfoDot, _CircleCheck, _MiniSwitch, ROW_H,
)
from ui.widgets.ckpt_settings_dialog import _TitleBar


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

LABEL_W = 170
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
    return (
        "QSpinBox{"
        "font-family:'Courier New',monospace;font-size:13px;font-weight:bold;"
        f"color:{theme_manager.accent};background:{t.input_bg};"
        f"border:1px solid {t.border_dim};border-radius:4px;padding:2px 6px;}}"
        "QSpinBox::up-button, QSpinBox::down-button{width:0px;border:none;}"
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


def _cancel_btn_ss():
    t = theme_manager.theme
    e = QColor(t.error)
    return (
        "QPushButton{"
        f"background:{t.surface};color:{t.text_muted};"
        f"border:1px solid {t.border_dim};border-radius:4px;"
        "font-family:'Montserrat',sans-serif;font-weight:600;font-size:10px;}"
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
    like RESUME CHECKPOINT need the room) with the ⓘ dot right after the
    text. A label that cannot fit on one line wraps inside the row. The
    width is measured explicitly: a word-wrapped QLabel's own sizeHint uses
    a narrow aspect-ratio heuristic and would wrap MODEL TYPE too."""
    wrap = QWidget()
    wrap.setStyleSheet("background:transparent;")
    wrap.setFixedWidth(LABEL_W)
    hl = QHBoxLayout(wrap)
    hl.setContentsMargins(0, 0, 8, 0)
    hl.setSpacing(5)
    lb = QLabel(label.upper())
    lb.setStyleSheet(_lbl_ss())
    lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    max_w = LABEL_W - 8 - (19 if tooltip else 0)
    natural = QFontMetrics(_label_font()).horizontalAdvance(label.upper()) + 2
    if natural <= max_w:
        lb.setFixedWidth(natural)
    else:
        lb.setWordWrap(True)
        lb.setFixedWidth(max_w)
    hl.addWidget(lb, 0, Qt.AlignVCenter)
    if tooltip:
        hl.addWidget(_InfoDot(tooltip), 0, Qt.AlignVCenter)
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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

    def _fitted(self, fm, w):
        s = self._text
        if not s:
            return self._placeholder
        avail = self.MAX_LINES * fm.height() + 1
        flags = Qt.TextWrapAnywhere | Qt.AlignLeft
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
        s = self._fitted(QFontMetrics(f), self.width())
        p.drawText(self.rect(), Qt.TextWrapAnywhere | Qt.AlignLeft | Qt.AlignVCenter, s)
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

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._color = QColor("#FFFFFF")
        self.setFixedSize(16, 16)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_color(self, c):
        self._color = css_color(c)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
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
    """Small outline button with a painted leading icon (Open Results Folder /
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
        self._val = _WrapLabel(placeholder)
        hl.addWidget(self._val, 1)
        self._btn = EllipsisButton()
        self._btn.clicked.connect(self._browse)
        hl.addWidget(self._btn)

    def add_extra(self, w):
        """Dock a widget (e.g. the LATEST chip) between the value and '···'."""
        hl = self.layout()
        hl.insertSpacing(2, 8)
        hl.insertWidget(3, w, 0, Qt.AlignVCenter)
        hl.insertSpacing(4, 4)

    # — values —
    def paths(self):
        return list(self._paths)

    def value(self):
        return self._paths[0] if self._paths else ""

    def set_paths(self, paths):
        self._paths = [p for p in (paths or []) if p]
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
            hl.addSpacing(6)
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


def _option_row(check, title, desc=""):
    """check widget + title (+ dim description) for the dialogs' lists."""
    t = theme_manager.theme
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    hl = QHBoxLayout(w)
    hl.setContentsMargins(2, 3, 2, 3)
    hl.setSpacing(10)
    hl.addWidget(check, 0, Qt.AlignTop if desc else Qt.AlignVCenter)
    col = QVBoxLayout()
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(1)
    tl = QLabel(title)
    tl.setStyleSheet(f"font-family:'Montserrat';font-size:11px;color:{t.text};background:transparent;")
    col.addWidget(tl)
    if desc:
        dl = QLabel(desc)
        dl.setWordWrap(True)
        dl.setStyleSheet(f"font-family:'Montserrat';font-size:9px;color:{t.text_muted};background:transparent;")
        col.addWidget(dl)
    hl.addLayout(col, 1)
    w.setCursor(Qt.PointingHandCursor)
    w.mousePressEvent = lambda e, c=check: c.toggle() if hasattr(c, "toggle") else c.set_checked(not c.is_checked())
    return w


class _MultiSelectDialog(_DialogBase):
    """Pick several entries from a catalogue (losses / metrics)."""

    def __init__(self, title, options, selected, hint="", toggle=None, parent=None):
        super().__init__(title, parent, size=(480, 520))
        self._options = options
        self._checks = {}
        self._toggle_sw = None
        if hint:
            self.hint(hint)
        sc = self._scroll()
        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(0, 0, 6, 0)
        vl.setSpacing(2)
        sel = set(selected or [])
        for key, label, desc in options:
            cb = _CircleCheck()
            cb.set_checked(key in sel)
            self._checks[key] = cb
            vl.addWidget(_option_row(cb, f"{label}  ·  {key}", desc))
        vl.addStretch()
        sc.setWidget(inner)
        self.body.addWidget(sc, 1)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        for text, fn in (("Select All", lambda: self._set_all(True)),
                         ("Clear", lambda: self._set_all(False))):
            b = QPushButton(text)
            b.setFixedHeight(26)
            b.setStyleSheet(_small_button_ss())
            b.clicked.connect(fn)
            actions.addWidget(b)
        actions.addStretch()
        self.body.addLayout(actions)

        if toggle is not None:
            label, checked, tip = toggle
            row = QHBoxLayout()
            row.setContentsMargins(2, 2, 2, 0)
            row.setSpacing(10)
            self._toggle_sw = _MiniSwitch(checked)
            row.addWidget(self._toggle_sw)
            tl = QLabel(label)
            tl.setToolTip(tip)
            tl.setStyleSheet(f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;")
            row.addWidget(tl, 1)
            self.body.addLayout(row)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{theme_manager.theme.border};border:none;")
        self.body.addWidget(sep)
        self.add_buttons("Apply", self.accept)

    def _set_all(self, on):
        for cb in self._checks.values():
            cb.set_checked(on)

    def selected(self):
        return [k for k, _l, _d in self._options if self._checks[k].is_checked()]

    def toggle_value(self):
        return self._toggle_sw.is_checked() if self._toggle_sw is not None else False


class _RunOptionsDialog(_DialogBase):
    """GPUs / workers / seed plus the run flags train.py accepts."""

    def __init__(self, state, parent=None):
        super().__init__("GPUS / WORKERS / SEED", parent, size=(500, 600))
        t = theme_manager.theme
        self._gpu_checks = {}
        self._switches = {}

        sc = self._scroll()
        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(0, 0, 6, 0)
        vl.setSpacing(6)

        def section(text):
            lb = QLabel(text.upper())
            lb.setStyleSheet(_card_title_ss())
            vl.addSpacing(4)
            vl.addWidget(lb)

        section("Devices")
        gpus = [g for g in list_gpus() if not g.startswith("CPU")]
        if not gpus:
            self.hint_lbl = QLabel("No CUDA GPU detected — training runs on the CPU (very slow).")
            self.hint_lbl.setWordWrap(True)
            self.hint_lbl.setStyleSheet(f"font-family:'Montserrat';font-size:10px;color:{t.warning};background:transparent;")
            vl.addWidget(self.hint_lbl)
        selected_ids = set(state.get("device_ids") or [0])
        for label in gpus:
            try:
                idx = int(label.split("GPU")[1].split(":")[0].strip())
            except Exception:
                continue
            cb = _CircleCheck()
            cb.set_checked(idx in selected_ids)
            self._gpu_checks[idx] = cb
            vl.addWidget(_option_row(cb, label, "DataParallel across every checked GPU" if len(gpus) > 1 else ""))
        if gpus:
            cpu = _CircleCheck()
            cpu.set_checked(state.get("force_cpu", False))
            self._gpu_checks["cpu"] = cpu
            vl.addWidget(_option_row(cpu, "CPU only", "Hide the GPUs from the job (CUDA_VISIBLE_DEVICES=\"\")."))

        section("Data loading")
        grid = QGridLayout()
        grid.setContentsMargins(2, 0, 2, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        def spin(lo, hi, val):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(int(val))
            s.setFixedSize(110, 30)
            s.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            s.setStyleSheet(_spin_ss())
            return s

        def grid_label(text, tip):
            lb = QLabel(text.upper())
            lb.setStyleSheet(_lbl_ss())
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

        def switch_row(key, label, tip, default=False):
            row = QHBoxLayout()
            row.setContentsMargins(2, 0, 2, 0)
            row.setSpacing(10)
            sw = _MiniSwitch(bool(state.get(key, default)))
            self._switches[key] = sw
            row.addWidget(sw)
            lb = QLabel(label)
            lb.setToolTip(tip)
            lb.setStyleSheet(f"font-family:'Montserrat';font-size:11px;color:{t.text};background:transparent;")
            lb.setCursor(Qt.PointingHandCursor)
            lb.mousePressEvent = lambda e, s=sw: s.set_checked(not s.is_checked())
            row.addWidget(lb, 1)
            vl.addLayout(row)

        switch_row("pin_memory", "Pin memory", "--pin_memory: faster host→GPU copies.")
        section("Run options")
        switch_row("pre_valid", "Validate before training", "--pre_valid: run validation once before the first epoch.")
        switch_row("save_every_epoch", "Save weights every epoch", "--save_weights_every_epoch: keep a checkpoint per epoch (with all metrics in the name).")
        switch_row("each_metrics_in_name", "Per-stem metrics in checkpoint names", "--each_metrics_in_name.")
        section("When resuming from a checkpoint")
        switch_row("load_optimizer", "Load optimizer state", "--load_optimizer")
        switch_row("load_scheduler", "Load scheduler state", "--load_scheduler")
        switch_row("load_epoch", "Continue epoch numbering", "--load_epoch")
        switch_row("load_best_metric", "Load best metric so far", "--load_best_metric")
        vl.addStretch()
        sc.setWidget(inner)
        self.body.addWidget(sc, 1)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{t.border};border:none;")
        self.body.addWidget(sep)
        self.add_buttons("Apply", self.accept)

    def values(self):
        ids = sorted(i for i, cb in self._gpu_checks.items() if i != "cpu" and cb.is_checked())
        cpu = self._gpu_checks.get("cpu")
        out = {
            "device_ids": ids or [0],
            "force_cpu": bool(cpu.is_checked()) if cpu is not None else False,
            "num_workers": self._workers.value(),
            "seed": self._seed.value(),
        }
        for k, sw in self._switches.items():
            out[k] = sw.is_checked()
        return out


class _ConfigEditorDialog(_DialogBase):
    """Raw YAML editor for the selected config (validated before saving)."""
    saved = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__("EDIT CONFIGURATION", parent, size=(780, 620))
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
        self.body.addWidget(self._edit, 1)
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
        self._fp16 = _MiniSwitch(False)
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
        self._log_file = None
        self._run_opts = {
            "device_ids": [0], "force_cpu": False, "num_workers": 4, "seed": 0,
            "pin_memory": False, "pre_valid": False, "save_every_epoch": False,
            "each_metrics_in_name": False, "load_optimizer": False,
            "load_scheduler": False, "load_epoch": False, "load_best_metric": False,
        }
        self._losses = catalog.default_losses()
        self._use_standard_loss = False
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
        ))
        root.addWidget(header_w)

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
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
        )
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
        w, ll = self._column((32, 32, 10, 16))
        ll.addWidget(_sec_hdr("Configuration"))
        ll.addSpacing(16)
        cfg = QVBoxLayout()
        cfg.setSpacing(6)
        cfg.setContentsMargins(0, 0, 0, 0)

        self._model_row = _ComboRowT(
            "Model type", catalog.model_type_choices(),
            tooltip="Architecture to train (--model_type).\n"
                    "Picked automatically from the config file name when possible.")
        cfg.addWidget(self._model_row)

        self._config_row = _PathRow(
            "Config path", "Select or drop a model YAML config…",
            mode="file", file_filter="YAML config (*.yaml *.yml);;All files (*.*)",
            tooltip="Model / training YAML (--config_path).\n"
                    "The TRAINING SETTINGS column is filled from it.",
            start_dir=os.path.join(REPO_ROOT, "configs"))
        self._config_row.changed.connect(self._on_config_changed)
        cfg.addWidget(self._config_row)

        self._results_row = _PathRow(
            "Results path", "Select or drop the results folder…", mode="folder",
            tooltip="Where checkpoints, metadata cache and the training log go\n(--results_path).")
        self._results_row.changed.connect(self._refresh_latest_chip)
        cfg.addWidget(self._results_row)

        self._data_row = _PathRow(
            "Data path", "Select or drop the training dataset folder(s)…", mode="folder",
            tooltip="Training dataset root(s) (--data_path).\n"
                    "Drop several folders to train on all of them.", multi=True)
        cfg.addWidget(self._data_row)

        self._valid_row = _PathRow(
            "Valid path", "Select or drop the validation folder(s)…", mode="folder",
            tooltip="Validation set (--valid_path): one folder per song with\n"
                    "every stem plus mixture.wav.", multi=True)
        cfg.addWidget(self._valid_row)

        self._dataset_row = _ComboRowT(
            "Dataset type", [(k, k) for k, _d in DATASET_TYPES],
            tooltip="Dataset layout (--dataset_type):\n" + "\n".join(d for _k, d in DATASET_TYPES))
        cfg.addWidget(self._dataset_row)

        self._augment_row = _CheckRow(
            "Use augmentation for training",
            tooltip="Sets augmentations.enable in the config used for the run.\n"
                    "The augmentation details themselves live in the YAML.",
            checked=True)
        cfg.addWidget(self._augment_row)

        self._gpu_row = _ChevronRow(
            "GPUs / Workers / Seed",
            tooltip="Devices, DataLoader workers, random seed and the\n"
                    "run flags (pre-validation, resume options, …).",
            placeholder="Choose devices and run options…")
        self._gpu_row.clicked.connect(self._open_run_options)
        cfg.addWidget(self._gpu_row)

        self._ckpt_row = _PathRow(
            "Resume checkpoint", "Optional: checkpoint to start from…",
            mode="file", file_filter="Checkpoints (*.ckpt *.pth *.pt *.bin *.th *.chpt);;All files (*.*)",
            tooltip="Initial weights (--start_check_point). Leave empty to train\n"
                    "from scratch. LATEST picks the newest checkpoint in the\n"
                    "results folder.")
        self._latest_chip = QPushButton("Latest")
        self._latest_chip.setFixedHeight(18)
        self._latest_chip.setCursor(Qt.PointingHandCursor)
        self._latest_chip.setToolTip("Use the newest checkpoint from the results folder")
        self._latest_chip.setStyleSheet(_chip_ss(False))
        self._latest_chip.clicked.connect(self._pick_latest_ckpt)
        self._ckpt_row.add_extra(self._latest_chip)
        self._ckpt_row.changed.connect(self._refresh_latest_chip)
        cfg.addWidget(self._ckpt_row)

        self._edit_row = _ChevronRow("", placeholder="Edit Configuration", icon=_PencilIcon())
        self._edit_row.clicked.connect(self._open_config_editor)
        cfg.addWidget(self._edit_row)

        ll.addLayout(cfg)
        ll.addStretch()
        return w

    def _build_settings_column(self):
        w, ml = self._column((10, 32, 10, 16))
        ml.addWidget(_sec_hdr("Training Settings"))
        ml.addSpacing(16)
        st = QVBoxLayout()
        st.setSpacing(6)
        st.setContentsMargins(0, 0, 0, 0)

        self._batch_row = _EditRow("Batch size", "training.batch_size per GPU.", "from config", int_only=True)
        st.addWidget(self._batch_row)
        self._lr_row = _EditRow("Learning rate", "training.lr, e.g. 5e-05.", "from config")
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
        self._loss_row = _ChevronRow(
            "Loss", tooltip="Loss functions summed for training (--loss).\n"
                            "RoFormer / Conformer models use their internal loss unless\n"
                            "'use standard loss' is switched on in the dialog.",
            placeholder="Choose loss functions…")
        self._loss_row.clicked.connect(self._open_loss_dialog)
        st.addWidget(self._loss_row)
        self._metrics_row = _ChevronRow(
            "Metrics", tooltip="Validation metrics computed each epoch (--metrics).\n"
                               "More metrics = slower validation.",
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
        w, rl = self._column((10, 32, 32, 16))
        rl.addWidget(_sec_hdr("Training Monitor"))
        rl.addSpacing(16)

        # Progress
        self._prog_card = _Card("Training Progress")
        self._prog_lbl = QLabel("0 / 0 (0.00%)")
        self._prog_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{t.text_sec};background:transparent;"
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
        self._stat_box.setStyleSheet(_mono_box_ss())
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
            "QPlainTextEdit{" + _mono_box_ss() + "font-size:9px;}"
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
        self.btn_run = GlyphButton("Start Training", "▶", _solid_icon_color,
                                   glyph_size=18, text_size=12)
        self.btn_run.setFixedSize(200, 44)
        self.btn_run.setStyleSheet(solid_button_ss())
        self.btn_run.clicked.connect(self._run)
        self.btn_stop = GlyphButton("Stop", "■", _stop_icon_color,
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
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        left.addLayout(btn_row)
        hb.addLayout(left, 1)

        right = QHBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        self.btn_results = _IconTextButton("Open Results Folder", "folder")
        self.btn_results.setFixedSize(196, 40)
        self.btn_results.clicked.connect(self._open_results)
        self.btn_logs = _IconTextButton("View Logs", "doc")
        self.btn_logs.setFixedSize(150, 40)
        self.btn_logs.clicked.connect(self._open_logs)
        self.btn_export = _IconTextButton("Export Weights", "export")
        self.btn_export.setFixedSize(170, 40)
        self.btn_export.setToolTip("Strip a training checkpoint down to the model weights\n"
                                   "(no optimizer / scheduler / metrics history) for inference.")
        self.btn_export.clicked.connect(self._export_weights)
        right.addWidget(self.btn_export)
        right.addWidget(self.btn_results)
        right.addWidget(self.btn_logs)
        hb.addLayout(right, 0)
        hb.setAlignment(right, Qt.AlignBottom)
        return bar

    def reapply_theme(self):
        self.setStyleSheet(f"#trainingPage{{background:{theme_manager.theme.bg};}}")

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
            self._augment_row.set_checked(bool(aug.get("enable")))

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
        self._gpu_row.set_text(f"{dev}\n{n} / {o['num_workers']} / {o['seed']}")

    def _open_run_options(self):
        dlg = _RunOptionsDialog(dict(self._run_opts), self)
        if dlg.exec() == QDialog.Accepted:
            self._run_opts.update(dlg.values())
            self._refresh_run_text()

    def _open_loss_dialog(self):
        dlg = _MultiSelectDialog(
            "LOSS FUNCTIONS", catalog.loss_choices(), self._losses,
            hint="Selected losses are summed (each with its default coefficient). The "
                 "fullness / bleedless penalties are meant to accompany a primary loss.",
            toggle=("Use these losses for RoFormer / Conformer models too (--use_standard_loss)",
                    self._use_standard_loss,
                    "Without this, bs_roformer / mel_band_roformer / conformer models train with "
                    "their built-in loss and the selection above is ignored."),
            parent=self)
        if dlg.exec() == QDialog.Accepted:
            sel = dlg.selected()
            self._losses = sel or catalog.default_losses()
            self._use_standard_loss = dlg.toggle_value()
            self._refresh_loss_text()

    def _open_metrics_dialog(self):
        dlg = _MultiSelectDialog(
            "VALIDATION METRICS", catalog.metric_choices(), self._metrics,
            hint="Computed on the validation set after every epoch. The scheduler metric "
                 "is added automatically if it is not selected.",
            parent=self)
        if dlg.exec() == QDialog.Accepted:
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
        dlg.exec()

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
            "augment": self._augment_row.is_checked(),
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
        self._augment_row.set_checked(d.get("augment", True))
        ro = d.get("run_options") or {}
        if isinstance(ro, dict):
            self._run_opts.update({k: v for k, v in ro.items() if k in self._run_opts})
        self._ckpt_row.set_value(d.get("checkpoint", ""))
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
        self._use_standard_loss = bool(d.get("use_standard_loss", False))
        if d.get("metrics"):
            self._metrics = [m for m in d["metrics"] if m in catalog.metrics()] or catalog.default_metrics()
        self._refresh_loss_text()
        self._refresh_metrics_text()
        if d.get("metric_for_scheduler"):
            self._sched_row.set_key(d["metric_for_scheduler"])
        self._refresh_run_text()
        self._refresh_latest_chip()

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
            aug["enable"] = bool(self._augment_row.is_checked())
        elif self._augment_row.is_checked() is False:
            cfg["augmentations"] = {"enable": False}
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
        metrics = list(self._metrics)
        sched = self._sched_row.key() or metrics[0]
        if sched not in metrics:
            metrics.append(sched)
        cmd = [
            get_python_exe(), TRAIN_SCRIPT,
            "--model_type", model_type,
            "--config_path", cfg_path,
            "--results_path", results,
            "--data_path", *self._data_row.paths(),
            "--valid_path", *self._valid_row.paths(),
            "--dataset_type", str(self._dataset_row.key() or "1"),
            "--num_workers", str(o["num_workers"]),
            "--seed", str(o["seed"]),
            "--device_ids", *[str(i) for i in (o["device_ids"] or [0])],
            "--metrics", *metrics,
            "--metric_for_scheduler", sched,
            "--loss", *self._losses,
        ]
        ckpt = self._ckpt_row.value()
        if ckpt:
            cmd += ["--start_check_point", ckpt]
        flags = {
            "pin_memory": "--pin_memory", "pre_valid": "--pre_valid",
            "save_every_epoch": "--save_weights_every_epoch",
            "each_metrics_in_name": "--each_metrics_in_name",
            "load_optimizer": "--load_optimizer", "load_scheduler": "--load_scheduler",
            "load_epoch": "--load_epoch", "load_best_metric": "--load_best_metric",
        }
        for key, flag in flags.items():
            if o.get(key):
                cmd.append(flag)
        if self._use_standard_loss:
            cmd.append("--use_standard_loss")
        env = {"CUDA_VISIBLE_DEVICES": ""} if o.get("force_cpu") else None

        self._persist()
        self._reset_monitor()
        self._open_log_file(results)
        self._append_log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting training")
        self._append_log("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        self._append_log(f"Config written to: {cfg_path}")

        self._runner = ProcessRunner(cmd, cwd=REPO_ROOT, env=env)
        self._runner.log_line.connect(self._on_line)
        self._runner.finished.connect(self._on_finished)
        self.process_running.emit(True)
        self._runner.start()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self):
        if self._runner:
            self._append_log("[STOP] Stopping training…")
            self._runner.stop()

    def _on_finished(self, code):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self._append_log("Completed: training finished")
        else:
            self._append_log(f"ERROR: training exited with code {code}")
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
        if dlg.exec() != QDialog.Accepted:
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
