"""ui/widgets/common.py — kept minimal for new dark-theme design."""
import os, time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QFileDialog, QSizePolicy, QLabel, QFrame, QTextEdit, QComboBox,
)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from PySide6.QtGui import QTextCursor, QPainter, QPen, QColor
from ui.theme import theme_manager


def paint_chevron(painter, cx, cy, angle=0.0, hovered=False):
    """Draw the app's standard chevron — the MODEL LIBRARY arrow: a
    two-segment, round-capped `>` at rest, rotated by `angle` degrees.
    Rest color is a faint theme-text tint (alpha 51); accent blue when
    hovered. Shared by every chevron in the GUI so they all match."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.translate(cx, cy)
    painter.rotate(angle)
    _c = QColor(theme_manager.theme.text)
    _c.setAlpha(51)
    pen = QPen(QColor(theme_manager.accent) if hovered else _c, 2)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.drawLine(-5, -6, 0, 0)
    painter.drawLine(0, 0, -5, 6)
    painter.restore()


def _outline_icon_color(btn):
    """Icon color for outline buttons (Separate / Run Ensemble / Register /
    Download): accent at rest, light text while hovered (button fills), muted
    when disabled."""
    t = theme_manager.theme
    if not btn.isEnabled():
        return t.disabled_text
    return theme_manager._accent_text if btn._hovered else theme_manager.accent


def _solid_icon_color(btn):
    """Icon color for solid-accent buttons (Separate / Run Ensemble, matching
    the Install button): light text at rest and while hovered (only the
    background darkens), muted when disabled."""
    t = theme_manager.theme
    if not btn.isEnabled():
        return t.disabled_text
    return theme_manager._accent_text


def _stop_icon_color(btn):
    """Icon color for the Stop buttons: red once enabled, muted when off."""
    t = theme_manager.theme
    return t.error if btn.isEnabled() else t.text_muted


def _add_icon_color(btn):
    """Icon color for the inference '+ Add' button."""
    t = theme_manager.theme
    return t.text if btn._hovered else t.text_dim


def _addfile_icon_color(btn):
    """Icon color for the manual-ensemble '+ Add File' button."""
    t = theme_manager.theme
    return theme_manager.accent if btn._hovered else t.text_muted


class GlyphButton(QPushButton):
    """QPushButton whose leading glyph icon (play / stop / plus) renders at a
    larger size than the button text, which keeps its own small font. Both are
    placed in one centered, mouse-transparent container: a QHBoxLayout centers
    them on the same vertical line, so the icon sits right next to the text
    with no gap and no baseline offset. `icon_color` is called with the button
    to pick the color so the icon and text follow hover / enabled states."""

    def __init__(self, text, glyph, icon_color, glyph_size=18, text_size=12,
                 parent=None):
        super().__init__("", parent)
        self._hovered = False
        self._glyph_size = glyph_size
        self._text_size = text_size
        self._icon_color_cb = icon_color
        self._content = QWidget(self)
        self._content.setAttribute(Qt.WA_TransparentForMouseEvents)
        # own transparent background: page columns use bare `background:`
        # stylesheets that cascade into descendants — without this the
        # container paints an opaque box behind the icon/text.
        self._content.setStyleSheet("background:transparent;")
        h = QHBoxLayout(self._content)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self._glyph_lbl = QLabel(glyph)
        self._text_lbl = QLabel(text)
        h.addWidget(self._glyph_lbl)
        h.addWidget(self._text_lbl)
        self._content.raise_()
        self._refresh_icon()

    def _refresh_icon(self):
        # own transparent backgrounds: page containers use bare `background:`
        # stylesheets that cascade into descendants — without these the labels
        # would paint opaque boxes behind the icon/text.
        c = self._icon_color_cb(self)
        base = ("background:transparent;border:none;"
                "font-family:'Montserrat',sans-serif;"
                f"color:{c};")
        self._glyph_lbl.setStyleSheet(base + f"font-size:{self._glyph_size}px;")
        self._text_lbl.setStyleSheet(base + f"font-size:{self._text_size}px;")
        self._layout_icon()

    def _layout_icon(self):
        self._content.adjustSize()
        self._content.move((self.width() - self._content.width()) // 2,
                           (self.height() - self._content.height()) // 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_icon()

    def enterEvent(self, e):
        self._hovered = True
        self._refresh_icon()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self._refresh_icon()
        super().leaveEvent(e)

    def changeEvent(self, e):
        if e.type() == QEvent.Type.EnabledChange:
            self._refresh_icon()
        super().changeEvent(e)


def add_button_hover():
    """The '+ Add' hover rule (shared by Log / Clear / Copy Log / Check For
    Updates buttons): soft accent background, accent border and regular text
    color."""
    t = theme_manager.theme
    return (
        f"QPushButton:hover{{background:{theme_manager._accent_soft};"
        f"color:{t.text};border:1px solid {theme_manager.accent};}}"
    )


class EllipsisButton(QPushButton):
    """The '\u00b7\u00b7\u00b7' browse button used in SETTINGS (LOCAL FILES):
    a three-dot text button that tints accent on hover. Shared with the
    INFERENCE page's Input/Output rows so every browse button matches."""

    def __init__(self, parent=None):
        super().__init__("\u00b7\u00b7\u00b7", parent)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.PointingHandCursor)
        t = theme_manager.theme
        _c = QColor(theme_manager.accent)
        self.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_dim};"
            f"border:none;font-size:14px;font-weight:600;border-radius:4px;}}"
            f"QPushButton:hover{{color:{theme_manager.accent};"
            f"background:rgba({_c.red()},{_c.green()},{_c.blue()},0.12);}}"
        )


class ChevronCombo(QComboBox):
    """QComboBox with the app's standard chevron painted over its right edge
    (same shape/colors as the MODEL LIBRARY arrow; the native arrow is hidden
    by the combo's stylesheet). Rotates to point down while the popup is open
    and turns accent blue on hover."""

    popupOpened = Signal()
    popupClosed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0.0
        self.popupOpened.connect(self._on_popup_opened)
        self.popupClosed.connect(self._on_popup_closed)

    def showPopup(self):
        super().showPopup()
        self.popupOpened.emit()

    def hidePopup(self):
        super().hidePopup()
        self.popupClosed.emit()

    def _on_popup_opened(self):
        self._angle = 90.0
        self.update()

    def _on_popup_closed(self):
        self._angle = 0.0
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        r = self.rect()
        paint_chevron(p, r.right() - 18, r.center().y(), self._angle, self.underMouse())
        p.end()


def outline_button_ss(font_size=12):
    """Primary action-button stylesheet matching the ENSEMBLE Select button:
    Montserrat 600 at the same size, accent outline + accent text by default,
    filling solid with the accent (light text) on hover. Shared by the primary
    action buttons across pages so they all match in font and look."""
    t = theme_manager.theme
    return (
        "QPushButton{"
        "background:transparent;"
        f"border:1px solid {theme_manager.accent};border-radius:8px;"
        f"color:{theme_manager.accent};"
        "font-family:'Montserrat',sans-serif;font-weight:600;"
        f"font-size:{font_size}px;}}"
        f"QPushButton:hover{{background:{theme_manager.accent};color:{theme_manager._accent_text};}}"
        f"QPushButton:pressed{{background:{theme_manager._accent_hover};color:{theme_manager._accent_text};}}"
        f"QPushButton:disabled{{background:{t.disabled_bg};color:{t.disabled_text};border:1px solid {t.border};}}"
    )


def solid_button_ss(font_size=12):
    """Solid-accent primary button, matching the Install button: filled accent
    with light text, darkening to the accent-hover color on hover / press."""
    t = theme_manager.theme
    return (
        "QPushButton{"
        f"background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
        "border-radius:8px;"
        "font-family:'Montserrat',sans-serif;font-weight:600;"
        f"font-size:{font_size}px;}}"
        f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
        f"QPushButton:pressed{{background:{theme_manager._accent_hover};}}"
        f"QPushButton:disabled{{background:{t.disabled_bg};color:{t.disabled_text};}}"
    )


class FilePicker(QWidget):
    path_changed = Signal(str)
    def __init__(self, mode="file", filter="All (*.*)", placeholder="", drag_drop=False, parent=None):
        super().__init__(parent)
        self._mode = mode; self._filter = filter
        hl = QHBoxLayout(self); hl.setContentsMargins(0,0,0,0); hl.setSpacing(0)
        self.line = QLineEdit(); self.line.setPlaceholderText(placeholder or f"Select {mode}…")
        self.line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line.textChanged.connect(self.path_changed); hl.addWidget(self.line, 1)
        self._btn = QPushButton("..."); self._btn.setFixedSize(44, 38)
        self._apply_style()
        self._btn.clicked.connect(self._browse); hl.addWidget(self._btn)
    def _apply_style(self):
        self._btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};border:none;}}"
            f"QPushButton:hover{{background:{theme_manager.accent};color:{theme_manager._accent_text};}}"
        )
    def reapply_theme(self):
        self._apply_style()
    def value(self): return self.line.text().strip()
    def set_value(self, v): self.line.setText(v)
    def _browse(self):
        if self._mode == "folder": path = QFileDialog.getExistingDirectory(self, "Select folder")
        else: path, _ = QFileDialog.getOpenFileName(self, "Select file", filter=self._filter)
        if path: self.line.setText(path)

class PageHeader(QWidget):
    """mvsep-style page header: left accent bar, big uppercase title, and
    a subtitle with an accent-highlighted phrase.
    Optional back button above the title; extra widgets dock on the right."""

    def __init__(self, title, subtitle="", highlight="", back=False, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Vertical accent bar running down the left edge
        bar = QFrame()
        bar.setFixedWidth(4)
        bar.setStyleSheet(
            f"background:{theme_manager.accent};border:none;border-radius:2px;"
        )
        root.addWidget(bar)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        self.back_btn = None
        if back:
            self.back_btn = QPushButton("← Back")
            self.back_btn.setMinimumHeight(32)
            self.back_btn.setCursor(Qt.PointingHandCursor)
            self.back_btn.setStyleSheet(self._back_ss())
            col.addWidget(self.back_btn)
            col.addSpacing(14)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:32px;font-weight:bold;color:"
            f"{theme_manager.theme.text};background:transparent;border:none;letter-spacing:-0.5px;"
        )
        col.addWidget(self.title_lbl)

        self.sub_lbl = None
        if subtitle:
            if highlight and highlight in subtitle:
                subtitle = subtitle.replace(
                    highlight,
                    '<span style="color:'
                    f'{theme_manager.accent!s};font-weight:bold;">'
                    f'{highlight!s}</span>',
                )
            self.sub_lbl = QLabel(subtitle)
            self.sub_lbl.setTextFormat(Qt.RichText)
            self.sub_lbl.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;color:"
                f"{theme_manager.theme.text_muted};background:transparent;border:none;letter-spacing:1px;"
            )
            col.addWidget(self.sub_lbl)

        root.addLayout(col, 1)

    def _back_ss(self):
        t = theme_manager.theme
        return (
            "QPushButton{background:transparent;color:"
            f"{t.text_muted};border:1px solid "
            f"{t.border_visible};font-family:'Montserrat';font-size:12px;border-radius:6px;padding:0 16px;}}"
            "QPushButton:hover{background:"
            f"{t.border};color:"
            f"{t.text};border-color:"
            f"{theme_manager.accent};}}"
        )

    def add_extra(self, widget):
        self.layout().addWidget(widget)
        return widget

    def set_title(self, text):
        self.title_lbl.setText(text.upper())

    def set_subtitle(self, text, highlight=""):
        if self.sub_lbl is None:
            return None
        if highlight and highlight in text:
            text = text.replace(
                highlight,
                '<span style="color:'
                f'{theme_manager.accent!s};font-weight:bold;">'
                f'{highlight!s}</span>',
            )
        self.sub_lbl.setText(text)

class SectionHeader(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 8, 0, 4)
        self._lbl = QLabel(title.upper())
        self._apply_style()
        hl.addWidget(self._lbl)
        hl.addStretch()

    def _apply_style(self):
        self._lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;color:"
            f"{theme_manager.theme.text};background:transparent;padding-left:10px;border-left:4px solid "
            f"{theme_manager.accent};letter-spacing:1px;"
        )

    def reapply_theme(self):
        self._apply_style()

class ConsoleLog(QWidget):
    _GREEN_TOKENS = (">", "[INFO]", "[PROCESS]", "[PROGRESS]", "[GPU]", "[STATUS]", "[WARN]", "[ERROR]")

    def __init__(self, parent=None):
        super().__init__(parent)
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        self._edit = QTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setObjectName("consoleLog")
        self._edit.setLineWrapMode(QTextEdit.NoWrap)
        self._apply_style()
        vl.addWidget(self._edit)
        self.setStyleSheet(f"background:{theme_manager.theme.console_bg};border:none;")
        for line in ("> MSS TOOL v1.0.0", "> Ready.", "[INFO] Waiting for input…"):
            self._insert(line)

    def _apply_style(self):
        self._edit.setStyleSheet(
            "QTextEdit#consoleLog{background:"
            f"{theme_manager.theme.console_bg};color:"
            f"{theme_manager.theme.text};"
            "font-family:'Courier New','Consolas',monospace;font-size:11px;border:none;padding:10px 12px;}"
        )

    def _colorize(self, text):
        import html
        for t in self._GREEN_TOKENS:
            if text.strip().startswith(t):
                rest = text[text.index(t) + len(t):]
                return ('<span style="color:'
                        f'{theme_manager.accent};font-weight:bold;">'
                        f'{html.escape(t)}</span><span style="color:'
                        f'{theme_manager.theme.text};">'
                        f'{html.escape(rest)}</span>')
        import html as _h
        return ('<span style="color:'
                f'{theme_manager.theme.text};">'
                f'{_h.escape(text)}</span>')

    def _insert(self, text):
        c = self._edit.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        self._edit.setTextCursor(c)
        self._edit.insertHtml(self._colorize(text) + "<br>")
        self._edit.verticalScrollBar().setValue(self._edit.verticalScrollBar().maximum())

    def append_line(self, text):
        self._insert(text)

    def clear_log(self):
        self._edit.clear()

    def reapply_theme(self):
        self._apply_style()
        self.setStyleSheet(f"background:{theme_manager.theme.console_bg};border:none;")

class SpectrogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

    def set_audio(self, path):
        return None

    def clear_audio(self):
        return None

    def set_active(self, v):
        return None

    def reapply_theme(self):
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

class WaveformPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

    def set_audio(self, path):
        return None

    def clear_audio(self):
        return None

    def set_active(self, v):
        return None

    def reapply_theme(self):
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

class ProcessingStatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

    def start_timer(self):
        return None

    def stop_timer(self):
        return None

    def update_stats(self, progress=""):
        return None

    def reapply_theme(self):
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

