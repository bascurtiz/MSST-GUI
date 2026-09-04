"""ui/widgets/common.py — kept minimal for new dark-theme design."""
import os, time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QFileDialog, QSizePolicy, QLabel, QFrame, QTextEdit, QComboBox,
)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from PySide6.QtGui import QTextCursor, QPainter, QPen, QColor, QFont, QFontMetrics, QImage
from ui.theme import theme_manager, FONT_FAMILY as FONT_FAMILY_DEFAULT
from backend.version import APP_VERSION


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


def css_color(value, fallback="#808080"):
    """QColor from a theme token, including CSS rgba() strings that
    QColor() itself cannot parse (invalid → black when painted)."""
    import re as _re
    c = QColor(value)
    if c.isValid():
        return c
    m = _re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", str(value).strip())
    if m:
        c = QColor(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        c.setAlphaF(float(m.group(4)))
        return c
    return QColor(fallback)


DOWNLOAD_GLYPH = "download"  # marker: draw a download icon instead of text


def run_blurred_dialog(dialog):
    """Run a modal dialog behind the app's frosted-backdrop blur.

    Walks up from the dialog to the owning window: if it exposes
    show_blurred_dialog() (MainWindow does), the backdrop blurs while the
    dialog is up and the result is returned as usual. Falls back to a plain
    exec() in standalone/offscreen contexts (no window, or no blur support),
    so dialogs never break just because there's no MainWindow.
    """
    w = dialog.parentWidget()
    while w is not None:
        opener = getattr(w, "show_blurred_dialog", None)
        if callable(opener):
            return opener(dialog)
        w = w.parentWidget()
    return dialog.exec()


def _painted_ink_center(w):
    """Vertical ink center of a small widget rendered offscreen, using the
    probe red (#FF2020) set by the caller; None if nothing paints."""
    img = QImage(w.width(), w.height(), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    w.render(img)
    rows = []
    for y in range(img.height()):
        for x in range(img.width()):
            c = QColor(img.pixel(x, y))
            if c.red() > 150 and c.green() < 100 and c.blue() < 100:
                rows.append(y)
                break
    if not rows:
        return None
    return (min(rows) + max(rows)) / 2.0


class _GlyphWidget(QWidget):
    """Leading glyph of a GlyphButton, hand-painted with an optical vertical
    offset. Symbols like '+' ride high inside their em box, so a plainly
    centered 18px '+' floats ~3px above its 12px text label; the offset is
    computed from the font metrics so the glyph's ink center lands on the
    text's cap-height center (what the eye compares against).
    The special DOWNLOAD_GLYPH marker draws a download icon (arrow down with
    a dash below) instead of text."""

    def __init__(self, text, size, family, parent=None):
        super().__init__(parent)
        self._text = text
        self._size = size
        self._family = family
        self._color = QColor("#FFFFFF")
        self._custom = text == DOWNLOAD_GLYPH
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        fm = self._metrics()
        from math import ceil
        if self._custom:
            self.setFixedSize(ceil(size * 0.72), fm.height())
            self._dy = -2  # optically center on the neighboring text
            return
        self.setFixedSize(ceil(fm.horizontalAdvance(text)) + 2, fm.height())
        self._dy = 0

    def _font(self):
        f = QFont(self._family)
        f.setPixelSize(self._size)
        return f

    def _metrics(self):
        return QFontMetrics(self._font())

    def set_color(self, color):
        # Theme tokens may be CSS rgba() strings — parse them properly or the
        # glyph paints invalid/black (e.g. text_muted, text_dim).
        self._color = css_color(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.translate(0, self._dy)
        if self._custom:
            self._paint_download(p)
            return
        p.setFont(self._font())
        p.setPen(QPen(self._color))
        rect = self.rect()
        p.drawText(rect, Qt.AlignCenter, self._text)

    def _paint_download(self, p):
        """Download icon: arrow pointing down with a dash below it. Drawn at
        ~60% of the glyph box so it reads as the same visual size as the
        small caps text next to it."""
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self._color, max(1.3, self._size / 13.0))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        # Centered drawing area: ~52% of the box height, 72% of its width —
        # sized to read like the small caps text, not larger.
        bw = self.width() * 0.72
        bh = self.height() * 0.52
        x0 = (self.width() - bw) / 2.0
        y0 = (self.height() - bh) / 2.0
        cx = self.width() / 2.0
        top = y0 + bh * 0.02
        tip = y0 + bh * 0.66
        head = bw * 0.32
        # shaft
        p.drawLine(cx, top, cx, tip)
        # arrow head
        p.drawLine(cx - head, tip - head, cx, tip)
        p.drawLine(cx + head, tip - head, cx, tip)
        # dash below
        dash_y = y0 + bh * 0.96
        dash = bw * 0.40
        p.drawLine(cx - dash, dash_y, cx + dash, dash_y)


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
        family = self.font().family() or FONT_FAMILY_DEFAULT
        self._glyph_lbl = _GlyphWidget(glyph, glyph_size, family)
        self._glyph_size = glyph_size
        self._text_lbl = QLabel(text)
        h.addWidget(self._glyph_lbl, 0, Qt.AlignVCenter)
        h.addWidget(self._text_lbl, 0, Qt.AlignVCenter)
        self._content.raise_()
        self._refresh_icon()
        self._optical_align(text_size)

    def _optical_align(self, text_size):
        """Calibrate the glyph's vertical offset from painted pixels: font
        metric bounds don't match the hinted rendering closely enough (a
        metric-derived nudge left \u25B6/\u25A0 sitting visibly low). Renders
        glyph and text once with a probe color and aligns the glyph's ink
        center with the text's CAP-BAND center: the probe is the uppercase
        text, whose ink spans exactly cap-top..baseline. (The mixed-case ink
        center is dragged down by descenders, which makes the glyph float.)"""
        real_cb = self._icon_color_cb
        self._icon_color_cb = lambda b: "#FF2020"  # probe color for ink scan
        self._glyph_lbl._dy = 0
        self._refresh_icon()
        original_text = self._text_lbl.text()
        try:
            self._text_lbl.setText(original_text.upper())
            g = _painted_ink_center(self._glyph_lbl)
            t = _painted_ink_center(self._text_lbl)
        finally:
            self._text_lbl.setText(original_text)
            self._icon_color_cb = real_cb
            self._refresh_icon()
        if g is not None and t is not None:
            # Both tight boxes are centered against each other in the row —
            # map the text's ink center into the (taller) glyph box's space
            # before comparing, or the offset absorbs the box height delta.
            top_off = (self._glyph_lbl.height() - self._text_lbl.height()) / 2.0
            dy = (t + top_off) - g
            # Damped: apply half the measured correction. Full-strength
            # nudges overcorrect depending on the machine's font hinting
            # (±2px swings on the small '+ Add' label), and a residual of
            # ~0.5px is invisible while a 2px overshoot is not.
            dy *= 0.5
            self._glyph_lbl._dy = int(round(max(-2.0, min(2.0, dy))))
            self._glyph_lbl.update()

    def _refresh_icon(self):
        # own transparent backgrounds: page containers use bare `background:`
        # stylesheets that cascade into descendants — without these the labels
        # would paint opaque boxes behind the icon/text.
        c = self._icon_color_cb(self)
        base = ("background:transparent;border:none;"
                "font-family:'Montserrat',sans-serif;"
                f"color:{c};")
        self._glyph_lbl.set_color(c)
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


def dark_menu_qss():
    """The app's always-dark QMenu look, as a widget-level stylesheet.

    Menus parented inside page columns with bare `background:` stylesheets
    inherit that cascade, which overrides the app-level dark QMenu rule in
    light mode (white menu, near-white text — unreadable). Apply this to any
    QMenu created under such an ancestor."""
    t = theme_manager.theme
    return (
        f"QMenu{{background:{t.menu_bg};color:{t.menu_text};"
        f"border:1px solid {t.menu_border};border-radius:6px;padding:6px;}}"
        "QMenu::item{background:transparent;color:" + t.menu_text + ";"
        "padding:7px 20px;border-radius:4px;font-size:11px;}"
        "QMenu::item:selected{background:" + theme_manager.accent + ";color:#FFFFFF;}"
        "QMenu::item:disabled{color:" + t.menu_disabled + ";}"
        "QMenu::separator{height:1px;background:" + t.menu_sep + ";margin:5px 10px;}"
        "QMenu::icon{padding-left:8px;}"
    )


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


# Type-badge colors per model type — reuses the waveform stem palette where
# the two overlap (vocals, instrumental, drums, bass, piano, guitar...). Used
# by the model library and auto-ensemble model cards.
MODEL_TYPE_COLORS = {
    "vocals": "#A855F7",
    "instrumental": "#60A5FA",
    "dereverb / deecho": "#DDDDDD",
    "denoise": "#888888",
    "phantom centre": "#E4FF76",
    "karaoke": "#F07CA8",
    "dual target (instrumental & vocals)": "#10B981",
    "multi stems": "#FFCA28",
    "super resolution": "#B55064",
    "drums": "#F59E0B",
    "bass": "#EF4444",
    "piano": "#485FAB",
    "guitar": "#C1090B",
    "wind": "#00B8D3",
    "strings": "#76C043",
    "percussion": "#F36E21",
    "keys": "#485FAB",
    "effects": "#FFFFFF",
    "crowd": "#94A3B8",
}


def _type_badge_color(model_type):
    """Badge tint color for a model type, or None to fall back to the theme's
    neutral badge chip."""
    return MODEL_TYPE_COLORS.get((model_type or "").lower())


def _type_badge_ss(model_type):
    """Stylesheet for a model-type badge: text tinted with the type's color on
    a translucent tint of the same color. The bright palette is dimmed on the
    dark theme (kept distinguishable but less glaring) and darkened further on
    the light theme for contrast against the light badge chip."""
    t = theme_manager.theme
    hexc = _type_badge_color(model_type)
    if not hexc:
        return (
            "font-family:'Montserrat';font-size:8px;font-weight:700;"
            f"color:{t.text_label};background:{t.surface_alt};"
            "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
        )
    c = QColor(hexc)
    rgb = f"{c.red()},{c.green()},{c.blue()}"
    if theme_manager.mode == "light":
        text = c.darker(230).name()
        bg_a, bd_a = 32, 70
    else:
        text = c.darker(140).name()
        bg_a, bd_a = 22, 48
    return (
        "font-family:'Montserrat';font-size:8px;font-weight:700;"
        f"color:{text};"
        f"background:rgba({rgb},{bg_a});"
        f"border:1px solid rgba({rgb},{bd_a});"
        "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
    )


# Compact titles for the category badges / "sort by target" grouping rows.
_TYPE_TITLES = {
    "dual target (instrumental & vocals)": "DUAL TARGET",
    "dereverb / deecho": "DEREVERB / DEECHO",
    "phantom centre": "PHANTOM CENTRE",
    "multi stems": "MULTI STEMS",
    "super resolution": "SUPER RESOLUTION",
    "vocals": "VOCALS",
    "instrumental": "INSTRUMENTAL",
    "denoise": "DENOISE",
    "karaoke": "KARAOKE",
    "drums": "DRUMS",
    "bass": "BASS",
    "piano": "PIANO",
    "guitar": "GUITAR",
    "wind": "WIND",
    "strings": "STRINGS",
    "percussion": "PERCUSSION",
    "keys": "KEYS",
    "effects": "EFFECTS",
    "crowd": "CROWD",
}


def _type_title(type_key):
    return _TYPE_TITLES.get(type_key or "", (type_key or "").upper() or "UNKNOWN")


def _custom_badge_ss():
    """Stylesheet for the CUSTOM model-control badge — grayscale instead of the
    accent tint, matching the neutral 'rest' gray of the waveform palette."""
    t = theme_manager.theme
    base = "#9A9FB3"
    c = QColor(base)
    rgb = f"{c.red()},{c.green()},{c.blue()}"
    text = c.darker(230).name() if theme_manager.mode == "light" else base
    return (
        "font-family:'Montserrat';font-size:8px;font-weight:700;"
        f"color:{text};"
        f"background:rgba({rgb},32);"
        f"border:1px solid rgba({rgb},70);"
        "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
    )


def _blocked_badge_ss():
    """Stylesheet for the NOT-RUNNABLE badge on model cards whose type has
    no branch in the inference engine — red error tint on a translucent
    chip, mirroring the custom/type badge shapes."""
    t = theme_manager.theme
    c = QColor(t.error)
    rgb = f"{c.red()},{c.green()},{c.blue()}"
    text = c.darker(210).name() if theme_manager.mode == "light" else c.name()
    return (
        "font-family:'Montserrat';font-size:8px;font-weight:700;"
        f"color:{text};"
        f"background:rgba({rgb},26);"
        f"border:1px solid rgba({rgb},60);"
        "padding:1px 6px;border-radius:3px;letter-spacing:0.5px;"
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
        for line in (f"> MSS TOOL v{APP_VERSION}", "> Ready.", "[INFO] Waiting for input…"):
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

