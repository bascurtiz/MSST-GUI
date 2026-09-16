"""ui/widgets/common.py — kept minimal for new dark-theme design."""
import math
import os, time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QFileDialog, QSizePolicy, QLabel, QFrame, QTextEdit, QComboBox,
    QGraphicsOpacityEffect, QDialog, QScrollArea, QAbstractScrollArea,
)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QRectF, QPointF, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import (
    QTextCursor, QPainter, QPen, QColor, QFont, QFontMetrics, QImage, QPixmap,
    QPainterPath,
)
from ui.theme import theme_manager, FONT_FAMILY as FONT_FAMILY_DEFAULT, UIConstants
from ui.strings import (
    PAGE_HELP, T_PAGE_HELP_TOOLTIP,
    HELP_REQUIRED_CAPTION, HELP_OPTIONAL_CAPTION,
)
from backend.version import APP_VERSION


def paint_chevron(painter, cx, cy, angle=0.0, hovered=False, color=None):
    """Draw the app's standard chevron — the MODEL LIBRARY arrow: a
    two-segment, round-capped `>` at rest, rotated by `angle` degrees.
    Rest color is a faint theme-text tint (alpha 51); accent blue when
    hovered. Pass `color` to override both. Shared by every chevron in
    the GUI so they all match."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.translate(cx, cy)
    painter.rotate(angle)
    if color is None:
        _c = QColor(theme_manager.theme.text)
        _c.setAlpha(51)
        color = QColor(theme_manager.accent) if hovered else _c
    pen = QPen(QColor(color), 2)
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


def _pause_icon_color(btn):
    """Icon color for outline Pause / Open Output buttons."""
    t = theme_manager.theme
    if not btn.isEnabled():
        return t.disabled_text
    return t.text if btn._hovered else t.text_dim


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
PAUSE_GLYPH = "pause"        # two vertical bars
FOLDER_GLYPH = "folder"      # outline folder


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
    The special DOWNLOAD_GLYPH / PAUSE_GLYPH / FOLDER_GLYPH markers draw
    icons instead of text."""

    _CUSTOM = {DOWNLOAD_GLYPH, PAUSE_GLYPH, FOLDER_GLYPH}

    def __init__(self, text, size, family, parent=None):
        super().__init__(parent)
        self._text = text
        self._size = size
        self._family = family
        self._color = QColor("#FFFFFF")
        self._custom = text in self._CUSTOM
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
        if not p.isActive():
            return
        p.translate(0, self._dy)
        if self._custom:
            if self._text == PAUSE_GLYPH:
                self._paint_pause(p)
            elif self._text == FOLDER_GLYPH:
                self._paint_folder(p)
            else:
                self._paint_download(p)
            p.end()
            return
        p.setFont(self._font())
        p.setPen(QPen(self._color))
        p.drawText(self.rect(), Qt.AlignCenter, self._text)
        p.end()

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

    def _paint_pause(self, p):
        """Two vertical bars, optically matching the play/stop glyphs."""
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        bw = self.width() * 0.72
        bh = self.height() * 0.52
        x0 = (self.width() - bw) / 2.0
        y0 = (self.height() - bh) / 2.0
        bar_w = max(2.0, bw * 0.22)
        gap = bw * 0.22
        left = x0 + (bw - (bar_w * 2 + gap)) / 2.0
        p.drawRoundedRect(QRectF(left, y0, bar_w, bh), 0.8, 0.8)
        p.drawRoundedRect(QRectF(left + bar_w + gap, y0, bar_w, bh), 0.8, 0.8)

    def _paint_folder(self, p):
        """Outline folder matching the Training page's folder button icon."""
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self._color, max(1.3, self._size / 13.0))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        bw = self.width() * 0.84
        bh = self.height() * 0.56
        x0 = (self.width() - bw) / 2.0
        y0 = (self.height() - bh) / 2.0
        tab = bw * 0.28
        path = QPainterPath()
        path.moveTo(x0, y0 + bh * 0.22)
        path.lineTo(x0 + tab, y0 + bh * 0.22)
        path.lineTo(x0 + tab + bw * 0.12, y0)
        path.lineTo(x0 + bw, y0)
        path.lineTo(x0 + bw, y0 + bh)
        path.lineTo(x0, y0 + bh)
        path.closeSubpath()
        p.drawPath(path)


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
        self._fit_h = None
        self._fit_pad = 28
        self._refresh_icon()
        self._optical_align(text_size)

    def fit_contents(self, height=40, h_pad=28):
        """Lock height; width hugs the glyph + label plus side padding."""
        self._fit_h = height
        self._fit_pad = h_pad
        self._content.adjustSize()
        self.setFixedSize(self._content.width() + h_pad, height)
        self._layout_icon()

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

    def set_label(self, text):
        """Replace the visible label. (The QPushButton text itself is unused —
        GlyphButton paints its text through an inner QLabel so the glyph can
        sit beside it.) Re-centers the content when the new text's width
        differs."""
        if self._text_lbl.text() == text:
            return
        self._text_lbl.setText(text)
        self._layout_icon()
        if self._fit_h is not None:
            self.fit_contents(self._fit_h, self._fit_pad)

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
    "dual target (instrumental & vocals)": "Dual Target",
    "dereverb / deecho": "Dereverb / Deecho",
    "phantom centre": "Phantom Centre",
    "multi stems": "Multi Stems",
    "super resolution": "Super Resolution",
    "vocals": "Vocals",
    "instrumental": "Instrumental",
    "denoise": "Denoise",
    "karaoke": "Karaoke",
    "drums": "Drums",
    "bass": "Bass",
    "piano": "Piano",
    "guitar": "Guitar",
    "wind": "Wind",
    "strings": "Strings",
    "percussion": "Percussion",
    "keys": "Keys",
    "effects": "Effects",
    "crowd": "Crowd",
}


def _type_title(type_key):
    return _TYPE_TITLES.get(type_key or "", (type_key or "").title() or "Unknown")


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

class _BackChevron(QPushButton):
    """Compact left-pointing chevron for PageHeader(back=True). Sits after
    the accent bar, beside the title — not a full-width bar above it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hovered = False
        self.setFixedSize(24, 24)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip("Back")
        self.setStyleSheet(
            "QPushButton{background:transparent;border:none;padding:0;}"
        )

    def enterEvent(self, e):
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, event):
        p = QPainter(self)
        if self._hovered:
            color = theme_manager.accent
        elif theme_manager.mode == "dark":
            color = "#FFFFFF"
        else:
            color = theme_manager.theme.text_muted
        paint_chevron(p, self.width() / 2, self.height() / 2, 180.0,
                      color=color)
        p.end()


class HelpButton(QPushButton):
    """30px circular '?' that opens the page Help dialog. Painted so a
    theme switch restyles it without a stylesheet rebuild."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageHelpBtn")
        self.setFixedSize(30, 30)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip(T_PAGE_HELP_TOOLTIP)
        self.setStyleSheet("QPushButton{background:transparent;border:none;}")
        self._hovered = False

    def enterEvent(self, e):
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, event):
        p = QPainter(self)
        if not p.isActive():
            return
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        accent = QColor(theme_manager.accent)
        if self._hovered:
            p.setPen(Qt.NoPen)
            p.setBrush(accent)
            p.drawEllipse(r)
            text = QColor(theme_manager._accent_text)
        else:
            pen = QPen(accent, 1.5)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(r)
            text = accent
        font = QFont(FONT_FAMILY_DEFAULT, 12)
        font.setBold(True)
        p.setFont(font)
        p.setPen(text)
        p.drawText(self.rect(), Qt.AlignCenter, "?")
        p.end()


class _HelpChevron(QWidget):
    """Same 24×24 `>` as the CONFIGURATION combo rows, so Optional lines
    up with Quality / Stems / Device."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self._angle = 0.0
        self._hovered = False

    def set_expanded(self, on):
        self._angle = 90.0 if on else 0.0
        self.update()

    def set_hovered(self, on):
        self._hovered = on
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        if not p.isActive():
            return
        paint_chevron(p, 12, 12, self._angle, hovered=self._hovered)
        p.end()


class _HelpHeader(QWidget):
    """Clickable badge + caption row for the OPTIONAL card."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
            e.accept()
            return
        super().mousePressEvent(e)


def _help_error_hover():
    c = QColor(theme_manager.theme.error)
    return f"rgba({c.red()},{c.green()},{c.blue()},51)"


def _help_num_badge(num, accent=True):
    """Same 24×24 square chip as How Ensemble Works (INPUT / PROCESS)."""
    t = theme_manager.theme
    b = QLabel(str(num))
    b.setFixedSize(24, 24)
    b.setAlignment(Qt.AlignCenter)
    if accent:
        b.setStyleSheet(
            "background:" + theme_manager.accent + ";"
            "color:" + theme_manager._accent_text + ";"
            "font-family:'Montserrat',sans-serif;"
            "font-size:11px;font-weight:bold;"
            "border-radius:3px;"
        )
    else:
        b.setStyleSheet(
            f"background:{t.surface};color:{t.text_dim};"
            f"border:1px solid {t.border_visible};"
            "font-family:'Montserrat',sans-serif;"
            "font-size:11px;font-weight:bold;"
            "border-radius:3px;"
        )
    return b


def _help_step_row(n, text, required):
    """One to-do: square index + 13px secondary copy, no card chrome."""
    t = theme_manager.theme
    row = QWidget()
    row.setStyleSheet("background:transparent;")
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 4, 0, 4)
    hl.setSpacing(12)
    hl.addWidget(_help_num_badge(n, accent=False), 0, Qt.AlignTop)
    body = QLabel(text)
    body.setWordWrap(True)
    body.setStyleSheet(
        "font-family:'Montserrat';font-size:13px;"
        f"color:{t.text_sec};background:transparent;"
    )
    hl.addWidget(body, 1)
    return row


class _HelpSection(QFrame):
    """Open REQUIRED / OPTIONAL block — no card, accent title + square index
    like How Ensemble Works' INPUT / PROCESS sections."""
    toggled = Signal()

    def __init__(self, kind, items, caption, index=1, parent=None):
        super().__init__(parent)
        t = theme_manager.theme
        required = kind == "required"
        self.setObjectName("helpRequired" if required else "helpOptional")
        self.setStyleSheet(
            f"QFrame#{self.objectName()}{{background:transparent;border:none;}}"
        )
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(14)
        vl.addSpacing(8)

        hr = QHBoxLayout()
        hr.setSpacing(8)
        title = QLabel("REQUIRED" if required else "OPTIONAL")
        title.setObjectName("helpRequiredBadge" if required else "helpOptionalBadge")
        title.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_TITLE_FONT_SIZE}px;"
            "font-weight:bold;letter-spacing:1px;"
            f"color:{theme_manager.accent};background:transparent;"
        )
        hr.addWidget(title)
        hr.addStretch()
        vl.addLayout(hr)

        cap = QLabel(caption)
        cap.setWordWrap(True)
        cap.setStyleSheet(
            "font-family:'Montserrat';font-size:13px;"
            f"color:{t.text_sec};background:transparent;"
        )
        vl.addWidget(cap)

        self._body = QWidget()
        self._body.setObjectName(
            "helpRequiredBody" if required else "helpOptionalBody")
        self._body.setStyleSheet("background:transparent;")
        bl = QVBoxLayout(self._body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        for i, step in enumerate(items, 1):
            bl.addWidget(_help_step_row(i, step, required))
        vl.addWidget(self._body)


def _help_section(kind, items, caption, index=1):
    return _HelpSection(kind, items, caption, index=index)


# Match inference/training CONFIGURATION row height so Optional cards are
# never squeezed below the Model Library architecture cards (42px).
_OPTIONAL_ROW_H = 42


def _optional_scroll_ss(gutter=True):
    t = theme_manager.theme
    # Inference Optional sits next to the Model Library bar, so a 16px
    # lane keeps SPECTRO / Skip errors off the thumb. Training Optional
    # has no neighbor bar — that lane capped the cards on the right.
    bar = (
        "QScrollBar:vertical{width:16px;background:transparent;margin:0;}"
        "QScrollBar::handle:vertical{"
        f"background:{t.scrollbar_handle};"
        "border-radius:2px;min-height:30px;margin-left:12px;}"
        f"QScrollBar::handle:vertical:hover{{background:{t.scrollbar_hover};}}"
        "QScrollBar::add-line:vertical{height:0;}"
        "QScrollBar::sub-line:vertical{height:0;}"
        "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{"
        "background:transparent;}"
    ) if gutter else (
        "QScrollBar:vertical{width:0px;height:0px;}"
    )
    return (
        "QScrollArea{background:transparent;border:none;}"
        + bar
    )


def _page_edge_scroll_ss():
    """Full-page vertical bar: 4px thumb inset 12px from the window edge
    (Training / Auto Ensemble). Opposite gutter of Optional — here the
    complaint is the chrome, not the cards."""
    t = theme_manager.theme
    return (
        "QScrollArea{background:transparent;border:none;}"
        "QScrollBar:vertical{width:16px;background:transparent;margin:0;}"
        "QScrollBar::handle:vertical{"
        f"background:{t.scrollbar_handle};"
        "border-radius:2px;min-height:30px;margin-right:12px;}"
        f"QScrollBar::handle:vertical:hover{{background:{t.scrollbar_hover};}}"
        "QScrollBar::add-line:vertical{height:0;}"
        "QScrollBar::sub-line:vertical{height:0;}"
        "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{"
        "background:transparent;}"
    )


class OptionalFold(QWidget):
    """Page-level Optional disclosure. Starts closed so the first view is
    only the required path; click the header to reveal extras underneath.

    fill_leftover=True (Inference): the body is an inner scroller
    that eats leftover column space so rows stay 42px instead of squashing.

    fill_leftover=False (Training / Iterative Ensemble): the body sizes to
    its content at full column width — no 16px scrollbar lane."""
    toggled = Signal(bool)

    def __init__(self, title="Optional", spacing=6, parent=None, fill_leftover=True):
        super().__init__(parent)
        self.setObjectName("optionalFold")
        self._fill_leftover = fill_leftover
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding if fill_leftover else QSizePolicy.Preferred,
        )
        self._expanded = False
        self._spacing = spacing
        root = QVBoxLayout(self)
        # Same 19px that sits under CONFIGURATION — above the OPTIONAL
        # title and again before its first row (every page that shows it).
        root.setContentsMargins(0, 19, 0, 0)
        root.setSpacing(19)
        root.setAlignment(Qt.AlignTop)

        header = _HelpHeader(self)
        header.setObjectName("optionalFoldHeader")
        header.setFixedHeight(24)
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hl = QHBoxLayout(header)
        # 14px right margin matches cfgRow so this `>` stacks with Quality /
        # Stems / Device (those rows use a 24×24 arrow and 14px inset).
        hl.setContentsMargins(0, 0, 14, 0)
        hl.setSpacing(0)
        t = theme_manager.theme
        self._lbl = QLabel(title.upper())
        self._lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
            f"color:{t.text_muted};background:transparent;padding-left:8px;"
            f"border-left:3px solid {t.border_visible};letter-spacing:1.5px;"
        )
        hl.addWidget(self._lbl, 1)
        self._chevron = _HelpChevron()
        hl.addWidget(self._chevron, 0, Qt.AlignVCenter)
        self._header = header
        header.clicked.connect(self._toggle)
        header.installEventFilter(self)
        header.setToolTip("Show optional settings")
        root.addWidget(header)

        self._body = QWidget()
        self._body.setObjectName("optionalFoldBody")
        self._body.setStyleSheet("background:transparent;")
        self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.body = QVBoxLayout(self._body)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(spacing)
        self.body.setAlignment(Qt.AlignTop)
        self.body.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        # Leftover column height (maximized Inference) must land below the
        # cards, not as gaps between them.
        self.body.addStretch(1)

        if fill_leftover:
            self._scroll = QScrollArea()
            self._scroll.setObjectName("optionalFoldScroll")
            self._scroll.setWidgetResizable(True)
            self._scroll.setFrameShape(QFrame.NoFrame)
            self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._scroll.setStyleSheet(_optional_scroll_ss(gutter=True))
            self._scroll.setWidget(self._body)
            root.addWidget(self._scroll, 1)
        else:
            # No inner QScrollArea: a viewport with horizontal scroll off
            # clips Resume / Augmentation / Edit Configuration to the right
            # (their min width exceeds the pane, while CONFIGURATION rows
            # above are laid out at full column width). The page scroller
            # already handles overflow.
            self._scroll = QWidget()
            self._scroll.setObjectName("optionalFoldScroll")
            self._scroll.setStyleSheet("background:transparent;border:none;")
            self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            wrap = QVBoxLayout(self._scroll)
            wrap.setContentsMargins(0, 0, 0, 0)
            wrap.setSpacing(0)
            wrap.addWidget(self._body)
            root.addWidget(self._scroll, 0)
        self._apply()

    def addWidget(self, w, stretch=0):
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.body.insertWidget(max(0, self.body.count() - 1), w, stretch)
        w.installEventFilter(self)
        self._lock_width_to_body()

    def addLayout(self, layout, stretch=0):
        self.body.insertLayout(max(0, self.body.count() - 1), layout, stretch)
        self._lock_width_to_body()

    def _lock_width_to_body(self):
        """Keep the fold (and its column) as wide as the open cards, even
        while OPTIONAL is collapsed — otherwise Training's left column
        jumps when Resume / Augmentation appear."""
        if self._fill_leftover:
            return
        w = max(self._body.minimumSizeHint().width(), self._body.sizeHint().width())
        if w > 0:
            self.setMinimumWidth(w)

    def showEvent(self, event):
        super().showEvent(event)
        self._lock_width_to_body()

    def is_expanded(self):
        return self._expanded

    def eventFilter(self, obj, event):
        if obj is self._header:
            if event.type() == QEvent.Enter:
                self._chevron.set_hovered(True)
            elif event.type() == QEvent.Leave:
                self._chevron.set_hovered(False)
        elif event.type() in (QEvent.Show, QEvent.Hide) and obj is not self._header:
            self._refresh_body_min()
        return super().eventFilter(obj, event)

    def _content_h(self):
        heights = []
        for i in range(self.body.count()):
            item = self.body.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                if w.isHidden():
                    continue
                h = w.minimumHeight() or w.sizeHint().height()
                heights.append(max(h, _OPTIONAL_ROW_H) if h > 0 else _OPTIONAL_ROW_H)
                continue
            lay = item.layout()
            if lay is not None:
                h = lay.sizeHint().height()
                if h > 0:
                    heights.append(h)
        if not heights:
            return 0
        return sum(heights) + self._spacing * (len(heights) - 1)

    def _min_viewport_h(self):
        content = self._content_h()
        if content <= 0:
            return _OPTIONAL_ROW_H
        if not self._fill_leftover:
            return content
        return min(content, _OPTIONAL_ROW_H)

    def _refresh_body_min(self):
        content = self._content_h()
        self._body.setMinimumHeight(content)
        if self._expanded:
            self._scroll.setMinimumHeight(self._min_viewport_h())
            if not self._fill_leftover:
                self._scroll.setMaximumHeight(max(content, _OPTIONAL_ROW_H))
        else:
            self._scroll.setMinimumHeight(0)
            if not self._fill_leftover:
                self._scroll.setMaximumHeight(0)
        self.updateGeometry()

    def _toggle(self):
        self._expanded = not self._expanded
        self._apply()
        self.toggled.emit(self._expanded)

    def _apply(self):
        self._body.setVisible(self._expanded)
        self._scroll.setVisible(True)
        if self._fill_leftover:
            # Keep the scroll pane in the layout even while collapsed — it
            # eats leftover column space *below* the header so OPTIONAL stays
            # parked under Device instead of floating in the gap.
            self._scroll.setMaximumHeight(16777215)
            self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            if self._expanded:
                self._refresh_body_min()
                self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                self._header.setToolTip("Hide optional settings")
            else:
                self._scroll.setMinimumHeight(0)
                self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self._header.setToolTip("Show optional settings")
        else:
            self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            if self._expanded:
                self._refresh_body_min()
                self._header.setToolTip("Hide optional settings")
            else:
                self._scroll.setMinimumHeight(0)
                self._scroll.setMaximumHeight(0)
                self._header.setToolTip("Show optional settings")
        self._chevron.set_expanded(self._expanded)
        self.updateGeometry()

    def reapply_theme(self):
        t = theme_manager.theme
        self._lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
            f"color:{t.text_muted};background:transparent;padding-left:8px;"
            f"border-left:3px solid {t.border_visible};letter-spacing:1.5px;"
        )
        self._chevron.update()
        if self._fill_leftover:
            self._scroll.setStyleSheet(_optional_scroll_ss(gutter=True))


# Side pads 20+14, square chip 24, gap 12, plus the 32px column gutter
# when REQUIRED and OPTIONAL sit side by side.
_HELP_STEP_CHROME = 20 + 14 + 24 + 12
_HELP_COL_GUTTER = 32
_HELP_MIN_W = 440
_HELP_MAX_W = 960
_HELP_PAD_L = 20
_HELP_PAD_R = 14
_HELP_PAD_Y = 20


def _help_line_px(text, font):
    fm = QFontMetrics(font)
    widest = 0
    for line in (text or "").split("\n"):
        line = line.strip()
        if line:
            widest = max(widest, fm.boundingRect(line).width())
    return widest


def _help_col_px(items, caption, font):
    widest = _help_line_px(caption or "", font)
    for step in items or []:
        widest = max(widest, _help_line_px(step, font))
    return widest


def _help_preferred_width(data, parent=None):
    """Dialog width from the two side-by-side columns (or one if a list
    is empty), clamped to the parent window so a short page stays compact."""
    font = QFont(FONT_FAMILY_DEFAULT, 13)
    intro_w = _help_line_px(data.get("intro") or "", font)
    req = data.get("required") or []
    opt = data.get("optional") or []
    req_w = _help_col_px(req, data.get("required_caption") or HELP_REQUIRED_CAPTION, font)
    opt_w = _help_col_px(opt, data.get("optional_caption") or HELP_OPTIONAL_CAPTION, font)
    if req and opt:
        content = req_w + _HELP_STEP_CHROME + _HELP_COL_GUTTER + opt_w + _HELP_STEP_CHROME
    else:
        content = max(req_w, opt_w) + _HELP_STEP_CHROME
    content = max(content, intro_w + _HELP_PAD_L + _HELP_PAD_R)
    title_font = QFont(FONT_FAMILY_DEFAULT, UIConstants.SEC_TITLE_FONT_SIZE)
    title_font.setBold(True)
    title = f"HELP  ·  {data.get('title', '')}"
    title_w = _HELP_PAD_L + QFontMetrics(title_font).horizontalAdvance(title) + 8 + 24 + _HELP_PAD_R
    min_w = 640 if req and opt else _HELP_MIN_W
    w = max(content, title_w, min_w)
    max_w = _HELP_MAX_W
    win = parent.window() if parent is not None else None
    if win is not None and win.width() > 280:
        max_w = min(max_w, max(min_w, win.width() - 64))
    return int(min(w, max_w))


class PageHelpDialog(QDialog):
    """Frameless Help sheet — How Ensemble Works layout: title + ×,
    REQUIRED and OPTIONAL side by side, no card chrome."""

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setObjectName("pageHelpDialog")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setModal(True)
        self._drag_pos = None
        t = theme_manager.theme
        # Split border-* — Qt QSS cannot parse `border:1px solid rgba(...)`.
        self.setStyleSheet(
            f"QDialog#pageHelpDialog{{background:{t.bg};"
            "border-width:1px;border-style:solid;"
            f"border-color:{t.border_dim};border-radius:8px;}}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(_HELP_PAD_L, _HELP_PAD_Y, _HELP_PAD_R, _HELP_PAD_Y)
        root.setSpacing(0)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)
        title = QLabel(f"HELP  ·  {data.get('title', '')}".strip())
        title.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_TITLE_FONT_SIZE}px;"
            f"font-weight:bold;color:{t.text};background:transparent;letter-spacing:1.5px;"
        )
        hdr.addWidget(title)
        hdr.addStretch()
        close = QPushButton("\u2715")
        close.setFixedSize(24, 24)
        close.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_muted};"
            "border:none;font-size:11px;border-radius:4px;}"
            f"QPushButton:hover{{background:{_help_error_hover()};color:{t.error};}}"
        )
        close.clicked.connect(self.accept)
        hdr.addWidget(close)
        root.addLayout(hdr)
        root.addSpacing(16)

        sc = QScrollArea()
        sc.setWidgetResizable(False)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setStyleSheet(_optional_scroll_ss())
        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        body = QVBoxLayout(inner)
        body.setContentsMargins(0, 0, 10, 0)
        body.setSpacing(0)

        intro = QLabel(data.get("intro", ""))
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "font-family:'Montserrat';font-size:13px;"
            f"color:{t.text_sec};background:transparent;"
        )
        body.addWidget(intro)
        body.addSpacing(24)

        required = data.get("required") or []
        optional = data.get("optional") or []
        cols = QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(_HELP_COL_GUTTER)
        cols.setAlignment(Qt.AlignTop)
        n = 1
        if required:
            cols.addWidget(_help_section(
                "required", required,
                data.get("required_caption") or HELP_REQUIRED_CAPTION,
                index=n,
            ), 1, Qt.AlignTop)
            n += 1
        if required and optional:
            vdiv = QFrame()
            vdiv.setFixedWidth(1)
            vdiv.setStyleSheet(f"background:{t.border_dim};border:none;")
            cols.addWidget(vdiv)
        if optional:
            cols.addWidget(_help_section(
                "optional", optional,
                data.get("optional_caption") or HELP_OPTIONAL_CAPTION,
                index=n,
            ), 1, Qt.AlignTop)
        body.addLayout(cols)
        body.addStretch()
        sc.setWidget(inner)
        root.addWidget(sc, 1)

        self._help_data = data
        self._sc = sc
        self._inner = inner
        self._hdr = hdr
        self._fit_to_content()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and e.pos().y() < 48:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and e.buttons() == Qt.LeftButton:
            self.move(e.globalPos() - self._drag_pos)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None
        super().mouseReleaseEvent(e)

    def _refit(self):
        self._fit_to_content()

    def _fit_to_content(self, data=None, sc=None, inner=None, footer=None):
        data = data if data is not None else self._help_data
        sc = sc if sc is not None else self._sc
        inner = inner if inner is not None else self._inner
        width = _help_preferred_width(data, self.parentWidget())
        inner_w = max(200, width - _HELP_PAD_L - _HELP_PAD_R)
        inner.setMinimumSize(0, 0)
        inner.setMaximumSize(16777215, 16777215)
        inner.setFixedWidth(inner_w)
        lay = inner.layout()
        if lay is not None:
            lay.invalidate()
            lay.activate()
        inner.adjustSize()
        content_h = max(inner.sizeHint().height(), inner.minimumSizeHint().height())
        if inner.hasHeightForWidth():
            hfw = inner.heightForWidth(inner_w)
            if hfw > 0:
                content_h = max(content_h, hfw)
        inner.setFixedSize(inner_w, content_h)

        header_h = 24
        chrome = _HELP_PAD_Y + _HELP_PAD_Y + header_h + 16

        max_h = 820
        scr = self.screen()
        if scr is not None:
            max_h = min(max_h, max(chrome + 160, int(scr.availableGeometry().height() * 0.88)))
        if content_h <= max_h - chrome:
            sc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setFixedSize(inner_w, content_h)
            self.setFixedSize(width, chrome + content_h)
        else:
            view_h = max_h - chrome
            sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sc.setFixedSize(inner_w, view_h)
            self.setFixedSize(width, max_h)


def show_page_help(parent, page_key):
    data = PAGE_HELP.get(page_key)
    if not data:
        return
    run_blurred_dialog(PageHelpDialog(data, parent))


class PageHeader(QWidget):
    """mvsep-style page header: left accent bar, big uppercase title, and
    a subtitle with an accent-highlighted phrase.
    Optional back chevron sits after the bar, left of the title;
    extra widgets dock on the right. Pass help_key to dock a Help '?'
    as the rightmost extra (Required vs Optional to-do for that page)."""

    def __init__(self, title, subtitle="", highlight="", back=False, parent=None,
                 help_key=None):
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

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        self.back_btn = None
        if back:
            self.back_btn = _BackChevron()
            title_row.addWidget(self.back_btn, 0, Qt.AlignVCenter)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:32px;font-weight:bold;color:"
            f"{theme_manager.theme.text};background:transparent;border:none;letter-spacing:-0.5px;"
        )
        title_row.addWidget(self.title_lbl)
        col.addLayout(title_row)

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
            if back:
                # Layout spacer — not QLabel contentsMargins. Parent
                # stylesheets (Auto/Manual header wrappers) polish children
                # and reset widget margins, which left those subtitles flush
                # left while Iterative (no wrapper stylesheet) stayed indented.
                sub_row = QHBoxLayout()
                sub_row.setContentsMargins(0, 0, 0, 0)
                sub_row.setSpacing(0)
                sub_row.addSpacing(24 + title_row.spacing())
                sub_row.addWidget(self.sub_lbl, 1)
                col.addLayout(sub_row)
            else:
                col.addWidget(self.sub_lbl)

        root.addLayout(col, 1)

        self._help_btn = None
        if help_key:
            self._help_btn = HelpButton(self)
            self._help_btn.clicked.connect(
                lambda: show_page_help(self.window() or self, help_key))
            root.addWidget(self._help_btn, 0, Qt.AlignVCenter)

    def add_extra(self, widget):
        # Help stays the rightmost extra so the '?' is always where
        # users look for it, even when Log / GitHub / badges are added.
        if self._help_btn is not None:
            idx = self.layout().indexOf(self._help_btn)
            self.layout().insertWidget(idx, widget)
        else:
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


class SyncingLabel(QLabel):
    """Muted two-line sync caption with a soft opacity pulse while active."""

    _TEXT = "Syncing metrics...\nPlease wait..."

    def __init__(self, parent=None):
        super().__init__(self._TEXT, parent)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        font = QFont("Montserrat", 9)
        font.setWeight(QFont.Weight(600))
        self.setFont(font)
        self.setStyleSheet(
            "font-family:'Montserrat';font-size:9px;font-weight:600;"
            f"color:{theme_manager.theme.text_muted};background:transparent;"
            "line-height:120%;")
        fm = QFontMetrics(font)
        lines = self._TEXT.split("\n")
        w = max(fm.horizontalAdvance(line) for line in lines)
        h = fm.lineSpacing() * len(lines)
        self.setFixedSize(w, h)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._effect.setOpacity(0.0)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(1800)
        self._anim.setStartValue(0.35)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim.setLoopCount(-1)

    def start(self):
        self._anim.start()

    def stop(self):
        self._anim.stop()
        self._effect.setOpacity(0.0)


class ScoresRefreshButton(QPushButton):
    """Square sync icon — re-downloads mvsep score listings."""

    # Material-style sync mark (24×24 viewBox) — rendered via SVG so it
    # stays crisp at 28×28; hand-drawn arcs clipped or vanished at this size.
    _SYNC_PATH = (
        "M12,4V1L8,5l4,4V6c3.31,0 6,2.69 6,6c0,1.01 -0.25,1.97 -0.7,2.8"
        "l1.46,1.46C19.54,15.03 20,13.57 20,12c0,-4.42 -3.58,-8 -8,-8"
        "m0,14c-3.31,0 -6,-2.69 -6,-6c0,-1.01 0.25,-1.97 0.7,-2.8"
        "L5.24,7.74C4.46,8.97 4,10.43 4,12c0,4.42 3.58,8 8,8v3l4,-4l-4,-4v3z"
    )
    _ICON_PX = 16
    _ICON_ROT = 90
    _pix_cache = {}

    def __init__(self, parent=None, tooltip=None):
        super().__init__(parent)
        self._hovered = False
        self._busy = False
        self._spin_deg = 0.0
        self.setFixedSize(28, 28)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip or (
            "Refresh mvsep Quality Checker scores\n"
            "(new models, updated links)."))
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(40)
        self._spin_timer.timeout.connect(self._tick_spin)
        self._apply_style()

    def _apply_style(self):
        t = theme_manager.theme
        # Match Check For Updates / GitHub header buttons.
        self.setStyleSheet(
            f"QPushButton{{background:{t.surface};"
            f"color:{t.text_dim};"
            f"border:1px solid {t.border_dim};border-radius:4px;}}"
            + add_button_hover()
            + f"QPushButton:disabled{{color:{t.disabled_text};}}")

    @classmethod
    def _icon_pixmap(cls, color):
        # Cache opaque glyphs; alpha comes from the theme token at paint time
        # (SVG fill alpha is unreliable across Qt builds).
        key = (color.red(), color.green(), color.blue())
        pix = cls._pix_cache.get(key)
        if pix is None:
            from PySide6.QtCore import QByteArray
            from PySide6.QtSvg import QSvgRenderer
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                f'<path fill="rgb({color.red()},{color.green()},{color.blue()})" '
                f'd="{cls._SYNC_PATH}"/></svg>'
            )
            renderer = QSvgRenderer(QByteArray(svg.encode()))
            img = QImage(64, 64, QImage.Format.Format_ARGB32)
            img.fill(Qt.transparent)
            painter = QPainter(img)
            painter.setRenderHint(QPainter.Antialiasing)
            renderer.render(painter)
            painter.end()
            pix = QPixmap.fromImage(img)
            cls._pix_cache[key] = pix
        return pix

    def set_busy(self, busy: bool):
        self._busy = bool(busy)
        if self._busy:
            self._spin_timer.start()
        else:
            self._spin_timer.stop()
            self._spin_deg = 0.0
        self.update()

    def _tick_spin(self):
        self._spin_deg = (self._spin_deg + 10.0) % 360.0
        self.update()

    def mousePressEvent(self, e):
        if self._busy:
            return
        super().mousePressEvent(e)

    def enterEvent(self, e):
        if not self._busy:
            self._hovered = True
            self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        t = theme_manager.theme
        if self._hovered and not self._busy:
            color = css_color(t.text)
        else:
            color = css_color(t.text_dim)
        icon = self._ICON_PX
        x = (self.width() - icon) // 2
        y = (self.height() - icon) // 2
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.translate(self.width() / 2.0, self.height() / 2.0)
        p.rotate(self._ICON_ROT + (self._spin_deg if self._busy else 0.0))
        p.translate(-self.width() / 2.0, -self.height() / 2.0)
        p.setOpacity(color.alphaF())
        p.drawPixmap(x, y, icon, icon, self._icon_pixmap(color))
        p.end()

