"""
ui/theme.py — mvsep.com-inspired theme (dark + light), Montserrat font.

Colors and fonts are modelled on mvsep.com's visual identity (sampled from
screenshots of the site):    * Font: Montserrat (bundled in resources/, SIL OFL 1.1)
    * Accent: mvsep brand blue #0F7FB3, white text on accent buttons
  * Light mode (default, like the site): #F7F8FA page, white cards,
    navy #203048 headings, light-blue-gray disabled buttons
  * Dark mode: blue-tinted grays with the same brand blue accent
"""
import os

from PySide6.QtCore import QObject, Signal, Qt, QEvent, SignalInstance
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QLabel, QWidget, QAbstractButton, QComboBox,
    QAbstractSpinBox, QSlider, QTabBar, QLineEdit, QMenu,
)

import backend.settings as settings_store


FONT_FAMILY = "Montserrat"
FONT_STACK = "'Montserrat','Segoe UI',Arial,sans-serif"
MONO_STACK = "'Courier New','Consolas',monospace"

# Bundled font files live directly in the resources folder.
_FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources",
)

# ── Palettes ─────────────────────────────────────────────────────────────────

class _DarkTheme:
    """mvsep.com dark palette (brand blue accent on dark grays)."""
    name = "dark"
    bg = "#14161A"
    bg_deep = "#0E1013"
    surface = "#1D2127"
    surface_alt = "#262B33"
    card = "#191D22"
    header_bg = "#101318"
    text = "#E8EDF3"
    text_sec = "rgba(232,237,243,0.72)"
    text_dim = "rgba(232,237,243,0.50)"
    text_muted = "rgba(232,237,243,0.38)"
    text_label = "rgba(232,237,243,0.26)"
    border = "rgba(255,255,255,0.09)"
    border_visible = "#2E3640"
    border_dim = "rgba(255,255,255,0.15)"
    input_bg = "#1D2127"
    input_hover = "#2C323B"
    scrollbar_bg = "rgba(255,255,255,0.04)"
    scrollbar_handle = "rgba(255,255,255,0.18)"
    tooltip_bg = "#232831"
    tooltip_text = "#E8EDF3"
    tooltip_border = "rgba(255,255,255,0.14)"
    # Menus stay dark in both themes (same always-dark look as tooltips).
    menu_bg = "#232831"
    menu_text = "#E8EDF3"
    menu_hover = "rgba(15,127,179,0.30)"
    menu_sep = "rgba(255,255,255,0.08)"
    menu_border = "rgba(255,255,255,0.14)"
    menu_disabled = "rgba(232,237,243,0.40)"
    console_bg = "#0E1013"
    console_text = "#C3CCD6"
    disabled_bg = "#2A2F37"
    disabled_text = "#6B7480"
    success = "#4CC38A"
    warning = "#FFB020"
    error = "#FF5C5C"
    purple = "#B388FF"
    arch_dot_vr = "#4FC3F7"
    arch_dot_mdx = "#FF7043"
    arch_dot_demucs = "#AB47BC"
    arch_dot_bs = "#66BB6A"
    arch_dot_melband = "#FFCA28"
    arch_dot_scnet = "#EC407A"
    arch_dot_apollo = "#42A5F5"
    arch_dot_bandit = "#8D6E63"
    accent = "#0F7FB3"
    accent_hover = "#0B6A99"
    accent_soft = "rgba(15,127,179,0.18)"
    accent_glow = "rgba(15,127,179,0.35)"
    accent_text = "#FFFFFF"


class _LightTheme:
    """mvsep.com light palette (sampled from their site: bg #F7F8FA, brand blue #0F7FB3)."""
    name = "light"
    bg = "#F7F8FA"
    bg_deep = "#EEF1F5"
    surface = "#FFFFFF"
    surface_alt = "#F0F3F6"
    card = "#FFFFFF"
    header_bg = "#FFFFFF"
    text = "#203048"
    text_sec = "rgba(32,48,72,0.66)"
    text_dim = "rgba(32,48,72,0.48)"
    text_muted = "rgba(32,48,72,0.38)"
    text_label = "rgba(32,48,72,0.28)"
    border = "rgba(0,0,0,0.16)"
    border_visible = "#D9E1E8"
    border_dim = "rgba(0,0,0,0.22)"
    input_bg = "#FFFFFF"
    input_hover = "#FFFFFF"
    scrollbar_bg = "rgba(0,0,0,0.05)"
    scrollbar_handle = "rgba(0,0,0,0.20)"
    # Tooltips and menus keep the dark appearance in both themes (white text
    # on a dark surface), matching mvsep.com's always-dark look.
    tooltip_bg = "#232831"
    tooltip_text = "#E8EDF3"
    tooltip_border = "rgba(255,255,255,0.14)"
    menu_bg = "#232831"
    menu_text = "#E8EDF3"
    menu_hover = "rgba(15,127,179,0.30)"
    menu_sep = "rgba(255,255,255,0.08)"
    menu_border = "rgba(255,255,255,0.14)"
    menu_disabled = "rgba(232,237,243,0.40)"
    console_bg = "#FFFFFF"
    console_text = "#3A4452"
    disabled_bg = "#F4F8F8"
    disabled_text = "#A7B2BE"
    success = "#2E9E5B"
    warning = "#E08A00"
    error = "#D93025"
    purple = "#7C4DFF"
    arch_dot_vr = "#0288D1"
    arch_dot_mdx = "#E64A19"
    arch_dot_demucs = "#8E24AA"
    arch_dot_bs = "#2E7D32"
    arch_dot_melband = "#F9A825"
    arch_dot_scnet = "#C2185B"
    arch_dot_apollo = "#1565C0"
    arch_dot_bandit = "#6D4C41"
    accent = "#0F7FB3"
    accent_hover = "#0B6A99"
    accent_soft = "rgba(15,127,179,0.13)"
    accent_glow = "rgba(15,127,179,0.30)"
    accent_text = "#FFFFFF"


_THEMES = {"dark": _DarkTheme, "light": _LightTheme}


# ── Global stylesheet (rendered per theme from resources/style.qss) ─────────

_QSS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "style.qss",
)


def build_stylesheet(theme) -> str:
    """Render the global QSS template with the given theme's tokens.

    Uses literal replacement instead of str.format because the stylesheet
    itself is full of CSS braces that format() would misinterpret.
    """
    with open(_QSS_PATH, "r", encoding="utf-8") as f:
        s = f.read()
    token_attrs = (
        "bg", "bg_deep", "surface", "surface_alt", "card", "header_bg",
        "text", "text_sec", "text_dim", "text_muted", "text_label",
        "border", "border_visible", "border_dim", "input_bg", "input_hover",
        "scrollbar_bg", "scrollbar_handle", "tooltip_bg", "tooltip_text",
        "tooltip_border", "menu_bg", "menu_text", "menu_hover", "menu_sep",
        "menu_border", "menu_disabled", "console_bg", "console_text", "disabled_bg",
        "disabled_text", "accent", "accent_hover", "accent_soft",
        "accent_glow", "accent_text",
    )
    s = s.replace("{font_family}", FONT_STACK)
    s = s.replace("{mono_family}", MONO_STACK)
    for attr in token_attrs:
        s = s.replace("{" + attr + "}", getattr(theme, attr))
    return s


# ── Theme manager ────────────────────────────────────────────────────────────

class ThemeManager(QObject):
    """Holds the active palette and re-applies the global stylesheet on switch."""

    theme_changed = Signal()
    # Emitted after the new palette is active but BEFORE the global stylesheet
    # is re-applied, so listeners can shed weight first (e.g. the main window
    # removes the pages that are about to be rebuilt anyway). Re-polishing a
    # small tree instead of the whole one makes switching dramatically faster.
    theme_about_to_change = Signal()

    def __init__(self):
        super().__init__()
        self._mode = "dark"  # mvsep.com's default look, and ours
        self.theme = _DarkTheme()
        self._apply_accent()

    # — public API (backward compatible) —
    @property
    def mode(self):
        return self._mode

    def init_app(self, app):
        register_fonts()
        apply_palette(app)

    def set_mode(self, mode, persist=True):
        mode = mode if mode in _THEMES else "dark"
        if mode == self._mode:
            self._apply_global()
            return
        self._mode = mode
        self.theme = _THEMES[mode]()
        self._apply_accent()
        if persist:
            self._persist()
        self.theme_about_to_change.emit()
        self._apply_global()
        self.theme_changed.emit()

    def toggle(self, persist=True):
        self.set_mode("light" if self._mode == "dark" else "dark", persist=persist)

    def set_accent(self, hex_str):
        """Override the accent color (kept for API compatibility)."""
        self.theme.accent = hex_str
        self._apply_accent()
        self._apply_global()

    def reset_default(self):
        self.set_mode("dark")

    # — internals —
    def _apply_accent(self):
        self.accent = self.theme.accent
        self._accent_hover = self.theme.accent_hover
        self._accent_soft = self.theme.accent_soft
        self._accent_glow = self.theme.accent_glow
        self._accent_text = self.theme.accent_text

    def _persist(self):
        try:
            data = settings_store.load()
            data.setdefault("ui", {})["theme"] = self._mode
            settings_store.save(data)
        except Exception:
            pass

    def _apply_global(self):
        app = QApplication.instance()
        if app is None:
            return
        app.setStyleSheet(build_stylesheet(self.theme))
        for w in app.topLevelWidgets():
            try:
                w.style().unpolish(w)
                w.style().polish(w)
            except Exception:
                pass


theme_manager = ThemeManager()

# Backward compatibility: main.py used to import a static STYLESHEET.
STYLESHEET = build_stylesheet(_DarkTheme())


# ── Fonts ────────────────────────────────────────────────────────────────────

def register_fonts():
    """Register bundled Montserrat faces (Regular/Bold) with Qt."""
    from PySide6.QtGui import QFontDatabase

    for fname in ("Montserrat-Regular.ttf", "Montserrat-Bold.ttf"):
        path = os.path.join(_FONT_DIR, fname)
        if os.path.isfile(path):
            QFontDatabase.addApplicationFont(path)


def apply_palette(app):
    """Register Montserrat and set it as the application font."""
    register_fonts()
    font = QFont(FONT_FAMILY, 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    install_styled_tooltips()
    install_interactive_cursors()


# ── Interactive cursors ──────────────────────────────────────────────────────
# Qt stylesheets cannot set cursors, so every interactive widget gets
# Qt.PointingHandCursor programmatically. The ChildAdded filter covers
# widgets created at any time (model cards, rows, dialogs) without touching
# every construction site.

_INTERACTIVE_BASES = (
    QAbstractButton,   # QPushButton, QToolButton, QCheckBox, QRadioButton…
    QComboBox,
    QAbstractSpinBox,  # QSpinBox / QDoubleSpinBox (knobs)
    QSlider,
    QTabBar,
    QLineEdit,         # clickable input fields
    QMenu,             # popup menus (items show the menu widget's cursor)
)

# Custom widgets that handle mouse presses but don't expose a `clicked`
# signal (matched by class name to avoid importing page modules here).
_CLICKABLE_CLASS_NAMES = {
    "_ComboRow",       # inference: row click opens the combo popup
    "_ModelItem",      # inference: row click selects the model
    "_OutputStemsRow", # inference: row click opens the stem dialog
}


def _is_interactive(widget):
    if isinstance(widget, _INTERACTIVE_BASES):
        return True
    if isinstance(getattr(widget, "clicked", None), SignalInstance):
        return True
    return widget.__class__.__name__ in _CLICKABLE_CLASS_NAMES


class _InteractiveCursorFilter(QObject):
    """Hands a hand-cursor to interactive widgets when they are polished."""

    def eventFilter(self, obj, event):
        # QEvent.Polish is sent to every widget after its constructor
        # finishes (and again on every style/theme repolish), so it reliably
        # covers widgets created at any time — including ones created with a
        # visible parent, where ChildAdded is unreliable — and it re-asserts
        # the cursor after theme switches.
        if event.type() == QEvent.Type.Polish:
            if isinstance(obj, QWidget) and _is_interactive(obj):
                obj.setCursor(Qt.PointingHandCursor)
        return False


_CURSOR_FILTER_INSTALLED = False


def install_interactive_cursors():
    """Install the app-wide hand-cursor filter (idempotent)."""
    global _CURSOR_FILTER_INSTALLED
    if _CURSOR_FILTER_INSTALLED:
        return
    app = QApplication.instance()
    if app is None:
        return
    app.installEventFilter(_InteractiveCursorFilter(app))
    # Cover anything that already exists (e.g. created before this ran),
    # without clobbering deliberately-set cursors.
    for w in app.allWidgets():
        if _is_interactive(w) and w.cursor().shape() == Qt.CursorShape.ArrowCursor:
            w.setCursor(Qt.PointingHandCursor)
    _CURSOR_FILTER_INSTALLED = True


# ── Styled tooltip ───────────────────────────────────────────────────────────
# QToolTip's default rendering ignores the app stylesheet on some platforms,
# so every tooltip is drawn by this custom label: a dark pill with white text
# in both themes, matching mvsep.com's always-dark tooltip.

class _StyledToolTip(QLabel):
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()
        self._apply_style()

    def _apply_style(self):
        t = theme_manager.theme
        self.setStyleSheet(
            f"background:{t.tooltip_bg};color:{t.tooltip_text};"
            f"border:1px solid {t.tooltip_border};padding:4px 8px;"
            "border-radius:4px;font-size:11px;"
        )

    def show_tip(self, pos, text):
        self.setText(text)
        self._apply_style()
        self.adjustSize()
        self.move(pos.x() + 2, pos.y() + 18)
        self.show()
        self.raise_()

    def hide_tip(self):
        self.hide()


class _ToolTipFilter(QObject):
    """Application-wide filter that intercepts real hover tooltips
    (QEvent.ToolTip) and shows the custom styled tooltip instead, so Qt's
    default tooltip rendering never appears."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tip = _StyledToolTip.instance()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.ToolTip:
            # Qt never populates QHelpEvent.tip(); the widget's own toolTip()
            # property is the actual text (QWidget::event uses it directly).
            text = obj.toolTip() if hasattr(obj, "toolTip") else ""
            if text:
                self._tip.show_tip(event.globalPos(), text)
            else:
                self._tip.hide_tip()
            return True  # consume: Qt's default tooltip never shows
        if event.type() in (QEvent.Type.Leave, QEvent.Type.MouseButtonPress,
                            QEvent.Type.WindowDeactivate):
            self._tip.hide_tip()
        return False


_FILTER_INSTALLED = False


def install_styled_tooltips():
    """Install the app-wide tooltip filter (idempotent)."""
    global _FILTER_INSTALLED
    if _FILTER_INSTALLED:
        return
    app = QApplication.instance()
    if app is None:
        return
    app.installEventFilter(_ToolTipFilter(app))
    theme_manager.theme_changed.connect(_StyledToolTip.instance()._apply_style)
    _FILTER_INSTALLED = True


# ── Design Tokens (source-of-truth constants) ───────────────────────────────

class UIConstants:
    """Single source of truth for repeated UI sizing/shaping values.

    Font families point at Montserrat (bundled); sizing matches mvsep.com's
    rounded, compact controls.
    """
    # ── Buttons ──
    BTN_HEIGHT = 36
    BTN_RADIUS = 6
    BTN_FONT_FAMILY = "'Montserrat',sans-serif"
    BTN_FONT_SIZE = 11
    BTN_PADDING_H = 16
    BTN_MIN_WIDTH = 80

    # ── Primary action (Start / Stop / Run) ──
    ACTION_RADIUS = 8
    ACTION_FONT_FAMILY = "'Montserrat',sans-serif"
    ACTION_FONT_SIZE = 11

    # ── Cards ──
    CARD_RADIUS_STYLESHEET = 8
    CARD_RADIUS_PAINT = 16
    CARD_MARGIN_TOP = 12
    CARD_MARGIN_BOTTOM = 14
    CARD_MARGIN_LEFT = 20
    CARD_MARGIN_RIGHT = 20

    # ── Section headers ──
    SEC_HDR_FONT_FAMILY = "'Montserrat',sans-serif"
    SEC_HDR_FONT_SIZE = 9
    SEC_HDR_PADDING_LEFT = 8
    SEC_HDR_HEIGHT = 18
    SEC_TITLE_FONT_SIZE = 13  # step titles (INPUT / PROCESS / OUTPUT)

    # ── Field labels & values ──
    FIELD_LABEL_FONT_FAMILY = "'Montserrat',sans-serif"
    FIELD_LABEL_FONT_SIZE = 9
    FIELD_VALUE_FONT_FAMILY = "'Montserrat',sans-serif"
    FIELD_VALUE_FONT_SIZE = 11

    # ── Input fields / combos ──
    INPUT_HEIGHT = 48
    INPUT_RADIUS = 6

    # ── Tags / badges ──
    TAG_RADIUS = 3
    TAG_PADDING_V = 1
    TAG_PADDING_H = 6
    TAG_FONT_SIZE = 8
    TAG_FONT_FAMILY = "'Montserrat',sans-serif"
    TAG_HEIGHT = 18

    # ── Icons ──
    ICON_SIZE = 24
    ICON_SIZE_SMALL = 20
    ICON_DOT = 8

    # ── Layout ──
    PAGE_MARGIN_LR = 28
    PAGE_MARGIN_TOP = 16
    PAGE_MARGIN_BOTTOM = 12
    SECTION_SPACING = 24
    INNER_SPACING = 14
    ITEM_SPACING = 8
    GRID_SPACING = 8

    # ── Separators ──
    SEP_HEIGHT = 1

    # ── Scrollbar ──
    SCROLLBAR_WIDTH = 4
    SCROLLBAR_RADIUS = 2
    SCROLLBAR_MIN_HANDLE = 30

    # ── Action bar ──
    ACTION_BAR_MARGIN_LR = 32
    ACTION_BAR_MARGIN_TOP = 8
    ACTION_BAR_MARGIN_BOTTOM = 10
    ACTION_BAR_SPACING = 10

    # ── Tabs (nav) ──
    TAB_PADDING_LR = 20
    TAB_FONT_SIZE = 11
    TAB_FONT_FAMILY = "'Montserrat',sans-serif"
    TAB_SPACING_BOTTOM = 6

    # ── Header bar ──
    HEADER_HEIGHT = 52
    HEADER_MARGIN_LEFT = 28
    HEADER_MARGIN_RIGHT = 16
