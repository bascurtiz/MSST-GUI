"""HiDPI helpers: Windows Per-Monitor V2, fractional Qt scale, DPR pixmaps.

UVR5 (Tkinter) opts *out* of DPI awareness so Windows bitmap-stretches a
1080p-calibrated window — large at 125%, and blurry. We do the opposite:
declare Per-Monitor V2, keep Qt's PassThrough factor (1.25 stays 1.25),
and paint logos/icons at device pixels so 125% stays sharp.

Default size stays near 1280x800 so the two-column layout does not stretch
into empty space on 1440p. 1920x1080 @ 125% (~1536x864, ~816 after the
taskbar) still fits that size out of the box; larger desktops only bump
a little, never toward filling the screen.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QPixmap

# Windows DPI_AWARENESS_CONTEXT values (HANDLE / intptr).
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
# PROCESS_PER_MONITOR_DPI_AWARE (SetProcessDpiAwareness fallback).
_PROCESS_PER_MONITOR_DPI_AWARE = 2

PREFERRED_WINDOW_W = 1280
PREFERRED_WINDOW_H = 800
# Logical available width above 1080p @ 100%. 1440p @ 100% is 2560; @ 125%
# is ~2048. Grow a little past preferred, then stop — 72% of 2560 made the
# Inference columns look like empty slabs.
_GROW_FROM_W = 1920
MAX_DEFAULT_W = 1400
MAX_DEFAULT_H = 880

HIDPI_ROUNDING_POLICY = Qt.HighDpiScaleFactorRoundingPolicy.PassThrough


def enable_windows_dpi_awareness() -> bool:
    """Ask Windows for Per-Monitor V2 so the process is never bitmap-scaled.

    Must run before any HWND exists (before QApplication). Returns True if
    a DPI-awareness call succeeded. Failure is non-fatal: Qt 6 still tries
    on its own, and a later call is ignored once awareness is locked.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ctx = ctypes.c_void_p(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctx):
            return True
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE)
        return True
    except Exception:
        return False


def apply_hidpi_rounding() -> None:
    """Keep 125% as 1.25 instead of rounding to 1.0 (too small) or 2.0."""
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(HIDPI_ROUNDING_POLICY)


def enable_hidpi() -> None:
    """Process-wide HiDPI bootstrap. Call once, before QApplication()."""
    enable_windows_dpi_awareness()
    apply_hidpi_rounding()


def current_dpr(widget=None) -> float:
    """Device pixel ratio for a widget, else the primary screen, else 1.0."""
    if widget is not None:
        try:
            dpr = float(widget.devicePixelRatioF())
            if dpr > 0:
                return dpr
        except Exception:
            pass
    app = QGuiApplication.instance()
    if app is not None:
        screen = app.primaryScreen()
        if screen is not None:
            try:
                dpr = float(screen.devicePixelRatio())
                if dpr > 0:
                    return dpr
            except Exception:
                pass
    return 1.0


def _positive_dpr(dpr) -> float:
    try:
        value = float(dpr)
    except (TypeError, ValueError):
        value = 1.0
    return value if value > 0 else 1.0


def make_pixmap(logical_w, logical_h, dpr=None) -> QPixmap:
    """Allocate a pixmap covering `logical_*` CSS pixels at `dpr`."""
    dpr = _positive_dpr(dpr if dpr is not None else current_dpr())
    pw = max(1, int(round(float(logical_w) * dpr)))
    ph = max(1, int(round(float(logical_h) * dpr)))
    pm = QPixmap(pw, ph)
    pm.setDevicePixelRatio(dpr)
    return pm


def scale_pixmap(pm, logical_width=None, logical_height=None, dpr=None):
    """Scale a source pixmap to logical size at `dpr` (sharp at 125%/150%)."""
    if pm is None or pm.isNull():
        return pm
    dpr = _positive_dpr(dpr if dpr is not None else current_dpr())
    if logical_width is None and logical_height is None:
        scaled = QPixmap(pm)
        scaled.setDevicePixelRatio(dpr)
        return scaled
    if logical_height is not None and logical_width is None:
        target = max(1, int(round(float(logical_height) * dpr)))
        scaled = pm.scaledToHeight(target, Qt.SmoothTransformation)
    elif logical_width is not None and logical_height is None:
        target = max(1, int(round(float(logical_width) * dpr)))
        scaled = pm.scaledToWidth(target, Qt.SmoothTransformation)
    else:
        scaled = pm.scaled(
            max(1, int(round(float(logical_width) * dpr))),
            max(1, int(round(float(logical_height) * dpr))),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    scaled.setDevicePixelRatio(dpr)
    return scaled


def default_window_size(avail_w, avail_h):
    """Pick a default window size that fits `availableGeometry`.

    1080p @ 100% (~1920x1040) and 1080p @ 125% (~1536x816) stay 1280x800
    when that rectangle fits. Wider logical desktops (1440p @ 100%/125%)
    bump toward MAX_DEFAULT_* so the window is not a postage stamp, then
    stop — never 72% of a 2560-wide desktop.
    """
    avail_w = max(1, int(avail_w))
    avail_h = max(1, int(avail_h))
    w = min(PREFERRED_WINDOW_W, avail_w)
    h = min(PREFERRED_WINDOW_H, avail_h)
    if avail_w > _GROW_FROM_W:
        w = min(avail_w, max(w, MAX_DEFAULT_W))
        h = min(avail_h, max(h, MAX_DEFAULT_H))
    return w, h


# Logical-pixel thickness of the invisible resize grips. Kept small so the
# close/max buttons still get most of their click target.
RESIZE_BORDER = 6

# 8 grips: edges first, then corners. Matches Qt.Edge flags used by
# QWindow.startSystemResize (no WS_THICKFRAME, so no legacy Win32 frame).
GRIP_EDGES = (
    Qt.LeftEdge,
    Qt.RightEdge,
    Qt.TopEdge,
    Qt.BottomEdge,
    Qt.LeftEdge | Qt.TopEdge,
    Qt.RightEdge | Qt.TopEdge,
    Qt.LeftEdge | Qt.BottomEdge,
    Qt.RightEdge | Qt.BottomEdge,
)


def grip_rects(width, height, border=RESIZE_BORDER):
    """Return 8 (x, y, w, h) rects for left/right/top/bottom + 4 corners."""
    b = max(1, int(border))
    w = max(0, int(width))
    h = max(0, int(height))
    inner_w = max(0, w - 2 * b)
    inner_h = max(0, h - 2 * b)
    right = max(0, w - b)
    bottom = max(0, h - b)
    return (
        (0, b, b, inner_h),
        (right, b, b, inner_h),
        (b, 0, inner_w, b),
        (b, bottom, inner_w, b),
        (0, 0, b, b),
        (right, 0, b, b),
        (0, bottom, b, b),
        (right, bottom, b, b),
    )


def grip_cursor(edges):
    """Cursor shape for a Qt.Edges resize grip."""
    left = bool(edges & Qt.LeftEdge)
    right = bool(edges & Qt.RightEdge)
    top = bool(edges & Qt.TopEdge)
    bottom = bool(edges & Qt.BottomEdge)
    if (left and top) or (right and bottom):
        return Qt.SizeFDiagCursor
    if (right and top) or (left and bottom):
        return Qt.SizeBDiagCursor
    if left or right:
        return Qt.SizeHorCursor
    return Qt.SizeVerCursor
