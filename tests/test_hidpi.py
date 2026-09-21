"""HiDPI: PassThrough rounding, DPR pixmaps, default window size."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QGuiApplication, QPixmap, QColor, QPainter
from PySide6.QtWidgets import QApplication

from ui.dpi import (
    HIDPI_ROUNDING_POLICY, apply_hidpi_rounding, current_dpr, default_window_size,
    make_pixmap, scale_pixmap, RESIZE_BORDER, GRIP_EDGES, grip_rects, grip_cursor,
    MAX_DEFAULT_W, MAX_DEFAULT_H,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def _fits(size, avail):
    return size[0] <= avail[0] and size[1] <= avail[1]


def main():
    apply_hidpi_rounding()
    app = QApplication.instance() or QApplication([])

    check(
        HIDPI_ROUNDING_POLICY == Qt.HighDpiScaleFactorRoundingPolicy.PassThrough,
        "HIDPI_ROUNDING_POLICY is PassThrough",
    )
    check(
        QGuiApplication.highDpiScaleFactorRoundingPolicy()
        == Qt.HighDpiScaleFactorRoundingPolicy.PassThrough,
        "QGuiApplication rounding policy is PassThrough",
    )

    src = QPixmap(40, 20)
    src.fill(QColor("#0F7FB3"))
    for dpr in (1.0, 1.25, 1.5, 2.0):
        scaled_h = scale_pixmap(src, logical_height=32, dpr=dpr)
        check(not scaled_h.isNull(), f"scale_pixmap height at dpr={dpr} is valid")
        check(
            abs(scaled_h.devicePixelRatio() - dpr) < 1e-6,
            f"scale_pixmap height dpr={dpr} is tagged",
        )
        check(
            scaled_h.height() == max(1, int(round(32 * dpr))),
            f"scale_pixmap height device px at dpr={dpr}: {scaled_h.height()}",
        )

        scaled_w = scale_pixmap(src, logical_width=260, dpr=dpr)
        check(
            scaled_w.width() == max(1, int(round(260 * dpr))),
            f"scale_pixmap width device px at dpr={dpr}: {scaled_w.width()}",
        )

        made = make_pixmap(16, 16, dpr)
        check(
            made.width() == max(1, int(round(16 * dpr)))
            and made.height() == max(1, int(round(16 * dpr))),
            f"make_pixmap 16x16 device size at dpr={dpr}",
        )
        check(
            abs(made.devicePixelRatio() - dpr) < 1e-6,
            f"make_pixmap dpr tagged at {dpr}",
        )
        made.fill(Qt.transparent)
        painter = QPainter(made)
        painter.end()

    # Keep-aspect scale of a 2:1 source to 16x16 logical at 1.25.
    boxed = scale_pixmap(src, logical_width=16, logical_height=16, dpr=1.25)
    check(boxed.width() == 20, f"keep-aspect 16x16@1.25 width is 20, got {boxed.width()}")
    check(boxed.height() == 10, f"keep-aspect 16x16@1.25 height is 10, got {boxed.height()}")

    check(current_dpr() > 0, "current_dpr is positive")

    # 1080p @ 100% (taskbar trimmed).
    s_1080 = default_window_size(1920, 1040)
    check(s_1080 == (1280, 800), f"1080p@100% stays 1280x800, got {s_1080}")
    check(_fits(s_1080, (1920, 1040)), "1080p@100% fits available")

    # 1080p @ 125% — must fit out of the box (1536x864 logical, ~816 after taskbar).
    s_1080_125 = default_window_size(1536, 816)
    check(s_1080_125 == (1280, 800), f"1080p@125% stays 1280x800, got {s_1080_125}")
    check(_fits(s_1080_125, (1536, 816)), "1080p@125% fits available")

    # Tight available: never overflow.
    s_tight = default_window_size(1200, 720)
    check(s_tight == (1200, 720), f"tight available is clamped, got {s_tight}")
    check(_fits(s_tight, (1200, 720)), "tight available still fits")

    # 1440p @ 125% (~2048x1152 logical, ~1100 after taskbar).
    s_1440 = default_window_size(2048, 1100)
    check(s_1440 == (MAX_DEFAULT_W, MAX_DEFAULT_H),
          f"1440p@125% uses the capped default, got {s_1440}")
    check(s_1440[0] > 1280, f"1440p@125% width grows past 1280, got {s_1440}")
    check(s_1440[1] > 800, f"1440p@125% height grows past 800, got {s_1440}")
    check(_fits(s_1440, (2048, 1100)), "1440p@125% fits available")
    check(s_1440[0] <= 1400, f"1440p@125% stays compact, got {s_1440}")

    # 1440p @ 100% (2560x1440, ~1400 after taskbar) — must not fill ~72%.
    s_1440_100 = default_window_size(2560, 1400)
    check(s_1440_100 == (MAX_DEFAULT_W, MAX_DEFAULT_H),
          f"1440p@100% uses the capped default, got {s_1440_100}")
    check(s_1440_100[0] < int(2560 * 0.72),
          f"1440p@100% is not 72% of the desktop, got {s_1440_100}")
    check(_fits(s_1440_100, (2560, 1400)), "1440p@100% fits available")

    rects = grip_rects(100, 80, 6)
    check(len(rects) == 8, f"eight grip rects, got {len(rects)}")
    check(len(GRIP_EDGES) == 8, "eight grip edge flags")
    check(rects[0] == (0, 6, 6, 68), f"left grip {rects[0]}")
    check(rects[1] == (94, 6, 6, 68), f"right grip {rects[1]}")
    check(rects[2] == (6, 0, 88, 6), f"top grip {rects[2]}")
    check(rects[3] == (6, 74, 88, 6), f"bottom grip {rects[3]}")
    check(rects[4] == (0, 0, 6, 6), f"top-left {rects[4]}")
    check(rects[5] == (94, 0, 6, 6), f"top-right {rects[5]}")
    check(rects[6] == (0, 74, 6, 6), f"bottom-left {rects[6]}")
    check(rects[7] == (94, 74, 6, 6), f"bottom-right {rects[7]}")
    check(RESIZE_BORDER == 6, "resize border stays 6px")
    check(grip_cursor(Qt.LeftEdge) == Qt.SizeHorCursor, "left grip is horizontal")
    check(grip_cursor(Qt.TopEdge) == Qt.SizeVerCursor, "top grip is vertical")
    check(
        grip_cursor(Qt.LeftEdge | Qt.TopEdge) == Qt.SizeFDiagCursor,
        "top-left grip is diagonal",
    )
    check(
        grip_cursor(Qt.RightEdge | Qt.TopEdge) == Qt.SizeBDiagCursor,
        "top-right grip is back-diagonal",
    )

    from ui.pages.settings_page import _RadioCheck, _GitHubIconButton
    from ui.theme import theme_manager
    from PySide6.QtWidgets import QLabel
    theme_manager.init_app(app)

    radio = _RadioCheck("MODEL MANAGER", checked=True)
    check(radio._circle.size() == QSize(16, 16),
          f"mode radio stays 16px logical, got {radio._circle.size()}")
    check(not isinstance(radio._circle, QLabel),
          "mode radio is painted, not a clipped QLabel pixmap")

    gh = _GitHubIconButton("https://example.com", "GitHub")
    check(gh.size() == QSize(30, 30), f"GitHub button stays 30px, got {gh.size()}")
    check(gh._ICON_PX == 14, f"GitHub mark is 14px logical, got {gh._ICON_PX}")
    gh_pix = gh._mark_pixmap(QColor(0, 0, 0))
    check(gh_pix.devicePixelRatio() == 1.0,
          f"GitHub mark pixmap is not DPR-tagged, got {gh_pix.devicePixelRatio()}")
    check(gh_pix.width() == 64, f"GitHub mark source is 64px, got {gh_pix.width()}")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
