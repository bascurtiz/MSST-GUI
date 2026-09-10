"""Small regressions for inference options and startup window hygiene."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QWidget

from utils.settings import parse_args_inference
from main import _SuppressUntitledWindows
from ui.pages.inference_page import (
    _SearchBar, _SortCombo, SEARCH_FIELD_WIDTH, SORT_COMBO_WIDTH,
)


def main():
    app = QApplication.instance() or QApplication([])
    failures = []

    args = parse_args_inference({"bigshifts": 4})
    if args.bigshifts != 4:
        failures.append("parse_args_inference preserves --bigshifts")
    defaults = parse_args_inference({})
    if defaults.bigshifts != 1:
        failures.append("Big Shifts defaults to one pass")

    # Both page headers use the same compact control dimensions: the folder
    # placeholder must fit in full, while the sort pill stays the inference
    # width when reused by Settings.
    search = _SearchBar("Search folder...")
    sort = _SortCombo()
    if search.sizeHint().width() != SEARCH_FIELD_WIDTH:
        failures.append("shared search field width")
    if sort.width() != SORT_COMBO_WIDTH:
        failures.append("shared sort combo width")
    search.deleteLater()
    sort.deleteLater()

    filt = _SuppressUntitledWindows()
    stray = QWidget()
    stray.setWindowTitle("MSST")
    stray.show()
    filt.eventFilter(stray, QEvent(QEvent.Type.Show))
    app.processEvents()
    if stray.isVisible():
        failures.append("untitled MSST helper window is hidden")
    stray.deleteLater()

    splash = QWidget()
    splash.setObjectName("startupSplash")
    splash.setWindowTitle("MSST")
    splash.show()
    filt.eventFilter(splash, QEvent(QEvent.Type.Show))
    app.processEvents()
    if not splash.isVisible():
        failures.append("named startup splash remains visible")
    splash.close()
    splash.deleteLater()
    app.processEvents()

    from ui.theme import _StyledToolTip, _ToolTipFilter
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtCore import QPoint
    tip = _StyledToolTip.instance()
    tip.show()
    filt.eventFilter(tip, QEvent(QEvent.Type.Show))
    app.processEvents()
    if not tip.isVisible():
        failures.append("styled tooltip window is not suppressed")
    tip.hide()

    for msg in failures:
        print("FAIL " + msg)
    if not failures:
        print("ALL CHECKS PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
