"""Headless verification of theme-toggle coverage.

Builds the real MainWindow offscreen and drives the real theme pipeline
(ThemeManager.set_mode -> MainWindow._on_theme_changed -> _rebuild_pages),
then asserts that every page and key child widgets actually restyled.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.theme import theme_manager  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def page_bg_tokens(mw):
    """Per-page stylesheet strings after the current theme state."""
    pages = {
        "inference": mw.inference_page,
        "training": mw.training_page,
        "ensemble_landing": mw.ensemble_landing,
        "auto_ensemble": mw.auto_ensemble,
        "manual_ensemble": mw.manual_ensemble,
        "iterative_ensemble": mw.iterative_ensemble,
        "console": mw.console_page,
        "settings": mw.settings_page,
    }
    return {k: (w.styleSheet() or "") for k, w in pages.items()}


def pump_until(app, cond, timeout_s=10.0):
    """Run the event loop until cond() is true (chunked rebuilds are async)."""
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        if cond():
            return True
    return False


def find_stem_type_button(page):
    from ui.pages.auto_ensemble_page import _StemTypeButton
    return page.findChild(_StemTypeButton)


def main():
    app = QApplication.instance() or QApplication([])
    theme_manager.init_app(app)
    mw = MainWindow()

    check("startup theme is dark", theme_manager.mode == "dark")
    check("central bg uses dark token",
          theme_manager.theme.bg in mw._central.styleSheet())

    # ── switch to light and back via the real pipeline ──────────────────
    theme_manager.set_mode("light")
    check("chunked rebuild completed (to light)", pump_until(app, lambda: not mw._switching))
    theme_manager.set_mode("dark")
    check("chunked rebuild completed", pump_until(app, lambda: not mw._switching))
    dark_bg = theme_manager.theme.bg
    check("mode is dark", theme_manager.mode == "dark")
    check("central bg switched to dark", dark_bg in mw._central.styleSheet())
    check("header bg switched to dark",
          theme_manager.theme.header_bg in mw._header.styleSheet())

    bgs = page_bg_tokens(mw)
    # The iterative page paints its background live from theme tokens in
    # _AtmosphericBackground.paintEvent (empty top-level stylesheet by design).
    for name, ss in bgs.items():
        if name == "iterative_ensemble":
            continue
        check(f"page '{name}' restyled after toggle (rebuild)",
              ss != "" and dark_bg in ss,
              )
    from ui.pages.iterative_ensemble_page import _AtmosphericBackground
    atmos = mw.iterative_ensemble.findChild(_AtmosphericBackground)
    check("iterative page has live-painted background", atmos is not None)

    # ── stem type buttons: name/count labels must carry valid token
    #    colors (regression for the missing-f-prefix bug) ─────────────────
    btn = find_stem_type_button(mw.auto_ensemble)
    check("stem type button exists", btn is not None)
    if btn is not None:
        from ui.pages.auto_ensemble_page import _StemTypeButton
        name_lbl = btn.findChild(type(btn._name_lbl))
        btn.setChecked(False) if hasattr(btn, "setChecked") else None
        btn.set_selected(False)
        btn._update_style()
        nl = btn._name_lbl.styleSheet()
        cl = btn._count_lbl.styleSheet()
        check("unselected name label QSS has no raw braces",
              "{" not in nl or "theme_manager" not in nl)
        check("unselected name label carries theme token color",
              theme_manager.theme.text_sec in nl)
        check("unselected count label carries theme token color",
              theme_manager.theme.text_muted in cl)
        btn.set_selected(True)
        sel = btn._name_lbl.styleSheet()
        check("selected name label uses accent text", sel.count("#") >= 1)

    # ── model card CUSTOM badge uses theme token ────────────────────────
    from ui.pages.auto_ensemble_page import _ModelCard
    card = mw.auto_ensemble.findChild(_ModelCard)
    if card is not None:
        card._apply_badge_styles()
        badge_ss = card._official_badge.styleSheet()
        check("OFFICIAL/CUSTOM badge styled from tokens",
              ("arch" not in badge_ss) and ("{" not in badge_ss))

    # ── console log history survives page rebuild ───────────────────────
    marker = "[TEST] history marker line"
    mw.console_page.append_log(marker)
    check("history buffer captured line",
          marker in mw.console_page._LOG_HISTORY)
    old_console_id = id(mw.console_page)
    theme_manager.set_mode("light")  # triggers another full rebuild
    check("chunked rebuild completed (2nd)", pump_until(app, lambda: not mw._switching))
    check("console page was rebuilt",
          id(mw.console_page) != old_console_id)
    check("log history survived rebuild",
          marker in mw.console_page._LOG_HISTORY)
    check("history re-rendered into new console widget",
          marker in mw.console_page._log_edit.toPlainText())

    # ── deferred rebuild while processing ───────────────────────────────
    theme_manager.set_mode("dark")
    check("chunked rebuild completed (3rd)", pump_until(app, lambda: not mw._switching))
    mw._processing = True
    stale_page_id = id(mw.inference_page)
    theme_manager.set_mode("light")
    app.processEvents()
    check("no rebuild while processing", id(mw.inference_page) == stale_page_id)
    check("pending flag set while processing", mw._theme_rebuild_pending)
    mw._processing = False
    mw._on_process_state(False)
    check("chunked rebuild completed (deferred)", pump_until(app, lambda: not mw._switching))
    check("deferred rebuild applied after run ends",
          not mw._theme_rebuild_pending)
    check("pages rebuilt after deferred apply",
          id(mw.inference_page) != stale_page_id)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
