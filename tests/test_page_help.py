"""Help dialog: every page lists Required vs Optional to-dos separately."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton, QWidget

from ui.strings import PAGE_HELP, HELP_REQUIRED_CAPTION, HELP_OPTIONAL_CAPTION
from ui.theme import theme_manager
from ui.widgets.common import HelpButton, PageHeader, PageHelpDialog, OptionalFold
import ui.widgets.common as common


EXPECTED_KEYS = (
    "inference", "training", "ensemble", "auto_ensemble",
    "manual_ensemble", "iterative_ensemble", "console", "settings",
)

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def main():
    app = QApplication.instance() or QApplication([])
    theme_manager.init_app(app)

    check("all page keys present", set(EXPECTED_KEYS) <= set(PAGE_HELP))
    for key in EXPECTED_KEYS:
        data = PAGE_HELP[key]
        check(f"{key} has title", bool(data.get("title")))
        check(f"{key} has intro", bool(data.get("intro")))
        req = data.get("required") or []
        opt = data.get("optional") or []
        check(f"{key} has required steps", len(req) >= 1)
        check(f"{key} has optional steps", len(opt) >= 1)
        check(f"{key} required are strings", all(isinstance(s, str) and s for s in req))
        check(f"{key} optional are strings", all(isinstance(s, str) and s for s in opt))

    hdr = PageHeader("INFERENCE", "TEST SUB", help_key="inference")
    btn = hdr.findChild(HelpButton, "pageHelpBtn")
    check("header docks a Help button", btn is not None)
    extra = QPushButton("Log")
    extra.setObjectName("logExtra")
    hdr.add_extra(extra)
    lay = hdr.layout()
    check(
        "Help stays right of other extras",
        lay.indexOf(btn) > lay.indexOf(extra) > 0,
    )

    bare = PageHeader("X")
    check("no Help without help_key", bare.findChild(HelpButton) is None)

    data = PAGE_HELP["inference"]
    dlg = PageHelpDialog(data)
    labels = [l.text() for l in dlg.findChildren(QLabel)]
    check("dialog shows REQUIRED badge", "REQUIRED" in labels)
    check("dialog shows OPTIONAL badge", "OPTIONAL" in labels)
    check("dialog uses required caption", HELP_REQUIRED_CAPTION in labels)
    check("dialog uses optional caption", HELP_OPTIONAL_CAPTION in labels)
    joined = " ".join(labels)
    check("dialog lists a required inference step", "Choose input audio" in joined)
    check("dialog lists an optional inference step", "Output format" in joined)
    check(
        "required and optional sections are distinct frames",
        dlg.findChild(QFrame, "helpRequired") is not None
        and dlg.findChild(QFrame, "helpOptional") is not None,
    )

    console = PageHelpDialog(PAGE_HELP["console"])
    clabels = [l.text() for l in console.findChildren(QLabel)]
    check(
        "console required caption is page-specific",
        PAGE_HELP["console"]["required_caption"] in clabels,
    )
    check(
        "console does not use the start-action required caption",
        HELP_REQUIRED_CAPTION not in clabels,
    )

    train_opt = PAGE_HELP["training"]["optional"][0]
    check(
        "training splits YAML onto the next line",
        "and metrics —" in train_opt.split("\n")[0]
        and train_opt.split("\n")[1].startswith("YAML values"),
    )
    ens_opt = PAGE_HELP["ensemble"]["optional"][0]
    ens_lines = ens_opt.split("\n")
    check(
        "ensemble puts Auto picks on its own line",
        ens_lines[0].endswith("choosing —")
        and ens_lines[1].startswith("Auto picks compatible models,"),
    )

    inf = PageHelpDialog(PAGE_HELP["inference"])
    ens = PageHelpDialog(PAGE_HELP["ensemble"])
    check("ensemble is shorter than inference", ens.height() < inf.height())
    check("widths follow copy, not a fixed 760 slab",
          inf.width() != 760 or ens.width() != 760)
    check("ensemble is at least as narrow as inference",
          ens.width() <= inf.width())

    opened = []
    real = common.run_blurred_dialog
    common.run_blurred_dialog = lambda dlg: opened.append(dlg)
    try:
        common.show_page_help(None, "inference")
    finally:
        common.run_blurred_dialog = real
    check("help opens a single dialog", len(opened) == 1)

    fold = OptionalFold()
    extra = QLabel("hidden extra")
    extra.setFixedHeight(42)
    fold.addWidget(extra)
    check("page Optional starts collapsed", fold._body.isHidden())
    fold._toggle()
    check("page Optional opens below the header", not fold._body.isHidden() and fold.is_expanded())
    check("page Optional scroll is on while open", not fold._scroll.isHidden())
    check("page Optional keeps full-height cards", extra.minimumHeight() == 42)
    fold._toggle()
    check("page Optional hides again", fold._body.isHidden())
    check("page Optional header stays mounted while closed", not fold._header.isHidden())

    page = OptionalFold(fill_leftover=False)
    tall = QLabel("post-processing")
    tall.setFixedHeight(220)
    page.addWidget(tall)
    page._toggle()
    check(
        "page Optional without leftover fill opens to full content height",
        page._scroll.minimumHeight() >= 220 and page._scroll.maximumHeight() >= 220,
    )

    if FAILURES:
        print("FAILURES:", FAILURES)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
