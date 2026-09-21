"""Help dialog: every page lists Required vs Optional to-dos separately."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton, QWidget

from ui.strings import PAGE_HELP, HELP_REQUIRED_CAPTION, HELP_OPTIONAL_CAPTION
from ui.theme import theme_manager
from ui.widgets.common import (
    HelpButton, PageHeader, PageHelpDialog, OptionalFold, _HELP_SCROLL_LANE,
    _HELP_PAD_L, _HELP_PAD_R, _HELP_PAD_Y, _HELP_SEC_PAD_X, _HELP_SEC_PAD_Y,
    css_color,
)
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
    m = dlg.layout().contentsMargins()
    check(
        "help sheet has extra outer margin",
        m.left() == _HELP_PAD_L and m.right() == _HELP_PAD_R
        and m.top() == _HELP_PAD_Y and m.bottom() == _HELP_PAD_Y,
    )
    req = dlg.findChild(QFrame, "helpRequired")
    sm = req.layout().contentsMargins()
    check(
        "REQUIRED section has inner inset",
        sm.left() == _HELP_SEC_PAD_X and sm.right() == _HELP_SEC_PAD_X
        and sm.top() == _HELP_SEC_PAD_Y and sm.bottom() == _HELP_SEC_PAD_Y,
    )
    opt = dlg.findChild(QFrame, "helpOptional")
    om = opt.layout().contentsMargins()
    check(
        "OPTIONAL section has the same inner inset",
        om.left() == sm.left() and om.right() == sm.right()
        and om.top() == sm.top() and om.bottom() == sm.bottom(),
    )

    three = PageHelpDialog({
        "title": "TEST",
        "heading": "GPUS / WORKERS / SEED  ·  TEST",
        "intro": "Three columns.",
        "required": ["Pick a GPU."],
        "middle": ["Set workers."],
        "optional": ["Flip a flag."],
        "required_title": "GPUS",
        "middle_title": "WORKERS / SEED",
        "required_caption": "Devices.",
        "middle_caption": "DataLoader.",
        "optional_caption": "Flags.",
    })
    three_labels = [l.text() for l in three.findChildren(QLabel)]
    check("three-col GPUS badge", "GPUS" in three_labels)
    check("three-col WORKERS / SEED badge", "WORKERS / SEED" in three_labels)
    check("three-col OPTIONAL badge", "OPTIONAL" in three_labels)
    check(
        "three-col keeps two required panes",
        len([f for f in three.findChildren(QFrame)
             if f.objectName() == "helpRequired"]) == 2,
    )

    stacked = PageHelpDialog({
        "title": "TEST",
        "heading": "GPUS / WORKERS / SEED  ·  TEST",
        "intro": "Optional below.",
        "required": ["Pick a GPU."],
        "middle": ["Set workers."],
        "optional": ["Flip a flag."],
        "required_title": "GPUS",
        "middle_title": "WORKERS / SEED",
        "required_caption": "Devices.",
        "middle_caption": "DataLoader.",
        "optional_caption": "Flags.",
        "optional_below": True,
    })
    stacked.show()
    app.processEvents()
    stacked_labels = [l.text() for l in stacked.findChildren(QLabel)]
    check("stacked OPTIONAL badge", "OPTIONAL" in stacked_labels)
    check(
        "stacked does not grow a third side pane",
        len([f for f in stacked.findChildren(QFrame)
             if f.objectName() == "helpRequired"]) == 2
        and len([f for f in stacked.findChildren(QFrame)
                 if f.objectName() == "helpOptional"]) == 1,
    )

    def _lab_pos(dlg, text):
        for lab in dlg.findChildren(QLabel):
            if lab.text() == text:
                return lab.mapTo(dlg, lab.rect().topLeft())
        return None

    gpus = _lab_pos(stacked, "GPUS")
    opt = _lab_pos(stacked, "OPTIONAL")
    check("stacked optional is below GPUS",
          gpus is not None and opt is not None and opt.y() > gpus.y() + 20)
    stacked.close()

    tall = PageHelpDialog({
        "title": "TEST",
        "heading": "SCROLL  ·  TEST",
        "intro": "Tall enough to grow a vertical bar.",
        "required": ["Pick a GPU."],
        "middle": ["Set workers."],
        "optional": [f"Optional step {i}." for i in range(40)],
        "required_title": "GPUS",
        "middle_title": "WORKERS / SEED",
        "required_caption": "Devices.",
        "middle_caption": "DataLoader.",
        "optional_caption": "Flags.",
        "optional_below": True,
    })
    tall.show()
    app.processEvents()
    check(
        "tall sheet enables a vertical bar",
        tall._sc.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded,
    )
    check(
        "inner width leaves the scrollbar lane",
        tall._inner.width() + _HELP_SCROLL_LANE <= tall._sc.width() + 1,
    )
    tall.close()

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

    GUIDE_URL = "https://msst-guide.pages.dev/"
    train_intro = PAGE_HELP["training"]["intro"]
    check("training intro keeps the guide URL as plain text", GUIDE_URL in train_intro)
    train = PageHelpDialog(PAGE_HELP["training"])
    check("training intro is rich text", train._intro.textFormat() == Qt.RichText)
    check("training intro opens the guide in a browser", train._intro.openExternalLinks())
    check(
        "training intro wraps the guide URL",
        f'href="{GUIDE_URL}"' in train._intro.text(),
    )
    check(
        "training intro uses accent for links",
        train._intro.palette().color(QPalette.ColorRole.Link).name().lower()
        == css_color(theme_manager.accent).name().lower(),
    )
    train.close()

    train_opt = PAGE_HELP["training"]["optional"][0]
    train_lines = train_opt.split("\n")
    check(
        "training splits losses onto the next line",
        train_lines[0].endswith("optimizer,")
        and train_lines[1].startswith("losses, and metrics"),
    )
    check(
        "training splits YAML onto the next line",
        train_lines[2].startswith("YAML values"),
    )
    wizard_opt = PAGE_HELP["training"]["optional"][5]
    wizard_lines = wizard_opt.split("\n")
    check(
        "training splits Fit GPU onto the next line",
        wizard_lines[0].endswith("wandb,")
        and wizard_lines[1].startswith("Fit GPU, and Train."),
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
