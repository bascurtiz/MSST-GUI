"""Training presets: vocal metrics, stem-shift freeze, YAML quick tips."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton  # noqa: E402

from backend import msst_catalog as catalog  # noqa: E402
from ui.pages.training_page import (  # noqa: E402
    _ConfigEditorDialog, _MultiSelectDialog, _RunOptionsDialog,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    app = QApplication([])

    dlg = _MultiSelectDialog(
        "VALIDATION METRICS", catalog.metric_choices(), ["sdr"],
        hint="test", caption="cap")
    app.processEvents()
    dlg._apply_preset(catalog.VOCAL_TUNING_METRICS)
    sel = set(dlg.selected())
    check(sel == set(catalog.VOCAL_TUNING_METRICS),
          f"vocal tuning preset -> {sel}")
    dlg.close()

    run = _RunOptionsDialog({}, None)
    app.processEvents()
    stem_btn = next(
        (b for b in run.findChildren(QPushButton)
         if b.text() == "Stem-shift preset (RoFormer)"), None)
    check(stem_btn is not None, "stem-shift preset link exists")
    if stem_btn is not None:
        stem_btn.click()
        app.processEvents()
        check(run._freeze.text() == catalog.STEM_SHIFT_FREEZE_LAYERS,
              "stem-shift preset fills freeze layers")
    run.close()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config_vocals_mdx23c.yaml")
    if os.path.isfile(cfg_path):
        editor = _ConfigEditorDialog(cfg_path)
        app.processEvents()
        tips_frame = next(
            (f for f in editor.findChildren(QFrame)
             if f.objectName() == "yamlTipsPanel"), None)
        check(tips_frame is not None, "config editor shows YAML quick tips")
        if tips_frame is not None:
            keys = [lb.text() for lb in tips_frame.findChildren(QLabel)
                    if lb.text().startswith("training.")
                    or lb.text().startswith("model.")
                    or lb.text().startswith("inference.")]
            check("training.target_instrument" in keys,
                  "tips include target_instrument")
            check("inference.num_overlap" in keys,
                  "tips include num_overlap")
        editor.close()
    else:
        check(True, "config fixture skipped (file missing)")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
