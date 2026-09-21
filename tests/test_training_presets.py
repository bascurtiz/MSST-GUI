"""Training presets: vocal metrics, stem-shift freeze, YAML quick tips."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton  # noqa: E402

from backend import msst_catalog as catalog  # noqa: E402
from ui.pages.training_page import (  # noqa: E402
    _ConfigEditorDialog, _MultiSelectDialog, _RunOptionsDialog,
)
from ui.widgets.common import (  # noqa: E402
    _HELP_PAD_L, _HELP_PAD_R, _HELP_PAD_Y, _HELP_SEC_PAD_X, _HELP_SEC_PAD_Y,
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

    def _outer_pad(dialog):
        m = dialog.layout().contentsMargins()
        return m.left(), m.right(), m.top(), m.bottom()

    def _col_pad(dialog):
        body, _vl = dialog._col_body()
        m = body.layout().contentsMargins()
        body.deleteLater()
        return m.left(), m.right(), m.top(), m.bottom()

    dlg = _MultiSelectDialog(
        "VALIDATION METRICS", catalog.metric_choices(), ["sdr"],
        hint="test", caption="cap")
    app.processEvents()
    check(_outer_pad(dlg) == (_HELP_PAD_L, _HELP_PAD_L, _HELP_PAD_Y, _HELP_PAD_Y),
          f"metrics sheet outer margin {_outer_pad(dlg)}")
    check(_col_pad(dlg) == (_HELP_SEC_PAD_X, _HELP_SEC_PAD_X, 8, _HELP_SEC_PAD_Y),
          f"metrics list inset {_col_pad(dlg)}")
    dlg._apply_preset(catalog.VOCAL_TUNING_METRICS)
    sel = set(dlg.selected())
    check(sel == set(catalog.VOCAL_TUNING_METRICS),
          f"vocal tuning preset -> {sel}")
    dlg.close()

    loss = _MultiSelectDialog(
        "LOSS FUNCTIONS", catalog.loss_choices(), catalog.default_losses(),
        hint="test", caption="cap")
    app.processEvents()
    check(_outer_pad(loss) == (_HELP_PAD_L, _HELP_PAD_R, _HELP_PAD_Y, _HELP_PAD_Y),
          f"loss sheet outer margin {_outer_pad(loss)}")
    check(_col_pad(loss) == (_HELP_SEC_PAD_X, _HELP_SEC_PAD_X, 8, _HELP_SEC_PAD_Y),
          f"loss list inset {_col_pad(loss)}")
    check(not loss._help_data.get("hide_scrollbar"),
          "loss sheet does not force-hide the scrollbar")
    loss._help_data["max_height"] = 400
    loss._refit()
    app.processEvents()
    check(loss._sc.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded,
          "loss sheet shows a scrollbar when the list is taller than the sheet")
    loss.close()

    run = _RunOptionsDialog({}, None)
    app.processEvents()
    check(_outer_pad(run) == (_HELP_PAD_L, _HELP_PAD_L, _HELP_PAD_Y, _HELP_PAD_Y),
          f"gpus/workers/seed outer margin {_outer_pad(run)}")
    check(_col_pad(run) == (_HELP_SEC_PAD_X, _HELP_SEC_PAD_X, 8, _HELP_SEC_PAD_Y),
          f"gpus/workers/seed list inset {_col_pad(run)}")
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
