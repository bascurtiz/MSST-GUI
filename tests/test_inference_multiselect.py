"""Regression test: multi-run selection on the inference page.

The "Batch model mode" button arms multi-select mode: the model rows'
round selectors morph into square checkboxes, a Select all control appears
above the library (with a partial-dash state), and the Separate button then
processes the input with every checked model. Exercises the real
InferencePage offscreen with a stubbed settings store — no torch, no
network, no subprocess:

  * single mode stays radio: checking one model deselects the others,
  * arming multi mode shows Select all and squares the selectors,
  * checking several models keeps them all (no mutual exclusion),
  * unchecking the primary falls back to another checked model,
  * checks mirror across the arch/target groupings,
  * Select all ticks everything and shows the partial dash for a subset,
  * disarming collapses the selection back to the single primary,
  * Separate in multi mode with nothing checked explains instead of running.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt, QPoint  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.theme import theme_manager  # noqa: E402
import ui.pages.inference_page as ip  # noqa: E402
import backend.settings as bs  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


class _NamesFetchThread(ip._NamesFetchThread):
    """Never touch the network in the harness."""

    def start(self):
        pass


class _FakeMsgBox:
    warns = []

    @staticmethod
    def warning(parent, title, text):
        _FakeMsgBox.warns.append(("warning", title, text))
        return None

    @staticmethod
    def information(parent, title, text):
        _FakeMsgBox.warns.append(("info", title, text))
        return None


def _all_items(page):
    items = []
    for card in list(page._arch_cards.values()) + list(page._target_cards.values()):
        for i in range(card._list_vl.count()):
            w = card._list_vl.itemAt(i).widget()
            if isinstance(w, ip._ModelItem):
                items.append(w)
    return items


def _by_name(page):
    return {it._name: it for it in _all_items(page)}


def _checked_names(page):
    """Unique model names checked across both groupings (each model exists
    in the arch grouping AND the target grouping, so rows are mirrored)."""
    return {it._name for it in _all_items(page) if it.is_checked()}


def main():
    app = QApplication([])
    theme_manager.init_app(app)
    ip._NamesFetchThread = _NamesFetchThread
    real_msgbox = ip.QMessageBox
    ip.QMessageBox = _FakeMsgBox
    _FakeMsgBox.warns = []

    tmp = tempfile.mkdtemp(prefix="msst_multiselect_")
    audio = os.path.join(tmp, "in.wav")
    with open(audio, "wb") as f:
        f.write(b"RIFF" + b"\0" * 256)  # never decoded
    out = os.path.join(tmp, "out")
    os.makedirs(out)

    made = []
    for base in ("alpha", "beta", "zeta"):
        for ext in (".ckpt", ".yaml"):
            p = os.path.join(tmp, base + ext)
            with open(p, "w", encoding="utf-8") as f:
                f.write("")
            made.append(p)

    models = [
        {"name": "alpha_mdx", "ckpt": os.path.join(tmp, "alpha.ckpt"),
         "yaml": os.path.join(tmp, "alpha.yaml"),
         "arch": "MDX23c Architecture", "model_type": "mdx23c", "type": "vocals"},
        {"name": "beta_bs", "ckpt": os.path.join(tmp, "beta.ckpt"),
         "yaml": os.path.join(tmp, "beta.yaml"),
         "arch": "BS Roformer Architecture", "model_type": "bs_roformer",
         "type": "multi stems"},
        {"name": "zeta_bs", "ckpt": os.path.join(tmp, "zeta.ckpt"),
         "yaml": os.path.join(tmp, "zeta.yaml"),
         "arch": "BS Roformer Architecture", "model_type": "bs_roformer",
         "type": "multi stems"},
    ]

    real_load = bs.load

    def fake_load():
        d = dict(real_load())
        d["registered_models"] = list(models)
        return d

    def _register_all(page):
        """The app's main window feeds registered models into the page via
        on_model_registered; the harness must do the same so library rows
        exist before select-all / batch are exercised."""
        for m in models:
            page.on_model_registered(m)
        page._flush_library_visibility()

    bs.load = fake_load
    try:
        page = ip.InferencePage()
        _register_all(page)
        page._input_row.set_value([audio])
        page._output_row.set_value(out)

        # 1. Defaults: single mode — radio circles, no Select all.
        check("button reads 'Multi-select Models'",
              page.btn_test_all.text() == "Multi-select Models")
        tt = page.btn_test_all.toolTip()
        check("tooltip comes from the strings registry, short, per-sentence",
              tt == ip.T_MULTI_SELECT_MODE and tt.count("\n") == 1
              and len(tt) < 140)
        check("starts in single mode", page._multi_mode is False)
        check("single mode: Select all row hidden",
              page._sel_all_row.isHidden())
        items = _all_items(page)
        # Each model exists twice (arch grouping + target grouping).
        check("three distinct models in the library",
              len({it._name for it in items}) == 3
              and len(items) == 6)
        check("single mode: selectors are circles",
              items and all(not it._circle._square for it in items))

        # 2. Single mode stays radio: checking one deselects everything else.
        _by_name(page)["alpha_mdx"]._on_circle_toggled(True)
        check("single mode: check selects the primary",
              page._selected_model
              and page._selected_model["name"] == "alpha_mdx")
        check("single mode: exactly one model checked",
              _checked_names(page) == {"alpha_mdx"})
        check("single mode: primary mirrors into the target grouping",
              len([it for it in _all_items(page)
                   if it._name == "alpha_mdx" and it.is_checked()]) == 2)

        # 3. Arm multi mode: checkboxes + Select all appear.
        page._toggle_multi_mode()
        check("multi mode armed", page._multi_mode is True)
        check("multi mode: button reflects armed state",
              page.btn_test_all.isChecked())
        check("multi mode: Select all row shown",
              not page._sel_all_row.isHidden())
        check("multi mode: selectors are square checkboxes",
              all(it._circle._square for it in _all_items(page)))
        # alpha was checked in single mode and stays checked on arm.
        check("Separate label shows the batch count on arm",
              page.btn_run._text_lbl.text() == "Separate (1/3)")

        # 4. Multi-select: checking a second model keeps the first.
        _by_name(page)["beta_bs"]._on_circle_toggled(True)
        check("multi mode: two models checked simultaneously",
              _checked_names(page) == {"alpha_mdx", "beta_bs"})
        check("multi mode: primary is the last checked",
              page._selected_model
              and page._selected_model["name"] == "beta_bs")
        check("multi mode: a check mirrors into both groupings",
              len([it for it in _all_items(page)
                   if it._name == "beta_bs" and it.is_checked()]) == 2)
        check("Separate label follows the checked count",
              page.btn_run._text_lbl.text() == "Separate (2/3)")

        # 4b. Unchecking one model drops the count on the button.
        _by_name(page)["alpha_mdx"]._on_circle_toggled(False)
        check("Separate label drops after an uncheck",
              page.btn_run._text_lbl.text() == "Separate (1/3)")
        _by_name(page)["alpha_mdx"]._on_circle_toggled(True)
        check("Separate label rises after a re-check",
              page.btn_run._text_lbl.text() == "Separate (2/3)")

        # 5. Unchecking the primary falls back to another checked model.
        _by_name(page)["beta_bs"]._on_circle_toggled(False)
        check("primary falls back to the remaining checked model",
              page._selected_model
              and page._selected_model["name"] == "alpha_mdx")

        # 6. Select all ticks every model (full state, not partial).
        page._on_select_all_toggled(True)
        check("Select all ticks every model",
              _checked_names(page) == {"alpha_mdx", "beta_bs", "zeta_bs"})
        check("Select all shows the full state",
              page._sel_all.is_checked() and not page._sel_all.is_partial())

        # 7. Dropping one model flips Select all to the partial dash.
        _by_name(page)["zeta_bs"]._on_circle_toggled(False)
        check("partial Select all after unchecking one",
              page._sel_all.is_checked() and page._sel_all.is_partial())
        checked = page._checked_models()
        check("checked-models list matches the selection",
              sorted(m["name"] for m in checked) == ["alpha_mdx", "beta_bs"])
        check("checked-models carry the engine type",
              all(m.get("model_type") for m in checked))

        # 8. Disarming collapses back to the single primary.
        page._toggle_multi_mode()
        check("multi mode disarmed", page._multi_mode is False)
        check("Select all hidden again", page._sel_all_row.isHidden())
        check("collapse keeps only the primary checked",
              _checked_names(page) == {"alpha_mdx"}
              and page._selected_model
              and page._selected_model["name"] == "alpha_mdx")
        check("Separate label returns to plain text on disarm",
              page.btn_run._text_lbl.text() == "Separate")

        # 9. Separate in multi mode with nothing checked explains itself.
        page._toggle_multi_mode()
        page._on_select_all_toggled(False)
        warns_before = len(_FakeMsgBox.warns)
        page._run()
        check("no-selection run shows an info dialog",
              len(_FakeMsgBox.warns) == warns_before + 1
              and _FakeMsgBox.warns[-1][0] == "info"
              and "No models selected" in _FakeMsgBox.warns[-1][1])
        check("no-selection run keeps multi mode armed",
              page._multi_mode is True)

        # 10. Unchecking the primary in single mode clears the selection.
        page._toggle_multi_mode()  # back to single mode
        _by_name(page)["alpha_mdx"]._on_circle_toggled(False)
        check("single mode: unchecking the primary clears it",
              page._selected_model is None)

        # 11. Physical clicks toggle exactly once (checkbox AND row). A press
        # on the checkbox used to propagate up to the row's own press handler
        # and toggle twice — on then off in the same click — so per-model
        # checks in batch mode flickered away.
        item = ip._ModelItem("alpha_mdx", "", "", "MDX23c Architecture",
                             "vocals", "mdx23c", "", False)
        item.resize(400, 38)
        item.show()
        toggles = []
        item._circle.toggled.connect(lambda v: toggles.append(v))
        QTest.mouseClick(item._circle, Qt.LeftButton, pos=QPoint(9, 9))
        check("checkbox click checks exactly once",
              item.is_checked() is True and toggles == [True])
        toggles.clear()
        QTest.mouseClick(item, Qt.LeftButton, pos=QPoint(200, 19))
        check("row click unchecks exactly once",
              item.is_checked() is False and toggles == [False])
        toggles.clear()
        QTest.mouseClick(item._circle, Qt.LeftButton, pos=QPoint(9, 9))
        check("checkbox click re-checks exactly once",
              item.is_checked() is True and toggles == [True])
        item.deleteLater()
        app.processEvents()

        # 12. The batch selection survives a restart: save_settings stores
        # the armed flag + checked names + primary, and a fresh page restores
        # them after the models re-register (rows exist once the visibility
        # flush runs, the same hook startup uses).
        page._toggle_multi_mode()            # re-arm
        page._on_select_all_toggled(False)   # clear
        _by_name(page)["alpha_mdx"]._on_circle_toggled(True)
        _by_name(page)["beta_bs"]._on_circle_toggled(True)  # becomes primary
        saved = page.save_settings()
        ms = saved.get("multi_select", {})
        check("save stores the armed flag", ms.get("armed") is True)
        check("save stores the checked batch",
              ms.get("batch") == ["alpha_mdx", "beta_bs"])
        check("save stores the last primary", ms.get("primary") == "beta_bs")

        page2 = ip.InferencePage()
        page2.load_settings(saved)   # called before rows exist, like _load_all
        _register_all(page2)         # registers models -> flush applies batch
        check("restart: multi mode re-armed", page2._multi_mode is True
              and page2.btn_test_all.isChecked()
              and not page2._sel_all_row.isHidden())
        check("restart: batch checks restored",
              _checked_names(page2) == {"alpha_mdx", "beta_bs"})
        check("restart: primary restored",
              page2._selected_model
              and page2._selected_model["name"] == "beta_bs")
        check("restart: Separate label shows the restored count",
              page2.btn_run._text_lbl.text() == "Separate (2/3)")
        # A second flush must not re-apply (pending is consumed once).
        _checked_names_before = _checked_names(page2)
        page2._flush_library_visibility()
        check("restart: pending batch applied exactly once",
              _checked_names(page2) == _checked_names_before)

        # 13. A non-armed save (plain single mode) restores nothing.
        page2._toggle_multi_mode()  # disarm -> collapse to primary
        saved2 = page2.save_settings()
        page3 = ip.InferencePage()
        page3.load_settings(saved2)
        _register_all(page3)
        check("disarmed save: no auto re-arm on restart",
              page3._multi_mode is False)
        page2.deleteLater()
        page3.deleteLater()
        app.processEvents()
    finally:
        bs.load = real_load
        ip.QMessageBox = real_msgbox

    print(f"RESULT: {len(FAILURES)} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())