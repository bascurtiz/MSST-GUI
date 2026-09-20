"""Training Setup wizard: start → model → data → Fit GPU → review."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton  # noqa: E402

from ui.pages.training_page import (  # noqa: E402
    TrainingPage, _TrainSetupDialog, _starter_yamls, _SETUP_STEPS, _ReviewRow,
)
from ui.theme import theme_manager  # noqa: E402
from backend.paths import REPO_ROOT  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    app = QApplication.instance() or QApplication([])
    page = TrainingPage()
    check(hasattr(page, "btn_setup"), "Wizard button exists")
    check(page.btn_setup._text_lbl.text() == "Wizard", "Wizard label")
    check(page.btn_results._lbl.text() == "Open Output", "Open Output on the action bar")
    check(page.btn_logs._lbl.text() == "View Logs", "View Logs label")

    dlg = _TrainSetupDialog(page)
    dlg.show()
    app.processEvents()
    check(dlg.width() >= 640, f"sheet is wide enough for path rows, got {dlg.width()}")
    check("font-size:12px" in dlg._probe_btn.styleSheet()
          and "padding:0 18px" in dlg._probe_btn.styleSheet(),
          "wizard probe button uses 12px type with side padding")
    check(dlg._probe_btn.height() == 40, "wizard probe button is 40px tall")
    check(dlg._stack.count() == len(_SETUP_STEPS), "six setup steps")
    check(dlg._step == 0, "opens on START")
    check(dlg._mode == "scratch", "no checkpoint defaults to scratch")
    check(dlg._choice_scratch.is_on() and not dlg._choice_resume.is_on(),
          "from-scratch card is selected")
    check(dlg._resume_host.isHidden(), "checkpoint row hidden for scratch")
    check(dlg._next_btn is not None and dlg._next_btn.text() == "NEXT",
          "NEXT on the first step")
    check(dlg._back_btn.isHidden(), "BACK hidden on the first step")
    labels = [l.text() for l in dlg.findChildren(QLabel)]
    check("01  START" in labels, "step badge")
    check("WIZARD  ·  TRAINING" in labels, "wizard heading")
    check(len(dlg._step_ticks._pills) == len(_SETUP_STEPS),
          "six step ticks beside the badge")
    check(dlg._step_ticks.parent() is dlg._step_badge.parentWidget(),
          "ticks sit beside the step badge")
    check(theme_manager.accent in dlg._choice_scratch.styleSheet(),
          "selected start card uses accent fill")
    check(theme_manager.accent not in dlg._choice_resume.styleSheet(),
          "unselected start card is not filled")
    dlg._choice_resume._hover = True
    dlg._choice_resume._paint()
    check(theme_manager.accent in dlg._choice_resume.styleSheet(),
          "hover lights up an unselected card")
    dlg._choice_resume._pressed = True
    dlg._choice_resume._paint()
    check(theme_manager.theme.surface_alt in dlg._choice_resume.styleSheet(),
          "press uses surface_alt")
    dlg._choice_resume._pressed = False
    dlg._choice_resume._hover = False
    dlg._choice_resume._paint()

    dlg._set_mode("resume")
    check(not dlg._resume_host.isHidden(), "fine-tune shows checkpoint + catalog")
    dlg._set_mode("scratch")
    dlg._on_next()
    check(dlg._step == 1, "NEXT moves to MODEL")
    check(dlg._step_ticks._index == 1, "ticks follow the MODEL step")
    check("02  MODEL" in [l.text() for l in dlg.findChildren(QLabel)],
          "MODEL badge after NEXT")
    check(not dlg._back_btn.isHidden(), "BACK after the first step")
    check("font-size:9px" in dlg._back_btn.styleSheet()
          and "font-size:9px" in dlg._next_btn.styleSheet(),
          "BACK and NEXT share the header 9px type")

    dlg._config.set_value("")
    dlg._on_next()
    check(dlg._step == 1, "MODEL without a YAML stays put")
    check(not dlg._gate.isHidden() and "YAML" in dlg._gate.text(),
          "gate explains the missing YAML")

    tmp = tempfile.mkdtemp(prefix="msst-setup-")
    yaml_path = os.path.join(tmp, "cfg.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("audio:\n  chunk_size: 131072\ntraining:\n  batch_size: 6\n")
    data_dir = os.path.join(tmp, "train")
    valid_dir = os.path.join(tmp, "valid")
    results_dir = os.path.join(tmp, "results")
    os.makedirs(data_dir)
    os.makedirs(valid_dir)
    os.makedirs(results_dir)
    dlg._model.set_key("bs_conformer")
    dlg._config.set_value(yaml_path)
    check(dlg._step_error() is None, "MODEL is complete with architecture + YAML")
    dlg._on_next()
    check(dlg._step == 2, "NEXT moves to DATA")
    check(page._batch_row.value() == "6",
          "first YAML sync fills batch from the file")

    dlg._on_next()
    check(dlg._step == 2, "DATA without folders stays put")
    dlg._data.set_paths([data_dir])
    dlg._valid.set_paths([valid_dir])
    dlg._results.set_value(results_dir)
    check(dlg._step_error() is None, "DATA is complete")
    dlg._on_next()
    check(dlg._step == 3, "NEXT moves to WANDB")
    check("04  WANDB" in [l.text() for l in dlg.findChildren(QLabel)],
          "WANDB badge")
    check(theme_manager.accent.lower() in dlg._wandb_links.text().lower(),
          "wandb links use the GUI accent blue")
    check(dlg._wandb_links.palette().color(QPalette.ColorRole.Link).name().lower()
          == QColor(theme_manager.accent).name().lower(),
          "link palette is accent, not the default dark blue")
    check(dlg._wandb_chart.objectName() == "wandbPreviewChart"
          and dlg._wandb_chart.width() >= 120,
          "wandb step shows a metrics chart preview")
    check(not any(b.text() == "Open wandb" for b in dlg.findChildren(QPushButton)),
          "wizard wandb step has no extra Open wandb button")
    check(dlg._wandb_mode_lbl.text() == "Log online (upload to wandb.ai)",
          f"online caption matches the switch, got {dlg._wandb_mode_lbl.text()!r}")
    dlg._wandb_offline.set_checked(True)
    check(dlg._wandb_mode_lbl.text() == "Log offline (no upload)",
          "offline caption follows the switch")
    dlg._wandb_offline.set_checked(False)
    dlg._wandb.setText("test-key-xyz")
    dlg._on_next()
    check(dlg._step == 4, "wandb does not block NEXT")
    check(dlg._step_error() is None, "Fit GPU is skippable")
    dlg._page._open_fit_gpu = lambda **_kw: False
    dlg._probe()
    check(dlg._step == 4, "closing Fit GPU without Apply stays on GPU")
    dlg._on_next()
    check(dlg._step == 5, "Fit GPU is skippable")
    check(dlg._next_btn.text() == "TRAIN", "last step is TRAIN")
    review_copy = dlg._review_text()
    check("NEED" in review_copy and "Fit GPU" in review_copy,
          "review flags an unprobed GPU as recommended")
    check("not probed" in review_copy, "review copy says the GPU was not probed")
    check("OK" in review_copy and "Config" in review_copy,
          "review marks the YAML as present")
    check(len(dlg._review.findChildren(QFrame, "setupReviewRow")) == 8,
          "review rows are hoverable frames")
    review_row = dlg._review.findChildren(QFrame, "setupReviewRow")[0]
    check(review_row.minimumHeight() >= 48, "review rows have room for 13px type")
    long_row = _ReviewRow(True, "Checkpoint", "model_vocals_mdx23c_sdr_10.17.ckpt")
    check(long_row.sizeHint().height() >= 48,
          "long values still get a full-height card")
    review_row._hover = True
    review_row._paint()
    check(theme_manager.accent in review_row.styleSheet(),
          "review row lights up on hover")
    review_row._hover = False
    review_row._paint()

    dlg._show_step(4)
    dlg._page._open_fit_gpu = lambda **_kw: True
    dlg._probe()
    check(dlg._fitted is True, "successful Apply marks Fit GPU applied")
    check(dlg._step == 5, "successful probe continues to review")

    # Fit GPU wrote batch 4 onto the tab; later wizard syncs must not reload
    # YAML batch_size: 6.
    page._batch_row.set_value(4)
    page._accum_row.set_value(2)
    if page._optim_row.combo.findData("adamw") >= 0:
        page._optim_row.set_key("adamw")
    dlg._fitted = True
    dlg._sync_to_page()
    check(page._batch_row.value() == "4",
          "same-path sync keeps the Fit GPU batch")
    check(page._accum_row.value() == "2",
          "same-path sync keeps the Fit GPU accum")
    if page._optim_row.combo.findData("adamw") >= 0:
        check(page._optim_row.key() == "adamw",
              "same-path sync keeps the Fit GPU optimizer")
    dlg._refresh_review()
    check("applied · batch 4" in dlg._review_text(),
          "review shows the applied batch")

    dlg._on_next()
    check(dlg.start_train is True, "TRAIN arms start")
    check(page._batch_row.value() == "4", "TRAIN keeps the probed batch")
    check(page._accum_row.value() == "2", "TRAIN keeps the probed accum")
    check(page._config_row.value() == yaml_path, "wizard writes the YAML onto the tab")
    check(page._data_row.paths() == [data_dir], "wizard writes the data folder")
    check(page._results_row.value() == results_dir, "wizard writes the results folder")
    check(not page._ckpt_row.value(), "scratch clears the resume checkpoint")
    check(page._run_opts.get("wandb_key") == "test-key-xyz",
          "wizard writes the wandb key onto run options")
    check(page.save_settings()["run_options"].get("wandb_key") == "test-key-xyz",
          "save_settings keeps the wandb key")

    starters = _starter_yamls("bs_conformer")
    check(isinstance(starters, list), "starter YAML list")
    cfg_root = os.path.join(REPO_ROOT, "configs")
    if os.path.isdir(cfg_root):
        check(any("conformer" in os.path.basename(p).lower() for p in starters)
              or not starters,
              "conformer starters prefer matching names")

    dlg2 = _TrainSetupDialog(page)
    check(dlg2._step_error() is None, "returning opens with the tab already filled")
    keep_cfg, keep_model = dlg2._config.value(), dlg2._model.key()
    dlg2._model.set_key("bs_roformer")
    dlg2._on_pretrained("config_vocals_mel_band_roformer.yaml", "")
    check(dlg2._model.key() == "bs_roformer",
          "catalog pick does not overwrite a locked architecture")
    dlg2._model.set_key(keep_model)
    dlg2._config.set_value(keep_cfg)
    # Jump to review without re-probing.
    dlg2._show_step(5)
    check(dlg2._next_btn.text() == "TRAIN", "review still offers TRAIN")
    names = [b.objectName() for b in dlg2.findChildren(QPushButton)]
    check("helpPrimaryBtn" in names and "helpBackBtn" in names, "NEXT / BACK chrome")
    dlg2.close()

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_train_setup():
    assert main() == 0
