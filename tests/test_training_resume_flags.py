"""Regression: selecting a resume checkpoint must arm the resume flags.

The TRAINING tab's Run Options dialog has switches for --load_optimizer /
--load_scheduler / --load_epoch / --load_best_metric, but they defaulted off
and nothing turned them on, so every resume-from-checkpoint run restarted at
epoch 0 with a fresh optimizer. Picking a checkpoint in the "Resume
checkpoint" row now arms all four flags (each is only consumed by the engine
when the checkpoint actually carries that piece, so state-only pre-trained
checkpoints are unaffected); the user can still uncheck any switch in Run
Options for a specific run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFontMetrics  # noqa: E402
from PySide6.QtWidgets import QApplication, QFrame, QLabel  # noqa: E402

from ui.pages.training_page import TrainingPage, _RunOptionsDialog  # noqa: E402
from ui.widgets.common import _HELP_PAD_L, _HELP_PAD_Y, _HELP_SCROLL_LANE  # noqa: E402

FAILURES = []
CHECKS = 0

RESUME_KEYS = ("load_optimizer", "load_scheduler", "load_epoch",
               "load_best_metric")


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    app = QApplication([])
    page = TrainingPage()

    # Defaults: no checkpoint -> resume flags off.
    check(all(not page._run_opts.get(k) for k in RESUME_KEYS),
          f"defaults not off: {[k for k in RESUME_KEYS if page._run_opts.get(k)]}")

    # Picking a checkpoint arms all four resume flags.
    page._ckpt_row.set_value("C:/training/resume.ckpt")
    check(all(page._run_opts.get(k) for k in RESUME_KEYS),
          f"picking a checkpoint did not arm all flags: "
          f"{[k for k in RESUME_KEYS if not page._run_opts.get(k)]}")

    # The Run Options dialog reflects the armed state (switches checked).
    dlg = _RunOptionsDialog(dict(page._run_opts), page)
    for k in RESUME_KEYS:
        check(dlg._switches[k].is_checked(),
              f"dialog switch '{k}' not checked after arming")

    # Other switches are untouched (only the resume group is armed).
    for k in ("pin_memory", "pre_valid", "save_every_epoch",
              "each_metrics_in_name", "safe_mode", "persistent_workers",
              "wandb_offline", "load_all_metrics", "load_all_losses"):
        check(not page._run_opts.get(k),
              f"unrelated switch '{k}' was armed")

    check(page._run_opts.get("launcher") == "standard",
          "launcher defaults to standard")
    check(page._run_opts.get("lora_mode") == "off",
          "LoRA defaults off")
    check(dlg._launcher.currentData() == "standard",
          "dialog launcher combo is standard")
    check(dlg._lora_mode.currentData() == "off",
          "dialog LoRA combo is off")

    labels = [l.text() for l in dlg.findChildren(QLabel)]
    check("GPUS" in labels, "help GPUS column")
    check("WORKERS / SEED" in labels, "help WORKERS / SEED column")
    check("OPTIONAL" in labels, "help OPTIONAL column")
    check("REQUIRED" not in labels, "REQUIRED split into GPUS / WORKERS / SEED")
    check(any("GPUS / WORKERS / SEED" in (l or "") for l in labels),
          "help heading")
    req_frames = [f for f in dlg.findChildren(QFrame)
                  if f.objectName() == "helpRequired"]
    opt_frames = [f for f in dlg.findChildren(QFrame)
                  if f.objectName() == "helpOptional"]
    check(len(req_frames) == 2, "GPUS and WORKERS / SEED are required panes")
    check(len(opt_frames) == 1, "training flags stay in OPTIONAL")

    m = dlg.layout().contentsMargins()
    check(
        (m.left(), m.right(), m.top(), m.bottom())
        == (_HELP_PAD_L, _HELP_PAD_L, _HELP_PAD_Y, _HELP_PAD_Y),
        f"gpus/workers/seed outer margin "
        f"{(m.left(), m.right(), m.top(), m.bottom())}",
    )

    dlg.show()
    app.processEvents()

    def pos(key):
        w = dlg._switches[key]
        return w.mapTo(dlg, w.rect().topLeft())

    def pos_label(text):
        for lab in dlg.findChildren(QLabel):
            if lab.text() == text:
                return lab.mapTo(dlg, lab.rect().topLeft())
        return None

    run, freeze = pos_label("RUN OPTIONS"), pos_label("FREEZE LAYERS")
    check(run is not None and freeze is not None, "run / freeze group labels")
    gpus, workers = pos_label("GPUS"), pos_label("WORKERS / SEED")
    check(gpus is not None and workers is not None, "GPUS / WORKERS badges")
    check(workers.y() > gpus.y() + 20, "workers sits below GPUS")
    check(abs(workers.x() - gpus.x()) <= 8, "workers aligns under GPUS")
    optional = pos_label("OPTIONAL")
    check(optional is not None, "OPTIONAL badge")
    check(optional.x() > gpus.x() + 40, "OPTIONAL sits next to GPUS")
    check(abs(optional.y() - gpus.y()) <= 8, "OPTIONAL aligns with GPUS")
    lora, wandb = pos_label("LORA"), pos_label("WEIGHTS & BIASES (wandb)")
    check(lora is not None and wandb is not None, "LoRA / wandb group labels")
    resume = pos_label("WHEN RESUMING FROM A CHECKPOINT")
    check(resume is not None, "resume group label")
    check(resume.x() > run.x() + 40, "resume sits next to run options")
    check(abs(resume.y() - run.y()) <= 8, "resume aligns with run options")
    check(lora.y() > run.y() + 20, "LoRA sits below run options")
    check(abs(lora.x() - run.x()) <= 8, "LoRA aligns under run options")
    check(freeze.y() > lora.y() + 20, "freeze sits below LoRA")
    check(abs(freeze.x() - run.x()) <= 8, "freeze aligns under run options")
    check(wandb.y() > freeze.y() + 20, "wandb sits below freeze")
    check(abs(wandb.x() - freeze.x()) <= 8, "wandb aligns under freeze")
    check(dlg.width() >= 1120, "sheet is wide enough for the resume title")
    check(dlg.width() <= 1180, "sheet is not over-wide")
    opt_right = opt_frames[0].mapTo(dlg, opt_frames[0].rect().topRight()).x()
    left_w = resume.x() - run.x()
    right_w = opt_right - resume.x()
    check(left_w > right_w, "left optional column is wider than resume")
    resume_lab = next(
        lab for lab in dlg.findChildren(QLabel)
        if lab.text() == "WHEN RESUMING FROM A CHECKPOINT")
    check(resume_lab.text() == "WHEN RESUMING FROM A CHECKPOINT",
          "resume title text is complete")
    if dlg._inner.height() <= dlg._sc.height() + 1:
        check(dlg._sc.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff,
              "sheet drops the vertical bar when content fits")
    h0, bar0 = dlg.height(), dlg._sc.verticalScrollBarPolicy()
    dlg._lora_on_sw.set_checked(True)
    app.processEvents()
    check(dlg.height() == h0, "LoRA on keeps the same modal height")
    if dlg._inner.height() <= dlg._sc.height() + 1:
        check(dlg._sc.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff,
              "LoRA extras keep the vertical bar hidden")
    dlg._lora_on_sw.set_checked(False)
    app.processEvents()
    check(dlg.height() <= h0 + 8, "LoRA off returns to the compact height")

    opt, sch = pos("load_optimizer"), pos("load_scheduler")
    check(sch.y() > opt.y() + 8, "scheduler sits below optimizer")
    check(abs(sch.x() - opt.x()) <= 8, "scheduler aligns under optimizer")
    met, loss = pos("load_all_metrics"), pos("load_all_losses")
    check(loss.y() > met.y() + 8, "all losses sits below all metrics")
    check(abs(loss.x() - met.x()) <= 8, "all losses aligns under all metrics")
    if dlg._sc.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded:
        check(dlg._inner.width() + _HELP_SCROLL_LANE <= dlg._sc.width() + 1,
              "freeze/wandb fields leave the scrollbar lane")
    dlg.close()

    # Clearing the checkpoint keeps the armed flags (harmless no-ops for a
    # from-scratch run — the engine only reads them with --start_check_point).
    page._ckpt_row.set_value("")
    check(all(page._run_opts.get(k) for k in RESUME_KEYS),
          "clearing the checkpoint unexpectedly disarmed the resume flags")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())