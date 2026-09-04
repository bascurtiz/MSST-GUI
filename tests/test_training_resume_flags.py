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

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.pages.training_page import TrainingPage, _RunOptionsDialog  # noqa: E402

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
              "each_metrics_in_name"):
        check(not page._run_opts.get(k),
              f"unrelated switch '{k}' was armed")

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