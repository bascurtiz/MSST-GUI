"""Fit GPU action-bar button applies advisor knobs onto the TRAINING tab."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace as dc_replace  # noqa: E402

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QLabel, QPushButton, QSpinBox, QLineEdit,
)

from backend.train_advisor import advise  # noqa: E402
from ui.pages.training_page import (  # noqa: E402
    TrainingPage, _FitGpuDialog, _probe_eta_remain,
    _PROBE_TRIAL_HINT, _PROBE_LOAD_HINT, _PROBE_ETA_MIN_REMAIN,
)
from ui.widgets.common import CHIP_GLYPH, BOLT_GLYPH  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    r0 = _probe_eta_remain(0)
    check(r0 >= _PROBE_LOAD_HINT + 4 * _PROBE_TRIAL_HINT - 0.01,
          f"cold start leaves load+trials, got {r0}")
    r30 = _probe_eta_remain(30, step=0, total=6, phase="load")
    check(r30 > 40, f"30s into load still has trial time left, got {r30}")
    r100 = _probe_eta_remain(100, step=0, total=6, phase="load")
    check(r100 > 20, f"long load does not collapse to a 12s floor, got {r100}")
    r_load = _probe_eta_remain(20, step=1, total=6, phase="loaded", load_elapsed=20)
    check(abs(r_load - 5 * _PROBE_TRIAL_HINT) < 0.01,
          f"after load guesses 5 trials, got {r_load}")
    r_trial = _probe_eta_remain(
        35, step=2, total=6, phase="trial", load_elapsed=20, since_step=0)
    check(abs(r_trial - 60) < 0.01,
          f"first finished trial leaves 60s, got {r_trial}")
    r_mid = _probe_eta_remain(
        40, step=2, total=6, phase="trial", load_elapsed=20, since_step=5)
    check(r_mid < r_trial,
          f"ETA counts down during a trial, {r_mid} vs {r_trial}")
    check(abs(r_mid - 55) < 0.01, f"5s into next trial leaves 55s, got {r_mid}")
    r_later = _probe_eta_remain(
        50, step=2, total=6, phase="trial", load_elapsed=20, since_step=15)
    check(r_later <= r_mid + 0.01,
          f"ETA must not climb during a trial, {r_later} vs {r_mid}")
    r_last = _probe_eta_remain(
        80, step=5, total=6, phase="trial", load_elapsed=20, since_step=5)
    check(r_last > 0, f"in-flight last trial is not 0, got {r_last}")
    check(r_last < 20, f"last trial leftover is short, got {r_last}")
    r_done = _probe_eta_remain(80, step=6, total=6, phase="trial", load_elapsed=20)
    check(r_done == 0.0, f"completed probe is 0, got {r_done}")
    check(_PROBE_ETA_MIN_REMAIN <= 1.0, "floor is 1s, not a 12s stall")

    app = QApplication.instance() or QApplication([])
    page = TrainingPage()
    check(hasattr(page, "btn_fit"), "Fit GPU is an action-bar button")
    check(not hasattr(page, "_fit_row"), "Fit GPU is not a settings chevron")
    check(page.btn_fit._glyph_lbl._text == CHIP_GLYPH, "Fit GPU uses a chip icon")
    check(page.btn_setup._glyph_lbl._text == BOLT_GLYPH, "Wizard uses a bolt icon")
    check(hasattr(page, "btn_wandb") and page.btn_wandb._lbl.text() == "Open wandb",
          "Open wandb sits with Export Weights")
    from ui.pages.training_page import _WANDB_ICON_PATH  # noqa: E402
    from PySide6.QtGui import QPixmap  # noqa: E402
    check(os.path.isfile(_WANDB_ICON_PATH) and not QPixmap(_WANDB_ICON_PATH).isNull(),
          "Open wandb uses the official wandb mark")
    check(page.btn_wandb._icon._kind == "wandb"
          and not page.btn_wandb._icon._wandb_pixmap().isNull(),
          "wandb button paints the bundled favicon")
    check(hasattr(page, "btn_setup") and page.btn_setup._text_lbl.text() == "Wizard",
          "Wizard is an action-bar button")
    check(page.btn_run.parentWidget() is page.btn_fit.parentWidget(),
          "Fit GPU sits with Train")
    page.resize(1400, 900)
    page.show()
    app.processEvents()
    check(page.btn_run.x() < page.btn_setup.x() < page.btn_fit.x() < page.btn_stop.x(),
          "order is Train · Wizard · Fit GPU · Stop")
    check(page.btn_results._lbl.text() == "Open Output",
          "results button is Open Output")
    check(page.btn_logs.x() < page.btn_wandb.x() < page.btn_export.x() < page.btn_results.x(),
          "right actions are View Logs · wandb · Export · Open Output")
    check("FIT GPU" not in [l.text() for l in page.findChildren(QLabel)],
          "Fit GPU is not a settings-row label")

    page._model_row.set_key("bs_conformer")
    advice = advise(
        model_type="bs_conformer", gpu_mem_gb=32,
        gpu_label="GPU 0: NVIDIA GeForce RTX 5090 (32 GB)",
        chunk_size=588800, current_batch=2, current_accum=1)
    page._apply_advice(advice)
    check(page._batch_row.value() == "1", f"applied batch 1, got {page._batch_row.value()}")
    check(page._accum_row.value() == "2", f"applied accum 2, got {page._accum_row.value()}")
    check(page._checkpoint_override is True, "checkpoint override armed")
    check(page._run_opts.get("pin_memory") is True, "pin memory armed")

    dlg = _FitGpuDialog(advice, {
        "model_type": "bs_conformer", "batch": "2", "accum": "1",
        "checkpoint": False, "optimizer": "prodigy",
    }, page)
    check(dlg.advice.batch_size == 1, "dialog keeps the advice")
    check(dlg._batch.value() == 1, "batch lives in a blue spin")
    check(dlg._accum.value() == 2, "accum lives in a blue spin")
    check(dlg._eff.text() == "2", "effective batch is its own field")
    check(abs(dlg._chunk.value() - 588800 / 44100) < 0.15,
          "chunk spin is seconds")
    dlg._chunk.setValue(10.0)
    check(dlg.values().chunk_size == 441000, "10s converts to samples at 44.1 kHz")
    check(dlg._peak.isReadOnly(), "peak allocation is a field")
    labels = [l.text() for l in dlg.findChildren(QLabel)]
    check("RECOMMENDED" in labels, "help RECOMMENDED column")
    check("LIVE" in labels, "help LIVE column")
    check("REQUIRED" not in labels, "Fit GPU is not a required/optional to-do")
    check("OPTIONAL" not in labels, "live allocation is not OPTIONAL")
    joined = " ".join(labels)
    check("FIT GPU  ·  TRAINING" in joined, "help heading")
    check("Batch size" in joined, "batch label")
    check("Peak this step" in joined, "live peak field")
    check("Batch size — 1" not in joined, "knobs are not static sentences")
    names = [b.objectName() for b in dlg.findChildren(QPushButton)]
    check("helpPrimaryBtn" in names, "APPLY in the header")
    apply = dlg.findChild(QPushButton, "helpPrimaryBtn")
    check(apply is not None and apply.height() == 30, "APPLY is Copy Log height")
    check("border:1px solid" in apply.styleSheet(), "APPLY uses Copy Log outline")
    close = dlg.findChild(QPushButton, "helpCloseBtn")
    check(close is not None and close.text() == "\u2715", "help × close")
    spins = dlg.findChildren(QSpinBox)
    check(any(s.value() == 1 for s in spins), "spin shows batch 1")
    blues = [e.styleSheet() for e in dlg.findChildren(QLineEdit)]
    check(any("Courier New" in s and "font-weight:bold" in s for s in blues),
          "measured values use the blue mono field")
    got = dlg.values()
    check(got.batch_size == 1 and got.grad_accum == 2, "values() reads the fields")
    check(dlg._eff.width() == 110 and dlg._peak.width() == 110,
          "blue fields share the 110px spin width")
    check("not a live VRAM probe" not in joined, "no contradictory heuristic disclaimer")
    check("A train step is allocated" not in joined,
          "intro does not claim a probe without a YAML")
    check("Select a config YAML" in joined or "Pick a config YAML" in joined,
          "intro/status asks for a YAML")
    check("not a lookup table" not in joined, "optional caption is not a live-probe claim")
    check("NOTES" in labels, "notes sit in their own section")
    check(hasattr(dlg, "_probe_bar"), "probe progress bar exists")
    check(hasattr(dlg, "_probe_eta") and dlg._probe_eta.objectName() == "fitProbeEta",
          "probe ETA sits beside the bar")
    dlg._set_probing(True, "Allocating")
    check(dlg._probe_row.isVisibleTo(dlg) and dlg._probe_eta.isVisibleTo(dlg),
          "ETA shows while the probe bar is up")
    check(dlg._probe_eta.text().startswith("ETA"),
          f"ETA label is populated, got {dlg._probe_eta.text()!r}")
    dlg._on_probe_progress({"step": 2, "total": 6, "phase": "trial"})
    dlg._probe_t0 = __import__("time").monotonic() - 10
    dlg._probe_loaded_t = dlg._probe_t0 + 4
    dlg._refresh_probe_eta()
    check("ETA" in dlg._probe_eta.text() and dlg._probe_eta.text() != "ETA —",
          f"progress updates ETA, got {dlg._probe_eta.text()!r}")
    check(dlg._probe_eta.text() != "ETA 0:00",
          f"mid-probe ETA is not zero, got {dlg._probe_eta.text()!r}")
    dlg._set_probing(False)
    check(not dlg._probe_row.isVisibleTo(dlg), "ETA hides when the probe ends")
    check(labels.count("Off") >= 2 and labels.count("On") >= 2,
          "checkpoint and pin memory both show Off / On")
    from ui.pages.inference_page import _InfoDot
    dots = dlg.findChildren(_InfoDot)
    check(len(dots) >= 10, f"knobs and live fields have i-icons, got {len(dots)}")
    check(any("--pin_memory" in (d.toolTip() or "") for d in dots),
          "pin memory tooltip lives on the i-icon")
    dlg.show()
    app.processEvents()
    notes_y = dlg._notes_lbl.mapTo(dlg, dlg._notes_lbl.rect().topLeft()).y()
    peak_y = dlg._peak.mapTo(dlg, dlg._peak.rect().topLeft()).y()
    check(notes_y > peak_y, "notes sit below both columns")
    dlg.close()

    import tempfile
    yaml_path = os.path.join(tempfile.mkdtemp(prefix="msst-fit-"), "cfg.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("audio:\n  chunk_size: 588800\ntraining:\n  batch_size: 2\n")
    page._config_row.set_value(yaml_path)
    page._apply_advice(got)
    check(page._chunk_override == 441000, "Apply stores the chunk override")
    written = open(yaml_path, encoding="utf-8").read()
    check("441000" in written, "Apply writes audio.chunk_size into the YAML")

    check(not dlg._needs_verify(), "no YAML skips the APPLY re-probe")
    dlg._set_probing(True, "Allocating")
    check(not dlg._apply_btn.isEnabled(), "APPLY waits while a probe is running")
    dlg._set_probing(False)
    check(dlg._apply_btn.isEnabled(), "APPLY returns after the probe")

    dlg._probe_spec = {"config_path": yaml_path, "model_type": "bs_conformer"}
    check(dlg._needs_verify(), "unmeasured knobs need an APPLY re-probe")
    kicked = []
    dlg._kick_probe = lambda spec=None, verifying=False: kicked.append(
        (spec, verifying))
    dlg.accept()
    check(kicked and kicked[0][1] is True, "APPLY re-probes instead of closing")
    check(dlg.result() != QDialog.Accepted, "OOM-unchecked APPLY does not write")
    spec = kicked[0][0]
    check(spec.get("verify_batch") == 1, "verify pins the field batch")
    check(spec.get("chunk_size") == 441000, "verify pins the field chunk")

    dlg.advice = dc_replace(
        dlg.advice, source="probe", peak_gb=18.4,
        batch_size=int(dlg._batch.value()),
        chunk_size=int(dlg._chunk_samples),
        use_torch_checkpoint=dlg.advice.use_torch_checkpoint)
    check(not dlg._needs_verify(), "successful probe at these knobs skips re-probe")
    dlg._chunk.setValue(8.0)
    check(dlg._needs_verify(), "edited chunk needs an APPLY re-probe")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_fit_gpu():
    assert main() == 0
