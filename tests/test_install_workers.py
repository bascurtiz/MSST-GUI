"""Regression: model installs run on plain Python threads, never QThreads.

The model-manager install dialog historically crashed with a native
'access violation' inside QThread.__init__ the moment a new install thread
was created (faulthandler dump: model_manager_dialog.py:67 -> super().__init__()
called from _start_install). QThread objects that are destroyed while running
corrupt Qt's thread state, which surfaces later as random native crashes.

The fix: _InstallWorker/_FetchWorker are QObject signal emitters driven by
plain daemon threads, kept referenced in a module-level registry until they
finish — no QThread object exists to leak or corrupt, and a closed dialog can
no longer GC a running worker.

These checks run headless (offscreen QApplication) with install/fetch stubbed.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui.pages.model_manager_dialog as mmd  # noqa: E402
import ui.widgets.pretrained_models_dialog as pmd  # noqa: E402
from backend.model_manager import ModelInfo  # noqa: E402
from backend.pretrained_catalog import PretrainedModel  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def pump_until(pred, timeout=5.0):
    """Run the (offscreen) event loop until pred() is true or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        QApplication.processEvents()
        if pred():
            return True
        time.sleep(0.005)
    QApplication.processEvents()
    return False


def drain(delay=0.05):
    """Deliver any queued cross-thread signal events still in flight."""
    t0 = time.time()
    while time.time() - t0 < delay:
        QApplication.processEvents()
        time.sleep(0.003)
    QApplication.processEvents()


def _model_info():
    return ModelInfo(
        key="k", full_name="Test Model", arch="VR Architecture",
        stem_type="vocals", category="", model_type="vr", stems=["vocals"],
        target_instrument="", checkpoint_url="https://x/1.ckpt",
        config_url="https://x/1.yaml", backend_script_url="",
        file_size=1024 * 1024,
    )


def _pretrained_model():
    return PretrainedModel(
        name="bs_roformer_ep317_sdr_12.9765.ckpt", section="BS-Roformer",
        instruments="vocals", metrics="", config_url="https://x/a.yaml",
        checkpoint_url="https://x/a.ckpt", source="pretrained",
        arch_hint="BS Roformer Architecture",
    )


def main():
    app = QApplication.instance() or QApplication([])

    # ── Model-manager install worker ──────────────────────────────────────
    def fake_install(info, progress_callback=None, status_callback=None,
                     cancel_callback=None, speed_callback=None):
        for i in range(4):
            if cancel_callback and cancel_callback():
                return False, "cancelled"
            status_callback(f"step {i}")
            speed_callback(10.0)
            progress_callback("x", i, 3)
            time.sleep(0.01)
        return True, "ok"

    mmd.install_model = fake_install
    worker = mmd._InstallWorker(_model_info(), "VR Architecture", "vocals")
    check(not isinstance(worker, QThread),
          "install worker must not be a QThread subclass")
    events = []
    worker.status.connect(lambda s: events.append(("status", s)))
    worker.finished_signal.connect(lambda ok, msg: events.append(("done", ok, msg)))
    worker.done.connect(lambda: mmd._ACTIVE_WORKERS.discard(worker))
    check(not worker.isRunning(), "worker starts not-running")
    worker.start()
    check(worker.isRunning(), "worker running right after start")
    check(worker in mmd._ACTIVE_WORKERS,
          "running worker must be in the module registry")
    ok = pump_until(lambda: not worker.isRunning())
    drain()
    check(ok, "install worker must finish")
    check(any(e[0] == "done" and e[1] for e in events),
          "finished_signal must deliver success")
    check(worker not in mmd._ACTIVE_WORKERS,
          "registry must drop the worker once done")
    check(worker._cancelled is False, "cancel flag untouched on success")

    # ── Cancel path ───────────────────────────────────────────────────────
    def slow_install(info, progress_callback=None, status_callback=None,
                     cancel_callback=None, speed_callback=None):
        for i in range(200):
            if cancel_callback and cancel_callback():
                return False, "cancelled"
            time.sleep(0.01)
        return True, "ok"

    mmd.install_model = slow_install
    w2 = mmd._InstallWorker(_model_info(), "VR Architecture", "vocals")
    results = []
    w2.finished_signal.connect(lambda ok, msg: results.append(ok))
    w2.done.connect(lambda: mmd._ACTIVE_WORKERS.discard(w2))
    w2.start()
    w2.cancel()
    ok = pump_until(lambda: not w2.isRunning())
    drain()
    check(ok, "cancelled worker must finish")
    check(results == [False], "cancelled install must report failure")
    check(w2 not in mmd._ACTIVE_WORKERS, "cancelled worker leaves registry")

    # ── Pretrained dialog fetch + install workers ─────────────────────────
    pmd.catalog.fetch_catalog = lambda: [_pretrained_model()]

    def fake_cat_install(model, progress_callback=None, status_callback=None,
                         speed_callback=None, cancel_callback=None):
        status_callback("downloading")
        progress_callback("a.ckpt", 1, 2)
        return True, "ok"

    pmd.catalog.install_model = fake_cat_install

    fetch = pmd._FetchWorker()
    check(not isinstance(fetch, QThread), "fetch worker must not be a QThread")
    fetched = []
    fetch.loaded.connect(lambda m: fetched.extend(m))
    fetch.done.connect(lambda: pmd._ACTIVE_WORKERS.discard(fetch))
    fetch.start()
    ok = pump_until(lambda: fetched)
    drain()
    check(ok, "fetch worker must deliver the catalog")
    check(not fetch.isRunning(), "fetch worker finished")
    check(fetch not in pmd._ACTIVE_WORKERS, "fetch worker leaves registry")

    inst = pmd._InstallWorker(_pretrained_model())
    check(not isinstance(inst, QThread), "install worker must not be a QThread")
    inst_events = []
    inst.status.connect(lambda s: inst_events.append(s))
    inst.finished_signal.connect(lambda ok, msg: inst_events.append(("ok", ok)))
    inst.done.connect(lambda: pmd._ACTIVE_WORKERS.discard(inst))
    inst.start()
    ok = pump_until(lambda: not inst.isRunning())
    drain()
    check(ok, "pretrained install worker must finish")
    check(("ok", True) in inst_events, "pretrained install reports success")
    check(inst not in pmd._ACTIVE_WORKERS, "install worker leaves registry")

    # ── Full dialog flow (the exact crash path) ───────────────────────────
    dlg = mmd.ModelInstallDialog(_model_info())
    dlg._start_install()
    ok = pump_until(lambda: dlg._completed)
    drain()
    check(ok, "dialog must complete after _start_install")
    check(dlg._status_lbl.text().startswith("Installation complete"),
          "dialog must show the completion status")
    check(len(mmd._ACTIVE_WORKERS) == 0, "no workers leaked after dialog flow")

    # ── Huge byte counts must not overflow the progress signal ───────────
    # A 6+ GB model's byte count (6,411,087,858) exceeds a 32-bit C int; the
    # signal used to be Signal(str, int, int) and shiboken raised an
    # OverflowError that escaped as a fatal app crash. Float-typed now, it
    # must round-trip intact.
    big = mmd._InstallWorker(_model_info(), "VR Architecture", "vocals")
    got = []
    big.progress.connect(lambda n, c, t: got.append((c, t)))
    BIG = 6411087858
    big.progress.emit("ckpt", BIG, BIG)
    check(got and got[0] == (BIG, BIG),
          f"progress signal must carry >2GiB byte counts, got {got}")

    app.processEvents()

    if FAILURES:
        print(f"FAIL ({len(FAILURES)}/{CHECKS} checks):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print(f"ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()