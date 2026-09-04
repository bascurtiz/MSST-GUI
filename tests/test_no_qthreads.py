"""Regression: no raw QThread subclass may exist in the app's worker set.

The GUI historically crashed with native "Windows fatal exception: access
violation" crashes whose faulthandler dumps pointed at *other* code than the
true culprit: QThread wrappers destroyed while their thread is still winding
down corrupt Qt's thread state, and the damage surfaces later as a random
native crash on the main thread (a plain QPushButton constructor, the next
QThread creation, a re-render, ...). Each conversion below removes one such
hazard; this suite pins the final three conversions so they cannot regress:

- ui/widgets/runtime_dialog._SetupThread   (GPU runtime setup dialog)
- backend.auto_ensemble_runner.AutoEnsembleRunner
- backend.iterative_ensemble.runner.IterativeEnsembleRunner

All are now plain QObject signal emitters driven by daemon threads and kept
referenced in module-level registries until they finish.

Headless (offscreen QApplication); runtime install is stubbed, no torch, no
network, no real pipeline runs.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui.widgets.runtime_dialog as rd  # noqa: E402
import backend.auto_ensemble_runner as aer  # noqa: E402
import backend.iterative_ensemble.runner as ier  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def pump_until(pred, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        QApplication.processEvents()
        if pred():
            return True
        time.sleep(0.004)
    QApplication.processEvents()
    return False


def main():
    app = QApplication([])

    classes = (
        (rd._SetupThread, "_ACTIVE_SETUP_WORKERS", rd),
        (aer.AutoEnsembleRunner, "_ACTIVE_ENSEMBLE_RUNNERS", aer),
        (ier.IterativeEnsembleRunner, "_ACTIVE_ITERATIVE_RUNNERS", ier),
    )

    # Static: none are QThreads; each keeps its public start/isRunning API.
    for cls, reg, mod in classes:
        check(not issubclass(cls, QThread), f"{cls.__name__} is not a QThread")
        check(hasattr(cls, "start") and hasattr(cls, "isRunning"),
              f"{cls.__name__} keeps start/isRunning")
        check(not issubclass(cls, QThread) and cls.__mro__[1] is not QThread,
              f"{cls.__name__} base is QObject, not QThread")

    # 1. Runtime setup worker: run a stubbed install to completion, then the
    #    registry drains. Cancel path: flag aborts the stub install.
    import backend.runtime_setup as rs

    def fake_install(log, progress, cancelled, top_up=False):
        for i in range(5):
            if cancelled():
                return False, "Cancelled"
            log(f"step {i}")
            progress((i + 1) / 5.0)
            time.sleep(0.01)
        return True, "done"

    rs.install_runtime = fake_install
    w = rd._SetupThread(top_up=False)
    logs, progs, results = [], [], []
    w.log_line.connect(lambda *a: logs.append(a))
    w.progress.connect(lambda *a: progs.append(a))
    w.finished_ok.connect(lambda *a: results.append(a))
    w.start()
    check(w in rd._ACTIVE_SETUP_WORKERS, "setup worker registered")
    ok = pump_until(lambda: results)
    check(ok, f"setup worker finished (results={results})")
    check(results and results[0][0] is True, "setup install reported ok")
    check(logs and progs, "setup worker streamed log/progress")
    ok = pump_until(lambda: len(rd._ACTIVE_SETUP_WORKERS) == 0)
    check(ok, f"setup registry drained: {len(rd._ACTIVE_SETUP_WORKERS)}")

    # 2. Runtime setup cancel path: worker keeps cancel() semantics.
    w2 = rd._SetupThread(top_up=False)
    res2 = []
    w2.finished_ok.connect(lambda *a: res2.append(a))
    w2.start()
    w2.cancel()
    ok = pump_until(lambda: res2)
    check(ok and res2 and res2[0][0] is False,
          f"setup cancel aborted install (res2={res2})")
    ok = pump_until(lambda: len(rd._ACTIVE_SETUP_WORKERS) == 0)
    check(ok, "setup registry drained after cancel")

    # 3. Auto ensemble runner: run() is a plain callable (the preflight test
    #    drives it synchronously) and start() registers + drains the registry.
    m = [{"name": "A", "ckpt": "x.ckpt", "yaml": "x.yaml", "arch": "x",
          "type": "x", "model_type": "mdx23c"}]
    r = aer.AutoEnsembleRunner(m, "nope.mp3", "vocals", "avg", "out")
    check(callable(getattr(r, "run", None)), "AutoEnsembleRunner.run callable")

    # start() registers and a cancelled job drains the registry without a
    # real pipeline: patch run to a stub that sleeps until cancelled.
    def stub_run(self):
        while not self._cancelled:
            time.sleep(0.01)
        self.finished.emit(False, "Cancelled", "")

    r2 = aer.AutoEnsembleRunner(m, "nope.mp3", "vocals", "avg", "out")
    r2.run = stub_run.__get__(r2)
    done2 = []
    r2.finished.connect(lambda *a: done2.append(a))
    r2.start()
    check(r2 in aer._ACTIVE_ENSEMBLE_RUNNERS, "auto runner registered")
    time.sleep(0.15)
    check(r2.isRunning(), "auto runner running")
    r2.cancel()
    ok = pump_until(lambda: done2)
    check(ok, f"auto runner finished after cancel (done2={done2})")
    ok = pump_until(lambda: len(aer._ACTIVE_ENSEMBLE_RUNNERS) == 0)
    check(ok, f"auto registry drained: {len(aer._ACTIVE_ENSEMBLE_RUNNERS)}")

    # 4. Iterative ensemble runner: same start/cancel/registry lifecycle.
    ir2 = ier.IterativeEnsembleRunner({"input_files": [], "iterations": 1})
    ir2.run = stub_run.__get__(ir2)
    done3 = []
    ir2.finished.connect(lambda *a: done3.append(a))
    ir2.start()
    check(ir2 in ier._ACTIVE_ITERATIVE_RUNNERS, "iterative runner registered")
    time.sleep(0.15)
    check(ir2.isRunning(), "iterative runner running")
    check(ir2._paused is False, "pause flag defaults to False")
    ir2.pause()
    check(ir2._paused is True, "pause() sets the flag")
    ir2.resume()
    check(ir2._paused is False, "resume() clears the flag")
    ir2.cancel()
    ok = pump_until(lambda: done3)
    check(ok, f"iterative runner finished after cancel (done3={done3})")
    ok = pump_until(lambda: len(ier._ACTIVE_ITERATIVE_RUNNERS) == 0)
    check(ok, f"iterative registry drained: {len(ier._ACTIVE_ITERATIVE_RUNNERS)}")

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        for f in FAILURES:
            print("  FAIL:", f)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
