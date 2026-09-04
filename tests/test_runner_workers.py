"""Regression: inference/training subprocess runners use plain Python
threads, never QThreads.

The TEST ALL MODELS batch historically crashed with a native 'access
violation' inside ProcessRunner.__init__ -> QThread.__init__ (faulthandler
dump: backend/runner.py:30, called from inference_page._run_inner in the
batch loop). QThread wrappers destroyed while their thread is still winding
down corrupt Qt's thread state; the damage surfaces later as a random
native crash — exactly where the next QThread gets created.

The fix: ProcessRunner is a QObject signal emitter driven by a plain daemon
thread and kept referenced in a module-level registry until the job
finishes — no QThread object exists to leak or corrupt.

These checks run headless (offscreen QApplication) and spawn only
`sys.executable` as the child process — no torch, no network.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from backend.runner import ProcessRunner, _ACTIVE_RUNNERS  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def pump_until(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        QApplication.processEvents()
        if pred():
            return True
        time.sleep(0.005)
    QApplication.processEvents()
    return False


def main():
    app = QApplication([])

    check(not issubclass(ProcessRunner, QThread),
          "ProcessRunner is not a QThread")
    for sig in ("log_line", "progress", "finished"):
        check(sig in ProcessRunner.__dict__,
              f"signal preserved: {sig}")

    # 1. Happy path: a real child process prints lines + a tqdm-style line.
    runner = ProcessRunner([
        sys.executable, "-c",
        "print('hello runner'); print('50%|#######| 5/10 [00:01<00:01]'); print('bye')",
    ])
    lines, pcts, codes = [], [], []
    runner.log_line.connect(lines.append)
    runner.progress.connect(pcts.append)
    runner.finished.connect(codes.append)
    check(not runner.isRunning(), "not running before start")
    runner.start()
    check(runner.isRunning(), "running right after start")
    check(runner in _ACTIVE_RUNNERS, "registered while running")
    ok = pump_until(lambda: codes)
    check(ok, "finished emitted")
    check(codes == [0], f"exit code 0, got {codes}")
    check(any("hello runner" in l for l in lines), "stdout streamed")
    check(any("bye" in l for l in lines), "final line streamed")
    check(50 in pcts, "tqdm percent parsed and emitted")
    check(not runner.isRunning(), "not running after finish")
    check(runner not in _ACTIVE_RUNNERS, "registry released after finish")

    # 2. stop(): terminates a long-running child.
    runner2 = ProcessRunner([sys.executable, "-c",
                             "import time; time.sleep(60)"])
    codes2 = []
    runner2.finished.connect(codes2.append)
    runner2.start()
    time.sleep(0.3)
    QApplication.processEvents()
    check(runner2.isRunning(), "long child running")
    runner2.stop()
    ok = pump_until(lambda: codes2, timeout=8.0)
    check(ok, "stop() terminates child -> finished emitted")
    check(codes2 == [-15] or codes2 == [1] or codes2 == [0],
          f"terminated child reports a code, got {codes2}")
    check(runner2 not in _ACTIVE_RUNNERS, "registry released after stop")

    # 3. stop() with nothing running is a no-op (no crash).
    runner3 = ProcessRunner([sys.executable, "-c", "pass"])
    runner3.stop()
    check(True, "stop() before start is safe")

    # 4. The inference page's names fetcher got the same treatment.
    import ui.pages.inference_page as ip
    check(not issubclass(ip._NamesFetchThread, QThread),
          "_NamesFetchThread is not a QThread")

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        for f in FAILURES:
            print("  FAIL:", f)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()