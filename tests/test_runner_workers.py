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
import subprocess
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


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _force_kill(pid):
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.kill(pid, 9)
    except Exception:
        pass


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
    check(bool(codes2),
          f"terminated child reports a code, got {codes2}")
    check(runner2 not in _ACTIVE_RUNNERS, "registry released after stop")

    # 3. stop() with nothing running is a no-op (no crash).
    runner3 = ProcessRunner([sys.executable, "-c", "pass"])
    runner3.stop()
    check(True, "stop() before start is safe")

    # 4. stop() must finish when a grandchild inherits stdout and keeps
    # it open. Parent-only terminate() leaves that writer alive, so the
    # stdout iterator never EOFs and finished never fires (the training
    # Stop hang: DataLoader workers holding the pipe).
    parent_src = (
        "import os, sys, time, subprocess\n"
        "kw = {}\n"
        "if os.name == 'nt':\n"
        "    kw['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c',\n"
        "     'import sys,time; sys.stdout.write(\"grandchild\\\\n\");"
        " sys.stdout.flush(); time.sleep(60)'],\n"
        "    **kw)\n"
        "print('child-pid', child.pid, flush=True)\n"
        "print('parent-ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    lines5, codes5 = [], []
    runner5 = ProcessRunner([sys.executable, "-c", parent_src])
    runner5.log_line.connect(lines5.append)
    runner5.finished.connect(codes5.append)
    child_pid = None
    runner5.start()
    try:
        ready = pump_until(
            lambda: any("parent-ready" in l for l in lines5), timeout=8.0)
        check(ready, "grandchild-tree parent became ready")
        for line in lines5:
            if "child-pid" in line:
                try:
                    child_pid = int(line.split()[-1])
                except ValueError:
                    pass
        runner5.stop()
        ok = pump_until(lambda: codes5, timeout=12.0)
        check(ok, "stop() with grandchild holding stdout emits finished")
        check(bool(codes5),
              f"tree-kill stop delivered a code, got {codes5}")
        check(runner5 not in _ACTIVE_RUNNERS,
              "registry released after tree-kill stop")
        if child_pid is not None:
            time.sleep(0.4)
            QApplication.processEvents()
            alive = _pid_alive(child_pid)
            check(not alive,
                  f"grandchild pid {child_pid} was killed with the tree")
    finally:
        if not codes5:
            runner5.stop()
            pump_until(lambda: codes5, timeout=5.0)
        if child_pid is not None and _pid_alive(child_pid):
            _force_kill(child_pid)

    # 5. The inference page's names fetcher got the same treatment.
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