"""Regression: exit codes from crashed subprocesses must never overflow Qt
signals, crash codes must be readable, and fatal dialogs must never be
built off the main thread.

History: 'TEST ALL MODELS' crashed with a native access violation. A child
inference process died natively (STATUS_ACCESS_VIOLATION = 3221225477
unsigned) and the runner emitted that raw value through
`finished = Signal(int)`. Shiboken's int32 conversion of the out-of-range
value leaves a pending exception in the emitter thread that surfaced later
inside the worker, reached sys.excepthook, and _excepthook built a
QMessageBox *on the worker thread* — 'QWidget::repaint: Recursive repaint'
+ access violation (faulthandler dump: main.py _show_fatal_dialog <-
_excepthook <- backend/runner.py:135 in run).

Fixes under test:
- backend/runner._coerce_exit_code() normalizes the unsigned NTSTATUS to
  the signed 32-bit value Windows itself reports before signal emission,
  so Signal(int) can never overflow.
- backend/runner.describe_exit_code() renders crash codes readably
  ('0xC0000005 (STATUS_ACCESS_VIOLATION)').
- main._FatalBridge / _marshal_fatal_dialog route the fatal dialog onto
  the main thread (queued connection) instead of touching widgets on a
  worker thread, with a re-entrancy guard inside _show_fatal_dialog.

Runs headless (offscreen QApplication); the only child spawned is
`sys.executable`.
"""
import contextlib
import io
import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from backend.runner import (  # noqa: E402
    ProcessRunner,
    _ACTIVE_RUNNERS,
    _coerce_exit_code,
    describe_exit_code,
)

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


def test_coerce():
    # Normal exit / small codes pass through untouched.
    check(_coerce_exit_code(0) == 0, "coerce: 0 stays 0")
    check(_coerce_exit_code(1) == 1, "coerce: 1 stays 1")
    check(_coerce_exit_code(127) == 127, "coerce: 127 stays 127")
    check(_coerce_exit_code(-15) == -15, "coerce: -15 stays -15")
    # The exact field crash: unsigned STATUS_ACCESS_VIOLATION.
    check(_coerce_exit_code(3221225477) == -1073741819,
          f"coerce: 0xC0000005 wraps to signed, got "
          f"{_coerce_exit_code(3221225477)}")
    # Boundary: 2^31 - 1 fits, 2^31 does not.
    check(_coerce_exit_code(2147483647) == 2147483647, "coerce: INT_MAX stays")
    check(_coerce_exit_code(2147483648) == -2147483648,
          "coerce: 2^31 wraps to INT_MIN")
    # None (no return code) maps to the runner's generic failure code.
    check(_coerce_exit_code(None) == -1, "coerce: None -> -1")


def test_describe():
    check(describe_exit_code(0) == "0", "describe: 0")
    check(describe_exit_code(1) == "1", "describe: 1")
    check(describe_exit_code(None) == "-1", "describe: None")
    # Coerced signed form of the field crash.
    check(describe_exit_code(-1073741819) == "0xC0000005 (STATUS_ACCESS_VIOLATION)",
          f"describe: access violation, got {describe_exit_code(-1073741819)}")
    # Raw unsigned form from a caller that didn't coerce.
    check(describe_exit_code(3221225477) == "0xC0000005 (STATUS_ACCESS_VIOLATION)",
          f"describe: raw unsigned access violation, got "
          f"{describe_exit_code(3221225477)}")
    check(describe_exit_code(-1073741811) == "0xC000000D (STATUS_INVALID_PARAMETER)",
          f"describe: invalid parameter, got {describe_exit_code(-1073741811)}")
    # Unknown NTSTATUS still gets the hex value.
    check(describe_exit_code(-1073741776).startswith("0xC0000030 (STATUS_0xC0000030)"),
          f"describe: unknown status keeps hex, got {describe_exit_code(-1073741776)}")


def test_signal_delivery_from_worker():
    """Emit a coerced exit code across threads: delivered queued to the main
    thread, in-range, and no shiboken Overflow warning/error surfaces."""
    from PySide6.QtCore import QObject, Signal

    class R(QObject):
        fin = Signal(int)

    r = R()
    code_box = {}
    loop = QEventLoop()
    r.fin.connect(lambda c: (code_box.update(code=c), loop.quit()))

    def worker():
        r.fin.emit(_coerce_exit_code(3221225477))

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        t = threading.Thread(target=worker)
        t.start()
        loop.exec()
        t.join()
    check("code" in code_box, "worker emit delivered")
    code = code_box.get("code")
    check(code == -1073741819, f"delivered signed value, got {code}")
    check(-(1 << 31) <= code < (1 << 31), "delivered value fits int32")
    check("libshiboken: Overflow" not in err.getvalue(),
          "no overflow warning on coerced emit")
    check("OverflowError" not in err.getvalue(),
          "no overflow error on coerced emit")
    # The emitter thread must not be left with a pending exception: another
    # plain emit right after must deliver fine.
    ok = pump_until(lambda: True)
    check(ok, "event loop still healthy after coerced emit")


def test_runner_reports_crashed_child():
    """End-to-end: a child whose exit code exceeds int32 — the shape native
    crashes produce on Windows (STATUS_ACCESS_VIOLATION = 0xC0000005 =
    3221225477) — must surface through ProcessRunner.finished as an
    in-range signed value with no shiboken overflow in stderr. (The venv
    Python catches an SEH raised inside a `-c` child and maps it to exit 1,
    so the exact 0xC0000005 value is covered at the unit + signal layer
    above; sys.exit(4294967295) yields the same unsigned-high returncode
    shape through a real child.)"""
    runner = ProcessRunner([
        sys.executable, "-c", "import sys; sys.exit(4294967295)",
    ])
    codes = []
    runner.finished.connect(codes.append)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        runner.start()
        ok = pump_until(lambda: codes)
    check(ok, "unsigned-high child finished emitted")
    check(bool(codes), "unsigned-high child delivered a code")
    code = codes[0] if codes else None
    check(code == -1, f"0xFFFFFFFF wraps to signed -1, got {code}")
    check(-(1 << 31) <= code < (1 << 31), "unsigned-high child code fits int32")
    check("libshiboken: Overflow" not in err.getvalue(),
          "no overflow warning from unsigned-high child")
    check(runner not in _ACTIVE_RUNNERS, "registry released after child")

    # Normal non-zero exits pass through unchanged.
    runner2 = ProcessRunner([sys.executable, "-c", "import sys; sys.exit(2)"])
    codes2 = []
    runner2.finished.connect(codes2.append)
    runner2.start()
    ok2 = pump_until(lambda: codes2)
    check(ok2, "exit-2 child finished emitted")
    check(codes2 == [2], f"exit 2 delivered unchanged, got {codes2}")


def test_fatal_bridge():
    """The fatal dialog must be marshaled onto the main thread from worker
    threads, show synchronously on the main thread, and never recurse."""
    import main as main_mod

    recorded = []

    def recorder(title, summary, trace):
        recorded.append((title, summary, trace))

    # Install the bridge AFTER stubbing so the connection captures the stub.
    main_mod._show_fatal_dialog = recorder
    main_mod._fatal_bridge = None
    main_mod._install_fatal_bridge()
    check(main_mod._fatal_bridge is not None, "bridge installed")

    # Worker thread -> queued onto the main thread.
    def worker():
        main_mod._marshal_fatal_dialog("t-worker", "s-worker", "tb-worker")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    QApplication.processEvents()
    check(any(r[0] == "t-worker" for r in recorded),
          "worker-thread fatal delivered on the main thread")

    # Main thread -> synchronous (direct connection).
    recorded.clear()
    main_mod._marshal_fatal_dialog("t-main", "s-main", "tb-main")
    check(recorded == [("t-main", "s-main", "tb-main")],
          "main-thread fatal shows synchronously")

    # Re-entrancy guard: while a dialog is active, a second call returns
    # immediately without touching widgets.
    called = []
    original_active = main_mod._dialog_active
    main_mod._dialog_active = True
    try:
        main_mod._show_fatal_dialog("t-2", "s-2", "tb-2")
    finally:
        main_mod._dialog_active = original_active
    check(not called, "guarded dialog did not recurse")

    # Worker-thread fatal with NO bridge yet must not touch widgets either
    # (fallback path is log-only for non-main threads).
    main_mod._fatal_bridge = None
    recorded.clear()

    def worker2():
        main_mod._marshal_fatal_dialog("t-nobridge", "s", "tb")

    t2 = threading.Thread(target=worker2)
    t2.start()
    t2.join()
    QApplication.processEvents()
    check(not recorded, "no-bridge worker fatal did not touch widgets")
    # Restore the bridge for any later checks.
    main_mod._install_fatal_bridge()


def main():
    app = QApplication([])
    test_coerce()
    test_describe()
    test_signal_delivery_from_worker()
    test_runner_reports_crashed_child()
    test_fatal_bridge()

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        for f in FAILURES:
            print("  FAIL:", f)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()