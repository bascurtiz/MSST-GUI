"""Regression: fatal errors surface an in-app dialog with the traceback.

main.py's hooks must format the full traceback, keep printing it to the
console/log, and hand it to a Qt dialog. The dialog path must degrade
gracefully (no crash, no block) when no QApplication exists — e.g. in
headless runs and during startup failures before the GUI is up.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_mod  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


class _FakeStdin:
    """isatty()=False so the fatal-hook's 'Press Enter' hold never blocks."""

    def isatty(self):
        return False


def main():
    try:
        raise RuntimeError("kaboom-in-app-dialog-test")
    except RuntimeError:
        tp, val, tb = sys.exc_info()

    trace = main_mod._format_traceback(tp, val, tb)
    check("RuntimeError" in trace, "traceback must mention the exception type")
    check("kaboom-in-app-dialog-test" in trace,
          "traceback must include the message")
    check(main_mod._log_hint().endswith("msst-gui.log"),
          "log hint must point at msst-gui.log")

    # Main-thread hook: prints + attempts the dialog; must not raise and must
    # not block (no QApplication instance exists here -> no dialog shown).
    real_stdin = sys.stdin
    sys.stdin = _FakeStdin()
    try:
        main_mod._excepthook(tp, val, tb)
    finally:
        sys.stdin = real_stdin

    # Thread crash hook: banner + marshals the dialog to the main thread;
    # with no event loop the marshal is a safe no-op.
    class FakeArgs:
        exc_type = RuntimeError
        exc_value = RuntimeError("thread-kaboom")
        exc_traceback = None

    main_mod._thread_excepthook(FakeArgs())

    # Dialog helpers are no-ops without a QApplication rather than crashing.
    main_mod._show_fatal_dialog("t", "s", trace)
    main_mod._marshal_fatal_dialog("t", "s", trace)

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())