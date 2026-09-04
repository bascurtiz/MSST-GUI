"""Regression: non-ASCII in job output must round-trip cleanly to the GUI.

backend/runner.py (and the auto/iterative ensemble runners) decode the
child's stdout as utf-8 with errors="replace", but on Windows a
pipe-inheriting python writes its ANSI code page (cp1252) unless told
otherwise — so accented track names ("Betül", "Néonheart") and the em-dash
in the dataset messages arrived as U+FFFD replacement chars in the GUI
logs. The runners now force PYTHONUTF8=1 on every child so the bytes are
utf-8 end to end.

This drives the real ProcessRunner (daemon thread + queued signals) with a
child that prints non-ASCII: the default run must be clean, and the override
case (PYTHONUTF8=0, i.e. a child not in UTF-8 mode) documents the failure
mode the fix eliminates.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from backend.runner import ProcessRunner  # noqa: E402

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
        time.sleep(0.004)
    QApplication.processEvents()
    return False


CHILD = ("import sys; print('Bet\\u00fcl Damla — N\\u00e9onheart'); "
         "print('Missing stem(s): other - skipping track.')")


def collect(env_override=None):
    lines = []
    runner = ProcessRunner([sys.executable, "-c", CHILD], env=env_override)
    runner.log_line.connect(lines.append)
    runner.start()
    ok = pump_until(lambda: not runner.isRunning())
    runner.wait()
    return "\n".join(lines), ok


def main():
    app = QApplication([])

    # Default runner: PYTHONUTF8=1 makes the child emit utf-8, so the
    # parent's utf-8 decode gets the real characters back.
    text, ok = collect()
    check(ok, "default runner never finished")
    check("Betül Damla — Néonheart" in text,
          f"non-ASCII lost in default run: {text!r}")
    check("\ufffd" not in text,
          f"replacement chars in default run: {text!r}")

    # Override: PYTHONUTF8=0 simulates a child NOT in UTF-8 mode (cp1252
    # bytes into the utf-8 decode) — the mojibake the fix prevents. Only
    # asserted on Windows, where the ANSI code page is not UTF-8.
    if sys.platform == "win32":
        text, ok = collect({"PYTHONUTF8": "0"})
        check(ok, "override runner never finished")
        check("\ufffd" in text,
              "override run should show replacement chars (documents the "
              f"failure mode): {text!r}")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())