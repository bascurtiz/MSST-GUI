"""Regression & performance verification for GUI unresponsiveness fixes:
1. _ConsoleEdit document block capping (maximumBlockCount <= 2000).
2. _LoadingSpinner label pulse without setStyleSheet.
3. _read_audio_envelope in-memory caching.
4. ProcessRunner tqdm progress throttling.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint

from ui.theme import theme_manager
from ui.pages.console_page import (
    _ConsoleEdit, _LoadingSpinner, _LoadingLabel, _read_audio_envelope, _ENVELOPE_CACHE
)
from backend.runner import ProcessRunner

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    app = QApplication.instance() or QApplication([])
    theme_manager.init_app(app)

    # 1. Test _ConsoleEdit block capping under stress
    edit = _ConsoleEdit()
    check(edit.document().maximumBlockCount() == 2000, "maximumBlockCount is set to 2000")

    t0 = time.time()
    for i in range(5000):
        edit.append_line(f"Processing chunk {i}: {i % 100}%|########| {i}/5000 [00:01<00:01]")
    t_elapsed = time.time() - t0

    # The document block count must be capped at 2000
    check(edit.document().blockCount() <= 2001, f"blockCount capped: got {edit.document().blockCount()}")
    check(t_elapsed < 5.0, f"5000 log lines inserted rapidly: took {t_elapsed:.2f}s (<5.0s)")

    # 2. Test _LoadingSpinner & _LoadingLabel
    spinner = _LoadingSpinner()
    check(isinstance(spinner._label, _LoadingLabel), "_LoadingSpinner uses _LoadingLabel")

    initial_sheet = spinner._label.styleSheet()
    spinner._set_pulse(0.8)
    check(spinner._label.styleSheet() == initial_sheet,
          "pulsing _LoadingLabel does NOT invoke setStyleSheet")
    check(abs(spinner._label._alpha - 0.8) < 1e-4, "_LoadingLabel alpha updated cleanly")

    # 3. Test _read_audio_envelope caching
    import soundfile as sf
    import numpy as np

    tmp_dir = tempfile.mkdtemp(prefix="msst_perf_test_")
    audio_path = os.path.join(tmp_dir, "test.wav")
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 44100)).astype(np.float32)
    sf.write(audio_path, samples, 44100)

    _ENVELOPE_CACHE.clear()
    env1, pk1 = _read_audio_envelope(audio_path)
    check(env1 is not None and pk1 is not None, "audio envelope computed")
    check(len(_ENVELOPE_CACHE) == 1, "envelope added to cache")

    t_cached0 = time.perf_counter()
    env2, pk2 = _read_audio_envelope(audio_path)
    t_cached = time.perf_counter() - t_cached0

    check(env2 is env1, "cached envelope returned directly")
    check(t_cached < 0.001, f"cached envelope lookup is instantaneous: {t_cached*1000:.3f}ms")

    # 4. Test ProcessRunner tqdm throttling
    runner = ProcessRunner([
        sys.executable, "-c",
        "; ".join([
            "import sys, time",
            "print('START')",
            # Output 50 rapid tqdm lines at the same 50%
            *[f"sys.stdout.write(' 50%|#####| {i}/100 [00:00<00:00]\\r')" for i in range(50)],
            "sys.stdout.write('\\n')",
            "print('END')",
            "sys.stdout.flush()",
        ])
    ])

    lines = []
    runner.log_line.connect(lines.append)
    codes = []
    runner.finished.connect(codes.append)

    runner.start()
    t0 = time.time()
    while not codes and time.time() - t0 < 10.0:
        app.processEvents()
        time.sleep(0.01)

    check(codes == [0], f"runner completed with code 0: got {codes}")
    check("START" in lines, "'START' received")
    check("END" in lines, "'END' received")

    # The 50 duplicate 50% lines should be throttled down to a handful
    tqdm_lines = [l for l in lines if "50%" in l and "|" in l]
    check(len(tqdm_lines) < 20,
          f"rapid identical tqdm lines throttled from 50 to {len(tqdm_lines)}")

    if FAILURES:
        print(f"\\n{CHECKS} checks, {len(FAILURES)} failures:")
        for f in FAILURES:
            print(f"  FAIL: {f}")
        sys.exit(1)
    else:
        print(f"\\nALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()
