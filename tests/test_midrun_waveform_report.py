"""Regression test: stems are reported while the run is still going.

The CONSOLE builds a song card's waveform from its attached outputs, but the
inference page only scanned the output folder when the run FINISHED — so a
long folder batch showed empty waveform panels on every completed card until
the very end. The page now re-scans on a timer during the run, emitting
"Wrote file:" lines (deduped) so each card picks up its stems shortly after
the engine writes them.

Covered here (offscreen, no runner / subprocess / torch):
  * files written during the run are reported as "Wrote file:" lines,
  * re-scanning does not re-report the same paths,
  * newly appearing files are picked up on a later tick,
  * files older than the run start (a previous run) are never reported.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.theme import theme_manager  # noqa: E402
import ui.pages.inference_page as ip  # noqa: E402
import backend.settings as bs  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


class _NamesFetchThread(ip._NamesFetchThread):
    def start(self):
        pass


class _FakeMsgBox:
    @staticmethod
    def warning(parent, title, text):
        return None

    @staticmethod
    def information(parent, title, text):
        return None


def _build_page():
    theme_manager.init_app(QApplication.instance() or QApplication([]))
    ip._NamesFetchThread = _NamesFetchThread
    ip.QMessageBox = _FakeMsgBox
    real_load = bs.load

    def fake_load():
        d = dict(real_load())
        d["registered_models"] = []
        return d

    bs.load = fake_load
    try:
        from ui.pages.inference_page import InferencePage
        page = InferencePage()
    finally:
        bs.load = real_load
    return page


def _write(path, age_s):
    with open(path, "wb") as f:
        f.write(b"x")
    t = time.time() - age_s
    os.utime(path, (t, t))


def main():
    app = QApplication.instance() or QApplication([])
    page = _build_page()

    out = tempfile.mkdtemp(prefix="msst_report_")
    page._last_store_dir = out
    page._run_started = time.time() - 30
    page._reported_files = set()

    lines = []
    page.log_output.connect(lambda t: lines.append(t))

    def wrote():
        return [l for l in lines if l.startswith("Wrote file: ")]

    # Files written during the run appear on the first scan.
    _write(os.path.join(out, "song_dnr_000_speech.flac"), 10)
    _write(os.path.join(out, "song_dnr_000_music.flac"), 9)
    page._report_written_files()
    check(len(wrote()) == 2, f"first tick must report 2 files, got {len(wrote())}")

    # Re-scanning must not re-report (the console card dedupes by basename
    # anyway, but the stream should stay clean).
    page._report_written_files()
    check(len(wrote()) == 2, f"re-scan must not re-report, got {len(wrote())}")

    # A file that appears later is picked up on the next tick only.
    _write(os.path.join(out, "song_dnr_001_speech.flac"), 2)
    page._report_written_files()
    check(len(wrote()) == 3, f"later file must be reported, got {len(wrote())}")

    # Non-audio files are ignored.
    _write(os.path.join(out, "song_dnr_000.txt"), 5)
    page._report_written_files()
    check(len(wrote()) == 3, "non-audio files must be ignored")

    # A file older than the run start (previous run) is never reported.
    _write(os.path.join(out, "stale_old.flac"), 3600)
    page._report_written_files()
    check(len(wrote()) == 3, "files older than the run must not be reported")

    # A fresh run resets the dedupe set (new run, new files): every file
    # inside the new run's window is reported again, so a fresh run of the
    # same model is never starved by the previous run's dedupe state.
    page._reported_files = set()
    _write(os.path.join(out, "song_dnr_000_speech.flac"), 1)
    page._report_written_files()
    check(len(wrote()) == 6,
          f"fresh run re-reports its run-window files, got {len(wrote())}")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print(f"ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()