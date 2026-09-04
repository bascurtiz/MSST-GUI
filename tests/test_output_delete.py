"""Regression test: CONSOLE output deletion must remove the files on disk.

Reported bug: deleting an output (card context menu or the detail view's
trash button) removed the GUI entry but left the files on disk. The detail
view's waveform keeps every displayed output file open through QMediaPlayer
(Windows Media Foundation), so os.remove issued right after the click failed
with PermissionError.

Fixes under test:

* both delete entry points route through ConsolePage._on_delete_requested,
  which stops/unloads the waveform FIRST (releasing the app's own handles),
  then deletes with _remove_files_with_retry — retrying briefly to ride out
  WMF's async handle release and antivirus scans, and clearing the read-only
  attribute before a retry.
* when a file genuinely cannot be deleted, the card is KEPT (the entry stays
  true to what is still on disk), the surviving paths remain on the card so a
  retry is possible, and the user is told exactly which files are left.

Drives the real ConsolePage log parser offscreen; no torch, no network, no
subprocess.
"""
import os
import stat
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from ui.pages.console_page import ConsolePage, _remove_files_with_retry  # noqa: E402
import ui.pages.console_page as cp  # noqa: E402

FAILURES = []
SONG = "3 Doors Down - The Better Life - [07] - Better Life"


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


class _FakeQMessageBox:
    Yes = QMessageBox.Yes
    No = QMessageBox.No
    answers = []
    warns = []

    @staticmethod
    def question(parent, title, text, *args, **kwargs):
        return (_FakeQMessageBox.answers.pop(0)
                if _FakeQMessageBox.answers else QMessageBox.No)

    @staticmethod
    def warning(parent, title, text):
        _FakeQMessageBox.warns.append(text)
        return None


def _make_card(page, outdir):
    """Feed the log parser a completed run whose outputs really exist."""
    f1 = os.path.join(outdir, f"{SONG} (vocals).flac")
    f2 = os.path.join(outdir, f"{SONG} (other).flac")
    for p in (f1, f2):
        with open(p, "wb") as f:
            f.write(b"fLaC" + b"\0" * 64)
    page.append_log(f"Output directory: {outdir}\n")
    page.append_log(f"Processing: {SONG}.mp3\n")
    page.append_log(f"Wrote file: {f1}\n")
    page.append_log(f"Wrote file: {f2}\n")
    page.append_log("Completed: processing\n")
    cards = list(page._song_cards.values())
    assert cards, "no card created by the log parser"
    return cards[-1], f1, f2


def main():
    app = QApplication.instance() or QApplication([])
    real_msgbox = cp.QMessageBox
    cp.QMessageBox = _FakeQMessageBox

    base = tempfile.mkdtemp(prefix="msst_del_")

    # 1. Helper: normal delete + nonexistent paths are skipped quietly.
    f_plain = os.path.join(base, "plain.wav")
    with open(f_plain, "wb") as f:
        f.write(b"\0" * 16)
    failed = _remove_files_with_retry([f_plain, os.path.join(base, "gone.wav")])
    check(failed == [], f"helper reported failures: {failed}")
    check(not os.path.exists(f_plain), "helper deleted the plain file")

    # 2. Helper: read-only file is cleared and then deleted.
    f_ro = os.path.join(base, "readonly.txt")
    with open(f_ro, "w") as f:
        f.write("x")
    os.chmod(f_ro, stat.S_IREAD)
    failed = _remove_files_with_retry([f_ro])
    check(failed == [], f"read-only file not deleted: {failed}")
    check(not os.path.exists(f_ro), "read-only file deleted")

    # 3. Full page flow (what the card menu and trash button both route to):
    #    files vanish from disk and the card entry is removed, no warning.
    outdir = os.path.join(base, "m_model")
    os.makedirs(outdir)
    page = ConsolePage()
    page.set_input_files(["D:/in/x.mp3"])
    page.set_job_active(True)
    card, f1, f2 = _make_card(page, outdir)
    check(sorted(os.path.basename(p) for p in card._output_paths)
          == [f"{SONG} (other).flac", f"{SONG} (vocals).flac"],
          "card knows both stem files")

    _FakeQMessageBox.answers = [QMessageBox.Yes]
    _FakeQMessageBox.warns = []
    page._on_delete_requested(card)
    check(not os.path.exists(f1) and not os.path.exists(f2),
          "page delete removed the files from disk")
    check(not card._output_paths, "card output paths cleared after delete")
    check(page._song_cards.get(getattr(card, "_key", None)) is not card,
          "card entry removed from the page")
    check(_FakeQMessageBox.warns == [],
          "clean delete shows no warning dialog")
    check(not os.path.isdir(outdir),
          "now-empty output directory removed too")

    # 4. Failure path: when a file cannot be deleted, the card is KEPT with
    #    the surviving paths and the user is warned — the entry never lies
    #    about what is still on disk.
    outdir2 = os.path.join(base, "locked_model")
    os.makedirs(outdir2)
    page2 = ConsolePage()
    page2.set_input_files(["D:/in/x.mp3"])
    page2.set_job_active(True)
    card2, g1, g2 = _make_card(page2, outdir2)

    orig_helper = cp._remove_files_with_retry
    cp._remove_files_with_retry = lambda paths, attempts=8, delay=0.15: list(paths)
    try:
        _FakeQMessageBox.answers = [QMessageBox.Yes]
        _FakeQMessageBox.warns = []
        page2._on_delete_requested(card2)
        check(_FakeQMessageBox.warns,
              "undeletable files produce a warning dialog")
        check(page2._song_cards.get(getattr(card2, "_key", None)) is card2,
              "card kept when files remain on disk")
        check(set(card2._output_paths) == {g1, g2},
              "remaining paths stay on the card for a retry")
    finally:
        cp._remove_files_with_retry = orig_helper

    # 5. Declining the confirmation deletes nothing.
    outdir3 = os.path.join(base, "keep_model")
    os.makedirs(outdir3)
    page3 = ConsolePage()
    page3.set_input_files(["D:/in/x.mp3"])
    page3.set_job_active(True)
    card3, h1, h2 = _make_card(page3, outdir3)
    _FakeQMessageBox.answers = [QMessageBox.No]
    _FakeQMessageBox.warns = []
    page3._on_delete_requested(card3)
    check(os.path.exists(h1) and os.path.exists(h2),
          "declining the dialog keeps the files")
    check(page3._song_cards.get(getattr(card3, "_key", None)) is card3,
          "declining keeps the card")

    cp.QMessageBox = real_msgbox
    print(f"RESULT: {len(FAILURES)} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())