"""Regression: spectro JPG previews must not appear as Console waveforms.

With Spectro enabled, inference writes `{stem}.jpg` next to each audio stem
and used to log it as `Wrote file:`. Console treated every `Wrote file:` path
as a stem, so a 2-stem run showed duplicate Vocals/Other waveform rows.

This drives the real ConsolePage log parser offscreen; no torch, no network,
no subprocess.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.pages.console_page import ConsolePage, _stem_label  # noqa: E402

FAILURES = []
CHECKS = 0

SONG = "Ngiah Tax Olo Fotsy, Onilahy - Nahay Trotraka"


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def feed(page, text):
    page.append_log(text + "\n")


def main():
    app = QApplication.instance() or QApplication([])  # noqa: F841

    out = tempfile.mkdtemp(prefix="msst_spectro_")
    vocals_flac = os.path.join(out, f"{SONG} (vocals).flac")
    vocals_jpg = os.path.join(out, f"{SONG} (vocals).jpg")
    other_flac = os.path.join(out, f"{SONG} (other).flac")
    other_jpg = os.path.join(out, f"{SONG} (other).jpg")

    page = ConsolePage()
    page.set_input_files(["D:/in/x.mp3"])
    page.set_job_active(True)

    feed(page, f"Output directory: {out}")
    feed(page, f"Processing: {SONG}.mp3")
    feed(page, f"Wrote file: {vocals_flac}")
    feed(page, f"Wrote file: {vocals_jpg}")
    feed(page, f"Wrote file: {other_flac}")
    feed(page, f"Wrote file: {other_jpg}")
    feed(page, f"Wrote spectrogram: {vocals_jpg}")
    feed(page, f"Wrote spectrogram: {other_jpg}")
    feed(page, "Completed: processing")

    cards = list(page._song_cards.values())
    check(len(cards) == 1, f"expected 1 card, got {len(cards)}")
    if not cards:
        _report()
        return 1

    card = cards[0]
    paths = list(card._output_paths)
    check(len(paths) == 2, f"expected 2 audio paths, got {paths}")
    check(all(os.path.splitext(p)[1].lower() in (".flac", ".wav", ".mp3", ".ogg", ".m4a")
              for p in paths),
          f"non-audio path attached: {paths}")
    check(set(os.path.normpath(p) for p in paths)
          == {os.path.normpath(vocals_flac), os.path.normpath(other_flac)},
          f"paths should be the two FLACs, got {paths}")

    labels = [_stem_label(p, song_base=SONG) for p in paths]
    check(sorted(labels) == ["Other", "Vocals"],
          f"expected Vocals + Other chips, got {labels}")

    # Safety net: add_output itself must refuse a JPG even if a caller
    # bypasses the log-line filter (e.g. unmatched-export reconcile).
    check(card.add_output(vocals_jpg) is False,
          "add_output must reject spectro JPGs")
    check(len(card._output_paths) == 2,
          "add_output JPG reject must not grow the path list")

    return _report()


def _report():
    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
