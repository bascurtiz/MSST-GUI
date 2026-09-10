"""Regression: CONSOLE overall batch progress bar + completion chime.

The right column (above the waveform card) shows elapsed / percent / ETA
for the whole job. resources/chime.mp3 plays once when the job ends.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from ui.pages.console_page import (  # noqa: E402
    ConsolePage, _fmt_hms, _CHIME_PATH, _OverallProgress,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def feed(page, text):
    page.append_log(text + "\n")


def main():
    app = QApplication.instance() or QApplication([])
    tmp = tempfile.mkdtemp(prefix="msst_overall_")
    songs = [f"song_karaoke_{i:03d}_mixture" for i in range(4)]
    paths = [os.path.join(tmp, s + ".wav") for s in songs]

    check(_fmt_hms(7, pad_hours=False) == "0:00:07",
          "elapsed format is H:MM:SS")
    check(_fmt_hms(756, pad_hours=True) == "00:12:36",
          "ETA format is HH:MM:SS")
    check(os.path.isfile(_CHIME_PATH),
          f"chime.mp3 is bundled at {_CHIME_PATH}")

    page = ConsolePage()
    check(isinstance(page._overall, _OverallProgress),
          "CONSOLE has an overall progress widget")
    check(page._overall.isHidden(),
          "overall bar hidden before a job starts")

    page.set_input_files(paths)
    page.set_job_active(True)
    check(not page._overall.isHidden(),
          "overall bar appears when the job starts")
    check(page._overall._pct_lbl.text() == "0%",
          f"starts at 0%, got {page._overall._pct_lbl.text()}")

    out = os.path.join(tmp, "out")
    os.makedirs(out, exist_ok=True)
    feed(page, f"Output directory: {out}")
    for s in songs:
        feed(page, f"Queued: {s}.wav")

    pct, _, _ = page._overall_progress_state()
    check(abs(pct - 0) < 0.5, f"queued-only overall is 0%, got {pct}")

    feed(page, f"Processing: {songs[0]}.wav")
    feed(page, "Processing audio chunks: 50%|#####|")
    pct, _, _ = page._overall_progress_state()
    # 1 of 4 songs at 50% → 12.5%
    check(abs(pct - 12.5) < 0.6,
          f"1/4 songs at 50% → ~12.5%, got {pct}")
    check("12%" in page._overall._pct_lbl.text()
          or "13%" in page._overall._pct_lbl.text(),
          f"label shows ~12%, got {page._overall._pct_lbl.text()}")
    check(page._overall._elapsed_lbl.text().startswith("Elapsed:"),
          f"elapsed label present: {page._overall._elapsed_lbl.text()}")
    check(page._overall._eta_lbl.text().startswith("ETA:"),
          f"ETA label present: {page._overall._eta_lbl.text()}")

    feed(page, f"Processing: {songs[1]}.wav")
    # song 0 complete (100) + song 1 loading (0) + 2 queued = 25%
    pct, _, _ = page._overall_progress_state()
    check(abs(pct - 25.0) < 0.6,
          f"1 complete of 4 → 25%, got {pct}")

    for s in songs[1:]:
        feed(page, f"Processing: {s}.wav")
        feed(page, "Processing audio chunks: 100%|##########|")
    feed(page, "Completed: processing")
    page.set_job_active(False)

    pct, _, eta = page._overall_progress_state(finished=True)
    check(abs(pct - 100.0) < 0.5, f"job done → 100%, got {pct}")
    check(page._overall._pct_lbl.text() == "100%",
          f"label 100% after job, got {page._overall._pct_lbl.text()}")
    check(eta == 0, f"finished ETA is 0, got {eta}")
    check(page._chime_player is not None or not os.path.isfile(_CHIME_PATH),
          "completion chime player was created when the job ended")

    # Multi-model session: 2 models × 4 songs → near end of model 1 ≈ 50%,
    # not ≈ 100%.
    page2 = ConsolePage()
    page2.set_input_files(paths)
    page2.set_job_active(True)
    feed(page2, "── RUNNING SELECTED MODELS (2) — same input, one run per model ──")
    feed(page2, "[1/2] Model A")
    check(page2._session_model_count == 2,
          f"session model count is 2, got {page2._session_model_count}")
    out2 = os.path.join(tmp, "out2")
    os.makedirs(out2, exist_ok=True)
    feed(page2, f"Output directory: {out2}")
    for s in songs:
        feed(page2, f"Queued: {s}.wav")
    for s in songs:
        feed(page2, f"Processing: {s}.wav")
        feed(page2, "Processing audio chunks: 100%|##########|")
    feed(page2, "Completed: processing")
    # Mid-batch True must not reset the session (model 2 starting).
    page2.set_job_active(True)
    feed(page2, "[2/2] Model B")
    pct, _, _ = page2._overall_progress_state()
    check(abs(pct - 50.0) < 1.0,
          f"after model 1 of 2 → ~50%, got {pct}")
    # Almost done with last song of model 1 would have been ~49% with
    # in-progress: simulate mid-model-1 at 98% of songs.
    page3 = ConsolePage()
    page3.set_input_files(paths)
    page3.set_job_active(True)
    feed(page3, "── RUNNING SELECTED MODELS (2) — same input, one run per model ──")
    feed(page3, "[1/2] Model A")
    for s in songs:
        feed(page3, f"Queued: {s}.wav")
    for s in songs[:-1]:
        feed(page3, f"Processing: {s}.wav")
        feed(page3, "Processing audio chunks: 100%|##########|")
    feed(page3, f"Processing: {songs[-1]}.wav")
    feed(page3, "Processing audio chunks: 96%|######### |")
    pct, _, _ = page3._overall_progress_state()
    # (3*100 + 96) / (4*2*100) * 100 = 396/800*100 = 49.5%
    check(abs(pct - 49.5) < 1.0,
          f"2-model session near end of model 1 → ~49.5%, got {pct}")

    # A second job must not count the previous run's cards.
    page.set_input_files(paths[:2])
    page.set_job_active(True)
    feed(page, f"Queued: {songs[0]}.wav")
    feed(page, f"Queued: {songs[1]}.wav")
    feed(page, f"Processing: {songs[0]}.wav")
    feed(page, "Processing audio chunks: 50%|#####|")
    pct, _, _ = page._overall_progress_state()
    check(abs(pct - 25.0) < 0.6,
          f"re-run 2-song job at 50% of first → ~25%, got {pct}")

    page._clear()
    check(page._overall.isHidden(),
          "overall bar hidden after Clear")

    labels = page._overall.findChildren(QLabel)
    check(any("Elapsed" in (w.text() or "") or w.objectName() == "overallElapsed"
              for w in labels),
          "overall bar has an Elapsed label")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print(f"ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()
