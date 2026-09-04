"""Regression test: CONSOLE stem outputs must never leak across model runs.

Reported bug: running several models on the same song produced mixed
waveforms — e.g. `bandit_v2_multi` (yaml stems speech/music/sfx) showed a
fourth `effects` track from the neighbouring model, `bs_mega_53stem` showed
an `instrument` track that is not in its 53-stem yaml, and `bs_inst_large2`
lost its `vocals` waveform.

Root causes fixed:

1. A job whose card never receives a completion line (failed/errored/stopped
   run) left the card stuck at "Processing...". Because the console picked
   the *alphabetically first* incomplete card as "active", that stale card
   then stole the next job's stem exports — and even the next job's
   "Completed:" mark — mixing stems across models. Fixes:
   - `_finalize_superseded_same_song`: creating a fresh card for a song
     marks any older same-song Loading/Processing card as failed (runs are
     strictly sequential per song, so it ended without completing).
   - `_ensure_active_card` / `_active_card` never select a failed card and
     never prefer an older duplicate of the same song over the newest one.
   - Reusing a leftover card with stale outputs (e.g. the very first run
     failed) resets it so old stems can't mix with the new job's.

2. A late "Wrote file:" line re-reported for a previous run's folder could
   land on the wrong card via loose song-name matching. Fix: `_match_card_
   for_export` first matches the export's own folder against each card's
   output dir (each run writes into its own per-checkpoint subfolder), and
   `_report_written_files` scans only the current model's subfolder.

This drives the real ConsolePage log parser offscreen; no torch, no network,
no subprocess.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.pages.console_page import ConsolePage  # noqa: E402

FAILURES = []
CHECKS = 0

SONG = "3 Doors Down - The Better Life - [07] - Better Life"


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def stems_of(card):
    return sorted(os.path.basename(p) for p in card._output_paths)


def feed(page, text):
    page.append_log(text + "\n")


def main():
    app = QApplication.instance() or QApplication([])

    # ── 1) sequential models, one failing mid-batch ─────────────────────
    page = ConsolePage()
    page.set_input_files(["D:/in/x.mp3"])
    page.set_job_active(True)  # whole batch keeps _job_active=True
    base = tempfile.mkdtemp(prefix="msst_leak_")
    dirs = {}
    for m in ("bs_inst_large2_unwa", "bandit_v2_multi", "bandit_plus"):
        d = os.path.join(base, m)
        os.makedirs(d, exist_ok=True)
        dirs[m] = d

    feed(page, f"Output directory: {dirs['bs_inst_large2_unwa']}")
    feed(page, f"Processing: {SONG}.mp3")
    feed(page, f"Wrote file: {dirs['bs_inst_large2_unwa']}/{SONG} (vocals).wav")
    feed(page, f"Wrote file: {dirs['bs_inst_large2_unwa']}/{SONG} (instrument).wav")
    feed(page, "Completed: processing")

    feed(page, f"Output directory: {dirs['bandit_v2_multi']}")
    feed(page, f"Processing: {SONG}.mp3")
    feed(page, f"Wrote file: {dirs['bandit_v2_multi']}/{SONG} (speech).wav")
    feed(page, "Traceback (most recent call last):")
    feed(page, "ERROR: processing failed")

    feed(page, f"Output directory: {dirs['bandit_plus']}")
    feed(page, f"Processing: {SONG}.mp3")
    feed(page, f"Wrote file: {dirs['bandit_plus']}/{SONG} (speech).wav")
    feed(page, f"Wrote file: {dirs['bandit_plus']}/{SONG} (effects).wav")
    feed(page, "Completed: processing")

    cards = dict(sorted(page._song_cards.items()))
    keys = list(cards)
    check(len(keys) == 3, f"expected 3 cards, got {keys}")
    c1, c2, c3 = (cards[k] for k in keys)

    check(c1._is_complete and not c1._failed,
          "run 1 card should be Complete")
    check(stems_of(c1) == [f"{SONG} (instrument).wav", f"{SONG} (vocals).wav"],
          f"run 1 stems wrong: {stems_of(c1)}")

    check(not c2._is_complete and c2._failed and c2._status_lbl.text() == "Failed",
          "failed run's card must be marked Failed, not stuck Processing")
    check(stems_of(c2) == [f"{SONG} (speech).wav"],
          f"failed run's own stem lost: {stems_of(c2)}")

    check(c3._is_complete and not c3._failed,
          "run 3 card should be Complete")
    check(stems_of(c3) == [f"{SONG} (effects).wav", f"{SONG} (speech).wav"],
          f"run 3 stems leaked/wrong: {stems_of(c3)}")
    check(page._unmatched_exports == [], "no exports may be stranded")

    # ── 2) late re-reported line for a previous run's folder ────────────
    page2 = ConsolePage()
    page2.set_input_files(["D:/in/x.mp3"])
    d1 = dirs["bs_inst_large2_unwa"]
    d2 = dirs["bandit_plus"]
    feed(page2, f"Output directory: {d1}")
    feed(page2, f"Processing: {SONG}.mp3")
    feed(page2, f"Wrote file: {d1}/{SONG} (instrument).wav")
    feed(page2, "Completed: processing")
    feed(page2, f"Output directory: {d2}")
    feed(page2, f"Processing: {SONG}.mp3")
    feed(page2, f"Wrote file: {d2}/{SONG} (speech).wav")
    # a late/re-reported line from run 1's folder must go back to card 1
    feed(page2, f"Wrote file: {d1}/{SONG} (instrument).wav")
    feed(page2, f"Wrote file: {d2}/{SONG} (effects).wav")
    feed(page2, "Completed: processing")
    cards2 = dict(sorted(page2._song_cards.items()))
    k2 = list(cards2)
    check(len(k2) == 2, f"expected 2 cards, got {k2}")
    a, b = (cards2[kk] for kk in k2)
    check(stems_of(a) == [f"{SONG} (instrument).wav"],
          f"card 1 got foreign stems: {stems_of(a)}")
    check(stems_of(b) == [f"{SONG} (effects).wav", f"{SONG} (speech).wav"],
          f"card 2 leaked stems: {stems_of(b)}")

    # ── 3) stale base-card reuse after a failed first run ───────────────
    page3 = ConsolePage()
    page3.set_input_files(["D:/in/x.mp3"])
    page3.set_job_active(True)
    feed(page3, f"Output directory: {d1}")
    feed(page3, f"Processing: {SONG}.mp3")
    feed(page3, f"Wrote file: {d1}/{SONG} (vocals).wav")
    feed(page3, "ERROR: processing failed")
    feed(page3, f"Output directory: {d2}")
    feed(page3, f"Processing: {SONG}.mp3")
    feed(page3, f"Wrote file: {d2}/{SONG} (speech).wav")
    feed(page3, "Completed: processing")
    cards3 = list(page3._song_cards.values())
    check(len(cards3) == 1, f"reused card expected, got {len(cards3)} cards")
    rc = cards3[0]
    check(rc._is_complete and not rc._failed,
          "reused card should end Complete")
    check(stems_of(rc) == [f"{SONG} (speech).wav"],
          f"stale stems survived reuse: {stems_of(rc)}")

    # ── 4) _ensure_active_card must never revive a failed card ──────────
    page4 = ConsolePage()
    page4.set_input_files(["D:/in/x.mp3"])
    page4.set_job_active(True)
    feed(page4, f"Output directory: {d1}")
    feed(page4, f"Processing: {SONG}.mp3")
    feed(page4, f"Wrote file: {d1}/{SONG} (a).wav")
    feed(page4, "ERROR: processing failed")
    feed(page4, f"Output directory: {d2}")
    feed(page4, f"Processing: {SONG}.mp3")
    feed(page4, f"Wrote file: {d2}/{SONG} (b).wav")
    feed(page4, "Completed: processing")
    for k, c in sorted(page4._song_cards.items()):
        if c._failed:
            check(c._status_lbl.text() == "Failed",
                  f"failed card was revived: {k} -> {c._status_lbl.text()}")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())