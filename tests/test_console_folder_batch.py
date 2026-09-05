"""Regression test: CONSOLE folder-batch runs must advance card-by-card.

Reported bug: selecting a folder as input produced 312 job cards, but ALL
processing progress accumulated on the first card (`song_000`) and — when
done — every song's waveforms appeared on that same card, while the other
cards stayed "Queued" forever. The list also ordered cards newest-first
(`song_103, song_102, song_101 ...`) instead of the processing order.

Root causes fixed in `console_page.py`:

1. Per-song "Processing: <song>" lines for cards that already existed (the
   folder batch pre-creates every card from the "Queued:" lines) only ran
   the create-a-new-card branch's bookkeeping, so the active song never
   advanced. Fix: a "Processing:" line naming a DIFFERENT song marks the
   previously-active card Complete and hands the active slot to the new
   song, so its own tqdm progress and exports land on its own card.

2. `_match_card_for_export` trusted the export's folder as the strongest
   signal — but in a folder batch every card shares one output folder, so
   every "Wrote file:" line resolved to the first card (song_000). Fix:
   the folder pins a card only when it is unambiguous; otherwise the song
   name in the filename picks the correct card.

3. Cards were ordered newest-first. Now: the most recently
   processed/completed song sits on top (so the user always sees the latest
   progress), with still-queued songs below in natural order.

This drives the real ConsolePage log parser offscreen; no torch, no
network, no subprocess.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.pages.console_page import ConsolePage  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def feed(page, text):
    page.append_log(text + "\n")


def stems_of(card):
    return sorted(os.path.basename(p) for p in card._output_paths)


def display_order(page):
    """Card song names in the visual list, top to bottom."""
    order = []
    layout = page._output_list._card_layout
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget() if item else None
        if w is None or w.layout() is None:
            continue
        row = w.layout()
        if row.count() == 0:
            continue
        cw = row.itemAt(0).widget()
        if cw is not None and hasattr(cw, "_song_name"):
            order.append(cw._song_name)
    return order


SONGS = [f"song_dnr_{i:03d}_mixture" for i in range(5)]


def feed_batch(page, out, per_song_processing=True):
    """Feed the log stream of a 5-song folder batch. With
    per_song_processing the engine prints a 'Processing:' line per file;
    without it only the first file's line appears (export-driven mode)."""
    feed(page, f"Output directory: {out}")
    for s in SONGS:
        feed(page, f"Queued: {s}.wav")
    for i, s in enumerate(SONGS):
        if i == 0 or per_song_processing:
            feed(page, f"Processing: {s}.wav")
        feed(page, f"Processing audio chunks: {10 + i * 20}%|{'#' * (i + 1)}|")
        feed(page, f"Wrote file: {out}/{s} (speech).flac")
        if i == 0:
            feed(page, f"Wrote file: {out}/{s} (music).flac")
    feed(page, "Completed: processing")


def main():
    app = QApplication.instance() or QApplication([])

    # ── 1) folder batch with per-song Processing lines ─────────────────
    page = ConsolePage()
    page.set_input_files([f"D:/in/{s}.wav" for s in SONGS])
    page.set_job_active(True)
    out = os.path.join(tempfile.mkdtemp(prefix="msst_batch_"),
                       "bandit_30_zfturbo")
    os.makedirs(out, exist_ok=True)

    feed(page, f"Output directory: {out}")
    for s in SONGS:
        feed(page, f"Queued: {s}.wav")

    # First song starts processing; its own progress lands on its own card.
    feed(page, "Processing: song_dnr_000_mixture.wav")
    feed(page, "Processing audio chunks: 40%|####|")
    feed(page, f"Wrote file: {out}/song_dnr_000_mixture (speech).flac")
    feed(page, f"Wrote file: {out}/song_dnr_000_mixture (music).flac")
    c000 = page._output_list.get_card("song_dnr_000_mixture")
    check(c000 is not None and c000._progress == 40,
          "first song's progress must land on its own card")
    check(c000._status_lbl.text() == "Processing...",
          "first card should be processing, not Queued")

    # Moving on to song_001 completes song_000 and activates song_001.
    feed(page, "Processing: song_dnr_001_mixture.wav")
    feed(page, "Processing audio chunks: 70%|#######|")
    feed(page, f"Wrote file: {out}/song_dnr_001_mixture (speech).flac")
    check(c000._is_complete and c000._status_lbl.text() == "Complete",
          "previous song's card must complete when the next song starts")
    c001 = page._output_list.get_card("song_dnr_001_mixture")
    check(c001 is not None and c001._progress == 70
          and c001._status_lbl.text() == "Processing...",
          "next song's card must be active with its own progress")

    # The rest of the batch: each song's own Processing → progress → export.
    for i, s in enumerate(SONGS):
        if i < 2:
            continue
        feed(page, f"Processing: {s}.wav")
        feed(page, f"Processing audio chunks: {10 + i * 10}%|{'#' * (i + 1)}|")
        feed(page, f"Wrote file: {out}/{s} (speech).flac")
    feed(page, "Completed: processing")

    cards = list(page._output_list.get_cards())
    check(len(cards) == 5, f"expected 5 cards, got {len(cards)}")

    by_name = {c._song_name: c for c in cards}
    # No card may carry another song's stems (the "all 312 waveforms in
    # song_000" symptom).
    for s in SONGS:
        c = by_name[s]
        own = [f"{s} (speech).flac"]
        if s == SONGS[0]:
            own.append(f"{s} (music).flac")
        check(stems_of(c) == sorted(own),
              f"{s} stems wrong: {stems_of(c)}")
        check(c._is_complete and not c._failed,
              f"{s} should be Complete, not queued/failed")
    check(display_order(page) == SONGS[::-1],
          f"most recently processed song must be on top, got "
          f"{display_order(page)}")

    # ── 2) export-driven mode: no per-song Processing lines ────────────
    page2 = ConsolePage()
    page2.set_input_files([f"D:/in/{s}.wav" for s in SONGS])
    page2.set_job_active(True)
    out2 = os.path.join(tempfile.mkdtemp(prefix="msst_batch2_"),
                        "bandit_30_zfturbo")
    os.makedirs(out2, exist_ok=True)
    feed_batch(page2, out2, per_song_processing=False)

    cards2 = list(page2._output_list.get_cards())
    check(len(cards2) == 5, f"export-driven: expected 5 cards, got {len(cards2)}")
    by_name2 = {c._song_name: c for c in cards2}
    for s in SONGS:
        c = by_name2[s]
        own = [f"{s} (speech).flac"]
        if s == SONGS[0]:
            own.append(f"{s} (music).flac")
        check(stems_of(c) == sorted(own),
              f"export-driven {s} stems wrong: {stems_of(c)}")
        check(c._is_complete and not c._failed,
              f"export-driven {s} should be Complete")
    check(display_order(page2) == SONGS[::-1],
          f"export-driven list must put the newest song on top, got "
          f"{display_order(page2)}")

    # ── 3) single song: existing flow still completes cleanly ──────────
    page3 = ConsolePage()
    page3.set_input_files(["D:/in/only_one.wav"])
    page3.set_job_active(True)
    out3 = os.path.join(tempfile.mkdtemp(prefix="msst_batch3_"), "model_dir")
    os.makedirs(out3, exist_ok=True)
    feed(page3, f"Output directory: {out3}")
    feed(page3, "Queued: only_one.wav")
    feed(page3, "Processing: only_one.wav")
    feed(page3, "Processing audio chunks: 100%|##########|")
    feed(page3, f"Wrote file: {out3}/only_one (speech).flac")
    feed(page3, "Completed: processing")
    c3 = page3._output_list.get_card("only_one")
    check(c3 is not None and c3._is_complete
          and stems_of(c3) == ["only_one (speech).flac"],
          f"single song should complete with its stem, got "
          f"{stems_of(c3) if c3 else None}")

    # ── 4) real engine flow: Processing lines only, exports arrive in ──
    #    a single burst at the end (the engine prints no mid-run
    #    "Wrote file:" lines; the GUI scans the folder on completion).
    page5 = ConsolePage()
    page5.set_input_files([f"D:/in/{s}.flac" for s in SONGS[:4]])
    page5.set_job_active(True)
    out5 = os.path.join(tempfile.mkdtemp(prefix="msst_batch5_"),
                        "bandit_30_zfturbo")
    os.makedirs(out5, exist_ok=True)
    feed(page5, f"Output directory: {out5}")
    for s in SONGS[:4]:
        feed(page5, f"Queued: {s}.flac")
    # Engine loop: per-song "Processing:" announcements + nameless bars.
    for i, s in enumerate(SONGS[:4]):
        feed(page5, f"Processing: {s}.flac")
        feed(page5, f"Processing audio chunks:  {25 + i * 20}%|{'#' * (i + 1)}|")
        if i == 0:
            # Only the FIRST song is active right after its announcement
            # (mid-run snapshot before the next announcement).
            c0 = page5._output_list.get_card(SONGS[0])
            check(c0 is not None and c0._progress == 25
                  and c0._status_lbl.text() == "Processing...",
                  "first song active with its own progress before switch")
        elif i == 1:
            c0 = page5._output_list.get_card(SONGS[0])
            c1 = page5._output_list.get_card(SONGS[1])
            check(c0 is not None and c0._is_complete
                  and c0._status_lbl.text() == "Complete",
                  "previous song completes when the engine announces the next")
            check(c1 is not None and c1._progress == 45
                  and c1._status_lbl.text() == "Processing...",
                  "next song's own progress lands on its card")
    # Completion: the folder scan re-reports every written file at once.
    for s in SONGS[:4]:
        feed(page5, f"Wrote file: {out5}/{s} (speech).flac")
        feed(page5, f"Wrote file: {out5}/{s} (music).flac")
        feed(page5, f"Wrote file: {out5}/{s} (effects).flac")
    feed(page5, "Completed: processing")
    cards5 = list(page5._output_list.get_cards())
    check(len(cards5) == 4, f"real flow: expected 4 cards, got {len(cards5)}")
    by_name5 = {c._song_name: c for c in cards5}
    for s in SONGS[:4]:
        c = by_name5[s]
        check(stems_of(c) == [f"{s} (effects).flac", f"{s} (music).flac",
                              f"{s} (speech).flac"],
              f"real flow {s} stems wrong: {stems_of(c)}")
        check(c._is_complete and not c._failed,
              f"real flow {s} should be Complete")
    check(display_order(page5) == SONGS[:4][::-1],
          f"real flow list must put the newest song on top, got "
          f"{display_order(page5)}")

    # ── 5) re-run duplicate still routes exports to the NEW card ───────
    page4 = ConsolePage()
    page4.set_input_files(["D:/in/dup_song.wav"])
    page4.set_job_active(True)
    d1 = os.path.join(tempfile.mkdtemp(prefix="msst_batch4_"), "modelA")
    d2 = os.path.join(tempfile.mkdtemp(prefix="msst_batch4_"), "modelB")
    os.makedirs(d1, exist_ok=True)
    os.makedirs(d2, exist_ok=True)
    feed(page4, f"Output directory: {d1}")
    feed(page4, "Queued: dup_song.wav")
    feed(page4, "Processing: dup_song.wav")
    feed(page4, f"Wrote file: {d1}/dup_song (speech).flac")
    feed(page4, "Completed: processing")
    feed(page4, f"Output directory: {d2}")
    feed(page4, "Processing: dup_song.wav")
    feed(page4, f"Wrote file: {d2}/dup_song (speech).flac")
    feed(page4, f"Wrote file: {d2}/dup_song (effects).flac")
    feed(page4, "Completed: processing")
    cards4 = [c for _k, c in sorted(page4._song_cards.items())]
    check(len(cards4) == 2, f"re-run: expected 2 cards, got {len(cards4)}")
    first, second = cards4
    check(stems_of(first) == ["dup_song (speech).flac"],
          f"first run's stems wrong: {stems_of(first)}")
    check(stems_of(second) == ["dup_song (effects).flac",
                               "dup_song (speech).flac"],
          f"re-run stems must land on the NEW card, got {stems_of(second)}")

    # ── 6) auto-follow: the list scrolls to the busy song ─────────────
    page6 = ConsolePage()
    page6.set_input_files([f"D:/in/{s}.wav" for s in SONGS])
    page6.set_job_active(True)
    seen = []
    orig_scroll = page6._output_list.scroll_to_card

    def spy(card):
        seen.append(getattr(card, "_song_name", None) if card else None)
        orig_scroll(card)

    page6._output_list.scroll_to_card = spy
    out6 = os.path.join(tempfile.mkdtemp(prefix="msst_batch6_"), "m")
    os.makedirs(out6, exist_ok=True)
    feed(page6, f"Output directory: {out6}")
    for s in SONGS:
        feed(page6, f"Queued: {s}.wav")
    feed(page6, "Processing: song_dnr_000_mixture.wav")
    feed(page6, "Processing: song_dnr_001_mixture.wav")
    feed(page6, "Processing: song_dnr_002_mixture.wav")
    check("song_dnr_002_mixture" in seen,
          f"list must follow the busy song; scroll_to_card saw {seen}")
    # Mixed state: the active song on top, completed below most-recent-
    # first, still-queued songs below in natural order.
    check(display_order(page6)
          == ["song_dnr_002_mixture", "song_dnr_001_mixture",
              "song_dnr_000_mixture", "song_dnr_003_mixture",
              "song_dnr_004_mixture"],
          f"mixed list must put the busy song on top, got {display_order(page6)}")

    # The scroll call actually moves the scrollbar once the list outgrows
    # the viewport (offscreen geometry still lays out real sizes).
    page7 = ConsolePage()
    page7.set_input_files([f"D:/in/{s}.wav" for s in SONGS])
    page7.set_job_active(True)
    out7 = os.path.join(tempfile.mkdtemp(prefix="msst_batch7_"), "m")
    os.makedirs(out7, exist_ok=True)
    feed(page7, f"Output directory: {out7}")
    for s in SONGS:
        feed(page7, f"Queued: {s}.wav")
    page7.resize(320, 180)
    page7.show()
    app.processEvents()
    sb = page7._output_list._scroll.verticalScrollBar()
    c_last = page7._output_list.get_card("song_dnr_004_mixture")
    page7._output_list.scroll_to_card(c_last)
    app.processEvents()
    check(sb.maximum() > 0 and sb.value() > 0,
          f"scroll_to_card must move the scrollbar "
          f"(max={sb.maximum()}, value={sb.value()})")
    page7.hide()

    # Reorder + follow must land on the NEW position of the busy card, not
    # its stale pre-reorder geometry (symptom: the list scrolled DOWN when a
    # song completed). Drive a real run until the list outgrows the
    # viewport, then check the active card is actually visible after the
    # deferred re-scroll fires.
    page9 = ConsolePage()
    page9.set_input_files([f"D:/in/{s}.wav" for s in SONGS])
    page9.set_job_active(True)
    out9 = os.path.join(tempfile.mkdtemp(prefix="msst_batch9_"), "m")
    os.makedirs(out9, exist_ok=True)
    feed(page9, f"Output directory: {out9}")
    for s in SONGS:
        feed(page9, f"Queued: {s}.wav")
    page9.resize(320, 200)
    page9.show()
    app.processEvents()
    for s in SONGS:
        feed(page9, f"Processing: {s}.wav")
        app.processEvents()
    scroll9 = page9._output_list._scroll
    active = page9._output_list.get_card(SONGS[-1])
    # Let the deferred re-scroll (QTimer.singleShot(0)) fire.
    app.processEvents()
    sb9 = scroll9.verticalScrollBar()
    card_top = active.mapTo(scroll9.widget(), QPoint(0, 0)).y()
    card_bottom = card_top + active.height()
    in_view = (card_top >= sb9.value()
               and card_bottom <= sb9.value() + scroll9.viewport().height())
    check(in_view,
          f"after reorder the busy card must be in view "
          f"(card y={card_top}..{card_bottom}, scroll={sb9.value()}, "
          f"viewport h={scroll9.viewport().height()})")
    page9.hide()

    # ── 7) "Open folder" works mid-run: the engine announces the output
    #        folder before creating it, so cards must remember the path and
    #        only require the folder to exist when the user actually opens.
    page8 = ConsolePage()
    page8.set_input_files(["D:/in/song_a.wav", "D:/in/song_b.wav"])
    page8.set_job_active(True)
    out8 = os.path.join(tempfile.mkdtemp(prefix="msst_batch8_"), "ckpt_dir")
    feed(page8, f"Output directory: {out8}")  # dir does NOT exist yet
    feed(page8, "Queued: song_a.wav")
    feed(page8, "Queued: song_b.wav")
    ca = page8._output_list.get_card("song_a")
    check(ca._open_folder_path() is None,
          "open folder must stay inert while the dir does not exist")
    os.makedirs(out8, exist_ok=True)  # engine creates it a few seconds in
    check(ca._open_folder_path() == out8,
          "open folder must work mid-run once the dir exists "
          f"(got {ca._open_folder_path()!r})")
    check(ca._output_paths == [],
          "mid-run open folder works without any stems attached yet")

    # Cards created BEFORE the "Output directory:" line also get the path.
    page9 = ConsolePage()
    page9.set_input_files(["D:/in/song_a.wav"])
    page9.set_job_active(True)
    feed(page9, "Queued: song_a.wav")  # card exists before the dir line
    out9 = os.path.join(tempfile.mkdtemp(prefix="msst_batch9_"), "ckpt_dir")
    feed(page9, f"Output directory: {out9}")
    os.makedirs(out9, exist_ok=True)
    c9 = page9._output_list.get_card("song_a")
    check(c9._open_folder_path() == out9,
          "output-dir line must reach cards created before it "
          f"(got {c9._open_folder_path()!r})")

    # ── 8) mid-run stem reporter must NOT complete the active card ──────
    # The page re-scans the output folder on a timer, so stems for OLDER
    # (already-completed) songs keep arriving while a newer song is still
    # processing. Those late exports must attach to their own cards without
    # closing out the currently-running one.
    pageA = ConsolePage()
    pageA.set_input_files([f"D:/in/{s}.wav" for s in SONGS])
    pageA.set_job_active(True)
    outA = os.path.join(tempfile.mkdtemp(prefix="msst_batchA_"), "ckpt_dir")
    feed(pageA, f"Output directory: {outA}")
    for s in SONGS:
        feed(pageA, f"Queued: {s}.wav")
    feed(pageA, "Processing: song_dnr_000_mixture.wav")
    feed(pageA, "Processing: song_dnr_001_mixture.wav")
    feed(pageA, "Processing: song_dnr_002_mixture.wav")
    cA2 = pageA._output_list.get_card("song_dnr_002_mixture")
    check(cA2._status_lbl.text() == "Processing...",
          "song_002 must be the active card")
    # Timer tick: late stems for songs the engine already moved past.
    feed(pageA, f"Wrote file: {outA}/song_dnr_000_mixture (speech).flac")
    feed(pageA, f"Wrote file: {outA}/song_dnr_000_mixture (music).flac")
    feed(pageA, f"Wrote file: {outA}/song_dnr_001_mixture (speech).flac")
    check(not cA2._is_complete and cA2._status_lbl.text() == "Processing...",
          "late stems for older songs must NOT complete the active card")
    cA0 = pageA._output_list.get_card("song_dnr_000_mixture")
    check(len(cA0._output_paths) == 2,
          f"late stems must attach to their own (older) card, got "
          f"{len(cA0._output_paths)}")
    # The run's real end: the generic completion closes out the active card.
    feed(pageA, f"Wrote file: {outA}/song_dnr_002_mixture (speech).flac")
    feed(pageA, "Completed: processing")
    check(cA2._is_complete and cA2._status_lbl.text() == "Complete",
          "generic completion must close out the final active card")

    # ── 8) auto-select: detail view follows the top card during a run ──
    # While a job runs, the detail panel must show the newest
    # processed/completed card (the list's top entry). A manual click pins
    # the detail to that card until the next song activates.
    pageB = ConsolePage()
    pageB.set_input_files([f"D:/in/{s}.wav" for s in SONGS])
    pageB.set_job_active(True)
    outB = os.path.join(tempfile.mkdtemp(prefix="msst_batchB_"), "ckpt_dir")
    feed(pageB, f"Output directory: {outB}")
    for s in SONGS:
        feed(pageB, f"Queued: {s}.wav")
    feed(pageB, "Processing: song_dnr_000_mixture.wav")
    dB = pageB._detail_view
    olB = pageB._output_list
    cB0 = olB.get_card("song_dnr_000_mixture")
    check(dB._card is cB0,
          "job start: detail must auto-select the active (top) card")

    feed(pageB, "Processing: song_dnr_001_mixture.wav")
    cB1 = olB.get_card("song_dnr_001_mixture")
    check(dB._card is cB1,
          "next song: detail must follow the new top card")

    # Manual click on an older card pins the detail view there...
    olB._on_card_selected(cB0)
    check(dB._card is cB0, "manual click must pin the detail to that card")
    # ...and late reorder/stem activity must NOT yank the detail away.
    feed(pageB, f"Wrote file: {outB}/song_dnr_000_mixture (speech).flac")
    check(dB._card is cB0,
          "manual pick must survive reorder while the song is still active")

    # When the engine moves on, following resumes onto the new top card.
    feed(pageB, "Processing: song_dnr_002_mixture.wav")
    cB2 = olB.get_card("song_dnr_002_mixture")
    check(dB._card is cB2,
          "next song activation must resume auto-follow onto the new top")

    # Outside a job (idle), sorting must never steal the selection.
    pageB.set_job_active(False)
    olB._on_card_selected(cB0)
    pageB._sort_card_order()
    check(dB._card is cB0,
          "idle reorder must never change a manual selection")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print(f"ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()