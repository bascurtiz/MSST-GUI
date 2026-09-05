"""Regression test: Quality Checker Test (mvsep quality-checker naming).

The inference page gained a Quality Checker Test checkbox (under DEVICE)
with a dataset dropdown. When enabled, outputs are written the way mvsep's
quality_checker expects: the input's trailing "_mixture" is dropped and each
stem is suffixed per the selected dataset (multisong/synthetic -> _instrum /
_vocals, 4-stem -> _instrum/_vocals/_bass/_drums/_other, DNR v3 ->
_music/_sfx/_speech, Super Resolution -> _restored, ...).

Covered here (offscreen, no torch / network / subprocess):

  * the dataset catalog matches the mvsep leaderboards' naming rules,
  * the CONFIGURATION row shows the checkbox + a dataset dropdown that
    appears only while enabled, and the choice persists via settings,
  * the engine-side helpers (strip _mixture + stem suffix map) produce the
    exact mvsep filenames,
  * the console routes "<song>_<stem>" exports to the right cards and
    renders proper stem chips for them,
  * resample_to_native writes stems at the mixture's native rate and exact
    frame count so the checker needs no server-side resample (soxr
    44.1k->48k can come out one sample longer and be rejected).
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from ui.theme import theme_manager  # noqa: E402
import ui.pages.inference_page as ip  # noqa: E402
import backend.settings as bs  # noqa: E402
from backend.audio_names import (  # noqa: E402
    SDR_FILENAME_TEMPLATE, strip_mixture_name, parse_stem_suffix_map,
    stem_suffix_for, resample_to_native,
)
import numpy as np  # noqa: E402
from ui.pages.console_page import ConsolePage, _stem_label  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


EXPECTED_DATASETS = [
    "Multisong", "Synthetic", "Guitar", "Piano", "Medley Vox", "Strings",
    "Wind", "DNR v3", "Super Resolution", "Lead/Back Vocals", "Drums",
    "Male/Female Vocals", "Phantom Center", "Synth Vocals 2026", "MUSDB18",
]


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


def _build_page(tmp, models):
    theme_manager.init_app(QApplication.instance() or QApplication([]))
    ip._NamesFetchThread = _NamesFetchThread
    ip.QMessageBox = _FakeMsgBox
    real_load = bs.load

    def fake_load():
        d = dict(real_load())
        d["registered_models"] = list(models)
        return d

    bs.load = fake_load
    try:
        from ui.pages.inference_page import InferencePage
        page = InferencePage()
    finally:
        bs.load = real_load
    for m in models:
        page.on_model_registered(m)
    return page


def _model(tmp, name, model_type="bs_roformer"):
    ckpt = os.path.join(tmp, name + ".ckpt")
    yaml = os.path.join(tmp, name + ".yaml")
    open(ckpt, "w", encoding="utf-8").close()
    open(yaml, "w", encoding="utf-8").close()
    return {"name": name, "ckpt": ckpt, "yaml": yaml,
            "arch": "BS Roformer Architecture", "model_type": model_type,
            "type": "multi stems"}


def main():
    app = QApplication.instance() or QApplication([])
    tmp = tempfile.mkdtemp(prefix="msst_qc_")

    # ── 1) catalog matches the mvsep leaderboards ─────────────────────
    names = [d[0] for d in ip.SDR_DATASETS]
    check(names == EXPECTED_DATASETS,
          f"catalog lists the expected datasets, got {names}")

    multisong_map, needs = ip._sdr_dataset("Multisong")
    check(stem_suffix_for("vocals", multisong_map) == "vocals"
          and stem_suffix_for("other", multisong_map) == "instrum"
          and stem_suffix_for("instrumental", multisong_map) == "instrum",
          "multisong/synthetic maps vocals + instrumental names")
    check(needs is False, "multisong does not need extract_instrumental")

    musdb_map, musdb_needs = ip._sdr_dataset("MUSDB18")
    check(musdb_needs is True, "MUSDB18 derives _instrum via mix minus vocals")
    check(stem_suffix_for("instrumental", musdb_map) == "instrum"
          and stem_suffix_for("other", musdb_map) == "other",
          "MUSDB18 keeps _other and maps the engine instrumental to _instrum")

    dnr_map, _ = ip._sdr_dataset("DNR v3")
    check(stem_suffix_for("effects", dnr_map) == "sfx"
          and stem_suffix_for("speech", dnr_map) == "speech"
          and stem_suffix_for("music", dnr_map) == "music",
          "DNR v3 maps effects->sfx and keeps speech/music")

    sr_map, _ = ip._sdr_dataset("Super Resolution")
    check(stem_suffix_for("anything", sr_map) == "restored",
          "Super Resolution catch-all names every output _restored")

    guitar_map, _ = ip._sdr_dataset("Guitar")
    check(stem_suffix_for("guitar", guitar_map) == "guitar"
          and stem_suffix_for("other", guitar_map) == "other",
          "guitar/piano/strings/wind keep identity stems")

    # ── 2) engine naming: strip _mixture + dataset suffix ─────────────
    def qc_name(input_base, instr, stem_map):
        suffix = stem_suffix_for(instr, stem_map)
        return SDR_FILENAME_TEMPLATE.format(
            file_name=strip_mixture_name(input_base), instr=suffix)

    check(qc_name("song_dnr_016_mixture", "effects", dnr_map)
          == "song_dnr_016_sfx", "DNR naming: song_dnr_016_mixture + effects")
    check(qc_name("melody_086_mixture", "vocals", multisong_map)
          == "melody_086_vocals", "multisong naming: melody_086_vocals")
    check(qc_name("melody_086_mixture", "other", multisong_map)
          == "melody_086_instrum", "multisong naming: melody_086_instrum")
    check(qc_name("song_plain", "vocals", multisong_map) == "song_plain_vocals",
          "no _mixture suffix is stripped only when present")
    check(qc_name("musdb_086_mixture", "instrumental", musdb_map)
          == "musdb_086_instrum", "MUSDB18 naming: musdb_086_instrum")
    check(qc_name("song_sr_016_mixture", "restored", sr_map)
          == "song_sr_016_restored", "SR naming: song_sr_016_restored")
    check(parse_stem_suffix_map(ip._sdr_map_text(dnr_map))
          == dnr_map, "map text round-trips through the engine parser")

    # ── 3) UI row: checkbox + dataset dropdown visibility ─────────────
    page = _build_page(tmp, [_model(tmp, "qc_model")])
    row = page._sdr_row
    labels = [w.text() for w in row.findChildren(QLabel)]
    check(any("Quality Checker Test" == t for t in labels),
          f"row label reads 'Quality Checker Test', got {labels}")
    check(row.combo.count() == len(EXPECTED_DATASETS),
          f"dropdown lists all datasets, got {row.combo.count()}")
    row.check.setChecked(False)
    app.processEvents()
    check(row.combo.isHidden() and row._arrow.isHidden(),
          "dropdown hidden while disabled")
    row.check.setChecked(True)
    app.processEvents()
    check(not row.combo.isHidden() and not row._arrow.isHidden(),
          "dropdown appears once enabled")
    check(row.combo.currentText() == "Multisong",
          "dropdown defaults to Multisong")

    # The empty-text checkbox must not expand into the hidden dropdown's
    # space (Qt would center the indicator); both states render at the
    # same natural width so the indicator stays left-aligned.
    check(row.check.sizePolicy().horizontalPolicy().name == "Fixed",
          "checkbox pinned to natural width (Fixed policy)")
    page.resize(760, 1000)
    page.show()
    app.processEvents()
    row.check.setChecked(True)
    app.processEvents()
    checked_right = row.check.geometry().right()
    row.check.setChecked(False)
    app.processEvents()
    unchecked_right = row.check.geometry().right()
    check(checked_right > 0 and checked_right == unchecked_right,
          "checkbox stays in the right icon column in both states "
          "(right edge %d vs %d)" % (checked_right, unchecked_right))
    page.hide()

    # ── 4) persistence via settings ───────────────────────────────────
    row.check.setChecked(True)
    row.combo.setCurrentIndex(row.combo.findText("DNR v3"))
    saved = page.save_settings()
    check(saved.get("sdr_test") is True
          and saved.get("sdr_dataset") == "DNR v3",
          "save_settings stores sdr_test + sdr_dataset")
    page2 = _build_page(tmp, [_model(tmp, "qc_model")])
    page2.load_settings({"sdr_test": True, "sdr_dataset": "DNR v3"})
    check(page2._sdr_check.isChecked()
          and page2._sdr_combo.currentText() == "DNR v3",
          "load_settings restores the checkbox and dataset")

    # ── 5) console routing + stem chips for <song>_<stem> names ───────
    out = os.path.join(tempfile.mkdtemp(prefix="msst_qc_console_"),
                       "bandit_30_zfturbo")
    os.makedirs(out)
    cpage = ConsolePage()
    cpage.set_input_files([f"{out}/../song_dnr_{i:03d}_mixture.flac"
                           for i in range(2)])
    cpage.set_job_active(True)

    def feed(t):
        cpage.append_log(t + "\n")

    feed(f"Output directory: {out}")
    feed("Queued: song_dnr_000_mixture.flac")
    feed("Queued: song_dnr_001_mixture.flac")
    feed("Processing: song_dnr_000_mixture.flac")
    feed("Processing audio chunks: 50%|#####|")
    feed("Processing: song_dnr_001_mixture.flac")
    feed("Processing audio chunks: 80%|########|")
    # End-of-run folder scan reports every written file at once (the engine
    # prints no mid-run "Wrote file:" lines).
    for i in range(2):
        for stem in ("speech", "music", "sfx"):
            feed(f"Wrote file: {out}/song_dnr_{i:03d}_{stem}.flac")
    feed("Completed: processing")

    cards = {c._song_name: c for c in cpage._output_list.get_cards()}
    check(len(cards) == 2, f"two cards created, got {list(cards)}")
    for i in range(2):
        c = cards[f"song_dnr_{i:03d}_mixture"]
        check(sorted(os.path.basename(p) for p in c._output_paths)
              == [f"song_dnr_{i:03d}_music.flac",
                  f"song_dnr_{i:03d}_sfx.flac",
                  f"song_dnr_{i:03d}_speech.flac"],
              f"song_dnr_{i:03d}: mvsep stems on its own card")
    check(_stem_label(f"{out}/song_dnr_000_speech.flac",
                      "song_dnr_000_mixture") == "Speech",
          "stem chip parses _speech suffix")
    check(_stem_label(f"{out}/song_dnr_000_sfx.flac",
                      "song_dnr_000_mixture") == "Sfx",
          "stem chip parses _sfx suffix")
    check(_stem_label(f"{out}/song_dnr_000_mixture (speech).flac",
                      "song_dnr_000_mixture") == "Speech",
          "classic '(speech)' naming still parses")

    # ── 6) stems match the mixture's native rate + frame count ────────
    # mvsep's checker compares uploaded stems against the reference mixture
    # frame-for-frame at its native rate. A 44.1 kHz model on a 48 kHz DNR v3
    # mixture uploads stems at 44.1 kHz; the server-side soxr resample can
    # then come out one sample LONGER (2646000 -> 2880001) and be rejected
    # with "Different shapes for wav file ... 2880001 != 2880000". The helper
    # must write at the mixture rate and exact frame count.
    a = np.random.RandomState(0).randn(2, 2646000).astype(np.float32)
    b = resample_to_native(a, 44100, 48000, 2880000)
    check(b.shape == (2, 2880000),
          f"44.1k->48k stem matches mixture: {b.shape} != (2, 2880000)")
    check(not np.isnan(b).any() and np.abs(b).max() <= 6.0,
          "resampled stem is finite and sane (peak %.3f)" % np.abs(b).max())

    # Same rate, longer output (e.g. chunk padding survived): trimmed to the
    # exact mixture length, keeping the head samples.
    c = np.arange(10, dtype=np.float32).reshape(1, 10)
    d = resample_to_native(c, 48000, 48000, 5)
    check(d.shape == (1, 5) and np.array_equal(d[0], c[0, :5]),
          "same-rate longer stem is trimmed to the mixture length")

    # Same rate, shorter output: zero-padded at the tail.
    e = resample_to_native(c, 48000, 48000, 12)
    check(e.shape == (1, 12) and np.array_equal(e[0, :10], c[0])
          and np.count_nonzero(e[0, 10:]) == 0,
          "same-rate shorter stem is zero-padded to the mixture length")

    # Same rate and length: untouched (identity, no resample/trimming).
    f = np.ascontiguousarray(c)
    g = resample_to_native(f, 48000, 48000, 10)
    check(g.shape == (1, 10) and np.array_equal(g, f),
          "matching rate/length passes through unchanged")

    # Channel layout must match the mixture too: the engine duplicates a
    # mono input to stereo for stereo models, so stems come out 2-channel
    # while the reference (e.g. a mono DNR v3 mixture) is 1-channel. The
    # helper must downmix stereo stems back to mono by averaging.
    st = np.stack([np.arange(1, 11, dtype=np.float32),
                   np.arange(11, 21, dtype=np.float32)])  # (2, 10)
    mono = resample_to_native(st, 48000, 48000, 10, target_channels=1)
    expect = ((st[0] + st[1]) / 2.0).reshape(1, 10)
    check(mono.shape == (1, 10)
          and np.allclose(mono, expect, atol=1e-6),
          "stereo stem downmixed to mono by averaging")

    # Symmetric upmix: a mono model output duplicated back to stereo.
    up = resample_to_native(mono, 48000, 48000, 10, target_channels=2)
    check(up.shape == (2, 10) and np.array_equal(up[0], up[1])
          and np.allclose(up[0], expect[0], atol=1e-6),
          "mono stem upmixed to stereo by duplication")

    # Combined: stereo 44.1 kHz stem -> mono 48 kHz mixture, exact length.
    big = np.random.RandomState(1).randn(2, 2646000).astype(np.float32)
    comb = resample_to_native(big, 44100, 48000, 2880000,
                              target_channels=1)
    check(comb.shape == (1, 2880000) and not np.isnan(comb).any(),
          "resample + downmix in one call matches the mono mixture")

    # Matching channel count is left alone (no downmix/upmix).
    keep = resample_to_native(mono, 48000, 48000, 10, target_channels=1)
    check(keep.shape == (1, 10) and np.allclose(keep, expect, atol=1e-6),
          "matching channel count passes through unchanged")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print(f"ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()