"""Regression tests: mvsep Quality Checker scores in the Model Library and
Model Manager.

A published Google Sheet maps checkpoint filenames to mvsep quality-checker
entry URLs; each entry page lists per-stem metrics (SDR, SI-SDR, L1-FREQ,
LOG-WMSE, AURA-STFT, AURA-MRSTFT, BLEEDLESS, FULLNESS). The Model Library
shows a link icon + a per-stem SDR line and can sort by any metric; the Model
Manager can sort by upload date or a metric and swaps the 'Updated … size'
line for the metric's per-stem values.

Covered here (offscreen, no network):

  * sheet-style CSV rows parse into filename -> mvsep URL entries,
  * entry-page HTML parses into stems + per-stem metrics for 2-, 3- and
    4-stem models (the page renders metrics with <br />, not newlines),
  * the display lines ('SDR effects: 10.74 | music: 8.28 | sfx: 9.45') and
    mean-metric sort keys match for every shape,
  * the disk cache round-trips and stale entries are refetched,
  * the library row grows its SDR line and link icon when scores exist, and
    _ArchCard.sort_models orders rows by the chosen metric.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from ui.theme import theme_manager  # noqa: E402
from ui.pages import inference_page as ip  # noqa: E402
from ui.pages.inference_page import (_ArchCard, _ModelItem, _SortCombo,
                                      _score_lookup_keys)  # noqa: E402
import backend.mvsep_scores as ms  # noqa: E402

FAILURES = []
CHECKS = 0

SAMPLE_ENTRIES_CSV = """Filename,URL,Architecture
bandit_30_zfturbo.ckpt,https://mvsep.com/quality_checker/entry/10393,Bandit
bs_pope_4stem_09072026_aname.ckpt,https://mvsep.com/quality_checker/entry/10412,BS Roformer
mdx_1_9703.onnx,https://mvsep.com/quality_checker/entry/10455,MDX
mbr_wsa.ckpt,https://mvsep.com/quality_checker/entry/10400,Bandit
"""


def _use_entries_cache(tmp_dir):
    """Point the entries cache at a temp dir and reset the in-memory map."""
    ms.ENTRIES_CACHE_PATH = os.path.join(tmp_dir, "entries_cache.json")
    ms._ENTRIES = None


def _seed_entries_cache():
    entries = ms._parse_entries_csv(SAMPLE_ENTRIES_CSV)
    ms._save_entries_cache(entries)
    ms._ENTRIES = None
    return entries


def check(msg, cond):
    global CHECKS
    CHECKS += 1
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


# ── Fixtures: page HTML exactly as mvsep renders it (metrics on <br /> lines)

HTML_2STEM = """<b>Metrics:</b><br />
Metric sdr for instrum: 14.6307<br />
Metric si_sdr for instrum: 14.5121<br />
Metric l1_freq for instrum: 34.4169<br />
Metric log_wmse for instrum: 11.6391<br />
Metric aura_stft for instrum: 12.4927<br />
Metric aura_mrstft for instrum: 15.1513<br />
Metric bleedless for instrum: 33.8754<br />
Metric fullness for instrum: 29.8576<br />
Metric sdr for vocals: 8.3232<br />
Metric si_sdr for vocals: 7.4393<br />
Metric l1_freq for vocals: 31.2333<br />
Metric log_wmse for vocals: 11.6391<br />
Metric aura_stft for vocals: 5.2792<br />
Metric aura_mrstft for vocals: 5.6483<br />
Metric bleedless for vocals: 30.3629<br />
Metric fullness for vocals: 11.4191<br />
"""

HTML_3STEM = """<b>Metrics:</b><br />
Metric sdr for music: 8.2824<br />
Metric si_sdr for music: 7.3786<br />
Metric l1_freq for music: 64.2916<br />
Metric log_wmse for music: 14.2247<br />
Metric aura_stft for music: 6.8169<br />
Metric aura_mrstft for music: 7.5462<br />
Metric bleedless for music: 15.5726<br />
Metric fullness for music: 16.8895<br />
Metric sdr for sfx: 9.4532<br />
Metric si_sdr for sfx: 8.9623<br />
Metric l1_freq for sfx: 58.5955<br />
Metric log_wmse for sfx: 12.9684<br />
Metric aura_stft for sfx: 9.4918<br />
Metric aura_mrstft for sfx: 10.5272<br />
Metric bleedless for sfx: 22.8320<br />
Metric fullness for sfx: 25.0946<br />
Metric sdr for speech: 10.7387<br />
Metric si_sdr for speech: 10.3070<br />
Metric l1_freq for speech: 72.5506<br />
Metric log_wmse for speech: 15.9120<br />
Metric aura_stft for speech: 5.8109<br />
Metric aura_mrstft for speech: 5.3945<br />
Metric bleedless for speech: 22.6031<br />
Metric fullness for speech: 22.3444<br />
"""

HTML_4STEM = """<b>Metrics:</b><br />
Metric sdr for vocals: 5.0231<br />
Metric si_sdr for vocals: 3.3197<br />
Metric l1_freq for vocals: 26.0042<br />
Metric log_wmse for vocals: 8.8673<br />
Metric aura_stft for vocals: 5.8286<br />
Metric aura_mrstft for vocals: 5.9901<br />
Metric bleedless for vocals: 23.1504<br />
Metric fullness for vocals: 12.1634<br />
Metric sdr for bass: 7.4481<br />
Metric si_sdr for bass: 6.5344<br />
Metric l1_freq for bass: 41.0445<br />
Metric log_wmse for bass: 13.6917<br />
Metric aura_stft for bass: 5.0711<br />
Metric aura_mrstft for bass: 3.8860<br />
Metric bleedless for bass: 23.3530<br />
Metric fullness for bass: 17.1725<br />
Metric sdr for drums: 7.7484<br />
Metric si_sdr for drums: 6.6948<br />
Metric l1_freq for drums: 30.8083<br />
Metric log_wmse for drums: 13.0691<br />
Metric aura_stft for drums: 7.2608<br />
Metric aura_mrstft for drums: 6.2834<br />
Metric bleedless for drums: 17.5832<br />
Metric fullness for drums: 15.7011<br />
Metric sdr for other: 3.9940<br />
Metric si_sdr for other: 1.8213<br />
Metric l1_freq for other: 26.7301<br />
Metric log_wmse for other: 9.1520<br />
Metric aura_stft for other: 5.2161<br />
Metric aura_mrstft for other: 7.2870<br />
Metric bleedless for other: 14.7484<br />
Metric fullness for other: 15.9727<br />
"""


def test_parsing():
    p2 = ms.parse_entry_html(HTML_2STEM)
    check("2-stem stems in page order", p2["stems"] == ["instrum", "vocals"])
    check("2-stem sdr value", p2["metrics"]["vocals"]["sdr"] == 8.3232)
    check("2-stem si_sdr value", p2["metrics"]["instrum"]["si_sdr"] == 14.5121)

    p3 = ms.parse_entry_html(HTML_3STEM)
    check("3-stem stems", p3["stems"] == ["music", "sfx", "speech"])
    check("3-stem sdr speech", p3["metrics"]["speech"]["sdr"] == 10.7387)
    check("3-stem fullness sfx", p3["metrics"]["sfx"]["fullness"] == 25.0946)

    p4 = ms.parse_entry_html(HTML_4STEM)
    check("4-stem stems", p4["stems"] == ["vocals", "bass", "drums", "other"])
    check("4-stem sdr other", p4["metrics"]["other"]["sdr"] == 3.9940)
    check("4-stem l1_freq drums", p4["metrics"]["drums"]["l1_freq"] == 30.8083)
    check("8 metrics per stem", len(p4["metrics"]["vocals"]) == 8)


def test_lines_and_keys():
    p4 = ms.parse_entry_html(HTML_4STEM)
    line = ms.sdr_line(p4)
    check("4-stem SDR line", line ==
          "SDR vocals: 5.02 | bass: 7.45 | drums: 7.75 | other: 3.99")
    check("4-stem mean sdr", abs(ms.mean_metric(p4, "sdr") - 6.0534) < 1e-3)
    check("4-stem mean aura_stft",
          abs(ms.mean_metric(p4, "aura_stft") - 5.8442) < 1e-3)

    p3 = ms.parse_entry_html(HTML_3STEM)
    check("3-stem SDR line", ms.sdr_line(p3) ==
          "SDR music: 8.28 | sfx: 9.45 | speech: 10.74")
    check("3-stem bleedless line", ms.metric_line(p3, "bleedless") ==
          "BLEEDLESS music: 15.57 | sfx: 22.83 | speech: 22.60")

    p2 = ms.parse_entry_html(HTML_2STEM)
    check("2-stem SDR line", ms.sdr_line(p2) ==
          "SDR instrum: 14.63 | vocals: 8.32")

    check("no scores -> empty line", ms.sdr_line(None) == "")
    check("no scores -> no mean", ms.mean_metric(None, "sdr") is None)
    check("missing metric -> no mean",
          ms.mean_metric(p2, "does_not_exist") is None)


def test_sheet_entries():
    tmp = tempfile.mkdtemp(prefix="msst_entries_")
    old = ms.ENTRIES_CACHE_PATH
    try:
        _use_entries_cache(tmp)
        parsed = ms._parse_entries_csv(SAMPLE_ENTRIES_CSV)
        check("sample sheet parses four entries", len(parsed) == 4)
        check("bandit url", parsed.get("bandit_30_zfturbo.ckpt") ==
              "https://mvsep.com/quality_checker/entry/10393")
        check("pope url", parsed.get("bs_pope_4stem_09072026_aname.ckpt") ==
              "https://mvsep.com/quality_checker/entry/10412")
        check("onnx url", parsed.get("mdx_1_9703.onnx") ==
              "https://mvsep.com/quality_checker/entry/10455")
        check("keys stored lowercase", "mbr_wsa.ckpt" in parsed and
              "MBR_WSA.CKPT" not in parsed)
        bad = [k for k, v in parsed.items()
               if not v.startswith("https://mvsep.com/quality_checker/entry/")]
        check("all urls are mvsep entries", not bad)
        seeded = _seed_entries_cache()
        loaded = ms.load_entries()
        check("load_entries reads cached sheet snapshot", loaded == seeded)
    finally:
        ms.ENTRIES_CACHE_PATH = old
        ms._ENTRIES = None


def test_cache():
    tmp = tempfile.mkdtemp(prefix="msst_scores_")
    old = ms.CACHE_PATH
    ms.CACHE_PATH = os.path.join(tmp, "cache.json")
    try:
        cache = {}
        entry = {
            "url": "https://mvsep.com/quality_checker/entry/10412",
            "fetched_at": "2026-09-08T12:00:00+00:00",
            "stems": ["vocals", "bass"],
            "metrics": {"vocals": {"sdr": 5.02}, "bass": {"sdr": 7.45}},
        }
        cache["bs_pope_4stem_09072026_aname.ckpt"] = entry
        ms._save_cache(cache)
        loaded = ms._load_cache()
        check("cache round-trips", loaded == cache)
        check("fresh entry not stale", not ms._is_stale(loaded["bs_pope_4stem_09072026_aname.ckpt"]))
        stale = dict(entry, fetched_at="2020-01-01T00:00:00+00:00")
        check("old entry stale", ms._is_stale(stale))
        check("broken date stale", ms._is_stale({"fetched_at": "nope"}))
    finally:
        ms.CACHE_PATH = old


def test_store_no_network():
    tmp = tempfile.mkdtemp(prefix="msst_scores_store_")
    old_cache = ms.CACHE_PATH
    old_entries = ms.ENTRIES_CACHE_PATH
    ms.CACHE_PATH = os.path.join(tmp, "cache.json")
    _use_entries_cache(tmp)
    try:
        _seed_entries_cache()
        _test_store_no_network()
    finally:
        ms.CACHE_PATH = old_cache
        ms.ENTRIES_CACHE_PATH = old_entries
        ms._ENTRIES = None


def _test_store_no_network():
    open(ms.CACHE_PATH, "w", encoding="utf-8").write("{}")
    store = ms.ScoresStore()
    store.request("bandit_30_zfturbo.ckpt")
    store.request("not_in_sheet.ckpt")
    check("request accepted sheet entry", "bandit_30_zfturbo.ckpt" in store._needed)
    check("request ignored unknown", "not_in_sheet.ckpt" not in store._needed)
    check("entry_url known", store.entry_url("bandit_30_zfturbo.ckpt") ==
          "https://mvsep.com/quality_checker/entry/10393")
    check("entry_url unknown empty", store.entry_url("nope.ckpt") == "")
    check("get missing -> None", store.get("bandit_30_zfturbo.ckpt") is None)


def test_library_row_and_sort():
    app = QApplication.instance() or QApplication([])
    theme_manager.init_app(app)
    tmp = tempfile.mkdtemp(prefix="msst_scores_ui_")
    yaml_path = os.path.join(tmp, "m.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("training:\n  instruments: [vocals, other]\n")

    card = _ArchCard("Melband Roformer Architecture")
    url = "https://mvsep.com/quality_checker/entry/10412"
    p2 = ms.parse_entry_html(HTML_2STEM)   # mean sdr 11.477
    p4 = ms.parse_entry_html(HTML_4STEM)   # mean sdr 6.053
    card.add_model("a_vocals.ckpt", os.path.join(tmp, "a_vocals.ckpt"),
                   yaml_path, "Melband Roformer Architecture", "vocals",
                   "mel_band_roformer", scores=p2, scores_url=url)
    card.add_model("b_inst.ckpt", os.path.join(tmp, "b_inst.ckpt"),
                   yaml_path, "Melband Roformer Architecture", "instrumental",
                   "mel_band_roformer")
    card.add_model("c_4stem.ckpt", os.path.join(tmp, "c_4stem.ckpt"),
                   yaml_path, "Melband Roformer Architecture", "multi stems",
                   "mel_band_roformer", scores=p4)

    items = {w._name: w for w in card.findChildren(_ModelItem)}
    check("three models added", len(items) == 3)

    scored = items["a_vocals.ckpt"]
    check("scored row has SDR line visible", not scored._scores_lbl.isHidden())
    # Two-row metric block: name (lowercase) on the left, per-stem label
    # above value.
    check("scored row shows per-stem SDR",
          scored._scores_lbl._metric_lbl.text() == "sdr"
          and scored._scores_lbl._labels[0].text() == "INSTRUM"
          and scored._scores_lbl._values[0].text() == "14.63")
    check("scored row taller", scored.height() > 38)
    check("scored row has link badge", scored._link is not None)

    # Real registrations may use a friendly/display name while the checkpoint
    # path carries the CSV filename. Both identities must attach the same
    # cached scores (the production path uses the latter first).
    path_identity = _ModelItem(
        "Friendly model name", os.path.join(tmp, "a_vocals.ckpt"),
        yaml_path, "Melband Roformer Architecture", scores=p2,
        scores_url=url)
    check("score row accepts checkpoint-path identity",
          path_identity._scores_lbl._values[0].text() == "14.63")
    path_identity.deleteLater()
    scored.set_metric("aura_stft")
    check("library row follows selected metric",
          scored._scores_lbl._metric_lbl.text() == "aura-stft"
          and scored._scores_lbl._values[0].text() == "12.49")
    scored.set_metric("sdr")

    unscored = items["b_inst.ckpt"]
    check("unscored row no SDR line", unscored._scores_lbl.isHidden())
    check("unscored row no link", unscored._link is None)
    check("unscored row base height", unscored.height() == 39)

    # Sort by SDR, high→low, unscored last.
    card.sort_models(lambda w: (
        1, 0.0) if not w._scores else (0, -ms.mean_metric(w._scores, "sdr")))
    order = []
    for i in range(card._list_vl.count()):
        w = card._list_vl.itemAt(i).widget()
        if isinstance(w, _ModelItem):
            order.append(w._name)
    # a (mean 11.48) > c (mean 6.05) > b (no scores)
    check("metric sort high->low, unscored last",
          order == ["a_vocals.ckpt", "c_4stem.ckpt", "b_inst.ckpt"])

    # Name sort restores alphabetical order.
    card.sort_models(lambda w: (0, (w._display or w._name).lower()))
    order = []
    for i in range(card._list_vl.count()):
        w = card._list_vl.itemAt(i).widget()
        if isinstance(w, _ModelItem):
            order.append(w._name)
    check("name sort alphabetical", order ==
          ["a_vocals.ckpt", "b_inst.ckpt", "c_4stem.ckpt"])

    # Scores landing later (async fetch) — row grows + text appears.
    unscored.set_scores(ms.parse_entry_html(HTML_2STEM))
    check("late scores attach", not unscored._scores_lbl.isHidden())
    check("late scores text",
          unscored._scores_lbl._metric_lbl.text() == "sdr"
          and unscored._scores_lbl._labels[0].text() == "INSTRUM"
          and unscored._scores_lbl._values[0].text() == "14.63")
    check("stem labels are uppercase",
          all(lbl.text() == lbl.text().upper()
              for lbl in unscored._scores_lbl._labels))
    check("late scores height", unscored.height() > 38)


def test_score_lookup_keys_ckpt_suffix():
    """Stem-only model names still resolve to CSV keys ending in .ckpt."""
    keys = _score_lookup_keys({"name": "mbr_deux_becruily", "ckpt": ""})
    check("stem expands to .ckpt key", "mbr_deux_becruily.ckpt" in keys)


def test_score_lookup_keys_friendly():
    """Reverse-map zoo full names back to CSV checkpoint keys."""
    friendly = {
        "bs_leap_xe_voc_unwa.ckpt":
        "BS Roformer Leap XE (90 bands) Vocals by Unwa",
    }
    keys = _score_lookup_keys(
        {"name": "BS Roformer Leap XE (90 bands) Vocals by Unwa", "ckpt": ""},
        friendly,
    )
    check("friendly name resolves to ckpt key",
          "bs_leap_xe_voc_unwa.ckpt" in keys)
    keys2 = _score_lookup_keys(
        {"name": "other.ckpt", "ckpt": "", "display": friendly[
            "bs_leap_xe_voc_unwa.ckpt"]},
        friendly,
    )
    check("display label resolves to ckpt key",
          "bs_leap_xe_voc_unwa.ckpt" in keys2)


def test_l1_freq_row_height():
    """Metric block uses dynamic height and shows l1-freq per-stem values."""
    app = QApplication.instance() or QApplication([])
    theme_manager.init_app(app)
    p2 = ms.parse_entry_html(HTML_2STEM)
    item = _ModelItem("a_vocals.ckpt", scores=p2)
    item.set_metric("l1_freq")
    check("l1-freq metric label",
          item._scores_lbl._metric_lbl.text() == "l1-freq")
    check("l1-freq instrum value",
          item._scores_lbl._values[0].text() == "34.42")
    check("l1-freq row taller than name row", item.height() > 38)
    check("l1-freq stem labels uppercase",
          item._scores_lbl._labels[0].text() == "INSTRUM"
          and item._scores_lbl._labels[1].text() == "VOCALS")
    item.deleteLater()


def test_attach_cached_scores_without_signal():
    """Cached scores attach via _attach_cached_scores when scores_ready
    never fired (fresh disk cache entries are not re-emitted)."""
    app = QApplication.instance() or QApplication([])
    theme_manager.init_app(app)
    tmp = tempfile.mkdtemp(prefix="msst_attach_")
    ckpt_path = os.path.join(tmp, "a_vocals.ckpt")
    yaml_path = os.path.join(tmp, "m.yaml")
    with open(ckpt_path, "w", encoding="utf-8"):
        pass
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("training:\n  instruments: [vocals, other]\n")

    store = ms.get_scores_store()
    parsed = ms.parse_entry_html(HTML_2STEM)
    store._cache.pop("a_vocals.ckpt", None)

    page = ip.InferencePage()
    page.on_model_registered({
        "name": "a_vocals.ckpt",
        "ckpt": ckpt_path,
        "yaml": yaml_path,
        "arch": "Melband Roformer Architecture",
        "type": "vocals",
        "model_type": "mel_band_roformer",
    })
    page._flush_library_visibility()

    card = page._arch_cards["Melband Roformer Architecture"]
    item = card.findChildren(_ModelItem)[0]
    check("row starts without cached scores", item._scores_lbl.isHidden())

    store._cache["a_vocals.ckpt"] = {
        "url": "https://mvsep.com/quality_checker/entry/10412",
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "stems": parsed["stems"],
        "metrics": parsed["metrics"],
    }
    page._attach_cached_scores()
    item.set_metric("sdr")
    check("re-attach from cache without signal",
          not item._scores_lbl.isHidden()
          and item._scores_lbl.has_content()
          and item._scores_lbl._values[0].text() == "14.63"
          and item.height() > 38)


def test_worker_refetches_when_url_changes():
    """A changed mvsep entry URL must not reuse the old cached scores."""
    tmp = tempfile.mkdtemp(prefix="msst_url_change_")
    old_cache = ms.CACHE_PATH
    old_entries = ms.ENTRIES_CACHE_PATH
    ms.CACHE_PATH = os.path.join(tmp, "cache.json")
    _use_entries_cache(tmp)
    try:
        key = "bs_vox_xlancer.ckpt"
        old_url = "https://mvsep.com/quality_checker/entry/10000"
        new_url = "https://mvsep.com/quality_checker/entry/10626"
        _seed_entries_cache()
        ms._save_entries_cache({key: new_url})
        ms._ENTRIES = None
        ms._save_cache({
            key: {
                "url": old_url,
                "fetched_at": "2026-09-08T12:00:00+00:00",
                "stems": ["instrum"],
                "metrics": {"instrum": {"sdr": 1.0}},
            }
        })
        def _fake_fetch(url):
            assert url == new_url
            return {
                "stems": ["instrum", "vocals"],
                "metrics": {
                    "instrum": {"sdr": 10.0},
                    "vocals": {"sdr": 5.0},
                },
            }

        old_fetch = ms.fetch_entry
        ms.fetch_entry = _fake_fetch
        store = ms.ScoresStore()
        store._needed.add(key)
        try:
            with store._worker_lock:
                store._worker_pass({key: new_url})
        finally:
            ms.fetch_entry = old_fetch
        got = store.get(key)
        check("url change triggers refetch", got is not None)
        check("refetched url stored", store._cache[key]["url"] == new_url)
        check("old single-stem cache replaced",
              "vocals" in (got.get("metrics") or {}))
    finally:
        ms.CACHE_PATH = old_cache
        ms.ENTRIES_CACHE_PATH = old_entries
        ms._ENTRIES = None


def test_request_queues_cached_refresh():
    """request() must emit scores_ready for fresh cache hits (no refetch)."""
    tmp = tempfile.mkdtemp(prefix="msst_scores_refresh_")
    old = ms.CACHE_PATH
    old_entries = ms.ENTRIES_CACHE_PATH
    ms.CACHE_PATH = os.path.join(tmp, "cache.json")
    _use_entries_cache(tmp)
    try:
        _seed_entries_cache()
        entry = {
            "url": "https://mvsep.com/quality_checker/entry/10412",
            "fetched_at": "2026-09-08T12:00:00+00:00",
            "stems": ["instrum", "vocals"],
            "metrics": {"instrum": {"sdr": 14.63}, "vocals": {"sdr": 8.32}},
        }
        key = "bandit_30_zfturbo.ckpt"
        ms._save_cache({key: entry})
        store = ms.ScoresStore()
        seen = []
        store.scores_ready.connect(lambda fn: seen.append(fn))
        store.start()
        store.request(key)
        app = QApplication.instance()
        for _ in range(40):
            if app is not None:
                app.processEvents()
            store._poll()
            if seen:
                break
            import time
            time.sleep(0.05)
        check("cached request queues scores_ready", seen == [key])
    finally:
        ms.CACHE_PATH = old
        ms.ENTRIES_CACHE_PATH = old_entries
        ms._ENTRIES = None


def test_sort_combo_options():
    """Sort dropdown: metric options are lowercase (less width next to the
    search field) and an older save storing the uppercase label still
    restores via the case-insensitive findText fallback."""
    app = QApplication.instance() or QApplication([])
    theme_manager.init_app(app)
    sc = _SortCombo()
    sc.addItem("Name")
    for m in ms.METRICS:
        sc.addItem(ms.METRIC_LABELS[m].lower())
    check("dropdown has Name + 8 metric options", sc.combo.count() == 9)
    check("options are lowercase", all(
        sc.combo.itemText(i) == sc.combo.itemText(i).lower()
        for i in range(1, sc.combo.count())))
    check("popup widens to its longest item",
          getattr(sc.combo, "fit_popup_to_items", False))
    # an old settings save (uppercase label) still resolves
    idx = sc.findText("BLEEDLESS")
    check("case-insensitive restore finds uppercase save", idx == sc.findText("bleedless"))
    check("name restores from uppercase save", sc.findText("NAME") == sc.findText("name"))


def main():
    test_parsing()
    test_lines_and_keys()
    test_sheet_entries()
    test_cache()
    test_store_no_network()
    test_library_row_and_sort()
    test_score_lookup_keys_ckpt_suffix()
    test_score_lookup_keys_friendly()
    test_l1_freq_row_height()
    test_attach_cached_scores_without_signal()
    test_sort_combo_options()
    test_worker_refetches_when_url_changes()
    test_request_queues_cached_refresh()
    print(f"{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()