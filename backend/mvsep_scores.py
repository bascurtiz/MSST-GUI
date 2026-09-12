"""
backend/mvsep_scores.py
------------------------
Scores for models from the mvsep Quality Checker leaderboards.

A published Google Sheet maps checkpoint filenames to their mvsep entry URLs
(see ``SHEET_EXPORT_URL``). The sheet can be edited in the cloud so new models
appear without an app update. Fetches are cached next to the app
(``mvsep_quality_checker_cache.json``); a stale or missing cache is refreshed
from the sheet on startup (network required for the first run).

Each entry page lists per-stem metrics — SDR, SI-SDR, L1-FREQ, LOG-WMSE,
AURA-STFT, AURA-MRSTFT, BLEEDLESS, FULLNESS — in the form::

    Metric sdr for vocals: 5.0231
    Metric si_sdr for vocals: 3.3197
    ...

This module parses those pages, caches the result to a JSON file next to the
app (mvsep_scores_cache.json, refreshed when older than STALE_DAYS), and
exposes a GUI-side `ScoresStore` singleton that fetches missing entries in a
background thread and emits `scores_ready(filename)` as each one lands, so the
Model Library and Model Manager can light up per-model once data arrives.

No score entry (no URL in the sheet) simply means the model is not listed yet —
the UI shows nothing extra for it. Denoise and dereverb/deecho models have no
public validation set at all; the library and manager show
``NO_VALIDATION_SET_LABEL`` in the metric slot instead of a blank SDR line.
"""
import csv
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

from PySide6.QtCore import QObject, QTimer, Signal

from backend.paths import APP_DIR, REPO_ROOT

# Ordered metric list — also the sort-dropdown order in the Model Library and
# Model Manager. `name` is not a real metric; it is handled by the callers.
METRICS = ["sdr", "si_sdr", "l1_freq", "log_wmse",
           "aura_stft", "aura_mrstft", "bleedless", "fullness"]

METRIC_LABELS = {
    "sdr": "SDR",
    "si_sdr": "SI-SDR",
    "l1_freq": "L1-FREQ",
    "log_wmse": "LOG-WMSE",
    "aura_stft": "AURA-STFT",
    "aura_mrstft": "AURA-MRSTFT",
    "bleedless": "BLEEDLESS",
    "fullness": "FULLNESS",
}

# Denoise / dereverb-deecho models have no mvsep quality-checker validation
# set, so the UI shows this note in the SDR / SI-SDR slot instead of nothing.
NO_VALIDATION_STEM_TYPES = frozenset({"denoise", "dereverb / deecho"})
NO_VALIDATION_SET_LABEL = "No validation set available"


def lacks_validation_set(stem_type: str) -> bool:
    """True for denoise and dereverb/deecho — no public SDR validation set."""
    return (stem_type or "").strip().lower() in NO_VALIDATION_STEM_TYPES

# Published Google Sheet (Filename, URL, Architecture columns).
SHEET_EXPORT_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1nUZgS26ZD2p7W-815xwAZ2LKAurixq9rUCnnlrVQNek/export?format=csv&gid=0"
)
ENTRIES_CACHE_PATH = os.path.join(APP_DIR, "mvsep_quality_checker_cache.json")
CACHE_PATH = os.path.join(APP_DIR, "mvsep_scores_cache.json")
STALE_DAYS = 7
# Bump when the HTML parser changes so existing disk/seed entries refetch.
# v2: signed metric values (e.g. piano SI-SDR -1.75) were previously dropped.
PARSER_VERSION = 2
ENTRIES_STALE_DAYS = 1  # re-fetch the sheet URL list daily
_FETCH_DELAY = 0.15  # politeness gap between page fetches

_ENTRIES = None  # lazy: filename(lower) -> url
_ENTRIES_LOCK = threading.Lock()


def _parse_entries_csv(text: str) -> dict:
    """Parse Filename,URL[,…] rows into filename(lower) -> mvsep URL."""
    entries = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 2 and row[0] and row[1].strip().startswith("http"):
            entries[row[0].strip().lower()] = row[1].strip()
    return entries


def _load_entries_cache() -> dict | None:
    """Return {"fetched_at": iso, "entries": {…}} or None."""
    try:
        with open(ENTRIES_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        entries = data.get("entries")
        if not isinstance(entries, dict) or not entries:
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _save_entries_cache(entries: dict) -> None:
    try:
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "entries": entries,
        }
        tmp = ENTRIES_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, ENTRIES_CACHE_PATH)
    except OSError:
        pass


def _entries_cache_stale(cache: dict) -> bool:
    return _is_stale({"fetched_at": cache.get("fetched_at", "")}, ENTRIES_STALE_DAYS)


def _fetch_remote_entries() -> dict:
    """Download the published sheet CSV. Raises on failure."""
    import requests
    resp = requests.get(SHEET_EXPORT_URL, timeout=20)
    resp.raise_for_status()
    entries = _parse_entries_csv(resp.text)
    if not entries:
        raise ValueError("no entries parsed from Google Sheet")
    return entries


def _resolve_initial_entries() -> dict:
    """Load the last cached sheet snapshot, or {} before the first fetch."""
    cached = _load_entries_cache()
    if cached:
        return dict(cached["entries"])
    return {}


def load_entries() -> dict:
    global _ENTRIES
    with _ENTRIES_LOCK:
        if _ENTRIES is None:
            _ENTRIES = _resolve_initial_entries()
        return _ENTRIES


def _set_entries(entries: dict) -> dict:
    """Replace the in-memory entry map; return the previous map."""
    global _ENTRIES
    with _ENTRIES_LOCK:
        old = dict(_ENTRIES) if _ENTRIES is not None else {}
        _ENTRIES = dict(entries)
        return old


# "Metric sdr for vocals: 5.0231" / "Metric si_sdr for piano: -1.74664"
_METRIC_RE = re.compile(
    r"Metric\s+([a-z0-9_]+)\s+for\s+([A-Za-z0-9 _\-]+?):\s*([-+]?\d+(?:\.\d+)?)\s*$"
)


def parse_entry_html(html: str) -> dict:
    """Extract per-stem metrics from an mvsep quality-checker entry page.

    Returns {"stems": [...], "metrics": {stem: {metric: float}}} with stems in
    the order they appear on the page. Only stems that actually have metrics
    are included (pages print '---' in the headline table for absent stems,
    but the Metrics section only ever lists real values).
    """
    stems: list[str] = []
    metrics: dict[str, dict[str, float]] = {}
    # The page renders each metric on its own line with <br /> tags rather
    # than newlines — normalize both so the per-line regex sees one metric
    # per line.
    html = html.replace("<br />", "\n").replace("<br>", "\n")
    for line in html.splitlines():
        m = _METRIC_RE.search(line.strip())
        if not m:
            continue
        metric, stem, val = m.group(1), m.group(2), float(m.group(3))
        stem = stem.strip()
        if stem not in metrics:
            metrics[stem] = {}
            stems.append(stem)
        metrics[stem][metric] = val
    return {"stems": stems, "metrics": metrics}


def fetch_entry(url: str) -> dict:
    """Fetch an entry page and parse it. Raises on network/parse failure."""
    import requests
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    parsed = parse_entry_html(resp.text)
    if not parsed["stems"]:
        raise ValueError(f"no metrics found at {url}")
    return parsed


# ── Disk cache ────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    """Load the writable cache, falling back to the bundled seed file.

    Fresh installs (especially the frozen build) start with no
    ``mvsep_scores_cache.json`` next to the executable. The entry list still
    provides mvsep links, but metrics need cached score payloads — so copy
    the shipped seed into the writable app dir on first load.
    """
    if os.path.isfile(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    seed = os.path.join(REPO_ROOT, "resources", "mvsep_scores_cache.json")
    try:
        with open(seed, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            _save_cache(data)
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass


def _is_stale(entry: dict, stale_days: int = STALE_DAYS) -> bool:
    if entry.get("parser") != PARSER_VERSION:
        return True
    try:
        fd = datetime.fromisoformat(entry.get("fetched_at", "")).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fd).total_seconds() > stale_days * 86400
    except (ValueError, TypeError):
        return True


# ── Display helpers ───────────────────────────────────────────────────────

def mean_metric(scores: dict, metric: str):
    """Average of one metric across the model's stems — the sort key. Returns
    None when the model has no scores or the metric is missing entirely."""
    if not scores:
        return None
    vals = [m[metric] for m in scores.get("metrics", {}).values()
            if isinstance(m.get(metric), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def metric_line(scores: dict, metric: str) -> str:
    """'SDR effects: 10.74 | music: 8.28 | sfx: 9.45' style line. Stems keep
    the page's own order/names so it is obvious which value belongs to which
    stem, for 2-, 3- and 4-stem models alike."""
    if not scores:
        return ""
    label = METRIC_LABELS.get(metric, metric.upper())
    parts = []
    for stem in scores.get("stems", []):
        val = scores.get("metrics", {}).get(stem, {}).get(metric)
        if isinstance(val, (int, float)):
            parts.append(f"{stem}: {val:.2f}")
    if not parts:
        return ""
    return f"{label} " + " | ".join(parts)


def sdr_line(scores: dict) -> str:
    return metric_line(scores, "sdr")


# ── GUI-side store ────────────────────────────────────────────────────────

class ScoresStore(QObject):
    """Singleton that owns the background mvsep fetch loop and the disk cache.

    Pages call `request(filename)` for every model they display and connect to
    `scores_ready(filename)` to refresh rows when data arrives. Needed entries
    (models the user actually has) are fetched before the rest of the sheet,
    so the Model Library lights up first. Cached-fresh entries are never
    re-fetched.

    `entries_updated` fires when the Google Sheet URL list changes (new rows
    or updated links) so pages can re-request scores for newly listed models.

    Call ``refresh_all()`` (or use the refresh icon in the Model Library /
    Model Manager) to force a sheet re-download and re-fetch mvsep scores for
    displayed models — needed after a leaderboard URL changes or a model is
    re-uploaded under a new entry.
    """
    scores_ready = Signal(str)  # checkpoint filename (lowercase)
    entries_updated = Signal()
    refresh_started = Signal()
    refresh_finished = Signal(bool)  # success

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache = _load_cache()
        self._needed = set()
        self._results = []          # (filename, scores) produced by the thread
        self._refresh_pending = set()  # cached keys already queued this session
        self._started = False
        self._refresh_running = False
        self._worker_lock = threading.Lock()
        self._thread = None
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll)

    # ── Public API ──

    def refresh_from_disk(self) -> None:
        """Merge any on-disk cache entries into memory (e.g. Model Manager
        fetched scores while Inference was on screen)."""
        disk = _load_cache()
        if disk:
            self._cache.update(disk)

    def get(self, filename: str):
        """Cached scores dict (or None) for a checkpoint filename."""
        raw = self._cache.get((filename or "").lower())
        if not raw:
            return None
        stems = raw.get("stems")
        metrics = raw.get("metrics")
        if not stems or not metrics:
            return None
        return {"stems": stems, "metrics": metrics}

    def entry_url(self, filename: str) -> str:
        return load_entries().get((filename or "").lower(), "")

    def _queue_cached_refresh(self, fn: str) -> None:
        """Notify UI listeners that a fresh cache entry is already available.

        The background worker skips network fetches for non-stale cache hits
        and therefore never emits ``scores_ready`` for them. Rows that were
        built before the cache warmed up depend on this refresh signal.
        """
        cached = self._cache.get(fn)
        if not cached or not cached.get("stems") or _is_stale(cached):
            return
        if fn in self._refresh_pending:
            return
        self._refresh_pending.add(fn)
        self._results.append((fn, {
            "stems": cached["stems"],
            "metrics": cached["metrics"],
        }))

    def request(self, filename: str) -> None:
        """Mark a checkpoint filename as wanted by the UI. Idempotent; safe to
        call for every displayed model."""
        fn = (filename or "").lower()
        if fn and fn in load_entries():
            self._needed.add(fn)
            self._queue_cached_refresh(fn)

    def refresh_all(self) -> None:
        """Force a Google Sheet download and re-fetch mvsep scores for models
        the UI has requested. Safe to call from the main thread."""
        if self._refresh_running:
            return
        if not self._started:
            self.start()
        threading.Thread(target=self._manual_refresh, daemon=True).start()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._timer.start()
        self._thread = threading.Thread(target=self._bootstrap, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._timer.stop()

    # ── Workers ──

    def _bootstrap(self) -> None:
        """Refresh the sheet URL list when needed, then crawl mvsep scores."""
        if self._sync_entries_from_sheet():
            self.entries_updated.emit()
        with self._worker_lock:
            self._worker_pass(load_entries())

    def _sync_entries_from_sheet(self, force: bool = False) -> bool:
        """Fetch the published Google Sheet when the local snapshot is stale.

        Returns True when a new snapshot was downloaded and applied.
        """
        cached = _load_entries_cache()
        if not force and cached and not _entries_cache_stale(cached):
            return False
        try:
            entries = _fetch_remote_entries()
        except Exception:
            return False
        _save_entries_cache(entries)
        _set_entries(entries)
        return True

    def _manual_refresh(self) -> None:
        self._refresh_running = True
        self.refresh_started.emit()
        success = False
        try:
            if not self._sync_entries_from_sheet(force=True):
                return
            entries = load_entries()
            if not entries:
                return
            force_scores = set(self._needed)
            for fn, url in entries.items():
                cached = self._cache.get(fn)
                if not cached or cached.get("url") != url:
                    self._cache.pop(fn, None)
                    self._refresh_pending.discard(fn)
                    force_scores.add(fn)
            _save_cache(self._cache)
            with self._worker_lock:
                self._worker_pass(entries, force=force_scores)
            self.entries_updated.emit()
            success = True
        except Exception:
            pass
        self._refresh_running = False
        self.refresh_finished.emit(success)

    def _worker_pass(self, entries: dict, force: set | None = None) -> None:
        # Needed (installed/displayed) models first, then the rest.
        order = sorted(entries, key=lambda fn: (fn not in self._needed, fn))
        for fn in order:
            url = entries[fn]
            cached = self._cache.get(fn)
            if force is None or fn not in force:
                if (cached and not _is_stale(cached)
                        and cached.get("url") == url):
                    continue
            try:
                parsed = fetch_entry(url)
            except Exception:
                continue  # leave it for a later run
            entry = {
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "parser": PARSER_VERSION,
                "stems": parsed["stems"],
                "metrics": parsed["metrics"],
            }
            self._cache[fn] = entry
            self._results.append((fn, parsed))
            _save_cache(self._cache)
            time.sleep(_FETCH_DELAY)
        _save_cache(self._cache)

    def _poll(self) -> None:
        while self._results:
            fn, parsed = self._results.pop(0)
            self.scores_ready.emit(fn)


# Module-level singleton: created lazily on first use (needs a QApplication
# for the QTimer), so importing this module in tests is side-effect free.
scores_store = None


def get_scores_store() -> ScoresStore:
    global scores_store
    if scores_store is None:
        scores_store = ScoresStore()
    return scores_store
