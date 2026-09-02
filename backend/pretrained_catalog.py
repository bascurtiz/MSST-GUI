"""
backend/pretrained_catalog.py
-----------------------------
Pre-trained checkpoint catalog for the TRAINING tab, tracking the upstream
ZFTurbo/Music-Source-Separation-Training docs.

Unlike the inference hub (backend/model_manager.py, which reads a curated
models.json), this catalog is *fetched at runtime*: it pulls the two upstream
markdown files and parses their tables into structured PretrainedModel rows.
That keeps the app current when ZFTurbo adds checkpoints, at the cost of
needing a network connection to browse the list.

Two sources, both fetched live:
  * docs/pretrained_models.md            — the main pre-trained table
  * docs/mel_roformer_experiments.md     — mel-band RoFormer experiments

Each row carries the download links for a matching config (.yaml) and
checkpoint (.ckpt / .pt / .th / .bin), which is exactly what train.py wants
for a fine-tune start.

`install_model` reuses the app's parallel downloader so large checkpoints
survive CDN throttling and stalled reads, and supports cancellation and
progress exactly like the inference install path.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.paths import REPO_ROOT, APP_DIR
from backend.download_utils import parallel_download

# Upstream docs, fetched at runtime.
PRETRAINED_DOC_URL = (
    "https://raw.githubusercontent.com/ZFTurbo/"
    "Music-Source-Separation-Training/main/docs/pretrained_models.md"
)
MEL_DOC_URL = (
    "https://raw.githubusercontent.com/ZFTurbo/"
    "Music-Source-Separation-Training/main/docs/mel_roformer_experiments.md"
)

# Where downloaded pre-trained weights + configs land (writable, not the
# bundled read-only configs/). Kept as its own folder so they never collide
# with the inference registered models or the bundled training configs.
PRETRAINED_ROOT = os.path.join(APP_DIR, "models", "pretrained")

# Extension checks for classifying a download link.
_CONFIG_EXTS = (".yaml", ".yml")
_CKPT_EXTS = (".ckpt", ".pth", ".pt", ".th", ".bin", ".chpt", ".safetensors")
_SKIP_EXTS = (".zip.001", ".zip.002", ".zip.003", ".zip")  # multi-part archives


@dataclass
class PretrainedModel:
    """One pre-trained checkpoint + its config, ready to fine-tune from.

    The upstream docs are markdown tables with two link columns (Config /
    Checkpoint), so `config_url` and `checkpoint_url` are resolved together.
    `section` is the doc heading the row came from (e.g. "Vocal models").
    """

    name: str
    section: str
    instruments: str
    metrics: str
    config_url: str
    checkpoint_url: str
    source: str = "pretrained"  # "pretrained" or "mel_roformer"
    arch_hint: str = ""         # a best-effort architecture label for the UI
    _dest_dir: str = field(default="", repr=False)

    @property
    def is_installed(self) -> bool:
        """True once both files exist on disk under the Pretrained folder.

        Uses the computed dest_dir rather than the cached _dest_dir: the
        downloader writes to dest_dir() even when no explicit destination is
        passed, so relying on the cache would report a fresh install as not
        installed."""
        ckpt = self.checkpoint_path
        if not ckpt or not os.path.isfile(ckpt):
            return False
        cfp = self.config_path
        return bool(cfp) and os.path.isfile(cfp)

    @property
    def dest_dir(self) -> str:
        """Folder the files are/will be downloaded to."""
        if self._dest_dir:
            return self._dest_dir
        return os.path.join(PRETRAINED_ROOT, self._safe_name())

    def _safe_name(self) -> str:
        return re.sub(r"[^\w.-]+", "_", self.name)[:80]

    @property
    def checkpoint_name(self) -> str:
        return os.path.basename(self.checkpoint_url.split("?")[0])

    @property
    def config_name(self) -> str:
        return os.path.basename(self.config_url.split("?")[0])

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.dest_dir, self.checkpoint_name)

    @property
    def config_path(self) -> str:
        return os.path.join(self.dest_dir, self.config_name)


# ── URL helpers ──────────────────────────────────────────────────────────────

def _is_config_url(url: str) -> bool:
    path = url.split("?")[0].lower()
    return path.endswith(_CONFIG_EXTS)


def _is_checkpoint_url(url: str) -> bool:
    path = url.split("?")[0].lower()
    if any(path.endswith(ext) for ext in _SKIP_EXTS):
        return False
    return path.endswith(_CKPT_EXTS)


# Single-stem instrument words that appear alone (no " / " separator) in the
# upstream "Instruments" column, so we can still pick them up.
_KNOWN_INST = {
    "vocals", "drums", "bass", "other", "crowd", "dereverb", "denoise",
    "restored", "aspiration", "dry", "similarity", "piano", "guitar", "keys",
}


def _plain_label(raw: str) -> str:
    """Strip markdown link markup from a table cell: "[Model](url)" -> "Model"
    and remove bold/italic markers, leaving readable text."""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"[*_`]", "", s)
    s = re.sub(r"<br\s*/?>", " ", s)
    return s.strip()


def _looks_like_heading(line: str) -> Optional[str]:
    """Return the heading text if a line is a `### ...` (or `## ...`) heading."""
    m = re.match(r"^#{2,3}\s+(.*)$", line.strip())
    return m.group(1).strip() if m else None


def _link(url: str) -> str:
    return url.strip().rstrip(")")


def _first_metric(texts) -> str:
    for t in texts:
        low = t.lower()
        if any(k in low for k in ("sdr", "l1freq", "avg", "dnr", "score", "l1")):
            return t
    return ""


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_config_and_ckpt(cells):
    """Given the raw cells of a table row (each cell still carrying markdown),
    find the config link and the checkpoint link by their URL extension.

    Returns (config_url, checkpoint_url, name, instruments, metrics).
    Cells are the columns of the row; the columns differ between the two
    documents, so we scan for link-URLs rather than fixed positions.
    """
    # Columns differ per table section, so classify by *content* instead of
    # position: a cell whose URL ends in .yaml/.yml is the Config column and
    # one whose URL ends in a weight extension is the Checkpoint column.
    # The remaining readable cells are Model Type / Instruments / Metrics;
    # to avoid a single-stem name like "HTDemucs4 FT Vocals" being confused
    # with the instruments "vocals", the Model Type is always the first
    # readable cell, Instruments the next, Metrics the one with a metric
    # keyword kept as a label.
    config_url = ""
    checkpoint_url = ""
    download_cells = []
    text_cells = []

    for cell in cells:
        links = re.findall(r"\[([^\]]*)\]\(([^)\s]+)\)", cell)
        # A cell is a download column only if one of its links is a config or
        # checkpoint URL. Author tags (like "([viperx](url) edition)") are
        # links too but belong to the Model Type text, so they stay in the
        # readable text cells rather than being treated as a download column.
        is_download_cell = any(
            _is_config_url(_link(u)) or _is_checkpoint_url(_link(u))
            for _l, u in links)
        plain = _plain_label(cell)
        if is_download_cell:
            for _label, url in links:
                url = _link(url)
                if not config_url and _is_config_url(url):
                    config_url = url
                elif not checkpoint_url and _is_checkpoint_url(url):
                    checkpoint_url = url
        elif plain:
            text_cells.append(plain)

    # Reorder: text_cells holds every non-download column in table order
    # (Model Type, Instruments, Metrics), regardless of where the Config /
    # Checkpoint cells sit. Fill name/instruments/metrics in that order.
    name = ""
    instruments = ""
    metrics = ""
    for plain in text_cells:
        low = plain.lower()
        is_metric = any(k in low for k in ("sdr", "l1freq", "avg", "dnr", "score", "l1"))
        if not name:
            name = plain
        elif not metrics and is_metric:
            metrics = plain
        elif not instruments:
            instruments = plain
        else:
            # A further readable cell (e.g. a second author tag) — only use it
            # to flesh out a bare name.
            if not metrics and len(name) < 6:
                name = plain

    # Fall back to the first metric keyword found anywhere in the row.
    if not metrics:
        metrics = _first_metric(text_cells)
    return config_url, checkpoint_url, name, instruments, metrics


def _parse_pretrained_doc(text: str) -> list[PretrainedModel]:
    """Parse docs/pretrained_models.md into PretrainedModel rows."""
    models: list[PretrainedModel] = []
    section = ""
    in_table = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = _looks_like_heading(line)
        if heading:
            section = heading
            in_table = False
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c for c in line.split("|")]
        # strip the empty leading/trailing cells produced by split("|")
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        # header row / separator row: no downloadable links
        if not re.search(r"\[[^\]]*\]\([^)]*\)", line):
            continue
        config_url, ckpt_url, name, instruments, metrics = _parse_config_and_ckpt(cells)
        if not ckpt_url or not config_url:
            continue
        if not name:
            name = f"{section} model"
        model = PretrainedModel(
            name=_plain_label(name),
            section=section,
            instruments=instruments,
            metrics=metrics,
            config_url=config_url,
            checkpoint_url=ckpt_url,
            source="pretrained",
            arch_hint=_guess_arch(config_url + " " + ckpt_url),
        )
        models.append(model)

    return models


def _parse_mel_doc(text: str) -> list[PretrainedModel]:
    """Parse docs/mel_roformer_experiments.md into PretrainedModel rows.

    This table has a single "DL Checkpoint" column holding "Config / Weights"
    links side by side, plus a numeric-average column, so it needs its own
    extraction rather than the pretrained table's column layout.
    """
    models: list[PretrainedModel] = []
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = _looks_like_heading(line)
        if heading:
            section = heading
            continue
        if not line.startswith("|"):
            continue
        if not re.search(r"\[[^\]]*\]\([^)]*\)", line):
            continue
        cells = [c for c in line.split("|")]
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]

        config_url = ""
        ckpt_url = ""
        avg = ""
        # The average SDR is the first numeric-ish cell.
        for cell in cells:
            m = re.search(r"(\d+\.\d+)", cell)
            if m and not avg:
                avg = m.group(1)
                break
        # Collect Config / Weights links anywhere in the row.
        for cell in cells:
            for _label, url in re.findall(r"\[([^\]]*)\]\(([^)\s]+)\)", cell):
                url = _link(url)
                if not config_url and _is_config_url(url):
                    config_url = url
                elif not ckpt_url and _is_checkpoint_url(url):
                    ckpt_url = url
        if not ckpt_url or not config_url:
            continue
        name = f"MelBand Roformer (SDR {avg})" if avg else "MelBand Roformer experiment"
        model = PretrainedModel(
            name=name,
            section=section,
            instruments="bass / drums / vocals / other",
            metrics=f"SDR: {avg}" if avg else "",
            config_url=config_url,
            checkpoint_url=ckpt_url,
            source="mel_roformer",
            arch_hint="Melband Roformer Architecture",
        )
        models.append(model)

    return models


def _guess_arch(blob: str) -> str:
    """Best-effort architecture label from a config/checkpoint URL or name."""
    low = blob.lower()
    if "mdx23c" in low or "mdx" in low:
        return "MDX23c Architecture"
    if "mel_band_roformer" in low or "melband" in low:
        return "Melband Roformer Architecture"
    if "bs_roformer" in low:
        return "BS Roformer Architecture"
    if "scnet" in low:
        return "SCNet Architecture"
    if "htdemucs" in low or "demucs" in low:
        return "Demucs Architecture"
    if "bs_mamba2" in low:
        return "BSMamba2 Architecture"
    if "segm" in low or "vitlarge" in low:
        return "VitLarge23 Architecture"
    if "swin_upernet" in low:
        return "Swin Upernet Architecture"
    if "conformer" in low:
        return "Conformer Architecture"
    if "dttnet" in low or "dtt_net" in low:
        return "DTTNet Architecture"
    if "apollo" in low:
        return "Apollo Architecture"
    if "bandit" in low:
        return "Bandit Architecture"
    return ""


# ── Fetch ────────────────────────────────────────────────────────────────────

def fetch_catalog(
    timeout: tuple = (20, 20),
) -> list[PretrainedModel]:
    """Fetch + parse both upstream docs. Raises on network/parse failure.

    The caller (the dialog) fetches this off the UI thread and surfaces the
    error to the user; the catalog ships with a sensible empty fallback so a
    fetch failure never crashes the TRAINING tab.
    """
    import requests

    pretrained = _fetch_doc(PRETRAINED_DOC_URL, _parse_pretrained_doc, timeout)
    mel = _fetch_doc(MEL_DOC_URL, _parse_mel_doc, timeout)
    # Deduplicate by config+ckpt URL pairs, preserving order (mel first so the
    # experiment rows appear above the curated table).
    seen = set()
    out: list[PretrainedModel] = []
    for m in mel + pretrained:
        key = (m.config_url, m.checkpoint_url)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _fetch_doc(url: str, parser, timeout: tuple) -> list[PretrainedModel]:
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    if not text or "|" not in text:
        return []
    return parser(text)


# ── Install ──────────────────────────────────────────────────────────────────

def install_model(
    model: PretrainedModel,
    dest_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    speed_callback: Optional[Callable[[float], None]] = None,
) -> tuple[bool, str]:
    """Download the config + checkpoint into the pretrained folder.

    Returns (success, message). Both files are required: a pre-trained start
    for train.py is useless without its matching yaml config.
    """
    if dest_dir:
        model._dest_dir = dest_dir
    ckpt_dest = model.checkpoint_path
    yaml_dest = model.config_path
    os.makedirs(model.dest_dir, exist_ok=True)

    if status_callback:
        status_callback("Downloading configuration...")
    ok, msg = _download(model.config_url, yaml_dest, progress_callback,
                        status_callback, cancel_callback, speed_callback)
    if not ok:
        _cleanup(yaml_dest)
        return False, f"Config download failed: {msg}"
    if cancel_callback and cancel_callback():
        _cleanup(yaml_dest)
        return False, "Cancelled"

    if status_callback:
        status_callback("Downloading checkpoint (large file)...")
    ok, msg = _download(model.checkpoint_url, ckpt_dest, progress_callback,
                        status_callback, cancel_callback, speed_callback)
    if not ok:
        _cleanup(yaml_dest, ckpt_dest)
        return False, f"Checkpoint download failed: {msg}"
    if cancel_callback and cancel_callback():
        _cleanup(yaml_dest, ckpt_dest)
        return False, "Cancelled"

    if status_callback:
        status_callback("Installed!")
    return True, "Installation complete"


def _download(url, dest, progress_callback, status_callback, cancel_callback, speed_callback):
    return parallel_download(
        url, dest,
        progress_callback=(
            lambda n, c, t: progress_callback(n, c, t) if progress_callback else None
        ),
        speed_callback=speed_callback,
        should_cancel=cancel_callback,
        timeout=(60, 60),
    )


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass
