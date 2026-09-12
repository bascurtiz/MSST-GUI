"""
backend/sdr_datasets.py
-----------------------
mvsep quality-checker validation dataset download URLs and helpers.
"""
from __future__ import annotations

import os
import threading
import zipfile
from typing import Callable, Optional, Sequence

from PySide6.QtCore import QObject, Signal

# Core per-kit stems scored on mvsep's Drums Separation (5 stems) benchmark.
DRUMS_QC_CORE_SUFFIXES = frozenset({"kick", "snare", "toms", "hh", "cymbals"})

# DrumSep models whose output stems align with the Drums QC naming rules.
DRUMSEP_QC_RECOMMENDED_MODELS = (
    "mdx23c_drumsep_5stem_aufr33_jarredou",
    "mdx23c_drumsep_6stem_aufr33_jarredou",
    "demucs4_drumsep_4stem_inagoy",
)

# Display names must match SDR_DATASETS entries in ui/pages/inference_page.py.
SDR_DATASET_URLS: dict[str, str] = {
    "Multisong": "https://mvsep.com/storage/public/multisong_dataset.zip",
    "Synthetic": "https://mvsep.com/storage/public/synth_dataset.zip",
    "Guitar": "https://mvsep.com/storage/public/guitar_validation.zip",
    "Piano": "https://mvsep.com/storage/public/piano_mixtures.zip",
    "Medley Vox": "https://mvsep.com/storage/public/medley_vox_mixtures.zip",
    "Strings": "https://mvsep.com/storage/public/strings_mixtures.zip",
    "Wind": "https://mvsep.com/storage/public/wind_mixtures.zip",
    "DNR v3": "https://mvsep.com/storage/public/dnr_v3_mixtures.zip",
    "Super Resolution": (
        "https://mvsep.com/storage/public/super_resolution_check_mixtures.zip"
    ),
    "Lead/Back Vocals": (
        "https://mvsep.com/storage/public/lead_back_vocals_mixtures.zip"
    ),
    "Drums": "https://mvsep.com/storage/public/drumsep5_mixtures.zip",
    "Male/Female Vocals": (
        "https://mvsep.com/storage/public/male_female_mixtures.zip"
    ),
    "Phantom Center": (
        "https://mvsep.com/storage/public/phantom_center_mixtures.zip"
    ),
    "Synth Vocals 2026": (
        "https://mvsep.com/storage/public/synth_dataset_2026.zip"
    ),
    "MUSDB18": "https://mvsep.com/storage/public/musdb18_mixtures.zip",
}


def dataset_download_url(display_name: str) -> Optional[str]:
    """Return the mvsep zip URL for a quality-checker dataset name."""
    return SDR_DATASET_URLS.get(display_name)


def drums_qc_predicted_suffixes(
    output_stems: Sequence[str],
    stem_map: dict,
    trained_instruments: Sequence[str],
) -> set[str]:
    """Output suffixes the model will write for Drums QC given its stems."""
    from backend.audio_names import qc_suffix_map_for_model, stem_suffix_for

    adj = qc_suffix_map_for_model(stem_map or {}, list(trained_instruments or []))
    return {stem_suffix_for(s, adj) for s in (output_stems or [])}


def drums_qc_incompatible_message(
    output_stems: Sequence[str],
    stem_map: dict,
    trained_instruments: Sequence[str],
) -> Optional[str]:
    """Return a user-facing warning when the model cannot produce drum-kit stems.

    Full-mix drum extractors (drums/other) are the common mismatch: they
    separate drums from the rest of a song, while the Drums QC benchmark
    expects intra-drum kit separation (kick, snare, toms, hh, cymbals).
    """
    if not output_stems or not stem_map:
        return None
    predicted = drums_qc_predicted_suffixes(
        output_stems, stem_map, trained_instruments)
    if predicted & DRUMS_QC_CORE_SUFFIXES:
        return None
    if "drums" in predicted or predicted <= {"drums", "other"}:
        models = "\n".join(f"  • {m}" for m in DRUMSEP_QC_RECOMMENDED_MODELS)
        return (
            "The selected model outputs drums/other (full-mix drum extraction), "
            "not per-kit drum stems.\n\n"
            "The Drums QC benchmark expects kick, snare, toms, hh, and/or "
            "cymbals.\n\n"
            "Use a DrumSep model instead, for example:\n"
            f"{models}\n\n"
            "Continue anyway?"
        )
    return None


def dataset_folder_name(url: str) -> str:
    """Subfolder name for a dataset zip (e.g. strings_mixtures.zip -> strings_mixtures)."""
    return os.path.splitext(os.path.basename(url))[0]


def dataset_extract_path(parent_dir: str, url: str) -> str:
    """Full path where a dataset zip is extracted: parent_dir/<zip_stem>/."""
    return os.path.join(parent_dir, dataset_folder_name(url))


def download_and_extract_dataset(
    url: str,
    dest_dir: str,
    *,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[bool, str]:
    """Download a validation zip into *dest_dir* and extract to a subfolder.

    The user picks a parent folder; contents land in
    ``dest_dir/<zip_basename_without_ext>/`` (e.g. ``D:\\datasets\\strings_mixtures``).

    Returns (ok, message). On success *message* is the extract folder path.
    """
    from backend.download_utils import _make_session, parallel_download

    os.makedirs(dest_dir, exist_ok=True)
    extract_dir = dataset_extract_path(dest_dir, url)
    os.makedirs(extract_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, os.path.basename(url))

    if status_callback:
        status_callback(f"Downloading {os.path.basename(url)}…")

    session = _make_session()
    try:
        ok, msg = parallel_download(
            url,
            zip_path,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
            session=session,
        )
    finally:
        session.close()

    if not ok:
        return False, msg

    if status_callback:
        status_callback("Extracting archive…")

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        return False, f"Downloaded file is not a valid zip: {exc}"
    except OSError as exc:
        return False, f"Could not extract archive: {exc}"

    return True, extract_dir


_ACTIVE_DATASET_WORKERS: set["SdrDatasetDownloadWorker"] = set()
_DATASET_WORKERS_LOCK = threading.Lock()


class SdrDatasetDownloadWorker(QObject):
    """Background worker: download + extract one mvsep validation dataset."""

    progress = Signal(str, float, float)
    status = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, url: str, dest_dir: str):
        super().__init__()
        self._url = url
        self._dest_dir = dest_dir
        self._cancelled = False
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self.is_running():
            return
        with _DATASET_WORKERS_LOCK:
            _ACTIVE_DATASET_WORKERS.add(self)
        self._thread = threading.Thread(
            target=self._run_wrapped, name="sdr-dataset-download", daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled = True

    def _run_wrapped(self):
        try:
            self._run()
        finally:
            with _DATASET_WORKERS_LOCK:
                _ACTIVE_DATASET_WORKERS.discard(self)

    def _run(self):
        def _progress(_name, downloaded, total):
            self.progress.emit(os.path.basename(self._url), float(downloaded),
                               float(total))

        ok, msg = download_and_extract_dataset(
            self._url,
            self._dest_dir,
            progress_callback=_progress,
            status_callback=self.status.emit,
            should_cancel=lambda: self._cancelled,
        )
        self.finished.emit(ok, msg)
