"""
backend/model_manager.py
------------------------
Fetches, parses, and installs models from the mvsepless_resources HuggingFace repo.
"""

from __future__ import annotations
import os
import json
import shutil
import requests
from dataclasses import dataclass, field
from typing import Callable, Optional
from backend.paths import REPO_ROOT, APP_DIR
from backend import settings as settings_store
from backend.download_utils import parallel_download

MODELS_JSON_URL = (
    "https://huggingface.co/noblebarkrr/mvsepless_resources/raw/main/models.json"
)
REPO_API_URL = (
    "https://huggingface.co/api/models/noblebarkrr/mvsepless_resources"
)
TREE_INFO_URL = (
    "https://huggingface.co/api/models/noblebarkrr/mvsepless_resources/tree/main?expand=true"
)
BASE_DL_URL = (
    "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main"
)

MODEL_TYPE_TO_ARCH = {
    "bs_roformer": "BS Roformer Architecture",
    "mel_band_roformer": "Melband Roformer Architecture",
    "scnet": "SCNet Architecture",
    "mdx23c": "MDX23c Architecture",
    "vr": "VR Architecture",
    "htdemucs": "Demucs Architecture",
    "bandit": "Bandit Architecture",
    "bandit_v2": "Bandit Architecture",
    "mdxnet": "MDX-Net Architecture",
    "scnet_masked": "SCNet Architecture",
    "scnet_tran": "SCNet Architecture",
    "medley_vox": "Medley Vox Architecture",
    "apollo": "Apollo Architecture",
    "segm_models": "VitLarge23 Architecture",
    "torchseg": "TorchSeg Architecture",
    "swin_upernet": "Swin Upernet Architecture",
    "bs_mamba2": "BSMamba2 Architecture",
    "conformer": "Conformer Architecture",
    "dtt_net": "DTTNet Architecture",
}

STEM_MAP = {
    # Core families
    "Вокал": "vocals",
    "Инструментал": "instrumental",
    "Инструментал и вокал": "dual target (instrumental & vocals)",
    "Караоке": "karaoke",
    "4 стема": "multi stems",
    "Гитара": "guitar",
    "Реверб": "dereverb / deecho",
    "Шум": "denoise",
    "Дуэт": "vocals",
    # Instruments (RU → app type vocabulary)
    "Ударные": "drums",                  # drums
    "DrumSep": "drums",
    "Бас": "bass",
    "Басс": "bass",                      # zoo spelling variant
    "Перкуссия": "percussion",
    "Клавишные": "keys",                 # keys
    "Синтезатор": "keys",                # synthesizer → keyboard family
    "Струнные": "strings",               # strings
    "Смычковые струнные": "strings",     # bowed strings
    "Щипковые струнные": "strings",      # plucked strings
    "Оркестр": "strings",                # orchestra (string-led ensemble)
    "Духовые": "wind",                   # wind instruments
    "Деревянные духовые": "wind",        # woodwinds
    "Медные духовые": "wind",            # brass
    "Саксофон": "wind",                  # saxophone
    "Гармоники": "wind",                 # free-reed aerophones (accordion/harmonica)
    # Cleanup / processing
    "Реверб и эхо": "dereverb / deecho",  # reverb + echo
    "Эхо": "dereverb / deecho",          # echo
    "Дыхание": "denoise",                # de-breath
    "Скретч": "denoise",                 # scratch removal
    "Фантомный центр": "phantom centre",
    # Multi-stem splits
    "6 стемов": "multi stems",
    "Все стемы": "multi stems",
    "Объёмный звук": "multi stems",       # surround channel split
    # Vocal family
    "Хор": "vocals",                     # choir / voice parts
    "Разделение голосов": "vocals",       # singing-voice separation
    "Мужской/Женский вокал": "vocals",    # male/female vocal
    # Instrumental extractors ("other"-target models)
    "Прочее": "instrumental",             # misc → other/instrumental
    # Movie-audio family (bgm / musicless / sfx, mixed stems)
    "Кинематограф": "multi stems",
    # Sound design
    "Звуковые эффекты": "effects",        # sound effects (ambience/explosions/…)
    "Звуки толпы": "crowd",               # crowd sounds
}


@dataclass
class ModelInfo:
    key: str
    full_name: str
    arch: str
    stem_type: str
    category: str
    model_type: str
    stems: list[str]
    target_instrument: Optional[str]
    checkpoint_url: str
    config_url: str
    backend_script_url: str = ""
    file_size: int = 0


def fetch_model_index() -> list[ModelInfo]:
    """Fetch and parse models.json from HuggingFace."""
    resp = requests.get(MODELS_JSON_URL, timeout=60)
    resp.raise_for_status()
    raw = resp.json()
    models: list[ModelInfo] = []
    for key, entry in raw.items():
        mt = entry.get("model_type", "")
        arch = MODEL_TYPE_TO_ARCH.get(mt, "BS Roformer Architecture")
        raw_cat = entry.get("category", "")
        stem_type = STEM_MAP.get(raw_cat, "vocals")
        ckpt_url = entry.get("checkpoint_url", "")
        cfg_url = entry.get("config_url", "")
        models.append(ModelInfo(
            key=key,
            full_name=entry.get("full_name", key),
            arch=arch,
            stem_type=stem_type,
            category=raw_cat,
            model_type=mt,
            stems=entry.get("stems", []),
            target_instrument=entry.get("target_instrument"),
            checkpoint_url=ckpt_url,
            config_url=cfg_url,
            backend_script_url=entry.get("backend_url", ""),
            file_size=entry.get("file_size", 0),
        ))
    return models


def fetch_repo_meta() -> str:
    """Fetch repo metadata, return lastModified ISO string."""
    resp = requests.get(REPO_API_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()["lastModified"]


def fetch_tree_info() -> dict[str, str]:
    """Return {directory_name: lastCommit_date} for each subdirectory."""
    resp = requests.get(TREE_INFO_URL, timeout=30)
    resp.raise_for_status()
    result: dict[str, str] = {}
    for entry in resp.json():
        if entry.get("type") == "directory":
            lc = entry.get("lastCommit")
            if lc:
                result[entry["path"]] = lc["date"]
    return result


def fetch_folder_tree(folder_key: str) -> tuple[dict[str, str], dict[str, int]]:
    """Return ({filename: lastCommit_date}, {filename: size_in_bytes}) for files in the given subdirectory."""
    base = TREE_INFO_URL.removesuffix("?expand=true")
    dates: dict[str, str] = {}
    sizes: dict[str, int] = {}
    url: str | None = f"{base}/{folder_key}?expand=true"
    while url:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        for entry in resp.json():
            if entry.get("type") == "file" and "lastCommit" in entry:
                fname = entry["path"].split("/")[-1]
                dates[fname] = entry["lastCommit"]["date"]
                if "size" in entry:
                    sizes[fname] = entry["size"]
        url = None
        link = resp.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part:
                start = part.index("<") + 1
                end = part.index(">")
                url = part[start:end]
                break
    return dates, sizes


def group_by_arch(models: list[ModelInfo]) -> dict[str, list[ModelInfo]]:
    """Group models by their architecture label."""
    groups: dict[str, list[ModelInfo]] = {}
    for m in models:
        groups.setdefault(m.arch, []).append(m)
    return groups


def _arch_folder(arch: str) -> str:
    from backend.yaml_analyzer import classify_model_type
    mapping = {
        "BS Roformer Architecture": "bs_roformer",
        "Melband Roformer Architecture": "melband_roformer",
        "SCNet Architecture": "scnet",
        "MDX Architecture": "mdx",  # legacy label from before the MDX split
        "MDX23c Architecture": "mdx",
        "MDX-Net Architecture": "mdxnet",
        "VR Architecture": "vr",
        "Medley Vox Architecture": "medley_vox",
        "Demucs Architecture": "demucs",
        "Apollo Architecture": "apollo",
        "Bandit Architecture": "bandit",
        "VitLarge23 Architecture": "segm_models",
        "TorchSeg Architecture": "torchseg",
        "Swin Upernet Architecture": "swin_upernet",
        "BSMamba2 Architecture": "bs_mamba2",
        "Conformer Architecture": "conformer",
        "DTTNet Architecture": "dtt_net",
    }
    return mapping.get(arch, "bs_roformer")


def _ckpt_name_from_url(url: str) -> str:
    return url.split("/")[-1].split("?")[0]


def _yaml_name_from_url(url: str) -> str:
    return url.split("/")[-1].split("?")[0]


def is_installed(info: ModelInfo) -> bool:
    """Check if model already registered in settings."""
    data = settings_store.load()
    models = data.get("registered_models", [])
    ckpt_name = _ckpt_name_from_url(info.checkpoint_url)
    for m in models:
        if os.path.basename(m.get("ckpt", "")).lower() == ckpt_name.lower():
            return True
    return False


def install_model(
    info: ModelInfo,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    speed_callback: Optional[Callable[[float], None]] = None,
) -> tuple[bool, str]:
    """Download checkpoint + config, register in settings.

    Returns (success, message).
    """
    ckpt_name = _ckpt_name_from_url(info.checkpoint_url)
    yaml_name = _yaml_name_from_url(info.config_url)
    arch = info.arch
    arch_dir = _arch_folder(arch)

    ckpt_dest = os.path.join(APP_DIR, "models", arch_dir, ckpt_name)
    yaml_dest = os.path.join(APP_DIR, "configs", yaml_name)
    custom_backend = False
    backend_module = ""

    if info.backend_script_url:
        backend_module = os.path.splitext(ckpt_name)[0]
        custom_backend_dir = os.path.join(REPO_ROOT, "models", "custom", backend_module)
        backend_dest = os.path.join(custom_backend_dir, "bs_roformer.py")
        custom_backend = True

    os.makedirs(os.path.dirname(ckpt_dest), exist_ok=True)
    os.makedirs(os.path.dirname(yaml_dest), exist_ok=True)
    if custom_backend:
        os.makedirs(custom_backend_dir, exist_ok=True)
        # Touch __init__.py
        init_file = os.path.join(custom_backend_dir, "__init__.py")
        if not os.path.isfile(init_file):
            try:
                with open(init_file, "w") as f:
                    f.write("")
            except OSError:
                pass

    if status_callback:
        status_callback("Downloading checkpoint...")

    ok, msg = _download_file(info.checkpoint_url, ckpt_dest,
                              progress_callback, status_callback, cancel_callback,
                              speed_callback)
    if not ok:
        _cleanup(ckpt_dest)
        return False, f"Checkpoint download failed: {msg}"

    if cancel_callback and cancel_callback():
        _cleanup(ckpt_dest)
        return False, "Cancelled"

    if status_callback:
        status_callback("Downloading configuration...")

    ok, msg = _download_file(info.config_url, yaml_dest,
                              progress_callback, status_callback, cancel_callback,
                              speed_callback)
    if not ok:
        _cleanup(ckpt_dest, yaml_dest)
        return False, f"Config download failed: {msg}"

    if cancel_callback and cancel_callback():
        _cleanup(ckpt_dest, yaml_dest)
        return False, "Cancelled"

    if custom_backend:
        if status_callback:
            status_callback("Downloading backend script...")
        ok, msg = _download_file(info.backend_script_url, backend_dest,
                                  progress_callback, status_callback, cancel_callback,
                                  speed_callback)
        if not ok:
            _cleanup(ckpt_dest, yaml_dest, backend_dest)
            return False, f"Backend script download failed: {msg}"

        if cancel_callback and cancel_callback():
            _cleanup(ckpt_dest, yaml_dest, backend_dest)
            return False, "Cancelled"

    # Register in settings
    data = settings_store.load()
    models = data.setdefault("registered_models", [])
    entry = {
        "name": ckpt_name,
        "ckpt": os.path.normpath(ckpt_dest),
        "yaml": os.path.normpath(yaml_dest),
        "arch": arch,
        "type": info.stem_type,
        "backend_module": backend_module,
        "custom_backend_enabled": custom_backend,
    }
    models.append(entry)
    settings_store.save(data)

    if status_callback:
        status_callback("Model installed successfully!")

    return True, "Installation complete"


def _download_file(
    url: str,
    dest: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    speed_callback: Optional[Callable[[float], None]] = None,
) -> tuple[bool, str]:
    """Download a single file with progress reporting.

    Uses the parallel (multi-connection) downloader so large checkpoints
    bypass per-connection CDN throttling; it automatically falls back to a
    single stream for small files or servers without Range support.
    """
    if status_callback:
        status_callback("Downloading...")
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
