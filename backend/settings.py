"""
backend/settings.py
--------------------
Saves and restores user settings to a JSON file next to the GUI package.
"""
import json
import os

from backend.paths import SETTINGS_PATH

_SETTINGS_FILE = SETTINGS_PATH


def load() -> dict:
    try:
        if not os.path.exists(_SETTINGS_FILE):
            return {}
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, PermissionError, OSError) as e:
        print(f"[settings] Failed to load settings: {e}")
        return {}


def save(data: dict) -> None:
    try:
        tmp = _SETTINGS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _SETTINGS_FILE)
    except (PermissionError, OSError) as e:
        print(f"[settings] Failed to save settings: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def load_ckpt_settings() -> dict:
    data = load()
    return data.get("ckpt_settings", {})


def save_ckpt_settings(ckpt_name: str, settings: dict) -> None:
    data = load()
    if "ckpt_settings" not in data:
        data["ckpt_settings"] = {}
    data["ckpt_settings"][ckpt_name] = settings
    save(data)


def load_iterative_settings() -> dict:
    data = load()
    return data.get("iterative_ensemble", {})


def save_iterative_settings(settings: dict) -> None:
    data = load()
    data["iterative_ensemble"] = settings
    save(data)


def load_pretrained_filters() -> dict:
    """Last-used pre-trained dialog filter choices ({"target": ..., "arch": ...})."""
    data = load()
    return data.get("pretrained_filters", {})


def save_pretrained_filters(filters: dict) -> None:
    data = load()
    data["pretrained_filters"] = filters
    save(data)



