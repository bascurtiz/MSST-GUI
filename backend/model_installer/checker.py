"""backend/model_installer/checker.py
Detects which required models are already installed.
"""
import os
from typing import List, Tuple

from backend.paths import APP_DIR
from backend.model_installer.registry import REQUIRED_MODELS, IterativeModel


def _find_ckpt(model: IterativeModel) -> str:
    sub = os.path.join(APP_DIR, "models", model.subfolder) if model.subfolder else os.path.join(APP_DIR, "models")
    return os.path.join(sub, model.ckpt_filename)


def _find_yaml(model: IterativeModel) -> str:
    return os.path.join(APP_DIR, "configs", model.yaml_filename)


def _is_registered_in_app(model: IterativeModel, registered_models: list) -> bool:
    for rm in registered_models:
        if rm.get("name") == model.ckpt_filename:
            return True
    return False


def check_model(model: IterativeModel, registered_models: list) -> Tuple[bool, str, str]:
    ckpt_path = _find_ckpt(model)
    yaml_path = _find_yaml(model)

    ckpt_exists = os.path.isfile(ckpt_path)
    yaml_exists = os.path.isfile(yaml_path)
    is_registered = _is_registered_in_app(model, registered_models)

    if ckpt_exists and yaml_exists and is_registered:
        return True, "Installed", ckpt_path

    status_parts = []
    if not ckpt_exists:
        status_parts.append("ckpt missing")
    if not yaml_exists:
        status_parts.append("yaml missing")
    if not is_registered:
        status_parts.append("not registered")

    status = ", ".join(status_parts)
    return False, status, ckpt_path


def check_models(registered_models: list) -> List[Tuple[IterativeModel, bool, str, str]]:
    results = []
    for model in REQUIRED_MODELS:
        installed, status, path = check_model(model, registered_models)
        results.append((model, installed, status, path))
    return results
