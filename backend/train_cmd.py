"""Build the training subprocess command from GUI run options.

Kept free of Qt so tests can assert LoRA / DDP / wandb / freeze flags without
spinning up the TRAINING tab. The GUI still owns YAML I/O and process spawn.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable, Mapping, Optional, Sequence

from backend.paths import TRAIN_ACCELERATE_SCRIPT, TRAIN_DDP_SCRIPT, TRAIN_SCRIPT

LAUNCHERS = ("standard", "ddp", "accelerate")
LORA_MODES = ("off", "peft", "loralib")

DEFAULT_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "merge_weights": False,
    "fan_in_fan_out": False,
    "enable_lora": [True],
}

DEFAULT_RUN_OPTS = {
    "device_ids": [0],
    "force_cpu": False,
    "num_workers": 4,
    "seed": 0,
    "pin_memory": False,
    "pre_valid": False,
    "save_every_epoch": False,
    "each_metrics_in_name": False,
    "load_optimizer": False,
    "load_scheduler": False,
    "load_epoch": False,
    "load_best_metric": False,
    "load_all_metrics": False,
    "load_all_losses": False,
    "safe_mode": False,
    "persistent_workers": False,
    "launcher": "standard",
    "lora_mode": "off",
    "lora_checkpoint": "",
    "freeze_layers": "",
    "wandb_key": "",
    "wandb_offline": False,
}

_SWITCH_FLAGS = {
    "pin_memory": "--pin_memory",
    "pre_valid": "--pre_valid",
    "save_every_epoch": "--save_weights_every_epoch",
    "each_metrics_in_name": "--each_metrics_in_name",
    "load_optimizer": "--load_optimizer",
    "load_scheduler": "--load_scheduler",
    "load_epoch": "--load_epoch",
    "load_best_metric": "--load_best_metric",
    "load_all_metrics": "--load_all_metrics",
    "load_all_losses": "--load_all_losses",
    "safe_mode": "--safe_mode",
    "persistent_workers": "--persistent_workers",
}

_BACKEND_FILENAMES = ("bs_roformer.py", "model.py", "models.py")


def detect_custom_backend(config_path: str) -> str:
    """Folder next to a config that ships an author backend .py, or ''."""
    if not config_path or not os.path.isfile(config_path):
        return ""
    folder = os.path.dirname(config_path)
    for name in _BACKEND_FILENAMES:
        if os.path.isfile(os.path.join(folder, name)):
            return folder
    return ""


def inject_lora_defaults(cfg: dict) -> dict:
    """Ensure a LoRA block exists when the GUI is about to train with LoRA."""
    if not isinstance(cfg, dict):
        return cfg
    existing = cfg.get("lora")
    if not isinstance(existing, dict) or not existing:
        cfg["lora"] = dict(DEFAULT_LORA_CONFIG)
    return cfg


def accelerate_available(python_exe: str | None = None) -> bool:
    """True when the job interpreter can import Hugging Face Accelerate.

    Probed in a subprocess so PyInstaller never traces ``accelerate`` (and
    torch) into the GUI bundle.
    """
    py = python_exe or sys.executable
    try:
        r = subprocess.run(
            [py, "-c", "import accelerate"],
            capture_output=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return r.returncode == 0
    except Exception:
        return False


def resolve_launcher(
    opts: Mapping,
    *,
    has_custom_backend: bool = False,
    accelerate_ok: Optional[bool] = None,
) -> tuple[str, Optional[str]]:
    """Pick a launcher, falling back to standard when a constraint is unmet.

    Returns ``(launcher, warning_or_None)``.
    """
    launcher = str(opts.get("launcher") or "standard").strip().lower()
    if launcher not in LAUNCHERS:
        launcher = "standard"
    force_cpu = bool(opts.get("force_cpu"))
    device_ids = list(opts.get("device_ids") or [0])
    gpu_count = 0 if force_cpu else len(device_ids)
    lora_on = str(opts.get("lora_mode") or "off") != "off"
    freeze_on = bool(str(opts.get("freeze_layers") or "").strip())

    if launcher == "ddp":
        if gpu_count < 2:
            return "standard", (
                "DDP needs at least 2 GPUs — falling back to Standard "
                "(DataParallel / single GPU)."
            )
        return "ddp", None

    if launcher == "accelerate":
        if lora_on or freeze_on or has_custom_backend:
            return "standard", (
                "Accelerate uses the older train_accelerate.py loop "
                "(no LoRA, freeze layers, or custom backends) — "
                "falling back to Standard."
            )
        ok = accelerate_available() if accelerate_ok is None else accelerate_ok
        if not ok:
            return "standard", (
                "accelerate is not installed — falling back to Standard."
            )
        return "accelerate", None

    return "standard", None


def _cli_device_ids(opts: Mapping, launcher: str) -> list[int]:
    """Device ids passed on the command line after CUDA_VISIBLE_DEVICES remap."""
    ids = [int(i) for i in (opts.get("device_ids") or [0])]
    if launcher in ("ddp", "accelerate") and not opts.get("force_cpu"):
        return list(range(len(ids)))
    return ids or [0]


def subprocess_env(opts: Mapping, launcher: str) -> Optional[dict]:
    """Extra env for the job, or None when the parent env is fine."""
    if opts.get("force_cpu"):
        return {"CUDA_VISIBLE_DEVICES": ""}
    ids = [int(i) for i in (opts.get("device_ids") or [0])]
    if launcher in ("ddp", "accelerate") and ids:
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in ids)}
    return None


def _append_standard_flags(cmd: list[str], opts: Mapping, *,
                           metrics: Sequence[str],
                           metric_for_scheduler: str,
                           losses: Sequence[str],
                           use_standard_loss: bool) -> None:
    cmd += ["--metrics", *metrics, "--metric_for_scheduler", metric_for_scheduler]
    if losses:
        cmd += ["--loss", *losses]
    if use_standard_loss:
        cmd.append("--use_standard_loss")
    for key, flag in _SWITCH_FLAGS.items():
        if opts.get(key):
            cmd.append(flag)
    mode = str(opts.get("lora_mode") or "off")
    ckpt = str(opts.get("lora_checkpoint") or "").strip()
    if mode == "peft":
        cmd.append("--train_lora_peft")
        if ckpt:
            cmd += ["--lora_checkpoint_peft", ckpt]
    elif mode == "loralib":
        cmd.append("--train_lora_loralib")
        if ckpt:
            cmd += ["--lora_checkpoint_loralib", ckpt]
    freeze = str(opts.get("freeze_layers") or "").split()
    if freeze:
        cmd += ["--freeze_layers", *freeze]
    key = str(opts.get("wandb_key") or "").strip()
    if key:
        cmd += ["--wandb_key", key]
    if opts.get("wandb_offline"):
        cmd.append("--wandb_offline")


def _append_accelerate_flags(cmd: list[str], opts: Mapping, *,
                             losses: Sequence[str]) -> None:
    # train_accelerate.py is an older loop with a smaller argparse surface.
    if opts.get("pre_valid"):
        cmd.append("--pre_valid")
    # type=bool on --pin_memory treats any non-empty string as True; skip it.
    if "multistft_loss" in losses:
        cmd.append("--use_multistft_loss")
    if "mse_loss" in losses:
        cmd.append("--use_mse_loss")
    if "l1_loss" in losses:
        cmd.append("--use_l1_loss")
    key = str(opts.get("wandb_key") or "").strip()
    if key:
        cmd += ["--wandb_key", key]


def build_train_command(
    *,
    python_exe: str,
    opts: Mapping,
    model_type: str,
    config_path: str,
    results_path: str,
    data_paths: Iterable[str],
    valid_paths: Iterable[str],
    dataset_type: str | int = "1",
    metrics: Sequence[str] = ("sdr",),
    metric_for_scheduler: str = "sdr",
    losses: Sequence[str] = ("masked_loss",),
    checkpoint: str = "",
    custom_backend: str = "",
    use_standard_loss: bool = False,
    launcher: str = "standard",
) -> list[str]:
    """Return the argv for a training subprocess (no env)."""
    data_paths = list(data_paths)
    valid_paths = list(valid_paths)
    metrics = list(metrics) or ["sdr"]
    losses = list(losses) or ["masked_loss"]
    device_ids = _cli_device_ids(opts, launcher)
    num_workers = int(opts.get("num_workers", 4))
    seed = int(opts.get("seed", 0))

    shared = [
        "--model_type", model_type,
        "--config_path", config_path,
        "--results_path", results_path,
        "--data_path", *data_paths,
        "--valid_path", *valid_paths,
        "--dataset_type", str(dataset_type or "1"),
        "--num_workers", str(num_workers),
        "--seed", str(seed),
        "--device_ids", *[str(i) for i in device_ids],
    ]
    ckpt = (checkpoint or "").strip()
    if ckpt:
        shared += ["--start_check_point", ckpt]

    if launcher == "ddp":
        cmd = [python_exe, TRAIN_DDP_SCRIPT, *shared]
        if custom_backend:
            cmd += ["--custom_backend", custom_backend]
        _append_standard_flags(
            cmd, opts, metrics=metrics,
            metric_for_scheduler=metric_for_scheduler,
            losses=losses, use_standard_loss=use_standard_loss)
        return cmd

    if launcher == "accelerate":
        nproc = max(1, len(device_ids))
        cmd = [
            python_exe, "-m", "accelerate.commands.launch",
            "--num_processes", str(nproc),
            TRAIN_ACCELERATE_SCRIPT, *shared,
        ]
        _append_accelerate_flags(cmd, opts, losses=losses)
        return cmd

    cmd = [python_exe, TRAIN_SCRIPT, *shared]
    if custom_backend:
        cmd += ["--custom_backend", custom_backend]
    _append_standard_flags(
        cmd, opts, metrics=metrics,
        metric_for_scheduler=metric_for_scheduler,
        losses=losses, use_standard_loss=use_standard_loss)
    return cmd
