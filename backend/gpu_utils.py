"""
backend/gpu_utils.py
---------------------
Detects available CUDA GPUs without importing torch (so the GUI starts fast
— torch loads in seconds and the GUI process never needs it anyway, since
inference runs in a subprocess). NVIDIA cards are queried through the
driver's own nvidia-smi tool; torch is only used as a fallback.
"""
from __future__ import annotations

import shutil
import subprocess

_cached_gpus: list[str] | None = None


def _gpus_via_nvidia_smi() -> list[str] | None:
    """Query the NVIDIA driver tool directly. Returns None when unavailable."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode != 0:
            return None
        devices = []
        for i, line in enumerate(out.stdout.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            name, mib = parts[0], parts[1]
            try:
                mem_gb = float(mib) * (1024 * 1024) / (1024 ** 3)
            except ValueError:
                continue
            devices.append(f"GPU {i}: {name} ({mem_gb:.0f} GB)")
        return devices or None
    except Exception:
        return None


def _gpus_via_torch() -> list[str]:
    """Original torch-based detection, kept as a fallback."""
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024 ** 3)
                devices.append(f"GPU {i}: {props.name} ({mem_gb:.0f} GB)")
    except Exception:
        pass
    return devices


def list_gpus(refresh: bool = False) -> list[str]:
    """
    Returns a list like ['CPU', 'GPU 0: NVIDIA RTX 4090 (24 GB)', …].
    Falls back gracefully when torch, CUDA or nvidia-smi is not available.
    Results are cached; pass refresh=True to re-detect.
    """
    global _cached_gpus
    if _cached_gpus is not None and not refresh:
        return list(_cached_gpus)
    devices = ["CPU"]
    gpus = _gpus_via_nvidia_smi()
    if gpus is None:
        gpus = _gpus_via_torch()
    devices.extend(gpus)
    _cached_gpus = devices
    return devices


def device_ids_from_selection(label: str) -> list[int] | None:
    """
    Convert a GPU label string back to a device-id list.
    'CPU'           → None  (caller should set force_cpu=True)
    'GPU 0: …'     → [0]
    """
    if label.startswith("CPU"):
        return None
    try:
        idx = int(label.split("GPU")[1].split(":")[0].strip())
        return [idx]
    except Exception:
        return [0]
