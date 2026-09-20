"""
backend/gpu_utils.py
---------------------
Detects available CUDA GPUs without importing torch (so the GUI starts fast
— torch loads in seconds and the GUI process never needs it anyway, since
inference runs in a subprocess). NVIDIA cards are queried through the
driver's own nvidia-smi tool; torch is only used as a fallback.
"""
from __future__ import annotations

import re
import shutil
import time
from typing import Iterable, Optional

from backend.win_startup import hidden_run

_GPU_MEM_RE = re.compile(r"\(([\d.]+)\s*GB\)", re.I)
HEAVY_CONFORMER_TYPES = frozenset({"bs_conformer", "mel_band_conformer"})
CONFORMER_OOM_VRAM_GB = 40.0

_cached_gpus: list[str] | None = None
_mem_cache: list[dict] | None = None
_mem_cache_t: float = 0.0
_MEM_CACHE_S = 2.0


def _gpus_via_nvidia_smi() -> list[str] | None:
    """Query the NVIDIA driver tool directly. Returns None when unavailable."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = hidden_run(
            [smi, "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=4,
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


def parse_gpu_memory_gb(label: str) -> Optional[float]:
    """VRAM in GB from a ``list_gpus`` label, or None for CPU / unknown."""
    if not label or str(label).startswith("CPU"):
        return None
    m = _GPU_MEM_RE.search(label)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_smi_memory_rows(stdout: str) -> list[dict]:
    """Parse nvidia-smi csv (index, name, total MiB, used MiB, free MiB)."""
    rows = []
    for line in (stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            total = float(parts[2]) / 1024.0
            used = float(parts[3]) / 1024.0
            free = float(parts[4]) / 1024.0
        except ValueError:
            continue
        rows.append({
            "index": idx,
            "name": parts[1],
            "total_gb": total,
            "used_gb": used,
            "free_gb": free,
        })
    return rows


def query_gpu_memory(refresh: bool = False) -> list[dict]:
    """Live used/free/total VRAM per CUDA index via nvidia-smi."""
    global _mem_cache, _mem_cache_t
    now = time.monotonic()
    if not refresh and _mem_cache is not None and (now - _mem_cache_t) < _MEM_CACHE_S:
        return list(_mem_cache)
    smi = shutil.which("nvidia-smi")
    rows: list[dict] = []
    if smi:
        try:
            out = hidden_run(
                [smi, "--query-gpu=index,name,memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=4,
            )
            if out.returncode == 0:
                rows = parse_smi_memory_rows(out.stdout)
        except Exception:
            rows = []
    _mem_cache = rows
    _mem_cache_t = now
    return list(rows)


def selected_gpu_snapshot(
    device_ids: Iterable[int] | None,
    rows: Iterable[dict] | None = None,
) -> Optional[dict]:
    """Tightest-free snapshot among selected GPUs."""
    rows = list(query_gpu_memory() if rows is None else rows)
    ids = list(device_ids or [0])
    picked = [r for r in rows if r.get("index") in ids]
    if not picked:
        return None
    return min(picked, key=lambda r: float(r.get("free_gb") or 0))


def selected_gpu_memory_gb(
    device_ids: Iterable[int] | None,
    labels: Iterable[str] | None = None,
) -> Optional[float]:
    """Smallest VRAM among the selected CUDA devices, or None if unknown."""
    labels = list_gpus() if labels is None else list(labels)
    by_id: dict[int, float] = {}
    for g in labels:
        if str(g).startswith("CPU"):
            continue
        try:
            idx = int(str(g).split("GPU")[1].split(":")[0].strip())
        except Exception:
            continue
        mem = parse_gpu_memory_gb(g)
        if mem is not None:
            by_id[idx] = mem
    ids = list(device_ids or [0])
    mems = [by_id[i] for i in ids if i in by_id]
    return min(mems) if mems else None


def conformer_vram_warning(
    *,
    model_type: str,
    batch_size: int,
    use_torch_checkpoint: bool,
    gpu_mem_gb: float | None,
) -> Optional[str]:
    """Warn when PolarFormer / BS-Conformer will likely CUDA OOM.

    These models at batch size > 1 without activation checkpointing
    typically need ~40 GB+. A 32 GB card OOM'd at ~45 GB allocated.
    """
    if str(model_type or "") not in HEAVY_CONFORMER_TYPES:
        return None
    if gpu_mem_gb is None or gpu_mem_gb >= CONFORMER_OOM_VRAM_GB:
        return None
    try:
        bs = int(batch_size)
    except (TypeError, ValueError):
        bs = 1
    if bs <= 1 and use_torch_checkpoint:
        return None
    parts = []
    if bs > 1:
        parts.append(f"batch size {bs}")
    if not use_torch_checkpoint:
        parts.append("activation checkpointing off")
    why = " and ".join(parts) or "this setup"
    return (
        f"BS-Conformer / PolarFormer with {why} typically needs ~40 GB+; "
        f"this GPU has {gpu_mem_gb:.0f} GB and is likely to CUDA OOM. "
        f"Set batch size to 1, raise grad accum steps to keep the effective "
        f"batch, and set model.use_torch_checkpoint: true in the config."
    )
