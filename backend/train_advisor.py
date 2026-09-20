"""Recommend training knobs from GPU VRAM + architecture family.

This is a lookup table, not a measured dry-run. A real allocation probe would
import torch into the GUI (slow start) and can CUDA OOM the machine we are
trying to protect. The numbers track community practice plus the PolarFormer
OOM we recorded: batch 2 without checkpointing allocated ~45 GB on a 32 GB
card.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from backend.gpu_utils import list_gpus, selected_gpu_memory_gb

# PolarFormer / long RoFormer windows are ~13 s of 44.1 kHz audio.
LONG_CHUNK_SAMPLES = 400_000
# Floor when halving a window that still CUDA OOM'd at batch 1.
MIN_CHUNK_SAMPLES = 32_768

FAMILY_LABELS = {
    "conformer": "Conformer / PolarFormer",
    "roformer": "RoFormer / band-split transformer",
    "mamba": "BS-Mamba",
    "scnet": "SCNet",
    "bandit": "Bandit",
    "demucs": "Demucs / HTDemucs",
    "mdx": "MDX-Net / MDX23C",
    "seg": "Segmentation / VR",
    "other": "Other",
}

_HEAVY = frozenset({"conformer", "roformer", "mamba"})
_SUPPORTS_CHECKPOINT = frozenset({"conformer", "roformer"})


@dataclass(frozen=True)
class TrainAdvice:
    family: str
    family_label: str
    gpu_label: str
    gpu_mem_gb: Optional[float]
    batch_size: int
    grad_accum: int
    use_torch_checkpoint: Optional[bool]
    pin_memory: bool
    optimizer: Optional[str]
    risk: str  # "ok" | "tight" | "cpu"
    notes: tuple[str, ...] = field(default_factory=tuple)
    source: str = "heuristic"  # "heuristic" | "probe"
    peak_gb: Optional[float] = None
    free_gb: Optional[float] = None
    weights_gb: Optional[float] = None
    chunk_size: Optional[int] = None
    sample_rate: int = 44100

    @property
    def effective_batch(self) -> int:
        return int(self.batch_size) * int(self.grad_accum)

    @property
    def gpu_short(self) -> str:
        label = self.gpu_label or "GPU"
        if "(" in label:
            return label.split(":", 1)[-1].strip()
        return label


def family_of(model_type: str) -> str:
    t = (model_type or "").strip().lower()
    if not t:
        return "other"
    if t in ("bs_conformer", "mel_band_conformer") or "conformer" in t:
        return "conformer"
    if t == "bs_mamba2":
        return "mamba"
    if "roformer" in t:
        return "roformer"
    if t.startswith("scnet"):
        return "scnet"
    if t.startswith("bandit"):
        return "bandit"
    if "demucs" in t:
        return "demucs"
    if t in ("mdx23c", "experimental_mdx23c_stht", "dttnet", "dtt_net", "mdxnet"):
        return "mdx"
    if t in ("segm_models", "torchseg", "swin_upernet", "vr"):
        return "seg"
    return "other"


def supports_torch_checkpoint(model_type: str) -> bool:
    return family_of(model_type) in _SUPPORTS_CHECKPOINT


def gpu_label_for_ids(
    device_ids: Iterable[int] | None,
    labels: Iterable[str] | None = None,
    *,
    force_cpu: bool = False,
) -> str:
    if force_cpu:
        return "CPU"
    labels = list(list_gpus() if labels is None else labels)
    ids = list(device_ids or [0])
    named = []
    for g in labels:
        if str(g).startswith("CPU"):
            continue
        try:
            idx = int(str(g).split("GPU")[1].split(":")[0].strip())
        except Exception:
            continue
        if idx in ids:
            named.append(g)
    if named:
        return " + ".join(named)
    gpus = [g for g in labels if not str(g).startswith("CPU")]
    return gpus[0] if gpus else "CPU"


def suggest_chunk_size(current: Optional[int], *, oom: bool = False) -> Optional[int]:
    """Halve a window that still OOM'd at batch 1; otherwise keep it."""
    if current is None:
        return None
    try:
        cur = int(current)
    except (TypeError, ValueError):
        return None
    if cur <= 0:
        return None
    if not oom:
        return cur
    halved = max(MIN_CHUNK_SAMPLES, cur // 2)
    halved -= halved % 2
    return halved if halved < cur else cur


def _ceil_div(a: int, b: int) -> int:
    return max(1, (int(a) + int(b) - 1) // int(b))


def _keep_effective(new_bs: int, current_bs: Optional[int],
                    current_accum: Optional[int], default_eff: int) -> int:
    cur_b = current_bs if current_bs and current_bs > 0 else new_bs
    cur_a = current_accum if current_accum and current_accum > 0 else 1
    target = max(cur_b * cur_a, default_eff)
    return _ceil_div(target, new_bs)


def _heavy_batch(mem: Optional[float], long_chunk: bool) -> tuple[int, bool, str]:
    """Returns (batch_size, use_checkpoint, risk)."""
    if mem is None or mem <= 0:
        return 1, True, "cpu"
    # Measured: PolarFormer bs=2, checkpoint off → ~45 GB allocated.
    # Long windows make a 32 GB card behave like a smaller one.
    if long_chunk:
        if mem < 48:
            risk = "tight" if mem < 24 else "ok"
            return 1, True, risk
        if mem < 80:
            return 2, True, "ok"
        return 4, False, "ok"
    if mem < 16:
        return 1, True, "tight"
    if mem < 40:
        risk = "tight" if mem < 24 else "ok"
        return 1, True, risk
    if mem < 48:
        return 2, True, "ok"
    if mem < 80:
        return 2, False, "ok"
    return 4, False, "ok"


def _medium_batch(mem: Optional[float]) -> tuple[int, str]:
    if mem is None or mem <= 0:
        return 1, "cpu"
    if mem < 10:
        return 1, "tight"
    if mem < 14:
        return 2, "ok"
    if mem < 20:
        return 4, "ok"
    if mem < 28:
        return 6, "ok"
    return 8, "ok"


def advise(
    *,
    model_type: str,
    gpu_mem_gb: Optional[float],
    gpu_label: str = "",
    chunk_size: Optional[int] = None,
    sample_rate: int = 44100,
    current_batch: Optional[int] = None,
    current_accum: Optional[int] = None,
    force_cpu: bool = False,
) -> TrainAdvice:
    family = family_of(model_type)
    label = FAMILY_LABELS[family]
    notes: list[str] = []
    long_chunk = bool(chunk_size and int(chunk_size) >= LONG_CHUNK_SAMPLES)
    if force_cpu:
        gpu_mem_gb = None
        gpu_label = gpu_label or "CPU"

    optimizer = None
    pin_memory = not force_cpu and gpu_mem_gb is not None and gpu_mem_gb > 0
    ckpt: Optional[bool] = None

    if family in _HEAVY:
        bs, ckpt_on, risk = _heavy_batch(gpu_mem_gb, long_chunk)
        ckpt = ckpt_on if supports_torch_checkpoint(model_type) else None
        accum = _keep_effective(bs, current_batch, current_accum, default_eff=2)
        if gpu_mem_gb is not None and gpu_mem_gb < 16 and not force_cpu:
            optimizer = "adamw8bit"
            notes.append(
                "adamw8bit stores optimizer state in 8-bit — the Prodigy / "
                "Adam copies of a 100M+ transformer will not fit a small card."
            )
        if long_chunk and sample_rate:
            secs = int(chunk_size) / float(sample_rate)
            notes.append(
                f"Chunk size is ~{secs:.1f}s of audio. Long windows dominate "
                "activation memory; batch 2 without checkpointing OOM'd a "
                "32 GB GPU at ~45 GB allocated."
            )
        elif ckpt:
            notes.append(
                "Activation checkpointing trades a slower step for much less "
                "VRAM. Leave it on unless the card has 48 GB+."
            )
    else:
        bs, risk = _medium_batch(gpu_mem_gb)
        accum = _keep_effective(bs, current_batch, current_accum, default_eff=bs)
        notes.append(
            f"{label} models are lighter than RoFormer / Conformer. Batch "
            "size is still per GPU; raise grad accum if you need a larger "
            "effective batch."
        )

    if risk == "cpu":
        notes.insert(0, "No CUDA GPU is selected. Source-separation training on CPU is not practical.")
        pin_memory = False
        ckpt = True if family in _SUPPORTS_CHECKPOINT else ckpt
        bs, accum = 1, 1
    elif risk == "tight":
        notes.insert(0, "This GPU is under the usual VRAM for this architecture. Start here; drop chunk size only if it still OOM's.")

    notes.append(
        "These are best-practice heuristics, not a live VRAM probe. "
        "Mixed precision (use_amp) should stay on."
    )

    return TrainAdvice(
        family=family,
        family_label=label,
        gpu_label=gpu_label or ("CPU" if risk == "cpu" else "GPU"),
        gpu_mem_gb=gpu_mem_gb,
        batch_size=int(bs),
        grad_accum=int(accum),
        use_torch_checkpoint=ckpt,
        pin_memory=bool(pin_memory),
        optimizer=optimizer,
        risk=risk,
        notes=tuple(notes),
        chunk_size=int(chunk_size) if chunk_size else None,
        sample_rate=int(sample_rate or 44100),
    )


def advise_from_selection(
    *,
    model_type: str,
    device_ids: Iterable[int] | None,
    force_cpu: bool = False,
    labels: Iterable[str] | None = None,
    chunk_size: Optional[int] = None,
    sample_rate: int = 44100,
    current_batch: Optional[int] = None,
    current_accum: Optional[int] = None,
) -> TrainAdvice:
    labels = list(list_gpus() if labels is None else labels)
    gpu_label = gpu_label_for_ids(device_ids, labels, force_cpu=force_cpu)
    mem = None if (force_cpu or gpu_label == "CPU") else selected_gpu_memory_gb(
        device_ids, labels)
    return advise(
        model_type=model_type,
        gpu_mem_gb=mem,
        gpu_label=gpu_label,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        current_batch=current_batch,
        current_accum=current_accum,
        force_cpu=force_cpu or gpu_label == "CPU",
    )


_PROBE_ERR = {
    "no_config": "Pick a config YAML to measure a real train step on this GPU.",
    "cpu": "CPU is selected — there is nothing to allocate on.",
    "timeout": "The live train-step timed out. Showing the usual knobs.",
    "no_cuda": "CUDA is not available to the probe. Showing the usual knobs.",
    "no_json": "The live train-step produced no result. Showing the usual knobs.",
    "bad_json": "The live train-step result was unreadable. Showing the usual knobs.",
}


def explain_probe_error(err) -> str:
    """Turn a probe error token into one sentence for the Fit GPU sheet."""
    raw = str(err or "").strip().strip("'\"")
    if not raw:
        return "The live train-step could not start. Showing the usual knobs."
    if raw in _PROBE_ERR:
        return _PROBE_ERR[raw]
    if raw.startswith("missing_yaml:"):
        key = raw.split(":", 1)[-1].strip().strip("'\"") or "model"
        return (
            f"The config YAML has no `{key}:` block, so the live train-step "
            "could not load the model. Showing the usual knobs."
        )
    if raw in ("model", "audio", "training"):
        return (
            f"The config YAML has no `{raw}:` block, so the live train-step "
            "could not load the model. Showing the usual knobs."
        )
    one = raw.splitlines()[0].strip()
    if len(one) > 160:
        one = one[:157] + "..."
    return f"The live train-step could not start: {one} Showing the usual knobs."


def advise_from_probe(
    result: dict,
    *,
    fallback: TrainAdvice,
) -> TrainAdvice:
    """Overlay a live allocation-probe result onto the heuristic advice."""
    if not result or not result.get("ok"):
        notes = list(fallback.notes)
        err = (result or {}).get("error")
        if err:
            notes.insert(0, explain_probe_error(err))
        return replace(fallback, notes=tuple(notes), source=fallback.source)

    notes = list(result.get("notes") or ())
    notes.append("Learning rate and epochs stay as they are.")
    ckpt = result.get("use_torch_checkpoint")
    if ckpt is not None:
        ckpt = bool(ckpt)
    peak = result.get("peak_gb")
    free = result.get("free_before_gb")
    total = result.get("total_gb")
    gpu_mem = fallback.gpu_mem_gb
    if total:
        gpu_mem = float(total)
    risk = str(result.get("risk") or fallback.risk)
    chunk = result.get("chunk_size")
    if chunk is not None:
        try:
            chunk = int(chunk)
        except (TypeError, ValueError):
            chunk = fallback.chunk_size
    else:
        chunk = fallback.chunk_size
    return replace(
        fallback,
        batch_size=int(result.get("batch_size") or fallback.batch_size),
        grad_accum=int(result.get("grad_accum") or fallback.grad_accum),
        use_torch_checkpoint=ckpt if ckpt is not None else fallback.use_torch_checkpoint,
        pin_memory=True,
        risk=risk,
        notes=tuple(notes),
        source="probe",
        peak_gb=float(peak) if peak is not None else None,
        free_gb=float(free) if free is not None else None,
        weights_gb=float(result["weights_gb"]) if result.get("weights_gb") is not None else None,
        gpu_mem_gb=gpu_mem,
        chunk_size=chunk,
    )
