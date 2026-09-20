"""Live CUDA allocation probe for Fit GPU.

The GUI never imports torch. A subprocess loads the selected model, runs one
train step (forward + backward + optimizer step) at increasing batch sizes,
and prints a JSON line. CUDA OOM stays in the child.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from typing import Any, Mapping, Optional, Sequence

from backend.paths import REPO_ROOT, get_python_exe
from backend.train_advisor import (
    _keep_effective,
    family_of,
    suggest_chunk_size,
    supports_torch_checkpoint,
)

PROBE_PREFIX = "MSST_PROBE "
HEADROOM = 0.90
PROBE_TIMEOUT_S = 600
BATCH_CANDIDATES = (1, 2, 4, 8)
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_active_proc: subprocess.Popen | None = None


def parse_probe_line(line: str) -> Optional[dict]:
    """One ``MSST_PROBE`` JSON object, or None if the line is not probe output."""
    text = (line or "").strip()
    if not text.startswith(PROBE_PREFIX):
        return None
    try:
        data = json.loads(text[len(PROBE_PREFIX):])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_probe_stdout(stdout: str) -> dict:
    """Last non-progress ``MSST_PROBE`` JSON line from the child, or an error dict."""
    for line in reversed((stdout or "").splitlines()):
        data = parse_probe_line(line)
        if data is None:
            text = (line or "").strip()
            if text.startswith(PROBE_PREFIX):
                return {"ok": False, "error": "bad_json", "source": "probe"}
            continue
        if data.get("progress"):
            continue
        return data
    return {"ok": False, "error": "no_json", "source": "probe"}


def pick_trial(trials: Sequence[Mapping]) -> Optional[dict]:
    """Largest successful batch, preferring checkpoint-off when it also fit."""
    ok = [dict(t) for t in trials if t.get("ok")]
    if not ok:
        return None
    best = max(ok, key=lambda t: (int(t.get("batch") or 0),
                                  0 if t.get("checkpoint") else 1))
    return best


def plan_from_trials(
    trials: Sequence[Mapping],
    *,
    current_batch: Optional[int] = None,
    current_accum: Optional[int] = None,
    supports_ckpt: bool = False,
    total_gb: Optional[float] = None,
    peak_gb: Optional[float] = None,
) -> dict:
    """Map measured trials onto batch / accum / checkpoint knobs."""
    best = pick_trial(trials)
    if best is None:
        return {
            "batch_size": 1,
            "grad_accum": _keep_effective(1, current_batch, current_accum, 2),
            "use_torch_checkpoint": True if supports_ckpt else None,
            "risk": "tight",
            "oom": True,
        }
    bs = max(1, int(best.get("batch") or 1))
    ckpt = best.get("checkpoint")
    if not supports_ckpt:
        ckpt = None
    peak = peak_gb if peak_gb is not None else best.get("peak_gb")
    risk = "ok"
    if total_gb and peak and float(peak) >= HEADROOM * float(total_gb):
        risk = "tight"
    return {
        "batch_size": bs,
        "grad_accum": _keep_effective(bs, current_batch, current_accum, 2),
        "use_torch_checkpoint": ckpt,
        "risk": risk,
        "oom": False,
    }


def probe_argv(spec: Mapping[str, Any]) -> list[str]:
    """Child argv for a Fit GPU probe. Never imports torch."""
    config = str(spec.get("config_path") or "").strip()
    cmd = [
        get_python_exe(), "-m", "backend.vram_probe",
        "--model-type", str(spec.get("model_type") or ""),
        "--config", config,
        "--device", str(int(spec.get("device_id") or 0)),
        "--current-batch", str(int(spec.get("current_batch") or 0)),
        "--current-accum", str(int(spec.get("current_accum") or 0)),
    ]
    backend = str(spec.get("custom_backend") or "").strip()
    if backend:
        cmd += ["--custom-backend", backend]
    chunk = int(spec.get("chunk_size") or 0)
    if chunk > 0:
        cmd += ["--chunk-size", str(chunk)]
    verify_batch = int(spec.get("verify_batch") or 0)
    if verify_batch > 0:
        cmd += ["--verify-batch", str(verify_batch)]
    if spec.get("checkpoint") is not None:
        cmd += ["--checkpoint", str(int(spec.get("checkpoint")))]
    return cmd


def run_probe_process(spec: Mapping[str, Any]) -> dict:
    """Spawn the allocation probe. Never imports torch in this process."""
    config = str(spec.get("config_path") or "").strip()
    if spec.get("force_cpu"):
        return {"ok": False, "error": "cpu", "source": "heuristic"}
    if not config or not os.path.isfile(config):
        return {"ok": False, "error": "no_config", "source": "heuristic"}
    cmd = probe_argv(spec)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["CUDA_VISIBLE_DEVICES"] = str(int(spec.get("device_id") or 0))
    env["PYTHONUNBUFFERED"] = "1"
    on_progress = spec.get("on_progress")
    timeout = int(spec.get("timeout") or PROBE_TIMEOUT_S)
    global _active_proc
    kw: dict[str, Any] = dict(
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    if sys.platform == "win32":
        kw["creationflags"] = _CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kw["startupinfo"] = si
    chunks: list[str] = []
    timed_out = False

    def _kill():
        nonlocal timed_out
        timed_out = True
        abort_probe()

    try:
        proc = subprocess.Popen(cmd, **kw)
        _active_proc = proc
        killer = threading.Timer(timeout, _kill)
        killer.daemon = True
        killer.start()
        try:
            stdout_f = proc.stdout
            while stdout_f is not None:
                line = stdout_f.readline()
                if line:
                    chunks.append(line)
                    data = parse_probe_line(line)
                    if data and data.get("progress") and callable(on_progress):
                        try:
                            on_progress(data)
                        except Exception:
                            pass
                    continue
                if proc.poll() is not None:
                    rest = stdout_f.read()
                    if rest:
                        chunks.append(rest)
                    break
                break
        finally:
            killer.cancel()
        code = proc.wait()
    except Exception as exc:
        abort_probe()
        return {"ok": False, "error": str(exc), "source": "probe"}
    finally:
        _active_proc = None
    stdout = "".join(chunks)
    result = parse_probe_stdout(stdout or "")
    result.setdefault("source", "probe")
    if timed_out and not result.get("ok"):
        result["error"] = "timeout"
    elif code not in (0, 1) and not result.get("ok"):
        result["error"] = result.get("error") or f"exit_{code}"
    return result


def abort_probe() -> None:
    """Kill a running Fit GPU child so closing the sheet frees the GPU."""
    global _active_proc
    proc = _active_proc
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _active_proc = None


def _gb(n: int | float) -> float:
    return round(float(n) / (1024 ** 3), 2)


def _emit_progress(phase: str, step: int, total: int, **extra: Any) -> None:
    payload = {"progress": True, "phase": phase, "step": int(step), "total": int(total)}
    payload.update(extra)
    print(PROBE_PREFIX + json.dumps(payload), flush=True)


def _dummy_batch(config, batch_size: int, device):
    import torch
    audio = getattr(config, "audio", None)
    chunk = int(getattr(audio, "chunk_size", None) or 131072)
    mix_ch = int(getattr(audio, "num_channels", None) or 2)
    training = getattr(config, "training", None)
    instruments = list(getattr(training, "instruments", None) or ["vocals", "other"])
    target = getattr(training, "target_instrument", None)
    n_stems = 1 if target else max(1, len(instruments))
    x = torch.randn(batch_size, mix_ch, chunk, device=device)
    y = torch.randn(batch_size, n_stems, mix_ch, chunk, device=device)
    return x, y


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _set_checkpoint(model, on: Optional[bool]) -> None:
    if on is None or not hasattr(model, "use_torch_checkpoint"):
        return
    model.use_torch_checkpoint = bool(on)


def probe_allocation(
    *,
    model_type: str,
    config_path: str,
    device: int = 0,
    current_batch: int = 0,
    current_accum: int = 0,
    custom_backend: str = "",
    chunk_size: int = 0,
    verify_batch: int = 0,
    checkpoint: int = -1,
) -> dict:
    """Load the model and measure a real train step. Runs in the child."""
    import torch
    from torch.cuda.amp.grad_scaler import GradScaler

    from utils.model_utils import effective_use_amp, get_optimizer
    from utils.settings import get_model_from_config

    if not torch.cuda.is_available():
        return {"ok": False, "error": "no_cuda", "source": "heuristic"}

    device_id = 0  # CUDA_VISIBLE_DEVICES already picked the card
    torch.cuda.set_device(device_id)
    props = torch.cuda.get_device_properties(device_id)
    total_gb = _gb(props.total_memory)
    free_before, _total = torch.cuda.mem_get_info(device_id)
    free_before_gb = _gb(free_before)

    verify = bool(verify_batch and int(verify_batch) > 0)
    ckpt_likely = supports_torch_checkpoint(model_type)
    if verify:
        n_trials = 1 if (not ckpt_likely or int(checkpoint) >= 0) else 2
    else:
        n_trials = len(BATCH_CANDIDATES) + (1 if ckpt_likely else 0)
    total_steps = 1 + max(1, n_trials)
    step = 0
    _emit_progress("load", step, total_steps)

    model, config = get_model_from_config(
        model_type, config_path,
        custom_backend=custom_backend or None)
    if chunk_size and int(chunk_size) > 0:
        audio = getattr(config, "audio", None)
        if audio is not None:
            try:
                audio.chunk_size = int(chunk_size)
            except Exception:
                pass
    model.train()
    model.to(device_id)
    torch.cuda.synchronize()
    weights_gb = _gb(torch.cuda.memory_allocated(device_id))
    step = 1
    _emit_progress("loaded", step, total_steps)

    try:
        optimizer = get_optimizer(config, model)
    except Exception:
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=1e-4)

    use_amp = bool(effective_use_amp(config))
    scaler = GradScaler(enabled=use_amp)
    internal = model_type in (
        "mel_band_roformer", "bs_roformer", "bs_mamba2",
        "mel_band_conformer", "bs_conformer",
    )
    ckpt_ok = ckpt_likely and hasattr(model, "use_torch_checkpoint")
    ckpt_order = (True, False) if ckpt_ok else (None,)
    batches = BATCH_CANDIDATES
    if verify:
        batches = (int(verify_batch),)
        if ckpt_ok and int(checkpoint) >= 0:
            ckpt_order = (bool(int(checkpoint)),)

    trials = []
    last_ok = None
    stop = False
    for ckpt in ckpt_order:
        if stop:
            break
        _set_checkpoint(model, ckpt)
        for bs in batches:
            if ckpt is False and not verify:
                if last_ok is None or bs != int(last_ok.get("batch") or 1):
                    continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_id)
            optimizer.zero_grad(set_to_none=True)
            try:
                x, y = _dummy_batch(config, bs, device_id)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if internal:
                        loss = model(x, y)
                        if hasattr(loss, "mean"):
                            loss = loss.mean()
                    else:
                        pred = model(x)
                        loss = torch.nn.functional.l1_loss(pred, y[:, :pred.shape[1]])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                peak = _gb(torch.cuda.max_memory_allocated(device_id))
                row = {
                    "batch": bs, "checkpoint": ckpt, "ok": True,
                    "peak_gb": peak,
                }
                trials.append(row)
                last_ok = row
                del x, y, loss
                if peak >= HEADROOM * total_gb:
                    stop = True
                    break
            except Exception as exc:
                trials.append({
                    "batch": bs, "checkpoint": ckpt, "ok": False,
                    "error": "oom" if _is_oom(exc) else str(exc)[:240],
                })
                del optimizer
                try:
                    optimizer = get_optimizer(config, model)
                except Exception:
                    optimizer = torch.optim.AdamW(
                        (p for p in model.parameters() if p.requires_grad), lr=1e-4)
                torch.cuda.empty_cache()
                if ckpt is False:
                    stop = True
                break
            finally:
                # Count the attempt after it finishes so ETA does not treat
                # the in-flight forward/backward as already done.
                step += 1
                _emit_progress("trial", step, total_steps, batch=int(bs))

    planned = plan_from_trials(
        trials,
        current_batch=current_batch or None,
        current_accum=current_accum or None,
        supports_ckpt=ckpt_ok,
        total_gb=total_gb,
        peak_gb=(last_ok or {}).get("peak_gb"),
    )
    notes = []
    audio = getattr(config, "audio", None)
    try:
        chunk = int(getattr(audio, "chunk_size", None) or 131072)
    except (TypeError, ValueError):
        chunk = 131072
    try:
        sr = int(getattr(audio, "sample_rate", None) or 44100) or 44100
    except (TypeError, ValueError):
        sr = 44100
    oom = last_ok is None
    halve = oom and (not verify or int(verify_batch) <= 1)
    out_chunk = suggest_chunk_size(chunk, oom=halve) or chunk
    if last_ok:
        notes.append(
            f"Live train step peaked at {last_ok['peak_gb']:.1f} GB "
            f"of {total_gb:.0f} GB (batch {last_ok['batch']}"
            + (", checkpoint on" if last_ok.get("checkpoint") else "")
            + ")."
        )
        if planned.get("use_torch_checkpoint"):
            notes.append(
                "Activation checkpointing stayed on — it is what let the step fit."
            )
    else:
        secs = out_chunk / float(sr) if sr else 0
        if verify:
            notes.append(
                f"Those knobs still CUDA OOM'd at batch {int(verify_batch)}. "
                f"Chunk size is set to {secs:.1f}s ({out_chunk} samples). "
                "Apply re-checks before writing the YAML."
            )
        else:
            notes.append(
                f"Batch 1 still CUDA OOM'd. Chunk size is set to {secs:.1f}s "
                f"({out_chunk} samples). Apply re-checks that window before "
                "writing the YAML."
            )
        if out_chunk >= chunk:
            notes.append("Pick a larger GPU if a shorter window still will not fit.")
    return {
        "ok": True,
        "source": "probe",
        "gpu_name": props.name,
        "total_gb": total_gb,
        "free_before_gb": free_before_gb,
        "weights_gb": weights_gb,
        "peak_gb": (last_ok or {}).get("peak_gb"),
        "trials": trials,
        "notes": notes,
        "family": family_of(model_type),
        "chunk_size": out_chunk,
        **planned,
        "verify": verify,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fit GPU live allocation probe")
    p.add_argument("--model-type", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--current-batch", type=int, default=0)
    p.add_argument("--current-accum", type=int, default=0)
    p.add_argument("--custom-backend", default="")
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--verify-batch", type=int, default=0)
    p.add_argument("--checkpoint", type=int, default=-1)
    args = p.parse_args(argv)
    try:
        result = probe_allocation(
            model_type=args.model_type,
            config_path=args.config,
            device=args.device,
            current_batch=args.current_batch,
            current_accum=args.current_accum,
            custom_backend=args.custom_backend,
            chunk_size=args.chunk_size,
            verify_batch=args.verify_batch,
            checkpoint=args.checkpoint,
        )
    except KeyError as exc:
        key = exc.args[0] if exc.args else "unknown"
        result = {"ok": False, "error": f"missing_yaml:{key}", "source": "probe"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)[:400], "source": "probe"}
    print(PROBE_PREFIX + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
