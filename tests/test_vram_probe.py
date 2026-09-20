"""Live Fit GPU allocation probe: JSON parse + trial planner (no CUDA)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect  # noqa: E402

from backend.train_advisor import advise, advise_from_probe, explain_probe_error  # noqa: E402
from backend.vram_probe import (  # noqa: E402
    parse_probe_stdout, plan_from_trials, pick_trial, probe_argv, PROBE_PREFIX,
    probe_allocation,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    raw = "loading model...\n" + PROBE_PREFIX + (
        '{"ok": true, "batch_size": 1, "peak_gb": 18.4}'
    )
    parsed = parse_probe_stdout(raw)
    check(parsed.get("ok") is True and parsed.get("peak_gb") == 18.4,
          "parse last MSST_PROBE line")
    mixed = (
        PROBE_PREFIX + '{"progress": true, "step": 1, "total": 6}\n'
        + PROBE_PREFIX + '{"ok": true, "batch_size": 2, "peak_gb": 11.0}\n'
    )
    mixed_parsed = parse_probe_stdout(mixed)
    check(mixed_parsed.get("ok") is True and mixed_parsed.get("batch_size") == 2,
          "progress lines do not hide the result")
    src = inspect.getsource(probe_allocation)
    dummy = src.find("_dummy_batch")
    trial = src.find('_emit_progress("trial"')
    check(dummy >= 0 and trial > dummy,
          "trial progress emits after the CUDA attempt")
    check(parse_probe_stdout("no json here").get("error") == "no_json",
          "missing probe line")

    trials = [
        {"batch": 1, "checkpoint": True, "ok": True, "peak_gb": 18.4},
        {"batch": 2, "checkpoint": True, "ok": False, "error": "oom"},
    ]
    plan = plan_from_trials(
        trials, current_batch=2, current_accum=1, supports_ckpt=True,
        total_gb=32, peak_gb=18.4)
    check(plan["batch_size"] == 1, f"OOM at 2 → batch 1, got {plan['batch_size']}")
    check(plan["grad_accum"] == 2, f"keep effective 2, got {plan['grad_accum']}")
    check(plan["use_torch_checkpoint"] is True, "checkpoint stays on")
    check(plan["risk"] == "ok", "18 GB of 32 is not tight")

    both = trials + [
        {"batch": 1, "checkpoint": False, "ok": True, "peak_gb": 22.0},
    ]
    best = pick_trial(both)
    check(best and best.get("checkpoint") is False and best.get("batch") == 1,
          "prefer checkpoint-off when the same batch still fits")

    none = plan_from_trials(
        [{"batch": 1, "checkpoint": True, "ok": False, "error": "oom"}],
        supports_ckpt=True, total_gb=24)
    check(none["batch_size"] == 1 and none["risk"] == "tight" and none["oom"],
          "batch 1 OOM is tight")

    fallback = advise(
        model_type="bs_conformer", gpu_mem_gb=32,
        gpu_label="GPU 0: RTX 5090 (32 GB)",
        chunk_size=588800, current_batch=2, current_accum=1)
    probed = advise_from_probe({
        "ok": True, "batch_size": 1, "grad_accum": 2,
        "use_torch_checkpoint": True, "risk": "ok", "peak_gb": 18.4,
        "free_before_gb": 30.1, "weights_gb": 2.2, "total_gb": 31.8,
        "notes": ["Live train step peaked at 18.4 GB of 32 GB (batch 1, checkpoint on)."],
    }, fallback=fallback)
    check(probed.source == "probe" and probed.peak_gb == 18.4,
          "advise_from_probe records live peak")
    check(probed.batch_size == 1 and probed.free_gb == 30.1, "overlay knobs + free")
    check(probed.chunk_size == 588800, "fitting probe keeps the YAML chunk")

    oom_probe = advise_from_probe({
        "ok": True, "batch_size": 1, "grad_accum": 2,
        "use_torch_checkpoint": True, "risk": "tight", "oom": True,
        "notes": ["Batch 1 still CUDA OOM'd. Chunk size is set to 6.7s (294400 samples). Apply writes it into the YAML."],
        "chunk_size": 294400,
    }, fallback=fallback)
    check(oom_probe.chunk_size == 294400, "OOM probe halves chunk onto advice")
    check(not any("chunk size stay as they are" in n.lower() for n in oom_probe.notes),
          "chunk is no longer frozen in the notes")

    skipped = advise_from_probe({"ok": False, "error": "no_config"}, fallback=fallback)
    check(skipped.source == "heuristic", "no config stays heuristic")
    check(any("config YAML" in n for n in skipped.notes), "asks for a config")

    check("no `model:` block" in explain_probe_error("model"),
          "KeyError 'model' is explained")
    check("no `model:` block" in explain_probe_error("missing_yaml:model"),
          "missing_yaml:model is explained")
    keyed = advise_from_probe({"ok": False, "error": "'model'"}, fallback=fallback)
    check(any("`model:`" in n for n in keyed.notes), "notes explain missing model block")
    check(not any("did not finish" in n for n in keyed.notes),
          "raw 'did not finish (model)' is gone")

    argv = probe_argv({
        "config_path": "c.yaml", "model_type": "bs_conformer",
        "device_id": 0, "chunk_size": 441000, "verify_batch": 1,
        "checkpoint": 1,
    })
    joined = " ".join(argv)
    check("--chunk-size" in argv and "441000" in argv, "verify passes chunk samples")
    check("--verify-batch" in argv and "1" in argv, "verify pins the batch")
    check("--checkpoint" in argv, "verify pins checkpoint")
    check("-m" in argv and "backend.vram_probe" in joined, "child is the probe module")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_vram_probe():
    assert main() == 0
