"""Regression: resuming training from --start_check_point must not crash.

The ZIP64 checkpoint fix freed the loaded checkpoint dict with a
``finally: del checkpoint`` right after the weights were copied out, but the
resume-metadata blocks below (optimizer/scheduler/epoch/best-metric/
all-metrics/all-losses) still referenced ``checkpoint`` — every resume run
died with ``UnboundLocalError: cannot access local variable 'checkpoint'``.
The fix extracts those keys into a small ``resume_meta`` dict before the
delete, so the metadata survives while the (possibly multi-GB) state dict is
still freed early.

This test drives the real ``train_model`` with the heavy outer layers
stubbed (model build, dataset, device, optimizer construction) while the
checkpoint load, ``del``, ``resume_meta`` extraction and resume blocks run
for real, then asserts the resume metadata is consumed exactly as before.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from ml_collections import ConfigDict  # noqa: E402

import train as train_mod  # noqa: E402
import utils.model_utils  # noqa: E402
import utils.dataset  # noqa: E402
import utils.losses  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


class _Recorder:
    """Stands in for optimizer/scheduler; records load_state_dict calls."""

    def __init__(self):
        self.calls = []

    def load_state_dict(self, state):
        self.calls.append(state)


def _stub_config():
    return ConfigDict({"training": {
        "batch_size": 1,
        "num_epochs": 0,          # loop body never runs in this test
        "instruments": ["vocals", "other"],
        "patience": 2,
        "reduce_factor": 0.95,
        "optimizer": "adam",
        "lr": 1e-4,
    }})


def _make_args(results, checkpoint_path, load_flags):
    return types.SimpleNamespace(
        model_type="mdx23c", config_path="stub.yaml",
        custom_backend=None, start_check_point=checkpoint_path,
        results_path=results, seed=0, device_ids="0", pre_valid=False,
        metrics=["sdr"], metric_for_scheduler="sdr", loss="masked_loss",
        freeze_layers=None, set_per_process_memory_fraction=None,
        safe_mode=False, num_workers=0, data_path="stub", valid_path="stub",
        dataset_type=1, load_only_compatible_weights=False,
        lora_checkpoint="", lora_checkpoint_loralib="",
        load_optimizer=("load_optimizer" in load_flags),
        load_scheduler=("load_scheduler" in load_flags),
        load_epoch=("load_epoch" in load_flags),
        load_best_metric=("load_best_metric" in load_flags),
        load_all_metrics=("load_all_metrics" in load_flags),
        load_all_losses=("load_all_losses" in load_flags),
    )


def _run_train_model(args, model, checkpoint, optimizer, scheduler):
    """Run train_model with the heavy outer layers stubbed; the checkpoint
    load + resume_meta + resume blocks execute for real. Returns the
    exception raised (or None)."""
    model_utils = utils.model_utils
    originals = {}
    patches = []

    def patch(target, attr, value):
        originals[(target, attr)] = getattr(target, attr)
        setattr(target, attr, value)
        patches.append((target, attr))

    def restore():
        for target, attr in patches:
            setattr(target, attr, originals[(target, attr)])

    try:
        # Outer layers: model/dataset/device/optimizer construction.
        patch(train_mod, "parse_args_train", lambda _a: args)
        patch(train_mod, "initialize_environment", lambda *a, **k: None)
        patch(train_mod, "wandb_init", lambda *a, **k: None)
        patch(train_mod, "get_model_from_config",
              lambda *a, **k: (model, _stub_config()))
        patch(train_mod, "initialize_model_and_device",
              lambda m, device_ids: (torch.device("cpu"), m))
        patch(train_mod, "get_scheduler", lambda config, opt: scheduler)
        patch(utils.dataset, "prepare_data", lambda *a, **k: [])
        patch(utils.losses, "choice_loss",
              lambda *a, **k: (lambda *x, **y: torch.tensor(0.0)))
        patch(model_utils, "effective_use_amp", lambda config: False)
        patch(model_utils, "get_optimizer", lambda config, m: optimizer)
        patch(model_utils, "get_lora", lambda args, config, m: m)
        patch(model_utils, "log_model_info", lambda *a, **k: None)

        # Real path under test: ensure_readable_checkpoint -> torch.load ->
        # load_start_checkpoint -> del checkpoint -> resume_meta blocks.
        try:
            train_mod.train_model(None)
            return None
        except BaseException as e:  # noqa: BLE001 - reported by caller
            return e
    finally:
        restore()


def main():
    # Tiny model + a checkpoint carrying every resume-metadata key.
    model = torch.nn.Linear(4, 4)
    ckpt = {
        "state": model.state_dict(),
        "optimizer_state_dict": {"adam": 1},
        "scheduler_state_dict": {"plateau": 2},
        "epoch": 4,
        "best_metric": 1.234,
        "all_metrics": {"epoch_0": {"sdr": 1.0}},
        "all_losses": {"0": 0.5},
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "resume.ckpt")
        torch.save(ckpt, ckpt_path)

        # 1. Resume with every --load_* flag: all metadata is consumed.
        opt = _Recorder()
        sched = _Recorder()
        err = _run_train_model(
            _make_args(tmp, ckpt_path, {"load_optimizer", "load_scheduler",
                                       "load_epoch", "load_best_metric",
                                       "load_all_metrics", "load_all_losses"}),
            model, ckpt, opt, sched)
        check(err is None, f"resume+flags raised: {err!r}")
        check(opt.calls == [ckpt["optimizer_state_dict"]],
              "optimizer state not loaded from resume checkpoint")
        check(sched.calls == [ckpt["scheduler_state_dict"]],
              "scheduler state not loaded from resume checkpoint")

        # 2. Resume without load flags: no crash, nothing consumed.
        opt = _Recorder()
        sched = _Recorder()
        err = _run_train_model(
            _make_args(tmp, ckpt_path, set()), model, ckpt, opt, sched)
        check(err is None, f"resume plain raised: {err!r}")
        check(opt.calls == [], "optimizer state loaded despite no flag")
        check(sched.calls == [], "scheduler state loaded despite no flag")

        # 3. No checkpoint: the plain training path is unchanged.
        opt = _Recorder()
        sched = _Recorder()
        err = _run_train_model(
            _make_args(tmp, "", set()), model, ckpt, opt, sched)
        check(err is None, f"plain path raised: {err!r}")
        check(opt.calls == [], "optimizer state loaded without checkpoint")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())