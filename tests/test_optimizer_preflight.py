"""Regression: selecting adamw8bit without CUDA must fail with a clear message.

get_optimizer (utils/model_utils.py) raises a ValueError naming the optimizer
and the missing GPU before touching bitsandbytes, so a CPU-only runtime gets a
readable failure in the job log instead of a cryptic bitsandbytes/optimizer
error mid-run. The GUI additionally blocks the launch (training_page.py) — the
engine-side guard is the backstop for CLI / config-driven runs and for the
"CPU only" job option (CUDA_VISIBLE_DEVICES=\"\").
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ml_collections import ConfigDict  # noqa: E402

from utils.model_utils import (  # noqa: E402
    get_optimizer, assert_cuda_available, effective_use_amp)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def _config():
    return ConfigDict({"training": {"optimizer": "adamw8bit", "lr": 1e-4},
                       "optimizer": {}})


def main():
    model = torch.nn.Linear(4, 4)
    cfg = _config()

    # Simulate a CPU-only runtime: torch.cuda.is_available() == False.
    real_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        try:
            get_optimizer(cfg, model)
            check(False, "adamw8bit without CUDA must raise ValueError")
        except ValueError as exc:
            msg = str(exc)
            check("adamw8bit" in msg, "message must name the optimizer")
            check("CUDA" in msg, "message must name CUDA/GPU as the requirement")
            check("'adamw' or 'adam'" in msg,
                  "message must suggest a working alternative")
    finally:
        torch.cuda.is_available = real_is_available

    # With CUDA present the guard must pass through to the real optimizer.
    if torch.cuda.is_available():
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError:
            print("note: bitsandbytes not installed; positive-path check skipped")
        else:
            opt = get_optimizer(_config(), model)
            check(isinstance(opt, AdamW8bit),
                  "adamw8bit with CUDA must construct bitsandbytes AdamW8bit")
    else:
        print("note: dev env has no CUDA; positive-path check skipped")

    # assert_cuda_available: shared guard behind every CUDA-only option.
    torch.cuda.is_available = lambda: False
    try:
        try:
            assert_cuda_available("--set_per_process_memory_fraction (cap VRAM)",
                                  "Remove the flag to train without a GPU.")
            check(False, "assert_cuda_available without CUDA must raise")
        except ValueError as exc:
            msg = str(exc)
            check("requires a CUDA GPU" in msg,
                  "guard message must state the CUDA requirement")
            check("Remove the flag" in msg,
                  "guard message must carry the option-specific remedy")
    finally:
        torch.cuda.is_available = real_is_available
    # No CUDA -> no raise.
    try:
        assert_cuda_available("x", "y.")
        check(True, "assert_cuda_available with CUDA must not raise")
    except ValueError:
        check(False, "assert_cuda_available with CUDA raised unexpectedly")

    # effective_use_amp: AMP is CUDA-bound — disabled on CPU-only runtimes.
    amp_cfg = ConfigDict({"training": {"use_amp": True}})
    torch.cuda.is_available = lambda: False
    try:
        check(effective_use_amp(amp_cfg) is False,
              "AMP must be disabled when no CUDA is available")
    finally:
        torch.cuda.is_available = real_is_available
    check(effective_use_amp(amp_cfg) is True,
          "AMP must stay enabled when CUDA is available")
    amp_cfg.training.use_amp = False
    check(effective_use_amp(amp_cfg) is False,
          "explicit use_amp: false must stay off")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())