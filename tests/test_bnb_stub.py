"""Regression: checkpoints referencing bitsandbytes classes must load.

Several published checkpoints (scnet_huge_4stem1.2_aname, ..._bleedless,
..._fullness, ..._str_fullness, ...) are full trainer saves whose optimizer
state references ``bitsandbytes.optim.adamw.AdamW8bit`` and friends. The real
package is part of the runtime (adamw8bit training), but the stub in
utils/bnb_stub.py must let torch.load unroll that object graph wherever it is
absent so the plain ``model_state_dict`` weights reach load_state_dict.
"""
import io
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import utils.bnb_stub  # noqa: E402  (installs the finder on import)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    # Runs in both worlds: dev environments now ship the real bitsandbytes
    # (requirements_gui.txt), while a runtime without it exercises the stub.
    # The stub must not shadow a real install, and must answer arbitrary
    # submodule imports when it is the active provider.
    import bitsandbytes.optim.adamw  # noqa: F401
    AdamW8bit = bitsandbytes.optim.adamw.AdamW8bit
    check(callable(AdamW8bit), "AdamW8bit must resolve (real package or stub)")

    # A minimal parameter set keeps the real bitsandbytes happy (its
    # constructor requires `params`); the stub's dummy accepts anything.
    sample_params = [torch.nn.Parameter(torch.randn(2))]

    # Round-trip a pickled instance through the module/class.
    buf = io.BytesIO()
    pickle.dump(AdamW8bit(sample_params), buf)
    buf.seek(0)
    obj = pickle.load(buf)
    check(obj is not None, "pickled optimizer instance must reconstruct")

    # torch.load of a trainer-style checkpoint: optimizer state carrying a
    # bnb instance + plain tensor state dict.
    ck = {
        "epoch": 12,
        "optimizer_name": "adamw8bit",
        "optimizer_state_dict": {"state": {0: AdamW8bit(sample_params)}},
        "scheduler_state_dict": {},
        "best_metric": 1.0,
        "model_state_dict": {"conv.weight": torch.randn(3, 3)},
    }
    buf = io.BytesIO()
    torch.save(ck, buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location="cpu", weights_only=False)
    check("model_state_dict" in loaded, "checkpoint must reload with weights")
    check("conv.weight" in loaded["model_state_dict"],
          "state dict tensors must survive the reload")

    # Idempotence: repeated installs must not stack finders — one finder in
    # the stub world, zero when the real package is present (install() bails).
    from utils import bnb_stub
    before = [f for f in sys.meta_path if isinstance(f, bnb_stub._BnbFinder)]
    bnb_stub.install()
    after = [f for f in sys.meta_path if isinstance(f, bnb_stub._BnbFinder)]
    check(len(after) == len(before), "install() must be idempotent")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())