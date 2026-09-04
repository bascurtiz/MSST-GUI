"""Regression: --train_lora_peft lifecycle (peft) must work end to end.

train.py:355 wraps the model with get_lora() (get_peft_model), training steps
then move only the LoRA adapters (base frozen), saves go through
model.save_pretrained(), and valid.py:868 / the --lora_checkpoint_peft
continuation reload them with PeftModel.from_pretrained. This test exercises
exactly those engine code paths on a tiny CPU MLP so the suite needs no GPU
and no real architecture: the peft mechanics are architecture-agnostic.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import peft  # noqa: F401
    from peft import PeftModel, LoraConfig
    import torch.nn as nn
    import torch
except ImportError:
    # peft is now part of requirements-runtime / requirements_gui; on an env
    # that predates it there is nothing meaningful to regression-test.
    print("SKIP: peft not installed")
    sys.exit(0)

from ml_collections import ConfigDict

from utils.model_utils import get_lora
from utils.settings import parse_args_train

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    torch.manual_seed(0)
    model = TinyNet()
    sd_base = {k: v.detach().clone() for k, v in model.state_dict().items()}

    config = ConfigDict()
    config["lora"] = {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["fc1", "fc2"],
    }
    args = parse_args_train({"model_type": "bs_roformer", "results_path": ".",
                             "data_path": []})
    args.freeze_layers = None
    args.train_lora_peft = True
    args.lora_checkpoint_peft = ""

    lora_model = get_lora(args, config, model)
    check(type(lora_model).__name__ == "PeftModel",
          "get_lora(peft) must return a PeftModel")

    trainable = [p for p in lora_model.parameters() if p.requires_grad]
    stray = [n for n, p in lora_model.named_parameters()
             if p.requires_grad and "lora" not in n.lower()]
    check(len(trainable) > 0, "expected trainable LoRA params")
    check(not stray, f"base params must be frozen, got: {stray[:3]}")

    # Real forward/backward/step: loss must decrease, only adapters move.
    x = torch.randn(4, 16)
    target = torch.randn(4, 16)
    optim = torch.optim.AdamW(trainable, lr=1e-2)
    losses = []
    for _ in range(8):
        lora_model.train()
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(lora_model(x), target)
        loss.backward()
        optim.step()
        losses.append(float(loss))
    check(losses[-1] < losses[0], f"loss must decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

    lora_model.eval()
    with torch.no_grad():
        y_trained = lora_model(x)

    # Save via the engine's branch: model.save_pretrained(store_path + '_lora_')
    tmp = tempfile.mkdtemp(prefix="lora_peft_test_")
    adapter_dir = tmp + "_lora_"
    try:
        lora_model.save_pretrained(adapter_dir)
        check(all(os.path.isfile(os.path.join(adapter_dir, f))
                  for f in ("adapter_config.json", "adapter_model.safetensors")),
              "save_pretrained must write adapter_config + adapter weights")

        # valid.py:868 reload onto the same base weights -> bit-identical output.
        model2 = TinyNet()
        model2.load_state_dict(sd_base)
        loaded = PeftModel.from_pretrained(model2, adapter_dir)
        loaded.eval()
        with torch.no_grad():
            y_loaded = loaded(x)
        diff = float((y_trained - y_loaded).abs().max())
        check(diff < 1e-5, f"reloaded adapter diverged from trained (max-abs {diff:.2e})")

        # get_lora continuation branch (--lora_checkpoint_peft).
        model3 = TinyNet()
        model3.load_state_dict(sd_base)
        args3 = parse_args_train({"model_type": "bs_roformer",
                                  "results_path": ".", "data_path": []})
        args3.freeze_layers = None
        args3.train_lora_peft = True
        args3.lora_checkpoint_peft = adapter_dir
        cont = get_lora(args3, config, model3)
        cont.eval()
        with torch.no_grad():
            y_cont = cont(x)
        diff2 = float((y_trained - y_cont).abs().max())
        check(diff2 < 1e-5, f"continuation load diverged (max-abs {diff2:.2e})")

        # Adapters genuinely change output vs the clean base.
        model4 = TinyNet()
        model4.load_state_dict(sd_base)
        model4.eval()
        with torch.no_grad():
            y_base = model4(x)
        moved = float((y_base - y_trained).abs().mean())
        check(moved > 1e-5, "adapters had no effect on output")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
