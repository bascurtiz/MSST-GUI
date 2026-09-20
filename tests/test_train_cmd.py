"""Training command construction: LoRA, DDP, wandb, freeze, launchers.

No Qt — exercises backend/train_cmd.py, the same helper the TRAINING tab uses.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.paths import TRAIN_ACCELERATE_SCRIPT, TRAIN_DDP_SCRIPT, TRAIN_SCRIPT  # noqa: E402
from backend.train_cmd import (  # noqa: E402
    DEFAULT_RUN_OPTS,
    build_train_command,
    detect_custom_backend,
    inject_lora_defaults,
    redact_train_cmd,
    resolve_launcher,
    subprocess_env,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def _opts(**kwargs):
    o = dict(DEFAULT_RUN_OPTS)
    o.update(kwargs)
    return o


def _cmd(opts, launcher="standard", **kwargs):
    return build_train_command(
        python_exe="python",
        opts=opts,
        model_type="bs_roformer",
        config_path="cfg.yaml",
        results_path="results",
        data_paths=["data"],
        valid_paths=["valid"],
        dataset_type="1",
        metrics=["sdr"],
        metric_for_scheduler="sdr",
        losses=["masked_loss"],
        checkpoint=kwargs.get("checkpoint", ""),
        custom_backend=kwargs.get("custom_backend", ""),
        use_standard_loss=kwargs.get("use_standard_loss", False),
        launcher=launcher,
    )


def main():
    std = _cmd(_opts(), "standard")
    check(TRAIN_SCRIPT in std, "standard launcher uses train.py")
    check(TRAIN_DDP_SCRIPT not in std, "standard does not use train_ddp.py")

    peft = _cmd(_opts(lora_mode="peft", lora_checkpoint="adapter.ckpt"), "standard")
    check("--train_lora_peft" in peft, "PEFT LoRA flag")
    check("--lora_checkpoint_peft" in peft and "adapter.ckpt" in peft,
          "PEFT LoRA checkpoint")

    ll = _cmd(_opts(lora_mode="loralib", lora_checkpoint="a.bin"), "standard")
    check("--train_lora_loralib" in ll and "--lora_checkpoint_loralib" in ll,
          "loralib LoRA flags")

    fr = _cmd(_opts(freeze_layers="layer1 attn.q"), "standard")
    i = fr.index("--freeze_layers")
    check(fr[i + 1] == "layer1" and fr[i + 2] == "attn.q", "freeze prefixes split")

    wb = _cmd(_opts(wandb_key="secret", wandb_offline=True), "standard")
    check("--wandb_key" not in wb and "secret" not in wb,
          "wandb key stays off the command line")
    check("--wandb_offline" in wb, "wandb offline")
    wb_env = subprocess_env(_opts(wandb_key="secret"), "standard")
    check(wb_env == {"WANDB_API_KEY": "secret"},
          "wandb key is injected as WANDB_API_KEY")
    redacted = redact_train_cmd(["python", "train.py", "--wandb_key", "secret"])
    check(redacted[-1] == "<redacted>" and "secret" not in redacted,
          "redact_train_cmd hides --wandb_key values")

    extra = _cmd(_opts(safe_mode=True, persistent_workers=True,
                      load_all_metrics=True, load_all_losses=True), "standard")
    for flag in ("--safe_mode", "--persistent_workers",
                 "--load_all_metrics", "--load_all_losses"):
        check(flag in extra, f"optional flag {flag}")

    ddp_opts = _opts(device_ids=[1, 3], launcher="ddp")
    launcher, warn = resolve_launcher(ddp_opts)
    check(launcher == "ddp" and warn is None, "DDP allowed with 2 GPUs")
    ddp = _cmd(ddp_opts, "ddp")
    check(TRAIN_DDP_SCRIPT in ddp, "DDP launcher uses train_ddp.py")
    # After CUDA_VISIBLE_DEVICES remap, the job sees 0,1 not 1,3.
    ids_at = ddp.index("--device_ids")
    check(ddp[ids_at + 1] == "0" and ddp[ids_at + 2] == "1",
          "DDP remaps device ids to 0..n-1")
    env = subprocess_env(ddp_opts, "ddp")
    check(env == {"CUDA_VISIBLE_DEVICES": "1,3"}, "DDP CUDA_VISIBLE_DEVICES")

    one = _opts(device_ids=[0], launcher="ddp")
    launcher, warn = resolve_launcher(one)
    check(launcher == "standard" and warn, "DDP falls back with one GPU")

    acc_blocked = resolve_launcher(
        _opts(launcher="accelerate", lora_mode="peft"),
        accelerate_ok=True)
    check(acc_blocked[0] == "standard" and acc_blocked[1],
          "Accelerate blocked when LoRA is on")

    acc_ok, acc_warn = resolve_launcher(
        _opts(launcher="accelerate", device_ids=[0, 1]),
        accelerate_ok=True)
    check(acc_ok == "accelerate" and acc_warn is None, "Accelerate allowed")
    acc = _cmd(_opts(device_ids=[0, 1]), "accelerate")
    check("-m" in acc and "accelerate.commands.launch" in acc,
          "Accelerate uses python -m accelerate")
    check(TRAIN_ACCELERATE_SCRIPT in acc, "Accelerate script")
    check("--loss" not in acc, "Accelerate does not pass train.py --loss")

    missing = resolve_launcher(
        _opts(launcher="accelerate"), accelerate_ok=False)
    check(missing[0] == "standard" and missing[1],
          "Accelerate falls back when package missing")

    cfg = {"training": {}}
    inject_lora_defaults(cfg)
    check(cfg["lora"]["r"] == 8 and cfg["lora"]["lora_alpha"] == 16,
          "LoRA defaults injected")
    cfg2 = {"lora": {"r": 4}}
    inject_lora_defaults(cfg2)
    check(cfg2["lora"]["r"] == 4, "existing LoRA block kept")

    check(detect_custom_backend("") == "", "empty config has no backend")
    check(detect_custom_backend("no-such.yaml") == "", "missing file has no backend")

    cpu_env = subprocess_env(_opts(force_cpu=True), "standard")
    check(cpu_env == {"CUDA_VISIBLE_DEVICES": ""}, "CPU-only hides GPUs")
    cpu_wb = subprocess_env(_opts(force_cpu=True, wandb_key="secret"), "standard")
    check(cpu_wb == {"CUDA_VISIBLE_DEVICES": "", "WANDB_API_KEY": "secret"},
          "CPU-only still injects wandb env")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_train_cmd():
    assert main() == 0
