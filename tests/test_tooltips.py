"""Styled tooltips wrap after each sentence so a hover cannot go ultra-wide."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.theme import wrap_tooltip_sentences  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    wandb = (
        "Injected into the job as WANDB_API_KEY so it does not appear in "
        "the spawned command or train_log.txt. If a key was ever printed "
        "in a log, revoke it at https://wandb.ai/settings and paste a new one."
    )
    out = wrap_tooltip_sentences(wandb)
    lines = out.split("\n")
    check(len(lines) == 2, f"wandb tooltip is two sentences, got {len(lines)}")
    check(lines[0].endswith("train_log.txt."),
          "first line ends at the first sentence")
    check(lines[1].startswith("If a key"),
          "second line is the revoke sentence")
    check("https://wandb.ai/settings" in lines[1],
          "URL stays on its sentence, not split at dots")

    check(wrap_tooltip_sentences("Show Help") == "Show Help",
          "single short tooltip is unchanged")
    check(wrap_tooltip_sentences("Foo.\nBar.") == "Foo.\nBar.",
          "existing sentence newlines stay")
    check(
        wrap_tooltip_sentences("Prefixes, space-separated (e.g. layer1 attn).")
        == "Prefixes, space-separated (e.g. layer1 attn).",
        "e.g. is not a sentence break",
    )

    launcher = (
        "standard: train.py (current). ddp: one process per GPU via "
        "CUDA_VISIBLE_DEVICES. accelerate: older train_accelerate.py "
        "(no LoRA / freeze / custom backends)."
    )
    launch_lines = wrap_tooltip_sentences(launcher).split("\n")
    check(len(launch_lines) == 3, f"launcher tooltip is three sentences, got {len(launch_lines)}")
    check(launch_lines[0].endswith("(current)."), "first launcher sentence")
    check(launch_lines[1].startswith("ddp:"), "ddp stays after the first dot")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
