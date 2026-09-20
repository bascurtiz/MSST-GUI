"""VRAM advisor heuristics: GPU + architecture → batch / accum / checkpoint."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.train_advisor import (  # noqa: E402
    advise, family_of, supports_torch_checkpoint,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    check(family_of("bs_conformer") == "conformer", "PolarFormer family")
    check(family_of("bs_roformer") == "roformer", "RoFormer family")
    check(family_of("mdx23c") == "mdx", "MDX family")
    check(supports_torch_checkpoint("bs_conformer"), "conformer has checkpoint")
    check(not supports_torch_checkpoint("mdx23c"), "mdx has no checkpoint flag")

    polar = advise(
        model_type="bs_conformer", gpu_mem_gb=32, gpu_label="GPU 0: RTX 5090 (32 GB)",
        chunk_size=588800, sample_rate=44100, current_batch=2, current_accum=1)
    check(polar.batch_size == 1, f"32 GB PolarFormer batch 1, got {polar.batch_size}")
    check(polar.grad_accum == 2, f"keep effective batch 2, got accum {polar.grad_accum}")
    check(polar.use_torch_checkpoint is True, "checkpoint on for 32 GB PolarFormer")
    check(polar.pin_memory is True, "pin memory on GPU")
    check(polar.optimizer is None, "do not swap Prodigy on 32 GB")
    check(polar.risk == "ok", f"32 GB PolarFormer risk ok, got {polar.risk}")
    check(any("45 GB" in n for n in polar.notes), "mentions the measured OOM")
    check(polar.chunk_size == 588800, "advice carries the YAML chunk")

    from backend.train_advisor import suggest_chunk_size
    check(suggest_chunk_size(588800, oom=True) == 294400,
          "OOM halves a long PolarFormer window")
    check(suggest_chunk_size(588800, oom=False) == 588800,
          "a fitting step keeps the YAML chunk")

    small = advise(
        model_type="bs_conformer", gpu_mem_gb=12, gpu_label="GPU 0: (12 GB)",
        chunk_size=588800, current_batch=2, current_accum=1)
    check(small.batch_size == 1 and small.use_torch_checkpoint is True,
          "12 GB still batch 1 + checkpoint")
    check(small.optimizer == "adamw8bit", "8-bit optimizer on tiny GPU")
    check(small.risk == "tight", "12 GB is tight")

    fat = advise(
        model_type="bs_conformer", gpu_mem_gb=80, gpu_label="GPU 0: (80 GB)",
        chunk_size=588800, current_batch=2, current_accum=1)
    check(fat.batch_size == 4, f"80 GB can batch 4, got {fat.batch_size}")
    check(fat.use_torch_checkpoint is False, "checkpoint off on 80 GB for speed")
    check(fat.grad_accum == 1, "accum 1 when batch already >= effective 2")

    mdx = advise(
        model_type="mdx23c", gpu_mem_gb=32, gpu_label="GPU 0: (32 GB)",
        current_batch=6, current_accum=1)
    check(mdx.batch_size == 8, f"MDX on 32 GB batch 8, got {mdx.batch_size}")
    check(mdx.use_torch_checkpoint is None, "MDX has no checkpoint knob")
    check(mdx.grad_accum == 1, "MDX accum stays 1 when batch >= current effective")

    cpu = advise(model_type="bs_roformer", gpu_mem_gb=None, force_cpu=True)
    check(cpu.risk == "cpu" and cpu.batch_size == 1 and cpu.pin_memory is False,
          "CPU is flagged and not practical")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_train_advisor():
    assert main() == 0
