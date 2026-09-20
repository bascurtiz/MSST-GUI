"""GPU label parsing and PolarFormer / BS-Conformer VRAM warnings."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.gpu_utils import (  # noqa: E402
    conformer_vram_warning,
    parse_gpu_memory_gb,
    parse_smi_memory_rows,
    selected_gpu_memory_gb,
    selected_gpu_snapshot,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    check(parse_gpu_memory_gb("GPU 0: NVIDIA GeForce RTX 5090 (32 GB)") == 32.0,
          "parse 32 GB label")
    check(parse_gpu_memory_gb("CPU") is None, "CPU has no VRAM")
    labels = [
        "CPU",
        "GPU 0: NVIDIA GeForce RTX 5090 (32 GB)",
        "GPU 1: NVIDIA RTX A6000 (48 GB)",
    ]
    check(selected_gpu_memory_gb([0], labels) == 32.0, "selected GPU 0")
    check(selected_gpu_memory_gb([0, 1], labels) == 32.0,
          "min VRAM across selected GPUs")

    oom = conformer_vram_warning(
        model_type="bs_conformer", batch_size=2,
        use_torch_checkpoint=False, gpu_mem_gb=32)
    check(oom and "CUDA OOM" in oom, "warn PolarFormer batch 2 no checkpoint")

    check(conformer_vram_warning(
        model_type="bs_conformer", batch_size=1,
        use_torch_checkpoint=True, gpu_mem_gb=32) is None,
          "silent when batch 1 + checkpoint on 32 GB")
    check(conformer_vram_warning(
        model_type="bs_roformer", batch_size=2,
        use_torch_checkpoint=False, gpu_mem_gb=32) is None,
          "other architectures are not this warning")
    check(conformer_vram_warning(
        model_type="bs_conformer", batch_size=2,
        use_torch_checkpoint=False, gpu_mem_gb=48) is None,
          "silent on 48 GB")
    check(conformer_vram_warning(
        model_type="mel_band_conformer", batch_size=2,
        use_torch_checkpoint=True, gpu_mem_gb=24) and "batch size 2" in
          conformer_vram_warning(
              model_type="mel_band_conformer", batch_size=2,
              use_torch_checkpoint=True, gpu_mem_gb=24),
          "warn batch 2 even with checkpoint on small GPU")

    rows = parse_smi_memory_rows(
        "0, NVIDIA GeForce RTX 5090, 32607, 1200, 31407\n"
        "1, NVIDIA GeForce RTX 4090, 24576, 8000, 16576\n")
    check(len(rows) == 2 and rows[0]["index"] == 0, "parse two smi memory rows")
    check(abs(rows[0]["total_gb"] - 31.84) < 0.1, "5090 total ~32 GB")
    check(rows[0]["free_gb"] > rows[1]["free_gb"], "5090 has more free")
    snap = selected_gpu_snapshot([0, 1], rows)
    check(snap and snap["index"] == 1, "snapshot picks the tighter GPU")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_gpu_utils():
    assert main() == 0
