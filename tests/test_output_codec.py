"""Regression: every stem must honor the configured output format.

The old per-stem fallback wrote stems peaking above 1.0 as WAV even when
FLAC (PCM_16/24) was selected, so a single run produced mixed .wav/.flac
folders. The chosen PCM type must decide the format for ALL stems: integer
types always yield FLAC (hot stems get peak-normalized so the integer
codec doesn't clip), FLOAT always yields 32-bit WAV.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    # WAV (32-bit float) — always wav, never normalized.
    check(inference.output_codec("FLOAT", 0.5) == ("wav", None),
          "FLOAT + quiet stem -> wav, no scale")
    check(inference.output_codec("FLOAT", 2.5) == ("wav", None),
          "FLOAT + hot stem -> still wav, no scale")

    # FLAC 16/24-bit — always flac; hot stems get a scale-down factor.
    check(inference.output_codec("PCM_16", 0.5) == ("flac", None),
          "PCM_16 + quiet stem -> flac, no scale")
    check(inference.output_codec("PCM_24", 1.0) == ("flac", None),
          "PCM_24 + peak exactly 1.0 -> flac, no scale")
    codec, scale = inference.output_codec("PCM_16", 1.4)
    check(codec == "flac" and scale is not None and abs(scale - 1.0 / 1.4) < 1e-9,
          "PCM_16 + hot stem -> flac with 1/peak scale")
    codec, scale = inference.output_codec("PCM_24", 4.0)
    check(codec == "flac" and scale == 0.25,
          "PCM_24 + hot stem -> flac with 1/peak scale")

    # Never a zero-division: a fully silent stem (peak 0) is not "hot".
    check(inference.output_codec("PCM_16", 0.0) == ("flac", None),
          "silent stem must not produce a scale")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())