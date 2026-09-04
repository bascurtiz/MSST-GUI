"""Regression: VR band resampling must not require the `samplerate` package.

The 32000 Hz UVR configs (10/11_sp-uvr-2b-32000, mgm-v5-karokee-32000-*)
use `sinc_*` res_types, which need the optional `samplerate` backend. On
runtimes without it, inference crashed with ModuleNotFoundError in BOTH the
band downsampling (models/vr_arch) and the upsampling legs inside
cmb_spectrogram_to_wave (models/vr/spec_utils). The shared fallback in
spec_utils.resample_audio switches to the scipy-backed polyphase resampler.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models import vr_arch  # noqa: E402
from models.vr import spec_utils  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    x = np.random.randn(2, 44100).astype(np.float32)

    saved = spec_utils._samplerate_available
    try:
        # Force the "samplerate missing" state regardless of the dev machine.
        spec_utils._samplerate_available = False

        # Downsampling leg (band 32000 -> 16000, as in the failing configs).
        y = spec_utils.resample_audio(x, 44100, 32000, "sinc_medium")
        check(y.shape == (2, 32000),
              f"sinc_medium fallback shape {y.shape} != (2, 32000)")
        check(np.isfinite(y).all(), "fallback output must be finite")

        y2 = spec_utils.resample_audio(x, 44100, 16000, "sinc_fastest")
        check(y2.shape == (2, 16000),
              f"sinc_fastest fallback shape {y2.shape} != (2, 16000)")

        # Non-sinc types are untouched.
        y3 = spec_utils.resample_audio(x, 44100, 11025, "polyphase")
        check(y3.shape == (2, 11025), "polyphase path must still work")

        # vr_arch._resample_band delegates to the same fallback (this is
        # what models/vr_arch.py calls for the band downsampling).
        y4 = vr_arch._resample_band(x, 44100, 32000, "sinc_medium")
        check(y4.shape == (2, 32000),
              f"vr_arch._resample_band fallback shape {y4.shape} != (2, 32000)")
    finally:
        spec_utils._samplerate_available = saved

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())