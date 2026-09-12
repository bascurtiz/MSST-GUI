"""Regression: max_fft must produce a waveform (not crash on absmax).

NumPy's np.ogrid[...] now returns a tuple, so the old absmax/absmin helpers
died on ``indices.insert(...)`` and ensemble.py never wrote the output file.
max_fft now uses the same lambda_max path as max_wave / min_fft.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ensemble as ens  # noqa: E402

FAILURES = []
CHECKS = 0


def check(msg, cond):
    global CHECKS
    CHECKS += 1
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def test_absmax_ogrid_tuple():
    """absmax/absmin must accept current NumPy ogrid (tuple, not list)."""
    a = np.array([
        [[1.0, -2.0], [0.5, 0.1]],
        [[0.5, 3.0], [-4.0, 0.2]],
    ])
    got = ens.absmax(a, axis=0)
    check("absmax picks larger-abs values",
          np.allclose(got, np.array([[1.0, 3.0], [-4.0, 0.2]])))
    got_min = ens.absmin(a, axis=0)
    check("absmin picks smaller-abs values",
          np.allclose(got_min, np.array([[0.5, -2.0], [0.5, 0.1]])))


def test_max_fft_writes_shape():
    rng = np.random.RandomState(0)
    w1 = rng.randn(2, 4096).astype(np.float32)
    w2 = rng.randn(2, 4096).astype(np.float32)
    out = ens.average_waveforms([w1, w2], [1.0, 1.0], "max_fft")
    check("max_fft returns stereo", out.shape[0] == 2)
    check("max_fft keeps length", out.shape[-1] == 4096)
    check("max_fft is finite", np.isfinite(out).all())


def test_min_fft_and_max_wave_still_work():
    rng = np.random.RandomState(1)
    w1 = rng.randn(2, 4096).astype(np.float32)
    w2 = rng.randn(2, 4096).astype(np.float32)
    mn = ens.average_waveforms([w1, w2], [1.0, 1.0], "min_fft")
    mx = ens.average_waveforms([w1, w2], [1.0, 1.0], "max_wave")
    check("min_fft returns stereo", mn.shape == (2, 4096) and np.isfinite(mn).all())
    check("max_wave returns stereo", mx.shape == (2, 4096) and np.isfinite(mx).all())


def main():
    test_absmax_ogrid_tuple()
    test_max_fft_writes_shape()
    test_min_fft_and_max_wave_still_work()
    print(f"{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
