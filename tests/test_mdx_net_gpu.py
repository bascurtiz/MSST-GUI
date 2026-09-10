"""Regression test: MDX-Net ONNX inference must run on the GPU.

Bug history: the bundled runtime's onnxruntime-gpu was resolved to 1.29.0,
which is built against CUDA 13 (its provider links cublas64_13.dll). The
runtime's torch is CUDA 12.8 and bundles only cublas64_12.dll, so the CUDA
provider DLL failed to initialise and onnxruntime silently fell back to CPU —
every MDX-Net model ran on CPU (~2x realtime, GPU idle) while the app claimed
CUDA was in use.

The fix (runtime_setup.py) pins onnxruntime-gpu to the CUDA 12.8 builds
(>=1.21,<1.27) and (models/mdx_net.py) preloads torch's CUDA libs, requests
CUDAExecutionProvider when available, and runs the STFT/ISTFT on the same
device as the session so the GPU stays busy.

This test runs without a real onnxruntime: the ort module is faked so the
provider-selection logic and the STFT/ISTFT device routing can be verified
offline (torch only).
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from ml_collections import ConfigDict

import models.mdx_net as mdx_net
from models.mdx_net import MDXNetModel

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


class _FakeSession:
    """Identity passthrough: returns the fed spectrogram unchanged."""

    def __init__(self, path, providers=None):
        self._providers = list(providers) if providers else []

    def get_providers(self):
        return self._providers

    def run(self, out_names, feed):
        return [feed["input"]]


class _FakeOrt:
    def __init__(self, avail):
        self._avail = list(avail)
        self.last_providers = None
        self.preload_called = False

    def get_available_providers(self):
        return list(self._avail)

    def preload_dlls(self, *a, **k):
        self.preload_called = True

    def InferenceSession(self, path, providers=None, **kw):
        self.last_providers = providers
        return _FakeSession(path, providers)


def _cfg():
    return ConfigDict({
        "model": ConfigDict({
            "dim_f": 64, "dim_t": 8, "n_fft": 128, "hop_length": 16,
            "compensation": 1.0, "primary_stem": "vocals",
        }),
        "training": ConfigDict({"instruments": ["vocals", "instrumental"]}),
        "inference": ConfigDict({"num_overlap": 2, "batch_size": 1}),
    })


def main():
    torch.manual_seed(0)

    # ── 1. Provider selection: CUDA requested only when available ────────
    fake = _FakeOrt(avail=["CPUExecutionProvider"])
    orig_ort = mdx_net.ort
    try:
        mdx_net.ort = fake
        model = MDXNetModel(_cfg(), __file__)  # file exists, not a real onnx
        check("preload_dlls called (1.21+ API)",
              fake.preload_called)
        check("CPU-only runtime requests CPU provider",
              fake.last_providers == ["CPUExecutionProvider"])

        fake2 = _FakeOrt(avail=["CUDAExecutionProvider", "CPUExecutionProvider"])
        mdx_net.ort = fake2
        model2 = MDXNetModel(_cfg(), __file__)
        check("CUDA runtime requests CUDA provider first",
              fake2.last_providers and fake2.last_providers[0] == "CUDAExecutionProvider")
    finally:
        mdx_net.ort = orig_ort

    # ── 2. STFT/ISTFT run on the requested device (GPU when available) ───
    # device "cuda:0" when CUDA is present, else CPU — the test asserts the
    # transforms followed the device either way.
    want_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if want_gpu else "cpu")

    fake3 = _FakeOrt(avail=["CUDAExecutionProvider", "CPUExecutionProvider"] if want_gpu
                     else ["CPUExecutionProvider"])
    mdx_net.ort = fake3
    try:
        model = MDXNetModel(_cfg(), __file__)
    finally:
        mdx_net.ort = orig_ort

    # Spy at the class level: Python looks up the __call__ dunder on the
    # type, not the instance, so an instance-level assignment would be
    # silently bypassed.
    seen = []
    stft_cls = mdx_net._Stft
    orig_call = stft_cls.__call__
    orig_inv = stft_cls.inverse

    def spy_call(self, x):
        seen.append(("stft", x.device.type))
        return orig_call(self, x)

    def spy_inv(self, x):
        seen.append(("istft", x.device.type))
        return orig_inv(self, x)

    stft_cls.__call__ = spy_call
    stft_cls.inverse = spy_inv
    try:
        chunks = np.random.randn(2, 2, 112).astype(np.float32)
        out = model._process_batch(chunks, device)
    finally:
        stft_cls.__call__ = orig_call
        stft_cls.inverse = orig_inv
    check("process_batch output shape [B, C, chunk]",
          out.shape == chunks.shape)
    devs = {d for _, d in seen}
    expected = "cuda" if want_gpu else "cpu"
    check(f"STFT/ISTFT ran on {expected} (saw {sorted(devs)})",
          devs == {expected} and len(seen) == 2)

    # ── 3. End-to-end demix returns primary + secondary stems ───────────
    mix = np.random.randn(2, 16 * 64).astype(np.float32)  # 1024 samples
    res = model.demix(mix, device, pbar=False)
    check("demix returns both stems with input shape",
          set(res.keys()) == {"vocals", "instrumental"}
          and res["vocals"].shape == mix.shape
          and res["instrumental"].shape == mix.shape)
    # Identity passthrough => primary should reconstruct the mix (the UVR
    # first-3-bin zeroing and overlap-add leave edge residue, so assert on
    # correlation — a broken pipeline would return uncorrelated garbage).
    corr = np.corrcoef(res["vocals"].ravel(), mix.ravel())[0, 1]
    check(f"identity session reconstructs mix (corr={corr:.3f})",
          corr > 0.95)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()