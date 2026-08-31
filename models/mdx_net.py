# coding: utf-8
"""
models/mdx_net.py
-----------------
MDX-Net (kuielab-style) inference backend.

The model zoo ships 45 "MDX-Net Architecture" entries as ONNX checkpoints
(mdx_*.onnx) whose configs use the kuielab layout (model.dim_f / dim_t /
n_fft / hop_length / compensation / primary_stem). These are NOT PyTorch
checkpoints and cannot go through get_model_from_config's torch branches,
so they run through onnxruntime with the UVR-style demix loop: STFT the
mixture, run the ONNX session, inverse-STFT the predicted primary stem,
apply the config's compensation factor, and derive the secondary stem as
mix - primary.

The onnxruntime package is installed into the bundled runtime via
requirements-runtime.txt (runtime_setup.py upgrades it to the CUDA wheel
whenever the runtime's torch is a CUDA build).
"""
import os

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import onnxruntime as ort
except ImportError:  # dev env without onnxruntime — raise a clear error later
    ort = None

from models.mdx23c_tfc_tdf_v3 import STFT as _Stft


class MDXNetModel:
    """Loads an MDX-Net ONNX checkpoint and demixes audio UVR-style.

    The wrapped object intentionally has no `forward`/`.to()` — it is not a
    torch module. The generic chunked demix loop in utils/model_utils.py
    dispatches model_type == 'mdxnet' straight to :meth:`demix`.
    """

    def __init__(self, config, model_path: str):
        if ort is None:
            raise ImportError(
                "onnxruntime is required for MDX-Net models. "
                "Install it with: pip install onnxruntime")
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"MDX-Net ONNX checkpoint not found: {model_path}")
        self.config = config
        self.model_path = model_path

        m = config.model
        self.dim_f = int(m.dim_f)
        self.dim_t = int(m.dim_t)
        self.n_fft = int(m.n_fft)
        self.hop_length = int(m.hop_length)
        self.compensation = float(getattr(m, "compensation", 1.0))
        self.primary_stem = str(getattr(m, "primary_stem", "")).strip().lower()

        # Normalize instrument names (some zoo configs carry trailing spaces)
        inst = [str(s).strip() for s in (config.training.instruments or [])]
        config.training.instruments = inst
        if getattr(config.training, "target_instrument", None):
            config.training.target_instrument = \
                str(config.training.target_instrument).strip()

        # The CUDA provider needs cudart/cublas/cuDNN at provider-init time.
        # onnxruntime-gpu wheels do not bundle them, but the CUDA torch wheel
        # ships matching builds in torch/lib — expose that directory to the
        # Windows DLL loader before the session is created.
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

        providers = ["CUDAExecutionProvider", "DmlExecutionProvider",
                     "CPUExecutionProvider"]
        try:
            avail = ort.get_available_providers()
        except Exception:
            avail = []
        providers = [p for p in providers if p in avail] or None
        self.session = ort.InferenceSession(model_path, providers=providers)
        active = self.session.get_providers()
        print(f"ONNX providers: {', '.join(active)}", flush=True)
        if "CUDAExecutionProvider" not in active \
                and "DmlExecutionProvider" not in active:
            print(
                "WARNING: onnxruntime has no GPU provider active — MDX-Net "
                "models run on CPU (several times slower). Re-run the GPU "
                "setup from Settings to upgrade to onnxruntime-gpu.",
                flush=True)

        self.stft = _Stft(type("_C", (), {
            "n_fft": self.n_fft, "hop_length": self.hop_length,
            "dim_f": self.dim_f})())

    def eval(self):
        return self

    def to(self, device):
        # onnxruntime is not a torch module; device handled by providers
        return self

    # ── UVR-style demix ──────────────────────────────────────────────

    def demix(self, mix, device, pbar: bool = False) -> dict:
        """Separate one mixture.

        Args:
            mix: np.ndarray float32 [C, T] (C == 1 mono or 2 stereo).
            device: torch device (unused by onnxruntime; kept for interface
                parity with the generic demix loop).
            pbar: stream per-chunk tqdm progress (consumed by the GUI).

        Returns:
            {primary_stem: [C, T], secondary_stem: [C, T]} — the model's
            predicted stem and its complement (mix - primary).
        """
        if isinstance(mix, torch.Tensor):
            mix = mix.detach().cpu().numpy()
        mix = np.asarray(mix, dtype=np.float32)
        if mix.ndim == 1:
            mix = mix[None]
        orig_channels = mix.shape[0]

        # The ONNX graph expects a stereo complex spectrogram [1, 4, F, T]
        # (2 channels x real/imag). Duplicate mono internally, restore the
        # original channel count on the way out.
        if orig_channels == 1:
            mix_work = np.repeat(mix, 2, axis=0)
        else:
            mix_work = mix[:2]

        primary = self._demix_stereo(mix_work, pbar=pbar)
        if orig_channels == 1:
            primary = primary[:1]

        secondary = mix[: primary.shape[0]] - primary
        inst = list(getattr(self.config.training, "instruments", []) or [])
        primary_name = self.primary_stem or (inst[0] if inst else "primary")
        secondary_name = next((s for s in inst if s.lower() != primary_name),
                              "secondary")
        return {primary_name: primary, secondary_name: secondary}

    def _demix_stereo(self, mix: np.ndarray, pbar: bool = False) -> np.ndarray:
        """Chunked overlap-add demix on [2, T] float32; returns [2, T'].

        Chunks are grouped into small batches per onnxruntime call: each
        chunk is independent (overlap-add weights them separately), so
        batching only cuts per-call overhead. When `pbar` is set a tqdm bar
        streams chunk progress; the GUI parses those lines to move the
        card's progress bar incrementally during demixing.
        """
        c, total = mix.shape

        chunk_size = self.hop_length * (self.dim_t - 1)
        trim = self.n_fft // 2
        gen_size = chunk_size - 2 * trim
        pad = gen_size + trim - (total % gen_size)
        mixture = np.concatenate(
            (np.zeros((c, trim), np.float32), mix,
             np.zeros((c, pad), np.float32)), axis=1)

        num_overlap = int(getattr(self.config.inference, "num_overlap", 2)) or 2
        step = chunk_size // num_overlap
        window = np.hanning(chunk_size).astype(np.float32)

        # Chunks are small ([C, hop*(dim_t-1)] each); even the config's
        # batch_size=1 would spend most of the run on per-call overhead, so
        # group at least 4 chunks per session.run.
        batch = max(4, int(getattr(self.config.inference, "batch_size", 1) or 1))

        starts = list(range(0, mixture.shape[1] - chunk_size + 1, step))
        result = np.zeros((c, mixture.shape[1]), np.float32)
        divider = np.zeros((c, mixture.shape[1]), np.float32)

        if pbar:
            bar = tqdm(total=len(starts), desc="Processing audio chunks",
                       leave=False)
        else:
            bar = None

        i = 0
        while i < len(starts):
            idx = starts[i:i + batch]
            parts = np.stack([mixture[:, s:s + chunk_size] for s in idx])
            ests = self._process_batch(parts)            # [B, C, chunk]
            for b, s in enumerate(idx):
                result[:, s:s + chunk_size] += ests[b] * window
                divider[:, s:s + chunk_size] += window
            i += len(idx)
            if bar is not None:
                bar.update(len(idx))
        if bar is not None:
            bar.close()

        out = result / np.maximum(divider, 1e-8)
        out = out[:, trim:trim + total]
        out *= self.compensation
        return out

    def _process_batch(self, chunks: np.ndarray) -> np.ndarray:
        """STFT -> ONNX -> ISTFT for a stacked batch [B, C, chunk]."""
        x = torch.from_numpy(chunks).float()                 # [B, C, T]
        spek = self.stft(x)                                  # [B, 2C, F, T']
        # UVR zeroes the first 3 frequency bins before inference
        spek[:, :, :3, :] *= 0
        inp = spek.detach().cpu().numpy()
        pred = self.session.run(None, {"input": inp})[0]     # [B, 2C, F, T']
        out = self.stft.inverse(torch.from_numpy(pred).float())  # [B, C, T]
        return out.numpy()
