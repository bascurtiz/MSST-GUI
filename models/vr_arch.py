"""VR (UVR5) architecture for MSST-style inference.

Wraps the band-split VR networks (``models/vr``) in a single ``nn.Module``
that consumes raw audio — ``(batch, channels, samples)`` — and produces the
separated stems ``(batch, 2, channels, samples)`` where index 0 is the
model's primary stem and index 1 the complement, matching the instrument
order in the model config.

The wrapper registers the network's parameters directly on itself (no
``model_run.`` prefix) so the ``state_dict`` keys match the raw UVR
checkpoints and ``load_state_dict`` / ``load_start_checkpoint`` work as-is.
"""
import math

import librosa
import numpy as np
import torch
import torch.nn as nn

from models.vr import nets, nets_new, spec_utils
from models.vr import v6_spec_utils

# VR 5.1 models use the CascadedNet network from nets_new.
VR_51_ARCH_SIZES = (56817, 218409)


def _resample_band(wave, orig_sr, target_sr, res_type):
    """Resample a band, tolerating a runtime without the `samplerate` backend.

    Delegates to spec_utils.resample_audio, which falls back to the
    scipy-backed `polyphase` resampler when a `sinc_*` res_type can't find
    the optional `samplerate` package (shared with the upsampling legs in
    cmb_spectrogram_to_wave, so the whole VR pipeline survives a missing
    backend).
    """
    return spec_utils.resample_audio(
        wave, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type
    )

# Non-accompaniment stems get the aggression value inverted, matching
# audio-separator's CommonSeparator.NON_ACCOM_STEMS.
NON_ACCOM_STEMS = (
    "Vocals", "Other", "Bass", "Drums", "Guitar", "Piano",
    "Synthesizer", "Strings", "Woodwinds", "Brass", "Wind Inst",
)


class VRNet(nn.Module):
    """End-to-end UVR5 VR source separator."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.model

        # VR6 models (tsurumeso vocal-remover `v6` branch): a CascadedNet
        # variant with a per-model complex mode whose `out` layer maps
        # nin*2 mask channels (real: 2 masks; complex: y+v complex masks).
        # Their configs carry no `model.nn_arch_size`, so build them off the
        # `is_vr6` flag before anything that requires it.
        self._is_v6 = bool(model_cfg.get("is_vr6", False))
        self._v6_complex = bool(model_cfg.get("complex", False))

        if self._is_v6:
            from models.vr.v6_nets import CascadedNet as V6CascadedNet
            self._hop_length = int(model_cfg["hop_length"])
            net = V6CascadedNet(
                int(model_cfg["n_fft"]),
                self._hop_length,
                nout=32,
                nout_lstm=128,
                is_complex=self._v6_complex,
            )
            self.nn_arch_size = 0
            self.is_vr_51 = False
            self._nets = [net]
            for name, child in net.named_children():
                self.add_module(name, child)
            self._offset = net.offset
            mp = dict(model_cfg.model_params)
            mp["band"] = {int(k): dict(v) for k, v in mp["band"].items()}
            self.model_params = mp
        else:
            self.nn_arch_size = int(model_cfg.nn_arch_size)
            self.is_vr_51 = bool(model_cfg.is_vr5) or self.nn_arch_size in VR_51_ARCH_SIZES
            nout = model_cfg.get("nout", None)
            nout_lstm = model_cfg.get("nout_lstm", None)

            # Band-split model parameters (already merged from the YAML).
            mp = dict(model_cfg.model_params)
            for k in ("mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"):
                if k not in mp:
                    mp[k] = False
            mp["band"] = {int(k): dict(v) for k, v in mp["band"].items()}
            for band in mp["band"].values():
                if "convert_channels" not in band:
                    band["convert_channels"] = None
            if "aggr_correction" not in mp:
                mp["aggr_correction"] = None
            self.model_params = mp

            if self.is_vr_51:
                net = nets_new.CascadedNet(
                    self.model_params["bins"] * 2,
                    self.nn_arch_size,
                    nout=nout or 32,
                    nout_lstm=nout_lstm or 128,
                )
            else:
                net = nets.determine_model_capacity(
                    self.model_params["bins"] * 2, self.nn_arch_size
                )

            # Flatten the network's children onto this module so state_dict keys
            # match the raw checkpoint (no extra prefix).
            self._nets = [net]
            for name, child in net.named_children():
                self.add_module(name, child)

            self._offset = net.offset

        inf = config.inference
        self.window_size = int(getattr(inf, "window_size", 512))
        self.batch_size = max(1, int(getattr(inf, "batch_size", 1)))
        self.enable_tta = bool(getattr(inf, "enable_tta", False))
        self.enable_post_process = bool(getattr(inf, "enable_post_process", False))
        self.post_process_threshold = float(getattr(inf, "post_process_threshold", 0.2))
        self.high_end_process = bool(getattr(inf, "high_end_process", False))
        self.aggression = float(int(getattr(inf, "aggression", 5)) / 100)

        instruments = list(getattr(config.training, "instruments", []) or [])
        self.primary_stem = instruments[0] if instruments else "Vocals"

        self.input_high_end_h = None
        self.input_high_end = None

        self.eval()

    # ── module plumbing ────────────────────────────────────────────────────

    def train(self, mode=True):
        super().train(mode)
        for net in self._nets:
            net.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        """Separate a batch of raw audio.

        Args:
            x: (batch, channels, samples) float32 tensor on the model device.

        Returns:
            (batch, 2, channels, samples) — [primary, secondary] stems.
        """
        B, C, N = x.shape
        x_np = x.detach().float().cpu().numpy()
        outs = []
        for b in range(B):
            audio = x_np[b]
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            if audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)
            y, v = self._separate(audio.astype(np.float32, copy=False))
            outs.append(np.stack([y, v], axis=0))
        out = np.stack(outs, axis=0)
        return torch.from_numpy(out).to(x.device)

    # ── VR pipeline ────────────────────────────────────────────────────────

    def _separate(self, audio):
        """Separate one stereo track; returns (primary, secondary) waveforms
        of exactly the input length."""
        if self._is_v6:
            return self._separate_v6(audio)
        mp = self.model_params
        bands_n = len(mp["band"])

        X_wave = {}
        X_spec_s = {}
        for d in range(bands_n, 0, -1):
            bp = mp["band"][d]
            if d == bands_n:  # high-end band (full sample rate)
                X_wave[d] = audio
                X_spec_s[d] = spec_utils.wave_to_spectrogram(
                    X_wave[d], bp["hl"], bp["n_fft"], mp, band=d, is_v51_model=self.is_vr_51
                )
            else:  # lower bands: resample down from the band above
                X_wave[d] = _resample_band(
                    X_wave[d + 1],
                    orig_sr=mp["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )
                X_spec_s[d] = spec_utils.wave_to_spectrogram(
                    X_wave[d], bp["hl"], bp["n_fft"], mp, band=d, is_v51_model=self.is_vr_51
                )

            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (
                    (bp["n_fft"] // 2 - bp["crop_stop"])
                    + (mp["pre_filter_stop"] - mp["pre_filter_start"])
                )
                self.input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, mp, is_v51_model=self.is_vr_51)

        y_spec, v_spec = self._inference(X_spec)

        y = self._spec_to_wav(y_spec)
        v = self._spec_to_wav(v_spec)

        return self._fit_length(y, audio.shape[-1]), self._fit_length(v, audio.shape[-1])

    def _separate_v6(self, audio):
        """VR6 pipeline (tsurumeso v6 branch): one full-band STFT, windowed
        mask prediction, then the two output masks are decoded per model mode."""
        X_spec = v6_spec_utils.wave_to_spectrogram(
            audio, self._hop_length, self.model_params["n_fft"]
        )
        y_spec, v_spec = self._inference_v6(X_spec)
        y = self._spec_to_wav(y_spec)
        v = self._spec_to_wav(v_spec)
        return self._fit_length(y, audio.shape[-1]), self._fit_length(v, audio.shape[-1])

    def _inference_v6(self, X_spec):
        net = self._nets[0]
        offset = net.offset
        window = self.window_size
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, window, offset)

        X_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        X_pad /= np.abs(X_spec).max()
        mask = self._execute_v6(X_pad, roi_size)
        mask = mask[:, :, :n_frame]

        if self.enable_tta:
            shift = roi_size // 2
            X_pad2 = np.pad(X_spec, ((0, 0), (0, 0), (pad_l + shift, pad_r + shift)),
                            mode="constant")
            X_pad2 /= np.abs(X_spec).max()
            mask_tta = self._execute_v6(X_pad2, roi_size)[:, :, shift:]
            mask = (mask + mask_tta[:, :, :n_frame]) * 0.5

        if self._v6_complex:
            y_spec = X_spec * mask[:2]
            v_spec = X_spec * mask[2:]
        else:
            X_mag = np.abs(X_spec)
            X_phase = np.exp(1.0j * np.angle(X_spec))
            y_spec = X_mag * mask[:2] * X_phase
            v_spec = X_mag * mask[2:] * X_phase
        return y_spec, v_spec

    def _execute_v6(self, X_pad, roi_size):
        net = self._nets[0]
        offset = net.offset
        window = self.window_size
        patches = (X_pad.shape[2] - 2 * offset) // roi_size
        if patches < 1:
            raise ValueError(
                f"Audio chunk too short for VR windowing (need >= {window} frames, "
                f"got {X_pad.shape[2]}); use a longer track or larger chunk size."
            )
        device = next(self.parameters()).device
        masks = []
        net.eval()
        # fp32 only: the replicate-pad inside the net is not implemented for
        # ComplexHalf (fp16 autocast would produce it on complex models).
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for i in range(patches):
                start = i * roi_size
                X_crop = X_pad[:, :, start:start + window]
                X_batch = torch.from_numpy(np.ascontiguousarray(X_crop)[None]).to(device)
                if self._v6_complex:
                    # keep the complex dtype explicitly — `.float()` on a
                    # complex tensor would silently drop the imaginary part
                    X_batch = X_batch.to(torch.complex64)
                else:
                    X_batch = torch.abs(X_batch)
                pred = net.predict_mask(X_batch)
                masks.append(pred[0].detach().cpu().numpy())
        return np.concatenate(masks, axis=2)

    def _spec_to_wav(self, spec):
        if self._is_v6:
            return v6_spec_utils.spectrogram_to_wave(spec, hop_length=self._hop_length)
        mp = self.model_params
        if self.high_end_process and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:
            input_high_end_ = spec_utils.mirroring("mirroring", spec, self.input_high_end, mp)
            wav = spec_utils.cmb_spectrogram_to_wave(
                spec, mp, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51
            )
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, mp, is_v51_model=self.is_vr_51)
        return wav

    def _inference(self, X_spec):
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self._offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        X_mag_pad /= X_mag_pad.max()
        mask = self._execute(X_mag_pad, roi_size)

        if self.enable_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
            X_mag_pad /= X_mag_pad.max()
            mask_tta = self._execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        mask = self._postprocess(mask)

        y_spec = mask * X_mag * np.exp(1.0j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)

        return y_spec, v_spec

    def _execute(self, X_mag_pad, roi_size):
        net = self._nets[0]
        patches = (X_mag_pad.shape[2] - 2 * self._offset) // roi_size
        if patches < 1:
            raise ValueError(
                f"Audio chunk too short for VR windowing (need >= {self.window_size} frames, "
                f"got {X_mag_pad.shape[2]}); use a longer track or larger chunk size."
            )

        device = next(self.parameters()).device
        masks = []
        net.eval()
        with torch.no_grad():
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_batch = torch.from_numpy(np.ascontiguousarray(X_mag_window)[None]).to(device)
                pred = net.predict_mask(X_batch)
                pred = pred[0].detach().cpu().numpy()
                masks.append(pred)
        return np.concatenate(masks, axis=2)

    def _postprocess(self, mask):
        is_non_accom = any(s.lower() == self.primary_stem.lower() for s in NON_ACCOM_STEMS)
        aggressiveness = {
            "value": self.aggression,
            "split_bin": self.model_params["band"][1]["crop_stop"],
            "aggr_correction": self.model_params.get("aggr_correction"),
        }
        mask = spec_utils.adjust_aggr(mask, is_non_accom, aggressiveness)
        if self.enable_post_process:
            mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)
        return mask

    @staticmethod
    def _fit_length(wave, n):
        if wave.ndim == 1:
            wave = np.stack([wave, wave])
        if wave.shape[1] > n:
            wave = wave[:, :n]
        elif wave.shape[1] < n:
            wave = np.pad(wave, ((0, 0), (0, n - wave.shape[1])))
        return wave
