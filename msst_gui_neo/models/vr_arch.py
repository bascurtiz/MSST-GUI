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

# VR 5.1 models use the CascadedNet network from nets_new.
VR_51_ARCH_SIZES = (56817, 218409)

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
                X_wave[d] = librosa.resample(
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

    def _spec_to_wav(self, spec):
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
