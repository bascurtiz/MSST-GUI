"""Band-split spectrogram helpers for the UVR5 VR architecture.

Ported from audio-separator's ``uvr_lib_v5.spec_utils`` (MIT), keeping only
what the VR networks need and dropping the ``audio_separator`` imports.
"""
import math

import librosa
import numpy as np
import torch

# `sinc_*` res_types map to the optional `samplerate` (libsamplerate)
# backend, which some runtimes don't ship (the 32000 Hz UVR configs use
# them and died with `ModuleNotFoundError: No module named 'samplerate'`
# mid-inference). Fall back to the scipy-backed `polyphase` resampler that
# the rest of the VR configs already use.
_SINC_RES_TYPES = {"sinc_best", "sinc_medium", "sinc_fast", "sinc_fastest", "sinc_hq"}
_samplerate_available = None


def resample_audio(wave, orig_sr, target_sr, res_type):
    """librosa.resample wrapper that tolerates a missing `samplerate` backend.

    When ``res_type`` is a ``sinc_*`` variant and the optional ``samplerate``
    package isn't importable, resample with the scipy-backed ``polyphase``
    instead of crashing. All VR band resampling (down in ``vr_arch._separate``
    and back up in ``cmb_spectrogram_to_wave``) goes through here.
    """
    global _samplerate_available
    if res_type in _SINC_RES_TYPES:
        if _samplerate_available is None:
            try:
                import samplerate  # noqa: F401
                _samplerate_available = True
            except ImportError:
                _samplerate_available = False
        if not _samplerate_available:
            res_type = "polyphase"
    return librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)


def crop_center(h1, h2):
    """Crop the centre of ``h1`` (time axis) to the width of ``h2``."""
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def preprocess(X_spec):
    """Split a complex spectrogram into magnitude and phase."""
    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)

    return X_mag, X_phase


def make_padding(width, cropsize, offset):
    """Padding needed to make ``width`` divisible by the crop size."""
    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def convert_channels(spec, mp, band):
    """Mid-side / stereo conversions for v5.1 models (in-place channel mix)."""
    cc = mp["band"][band].get("convert_channels")

    if "mid_side_c" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25)
        spec_right = np.subtract(spec[1], spec[0] * 0.25)
    elif "mid_side" == cc:
        spec_left = np.add(spec[0], spec[1]) / 2
        spec_right = np.subtract(spec[0], spec[1])
    elif "stereo_n" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25) / 0.9375
        spec_right = np.add(spec[1], spec[0] * 0.25) / 0.9375
    else:
        return spec

    return np.asfortranarray([spec_left, spec_right])


def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False):
    """STFT of a waveform into a 2-channel complex spectrogram, honouring the
    band's channel / mid-side / reverse settings."""
    if wave.ndim == 1:
        wave = np.asfortranarray([wave, wave])

    if not is_v51_model:
        if mp["reverse"]:
            wave_left = np.flip(np.asfortranarray(wave[0]))
            wave_right = np.flip(np.asfortranarray(wave[1]))
        elif mp["mid_side"]:
            wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mp["mid_side_b2"]:
            wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
        else:
            wave_left = np.asfortranarray(wave[0])
            wave_right = np.asfortranarray(wave[1])
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])

    if is_v51_model:
        spec = convert_channels(spec, mp, band)

    return spec


def combine_spectrograms(specs, mp, is_v51_model=False):
    """Combine per-band spectrograms into a single full-band spectrogram."""
    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp["band"])

    for d in range(1, bands_n + 1):
        h = mp["band"][d]["crop_stop"] - mp["band"][d]["crop_start"]
        spec_c[:, offset:offset + h, :l] = specs[d][:, mp["band"][d]["crop_start"]:mp["band"][d]["crop_stop"], :l]
        offset += h

    if offset > mp["bins"]:
        raise ValueError("Too much bins")

    # lowpass filter
    if mp["pre_filter_start"] > 0:
        if is_v51_model:
            spec_c *= get_lp_filter_mask(spec_c.shape[1], mp["pre_filter_start"], mp["pre_filter_stop"])
        else:
            if bands_n == 1:
                spec_c = fft_lp_filter(spec_c, mp["pre_filter_start"], mp["pre_filter_stop"])
            else:
                gp = 1
                for b in range(mp["pre_filter_start"] + 1, mp["pre_filter_stop"]):
                    g = math.pow(10, -(b - mp["pre_filter_start"]) * (3.5 - gp) / 20.0)
                    gp = g
                    spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


def spectrogram_to_wave(spec, hop_length=1024, mp={}, band=0, is_v51_model=True):
    """Inverse STFT of one band's spectrogram into a waveform."""
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if is_v51_model:
        cc = mp["band"][band].get("convert_channels")
        if "mid_side_c" == cc:
            return np.asfortranarray([np.subtract(wave_left / 1.0625, wave_right / 4.25), np.add(wave_right / 1.0625, wave_left / 4.25)])
        elif "mid_side" == cc:
            return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif "stereo_n" == cc:
            return np.asfortranarray([np.subtract(wave_left, wave_right * 0.25), np.subtract(wave_right, wave_left * 0.25)])
    else:
        if mp["reverse"]:
            return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
        elif mp["mid_side"]:
            return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif mp["mid_side_b2"]:
            return np.asfortranarray([np.add(wave_right / 1.25, 0.4 * wave_left), np.subtract(wave_left / 1.25, 0.4 * wave_right)])

    return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False):
    """Convert the combined full-band spectrogram back to a waveform by
    splitting it into per-band spectrograms, inverting each and resampling."""
    bands_n = len(mp["band"])
    offset = 0
    wave = None

    for d in range(1, bands_n + 1):
        bp = mp["band"][d]
        spec_s = np.zeros(shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"]:bp["crop_stop"], :] = spec_m[:, offset:offset + h, :]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h:max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp["hpf_start"] > 0:
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
            else:
                wave = np.add(wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model))
        else:
            sr = mp["band"][d + 1]["sr"]
            if d == 1:  # lower
                if is_v51_model:
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                wave = resample_audio(
                    spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model),
                    orig_sr=bp["sr"], target_sr=sr, res_type=bp["res_type"],
                )
            else:  # mid
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                wave2 = np.add(wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model))
                wave = resample_audio(wave2, orig_sr=bp["sr"], target_sr=sr, res_type=bp["res_type"])

    return wave


def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    mask = np.concatenate(
        [np.ones((bin_start - 1, 1)), np.linspace(1, 0, bin_stop - bin_start + 1)[:, None], np.zeros((n_bins - bin_stop, 1))],
        axis=0,
    )

    return mask


def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    mask = np.concatenate(
        [np.zeros((bin_stop + 1, 1)), np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None], np.ones((n_bins - bin_start - 2, 1))],
        axis=0,
    )

    return mask


def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0:bin_stop + 1, :] *= 0

    return spec


def mirroring(a, spec_m, input_high_end, mp):
    """Mirror the missing high-frequency range for ``high_end_process``."""
    if "mirroring" == a:
        mirror = np.flip(np.abs(spec_m[:, mp["pre_filter_start"] - 10 - input_high_end.shape[1]:mp["pre_filter_start"] - 10, :]), 1)
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)

    if "mirroring2" == a:
        mirror = np.flip(np.abs(spec_m[:, mp["pre_filter_start"] - 10 - input_high_end.shape[1]:mp["pre_filter_start"] - 10, :]), 1)
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)


def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    aggr = aggressiveness["value"] * 2

    if aggr != 0:
        if is_non_accom_stem:
            aggr = 1 - aggr

        if np.any(aggr > 10) or np.any(aggr < -10):
            print(f"Warning: Extreme aggressiveness values detected: {aggr}")

        aggr = [aggr, aggr]

        if aggressiveness["aggr_correction"] is not None:
            aggr[0] += aggressiveness["aggr_correction"]["left"]
            aggr[1] += aggressiveness["aggr_correction"]["right"]

        for ch in range(2):
            mask[ch, :aggressiveness["split_bin"]] = np.power(mask[ch, :aggressiveness["split_bin"]], 1 + aggr[ch] / 3)
            mask[ch, aggressiveness["split_bin"]:] = np.power(mask[ch, aggressiveness["split_bin"]:], 1 + aggr[ch])

    return mask


def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    mask = y_mask

    try:
        if min_range < fade_size * 2:
            raise ValueError("min_range must be >= fade_size * 2")

        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)
        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None
            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size:
                    s = old_e - fade_size * 2

                if s != 0:
                    weight[:, :, s:s + fade_size] = np.linspace(0, 1, fade_size)
                else:
                    s -= fade_size

                if e != y_mask.shape[2]:
                    weight[:, :, e - fade_size:e] = np.linspace(1, 0, fade_size)
                else:
                    e += fade_size

                weight[:, :, s + fade_size:e - fade_size] = 1
                old_e = e

        v_mask = 1 - y_mask
        y_mask += weight * v_mask

        mask = y_mask
    except Exception:
        pass

    return mask
