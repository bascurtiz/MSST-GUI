# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'  # forked by jarredou
# Enhanced with folder support, PyTorch STFT, GPU acceleration, and soundfile

import os
import math
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch


def load_audio(file_path):
    """
    Load audio file using soundfile (with librosa fallback) and convert to torch tensor.
    Returns: (waveform, sample_rate)
    waveform shape: (channels, samples)
    """
    try:
        audio_np, sample_rate = sf.read(file_path, always_2d=True)
        # soundfile returns (samples, channels), transpose to (channels, samples)
        audio_np = audio_np.T
    except Exception:
        import librosa
        audio_np, sample_rate = librosa.load(file_path, sr=None, mono=False)
        if audio_np.ndim == 1:
            audio_np = np.expand_dims(audio_np, axis=0)

    # Convert to torch tensor
    waveform = torch.from_numpy(audio_np).float()
    return waveform, sample_rate


def resample_audio(waveform, orig_sr, target_sr):
    """
    Resample audio using torchaudio if available, else linear interpolation.
    """
    if orig_sr == target_sr:
        return waveform

    try:
        import torchaudio.functional as taf
        return taf.resample(waveform, orig_sr, target_sr)
    except Exception:
        ratio = target_sr / orig_sr
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # (1, channels, samples)

        resampled = torch.nn.functional.interpolate(
            waveform,
            scale_factor=ratio,
            mode='linear',
            align_corners=False
        )
        return resampled.squeeze(0)


def linear_to_db(value):
    """Convert linear amplitude value to decibels"""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if value <= 0:
        return float('-inf')
    return 20.0 * math.log10(value)


def stft(wave, n_fft, hop_length):
    """
    Compute STFT for stereo/multi-channel audio using PyTorch.
    :param wave: tensor of shape (channels, length)
    :param n_fft: FFT size
    :param hop_length: hop length
    :return: complex spectrogram of shape (channels, freq_bins, time_frames)
    """
    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=wave.device),
        return_complex=True
    )
    return spec


def istft(spec, hop_length, length):
    """
    Compute inverse STFT for stereo/multi-channel audio using PyTorch.
    :param spec: complex spectrogram of shape (channels, freq_bins, time_frames)
    :param hop_length: hop length
    :param length: target length
    :return: waveform of shape (channels, length)
    """
    n_fft = (spec.shape[1] - 1) * 2
    wave = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=spec.device),
        length=length
    )
    return wave


def absmax(a, axis=None, *, dim=None):
    """Find values with maximum absolute value along dimension/axis."""
    if dim is not None:
        axis = dim
    if isinstance(a, torch.Tensor):
        abs_a = torch.abs(a)
        indices = torch.argmax(abs_a, dim=axis, keepdim=True)
        return torch.gather(a, axis, indices).squeeze(axis)
    else:
        dims = list(a.shape)
        dims.pop(axis)
        indices = list(np.ogrid[tuple(slice(0, d) for d in dims)])
        argmax = np.abs(a).argmax(axis=axis)
        indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
        return a[tuple(indices)]


def absmin(a, axis=None, *, dim=None):
    """Find values with minimum absolute value along dimension/axis."""
    if dim is not None:
        axis = dim
    if isinstance(a, torch.Tensor):
        abs_a = torch.abs(a)
        indices = torch.argmin(abs_a, dim=axis, keepdim=True)
        return torch.gather(a, axis, indices).squeeze(axis)
    else:
        dims = list(a.shape)
        dims.pop(axis)
        indices = list(np.ogrid[tuple(slice(0, d) for d in dims)])
        argmin = np.abs(a).argmin(axis=axis)
        indices.insert((len(a.shape) + axis) % len(a.shape), argmin)
        return a[tuple(indices)]


def lambda_max(arr, dim=None, key=None, axis=None, keepdims=False):
    """Find elements with maximum key value along dimension."""
    if axis is not None:
        dim = axis
    if isinstance(arr, torch.Tensor):
        if key is None:
            key = lambda x: x
        key_values = key(arr)
        indices = torch.argmax(key_values, dim=dim, keepdim=True)
        res = torch.gather(arr, dim, indices)
        return res if keepdims else res.squeeze(dim)
    else:
        if key is None:
            key = lambda x: x
        idxs = np.argmax(key(arr), axis=dim)
        if dim is not None:
            idxs = np.expand_dims(idxs, dim)
            result = np.take_along_axis(arr, idxs, dim)
            if not keepdims:
                result = np.squeeze(result, axis=dim)
            return result
        else:
            return arr.flatten()[idxs]


def lambda_min(arr, dim=None, key=None, axis=None, keepdims=False):
    """Find elements with minimum key value along dimension."""
    if axis is not None:
        dim = axis
    if isinstance(arr, torch.Tensor):
        if key is None:
            key = lambda x: x
        key_values = key(arr)
        indices = torch.argmin(key_values, dim=dim, keepdim=True)
        res = torch.gather(arr, dim, indices)
        return res if keepdims else res.squeeze(dim)
    else:
        if key is None:
            key = lambda x: x
        idxs = np.argmin(key(arr), axis=dim)
        if dim is not None:
            idxs = np.expand_dims(idxs, dim)
            result = np.take_along_axis(arr, idxs, dim)
            if not keepdims:
                result = np.squeeze(result, axis=dim)
            return result
        else:
            return arr.flatten()[idxs]


def lambda_median(arr, dim=0, key=None):
    """
    Find elements with median key value along dimension.
    For complex STFT spectrograms, this selects the complex bin corresponding
    to the median magnitude, avoiding ComplexFloat sorting errors and preserving phase.
    """
    if key is None:
        key = lambda x: x
    key_values = key(arr)
    sorted_indices = torch.argsort(key_values, dim=dim)
    median_pos = arr.shape[dim] // 2
    median_idx = torch.select(sorted_indices, dim=dim, index=median_pos).unsqueeze(dim)
    return torch.gather(arr, dim, median_idx).squeeze(dim)


def match_tensor_shapes(tensor_1, tensor_2):
    """Match the time dimension of two tensors by padding or trimming"""
    target_length = tensor_2.shape[-1]
    current_length = tensor_1.shape[-1]

    if current_length > target_length:
        tensor_1 = tensor_1[..., :target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        tensor_1 = torch.nn.functional.pad(tensor_1, (0, padding), 'constant', 0)

    return tensor_1


def average_waveforms(pred_track, weights, algorithm, n_fft=2048, hop_length=1024):
    """
    Combine tracks using the chosen ensemble algorithm.

    :param pred_track: list or tensor/ndarray of shape (num, channels, length)
    :param weights: weights of shape (num,)
    :param algorithm: One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft
    :param n_fft: FFT size for STFT operations (default: 2048)
    :param hop_length: hop length for STFT operations (default: 1024)
    :return: averaged waveform in shape (channels, length)
    """
    was_numpy = False
    if isinstance(pred_track, np.ndarray):
        was_numpy = True
        pred_track = [torch.from_numpy(p).float() for p in pred_track]
    elif len(pred_track) > 0 and isinstance(pred_track[0], np.ndarray):
        was_numpy = True
        pred_track = [torch.from_numpy(p).float() for p in pred_track]

    pred_track = torch.stack(pred_track)
    final_length = pred_track.shape[-1]
    weights = torch.as_tensor(weights, dtype=pred_track.dtype, device=pred_track.device)

    total_weight = torch.sum(weights)
    if total_weight == 0:
        total_weight = 1.0

    if algorithm in ['avg_wave', 'median_wave', 'min_wave', 'max_wave']:
        # Waveform domain operations
        if algorithm == 'avg_wave':
            weighted_tracks = pred_track * weights.view(-1, 1, 1)
            result = torch.sum(weighted_tracks, dim=0) / total_weight
        elif algorithm == 'median_wave':
            result = torch.median(pred_track, dim=0)[0]
        elif algorithm == 'min_wave':
            result = lambda_min(pred_track, dim=0, key=torch.abs)
        elif algorithm == 'max_wave':
            result = lambda_max(pred_track, dim=0, key=torch.abs)
        else:
            raise ValueError(f"Unknown waveform algorithm: {algorithm}")

    elif algorithm in ['avg_fft', 'median_fft', 'min_fft', 'max_fft']:
        # Frequency domain operations
        spec_tracks = []
        for i in range(pred_track.shape[0]):
            spec = stft(pred_track[i], n_fft, hop_length)
            spec_tracks.append(spec)

        spec_tracks = torch.stack(spec_tracks)

        if algorithm == 'avg_fft':
            weighted_specs = spec_tracks * weights.view(-1, 1, 1, 1)
            avg_spec = torch.sum(weighted_specs, dim=0) / total_weight
            result = istft(avg_spec, hop_length, final_length)
        elif algorithm == 'median_fft':
            median_spec = lambda_median(spec_tracks, dim=0, key=torch.abs)
            result = istft(median_spec, hop_length, final_length)
        elif algorithm == 'min_fft':
            min_spec = lambda_min(spec_tracks, dim=0, key=torch.abs)
            result = istft(min_spec, hop_length, final_length)
        elif algorithm == 'max_fft':
            max_spec = absmax(spec_tracks, dim=0)
            result = istft(max_spec, hop_length, final_length)
        else:
            raise ValueError(f"Unknown FFT algorithm: {algorithm}")
    else:
        raise ValueError(f"Unknown ensemble algorithm: {algorithm}")

    if was_numpy and isinstance(result, torch.Tensor):
        result = result.detach().cpu().numpy()

    return result


def save_audio(waveform, sample_rate, output_path, output_format=None):
    """Save audio with format normalization and peak clipping protection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    # Ensure waveform is in the right format (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to CPU numpy array if on GPU
    if waveform.is_cuda:
        waveform = waveform.cpu()

    # Convert to numpy and transpose to (samples, channels) for soundfile
    audio_np = waveform.numpy().T

    # Normalize format alias
    fmt = (output_format or "").upper()
    if fmt == 'FLOAT':
        fmt = 'WAV_FLOAT'
    elif fmt == 'PCM_16':
        fmt = 'FLAC_16'
    elif fmt == 'PCM_24':
        fmt = 'FLAC_24'

    # Auto-detect from file extension if not explicitly specified
    if not fmt:
        if output_path.suffix.lower() == '.flac':
            fmt = 'FLAC_24'
        else:
            fmt = 'WAV_FLOAT'

    if fmt == 'WAV_FLOAT':
        sf.write(str(output_path), audio_np, sample_rate, subtype='FLOAT')
    elif fmt in ['FLAC_16', 'FLAC_24']:
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1.0:
            peak_db = linear_to_db(max_val)
            print(f"Clipping detected, normalizing: {peak_db:.2f} dB")
            audio_np = audio_np / max_val.item()

        subtype = 'PCM_16' if fmt == 'FLAC_16' else 'PCM_24'
        sf.write(str(output_path), audio_np, sample_rate, subtype=subtype)
    else:
        sf.write(str(output_path), audio_np, sample_rate)


def get_audio_files_from_folders(folders):
    """
    Scan folders recursively and group audio files by relative path.
    Returns dict: {rel_path: [path_in_folder1, path_in_folder2, ...]}
    """
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.opus', '.aiff'}
    files_by_rel_path = defaultdict(list)

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f'Warning: Folder does not exist: {folder}')
            continue

        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                rel_path = str(file_path.relative_to(folder_path))
                files_by_rel_path[rel_path].append(str(file_path))

    return files_by_rel_path


def process_files(args, device):
    """Process individual files"""
    if args.weights is not None:
        if len(args.weights) != len(args.files):
            raise ValueError("Number of weights must match number of files")
        weights = [float(w) for w in args.weights]
    else:
        weights = [1.0] * len(args.files)

    print(f'Number of input files: {len(args.files)}')
    print(f'Weights: {weights}')
    print(f'Output file: {args.output}')

    data = []
    max_len = 0
    min_len = float('inf')
    sample_rate = None

    for f in args.files:
        if not os.path.isfile(f):
            print(f"Error. Can't find file: {f}. Check paths.")
            return

        print(f'Reading file: {f}')
        wav, sr = load_audio(f)

        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            print(f'Warning: Sample rate mismatch. Expected {sample_rate}, got {sr}. Resampling...')
            wav = resample_audio(wav, sr, sample_rate)

        # Ensure stereo (2 channels)
        if wav.shape[0] == 1:
            print("Mono detected. Converting to stereo by duplication.")
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            print(f"Multi-channel audio detected ({wav.shape[0]} channels). Using first 2 channels.")
            wav = wav[:2]

        wav = wav.to(device)
        print(f"Waveform shape: {wav.shape} sample rate: {sr}")
        data.append(wav)
        max_len = max(max_len, wav.shape[1])
        min_len = min(min_len, wav.shape[1])

    # Choose target length
    target_len = min_len if args.trim_to_shortest else max_len
    target_tensor = torch.zeros(2, target_len, device=device)
    data = [match_tensor_shapes(wav, target_tensor) for wav in data]

    print(f"Target length: {target_len} ({'shortest' if args.trim_to_shortest else 'longest'} input file)")
    print("Starting ensemble processing...")
    result = average_waveforms(data, weights, args.type, args.n_fft, args.hop_length)
    print(f'Result shape: {result.shape}')

    save_audio(result, sample_rate, args.output, args.output_format)
    print(f'Ensemble saved to: {args.output}')


def process_folders(args, device):
    """Process folders - ensemble matching files from each folder"""
    if args.weights is not None:
        if len(args.weights) != len(args.folders):
            raise ValueError("Number of weights must match number of folders")
        weights = [float(w) for w in args.weights]
    else:
        weights = [1.0] * len(args.folders)

    print(f'Number of input folders: {len(args.folders)}')
    print(f'Folders: {args.folders}')
    print(f'Weights: {weights}')
    print(f'Output folder: {args.output}')

    files_by_rel_path = get_audio_files_from_folders(args.folders)

    if not files_by_rel_path:
        print("Error: No audio files found in the specified folders")
        return

    # Filter to only files that appear in all folders
    valid_files = {name: paths for name, paths in files_by_rel_path.items()
                   if len(paths) == len(args.folders)}

    if not valid_files:
        print("Error: No matching files found across all folders")
        print("\nFiles found per folder:")
        for name, paths in sorted(files_by_rel_path.items()):
            print(f"  {name}: found in {len(paths)}/{len(args.folders)} folders")
        return

    print(f'\nFound {len(valid_files)} matching files to ensemble:')
    for name in sorted(valid_files.keys()):
        print(f'  - {name}')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine output extension based on output_format
    fmt = (args.output_format or "").upper()
    if fmt in ['FLAC_16', 'FLAC_24', 'PCM_16', 'PCM_24']:
        output_ext = '.flac'
    else:
        output_ext = '.wav'

    total_files = len(valid_files)
    for idx, (filename, file_paths) in enumerate(sorted(valid_files.items()), 1):
        print(f'\n{"="*80}')
        print(f'Processing {idx}/{total_files}: {filename}')
        print(f'{"="*80}')

        output_filename = Path(filename).with_suffix(output_ext)
        dest_file = output_path / output_filename
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        file_args = argparse.Namespace(
            files=file_paths,
            folders=None,
            type=args.type,
            weights=weights,
            output=str(dest_file),
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            device=args.device,
            trim_to_shortest=args.trim_to_shortest,
            output_format=args.output_format
        )
        process_files(file_args, device)

    print(f'\n{"="*80}')
    print(f'All files processed! Output folder: {args.output}')
    print(f'{"="*80}')


def ensemble_files(args):
    parser = argparse.ArgumentParser(description="Audio ensemble tool using PyTorch and soundfile")
    parser.add_argument("--files", type=str, nargs='+',
                        help="Path to all audio-files to ensemble")
    parser.add_argument("--folders", type=str, nargs='+',
                        help="Path to folders containing audio files to ensemble (will match by filename)")
    parser.add_argument("--type", type=str, default='avg_wave',
                        help="One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft")
    parser.add_argument("--weights", type=float, nargs='+',
                        help="Weights to create ensemble. Number of weights must be equal to number of files/folders")
    parser.add_argument("--output", type=str,
                        help="Path to output file (for --files mode) or output folder (for --folders mode)")
    parser.add_argument("--n_fft", type=int, default=2048,
                        help="FFT size for STFT operations (default: 2048)")
    parser.add_argument("--hop_length", type=int, default=1024,
                        help="Hop length for STFT operations (default: 1024)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)")
    parser.add_argument("--trim_to_shortest", action="store_true",
                        help="Trim output to shortest input file length to avoid padding artifacts")
    parser.add_argument("--output_format", type=str, default=None,
                        choices=["WAV_FLOAT", "FLAC_16", "FLAC_24", "FLOAT", "PCM_16", "PCM_24"],
                        help="Output format: WAV_FLOAT (32-bit float WAV), FLAC_16 (16-bit FLAC), FLAC_24 (24-bit FLAC)")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.files and args.folders:
        print("Error: Cannot use both --files and --folders at the same time")
        return

    if not args.files and not args.folders:
        print("Error: Must specify either --files or --folders")
        parser.print_help()
        return

    if not args.output:
        print("Error: --output is required")
        return

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')
    print(f'Ensemble type: {args.type}')
    print(f'N_FFT: {args.n_fft}, Hop length: {args.hop_length}')

    if args.files:
        process_files(args, device)
    elif args.folders:
        process_folders(args, device)


if __name__ == "__main__":
    ensemble_files(None)
