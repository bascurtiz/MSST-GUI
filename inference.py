# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, load_config, parse_args_inference
from utils.model_utils import bigshifts_wrapper
from utils.model_utils import (
    prefer_target_instrument, apply_tta, load_start_checkpoint,
    ensure_readable_checkpoint)
from utils.stem_planning import complement_stem_name

import warnings

warnings.filterwarnings("ignore")


def run_folder(
    model: "torch.nn.Module",
    args: "argparse.Namespace",
    config: dict,
    device: "torch.device",
    verbose: bool = False
) -> None:
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : argparse.Namespace
        Arguments containing input folder, output folder, and processing options.
    config : dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()

    # Recursively collect all files from input directory
    mixture_paths = sorted(
        glob.glob(os.path.join(args.input_folder, "**/*.*"), recursive=True)
    )
    mixture_paths = [p for p in mixture_paths if os.path.isfile(p)]

    sample_rate: int = getattr(config.audio, "sample_rate", 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments: list[str] = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    # Wrap paths with progress bar if not in verbose mode
    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    # Determine whether to use detailed progress bar
    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    for path in mixture_paths:
        # Get relative path from input folder
        relative_path: str = os.path.relpath(path, args.input_folder)
        # Extract directory and file name
        dir_name: str = os.path.dirname(relative_path)
        file_name: str = os.path.splitext(os.path.basename(path))[0]

        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f"Cannot read track: {format(path)}")
            print(f"Error message: {str(e)}")
            continue

        # Convert mono audio to expected channel format if needed
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if "num_channels" in config.audio:
                if config.audio["num_channels"] == 2:
                    print("Convert mono track to stereo...")
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()

        # Mono models (stereo: false — e.g. 16 kHz speech-denoising mel-band
        # checkpoints with num_channels: 1) only accept a single channel.
        # Downmix stereo sources for the model, then re-expand the separated
        # stems to two channels below so output files keep the source's
        # channel layout and mix-minus complements stay channel-consistent.
        model_stereo = getattr(model, 'stereo', True)
        if model_stereo is None:
            model_stereo = True
        downmixed = False
        if not model_stereo and mix.shape[0] > 1:
            print("Model is mono (stereo: false) - downmixing stereo input to mono...")
            mix = np.mean(mix, axis=0, keepdims=True)
            downmixed = True

        # Normalize input audio if enabled
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                mix, norm_params = normalize_audio(mix)

        # Perform source separation
        waveforms_orig = bigshifts_wrapper(
            config,
            model,
            mix,
            device,
            model_type=args.model_type,
            pbar=detailed_pbar,
            bigshifts=args.bigshifts
        )

        # Apply test-time augmentation if enabled
        if args.use_tta:
            waveforms_orig = apply_tta(
                config,
                model,
                mix,
                waveforms_orig,
                device,
                args.model_type,
                bigshifts=args.bigshifts,
                pbar=detailed_pbar
            )

        # Mono-model run: give the separated stems the same two-channel
        # layout as the source track (they were produced from the downmixed
        # mono input).
        if downmixed:
            waveforms_orig = {
                k: (np.repeat(v, 2, axis=0)
                    if v.ndim == 2 and v.shape[0] == 1 else v)
                for k, v in waveforms_orig.items()
            }

        # Extract the complement track if requested (mix minus the separated
        # target). Name it after the model's *other* trained stem when there is
        # exactly one (e.g. a vocals/instrument model yields real "vocals",
        # not a misleading "instrumental"); for multi-stem models the mix
        # minus one target is not any single trained stem, so keep upstream's
        # generic "instrumental" label. (Shared helper: utils/stem_planning.py)
        if args.extract_instrumental:
            all_instruments = list(getattr(config.training, 'instruments', []) or [])
            complement_name = complement_stem_name(all_instruments, instruments)
            instr = "vocals" if "vocals" in instruments else instruments[0]
            waveforms_orig[complement_name] = mix_orig - waveforms_orig[instr]
            if complement_name not in instruments:
                instruments.append(complement_name)

        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Denormalize output audio if normalization was applied
            if "normalize" in config.inference:
                if config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            peak: float = float(np.abs(estimates).max())
            codec, norm_scale = output_codec(args.pcm_type, peak)
            if norm_scale is not None:
                print(f"Note: stem '{instr}' peaks at {peak:.2f} (>1.0) \u2014 "
                      "peak-normalizing to keep the selected FLAC format "
                      "without clipping.")
                estimates = estimates * norm_scale

            subtype = args.pcm_type

            # Generate output directory structure using relative paths
            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=file_name,
                dir_name=dir_name,
                model_type=args.model_type,
                model=os.path.splitext(
                    os.path.basename(args.start_check_point)
                )[0],
            )

            # Create output directory
            output_dir: str = os.path.join(args.store_dir, *dirnames)
            os.makedirs(output_dir, exist_ok=True)

            output_path: str = os.path.join(output_dir, f"{fname}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)

            # Draw and save spectrogram if enabled
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

def output_codec(pcm_type: str, peak: float):
    """Pick the output codec (and optional peak-normalize scale) for a stem.

    The configured PCM type decides the format for EVERY stem — the old
    per-stem fallback wrote hot stems as WAV even when FLAC was selected,
    producing mixed .wav/.flac folders. Integer types (FLAC PCM_16/24)
    always yield '.flac'; stems peaking above 1.0 get a scale-down factor so
    the integer codec never clips. FLOAT always yields '.wav' (32-bit float).
    """
    if pcm_type != 'FLOAT':
        scale = 1.0 / peak if peak > 1.0 else None
        return "flac", scale
    return "wav", None


def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname

def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    if args.model_type == 'mdxnet':
        # MDX-Net zoo models ship as ONNX checkpoints with a kuielab-layout
        # config; they can't go through get_model_from_config's torch branches.
        # MDXNetModel raises a clear error if onnxruntime is missing.
        from models.mdx_net import MDXNetModel
        config = load_config(args.model_type, args.config_path)
        model = MDXNetModel(config, args.start_check_point)
    else:
        # custom_backend: author-shipped architecture file (fork models such
        # as pcunwa's HyperACE / BS-Roformer-Large-Inst) — the model class is
        # loaded from that folder's bs_roformer.py instead of the bundled code.
        try:
            model, config = get_model_from_config(
                args.model_type, args.config_path,
                custom_backend=getattr(args, 'custom_backend', None),
                checkpoint_path=args.start_check_point)
        except ValueError as exc:
            # A model whose type has no dispatch branch used to die here as
            # a raw traceback; the GUI pre-validates registered models, but
            # manual CLI / ensemble launches can still reach this point, so
            # report it clearly and exit instead.
            if str(exc).startswith('Unknown model type'):
                print(f"ERROR: {exc}")
                print("This model type is not supported by this build's "
                      "inference engine. Pick another model or update the app.")
                sys.exit(1)
            raise
    if 'model_type' in config.training and args.model_type != 'mdxnet':
        args.model_type = config.training.model_type
    # MDX-Net checkpoints are ONNX, so torch.load/load_start_checkpoint don't
    # apply — the ONNX session is built inside MDXNetModel from the checkpoint.
    if args.start_check_point and args.model_type != 'mdxnet':
        # >=4 GiB checkpoints are ZIP64 archives that torch 2.11's C++
        # reader crashes on natively when an older torch wrote them (access
        # violation in get_storage_from_record). mmap=True reads them but
        # its mapped storages intermittently fault during the state-dict
        # copy on Windows, so the deterministic path is: rewrite the
        # archive once with python's zipfile (ensure_readable_checkpoint)
        # and plain-load that — torch reads python-written ZIP64 fine.
        checkpoint = torch.load(
            ensure_readable_checkpoint(args.start_check_point),
            weights_only=False, map_location='cpu')
        try:
            load_start_checkpoint(args, model, checkpoint, type_='inference')
        finally:
            # Free the source state dict before model.to(device) to keep
            # peak RAM (and commit) low; load_start_checkpoint already
            # copied the weights into the model's own parameters.
            del checkpoint

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if (isinstance(args.device_ids, list) and len(args.device_ids) > 1
            and not args.force_cpu and args.model_type != 'mdxnet'):
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
