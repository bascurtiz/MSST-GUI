# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import shutil
import subprocess
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram, get_audio_metadata, sanitize_filename
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import bigshifts_wrapper
from utils.model_utils import apply_tta, load_start_checkpoint

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

    instruments: list[str] = list(getattr(config.training, "instruments", []) or [])[:]
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
        # Try to use metadata for display name
        artist, title = get_audio_metadata(path)
        if artist and title:
            display_name = f"{sanitize_filename(artist)} - {sanitize_filename(title)}"
        else:
            display_name = file_name

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

        # Extract instrumental track if requested
        if args.extract_instrumental:
            target = getattr(config.training, "target_instrument", None) or "vocals"
            if target in waveforms_orig:
                waveforms_orig["instrumental"] = mix_orig - waveforms_orig[target]
                if "instrumental" not in instruments:
                    instruments.append("instrumental")

        # Only consider stems the model actually produced
        instruments = [i for i in instruments if i in waveforms_orig]

        # Filter to only save specified stems
        if args.save_stems:
            save_list = [s.strip().lower() for s in args.save_stems.split(',')]
            instruments = [i for i in instruments if i.lower() in save_list]

        # Compute rest (complement = mix − sum of selected stems)
        if args.save_rest:
            stems_to_sum = [s for s in instruments
                            if s.lower() != 'rest' and s.lower() != 'instrumental']
            if stems_to_sum:
                sum_selected = None
                for s in stems_to_sum:
                    if sum_selected is None:
                        sum_selected = waveforms_orig[s].copy()
                    else:
                        sum_selected += waveforms_orig[s]
                if sum_selected is not None:
                    waveforms_orig["rest"] = mix_orig - sum_selected
                    if "rest" not in instruments:
                        instruments.append("rest")

        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Denormalize output audio if normalization was applied
            if "normalize" in config.inference:
                if config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            codec = args.output_format
            subtype = args.pcm_type
            if codec == "flac" and subtype == "FLOAT":
                # FLAC cannot store float samples; fall back to 16-bit PCM.
                subtype = "PCM_16"

            # Generate output filename using template
            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=display_name,
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
            if codec == "mp3":
                tmp_wav = os.path.join(output_dir, f"{fname}.wav")
                sf.write(tmp_wav, estimates.T, sr, subtype="PCM_16")
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg is None:
                    print(f"WARNING: ffmpeg not found, wrote {fname}.wav (PCM 16) instead of mp3")
                    print("Wrote file:", tmp_wav)
                else:
                    res = subprocess.run(
                        [ffmpeg, "-y", "-i", tmp_wav, "-codec:a", "libmp3lame",
                         "-b:a", f"{args.mp3_bitrate}k", output_path],
                        capture_output=True, text=True,
                        creationflags=0x08000000 if os.name == "nt" else 0
                    )
                    if res.returncode == 0 and os.path.isfile(output_path):
                        os.remove(tmp_wav)
                        print("Wrote file:", output_path)
                    else:
                        print("WARNING: mp3 conversion failed, kept wav output instead")
                        print("Wrote file:", tmp_wav)
            else:
                sf.write(output_path, estimates.T, sr, subtype=subtype)
                print("Wrote file:", output_path)

            # Draw and save spectrogram if enabled
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

        print(f"Completed: {display_name}")

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

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

    model, config = get_model_from_config(args.model_type, args.config_path,
                                           custom_backend=args.custom_backend)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
