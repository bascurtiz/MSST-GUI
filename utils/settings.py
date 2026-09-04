import os
import random
import time
import yaml
import numpy as np
import torch
import argparse
import inspect
import socket
from typing import Dict, List, Tuple, Union
from omegaconf import OmegaConf
from ml_collections import ConfigDict
import torch.distributed as dist
from torch import nn
import soundfile as sf

# Stand-in for the bitsandbytes package so torch.load can unpickle
# checkpoints saved with bnb optimizers (scnet_huge_* family, etc.) even
# though the package isn't part of the runtime. See utils/bnb_stub.py.
import utils.bnb_stub  # noqa: F401  (self-installs on import)


def parse_args_train(dict_args: Union[argparse.Namespace, Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.

    This function constructs an argument parser for model, dataset, training, and logging
    options, merges overrides from a provided dictionary (if any), and returns the parsed
    arguments. If `dict_args` is None, the arguments are parsed from `sys.argv`.

    Args:
        dict_args (Dict | None): Optional dictionary of argument overrides. Keys should
            match the defined CLI options.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all configuration
        values required for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--load_optimizer", action='store_true',
                        help="Load optimizer state from checkpoint (if available)")
    parser.add_argument("--load_scheduler", action='store_true',
                        help="Load scheduler state from checkpoint (if available)")
    parser.add_argument("--load_epoch", action='store_true', help="Load epoch number from checkpoint (if available)")
    parser.add_argument("--load_best_metric", action='store_true',
                        help="Load best metric from checkpoint (if available)")
    parser.add_argument("--load_all_metrics", action='store_true',
                        help="Load all metrics from checkpoint (if available)")
    parser.add_argument("--load_all_losses", action='store_true',
                        help="Load all losses from checkpoint (if available)")
    parser.add_argument("--safe_mode", action='store_true',
                        help="Ignore forward errors")
    parser.add_argument("--results_path", type=str,
                        help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3, 4, 5, 6, 7. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str,
                        help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--loss", type=str, nargs='+', choices=[
        'masked_loss', 'mse_loss', 'l1_loss', 'multistft_loss', 'spec_masked_loss', 'spec_rmse_loss',
        'log_wmse_loss', 'l1_snr_loss', 'l1_snr_db_loss', 'stft_l1_snr_db_loss', 'multi_l1_snr_db_loss',
        'fullness_penalty_loss', 'bleedless_penalty_loss'
    ], default=['masked_loss'], help="List of loss functions to use")
    parser.add_argument("--masked_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--mse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--l1_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--log_wmse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--multistft_loss_coef", type=float, default=0.001, help="Coef for loss")
    parser.add_argument("--spec_masked_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--spec_rmse_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--l1_snr_loss_coef", type=float, default=1., help="Coef for L1-SNR loss")
    parser.add_argument("--l1_snr_db_loss_coef", type=float, default=1., help="Coef for L1-SNR-DB loss")
    parser.add_argument("--stft_l1_snr_db_loss_coef", type=float, default=1., help="Coef for STFT-L1-SNR-DB loss")
    parser.add_argument("--multi_l1_snr_db_loss_coef", type=float, default=1., help="Coef for Multi-L1-SNR-DB loss")
    parser.add_argument("--fullness_penalty_loss_coef", type=float, default=0.00002,
                        help="Coef for the fullness penalty loss. This loss should be used in combination with a primary loss function.")
    parser.add_argument("--bleedless_penalty_loss_coef", type=float, default=0.00002,
                        help="Coef for the bleedless penalty loss. This loss should be used in combination with a primary loss function.")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--wandb_offline", action='store_true', help='local wandb')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness', 'l1_snr', 'bleedless_mr', 'fullness_mr'],
                        help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['k_sdr','sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness', 'l1_snr', 'bleedless_mr', 'fullness_mr'],
                        help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora_peft", action='store_true', help="Training with LoRA from peft")
    parser.add_argument("--train_lora_loralib", action='store_true', help="Training with LoRA from loralib")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--each_metrics_in_name", action='store_true',
                        help="All stems in naming checkpoints")
    parser.add_argument("--use_standard_loss", action='store_true',
                        help="Roformers will use provided loss instead of internal")
    parser.add_argument("--custom_backend", type=str, default=None,
                        help="Path to a folder containing an author-provided backend .py (e.g. bs_roformer.py). "
                             "When given, the model class is loaded dynamically from that file instead of the "
                             "bundled model code — used for fine-tune starts from fork-architecture checkpoints.")
    parser.add_argument("--save_weights_every_epoch", action='store_true',
                        help="Weights will be saved every epoch with all metric values")
    parser.add_argument("--persistent_workers", action='store_true',
                        help="dataloader persistent_workers")
    parser.add_argument("--prefetch_factor", type=int, default=None,
                        help="dataloader prefetch_factor")
    parser.add_argument("--set_per_process_memory_fraction", action='store_true',
                        help="using only VRAM, no RAM")
    parser.add_argument("--load_only_compatible_weights", action='store_true',
                        help="using only VRAM, no RAM")
    parser.add_argument("--freeze_layers", nargs="+", type=str,
                        help="List of layers to freeze. Use prefixes e.g. layer1 - will freeze all layers whose names "
                             "starts with layer1. You can set mulitple parameters.")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    get_internal_loss = (args.model_type in ('mel_band_conformer',) or 'roformer' in args.model_type
                         ) and not args.use_standard_loss
    if get_internal_loss:
        args.loss = [f'{args.model_type}_loss']
    return args


def parse_args_valid(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for validation configuration.

    Builds the CLI for model selection, configuration paths, validation data
    locations, output/spectrogram saving options, device/runtime settings, and
    evaluation metrics. If `dict_args` is provided, its key–value pairs override
    or set the parsed arguments; otherwise arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all validation
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer,"
                             " bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint"
                                                                          " to valid weights")
    parser.add_argument("--valid_path", nargs="+", type=str, help="Validate path")
    parser.add_argument("--store_dir", type=str, default="", help="Path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="If --store_dir is set then code will generate spectrograms for resulted stems as well."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='List of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="Dataloader pin_memory")
    parser.add_argument("--extension", type=str, default='wav', help="Choose extension for validation")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                             "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness', 'l1_snr', 'bleedless_mr', 'fullness_mr'],
                        help='List of metrics to use.')
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")


    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def parse_args_inference(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for inference configuration.

    Builds the CLI for model selection, configuration path, input/output handling,
    device/runtime options, test-time augmentation, and optional LoRA checkpoints.
    If `dict_args` is provided, its key–value pairs override or supply CLI options
    programmatically; otherwise, arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all inference
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Code will generate spectrograms for resulted stems."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--bigshifts", type=int, default=1,
                        help="Number of circular time shifts to average during demix. Values <= 0 are treated as 1.")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--custom_backend", type=str, default=None,
                        help="Path to a folder containing an author-provided backend .py (e.g. bs_roformer.py). "
                             "When given, the model class is loaded dynamically from that file instead of the "
                             "bundled model code — lets fork architectures (pcunwa HyperACE/Large-Inst, etc.) run "
                             "without app code changes.")
    parser.add_argument("--filename_template", type=str, default='{file_name}/{instr}',
                        help="Output filename template, without extension, using '/' for subdirectories. Default: '{file_name}/{instr}'")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()
    args.pcm_type = validate_sndfile_subtype(args)

    return args


def validate_sndfile_subtype(args):
    codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
    subtype = args.pcm_type
    if subtype in sf.available_subtypes(codec):
        return subtype
    default = sf.default_subtype(codec)
    print(f"WARNING: codec {codec} doesn't support subtype {subtype}, defaulting to {default}")
    return default


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """
    Load a model configuration from a file.

    Based on `model_type`, returns either an OmegaConf (e.g., for 'htdemucs')
    or a YAML-parsed ConfigDict for other models.

    Args:
        model_type (str): Model identifier that determines the loader behavior
            (e.g., 'htdemucs', 'mdx23c', etc.).
        config_path (str): Path to the configuration file (YAML/OmegaConf).

    Returns:
        Union[ConfigDict, OmegaConf]: Loaded configuration object.

    Raises:
        FileNotFoundError: If `config_path` does not point to an existing file.
        ValueError: If the configuration cannot be parsed or is otherwise invalid.
    """
    try:
        with open(config_path, 'rb') as f:
            raw = f.read()
        # Config files are UTF-8 (some carry non-ASCII comments, e.g. the
        # tsurumeso vr6 family has Russian text); opening with the locale
        # default (cp1252 on Windows) choked on those. Decode explicitly with
        # sensible fallbacks so any legacy cp1252/latin-1 config still loads.
        text = None
        for encoding in ('utf-8', 'cp1252', 'latin-1'):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode('utf-8', errors='replace')
        if model_type == 'htdemucs':
            config = OmegaConf.load(config_path)
        else:
            config = ConfigDict(yaml.load(text, Loader=yaml.FullLoader))
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def _load_custom_backend(model_type, config, custom_backend):
    """Build the model from an author-provided backend .py (fork architectures).

    Some models ship their own architecture file alongside the checkpoint
    (e.g. pcunwa's BS-Roformer-Large-Inst / HyperACE publish a bs_roformer.py
    next to the weights). Loading the class from that file lets those forks
    run without vendoring code into the app. The folder is expected to
    contain the backend file named bs_roformer.py.
    """
    import importlib.util, sys
    # Fork authors name their side-car differently (pcunwa ships
    # bs_roformer.py, others model.py); try the known names in order.
    backend_file = ""
    for name in ("bs_roformer.py", "model.py", "models.py"):
        candidate = os.path.join(custom_backend, name)
        if os.path.isfile(candidate):
            backend_file = candidate
            break
    if not backend_file:
        raise ImportError(f"Custom backend module not found in {custom_backend}")
    sys.path.insert(0, custom_backend)
    try:
        spec = importlib.util.spec_from_file_location("custom_backend", backend_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    CLASS_NAME_MAP = {
        "bs_roformer": "BSRoformer", "bs_conformer": "BSConformer",
        "mel_band_roformer": "MelBandRoformer", "mel_band_conformer": "MelBandConformer",
        "bs_roformer_experimental": "BSRoformer",
        "mel_band_roformer_experimental": "MelBandRoformer",
        "bs_mamba2": "BSMamba2Model",
        "scnet": "SCNet", "scnet_unofficial": "SCNet",
        "apollo": "BaseModel", "bandit": "MultiMaskMultiSourceBandSplitRNNSimple",
        "htdemucs": "get_model", "mdx23c": "TFC_TDF_net",
        # DTTNet (upstream Music-Source-Separation-Training spells it
        # both 'dttnet' and 'dtt_net'); the top-level model class is DPTDFNet.
        "dtt_net": "DPTDFNet", "dttnet": "DPTDFNet",
    }
    class_name = CLASS_NAME_MAP.get(model_type, "BSRoformer")
    model_class = getattr(module, class_name, None)
    if model_class is None:
        available = [n for n in dir(module) if not n.startswith('_')]
        raise ImportError(
            f"Class '{class_name}' not found in custom backend.\n"
            f"Available exports: {', '.join(available[:20])}")
    if model_type in ('htdemucs',):
        model = model_class(config)
    elif model_type in ('mdx23c', 'dtt_net', 'dttnet', 'segm_models',
                        'torchseg', 'swin_upernet',
                        'experimental_mdx23c_stht'):
        model = model_class(config)
    elif model_type == 'apollo':
        model = model_class.apollo(**_fit_model_kwargs(model_class.apollo, dict(config.model)))
    elif model_type == 'bandit':
        model = model_class(**_fit_model_kwargs(model_class, dict(config.model)))
    elif model_type == 'conformer':
        from models.conformer_model import NeuralModel
        model = model_class(
            core=NeuralModel(**_fit_model_kwargs(NeuralModel, dict(config.model))),
            n_fft=config.stft.n_fft,
            hop_length=config.stft.hop_length,
            win_length=getattr(config.stft, 'win_length', config.stft.n_fft),
            center=config.stft.center)
    else:
        model = model_class(**_fit_model_kwargs(model_class, dict(config.model)))
    return model, config


def _resolve_bs_roformer_variant(config, checkpoint_path=None):
    """Build the right BS-Roformer class for a config (+ optionally checkpoint).

    Several distinct architectures register under the single "BS Roformer
    Architecture" group, and their checkpoints are only strictly loadable by
    the matching class:

    * top-level ``conformer: true``  -> BSConformer
    * top-level ``siamese: true``    -> BSRoformer(siamese=True) two-stream trunk
    * top-level ``sw: true``         -> BSRoformerSW(learned positions)
    * 6-stem configs with no flag    -> BSRoformerSW(rope) shared-bias "Logic"
    * unwa's "Instrumental Large v2" fork adds a 4-layer axial TransformerBlock
      inside the MaskEstimator, so its state dicts carry
      ``mask_estimators.N.layers...`` / ``mask_estimators.N.norm.gamma`` keys
      the stock classes never create. When no explicit marker is present and
      a checkpoint is available, the checkpoint's own keys are sniffed and
      the vendored fork class (models/bs_roformer/bs_roformer_unwa_large.py)
      is used so already-installed fork checkpoints keep loading even without
      their author side-car file.
    """
    from models.bs_roformer import BSRoformer, BSConformer
    from models.bs_roformer.bs_roformer_sw import BSRoformerSW

    _model_cfg = dict(config.model)
    if getattr(config, 'conformer', None) is True:
        return BSConformer(**_fit_model_kwargs(BSConformer, _model_cfg))
    if getattr(config, 'siamese', None) is True:
        fit = _fit_model_kwargs(BSRoformer, _model_cfg)
        return BSRoformer(siamese=True, **fit)
    if getattr(config, 'sw', None) is True:
        fit = _fit_model_kwargs(BSRoformerSW, _model_cfg)
        return BSRoformerSW(position_mode='learned', **fit)
    if _model_cfg.get('num_stems', 1) == 6:
        fit = _fit_model_kwargs(BSRoformerSW, _model_cfg)
        return BSRoformerSW(position_mode='rope', **fit)
    if checkpoint_path:
        fork_cls = _sniff_unwa_large_fork(checkpoint_path)
        if fork_cls is not None:
            fit = _fit_model_kwargs(fork_cls, _model_cfg)
            return fork_cls(**fit)
    return BSRoformer(**_fit_model_kwargs(BSRoformer, _model_cfg))


def _read_ckpt_keys(checkpoint_path):
    """Read a checkpoint's state-dict key names without materializing any
    tensor data, using zipfile + pickle interception on the archive's
    ``data.pkl`` member.

    Why: the sniffers used to ``torch.load`` the checkpoint just to look at
    key names, and inference.py loads the same checkpoint again right after.
    torch 2.11 on Windows returns a *broken* mapping from the second mmap
    load of a ZIP64 (>4 GB) archive — reading any storage from that second
    load dies with a native access violation. Reading keys from the pickle
    alone means the engine performs exactly one torch.load per checkpoint.

    Handles old (``data.pkl`` at any prefix, e.g. Lightning checkpoints
    like ``last_mel_band_roformer/data.pkl``) and new
    (``archive/data.pkl``) torch zip layouts, and multi-archive Lightning
    files by trying every ``data.pkl`` member. Returns the unwrapped key
    list, or None if nothing readable was found.
    """
    import io
    import pickle
    import zipfile

    try:
        with zipfile.ZipFile(checkpoint_path) as z:
            # Read the small metadata pickles inside the with-block; the
            # handle is closed once we leave it.
            payloads = []
            for n in z.namelist():
                if n.endswith('data.pkl'):
                    try:
                        payloads.append(z.read(n))
                    except Exception:
                        continue
    except Exception:
        return None

    class _KeysUnpickler(pickle.Unpickler):
        """Materialize the object *structure* but never tensor storage:
        every torch global is stubbed and storage persistent-ids return
        None, so only plain containers/strings (the keys) survive."""
        def find_class(self, module, name):
            if module.startswith('torch'):
                return lambda *a: None
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return None

    for raw in payloads:
        try:
            obj = _KeysUnpickler(io.BytesIO(raw)).load()
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for wrapper in ('state_dict', 'state', 'model_state_dict'):
            if isinstance(obj.get(wrapper), dict):
                obj = obj[wrapper]
                break
        return list(obj.keys())
    return None


def _torch_load_ckpt_keys(checkpoint_path):
    """Fallback key reader using torch.load (full load, lazy mmap first).
    Only reached when the pickle-interception reader fails on an exotic
    checkpoint layout."""
    try:
        sd = torch.load(checkpoint_path, map_location='cpu', mmap=True,
                        weights_only=True)
    except Exception:
        try:
            sd = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
        except Exception:
            return None
    if not isinstance(sd, dict):
        return None
    for wrapper in ('state_dict', 'state', 'model_state_dict'):
        if isinstance(sd.get(wrapper), dict):
            sd = sd[wrapper]
            break
    return list(sd.keys())


def _sniff_unwa_large_fork(checkpoint_path):
    """Return the vendored fork class if the checkpoint is unwa's Large-Inst.

    The fork's MaskEstimator contains axial TransformerBlocks (rotary
    embeddings, attention qkv/gates, GLU MLPs) plus a final RMSNorm — keys
    like ``mask_estimators.0.layers.0.0.layers.0.0.to_qkv.weight`` and
    ``mask_estimators.0.norm.gamma`` that no other BS-Roformer variant
    produces. Keys are read without materializing tensor data (see
    _read_ckpt_keys). Returns the class, or None if the checkpoint doesn't
    match.
    """
    keys = _read_ckpt_keys(checkpoint_path)
    if keys is None:
        keys = _torch_load_ckpt_keys(checkpoint_path)
    if not keys:
        return None
    has_est_layers = any(k.startswith('mask_estimators.0.layers.')
                         for k in keys)
    has_est_norm = any(k.startswith('mask_estimators.0.norm.') for k in keys)
    if not (has_est_layers and has_est_norm):
        return None
    from models.bs_roformer.bs_roformer_unwa_large import BSRoformer
    return BSRoformer


def _sniff_melband_mask_estimator_depth(checkpoint_path):
    """Return the mask-estimator MLP depth baked into a mel-band checkpoint.

    MelBandRoformer's per-band mask-estimator head is
    ``MLP(dim, dim_in*2, depth=...) -> GLU`` — i.e. ``depth+1`` Linear layers
    named ``mask_estimators.<stem>.to_freqs.<band>.0.<2*i>``. Some checkpoints
    (JazzPear's ``mbr_expl_jazzpear`` among them) were trained with a
    different ``mask_estimator_depth`` than the side-car YAML declares;
    every other weight matches perfectly, so the strict load dies only on the
    head's layer count. Sniffing the actual depth from the checkpoint lets us
    build the matching architecture instead of failing with a wall of
    "missing key / size mismatch" errors.

    Keys are read without materializing tensor data (see _read_ckpt_keys),
    so the later torch.load in inference.py is the only load of the
    checkpoint. Returns None if the checkpoint can't be read or has no
    mel-band mask-estimator keys.
    """
    keys = _read_ckpt_keys(checkpoint_path)
    if keys is None:
        keys = _torch_load_ckpt_keys(checkpoint_path)
    if not keys:
        return None
    # Largest per-band linear index inside the first stem's heads. Linear
    # layers of the MLP live at even indices 0, 2, 4, ... (odd slots are the
    # activations / GLU), so max_idx // 2 == mask_estimator_depth.
    max_linear_idx = -1
    for key in keys:
        if (key.startswith('mask_estimators.0.to_freqs.')
                and key.endswith('.weight')):
            try:
                idx = int(key.rsplit('.', 2)[1])
            except (ValueError, IndexError):
                continue
            if idx > max_linear_idx:
                max_linear_idx = idx
    if max_linear_idx < 0 or max_linear_idx % 2 != 0:
        return None
    return max_linear_idx // 2


def _fit_model_kwargs(cls, kwargs: dict) -> dict:
    """Drop config keys a model constructor doesn't accept.

    The model library ships newer YAML configs than the bundled model code
    sometimes supports (e.g. `sage_attention`, added in a later upstream
    revision). Passing those straight through crashes with an "unexpected
    keyword argument" TypeError. Filtering to the constructor's signature
    makes the app resilient to such config/model version drift — the extras
    are optional features that simply default off. If the signature can't be
    read or the class accepts **kwargs, the config is passed as-is.
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def get_model_from_config(model_type: str, config_path: str,
                          custom_backend: str = None,
                          checkpoint_path: str = None) -> Tuple[nn.Module, Union[ConfigDict, OmegaConf]]:
    """
    Load and instantiate a model using a configuration file.

    Given a `model_type` and a path to a configuration, this function loads the
    configuration (YAML or OmegaConf) and constructs the corresponding model.

    Args:
        model_type (str): Identifier of the model family (e.g., 'mdx23c', 'htdemucs',
            'scnet', 'mel_band_conformer', etc.).
        config_path (str): Filesystem path to the configuration file used to
            initialize the model.
        custom_backend (str, optional): Path to a folder containing an
            author-provided backend file (e.g. bs_roformer.py). When given,
            the model class is loaded dynamically from that file instead of
            the bundled model code — used for fork architectures.
        checkpoint_path (str, optional): Checkpoint path, used only by model
            types whose architecture is built from the checkpoint itself
            (e.g. mdxnet ONNX).

    Returns:
        Tuple[nn.Module, Union[ConfigDict, OmegaConf]]: A tuple containing the
        initialized PyTorch model and the loaded configuration object.

    Raises:
        ValueError: If `model_type` is unknown or model initialization fails.
        FileNotFoundError: If `config_path` does not exist (may be raised by the
            underlying config loader).
        ImportError: If `custom_backend` is provided but the module cannot be
            loaded or the expected class is not found.
    """

    config = load_config(model_type, config_path)
    if 'model_type' in config.training:
        model_type = config.training.model_type
    # Bandit v1 and v2 are indistinguishable by the "Bandit Architecture"
    # label the GUI / ensemble runners pass down (both map to 'bandit'), but
    # their config layouts differ: v1 nests hyper-parameters under `model:`,
    # v2 under `kwargs:` (see config_dnr_bandit_v2_mus64.yaml). Sniff the
    # layout so a v2 config isn't fed to the v1 branch (which would die on
    # the missing `config.model` with KeyError 'model').
    if model_type == 'bandit' and 'model' not in config and 'kwargs' in config:
        model_type = 'bandit_v2'
        # Record the refined type so callers that re-read config.training
        # (e.g. inference.py's args.model_type propagation) stay consistent.
        config.training.model_type = model_type
    # Same collapse problem for SCNet: the "SCNet Architecture" label maps
    # every variant (scnet, scnet_masked, scnet_tran) back to 'scnet', but
    # tran configs nest extra hyper-parameters under `tran_*` keys that the
    # plain SCNet constructor rejects (TypeError: unexpected keyword argument
    # 'tran_attn_dropout') — and building the wrong class would also break
    # the checkpoint load. Sniff the config and refine to the tran variant.
    if (model_type == 'scnet' and hasattr(config, 'model')
            and any(str(k).startswith('tran_') for k in dict(config.model))):
        model_type = 'scnet_tran'
        config.training.model_type = model_type
    # Fork architectures (unwa's "Instrumental Large v2", pcunwa HyperACE,
    # etc.) are NOT special-cased here any more: the engine builds the model
    # from the author's own side-car file via --custom_backend when one was
    # installed with the model. When a side-car is missing or fails to load,
    # the fall back below is a checkpoint-sniffing resolver (see
    # _resolve_bs_roformer_variant): it inspects the state dict's
    # MaskEstimator keys and routes to the vendored fork class
    # (models/bs_roformer/bs_roformer_unwa_large.py) so an already-installed
    # fork checkpoint still loads without its author file.
    if custom_backend:
        try:
            return _load_custom_backend(model_type, config, custom_backend)
        except Exception as exc:
            # A broken/incompatible side-car must never brick the job:
            # warn and fall back to the bundled model code (whose variant
            # sniffing may still handle the architecture).
            print(
                f"WARNING: could not load custom backend '{custom_backend}' ({exc}).\n"
                f"Falling back to the bundled model code."
            )
    if model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        kwargs = dict(config.model)
        # A checkpoint's mask-estimator MLP depth can disagree with the
        # side-car config (e.g. JazzPear's expl model was trained with depth
        # 1 while its YAML says 2). Build with the depth the weights actually
        # have so the strict load succeeds; otherwise every other layer
        # matches and only the head's layer count fails.
        depth = _sniff_melband_mask_estimator_depth(checkpoint_path)
        if depth is not None and kwargs.get('mask_estimator_depth') != depth:
            print(
                f"[mel_band_roformer] checkpoint mask-estimator depth is "
                f"{depth} (config: {kwargs.get('mask_estimator_depth')}) — "
                f"building with {depth}."
            )
            kwargs['mask_estimator_depth'] = depth
        model = MelBandRoformer(**_fit_model_kwargs(MelBandRoformer, kwargs))
    elif model_type == 'mel_band_conformer':
        from models.bs_roformer import MelBandConformer
        model = MelBandConformer(**_fit_model_kwargs(MelBandConformer, dict(config.model)))
    elif model_type == 'mel_band_roformer_experimental':
        from models.bs_roformer.mel_band_roformer_experimental import MelBandRoformer
        model = MelBandRoformer(**_fit_model_kwargs(MelBandRoformer, dict(config.model)))
    elif model_type == 'bs_roformer':
        model = _resolve_bs_roformer_variant(config, checkpoint_path)
    elif model_type == 'bs_roformer_unwa_large':
        from models.bs_roformer.bs_roformer_unwa_large import BSRoformer
        model = BSRoformer(**_fit_model_kwargs(BSRoformer, dict(config.model)))
    elif model_type == 'bs_conformer':
        from models.bs_roformer import BSConformer
        model = BSConformer(**_fit_model_kwargs(BSConformer, dict(config.model)))
    elif model_type == 'bs_roformer_experimental':
        from models.bs_roformer.bs_roformer_experimental import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'bs_mamba2':
        from models.bs_mamba2_code.bs_mamba2 import BSMamba2Model
        model = BSMamba2Model(**dict(config.model))
    elif model_type == 'swin_upernet':
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'bandit_v2':
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == 'scnet_unofficial':
        from models.scnet_unofficial import SCNet
        model = SCNet(**_fit_model_kwargs(SCNet, dict(config.model)))
    elif model_type == 'scnet':
        from models.scnet import SCNet
        model = SCNet(**_fit_model_kwargs(SCNet, dict(config.model)))
    elif model_type == 'scnet_tran':
        from models.scnet.scnet_tran import SCNet_Tran
        model = SCNet_Tran(**_fit_model_kwargs(SCNet_Tran, dict(config.model)))
    elif model_type == 'apollo':
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == 'experimental_mdx23c_stht':
        from models.mdx23c_tfc_tdf_v3_with_STHT import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'dttnet':
        from models.DTTNet import DPTDFNet
        model = DPTDFNet(config)
    elif model_type == 'scnet_masked':
        from models.scnet.scnet_masked import SCNet
        model = SCNet(**_fit_model_kwargs(SCNet, dict(config.model)))
    elif model_type == 'conformer':
        from models.conformer_model import ConformerMSS, NeuralModel
        model = ConformerMSS(
            core=NeuralModel(**config.model),
            n_fft=config.stft.n_fft,
            hop_length=config.stft.hop_length,
            win_length=getattr(config.stft, 'win_length', config.stft.n_fft),
            center=config.stft.center
        )
    elif model_type == 'mel_band_conformer':
        from models.mel_band_conformer import MelBandConformer
        model = MelBandConformer(**config.model)
    elif model_type == 'moises_light':
        from moises_light import MoisesLight
        model = MoisesLight(**dict(config.model))
    elif model_type == 'vr':
        # UVR5 VR-family models (band-split cascaded nets). VRNet is a
        # self-contained wrapper: it registers the network's children directly
        # on itself so raw UVR checkpoints strict-load as-is, and its forward()
        # consumes raw audio (batch, channels, samples) -> (batch, 2, channels,
        # samples) with [primary, secondary] stems matching the config's
        # instrument order.
        from models.vr_arch import VRNet
        model = VRNet(config)
    elif model_type == 'medley_vox':
        # Medley-Vox (Conv-TasNet + STFT) models. build_medley_vox returns the
        # checkpoint-compatible wrapper (online/ema DataParallel layout + the
        # load_state_dict override for the initted/step bookkeeping scalars)
        # and fills in inference defaults (chunk size from seq_dur, fp32).
        from models.medley_vox.medley_vox import build_medley_vox
        model = build_medley_vox(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def get_scheduler(config, optimizer):
    scheduler_name = config.training.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_name == 'linear_scheduler':
        from transformers import get_linear_schedule_with_warmup
        num_training_steps = config.training.num_epochs * config.training.num_steps
        num_warmup_steps = config.training.get('num_warmup_steps', 0)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_name == 'cosine_scheduler':
        num_training_steps = config.training.num_epochs * config.training.num_steps
        num_warmup_steps = config.training.get('num_warmup_steps', 0)
        # restart_cycle_epochs > 0 enables cosine warm restarts with that period
        # (in epochs); the initial warmup applies only to the first cycle.
        cycle_epochs = config.training.get('restart_cycle_epochs', 0)
        if cycle_epochs and cycle_epochs > 0:
            from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
            cycle_steps = cycle_epochs * config.training.num_steps
            num_cycles = max(1, round((num_training_steps - num_warmup_steps) / cycle_steps))
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles
            )
        else:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
    elif scheduler_name == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.get('patience', 10),
                                      factor=config.training.get('reduce_factor', 0.5))
    else:
        available_schedulers = ['linear_scheduler', 'cosine_scheduler', 'ReduceLROnPlateau']
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. "
            f"Available options: {available_schedulers}. "
            f"Check your config.training.scheduler setting."
        )
    scheduler.name = scheduler_name
    return scheduler


def logging(logs: List[str], text: str, verbose_logging: bool = False) -> Union[List[str], None]:
    """
    Print a log message and optionally append it to an in-memory list.

    In Distributed Data Parallel (DDP) contexts, the message is printed only on
    rank 0; when DDP is uninitialized, it prints unconditionally. If
    `verbose_logging` is True, the message is also appended to `logs`.

    Args:
        logs (List[str]): Mutable list to which the message is appended when
            `verbose_logging` is True.
        text (str): The log message to print (rank 0 only under DDP) and
            optionally store.
        verbose_logging (bool, optional): If True, append `text` to `logs`.
            Defaults to False.

    Returns:
        List[str]: The function prints and may mutate `logs` in place.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(text)
        if verbose_logging:
            logs.append(text)
    return logs

def write_results_in_file(store_dir: str, logs: List[str]) -> None:
    """
    Write accumulated log messages to a results file.

    Creates (or overwrites) a `results.txt` file inside `store_dir` and writes
    each entry from `logs` as a separate line. In Distributed Data Parallel (DDP)
    scenarios, writing is intended to occur only on rank 0.

    Args:
        store_dir (str): Directory path where `results.txt` will be saved.
        logs (List[str]): Ordered collection of log lines to write.

    Returns:
        None
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(f'{store_dir}/results.txt', 'w') as out:
            for item in logs:
                out.write(item + "\n")


def manual_seed(seed: int) -> None:
    """
    Initialize random seeds for reproducibility.

    Sets the seed across Python's `random`, NumPy, and PyTorch (CPU and CUDA)
    libraries, and updates the `PYTHONHASHSEED` environment variable. This helps
    ensure deterministic behavior where possible, though some GPU operations
    may still introduce nondeterminism.

    Args:
        seed (int): The seed value to use for all random number generators.

    Returns:
        None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """
    Initialize runtime environment settings.

    Sets random seeds for reproducibility, adjusts PyTorch cuDNN behavior,
    configures multiprocessing with the 'spawn' start method, and ensures
    the results directory exists.

    Args:
        seed (int): Random seed value for deterministic initialization.
        results_path (str): Filesystem path to create for saving results.

    Returns:
        None
    """

    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(results_path, exist_ok=True)


def initialize_environment_ddp(rank: int, world_size: int, seed: int = 0, resuls_path: str = None) -> None:
    """
    Initialize environment for Distributed Data Parallel (DDP) training/validation.

    Sets up the DDP process group, seeds random number generators, configures
    multiprocessing to use the 'spawn' method, and creates a results directory
    if provided.

    Args:
        rank (int): Rank of the current process within the DDP group.
        world_size (int): Total number of processes participating in DDP.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        resuls_path (str, optional): Directory path to create for storing results.
            If None, no directory is created. Defaults to None.

    Returns:
        None
    """
    seed = (seed + int(time.time())) % 55535 + 10000
    setup_ddp(rank, world_size, seed)
    manual_seed(seed)

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)  # force=True prevent errors
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            raise e
    if not (resuls_path is None):
        os.makedirs(resuls_path, exist_ok=True)


def gen_wandb_name(args, config) -> str:
    """
    Generate a descriptive name for a Weights & Biases (wandb) run.

    Combines the model type, a dash-joined list of training instruments,
    and the current date into a single string identifier.

    Args:
        args: Parsed arguments namespace containing at least `model_type`.
        config: Configuration object/dict with a `training.instruments` field.

    Returns:
        str: Formatted run name in the form
            "<model_type>_[<instrument1>-<instrument2>-...]_<YYYY-MM-DD>".
    """

    instrum = '-'.join(config['training']['instruments'])
    time_str = time.strftime("%Y-%m-%d")
    name = '{}_[{}]_{}'.format(args.model_type, instrum, time_str)
    return name


def wandb_init(args: argparse.Namespace, config: Union[ConfigDict, OmegaConf], batch_size: int) -> None:
    """
    Initialize Weights & Biases (wandb) for experiment tracking.

    Depending on the provided arguments, sets up wandb in one of three modes:
    - Offline mode when `args.wandb_offline` is True.
    - Disabled mode when no valid `wandb_key` is provided.
    - Online mode with authentication using `args.wandb_key`.

    Args:
        args (argparse.Namespace): Parsed arguments containing wandb options
            (`wandb_offline`, `wandb_key`, `device_ids`).
        config (Dict): Experiment configuration dictionary to log.
        batch_size (int): Training batch size to include in the run configuration.

    Returns:
        None
    """
    import wandb

    if args.wandb_offline:
        wandb.init(mode='offline',
                   project='msst',
                   name=gen_wandb_name(args, config),
                   config={'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size}
                   )
    elif args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project='msst',
            name=gen_wandb_name(args, config),
            config={'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size}
        )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))              # 0 → OS chooses free port
        return s.getsockname()[1]


def setup_ddp(rank: int, world_size: int, seed: int) -> None:
    """
    Initialize a Distributed Data Parallel (DDP) process group.

    Configures environment variables for the DDP master node, attempts to
    initialize the process group with the NCCL backend (preferred for GPUs),
    and falls back to the Gloo backend if NCCL is unavailable. Also sets the
    current CUDA device to match the process rank.

    Args:
        rank (int): Rank of the current process in the DDP group.
        world_size (int): Total number of processes participating in DDP.
        seed:
    Returns:
        None
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(seed)
    os.environ["USE_LIBUV"] = "0"
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        if dist.get_rank() == 0:
            print(f'NCCL are not available. Using "gloo" backend.')

    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """
    Finalize and clean up a Distributed Data Parallel (DDP) process group.

    Calls `torch.distributed.destroy_process_group()` to release resources
    associated with the current DDP environment.

    Returns:
        None
    """
    dist.destroy_process_group()
