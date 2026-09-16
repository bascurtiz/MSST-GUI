> **MSST-GUI note:** This file is copied from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). The GUI vendors the classic scripts (inference.py, train.py, valid.py) rather than pip install msst / msst.Separator. Use this document to understand upstream CLI and Python APIs; job subprocesses still call the vendored scripts.


# MSST Python API

MSST supports Python 3.10–3.13 and exposes a small typed API for inference,
validation, and training. Configs, checkpoints, and datasets are external
artifacts: they are passed as paths and are not embedded in the wheel.

## Contents

- [Installation](#installation)
- [API at a glance](#api-at-a-glance)
- [Inference](#inference)
- [Validation](#validation)
- [Dataset metadata](#dataset-metadata)
- [Preflight checks](#preflight-checks)
- [Training](#training)
- [Experiments and run tracking](#experiments-and-run-tracking)
- [Typing and errors](#typing-and-errors)

## Installation

Install the CUDA build of PyTorch appropriate for the target GPU first. Then
install the package with the extra matching the model architecture and task:

```bash
pip install "msst[bs_roformer]"
pip install "msst[validation,bs_roformer]"
pip install "msst[train,bs_roformer]"
```

Other common extras are `mel_band_roformer`, `scnet_unofficial`, `gui`,
`lora-peft`, and `lora-loralib`. See the complete
[model compatibility table](compatibility.md#model-extras).

## API at a glance

| Task | Public Python API | Return value | CLI |
| --- | --- | --- | --- |
| Separate one folder | `msst.inference(...)` | `list[str]` of written files | `msst inference` |
| Reuse one loaded model | `msst.Separator(...)` | `dict[str, numpy.ndarray]` per input | — |
| Evaluate a checkpoint | `msst.valid(...)` | `msst.ValidationResult` | `msst valid` |
| Train or fine-tune | `msst.train(...)` | `None`; artifacts are written to disk | `msst train` |
| Build training metadata | `msst.build_metadata(...)` | `msst.MetadataResult` | `msst metadata` |
| Check a training run | `msst.check(...)` | `msst.CheckReport` | `msst check` |
| Initialize experiment | `msst.init_experiment(...)` | `msst.Experiment` | `msst init` |

Workflow functions use keyword-only arguments. Experiment initialization and
loading accept their path positionally. Paths accept `str` or
`os.PathLike[str]`, including `pathlib.Path`. Run any command with `--help` for
the corresponding command-line contract; see the [CLI guide](cli.md) for
complete examples.

## Inference

### One-off folder inference

`msst.inference` loads a model, processes a folder, releases it, and returns
the paths it wrote:

```python
import msst

written = msst.inference(
    model_type="bs_roformer",
    config_path="models/dereverb-v4.yaml",
    checkpoint_path="models/dereverb-v4.ckpt",
    input_folder="audio/input",
    output_folder="audio/stems",
    device_ids=0,
)
```

| Parameter | Type / default | Meaning |
| --- | --- | --- |
| `config_path` | path, required | YAML model configuration. |
| `checkpoint_path` | path, required | Model checkpoint. |
| `input_folder` | path, required | Folder scanned recursively for supported audio. |
| `output_folder` | path, required | Destination for separated stems. |
| `model_type` | `str = "mdx23c"` | Architecture key matching the config and installed extra. |
| `device_ids` | `int \| Sequence[int] = 0` | CUDA device index or indices. |
| `force_cpu` | `bool = False` | Ignore CUDA and use CPU. |
| `use_tta` | `bool = False` | Enable test-time augmentation. |
| `bigshifts` | `int = 1` | Number of shift passes; must be at least one. |
| `extract_instrumental` | `bool = False` | Add an instrumental stem where supported. |
| `disable_detailed_pbar` | `bool = False` | Hide detailed progress output. |
| `draw_spectro` | `float = 0` | Seconds of spectrogram output; zero disables it. |
| `flac_file` | `bool = False` | Write FLAC instead of WAV. |
| `pcm_type` | `"PCM_16" \| "PCM_24" \| "FLOAT" \| None` | Output subtype; defaults to `FLOAT` for WAV and `PCM_24` for FLAC. |
| `filename_template` | `"{relative_path}/{instr}"` | Relative output naming template. |
| `lora_checkpoint_peft` | path or `None` | Optional PEFT LoRA weights. |
| `lora_checkpoint_loralib` | path or `None` | Optional loralib weights. |
| `skip_errors` | `bool = False` | Log unreadable inputs and continue instead of failing the batch. |
| `experiment` | path, `Experiment`, or `None` | Optional local run tracking. |
| `run_name` | `str` or `None` | Optional tracked run label. |

Available template fields are `{instr}`, `{start_time}`, `{file_name}`,
`{dir_name}`, `{relative_path}`, `{extension}`, `{model_type}`, and `{model}`.
Unsafe paths, unknown fields, malformed templates, and output collisions are
rejected. Use `{extension}` when files have the same basename in two codecs.

### Reusing a loaded model

`Separator` is preferable for services, notebooks, and pipelines because its
constructor loads weights only once:

```python
from msst import Separator

with Separator(
    model_type="bs_roformer",
    config_path="models/dereverb-v4.yaml",
    checkpoint_path="models/dereverb-v4.ckpt",
    device_ids=0,
) as separator:
    print(separator.device, separator.sample_rate, separator.instruments)
    first = separator.separate_file("audio/first.wav")
    second = separator.separate_file("audio/second.wav")
```

The constructor accepts the model/device options from `inference`, with
`detailed_progress=True` instead of `disable_detailed_pbar`.

| Member | Purpose |
| --- | --- |
| `sample_rate: int` | Sample rate expected and returned by the model. |
| `instruments: tuple[str, ...]` | Stem names produced by the configuration. |
| `device: str` | Resolved execution device. |
| `separate(audio, sample_rate=None, channels_first=True)` | Separate an in-memory array. |
| `separate_file(path)` | Decode and separate one file without writing it. |
| `separate_folder(input_folder, output_folder, ...)` | Separate a folder and return written paths. |
| `close()` | Release model references and clear the CUDA cache. |

`separate` and `separate_file` return a dictionary from stem name to a NumPy
array shaped `(channels, samples)`. Mono arrays and samples-first arrays are
accepted; set `channels_first=False` for `(samples, channels)`. Input is
resampled when `sample_rate` differs from `separator.sample_rate`.

Calls on one instance are serialized because model-specific implementations
are not assumed to be thread-safe. Use separate instances for deliberate
parallel model execution.

## Validation

`msst.valid` evaluates one checkpoint against reference stems. The dataset
layout and mixture/reference filenames come from the model configuration; see
[validation datasets](dataset_types.md#dataset-for-validation).

```python
import msst

result = msst.valid(
    model_type="bs_roformer",
    config_path="models/dereverb-v4.yaml",
    checkpoint_path="models/dereverb-v4.ckpt",
    valid_path="datasets/dereverb-test",
    metrics=("sdr", "k_sdr"),
    device_ids=0,
)

print(result.averages)              # {"sdr": ..., "k_sdr": ...}
print(result.per_track["sdr"])     # instrument -> individual scores
```

| Parameter | Type / default | Meaning |
| --- | --- | --- |
| `config_path` | path, required | YAML model configuration. |
| `checkpoint_path` | path, required | Checkpoint to evaluate. |
| `valid_path` | path or sequence, required | One or more validation dataset roots. |
| `model_type` | `str = "mdx23c"` | Architecture key. |
| `device_ids` | `int \| Sequence[int] = 0` | CUDA index or indices. |
| `force_cpu` | `bool = False` | Run validation on CPU. |
| `metrics` | metric or sequence = `("sdr",)` | Metrics to calculate. |
| `output_folder` | path or `None` | Optional separated stems and result output. |
| `extension` | `str = "wav"` | Validation audio extension. |
| `use_tta` | `bool = False` | Enable validation-time augmentation. |
| `draw_spectro` | `float = 0` | Seconds of saved spectrograms; zero disables them. |
| `lora_checkpoint_peft` | path or `None` | Optional PEFT LoRA weights. |
| `lora_checkpoint_loralib` | path or `None` | Optional loralib weights. |
| `verbose` | `bool = False` | Print paths, progress, and aggregate metrics. |
| `experiment` | path, `Experiment`, or `None` | Optional local run tracking. |
| `run_name` | `str` or `None` | Optional tracked run label. |

Supported metrics are exposed as `msst.SUPPORTED_VALIDATION_METRICS` and typed
as `msst.ValidationMetric`: `sdr`, `k_sdr`, `si_sdr`, `l1_freq`,
`neg_log_wmse`, `aura_stft`, `aura_mrstft`, `bleedless`, `fullness`, `l1_snr`,
`bleedless_mr`, and `fullness_mr`.

`ValidationResult.averages` is `dict[str, float]`.
`ValidationResult.per_track` is
`dict[str, dict[str, list[float]]]`, organized as metric → instrument → scores.
`ValidationConfig` can be used independently to validate and normalize public
arguments before a run.

## Dataset metadata

`build_metadata` is the metadata scanner used by the API, CLI, and training.
It scans the dataset directly and does not construct a DataLoader.

```python
result = msst.build_metadata(
    config_path="configs/model.yaml",
    data_path=("datasets/train-a", "datasets/train-b"),
    results_path="results/run-01",
    model_type="bs_roformer",
    dataset_type=1,
    num_workers=16,
)
print(result.path, result.track_count, result.fingerprint)
```

| Parameter | Type / default | Meaning |
| --- | --- | --- |
| `config_path` | path, required | Training/model YAML. |
| `data_path` | path or sequence, required | Dataset roots. |
| `results_path` | path or `None` | Receives `metadata_<type>.pkl`. |
| `metadata_path` | path or `None` | Explicit cache path, used instead of `results_path`. |
| `model_type` | `str = "mdx23c"` | Selects the correct config loader. |
| `dataset_type` | `int = 1` | Dataset layout from 1 through 8. |
| `num_workers` | `int` or `None` | Metadata reader worker count. |
| `refresh` | `bool = False` | Ignore existing cached entries. |
| `verbose` | `bool = True` | Print scan progress. |

The adjacent JSON manifest records config hash, resolved roots, entry count,
MSST version, and metadata fingerprint. The training runtime consumes the
pickle cache.

## Preflight checks

There are intentionally exactly two modes:

```python
report = msst.check(
    mode="full",
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="weights/start.ckpt",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="results/run-01",
    device_ids=0,
    dataset_type=1,
    memory_headroom=0.15,
)
report.raise_for_errors()
```

- `safe` checks paths, YAML/model construction, optional dependencies,
  checkpoint compatibility, actual dataset audio metadata, CUDA visibility,
  output writability, and a conservative lower-bound memory estimate. It does
  not execute forward or backward.
- `full` includes all safe checks and runs one canonical batch through forward,
  configured loss, backward, and one in-memory optimizer step. It never writes
  model weights. `peak_allocated_gib` and `peak_reserved_gib` report the measured
  GPU peak, including optimizer-state allocation.

`CheckReport.ok` is false on errors; `errors` and `warnings` contain typed
`CheckItem` values. A successful full step proves that representative batch
fits. Random augmentations and future batches can still differ, so this is not
an absolute OOM guarantee.

## Training

Training uses the packaged runtime through a lazy facade and writes checkpoints,
history, and metadata below `results_path`:

```python
import msst

msst.train(
    model_type="bs_roformer",
    config_path="configs/config_vocals_bs_roformer.yaml",
    checkpoint_path="weights/start.ckpt",
    results_path="results/run_01",
    data_path=("datasets/train_a", "datasets/train_b"),
    valid_path="datasets/validation",
    device_ids=(0, 1),
    launcher="ddp",
    experiment="experiments/vocals-v3",
    num_workers=8,
    wandb_offline=True,
)
```

| Parameter | Type / default | Meaning |
| --- | --- | --- |
| `config_path` | path, required | YAML training/model configuration. |
| `results_path` | path, required | Checkpoints, history, and metadata destination. |
| `data_path` | path or sequence, required | One or more training dataset roots. |
| `valid_path` | path or sequence, required | Validation roots used after every epoch. |
| `model_type` | `str = "mdx23c"` | Architecture key. |
| `checkpoint_path` | path or `None` | Optional initialization/resume checkpoint. |
| `device_ids` | `int \| Sequence[int] = (0,)` | Training CUDA devices. |
| `launcher` | `"standard" \| "ddp" \| "accelerate"` | Runtime; default is `standard`. |
| `experiment` | path, `Experiment`, or `None` | Optional run tracking. |
| `run_name` | `str` or `None` | Optional tracked run label. |
| `**options` | typed scalar/sequence values | Additional names listed below and by `msst train --help`. |

Accepted `**options` use the same names as the CLI:

| Group | Option names |
| --- | --- |
| Resume | `load_optimizer`, `load_scheduler`, `load_epoch`, `load_best_metric`, `load_all_metrics`, `load_all_losses`, `safe_mode`, `load_only_compatible_weights`, `checkpoint_exclude_prefixes` |
| Dataset/loading | `dataset_type`, `num_workers`, `pin_memory`, `seed`, `persistent_workers`, `prefetch_factor` |
| Loss | `loss`, `masked_loss_coef`, `mse_loss_coef`, `l1_loss_coef`, `log_wmse_loss_coef`, `multistft_loss_coef`, `spec_masked_loss_coef`, `spec_rmse_loss_coef`, `l1_snr_loss_coef`, `l1_snr_db_loss_coef`, `stft_l1_snr_db_loss_coef`, `multi_l1_snr_db_loss_coef`, `fullness_penalty_loss_coef`, `bleedless_penalty_loss_coef`, `use_standard_loss` |
| Validation/saving | `pre_valid`, `metrics`, `metric_for_scheduler`, `each_metrics_in_name`, `save_weights_every_epoch` |
| Weights/LoRA | `train_lora_peft`, `train_lora_loralib`, `lora_checkpoint_peft`, `lora_checkpoint_loralib`, `freeze_layers` |
| Weights & Biases | `wandb_key`, `wandb_offline`, `wandb_project`, `wandb_group`, `wandb_run_name`, `wandb_tags`, `wandb_dir` |
| Runtime | `set_per_process_memory_fraction` |

Exact defaults, choices, and CLI descriptions are available from
`msst train --help`; architecture, optimizer, schedule, batch size, segment
length, and epoch settings remain in the YAML config. `train` returns `None`.

The equivalent unified commands are:

```bash
msst train --launcher standard ...
msst train --launcher ddp ...
msst train --launcher accelerate ...
```

DDP sends rank-zero output to the main console and preserves other rank output
in `results_path/ddp_logs/rank_<N>.log`.

The DDP launcher spawns one process per requested `device_ids` entry.
Accelerate process creation remains owned by `accelerate launch`; selecting
`--launcher accelerate` chooses the runtime inside each launched process.

New checkpoints contain compatibility metadata such as `format_version`,
`msst_version`, `model_type`, selected stem, and stem count when known. Configs
and checkpoints have versions independent of the Python distribution.

## Experiments and run tracking

```python
experiment = msst.init_experiment("experiments/dereverb-v5")

msst.train(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="results/training",
    device_ids=(0,),
    experiment=experiment,
    run_name="baseline",
)
```

The initializer creates `configs/`, `checkpoints/`, `jobs/`, `logs/`,
`results/`, and `runs/` plus `experiment.yaml`. It does not create execution
scripts or environment-specific configuration.

Tracking is opt-in. Each call to `train`, `valid`, or `inference` with an
`experiment` creates `runs/<timestamp>-<task>-<id>/run.json`, snapshots the
config, hashes config/checkpoint inputs, records Python/PyTorch/CUDA/GPU and
platform details, and marks the run completed or failed.
The caller still chooses every data, results, and log path.

See [Experiments and run tracking](experiments.md) for the manifest schema,
run-record fields, lifecycle behavior, and recommended layout.

## Typing and errors

The wheel includes `py.typed`, so IDEs, Pyright, and mypy consume the public
annotations directly. Use `help(msst.inference)`, `help(msst.valid)`,
`help(msst.train)`, `help(msst.check)`, or `help(msst.build_metadata)` for the
installed version's docstrings and signatures.

Public package errors inherit from `msst.MSSTError`:

| Error | Meaning |
| --- | --- |
| `ConfigurationError` | Invalid YAML, missing sections, or model/config mismatch. |
| `CheckpointError` | Unreadable weights, incompatible metadata, or tensor mismatch. |
| `AudioFileError` | Unsupported, corrupt, or undecodable audio. |
| `DeviceError` | Invalid CUDA selection or model transfer failure. |
| `OutputFormatError` | Unsupported WAV/FLAC and subtype combination. |
| `OptionalDependencyError` | A model/workflow extra is missing or unsupported. |

Missing paths and invalid array arguments use standard `FileNotFoundError`,
`NotADirectoryError`, `TypeError`, and `ValueError` where appropriate.
