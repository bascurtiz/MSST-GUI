> **MSST-GUI note:** This file is copied from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). The GUI vendors the classic scripts (inference.py, train.py, valid.py) rather than pip install msst / msst.Separator. Use this document to understand upstream CLI and Python APIs; job subprocesses still call the vendored scripts.


# MSST command-line interface

Installing MSST adds the `msst` command. It exposes the same core workflows as
the Python API and is suitable for terminals, shell scripts, notebooks that
invoke commands, and external process runners.

```bash
msst --help
```

```text
usage: msst {init,metadata,check,train,valid,inference} ...
```

Every subcommand has its own help page:

```bash
msst <command> --help
```

## Commands at a glance

| Command | Main inputs | Main output |
| --- | --- | --- |
| `msst inference` | config, checkpoint, audio folder | separated audio files |
| `msst valid` | config, checkpoint, validation dataset | aggregate and per-track metrics |
| `msst train` | config, training and validation datasets | checkpoints and training history |
| `msst check` | planned training arguments | readiness report |
| `msst metadata` | config and training dataset | metadata cache and manifest |
| `msst init` | destination directory | experiment directory and manifest |

Commands return a non-zero exit status for invalid arguments, missing files,
package-specific errors, or failed checks. This makes them safe to use in
automated workflows.

## Inference

```bash
msst inference \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --start_check_point checkpoints/model.ckpt \
  --input_folder audio/input \
  --store_dir results/separated \
  --device_ids 0
```

Important options:

| Option | Meaning |
| --- | --- |
| `--model_type` | Model architecture key. |
| `--config_path` | YAML model configuration. |
| `--start_check_point` | Checkpoint to load. |
| `--input_folder` | Folder scanned recursively for supported audio. |
| `--store_dir` | Output root for separated stems. |
| `--device_ids` | One or more CUDA device indices. |
| `--force_cpu` | Ignore CUDA and use CPU. |
| `--use_tta` | Enable test-time augmentation. |
| `--flac_file` | Write FLAC instead of WAV. |
| `--pcm_type` | Select the output subtype. |
| `--filename_template` | Control relative output names. |
| `--skip_errors` | Continue after an unreadable input. |
| `--experiment` | Attach the run to an experiment directory. |
| `--run_name` | Label the tracked run. |

Run `msst inference --help` for model-specific and LoRA options. Output naming
and supported codecs are described in [Compatibility](compatibility.md).

## Validation

```bash
msst valid \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --start_check_point checkpoints/model.ckpt \
  --valid_path datasets/test \
  --metrics sdr k_sdr \
  --store_dir results/validation \
  --device_ids 0 \
  --verbose
```

`--valid_path` accepts one or more roots. `--device_ids` accepts multiple
indices for parallel validation. `--store_dir` is optional; when omitted, the
command computes metrics without retaining separated stems.

Supported metric names are listed by the command help and exposed in Python as
`msst.SUPPORTED_VALIDATION_METRICS`.

## Training

```bash
msst train \
  --launcher standard \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --start_check_point checkpoints/model.ckpt \
  --data_path datasets/train \
  --valid_path datasets/valid \
  --results_path results/training \
  --dataset_type 1 \
  --num_workers 8 \
  --device_ids 0
```

The required arguments are:

- `--config_path`;
- at least one `--data_path`;
- at least one `--valid_path`;
- `--results_path`.

`--start_check_point` is optional. Omit it when initializing a new model.

### Launchers

```bash
msst train --launcher standard ...
msst train --launcher ddp ...
msst train --launcher accelerate ...
```

| Launcher | Behavior |
| --- | --- |
| `standard` | CPU, one GPU, or PyTorch DataParallel. |
| `ddp` | One DistributedDataParallel worker per requested device. |
| `accelerate` | Uses the Hugging Face Accelerate runtime. |

All launchers consume the same training argument names. Architecture, batch
size, optimizer, schedule, chunk length, and epoch count come from the YAML
config. The CLI controls paths, devices, data loading, checkpoint behavior,
loss overrides, validation, LoRA, and run logging.

Use `msst train --help` as the authoritative list of current choices and
defaults. The parameters are grouped in
[Python API: Training](python_api.md#training).

## Preflight checks

The check command accepts the planned training arguments plus `--mode` and
`--memory_headroom`.

Safe mode:

```bash
msst check \
  --mode safe \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --start_check_point checkpoints/model.ckpt \
  --data_path datasets/train \
  --valid_path datasets/valid \
  --results_path results/training \
  --dataset_type 1 \
  --device_ids 0
```

Full representative train-step:

```bash
msst check \
  --mode full \
  --memory_headroom 0.15 \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --start_check_point checkpoints/model.ckpt \
  --data_path datasets/train \
  --valid_path datasets/valid \
  --results_path results/training \
  --dataset_type 1 \
  --device_ids 0
```

Each finding is printed as `PASS`, `WARNING`, or `ERROR`. The command exits
with an error when any required check fails. Full mode does not write a model
checkpoint.

## Dataset metadata

Write the standard cache below a result directory:

```bash
msst metadata \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --data_path datasets/train \
  --results_path results/training \
  --dataset_type 1 \
  --num_workers 8
```

Or choose the cache path explicitly:

```bash
msst metadata \
  --config_path configs/model.yaml \
  --data_path datasets/train-a datasets/train-b \
  --metadata_path cache/training.pkl \
  --dataset_type 1
```

`--results_path` and `--metadata_path` are mutually exclusive. Use
`--refresh` to rebuild an existing cache and `--quiet` to suppress scan
progress. The command prints the cache path, manifest path, track count, and
fingerprint.

## Initialize an experiment

```bash
msst init experiments/vocals
```

Set a display name independently of the directory name:

```bash
msst init experiments/vocals --name vocals-baseline
```

Initialization fails if `experiment.yaml` already exists. It never overwrites
an existing experiment. See [Experiments](experiments.md) for the complete
layout and tracking behavior.

## Paths and shell quoting

Relative paths are resolved from the command's current working directory. For
repeatable scripts, either change to a known project directory first or pass
absolute paths.

Quote paths containing spaces:

```bash
msst inference \
  --config_path "configs/my model.yaml" \
  --start_check_point "checkpoints/my model.ckpt" \
  --input_folder "audio/new tracks" \
  --store_dir "results/new tracks"
```

Options that accept multiple values, such as `--data_path`, `--valid_path`,
`--device_ids`, and `--metrics`, consume values until the next option begins.

## Choosing between CLI and Python

Use the CLI when one process performs one clearly defined operation. Use the
Python API when an application needs typed return values, in-memory audio,
custom control flow, or a reusable loaded `Separator`.

Both interfaces call the same packaged runtime. See [Python API](python_api.md)
for object types, return values, and exception behavior.
