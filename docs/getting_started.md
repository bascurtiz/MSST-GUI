> **MSST-GUI note:** This file is copied from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). The GUI vendors the classic scripts (inference.py, train.py, valid.py) rather than pip install msst / msst.Separator. Use this document to understand upstream CLI and Python APIs; job subprocesses still call the vendored scripts.


# Getting started with MSST

This guide covers a complete first workflow: install MSST, provide a model,
run inference, validate the checkpoint, check a training setup, and start
training. The examples use BSRoformer, but the workflow is the same for other
model types.

## Requirements

- CPython 3.10, 3.11, 3.12, or 3.13;
- a YAML model configuration;
- a compatible checkpoint for inference or validation;
- audio or a dataset in one of the supported layouts;
- a PyTorch installation suitable for the selected execution device.

The Python package contains model implementations and workflow code. Configs,
checkpoints, datasets, and generated results remain external files.

## Create an environment

Create and activate an isolated environment using the environment manager of
your choice. With the standard library:

```bash
python -m venv .venv
```

Activation commands differ by operating system. After activation, upgrade the
packaging tools:

```bash
python -m pip install --upgrade pip
```

On a CUDA system, install the appropriate PyTorch build first. MSST then adds
the model and workflow code around that PyTorch installation.

Different model families require different third-party libraries. MSST keeps
them in optional **model extras**, so installing one architecture does not pull
in dependencies for every other architecture. The extra normally has the same
name as `model_type`. For example, to use `model_type="bs_roformer"`, install:

```bash
python -m pip install "msst[bs_roformer]"
```

The complete mapping from model types to extras is available in
[Model extras](compatibility.md#model-extras). Multiple model extras can be
installed together:

```bash
python -m pip install "msst[bs_roformer,mel_band_roformer,htdemucs]"
```

To install the dependencies for all supported model families, use:

```bash
python -m pip install "msst[all-models]"
```

`all-models` can be useful for a shared model-testing environment, but it is a
large installation. Dependencies restricted to a narrow platform are not part
of this aggregate extra; they remain in the extra for their architecture. This
allows the generally available models to install even when a specialized model
cannot. For example, BSMamba2 is enabled separately with
`python -m pip install "msst[bs_mamba2]"` on a supported system.

Pip cannot ignore an arbitrary failed dependency inside one installation
transaction. Keeping specialized architectures in separate extras provides a
predictable environment and makes failures local to the requested model.

Training and validation have their own workflow extras. Combine a workflow
extra with the selected model extra:

```bash
python -m pip install "msst[validation,bs_roformer]"
python -m pip install "msst[train,bs_roformer]"
```

Confirm the installation:

```bash
python -c "import msst; print(msst.__version__); print(msst.__file__)"
msst --help
```

`msst.__file__` should point to the active environment, not to an unrelated
source checkout.

## Prepare a workspace

A simple workspace can look like this:

```text
my-project/
├── configs/
│   └── model.yaml
├── checkpoints/
│   └── model.ckpt
├── audio/
│   └── input/
│       └── song.wav
├── datasets/
│   ├── train/
│   ├── valid/
│   └── test/
└── results/
```

The config and checkpoint must describe the same architecture. In particular,
the `model_type`, tensor dimensions, target instruments, and checkpoint stem
layout must be compatible. Keep a published config/checkpoint pair together.

## Run inference

Create `run_inference.py`:

```python
from pathlib import Path

import msst

root = Path(__file__).resolve().parent

outputs = msst.inference(
    model_type="bs_roformer",
    config_path=root / "configs/model.yaml",
    checkpoint_path=root / "checkpoints/model.ckpt",
    input_folder=root / "audio/input",
    output_folder=root / "results/separated",
    device_ids=0,
)

for output in outputs:
    print(output)
```

Run it:

```bash
python run_inference.py
```

The same operation can be launched directly from the console:

```bash
msst inference \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --start_check_point checkpoints/model.ckpt \
  --input_folder audio/input \
  --store_dir results/separated \
  --device_ids 0
```

For one folder, `msst.inference()` is sufficient. For a notebook, service, or
pipeline that processes multiple inputs over time, construct one
`msst.Separator` and reuse it so the checkpoint is loaded only once.

Set `force_cpu=True` to run inference on CPU. CPU execution is useful for
functional checks but may be slow for large models.

See [MSST command-line interface](cli.md#inference) for output format,
filename, error-handling, and tracking options.

## Validate a checkpoint

Install the validation extra and point `valid_path` to a dataset containing
mixtures and reference stems:

```python
import msst

result = msst.valid(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/model.ckpt",
    valid_path="datasets/test",
    metrics=("sdr", "k_sdr"),
    device_ids=0,
    output_folder="results/validation",
    verbose=True,
)

print(result.averages)
```

Console equivalent:

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

`result.averages` contains aggregate metrics. `result.per_track` retains the
individual scores grouped by metric and instrument. Validation dataset naming
and directory rules are documented in
[Dataset types](dataset_types.md#dataset-for-validation). More validation CLI
options are listed in [MSST command-line interface](cli.md#validation).

## Prepare training metadata

Metadata scanning records usable audio paths and lengths. Building it before
training makes dataset problems visible separately from model execution:

```python
import msst

metadata = msst.build_metadata(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    data_path="datasets/train",
    results_path="results/training",
    dataset_type=1,
    num_workers=8,
)

print(metadata.path)
print(metadata.track_count)
print(metadata.fingerprint)
```

Console equivalent:

```bash
msst metadata \
  --model_type bs_roformer \
  --config_path configs/model.yaml \
  --data_path datasets/train \
  --results_path results/training \
  --dataset_type 1 \
  --num_workers 8
```

The result directory receives `metadata_<dataset_type>.pkl` and an adjacent
JSON manifest. Reuse is automatic when the cache signature still matches the
dataset and config. Set `refresh=True` to scan again.

Generating metadata in advance is optional. `msst.train()` creates or reuses
the same cache when training starts. A separate metadata step is useful because
metadata generation is CPU-only and does not require a GPU. Dataset scanning
and cache validation can finish ahead of time, leaving the training process to
start model work immediately instead of spending GPU time on metadata. It also
isolates missing tracks, invalid layouts, and audio decoding errors from the
training run.

## Check the training setup

Run the safe check first:

```python
import msst

report = msst.check(
    mode="safe",
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/model.ckpt",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="results/training",
    device_ids=(0,),
    dataset_type=1,
)

for item in report.items:
    print(f"[{item.status.upper()}] {item.name}: {item.message}")

report.raise_for_errors()
```

Console equivalent:

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

The safe mode checks the config, paths, audio metadata, dependencies,
checkpoint compatibility, selected devices, output writability, and a
conservative memory estimate without running forward or backward.

After it succeeds, use `mode="full"` to execute one representative training
step. The full mode performs forward, configured loss, backward, and one
in-memory optimizer update without writing a checkpoint. On CUDA, inspect
`report.peak_allocated_gib` and `report.peak_reserved_gib`.

## Start training

```python
import msst

msst.train(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/model.ckpt",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="results/training",
    device_ids=(0,),
    launcher="standard",
    dataset_type=1,
    num_workers=8,
)
```

Console equivalent:

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

Omit `checkpoint_path` to initialize a new model. Model architecture, batch
size, chunk length, optimizer, schedule, and epoch count are configured in the
YAML file. Runtime and data-loading options are function arguments.

For multiple devices, choose `launcher="ddp"`. The `accelerate` launcher is
available when the `accelerate` extra and its runtime are used. See
[Python API: Training](python_api.md#training) and
[MSST command-line interface](cli.md#training) for all accepted options.

## Track an experiment

```python
import msst

experiment = msst.init_experiment("experiments/vocals")

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

Tracking records parameters, input hashes, software details, status, results,
and artifacts under the experiment's `runs/` directory. It does not redirect
the operation's output paths. See [Experiments](experiments.md) for the manifest
and run-record formats.

## Next steps

- [Python API](python_api.md) — signatures, parameters, return values, and errors.
- [Command-line interface](cli.md) — command syntax and shell examples.
- [Compatibility](compatibility.md) — model extras, devices, and audio formats.
- [Dataset types](dataset_types.md) — training and validation layouts.
- [Experiments](experiments.md) — portable organization and tracking.
- [Augmentations](augmentations.md) — training augmentation settings.
- [LoRA](LoRA.md) — parameter-efficient fine-tuning.
