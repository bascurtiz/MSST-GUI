> **MSST-GUI note:** This file is copied from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). The GUI vendors the classic scripts (inference.py, train.py, valid.py) rather than pip install msst / msst.Separator. Use this document to understand upstream CLI and Python APIs; job subprocesses still call the vendored scripts.


# Experiments and run tracking

MSST experiments provide a portable directory structure and optional run
records. They organize model inputs, operation outputs, logs, scripts, and
reproducibility metadata without controlling how a process is launched.

Tracking is available for `msst.inference()`, `msst.valid()`, and
`msst.train()`. It is disabled unless the `experiment` argument is supplied.

## Create an experiment

With Python:

```python
import msst

experiment = msst.init_experiment(
    "experiments/vocals",
    name="vocals-baseline",
)

print(experiment.path)
print(experiment.name)
print(experiment.manifest_path)
```

With the command-line interface:

```bash
msst init experiments/vocals --name vocals-baseline
```

Initialization creates the following structure:

```text
vocals/
├── experiment.yaml
├── configs/
├── checkpoints/
├── jobs/
├── logs/
├── results/
└── runs/
```

| Path | Intended content |
| --- | --- |
| `experiment.yaml` | Stable experiment identity and directory mapping. |
| `configs/` | YAML configurations used by the experiment. |
| `checkpoints/` | Input or selected model checkpoints. |
| `jobs/` | User-owned launch or helper scripts. |
| `logs/` | Console captures and application logs. |
| `results/` | Separated audio, metrics, trained checkpoints, and reports. |
| `runs/` | Automatically generated individual run directories. |

MSST creates the directories but does not move existing files into them and
does not choose operation output paths. The caller remains responsible for
where configs, checkpoints, datasets, logs, and results live.

Initialization refuses to overwrite an existing `experiment.yaml`. Use
`msst.load_experiment(path)` to reopen an existing experiment.

## Experiment manifest

`experiment.yaml` has a small versioned schema:

```yaml
schema_version: 1
name: vocals-baseline
created_at: "2026-01-01T12:00:00+00:00"
paths:
  configs: configs
  checkpoints: checkpoints
  jobs: jobs
  logs: logs
  results: results
  runs: runs
```

`created_at` is stored in UTC. Directory values are relative to the experiment
root so the whole directory can be moved. Loading validates the schema and
name, then recreates any missing standard subdirectories.

## Track inference

```python
import msst

experiment = msst.load_experiment("experiments/vocals")

outputs = msst.inference(
    model_type="bs_roformer",
    config_path="experiments/vocals/configs/model.yaml",
    checkpoint_path="experiments/vocals/checkpoints/model.ckpt",
    input_folder="audio/input",
    output_folder="experiments/vocals/results/separated",
    device_ids=0,
    experiment=experiment,
    run_name="album-inference",
)
```

The tracked artifacts are the audio paths returned by inference.

## Track validation

```python
result = msst.valid(
    model_type="bs_roformer",
    config_path="experiments/vocals/configs/model.yaml",
    checkpoint_path="experiments/vocals/checkpoints/model.ckpt",
    valid_path="datasets/test",
    output_folder="experiments/vocals/results/validation",
    metrics=("sdr", "k_sdr"),
    device_ids=0,
    experiment=experiment,
    run_name="test-set",
)
```

The run record contains aggregate metric values. When `output_folder` is set,
that directory is also registered as an artifact.

## Track training

```python
msst.train(
    model_type="bs_roformer",
    config_path="experiments/vocals/configs/model.yaml",
    checkpoint_path="experiments/vocals/checkpoints/model.ckpt",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="experiments/vocals/results/training",
    device_ids=(0,),
    launcher="standard",
    experiment=experiment,
    run_name="baseline",
)
```

The training result directory is registered as an artifact. Training itself
continues to own checkpoint naming, history, model metadata, and metric files.

## Run directories

Every tracked call creates a unique directory:

```text
runs/
└── 20260101T120000Z-baseline-a1b2c3d4/
    ├── config.yaml
    └── run.json
```

The directory name contains:

1. a UTC timestamp;
2. the normalized `run_name`, or the task name when no label is supplied;
3. a random suffix that prevents collisions between simultaneous starts.

Characters outside letters, digits, `.`, `_`, and `-` are normalized in the
directory label. The original operation parameters remain in `run.json`.

If `config_path` points to a readable file, the run directory receives a
snapshot named `config.yaml`. Checkpoints are hashed and described but are not
copied because they can be large.

## Run record schema

A running record has this general shape:

```json
{
  "schema_version": 1,
  "run_id": "20260101T120000Z-baseline-a1b2c3d4",
  "experiment": "vocals-baseline",
  "task": "train",
  "status": "running",
  "started_at": "2026-01-01T12:00:00+00:00",
  "parameters": {},
  "environment": {
    "msst": "0.1.0",
    "python": "3.12.0",
    "platform": "...",
    "torch": "...",
    "cuda_build": "...",
    "cuda_available": true,
    "gpus": []
  },
  "inputs": {},
  "artifacts": []
}
```

The exact values depend on the operation and environment.

| Field | Meaning |
| --- | --- |
| `schema_version` | Run-record schema understood by MSST. |
| `run_id` | Unique run directory name. |
| `experiment` | Name from `experiment.yaml`. |
| `task` | `inference`, `valid`, or `train`. |
| `status` | `running`, `completed`, or `failed`. |
| `started_at`, `finished_at` | UTC lifecycle timestamps. |
| `parameters` | Serializable operation arguments. |
| `environment` | Package, Python, platform, PyTorch, and device details. |
| `inputs` | Resolved input paths, sizes, hashes, and config snapshot name. |
| `artifacts` | Paths produced or owned by the operation. |
| `result` | Compact structured result when the workflow returns one. |
| `error` | Exception type and message for a failed run. |

Paths and hashes make a record useful for reproducibility, but a run record is
not a backup. Preserve configs, checkpoints, datasets, and artifacts according
to the needs of the project.

## Lifecycle and failure recording

`run.json` is written when the operation starts with `status: running`. On a
normal return it is updated to `completed` and receives `finished_at`. If the
operation raises an exception, the status becomes `failed`, the exception type
and message are recorded, and the original exception is raised to the caller.

Updates use a temporary file followed by replacement, preventing readers from
observing partially written JSON. Separate runs can be created concurrently
because each receives a unique directory.

An interrupted process may leave `status: running`. This accurately indicates
that MSST did not observe normal completion or a handled failure.

## CLI tracking

Inference, validation, and training commands accept the same tracking fields:

```bash
msst inference ... \
  --experiment experiments/vocals \
  --run_name album-inference

msst valid ... \
  --experiment experiments/vocals \
  --run_name test-set

msst train ... \
  --experiment experiments/vocals \
  --run_name baseline
```

The experiment must already exist. Create it once with `msst init`.

## Recommended practices

- Use one experiment directory for runs that answer the same research or
  production question.
- Give runs short semantic names; uniqueness is added automatically.
- Keep the exact config used for a run in `configs/` even though every tracked
  run also receives a snapshot.
- Keep large checkpoint and dataset files outside `runs/`; rely on hashes to
  identify them.
- Write operation outputs under `results/` and console captures under `logs/`.
- Treat `run.json` as append-only provenance after a run finishes.
- Copy or move the complete experiment directory when transferring it; the
  manifest's standard paths are relative to its root.

## Using the same experiment from different workflows

An experiment directory can be used from a Python script, a notebook, the MSST
command-line interface, or another process runner. In every case, pass the same
experiment path and choose the operation's config, checkpoint, dataset, log,
and result paths normally.

This makes `experiment.yaml` and the records under `runs/` a common history for
the project, while each workflow remains free to choose how and where it runs.
