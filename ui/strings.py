"""
ui/strings.py
-------------
Central registry for user-facing UI strings that live outside inline widget
code — the action buttons' tooltips to start. English is the source language.

To localize the app, translate the values in this module (or generate a Qt
.ts/.qm from it and feed it through a QTranslator later); every consumer
reads from here, so no widget code needs to change.

Keep each entry one translateable unit: full sentences, no mid-string
template splicing, and the same line breaks the UI should show (tooltips
render each sentence on its own line by convention in this app).
"""

# Inference page — action row.
T_MULTI_SELECT_MODE = (
    "Tick the models to run together as a batch.\n"
    "Every checked model gets its own run."
)

# Training page — action row.
T_EXPORT_WEIGHTS = (
    "Strip a training checkpoint down to the model weights\n"
    "(no optimizer / scheduler / metrics history) for inference."
)

# Page-header Help button.
T_PAGE_HELP_TOOLTIP = "Show Help"

# Shared captions in the Help dialog. Required vs optional stay visually
# distinct (accent card vs muted card) so users do not mix the two lists.
HELP_REQUIRED_CAPTION = "Complete these or the action will not start."
HELP_OPTIONAL_CAPTION = "Already set to usable defaults. Change them only if you need to."

# Per-page Help copy. `required` is the numbered to-do that unblocks the
# page action; `optional` is everything else. Numbering restarts in each
# list so an optional step is never read as "step 5 of the run".
PAGE_HELP = {
    "inference": {
        "title": "INFERENCE",
        "intro": "Split a mix into stems with one or more models.",
        "required": [
            "Choose input audio — a file, several files, or a folder.",
            "Choose an output folder.",
            "Select at least one model in the library.\nTick several to run them as a batch.",
            "Click Separate.",
        ],
        "optional": [
            "Output format — WAV float is the default.\nUse FLAC 16/24 for smaller files.",
            "Stems — leave as all stems,\nor pick a subset (and optionally save the rest).",
            "Device — GPU is used when one is listed.",
            "TTA and Big Shifts — slower, often a bit cleaner.",
            "Spectro preview length, skip-unreadable-files (on by default),\nand a LoRA adapter.",
            "Quality Check — only if you need mvsep-style output names.",
            "Library search, filters, sort,\nand per-checkpoint chunk / overlap / batch.",
        ],
    },
    "training": {
        "title": "TRAINING",
        "intro": "Train or fine-tune a separation model from a YAML config and a dataset.",
        "required": [
            "Pick a model type.",
            "Select a config YAML (premade or your own).",
            "Select the training data folder.",
            "Select the validation folder.",
            "Select a results folder for checkpoints and logs.",
            "Click Train.",
        ],
        "optional": [
            "Batch size, learning rate, epochs, patience, optimizer, losses, and metrics —\nYAML values apply until you override them.",
            "Resume from a checkpoint.",
            "Run Options — launcher (Standard / DDP / Accelerate),\nLoRA, freeze layers, wandb, GPUs, workers.",
            "Edit the YAML, pick a premade model,\nor export weights from a finished checkpoint.",
        ],
    },
    "ensemble": {
        "title": "ENSEMBLE",
        "intro": "Combine several model outputs.\nPick a workflow, then follow that page's Help.",
        "required": [
            "Choose Auto Ensemble, Manual Ensemble,\nor Iterative Ensemble.",
        ],
        "optional": [
            "Read each card before choosing —\nAuto picks compatible models,\nManual blends files you already have,\nIterative runs a multi-pass pipeline.",
            "Use the back chevron on the next page to return here.",
        ],
    },
    "auto_ensemble": {
        "title": "AUTO ENSEMBLE",
        "intro": "Run several compatible models on one mix and blend their stems.",
        "required": [
            "Pick a stem type.",
            "Choose input audio.",
            "Tick at least two models of that stem type.",
            "Click Auto Ensemble.",
        ],
        "optional": [
            "Output folder — defaults to an ensemble folder\nin the working directory.",
            "Ensemble type (average, median, min, max —\nwaveform or spectrogram).",
            "Output format, including MP3 if you want a lossy preview.",
            "Target output name.",
        ],
    },
    "manual_ensemble": {
        "title": "MANUAL ENSEMBLE",
        "intro": "Blend stem files you already exported, with a weight per file.",
        "required": [
            "Add at least two input files —\nthe same stem from different models.",
            "Choose an output file path.",
            "Click Manual Ensemble.",
        ],
        "optional": [
            "Weights per file — equal weights are the default.",
            "Ensemble type — average waveform is the usual starting point.",
            "How Ensemble Works — the side panel explains each blend mode.",
        ],
    },
    "iterative_ensemble": {
        "title": "ITERATIVE ENSEMBLE",
        "intro": "Multi-pass instrumental extraction with local models and optional MVSep API stages.",
        "required": [
            "Choose input audio.",
            "Enable at least one local model, or an MVSep API model.",
            "If any MVSep API model is on, enter your MVSep API key.",
            "Click Iterative Ensemble.",
        ],
        "optional": [
            "Output folder — defaults to iterative_output\nin the working directory.",
            "Iterations, worker count, export format, and overlap.",
            "Restore side, amplify masked, auto-trim, delete previous pass.",
            "Per-model slowdown, post-separate, and finisher variants.",
        ],
    },
    "console": {
        "title": "CONSOLE",
        "intro": "Watch jobs, play output, and read the log.\nJobs are started on other pages.",
        "required_caption": "Nothing to fill in here — this page only shows work started elsewhere.",
        "required": [
            "Start a job from Inference, Training, or Ensemble.\nProgress and files appear here.",
        ],
        "optional_caption": "Use these when you want to inspect or manage output.",
        "optional": [
            "Switch to Log for the raw process output.",
            "Clear or Copy Log.",
            "Stop the running job.",
            "Play, inspect, or delete finished files.",
        ],
    },
    "settings": {
        "title": "SETTINGS",
        "intro": "Add models to the library, or manage the ones already registered.",
        "required": [
            "Pick one path: Local files (checkpoint + YAML),\nURL download (both URLs), or Model Manager (browse and install).",
            "Click Register, Download, or Install — depending on the path you chose.",
        ],
        "optional": [
            "Architecture type — usually detected from the YAML.",
            "Display name and custom backend folder.",
            "Search folders and sort in Model Manager.",
            "Check for updates, or open the GitHub repository.",
        ],
    },
}
