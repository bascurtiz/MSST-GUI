"""
backend/auto_ensemble_runner.py
Auto Ensemble pipeline — runs inference on multiple models, collects stem outputs,
performs ensemble, saves metadata, and cleans up temp files.
"""
import os
import re
import sys
import json
import shutil
import tempfile
import subprocess
import threading
from datetime import datetime
from PySide6.QtCore import QObject, Signal

from backend.runner import describe_exit_code  # noqa: E402

from backend.paths import REPO_ROOT, APP_DIR, get_python_exe
from backend.runner import ProcessRunner
from backend.audio_names import format_output_filename, INFERENCE_FILENAME_TEMPLATE
from backend.msst_catalog import (
    engine_supported_type_list,
    engine_unsupported_models,
)

TQDM_PATTERN = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

# Arch label → engine type fallback for manually registered models (zoo
# installs carry the precise type in model_type). Kept next to
# _get_model_type so the pre-flight check and the launch path can never
# disagree on what type a model will run as.
ARCH_TO_MODEL_TYPE = {
    "MDX Architecture": "mdx23c",
    "MDX23c Architecture": "mdx23c",
    "MDX-Net Architecture": "mdxnet",
    "VR Architecture": "vr",
    "Demucs Architecture": "htdemucs",
    "BS Roformer Architecture": "bs_roformer",
    "Melband Roformer Architecture": "mel_band_roformer",
    "Medley Vox Architecture": "medley_vox",
    "SCNet Architecture": "scnet",
    "Apollo Architecture": "apollo",
    "Bandit Architecture": "bandit",
    "Conformer Architecture": "conformer",
    "DTTNet Architecture": "dttnet",
    "BSMamba2 Architecture": "bs_mamba2",
    "Swin Upernet Architecture": "swin_upernet",
    "TorchSeg Architecture": "torchseg",
    "VitLarge23 Architecture": "segm_models",
}

STEM_NAME_MAP = {
    "vocals": ["vocals", "vocal", "voice"],
    "instrumental": ["instrumental", "inst", "other", "accompaniment", "no_vocal", "no_vocals"],
    "drums": ["drums", "drum"],
    "bass": ["bass"],
    "piano": ["piano"],
    "guitar": ["guitar"],
    "other": ["other", "rest", "remaining"],
    "karaoke": ["instrumental", "inst", "other", "accompaniment", "no_vocal", "no_vocals"],
    "denoise": ["denoise", "denoised", "clean"],
    "dereverb / deecho": ["dereverb", "deecho", "dry"],
}


def _is_error_line(line):
    error_keywords = ['error:', 'exception', 'traceback', 'assertion', 'failed', 'critical', 'fatal']
    lower = line.lower()
    return any(kw in lower for kw in error_keywords)


def _strip_tqdm(line):
    return TQDM_PATTERN.sub('', line).strip()


# Module-level registry for ensemble runners (same rationale as
# backend/runner.py's registry): a page can be torn down and rebuilt (theme
# switch) or closed while an ensemble is running, and a QThread wrapper
# destroyed while its thread is still winding down corrupts Qt's thread
# state — surfacing later as a random native access violation on the main
# thread. The runner is a plain QObject on a daemon thread, kept referenced
# until the job finishes — no QThread object exists to leak or corrupt.
_ACTIVE_ENSEMBLE_RUNNERS = set()
_ENSEMBLE_RUNNERS_LOCK = threading.Lock()


class AutoEnsembleRunner(QObject):
    stage_changed = Signal(str, int, int)
    model_progress = Signal(str, int)
    ensemble_progress = Signal(int)
    log_line = Signal(str)
    finished = Signal(bool, str, str)
    error = Signal(str)
    model_skipped = Signal(str, str)

    def __init__(self, models, input_path, target_output, ensemble_type, output_dir, parent=None):
        super().__init__()
        self._models = models
        self._input_path = input_path
        self._target_output = target_output
        self._ensemble_type = ensemble_type
        self._output_dir = output_dir
        self._temp_dir = None
        self._cancelled = False
        self._model_outputs = []
        self._skipped_models = []
        self._current_file = None
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API (mirrors the old QThread surface)
    # ------------------------------------------------------------------

    def start(self):
        """Spawn the pipeline on a plain daemon thread (never a QThread)."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        with _ENSEMBLE_RUNNERS_LOCK:
            _ACTIVE_ENSEMBLE_RUNNERS.add(self)
        self._thread = threading.Thread(
            target=self._run_wrapped, name="auto-ensemble", daemon=True)
        self._thread.start()

    def isRunning(self):
        return bool(self._thread and self._thread.is_alive())

    def _run_wrapped(self):
        """Thread entry: runs the pipeline, then releases the registry hold."""
        try:
            try:
                self.run()
            except Exception as exc:  # noqa: BLE001 — never die silently
                import traceback as _tb
                self.log_line.emit(f"FATAL ERROR: {exc}")
                self.log_line.emit(_tb.format_exc())
                try:
                    self.finished.emit(False, f"Error: {exc}", "")
                except RuntimeError:
                    pass
        finally:
            self._running = False
            with _ENSEMBLE_RUNNERS_LOCK:
                _ACTIVE_ENSEMBLE_RUNNERS.discard(self)

    def cancel(self):
        self._cancelled = True

    def _preflight_unsupported(self):
        """Return the selected models whose type has no engine branch (empty
        = all runnable). The job aborts up front on a non-empty result —
        instead of each model dying mid-run with a raw 'Unknown model type'
        traceback — emitting a clear error before any subprocess is spawned."""
        problems = engine_unsupported_models(self._models, ARCH_TO_MODEL_TYPE)
        if not problems:
            return []
        for model, eff in problems:
            name = model.get("name", "Unknown")
            self.log_line.emit(
                f"ERROR: model '{name}' cannot be run: model type '{eff}' "
                "has no branch in this build's inference engine.")
        self.log_line.emit(
            "Supported model types: " + ", ".join(engine_supported_type_list())
            + ".")
        self.log_line.emit(
            "Aborting the ensemble before any model ran: pick another "
            "model, or update the app if it needs a newer engine.")
        return problems

    def run(self):
        unsupported = self._preflight_unsupported()
        if unsupported:
            self.finished.emit(
                False,
                f"Aborted: {len(unsupported)} model(s) cannot run in this build.",
                "",
            )
            return
        self._temp_dir = tempfile.mkdtemp(prefix="msst_auto_ensemble_")

        if isinstance(self._input_path, list):
            files = sorted(f for f in self._input_path if os.path.isfile(f))
        elif os.path.isdir(self._input_path):
            AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.opus', '.wv')
            files = sorted(
                os.path.join(self._input_path, f)
                for f in os.listdir(self._input_path)
                if f.lower().endswith(AUDIO_EXTS)
            )
        else:
            files = [self._input_path] if os.path.isfile(self._input_path) else []

        if not files:
            self._cleanup()
            self.finished.emit(False, "No audio files found in input.", "")
            return

        total_files = len(files)
        succeeded = 0

        for idx, file_path in enumerate(files):
            if self._cancelled:
                self._cleanup()
                self.finished.emit(False, "Cancelled", "")
                return

            self._current_file = file_path
            self._model_outputs = []
            self._skipped_models = []
            self.log_line.emit(f"\n{'='*60}")
            self.log_line.emit(f"Processing file [{idx+1}/{total_files}]: {os.path.basename(file_path)}")
            self.log_line.emit(f"{'='*60}")

            try:
                self._run_inference_stage()
                if self._cancelled:
                    self._cleanup()
                    self.finished.emit(False, "Cancelled", "")
                    return

                valid_pairs = [pair for pair in self._model_outputs if pair[1] is not None]
                if len(valid_pairs) < 2:
                    self.log_line.emit(
                        f"SKIPPED: {os.path.basename(file_path)} — "
                        f"only {len(valid_pairs)} model(s) succeeded (need ≥2)."
                    )
                    continue

                self._run_ensemble_stage(valid_pairs)
                if self._cancelled:
                    self._cleanup()
                    self.finished.emit(False, "Cancelled", "")
                    return

                succeeded += 1
            except Exception as e:
                self.log_line.emit(f"ERROR processing {os.path.basename(file_path)}: {str(e)}")
                continue

        self._save_metadata()
        self._cleanup()
        if succeeded == 0:
            self.finished.emit(False, "All files failed.", "")
        else:
            self.finished.emit(True, f"Ensemble complete. Processed {succeeded}/{total_files} file(s).", self._output_dir)

    def _run_inference_stage(self):
        total = len(self._models)
        self.stage_changed.emit("inference", 0, total)
        self.log_line.emit(f"Starting inference stage with {total} model(s)")
        self.log_line.emit(f"Target output: {self._target_output}")

        for i, model in enumerate(self._models):
            if self._cancelled:
                return

            self.stage_changed.emit("inference", i + 1, total)
            model_name = model.get("name", "Unknown")
            self.log_line.emit(f"[{i+1}/{total}] Running inference with {model_name}...")
            self.log_line.emit(f"  Checkpoint: {model.get('ckpt', 'N/A')}")
            self.log_line.emit(f"  Config: {model.get('yaml', 'N/A')}")
            self.log_line.emit(f"  Architecture: {model.get('arch', 'N/A')}")

            compatible, compat_msg = self._check_model_compatibility(model)
            if not compatible:
                self.log_line.emit(f"  WARNING: {compat_msg}")
                self._model_outputs.append((model, None))
                self._skipped_models.append(model_name)
                self.model_skipped.emit(model_name, compat_msg)
                self.log_line.emit(f"WARNING: {model_name} skipped (incompatible).")
                continue

            self.log_line.emit(f"  Compatibility: {compat_msg}")

            success = False
            for attempt in range(2):
                if self._cancelled:
                    return
                if attempt > 0:
                    self.log_line.emit(f"Retrying {model_name}...")

                output = self._run_single_inference(model)
                if output is not None:
                    self._model_outputs.append((model, output))
                    success = True
                    break

            if not success:
                self._model_outputs.append((model, None))
                self._skipped_models.append(model_name)
                self.model_skipped.emit(model_name, "Inference failed after retry")
                self.log_line.emit(f"WARNING: {model_name} skipped.")

    def _resolve_target(self):
        target = self._target_output
        is_complement = False
        if target.lower().startswith("no "):
            target = target[3:]
            is_complement = True
        return target, is_complement

    def _get_patterns(self, stem):
        stem_lower = stem.lower()
        if stem_lower.startswith("no "):
            base_stem = stem_lower[3:]
            base_patterns = set(STEM_NAME_MAP.get(base_stem, [base_stem]))
            all_patterns = set()
            for pats in STEM_NAME_MAP.values():
                all_patterns.update(pats)
            return list(all_patterns - base_patterns)
        return STEM_NAME_MAP.get(stem_lower, [stem_lower])

    def _get_target_stems_from_yaml(self, yaml_path, base_target, is_complement):
        if not os.path.isfile(yaml_path):
            return None
        import yaml
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.unsafe_load(f)
        except Exception:
            return None
        training = config.get("training", {})
        if not training:
            return None
        instruments = training.get("instruments", [])
        target_instrument = training.get("target_instrument", None)
        if not instruments:
            return None
        if target_instrument:
            if is_complement:
                return [i for i in instruments if i.lower() != target_instrument.lower()]
            return [target_instrument]
        if is_complement:
            return [i for i in instruments if i.lower() != base_target.lower()]
        return [s for s in instruments if s.lower() == base_target.lower()]

    def _check_model_compatibility(self, model):
        yaml_path = model.get("yaml", "")
        if not os.path.isfile(yaml_path):
            return False, "YAML not found"

        import yaml
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.unsafe_load(f)
        except Exception:
            return False, "Failed to parse YAML"

        training = config.get("training", {})
        instruments = training.get("instruments", [])
        target_instrument = training.get("target_instrument", None)

        base_target, is_complement = self._resolve_target()
        target_lower = base_target.lower()

        if target_instrument is None:
            instruments_lower = [i.lower() for i in instruments]
            for inst in instruments_lower:
                if any(p in inst for p in STEM_NAME_MAP.get(target_lower, [target_lower])):
                    return True, f"Dual target model — outputs: {', '.join(instruments)}"
            if model.get("type", "").lower() == target_lower:
                return True, f"Model type '{model.get('type')}' matches target"
            return False, f"Target '{self._target_output}' not found in instruments: {instruments}"

        target_inst_lower = target_instrument.lower() if target_instrument else ""

        if any(p in target_inst_lower for p in STEM_NAME_MAP.get(target_lower, [target_lower])):
            return True, f"Single target: {target_instrument}"

        if target_inst_lower == "other" and target_lower in ["instrumental", "inst", "accompaniment"]:
            return True, f"Target 'other' maps to instrumental"

        if model.get("type", "").lower() == target_lower:
            return True, f"Model type '{model.get('type')}' matches target"

        return False, f"Model targets '{target_instrument}', not '{self._target_output}'"

    def _run_single_inference(self, model):
        model_output = os.path.join(self._temp_dir, model.get("name", "model"))
        os.makedirs(model_output, exist_ok=True)

        ckpt_path = model.get("ckpt", "")
        yaml_path = model.get("yaml", "")

        if not os.path.isfile(ckpt_path):
            self.log_line.emit(f"ERROR: Checkpoint file not found: {ckpt_path}")
            return None

        if not os.path.isfile(yaml_path):
            self.log_line.emit(f"ERROR: YAML file not found: {yaml_path}")
            return None

        input_dir = os.path.join(self._temp_dir, f"input_{model.get('name', 'model')}")
        os.makedirs(input_dir, exist_ok=True)

        input_filename = os.path.basename(self._current_file)
        input_copy = os.path.join(input_dir, input_filename)
        try:
            shutil.copy2(self._current_file, input_copy)
        except Exception as e:
            self.log_line.emit(f"ERROR: Failed to copy input file: {e}")
            return None

        cmd = [
            get_python_exe(), os.path.join(REPO_ROOT, "inference.py"),
            "--model_type", self._get_model_type(model),
            "--config_path", yaml_path,
            "--start_check_point", ckpt_path,
            "--input_folder", input_dir,
            "--store_dir", model_output,
        ]
        # upstream inference.py nests "{file_name}/{instr}" by default; keep the
        # flat layout the stem discovery below walks for.
        cmd += ["--filename_template", INFERENCE_FILENAME_TEMPLATE]
        # Fork architectures: load the model class from the author's
        # backend file (models/custom/<module> under the writable APP_DIR)
        # instead of the bundled code.
        if model.get("custom_backend_enabled"):
            bm = model.get("backend_module", "")
            if bm:
                cmd += ["--custom_backend",
                        os.path.join(APP_DIR, "models", "custom", bm)]

        self.log_line.emit(f"Command: {' '.join(cmd)}")
        self.log_line.emit(f"Working directory: {REPO_ROOT}")
        self.log_line.emit(f"Input folder: {input_dir} (contains only selected audio)")
        self.log_line.emit("---")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONUTF8": "1"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            stdout_lines = []
            stderr_lines = []

            def read_stream(stream, lines, is_stderr=False):
                for line in stream:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    clean = _strip_tqdm(line)
                    if not clean:
                        continue
                    lines.append(clean)
                    if is_stderr and _is_error_line(clean):
                        self.log_line.emit(f"ERR: {clean}")
                    else:
                        self.log_line.emit(clean)

            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, False))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, True))

            stdout_thread.start()
            stderr_thread.start()

            process.wait()
            stdout_thread.join()
            stderr_thread.join()

            if process.returncode == 0:
                self.log_line.emit(f"Inference completed successfully")
                return model_output
            else:
                self.log_line.emit(
                    f"Inference failed with return code: "
                    f"{describe_exit_code(process.returncode)}")
                if stderr_lines:
                    self.log_line.emit(f"Stderr output:")
                    for line in stderr_lines[-20:]:
                        self.log_line.emit(f"  {line}")
                return None
        except Exception as e:
            self.log_line.emit(f"Exception running inference: {str(e)}")
            import traceback
            self.log_line.emit(traceback.format_exc())
            return None

    def _get_model_type(self, model):
        # Precise engine type recorded at install time wins (e.g. 'scnet_tran',
        # 'bandit_v2' — the arch label collapses variants); fall back to the
        # arch map for manually registered models.
        stored = model.get("model_type", "")
        if stored:
            return stored
        return ARCH_TO_MODEL_TYPE.get(model.get("arch", ""), "bs_roformer")

    def _compute_complement(self, input_path, primary_path, output_path):
        import soundfile as sf
        import numpy as np
        mix, sr = sf.read(input_path)
        primary, _ = sf.read(primary_path)
        min_len = min(len(mix), len(primary))
        mix = mix[:min_len]
        primary = primary[:min_len]
        complement = mix - primary
        sf.write(output_path, complement, sr, subtype='FLOAT')

    def _process_complement(self, model, output_dir):
        yaml_path = model.get("yaml", "")
        base_target, _ = self._resolve_target()
        primary_stems = self._get_target_stems_from_yaml(
            yaml_path, base_target, False)
        if not primary_stems:
            return None
        primary_file = None
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.lower().endswith((".wav", ".flac")):
                    fname_lower = f.lower().replace(".wav", "").replace(".flac", "")
                    if any(s.lower() in fname_lower for s in primary_stems):
                        primary_file = os.path.join(root, f)
                        break
            if primary_file:
                break
        if not primary_file:
            return None
        model_name = model.get("name", os.path.basename(output_dir))
        complement_path = os.path.join(
            self._temp_dir, f"{model_name}_complement.wav")
        self._compute_complement(self._current_file, primary_file, complement_path)
        return complement_path

    def _run_ensemble_stage(self, valid_model_outputs):
        self.stage_changed.emit("ensemble", 0, 1)
        self.log_line.emit("Collecting stem files for ensemble...")

        base_target, is_complement = self._resolve_target()

        stem_files = []
        models_with_stems = 0

        for model, output_dir in valid_model_outputs:
            model_name = model.get("name", os.path.basename(output_dir))

            if is_complement:
                comp_path = self._process_complement(model, output_dir)
                if comp_path:
                    models_with_stems += 1
                    self.log_line.emit(
                        f"  Computed complement: {os.path.basename(comp_path)} (from {model_name})")
                    stem_files.append(comp_path)
                else:
                    self.log_line.emit(
                        f"  WARNING: Cannot compute complement for {model_name}")
            else:
                yaml_path = model.get("yaml", "")
                target_stems = self._get_target_stems_from_yaml(
                    yaml_path, base_target, is_complement)
                model_stems = []
                if target_stems:
                    self.log_line.emit(
                        f"  Looking for stems matching: {target_stems} (from {model_name})")
                    for root, dirs, files in os.walk(output_dir):
                        for f in files:
                            if f.lower().endswith((".wav", ".flac")):
                                fname_lower = f.lower().replace(".wav", "").replace(".flac", "")
                                if any(s.lower() in fname_lower for s in target_stems):
                                    model_stems.append(os.path.join(root, f))
                else:
                    patterns = self._get_patterns(self._target_output)
                    self.log_line.emit(
                        f"  Falling back to patterns: {patterns} (from {model_name})")
                    for root, dirs, files in os.walk(output_dir):
                        for f in files:
                            if f.lower().endswith((".wav", ".flac")):
                                fname_lower = f.lower().replace(".wav", "").replace(".flac", "")
                                if any(p in fname_lower for p in patterns):
                                    model_stems.append(os.path.join(root, f))

                if model_stems:
                    models_with_stems += 1
                    for sf in model_stems:
                        self.log_line.emit(f"  Found: {os.path.basename(sf)} (from {model_name})")
                    stem_files.extend(model_stems)
                else:
                    self.log_line.emit(f"  WARNING: No '{self._target_output}' stem found in {model_name}")
                    available = []
                    for root, dirs, files in os.walk(output_dir):
                        for f in files:
                            if f.lower().endswith((".wav", ".flac")):
                                available.append(f)
                    if available:
                        self.log_line.emit(f"    Available files: {', '.join(available)}")

        self.log_line.emit(f"Total stem files collected: {len(stem_files)} from {models_with_stems}/{len(valid_model_outputs)} models")

        if len(stem_files) < 2:
            self.error.emit(f"Not enough stem files for ensemble. Found {len(stem_files)}, need at least 2.")
            return

        ensemble_output = os.path.join(
            self._output_dir,
            format_output_filename(self._current_file, self._target_output))

        type_map = {
            "Average": "avg_wave",
            "Median": "median_wave",
            "Max Spec": "max_fft",
            "Min Spec": "min_fft",
        }
        ensemble_algo = type_map.get(self._ensemble_type, "avg_wave")

        cmd = [
            get_python_exe(), os.path.join(REPO_ROOT, "ensemble.py"),
            "--files", *stem_files,
            "--type", ensemble_algo,
            "--output", ensemble_output,
        ]

        self.log_line.emit(f"Running ensemble ({ensemble_algo}) on {len(stem_files)} files...")
        for sf in stem_files:
            self.log_line.emit(f"  Input: {os.path.basename(sf)}")

        runner = ProcessRunner(cmd, cwd=REPO_ROOT)
        runner.progress.connect(self.ensemble_progress.emit)
        runner.log_line.connect(self.log_line.emit)
        runner.start()
        runner.wait()

        if runner._process and runner._process.returncode == 0:
            self.ensemble_progress.emit(100)
            self.log_line.emit(f"Ensemble output saved to: {ensemble_output}")
        else:
            self.error.emit("Ensemble processing failed.")

    def _save_metadata(self):
        metadata_path = os.path.join(self._output_dir, "ensemble_info.json")

        if isinstance(self._input_path, list):
            input_name = f"{len(self._input_path)} files"
        else:
            input_name = os.path.basename(self._input_path.rstrip("/\\"))

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "input": input_name,
            "target_output": self._target_output,
            "ensemble_type": self._ensemble_type,
            "models": [
                {
                    "name": m.get("name", ""),
                    "arch": m.get("arch", ""),
                    "yaml": m.get("yaml", ""),
                }
                for m in self._models
            ],
            "skipped_models": self._skipped_models,
        }

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log_line.emit(f"Warning: Could not save metadata: {e}")

    def _cleanup(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except OSError:
                pass
            self._temp_dir = None
