"""backend/iterative_ensemble/runner.py
Iterative Ensemble pipeline runner.
Performs repeated separation passes, ensembles instrumental outputs,
attenuates vocals iteratively, and optionally runs finisher cleanup.
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess
import threading
import time
import re
from datetime import datetime
from PySide6.QtCore import QThread, Signal

from backend.paths import REPO_ROOT, get_python_exe
from backend.mvsep.api_client import MVSepApiClient, MVSepAPIError
from backend.mvsep.models import MVSepModel
from utils.audio_utils import format_output_filename


TQDM_PATTERN = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def _strip_tqdm(line):
    return TQDM_PATTERN.sub('', line).strip()


class IterativeEnsembleRunner(QThread):
    stage_changed = Signal(str, int, int)
    file_progress = Signal(str, int)
    iteration_progress = Signal(int, int)
    model_progress = Signal(str, str, int)
    queue_status = Signal(int, int)
    log_line = Signal(str)
    finished = Signal(bool, str, str)
    error = Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._temp_dir = None
        self._cancelled = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    def cancel(self):
        self._cancelled = True
        self._pause_event.set()

    def pause(self):
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        self._paused = False
        self._pause_event.set()

    def _log(self, msg):
        self.log_line.emit(msg)

    def _wait_if_paused(self):
        while self._paused and not self._cancelled:
            time.sleep(0.1)
            self._pause_event.wait(timeout=0.1)

    def run(self):
        self._temp_dir = tempfile.mkdtemp(prefix="msst_iterative_")
        try:
            self._run_pipeline()
        except Exception as e:
            self._log("FATAL ERROR: " + str(e))
            import traceback
            self._log(traceback.format_exc())
            self.finished.emit(False, "Error: " + str(e), "")
        finally:
            self._cleanup()

    def _run_pipeline(self):
        cfg = self._config
        input_files = cfg.get("input_files", [])
        output_dir = cfg.get("output_dir", "./iterative_output/")
        export_format = cfg.get("export_format", "wav FLOAT")
        iterations = cfg.get("iterations", 4)
        worker_count = cfg.get("worker_count", 3)
        mvsep_token = cfg.get("mvsep_token", "")
        api_no_credits = cfg.get("api_no_credits", True)

        models_local = cfg.get("models_local", {})
        models_api = cfg.get("models_api", {})
        use_slowdown = cfg.get("use_slowdown", {})
        post_separate = cfg.get("post_separate", {})
        auto_trim = cfg.get("auto_trim", False)
        auto_trim_model = cfg.get("auto_trim_model", False)
        restore_side = cfg.get("restore_side", True)
        amplify_masked = cfg.get("amplify_masked", True)
        finisher_variants = cfg.get("finisher_variants", ["mvsep_only"])
        overlap = cfg.get("overlap", 2)
        delete_prev_pass = cfg.get("delete_prev_pass", True)
        cleanup_intermediate = cfg.get("cleanup_intermediate", True)

        os.makedirs(output_dir, exist_ok=True)

        # Initialize MVSep client with error handling
        mvsep_client = None
        if mvsep_token:
            try:
                mvsep_client = MVSepApiClient(mvsep_token, log_callback=self._log)
                self._log("MVSep client initialized successfully")
            except Exception as e:
                self._log("WARNING: MVSep client initialization failed: " + str(e))
                self._log("Continuing with local models only")
                mvsep_client = None

        total_files = len(input_files)
        self._log("Iterative Ensemble: " + str(total_files) + " file(s), " + str(iterations) + " iteration(s)")

        for file_idx, input_file in enumerate(input_files):
            if self._cancelled:
                return

            self._wait_if_paused()
            song_name = os.path.splitext(os.path.basename(input_file))[0]
            self._log("\n" + "=" * 60)
            self._log("Processing [" + str(file_idx+1) + "/" + str(total_files) + "]: " + song_name)
            self.file_progress.emit(song_name, 0)

            song_output = os.path.join(output_dir, song_name)
            os.makedirs(song_output, exist_ok=True)

            current_input = input_file

            for iteration in range(1, iterations + 1):
                if self._cancelled:
                    return
                self._wait_if_paused()

                self.iteration_progress.emit(iteration, iterations)
                self._log("\n--- Iteration " + str(iteration) + "/" + str(iterations) + " ---")

                is_finisher = (iteration == iterations)

                if is_finisher:
                    self.stage_changed.emit("Finisher Pass", iteration, iterations)
                    self._run_finisher_pass(
                        current_input, song_output, song_name, output_dir, input_file,
                        mvsep_client, models_local, models_api, finisher_variants,
                        restore_side, export_format, overlap,
                    )
                else:
                    self.stage_changed.emit("Local Inference", iteration, iterations)
                    self._run_iterative_pass(
                        current_input, song_output, song_name, iteration,
                        models_local, models_api, mvsep_client,
                        use_slowdown, post_separate, auto_trim,
                        restore_side, amplify_masked, overlap,
                    )
                    self.stage_changed.emit("Repeat", iteration, iterations)

                if delete_prev_pass and iteration > 1:
                    prev_pass = iteration - 1
                    prev_dir = os.path.join(song_output, "pass_" + str(prev_pass))
                    if os.path.isdir(prev_dir):
                        try:
                            shutil.rmtree(prev_dir)
                            self._log("Cleaned up pass_" + str(prev_pass))
                        except OSError:
                            pass

                next_pass_input = os.path.join(song_output, "pass_" + str(iteration) + "_input.wav")
                if os.path.isfile(next_pass_input):
                    current_input = next_pass_input
                    self._log("Next pass input: " + os.path.basename(next_pass_input))

                self.file_progress.emit(song_name, int((iteration / iterations) * 100))

            self._log("\nCompleted: " + song_name)
            self.file_progress.emit(song_name, 100)

            if cleanup_intermediate and os.path.isdir(song_output):
                try:
                    shutil.rmtree(song_output)
                    self._log("Cleaned up intermediate files for: " + song_name)
                except OSError as e:
                    self._log("WARNING: could not clean up intermediates: " + str(e))

        self.stage_changed.emit("Output Saved", 1, 1)
        self.finished.emit(True, "Iterative ensemble complete", output_dir)

    def _run_iterative_pass(self, current_input, song_output, song_name, iteration,
                            models_local, models_api, mvsep_client, use_slowdown,
                            post_separate, auto_trim, restore_side, amplify_masked, overlap):
        pass_dir = os.path.join(song_output, "pass_" + str(iteration))
        os.makedirs(pass_dir, exist_ok=True)

        instrumental_files = []

        local_models = [m for m in models_local.values() if m.get("enabled", False)]
        api_models = {k: v for k, v in models_api.items() if v.get("enabled", False)}

        total_models = len(local_models) + len(api_models)
        self._log("Running " + str(total_models) + " model(s) for pass " + str(iteration) + "...")

        # Run local models
        for model_idx, model in enumerate(local_models):
            if self._cancelled:
                return
            self._wait_if_paused()

            self.model_progress.emit(model["name"], "local", int((model_idx / max(total_models, 1)) * 100))
            self._log("  [" + str(model_idx+1) + "/" + str(total_models) + "] " + model["name"] + " (local)...")

            model_output_dir = os.path.join(pass_dir, model["name"].replace(".ckpt", ""))
            os.makedirs(model_output_dir, exist_ok=True)

            success = self._run_local_inference(
                current_input, model, model_output_dir, overlap
            )

            if not success:
                raise RuntimeError(f"Local inference failed for {model['name']}")

            inst_file = self._find_instrumental_stem(model_output_dir, model.get("stem_type", "instrumental"))
            if not inst_file:
                raise RuntimeError(f"No instrumental stem found for {model['name']}")

            instrumental_files.append(inst_file)
            self._log("    -> " + os.path.basename(inst_file))

            if post_separate.get("bs_resurrect") and "resurrect" in model.get("name", "").lower():
                self._log("    Post-separate BS Resurrect vocals...")

        # Run API models
        if api_models:
            self.stage_changed.emit("MVSep API Processing", iteration, self._config.get("iterations", 4))
        for api_name, api_cfg in api_models.items():
            if self._cancelled:
                return
            self._wait_if_paused()

            if not mvsep_client:
                raise RuntimeError("MVSep client not available for API model: " + api_name)

            self.model_progress.emit(api_name, "api", int(((len(local_models) + list(api_models.keys()).index(api_name)) / max(total_models, 1)) * 100))
            self._log("  [" + str(len(local_models)+list(api_models.keys()).index(api_name)+1) + "/" + str(total_models) + "] " + api_name + " (MVSep API)...")

            api_output = os.path.join(pass_dir, api_name + "_output.wav")
            mvsep_model = MVSepModel.BS_ROFORMER_2025_07 if "mvsep" in api_name.lower() else MVSepModel.SCNET_XL_IHF_BECRUILY
            result = mvsep_client.process_file(
                current_input, mvsep_model, api_output,
                upload_progress_cb=lambda p: self.model_progress.emit(api_name, "upload", int(p)),
                process_progress_cb=lambda p: self.model_progress.emit(api_name, "processing", int(p)),
                download_progress_cb=lambda p: self.model_progress.emit(api_name, "download", int(p)),
            )
            instrumental_files.append(result)
            self._log("    -> " + os.path.basename(result))

        if len(instrumental_files) < 1:
            raise RuntimeError("No instrumental files produced in pass " + str(iteration))

        self.stage_changed.emit("Ensemble & Attenuation", iteration, self._config.get("iterations", 4))
        self._log("  Ensemble: " + str(len(instrumental_files)) + " instrumental file(s)")
        ensemble_output = os.path.join(pass_dir, "ensemble_instrumental.wav")
        ensemble_ok = self._run_ensemble(instrumental_files, ensemble_output, "max_fft")
        if not ensemble_ok:
            raise RuntimeError("Ensemble failed in pass " + str(iteration))

        if os.path.isfile(ensemble_output):
            self._log("  Ensemble output: " + os.path.basename(ensemble_output))

            next_input = os.path.join(song_output, "pass_" + str(iteration) + "_input.wav")
            self._attenuate_vocals(current_input, ensemble_output, next_input)

            if os.path.isfile(next_input):
                self._log("  Next pass input created: " + os.path.basename(next_input))

    def _run_finisher_pass(self, current_input, song_output, song_name,
                           output_dir, original_input,
                           mvsep_client, models_local, models_api, variants,
                           restore_side, export_format, overlap):
        finisher_dir = os.path.join(song_output, "finisher")
        os.makedirs(finisher_dir, exist_ok=True)

        self._log("Running finisher pass...")

        if not mvsep_client:
            self._log("WARNING: MVSep client not available — skipping finisher, using last pass output")
            last_pass_dir = os.path.join(song_output, "pass_" + str(self._config.get("iterations", 4) - 1))
            candidate = os.path.join(last_pass_dir, "ensemble_instrumental.wav")
            if os.path.isfile(candidate):
                final_output = os.path.join(
                    output_dir,
                    format_output_filename(original_input, "Iterative finisher"))
                shutil.copy2(candidate, final_output)
                self._log("  Copied " + os.path.basename(candidate) + " -> " + os.path.basename(final_output))
            return

        for variant in variants:
            if self._cancelled:
                return

            self._log("  Finisher variant: " + variant)
            signals_for_ensemble = []

            mvsep_model = MVSepModel.BS_ROFORMER_2025_07
            mvsep_output = os.path.join(finisher_dir, f"finisher_{variant}_mvsep.wav")
            result = mvsep_client.process_file(
                current_input, mvsep_model, mvsep_output,
                process_progress_cb=lambda p: self._log("    Processing: " + str(int(p)) + "%"),
            )
            self._log("    MVSep -> " + os.path.basename(result))
            signals_for_ensemble.append(result)

            if "resurrect" in variant.lower():
                local_bs = None
                for m in models_local.values():
                    if m.get("enabled") and "resurrect" in m.get("name", "").lower():
                        local_bs = m
                        break
                if local_bs:
                    self._log("    + bs_resurrect (local)...")
                    bs_output_dir = os.path.join(finisher_dir, f"finisher_{variant}_resurrect")
                    os.makedirs(bs_output_dir, exist_ok=True)
                    success = self._run_local_inference(current_input, local_bs, bs_output_dir, overlap)
                    if success:
                        bs_file = self._find_instrumental_stem(bs_output_dir, local_bs.get("stem_type", "instrumental"))
                        if bs_file:
                            self._log("    bs_resurrect -> " + os.path.basename(bs_file))
                            signals_for_ensemble.append(bs_file)
                        else:
                            self._log("    WARNING: bs_resurrect instrumental not found")
                    else:
                        self._log("    WARNING: bs_resurrect inference failed")
                else:
                    self._log("    WARNING: no bs_resurrect model enabled for variant " + variant)

            if len(signals_for_ensemble) > 1:
                ensemble_output = os.path.join(finisher_dir, f"finisher_{variant}_ensemble.wav")
                self._log("    Ensembling " + str(len(signals_for_ensemble)) + " signal(s) with max_fft...")
                self._run_ensemble(signals_for_ensemble, ensemble_output, "max_fft")
                final_source = ensemble_output
            else:
                final_source = signals_for_ensemble[0]

            final_output = os.path.join(
                output_dir,
                format_output_filename(original_input, "Iterative " + variant))
            shutil.copy2(final_source, final_output)
            self._log("  Final output: " + os.path.basename(final_output))

    def _run_local_inference(self, input_audio, model, output_dir, overlap):
        ckpt_path = model.get("ckpt", "")
        yaml_path = model.get("yaml", "")

        if not os.path.isfile(ckpt_path) or not os.path.isfile(yaml_path):
            self._log("    Missing files for " + model.get("name", "unknown"))
            return False

        input_dir = os.path.join(self._temp_dir, "iter_input_" + model.get("name", "model"))
        os.makedirs(input_dir, exist_ok=True)
        input_copy = os.path.join(input_dir, os.path.basename(input_audio))
        try:
            shutil.copy2(input_audio, input_copy)
        except Exception as e:
            self._log("    Copy error: " + str(e))
            return False

        arch = model.get("arch", "bs_roformer")
        model_type = self._get_model_type(arch)

        cmd = [
            get_python_exe(), os.path.join(REPO_ROOT, "inference.py"),
            "--model_type", model_type,
            "--config_path", yaml_path,
            "--start_check_point", ckpt_path,
            "--input_folder", input_dir,
            "--store_dir", output_dir,
        ]
        if model.get("custom_backend_enabled"):
            bm = model.get("backend_module", "")
            if bm:
                cmd += ["--custom_backend",
                        os.path.join(REPO_ROOT, "models", "custom", bm)]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", cwd=REPO_ROOT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            def read_stream(stream, prefix=""):
                for line in stream:
                    line = line.rstrip("\n")
                    if line:
                        clean = _strip_tqdm(line)
                        if clean:
                            self._log(prefix + clean)

            stdout_t = threading.Thread(target=read_stream, args=(process.stdout, ""))
            stderr_t = threading.Thread(target=read_stream, args=(process.stderr, "  "))
            stdout_t.start()
            stderr_t.start()
            process.wait()
            stdout_t.join()
            stderr_t.join()

            return process.returncode == 0
        except Exception as e:
            self._log("    Inference error: " + str(e))
            return False

    def _get_model_type(self, arch):
        mapping = {
            "MDX Architecture": "mdx23c",
            "Demucs Architecture": "htdemucs",
            "BS Roformer Architecture": "bs_roformer",
            "Melband Roformer Architecture": "mel_band_roformer",
            "SCNet Architecture": "scnet",
            "Apollo Architecture": "apollo",
            "Bandit Architecture": "bandit",
        }
        return mapping.get(arch, "bs_roformer")

    def _find_instrumental_stem(self, output_dir, stem_type):
        patterns = {
            "instrumental": ["instrumental", "inst", "other", "accompaniment"],
            "vocals": ["vocals", "vocal", "voice"],
        }
        search_patterns = patterns.get(stem_type, [stem_type])

        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.lower().endswith((".wav", ".flac")):
                    fname = f.lower().replace(".wav", "").replace(".flac", "")
                    if any(p in fname for p in search_patterns):
                        return os.path.join(root, f)

        return None

    def _run_ensemble(self, files, output, algo="max_fft"):
        if len(files) < 2:
            if len(files) == 1:
                shutil.copy2(files[0], output)
                return True
            return False

        cmd = [
            get_python_exe(), os.path.join(REPO_ROOT, "ensemble.py"),
            "--files", *files,
            "--type", algo,
            "--output", output,
        ]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", cwd=REPO_ROOT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            stdout, stderr = process.communicate()
            for line in stdout.splitlines():
                if line.strip():
                    self._log("    " + line.strip())
            return process.returncode == 0
        except Exception as e:
            self._log("    Ensemble error: " + str(e))
            return False

    def _attenuate_vocals(self, mixture_path, instrumental_path, output_path):
        try:
            import numpy as np
            import soundfile as sf

            mix, sr = sf.read(mixture_path)
            inst, sr2 = sf.read(instrumental_path)

            if sr != sr2:
                self._log("    Sample rate mismatch: " + str(sr) + " vs " + str(sr2))
                return

            min_len = min(len(mix), len(inst))
            mix = mix[:min_len]
            inst = inst[:min_len]

            if mix.ndim == 1:
                mix = np.column_stack([mix, mix])
            if inst.ndim == 1:
                inst = np.column_stack([inst, inst])

            vocal_estimate = mix - inst
            vocal_attenuated = vocal_estimate * 0.5
            new_mix = inst + vocal_attenuated

            sf.write(output_path, new_mix, sr, subtype='FLOAT')
            self._log("    Vocals attenuated by 50%")
        except ImportError:
            self._log("    WARNING: numpy/soundfile not available, skipping vocal attenuation")
            shutil.copy2(instrumental_path, output_path)
        except Exception as e:
            self._log("    Vocal attenuation error: " + str(e))
            shutil.copy2(instrumental_path, output_path)

    def _cleanup(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except OSError:
                pass
            self._temp_dir = None
