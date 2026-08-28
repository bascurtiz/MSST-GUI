"""
backend/runtime_setup.py
-------------------------
First-run GPU runtime provisioning for the frozen app.

One PyTorch build cannot cover every GPU: the CUDA 12.1 wheel line supports
compute capability 5.0–9.0 (GTX 10-series through RTX 40-series) but has no
kernels for Blackwell, while the CUDA 12.8 line (torch ≥ 2.7) supports
Blackwell (RTX 50-series) but dropped pre-Volta cards. The bundled runtime
therefore installs the correct line per detected GPU on first use:

    any compute capability ≥ 10.0  → CUDA 12.8 wheels (torch ≥ 2.7)
    else NVIDIA (5.0 – 9.0)        → CUDA 12.1 wheels (torch 2.4.1)
    no NVIDIA GPU                  → CPU wheels from PyPI
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

from backend.paths import get_app_dir, get_runtime_dir, get_runtime_python, REPO_ROOT

# Keep the bundled runtime hermetic: installed packages must never be
# resolved from the user's roaming site-packages (version conflicts), and
# inference subprocesses inherit this flag automatically.
if getattr(sys, "frozen", False):
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/{line}"

# torch specs per accelerator line (installed from the PyTorch wheel index)
_TORCH_SPECS = {
    "cu128": ["torch>=2.7.0,<3", "torchaudio"],
    "cu121": ["torch==2.4.1", "torchaudio==2.4.1"],
    "cpu": ["torch", "torchaudio"],
}


def nvidia_smi_query(fields: str):
    """Query nvidia-smi; returns list of value-strings per GPU, or None."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode != 0:
            return None
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return None


def detect_gpus():
    """[(name, compute_cap_float), …] via nvidia-smi; [] when unavailable."""
    rows = nvidia_smi_query("name,compute_cap")
    gpus = []
    for row in rows or []:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpus.append((parts[0], float(parts[1])))
        except ValueError:
            continue
    return gpus


def pick_torch_line(caps):
    """'cu128' | 'cu121' | 'cpu' for the detected compute capabilities."""
    if not caps:
        return "cpu"
    if any(c >= 10.0 for c in caps):
        return "cu128"
    return "cu121"


def runtime_ready():
    """True when the runtime interpreter exists and can import the real
    dependency chain (torch alone passing is not enough: inference imports
    requests/urllib3 via wandb, which is where broken codec bindings crash)."""
    py = get_runtime_python()
    if not os.path.isfile(py):
        return False
    try:
        r = subprocess.run(
            [py, "-c", "import torch, requests"],
            capture_output=True, timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return r.returncode == 0
    except Exception:
        return False


def _probe_torch(py):
    """(version, cuda_build|None, cuda_usable) for the runtime's torch.
    cuda_usable is only meaningful when cuda_build is set; None on probe
    failure (torch not importable yet, etc.)."""
    try:
        r = subprocess.run(
            [py, "-c",
             "import torch;print(torch.__version__, torch.version.cuda, "
             "torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if r.returncode != 0:
            return None
        ver, build, ok = r.stdout.split()[:3]
        return ver, (None if build == "None" else build), ok == "True"
    except Exception:
        return None


def runtime_needs_repair():
    """Runtime exists but the library chain is broken (e.g. missing/broken
    zstd binding picked by urllib3). Installer should offer Repair."""
    py = get_runtime_python()
    if not os.path.isfile(py):
        return False
    try:
        r = subprocess.run(
            [py, "-c", "import requests"],
            capture_output=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return r.returncode != 0
    except Exception:
        return False


def runtime_python_version():
    py = get_runtime_python()
    if not os.path.isfile(py):
        return None
    try:
        r = subprocess.run(
            [py, "-c", "import platform;print(platform.python_version())"],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def _requirements_runtime_file():
    """requirements-runtime.txt ships next to the code (dev) or in the
    read-only payload (frozen)."""
    for base in (REPO_ROOT, os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(base, "requirements-runtime.txt")
        if os.path.isfile(p):
            return p
    return None


def _requirements_hash():
    """SHA-256 of the shipped requirements-runtime.txt (None when absent)."""
    req = _requirements_runtime_file()
    if not req:
        return None
    try:
        with open(req, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _requirements_marker_path():
    return os.path.join(get_runtime_dir(), ".requirements-hash")


def _write_requirements_marker():
    """Record which requirements set the runtime was provisioned with."""
    want = _requirements_hash()
    if not want:
        return
    try:
        with open(_requirements_marker_path(), "w") as f:
            f.write(want)
    except OSError:
        pass


def runtime_requirements_current():
    """True when the runtime was provisioned with the currently shipped
    requirements-runtime.txt. Frozen-only concern: an app update that ships
    new requirements (e.g. a newly required model library) must top up the
    existing runtime before separation jobs can rely on it."""
    if not getattr(sys, "frozen", False):
        return True
    want = _requirements_hash()
    if want is None:
        return True
    try:
        with open(_requirements_marker_path(), "r") as f:
            return f.read().strip() == want
    except OSError:
        return False


def _bootstrap_runtime_dir():
    """Unpack the pristine bundled runtime into the writable app dir.

    Returns the runtime python path, or raises RuntimeError with a
    user-presentable message."""
    runtime = get_runtime_dir()
    py = get_runtime_python()
    if os.path.isfile(py):
        return py
    pristine = os.path.join(REPO_ROOT, "runtime_pristine")
    src_py = os.path.join(pristine, "python.exe")
    if not os.path.isdir(pristine) or not os.path.isfile(src_py):
        # Dev checkout without the payload — fall back to the running
        # interpreter so non-frozen use keeps working.
        if not getattr(sys, "frozen", False):
            return sys.executable
        raise RuntimeError(
            "The bundled Python runtime payload (runtime_pristine) is missing "
            "from the installation.")
    shutil.copytree(pristine, runtime, dirs_exist_ok=True)
    if not os.path.isfile(py):
        raise RuntimeError("Failed to unpack the bundled Python runtime.")
    return py


def _run_pip(py, args, log_cb, cancel_check, line_cb=None):
    # PIP_CONFIG_FILE -> devnull ignores machine/user pip.ini overrides
    # (e.g. NVIDIA PyIndex setups adding unreachable extra indexes); without
    # this every lookup can stall for ~10s on failed DNS retries.
    env = os.environ.copy()
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PYTHONNOUSERSITE"] = "1"
    if args and args[0] == "install":
        # The runtime's Scripts dir is never on PATH; without this flag pip
        # prints a 'scripts are installed in ...Scripts which is not on
        # PATH' WARNING per console script, flooding the setup log.
        args = ["install", "--no-warn-script-location"] + args[1:]
    cmd = [py, "-m", "pip"] + args + ["--retries", "3"]
    log_cb(" ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # Quiet log policy: every pip line feeds the progress counter via
    # line_cb, but only errors/warnings and the end-of-phase summary reach
    # the log — pip's per-package chatter (Collecting/Downloading/Using
    # cached/Installing) floods the dialog without telling the user more
    # than the progress bar already does.
    in_error = False
    for raw_line in proc.stdout:
        if cancel_check and cancel_check():
            proc.terminate()
            raise RuntimeError("Setup cancelled.")
        line = raw_line.rstrip()
        if not line:
            continue
        s = line.strip()
        if line_cb:
            try:
                line_cb(s)
            except Exception:
                pass
        u = s.upper()
        if u.startswith(("ERROR", "WARNING")):
            # Benign uninstall notices: the zstd cleanup targets stub
            # packages that are absent by design on most machines.
            if _BENIGN_WARN.match(s):
                continue
            in_error = True
            log_cb(line)
        elif in_error and line[:1] in (" ", "\t"):
            log_cb(line)  # indented error context (e.g. resolver details)
        elif s.startswith("Successfully installed"):
            in_error = False
            log_cb(line)
        else:
            in_error = False
    proc.wait()
    return proc.returncode


def _sleep_with_cancel(seconds, cancel_check):
    """Sleep in short slices so Cancel stays responsive. False on cancel."""
    end = time.time() + seconds
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return True
        if cancel_check and cancel_check():
            return False
        time.sleep(min(0.25, remaining))


_PIP_ATTEMPTS = 3
_PIP_BACKOFF_S = 3
_BENIGN_WARN = re.compile(
    r"WARNING: Skipping \S+ as it is not installed\.", re.I)


def _run_pip_retries(py, args, log_cb, cancel_check, line_cb=None):
    """Run a pip install step, retrying failures with a short backoff.

    A retry is a resume, not a restart: pip skips already-satisfied
    requirements and reuses cached wheels, so completed work is never
    redone."""
    for attempt in range(1, _PIP_ATTEMPTS + 1):
        rc = _run_pip(py, args, log_cb, cancel_check, line_cb)
        if rc == 0:
            return 0
        if cancel_check and cancel_check():
            raise RuntimeError("Setup cancelled.")
        if attempt < _PIP_ATTEMPTS:
            delay = _PIP_BACKOFF_S * attempt
            log_cb(f"Step failed (exit code {rc}) — retrying in {delay}s "
                   f"({attempt + 1}/{_PIP_ATTEMPTS}); "
                   f"completed downloads are kept…")
            if not _sleep_with_cancel(delay, cancel_check):
                raise RuntimeError("Setup cancelled.")
    return rc


def install_runtime(log_cb=print, progress_cb=None, cancel_check=None,
                    top_up=False):
    """Install the GPU-appropriate PyTorch build + libraries into the
    bundled runtime. Blocking; run it on a worker thread.

    With top_up=True the PyTorch step is skipped and only
    requirements-runtime.txt is (re)installed — used when an app update
    ships new requirements (pip installs just the missing libraries).

    Returns (True, summary) on success, (False, error) otherwise.
    """
    py = _bootstrap_runtime_dir()

    if top_up:
        req = _requirements_runtime_file()
        if not req:
            return False, "requirements-runtime.txt is missing from the installation."
        log_cb("Updating runtime libraries (PyTorch already installed)…")
        rc = _run_pip_retries(py, ["install", "-r", req], log_cb, cancel_check)
        if rc != 0:
            return False, (f"Library update failed (exit code {rc}). "
                           f"Click Resume to try again — installed packages are kept.")
        _write_requirements_marker()
        if not runtime_ready():
            return False, "Runtime updated but 'import torch' failed."
        return True, "Runtime libraries updated."

    gpus = detect_gpus()
    caps = [c for _, c in gpus]
    line = pick_torch_line(caps)

    gpu_txt = ", ".join(f"{n} (CC {c:.1f})" for n, c in gpus) or "no NVIDIA GPU detected"
    log_cb(f"Detected: {gpu_txt}")
    summary = {"cpu": "CPU-only PyTorch",
               "cu121": "PyTorch (CUDA 12.1) — supports GTX 10-series through RTX 40-series",
               "cu128": "PyTorch (CUDA 12.8) — supports RTX 50-series and newer"}[line]
    log_cb(f"Selected build: {summary}")
    log_cb("Installing PyTorch…")

    # Fine-grained progress: pip reports package downloads line by line, so
    # every wheel lands as a small, monotonic nudge inside the phase band.
    def _phase_counter(base, span, cap):
        state = {"n": 0}

        def on_line(line):
            if not progress_cb:
                return
            s = line.lstrip()
            if s.startswith(("Downloading ", "Using cached ")) and (
                    ".whl" in s or ".tar.gz" in s):
                state["n"] += 1
                frac = min(state["n"], cap) / float(cap)
                progress_cb(base + span * frac)

        return on_line

    if progress_cb:
        progress_cb(0.04)

    # 1. PyTorch + torchaudio from the matching wheel index
    torch_args = _TORCH_SPECS[line]
    torch_line_cb = _phase_counter(0.05, 0.45, cap=14)
    if line == "cpu":
        rc = _run_pip_retries(py, ["install", "--upgrade"] + torch_args,
                              log_cb, cancel_check, line_cb=torch_line_cb)
    else:
        index = _TORCH_CUDA_INDEX.format(line=line)
        rc = _run_pip_retries(py, ["install", "--upgrade"] + torch_args +
                              ["--index-url", index],
                              log_cb, cancel_check, line_cb=torch_line_cb)
    if rc != 0:
        return False, (f"PyTorch installation failed (exit code {rc}). "
                       f"Click Resume to try again — completed downloads are kept.")
    if progress_cb:
        progress_cb(0.55)

    # 1b. Remove known-broken zstd stub bindings: urllib3 2.6+ probes them and
    # crashes on import (AttributeError on ZstdError) when the canonical
    # `zstandard` package isn't what it finds first.
    _run_pip(py, ["uninstall", "-y", "backports.zstd", "zstd"],
             log_cb, cancel_check)

    # 2. Remaining inference libraries from PyPI (torch already satisfied)
    req = _requirements_runtime_file()
    if req:
        log_cb("Installing inference libraries…")
        req_line_cb = _phase_counter(0.56, 0.32, cap=40)
        rc = _run_pip_retries(py, ["install", "-r", req],
                              log_cb, cancel_check, line_cb=req_line_cb)
        if rc != 0:
            return False, (f"Library installation failed (exit code {rc}). "
                           f"Click Resume to try again — installed packages are kept.")
        _write_requirements_marker()
    if progress_cb:
        progress_cb(0.93)

    if not runtime_ready():
        return False, "Runtime installed but 'import torch' failed."

    # 3. Verify the CUDA build actually survived and works. Two silent
    # failure modes: a PyPI dependency can pull a CPU-only torch over the
    # pinned CUDA one, and an outdated NVIDIA driver leaves a correct CUDA
    # build unable to initialize. CPU fallback still counts as success —
    # but the completion message must not promise GPU that won't start.
    note = ""
    probe = _probe_torch(py)
    if line != "cpu":
        if probe and probe[1] is None:
            log_cb("A dependency replaced the CUDA PyTorch build with the CPU "
                   "build — reinstalling it from the CUDA wheel index…")
            index = _TORCH_CUDA_INDEX.format(line=line)
            if _run_pip_retries(py, ["install", "--upgrade"] + _TORCH_SPECS[line] +
                                ["--index-url", index],
                                log_cb, cancel_check) == 0:
                probe = _probe_torch(py)
        if not probe or probe[1] is None:
            note = (" A CPU-only PyTorch ended up installed — inference will "
                    "run on CPU. Re-run GPU setup from Settings.")
        elif not probe[2]:
            note = (" CUDA could not be initialized — the NVIDIA driver is "
                    "likely too old for this PyTorch build (>= 531.14 needed "
                    "for CUDA 12.1). Update the driver; inference uses CPU "
                    "meanwhile.")
        if note:
            log_cb(f"WARNING:{note}")
    if progress_cb:
        progress_cb(1.0)
    return True, f"{summary} installed.{note}"


def runtime_status_text():
    """Short human-readable status for the Settings page."""
    if runtime_ready():
        ver = runtime_python_version() or "?"
        gpus = detect_gpus()
        gpu_txt = gpus[0][0] if gpus else "GPU"
        extra = " · library update pending" if not runtime_requirements_current() else ""
        return f"Ready — Python {ver} · {gpu_txt}{extra}"
    if os.path.isfile(get_runtime_python()):
        return "Runtime present but PyTorch is not installed yet."
    return "Not installed — required to run separation jobs."
