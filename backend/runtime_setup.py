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
import os
import shutil
import subprocess
import sys

from backend.paths import get_app_dir, get_runtime_dir, get_runtime_python, REPO_ROOT

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
    """True when the runtime interpreter exists and can import torch."""
    py = get_runtime_python()
    if not os.path.isfile(py):
        return False
    try:
        r = subprocess.run(
            [py, "-c", "import torch"],
            capture_output=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return r.returncode == 0
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


def _run_pip(py, args, log_cb, cancel_check):
    cmd = [py, "-m", "pip"] + args
    log_cb(" ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for line in proc.stdout:
        if cancel_check and cancel_check():
            proc.terminate()
            raise RuntimeError("Setup cancelled.")
        line = line.rstrip()
        if line:
            log_cb(line)
    proc.wait()
    return proc.returncode


def install_runtime(log_cb=print, progress_cb=None, cancel_check=None):
    """Install the GPU-appropriate PyTorch build + libraries into the
    bundled runtime. Blocking; run it on a worker thread.

    Returns (True, summary) on success, (False, error) otherwise.
    """
    py = _bootstrap_runtime_dir()
    gpus = detect_gpus()
    caps = [c for _, c in gpus]
    line = pick_torch_line(caps)

    gpu_txt = ", ".join(f"{n} (CC {c:.1f})" for n, c in gpus) or "no NVIDIA GPU detected"
    log_cb(f"Detected: {gpu_txt}")
    summary = {"cpu": "CPU-only PyTorch",
               "cu121": "PyTorch (CUDA 12.1) — supports GTX 10-series through RTX 40-series",
               "cu128": "PyTorch (CUDA 12.8) — supports RTX 50-series and newer"}[line]
    log_cb(f"Selected build: {summary}")

    if progress_cb:
        progress_cb(0.05)

    # 1. PyTorch + torchaudio from the matching wheel index
    torch_args = _TORCH_SPECS[line]
    if line == "cpu":
        torch_args = torch_args  # PyPI default index already serves CPU wheels
        rc = _run_pip(py, ["install", "--upgrade"] + torch_args, log_cb, cancel_check)
    else:
        index = _TORCH_CUDA_INDEX.format(line=line)
        rc = _run_pip(py, ["install", "--upgrade"] + torch_args +
                      ["--index-url", index], log_cb, cancel_check)
    if rc != 0:
        return False, f"PyTorch installation failed (exit code {rc})."
    if progress_cb:
        progress_cb(0.55)

    # 2. Remaining inference libraries from PyPI (torch already satisfied)
    req = _requirements_runtime_file()
    if req:
        rc = _run_pip(py, ["install", "-r", req], log_cb, cancel_check)
        if rc != 0:
            return False, f"Library installation failed (exit code {rc})."
    if progress_cb:
        progress_cb(0.95)

    if not runtime_ready():
        return False, "Runtime installed but 'import torch' failed."
    if progress_cb:
        progress_cb(1.0)
    return True, f"{summary} installed."


def runtime_status_text():
    """Short human-readable status for the Settings page."""
    if runtime_ready():
        ver = runtime_python_version() or "?"
        gpus = detect_gpus()
        gpu_txt = gpus[0][0] if gpus else "GPU"
        return f"Ready — Python {ver} · {gpu_txt}"
    if os.path.isfile(get_runtime_python()):
        return "Runtime present but PyTorch is not installed yet."
    return "Not installed — required to run separation jobs."
