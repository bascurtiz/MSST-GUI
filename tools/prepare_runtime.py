"""
tools/prepare_runtime.py
------------------------
Build-time preparation of the bundled Python runtime used for inference
jobs in the frozen app. Produces build/runtime_pristine/ — an embedded
Python 3.11 distribution with pip pre-bootstrapped. The PyInstaller spec
ships it as a data payload; the app copies it to <exe dir>/runtime on
first use and pip-installs the GPU-appropriate PyTorch build into it.

Usage:  python tools/prepare_runtime.py
"""
import os
import sys
import subprocess
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "build", "runtime_pristine")

PY_EMBED_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def download(url, dest):
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def run(cmd):
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


def patch_pth(runtime_dir):
    """Embedded python needs 'import site' + site-packages on its path for
    pip and normal packages to work (ComfyUI-portable style)."""
    pth = os.path.join(runtime_dir, "python311._pth")
    with open(pth, "w", encoding="utf-8") as f:
        f.write("python311.zip\n.\nLib/site-packages\nimport site\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    zip_path = os.path.join(OUT, "_embed.zip")
    if not os.path.isfile(os.path.join(OUT, "python.exe")):
        if not os.path.isfile(zip_path):
            download(PY_EMBED_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(OUT)
        os.remove(zip_path)
        print("embedded python extracted")
    patch_pth(OUT)

    py = os.path.join(OUT, "python.exe")
    chk = subprocess.run([py, "-m", "pip", "--version"],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        get_pip = os.path.join(OUT, "_get-pip.py")
        download(GET_PIP_URL, get_pip)
        rc = run([py, get_pip])
        os.remove(get_pip)
        if rc != 0:
            print("pip bootstrap failed", file=sys.stderr)
            return 1
    if run([py, "-m", "pip", "--version"]) != 0:
        return 1
    print(f"runtime ready: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
