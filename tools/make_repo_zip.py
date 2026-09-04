"""
tools/make_repo_zip.py
----------------------
Packages everything that belongs in a published repository into
dist/msst-gui-repo.zip. Excludes user data (checkpoints, settings,
outputs), caches and build artifacts — mirroring .gitignore.

Usage:  python tools/make_repo_zip.py
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "dist")
OUT = os.path.join(OUT_DIR, "msst-gui-repo.zip")

# Directories never published (caches, tool state, build output, user data,
# and doc-site/ which holds live Google OAuth credentials and is unrelated
# to the app).
SKIP_DIRS = {
    "__pycache__", ".git", ".zcode", ".freebuff", ".pytest_cache",
    ".venv", "build", "dist", "runtime", "runtime_test", "iterative_output",
    "test_iterative", "tests_cache", "_dl_tmp", "temp",
    "doc-site", ".idea", ".vscode",
}

# models/ ships as source code only — never checkpoints or user downloads.
MODELS_EXTS = {".py", ".yaml", ".yml", ".json", ".txt", ".md"}

ROOT_FILES = [
    "README.md", "MSST-GUI.iss", "MSST-GUI.spec",
    "main.py", "inference.py", "ensemble.py", "train.py", "valid.py",
    "train_accelerate.py", "train_ddp.py", "valid_ddp.py",
    "requirements_gui.txt", "requirements-runtime.txt",
    "run_gui.bat", "run_install.bat",
]


def skip_dir(name):
    return name in SKIP_DIRS or name.endswith(".egg-info")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)

    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in ROOT_FILES:
            p = os.path.join(ROOT, f)
            if os.path.isfile(p):
                z.write(p, f)
                count += 1

        # Animated demo the README embeds — ship whatever version is current
        # so the link never dangles when the gif gets replaced.
        import glob as _glob
        for p in sorted(_glob.glob(os.path.join(ROOT, "msst-gui-ani-v*.gif"))):
            z.write(p, os.path.basename(p))
            count += 1

        # The ignore rules live in a dot-less file in the working tree (so
        # file managers show it); GitHub only honors the dot-name, so the zip
        # carries it as .gitignore.
        gi_src = os.path.join(ROOT, "gitignore")
        if os.path.isfile(gi_src):
            z.write(gi_src, ".gitignore")
            count += 1

        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if not skip_dir(d)]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext == ".pyc":
                    continue
                src = os.path.join(root, f)
                rel = os.path.relpath(src, ROOT)
                if os.sep not in rel:
                    continue  # root-level files are added explicitly above
                top = rel.split(os.sep)[0]
                if top == "models" and ext not in MODELS_EXTS:
                    continue  # checkpoints / user downloads stay out
                z.write(src, rel.replace(os.sep, "/"))
                count += 1

    size = os.path.getsize(OUT)
    print(f"wrote {OUT} ({size/1024/1024:.1f} MB, {count} files)")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
