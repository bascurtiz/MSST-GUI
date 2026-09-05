"""Regression test: the engine (inference.py) imports backend.audio_names.

The GUI's inference page passes the Quality Checker Test naming rules to the
engine via CLI args, and the engine resolves them with helpers in
backend/audio_names.py. But the engine runs as a *subprocess* from plain data
files in _internal/ (runtime python), where the frozen PYZ of the GUI does not
exist — so backend/audio_names.py must be shipped as data next to the engine.

This test pins both halves:

  * MSST-GUI.spec ships backend/__init__.py + backend/audio_names.py as data,
  * a bare python process whose sys.path contains only the engine layout (no
    PyInstaller, no GUI modules) can import and use the helpers — the exact
    situation the engine subprocess is in.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    # ── 1) the spec ships the engine-needed backend files as data ──────
    spec = open(os.path.join(ROOT, "MSST-GUI.spec"), encoding="utf-8").read()
    check('("backend", "__init__.py")' in spec.replace('"', "'").replace(
        "\\", "/") or 'os.path.join(ROOT, "backend", "__init__.py")' in spec,
        "spec ships backend/__init__.py as data")
    check('os.path.join(ROOT, "backend", "audio_names.py")' in spec,
          "spec ships backend/audio_names.py as data")

    # ── 2) a bare interpreter with only the engine layout can use them ──
    tmp = tempfile.mkdtemp(prefix="msst_engine_bk_")
    try:
        os.makedirs(os.path.join(tmp, "backend"))
        for f in ("__init__.py", "audio_names.py"):
            shutil.copyfile(os.path.join(ROOT, "backend", f),
                            os.path.join(tmp, "backend", f))
        probe = (
            "import sys\n"
            "sys.path.insert(0, r'%s')\n"
            "from backend.audio_names import (\n"
            "    strip_mixture_name, parse_stem_suffix_map, stem_suffix_for)\n"
            "assert strip_mixture_name('song_dnr_016_mixture') == 'song_dnr_016'\n"
            "m = parse_stem_suffix_map('effects=sfx,speech=speech,music=music')\n"
            "assert stem_suffix_for('effects', m) == 'sfx'\n"
            "assert stem_suffix_for('speech', m) == 'speech'\n"
            "assert strip_mixture_name('song_plain') == 'song_plain'\n"
            "print('ENGINE_IMPORT_OK')\n" % tmp.replace("\\", "\\\\"))
        r = subprocess.run([sys.executable, "-c", probe],
                           capture_output=True, text=True)
        check(r.returncode == 0 and "ENGINE_IMPORT_OK" in r.stdout,
              "bare engine-layout import works: %r %r" % (r.stdout, r.stderr))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("ALL %d CHECKS PASSED" % CHECKS if not FAILURES
          else "%d FAILURES: %s" % (len(FAILURES), "; ".join(FAILURES)))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()