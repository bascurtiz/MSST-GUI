"""Regression: load_config must not die on non-ASCII YAML configs.

The tsurumeso vr6 configs (vr6_bass_drypaint, vr6_last_baseline_tsurumeso,
vr6_soprano_drypaint) are UTF-8 with Russian comments; opening them with the
locale default (cp1252 on Windows) raised UnicodeDecodeError. Configs must
load as UTF-8, with cp1252/latin-1 fallbacks for legacy files.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.settings import load_config  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    tmp = tempfile.mkdtemp()

    # 1) UTF-8 with non-ASCII comments (the vr6 failure).
    utf8_path = os.path.join(tmp, "utf8_config.yaml")
    with open(utf8_path, "w", encoding="utf-8") as f:
        f.write("# размер: 0 # не используется — русский комментарий\n")
        f.write("model:\n  nn_arch_size: 31191\n")
        f.write("training:\n  instruments: ['Bass', 'Other']\n")
    cfg = load_config("vr", utf8_path)
    check(list(cfg.training.instruments) == ["Bass", "Other"],
          "UTF-8 config must parse with instruments intact")

    # 2) cp1252-encoded legacy config (smart quotes) must still load.
    cp_path = os.path.join(tmp, "cp1252_config.yaml")
    with open(cp_path, "w", encoding="cp1252") as f:
        f.write("# \u201csmart quotes\u201d in a comment\n")
        f.write("model:\n  nn_arch_size: 123\n")
        f.write("training:\n  instruments: ['Vocals', 'Instrumental']\n")
    cfg = load_config("vr", cp_path)
    check(list(cfg.training.instruments) == ["Vocals", "Instrumental"],
          "cp1252 config must load via fallback")

    # 3) Not-UTF-8 file (latin-1/cp1252 byte) must reach the fallback tier
    #    instead of raising UnicodeDecodeError.
    lat_path = os.path.join(tmp, "latin1_config.yaml")
    with open(lat_path, "w", encoding="latin-1") as f:
        f.write("# caf\xe9 (\xe9 = latin-1 e-acute) comment\n")
        f.write("model:\n  nn_arch_size: 456\n")
        f.write("training:\n  instruments: ['Drums', 'Other']\n")
    cfg = load_config("vr", lat_path)
    check(list(cfg.training.instruments) == ["Drums", "Other"],
          "latin-1-encoded config must load via fallback")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())