"""Regression: dataset_type-1 stem files must resolve flexibly.

Community pair datasets (SDR30+ 'Remixpack' etc.) name stems like
``Track - Lead Vocals.flac`` / ``Track (Instrumental).flac`` instead of the
exact ``vocals.wav`` / ``other.wav`` the old loader demanded — a mismatch that
made every folder fail and then crashed on an empty numpy reduction. The
loader now matches stem names case-insensitively with word boundaries
(``_find_instrument_file``), understands community aliases
(``acapella``/``vocal`` for vocals, ``instrumental``/``instrument``/``instr``
for the 'other' complement stem), reports unmatched folders with the actual
file list, and never crashes on a folder with no matches.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from ml_collections import ConfigDict  # noqa: E402

from utils.dataset import (  # noqa: E402
    MSSDataset, _find_instrument_file, get_track_set_length)

FAILURES = []
CHECKS = 0
FTYPES = ["wav", "flac", "mp3"]


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def write_stem(folder, name, frames=4410):
    path = os.path.join(folder, name)
    sf.write(path, np.random.randn(frames).astype("float32"), 44100)
    return path


def main():
    tmp = tempfile.mkdtemp(prefix="stem_match_")
    try:
        # Community 'pairs' naming: stems embedded in the file name.
        track = os.path.join(tmp, "Artist - Song")
        os.makedirs(track)
        write_stem(track, "Artist - Song - Lead Vocals.flac")
        write_stem(track, "Artist - Song (Instrumental).flac")

        check(_find_instrument_file(track, "vocals", FTYPES)
              == "Artist - Song - Lead Vocals.flac",
              "vocals must match the '- Lead Vocals' suffix form")
        check(_find_instrument_file(track, "other", FTYPES)
              == "Artist - Song (Instrumental).flac",
              "other must match the '(Instrumental)' alias form")
        length = get_track_set_length((track, ["vocals", "other"], FTYPES, 1))[1]
        check(int(length) > 0, "matched stems must report a real length")

        # Exact <instr>.<ext> files win over substring forms; a folder with
        # only one of the two stems is unusable (skipped, never a crash).
        exact = os.path.join(tmp, "Exact")
        os.makedirs(exact)
        write_stem(exact, "vocals.wav")
        write_stem(exact, "Exact - Vocals.flac")
        check(_find_instrument_file(exact, "vocals", FTYPES) == "vocals.wav",
              "exact '<instr>.<ext>' must win over substring matches")
        check(get_track_set_length((exact, ["vocals", "other"], FTYPES, 1))[1] == 0,
              "partially-matched folder must be skipped (length 0)")

        # Alias coverage: (Acapella) vocal + (Instr) backing.
        aliased = os.path.join(tmp, "Aliased")
        os.makedirs(aliased)
        write_stem(aliased, "Aliased (Acapella).flac")
        write_stem(aliased, "Aliased (Instr).flac")
        check(_find_instrument_file(aliased, "vocals", FTYPES).endswith("(Acapella).flac"),
              "vocals must match the (Acapella) alias")
        check(_find_instrument_file(aliased, "other", FTYPES).endswith("(Instr).flac"),
              "other must match the (Instr) alias")

        # Word boundaries: 'other' must not collide with Mother/Brother/Another.
        tricky = os.path.join(tmp, "Tricky")
        os.makedirs(tricky)
        write_stem(tricky, "Mother.flac")
        write_stem(tricky, "Brother.wav")
        check(_find_instrument_file(tricky, "other", FTYPES) is None,
              "'other' must not match Mother/Brother")

        # Folder with no matching stems: report length 0, never crash.
        empty = os.path.join(tmp, "Empty")
        os.makedirs(empty)
        with open(os.path.join(empty, "notes.txt"), "w") as f:
            f.write("not audio")  # no audio at all
        path0, len0 = get_track_set_length((empty, ["vocals", "other"], FTYPES, 1))
        check(len0 == 0, "unmatched folder must return length 0 (no crash)")

        # End-to-end dataset scan: all tracks usable, none dropped.
        cfg = ConfigDict({
            "audio": {"chunk_size": 4410, "min_mean_abs": 0.0},
            "training": {"instruments": ["vocals", "other"],
                         "batch_size": 1, "num_steps": 1,
                         "target_instrument": None},
        })
        ds = MSSDataset(cfg, [tmp], metadata_path=os.path.join(tmp, "meta.pkl"),
                        dataset_type=1, batch_size=1, verbose=False)
        usable = {os.path.basename(p) for p, l in ds.metadata if l > 0}
        check(usable == {"Artist - Song", "Aliased"},
              f"usable tracks must be exactly the fully-matched folders, got {usable}")
        check(ds.metadata[0][1] > 0, "metadata must carry positive lengths")

        # Loader paths: stems actually load through the resolver.
        torch.manual_seed(0)
        batch = ds[0]
        stems, mix = batch[0], batch[1]
        check(stems.shape[0] == 2 and mix.shape[0] == 2,
              "2-stem batch must load (stems + mixture derived from stems)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())