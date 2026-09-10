"""Regression test: multi-file folder runs with the mix-minus complement.

Bug history: run_folder derived its per-file stem list once before the loop
and then mutated it by appending the complement name (extract_instrumental).
On a 2-stem single-output config (instruments [Voices, Inst], target Voices)
the FIRST file's complement is correctly named "Inst"; the SECOND file then
sees "Inst" already in the (mutated) list, complement_stem_name falls back to
the generic "instrumental", and the write loop still iterates the stale "Inst"
name -> KeyError 'Inst' kills the whole batch right after song_000. Observed
on a 100-song DNR batch with bs_fnf2_mrdense67 (song_000 wrote both stems,
song_001 crashed with "KeyError: 'Inst'"). Multi-stem configs never hit this
because their complement is always the generic "instrumental", so the name is
stable across files.

This runs the real run_folder() end-to-end on two tiny wav files with a
stubbed bigshifts_wrapper (no model, no CUDA inference) and asserts BOTH
songs write BOTH stems under the correct names — the second song must not
crash and its complement must still be named "Inst".
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import inference as inf  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


class Flex:
    """Attr AND dict-style access — upstream reads config.audio both ways
    (config.audio.sample_rate attribute-style, "num_channels" via in/[])."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __contains__(self, k):
        return k in self.__dict__

    def __getitem__(self, k):
        return self.__dict__[k]


def _write_wav(path, sr=44100, seconds=0.25):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = 0.25 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, x.astype(np.float32), sr)  # mono input file


def main():
    tmp = tempfile.mkdtemp(prefix="msst_comp_rerun_")
    inp = os.path.join(tmp, "in")
    out = os.path.join(tmp, "out")
    os.makedirs(inp)
    for i in range(2):
        _write_wav(os.path.join(inp, f"song_{i:03d}_mixture.wav"))

    # Mirror the bs_fnf2_mrdense67 config shape: 2 trained stems, target the
    # first one (single-output run), mix-minus complement for the other.
    config = Flex(
        audio=Flex(sample_rate=44100, num_channels=2),
        training=Flex(instruments=["Voices", "Inst"],
                      target_instrument="Voices"),
        inference={"normalize": False},
    )

    args = types.SimpleNamespace(
        input_folder=inp,
        start_check_point="bs_fnf2_mrdense67.ckpt",
        store_dir=out,
        model_type="bs_roformer",
        bigshifts=1,
        use_tta=False,
        extract_instrumental=True,
        disable_detailed_pbar=True,
        filename_template="{file_name}_{instr}",
        pcm_type="FLOAT",
        strip_mixture=False,
        stem_suffix_map="",
        draw_spectro=0,
    )

    real_bigshifts = inf.bigshifts_wrapper

    def fake_bigshifts(config, model, mix, device, **kwargs):
        # One separated stem per the config target; run_folder derives the
        # complement itself (mix - target) in extract_instrumental mode.
        return {"Voices": np.zeros((2, mix.shape[1]), dtype=np.float32)}

    inf.bigshifts_wrapper = fake_bigshifts
    model = types.SimpleNamespace(eval=lambda: None, stereo=True)
    try:
        inf.run_folder(model, args, config, torch.device("cpu"), verbose=False)
    except KeyError as exc:
        check(False, f"run_folder crashed on a later song: {exc}")
    finally:
        inf.bigshifts_wrapper = real_bigshifts

    # os.listdir sorts case-sensitively: "Inst" (I) < "Voices" (V).
    expected = [
        "song_000_mixture_Inst.wav",
        "song_000_mixture_Voices.wav",
        "song_001_mixture_Inst.wav",
        "song_001_mixture_Voices.wav",
    ]
    got = sorted(os.listdir(out))
    check(got == expected,
          f"both songs wrote both stems under correct names: {got}")

    # ── Quality-Checker mode with a real 4-stem model ──────────────────
    # bs_pope_4stem_09072026_aname trains vocals/other/drums/bass. On the
    # Multisong dataset the raw map renames "other" to "instrum", which is
    # only right for 2-stem vocals models whose "other" IS the mix-minus
    # complement. A 4-stem model's "other" is a real trained stem and must
    # be written as _other (mvsep's 4-stem / MUSDB18 convention), with the
    # _mixture suffix stripped from the input name.
    out4 = os.path.join(tmp, "out4")
    os.makedirs(out4)
    config4 = Flex(
        audio=Flex(sample_rate=44100, num_channels=2),
        training=Flex(instruments=["vocals", "other", "drums", "bass"],
                      target_instrument=None),
        inference={"normalize": False},
    )
    args4 = types.SimpleNamespace(
        input_folder=inp,
        start_check_point="bs_pope_4stem_09072026_aname.ckpt",
        store_dir=out4,
        model_type="bs_roformer",
        bigshifts=1,
        use_tta=False,
        extract_instrumental=False,
        disable_detailed_pbar=True,
        filename_template="{file_name}_{instr}",
        pcm_type="FLOAT",
        strip_mixture=True,
        stem_suffix_map=(  # Multisong, serialized as the GUI sends it
            "vocals=vocals,voice=vocals,voices=vocals,vocal=vocals,"
            "other=instrum,instrumental=instrum,instrument=instrum,"
            "instr=instrum,inst=instrum,accompaniment=instrum,"
            "accomp=instrum"),
        draw_spectro=0,
    )

    def fake_bigshifts4(config, model, mix, device, **kwargs):
        return {k: np.zeros((2, mix.shape[1]), dtype=np.float32)
                for k in ("vocals", "other", "drums", "bass")}

    inf.bigshifts_wrapper = fake_bigshifts4
    try:
        inf.run_folder(model, args4, config4, torch.device("cpu"),
                       verbose=False)
    except Exception as exc:
        check(False, f"4-stem QC run crashed: {exc}")
    finally:
        inf.bigshifts_wrapper = real_bigshifts

    expected4 = sorted(
        f"song_{i:03d}_{stem}.wav"
        for i in range(2) for stem in ("vocals", "other", "drums", "bass")
    )
    got4 = sorted(os.listdir(out4))
    check(got4 == expected4,
          f"4-stem QC run: _other kept (not _instrum): {got4}")
    check(not any("_instrum" in f for f in got4),
          "no _instrum file for a 4-stem model on multisong")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print(f"ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()