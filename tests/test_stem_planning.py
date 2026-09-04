"""Regression test: stem mislabeling for single-output (karaoke /
instrumental-only) model configs.

Bug history: a 1-output config with instruments ``[vocals, instrument]``
targeting ``instrument`` produced files named ``(instrument)`` and
``(instrumental)`` — where "instrumental" actually contained the vocals.
The complement was hardcoded to "instrumental" instead of the config's real
second stem ("vocals").  This suite replays the full decision path
(GUI temp-yaml target resolution + engine complement naming) against the
single-output config shapes found in the pre-trained catalog audit and
asserts every stem comes out with its correct name.

Runs offline: no Qt, no torch, no network.  Pure logic from
utils/stem_planning.py — the same module inference.py and the inference
page call, so this test guards the production path, not a copy.
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stem_planning import (  # noqa: E402
    KEEP,
    complement_stem_name,
    is_single_output,
    plan_output_stems,
    resolve_target,
    rest_needed,
)

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


# ── Fixtures: real single-output config shapes from the catalog audit ──────
# (num_stems == 1 + target_instrument; 2-instrument lists.)

KARAOKE_VOCALS = {          # e.g. "BS Roformer (viperx edition)"
    "model": {"num_stems": 1},
    "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
}
KARAOKE_OTHER = {           # same model, "other" variant
    "model": {"num_stems": 1},
    "training": {"instruments": ["vocals", "other"], "target_instrument": "other"},
}
MEL_NOREVERB = {            # "MelBand Roformer (anvuew edition)"
    "model": {"num_stems": 1},
    "training": {"instruments": ["noreverb", "reverb"], "target_instrument": "noreverb"},
}
MEL_DENOISE = {             # "MelBand Roformer Denoise (by aufr33)"
    "model": {"num_stems": 1},
    "training": {"instruments": ["dry", "other"], "target_instrument": "dry"},
}
PHANTOM = {                 # "MDX23C Phantom Centre extraction"
    "model": {"num_stems": 1},
    "training": {"instruments": ["similarity", "difference"],
                 "target_instrument": "similarity"},
}
UNWA_INST = {               # "BS Roformer Instrumental Large v2" (pcunwa)
    "model": {"num_stems": 1},
    "training": {"instruments": ["vocals", "instrument"],
                 "target_instrument": "instrument"},
}
DTTNET = {                  # "DTTNet"
    "model": {"num_stems": 1},
    "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
}

# Multi-stem controls (must never invent a complement name):
BS_4STEM = {
    "model": {"num_stems": 4},
    "training": {"instruments": ["drums", "bass", "other", "vocals"]},
}
SCNET_4STEM = {
    "model": {"num_stems": 4},
    "training": {"instruments": ["drums", "bass", "other", "vocals"]},
}

SINGLE_OUTPUT_FIXTURES = {
    "karaoke vocals": KARAOKE_VOCALS,
    "karaoke other": KARAOKE_OTHER,
    "noreverb/reverb": MEL_NOREVERB,
    "denoise dry/other": MEL_DENOISE,
    "phantom similarity/difference": PHANTOM,
    "unwa vocal/instrument": UNWA_INST,
    "dttnet": DTTNET,
}


def main():
    # ── 1. is_single_output classification ────────────────────────────────
    for name, cfg in SINGLE_OUTPUT_FIXTURES.items():
        check(f"is_single_output({name})", is_single_output(cfg))
    check("is_single_output(4-stem) is False", not is_single_output(BS_4STEM))
    check("is_single_output(no target) is False",
          not is_single_output({"model": {"num_stems": 1},
                                "training": {"instruments": ["vocals", "other"]}}))
    check("is_single_output(missing num_stems, no target) is False",
          not is_single_output({"training": {"instruments": ["vocals", "other"]}}))

    # ── 2. Complement naming (the original mislabeling bug) ────────────────
    # A 1-output [vocals, instrument] config targeting instrument must name
    # the mix-complement "vocals" — never the generic "instrumental".
    check("unwa complement is 'vocals'",
          complement_stem_name(["vocals", "instrument"], ["instrument"]) == "vocals")
    check("karaoke complement is 'other'",
          complement_stem_name(["vocals", "other"], ["vocals"]) == "other")
    check("noreverb complement is 'reverb'",
          complement_stem_name(["noreverb", "reverb"], ["noreverb"]) == "reverb")
    check("phantom complement is 'difference'",
          complement_stem_name(["similarity", "difference"], ["similarity"]) == "difference")
    # Multi-stem: mix minus one target is not any single trained stem.
    check("4-stem complement stays generic 'instrumental'",
          complement_stem_name(["drums", "bass", "other", "vocals"], ["vocals"])
          == "instrumental")

    # ── 3. Full stem plan: selecting all stems ─────────────────────────────
    # The reported bug scenario: select everything, save-rest auto-fires,
    # and every written file must carry a real stem name.
    for name, cfg in SINGLE_OUTPUT_FIXTURES.items():
        inst = cfg["training"]["instruments"]
        target = cfg["training"].get("target_instrument")
        plan = plan_output_stems(cfg, inst)  # all stems selected
        others = [s for s in inst if s != target]
        expected = [target] + others  # target + correctly-named complement
        check(f"plan(all stems)={name} -> {expected}",
              plan == expected and "instrumental" not in plan)

    # ── 4. Full stem plan: selecting only the target ───────────────────────
    # Selecting just the target must yield the target alone (no complement).
    for name, cfg in SINGLE_OUTPUT_FIXTURES.items():
        target = cfg["training"].get("target_instrument")
        plan = plan_output_stems(cfg, [target])
        check(f"plan(target only)={name} -> [{target}]",
              plan == [target])

    # ── 5. Selecting the derived stem alone ────────────────────────────────
    # Picking the *other* stem must still emit the target (the model can
    # only separate its trained target) plus the derived stem, correctly
    # named — never a mislabeled single file.
    plan = plan_output_stems(UNWA_INST, ["vocals"])   # vocals is derived here
    check("unwa vocals-only -> [instrument, vocals]",
          plan == ["instrument", "vocals"])
    plan = plan_output_stems(KARAOKE_OTHER, ["vocals"])
    check("karaoke(other-target) vocals-only -> [other, vocals]",
          plan == ["other", "vocals"])

    # ── 6. Multi-stem controls: no complement invented ─────────────────────
    for name, cfg in (("BS 4-stem", BS_4STEM), ("SCNet 4-stem", SCNET_4STEM)):
        inst = cfg["training"]["instruments"]
        plan = plan_output_stems(cfg, inst)
        check(f"plan(all stems)={name} -> all 4 stems",
              sorted(plan) == sorted(inst) and "instrumental" not in plan)

    # ── 7. resolve_target / rest_needed unit checks ────────────────────────
    check("resolve_target keeps target for 1-output multi-select",
          resolve_target(UNWA_INST, ["vocals", "instrument"], True, False) is KEEP)
    check("resolve_target keeps target for 1-output single override",
          resolve_target(UNWA_INST, ["vocals"], False, True) is KEEP)
    check("resolve_target nulls target for multi-output multi-select",
          resolve_target(BS_4STEM, ["vocals", "drums"], True, True) is None)
    check("resolve_target sets stem for multi-output single select",
          resolve_target(BS_4STEM, ["vocals"], False, True) == "vocals")
    check("rest_needed auto-fires on non-target selection",
          rest_needed(UNWA_INST, ["vocals", "instrument"]) is True)
    check("rest_needed does not fire for target-only selection",
          rest_needed(UNWA_INST, ["instrument"]) is False)
    check("rest_needed does not fire for multi-stem all-selection",
          rest_needed(BS_4STEM, BS_4STEM["training"]["instruments"]) is False)

    # ── 8. Real installed config, when present ─────────────────────────────
    installed = os.path.join(
        os.path.expanduser("~"),
        "AppData", "Local", "Programs", "MSST-GUI", "configs",
        "bs_inst_large2_unwa_config.yaml")
    if os.path.isfile(installed):
        with open(installed, encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        inst = cfg["training"]["instruments"]
        plan = plan_output_stems(cfg, inst)
        check("installed unwa config plan -> [instrument, vocals]",
              plan == ["instrument", "vocals"])
        check("installed unwa config is single-output",
              is_single_output(cfg))
    else:
        print("SKIP installed-config check (not installed on this machine)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()