"""Catalog rows vs engine --model_type (wizard pretrained lock)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pretrained_catalog import (  # noqa: E402
    PretrainedModel, matches_model_type,
)

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def _m(**kw):
    row = dict(
        name="model",
        section="Vocal models",
        instruments="vocals",
        metrics="",
        config_url="",
        checkpoint_url="",
        arch_hint="",
    )
    row.update(kw)
    return PretrainedModel(**row)


def main():
    roformer = _m(
        name="BS Roformer vocals",
        config_url="https://example.com/config_vocals_bs_roformer.yaml",
        checkpoint_url="https://example.com/model_bs_roformer.ckpt",
    )
    check(matches_model_type(roformer, "bs_roformer"),
          "bs_roformer URL matches bs_roformer")
    check(not matches_model_type(roformer, "bs_conformer"),
          "bs_roformer URL does not match bs_conformer")

    conformer = _m(
        name="BS Conformer vocals",
        config_url="https://example.com/config_vocals_bs_conformer.yaml",
        checkpoint_url="https://example.com/model_bs_conformer.ckpt",
    )
    check(matches_model_type(conformer, "bs_conformer"),
          "bs_conformer URL matches bs_conformer")
    check(not matches_model_type(conformer, "bs_roformer"),
          "bs_conformer URL does not match bs_roformer")
    check(not matches_model_type(conformer, "conformer"),
          "bs_conformer is not the generic conformer type")

    unhinted = _m(
        name="Kim vocals",
        config_url="https://example.com/configs/config_vocals_mel_band_roformer.yaml",
        arch_hint="",
    )
    check(matches_model_type(unhinted, "mel_band_roformer"),
          "empty hint still matches from the config URL")
    check(not matches_model_type(unhinted, "bs_roformer"),
          "mel-band URL does not match bs_roformer")

    demucs = _m(
        name="Demucs4 choir",
        checkpoint_url="https://example.com/demucs4_choirsep.ckpt",
        arch_hint="Demucs Architecture",
    )
    check(matches_model_type(demucs, "htdemucs"),
          "Demucs arch_hint aliases to htdemucs")
    check(not matches_model_type(demucs, "mdx23c"),
          "Demucs hint does not match mdx23c")

    mamba = _m(arch_hint="BSMamba2 Architecture")
    check(matches_model_type(mamba, "bs_mamba2"),
          "BSMamba2 hint aliases to bs_mamba2")

    generic_conf = _m(
        name="Some Conformer",
        arch_hint="Conformer Architecture",
    )
    check(not matches_model_type(generic_conf, "bs_conformer"),
          "conformer family is not collapsed into bs_conformer")
    check(matches_model_type(generic_conf, "conformer"),
          "plain Conformer hint matches the conformer engine type")

    wrong_hint = _m(
        config_url="https://example.com/config_vocals_bs_conformer.yaml",
        arch_hint="BS Roformer Architecture",
    )
    check(not matches_model_type(wrong_hint, "bs_roformer"),
          "a concrete URL guess of a different type hides the row")

    check(matches_model_type(roformer, ""),
          "empty lock matches every row")
    check(matches_model_type(roformer, None),
          "None lock matches every row")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_pretrained_catalog_match():
    assert main() == 0
