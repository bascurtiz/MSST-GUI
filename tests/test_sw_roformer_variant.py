"""Regression: SW / Logic / SW-Fixed BS-Roformer dispatch and stem energy.

Bug history: 6-stem configs without ``sw: true`` were forced into
BSRoformerSW(rope). That is correct for Logic (``linear_62_bias_0``) but
wrong for jarredou's SW-Fixed checkpoint, which is stock BSRoformer
(``rotary_embed.freqs`` only). Original SW (``cos_emb_time``) also needs
the learned-position class, and its cos/sin tables must be the *same*
Parameter objects the attention layers apply — a re-wrap left apply()
reading a zero copy and collapsed every head except ``other``.

Runs offline for sniff / share / dispatch-class checks. The optional peak
pass needs CUDA plus the installed checkpoints and one Multisong clip.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from ml_collections import ConfigDict  # noqa: E402

from models.bs_roformer.bs_roformer_sw import BSRoformerSW  # noqa: E402
from utils.settings import (  # noqa: E402
    _resolve_bs_roformer_variant,
    _sniff_bs_roformer_sw_mode,
    load_config,
    resolve_sw_msst_paths,
)

FAILURES = []
CHECKS = 0

INSTALLED_CKPT = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Local", "Programs", "MSST-GUI", "models", "bs_roformer")
INSTALLED_CFG = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Local", "Programs", "MSST-GUI", "configs")
MULTISONG_CLIP = os.path.join(
    "D:\\", "VALIDATION DATASETS", "multisong_dataset (1)",
    "song_000_mixture.wav")


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def _save_ckpt(folder, name, keys):
    path = os.path.join(folder, name)
    torch.save({k: torch.zeros(2) for k in keys}, path)
    return path


def _tiny_cfg(sw=None, num_stems=6):
    data = {
        "model": {
            "dim": 64,
            "depth": 1,
            "stereo": True,
            "num_stems": num_stems,
            "time_transformer_depth": 1,
            "freq_transformer_depth": 1,
        },
        "training": {
            "instruments": ["bass", "drums", "other", "vocals",
                            "guitar", "piano"][:max(num_stems, 1)],
        },
    }
    if sw is True:
        data["sw"] = True
    return ConfigDict(data)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        learned = _save_ckpt(tmp, "learned.ckpt",
                             ["cos_emb_time", "shared_qkv_bias", "band_split.x"])
        rope = _save_ckpt(tmp, "rope.ckpt",
                          ["linear_62_bias_0", "linear_64_bias_0",
                           "layers.0.0.layers.0.0.rotary_embed.freqs"])
        stock = _save_ckpt(tmp, "stock.ckpt",
                           ["layers.0.0.layers.0.0.rotary_embed.freqs",
                            "mask_estimators.0.to_freqs.0.0.0.weight"])
        empty = _save_ckpt(tmp, "empty.ckpt", [])

        check(_sniff_bs_roformer_sw_mode(learned) == "learned",
              "cos_emb_time / shared_qkv_bias -> learned")
        check(_sniff_bs_roformer_sw_mode(rope) == "rope",
              "linear_62_bias_0 -> rope")
        check(_sniff_bs_roformer_sw_mode(stock) == "stock",
              "rotary_embed.freqs only -> stock")
        check(_sniff_bs_roformer_sw_mode(empty) is None,
              "empty checkpoint -> None")
        check(_sniff_bs_roformer_sw_mode(
            os.path.join(tmp, "missing.ckpt")) is None,
            "missing file -> None")

        # 6-stem yaml WITHOUT sw:true must not force rope when the ckpt is stock.
        stock_model = _resolve_bs_roformer_variant(
            _tiny_cfg(sw=None, num_stems=6), stock)
        check(type(stock_model).__name__ == "BSRoformer",
              "6-stem stock ckpt builds BSRoformer, not rope SW")
        check(not hasattr(stock_model, "linear_62_bias_0"),
              "stock 6-stem has no Logic shared bias")

        # yaml sw:true is only a hint when keys cannot be read.
        hinted = _resolve_bs_roformer_variant(_tiny_cfg(sw=True, num_stems=6))
        check(isinstance(hinted, BSRoformerSW),
              "yaml sw:true without ckpt builds learned SW")
        check(hasattr(hinted, "cos_emb_time"),
              "yaml sw:true without ckpt uses learned embeddings")

        # Keys win over a misleading yaml flag: stock ckpt + sw:true yaml.
        overridden = _resolve_bs_roformer_variant(
            _tiny_cfg(sw=True, num_stems=6), stock)
        check(type(overridden).__name__ == "BSRoformer",
              "stock ckpt wins over yaml sw:true")

        learned_model = _resolve_bs_roformer_variant(
            _tiny_cfg(sw=None, num_stems=6), learned)
        check(isinstance(learned_model, BSRoformerSW),
              "learned ckpt builds BSRoformerSW")
        check(hasattr(learned_model, "cos_emb_time"),
              "learned ckpt uses position_mode=learned")

        # ZFTurbo/MSST path: original SW + a Fixed sibling -> use Fixed.
        ckpt, yaml_path, redirected = resolve_sw_msst_paths(learned, None)
        check(ckpt == learned and redirected is False,
              "learned ckpt without sibling stays put")
        sibling = _save_ckpt(tmp, "bs_6stem_fixed.ckpt",
                             ["layers.0.0.layers.0.0.rotary_embed.freqs"])
        # Rename the learned file so the sibling-name lookup matches.
        orig_named = os.path.join(tmp, "bs_6stem.ckpt")
        os.replace(learned, orig_named)
        ckpt, yaml_path, redirected = resolve_sw_msst_paths(orig_named, None)
        check(redirected is True, "learned SW + Fixed sibling redirects")
        check(os.path.normcase(ckpt) == os.path.normcase(sibling),
              "redirect target is bs_6stem_fixed.ckpt")
        learned = orig_named  # keep later installed-ckpt checks independent

        rope_model = _resolve_bs_roformer_variant(
            _tiny_cfg(sw=None, num_stems=6), rope)
        check(isinstance(rope_model, BSRoformerSW),
              "Logic ckpt builds BSRoformerSW")
        check(hasattr(rope_model, "linear_62_bias_0"),
              "Logic ckpt uses position_mode=rope")

    # Shared Parameter identity — apply() must see the parent tables.
    shared = BSRoformerSW(
        dim=64, depth=1, stereo=True, num_stems=2,
        time_transformer_depth=1, freq_transformer_depth=1,
        position_mode="learned")
    time_attn = shared.layers[0][0].layers[0][0]
    check(time_attn.rotary_embed.cos_emb is shared.cos_emb_time,
          "time rotary cos_emb is the parent cos_emb_time Parameter")
    check(time_attn.rotary_embed.sin_emb is shared.sin_emb_time,
          "time rotary sin_emb is the parent sin_emb_time Parameter")
    check(time_attn.to_qkv.bias is shared.shared_qkv_bias,
          "attention qkv bias is the shared_qkv_bias Parameter")
    check(time_attn.to_out[0].bias is shared.shared_out_bias,
          "attention out bias is the shared_out_bias Parameter")

    # Installed checkpoints, when present.
    sw_ckpt = os.path.join(INSTALLED_CKPT, "bs_6stem.ckpt")
    fixed_ckpt = os.path.join(INSTALLED_CKPT, "bs_6stem_fixed.ckpt")
    logic_ckpt = os.path.join(INSTALLED_CKPT, "bs_logic_6stem.ckpt")
    sw_yaml = os.path.join(INSTALLED_CFG, "bs_6stem_config.yaml")
    fixed_yaml = os.path.join(INSTALLED_CFG, "bs_6stem_fixed_config.yaml")
    if os.path.isfile(sw_ckpt):
        check(_sniff_bs_roformer_sw_mode(sw_ckpt) == "learned",
              "installed bs_6stem.ckpt sniffs as learned")
    else:
        print("SKIP installed bs_6stem.ckpt sniff")
    if os.path.isfile(fixed_ckpt):
        check(_sniff_bs_roformer_sw_mode(fixed_ckpt) == "stock",
              "installed bs_6stem_fixed.ckpt sniffs as stock")
        if os.path.isfile(fixed_yaml):
            cfg = load_config("bs_roformer", fixed_yaml)
            built = _resolve_bs_roformer_variant(cfg, fixed_ckpt)
            check(type(built).__name__ == "BSRoformer",
                  "installed Fixed yaml+ckpt builds stock BSRoformer")
    else:
        print("SKIP installed bs_6stem_fixed.ckpt sniff")
    if os.path.isfile(logic_ckpt):
        check(_sniff_bs_roformer_sw_mode(logic_ckpt) == "rope",
              "installed bs_logic_6stem.ckpt sniffs as rope")
    else:
        print("SKIP installed bs_logic_6stem.ckpt sniff")

    # ZFTurbo/MSST have no SW class — they run SW-Fixed as stock
    # BSRoformer. When both files are installed, picking the original
    # 350 MB checkpoint must remap onto Fixed (otherwise vocals stay empty).
    if os.path.isfile(sw_ckpt) and os.path.isfile(fixed_ckpt):
        ckpt, yaml_path, redirected = resolve_sw_msst_paths(sw_ckpt, sw_yaml)
        check(redirected is True,
              "installed original SW + Fixed sibling redirects")
        check(os.path.normcase(ckpt) == os.path.normcase(fixed_ckpt),
              "installed redirect target is bs_6stem_fixed.ckpt")
        if os.path.isfile(fixed_yaml):
            check(os.path.normcase(os.path.abspath(yaml_path))
                  == os.path.normcase(os.path.abspath(fixed_yaml)),
                  "installed redirect pairs the Fixed yaml")
    else:
        print("SKIP installed SW -> Fixed redirect")

    # SW-Fixed is the working 6-stem model (vocals/bass/drums present).
    _maybe_peak_check(fixed_ckpt, fixed_yaml, "bs_6stem_fixed",
                      required=("vocals", "bass", "drums"))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES / {CHECKS} checks:")
        for item in FAILURES:
            print(" -", item)
        sys.exit(1)
    print(f"ALL {CHECKS} CHECKS PASSED")


def _maybe_peak_check(ckpt, yaml_path, label, required=(),
                      forbid_other_only=False):
    """Load a real SW/Fixed checkpoint and assert the named stems have energy."""
    if not (os.path.isfile(ckpt) and os.path.isfile(yaml_path)
            and os.path.isfile(MULTISONG_CLIP)):
        print(f"SKIP peak check ({label}: files missing)")
        return
    if not torch.cuda.is_available():
        print(f"SKIP peak check ({label}: no CUDA)")
        return

    import numpy as np
    import soundfile as sf
    from utils.model_utils import demix

    audio, sr = sf.read(MULTISONG_CLIP, always_2d=True)
    clip = audio[: sr * 4].T.astype(np.float32)
    model, config = __import__(
        "utils.settings", fromlist=["get_model_from_config"]
    ).get_model_from_config("bs_roformer", yaml_path, checkpoint_path=ckpt)
    weights = torch.load(ckpt, weights_only=False, map_location="cpu")
    if isinstance(weights, dict):
        for wrapper in ("state_dict", "state", "model_state_dict"):
            if isinstance(weights.get(wrapper), dict):
                weights = weights[wrapper]
                break
        weights = {k: v for k, v in weights.items() if k != "_metadata"}
    model.load_state_dict(weights)
    del weights
    device = torch.device("cuda")
    model = model.to(device).eval()
    out = demix(config, model, clip, device, "bs_roformer", pbar=False)
    mix_peak = float(np.abs(clip).max())
    floor = max(1e-3, mix_peak * 0.02)
    peaks = {name: float(np.abs(np.asarray(wav)).max())
             for name, wav in out.items()}
    for stem in required:
        if stem not in peaks:
            check(False, f"{label} missing stem {stem}")
            continue
        check(peaks[stem] >= floor,
              f"{label} {stem} peak {peaks[stem]:.5f} >= {floor:.5f} "
              f"(not collapsed)")
    if forbid_other_only and peaks:
        loud = [name for name, peak in peaks.items() if peak >= floor]
        check(loud != ["other"],
              f"{label} must not collapse into other-only (loud={loud})")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
