"""
backend/yaml_analyzer.py
Model classifier — parses YAML/JSON configs to determine architecture,
stem type, and output stems. Supports file paths, in-memory strings, and
parsed dictionaries.
"""
import os
import re
import json
import yaml


def _tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


try:
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor)
except Exception:
    pass
try:
    yaml.FullLoader.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor)
except Exception:
    pass


ARCH_TYPES = [
    "Apollo Architecture",
    "Bandit Architecture",
    "BS Roformer Architecture",
    "BSMamba2 Architecture",
    "Conformer Architecture",
    "Demucs Architecture",
    "DTTNet Architecture",
    "MDX23c Architecture",
    "MDX-Net Architecture",
    "Medley Vox Architecture",
    "Melband Roformer Architecture",
    "SCNet Architecture",
    "Swin Upernet Architecture",
    "TorchSeg Architecture",
    "VR Architecture",
    "VitLarge23 Architecture",
]

MODEL_TYPES = [
    "vocals",
    "instrumental",
    "dereverb / deecho",
    "denoise",
    "phantom centre",
    "karaoke",
    "dual target (instrumental & vocals)",
    "multi stems",
    "super resolution",
    "drums",
    "bass",
    "piano",
    "guitar",
    "wind",
    "strings",
    "percussion",
    "keys",
    "effects",
    "crowd",
]


def load_config_data(source):
    """Load config data from a dict, a file path, or raw YAML/JSON text."""
    if isinstance(source, dict):
        return source
    if not source or not isinstance(source, str):
        return None

    text = None
    if os.path.isfile(source):
        try:
            with open(source, "rb") as f:
                raw = f.read()
            for enc in ("utf-8", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                text = raw.decode("utf-8", errors="replace")
        except Exception:
            return None
    else:
        text = source

    if not text or not text.strip():
        return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            pass

    try:
        data = yaml.load(text, Loader=yaml.FullLoader)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    try:
        data = yaml.unsafe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def _load_yaml(path):
    """Backwards compatibility helper."""
    return load_config_data(path)


def get_stems_for_type(type_name):
    if not type_name:
        return []
    if type_name.lower() == "dual target (instrumental & vocals)":
        base = ["vocals", "instrumental"]
    else:
        base = [type_name]
    result = []
    for s in base:
        result.append(s)
        result.append(f"No {s}")
    return result


def classify_model_arch(source, filename=""):
    """
    Detect the architecture from a YAML/JSON config (path, text, or dict)
    and optional filename / URL.
    Returns one of ARCH_TYPES, or None.
    """
    fn = ""
    if filename:
        fn = os.path.basename(filename).lower()
    elif isinstance(source, str) and (os.path.isfile(source) or "/" in source or "\\" in source):
        fn = os.path.basename(source).lower()

    data = load_config_data(source)
    if not isinstance(data, dict):
        data = {}

    training = data.get("training", {}) if isinstance(data, dict) else {}
    if not isinstance(training, dict):
        training = {}

    # 1. Direct model_type key in training
    t_mt = str(training.get("model_type", "")).strip().lower()
    if t_mt:
        m_map = {
            "apollo": "Apollo Architecture",
            "bandit": "Bandit Architecture",
            "bandit_v2": "Bandit Architecture",
            "bs_roformer": "BS Roformer Architecture",
            "bs_roformer_experimental": "BS Roformer Architecture",
            "bs_roformer_unwa_large": "BS Roformer Architecture",
            "bs_mamba2": "BSMamba2 Architecture",
            "conformer": "Conformer Architecture",
            "bs_conformer": "Conformer Architecture",
            "mel_band_conformer": "Conformer Architecture",
            "htdemucs": "Demucs Architecture",
            "demucs": "Demucs Architecture",
            "demucs3": "Demucs Architecture",
            "demucs4": "Demucs Architecture",
            "dttnet": "DTTNet Architecture",
            "dtt_net": "DTTNet Architecture",
            "mdx23c": "MDX23c Architecture",
            "experimental_mdx23c_stht": "MDX23c Architecture",
            "mdxnet": "MDX-Net Architecture",
            "mdx_net": "MDX-Net Architecture",
            "medley_vox": "Medley Vox Architecture",
            "mel_band_roformer": "Melband Roformer Architecture",
            "mel_band_roformer_experimental": "Melband Roformer Architecture",
            "scnet": "SCNet Architecture",
            "scnet_large": "SCNet Architecture",
            "scnet_tran": "SCNet Architecture",
            "scnet_masked": "SCNet Architecture",
            "scnet_unofficial": "SCNet Architecture",
            "swin_upernet": "Swin Upernet Architecture",
            "torchseg": "TorchSeg Architecture",
            "segm_models": "VitLarge23 Architecture",
            "vr": "VR Architecture",
        }
        if t_mt in m_map:
            return m_map[t_mt]

    # 2. Top-level keys
    if data.get("conformer") is True:
        return "Conformer Architecture"
    if "cls" in data and "bandit" in str(data.get("cls")).lower():
        return "Bandit Architecture"
    if any(k in data for k in ("htdemucs", "demucs", "hdemucs")) or data.get("model") in ("htdemucs", "demucs", "hdemucs"):
        return "Demucs Architecture"
    if "kwargs" in data and isinstance(data["kwargs"], dict):
        kw = data["kwargs"]
        if any(k in kw for k in ("band_type", "n_sqm_modules")):
            return "Bandit Architecture"

    # JSON MDX-Net configs
    if "compensate" in data and any(k in data for k in ("dim_f", "dim_t", "n_fft", "hop_length")):
        return "MDX-Net Architecture"

    # 3. Model dictionary signatures
    model = data.get("model", {})
    if isinstance(model, dict):
        m_keys = set(model.keys())
        if any(k in m_keys for k in ("is_vr5", "nn_arch_size", "nout_lstm", "model_params")):
            return "VR Architecture"
        if any(k in m_keys for k in ("band_SR", "num_dplayer", "bandsplit_ratios")) or any(k.startswith("tran_") for k in m_keys):
            return "SCNet Architecture"
        if any(k in m_keys for k in ("bandsequence", "block_type")):
            return "DTTNet Architecture"
        if any(k in m_keys for k in ("above_freq", "bn_chan")) or model.get("architecture") == "Conv-TasNet":
            return "Medley Vox Architecture"
        if "band_specs" in m_keys:
            return "Bandit Architecture"
        if "bottleneck_factor" in m_keys and "num_blocks_per_scale" in m_keys:
            return "MDX23c Architecture"
        if "feature_dim" in m_keys and "win" in m_keys and "layer" in m_keys:
            return "Apollo Architecture"
        if "time_module_depth" in m_keys or model.get("module_type") == "mamba2" or "use_mamba" in m_keys or "mamba" in str(model.get("module_type", "")).lower():
            return "BSMamba2 Architecture"
        if any(k in m_keys for k in ("time_conformer_depth", "freq_conformer_depth", "freq_bins")):
            return "Conformer Architecture"
        if "encoder_name" in m_keys:
            enc = str(model.get("encoder_name", "")).lower()
            if enc.startswith("tu-") or "vit" in enc:
                return "VitLarge23 Architecture"
            return "TorchSeg Architecture"
        if m_keys == {"act", "num_channels", "num_subbands"}:
            return "Swin Upernet Architecture"
        if "freqs_per_bands" in m_keys:
            return "BS Roformer Architecture"
        if "num_bands" in m_keys:
            return "Melband Roformer Architecture"

    # 4. Training instruments signature
    instruments = [str(i).lower() for i in training.get("instruments", [])]
    if "restored" in instruments and "addition" in instruments:
        return "Apollo Architecture"

    # 5. Filename fallbacks
    if fn:
        if "apollo" in fn:
            return "Apollo Architecture"
        if "bandit" in fn:
            return "Bandit Architecture"
        if "mel_band" in fn or "melband" in fn or fn.startswith("mbr_"):
            return "Melband Roformer Architecture"
        if "bs_roformer" in fn or "bsrofo" in fn or (fn.startswith("bs_") and "mamba" not in fn and "conformer" not in fn):
            return "BS Roformer Architecture"
        if "mamba" in fn:
            return "BSMamba2 Architecture"
        if "conformer" in fn:
            return "Conformer Architecture"
        if "demucs" in fn or "htdemucs" in fn or "hdemucs" in fn:
            return "Demucs Architecture"
        if "dttnet" in fn or "dtt_net" in fn:
            return "DTTNet Architecture"
        if "mdx23" in fn:
            return "MDX23c Architecture"
        if "mdx" in fn or fn.endswith(".onnx") or fn.endswith(".json"):
            return "MDX-Net Architecture"
        if "medley_vox" in fn or "medleyvox" in fn:
            return "Medley Vox Architecture"
        if "scnet" in fn:
            return "SCNet Architecture"
        if "swin" in fn or "upernet" in fn:
            return "Swin Upernet Architecture"
        if "torchseg" in fn:
            return "TorchSeg Architecture"
        if "segm" in fn or "vitlarge" in fn or "vit_large" in fn:
            return "VitLarge23 Architecture"
        if "vr" in fn or "hp-karaoke" in fn:
            return "VR Architecture"

    return None


def classify_model_type(source, filename=""):
    """
    Detect the stem / model type from a YAML/JSON config (path, text, or dict)
    and optional filename / URL.
    Returns one of MODEL_TYPES, or None.
    """
    fn = ""
    if filename:
        fn = os.path.basename(filename).lower()
    elif isinstance(source, str) and (os.path.isfile(source) or "/" in source or "\\" in source):
        fn = os.path.basename(source).lower()

    data = load_config_data(source)
    if not isinstance(data, dict):
        data = {}

    training = data.get("training", {}) if isinstance(data, dict) else {}
    if not isinstance(training, dict):
        training = {}

    raw_instruments = training.get("instruments", [])
    instruments = [str(i).lower() for i in raw_instruments] if isinstance(raw_instruments, list) else []
    target = training.get("target_instrument")
    target_lower = str(target).lower().strip() if target else None
    model_name = str(training.get("model_name", "")).lower().strip()

    all_text = f"{fn} {model_name} {' '.join(instruments)} {target_lower or ''}".lower()

    # Special handling for JSON configs
    if not training and (fn.endswith(".json") or ("compensate" in data and "n_fft" in data)):
        if any(k in all_text for k in ("dereverb", "deecho")):
            return "dereverb / deecho"
        if "denoise" in all_text:
            return "denoise"
        if "karaoke" in all_text:
            return "karaoke"
        if any(k in all_text for k in ("inst", "instrumental")):
            return "instrumental"
        if any(k in all_text for k in ("voc", "vocal")):
            return "vocals"
        return "dual target (instrumental & vocals)"

    # 1. Processing keywords
    if any(k in all_text for k in ("super_resolution", "superresolution", "upscale", "restored", "restoration")) or "apollo" in fn:
        if not target_lower or target_lower in ("restored", "restoration", "sr", "none"):
            return "super resolution"

    if any(k in all_text for k in ("dereverb", "deecho", "de-reverb", "de-echo", "noreverb", "reverb", "dry", "room")):
        return "dereverb / deecho"

    if any(k in all_text for k in ("denoise", "noise", "debleed", "bleed", "debreath", "breath", "scratch", "clean", "invert_clean")):
        return "denoise"

    if any(k in all_text for k in ("phantom", "centre", "center", "similarity", "difference", "mid_side", "mid-side")):
        return "phantom centre"

    if "karaoke" in all_text:
        return "karaoke"

    if "crowd" in all_text:
        return "crowd"

    if any(k in all_text for k in ("effects", "sfx")):
        return "effects"

    # 2. Dual target detection
    if "instvoc" in fn or "inst_voc" in fn:
        return "dual target (instrumental & vocals)"
    if ("vocals" in instruments and "instrumental" in instruments and not target_lower):
        return "dual target (instrumental & vocals)"
    if any(k in instruments for k in ("vox_1", "vox_2")):
        return "vocals"

    # 3. Multi-stems detection
    non_aux = [i for i in instruments if i not in ("other", "none", "rest", "")]
    if not target_lower and (len(non_aux) >= 3 or len(instruments) >= 4 or any(k in fn for k in ("4stem", "6stem", "53stem", "multistem"))):
        return "multi stems"

    # 4. Target instrument specified
    if target_lower:
        if target_lower in ("vocals", "vocal", "vox"):
            return "vocals"
        if target_lower in ("other", "inst", "instrumental"):
            if "vocals" in instruments or "vocal" in instruments:
                return "instrumental"
        if any(k in target_lower for k in ("drum", "kick", "snare", "cymbal", "tom")):
            return "drums"
        if "bass" in target_lower:
            return "bass"
        if any(k in target_lower for k in ("piano", "key", "synth")):
            return "piano"
        if any(k in target_lower for k in ("guitar", "lead", "rhythm")):
            return "guitar"
        if any(k in target_lower for k in ("wind", "sax", "brass", "horn", "trumpet", "flute")):
            return "wind"
        if any(k in target_lower for k in ("string", "bowed_str", "violin", "cello", "orch")):
            return "strings"
        if "percussion" in target_lower:
            return "percussion"

    # 5. Check isolated instruments in instruments list
    inst_no_other = [i for i in instruments if i not in ("other", "none", "rest", "")]
    for instr in inst_no_other:
        if any(k in instr for k in ("drum", "kick", "snare", "cymbal", "tom", "hihat", "hh")):
            return "drums"
        if "bass" in instr:
            return "bass"
        if any(k in instr for k in ("piano", "digital-piano")):
            return "piano"
        if any(k in instr for k in ("guitar", "lead", "rhythm")):
            return "guitar"
        if any(k in instr for k in ("wind", "sax", "brass", "horn", "trumpet", "flute", "clarinet", "harmonica")):
            return "wind"
        if any(k in instr for k in ("string", "bowed_str", "violin", "cello", "viola", "orch")):
            return "strings"
        if "percussion" in instr:
            return "percussion"
        if any(k in instr for k in ("key", "synth", "synthesizer", "organ")):
            return "keys"

    # 6. Vocals / Instrumental logic
    if "vocals" in instruments or "vocal" in instruments:
        if "other" in instruments and target_lower == "other":
            return "instrumental"
        if "inst" in fn or "instrumental" in fn:
            return "instrumental"
        return "vocals"

    if "instrumental" in instruments or "inst" in instruments:
        return "instrumental"

    # 7. Final filename hints
    if "inst" in fn:
        return "instrumental"
    if "voc" in fn:
        return "vocals"

    return None


def classify_yaml_config(source, filename=""):
    """
    Convenience method returning both architecture and model type as a dict:
    {"arch": ..., "type": ...}
    """
    return {
        "arch": classify_model_arch(source, filename=filename),
        "type": classify_model_type(source, filename=filename),
    }
