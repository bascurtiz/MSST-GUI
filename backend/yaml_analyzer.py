"""
backend/yaml_analyzer.py
Model classifier — parses YAML configs to determine stem type and output stems.
"""
import yaml


def _load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        return None


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


def classify_model_type(yaml_path):
    config = _load_yaml(yaml_path)
    if not config:
        return None

    training = config.get("training", {})
    if not training:
        return None

    instruments = training.get("instruments", [])
    target = training.get("target_instrument", None)
    model_name = training.get("model_name", "")

    if not instruments:
        return None

    instruments_lower = [i.lower() for i in instruments]
    target_lower = target.lower() if target else None
    name_lower = model_name.lower() if model_name else ""

    # Check for processing models first (dereverb, denoise, etc.)
    processing_keywords = {
        "dereverb": "dereverb / deecho",
        "deecho": "dereverb / deecho",
        "denoise": "denoise",
        "super_resolution": "super resolution",
        "superresolution": "super resolution",
        "upscale": "super resolution",
        "phantom": "phantom centre",
        "centre": "phantom centre",
        "center": "phantom centre",
        "karaoke": "karaoke",
    }

    for keyword, model_type in processing_keywords.items():
        if keyword in name_lower or keyword in "_".join(instruments_lower):
            return model_type

    # Vocals / Instrumental logic
    if "other" in instruments_lower and "vocals" in instruments_lower:
        if target_lower == "other":
            return "instrumental"
        elif target_lower == "vocals":
            return "vocals"

    if "other" in instruments_lower and "instrumental" in instruments_lower:
        if target_lower == "other":
            return "vocals"

    # Dual target detection
    if ("vocals" in instruments_lower and "instrumental" in instruments_lower and
            target_lower is None):
        return "dual target (instrumental & vocals)"

    # Multi stems detection (3+ instruments, not just vocals/other)
    stem_instruments = [i for i in instruments_lower if i not in ("other", "vocals", "instrumental", "none")]
    if len(stem_instruments) >= 3:
        return "multi stems"

    # Specific instrument detection
    instrument_map = {
        "drums": "drums",
        "bass": "bass",
        "piano": "piano",
        "guitar": "guitar",
        "wind": "wind",
        "strings": "strings",
        "percussion": "percussion",
        "keys": "keys",
        "kick": "drums",
        "snare": "drums",
        "cymbals": "drums",
        "toms": "drums",
    }

    for instr, model_type in instrument_map.items():
        if instr in instruments_lower:
            return model_type

    # Fallback: if only vocals and other, default based on target
    if "vocals" in instruments_lower:
        return "vocals"
    if "instrumental" in instruments_lower:
        return "instrumental"

    return None



