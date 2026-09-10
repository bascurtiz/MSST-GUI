"""
backend/audio_names.py
----------------------
Output file naming helpers for the GUI's own job runners (auto / iterative
ensemble). These used to live in utils/audio_utils.py; the utils package is
now the untouched upstream Music-Source-Separation-Training code (which the
GUI process must never import — it pulls in torch / matplotlib), so the
GUI-side helpers live here.
"""
import os

# The flat "<song> (<stem>)" layout the GUI has always used, passed to
# inference.py as --filename_template (upstream's default nests one folder
# per song: "{file_name}/{instr}").
INFERENCE_FILENAME_TEMPLATE = "{file_name} ({instr})"

# mvsep quality-checker output naming (SDR-test mode): the input's trailing
# "_mixture" is stripped and each stem is written as "<song>_<suffix>", e.g.
# song_dnr_016_mixture.flac -> song_dnr_016_speech.flac.
SDR_FILENAME_TEMPLATE = "{file_name}_{instr}"
SDR_MIXTURE_SUFFIX = "_mixture"


def strip_mixture_name(name):
    """Drop a trailing "_mixture" from an input base name (mvsep
    quality-checker naming). No-op when the name doesn't end with it."""
    if isinstance(name, str) and name.endswith(SDR_MIXTURE_SUFFIX):
        return name[:-len(SDR_MIXTURE_SUFFIX)]
    return name


def parse_stem_suffix_map(text):
    """Parse "config_stem=output_suffix,other=instrum" into a dict.
    An entry "*=suffix" acts as a catch-all for stems without an explicit
    mapping (e.g. Super Resolution: every output becomes "_restored")."""
    result = {}
    if not text:
        return result
    for part in str(text).replace(" ", "").split(","):
        if not part or "=" not in part:
            continue
        key, _, val = part.partition("=")
        if key and val:
            result[key] = val
    return result


def qc_suffix_map_for_model(suffix_map, trained_instruments):
    """Adjust a dataset's stem-suffix map for the model actually running.

    mvsep's 2-stem vocal datasets (Multisong / Synthetic / Synth Vocals
    2026) score the instrumental part as "_instrum", so their maps rename
    the model's "other" stem to "instrum". That is only correct when
    "other" IS the mix-minus-vocals complement, i.e. a 2-stem vocals model.
    A multi-stem model (4 stems: vocals/drums/bass/other) has "other" as
    its own trained stem and mvsep names that "_other" (MUSDB18
    convention); renaming it "instrum" mislabels the uploaded file.
    Returns a new map with the other->instrum remap dropped when the model
    has >= 3 trained stems; otherwise the map is returned unchanged.

    Lead/Back Vocals: a 2-stem karaoke model (vocals + instrum) outputs
    lead vocals and back+instrumental combined. Every map entry that targets
    "_instrum" must become "_back-instrum" when the model has <= 2 trained
    stems; 3+ stem models keep "_instrum" for a real pure-instrumental stem.
    """
    if not suffix_map or not trained_instruments:
        return suffix_map

    adjusted = None

    if (len(trained_instruments) >= 3
            and suffix_map.get("other") == "instrum"):
        adjusted = dict(suffix_map)
        adjusted["other"] = "other"

    is_lead_back = (suffix_map.get("lead") == "lead"
                    and suffix_map.get("back-instrum") == "back-instrum")
    if is_lead_back and len(trained_instruments) <= 2:
        if adjusted is None:
            adjusted = dict(suffix_map)
        for key, val in list(adjusted.items()):
            if val == "instrum":
                adjusted[key] = "back-instrum"

    return adjusted if adjusted is not None else suffix_map


def stem_suffix_for(instr, suffix_map):
    """The mvsep output suffix for a config stem name, honouring an
    explicit entry first and a "*" catch-all second; identity otherwise.
    Lookup is case-insensitive (models commonly capitalize stems, e.g.
    "Voices" vs the map's "vocals") and the identity fallback is
    lowercased to match mvsep's always-lowercase suffixes."""
    if instr in suffix_map:
        return suffix_map[instr]
    low = instr.lower()
    if low in suffix_map:
        return suffix_map[low]
    if "*" in suffix_map:
        return suffix_map["*"]
    # Normalize case only when a map is active (Quality Checker mode, where
    # mvsep's suffixes are always lowercase); plain runs keep the model's
    # own stem names untouched.
    return low if suffix_map else instr


def resample_to_native(x, orig_sr, target_sr, target_len,
                       target_channels=None):
    """Resample channel-major audio ``x`` (channels x samples) from
    ``orig_sr`` to ``target_sr`` and force exactly ``target_len`` samples per
    channel (trimming the tail or zero-padding the end). When
    ``target_channels`` is given and differs from the array's channel count,
    the channel layout is converted first (downmix to mono by averaging,
    upmix to stereo by duplicating the single channel).

    mvsep's quality checker compares an uploaded stem against its reference
    mixture frame-for-frame at the mixture's native sample rate. When a model
    runs at a different rate than the input (e.g. a 44.1 kHz model on a
    48 kHz DNR v3 mixture), uploading the model-rate stem forces the checker
    to resample it server-side, which can come out one sample longer
    (``soxr``: 2646000 @ 44.1 kHz -> 2880001 @ 48 kHz) and be rejected with
    "Different shapes for wav file ... N != M". A mono mixture duplicated to
    stereo by the engine would similarly fail the checker's shape assert, so
    stems are also written with the input's own channel count. Writing stems
    at the input's own rate, frame count and channel layout avoids that
    server-side conversion entirely.

    Resamplers (in order): ``soxr`` (HQ), then ``scipy.signal.resample_poly``.
    """
    import numpy as np

    x = np.asarray(x)
    if target_channels is not None and x.shape[0] != target_channels:
        if target_channels == 1 and x.shape[0] > 1:
            x = np.mean(x, axis=0, keepdims=True)
        elif target_channels > 1 and x.shape[0] == 1:
            x = np.repeat(x, target_channels, axis=0)
        else:
            x = x[:target_channels]
    if orig_sr != target_sr:
        try:
            import soxr
            y = soxr.resample(x, orig_sr, target_sr, quality="HQ")
        except Exception:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(orig_sr), int(target_sr))
            y = resample_poly(
                x, int(target_sr) // g, int(orig_sr) // g, axis=-1)
        y = np.asarray(y)
    else:
        y = x
    n = y.shape[-1]
    if n > target_len:
        y = y[..., :target_len]
    elif n < target_len:
        pad = np.zeros(y.shape[:-1] + (int(target_len) - n,), dtype=y.dtype)
        y = np.concatenate([y, pad], axis=-1)
    return y


def get_audio_metadata(filepath):
    """(artist, title) from the file's tags, or (None, None)."""
    try:
        import mutagen
        audio = mutagen.File(filepath, easy=True)
        if audio is not None:
            artist = audio.get('artist', [None])[0]
            title = audio.get('title', [None])[0]
            return artist, title
    except Exception:
        pass
    return None, None


def sanitize_filename(name):
    if not name:
        return name
    invalid = '<>:"/\|?*'
    for ch in invalid:
        name = name.replace(ch, '')
    name = name.strip('. ')
    return name or 'Unknown'


def format_output_filename(input_path, target_name, ext='.wav'):
    """'<Artist - Title> (<target>)<ext>' when the input carries tags,
    otherwise '<input base name> (<target>)<ext>'."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    artist, title = get_audio_metadata(input_path)
    if artist and title:
        base = f"{sanitize_filename(artist)} - {sanitize_filename(title)}"
    return f"{base} ({target_name}){ext}"
