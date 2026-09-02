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
