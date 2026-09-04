"""Regression test: console waveforms must reflect real relative loudness.

The old _WaveformTrack.load_audio normalized every stem to its own peak and
decimated by picking isolated single samples. A near-silent stem (an 'sfx'
file that is basically empty) was therefore amplified to full height and its
sparse noise drawn as busy activity — while Audacity showed the truth: an
almost flat line.

The fix decodes the whole output set once (windowed max envelope + true peak
per file) and scales every track against the loudest stem of the set, so a
near-silent stem draws near-flat. This drives the real _WaveformContainer
load path offscreen with synthetic files; no torch, no network.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication

from ui.pages.console_page import _WaveformContainer, _WaveformTrack  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    import soundfile as sf

    app = QApplication([])
    tmp = tempfile.mkdtemp()
    sr = 44100
    t = np.linspace(0, 45.0, 45 * sr, endpoint=False)

    def wav(rel, name):
        sig = rel * 0.8 * np.sin(2 * np.pi * 220 * t) + rel * 1e-3 * np.random.randn(len(t))
        path = os.path.join(tmp, name)
        sf.write(path, sig.astype(np.float32), sr)
        return path

    music = wav(1.0, "song (music).wav")
    inst = wav(0.5, "song (instrument).wav")
    sfx = wav(2e-4, "song (sfx).wav")

    class Card:
        _model_type = ""
        _model_name = ""
        _target_stem = ""

    card = Card()

    # --- pure envelope math ---
    env_m, pk_m = _WaveformContainer._read_envelope(music)
    env_s, pk_s = _WaveformContainer._read_envelope(sfx)
    check(pk_m is not None and pk_s is not None, "envelope peaks must decode")
    check(pk_m > 0.5 and 1e-4 < pk_s < 1e-2, f"true peaks off: {pk_m:.3e} {pk_s:.3e}")
    check(not np.any(np.isnan(env_m)) and not np.any(np.isnan(env_s)), "envelope must be nan-free")
    check(abs(env_m.max() - 1.0) < 1e-9 and abs(env_s.max() - 1.0) < 1e-9,
          "envelope normalized to unit peak")
    check(env_s.shape == env_m.shape, "fixed bin count")

    # --- integrated container path (the real fix) ---
    card._output_paths = [music, inst, sfx]
    container = _WaveformContainer()
    container.load_tracks(card)

    by_label = {tr._label: tr for tr in container._tracks}
    check(set(by_label) == {"Music", "Instrument", "Sfx"}, f"labels: {list(by_label)}")

    max_music = float(np.max(np.abs(by_label["Music"]._samples)))
    max_inst = float(np.max(np.abs(by_label["Instrument"]._samples)))
    max_sfx = float(np.max(np.abs(by_label["Sfx"]._samples)))

    check(max_music > 0.95, f"loudest stem should fill the track, got {max_music:.4f}")
    check(0.4 < max_inst < 0.6, f"instrument must keep true ~half amplitude, got {max_inst:.4f}")
    # The regression itself: a near-silent stem used to be drawn at ~full height.
    check(max_sfx < 5e-3, f"near-silent sfx still amplified to {max_sfx:.3e}")

    # painting must not crash with tiny/zero-scale arrays
    from PySide6.QtGui import QPixmap
    for tr in container._tracks:
        tr.resize(700, 120)
        pm = QPixmap(700, 120)
        tr.render(pm)

    # an all-silent set stays flat (no div-by-zero artifacts)
    card._output_paths = [wav(0.0, "a (vocals).wav"), wav(0.0, "a (other).wav")]
    container2 = _WaveformContainer()
    container2.load_tracks(card)
    for tr in container2._tracks:
        check(float(np.max(np.abs(tr._samples))) == 0.0, "all-silent set must be flat")

    # plucked-string stems (banjo, ukulele, mandolin, ...) take the guitar
    # family color, not the fallback palette (banjo was a dark purple)
    from ui.pages.console_page import _stem_color
    guitar = _stem_color("guitar")
    for plucked in ("banjo", "ukulele", "mandolin", "dobro", "sitar"):
        check(_stem_color(plucked) == guitar,
              f"{plucked} must be guitar-family red, got {_stem_color(plucked)}")
    check(_stem_color("accordion") != guitar,
          "unrelated stem must keep its own color")

    # woodwind/brass stems (clarinet, saxophone, ...) share the teal
    # "wind" family color; flutes moved to their own teal-green hue
    teal = _stem_color("wind")
    for wood in ("clarinet", "saxophone", "oboe", "bassoon",
                 "french-horn", "trombone", "trumpet", "tuba", "woodwind",
                 "wind-chimes", "accordion"):
        check(_stem_color(wood) == teal,
              f"{wood} must be wind-family teal, got {_stem_color(wood)}")
    check(teal != guitar, "wind teal must differ from guitar red")
    flute_col = _stem_color("flute")
    check(flute_col != teal, "flutes must have their own color, not wind teal")
    for fl in ("duduk", "flute", "harmonica", "pan flute",
               "penny whistle", "recorder", "shakuhachi", "tin whistle",
               "whistle"):
        check(_stem_color(fl) == flute_col,
              f"{fl} must be flutes color {flute_col}, got {_stem_color(fl)}")

    # drum-kit stems (tambourine, triangle, hi-hat, ...) are amber like
    # drums; percussion + timpani are the distinct percussion orange
    amber = _stem_color("drums")
    for kit in ("tambourine", "triangle", "hh", "congas", "kick",
                "snare", "toms"):
        check(_stem_color(kit) == amber,
              f"{kit} must be drums-family amber, got {_stem_color(kit)}")
    orange = _stem_color("percussion")
    check(orange != amber, "percussion orange must differ from drums amber")
    for p in ("percussion", "timpani", "glockenspiel"):
        check(_stem_color(p) == orange,
              f"{p} must be percussion orange, got {_stem_color(p)}")
    check(amber != teal and amber != guitar,
          "drums amber must stay distinct")

    # singular 'vocal' joins the vocals purple
    purple = _stem_color("vocals")
    for v in ("vocal", "back-vocal", "lead-vocal"):
        check(_stem_color(v) == purple,
              f"{v} must be vocals purple, got {_stem_color(v)}")

    # viola / violin / double bass join the strings green
    green = _stem_color("strings")
    for v in ("viola", "violin", "double-bass", "orchestral", "staccato",
              "string-ensemble", "strings-melody"):
        check(_stem_color(v) == green,
              f"{v} must be strings green, got {_stem_color(v)}")

    # organs family gets its own ochre (no longer keys blue)
    organ_col = _stem_color("organ")
    check(organ_col != _stem_color("keys"),
          "organs must differ from keys blue")
    for og in ("b3", "combo-organ", "drawbar", "farfisa", "hammond",
               "organ", "pipe-organ", "organs"):
        check(_stem_color(og) == organ_col,
              f"{og} must be organs color {organ_col}, got {_stem_color(og)}")

    # 'instrument' (bs_inst_large2_unwa output) renders like 'instrumental'
    blue = _stem_color("instrumental")
    check(_stem_color("instrument") == blue,
          "instrument must share instrumental light blue, got "
          f"{_stem_color('instrument')}")
    # karaoke config short labels: 'instrum' / 'back instrum' (and the
    # underscore spelling as it appears on disk) are the backing track
    for short in ("instrum", "back_instrum", "back instrum",
                  "backing instrum", "backing_instrum"):
        check(_stem_color(short) == blue,
              f"{short} must share instrumental light blue, got "
              f"{_stem_color(short)}")

    # phantom-centre / mid-side + drum-kit stems: center/wide and similarity
    # get identity colors of their own; singular drum + cymbals/ride/crash
    # join the drums amber; short 'inst' = instrumental light blue
    lime = _stem_color("center")
    cyan = _stem_color("wide")
    check(cyan != lime, "wide must differ from center (no mid/side collision)")
    check(_stem_color("similarity") == "#6D28D9",
          f"similarity must be dark violet #6D28D9, got "
          f"{_stem_color('similarity')}")
    for kit in ("drum", "cymbals", "ride", "crash"):
        check(_stem_color(kit) == amber,
              f"{kit} must be drums-family amber, got {_stem_color(kit)}")
    check(_stem_color("inst") == blue,
          f"inst must share instrumental light blue, got {_stem_color('inst')}")
    # a phantom-centre model must not repaint its 'similarity' target with
    # the phantom badge lime once the stem has its own color
    from ui.pages.console_page import _stem_override_for_model
    class _Card:
        _model_type = "phantom centre"
        _model_name = "mdx23c similarity"
        _target_stem = "similarity"
    check(_stem_override_for_model(_Card(), "Similarity") is None,
          "phantom badge must not override similarity's own color")

    # a "keys"-type organ model must not repaint its 'organ' target with the
    # keys badge blue — the organ stem keeps its own ochre #996E10
    class _OrganCard:
        _model_type = "keys"
        _model_name = "organ separation"
        _target_stem = "organ"
    check(_stem_override_for_model(_OrganCard(), "Organ") is None,
          "keys badge must not override organ's own ochre")
    check(_stem_color("organ") == "#996E10",
          f"organ must stay ochre #996E10, got {_stem_color('organ')}")

    # Full mega-53 cluster spec: every catalog stem pins to a family color,
    # so nothing drifts onto the fallback palette.
    _53_SPEC = {
        "guitar": ["acoustic-guitar", "banjo", "dobro", "electric-guitar",
                   "guitar", "mandolin", "sitar", "ukulele"],
        "wind": ["accordion", "bassoon", "brass", "clarinet",
                 "french-horn", "oboe", "saxophone", "trombone",
                 "trumpet", "tuba", "wind", "wind-chimes", "woodwind"],
        "flutes": ["flute", "harmonica"],
        "vocals": ["back-vocal", "lead-vocal", "vocal"],
        "strings": ["bowed_strings", "cello", "double-bass", "harp",
                     "strings", "viola", "violin"],
        "bass": ["bass"],
        "drums": ["congas", "drums", "hh", "kick", "snare",
                   "tambourine", "toms", "triangle"],
        "percussion": ["bells", "glockenspiel", "marimba", "percussion",
                        "timpani"],
        "keys": ["digital-piano", "harpsichord", "keys", "piano"],
        "organs": ["organ"],
        "synth": ["synth"],
    }
    counted = 0
    for anchor, members in _53_SPEC.items():
        anchor_col = _stem_color(anchor)
        for m in members:
            counted += 1
            check(_stem_color(m) == anchor_col,
                  f"{m} must match {anchor} family color "
                  f"{anchor_col}, got {_stem_color(m)}")
    check(counted == 53, f"53-stem spec must cover 53 stems, covers {counted}")
    fam_colors = {_stem_color(a) for a in _53_SPEC}
    check(len(fam_colors) == len(_53_SPEC),
          "family anchors must stay distinct from each other")

    # fallback solo-load path (samples=None) still normalizes to its own peak
    tr_alone = _WaveformTrack("Solo", "#ffffff", "#222222", None)
    tr_alone.load_audio(music)
    check(float(np.max(np.abs(tr_alone._samples))) > 0.95, "solo fallback load broken")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
