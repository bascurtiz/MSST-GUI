"""
utils/stem_planning.py
----------------------
Pure stem-planning helpers shared by the inference engine (inference.py),
the GUI (ui/pages/inference_page.py) and the test suite.

They encode one coupled decision:

  * A single-output model (``model.num_stems <= 1`` with a declared
    ``training.target_instrument``) can only ever separate that one trained
    stem.  Every other stem listed in the config is *derived* as the
    mix-complement (mix minus the separated stems) and must be labeled with
    its real config name — e.g. a ``[vocals, instrument]`` config targeting
    ``instrument`` yields a derived stem named ``vocals``, not the generic
    ``instrumental``.
  * Multi-output models separate every trained stem directly and never
    invent a complement label.

Kept free of Qt and torch so the GUI process, the engine subprocess and the
offline tests can all import it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Sentinel: leave the config's target_instrument untouched.
KEEP = object()


def is_single_output(config: Dict[str, Any]) -> bool:
    """True when the model can only emit its trained target stem.

    A config is single-output when ``model.num_stems`` is 1 (or missing, the
    upstream default) *and* ``training.target_instrument`` is declared —
    without a target there is nothing to derive complements from.
    """
    try:
        num_stems = int((config.get("model", {}) or {}).get("num_stems", 1) or 1)
    except Exception:
        num_stems = 1
    target = (config.get("training", {}) or {}).get("target_instrument")
    return num_stems <= 1 and bool(target)


def complement_stem_name(
    all_instruments: Sequence[str],
    separated_stems: Sequence[str],
) -> str:
    """Name for the auto-derived rest stem (mix - separated stems).

    For a 2-instrument single-output config (e.g. ``[vocals, instrument]``
    targeting ``instrument``) the complement IS the other trained stem, so
    label it with that name ("vocals").  Multi-stem configs — where the
    complement is not any single trained stem — fall back to the generic
    upstream "instrumental" label.
    """
    others = [i for i in all_instruments if i not in separated_stems]
    if len(others) == 1:
        return others[0]
    return "instrumental"


def rest_needed(
    config: Dict[str, Any],
    selected_stems: Sequence[str],
    save_rest: bool = False,
    all_stems: Optional[Sequence[str]] = None,
) -> bool:
    """Mirror the GUI's auto-enable rule for ``--extract_instrumental``.

    Save-rest is turned on when the user checked it, or when a stem other
    than the config's target is selected (that stem can only be obtained as
    the mix-complement).
    """
    if save_rest:
        return True
    target = (config.get("training", {}) or {}).get("target_instrument") or ""
    target = str(target).lower()
    all_stems = (
        list(all_stems) if all_stems is not None
        else list((config.get("training", {}) or {}).get("instruments", []) or [])
    )
    sel_lower = {s.lower() for s in selected_stems}
    return bool(target and len(all_stems) > 1
                and any(s != target for s in sel_lower))


def resolve_target(
    config: Dict[str, Any],
    selected_stems: Sequence[str],
    effective_save_rest: bool,
    custom_selection: bool,
):
    """What ``training.target_instrument`` the temp yaml should carry.

    Returns one of:
      * ``KEEP``      — leave the yaml's value untouched
      * a stem name  — explicit single-stem selection (multi-output models
                       only; a single-output model can only ever emit its
                       trained target, so it is never overridden)
      * ``None``      — drop the target so the engine emits every trained
                        stem (multi-output models only)
    """
    try:
        num_stems = int((config.get("model", {}) or {}).get("num_stems", 1) or 1)
    except Exception:
        num_stems = 1
    if not custom_selection and not effective_save_rest:
        return KEEP
    if len(selected_stems) == 1 and not effective_save_rest:
        if num_stems > 1:
            return selected_stems[0]
        return KEEP
    if num_stems > 1 and "training" in config:
        # Drop the target so the engine emits every trained stem — but only
        # when the config has a training section to modify.
        return None
    return KEEP


def plan_output_stems(
    config: Dict[str, Any],
    selected_stems: Sequence[str],
    save_rest: bool = False,
    all_stems: Optional[Sequence[str]] = None,
) -> List[str]:
    """Predict the exact stem names the engine will write for a selection.

    Follows the same path as the GUI temp-yaml logic (``resolve_target`` +
    ``rest_needed``) and inference.py's complement step
    (``complement_stem_name``).  Used by the regression tests to catch stem
    mislabeling for single-output configs.
    """
    instruments = list((config.get("training", {}) or {}).get("instruments", []) or [])
    all_stems = list(all_stems) if all_stems is not None else instruments[:]

    custom_selection = (
        sorted(s.lower() for s in selected_stems) != sorted(s.lower() for s in all_stems)
    ) if all_stems else False

    rest = rest_needed(config, selected_stems, save_rest, all_stems)
    target = resolve_target(config, selected_stems, rest, custom_selection)
    if target is KEEP:
        target = (config.get("training", {}) or {}).get("target_instrument") or None

    # Engine: prefer_target_instrument(config) -> [target] or all stems.
    if target:
        separated = [str(target)]
    else:
        separated = instruments[:]

    out = separated[:]
    if rest:
        complement = complement_stem_name(instruments, separated)
        if complement not in out:
            out.append(complement)
    return out