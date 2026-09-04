"""Regression test: pre-run validation of registered model types against the
inference engine's actual dispatch branches.

Bug history: a model whose stored ``model_type`` had no branch in the engine
(e.g. a zoo model newer than the build, or a hand-registered config whose
type label drifted) launched inference.py fine, then died mid-start with a
raw ``ValueError: Unknown model type: ...`` traceback in the console.

The fix validates *before* spawning: the GUI resolves the effective type the
engine will build (yaml ``training.model_type`` overrides the CLI type,
mirroring utils/settings.get_model_from_config) and checks it against the
engine's real branch list — parsed out of the source by
backend.msst_catalog (the same catalog the TRAINING tab uses) plus the ONNX
'mdxnet' special case handled directly in inference.py.

Runs offline (no torch, no network). Qt is imported only to reach the pure
helper resolve_engine_model_type; nothing is instantiated.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import msst_catalog  # noqa: E402
from backend.msst_catalog import (  # noqa: E402
    engine_effective_type,
    engine_supported_type_list,
    engine_unsupported_models,
    guess_registry_engine_type,
)
from ui.pages.inference_page import (  # noqa: E402
    has_custom_sidecar,
    resolve_engine_model_type,
)

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def _write(tmp, text):
    p = os.path.join(tmp, "cfg.yaml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def main():
    parsed = msst_catalog.model_types()
    engine = msst_catalog.engine_model_types()

    # ── Catalog sanity: parse must succeed and mirror the engine ──────────
    check("catalog parsed get_model_from_config branches (26)", len(parsed) == 26)
    check("engine list = branches + mdxnet", len(engine) == len(parsed) + 1)
    check("mdxnet (ONNX special case) included", "mdxnet" in engine)
    check("no duplicate entries", len(engine) == len(set(engine)))

    # Every model_type the live zoo can hand a registered model must be
    # runnable by this build (all are covered by branches or the mdxnet
    # ONNX case). Hardcoded snapshot so the test also fails when a *new*
    # zoo type arrives without an engine branch.
    zoo_types = {
        "mel_band_roformer", "bs_roformer", "mdxnet", "vr", "mdx23c",
        "htdemucs", "scnet", "medley_vox", "bandit", "scnet_masked",
        "scnet_tran", "bandit_v2",
    }
    missing = sorted(zoo_types - set(engine))
    check("every zoo model_type has an engine branch", not missing)
    if missing:
        print("   zoo types WITHOUT an engine branch:", ", ".join(missing))

    # Junk / drifted labels must be caught.
    check("bogus type rejected", "hyperace_v9000" not in engine)
    check("variant types present", {"scnet_tran", "scnet_masked", "bandit_v2",
                                    "bs_roformer_unwa_large", "vr",
                                    "medley_vox"} <= set(engine))

    # ── Effective-type resolution (mirror of get_model_from_config) ───────
    with tempfile.TemporaryDirectory() as tmp:
        # yaml with training.model_type wins over the CLI type
        p = _write(tmp, "training:\n  model_type: scnet_tran\nmodel: {}\n")
        check("yaml training.model_type overrides",
              resolve_engine_model_type("scnet", p) == "scnet_tran")

        # yaml without training.model_type -> CLI type preserved
        p2 = _write(tmp, "training:\n  instruments: [vocals, other]\nmodel: {}\n")
        check("no override keeps CLI type",
              resolve_engine_model_type("vr", p2) == "vr")

        # variant sniffs mirror the engine: bandit kwargs layout -> bandit_v2
        p3 = _write(tmp, "training:\n  model_type: bandit\nkwargs: {}\n")
        check("bandit kwargs layout resolves to bandit_v2",
              resolve_engine_model_type("bandit", p3) == "bandit_v2")
        # ... and scnet tran_* model keys -> scnet_tran even without override
        p4 = _write(tmp, "training:\n  instruments: [vocals, other]\n"
                         "model:\n  tran_depth: 6\n  num_blocks: 2\n")
        check("scnet tran keys resolve to scnet_tran",
              resolve_engine_model_type("scnet", p4) == "scnet_tran")
        # a plain scnet yaml stays scnet
        p5 = _write(tmp, "training:\n  instruments: [vocals, other]\nmodel: {}\n")
        check("plain scnet stays scnet",
              resolve_engine_model_type("scnet", p5) == "scnet")

        # unreadable / missing yaml -> CLI type preserved (fail-open)
        check("missing yaml keeps CLI type",
              resolve_engine_model_type("vr", os.path.join(tmp, "nope.yaml"))
              == "vr")
        check("empty yaml path keeps CLI type",
              resolve_engine_model_type("mdx23c", "") == "mdx23c")

    # ── Membership rule the GUI applies before spawning ───────────────────
    # (catalog engine types are the same values the page unions with mdxnet)
    supported = set(parsed) | {"mdxnet"}
    for ok in ("mdx23c", "scnet_tran", "vr", "mdxnet", "bandit_v2",
               "medley_vox", "dttnet"):
        check(f"'{ok}' is supported", ok in supported)
    for bad in ("hyperace_v9000", "dtt_net", "scnet_v3", ""):
        check(f"'{bad or '<empty>'}' is rejected", bad not in supported)

    # ── Custom side-cars live under the writable app dir (survives updates) ─
    from backend.paths import APP_DIR, REPO_ROOT as _repo_root  # noqa: E402
    import inspect as _inspect
    from backend import msst_catalog as _cat  # noqa: E402
    dflt = _inspect.signature(_cat.has_custom_sidecar).parameters["repo_root"].default
    check("sidecar default root = writable APP_DIR", dflt == APP_DIR)
    check("dev checkout: APP_DIR is the repo root", APP_DIR == _repo_root)

    # ── Runner-level pre-flight helpers (GUI + both ensemble runners) ─────
    # engine_effective_type mirrors each launch's resolution: stored type or
    # arch fallback, then yaml override + variant sniffs (mdxnet exempt).
    mapping = {"MDX23c Architecture": "mdx23c",
               "MDX-Net Architecture": "mdxnet",
               "BS Roformer Architecture": "bs_roformer",
               "SCNet Architecture": "scnet"}
    good = {"name": "G", "arch": "MDX23c Architecture",
            "model_type": "mdx23c", "yaml": "", "ckpt": ""}
    stale = {"name": "S", "arch": "MDX23c Architecture",
             "model_type": "hyperace_v9000", "yaml": "", "ckpt": ""}
    archonly = {"name": "A", "arch": "SCNet Architecture",
                "yaml": "", "ckpt": ""}   # no stored type -> scnet
    check("effective type from stored type",
          engine_effective_type(good, mapping) == "mdx23c")
    check("effective type from arch fallback",
          engine_effective_type(archonly, mapping) == "scnet")
    check("mdxnet never yaml-overridden",
          engine_effective_type({"name": "N", "arch": "MDX-Net Architecture",
                                 "model_type": "mdxnet", "yaml": ""}, mapping)
          == "mdxnet")
    problems = engine_unsupported_models([good, stale, archonly], mapping)
    check("unsupported scan flags only the stale model",
          [(m["name"], eff) for m, eff in problems] == [("S", "hyperace_v9000")])
    check("supported type list includes all branches + mdxnet",
          len(engine_supported_type_list()) == len(set(parsed) | {"mdxnet"}))

    # ── Side-car exemption: fork models bypass the built-in branches ──────
    # (both file-style and folder-style layouts the engine accepts)
    from backend.paths import REPO_ROOT  # noqa: E402
    custom_dir = os.path.join(REPO_ROOT, "models", "custom")
    os.makedirs(custom_dir, exist_ok=True)
    probe = os.path.join(custom_dir, "_sidecar_probe.py")
    probe_dir = os.path.join(custom_dir, "_sidecar_folder")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("# probe")
        os.makedirs(probe_dir, exist_ok=True)
        with open(os.path.join(probe_dir, "bs_roformer.py"), "w",
                  encoding="utf-8") as f:
            f.write("# author backend")
        fork = {"custom_backend_enabled": True,
                "backend_module": "_sidecar_probe.py"}
        folder = {"custom_backend_enabled": True,
                  "backend_module": "_sidecar_folder"}
        check("enabled + file-style sidecar -> exempt",
              has_custom_sidecar(fork) is True)
        check("enabled + folder-style sidecar -> exempt",
              has_custom_sidecar(folder) is True)
        check("enabled but missing sidecar -> not exempt",
              has_custom_sidecar({"custom_backend_enabled": True,
                                  "backend_module": "_nope.py"}) is False)
        check("folder without backend file -> not exempt",
              has_custom_sidecar({"custom_backend_enabled": True,
                                  "backend_module": "models"}) is False
              or has_custom_sidecar(
                  {"custom_backend_enabled": True,
                   "backend_module": "_sidecar_folder_absent"}) is False)
        check("path traversal rejected",
              has_custom_sidecar({"custom_backend_enabled": True,
                                  "backend_module": "../x"}) is False)
        check("sidecar disabled -> not exempt",
              has_custom_sidecar({"custom_backend_enabled": False,
                                  "backend_module": "_sidecar_probe.py"})
              is False)
        check("empty model dict -> not exempt",
              has_custom_sidecar({}) is False)
    finally:
        if os.path.isfile(probe):
            os.remove(probe)
        import shutil as _sh
        if os.path.isdir(probe_dir):
            _sh.rmtree(probe_dir, ignore_errors=True)

    # ── DTTNet: arch label must resolve to the engine's 'dttnet' branch ───
    # (the pre-run validation / badge flagged a manual URL install as NOT
    # RUNNABLE because the arch map spelled the type 'dtt_net', which has no
    # bundled branch, and the folder-style side-car wasn't recognized.)
    dtt_mapping = {"DTTNet Architecture": "dttnet"}
    dtt = {"name": "dttnet_vocalsg32_ep4082_fix.ckpt",
           "arch": "DTTNet Architecture", "yaml": "", "ckpt": ""}
    check("DTTNet arch resolves to dttnet",
          engine_effective_type(dtt, dtt_mapping) == "dttnet")
    check("dttnet is an engine branch", "dttnet" in parsed)
    check("dtt_net is not a bundled branch", "dtt_net" not in parsed)
    check("DTTNet entry not flagged unsupported",
          engine_unsupported_models([dtt], dtt_mapping) == [])

    # ── Registry type detection for manual / URL installs ─────────────────
    with tempfile.TemporaryDirectory() as tmp2:
        p1 = _write(tmp2, "training:\n  model_type: dtt_net\n")
        check("registry: yaml declares dtt_net -> aliased to dttnet",
              guess_registry_engine_type(p1, "x.ckpt") == "dttnet")
        p2 = _write(tmp2, "training:\n  instruments: [vocals, other]\n")
        check("registry: no yaml type -> name match dttnet",
              guess_registry_engine_type(p2, "dttnet_vocalsg32_ep4082_fix.ckpt")
              == "dttnet")
    check("registry: name match mdx23c",
          guess_registry_engine_type("", "model_mdx23c_vocals.ckpt") == "mdx23c")
    check("registry: no match -> empty (arch map fallback)",
          guess_registry_engine_type("", "totally_unknown_model.ckpt") == "")

    # Real on-disk config (when present) routes to a supported variant.
    hits = []
    for root, _dirs, names in os.walk(os.path.join(REPO_ROOT, "configs")):
        for n in names:
            if n.endswith(".yaml") and "scnet_tran" in n:
                hits.append(os.path.join(root, n))
    if hits:
        eff = resolve_engine_model_type("scnet", hits[0])
        check(f"real {os.path.basename(hits[0])} resolves to {eff}",
              eff == "scnet_tran" and eff in supported)
    else:
        print("SKIP real-config check (no scnet_tran config in repo)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
