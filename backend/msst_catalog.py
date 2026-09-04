"""
backend/msst_catalog.py
-----------------------
What the upstream training code can actually do, read from the code itself.

The TRAINING tab must not carry its own copies of the model-type, optimizer,
loss and metric lists: when a new architecture, optimizer or loss lands in
Music-Source-Separation-Training, the tab should offer it after a plain file
sync. So these lists are parsed out of the real sources with `ast` — no
import of utils/ (that would pull torch into the GUI process):

  model types   utils/settings.py   get_model_from_config: `model_type == '…'`
                                    / `model_type in ('…', …)` branches
  optimizers    utils/model_utils.py get_optimizer: `name_optimizer == '…'`
  losses        utils/settings.py   parse_args_train --loss choices
                utils/losses.py     choice_loss docstring "- `name`: …" bullets
  metrics       utils/settings.py   parse_args_train --metrics choices
  schedulers    utils/settings.py   get_scheduler: `scheduler_name == '…'`
  model names   docs/README_MSST.md "* <Name> … Key: `key`" lines

Everything is cached after the first read; call refresh() to re-parse.
"""
import ast
import os
import re

from backend.paths import REPO_ROOT, APP_DIR

_SETTINGS = os.path.join(REPO_ROOT, "utils", "settings.py")
_MODEL_UTILS = os.path.join(REPO_ROOT, "utils", "model_utils.py")
_LOSSES = os.path.join(REPO_ROOT, "utils", "losses.py")
_README = os.path.join(REPO_ROOT, "docs", "README_MSST.md")

_cache = {}


def refresh():
    """Forget every parsed list (next access re-reads the sources)."""
    _cache.clear()


# ── source helpers ────────────────────────────────────────────────────────────

def _parse(path):
    try:
        with open(path, "rb") as f:
            return ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError) as exc:
        print(f"[catalog] cannot parse {path}: {exc}")
        return None


def _find_function(tree, name):
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _string_consts(node):
    """String literal(s) a Compare's right-hand side names: 'x' or ('x', 'y')."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _compared_literals(func, var_name):
    """Every string a function compares `var_name` against, in source order:
    `var_name == 'a'`, `var_name in ('a', 'b')`."""
    found = []
    if func is None:
        return found
    for node in ast.walk(func):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == var_name):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.In)):
                for s in _string_consts(comp):
                    if s not in found:
                        found.append(s)
    return found


def _argparse_choices(func, option):
    """The `choices=[...]` of `parser.add_argument("<option>", ...)` inside
    a function (upstream's parse_args_train), or [] when absent."""
    if func is None:
        return []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == option):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return _string_consts(kw.value)
    return []


def _argparse_default(func, option):
    if func is None:
        return None
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == option):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    return ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    return None
    return None


def _cached(key, builder):
    if key not in _cache:
        _cache[key] = builder()
    return _cache[key]


# ── model types ───────────────────────────────────────────────────────────────

def model_types():
    """Model-type keys get_model_from_config() knows how to build, in the
    order the code lists them (['mdx23c', 'htdemucs', ...])."""
    def build():
        func = _find_function(_parse(_SETTINGS), "get_model_from_config")
        return _compared_literals(func, "model_type")
    return _cached("model_types", build)


# The engine's accepted set is not only get_model_from_config's branches:
# inference.py handles 'mdxnet' (ONNX MDX-Net checkpoints) directly in
# proc_folder, before get_model_from_config is ever called. Kept here so the
# GUI can pre-validate a registered model's type against what the engine will
# actually run (see ui/pages/inference_page.py._run_inner).
_ONNX_SPECIAL_CASE = ("mdxnet",)


def engine_model_types():
    """Model-type values inference.py can actually run: every branch parsed
    out of get_model_from_config plus the ONNX 'mdxnet' special case handled
    directly in proc_folder.

    If utils/settings.py cannot be read, model_types() is empty and this
    returns only ['mdxnet'] — callers that want to *block* on an unsupported
    type must treat a failed parse as "unknown, don't block" (the GUI checks
    model_types() separately for that)."""
    def build():
        types = list(model_types())
        for t in _ONNX_SPECIAL_CASE:
            if t not in types:
                types.append(t)
        return types
    return _cached("engine_model_types", build)


# Fork side-cars live under the WRITABLE app-data root (APP_DIR) — in the
# frozen app REPO_ROOT is _internal, which the installer wipes on every
# update, so files stored there would vanish with the ckpt. APP_DIR equals
# REPO_ROOT in a dev checkout.
_CUSTOM_ROOT = APP_DIR


# ── per-model pre-run validation (GUI + ensemble runners) ───────────────────
# A registered model entry may carry a model_type the engine has no branch
# for (a zoo model newer than the build, a drifted label such as 'dtt_net'
# vs 'dttnet', or a hand-registered config). The helpers below resolve the
# type each launch would *actually* build — mirroring inference.py and
# get_model_from_config — and say whether it is runnable. They are pure and
# Qt-free, so the inference page and both ensemble runners share one verdict.


def resolve_engine_model_type(model_type: str, yaml_path: str) -> str:
    """The model type the engine will actually build for a launch, mirroring
    utils/settings.get_model_from_config end-to-end: a `training.model_type`
    recorded in the yaml overrides the type passed on the command line, then
    the variant sniffs refine coarse labels (bandit `kwargs:` layout -> 
    bandit_v2; scnet `tran_*` model keys -> scnet_tran). Falls back to the
    caller's type when there is no yaml or it can't be read."""
    if not (yaml_path and os.path.isfile(yaml_path)):
        return model_type
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.load(_f, Loader=yaml.FullLoader) or {}
        resolved = model_type
        _tm = (_cfg.get("training") or {}).get("model_type")
        if isinstance(_tm, str) and _tm.strip():
            resolved = _tm.strip()
        # Variant sniffs — same rules and order as the engine.
        if resolved == "bandit" and "model" not in _cfg and "kwargs" in _cfg:
            resolved = "bandit_v2"
        elif resolved == "scnet":
            _sm = _cfg.get("model") or {}
            if isinstance(_sm, dict) and any(
                    str(k).startswith("tran_") for k in _sm):
                resolved = "scnet_tran"
        return resolved
    except Exception:
        return model_type


_CUSTOM_BACKEND_FILE_NAMES = ("bs_roformer.py", "model.py", "models.py")


def has_custom_sidecar(model: dict, repo_root: str = _CUSTOM_ROOT) -> bool:
    """True when the model entry ships an author side-car backend that the
    engine will load instead of the built-in dispatch branches (fork
    architectures such as pcunwa's HyperACE / BS-Roformer-Large-Inst).

    Accepts both layouts the engine's _load_custom_backend supports:
      * file-style:  models/custom/<backend_module>            (a .py file)
      * folder-style: models/custom/<backend_module>/bs_roformer.py
        (the manual / URL install flows put the ckpt, yaml and the author
        file renamed to bs_roformer.py inside one folder)."""
    if not (model or {}).get("custom_backend_enabled"):
        return False
    bm = (model or {}).get("backend_module") or ""
    if not bm or os.path.isabs(bm) or ".." in bm.replace("\\", "/").split("/"):
        return False
    base = os.path.join(repo_root, "models", "custom", bm)
    if os.path.isfile(base):
        return True
    if os.path.isdir(base):
        return any(os.path.isfile(os.path.join(base, n))
                   for n in _CUSTOM_BACKEND_FILE_NAMES)
    return False


# Upstream spells DTTNet both ways (Music-Source-Separation-Training uses
# 'dtt_net' in places, the engine's dispatch branch is 'dttnet'); registries
# must store the spelling the engine can actually build.
_REGISTRY_TYPE_ALIASES = {"dtt_net": "dttnet"}


def guess_registry_engine_type(yaml_path: str, ckpt_name: str = "") -> str:
    """Best precise engine type for a registry entry being (re)registered:
    1) config.training.model_type when the yaml declares it (spelling
    normalized to what the engine can build), else
    2) a name match against the engine branch keys (longest first).

    Returns '' when neither source yields a confident match — callers then
    fall back to their arch→type map."""
    mt = ""
    if yaml_path and os.path.isfile(yaml_path):
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as _f:
                _cfg = yaml.load(_f, Loader=yaml.FullLoader) or {}
            _tm = (_cfg.get("training") or {}).get("model_type")
            if isinstance(_tm, str) and _tm.strip():
                mt = _tm.strip()
        except Exception:
            mt = ""
    if mt:
        return _REGISTRY_TYPE_ALIASES.get(mt, mt)
    for name in (yaml_path or "", ckpt_name or ""):
        if not name:
            continue
        guess = guess_model_type_from_name(name)
        if guess:
            return guess
    return ""


def engine_effective_type(model: dict, arch_to_model_type: dict) -> str:
    """The engine type a launch of this model entry will build: the stored
    precise type (install-time) or the arch→type fallback, then the yaml
    override + variant sniffs via resolve_engine_model_type. mdxnet is
    special-cased in inference.py before the config is consulted, so its
    yaml never overrides."""
    model = model or {}
    base = (model.get("model_type") or ""
            or arch_to_model_type.get(model.get("arch", ""), "bs_roformer"))
    if base == "mdxnet":
        return "mdxnet"
    return resolve_engine_model_type(base, model.get("yaml") or "")


def engine_unsupported_models(models, arch_to_model_type: dict,
                              repo_root: str = _CUSTOM_ROOT):
    """Every model entry whose effective engine type has no dispatch branch.

    Returns a list of ``(model, effective_type)`` tuples. Returns [] when the
    engine source cannot be parsed (fail open: an unknown catalog must never
    block a run). Custom side-car models are exempt — the engine loads their
    class from the author's file."""
    known = model_types()
    if not known:
        return []
    problems = []
    for model in models or []:
        eff = engine_effective_type(model, arch_to_model_type)
        if has_custom_sidecar(model, repo_root):
            continue
        if eff not in known and eff != "mdxnet":
            problems.append((model, eff))
    return problems


def engine_supported_type_list() -> list:
    """Sorted list of every model type the engine can run, for messages."""
    return sorted(set(engine_model_types()))


def model_display_names():
    """{key: human name} from the upstream README's model list
    ("* Mel-Band RoFormer [...] Key: `mel_band_roformer`.")."""
    def build():
        names = {}
        try:
            with open(_README, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = re.search(r"Key:\s*`([A-Za-z0-9_]+)`", line)
                    if not m or not line.lstrip().startswith("*"):
                        continue
                    key = m.group(1)
                    head = line.lstrip()[1:].strip()
                    # name = text up to the first link/"based on"/"Key:"
                    name = re.split(r"\s*(?:\[\[|\[|based on|Key:)", head, 1)[0].strip(" .")
                    if name:
                        names[key] = name
        except OSError:
            pass
        return names
    return _cached("model_names", build)


def model_display_name(key):
    name = model_display_names().get(key)
    if name:
        return name
    # bs_roformer_experimental -> "Bs Roformer Experimental"
    return " ".join(w.upper() if w in ("mdx23c", "stht", "bs", "dttnet") else w.capitalize()
                    for w in key.split("_"))


def model_type_choices():
    """[(key, display), ...] for a combo box."""
    return [(k, model_display_name(k)) for k in model_types()]


# ── optimizers / schedulers ───────────────────────────────────────────────────

def optimizers():
    """Optimizer names get_optimizer() accepts (config.training.optimizer)."""
    def build():
        func = _find_function(_parse(_MODEL_UTILS), "get_optimizer")
        return _compared_literals(func, "name_optimizer")
    return _cached("optimizers", build)


def schedulers():
    """Scheduler names get_scheduler() accepts (config.training.scheduler)."""
    def build():
        func = _find_function(_parse(_SETTINGS), "get_scheduler")
        return _compared_literals(func, "scheduler_name")
    return _cached("schedulers", build)


# ── losses / metrics ──────────────────────────────────────────────────────────

def losses():
    """Loss names accepted by train.py --loss, in declaration order."""
    def build():
        func = _find_function(_parse(_SETTINGS), "parse_args_train")
        return _argparse_choices(func, "--loss")
    return _cached("losses", build)


def loss_descriptions():
    """{loss: one-line description} from choice_loss()'s docstring bullets
    ("- `masked_loss`: robust, quantile-masked MSE ...")."""
    def build():
        func = _find_function(_parse(_LOSSES), "choice_loss")
        doc = ast.get_docstring(func) if func is not None else ""
        descs = {}
        for m in re.finditer(r"-\s*`([A-Za-z0-9_]+)`\s*(?:/\s*`([A-Za-z0-9_]+)`\s*)?:\s*(.+?)(?=\n\s*-\s*`|\n\s*\n|\Z)",
                             doc or "", re.S):
            text = " ".join(m.group(3).split())
            for key in (m.group(1), m.group(2)):
                if key:
                    descs[key] = text
        return descs
    return _cached("loss_descriptions", build)


def default_losses():
    func = _find_function(_parse(_SETTINGS), "parse_args_train")
    d = _argparse_default(func, "--loss")
    return list(d) if isinstance(d, (list, tuple)) else ["masked_loss"]


def metrics():
    """Metric names accepted by train.py --metrics, in declaration order."""
    def build():
        func = _find_function(_parse(_SETTINGS), "parse_args_train")
        return _argparse_choices(func, "--metrics")
    return _cached("metrics", build)


def default_metrics():
    func = _find_function(_parse(_SETTINGS), "parse_args_train")
    d = _argparse_default(func, "--metrics")
    return list(d) if isinstance(d, (list, tuple)) else ["sdr"]


def _pretty(key):
    return key.replace("_", " ")


def loss_choices():
    """[(key, label, description), ...] for the LOSS picker."""
    descs = loss_descriptions()
    return [(k, _pretty(k), descs.get(k, "")) for k in losses()]


def metric_choices():
    """[(key, label, description), ...] for the METRICS picker."""
    return [(k, _pretty(k), "") for k in metrics()]


# ── config-name → model type ──────────────────────────────────────────────────

def guess_model_type_from_name(filename):
    """Best model-type key for a config file name, matched against the keys
    the code knows (longest key first, underscores optional), e.g.
    'config_vocals_mel_band_roformer.yaml' -> 'mel_band_roformer',
    'bs_roformer_inst.yaml' -> 'bs_roformer', 'mbr_x.yaml' -> None."""
    name = os.path.basename(filename).lower()
    compact = name.replace("_", "").replace("-", "")
    for key in sorted(model_types(), key=len, reverse=True):
        if key in name or key.replace("_", "") in compact:
            return key
    # loose family matches derived from the keys themselves ("…roformer…")
    for key in sorted(model_types(), key=len, reverse=True):
        last = key.split("_")[-1]
        if len(last) >= 6 and last in name:
            return key
    return None
