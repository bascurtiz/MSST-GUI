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

from backend.paths import REPO_ROOT

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
