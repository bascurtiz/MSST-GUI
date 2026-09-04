"""utils/bnb_stub.py
-------------------
Make ``torch.load`` tolerate checkpoints that reference ``bitsandbytes``
classes even though the package is not installed in the runtime.

Several published SCNet / roformer checkpoints (e.g. the ``scnet_huge_*``
family by ``_aname``) were saved from trainer state that includes a
bitsandbytes 8-bit optimizer (``bitsandbytes.optim.adamw.AdamW8bit`` and
friends). ``torch.load`` re-imports those classes to unpickle the object
graph, and dies with ``ModuleNotFoundError: No module named 'bitsandbytes'``
before we ever get to the weights — even though the *weights* are plain
torch tensors that load into the bundled architecture.

We never need the real package: the bnb objects are only unpickled as
opaque blobs (optimizer state that inference ignores). Installing a
meta-path finder that fabricates permissive stand-in modules/classes under
the ``bitsandbytes.*`` namespace lets the pickle graph unroll; the state
dict is then extracted and loaded as usual.

The finder is only installed when the real package is NOT importable, so a
runtime that does ship bitsandbytes is never shadowed.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

_BASE = "bitsandbytes"


class _BnbDummy:
    """Maximally permissive stand-in class for any bnb class reference.

    Pickle may reconstruct instances through ``__new__``/``__setstate__``
    (optimizer classes ship a ``__getstate__`` returning
    ``(defaults, state, param_groups)``, which unpickling replays onto the
    reconstructed object). Accept anything and store nothing.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _BnbDummy()

    def __getattr__(self, name):
        return _BnbDummy()


class _BnbStubModule(types.ModuleType):
    """A fake package under ``bitsandbytes.*`` that fabricates members on
    demand (both nested packages and the classes pickle looks up)."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in sys.modules:  # already-registered subpackage
            return sys.modules[name]
        # Declare the fake module as the class's home so pickling an
        # instance also resolves back through the finder (round-trip safe).
        cls = type(name, (_BnbDummy,), {"__module__": self.__name__})
        setattr(self, name, cls)
        return cls


class _BnbFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Meta-path finder that answers any ``bitsandbytes[.*]`` import."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _BASE or fullname.startswith(_BASE + "."):
            return importlib.machinery.ModuleSpec(
                fullname, self, is_package=True
            )
        return None

    def create_module(self, spec):
        return _BnbStubModule(spec.name)

    def exec_module(self, module):
        pass  # stub packages have no real content


_installed = False


def install():
    """Register the stub finder (once) if the real package is absent."""
    global _installed
    if _installed:
        return
    try:
        if importlib.util.find_spec(_BASE) is not None:
            return  # real bitsandbytes present — do not shadow it
    except (ImportError, ValueError):
        pass
    sys.meta_path.insert(0, _BnbFinder())
    _installed = True


install()