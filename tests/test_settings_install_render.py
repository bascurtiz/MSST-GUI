"""Regression test: post-install re-render deferral + layout-clear fix.

The settings model-manager crashed natively (Windows "fatal exception:
access violation") three times across versions — always at the very first
widget allocation (a plain QLabel in _ElidedLabel.__init__) of the
re-render that followed a model install. That render used to run
synchronously inside the Install button's clicked emission while the
just-finished modal dialog's blur was animating out. This test pins the
fix: _install must NOT render or emit synchronously (the rebuild is
deferred with a 0-ms singleShot so the clicked emission fully unwinds
first), and _clear must take rows out of the layout instead of leaving
pending-delete widgets coexisting with the fresh render.

Exercises the real _FolderManagerWidget offscreen with the network
fetchers stubbed — no HuggingFace, no modal loop, no subprocess.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui.pages.settings_page as sp  # noqa: E402
import backend.settings as bs  # noqa: E402
from backend.model_manager import ModelInfo  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def _canned_models():
    return [
        ModelInfo(
            key="vocals_demo", full_name="Demo Vocal Model",
            arch="BS Roformer Architecture", stem_type="vocals",
            category="vocals", model_type="bs_roformer",
            stems=["vocals", "other"], target_instrument="vocals",
            checkpoint_url=("https://huggingface.co/demo/resolve/main/"
                            "vocals_demo.ckpt"),
            config_url=("https://huggingface.co/demo/resolve/main/"
                        "vocals_demo.yaml"),
        ),
        ModelInfo(
            key="drums_demo", full_name="Demo Drum Model",
            arch="MDX23c Architecture", stem_type="drums",
            category="drums", model_type="mdx23c",
            stems=["drums"], target_instrument="drums",
            checkpoint_url=("https://huggingface.co/demo/resolve/main/"
                            "drums_demo.ckpt"),
            config_url=("https://huggingface.co/demo/resolve/main/"
                        "drums_demo.yaml"),
        ),
    ]


def _install_info():
    return ModelInfo(
        key="bass_demo", full_name="Demo Bass Model",
        arch="MDX23c Architecture", stem_type="bass",
        category="bass", model_type="mdx23c",
        stems=["bass"], target_instrument="bass",
        checkpoint_url=("https://huggingface.co/demo/resolve/main/"
                        "bass_demo.ckpt"),
        config_url=("https://huggingface.co/demo/resolve/main/"
                    "bass_demo.yaml"),
    )


def main():
    app = QApplication.instance() or QApplication([])

    # Stub the network fetchers and the settings store — the manager's
    # constructor immediately spawns the zoo-index fetch thread.
    sp.fetch_model_index = lambda: _canned_models()
    sp.fetch_repo_meta = lambda: ""
    sp.fetch_tree_info = lambda: {}
    bs.load = lambda: {"registered_models": [], "_model_tracker": {}}
    bs.save = lambda data: None

    widget = sp._FolderManagerWidget()
    # Let the (stubbed) index fetch land and the initial render run. Poll
    # instead of one fixed wait — the fetch runs on a real daemon thread, so
    # arrival time varies with scheduler/GIL timing.
    for _ in range(60):
        if len(widget._folder_order) >= 2:
            break
        QTest.qWait(50)
    QTest.qWait(50)  # let the resulting render settle

    # ── 1. _clear removes rows from the layout instead of leaving them ──
    # After the initial render: one folder card per type + trailing stretch.
    check("initial render: stretch + 2 folder cards",
          widget._list_layout.count() == 3)
    check("initial render: folder cards present",
          widget._list_layout.itemAt(0) is not None
          and widget._list_layout.itemAt(0).widget() is not None
          and widget._list_layout.itemAt(1).widget() is not None)

    # A fresh render must not accumulate rows: the old cards are takeAt'd
    # (pending delete), so the layout holds exactly stretch + new cards.
    widget._render()
    check("re-render: layout holds stretch + 2 fresh cards (no duplicates)",
          widget._list_layout.count() == 3)

    # ── 2. _install defers the rebuild out of the click emission ──
    render_calls = []
    orig_render = widget._request_render
    widget._request_render = lambda *a, **k: (
        render_calls.append(1), orig_render(*a, **k))[1]
    installed = []
    widget.model_installed.connect(lambda: installed.append(1))

    sp.run_blurred_dialog = lambda dlg: sp.ModelInstallDialog.Accepted
    widget._install(_install_info())

    check("install: no synchronous re-render inside the click emission",
          len(render_calls) == 0)
    check("install: no synchronous model_installed inside the emission",
          len(installed) == 0)

    # The deferred rebuild runs from the main loop once the emission (and
    # the modal dialog it drove) has fully unwound.
    QTest.qWait(60)
    check("install: deferred re-render ran exactly once",
          len(render_calls) == 1)
    check("install: deferred model_installed fired exactly once",
          len(installed) == 1)
    check("install: list rebuilt after install",
          widget._list_layout.count() == 3)

    # ── 3. Sanity: no stray pending render timers re-fire renders ──
    before = len(render_calls)
    QTest.qWait(200)
    check("install: no duplicate renders from stray timers",
          len(render_calls) == before)

    widget.deleteLater()
    app.processEvents()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())