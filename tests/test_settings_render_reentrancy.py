"""Regression: a folder-fetch completing while the install dialog is open
must NOT rebuild the manager list.

A user session crashed repeatedly with a native
"Windows fatal exception: access violation" on the main thread at a plain
widget allocation (QVBoxLayout()/QLabel()/QPushButton() constructor) inside
settings_page.py `_render_inside`, called from `_install`'s post-dialog
re-render — even after every QThread in the app was converted to plain
daemon-thread workers. Root cause: the install dialog is opened from an
Install button *inside the list that gets rebuilt*. While that modal dialog's
nested event loop spins (minutes, during a download), a background
`_FolderFetchThread` can complete; its `_finish_folder_fetch` called
`self._render()`, which `deleteLater()`s every list widget — including the
Install button whose clicked emission is parked inside `dialog.exec()`. The
nested modal loop then processes those deletions underneath the live signal
emission; destroying a sender mid-emission is native use-after-free in Qt.
It corrupts the heap without crashing immediately, and explodes at the next
widget allocation on the main thread — the random access violations in the
re-render.

Fix: `_FolderManagerWidget._request_render()` defers (coalesced) any render
requested while `QApplication.activeModalWidget()` is non-None; the deferred
render runs exactly once after the modal closes.

This test reproduces the exact interleaving headless: a real modal install
dialog opened by clicking a real row's Install button, while a real
`_FolderFetchThread` for another folder lands mid-modal. It asserts zero
renders happened while the modal was up and that the deferred rebuild then
applied the fetch.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QPushButton, QLabel, QVBoxLayout,
)

import ui.pages.settings_page as sp  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def pump_until(pred, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        QApplication.processEvents()
        if pred():
            return True
        time.sleep(0.004)
    QApplication.processEvents()
    return False


def make_models():
    from backend.model_manager import ModelInfo
    out = []
    for i in range(10):
        out.append(ModelInfo(
            key=f"mbr_model_{i}", full_name=f"Mel-Band Test {i}",
            arch="Melband Roformer Architecture", stem_type="multi stems",
            category="4 стема", model_type="mel_band_roformer",
            stems=["vocals", "drums", "bass", "other"], target_instrument=None,
            checkpoint_url=f"https://huggingface.co/x/resolve/main/mbr_{i}.ckpt",
            config_url=f"https://huggingface.co/x/resolve/main/mbr_{i}.yaml",
            file_size=1000 + i,
        ))
    for i in range(8):
        out.append(ModelInfo(
            key=f"vr_model_{i}", full_name=f"VR Test {i}",
            arch="VR Architecture", stem_type="karaoke",
            category="Караоке", model_type="vr",
            stems=["vocals", "instrumental"], target_instrument=None,
            checkpoint_url=f"https://huggingface.co/x/resolve/main/vr_{i}.ckpt",
            config_url=f"https://huggingface.co/x/resolve/main/vr_{i}.yaml",
            file_size=500 + i,
        ))
    return out


class _FakeInstallDialog(QDialog):
    """Stand-in for ModelInstallDialog: opens modally like the real one and
    self-closes via accept after the vr folder fetch has had time to land
    inside the modal loop."""

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self._info = info
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("fake install"))
        # The real app's dialog stays open for the whole download (minutes);
        # here 60 ms is enough for the vr fetch (20 ms stub) to complete
        # inside the modal loop, then we close.
        QTimer.singleShot(200, self.accept)


def main():
    app = QApplication([])

    models = make_models()

    def fake_index():
        time.sleep(0.03)
        return models

    sp.fetch_model_index = fake_index
    sp.fetch_repo_meta = lambda: "2026-01-01T00:00:00Z"
    sp.fetch_tree_info = lambda: {"sub": {}, "meta": "2026-01-01T00:00:00Z"}

    def fake_folder_tree(key):
        time.sleep(0.02)  # lands mid-modal when the dialog is open
        dates = {m.checkpoint_url.split("/")[-1]: "2026-01-01T00:00:00Z"
                 for m in models if m.model_type == key}
        return dates, {n: 1234 for n in dates}

    sp.fetch_folder_tree = fake_folder_tree
    sp.is_installed = lambda info: False

    # Swap the real install dialog for the fake (module-level name the
    # manager resolves at call time).
    sp.ModelInstallDialog = _FakeInstallDialog

    mgr = sp._FolderManagerWidget()
    check(pump_until(lambda: getattr(mgr, "_models", None)),
          "manager index loaded")

    # Expand the mel-band folder fully (fetch done, rows with Install buttons).
    mgr._toggle_folder("mel_band_roformer")
    check(pump_until(lambda: "mel_band_roformer" in mgr._folder_meta),
          "mel_band folder meta fetched")
    check(pump_until(lambda: len(sp._ACTIVE_FETCH_WORKERS) == 0),
          "mel_band fetch registry drained")

    def count_install_btns():
        return sum(1 for b in mgr.findChildren(QPushButton)
                   if b.text() == "Install")

    n_before = count_install_btns()
    check(n_before >= 10, f">=10 mel-band Install buttons, got {n_before}")

    # Start a SECOND folder's fetch, then immediately click an Install button
    # from the first folder. The vr fetch (20 ms stub) completes while the
    # install dialog's modal loop is spinning — the exact interleaving that
    # used to deleteLater the whole tree under the parked click emission.
    mgr._toggle_folder("vr")
    check(pump_until(lambda: "vr" in mgr._loading_folders),
          "vr folder fetch started (in flight)")

    install_btns = [b for b in mgr.findChildren(QPushButton)
                    if b.text() == "Install"]
    clicked_btn = install_btns[0]

    # Instrument renders: record any that happen while a modal is up. With
    # the fix there must be none; without it, the vr fetch's completion
    # would call _render() mid-modal (deleteLater under the parked click).
    calls_while_modal = []
    orig_render = mgr._render

    def tracked_render(search_term=""):
        from PySide6.QtWidgets import QApplication
        if QApplication.activeModalWidget() is not None:
            calls_while_modal.append(search_term)
        return orig_render(search_term)

    mgr._render = tracked_render
    try:
        # Synchronous: runs _install -> fake dialog.exec() (nested modal
        # loop; vr fetch lands and the timer accept closes it).
        clicked_btn.click()

        # The deferred (post-modal) rebuild must have applied the vr fetch:
        # both folder groups' rows now render.
        check(pump_until(lambda: count_install_btns() >= 10 + 8),
              f"deferred rebuild rendered both groups, got "
              f"{count_install_btns()}")
    finally:
        mgr._render = orig_render

    check(calls_while_modal == [],
          f"no render ran while the modal was up (got {len(calls_while_modal)}"
          f" — a mid-modal rebuild deleteLater's the parked sender button)")
    check(mgr._render_pending is None,
          "no render left pending after the modal closed")
    check("vr" in mgr._folder_meta and "vr" in mgr._expanded,
          "vr folder fetch was applied")
    check(pump_until(lambda: len(sp._ACTIVE_FETCH_WORKERS) == 0),
          "fetch registry drained after install flow")
    # The clicked button's tree was rebuilt after the emission unwound — the
    # manager still has live, queryable children (no native crash / dangling).
    check(len(mgr.findChildren(QPushButton)) > 0,
          "manager widget tree alive after install flow")

    # Second scenario: fetch lands while NO modal — renders immediately.
    mgr2 = sp._FolderManagerWidget()
    check(pump_until(lambda: getattr(mgr2, "_models", None)),
          "mgr2 index loaded")
    mgr2._toggle_folder("mel_band_roformer")
    check(pump_until(lambda: "mel_band_roformer" in mgr2._folder_meta),
          "mgr2 folder fetch applied without modal")

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        for f in FAILURES:
            print("  FAIL:", f)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
