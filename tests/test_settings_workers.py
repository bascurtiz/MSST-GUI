"""Regression: the settings/console/update background workers are plain
QObjects on daemon threads — never QThreads.

A user session crashed while adding models in Settings with a native
"Windows fatal exception: access violation" inside a plain
`QPushButton("Install")` constructor (faulthandler dump:
settings_page.py:1566 _render_inside, called from _install's post-dialog
re-render). The crash path still ran raw QThread subclasses — _MgrFetchThread,
_FolderFetchThread, _DownloadWorker — including a destroy-while-running race
(_on_loaded dropped the only reference to the index QThread right after its
done.emit, while run() may not have returned yet) and an orphan path that
deleteLater'd QThread wrappers as Python GC released them. QThread wrappers
destroyed while their thread is still winding down corrupt Qt's thread state;
the damage surfaces later as a random native crash on the main thread.

The fix: all three settings workers plus the console friendly-names and
update-check workers are QObject signal emitters driven by plain daemon
threads and kept referenced in module-level registries until they finish —
no QThread object exists to leak, destroy, or corrupt.

These checks run headless (offscreen QApplication) with all network fetches
stubbed — no torch, no network.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

import ui.pages.settings_page as sp  # noqa: E402
import ui.pages.console_page as cp  # noqa: E402
import ui.widgets.update_dialog as ud  # noqa: E402

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
    for i in range(14):
        out.append(ModelInfo(
            key=f"mbr_model_{i}", full_name=f"Mel-Band Test {i}",
            arch="Melband Roformer Architecture", stem_type="multi stems",
            category="4 стема", model_type="mel_band_roformer",
            stems=["vocals", "drums", "bass", "other"],
            target_instrument=None,
            checkpoint_url=f"https://huggingface.co/x/resolve/main/mbr_{i}.ckpt",
            config_url=f"https://huggingface.co/x/resolve/main/mbr_{i}.yaml",
            file_size=1000 + i,
        ))
    for i in range(12):
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


def main():
    app = QApplication([])

    # Static: converted classes are not QThreads and keep their public API.
    for cls in (sp._MgrFetchThread, sp._FolderFetchThread, sp._DownloadWorker,
                cp._FriendlyNamesThread, ud._CheckThread):
        check(not issubclass(cls, QThread), f"{cls.__name__} is not a QThread")
        check(hasattr(cls, "start") and hasattr(cls, "isRunning"),
              f"{cls.__name__} keeps start/isRunning")
        check(not issubclass(cls, QThread) and cls.__mro__[1] is not QThread,
              f"{cls.__name__} base is not QThread")

    # Stub the network fetches the manager workers use.
    models = make_models()

    def fake_index():
        time.sleep(0.05)
        return models

    sp.fetch_model_index = fake_index
    sp.fetch_repo_meta = lambda: "2026-01-01T00:00:00Z"
    sp.fetch_tree_info = lambda: {"sub": {}, "meta": "2026-01-01T00:00:00Z"}

    def fake_folder_tree(key):
        time.sleep(0.02)
        dates = {m.checkpoint_url.split("/")[-1]: "2026-01-01T00:00:00Z"
                 for m in models if m.model_type == key}
        return dates, {n: 1234 for n in dates}

    sp.fetch_folder_tree = fake_folder_tree

    # 1. Build the manager: index worker loads + renders, registry drains.
    mgr = sp._FolderManagerWidget()
    ok = pump_until(lambda: getattr(mgr, "_models", None))
    check(ok, "manager index loaded")
    check(not mgr._fetch_thread or not mgr._fetch_thread.isRunning(),
          "index worker finished")
    ok = pump_until(lambda: len(sp._ACTIVE_FETCH_WORKERS) == 0)
    check(ok, f"fetch registry drained, got {len(sp._ACTIVE_FETCH_WORKERS)}")
    check(len(mgr._folder_order) == 2, f"two folder groups, got {mgr._folder_order}")

    # 2. Expand a folder: folder worker fetches, rows render with the exact
    #    QPushButton('Install') path from the crash dump.
    mgr._toggle_folder("mel_band_roformer")
    ok = pump_until(lambda: "mel_band_roformer" in mgr._folder_meta)
    check(ok, "folder meta fetched")
    ok = pump_until(lambda: len(sp._ACTIVE_FETCH_WORKERS) == 0)
    check(ok, f"folder registry drained, got {len(sp._ACTIVE_FETCH_WORKERS)}")

    def count_install_btns():
        return sum(1 for b in mgr.findChildren(QPushButton)
                   if b.text() == "Install")

    check(count_install_btns() >= 14,
          f">=14 Install buttons rendered, got {count_install_btns()}")

    # 3. Churn: repeatedly re-render (mirrors install-accept -> _render with
    #    135 models) while more folder workers start mid-flight.
    for i in range(40):
        mgr._render()
        QApplication.processEvents()
        if i in (5, 15, 25):
            mgr._toggle_folder("vr")
    ok = pump_until(lambda: "vr" in mgr._folder_meta)
    check(ok, "vr folder fetched under churn")
    check(count_install_btns() >= 14 + 12,
          f"rows render under churn: {count_install_btns()}")
    ok = pump_until(lambda: len(sp._ACTIVE_FETCH_WORKERS) == 0)
    check(ok, f"registry drained after churn, got {len(sp._ACTIVE_FETCH_WORKERS)}")

    # 4. Orphan-strip mid-flight: a fresh manager starts its index + folder
    #    fetches, then gets orphaned + deleted before they finish. No QThread
    #    exists to destroy; workers drain on their own and late signals die.
    mgr2 = sp._FolderManagerWidget()
    mgr2._toggle_folder("vr")
    pump_until(lambda: len(sp._ACTIVE_FETCH_WORKERS) >= 1, timeout=3.0)
    check(sp.orphan_fetch_threads(mgr2) is None,
          "orphan returns None (nothing to defer)")
    mgr2.deleteLater()
    ok = pump_until(lambda: len(sp._ACTIVE_FETCH_WORKERS) == 0, timeout=6.0)
    check(ok, f"orphaned workers still drain, got {len(sp._ACTIVE_FETCH_WORKERS)}")
    QApplication.processEvents()
    QApplication.processEvents()

    # 5. Console friendly-names worker: done emitted, registry drains.
    cp.fetch_model_index = lambda: models
    name_w = cp._FriendlyNamesThread()
    got = []
    name_w.done.connect(lambda *a: got.append(a))
    name_w.start()
    check(name_w in cp._ACTIVE_NAMES_WORKERS, "names worker registered")
    ok = pump_until(lambda: got)
    check(ok, "names worker emitted done")
    ok = pump_until(lambda: len(cp._ACTIVE_NAMES_WORKERS) == 0)
    check(ok, f"names registry drained: {len(cp._ACTIVE_NAMES_WORKERS)}")

    # 6. Update-check worker: done emitted, registry drains.
    ud.uc.check_for_update = lambda: (False, None)
    ck = ud._CheckThread()
    got2 = []
    ck.done.connect(lambda *a: got2.append(a))
    ck.start()
    check(ck in ud._ACTIVE_CHECK_WORKERS, "check worker registered")
    ok = pump_until(lambda: got2)
    check(ok, f"check worker emitted done (got2={got2})")
    ok = pump_until(lambda: len(ud._ACTIVE_CHECK_WORKERS) == 0)
    check(ok, f"check registry drained: {len(ud._ACTIVE_CHECK_WORKERS)}")

    # 7. Download worker: cancel + success paths, registry drains (network
    #    stubbed via the download_utils module the worker imports at runtime).
    from backend import download_utils

    class _FakeSession:
        def close(self):
            pass

    def _fake_stream(url, dest, progress_callback=None, should_cancel=None,
                     chunk_size=1048576, timeout=(60, 60), session=None):
        if progress_callback:
            progress_callback("stream", 100, 100)
        return True, "ok"

    def _fake_parallel(url, dest, progress_callback=None, speed_callback=None,
                       should_cancel=None, session=None):
        if progress_callback:
            progress_callback("parallel", 100, 100)
        if speed_callback:
            speed_callback(1.0)
        return True, "ok"

    download_utils.stream_download = _fake_stream
    download_utils.parallel_download = _fake_parallel
    sp.HuggingFaceDownloader.to_direct_download_url = staticmethod(lambda u: u)

    dw = sp._DownloadWorker("http://x/a.ckpt", "http://x/a.yaml",
                            "C:/nope/a.ckpt", "C:/nope/a.yaml")
    dw._session = _FakeSession()
    dw._cancelled = True  # downloads (stubbed) run, then the cancel path
    got3 = []
    dw.finished.connect(lambda *a: got3.append(a))
    dw.start()
    check(dw in sp._ACTIVE_DOWNLOAD_WORKERS, "download worker registered")
    ok = pump_until(lambda: got3, timeout=6.0)
    check(ok, f"download worker finished (got3={got3})")
    check(got3 and got3[0][0] is False, f"cancel reported, got {got3}")
    ok = pump_until(lambda: len(sp._ACTIVE_DOWNLOAD_WORKERS) == 0)
    check(ok, f"download registry drained: {len(sp._ACTIVE_DOWNLOAD_WORKERS)}")

    dw2 = sp._DownloadWorker("http://x/b.ckpt", "http://x/b.yaml",
                             "C:/nope/b.ckpt", "C:/nope/b.yaml")
    dw2._session = _FakeSession()
    got4 = []
    dw2.finished.connect(lambda *a: got4.append(a))
    dw2.start()
    ok = pump_until(lambda: got4, timeout=6.0)
    check(ok and got4 and got4[0][0] is True
          and got4[0][2].get("ckpt_name") == "b.ckpt",
          f"download success path emitted, got {got4}")
    ok = pump_until(lambda: len(sp._ACTIVE_DOWNLOAD_WORKERS) == 0)
    check(ok, "download registry drained (2)")

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        for f in FAILURES:
            print("  FAIL:", f)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
