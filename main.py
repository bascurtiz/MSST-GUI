"""
MSST GUI — entry point.
Run this from the root of the Music-Source-Separation-Training repo:
    python msst_gui/main.py

Debugging the frozen exe:
    The shipped exe is windowed (no terminal opens on launch). All output
    still lands in msst-gui.log (in the app-data dir, see
    backend.paths.get_app_dir): prints and tracebacks via a tee, Qt
    warnings/fatals via the installed Qt message handler, and hard crashes
    via faulthandler's all-thread stack dump. To see output live in the
    terminal you launched from, run:  MSST-GUI.exe --console
    A fatal error (main thread or a crashed background thread) also pops
    an in-app dialog showing the full traceback before the app closes.
"""
import sys
import os

if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS
    sys.path.insert(0, BASE)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if BASE not in sys.path:
        sys.path.insert(0, BASE)

# --- console + logging bootstrap (before any heavy imports) ---------------
_DEBUG_FLAGS = {"--console", "--debug"}
_LOG_MAX_BYTES = 2 * 1024 * 1024


def _wants_console():
    return any(a in _DEBUG_FLAGS for a in sys.argv[1:])


def _strip_flags(argv):
    return [a for a in argv if a not in _DEBUG_FLAGS]


def _attach_parent_console():
    """Print to the terminal the exe was launched from, when there is one."""
    if sys.platform != "win32":
        return
    import ctypes
    k32 = ctypes.windll.kernel32
    if k32.GetConsoleWindow():
        return  # already attached (pythonw builds never are)
    try:
        if not k32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            return
    except Exception:
        return
    try:
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except Exception:
        pass


class _Tee:
    """Fan stdout/stderr out to several streams (None entries ignored)."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _open_log_file(app_dir):
    """Append-mode log file with simple rotation (msst-gui.log -> .old)."""
    log_path = os.path.join(app_dir, "msst-gui.log")
    try:
        if os.path.isfile(log_path) and \
                os.path.getsize(log_path) > _LOG_MAX_BYTES:
            old = log_path + ".old"
            if os.path.isfile(old):
                os.remove(old)
            os.replace(log_path, old)
        return open(log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return None


from backend.paths import get_app_dir  # noqa: E402  (light: os/sys only)

_log_file = _open_log_file(get_app_dir())
if _wants_console():
    _attach_parent_console()

# Windowed PyInstaller builds start with stdout/stderr = None; tee whatever
# exists together with the log file so prints and tracebacks are never lost.
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
sys.argv = _strip_flags(sys.argv)

# Hard-crash visibility: a segfault/abort in native code (Qt multimedia,
# torch, etc.) kills the process without any Python traceback, so none of
# the excepthooks above fire and the log just ends silently. faulthandler
# dumps every thread's Python stack into the log file the instant a fatal
# signal hits (works for access violations + abort() on Windows too).
import faulthandler  # noqa: E402
import datetime  # noqa: E402
try:
    if _log_file is not None:
        faulthandler.enable(file=_log_file, all_threads=True)
except Exception:
    pass

# The log is append-mode across runs; a divider + timestamp makes each
# session's tail obvious, so a session that ends without an explicit
# "closing" line is visibly a crash.
try:
    from backend.version import APP_VERSION  # noqa: E402
    _ver = APP_VERSION
except Exception:
    _ver = "?"
try:
    print("\n" + "=" * 70)
    print(f"[MSST-GUI] session start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
          f"(v{_ver})")
    print("=" * 70)
except Exception:
    pass


def _format_traceback(tp, val, tb):
    import io
    import traceback
    buf = io.StringIO()
    traceback.print_exception(tp, val, tb, file=buf)
    return buf.getvalue()


def _log_hint():
    try:
        return os.path.join(get_app_dir(), "msst-gui.log")
    except Exception:
        return "msst-gui.log"


_dialog_active = False


def _show_fatal_dialog(title, summary, trace):
    """Present the traceback in an in-app modal dialog.

    Main thread only (worker-thread crashes are marshaled there by
    _marshal_fatal_dialog). Falls back to the console/log output when Qt
    isn't available, so the info is never lost."""
    global _dialog_active
    # A second fatal while the dialog is up must not recurse into widget
    # creation (recursive-repaint access violation) — the log already has
    # this one's traceback.
    if _dialog_active:
        return
    _dialog_active = True
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is None:
            return
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(summary)
        box.setDetailedText(trace)  # collapsible, selectable traceback
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()
    except Exception:
        pass  # console + log output still cover this case
    finally:
        _dialog_active = False


from PySide6.QtCore import QObject, Signal  # noqa: E402


class _FatalBridge(QObject):
    """Owned by the main thread (created there by _install_fatal_bridge).
    Emitting `fatal` from any thread delivers the dialog onto the main
    thread's event loop via the queued connection; from the main thread it
    runs synchronously (direct connection), like the old behavior."""
    fatal = Signal(str, str, str)


_fatal_bridge = None


def _install_fatal_bridge():
    """Create the bridge on the main thread; call once after QApplication."""
    global _fatal_bridge
    if _fatal_bridge is None:
        _fatal_bridge = _FatalBridge()
        _fatal_bridge.fatal.connect(_show_fatal_dialog)


def _marshal_fatal_dialog(title, summary, trace):
    """Show the fatal dialog safely from any thread.

    Worker threads can't touch widgets: emitting on the main-thread-owned
    bridge queues the call onto the main event loop. Before the bridge
    exists (early startup) only a main-thread call is safe to show
    directly."""
    bridge = _fatal_bridge
    if bridge is not None:
        try:
            bridge.fatal.emit(title, summary, trace)
            return
        except Exception:
            pass  # fall through to the log-only path
    try:
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is app.thread():
            _show_fatal_dialog(title, summary, trace)
    except Exception:
        pass


def _excepthook(tp, val, tb):
    trace = _format_traceback(tp, val, tb)
    print(trace, end="")
    log_path = _log_hint()
    print("\n[MSST-GUI] The app hit an unhandled error and is closing.")
    print(f"[MSST-GUI] Traceback also saved to: {log_path}")
    _marshal_fatal_dialog(
        "MSST-GUI — fatal error",
        "The app hit an unhandled error and will close.\n\n"
        "The full traceback is below (Details) and was saved to:\n"
        f"{log_path}",
        trace,
    )
    # Hold the terminal open on fatal errors so the traceback can be read
    # before the console closes with the process (skipped when no console /
    # interactive stdin exists, e.g. windowed or piped runs).
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input("Press Enter to close this window...")
    except Exception:
        pass


sys.excepthook = _excepthook


def _thread_excepthook(args):
    """Surface background-thread crashes (QThread.run, worker threads) with a
    clear banner and an in-app dialog instead of a bare traceback."""
    trace = _format_traceback(
        args.exc_type, args.exc_value, args.exc_traceback)
    print("\n[MSST-GUI] A background thread crashed:")
    print(trace, end="")
    log_path = _log_hint()
    print(f"[MSST-GUI] Traceback also saved to: {log_path}")
    _marshal_fatal_dialog(
        "MSST-GUI — background thread crash",
        "A background thread crashed and the app may close or behave "
        "unexpectedly.\n\n"
        "The full traceback is below (Details) and was saved to:\n"
        f"{log_path}",
        trace,
    )


import threading  # noqa: E402
try:
    threading.excepthook = _thread_excepthook
except Exception:
    pass

from PySide6.QtCore import qInstallMessageHandler, QtMsgType  # noqa: E402


def _qt_message_handler(mode, _context, message):
    print(f"[Qt] {message}")
    # Qt aborts right after a fatal message handler returns, and faulthandler
    # does not always fire for Qt's terminate path on Windows (a qFatal such
    # as 'QThread: Destroyed while thread is still running' can die without
    # any Python-level dump). Dump every thread's Python stack into the log
    # NOW so a fatal Qt condition always leaves a trace behind.
    if mode == QtMsgType.QtFatalMsg and _log_file is not None:
        try:
            _log_file.write("\n[Qt] FATAL — all-thread Python stacks before abort:\n")
            _log_file.flush()
            faulthandler.dump_traceback(file=_log_file, all_threads=True)
            _log_file.flush()
        except Exception:
            pass


qInstallMessageHandler(_qt_message_handler)
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import backend.settings as settings_store
import backend.runtime_setup  # noqa: F401  (sets PYTHONNOUSERSITE early when frozen)
from ui.main_window import MainWindow
from ui.theme import theme_manager, apply_palette


def main():
    # Named mutex so installers/updaters (Inno Setup AppMutex) can detect a
    # running instance and ask the user to close it before upgrading.
    try:
        import ctypes
        ctypes.windll.kernel32.CreateMutexW(None, False, "MSST-GUI-Mutex")
    except Exception:
        pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    # Worker-thread crashes must never touch widgets; the fatal-dialog
    # bridge lives here on the main thread and marshals queued dialog
    # requests onto this thread's event loop (see _marshal_fatal_dialog).
    _install_fatal_bridge()
    app.setApplicationName("MSST")
    app.setOrganizationName("ZFTurbo")

    # Qt's ICO reader crashes on the bundled favicon.ico (OS/2-style DIB),
    # so use the pre-decoded PNG copy for the window/taskbar icon.
    icon_path = os.path.join(BASE, "resources", "app_icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Diagnostic self-check (MSST_FATAL_SELFTEST=1): deliberately raise a Qt
    # fatal error right after boot so the crash-visibility chain — Qt message
    # handler prints into console + log, then all-thread stacks are dumped
    # before Qt aborts — can be verified on demand against the frozen exe.
    if os.environ.get("MSST_FATAL_SELFTEST"):
        from PySide6.QtCore import QTimer, qFatal as _qFatal
        print("[MSST-GUI] selftest: raising an intentional Qt fatal error...")
        QTimer.singleShot(
            0,
            lambda: _qFatal("MSST_FATAL_SELFTEST: intentional Qt fatal error"))
        app.exec()  # never returns: qFatal aborts the process
        sys.exit(1)

    apply_palette(app)

    saved_theme = settings_store.load().get("ui", {}).get("theme", "dark")
    theme_manager.set_mode(saved_theme, persist=False)

    # Startup splash: logo + title + a live progress bar over the heavy
    # synchronous startup phases (page construction, settings load).
    from ui.widgets.splash import SplashPanel
    try:
        from backend import update_checker as _uc
        _version = _uc.app_version()
    except Exception:
        _version = ""
    splash = SplashPanel(BASE, version=_version)
    splash.set_stage("Starting application...", 5)
    splash.show()
    app.processEvents()

    try:
        window = MainWindow(progress_cb=splash.set_stage)
    except Exception:
        splash.close_now()
        raise
    window.show()
    splash.raise_()
    # Startup phases are done — theme rebuilds must not drive the splash.
    window.set_progress_callback(None)

    # Kick off the background runtime probe immediately so the job-start gate
    # reads a cached verdict instead of cold-importing torch on the UI thread
    # the first (and every) time the user hits a job button. The splash shows
    # this as its last stage; the probe keeps running after the splash fades.
    from ui.widgets.runtime_dialog import prime_runtime_check
    splash.set_stage("Checking runtime environment...", 92)
    prime_runtime_check()
    splash.finish("Ready")

    # Log when the event loop ends so the log's last lines distinguish a
    # clean exit (window closed / quit() called -> "closing cleanly") from
    # a hard crash (log simply stops mid-session, caught by faulthandler).
    try:
        app.aboutToQuit.connect(
            lambda: print("[MSST-GUI] event loop closing cleanly..."))
    except Exception:
        pass

    rc = app.exec()
    print(f"[MSST-GUI] event loop exited with code {rc}.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
