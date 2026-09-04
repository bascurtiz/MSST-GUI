"""
backend/runner.py
-----------------
Runs inference or training as a subprocess and streams stdout/stderr
back to the GUI via Qt signals.  The GUI thread never blocks.

History: this class was originally a QThread subclass. QThread wrappers
that are garbage-collected while their thread is still winding down corrupt
Qt's thread state, surfacing later as random native "access violation"
crashes — faulthandler dump: backend/runner.py:30 (super().__init__()) hit
from inference_page._run_inner mid TEST ALL MODELS batch. It is now a plain
QObject signal emitter driven by a daemon thread, kept alive in a
module-level registry until the job finishes — no QThread object exists to
leak, destroy, or corrupt.
"""
import os
import subprocess
import sys
import threading

from PySide6.QtCore import QObject, Signal


# Module-level registry: a running worker is referenced here so the Python
# wrapper can never be garbage-collected mid-run (the crash the QThread
# rewrite eliminates). Entries are removed the moment the job finishes.
_ACTIVE_RUNNERS = set()
_ACTIVE_LOCK = threading.Lock()


class ProcessRunner(QObject):
    """
    Runs a subprocess on a plain daemon thread and emits its output line by
    line.  Signals are emitted from the worker thread and delivered queued
    to the GUI thread (the receiver lives there), so the UI never blocks.

    Signals
    -------
    log_line(str)      – one line of stdout/stderr text
    progress(int)      – estimated percent (0-100), parsed from tqdm output
    finished(int)      – return code when process exits
    """

    log_line = Signal(str)
    progress = Signal(int)
    finished = Signal(int)

    def __init__(self, cmd: list[str], cwd: str | None = None,
                 env: dict | None = None):
        super().__init__()
        self._cmd = cmd
        self._cwd = cwd or os.getcwd()
        # Extra environment for the child (merged over os.environ), e.g.
        # CUDA_VISIBLE_DEVICES="" for a CPU-only training run.
        self._extra_env = dict(env) if env else {}
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Spawn the worker thread and stream the child's output."""
        if self._thread and self._thread.is_alive():
            return
        with _ACTIVE_LOCK:
            _ACTIVE_RUNNERS.add(self)
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        """Ask the running process to terminate gracefully."""
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def isRunning(self):
        """True while the worker thread is still streaming the process."""
        return bool(self._thread and self._thread.is_alive())

    def wait(self, msecs: int | None = None) -> bool:
        """Block until the worker thread finishes (mirrors QThread.wait):
        returns True when the job is done, False on timeout."""
        if not self._thread:
            return True
        self._thread.join(msecs / 1000.0 if msecs else None)
        return not self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def run(self):
        """Stream the child process; runs on the worker thread."""
        try:
            env = os.environ.copy()
            # The child inherits a pipe, so cp.stdio buffers print()/tqdm
            # output in multi-KB blocks unless we force unbuffered mode —
            # otherwise the GUI sees progress in bursts, or only after the
            # process exits. Read that batch as it's produced.
            env['PYTHONUNBUFFERED'] = '1'
            # Force UTF-8 stdio on the child: on Windows a pipe-inheriting
            # python defaults to the ANSI code page (e.g. cp1252), so any
            # non-ASCII in the job output (accented track names, the em-dash
            # in dataset messages) would be read back as replacement chars
            # by the utf-8 decode below. UTF-8 mode also makes the child's
            # file writes (train_log.txt) utf-8, matching the GUI's own.
            env['PYTHONUTF8'] = '1'
            env.update(self._extra_env)
            self._process = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                cwd=self._cwd,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            # tqdm updates use '\r' (in-place redraw) not '\n', so a plain
            # line-by-line reader would only see progress after a real '\n'
            # arrives. Split on BOTH delimiters to stream every update.
            buf = ""
            for chunk in self._process.stdout:
                buf += chunk.replace("\r", "\n")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self.log_line.emit(line)
                    pct = _parse_tqdm_percent(line)
                    if pct is not None:
                        self.progress.emit(pct)
            if buf.strip():
                self.log_line.emit(buf.strip())

            self._process.wait()
            self.finished.emit(_coerce_exit_code(self._process.returncode))
        except Exception as exc:
            self.log_line.emit(f"[ERROR] {exc}")
            self.finished.emit(_coerce_exit_code(None))
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_RUNNERS.discard(self)


# ------------------------------------------------------------------
# Exit-code helpers
# ------------------------------------------------------------------

# Windows reports a child that died natively as an *unsigned* 32-bit
# NTSTATUS (e.g. an access violation is 3221225477, not -1). Emitting that
# raw value through Signal(int) makes shiboken print a libshiboken Overflow
# warning and leaves a pending exception in the emitter's thread that
# surfaces later as a SystemError at an unrelated point (in the field: an
# overflow escaping the worker into sys.excepthook, which then built a
# QMessageBox on the worker thread and died with a recursive-repaint access
# violation). Coerce to the signed 32-bit value Windows itself reports so
# signal delivery can never overflow.
_NTSTATUS_NAMES = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000000D: "STATUS_INVALID_PARAMETER",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC0000135: "STATUS_DLL_NOT_FOUND",
    0xC000013A: "STATUS_CONTROL_C_EXIT",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}


def _coerce_exit_code(code):
    """Return a value that always fits a Qt Signal(int) (signed 32-bit)."""
    if code is None:
        return -1
    code = int(code)
    if code >= 1 << 31:
        code -= 1 << 32
    return code


def describe_exit_code(code) -> str:
    """Human-readable exit code for console output. A child that died
    natively becomes '0xC0000005 (STATUS_ACCESS_VIOLATION)' instead of a
    cryptic bare '-1073741819'. Accepts the coerced signed form or a raw
    unsigned value from a caller that didn't coerce."""
    if code is None:
        return "-1"
    code = int(code)
    if code == 0:
        return "0"
    if code >= 1 << 31:
        code -= 1 << 32  # raw unsigned NTSTATUS (e.g. 3221225477)
    if code < 0:
        u = code + (1 << 32)
        name = _NTSTATUS_NAMES.get(u)
        if name:
            return f"0x{u:08X} ({name})"
        return f"0x{u:08X} (STATUS_0x{u:08X})"
    return str(code)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _parse_tqdm_percent(line: str) -> int | None:
    """
    Try to extract a percentage from a tqdm progress line such as:
        ' 42%|████      | 42/100 [00:05<00:07]'
    Returns an int 0-100 or None if the line doesn't contain one.
    """
    try:
        if "%" in line and "|" in line:
            pct_str = line.strip().split("%")[0].strip().split()[-1]
            return int(float(pct_str))
    except Exception:
        pass
    return None