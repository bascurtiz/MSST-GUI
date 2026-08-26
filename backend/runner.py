"""
backend/runner.py
-----------------
Runs inference or training as a subprocess and streams stdout/stderr
back to the GUI via Qt signals.  The GUI thread never blocks.
"""
import subprocess
import sys
import os
from PySide6.QtCore import QThread, Signal


class ProcessRunner(QThread):
    """
    QThread that spawns a subprocess and emits its output line by line.

    Signals
    -------
    log_line(str)      – one line of stdout/stderr text
    progress(int)      – estimated percent (0-100), parsed from tqdm output
    finished(int)      – return code when process exits
    """

    log_line = Signal(str)
    progress = Signal(int)
    finished = Signal(int)

    def __init__(self, cmd: list[str], cwd: str | None = None):
        super().__init__()
        self._cmd = cmd
        self._cwd = cwd or os.getcwd()
        self._process: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self):
        """Ask the running process to terminate gracefully."""
        if self._process and self._process.poll() is None:
            self._process.terminate()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def run(self):
        """Called automatically by QThread.start()."""
        try:
            self._process = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self._cwd,
            )

            for raw_line in self._process.stdout:
                line = raw_line.rstrip("\n")
                self.log_line.emit(line)
                pct = _parse_tqdm_percent(line)
                if pct is not None:
                    self.progress.emit(pct)

            self._process.wait()
            self.finished.emit(self._process.returncode)

        except Exception as exc:
            self.log_line.emit(f"[ERROR] {exc}")
            self.finished.emit(-1)


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
