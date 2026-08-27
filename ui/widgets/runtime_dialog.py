"""
ui/widgets/runtime_dialog.py
----------------------------
First-run GPU runtime setup dialog for the frozen app.

`ensure_runtime(parent)` is the single gate that job-start paths call:
returns True immediately when the runtime is usable (dev checkouts always
are), otherwise shows a modal dialog that installs the GPU-appropriate
PyTorch build into the bundled runtime.
"""
import os
import sys

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QProgressBar,
)

from backend import runtime_setup
from ui.theme import theme_manager


def runtime_usable() -> bool:
    """Quick check whether separation jobs can run right now."""
    if not getattr(sys, "frozen", False):
        return True  # dev checkout: the running venv is the runtime
    return runtime_setup.runtime_ready()


class _SetupThread(QThread):
    log_line = Signal(str)
    progress = Signal(float)
    finished_ok = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        def log(msg):
            self.log_line.emit(str(msg))

        def progress(f):
            self.progress.emit(float(f))

        def cancelled():
            return self._cancelled

        try:
            ok, msg = runtime_setup.install_runtime(log, progress, cancelled)
        except Exception as e:  # noqa: BLE001 — surface anything to the user
            ok, msg = False, str(e)
        self.finished_ok.emit(ok, msg)


class RuntimeSetupDialog(QDialog):
    """Modal first-run installer for the GPU runtime."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Runtime Setup")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._thread = None
        self._result = (False, "")

        t = theme_manager.theme
        self.setStyleSheet(
            f"QDialog{{background:{t.bg};}}"
            f"QLabel{{color:{t.text};background:transparent;}}"
            f"QPlainTextEdit{{background:{t.surface};color:{t.console_text};"
            f"border:1px solid {t.border};border-radius:6px;}}"
            f"QProgressBar{{background:{t.surface_alt};border:1px solid {t.border};"
            f"border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{theme_manager.accent};border-radius:3px;}}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)

        gpus = runtime_setup.detect_gpus()
        line = runtime_setup.pick_torch_line([c for _, c in gpus])
        line_txt = {
            "cu128": "CUDA 12.8 build — supports RTX 50-series and newer",
            "cu121": "CUDA 12.1 build — supports GTX 10-series through RTX 40-series",
            "cpu": "CPU build — no NVIDIA GPU detected",
        }[line]
        gpu_txt = ", ".join(n for n, _ in gpus) or "no NVIDIA GPU detected"

        title = QLabel("First-time setup — GPU runtime")
        title.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:14px;font-weight:bold;")
        root.addWidget(title)

        info = QLabel(
            f"Detected GPU: <b style='color:{theme_manager.accent};'>{gpu_txt}</b><br>"
            f"PyTorch to install: <b>{line_txt}</b><br><br>"
            f"Installation will take 5-10 minutes, depending on your hardware and "
            f"internet connection (2.5-3 GB).")
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        root.addWidget(info)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        root.addWidget(self._bar)

        # Progress reports arrive as coarse milestones (per pip package), so
        # the bar eases toward each target instead of snapping between them.
        self._shown = 0.0
        self._target = 0.0
        self._easer = QTimer(self)
        self._easer.setInterval(60)
        self._easer.timeout.connect(self._ease_progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(180)
        self._log.setPlaceholderText("Setup log…")
        root.addWidget(self._log)

        btns = QHBoxLayout()
        btns.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet("QPushButton{font-weight:600;}")
        self._cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self._cancel_btn)
        self._install_btn = QPushButton("Install")
        self._install_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.accent};color:#FFFFFF;"
            f"border:none;border-radius:6px;padding:0 18px;min-height:32px;"
            f"font-weight:600;}}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}")
        self._install_btn.clicked.connect(self._start_install)
        btns.addWidget(self._install_btn)
        self._close_btn = QPushButton("Close")
        self._close_btn.setStyleSheet("QPushButton{font-weight:600;}")
        self._close_btn.clicked.connect(self.reject)
        self._close_btn.setVisible(False)
        btns.addWidget(self._close_btn)
        root.addLayout(btns)

    def set_progress(self, fraction):
        """Set a progress target (0..1). The bar eases toward it so coarse
        milestone updates read as continuous movement."""
        self._target = max(self._target, min(1.0, float(fraction)))
        if not self._easer.isActive():
            self._easer.start()

    def _ease_progress(self):
        gap = self._target - self._shown
        if gap <= 0.001:
            self._shown = self._target
            self._easer.stop()
        else:
            # glide over a full-scale jump in ~4s; small nudges land faster
            self._shown += max(gap * 0.10, 0.002)
        self._bar.setValue(int(round(min(self._shown, 1.0) * 100)))

    def _start_install(self):
        self._install_btn.setEnabled(False)
        self._append_log("— preparing runtime —")
        self._thread = _SetupThread(self)
        self._thread.log_line.connect(self._append_log)
        self._thread.progress.connect(self.set_progress)
        self._thread.finished_ok.connect(self._on_finished)
        self._cancel_btn.setText("Cancel setup")
        try:
            self._cancel_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._cancel_btn.clicked.connect(self._thread.cancel)
        self._thread.start()

    def _on_finished(self, ok, msg):
        self._bar.setValue(100 if ok else self._bar.value())
        if ok:
            self._append_log(
                f'<span style="color:{theme_manager.theme.success};'
                f'font-weight:bold;">Installation complete!</span>', html=True)
        else:
            self._append_log(
                f'<span style="color:{theme_manager.theme.error};'
                f'font-weight:bold;">Installation failed:</span> {msg}', html=True)
        self._result = (ok, msg)
        self._cancel_btn.setVisible(False)
        self._close_btn.setVisible(True)
        self._install_btn.setVisible(not ok)
        self._install_btn.setText("Retry")
        self._install_btn.setEnabled(True)
        try:
            self._install_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._install_btn.clicked.connect(self._start_install)

    def _append_log(self, line, html=False):
        if html:
            self._log.appendHtml(line)
        else:
            self._log.appendPlainText(line)
        self._scroll_log()

    def _scroll_log(self):
        bar = self._log.verticalScrollBar()
        bar.setValue(bar.maximum())

    @property
    def succeeded(self):
        return self._result[0]


def ensure_runtime(parent=None) -> bool:
    """Gate for job-start paths. True when separation jobs can run."""
    if not getattr(sys, "frozen", False):
        return True  # dev checkout: the running venv is the runtime
    if runtime_setup.runtime_ready() and not runtime_setup.runtime_needs_repair():
        return True
    dlg = RuntimeSetupDialog(parent)
    dlg.exec()
    return dlg.succeeded and runtime_setup.runtime_ready()
