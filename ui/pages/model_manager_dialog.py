"""
ui/pages/model_manager_dialog.py
---------------------------------
Verification dialog for installing models from HuggingFace model manager.
Opened from _FolderManagerWidget when user clicks Install on a model folder.
"""

from __future__ import annotations
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QLineEdit, QScrollArea,
    QSizePolicy, QMessageBox, QProgressBar, QDialog,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer

from ui.theme import theme_manager, UIConstants
from backend.model_manager import install_model, ModelInfo


ARCH_TYPES = [
    "MDX Architecture", "Demucs Architecture",
    "BS Roformer Architecture", "Melband Roformer Architecture",
    "SCNet Architecture", "Apollo Architecture", "Bandit Architecture",
]

MODEL_TYPES = [
    "vocals", "instrumental", "dereverb / deecho", "denoise",
    "phantom centre", "karaoke", "dual target (instrumental & vocals)",
    "multi stems", "super resolution", "drums", "bass", "piano",
    "guitar", "wind", "strings", "percussion", "keys",
]


def _input_style():
    return (
        f"background:{theme_manager.theme.input_bg};"
        f"border:1px solid {theme_manager.theme.border_visible};border-radius:6px;"
        f"color:{theme_manager.theme.text};font-family:'Montserrat';font-size:12px;padding:0 14px;"
    )


def _combo_style():
    return (
        f"QComboBox{{background:transparent;border:none;"
        f"color:{theme_manager.theme.text_dim};font-family:'Montserrat';font-size:12px;padding:0 14px;}}"
        f"QComboBox::drop-down{{width:0;border:none;}}"
        f"QComboBox::down-arrow{{width:0;height:0;border:none;}}"
        f"QComboBox QAbstractItemView{{background:{theme_manager.theme.input_bg};"
        f"border:1px solid {theme_manager.theme.border_visible};"
        f"color:{theme_manager.theme.text};"
        f"selection-background-color:{theme_manager.accent};"
        f"selection-color:{theme_manager._accent_text};}}"
    )


class _InstallThread(QThread):
    progress = Signal(str, int, int)
    status = Signal(str)
    speed = Signal(float)   # megabits per second
    finished_signal = Signal(bool, str)
    error = Signal(str)

    def __init__(self, info: ModelInfo, arch: str, stem_type: str, backend_url: str = ""):
        super().__init__()
        self._info = info
        self._arch = arch
        self._stem_type = stem_type
        self._backend_url = backend_url
        self._cancelled = False

    def run(self):
        try:
            override_info = ModelInfo(
                key=self._info.key,
                full_name=self._info.full_name,
                arch=self._arch,
                stem_type=self._stem_type,
                category=self._info.category,
                model_type=self._info.model_type,
                stems=self._info.stems,
                target_instrument=self._info.target_instrument,
                checkpoint_url=self._info.checkpoint_url,
                config_url=self._info.config_url,
                backend_script_url=self._backend_url,
                file_size=self._info.file_size,
            )
            success, msg = install_model(
                override_info,
                progress_callback=lambda n, c, t: self.progress.emit(n, c, t),
                status_callback=lambda s: self.status.emit(s),
                cancel_callback=lambda: self._cancelled,
                speed_callback=self.speed.emit,
            )
            self.finished_signal.emit(success, msg)
        except Exception as e:
            self.error.emit(str(e))


class ModelInstallDialog(QDialog):
    def __init__(self, info: ModelInfo, parent=None):
        super().__init__(parent)
        self._info = info
        self._install_thread: Optional[_InstallThread] = None
        self._completed = False
        self._cancelled = False
        self._drag_pos = None

        has_backend = bool(info.backend_script_url)
        dlg_height = 440 if has_backend else 380

        self.setFixedSize(480, dlg_height)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        # ensure the QDialog{background;...} rule actually paints
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setModal(True)

        self.setStyleSheet(f"""
            QDialog{{
                background:{theme_manager.theme.bg};
                border:1px solid {theme_manager.theme.border_dim};
                border-radius:{UIConstants.CARD_RADIUS_PAINT}px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(0)

        title = QLabel("Install Model")
        title.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:16px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;border:none;"
        )
        root.addWidget(title)
        root.addSpacing(4)

        name_lbl = QLabel(info.full_name)
        name_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:12px;color:{theme_manager.theme.text_dim};background:transparent;border:none;"
        )
        name_lbl.setWordWrap(True)
        root.addWidget(name_lbl)
        root.addSpacing(4)

        sz = info.file_size
        if sz:
            size_text = f"{sz / 1024 / 1024:.1f} MB" if sz >= 1024 * 1024 else f"{sz / 1024:.0f} KB"
            size_lbl = QLabel(size_text)
            size_lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_muted};background:transparent;border:none;"
            )
            root.addWidget(size_lbl)
        root.addSpacing(18)

        # ── Architecture ──
        arch_lbl = QLabel("ARCHITECTURE")
        arch_lbl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;border:none;letter-spacing:1px;"
        )
        root.addWidget(arch_lbl)
        root.addSpacing(6)

        guessed_arch = info.arch
        self._arch_combo = QComboBox()
        self._arch_combo.addItems(ARCH_TYPES)
        idx = self._arch_combo.findText(guessed_arch)
        if idx >= 0:
            self._arch_combo.setCurrentIndex(idx)
        self._arch_combo.setMinimumHeight(40)
        self._arch_combo.setStyleSheet(_combo_style())
        arch_wrap = QWidget()
        arch_wrap.setStyleSheet(
            f"background:{theme_manager.theme.input_bg};"
            f"border:1px solid {theme_manager.theme.border_visible};border-radius:6px;"
        )
        arch_wrap.setMinimumHeight(40)
        awl = QHBoxLayout(arch_wrap)
        awl.setContentsMargins(0, 0, 14, 0)
        awl.addWidget(self._arch_combo, 1)
        root.addWidget(arch_wrap)
        root.addSpacing(12)

        # ── Type ──
        type_lbl = QLabel("TYPE")
        type_lbl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;border:none;letter-spacing:1px;"
        )
        root.addWidget(type_lbl)
        root.addSpacing(6)

        guessed_type = info.stem_type
        self._type_combo = QComboBox()
        self._type_combo.addItems(MODEL_TYPES)
        idx2 = self._type_combo.findText(guessed_type)
        if idx2 >= 0:
            self._type_combo.setCurrentIndex(idx2)
        self._type_combo.setMinimumHeight(40)
        self._type_combo.setStyleSheet(_combo_style())
        type_wrap = QWidget()
        type_wrap.setStyleSheet(
            f"background:{theme_manager.theme.input_bg};"
            f"border:1px solid {theme_manager.theme.border_visible};border-radius:6px;"
        )
        type_wrap.setMinimumHeight(40)
        twl = QHBoxLayout(type_wrap)
        twl.setContentsMargins(0, 0, 14, 0)
        twl.addWidget(self._type_combo, 1)
        root.addWidget(type_wrap)
        root.addSpacing(12)

        # ── Backend Script URL (optional) ──
        self._backend_group = QWidget()
        self._backend_group.setStyleSheet("background:transparent;")
        self._backend_group.setVisible(has_backend)
        bgl = QVBoxLayout(self._backend_group)
        bgl.setContentsMargins(0, 0, 0, 0)
        bgl.setSpacing(6)

        back_lbl = QLabel("BACKEND SCRIPT (.PY) [optional]")
        back_lbl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;border:none;letter-spacing:1px;"
        )
        bgl.addWidget(back_lbl)

        self._backend_input = QLineEdit()
        self._backend_input.setPlaceholderText("Paste backend .py URL (auto-filled if available)")
        self._backend_input.setText(info.backend_script_url)
        self._backend_input.setMinimumHeight(40)
        self._backend_input.setStyleSheet(_input_style())
        bgl.addWidget(self._backend_input)

        root.addWidget(self._backend_group)
        root.addSpacing(16)

        # ── Progress ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar{{background:{theme_manager.theme.input_bg};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{theme_manager.accent};border-radius:3px;}}"
        )
        root.addWidget(self._progress_bar)
        root.addSpacing(4)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(12)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;border:none;"
        )
        status_row.addWidget(self._status_lbl, 1)

        self._speed_lbl = QLabel("")
        self._speed_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;font-weight:700;color:{theme_manager.accent};background:transparent;border:none;"
        )
        status_row.addWidget(self._speed_lbl)
        root.addLayout(status_row)
        root.addStretch()

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(12)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setMinimumHeight(44)
        self._cancel_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text_dim};"
            f"border:1px solid {theme_manager.theme.border_dim};font-family:'Montserrat',sans-serif;"
            f"font-weight:600;font-size:11px;border-radius:8px;padding:0 32px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.border_visible};color:{theme_manager.theme.text};}}"
        )
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._install_btn = QPushButton("Install")
        self._install_btn.setMinimumHeight(44)
        self._install_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
            f"font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;border-radius:8px;padding:0 32px;}}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.disabled_bg};color:{theme_manager.theme.text_muted};}}"
        )
        self._install_btn.clicked.connect(self._start_install)
        btn_row.addWidget(self._install_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

    def _start_install(self):
        self._install_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_lbl.setText("Starting download...")

        arch = self._arch_combo.currentText()
        stem_type = self._type_combo.currentText()
        backend_url = self._backend_input.text().strip() if self._backend_group.isVisible() else ""

        self._install_thread = _InstallThread(self._info, arch, stem_type, backend_url)
        self._install_thread.progress.connect(self._on_progress)
        self._install_thread.status.connect(self._status_lbl.setText)
        self._install_thread.speed.connect(self._on_speed)
        self._install_thread.finished_signal.connect(self._on_install_done)
        self._install_thread.error.connect(self._on_install_error)
        self._install_thread.finished.connect(self._install_thread.deleteLater)
        self._install_thread.start()

    def _on_progress(self, name, cur, total):
        if total:
            self._progress_bar.setValue(int(cur * 100 / total))

    def _on_speed(self, mbps: float):
        mbs = mbps / 8  # megabits -> megabytes
        if mbs < 1:
            self._speed_lbl.setText(f"{mbs * 1024:.0f} KB/s")
        else:
            self._speed_lbl.setText(f"{mbs:.2f} MB/s")

    def _on_install_done(self, success, msg):
        self._completed = True
        self._speed_lbl.setText("")
        if self._cancelled:
            self.reject()
            return
        if success:
            self._status_lbl.setText("Installation complete!")
            self._progress_bar.setValue(100)
            self._cancel_btn.setText("Close")
            self._cancel_btn.setEnabled(True)
            QTimer.singleShot(600, self.accept)
        else:
            self._status_lbl.setText(f"Failed: {msg}")
            self._cancel_btn.setEnabled(True)
            self._install_btn.setEnabled(True)

    def _on_install_error(self, msg):
        self._completed = True
        if self._cancelled:
            self.reject()
            return
        self._status_lbl.setText(f"Error: {msg}")
        self._cancel_btn.setEnabled(True)
        self._install_btn.setEnabled(True)

    def _on_cancel(self):
        if self._install_thread and self._install_thread.isRunning():
            self._cancelled = True
            self._install_thread._cancelled = True
            self._cancel_btn.setEnabled(False)
            self._status_lbl.setText("Cancelling...")
        elif self._completed:
            self.accept()
        else:
            self.reject()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()
