"""ui/widgets/model_installer_dialog.py
Modal dialog for installing missing iterative ensemble models.
Uses plain Python threading.Thread - no QThread/moveToThread.
"""
import threading
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QProgressBar, QScrollArea, QSizePolicy, QWidget,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QGuiApplication

from backend.model_installer import REQUIRED_MODELS, ModelInstaller

from ui.theme import theme_manager


class _ModelInstallRow(QFrame):
    def __init__(self, model, status, path):
        super().__init__()
        self._model = model
        self.setStyleSheet(
            "QFrame{background:" + theme_manager.theme.surface + ";border:1px solid " + theme_manager.theme.border_visible + ";border-radius:10px;}"
        )
        self.setFixedHeight(70)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)

        name = QLabel(model.name)
        name.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
            "color:" + theme_manager.theme.text + ";background:transparent;border:none;"
        )
        top.addWidget(name, 1)

        self._status_label = QLabel(status)
        status_color = theme_manager.accent if "installed" in status.lower() or "already" in status.lower() else theme_manager.theme.warning
        self._status_label.setStyleSheet(
            "font-family:'Montserrat';font-size:10px;color:" + status_color + ";background:transparent;border:none;"
        )
        top.addWidget(self._status_label)
        root.addLayout(top)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            "QProgressBar{background:" + theme_manager.theme.border + ";border:none;border-radius:2px;}"
            "QProgressBar::chunk{background:" + theme_manager.accent + ";border-radius:2px;}"
        )
        self._progress.setValue(0)
        root.addWidget(self._progress)

    def set_progress(self, current, total):
        if total > 0:
            self._progress.setValue(int((current / total) * 100))

    def set_status(self, status, color=None):
        self._status_label.setText(status)
        if color:
            self._status_label.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;color:" + color + ";background:transparent;border:none;"
            )

    def reapply_theme(self):
        self.setStyleSheet(
            "QFrame{background:" + theme_manager.theme.surface + ";border:1px solid " + theme_manager.theme.border_visible + ";border-radius:10px;}"
        )
        status_color = theme_manager.accent if "installed" in self._status_label.text().lower() or "already" in self._status_label.text().lower() else theme_manager.theme.warning
        self._status_label.setStyleSheet(
            "font-family:'Montserrat';font-size:10px;color:" + status_color + ";background:transparent;border:none;"
        )
        self._progress.setStyleSheet(
            "QProgressBar{background:" + theme_manager.theme.border + ";border:none;border-radius:2px;}"
            "QProgressBar::chunk{background:" + theme_manager.accent + ";border-radius:2px;}"
        )

    @property
    def model(self):
        return self._model


class _ModelInstallerDialog(QDialog):
    installation_complete = Signal(list)

    def __init__(self, missing_models, registered_models, parent=None):
        super().__init__(parent)
        self._missing = missing_models
        self._registered = registered_models
        self._installed_models = []
        self._installer = None
        self._done = False
        self._install_thread = None
        self.setWindowTitle("Install Required Models")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(400)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._center_on_parent)

    def _center_on_parent(self):
        p = self.parent()
        if not p:
            return
        pg = p.geometry() if p.isMaximized() or p.isFullScreen() else p.frameGeometry()
        sg = QGuiApplication.primaryScreen().availableGeometry()
        g = self.frameGeometry()
        x = pg.x() + (pg.width() - g.width()) // 2
        y = pg.y() + (pg.height() - g.height()) // 2
        x = max(sg.x(), min(x, sg.x() + sg.width() - g.width()))
        y = max(sg.y(), min(y, sg.y() + sg.height() - g.height()))
        self.move(x, y)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        container = QFrame()
        container.setStyleSheet(
            "QFrame{background:" + theme_manager.theme.bg + ";border:1px solid " + theme_manager.theme.border_visible + ";border-radius:16px;}"
        )
        cl = QVBoxLayout(container)
        cl.setContentsMargins(32, 28, 32, 28)
        cl.setSpacing(20)

        title = QLabel("INSTALL REQUIRED MODELS")
        title.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:20px;font-weight:bold;"
            "color:" + theme_manager.accent + ";background:transparent;border:none;letter-spacing:1px;"
        )
        cl.addWidget(title)

        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet("background:" + theme_manager.accent + ";border:none;border-radius:1px;")
        cl.addWidget(divider)

        count = len(self._missing)
        subtitle = QLabel(str(count) + " model(s) need to be installed before using Iterative Ensemble.")
        subtitle.setStyleSheet(
            "font-family:'Montserrat';font-size:12px;color:" + theme_manager.theme.text_dim + ";background:transparent;border:none;"
        )
        cl.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;border:none;")

        rows_container = QWidget()
        rows_container.setStyleSheet("background:transparent;")
        self._rows_layout = QVBoxLayout(rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(10)

        self._rows = {}
        for model, status, path in self._missing:
            row = _ModelInstallRow(model, status, path)
            self._rows[model.id] = row
            self._rows_layout.addWidget(row)

        self._rows_layout.addStretch()
        scroll.setWidget(rows_container)
        cl.addWidget(scroll, 1)

        self._log_label = QLabel("")
        self._log_label.setStyleSheet(
            "font-family:'Courier New',monospace;font-size:10px;color:" + theme_manager.theme.text_dim + ";"
            "background:transparent;border:none;"
        )
        self._log_label.setVisible(False)
        cl.addWidget(self._log_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(44)
        self._cancel_btn.setStyleSheet(
            "QPushButton{background:" + theme_manager.theme.surface + ";color:" + theme_manager.theme.text_dim + ";border:1px solid " + theme_manager.theme.border_visible + ";"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;"
            "border-radius:8px;padding:0 24px;}"
            "QPushButton:hover{background:" + theme_manager.theme.surface_alt + ";color:" + theme_manager.theme.text + ";}"
        )
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._install_btn = QPushButton("Install All")
        self._install_btn.setFixedHeight(44)
        self._install_btn.setStyleSheet(
            "QPushButton{background:" + theme_manager.accent + ";color:" + theme_manager._accent_text + ";border:none;"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;"
            "border-radius:8px;padding:0 32px;}"
            "QPushButton:hover{background:" + theme_manager._accent_hover + ";}"
            "QPushButton:disabled{background:" + theme_manager.theme.disabled_bg + ";color:" + theme_manager.theme.disabled_text + ";}"
        )
        self._install_btn.clicked.connect(self._on_install)
        btn_row.addWidget(self._install_btn, 1)

        cl.addLayout(btn_row)
        root.addWidget(container, 1)

    def _on_install(self):
        if self._done:
            self.accept()
            return

        self._install_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._log_label.setVisible(True)
        self._log_label.setText("Starting installation...")

        models_to_install = [m for m, s, p in self._missing]

        self._installer = ModelInstaller()
        self._installer.progress.connect(self._on_progress)
        self._installer.status.connect(self._on_status)
        self._installer.model_finished.connect(self._on_model_finished)
        self._installer.model_error.connect(self._on_model_error)
        self._installer.all_finished.connect(self._on_all_finished)

        self._install_thread = threading.Thread(
            target=self._installer.install_all,
            args=(models_to_install,),
            daemon=True,
        )
        self._install_thread.start()

    def _on_progress(self, model_id, current, total):
        if model_id in self._rows:
            self._rows[model_id].set_progress(current, total)

    def _on_status(self, status):
        self._log_label.setText(status)

    def _on_model_finished(self, model_id, model_dict):
        if model_id in self._rows:
            self._rows[model_id].set_status("Installed", theme_manager.accent)
            self._rows[model_id].set_progress(100, 100)
        self._installed_models.append(model_dict)

    def _on_model_error(self, model_id, error):
        if model_id in self._rows:
            self._rows[model_id].set_status("Failed: " + error, theme_manager.theme.error)

    def _on_all_finished(self):
        self._done = True
        self._log_label.setText("Installation complete.")
        self._install_btn.setText("Done")
        self._install_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)

        if self._installed_models:
            self.installation_complete.emit(self._installed_models)

        if self._installer:
            self._installer.deleteLater()
            self._installer = None
        QTimer.singleShot(500, self.accept)

    def _on_cancel(self):
        if self._installer:
            self._installer.cancel()
        self._log_label.setText("Cancelling...")
        self._cancel_btn.setEnabled(False)
        QTimer.singleShot(500, self.reject)

    def closeEvent(self, event):
        if self._installer:
            self._installer.cancel()
        super().closeEvent(event)

    def reapply_theme(self):
        for row in self._rows.values():
            row.reapply_theme()
