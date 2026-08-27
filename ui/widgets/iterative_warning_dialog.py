"""ui/widgets/iterative_warning_dialog.py
First-time warning dialog for Iterative Ensemble.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QFrame,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QGuiApplication

from ui.theme import theme_manager


class IterativeWarningDialog(QDialog):
    continue_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Iterative Ensemble - Important Information")
        self.setModal(True)
        self.setMinimumWidth(520)
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

        self._container = QFrame()
        self._container.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.bg};border:1px solid {theme_manager.theme.border_visible};border-radius:16px;}}"
        )
        cl = QVBoxLayout(self._container)
        cl.setContentsMargins(32, 28, 32, 28)
        cl.setSpacing(20)

        title = QLabel("ITERATIVE ENSEMBLE")
        title.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:20px;font-weight:bold;"
            f"color:{theme_manager.accent};background:transparent;border:none;letter-spacing:1px;"
        )
        cl.addWidget(title)

        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet(f"background:{theme_manager.accent};border:none;border-radius:1px;")
        cl.addWidget(divider)

        warnings = [
            ("Internet Connection Required", "This workflow requires an active internet connection for API-based models."),
            ("MVSep API Key Required", "A valid MVSep API key is needed. Some models are processed exclusively through the MVSep API."),
            ("Long Processing Time", "Iterative ensemble performs multiple separation passes per file. Processing can take a very long time, especially with many files or iterations."),
            ("Multiple Separation Passes", "Each iteration separates the audio with multiple models, ensembles the results, and attenuates vocals before the next pass."),
        ]

        for icon_text, desc in warnings:
            item = QVBoxLayout()
            item.setSpacing(4)

            header = QLabel(f"\u26a0  {icon_text}")
            header.setStyleSheet(
                f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
                f"color:{theme_manager.theme.text};background:transparent;border:none;"
            )
            item.addWidget(header)

            body = QLabel(desc)
            body.setWordWrap(True)
            body.setStyleSheet(
                f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_dim};"
                f"background:transparent;border:none;line-height:1.5;"
            )
            item.addWidget(body)
            cl.addLayout(item)

        self._dont_show = QCheckBox("Don't show this warning again")
        self._dont_show.setStyleSheet(
            f"QCheckBox{{color:{theme_manager.theme.text_dim};font-family:'Montserrat';font-size:11px;}}"
            f"QCheckBox::indicator{{width:16px;height:16px;border-radius:4px;"
            f"border:2px solid {theme_manager.theme.border_visible};background:{theme_manager.theme.surface};}}"
            f"QCheckBox::indicator:checked{{background:{theme_manager.accent};border-color:{theme_manager.accent};}}"
        )
        cl.addWidget(self._dont_show)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(44)
        cancel_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};border:1px solid {theme_manager.theme.border_visible};"
            f"font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;"
            f"border-radius:8px;padding:0 24px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        continue_btn = QPushButton("Continue")
        continue_btn.setFixedHeight(44)
        continue_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
            f"font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;"
            f"border-radius:8px;padding:0 32px;}}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
        )
        continue_btn.clicked.connect(self._on_continue)
        btn_row.addWidget(continue_btn, 1)

        cl.addLayout(btn_row)
        root.addWidget(self._container, 1)

    def reapply_theme(self):
        self._container.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.bg};border:1px solid {theme_manager.theme.border_visible};border-radius:16px;}}"
        )
        self._dont_show.setStyleSheet(
            f"QCheckBox{{color:{theme_manager.theme.text_dim};font-family:'Montserrat';font-size:11px;}}"
            f"QCheckBox::indicator{{width:16px;height:16px;border-radius:4px;"
            f"border:2px solid {theme_manager.theme.border_visible};background:{theme_manager.theme.surface};}}"
            f"QCheckBox::indicator:checked{{background:{theme_manager.accent};border-color:{theme_manager.accent};}}"
        )

    def _on_continue(self):
        self.continue_clicked.emit()
        self.accept()

    @property
    def dont_show_again(self):
        return self._dont_show.isChecked()
