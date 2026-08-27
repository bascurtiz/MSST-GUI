"""
ui/widgets/progress_panel.py
Cinematic progress display for Auto Ensemble pipeline.
Atmospheric surfaces, layered lighting, editorial typography.
"""
import time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QFrame, QScrollArea, QPushButton,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import theme_manager


class _ModelProgressCard(QFrame):
    def __init__(self, model_name, parent=None):
        super().__init__(parent)
        self._name = model_name
        self._state = "normal"
        self.setFixedHeight(64)
        self.setObjectName("progressCard")
        self._apply_card_style()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)

        self._name_label = QLabel(model_name)
        self._name_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;"
        )
        top.addWidget(self._name_label, 1)

        self._status_label = QLabel("Waiting...")
        self._status_label.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;"
        )
        top.addWidget(self._status_label)

        root.addLayout(top)

        self._bar = QProgressBar()
        self._bar.setFixedHeight(4)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            f"QProgressBar{{background:{theme_manager.theme.border};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{theme_manager.accent};border-radius:2px;}}"
        )
        self._bar.setValue(0)
        root.addWidget(self._bar)

    def _apply_card_style(self):
        self.setStyleSheet(
            f"QFrame#progressCard{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {theme_manager.theme.surface_alt},stop:1 {theme_manager.theme.surface});"
            f"border:1px solid {theme_manager.theme.border};border-radius:10px;}}"
        )

    def reapply_theme(self):
        self._apply_card_style()
        self._status_label.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;"
        )
        self._bar.setStyleSheet(
            f"QProgressBar{{background:{theme_manager.theme.border};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{theme_manager.accent};border-radius:2px;}}"
        )
        if self._state == "complete":
            self._name_label.setStyleSheet(
                f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
                f"color:{theme_manager.accent};background:transparent;"
            )
        elif self._state == "skipped":
            self._name_label.setStyleSheet(
                f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
                f"color:{theme_manager.theme.text_dim};background:transparent;"
            )
        else:
            self._name_label.setStyleSheet(
                f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
                f"color:{theme_manager.theme.text};background:transparent;"
            )

    def update_progress(self, pct):
        self._bar.setValue(pct)
        self._status_label.setText(f"{pct}%")

    def set_status(self, status):
        self._status_label.setText(status)

    def set_complete(self):
        self._bar.setValue(100)
        self._status_label.setText("Complete")
        self._state = "complete"
        self._name_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
            f"color:{theme_manager.accent};background:transparent;"
        )

    def set_skipped(self):
        self._bar.setValue(0)
        self._status_label.setText("Skipped")
        self._state = "skipped"
        self._name_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:bold;"
            f"color:{theme_manager.theme.text_dim};background:transparent;"
        )


class ProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_time = None
        self._model_cards = {}

        self.setStyleSheet("background:transparent;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(20)

        overall = QFrame()
        overall.setObjectName("overallPanel")
        overall.setStyleSheet(
            f"QFrame#overallPanel{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {theme_manager.theme.surface_alt},stop:1 {theme_manager.theme.surface});"
            f"border:1px solid {theme_manager.theme.border};border-radius:12px;}}"
        )
        ol = QVBoxLayout(overall)
        ol.setContentsMargins(20, 16, 20, 16)
        ol.setSpacing(12)

        oh = QHBoxLayout()
        oh.setContentsMargins(0, 0, 0, 0)

        self._overall_label = QLabel("Overall Progress")
        self._overall_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;letter-spacing:1px;"
        )
        oh.addWidget(self._overall_label, 1)

        self._time_label = QLabel("00:00")
        self._time_label.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;"
        )
        oh.addWidget(self._time_label)

        ol.addLayout(oh)

        self._overall_bar = QProgressBar()
        self._overall_bar.setFixedHeight(6)
        self._overall_bar.setTextVisible(False)
        self._overall_bar.setStyleSheet(
            f"QProgressBar{{background:{theme_manager.theme.border};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{theme_manager.accent};border-radius:3px;}}"
        )
        self._overall_bar.setValue(0)
        ol.addWidget(self._overall_bar)

        root.addWidget(overall)

        self._models_label = QLabel("Model Progress")
        self._models_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:bold;"
            f"color:{theme_manager.theme.text_dim};background:transparent;letter-spacing:1px;"
        )
        root.addWidget(self._models_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet("background:transparent;border:none;")
        self._scroll.setFixedHeight(200)

        self._models_container = QWidget()
        self._models_container.setStyleSheet("background:transparent;")
        self._models_layout = QVBoxLayout(self._models_container)
        self._models_layout.setContentsMargins(0, 0, 0, 0)
        self._models_layout.setSpacing(10)
        self._models_layout.addStretch()

        self._scroll.setWidget(self._models_container)
        root.addWidget(self._scroll)

        log_header = QHBoxLayout()
        log_header.setContentsMargins(0, 0, 0, 0)

        self._log_label = QLabel("Log Output")
        self._log_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:bold;"
            f"color:{theme_manager.theme.text_dim};background:transparent;letter-spacing:1px;"
        )
        log_header.addWidget(self._log_label, 1)

        self._log_toggle = QPushButton("▼")
        self._log_toggle.setFixedSize(24, 24)
        self._log_toggle.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};border:1px solid {theme_manager.theme.border};"
            f"border-radius:4px;font-size:10px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
        )
        self._log_toggle.clicked.connect(self._toggle_log)
        log_header.addWidget(self._log_toggle)

        root.addLayout(log_header)

        self._log_frame = QFrame()
        self._log_frame.setObjectName("logFrame")
        self._log_frame.setStyleSheet(
            f"QFrame#logFrame{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.border};border-radius:8px;}}"
        )
        self._log_frame.setFixedHeight(120)

        log_scroll = QScrollArea()
        log_scroll.setWidgetResizable(True)
        log_scroll.setFrameShape(QFrame.NoFrame)
        log_scroll.setStyleSheet("background:transparent;border:none;")

        self._log_content = QLabel("")
        self._log_content.setStyleSheet(
            f"font-family:Consolas,Monaco,monospace;font-size:10px;color:{theme_manager.theme.text_dim};"
            f"background:transparent;padding:12px;"
        )
        self._log_content.setWordWrap(True)
        self._log_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        log_inner = QWidget()
        log_inner.setStyleSheet("background:transparent;")
        log_inner_layout = QVBoxLayout(log_inner)
        log_inner_layout.setContentsMargins(0, 0, 0, 0)
        log_inner_layout.addWidget(self._log_content)

        log_scroll.setWidget(log_inner)

        log_layout = QVBoxLayout(self._log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(log_scroll)

        root.addWidget(self._log_frame)

        self._log_visible = True

    def reapply_theme(self):
        self._apply_overall_styles()
        self._models_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:bold;"
            f"color:{theme_manager.theme.text_dim};background:transparent;letter-spacing:1px;"
        )
        self._log_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:bold;"
            f"color:{theme_manager.theme.text_dim};background:transparent;letter-spacing:1px;"
        )
        self._log_toggle.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};border:1px solid {theme_manager.theme.border};"
            f"border-radius:4px;font-size:10px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};}}"
        )
        self._log_frame.setStyleSheet(
            f"QFrame#logFrame{{background:{theme_manager.theme.surface};border:1px solid {theme_manager.theme.border};border-radius:8px;}}"
        )
        self._log_content.setStyleSheet(
            f"font-family:Consolas,Monaco,monospace;font-size:10px;color:{theme_manager.theme.text_dim};"
            f"background:transparent;padding:12px;"
        )
        for card in self._model_cards.values():
            card.reapply_theme()

    def _apply_overall_styles(self):
        overall = self.findChild(QFrame, "overallPanel")
        if overall:
            overall.setStyleSheet(
                f"QFrame#overallPanel{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                f"stop:0 {theme_manager.theme.surface_alt},stop:1 {theme_manager.theme.surface});"
                f"border:1px solid {theme_manager.theme.border};border-radius:12px;}}"
            )
        self._overall_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;letter-spacing:1px;"
        )
        self._time_label.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;"
        )

    def _toggle_log(self):
        self._log_visible = not self._log_visible
        self._log_frame.setVisible(self._log_visible)
        self._log_toggle.setText("▼" if self._log_visible else "▲")

    def start(self, models):
        self._start_time = time.time()
        self._model_cards = {}

        while self._models_layout.count() > 1:
            item = self._models_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for model in models:
            name = model.get("name", "Unknown")
            card = _ModelProgressCard(name)
            self._model_cards[name] = card
            self._models_layout.insertWidget(self._models_layout.count() - 1, card)

        self._overall_bar.setValue(0)
        self._overall_label.setText("Overall Progress")
        self._time_label.setText("00:00")
        self._log_content.setText("")

    def update_stage(self, stage, current, total):
        if stage == "inference":
            pct = int((current / total) * 100) if total > 0 else 0
            self._overall_bar.setValue(pct)
            self._overall_label.setText(f"Inference ({current}/{total})")
        elif stage == "ensemble":
            self._overall_bar.setValue(90)
            self._overall_label.setText("Ensemble")

    def update_model_progress(self, model_name, percentage):
        card = self._model_cards.get(model_name)
        if card:
            card.update_progress(percentage)
            card.set_status(f"{percentage}%")

    def update_ensemble_progress(self, percentage):
        self._overall_bar.setValue(90 + int(percentage * 0.1))

    def model_complete(self, model_name):
        card = self._model_cards.get(model_name)
        if card:
            card.set_complete()

    def model_skipped(self, model_name):
        card = self._model_cards.get(model_name)
        if card:
            card.set_skipped()

    def add_log(self, message):
        current = self._log_content.text()
        if current:
            self._log_content.setText(current + "\n" + message)
        else:
            self._log_content.setText(message)

        sb = self._log_content.parent().parent().verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def update_time(self):
        if self._start_time:
            elapsed = int(time.time() - self._start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self._time_label.setText(f"{mins:02d}:{secs:02d}")

    def finish(self, success):
        if success:
            self._overall_bar.setValue(100)
            self._overall_label.setText("Complete")
        else:
            self._overall_label.setText("Failed")
