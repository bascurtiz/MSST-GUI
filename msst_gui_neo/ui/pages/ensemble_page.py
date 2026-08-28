"""
ui/pages/ensemble_page.py
Ensemble landing page — category selection between Auto Ensemble and Manual Ensemble.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath

from ui.theme import theme_manager
from ui.widgets.common import PageHeader


def _acc_rgba(alpha: float) -> str:
    c = QColor(theme_manager.accent)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


class _IconPaint(QLabel):
    def __init__(self, icon_type, parent=None):
        super().__init__(parent)
        self._type = icon_type
        self.setFixedSize(28, 28)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        pen = QPen(QColor(theme_manager.accent), 2.0)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if self._type == "auto":
            self._draw_lightning(painter)
        elif self._type == "iterative":
            self._draw_iterative(painter)
        else:
            self._draw_wrench(painter)
        painter.end()

    def _draw_lightning(self, painter):
        path = QPainterPath()
        path.moveTo(16, 2)
        path.lineTo(8, 14)
        path.lineTo(13, 14)
        path.lineTo(10, 26)
        path.lineTo(20, 12)
        path.lineTo(15, 12)
        path.closeSubpath()
        painter.drawPath(path)

    def _draw_wrench(self, painter):
        cx, cy = 14, 14
        r = 7
        angle = -30

        import math
        rad = math.radians(angle)
        ox = cx + r * math.cos(rad)
        oy = cy + r * math.sin(rad)

        path = QPainterPath()
        path.addEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        handle_w = 4
        handle_len = 12
        hx = ox
        hy = oy

        angle2 = angle + 180
        rad2 = math.radians(angle2)
        ex = hx + handle_len * math.cos(rad2)
        ey = hy + handle_len * math.sin(rad2)

        perp_rad = rad + math.pi / 2
        dx = handle_w / 2 * math.cos(perp_rad)
        dy = handle_w / 2 * math.sin(perp_rad)

        path.moveTo(hx + dx, hy + dy)
        path.lineTo(ex + dx, ey + dy)
        path.lineTo(ex - dx, ey - dy)
        path.lineTo(hx - dx, hy - dy)
        path.closeSubpath()

        painter.drawPath(path)

    def _draw_iterative(self, painter):
        cx, cy = 14, 14
        num_circles = 3
        for i in range(num_circles):
            r = 4 + i * 3
            path = QPainterPath()
            path.addEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
            painter.drawPath(path)


class _IconBox(QFrame):
    def __init__(self, icon_type, parent=None):
        super().__init__(parent)
        self.setFixedSize(56, 56)
        self.setStyleSheet(
            f"QFrame{{background:transparent;border:1px solid {theme_manager.theme.border};border-radius:10px;}}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setAlignment(Qt.AlignCenter)

        icon = _IconPaint(icon_type)
        root.addWidget(icon)


class _EnsembleCard(QFrame):
    clicked = Signal()

    def __init__(self, icon_type, title, description, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.card};border:1px solid {theme_manager.theme.border};border-radius:20px;}}"
        )
        self.setMinimumHeight(260)
        self.setCursor(Qt.PointingHandCursor)
        self._pressed = False
        self._hovered = False

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 32)
        root.setSpacing(20)

        icon_box = _IconBox(icon_type)
        root.addWidget(icon_box)

        root.addSpacing(4)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:16px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;border:none;"
        )
        root.addWidget(self._title_label)

        self._desc_label = QLabel(description)
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{theme_manager.theme.text_dim};"
            f"background:transparent;border:none;line-height:1.6;"
        )
        self._desc_label.setMaximumWidth(340)
        root.addWidget(self._desc_label)

        root.addStretch()

        self._select_btn = QPushButton("Select")
        self._select_btn.setFixedHeight(48)
        self._select_btn.setCursor(Qt.PointingHandCursor)
        self._select_btn.setStyleSheet(self._btn_ss())
        self._select_btn.clicked.connect(self.clicked.emit)
        root.addWidget(self._select_btn)

    def reapply_theme(self):
        self.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.card};border:1px solid {theme_manager.theme.border};border-radius:20px;}}"
        )
        self._title_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:16px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;border:none;"
        )
        self._desc_label.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{theme_manager.theme.text_dim};"
            f"background:transparent;border:none;line-height:1.6;"
        )
        self._select_btn.setStyleSheet(self._btn_ss())

    def _btn_ss(self):
        if self._hovered:
            return (
                f"QPushButton{{background:{theme_manager.accent};border:none;"
                f"color:{theme_manager._accent_text};font-family:'Montserrat',sans-serif;font-weight:600;"
                f"font-size:12px;border-radius:8px;}}"
            )
        return (
            f"QPushButton{{background:transparent;border:1px solid {theme_manager.accent};"
            f"color:{theme_manager.accent};font-family:'Montserrat',sans-serif;font-weight:600;"
            f"font-size:12px;border-radius:8px;}}"
            f"QPushButton:hover{{background:{_acc_rgba(0.08)};}}"
            f"QPushButton:pressed{{background:{_acc_rgba(0.15)};}}"
        )

    def _set_card_ss(self):
        t = theme_manager.theme
        if self._pressed:
            bg, bd = t.surface_alt, theme_manager.accent
        elif self._hovered:
            bg, bd = theme_manager._accent_soft, theme_manager.accent
        else:
            bg, bd = t.card, t.border
        self.setStyleSheet(
            f"QFrame{{background:{bg};border:1px solid {bd};border-radius:20px;}}"
        )

    def enterEvent(self, event):
        self._hovered = True
        self._set_card_ss()
        self._select_btn.setStyleSheet(self._btn_ss())
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._pressed = False
        self._set_card_ss()
        self._select_btn.setStyleSheet(self._btn_ss())
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pressed = True
            self._set_card_ss()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        was_pressed = self._pressed
        self._pressed = False
        self._set_card_ss()
        if event.button() == Qt.LeftButton and was_pressed:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class EnsembleLandingPage(QWidget):
    auto_selected = Signal()
    manual_selected = Signal()
    iterative_selected = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ensembleLandingPage")
        # Object-name scoped so the background doesn't cascade into child
        # dialogs (QMessageBox etc.) and overwrite their button styles.
        self.setStyleSheet(f"#ensembleLandingPage{{background:{theme_manager.theme.bg};}}")
        self._build_ui()

    def reapply_theme(self):
        self.setStyleSheet(f"#ensembleLandingPage{{background:{theme_manager.theme.bg};}}")
        for card in self._cards:
            card.reapply_theme()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 40)
        root.setSpacing(0)

        hdr = PageHeader(
            "ENSEMBLE",
            "COMBINE MULTIPLE MODELS FOR BETTER RESULTS",
            highlight="MULTIPLE MODELS",
        )
        root.addWidget(hdr)

        root.addSpacing(40)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(20)
        cards_row.setContentsMargins(0, 0, 0, 0)

        auto_card = _EnsembleCard(
            "auto",
            "AUTO ENSEMBLE",
            "Automatically combine compatible models\nand generate optimized stem outputs.",
        )
        auto_card.clicked.connect(self.auto_selected.emit)
        cards_row.addWidget(auto_card, 1)

        manual_card = _EnsembleCard(
            "manual",
            "MANUAL ENSEMBLE",
            "Advanced manual ensemble workflow for\ncombining custom model outputs.",
        )
        manual_card.clicked.connect(self.manual_selected.emit)
        cards_row.addWidget(manual_card, 1)

        iterative_card = _EnsembleCard(
            "iterative",
            "ITERATIVE ENSEMBLE",
            "Multi-pass iterative separation with\nmax spec ensemble and vocal attenuation.",
        )
        iterative_card.clicked.connect(self.iterative_selected.emit)
        cards_row.addWidget(iterative_card, 1)

        self._cards = [auto_card, manual_card, iterative_card]

        root.addLayout(cards_row)
        root.addStretch()
