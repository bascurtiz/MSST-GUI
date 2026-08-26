"""ui/widgets/common.py — kept minimal for new dark-theme design."""
import os, time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QFileDialog, QSizePolicy, QLabel, QFrame, QTextEdit,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor
from ui.theme import theme_manager

class FilePicker(QWidget):
    path_changed = Signal(str)
    def __init__(self, mode="file", filter="All (*.*)", placeholder="", drag_drop=False, parent=None):
        super().__init__(parent)
        self._mode = mode; self._filter = filter
        hl = QHBoxLayout(self); hl.setContentsMargins(0,0,0,0); hl.setSpacing(0)
        self.line = QLineEdit(); self.line.setPlaceholderText(placeholder or f"Select {mode}…")
        self.line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line.textChanged.connect(self.path_changed); hl.addWidget(self.line, 1)
        self._btn = QPushButton("..."); self._btn.setFixedSize(44, 38)
        self._apply_style()
        self._btn.clicked.connect(self._browse); hl.addWidget(self._btn)
    def _apply_style(self):
        self._btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text};border:none;}}"
            f"QPushButton:hover{{background:{theme_manager.accent};color:{theme_manager._accent_text};}}"
        )
    def reapply_theme(self):
        self._apply_style()
    def value(self): return self.line.text().strip()
    def set_value(self, v): self.line.setText(v)
    def _browse(self):
        if self._mode == "folder": path = QFileDialog.getExistingDirectory(self, "Select folder")
        else: path, _ = QFileDialog.getOpenFileName(self, "Select file", filter=self._filter)
        if path: self.line.setText(path)

class PageHeader(QWidget):
    """mvsep-style page header: left accent bar, big uppercase title, and
    a subtitle with an accent-highlighted phrase.
    Optional back button above the title; extra widgets dock on the right."""

    def __init__(self, title, subtitle="", highlight="", back=False, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Vertical accent bar running down the left edge
        bar = QFrame()
        bar.setFixedWidth(4)
        bar.setStyleSheet(
            f"background:{theme_manager.accent};border:none;border-radius:2px;"
        )
        root.addWidget(bar)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        self.back_btn = None
        if back:
            self.back_btn = QPushButton("\u2190 Back")
            self.back_btn.setMinimumHeight(32)
            self.back_btn.setCursor(Qt.PointingHandCursor)
            self.back_btn.setStyleSheet(self._back_ss())
            col.addWidget(self.back_btn)
            col.addSpacing(14)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:32px;font-weight:900;"
            f"color:{theme_manager.theme.text};background:transparent;border:none;"
            "letter-spacing:-0.5px;"
        )
        col.addWidget(self.title_lbl)

        self.sub_lbl = None
        if subtitle:
            if highlight and highlight in subtitle:
                subtitle = subtitle.replace(
                    highlight,
                    '<span style="color:%s;font-weight:900;">%s</span>'
                    % (theme_manager.accent, highlight),
                )
            self.sub_lbl = QLabel(subtitle)
            self.sub_lbl.setTextFormat(Qt.RichText)
            self.sub_lbl.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;"
                f"color:{theme_manager.theme.text_muted};background:transparent;"
                "border:none;letter-spacing:1px;"
            )
            col.addWidget(self.sub_lbl)

        root.addLayout(col, 1)

    def _back_ss(self):
        t = theme_manager.theme
        return (
            f"QPushButton{{background:transparent;color:{t.text_muted};border:1px solid {t.border_visible};"
            "font-family:'Montserrat';font-size:12px;border-radius:6px;padding:0 16px;}"
            f"QPushButton:hover{{background:{t.border};color:{t.text};border-color:{theme_manager.accent};}}"
        )

    def add_extra(self, widget):
        """Dock a widget (badge, buttons, …) at the right end of the header."""
        self.layout().addWidget(widget)
        return widget

    def set_title(self, text):
        self.title_lbl.setText(text.upper())

    def set_subtitle(self, text, highlight=""):
        if self.sub_lbl is None:
            return
        if highlight and highlight in text:
            text = text.replace(
                highlight,
                '<span style="color:%s;font-weight:900;">%s</span>'
                % (theme_manager.accent, highlight),
            )
        self.sub_lbl.setText(text)


class SectionHeader(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        hl = QHBoxLayout(self); hl.setContentsMargins(0,8,0,4)
        self._lbl = QLabel(title.upper())
        self._apply_style()
        hl.addWidget(self._lbl); hl.addStretch()
    def _apply_style(self):
        self._lbl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:12px;font-weight:900;"
            f"color:{theme_manager.theme.text};background:transparent;padding-left:10px;"
            f"border-left:4px solid {theme_manager.accent};letter-spacing:1px;"
        )
    def reapply_theme(self):
        self._apply_style()

class ConsoleLog(QWidget):
    _GREEN_TOKENS = (">","[INFO]","[PROCESS]","[PROGRESS]","[GPU]","[STATUS]","[WARN]","[ERROR]")
    def __init__(self, parent=None):
        super().__init__(parent)
        vl = QVBoxLayout(self); vl.setContentsMargins(0,0,0,0)
        self._edit = QTextEdit(); self._edit.setReadOnly(True)
        self._edit.setObjectName("consoleLog"); self._edit.setLineWrapMode(QTextEdit.NoWrap)
        self._apply_style()
        vl.addWidget(self._edit); self.setStyleSheet(f"background:{theme_manager.theme.console_bg};border:none;")
        for line in ("> MSS TOOL v1.0.0", "> Ready.", "[INFO] Waiting for input…"):
            self._insert(line)
    def _apply_style(self):
        self._edit.setStyleSheet(
            f"QTextEdit#consoleLog{{background:{theme_manager.theme.console_bg};color:{theme_manager.theme.text};"
            f"font-family:'Courier New','Consolas',monospace;font-size:11px;"
            f"border:none;padding:10px 12px;}}"
        )
    def _colorize(self, text):
        import html
        for t in self._GREEN_TOKENS:
            if text.strip().startswith(t):
                rest = text[text.index(t)+len(t):]
                return (f'<span style="color:{theme_manager.accent};font-weight:bold;">{html.escape(t)}</span>'
                        f'<span style="color:{theme_manager.theme.text};">{html.escape(rest)}</span>')
        import html as _h; return f'<span style="color:{theme_manager.theme.text};">{_h.escape(text)}</span>'
    def _insert(self, text):
        c = self._edit.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        self._edit.setTextCursor(c)
        self._edit.insertHtml(self._colorize(text)+"<br>")
        self._edit.verticalScrollBar().setValue(self._edit.verticalScrollBar().maximum())
    def append_line(self, text): self._insert(text)
    def clear_log(self): self._edit.clear()
    def reapply_theme(self):
        self._apply_style()
        self.setStyleSheet(f"background:{theme_manager.theme.console_bg};border:none;")

class SpectrogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(80)
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")
    def set_audio(self, path): pass
    def clear_audio(self): pass
    def set_active(self, v): pass
    def reapply_theme(self):
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

class WaveformPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(60)
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")
    def set_audio(self, path): pass
    def clear_audio(self): pass
    def set_active(self, v): pass
    def reapply_theme(self):
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")

class ProcessingStatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")
    def start_timer(self): pass
    def stop_timer(self): pass
    def update_stats(self, progress=""): pass
    def reapply_theme(self):
        self.setStyleSheet(f"background:{theme_manager.theme.bg_deep};")
