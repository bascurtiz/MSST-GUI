"""
ui/widgets/drop_zone.py
------------------------
Cinematic drag-and-drop input zone for audio files and folders.
Visual styling comes entirely from theme QSS (objectName="dropZone").
"""
import os
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".opus"}


class DropZone(QWidget):
    """
    Accepts audio files or folders via drag-and-drop or click-to-browse.
    Emits `path_dropped(str)` — always a folder path.
    """
    path_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(116)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(6)

        self._icon = QLabel("🎵")
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setObjectName("dropZoneIcon")

        self._label = QLabel("Drop audio file or folder here\nor click to browse")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setObjectName("dropZoneLabel")
        self._label.setWordWrap(True)

        layout.addWidget(self._icon)
        layout.addWidget(self._label)

    # ── drag events ──────────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragOver", True)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("dragOver", False)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dragOver", False)
        self.style().polish(self)
        urls = event.mimeData().urls()
        if not urls:
            return
        self._emit(urls[0].toLocalFile())

    # ── click to browse ───────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Select input folder")
        if path:
            self._emit(path)

    # ── internal ──────────────────────────────────────────────────────────────
    def _emit(self, path: str):
        if os.path.isdir(path):
            self._label.setText(f"📁  {os.path.basename(path)}")
            self.path_dropped.emit(path)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                folder = os.path.dirname(path)
                self._label.setText(f"🎵  {os.path.basename(path)}")
                self.path_dropped.emit(folder)
            else:
                self._label.setText("⚠  Not a supported audio file")
        else:
            self._label.setText("⚠  Path not found")

    def set_path_text(self, text: str):
        self._label.setText(text)
