"""ui/pages/console_page.py — Modern console with progress cards and log view."""
import os
import re
import time
import html
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSizePolicy, QMessageBox, QApplication, QStackedWidget,
    QScrollArea, QProgressBar, QTextEdit, QStyle, QMenu,
)
from PySide6.QtCore import Qt, QTimer, Property, QUrl, QPropertyAnimation, QEasingCurve, Signal, QByteArray, QRectF, QRect, QPoint
from PySide6.QtGui import QTextCursor, QPainter, QPen, QColor, QPainterPath, QDesktopServices, QFont, QPixmap, QCursor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from ui.theme import theme_manager, FONT_FAMILY, FONT_STACK
from ui.widgets.common import PageHeader
from mutagen import File as _MutagenFile


# ── helpers ──────────────────────────────────────────────────────

def _accent_rgba(alpha):
    c = QColor(theme_manager.accent)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def _error_rgba(alpha):
    c = QColor(theme_manager.theme.error)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


# Stem colors fixed to a palette that fits the GUI.
#
# Known stems map to a set color; any other/unrecognized label falls back to
# the neutral "rest" gray. A matching dark background is derived from the same
# hue so the waveform track keeps its subtle color wash.
_STEM_COLORS = {
    "instrumental": "#60A5FA",
    "vocals": "#A855F7",
    "bass": "#EF4444",
    "drums": "#F59E0B",
    "other": "#10B981",
    "guitar": "#C1090B",
    "piano": "#485FAB",
}
_REST_COLOR = "#9A9FB3"

# Common aliases -> canonical stem key.
_STEM_ALIASES = {
    "no vocals": "instrumental",
    "no_vocals": "instrumental",
    "no-vocals": "instrumental",
    "accompaniment": "instrumental",
}


def _normalize_stem(label):
    key = label.lower().strip().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    return _STEM_ALIASES.get(key, key)


def _stem_color(label):
    return _STEM_COLORS.get(_normalize_stem(label), _REST_COLOR)


def _stem_bg_color(label):
    base = _STEM_COLORS.get(_normalize_stem(label), _REST_COLOR)
    c = QColor(base)
    h, _, _, _ = c.getHsvF()
    bg = QColor()
    if theme_manager.mode == "light":
        # Light tinted wash that sits well on the white card in bright theme.
        bg.setHsvF(h, 0.30, 0.94)
    else:
        bg.setHsvF(h, 0.55, 0.17)
    return bg.name()


# ── constants ─────────────────────────────────────────────────────
_AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

# ── regex ─────────────────────────────────────────────────────────

# Robust: capture whatever follows the first ":" or ">" after "Processing",
# regardless of "[i/n]" bracket style. e.g. "Processing 1/3: song.wav",
# "Processing file [2/5]: song.wav", "Processing: song.wav".
_RE_PROCESS = re.compile(r'Processing\b.*?[:>]\s*(.+)', re.IGNORECASE)
_RE_PROCESS_SIMPLE = re.compile(r'Processing[:\s]\s+(.+)', re.IGNORECASE)
_RE_PROGRESS_VAL = re.compile(r'Processing:\s*(\d+)%')
_RE_TQDM_PCT = re.compile(r'\b(\d+)%\s*\|')
_RE_TQDM_ETA = re.compile(r'<(\d+:\d+)')
_RE_EXPORT = re.compile(r'(?:Wrote\s+file|Ensemble\s+output(?:\s+saved\s+to)?)[:\s]\s*(.+)', re.IGNORECASE)
_RE_COMPLETED = re.compile(r'(?:Completed|Inference completed)[:\s]', re.IGNORECASE)
_RE_COMPLETED_NAME = re.compile(r'(?:Completed|Inference completed)[:\s]\s*(.+)', re.IGNORECASE)
_RE_DONE = re.compile(r'^Done\b', re.IGNORECASE)
_RE_ERROR = re.compile(r'(?:^ERROR|FATAL|Traceback|Error\s+message)', re.IGNORECASE)
_RE_OUTPUT_DIR = re.compile(r'Output\s+directory[:\s]\s*(.+)', re.IGNORECASE)
_RE_QUEUED = re.compile(r'Queued[:\s]\s*(.+)', re.IGNORECASE)


def _norm(s):
    """Normalize a song name for fuzzy card matching (lowercase, strip to
    alpha-numeric runs separated by single space). This catches every
    common separator style — hyphens, underscores, dots — so that
    "bts_-_come_over" matches "bts - come over" matches "bts.come.over"."""
    s = (s or "").lower().strip()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return s.strip()


def _extract_name(text):
    m = _RE_PROCESS.search(text)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        if '%' in name or not name:
            return None
        return os.path.splitext(os.path.basename(name))[0]
    m = _RE_PROCESS_SIMPLE.search(text)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        if '%' in name or not name:
            return None
        return os.path.splitext(os.path.basename(name))[0]
    return None


def _extract_raw_input_path(text):
    m = (_RE_PROCESS.search(text) or _RE_PROCESS_SIMPLE.search(text)
         or _RE_QUEUED.search(text))
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return None


def _extract_completed_name(text):
    m = _RE_COMPLETED_NAME.search(text)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        return os.path.splitext(os.path.basename(name))[0]
    return None


def _load_cover_pixmap(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        audio = _MutagenFile(path)
        if audio is None:
            return None
        data = None
        pics = getattr(audio, "pictures", None)
        if pics:
            data = pics[0].data
        elif "covr" in audio:
            data = bytes(audio["covr"][0])
        else:
            for k in list(audio.keys()):
                if k.upper().startswith("APIC"):
                    v = audio[k]
                    v = v[0] if isinstance(v, list) else v
                    data = getattr(v, "data", None)
                    if data:
                        break
        if not data:
            return None
        pm = QPixmap()
        pm.loadFromData(QByteArray(data))
        return pm if not pm.isNull() else None
    except Exception:
        return None


def _extract_progress(text):
    m = _RE_PROGRESS_VAL.search(text)
    if m:
        return int(m.group(1))
    m = _RE_TQDM_PCT.search(text)
    if m:
        return int(m.group(1))
    return None


def _extract_eta(text):
    m = _RE_TQDM_ETA.search(text)
    if m:
        return m.group(1)
    return None


def _extract_export(text):
    m = _RE_EXPORT.search(text)
    if m:
        return m.group(1).strip()
    return None


def _is_done(text):
    return bool(_RE_DONE.search(text) or _RE_COMPLETED.search(text))


def _is_progress_line(text):
    return bool(_RE_TQDM_PCT.search(text) or _RE_PROGRESS_VAL.search(text))


def _has_error(text):
    return bool(_RE_ERROR.search(text))


# ── smooth animated progress bar ─────────────────────────────────

class _SmoothBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(5)
        self._current = 0.0
        self._target = 0.0
        self._anim = QPropertyAnimation(self, b"animated_value", self)
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _get_animated(self):
        return self._current

    def _set_animated(self, v):
        self._current = v
        self.update()

    animated_value = Property(float, _get_animated, _set_animated)

    def setValue(self, pct):
        v = max(0.0, min(100.0, float(pct)))
        self._target = v
        self._anim.stop()
        self._anim.setStartValue(self._current)
        self._anim.setEndValue(v)
        self._anim.start()

    def setValueImmediate(self, pct):
        v = max(0.0, min(100.0, float(pct)))
        self._anim.stop()
        self._current = v
        self._target = v
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        bg = QColor(theme_manager.theme.border)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(r, 3, 3)
        if self._current > 0:
            fw = int(r.width() * self._current / 100.0)
            if fw > 0:
                p.setBrush(QColor(theme_manager.accent))
                p.drawRoundedRect(0, 0, fw, r.height(), 3, 3)
        p.end()


# ── music note icon ───────────────────────────────────────────────

class _MusicNoteIcon(QWidget):
    def __init__(self, parent=None, completed=False, size=42):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._completed = completed
        self._sz = size

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self._sz / 42.0

        c = theme_manager.accent if self._completed else theme_manager.theme.text
        bg = QColor(c)
        bg.setAlpha(45 if self._completed else 20)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(3 * s, 3 * s, 36 * s, 36 * s, 8 * s, 8 * s)

        pen = QPen(QColor(c), 2.2 * s)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QColor(c))

        p.drawEllipse(12 * s, 24 * s, 9 * s, 8 * s)
        p.drawLine(19 * s, 24 * s, 19 * s, 10 * s)
        flag = QPainterPath()
        flag.moveTo(19 * s, 10 * s)
        flag.cubicTo(27 * s, 10 * s, 27 * s, 18 * s, 22 * s, 21 * s)
        p.drawPath(flag)
        p.end()


class _SongCoverIcon(QWidget):
    def __init__(self, parent=None, size=44):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._sz = size
        self._cover = None
        self.__completed = False
        self._note = _MusicNoteIcon(completed=False, size=size)
        self._note.setParent(self)
        self._note.setGeometry(0, 0, size, size)

    @property
    def _completed(self):
        return self.__completed

    @_completed.setter
    def _completed(self, v):
        self.__completed = bool(v)
        self._note._completed = self.__completed
        self._note.update()
        self.update()

    def set_cover(self, pixmap):
        if pixmap is not None and not pixmap.isNull():
            self._cover = pixmap
            self._note.hide()
        else:
            self._cover = None
            self._note.show()
        self.update()

    def paintEvent(self, event):
        if self._cover is not None and not self._cover.isNull():
            p = QPainter(self)
            try:
                p.setRenderHint(QPainter.Antialiasing)
                path = QPainterPath()
                path.addRoundedRect(0, 0, self._sz, self._sz, 8, 8)
                p.setClipPath(path)
                scaled = self._cover.scaled(
                    self._sz, self._sz,
                    Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                p.drawPixmap(0, 0, scaled)
            finally:
                p.end()
        else:
            super().paintEvent(event)


# ── folder icon ────────────────────────────────────────────────────

class _FolderIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 22)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(theme_manager.theme.text)
        c.setAlpha(90)
        pen = QPen(c, 1.5)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        path = QPainterPath()
        path.moveTo(2, 7)
        path.lineTo(2, 19)
        path.lineTo(20, 19)
        path.lineTo(20, 7)
        path.lineTo(12, 7)
        path.lineTo(10, 4)
        path.lineTo(2, 4)
        path.closeSubpath()
        p.drawPath(path)

        p.end()


# ── trash icon ─────────────────────────────────────────────────────

class _TrashIcon(QWidget):
    def __init__(self, parent=None, size=18):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._sz = size

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self._sz / 18.0
        red = QColor(220, 40, 30)
        dark_red = QColor(150, 24, 18)

        # Filled, bold trash-can silhouette (reads clearly at small sizes).
        p.setPen(Qt.NoPen)
        p.setBrush(red)

        # Handle knob on top
        p.drawRoundedRect(7.2 * s, 1.0 * s, 3.6 * s, 1.9 * s, 0.9 * s, 0.9 * s)
        # Lid bar
        p.drawRoundedRect(2.0 * s, 2.7 * s, 14.0 * s, 2.2 * s, 1.0 * s, 1.0 * s)

        # Tapered can body with rounded bottom corners
        body = QPainterPath()
        body.moveTo(4.2 * s, 4.9 * s)
        body.lineTo(13.8 * s, 4.9 * s)
        body.lineTo(12.8 * s, 16.0 * s)
        body.quadTo(12.6 * s, 17.0 * s, 11.8 * s, 17.0 * s)
        body.lineTo(6.2 * s, 17.0 * s)
        body.quadTo(5.4 * s, 17.0 * s, 5.2 * s, 16.0 * s)
        body.closeSubpath()
        p.drawPath(body)

        # Vertical ridges (darker red) for the classic trash detail
        pen = QPen(dark_red, 1.5 * s)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(7.4 * s, 6.4 * s, 6.9 * s, 15.2 * s)
        p.drawLine(9.0 * s, 6.4 * s, 9.0 * s, 15.2 * s)
        p.drawLine(10.6 * s, 6.4 * s, 11.1 * s, 15.2 * s)
        p.end()


# ── playback icons ─────────────────────────────────────────────────

class _PlayPauseIcon(QWidget):
    def __init__(self, parent=None, playing=False, size=18):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._playing = playing
        self._sz = size

    def set_playing(self, v):
        self._playing = v
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(theme_manager.accent)
        s = self._sz / 18.0
        pen = QPen(c, 1.8 * s)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if self._playing:
            r = QPainterPath()
            r.addRoundedRect(4 * s, 3 * s, 4 * s, 12 * s, 1.5 * s, 1.5 * s)
            r.addRoundedRect(10 * s, 3 * s, 4 * s, 12 * s, 1.5 * s, 1.5 * s)
            p.drawPath(r)
        else:
            tri = QPainterPath()
            tri.moveTo(5 * s, 2 * s)
            tri.lineTo(16 * s, 9 * s)
            tri.lineTo(5 * s, 16 * s)
            tri.closeSubpath()
            p.drawPath(tri)
        p.end()


# ── waveform track ────────────────────────────────────────────────

def _fmt_time(ms):
    if ms is None or ms <= 0:
        return "00:00"
    s = int(ms // 1000)
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"


class _WaveformTrack(QWidget):
    play_toggled = Signal(object, bool)

    def __init__(self, label, color, bg_color, parent=None):
        super().__init__(parent)
        self._label = label
        self._color = color
        self._bg_color = bg_color
        self._samples = None
        self._playback_progress = 0.0
        self._seeking = False
        self._time_bubble_text = ""
        self._playing = False
        self._gutter = 46
        self._path = ""
        self._duration_ms = 0
        self._pending_seek_ms = None
        self._player = None
        self._audio_output = None
        self._chip_hovered = False
        self._chip_rect = None
        self._track_hovered = False
        self.setMinimumHeight(110)
        self.setMaximumHeight(130)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

        self._play_btn = QPushButton(self)
        self._play_btn.setFixedSize(34, 34)
        self._play_btn.setCursor(Qt.PointingHandCursor)
        self._play_icon = _PlayPauseIcon(self._play_btn, playing=False, size=16)
        self._play_icon.setGeometry(9, 9, 16, 16)
        self._play_btn.clicked.connect(self._on_btn_clicked)
        self._play_btn.setGeometry(6, 8, 34, 34)
        self._apply_play_style()

    def _apply_play_style(self):
        t = theme_manager.theme
        if self._playing:
            bg = f"background:{_accent_rgba(0.22)};border:1px solid {theme_manager.accent};"
            hbg = f"background:{_accent_rgba(0.34)};border:1px solid {theme_manager.accent};"
        else:
            bg = f"background:{t.border};border:1px solid {t.border_dim};"
            hbg = f"background:{t.border_dim};border:1px solid {t.border_visible};"
        self._play_btn.setStyleSheet(
            f"QPushButton{{border:none;border-radius:8px;{bg}}}"
            f"QPushButton:hover{{border:none;border-radius:8px;{hbg}}}"
        )
        if getattr(self, "_play_icon", None) is not None:
            self._play_icon.set_playing(self._playing)

    def _chip_is_hovered(self):
        return (
            self._track_hovered
            and self._chip_rect is not None
            and self._chip_rect.contains(self.mapFromGlobal(QCursor.pos()))
        )

    def enterEvent(self, event):
        self._track_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._track_hovered = False
        self._chip_hovered = False
        self.update()
        super().leaveEvent(event)

    def _ensure_player(self):
        if self._player is None:
            self._audio_output = QAudioOutput()
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._audio_output)
            self._player.positionChanged.connect(self._on_position)
            self._player.durationChanged.connect(self._on_duration)
            self._player.playbackStateChanged.connect(self._on_state)
            if self._path:
                self._player.setSource(QUrl.fromLocalFile(self._path))
        return self._player

    def _on_btn_clicked(self):
        self.toggle_playback()

    def _host(self):
        p = self.parentWidget()
        return p if isinstance(p, _WaveformContainer) else None

    def apply_sync(self, ms):
        """Drive this track's playhead from the shared container position."""
        if self._duration_ms > 0:
            self._playback_progress = min(1.0, max(0.0, ms / self._duration_ms))
            self._time_bubble_text = _fmt_time(ms) + " / " + _fmt_time(self._duration_ms)
        else:
            self._playback_progress = 0.0
            self._time_bubble_text = ""
        self.update()

    def toggle_playback(self):
        p = self._ensure_player()
        if p.playbackState() == QMediaPlayer.PlayingState:
            p.pause()
            return
        h = self._host()
        target = None
        if h is not None:
            shared = h.shared_position()
            if p.duration() > 0:
                if shared >= p.duration() - 120:
                    # Song already played to the end: restart from the top.
                    target = 0
                    h.sync_position(0)
                else:
                    target = min(shared, max(0, p.duration() - 50))
            elif shared > 0:
                # Duration not known yet; apply the seek once it is.
                self._pending_seek_ms = shared
        else:
            if p.duration() > 0 and p.position() >= p.duration() - 120:
                target = 0
        if target is not None and p.duration() > 0:
            p.setPosition(target)
        p.play()

    def pause(self):
        if self._player is not None and self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()

    def stop_and_unload(self):
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())

    def is_playing(self):
        return self._player is not None and self._player.playbackState() == QMediaPlayer.PlayingState

    def _on_state(self, state):
        playing = state == QMediaPlayer.PlayingState
        if playing != self._playing:
            self._playing = playing
            self._apply_play_style()
            self.play_toggled.emit(self, playing)
        if state == QMediaPlayer.StoppedState:
            h = self._host()
            if h is not None:
                h.sync_position(0)
            else:
                self._playback_progress = 0.0
                self._time_bubble_text = ""
                self.update()

    def _on_position(self, pos):
        h = self._host()
        if h is not None:
            h.sync_position(pos)
        elif self._duration_ms > 0:
            self._playback_progress = min(1.0, max(0.0, pos / self._duration_ms))
            self._time_bubble_text = _fmt_time(pos) + " / " + _fmt_time(self._duration_ms)
            self.update()

    def _on_duration(self, dur):
        self._duration_ms = dur if dur > 0 else 0
        if self._pending_seek_ms is not None:
            ms = self._pending_seek_ms
            self._pending_seek_ms = None
            if self._duration_ms > 0:
                self._player.setPosition(min(ms, max(0, self._duration_ms - 50)))

    def seek_to(self, ratio):
        ratio = min(1.0, max(0.0, ratio))
        p = self._ensure_player()
        if self._duration_ms > 0:
            p.setPosition(int(ratio * self._duration_ms))
        h = self._host()
        if h is not None:
            h.sync_position(int(ratio * self._duration_ms))
        else:
            self._playback_progress = ratio
            self._time_bubble_text = _fmt_time(int(ratio * self._duration_ms)) + " / " + _fmt_time(self._duration_ms)
            self.update()

    def load_audio(self, path):
        self._path = path
        # Ensure every track has a player (and thus a loaded duration) even
        # before it is ever played, so its playhead can follow the shared
        # position from the very first playback of any stem.
        self._ensure_player()
        if self._player is not None:
            self._player.setSource(QUrl.fromLocalFile(self._path))
        try:
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float64)
            step = max(1, len(data) // 400)
            data = data[::step]
            peak = np.max(np.abs(data)) or 1.0
            data = data / peak
            self._samples = data
            self._playback_progress = 0.0
            self.update()
        except Exception:
            self._samples = None
            self.update()

    def set_progress(self, ratio, time_text=""):
        self._playback_progress = max(0.0, min(1.0, ratio))
        self._time_bubble_text = time_text
        self.update()

    def _ratio_from_pos(self, pos):
        x0 = self._gutter
        pw = max(1, self.width() - x0 - 4)
        return max(0.0, min(1.0, (pos.x() - x0) / pw))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_play_btn", None) is not None:
            self._play_btn.setGeometry(6, (self.height() - 34) // 2, 34, 34)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._samples is not None:
            self._seeking = True
            self.seek_to(self._ratio_from_pos(event.pos()))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if self._seeking and self._samples is not None:
            self.seek_to(self._ratio_from_pos(pos))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._seeking = False
        event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            w = self.width()
            h = self.height()

            p.fillRect(0, 0, w, h, QColor(self._bg_color))

            x0 = self._gutter
            pd = 4
            pw = max(1, w - x0 - pd)
            ph = h - 2 * pd

            if self._samples is not None and len(self._samples) >= 2:
                n = len(self._samples)
                cy = pd + ph / 2
                half_h = ph * 0.40
                px = x0 + self._playback_progress * pw

                path = QPainterPath()
                path.moveTo(x0, cy)
                for i in range(n):
                    x = x0 + (i / (n - 1)) * pw
                    val = self._samples[i] * half_h
                    path.lineTo(x, cy - val)
                for i in range(n - 1, -1, -1):
                    x = x0 + (i / (n - 1)) * pw
                    val = self._samples[i] * half_h
                    path.lineTo(x, cy + val)
                path.closeSubpath()

                wave_c = QColor(self._color)

                p.save()
                p.setClipRect(x0, pd, int(px) - x0, ph)
                wf_bg = QColor(wave_c.red(), wave_c.green(), wave_c.blue(), 255)
                p.setBrush(wf_bg)
                p.setPen(Qt.NoPen)
                p.drawPath(path)
                p.restore()

                p.save()
                p.setClipRect(int(px), pd, x0 + pw - int(px), ph)
                dimmed = QColor(wave_c.red(), wave_c.green(), wave_c.blue(), 60)
                p.setBrush(dimmed)
                p.setPen(Qt.NoPen)
                p.drawPath(path)
                p.restore()

                # The audible track is highlighted: accent playhead + accent
                # time bubble while this stem's audio is actually playing.
                if self._playing:
                    p.setPen(QPen(QColor(theme_manager.accent), 2.0))
                else:
                    p.setPen(QPen(QColor(0, 0, 0, 160), 1.5))
                p.drawLine(int(px), pd, int(px), pd + ph)

                if self._playback_progress > 0.01:
                    bubble_w = 72
                    bubble_h = 20
                    bx = max(x0, min(int(px) - bubble_w // 2, x0 + pw - bubble_w))
                    by = pd + ph - bubble_h - 6
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(theme_manager.accent) if self._playing else QColor(0, 0, 0, 200))
                    p.drawRoundedRect(bx, by, bubble_w, bubble_h, 10, 10)
                    p.setPen(Qt.white)
                    p.setFont(QFont(FONT_FAMILY, 8))
                    p.drawText(bx, by, bubble_w, bubble_h, Qt.AlignCenter, self._time_bubble_text)

                # Small musical note under the play button while this stem is
                # the one actually audible; hidden otherwise.
                if self._playing:
                    btn_bottom = (self.height() - 34) // 2 + 34
                    note = QPainterPath()
                    note.setFillRule(Qt.WindingFill)
                    hx, hy = 23.0, btn_bottom + 18.0
                    note.addEllipse(QRectF(hx - 3.5, hy - 2.0, 7, 5))
                    note.addRect(QRectF(hx + 0.5, hy - 11, 2, 10))
                    note.moveTo(hx + 2.5, hy - 11)
                    note.quadTo(hx + 9, hy - 7, hx + 5, hy - 2)
                    note.quadTo(hx + 3, hy - 4, hx + 2.5, hy - 5.5)
                    note.closeSubpath()
                    p.save()
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(theme_manager.accent))
                    p.drawPath(note)
                    p.restore()

            if self._label:
                fm = p.fontMetrics()
                label_w = fm.horizontalAdvance(self._label) + 20
                label_h = 22
                label_x = x0 + 6
                label_y = 10
                self._chip_rect = QRect(label_x, label_y, label_w, label_h)
                col = QColor(self._color)
                # Darker fill so white text reads on every stem color.
                col = col.darker(135)
                self._chip_hovered = self._chip_is_hovered()
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawRoundedRect(self._chip_rect, 11, 11)
                if self._chip_hovered:
                    p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
                    p.setBrush(Qt.NoBrush)
                    p.drawRoundedRect(self._chip_rect.adjusted(1, 1, -1, -1), 11, 11)
                p.setPen(Qt.white)
                p.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.DemiBold))
                p.drawText(self._chip_rect, Qt.AlignCenter, self._label)
        finally:
            p.end()


# ── waveform container ────────────────────────────────────────────

class _WaveformContainer(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._card = None
        self._tracks = []
        self._shared_pos_ms = 0

        self.setStyleSheet(
            "_WaveformContainer {"
            "border-top-left-radius:0px;border-bottom-left-radius:0px;"
            "border-top-right-radius:16px;border-bottom-right-radius:16px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._track_layout = QVBoxLayout()
        self._track_layout.setContentsMargins(0, 0, 0, 0)
        self._track_layout.setSpacing(0)
        layout.addLayout(self._track_layout, 1)

    def shared_position(self):
        return self._shared_pos_ms

    def sync_position(self, ms):
        """Broadcast the shared playhead position to every track."""
        self._shared_pos_ms = ms
        for t in self._tracks:
            t.apply_sync(ms)

    def load_tracks(self, card_ref):
        self._card = card_ref
        self._shared_pos_ms = 0
        tracks = card_ref._output_paths
        old_count = len(self._tracks)
        for i, path in enumerate(tracks):
            basename = os.path.basename(path)
            name_no_ext, _ = os.path.splitext(basename)
            m = re.search(r'\(([^)]+)\)$', name_no_ext)
            label = m.group(1).strip().capitalize() if m else name_no_ext
            line_c = _stem_color(label)
            bg_c = _stem_bg_color(label)
            if i < old_count:
                track = self._tracks[i]
                track.load_audio(path)
                track._label = label
                track._color = line_c
                track._bg_color = bg_c
                track.setVisible(True)
            else:
                track = _WaveformTrack(label, line_c, bg_c, self)
                track.load_audio(path)
                track.play_toggled.connect(self._on_track_toggle)
                self._tracks.append(track)
                self._track_layout.addWidget(track)
        for i in range(len(tracks), old_count):
            self._tracks[i].setVisible(False)

    def refresh_tracks(self, card_ref):
        tracks = card_ref._output_paths
        old_count = len(self._tracks)
        for i, path in enumerate(tracks):
            basename = os.path.basename(path)
            name_no_ext, _ = os.path.splitext(basename)
            m = re.search(r'\(([^)]+)\)$', name_no_ext)
            label = m.group(1).strip().capitalize() if m else name_no_ext
            line_c = _stem_color(label)
            bg_c = _stem_bg_color(label)
            if i < old_count:
                track = self._tracks[i]
                track._label = label
                track._color = line_c
                track._bg_color = bg_c
                track.update()
            else:
                track = _WaveformTrack(label, line_c, bg_c, self)
                track.load_audio(path)
                track.play_toggled.connect(self._on_track_toggle)
                self._tracks.append(track)
                self._track_layout.addWidget(track)
        for i in range(len(tracks), old_count):
            self._tracks[i].setVisible(False)

    def _on_track_toggle(self, track, playing):
        if playing:
            for t in self._tracks:
                if t is not track:
                    t.pause()

    def pause(self):
        for t in self._tracks:
            t.pause()

    def stop_and_unload(self):
        for t in self._tracks:
            t.stop_and_unload()


# ── task card (left column) ────────────────────────────────────────

class _TaskCard(QFrame):
    selected = Signal(object)
    deleted = Signal(object)

    def __init__(self, song_name, input_path=None, parent=None):
        super().__init__(parent)
        self._song_name = song_name
        self._input_path = input_path
        self._progress = 0
        self._output_files = []
        self._output_dir = None
        self._output_paths = []
        self._start_time = time.time()
        self._is_complete = False
        self._failed = False
        self._is_selected = False
        self._cover = _load_cover_pixmap(input_path)
        self._bg_color = theme_manager.theme.card
        self._border = theme_manager.theme.border

        self.setObjectName("taskCard")
        self.setFixedHeight(120)
        self.setCursor(Qt.PointingHandCursor)
        self._hovered = False
        self._apply_style()

        self._elide_timer = None

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(12)

        self._icon = _MusicNoteIcon(completed=False, size=40)
        root.addWidget(self._icon)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)

        self._name_lbl = QLabel(song_name)
        self._raw_name = song_name
        # Long titles are elided on resize; don't let their full width force
        # the card (and the panel) wider than the viewport.
        self._name_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._name_lbl.setMinimumWidth(0)
        col.addWidget(self._name_lbl)

        self._output_lbl = QLabel()
        self._output_lbl.setVisible(False)
        self._raw_output = ""
        self._model_name = ""
        self._output_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._output_lbl.setMinimumWidth(0)
        col.addWidget(self._output_lbl)

        self._bar = _SmoothBar()
        col.addWidget(self._bar)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)

        self._pct_lbl = QLabel("0%")
        status_row.addWidget(self._pct_lbl)

        self._eta_lbl = QLabel("")
        status_row.addWidget(self._eta_lbl)

        status_row.addStretch()

        self._status_lbl = QLabel("Queued")
        status_row.addWidget(self._status_lbl)

        col.addLayout(status_row)
        root.addLayout(col, 1)

        self._menu_btn = QPushButton()
        self._menu_btn.setFixedSize(24, 24)
        self._menu_btn.setToolTip("Options")
        self._menu_btn.setText("⋮")
        root.addWidget(self._menu_btn)

        # Options popup — styled dark with hand-cursor items via style.qss
        # and the app-wide interactive-cursor filter.
        self._menu = QMenu(self)
        self._menu.addAction("Open output folder").triggered.connect(
            self._menu_open_folder)
        self._menu.addAction("Delete output files").triggered.connect(
            self._menu_delete_outputs)
        self._menu_btn.clicked.connect(self._show_options_menu)

        for w in (self._name_lbl, self._pct_lbl, self._eta_lbl,
                  self._status_lbl, self._output_lbl, self._bar, self._icon):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._apply_text_style()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _apply_style(self):
        if self._is_selected:
            border = theme_manager.accent
            self._bg_color = theme_manager.theme.surface_alt
        elif self._failed:
            border = _error_rgba(170)
            self._bg_color = theme_manager.theme.card
        elif self._is_complete:
            border = theme_manager._accent_glow
            self._bg_color = theme_manager.theme.card
        else:
            border = theme_manager.theme.border
            self._bg_color = theme_manager.theme.card
        self._border = border
        self.setStyleSheet(
            "#taskCard{"
            "background:transparent;"
            "border:none;"
            "border-radius:8px;"
            "}"
        )
        self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        try:
            with QPainter(self) as painter:
                painter.setRenderHint(QPainter.Antialiasing)
                rect = self.rect()
                path = QPainterPath()
                path.addRoundedRect(
                    QRectF(rect.x() + 0.5, rect.y() + 0.5,
                           rect.width() - 1, rect.height() - 1), 8, 8
                )
                painter.save()
                painter.setClipPath(path)
                if self._cover and not self._cover.isNull():
                    scaled = self._cover.scaled(
                        rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                    )
                    off_x = (scaled.width() - rect.width()) // 2
                    off_y = (scaled.height() - rect.height()) // 2
                    painter.drawPixmap(-off_x, -off_y, scaled)
                    painter.fillRect(rect, QColor(0, 0, 0, 140))
                else:
                    painter.fillRect(rect, QColor(self._bg_color))
                painter.restore()
                # Selected/hover: accent ring on top of cover/bg so it is
                # visible in both cases. Normal state keeps the previous look
                # (subtle border only when no cover hides it).
                if self._is_selected or self._hovered:
                    # 1px thicker in bright mode so the ring reads clearly.
                    pw = 2.5 if theme_manager.mode == "light" else 1.5
                    painter.setPen(QPen(QColor(theme_manager.accent), pw))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRoundedRect(
                        QRectF(rect.x() + pw / 2, rect.y() + pw / 2,
                               rect.width() - pw, rect.height() - pw), 8, 8
                    )
                elif not (self._cover is not None and not self._cover.isNull()):
                    painter.setPen(QPen(QColor(self._border), 1.0))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRoundedRect(
                        QRectF(rect.x() + 0.5, rect.y() + 0.5,
                               rect.width() - 1, rect.height() - 1), 8, 8
                    )
        except Exception:
            pass
        super().paintEvent(event)

    def set_selected(self, v):
        self._is_selected = v
        self._apply_style()

    def set_progress(self, pct):
        if pct < 0:
            pct = 0
        if pct > 100:
            pct = 100
        self._progress = pct
        self._bar.setValue(pct)
        self._pct_lbl.setText(f"{pct}%")

    def add_output(self, path):
        basename = os.path.basename(path)
        if basename not in self._output_files:
            self._output_files.append(basename)
            self._output_paths.append(path)
            if not self._output_dir:
                d = os.path.dirname(path)
                if d:
                    self._output_dir = d
            self._refresh_output_label()
            return True
        return False

    def set_model_name(self, name):
        self._model_name = name or ""
        self._refresh_output_label()

    def _refresh_output_label(self):
        if self._model_name:
            self._raw_output = self._model_name
        else:
            display = ", ".join(self._output_files)
            if len(display) > 60:
                display = f"{len(self._output_files)} files"
            self._raw_output = display
        self._elide_labels()
        self._output_lbl.setVisible(bool(self._raw_output))

    def set_output_dir(self, path):
        if path and os.path.isdir(path):
            self._output_dir = path

    def set_status(self, text):
        self._status_lbl.setText(text)

    def reset_progress(self):
        self._progress = 0
        self._output_files = []
        self._output_paths = []
        self._output_dir = None
        self._start_time = time.time()
        self._is_complete = False
        self._failed = False
        self._bar.setValueImmediate(0)
        self._pct_lbl.setText("0%")
        self._status_lbl.setText("Processing...")
        self._refresh_output_label()
        self._icon._completed = False
        self._icon.update()
        self._apply_style()

    def mark_complete(self):
        self._timer.stop()
        self._is_complete = True
        self._bar.setValueImmediate(100)
        self._pct_lbl.setText("100%")
        self._status_lbl.setText("Complete")
        self._icon._completed = True
        self._icon.update()
        elapsed = time.time() - self._start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self._eta_lbl.setText(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")
        self._apply_style()

    def mark_queued(self):
        self._progress = 0
        self._bar.setValueImmediate(0)
        self._pct_lbl.setText("0%")
        self._status_lbl.setText("Queued")
        self._eta_lbl.setText("")
        if self._timer.isActive():
            self._timer.stop()
        self._icon._completed = False
        self._icon.update()
        self._apply_style()

    def mark_active(self):
        self._start_time = time.time()
        self._status_lbl.setText("Processing...")
        if not self._timer.isActive():
            self._timer.start(1000)
        self._apply_style()

    def mark_failed(self):
        self._timer.stop()
        self._failed = True
        self._status_lbl.setText("Failed")
        self._icon._completed = False
        self._icon.update()
        self._apply_style()

    def _tick(self):
        elapsed = time.time() - self._start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self._eta_lbl.setText(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self)
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide_labels()

    def _elide_labels(self):
        fm = self._name_lbl.fontMetrics()
        max_w = self._name_lbl.width()
        if max_w > 0:
            self._name_lbl.setText(
                fm.elidedText(self._raw_name, Qt.ElideRight, max_w)
            )
        if self._raw_output:
            fm2 = self._output_lbl.fontMetrics()
            max_w2 = self._output_lbl.width()
            if max_w2 > 0:
                self._output_lbl.setText(
                    fm2.elidedText(self._raw_output, Qt.ElideRight, max_w2)
                )

    def _apply_text_style(self):
        """Text colors for the card. When cover art is painted the backdrop is
        always dark (image + overlay), so text must be light in both themes;
        without cover the themed card color is used and theme colors apply."""
        if self._cover is not None and not self._cover.isNull():
            name_c = "#FFFFFF"
            meta_c = theme_manager.accent
            dim_c = "rgba(255,255,255,0.78)"
            mut_c = "rgba(255,255,255,0.62)"
            btn_c = "rgba(255,255,255,0.85)"
        else:
            t = theme_manager.theme
            name_c = t.text
            meta_c = theme_manager.accent
            dim_c = t.text_dim
            mut_c = t.text_muted
            btn_c = t.text_muted
        self._name_lbl.setStyleSheet(
            "background:transparent;border:none;"
            f"font-size:14px;font-weight:bold;color:{name_c};"
        )
        self._output_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:11px;"
            f"color:{meta_c};"
        )
        self._pct_lbl.setStyleSheet(
            "background:transparent;border:none;"
            "font-weight:bold;font-size:11px;"
            f"color:{meta_c};"
        )
        self._eta_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:10px;"
            f"color:{dim_c};"
        )
        self._status_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:10px;"
            f"color:{mut_c};"
        )
        self._menu_btn.setStyleSheet(f"""
            QPushButton{{background:transparent;border:none;border-radius:4px;
            color:{btn_c};font-weight:600;font-size:16px;}}
            QPushButton:hover{{color:#FFFFFF;background:rgba(255,255,255,0.18);}}
        """)

    def _show_options_menu(self):
        pos = self._menu_btn.mapToGlobal(
            QPoint(0, self._menu_btn.height() + 2))
        self._menu.exec(pos)

    def _menu_open_folder(self):
        d = getattr(self, "_output_dir", None)
        if d and os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _menu_delete_outputs(self):
        song = self._song_name
        ret = QMessageBox.question(
            self, "Delete Output",
            f'Permanently delete output files for "{song}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        failed = []
        for p in list(self._output_paths):
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    failed.append(p)
        out_dir = getattr(self, "_output_dir", None)
        if out_dir and os.path.isdir(out_dir) and not os.listdir(out_dir):
            try:
                os.rmdir(out_dir)
            except Exception:
                pass
        self._output_paths = []
        self._output_files = []
        if failed:
            QMessageBox.warning(
                self, "Delete Output",
                "Some files failed to delete:\n" + "\n".join(failed),
            )
        self.deleted.emit(self)

    def reapply_theme(self):
        self._apply_style()
        self._apply_text_style()


# ── output list panel (left column) ────────────────────────────────

class _OutputListPanel(QWidget):
    cardSelected = Signal(object)
    cardDeleted = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards = {}
        self._selected_card = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea{{background:transparent;border:none;}}
            QScrollBar:vertical{{width:3px;background:transparent;margin:0;}}
            QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};
            border-radius:1px;min-height:20px;}}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical{{height:0;}}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical{{background:transparent;}}
        """)

        self._container = QWidget()
        self._container.setStyleSheet("background:transparent;")
        self._card_layout = QVBoxLayout(self._container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(6)
        self._card_layout.addStretch()

        scroll.setWidget(self._container)
        root.addWidget(scroll, 1)

    def _make_dot_widget(self, card):
        dot = QWidget()
        dot.setFixedSize(8, 8)
        dot.setAttribute(Qt.WA_TransparentForMouseEvents)
        return dot

    def _update_dot_color(self, dot, card):
        if card._failed:
            color = theme_manager.theme.error
        elif card._is_complete:
            color = theme_manager.theme.success
        elif card._progress > 0:
            c = QColor(theme_manager.accent)
            color = f"rgba({c.red()},{c.green()},{c.blue()},120)"
        else:
            color = theme_manager.theme.border_visible
        dot.setStyleSheet(f"background:{color};border:none;border-radius:4px;")

    def add_card(self, card):
        name = card._song_name
        self._cards[name] = card
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        card.selected.connect(self._on_card_selected)
        card.deleted.connect(self.cardDeleted.emit)
        row.addWidget(card, 1)
        dot = self._make_dot_widget(card)
        self._update_dot_color(dot, card)
        row.addWidget(dot, 0, Qt.AlignVCenter)

        container = QWidget()
        container.setStyleSheet("background:transparent;")
        container.setLayout(row)
        self._card_layout.insertWidget(self._card_layout.count() - 1, container)
        card._container = container

        # Track status changes to update dot
        orig_mark = card.mark_complete
        orig_fail = card.mark_failed

        def _mark_complete():
            orig_mark()
            self._update_dot_color(dot, card)

        def _mark_failed():
            orig_fail()
            self._update_dot_color(dot, card)

        card.mark_complete = _mark_complete
        card.mark_failed = _mark_failed

        if self._selected_card is None:
            self._select_card(card)
            self.cardSelected.emit(card)

    def _on_card_selected(self, card):
        self._select_card(card)
        self.cardSelected.emit(card)

    def _select_card(self, card):
        if self._selected_card:
            self._selected_card.set_selected(False)
        self._selected_card = card
        if card:
            card.set_selected(True)

    def get_card(self, name):
        return self._cards.get(name)

    def get_cards(self):
        return list(self._cards.values())

    def reorder(self, card_order):
        """Rearrange containers so cards appear in *card_order* sequence."""
        containers = []
        for card in card_order:
            c = getattr(card, '_container', None)
            if c:
                self._card_layout.removeWidget(c)
                containers.append(c)
        for i, c in enumerate(containers):
            self._card_layout.insertWidget(min(i, self._card_layout.count() - 1), c)

    def clear(self):
        self._cards.clear()
        self._selected_card = None
        for i in reversed(range(self._card_layout.count())):
            item = self._card_layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

    def remove_card(self, card):
        name = card._song_name
        self._cards.pop(name, None)
        if self._selected_card is card:
            self._selected_card = None
        for i in range(self._card_layout.count()):
            item = self._card_layout.itemAt(i)
            container = item.widget() if item else None
            if container is None or container.layout() is None:
                continue
            row = container.layout()
            for j in range(row.count()):
                cw = row.itemAt(j).widget() if row.itemAt(j) else None
                if cw is card:
                    container.setParent(None)
                    container.deleteLater()
                    return


# ── smooth loading indicator ────────────────────────────────────────

class _LoadingRing(QWidget):
    """Smooth 120Hz rotating ring driven by QPropertyAnimation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0.0
        self._anim = QPropertyAnimation(self, b"angle", self)
        self._anim.setDuration(1200)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(360.0)
        self._anim.setLoopCount(-1)
        self.setFixedSize(48, 48)

    def _get_angle(self):
        return self._angle

    def _set_angle(self, a):
        self._angle = a
        self.update()

    angle = Property(float, _get_angle, _set_angle)

    def start(self):
        self._anim.start()

    def stop(self):
        self._anim.stop()
        self._angle = 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(theme_manager.accent), 3)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        r = self.rect().adjusted(4, 4, -4, -4)
        p.drawArc(r, int(self._angle * 16), 300 * 16)


class _LoadingSpinner(QWidget):
    """Modern loading indicator: smooth ring + pulsing 'Loading' text.

    The pulse is done by animating the label's color alpha directly — a
    QGraphicsOpacityEffect here re-renders the label into a pixmap every
    frame and floods the console with 'Painter not active' warnings.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ring = _LoadingRing()
        self._label = QLabel("Loading")
        self._label.setAlignment(Qt.AlignCenter)
        self._pulse_val = 0.6
        self._apply_pulse()

        self._pulse = QPropertyAnimation(self, b"pulse", self)
        self._pulse.setDuration(1500)
        self._pulse.setStartValue(0.5)
        self._pulse.setEndValue(1.0)
        self._pulse.setEasingCurve(QEasingCurve.InOutSine)
        self._pulse.setLoopCount(-1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addStretch()
        layout.addWidget(self._ring, 0, Qt.AlignCenter)
        layout.addWidget(self._label)
        layout.addStretch()

    def _get_pulse(self):
        return self._pulse_val

    def _set_pulse(self, v):
        self._pulse_val = float(v)
        self._apply_pulse()

    pulse = Property(float, _get_pulse, _set_pulse)

    def _apply_pulse(self):
        c = QColor(theme_manager.theme.text)
        c.setAlpha(int(255 * self._pulse_val))
        self._label.setStyleSheet(
            "background:transparent;border:none;font-size:12px;"
            f"color:rgba({c.red()},{c.green()},{c.blue()},{c.alpha()});"
            f"font-family:{FONT_STACK};"
        )

    def start(self):
        self._ring.start()
        self._pulse.start()

    def stop(self):
        self._ring.stop()
        self._pulse.stop()
        self._set_pulse(0.6)


# ── detail view (right column) ─────────────────────────────────────

class _DetailView(QFrame):
    cardDeleted = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._card = None
        self._full_name = ""
        self._full_path = ""
        self.setStyleSheet(f"_DetailView{{background:{theme_manager.theme.card};border-radius:12px;}}")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(0)

        # Top info row
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(14)

        self._icon = _SongCoverIcon(size=64)
        info_row.addWidget(self._icon)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        self._name_lbl = QLabel("")
        self._name_lbl.setStyleSheet(
            f"background:transparent;border:none;font-size:18px;font-weight:bold;color:{theme_manager.theme.text};"
        )
        self._name_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._name_lbl.setWordWrap(False)
        text_col.addWidget(self._name_lbl)

        self._path_lbl = QLabel("")
        self._path_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:11px;"
            f"color:{theme_manager.accent};"
        )
        self._path_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._path_lbl.setWordWrap(False)
        text_col.addWidget(self._path_lbl)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(8)

        self._elapsed_lbl = QLabel("")
        self._elapsed_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:11px;"
            f"color:{theme_manager.theme.text_dim};"
        )
        meta_row.addWidget(self._elapsed_lbl)

        self._sep = QLabel("|")
        self._sep.setStyleSheet(
            "background:transparent;border:none;font-size:11px;"
            f"color:{theme_manager.theme.text_dim};"
        )
        meta_row.addWidget(self._sep)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:11px;"
            f"color:{theme_manager.accent};"
        )
        meta_row.addWidget(self._status_lbl)

        meta_row.addStretch()
        text_col.addLayout(meta_row)

        info_row.addLayout(text_col, 1)

        self._btn_delete = QPushButton()
        self._btn_delete.setFixedSize(36, 36)
        self._btn_delete.setToolTip("Delete output files")
        self._btn_delete.setStyleSheet(f"""
            QPushButton{{background:{theme_manager.theme.border};border:none;border-radius:6px;}}
            QPushButton:hover{{background:{_error_rgba(0.20)};}}
        """)
        del_icon = _TrashIcon(self._btn_delete, size=22)
        del_icon.setGeometry(7, 7, 22, 22)
        self._btn_delete.clicked.connect(self._delete_outputs)
        info_row.addWidget(self._btn_delete)

        self._btn_folder = QPushButton()
        self._btn_folder.setFixedSize(36, 36)
        self._btn_folder.setToolTip("Open output folder")
        self._btn_folder.setStyleSheet(f"""
            QPushButton{{background:{theme_manager.theme.border};border:none;border-radius:6px;}}
            QPushButton:hover{{background:{theme_manager.theme.border_dim};}}
        """)
        fo_icon = _FolderIcon(self._btn_folder)
        fo_icon.setGeometry(7, 7, 22, 22)
        self._btn_folder.clicked.connect(self._open_folder)
        info_row.addWidget(self._btn_folder)

        root.addLayout(info_row)
        root.addSpacing(30)

        # Waveform section
        self._wf_lbl = QLabel("WAVEFORM")
        self._wf_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:10px;font-weight:bold;"
            f"color:{theme_manager.theme.text_dim};letter-spacing:1px;"
        )
        root.addWidget(self._wf_lbl)
        root.addSpacing(10)

        # Waveform tracks (each track has its own play/pause button) inside a scroll area
        wave_row = QHBoxLayout()
        wave_row.setContentsMargins(0, 0, 0, 0)
        wave_row.setSpacing(0)

        self._waveform = _WaveformContainer()
        wave_row.addWidget(self._waveform, 1, Qt.AlignTop)

        self._wave_scroll = QScrollArea()
        self._wave_scroll.setWidgetResizable(True)
        self._wave_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._wave_scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
        )
        wave_inner = QWidget()
        wave_inner.setStyleSheet("background:transparent;")
        wave_inner.setLayout(wave_row)
        self._wave_scroll.setWidget(wave_inner)

        # Stack: page 0 = waveform, page 1 = loading spinner
        self._view_stack = QStackedWidget()
        self._view_stack.setStyleSheet("background:transparent;border:none;")
        self._view_stack.addWidget(self._wave_scroll)

        spinner_page = QWidget()
        spinner_page.setStyleSheet("background:transparent;")
        sp_lo = QVBoxLayout(spinner_page)
        sp_lo.setContentsMargins(0, 0, 0, 0)
        sp_lo.addStretch()
        self._spinner = _LoadingSpinner()
        sp_lo.addWidget(self._spinner, 0, Qt.AlignCenter)
        sp_lo.addStretch()
        self._view_stack.addWidget(spinner_page)

        root.addWidget(self._view_stack, 1)

        self._empty_lbl = QLabel("Select an output from the list\nto view its details and waveform.", self)
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:13px;"
            f"color:{theme_manager.theme.text_dim};"
        )

    def set_failed(self):
        """Processing ended with an error before any output existed: show the
        failure instead of leaving the detail view on Loading/Processing."""
        self._spinner.stop()
        self._view_stack.setCurrentIndex(0)
        self._waveform.setVisible(False)
        self._wf_lbl.setVisible(False)
        self._status_lbl.setText("Failed")
        self._status_lbl.setStyleSheet(
            "background:transparent;border:none;font-size:11px;font-weight:bold;"
            f"color:{theme_manager.theme.error};"
        )

    def show_card(self, card):
        self._card = card
        if card is None:
            self._show_empty()
            return

        self._waveform.pause()

        for w in (self._icon, self._name_lbl, self._path_lbl, self._elapsed_lbl,
                  self._sep, self._status_lbl, self._wf_lbl, self._view_stack):
            w.setVisible(True)
        self._empty_lbl.setVisible(False)

        self._full_name = card._song_name
        self._name_lbl.setText(self._full_name)
        files = card._output_files
        self._full_path = card._model_name or (files[0] if files else "")
        self._path_lbl.setText(self._full_path)
        QTimer.singleShot(0, self._elide)

        is_complete = card._is_complete
        is_failed = card._failed
        self._icon.set_cover(_load_cover_pixmap(card._input_path))
        self._icon._completed = is_complete
        self._icon.update()

        if is_complete:
            self._status_lbl.setText("Complete")
        elif is_failed:
            self._status_lbl.setText("Failed")
        else:
            self._status_lbl.setText("Processing...")

        elapsed_text = card._eta_lbl.text()
        self._elapsed_lbl.setText(elapsed_text)

        self._waveform.load_tracks(card)
        n = len(card._output_paths)
        show_waveform = n > 0 or is_complete or is_failed
        if show_waveform:
            self._view_stack.setCurrentIndex(0)
            self._wf_lbl.setVisible(True)
            self._spinner.stop()
            if n > 0:
                self._waveform.setFixedHeight(n * 130)
                self._waveform.setVisible(True)
            else:
                self._waveform.setVisible(False)
        else:
            self._view_stack.setCurrentIndex(1)
            self._wf_lbl.setVisible(False)
            self._spinner.start()

    def _elide(self):
        if self._name_lbl.width() > 0:
            self._name_lbl.setText(
                self._name_lbl.fontMetrics().elidedText(
                    self._full_name, Qt.ElideRight, self._name_lbl.width()))
        if self._path_lbl.width() > 0:
            self._path_lbl.setText(
                self._path_lbl.fontMetrics().elidedText(
                    self._full_path, Qt.ElideRight, self._path_lbl.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _refresh_waveform(self):
        if not self._card or not self._waveform:
            return
        self._waveform.refresh_tracks(self._card)
        n = len(self._card._output_paths)
        self._spinner.stop()
        self._view_stack.setCurrentIndex(0)
        self._wf_lbl.setVisible(True)
        if n > 0:
            self._waveform.setFixedHeight(n * 130)
            self._waveform.setVisible(True)

    def _show_empty(self):
        self._spinner.stop()
        for w in (self._icon, self._name_lbl, self._path_lbl, self._elapsed_lbl,
                  self._sep, self._status_lbl, self._wf_lbl, self._view_stack):
            w.setVisible(False)
        self._empty_lbl.setVisible(True)

        # Check if _empty_lbl is already added
        layout = self.layout()
        has_empty = False
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget() == self._empty_lbl:
                has_empty = True
                break
        if not has_empty:
            layout.addWidget(self._empty_lbl)

    def _delete_outputs(self):
        card = self._card
        if not card:
            return
        song = card._song_name
        ret = QMessageBox.question(
            self, "Delete Output",
            f'Permanently delete output files for "{song}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        # Stop playback and release file handles before deleting.
        self._waveform.stop_and_unload()

        failed = []
        for p in list(card._output_paths):
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    failed.append(p)
        # Remove the output directory if it is now empty.
        out_dir = getattr(card, "_output_dir", None)
        if out_dir and os.path.isdir(out_dir) and not os.listdir(out_dir):
            try:
                os.rmdir(out_dir)
            except Exception:
                pass

        card._output_paths = []
        card._output_files = []

        if failed:
            QMessageBox.warning(
                self, "Delete Output",
                "Some files failed to delete:\n" + "\n".join(failed),
            )

        self.cardDeleted.emit(card)
        self._show_empty()

    def _open_folder(self):
        card = self._card
        if card and card._output_dir and os.path.isdir(card._output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(card._output_dir))

    def reapply_theme(self):
        self.setStyleSheet(f"_DetailView{{background:{theme_manager.theme.card};border-radius:12px;}}")
        if self._card is not None:
            self._waveform.refresh_tracks(self._card)

class _ConsoleEdit(QTextEdit):
    _GREEN_TOKENS = (">", "[INFO]", "[PROCESS]", "[PROGRESS]", "[GPU]", "[STATUS]", "[WARN]", "[ERROR]")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("consoleEdit")
        self.setLineWrapMode(QTextEdit.NoWrap)
        self.setStyleSheet(
            f"QTextEdit#consoleEdit{{background:{theme_manager.theme.console_bg};color:{theme_manager.theme.console_text};"
            "font-family:'Courier New','Consolas',monospace;font-size:11px;"
            f"border:1px solid {theme_manager.theme.border};border-radius:6px;padding:12px;}}"
        )

    def _colorize(self, text):
        for t in self._GREEN_TOKENS:
            if text.strip().startswith(t):
                rest = text[text.index(t) + len(t):]
                return (f'<span style="color:{theme_manager.accent};font-weight:bold;">{html.escape(t)}</span>'
                        f'<span style="color:{theme_manager.theme.text};">{html.escape(rest)}</span>')
        return f'<span style="color:{theme_manager.theme.text};">{html.escape(text)}</span>'

    def _insert(self, text):
        c = self.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(c)
        self.insertHtml(self._colorize(text) + "<br>")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def append_line(self, text):
        self._insert(text)

    def clear_log(self):
        self.clear()

    def get_full_text(self):
        return self.toPlainText()


# ── console page (main) ───────────────────────────────────────────

class ConsolePage(QWidget):
    # Raw log lines kept at class level so the text survives the full page
    # recreation MainWindow performs on every theme switch; replayed through
    # the current colorizer on construction so old lines re-render in the
    # active theme instead of keeping baked-in colors.
    _LOG_HISTORY = []
    _LOG_HISTORY_MAX = 4000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job_active = False  # True while an inference/ensemble job runs
        self._song_cards = {}
        self._unmatched_exports = []
        self._current_song = None
        self._followed_song = None
        self._pending_output_dir = None
        self._output_dir = None
        self._input_path_map = {}
        self._current_model = ""
        self._input_order = []
        self._build_ui()
        self._debug_path = os.path.join(
            os.environ.get("TEMP", os.path.expanduser("~")),
            "msst_console_debug.txt",
        )
        try:
            with open(self._debug_path, "w", encoding="utf-8") as _f:
                _f.write("=== msst console debug log ===\n")
        except Exception:
            self._debug_path = None

    def set_job_active(self, active):
        """Driven by the pages' process_running signal so mid-run error text
        never marks a card FAILED before the job has actually ended."""
        self._job_active = bool(active)

    def set_current_model(self, name):
        self._current_model = name or ""
        card = self._song_cards.get(self._current_song)
        if card:
            card.set_model_name(self._current_model)

    def set_input_files(self, paths):
        self._input_order.clear()
        for p in paths or []:
            if not p or not isinstance(p, str):
                continue
            key = os.path.splitext(os.path.basename(p))[0]
            if key:
                self._input_path_map[key] = p
                self._input_path_map[key.lower()] = p
                self._input_order.append(_norm(key))
        self._sort_card_order()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 16)
        root.setSpacing(0)

        # Header
        self._header = PageHeader(
            "CONSOLE",
            "PROCESSING RESULTS & OUTPUTS",
            highlight="OUTPUTS",
        )

        self._btn_log = QPushButton("Log")
        self._btn_log.setFixedSize(70, 30)
        self._btn_log.setCheckable(True)
        self._btn_log.toggled.connect(self._toggle_view)
        self._header.add_extra(self._btn_log)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedSize(80, 30)
        self._btn_clear.clicked.connect(self._clear)
        self._header.add_extra(self._btn_clear)

        self._btn_copy = QPushButton("Copy Log")
        self._btn_copy.setFixedSize(90, 30)
        self._btn_copy.clicked.connect(self._copy)
        self._header.add_extra(self._btn_copy)

        root.addWidget(self._header)
        root.addSpacing(40)

        # 2-column content
        self._stack = QStackedWidget()

        content_widget = QWidget()
        content_widget.setStyleSheet("background:transparent;")
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        self._output_list = _OutputListPanel()
        self._output_list.cardSelected.connect(self._on_card_selected)
        self._output_list.cardDeleted.connect(self._on_card_deleted)
        content_layout.addWidget(self._output_list, 35)

        self._detail_view = _DetailView()
        self._detail_view.cardDeleted.connect(self._on_card_deleted)
        self._detail_view._show_empty()
        content_layout.addWidget(self._detail_view, 65)

        self._stack.addWidget(content_widget)

        self._log_edit = _ConsoleEdit()
        self._stack.addWidget(self._log_edit)
        for line in self._LOG_HISTORY:
            self._log_edit.append_line(line)
        self._stack.setCurrentIndex(0)

        root.addWidget(self._stack, 1)

        self._apply_styles()

    def _on_card_selected(self, card):
        self._detail_view.show_card(card)

    def _on_card_deleted(self, card):
        self._output_list.remove_card(card)
        self._song_cards.pop(getattr(card, "_key", None), None)
        self._song_cards.pop(_norm(card._song_name), None)
        if self._current_song == getattr(card, "_key", None):
            self._current_song = None
        # If the deleted card was shown in the detail view, clear it and
        # stop playback so file handles are released before deletion.
        if getattr(self._detail_view, "_card", None) is card:
            try:
                self._detail_view._waveform.stop_and_unload()
            except Exception:
                pass
            self._detail_view._show_empty()

    def _apply_styles(self):
        self.setObjectName("consolePage")
        # Object-name scoped so the background doesn't cascade into child
        # dialogs (QMessageBox etc.) and overwrite their button styles.
        self.setStyleSheet(f"#consolePage{{background:{theme_manager.theme.bg};}}")
        self._btn_log.setStyleSheet(
            f"QPushButton{{"
            f"background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};"
            f"border:1px solid {theme_manager.theme.border_dim};border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:9px;}"
            f"QPushButton:hover{{color:{theme_manager.theme.text};border:1px solid {theme_manager.theme.border_dim};}}"
            "QPushButton:checked{"
            f"background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
            "}"
        )
        self._btn_clear.setStyleSheet(
            f"QPushButton{{"
            f"background:{theme_manager.theme.surface};color:{theme_manager.theme.text_dim};"
            f"border:1px solid {theme_manager.theme.border_dim};border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:9px;}"
            f"QPushButton:hover{{color:{theme_manager.theme.error};border:1px solid {_error_rgba(0.40)};}}"
        )
        self._btn_copy.setStyleSheet(
            f"QPushButton{{"
            f"background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:9px;}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
            f"QPushButton:pressed{{background:{theme_manager.accent};}}"
        )

    def _toggle_view(self, checked):
        self._stack.setCurrentIndex(1 if checked else 0)
        if checked:
            self._header.set_title("CONSOLE")
            self._header.set_subtitle("INFERENCE PROCESS & PROGRESS", highlight="PROGRESS")
        else:
            self._header.set_title("CONSOLE")
            self._header.set_subtitle("PROCESSING RESULTS & OUTPUTS", highlight="OUTPUTS")

    def append_log(self, text):
        self._parse_and_update(text)
        self._LOG_HISTORY.append(text)
        if len(self._LOG_HISTORY) > self._LOG_HISTORY_MAX:
            del self._LOG_HISTORY[:len(self._LOG_HISTORY) - self._LOG_HISTORY_MAX]
        self._log_edit.append_line(text)
        if self._debug_path:
            try:
                with open(self._debug_path, "a", encoding="utf-8") as _f:
                    _f.write(text.rstrip("\r\n") + "\n")
            except Exception:
                self._debug_path = None

    def _is_media_file(self, raw):
        return bool(raw) and os.path.splitext(raw)[1].lower() in _AUDIO_EXTS

    def _ensure_active_card(self):
        """Enforce that _current_song points to the alphabetically first
        incomplete card.  MSST processes files in alphabetical order."""
        best = None
        for key, card in self._song_cards.items():
            if key and not card._is_complete and card._status_lbl.text() != "Complete":
                if best is None or key < best:
                    best = key
        if best is None:
            return
        if self._current_song == best:
            cur = self._song_cards.get(best)
            if cur and not cur._is_complete:
                return  # Already tracking the right card
        # Deactivate the previous card when switching to a better candidate.
        if self._current_song:
            old = self._song_cards.get(self._current_song)
            if old and old._status_lbl.text() == "Processing...":
                old.mark_queued()
        # Activate the target card.
        card = self._song_cards[best]
        if card._status_lbl.text() not in ("Processing...", "Failed"):
            card.reset_progress()
        card.mark_active()
        self._current_song = best
        self._followed_song = best

    def _parse_and_update(self, text):
        m_dir = _RE_OUTPUT_DIR.search(text)
        if m_dir:
            d = m_dir.group(1).strip()
            if d:
                self._pending_output_dir = d
                self._output_dir = d

        m_q = _RE_QUEUED.search(text)
        if m_q:
            raw = m_q.group(1).strip().strip('"').strip("'")
            # Only treat as a song if it is an actual audio file. This avoids
            # garbage cards from lines like "Queued: 5 files" or "Loading model.bin".
            if self._is_media_file(raw):
                name = os.path.splitext(os.path.basename(raw))[0]
                key = _norm(name)
                if key and key not in self._song_cards:
                    full = self._input_path_map.get(name) or self._input_path_map.get(name.lower()) or raw
                    card = _TaskCard(name, input_path=full)
                    card._key = key
                    self._song_cards[key] = card
                    self._output_list.add_card(card)
                    card.mark_queued()
                    card.set_model_name(self._current_model)
                    if self._output_dir:
                        card.set_output_dir(self._output_dir)
                    self._reconcile_unmatched()
                    self._sort_card_order()
                    self._ensure_active_card()

        qname = _extract_name(text)
        if qname:
            raw_q = _extract_raw_input_path(text)
            if self._is_media_file(raw_q):
                key = _norm(qname)
                if key not in self._song_cards:
                    full = (self._input_path_map.get(qname)
                            or self._input_path_map.get(qname.lower())
                            or raw_q or qname)
                    card = _TaskCard(qname, input_path=full)
                    card._key = key
                    self._song_cards[key] = card
                    self._output_list.add_card(card)
                    card.mark_queued()
                    card.set_model_name(self._current_model)
                    if self._output_dir:
                        card.set_output_dir(self._output_dir)
                    self._reconcile_unmatched()
                    self._sort_card_order()
                    self._ensure_active_card()

        # Handle "Completed: <name>" — marks the specific card that finished.
        completed_name = _extract_completed_name(text)
        if completed_name:
            key = _norm(completed_name)
            card = self._song_cards.get(key)
            # If direct match fails, try fuzzy: find a card whose song name is
            # contained in the completed name (handles metadata display names).
            if card is None:
                completed_lower = completed_name.lower()
                for ckey, c in self._song_cards.items():
                    if not c._is_complete and c._song_name.lower() in completed_lower:
                        card = c
                        break
            if card and not card._is_complete:
                card.mark_complete()
                if self._current_song == key or self._current_song == getattr(card, '_key', None):
                    self._current_song = None
                self._ensure_active_card()
                self._reconcile_unmatched()
                # Refresh the detail view if it's showing this card.
                if self._detail_view._card is card:
                    self._detail_view.show_card(card)
                return

        if _is_done(text) and self._current_song:
            card = self._song_cards.get(self._current_song)
            if card and not card._is_complete:
                card.mark_complete()
                self._current_song = None
                self._ensure_active_card()
                self._reconcile_unmatched()
                # Refresh the detail view if it's showing this card.
                if self._detail_view._card is card:
                    self._detail_view.show_card(card)
            return

        # Ensure the active card correctly reflects alphabetical order.
        # This is a no-op when _current_song is already correct.
        self._ensure_active_card()

        if not self._current_song:
            return

        card = self._song_cards.get(self._current_song)
        if not card:
            return

        if _has_error(text):
            if self._job_active:
                # Mid-run error-ish output (tracebacks, warnings) stays in the
                # log only; flipping the card to FAILED here desynced it from
                # the detail view, which was still loading/processing.
                return
            card.mark_failed()
            if getattr(self._detail_view, "_card", None) is card:
                self._detail_view.set_failed()
            return

        pct = _extract_progress(text)
        if pct is not None:
            card.set_progress(pct)

        export = _extract_export(text)
        if export:
            self._attach_export(None, export)
            return

    def _active_card(self):
        for c in self._song_cards.values():
            if c._status_lbl.text() == "Processing..." and not c._is_complete:
                return c
        return None

    def _attach_export(self, active_card, export):
        d = os.path.dirname(export)
        if d and (self._output_dir is None or not os.path.isdir(self._output_dir)):
            if os.path.isdir(d):
                self._output_dir = d
        target = self._match_card_for_export(export, allow_active=True)
        if target is None:
            # Buffer it; it may match once the right card appears or finishes.
            self._unmatched_exports.append(export)
            return
        self._assign_export(target, export)

    def _match_card_for_export(self, export, allow_active=True):
        basename = os.path.basename(export)
        name_no_ext, _ = os.path.splitext(basename)
        m = re.search(r'\((.+)\)$', name_no_ext)
        stemless = re.sub(r'\s*\([^)]*\)\s*$', '', name_no_ext).strip() if m else name_no_ext
        nstem = _norm(stemless)

        # 1) exact normalized stem (most reliable)
        if nstem in self._song_cards:
            return self._song_cards[nstem]
        # 2) output's parent directory often equals the song name
        #    (MSST writes each song into its own subfolder)
        parent = os.path.basename(os.path.dirname(export)).strip()
        if parent:
            nparent = _norm(parent)
            if nparent in self._song_cards:
                return self._song_cards[nparent]
        # 3) substring containment (handles sanitized / rearranged names)
        best = None
        for key, card in self._song_cards.items():
            if not key:
                continue
            if key in nstem or nstem in key:
                if best is None or len(key) > len(best[0]):
                    best = (key, card)
        if best:
            return best[1]
        # 4) last resort: the actively-processing card (never a queued one)
        if allow_active:
            return self._active_card() or active_card
        return None

    def _assign_export(self, target, export):
        # Activate queued cards when their first export arrives (the true
        # signal that MSST has started processing this song).
        was_queued = target._status_lbl.text() == "Queued"
        if was_queued:
            target.reset_progress()
            target.mark_active()
            self._current_song = target._key
            self._followed_song = target._key
        if target.add_output(export) and self._detail_view._card is target:
            self._detail_view._refresh_waveform()

    def _reconcile_unmatched(self):
        if not self._unmatched_exports:
            return
        still = []
        for export in self._unmatched_exports:
            target = self._match_card_for_export(export, allow_active=False)
            if target is None:
                still.append(export)
            else:
                self._assign_export(target, export)
        self._unmatched_exports = still



    def _sort_card_order(self):
        cards = sorted(self._song_cards.values(), key=lambda c: c._key or "")
        self._output_list.reorder(cards)

    def get_full_log(self):
        return self._log_edit.get_full_text()

    def _copy(self):
        full_text = self.get_full_log()
        if not full_text.strip():
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(full_text)
        self._show_copied_feedback()

    def _show_copied_feedback(self):
        self._btn_copy.setText("Copied!")
        self._btn_copy.setStyleSheet(
            f"QPushButton{{"
            f"background:{theme_manager._accent_soft};color:{theme_manager.accent};border:none;border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:9px;}"
        )
        QTimer.singleShot(1200, self._reset_copy_btn)

    def _reset_copy_btn(self):
        self._btn_copy.setText("Copy Log")
        self._btn_copy.setStyleSheet(
            f"QPushButton{{"
            f"background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:9px;}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
            f"QPushButton:pressed{{background:{theme_manager.accent};}}"
        )

    def _clear(self):
        self._output_list.clear()
        self._song_cards.clear()
        self._unmatched_exports = []
        self._input_order.clear()
        self._current_song = None
        self._followed_song = None
        self._pending_output_dir = None
        self._output_dir = None
        self._log_edit.clear_log()
        del self._LOG_HISTORY[:]
        self._detail_view._show_empty()

    def reapply_theme(self):
        self._apply_styles()
        for card in self._song_cards.values():
            card.reapply_theme()
        self._detail_view.reapply_theme()
