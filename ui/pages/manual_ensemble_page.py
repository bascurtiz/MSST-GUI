"""
ui/pages/ensemble_page.py
Ensemble page — combine results from multiple source separation models.
"""
import os, sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QLineEdit, QFileDialog,
    QScrollArea, QSizePolicy, QSlider, QMessageBox, QCheckBox,
)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent

import soundfile as sf

from backend.runner import ProcessRunner
from backend.paths import REPO_ROOT, get_python_exe
from ui.theme import theme_manager, UIConstants
from ui.widgets.common import (
    PageHeader, outline_button_ss, solid_button_ss, ChevronCombo, GlyphButton,
    EllipsisButton, FOLDER_GLYPH,
    _outline_icon_color, _solid_icon_color, _stop_icon_color, _addfile_icon_color,
    _pause_icon_color,
)
from ui.pages.inference_page import _ComboBox, _ExpandArrow

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".m4a", ".opus", ".wv"}
AUDIO_FILTER = "Audio files (*.wav *.flac *.mp3 *.ogg *.aiff *.m4a *.opus *.wv);;All files (*.*)"
ENSEMBLE_TYPES = [
    "avg_wave", "median_wave", "min_wave", "max_wave",
    "avg_fft", "median_fft", "min_fft", "max_fft",
]
ENSEMBLE_DESC = {
    "avg_wave": "Average waveform per sample (recommended)",
    "median_wave": "Median waveform per sample",
    "min_wave": "Minimum absolute value (conservative)",
    "max_wave": "Maximum absolute value (aggressive)",
    "avg_fft": "Average spectrogram (STFT) then inverse STFT",
    "median_fft": "Median spectrogram (good for 3+ models)",
    "min_fft": "Minimum spectrogram (most conservative)",
    "max_fft": "Maximum spectrogram (most aggressive)",
}

def _rgba_str(color_str: str, alpha: int) -> str:
    c = QColor(color_str)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def _row_ss():
    t = theme_manager.theme
    return (
        "QFrame#cfgRow{"
        f"background:{t.surface};"
        f"border:1px solid {t.border};"
        "border-radius:5px;}"
        "QFrame#cfgRow:hover{"
        f"border:1px solid {theme_manager._accent_hover}" ";}"
    )

def _icon_ss():
    t = theme_manager.theme
    return (
        f"font-size:13px;color:{t.text_muted};"
        "background:transparent;border:none;"
    )

def _lbl_ss():
    t = theme_manager.theme
    return (
        "font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
        f"color:{t.text_dim};background:transparent;letter-spacing:1px;"
    )

def _combo_ss():
    t = theme_manager.theme
    return (
        "QComboBox{background:transparent;border:none;"
        f"font-family:'Montserrat';font-size:11px;color:{t.text_sec};padding:0 4px;}}"
        "QComboBox::drop-down{border:none;width:0;}"
        "QComboBox::down-arrow{width:0;height:0;border:none;}"
        "QComboBox QAbstractItemView{"
        f"background:{t.surface_alt};"
        f"border:1px solid {t.border_dim};"
        f"color:{t.text};"
        f"selection-background-color:{theme_manager.accent};"
        f"selection-color:{theme_manager._accent_text};"
        "outline:none;}"
        "QComboBox QAbstractItemView::item{padding:6px 12px;min-height:26px;}"
    )

def _weight_slider_ss():
    t = theme_manager.theme
    return (
        "QSlider::groove:horizontal{"
        f"background:{t.border_dim};height:4px;border-radius:2px;}}"
        "QSlider::handle:horizontal{"
        f"background:{theme_manager.accent};width:14px;height:14px;margin:-5px 0;"
        "border:none;border-radius:7px;}"
        "QSlider::handle:horizontal:hover{"
        f"background:{theme_manager._accent_hover};}}"
        "QSlider::sub-page:horizontal{"
        f"background:{theme_manager.accent};border-radius:2px;height:4px" ";}"
    )

def _weight_val_ss():
    return (
        "font-family:'Courier New',monospace;font-size:11px;font-weight:bold;"
        f"color:{theme_manager.accent};background:transparent;"
    )

def _drop_zone_ss():
    t = theme_manager.theme
    return (
        "QFrame#dropZone{"
        f"background:{_rgba_str(t.text, 2)};"
        f"border:1px dashed {t.border_dim};"
        "border-radius:6px;}"
        "QFrame#dropZone:hover{"
        f"border-color:{theme_manager._accent_glow};"
        f"background:{theme_manager._accent_soft}" ";}"
        "QFrame#dropZone[dragOver=true]{"
        f"border-color:{theme_manager.accent};"
        f"background:{theme_manager._accent_soft}" ";}"
    )

def _file_row_ss():
    t = theme_manager.theme
    return (
        "QFrame#fileRow{"
        f"background:{t.surface};"
        f"border:1px solid {t.border};"
        "border-radius:6px;}"
        "QFrame#fileRow:hover{"
        f"border-color:{theme_manager._accent_hover}" ";}"
    )

def _guide_ss():
    t = theme_manager.theme
    return (
        "QFrame#guidePanel{"
        f"background:{t.bg};"
        f"border:1px solid {t.border_dim};"
        "border-radius:8px;}"
    )

def _badge_ss():
    return (
        "QLabel#sectionBadge{"
        f"background:{theme_manager.accent};"
        f"color:{theme_manager._accent_text};"
        "font-family:'Montserrat',sans-serif;"
        "font-size:10px;font-weight:bold;"
        "border-radius:4px;"
        "padding:2px 6px;}"
    )
ROW_H = 46


def _sec_hdr(text):
    t = theme_manager.theme
    w = QLabel(text.upper())
    w.setStyleSheet(
        "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
        f"color:{t.text};background:transparent;padding-left:8px;"
        f"border-left:3px solid {theme_manager.accent};letter-spacing:1.5px;"
    )
    w.setFixedHeight(18)
    return w


def _hdiv(opacity=20):
    f = QFrame(); f.setFixedHeight(1)
    t = theme_manager.theme
    f.setStyleSheet(
        f"background:{_rgba_str(t.text, int(opacity * 255 / 100))};border:none;"
    )
    return f


def get_audio_metadata(path):
    try:
        info = sf.info(path)
        fmt = info.format.upper()
        sr = f"{info.samplerate/1000:.1f} kHz"
        ch = "Stereo" if info.channels == 2 else "Mono"
        dur = f"{int(info.duration//60):02d}:{int(info.duration%60):02d}"
        return f"{fmt} \u2022 {sr} \u2022 {ch} \u2022 {dur}"
    except Exception:
        return "Unknown format \u2022 Unknown \u2022 Unknown \u2022 00:00"


def get_item_metadata(path):
    if not path or not os.path.exists(path):
        return ""
    if os.path.isdir(path):
        try:
            count = len([
                f for f in Path(path).rglob('*')
                if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
            ])
            return f"Folder \u2022 {count} stem file(s) found"
        except Exception:
            return "Folder"
    return get_audio_metadata(path)


def _is_audio(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS


class _DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setStyleSheet(_drop_zone_ss())
        self.setMinimumHeight(150)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(32, 28, 32, 28)
        vl.setSpacing(0)
        vl.setAlignment(Qt.AlignCenter)

        icon_wrapper = QWidget()
        icon_wrapper.setStyleSheet("background:transparent;")
        icon_layout = QHBoxLayout(icon_wrapper)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.addStretch()

        self._icon_container = QWidget()
        self._icon_container.setFixedSize(52, 60)
        icon_inner = QVBoxLayout(self._icon_container)
        icon_inner.setContentsMargins(0, 0, 0, 0)
        icon_inner.setAlignment(Qt.AlignCenter)

        self._icon = QLabel("\u266A")
        self._icon.setStyleSheet(
            "font-size:22px;color:" + theme_manager.accent + ";background:transparent;border:none;"
        )
        self._icon.setAlignment(Qt.AlignCenter)
        icon_inner.addWidget(self._icon)

        icon_layout.addWidget(self._icon_container)
        icon_layout.addStretch()
        vl.addWidget(icon_wrapper)

        vl.addSpacing(10)

        self._main_lbl = QLabel("Drag & drop audio files here")
        self._main_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;font-weight:600;color:{theme_manager.theme.text};"
            "background:transparent;"
        )
        self._main_lbl.setAlignment(Qt.AlignCenter)
        vl.addWidget(self._main_lbl)

        vl.addSpacing(3)

        self._sub_lbl = QLabel('or click "+ Add File" to browse')
        self._sub_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_dim};"
            "background:transparent;"
        )
        self._sub_lbl.setAlignment(Qt.AlignCenter)
        vl.addWidget(self._sub_lbl)

        vl.addSpacing(3)

        self._fmt_lbl = QLabel("Supports WAV, FLAC, MP3, OGG")
        self._fmt_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_muted};"
            "background:transparent;"
        )
        self._fmt_lbl.setAlignment(Qt.AlignCenter)
        vl.addWidget(self._fmt_lbl)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            has_valid = any(os.path.isdir(u.toLocalFile()) or _is_audio(u.toLocalFile()) for u in urls)
            if has_valid:
                event.acceptProposedAction()
                self.setProperty("dragOver", True)
                self.style().polish(self)
            else:
                event.ignore()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            has_valid = any(os.path.isdir(u.toLocalFile()) or _is_audio(u.toLocalFile()) for u in urls)
            if has_valid:
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragOver", False)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dragOver", False)
        self.style().polish(self)
        urls = event.mimeData().urls()
        paths = []
        for u in urls:
            p = u.toLocalFile()
            if os.path.isdir(p) or (_is_audio(p) and os.path.isfile(p)):
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
            self._main_lbl.setText(f"{len(paths)} item(s) added")
            QTimer.singleShot(2000, self._reset_text)

    def _reset_text(self):
        self._main_lbl.setText("Drag & drop audio files or folders here")

    def mousePressEvent(self, event):
        pass


class _EnsembleFileRow(QFrame):
    removed = Signal()

    def __init__(self, path="", parent=None):
        super().__init__(parent)
        self._path = path
        self.setObjectName("fileRow")
        self.setFixedHeight(64)
        self.setStyleSheet(_file_row_ss())

        hl = QHBoxLayout(self)
        hl.setContentsMargins(14, 10, 10, 10)
        hl.setSpacing(0)

        t = theme_manager.theme
        is_dir = os.path.isdir(path)
        ic = QLabel("\U0001F4C1" if is_dir else "\u266A")
        ic.setStyleSheet("font-size:16px;color:" + t.purple + ";background:transparent;border:none;")
        ic.setFixedWidth(28)
        ic.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        hl.addWidget(ic)

        info_vl = QVBoxLayout()
        info_vl.setSpacing(2)
        info_vl.setContentsMargins(0, 0, 0, 0)

        self._name_lbl = QLabel(os.path.basename(path) if path else "Unknown")
        self._name_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:12px;font-weight:bold;color:{t.text};"
            "background:transparent;"
        )
        info_vl.addWidget(self._name_lbl)

        self._meta_lbl = QLabel(get_item_metadata(path) if path else "")
        self._meta_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{t.text_dim};"
            "background:transparent;"
        )
        info_vl.addWidget(self._meta_lbl)

        hl.addLayout(info_vl, 1)

        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(30)
        sep.setStyleSheet(f"background:{t.border};border:none;")
        hl.addWidget(sep)
        hl.addSpacing(12)

        weight_vl = QVBoxLayout()
        weight_vl.setSpacing(2)
        weight_vl.setContentsMargins(0, 0, 0, 0)

        weight_lbl = QLabel("WEIGHT")
        weight_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:8px;font-weight:700;"
            f"color:{t.text_muted};background:transparent;letter-spacing:1px;"
        )
        weight_vl.addWidget(weight_lbl)

        slider_wrap = QWidget()
        slider_wrap.setStyleSheet("background:transparent;")
        sw = QHBoxLayout(slider_wrap)
        sw.setContentsMargins(0, 0, 50, 0)
        sw.setSpacing(0)

        self._weight_slider = QSlider(Qt.Horizontal)
        self._weight_slider.setMinimum(1)
        self._weight_slider.setMaximum(100)
        self._weight_slider.setValue(10)
        self._weight_slider.setStyleSheet(_weight_slider_ss())
        self._weight_slider.setFixedHeight(28)
        self._weight_slider.setFixedWidth(100)
        self._weight_slider.valueChanged.connect(self._on_weight_changed)
        sw.addWidget(self._weight_slider)

        self._weight_val_lbl = QLabel("1.0")
        self._weight_val_lbl.setStyleSheet(_weight_val_ss())
        self._weight_val_lbl.setFixedWidth(35)
        self._weight_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sw.addWidget(self._weight_val_lbl)

        weight_vl.addWidget(slider_wrap)
        hl.addLayout(weight_vl)

        self._btn_dots = QPushButton("\u22EE")
        self._btn_dots.setFixedSize(24, 24)
        self._btn_dots.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_label};"
            "border:none;font-size:14px;border-radius:4px;}"
            f"QPushButton:hover{{color:{theme_manager.accent};background:{theme_manager._accent_soft};}}"
        )
        hl.addWidget(self._btn_dots)

        self._btn_remove = QPushButton("\u2715")
        self._btn_remove.setFixedSize(24, 24)
        self._btn_remove.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_label};"
            "border:none;font-size:12px;border-radius:4px;}"
            f"QPushButton:hover{{background:{_rgba_str(t.error, 51)};color:{t.error};}}"
        )
        self._btn_remove.clicked.connect(self.removed.emit)
        hl.addWidget(self._btn_remove)

    def _on_weight_changed(self, val):
        weight = val / 10.0
        self._weight_val_lbl.setText(f"{weight:.1f}")

    def get_path(self):
        return self._path

    def get_weight(self):
        return self._weight_slider.value() / 10.0

    def apply_theme(self):
        t = theme_manager.theme
        self.setStyleSheet(_file_row_ss())
        self._name_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:12px;font-weight:bold;color:{t.text};"
            "background:transparent;"
        )
        self._meta_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{t.text_dim};"
            "background:transparent;"
        )
        self._weight_slider.setStyleSheet(_weight_slider_ss())
        self._weight_val_lbl.setStyleSheet(_weight_val_ss())
        self._btn_dots.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_label};"
            "border:none;font-size:14px;border-radius:4px;}"
            f"QPushButton:hover{{color:{theme_manager.accent};background:{theme_manager._accent_soft};}}"
        )
        self._btn_remove.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_label};"
            "border:none;font-size:12px;border-radius:4px;}"
            f"QPushButton:hover{{background:{_rgba_str(t.error, 51)};color:{t.error};}}"
        )


class _EnsembleGuidePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("guidePanel")
        self.setStyleSheet(_guide_ss())

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 14, 20)
        root.setSpacing(0)

        t = theme_manager.theme
        hdr = QHBoxLayout()
        title = QLabel("How Ensemble Works")
        title.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_TITLE_FONT_SIZE}px;font-weight:bold;"
            f"color:{t.text};background:transparent;letter-spacing:1.5px;"
        )
        hdr.addWidget(title)
        hdr.addStretch()

        self._close_btn = QPushButton("\u2715")
        self._close_btn.setFixedSize(24, 24)
        self._close_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_muted};"
            "border:none;font-size:11px;border-radius:4px;}"
            f"QPushButton:hover{{background:{_rgba_str(t.error, 51)};color:{t.error};}}"
        )
        hdr.addWidget(self._close_btn)
        root.addLayout(hdr)

        root.addSpacing(16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            f"QScrollBar:vertical{{width:4px;background:{t.scrollbar_bg};margin:0;}}"
            f"QScrollBar::handle:vertical{{background:{t.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            f"QScrollBar::handle:vertical:hover{{background:{t.scrollbar_hover};}}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,"
            "QScrollBar::sub-page:vertical{background:transparent;}"
        )
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content.setMinimumWidth(300)
        content.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 10, 0)
        cl.setSpacing(0)

        cl.addWidget(self._section_input(), 1)
        cl.addWidget(self._section_divider(), 1)
        cl.addWidget(self._section_process(), 1)
        cl.addWidget(self._section_divider(), 1)
        cl.addWidget(self._section_output(), 1)
        cl.addWidget(self._section_divider(), 1)
        cl.addWidget(self._section_weight(), 1)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _section_divider(self):
        t = theme_manager.theme
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        container.setStyleSheet("background:transparent;")
        container.setFixedHeight(28)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        left_div = QFrame()
        left_div.setFixedHeight(1)
        left_div.setStyleSheet(f"background:{t.border_dim};border:none;")
        row.addWidget(left_div, 1)
        arrow = QLabel("\u2193")
        arrow.setStyleSheet(
            "font-size:14px;color:" + t.text_dim + ";background:transparent;border:none;"
        )
        arrow.setAlignment(Qt.AlignCenter)
        row.addWidget(arrow)
        right_div = QFrame()
        right_div.setFixedHeight(1)
        right_div.setStyleSheet(f"background:{t.border_dim};border:none;")
        row.addWidget(right_div, 1)
        return container

    def _divider(self):
        t = theme_manager.theme
        f = QFrame()
        f.setFixedHeight(1)
        f.setStyleSheet(f"background:{t.border};border:none;")
        return f

    def _badge(self, num):
        b = QLabel(str(num))
        b.setFixedSize(24, 24)
        b.setAlignment(Qt.AlignCenter)
        b.setStyleSheet(
            "background:" + theme_manager.accent + ";"
            "color:" + theme_manager._accent_text + ";"
            "font-family:'Montserrat',sans-serif;"
            "font-size:11px;font-weight:bold;"
            "border-radius:3px;"
        )
        return b

    def _section_input(self):
        t = theme_manager.theme
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(14)
        vl.addSpacing(8)

        hr = QHBoxLayout()
        hr.setSpacing(8)
        hr.addWidget(self._badge(1))
        tl = QLabel("INPUT")
        tl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_TITLE_FONT_SIZE}px;font-weight:bold;"
            "color:" + theme_manager.accent + ";background:transparent;letter-spacing:1px;"
        )
        hr.addWidget(tl)
        hr.addStretch()
        vl.addLayout(hr)

        desc = QLabel(
            "You have multiple audio files that are the same stem "
            "from different models:"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{t.text_sec};"
            "background:transparent;line-height:1.6;"
        )
        vl.addWidget(desc)

        files_layout = QVBoxLayout()
        files_layout.setSpacing(8)

        for name, desc_text in [
            ("vocals_model_A.wav", "Result from model A"),
            ("vocals_model_B.wav", "Result from model B"),
            ("vocals_model_C.wav", "Result from model C"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(8)
            icon = QLabel("\u266A")
            icon.setStyleSheet(
                "font-size:14px;color:" + t.purple + ";background:transparent;border:none;"
            )
            row.addWidget(icon)
            fname = QLabel(name)
            fname.setStyleSheet(
                f"font-family:'Courier New',monospace;font-size:13px;color:{t.text_sec};"
                "background:transparent;"
            )
            row.addWidget(fname)
            arrow = QLabel("\u2190")
            arrow.setStyleSheet(
                f"font-size:11px;color:{t.text_muted};background:transparent;border:none;"
            )
            row.addWidget(arrow)
            d = QLabel(desc_text)
            d.setStyleSheet(
                f"font-family:'Montserrat';font-size:13px;color:{t.text_dim};"
                "background:transparent;"
            )
            row.addWidget(d, 1)
            files_layout.addLayout(row)

        vl.addLayout(files_layout)

        return w

    def _section_process(self):
        t = theme_manager.theme
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(14)
        vl.addSpacing(8)

        hr = QHBoxLayout()
        hr.setSpacing(8)
        hr.addWidget(self._badge(2))
        tl = QLabel("PROCESS")
        tl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_TITLE_FONT_SIZE}px;font-weight:bold;"
            "color:" + theme_manager.accent + ";background:transparent;letter-spacing:1px;"
        )
        hr.addWidget(tl)
        hr.addStretch()
        vl.addLayout(hr)

        desc = QLabel("Ensemble combines these files using one of 8 methods:")
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{t.text_sec};"
            "background:transparent;line-height:1.6;"
        )
        vl.addWidget(desc)

        tbl = QFrame()
        tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tbl.setMinimumWidth(250)
        tbl.setStyleSheet("QFrame{background:transparent;border:none;}")
        tbl_layout = QVBoxLayout(tbl)
        tbl_layout.setContentsMargins(0, 8, 0, 8)
        tbl_layout.setSpacing(0)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 6)
        header_row.setSpacing(16)
        h_method = QLabel("METHOD")
        h_method.setStyleSheet(
            "font-family:'Courier New',monospace;font-size:11px;font-weight:bold;"
            f"color:{t.text_dim};background:transparent;letter-spacing:0.5px;"
        )
        h_method.setFixedWidth(75)
        header_row.addWidget(h_method)
        h_works = QLabel("HOW IT WORKS")
        h_works.setStyleSheet(
            "font-family:'Montserrat';font-size:11px;font-weight:600;"
            f"color:{t.text_dim};background:transparent;"
        )
        header_row.addWidget(h_works, 1)
        tbl_layout.addLayout(header_row)

        tbl_layout.addWidget(self._divider())

        methods = [
            ("avg_wave", "Average every waveform sample"),
            ("median_wave", "Median of every waveform sample"),
            ("min_wave", "Take minimum absolute value"),
            ("max_wave", "Take maximum absolute value"),
            ("avg_fft", "Average spectrogram (STFT), then inverse STFT"),
            ("median_fft", "Median spectrogram"),
            ("min_fft", "Minimum spectrogram"),
            ("max_fft", "Maximum spectrogram"),
        ]

        for i, (m, d) in enumerate(methods):
            row = QHBoxLayout()
            row.setContentsMargins(0, 4, 0, 4)
            row.setSpacing(12)
            name = QLabel(m)
            name.setStyleSheet(
                "font-family:'Courier New',monospace;font-size:12px;font-weight:bold;"
                "color:" + theme_manager.accent + ";background:transparent;"
            )
            name.setFixedWidth(75)
            row.addWidget(name)
            desc_lbl = QLabel(d)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:12px;color:{t.text_sec};"
                "background:transparent;"
            )
            row.addWidget(desc_lbl, 1)
            tbl_layout.addLayout(row)
            if i < len(methods) - 1:
                tbl_layout.addWidget(self._divider())

        vl.addWidget(tbl)

        return w

    def _section_output(self):
        t = theme_manager.theme
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(14)
        vl.addSpacing(8)

        hr = QHBoxLayout()
        hr.setSpacing(8)
        hr.addWidget(self._badge(3))
        tl = QLabel("OUTPUT")
        tl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_TITLE_FONT_SIZE}px;font-weight:bold;"
            "color:" + theme_manager.accent + ";background:transparent;letter-spacing:1px;"
        )
        hr.addWidget(tl)
        hr.addStretch()
        vl.addLayout(hr)

        desc = QLabel("One combined audio file:")
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{t.text_sec};"
            "background:transparent;line-height:1.6;"
        )
        vl.addWidget(desc)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        out_icon = QLabel("\u266A")
        out_icon.setStyleSheet(
            "font-size:14px;color:" + theme_manager.accent + ";background:transparent;border:none;"
        )
        out_row.addWidget(out_icon)
        out_name = QLabel("vocals_ensemble.wav")
        out_name.setStyleSheet(
            f"font-family:'Courier New',monospace;font-size:13px;color:{t.text_sec};"
            "background:transparent;"
        )
        out_row.addWidget(out_name)
        arrow = QLabel("\u2190")
        arrow.setStyleSheet(
            f"font-size:11px;color:{t.text_muted};background:transparent;border:none;"
        )
        out_row.addWidget(arrow)
        out_desc = QLabel("Ensemble result")
        out_desc.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{t.text_dim};"
            "background:transparent;"
        )
        out_row.addWidget(out_desc, 1)
        vl.addLayout(out_row)

        return w

    def _section_weight(self):
        t = theme_manager.theme
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(14)
        vl.addSpacing(8)

        hr = QHBoxLayout()
        hr.setSpacing(8)
        hr.addWidget(self._badge(4))
        tl = QLabel("WEIGHT SYSTEM")
        tl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_TITLE_FONT_SIZE}px;font-weight:bold;"
            "color:" + theme_manager.accent + ";background:transparent;letter-spacing:1px;"
        )
        hr.addWidget(tl)
        hr.addStretch()
        vl.addLayout(hr)

        desc = QLabel("Each file can have a different weight:")
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{t.text_sec};"
            "background:transparent;line-height:1.6;"
        )
        vl.addWidget(desc)

        weight_box = QFrame()
        weight_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        weight_box.setStyleSheet("QFrame{background:transparent;border:none;}")
        wb_layout = QVBoxLayout(weight_box)
        wb_layout.setContentsMargins(0, 0, 0, 0)

        formula = QLabel("File A (weight 2.0) + File B (weight 1.0) + File C (weight 0.5)")
        formula.setWordWrap(True)
        formula.setStyleSheet(
            "font-family:'Courier New',monospace;font-size:12px;color:" + theme_manager.accent + ";"
            f"background:{theme_manager._accent_soft};"
            f"border:1px solid {theme_manager._accent_glow};"
            "border-radius:4px;"
            "padding:8px 12px;"
            "font-weight:bold;"
        )
        wb_layout.addWidget(formula, 1)
        vl.addWidget(weight_box)

        vl.addSpacing(10)

        cols_row = QHBoxLayout()
        cols_row.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_title = QLabel("How weight works:")
        left_title.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;"
            "color:" + theme_manager.accent + ";background:transparent;letter-spacing:0.5px;"
        )
        left_col.addWidget(left_title)

        bullets = [
            "avg_wave: (A\u00d72.0 + B\u00d71.0 + C\u00d70.5) / 3.5",
            "File A has 2x more influence than File B",
            "File C has half the influence of File B",
        ]
        for b in bullets:
            bl = QLabel("\u2022 " + b)
            bl.setWordWrap(True)
            bl.setStyleSheet(
                f"font-family:'Montserrat';font-size:12px;color:{t.text_sec};"
                "background:transparent;"
            )
            left_col.addWidget(bl)
        cols_row.addLayout(left_col, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_title = QLabel("Weight only applies to:")
        right_title.setWordWrap(True)
        right_title.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;"
            "color:" + theme_manager.accent + ";background:transparent;letter-spacing:0.5px;"
        )
        right_col.addWidget(right_title)

        for txt in ["avg_wave", "avg_fft"]:
            row = QHBoxLayout()
            row.setSpacing(6)
            check = QLabel("\u2713")
            check.setStyleSheet(
                "font-size:11px;color:" + t.success + ";background:transparent;border:none;"
            )
            row.addWidget(check)
            il = QLabel(txt)
            il.setWordWrap(True)
            il.setStyleSheet(
                f"font-family:'Montserrat';font-size:12px;color:{t.text_sec};"
                "background:transparent;"
            )
            row.addWidget(il)
            row.addStretch()
            right_col.addLayout(row)

        right_col.addSpacing(8)
        no_title = QLabel("Weight does NOT apply to:")
        no_title.setWordWrap(True)
        no_title.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:bold;"
            "color:" + t.error + ";background:transparent;letter-spacing:0.5px;"
        )
        right_col.addWidget(no_title)

        for txt in ["median_wave, min_wave, max_wave", "median_fft, min_fft, max_fft"]:
            row = QHBoxLayout()
            row.setSpacing(0)
            xmark = QLabel("\u2715")
            xmark.setStyleSheet(
                "font-size:11px;color:" + t.error + ";background:transparent;border:none;"
            )
            row.addWidget(xmark)
            nl = QLabel(txt)
            nl.setStyleSheet(
                f"font-family:'Montserrat';font-size:12px;color:{t.text_dim};"
                "background:transparent;"
            )
            row.addWidget(nl)
            row.addSpacing(12)
            sub = QLabel("(all files treated equally)")
            sub.setStyleSheet(
                f"font-family:'Montserrat';font-size:11px;color:{t.text_muted};"
                "background:transparent;"
            )
            row.addWidget(sub)
            row.addStretch()
            right_col.addLayout(row)

        cols_row.addLayout(right_col, 1)
        vl.addLayout(cols_row)

        return w


class ManualEnsemblePage(QWidget):
    navigate_back = Signal()
    log_output = Signal(str)
    process_running = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("manualEnsemblePage")
        self._runner = None
        self._file_rows = []
        self._mode = "files"
        self._build_ui()
        self.reapply_theme()

    def reapply_theme(self):
        t = theme_manager.theme
        # Object-name scoped so the background doesn't cascade into child
        # dialogs (QMessageBox etc.) and overwrite their button styles.
        self.setStyleSheet(f"#manualEnsemblePage{{background:{t.bg};}}")

        self._guide_panel.setStyleSheet(_guide_ss())
        self._drop_zone.setStyleSheet(_drop_zone_ss())
        if hasattr(self, "_action_bar"):
            self._action_bar.setStyleSheet(
                f"QWidget#actionBar{{background:{t.bg};border:none;}}"
            )

        for w in self.findChildren(QPushButton):
            if w is self.btn_run:
                w.setStyleSheet(solid_button_ss())
                w._refresh_icon()
            elif w is self.btn_stop:
                self._update_stop_style()
                w._refresh_icon()
            elif w is self._open_btn:
                self._update_open_style()
                w._refresh_icon()
            elif w is self._guide_btn:
                w.setStyleSheet(
                    f"QPushButton{{background:transparent;color:{t.text_dim};"
                    f"border:1px solid {t.border_dim};border-radius:4px;"
                    "padding-left:10px;padding-right:10px;"
                    "font-family:'Montserrat',sans-serif;font-weight:600;font-size:8px;"
                    "}"
                    f"QPushButton:hover{{color:{theme_manager.accent};border:1px solid {theme_manager._accent_glow};}}"
                )
            elif w is self._btn_add:
                w.setStyleSheet(
                    f"QPushButton{{background:transparent;color:{t.text_muted};"
                    f"border:1px solid {t.border_dim};border-radius:4px;"
                    "font-family:'Montserrat',sans-serif;font-weight:600;font-size:8px;"
                    "}"
                    f"QPushButton:hover{{color:{theme_manager.accent};border:1px solid {theme_manager._accent_glow};}}"
                )

        for w in self.findChildren(QComboBox):
            w.setStyleSheet(_combo_ss())

        if hasattr(self, "_trim_chk"):
            self._trim_chk.setStyleSheet(
                f"QCheckBox{{background:transparent;spacing:6px;font-family:'Montserrat';font-size:10px;color:{t.text_dim};}}"
                f"QCheckBox::indicator{{width:14px;height:14px;border-radius:3px;border:1px solid {t.border_dim};background:{t.surface};}}"
                f"QCheckBox::indicator:hover{{border-color:{theme_manager.accent};}}"
                f"QCheckBox::indicator:checked{{background:{theme_manager.accent};border-color:{theme_manager.accent};}}"
            )

        for row in self._file_rows:
            row.apply_theme()

        for w in self.findChildren(QLabel):
            obj = w.objectName()
            if obj == "sectionBadge":
                w.setStyleSheet(_badge_ss())

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header_w = QWidget()
        header_w.setStyleSheet("background:transparent;")
        hh = QHBoxLayout(header_w)
        hh.setContentsMargins(32, 32, 32, 0)
        hh.setSpacing(0)

        hdr = PageHeader(
            "MANUAL ENSEMBLE",
            "COMBINE MODELS WITH CUSTOM WEIGHTS",
            highlight="CUSTOM WEIGHTS",
            back=True,
            help_key="manual_ensemble",
        )
        self._back_btn = hdr.back_btn
        self._back_btn.clicked.connect(self.navigate_back.emit)
        hh.addWidget(hdr, 1)
        root.addWidget(header_w)
        root.addSpacing(UIConstants.HEADER_CONTENT_GAP)

        t = theme_manager.theme

        content = QWidget()
        content.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(32, 0, 32, 0)
        cl.setSpacing(0)

        self._main_split = QHBoxLayout()
        self._main_split.setContentsMargins(0, 0, 0, 0)
        self._main_split.setSpacing(0)

        workspace = QWidget()
        workspace.setStyleSheet("background:transparent;")
        wl = QVBoxLayout(workspace)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)

        settings_hdr = QHBoxLayout()
        settings_hdr.addWidget(_sec_hdr("Settings"))
        settings_hdr.addStretch()

        self._guide_btn = QPushButton("How Ensemble Works")
        self._guide_btn.setFixedHeight(24)
        self._guide_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_dim};"
            f"border:1px solid {t.border_dim};border-radius:4px;"
            "padding-left:10px;padding-right:10px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:8px;"
            "}"
            f"QPushButton:hover{{color:{theme_manager.accent};border:1px solid {theme_manager._accent_glow};}}"
        )
        self._guide_btn.clicked.connect(self._toggle_guide)
        settings_hdr.addWidget(self._guide_btn)
        wl.addLayout(settings_hdr)

        wl.addSpacing(10)

        cfg = QVBoxLayout()
        cfg.setSpacing(6)
        cfg.setContentsMargins(0, 0, 0, 0)

        # Mode Selection Row (Folder Mode vs File Mode)
        mode_row = QFrame()
        mode_row.setObjectName("cfgRow")
        mode_row.setFixedHeight(ROW_H)
        mode_row.setStyleSheet(_row_ss())
        mr_hl = QHBoxLayout(mode_row)
        mr_hl.setContentsMargins(14, 0, 14, 0)
        mr_hl.setSpacing(0)

        mr_ic = QLabel("\u29C9")
        mr_ic.setStyleSheet(_icon_ss())
        mr_ic.setFixedWidth(28)
        mr_ic.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        mr_hl.addWidget(mr_ic)

        mr_lb = QLabel("MODE")
        mr_lb.setStyleSheet(_lbl_ss())
        mr_lb.setFixedWidth(100)
        mr_lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        mr_hl.addWidget(mr_lb)

        sep_m = QFrame()
        sep_m.setFixedWidth(1); sep_m.setFixedHeight(24)
        sep_m.setStyleSheet(f"background:{t.border};border:none;")
        mr_hl.addWidget(sep_m)
        mr_hl.addSpacing(10)

        self._mode_combo = _ComboBox()
        self._mode_combo.addItem("File Mode (Single Track)", "Ensemble 2+ audio files into a single output file")
        self._mode_combo.addItem("Folder Mode (Batch Folders)", "Ensemble matching stem files across 2+ folders")
        self._mode_combo.setStyleSheet(_combo_ss())
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mr_hl.addWidget(self._mode_combo, 1)

        self._mode_arrow = _ExpandArrow()
        mr_hl.addWidget(self._mode_arrow)
        self._mode_combo.popupOpened.connect(lambda: self._mode_arrow.set_down(True))
        self._mode_combo.popupClosed.connect(lambda: self._mode_arrow.set_down(False))
        cfg.addWidget(mode_row)

        type_row = QFrame()
        type_row.setObjectName("cfgRow")
        type_row.setFixedHeight(ROW_H)
        type_row.setStyleSheet(_row_ss())
        tr_hl = QHBoxLayout(type_row)
        tr_hl.setContentsMargins(14, 0, 14, 0)
        tr_hl.setSpacing(0)

        tr_ic = QLabel("\u25C8")
        tr_ic.setStyleSheet(_icon_ss())
        tr_ic.setFixedWidth(28)
        tr_ic.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        tr_hl.addWidget(tr_ic)

        tr_lb = QLabel("TYPE")
        tr_lb.setStyleSheet(_lbl_ss())
        tr_lb.setFixedWidth(100)
        tr_lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        tr_hl.addWidget(tr_lb)

        sep = QFrame()
        sep.setFixedWidth(1); sep.setFixedHeight(24)
        sep.setStyleSheet(f"background:{t.border};border:none;")
        tr_hl.addWidget(sep)
        tr_hl.addSpacing(10)

        self._type_combo = _ComboBox()
        for et in ENSEMBLE_TYPES:
            self._type_combo.addItem(et, ENSEMBLE_DESC.get(et, ""))
        self._type_combo.setStyleSheet(_combo_ss())
        tr_hl.addWidget(self._type_combo, 1)

        self._type_arrow = _ExpandArrow()
        tr_hl.addWidget(self._type_arrow)
        self._type_combo.popupOpened.connect(lambda: self._type_arrow.set_down(True))
        self._type_combo.popupClosed.connect(lambda: self._type_arrow.set_down(False))
        cfg.addWidget(type_row)

        out_row = QFrame()
        out_row.setObjectName("cfgRow")
        out_row.setFixedHeight(ROW_H)
        out_row.setStyleSheet(_row_ss())
        or_hl = QHBoxLayout(out_row)
        or_hl.setContentsMargins(14, 0, 14, 0)
        or_hl.setSpacing(0)

        or_ic = QLabel("\u2193")
        or_ic.setStyleSheet(_icon_ss())
        or_ic.setFixedWidth(28)
        or_ic.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        or_hl.addWidget(or_ic)

        self._out_lb = QLabel("OUTPUT")
        self._out_lb.setStyleSheet(_lbl_ss())
        self._out_lb.setFixedWidth(100)
        self._out_lb.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        or_hl.addWidget(self._out_lb)

        sep2 = QFrame()
        sep2.setFixedWidth(1); sep2.setFixedHeight(24)
        sep2.setStyleSheet(f"background:{t.border};border:none;")
        or_hl.addWidget(sep2)
        or_hl.addSpacing(10)

        self._output_edit = QLineEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("Select output file\u2026")
        self._output_edit.setStyleSheet(
            "QLineEdit{background:transparent;border:none;"
            f"font-family:'Montserrat';font-size:11px;color:{t.text_dim};padding:0;}}"
        )
        self._output_edit.setFixedHeight(ROW_H)
        or_hl.addWidget(self._output_edit, 1)

        or_btn = EllipsisButton()
        or_btn.clicked.connect(self._browse_output)
        or_hl.addWidget(or_btn)
        cfg.addWidget(out_row)

        # Options Row: Device, Format, N_FFT, Hop, Trim
        opt_row = QFrame()
        opt_row.setObjectName("cfgRow")
        opt_row.setFixedHeight(ROW_H)
        opt_row.setStyleSheet(_row_ss())
        opt_hl = QHBoxLayout(opt_row)
        opt_hl.setContentsMargins(14, 0, 14, 0)
        opt_hl.setSpacing(0)

        opt_ic = QLabel("\u2699")
        opt_ic.setStyleSheet(_icon_ss())
        opt_ic.setFixedWidth(28)
        opt_ic.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        opt_hl.addWidget(opt_ic)

        opt_lb = QLabel("OPTIONS")
        opt_lb.setStyleSheet(_lbl_ss())
        opt_lb.setFixedWidth(65)
        opt_hl.addWidget(opt_lb)

        sep_o1 = QFrame()
        sep_o1.setFixedWidth(1); sep_o1.setFixedHeight(24)
        sep_o1.setStyleSheet(f"background:{t.border};border:none;")
        opt_hl.addWidget(sep_o1)
        opt_hl.addSpacing(8)

        dev_lbl = QLabel("DEVICE:")
        dev_lbl.setStyleSheet("font-family:'Montserrat';font-size:8px;font-weight:700;color:" + t.text_dim + ";")
        opt_hl.addWidget(dev_lbl)
        opt_hl.addSpacing(4)
        self._device_combo = _ComboBox()
        self._device_combo.addItems(["Auto", "CUDA", "CPU"])
        self._device_combo.setStyleSheet(_combo_ss())
        opt_hl.addWidget(self._device_combo)

        opt_hl.addSpacing(10)
        sep_o2 = QFrame()
        sep_o2.setFixedWidth(1); sep_o2.setFixedHeight(24)
        sep_o2.setStyleSheet(f"background:{t.border};border:none;")
        opt_hl.addWidget(sep_o2)
        opt_hl.addSpacing(8)

        nfft_lbl = QLabel("N_FFT:")
        nfft_lbl.setStyleSheet("font-family:'Montserrat';font-size:8px;font-weight:700;color:" + t.text_dim + ";")
        opt_hl.addWidget(nfft_lbl)
        opt_hl.addSpacing(4)
        self._n_fft_combo = _ComboBox()
        self._n_fft_combo.addItems(["2048", "4096", "8192", "1024", "512", "256"])
        self._n_fft_combo.setStyleSheet(_combo_ss())
        opt_hl.addWidget(self._n_fft_combo)

        opt_hl.addSpacing(10)
        sep_o3 = QFrame()
        sep_o3.setFixedWidth(1); sep_o3.setFixedHeight(24)
        sep_o3.setStyleSheet(f"background:{t.border};border:none;")
        opt_hl.addWidget(sep_o3)
        opt_hl.addSpacing(8)

        hop_lbl = QLabel("HOP:")
        hop_lbl.setStyleSheet("font-family:'Montserrat';font-size:8px;font-weight:700;color:" + t.text_dim + ";")
        opt_hl.addWidget(hop_lbl)
        opt_hl.addSpacing(4)
        self._hop_combo = _ComboBox()
        self._hop_combo.addItems(["n_fft / 2", "n_fft / 4", "n_fft / 8"])
        self._hop_combo.setStyleSheet(_combo_ss())
        opt_hl.addWidget(self._hop_combo)

        opt_hl.addSpacing(10)
        sep_o4 = QFrame()
        sep_o4.setFixedWidth(1); sep_o4.setFixedHeight(24)
        sep_o4.setStyleSheet(f"background:{t.border};border:none;")
        opt_hl.addWidget(sep_o4)
        opt_hl.addSpacing(8)

        fmt_lbl = QLabel("FORMAT:")
        fmt_lbl.setStyleSheet("font-family:'Montserrat';font-size:8px;font-weight:700;color:" + t.text_dim + ";")
        opt_hl.addWidget(fmt_lbl)
        opt_hl.addSpacing(4)
        self._format_combo = _ComboBox()
        self._format_combo.addItems(["Auto", "FLAC 24-bit", "FLAC 16-bit", "WAV Float"])
        self._format_combo.setStyleSheet(_combo_ss())
        opt_hl.addWidget(self._format_combo)

        opt_hl.addSpacing(10)
        sep_o5 = QFrame()
        sep_o5.setFixedWidth(1); sep_o5.setFixedHeight(24)
        sep_o5.setStyleSheet(f"background:{t.border};border:none;")
        opt_hl.addWidget(sep_o5)
        opt_hl.addSpacing(8)

        self._trim_chk = QCheckBox("Trim shortest")
        self._trim_chk.setToolTip("Trim output to the shortest input file instead of zero-padding")
        self._trim_chk.setStyleSheet(
            f"QCheckBox{{background:transparent;spacing:4px;font-family:'Montserrat';font-size:9px;color:{t.text_dim};}}"
            f"QCheckBox::indicator{{width:13px;height:13px;border-radius:3px;border:1px solid {t.border_dim};background:{t.surface};}}"
            f"QCheckBox::indicator:hover{{border-color:{theme_manager.accent};}}"
            f"QCheckBox::indicator:checked{{background:{theme_manager.accent};border-color:{theme_manager.accent};}}"
        )
        opt_hl.addWidget(self._trim_chk)

        cfg.addWidget(opt_row)
        wl.addLayout(cfg)

        files_outer = QWidget()
        files_outer.setStyleSheet(f"background:{theme_manager.theme.bg};")
        fl = QVBoxLayout(files_outer)
        fl.setContentsMargins(0, 14, 0, 0)
        fl.setSpacing(8)

        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(10)
        self._sec_hdr_lbl = _sec_hdr("Input Files")
        hdr_row.addWidget(self._sec_hdr_lbl)
        self._sub2_lbl = QLabel("Add at least 2 audio files to ensemble")
        self._sub2_lbl.setStyleSheet(
            f"font-size:9px;color:{t.text_label};"
            "background:transparent;font-family:'Montserrat';font-style:italic;")
        hdr_row.addWidget(self._sub2_lbl)
        hdr_row.addStretch()

        self._btn_add = GlyphButton("Add File", "+", _addfile_icon_color,
                                    glyph_size=16, text_size=8, parent=self)
        self._btn_add.setFixedSize(100, 26)
        self._btn_add.setStyleSheet(
            f"QPushButton{{background:transparent;color:{t.text_muted};"
            f"border:1px solid {t.border_dim};border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:8px;"
            "}"
            f"QPushButton:hover{{color:{theme_manager.accent};border:1px solid {theme_manager._accent_glow};}}"
        )
        self._btn_add.clicked.connect(self._add_file_row)
        hdr_row.addWidget(self._btn_add)
        fl.addLayout(hdr_row)

        self._drop_zone = _DropZone()
        self._drop_zone.files_dropped.connect(self._on_files_dropped)
        fl.addWidget(self._drop_zone)

        file_hdr = QFrame()
        file_hdr.setFixedHeight(24)
        file_hdr.setStyleSheet("background:transparent;border:none;")
        fh = QHBoxLayout(file_hdr)
        fh.setContentsMargins(14, 0, 8, 0)
        fh.setSpacing(0)

        fh_ic = QLabel("")
        fh_ic.setFixedWidth(28)
        fh.addWidget(fh_ic)

        self._fh_name = QLabel("FILE NAME")
        self._fh_name.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:8px;font-weight:700;"
            f"color:{t.text_label};background:transparent;letter-spacing:1px;"
        )
        fh.addWidget(self._fh_name, 1)

        fh_sep = QFrame()
        fh_sep.setFixedWidth(1); fh_sep.setFixedHeight(16)
        fh_sep.setStyleSheet(f"background:{t.border};border:none;")
        fh.addWidget(fh_sep)
        fh.addSpacing(6)

        fh_wt = QLabel("WEIGHT")
        fh_wt.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:8px;font-weight:700;"
            f"color:{t.text_label};background:transparent;letter-spacing:1px;"
        )
        fh_wt.setFixedWidth(160)
        fh_wt.setAlignment(Qt.AlignCenter)
        fh_wt.setToolTip("0.1 = minimal influence, 10.0 = maximum influence")
        fh.addWidget(fh_wt)

        fh_sp = QLabel("")
        fh_sp.setFixedWidth(30)
        fh.addWidget(fh_sp)

        fh_rm = QLabel("")
        fh_rm.setFixedWidth(28)
        fh.addWidget(fh_rm)
        fl.addWidget(file_hdr)

        self._files_container = QWidget()
        self._files_container.setStyleSheet("background:transparent;")
        self._files_layout = QVBoxLayout(self._files_container)
        self._files_layout.setContentsMargins(0, 0, 10, 0)
        self._files_layout.setSpacing(6)
        self._files_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            f"QScrollBar:vertical{{width:4px;background:{t.scrollbar_bg};margin:0;}}"
            f"QScrollBar::handle:vertical{{background:{t.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            f"QScrollBar::handle:vertical:hover{{background:{t.scrollbar_hover};}}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,"
            "QScrollBar::sub-page:vertical{background:transparent;}"
        )
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(self._files_container)
        scroll.setFixedHeight(200)
        fl.addWidget(scroll, 1)

        wl.addWidget(files_outer, 1)

        self._workspace = workspace
        self._workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._main_split.addWidget(self._workspace, 1)

        self._guide_panel = _EnsembleGuidePanel()
        self._guide_panel._close_btn.clicked.connect(self._toggle_guide)
        self._main_split.setSpacing(UIConstants.SECTION_SPACING)
        self._guide_panel.setMaximumWidth(self._guide_target_width())
        self._guide_btn.setText("\u2715")
        self._guide_sized = False
        self._main_split.addWidget(self._guide_panel, 0)

        self._guide_anim = QPropertyAnimation(self._guide_panel, b"maximumWidth")
        self._guide_anim.setDuration(400)
        self._guide_anim.setEasingCurve(QEasingCurve.OutCubic)

        cl.addLayout(self._main_split)
        root.addWidget(content, 1)
        self._build_action_bar(root)

    def _build_action_bar(self, outer):
        t = theme_manager.theme
        bar_container = QWidget()
        bar_container.setObjectName("actionBar")
        bar_container.setStyleSheet(
            f"QWidget#actionBar{{background:{t.bg};border:none;}}"
        )
        bar_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._action_bar = bar_container

        bar = QHBoxLayout(bar_container)
        bar.setContentsMargins(32, 24, 32, 32)
        bar.setSpacing(8)

        self.btn_run = GlyphButton("Manual Ensemble", "\u25B6", _solid_icon_color,
                                   glyph_size=18, text_size=12, parent=self)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setStyleSheet(solid_button_ss())
        self.btn_run.fit_contents()
        self.btn_run.clicked.connect(self._run)
        bar.addWidget(self.btn_run)

        self.btn_stop = GlyphButton("Stop", "\u25A0", _stop_icon_color,
                                    glyph_size=16, text_size=12, parent=self)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self._update_stop_style()
        self.btn_stop.fit_contents()
        bar.addWidget(self.btn_stop)

        bar.addStretch(1)

        self._open_btn = GlyphButton("Open Output", FOLDER_GLYPH, _pause_icon_color,
                                     glyph_size=16, text_size=10, parent=self)
        self._open_btn.setCursor(Qt.PointingHandCursor)
        self._open_btn.clicked.connect(self._open_output)
        self._update_open_style()
        self._open_btn.fit_contents()
        bar.addWidget(self._open_btn)

        outer.addWidget(bar_container, 0)

    def _update_stop_style(self):
        t = theme_manager.theme
        self.btn_stop.setStyleSheet(
            "QPushButton{"
            f"background:{t.surface};color:{t.text_muted};"
            f"border:1px solid {t.border};border-radius:6px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:12px;}"
            "QPushButton:enabled{"
            f"color:{t.error};border:1px solid {t.error};}}"
            f"QPushButton:hover:enabled{{background:{t.surface_alt};}}"
            f"QPushButton:disabled{{color:{t.text_muted};}}"
        )

    def _update_open_style(self):
        t = theme_manager.theme
        self._open_btn.setStyleSheet(
            f"QPushButton{{background:{t.surface};color:{t.text_dim};"
            f"border:1px solid {t.border};"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:10px;"
            "border-radius:6px;}"
            f"QPushButton:hover{{background:{t.surface_alt};color:{t.text};}}"
        )

    def showEvent(self, event):
        super().showEvent(event)
        if self._guide_sized:
            return
        self._guide_sized = True
        if self._guide_panel.maximumWidth() > 10:
            self._guide_panel.setMaximumWidth(self._guide_target_width())

    def _guide_target_width(self):
        return max(380, min(int(max(self.width(), 0) * 0.30), 500))

    def _toggle_guide(self):
        if self._guide_anim.state() == QPropertyAnimation.Running:
            return

        is_open = self._guide_panel.maximumWidth() > 10

        if not is_open:
            self._main_split.setSpacing(UIConstants.SECTION_SPACING)
            target_width = self._guide_target_width()
            self._guide_anim.setStartValue(0)
            self._guide_anim.setEndValue(target_width)
            self._guide_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._guide_btn.setText("\u2715")
        else:
            current = self._guide_panel.maximumWidth()
            self._guide_anim.setStartValue(current)
            self._guide_anim.setEndValue(0)
            self._guide_anim.setEasingCurve(QEasingCurve.InCubic)
            self._guide_anim.finished.connect(self._on_guide_closed)

        self._guide_anim.start()

    def _on_guide_closed(self):
        self._guide_btn.setText("How Ensemble Works")
        self._main_split.setSpacing(0)
        try:
            self._guide_anim.finished.disconnect(self._on_guide_closed)
        except TypeError:
            pass

    def _on_mode_changed(self, idx):
        if idx == 1:
            self._mode = "folders"
            self._out_lb.setText("OUTPUT DIR")
            self._output_edit.setPlaceholderText("Select output directory\u2026")
            self._sec_hdr_lbl.setText("Input Folders")
            self._sub2_lbl.setText("Add at least 2 folders containing matching stem files")
            self._btn_add.set_label("Add Folder")
            self._fh_name.setText("FOLDER NAME")
            self._drop_zone._main_lbl.setText("Drag & drop model output folders here")
            self._drop_zone._sub_lbl.setText('or click "+ Add Folder" to browse')
        else:
            self._mode = "files"
            self._out_lb.setText("OUTPUT")
            self._output_edit.setPlaceholderText("Select output file\u2026")
            self._sec_hdr_lbl.setText("Input Files")
            self._sub2_lbl.setText("Add at least 2 audio files to ensemble")
            self._btn_add.set_label("Add File")
            self._fh_name.setText("FILE NAME")
            self._drop_zone._main_lbl.setText("Drag & drop audio files here")
            self._drop_zone._sub_lbl.setText('or click "+ Add File" to browse')

    def _open_output(self):
        output = self._output_edit.text().strip()
        if output:
            if os.path.isdir(output):
                path = os.path.abspath(output)
            else:
                path = os.path.dirname(os.path.abspath(output))
        else:
            path = os.getcwd()
        if os.path.isdir(path):
            os.startfile(path)

    def _on_files_dropped(self, paths):
        if not paths:
            return
        all_dirs = all(os.path.isdir(p) for p in paths)
        all_files = all(os.path.isfile(p) for p in paths)
        if all_dirs and self._mode != "folders":
            self._mode_combo.setCurrentIndex(1)
        elif all_files and self._mode != "files":
            self._mode_combo.setCurrentIndex(0)

        existing = {row.get_path() for row in self._file_rows}
        for p in paths:
            if p not in existing:
                row = _EnsembleFileRow(p)
                row.removed.connect(lambda r=row: self._remove_file_row(r))
                self._files_layout.insertWidget(self._files_layout.count() - 1, row)
                self._file_rows.append(row)
                existing.add(p)

    def _browse_output(self):
        current = self._output_edit.text().strip()
        if self._mode == "folders":
            path = QFileDialog.getExistingDirectory(
                self, "Select output folder", current or "")
            if path:
                self._output_edit.setText(path)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Select output file", current or "",
                "FLAC files (*.flac);;WAV files (*.wav);;All files (*.*)")
            if path:
                self._output_edit.setText(path)

    def _add_file_row(self):
        existing = {row.get_path() for row in self._file_rows}
        if self._mode == "folders":
            path = QFileDialog.getExistingDirectory(self, "Select model output folder", "")
            if path and path not in existing:
                row = _EnsembleFileRow(path)
                row.removed.connect(lambda r=row: self._remove_file_row(r))
                self._files_layout.insertWidget(self._files_layout.count() - 1, row)
                self._file_rows.append(row)
        else:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select audio files", "", AUDIO_FILTER)
            for path in paths:
                if path and path not in existing:
                    row = _EnsembleFileRow(path)
                    row.removed.connect(lambda r=row: self._remove_file_row(r))
                    self._files_layout.insertWidget(self._files_layout.count() - 1, row)
                    self._file_rows.append(row)
                    existing.add(path)

    def _remove_file_row(self, row):
        if row in self._file_rows:
            self._file_rows.remove(row)
        row.deleteLater()

    def _validate(self):
        item_lbl = "folder" if self._mode == "folders" else "file"
        if len(self._file_rows) < 2:
            return f"Please add at least 2 input {item_lbl}s."
        items = []
        weights = []
        for row in self._file_rows:
            p = row.get_path()
            if not p:
                return f"Please select all input {item_lbl}s."
            if self._mode == "folders":
                if not os.path.isdir(p):
                    return f"Folder not found: {os.path.basename(p)}"
            else:
                if not os.path.isfile(p):
                    return f"File not found: {os.path.basename(p)}"
            items.append(p)
            weights.append(str(row.get_weight()))
        output = self._output_edit.text().strip()
        if not output:
            return f"Please select an output {'folder' if self._mode == 'folders' else 'file'}."
        return None, items, weights, output

    def _run(self):
        from ui.widgets.runtime_dialog import ensure_runtime
        if not ensure_runtime(self):
            return
        result = self._validate()
        if isinstance(result, str):
            QMessageBox.warning(self, "Missing input", result)
            return
        _, items, weights, output = result

        ensemble_type = self._type_combo.currentText()

        # Options
        device_choice = self._device_combo.currentText().strip().lower()
        n_fft_val = int(self._n_fft_combo.currentText().strip())
        hop_choice = self._hop_combo.currentText().strip()
        if "2" in hop_choice:
            hop_val = n_fft_val // 2
        elif "4" in hop_choice:
            hop_val = n_fft_val // 4
        elif "8" in hop_choice:
            hop_val = n_fft_val // 8
        else:
            hop_val = n_fft_val // 2

        fmt_choice = self._format_combo.currentText().strip()
        fmt_map = {
            "FLAC 24-bit": "FLAC_24",
            "FLAC 16-bit": "FLAC_16",
            "WAV Float": "WAV_FLOAT",
        }
        output_format = fmt_map.get(fmt_choice)

        mode_flag = "--folders" if self._mode == "folders" else "--files"

        cmd = [
            get_python_exe(), os.path.join(REPO_ROOT, "ensemble.py"),
            mode_flag,
        ] + items + [
            "--weights",
        ] + weights + [
            "--type", ensemble_type,
            "--output", output,
            "--device", device_choice,
            "--n_fft", str(n_fft_val),
            "--hop_length", str(hop_val),
        ]

        if output_format:
            cmd.extend(["--output_format", output_format])

        if self._trim_chk.isChecked():
            cmd.append("--trim_to_shortest")

        self._runner = ProcessRunner(cmd, cwd=REPO_ROOT)
        self._runner.log_line.connect(self.log_output.emit)
        self._runner.finished.connect(self._on_finished)
        self.process_running.emit(True)
        self._runner.start()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self):
        if self._runner:
            self._runner.stop()

    def _on_finished(self, code):
        self.process_running.emit(False)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
