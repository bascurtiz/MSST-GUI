"""ui/widgets/ckpt_settings_dialog.py — Per-checkpoint settings dialog with animated sliders."""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSlider, QWidget, QApplication, QSpinBox,
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtGui import QColor, QPalette

from ui.theme import theme_manager
from PySide6.QtGui import QColor



def _slider_ss():
    a = theme_manager.accent
    ah = theme_manager._accent_hover
    bd = theme_manager.theme.border_dim
    return (
        "QSlider::groove:horizontal{"
        "background:" + bd + ";height:4px;border-radius:2px;}"
        "QSlider::handle:horizontal{"
        "background:" + a + ";width:14px;height:14px;margin:-5px 0;"
        "border:none;border-radius:7px;}"
        "QSlider::handle:horizontal:hover{"
        "background:" + ah + ";}"
        "QSlider::sub-page:horizontal{"
        "background:" + a + ";border-radius:2px;height:4px;}"
    )

def _value_lbl_ss():
    return (
        "font-family:'Courier New',monospace;font-size:13px;font-weight:bold;"
        "color:" + theme_manager.accent + ";background:transparent;"
    )

def _dim_label_color():
    """Color for the dialog's hint line ("Configuration overrides …").

    Both themes use the `text_dim` token: the shared `text_label` is 26%
    alpha in dark (near-invisible on the near-black dialog) and 40% in light
    (washed-out gray on white); `text_dim` (50% in dark, 62% in light) keeps
    the hint subdued but clearly readable in both.
    """
    return theme_manager.theme.text_dim


def _row_label_color():
    """Color for the small-caps slider row labels (CHUNK SIZE / OVERLAP /
    BATCH SIZE): bright like the CHECKPOINT SETTINGS title in dark mode,
    near-black in light mode."""
    return "#FFFFFF" if theme_manager.mode == "dark" else "#101318"


def _title_lbl_ss():
    return (
        "font-family:'Montserrat',sans-serif;font-size:9px;font-weight:bold;"
        f"color:{_row_label_color()};background:transparent;letter-spacing:1px;"
    )


class _TitleBar(QWidget):
    # Fixed dark strip (matches the app's dark header token) with white text,
    # identical in both themes.
    _BAR_BG = "#101318"
    _BAR_BORDER = "#2E3640"

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self._drag_pos = None
        # QWidget subclasses skip stylesheet backgrounds unless this is set —
        # without it the bar renders unpainted (black) with dark text on top.
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setStyleSheet(self._bar_ss())

        hl = QHBoxLayout(self)
        hl.setContentsMargins(20, 0, 12, 0)
        hl.setSpacing(0)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:10px;font-weight:bold;"
            "color:#FFFFFF;background:transparent;letter-spacing:1px;"
        )
        hl.addWidget(title_lbl)
        hl.addStretch()

        _err = QColor(theme_manager.theme.error)
        self._close_btn = QPushButton("✕")
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.setStyleSheet(
            "QPushButton{background:transparent;color:rgba(255,255,255,0.55);"
            "border:none;font-size:14px;border-radius:4px;}"
            "QPushButton:hover{background:rgba(" + str(_err.red()) + "," + str(_err.green()) + "," + str(_err.blue()) + ",0.25);color:" + theme_manager.theme.error + ";}"
        )
        self._close_btn.clicked.connect(parent.close)
        hl.addWidget(self._close_btn)

    def _bar_ss(self):
        return (
            "QWidget{background:" + self._BAR_BG + ";"
            "border-bottom:1px solid " + self._BAR_BORDER + ";}"
        )

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.parent().frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() == Qt.LeftButton:
            self.parent().move(e.globalPos() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

    def reapply_theme(self):
        # The strip is intentionally dark in both themes — nothing to re-apply.
        self.setStyleSheet(self._bar_ss())



class _SliderRow(QWidget):
    def __init__(self, label_text, min_val, max_val, default_val, parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)

        # Single row: title + slider + value
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self._title = QLabel(label_text.upper())
        self._title.setStyleSheet(_title_lbl_ss())
        self._title.setFixedWidth(90)
        row.addWidget(self._title)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(min_val)
        self._slider.setMaximum(max_val)
        self._slider.setValue(default_val)
        self._slider.setStyleSheet(_slider_ss())
        self._slider.setFixedHeight(28)
        self._slider.valueChanged.connect(self._on_value_changed)
        row.addWidget(self._slider, 1)

        self._value_lbl = QLabel(str(default_val))
        self._value_lbl.setStyleSheet(_value_lbl_ss())
        self._value_lbl.setFixedWidth(80)
        self._value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self._value_lbl)

        vl.addLayout(row)

        self._anim = QPropertyAnimation(self._slider, b"value")
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _on_value_changed(self, val):
        self._value_lbl.setText(str(val))

    def set_value(self, val):
        val = max(self._min, min(self._max, val))
        if val != self._slider.value():
            self._anim.setStartValue(self._slider.value())
            self._anim.setEndValue(val)
            self._anim.start()

    def get_value(self):
        return self._slider.value()

    def reapply_theme(self):
        self._slider.setStyleSheet(_slider_ss())
        self._value_lbl.setStyleSheet(_value_lbl_ss())


class _ChunkSlider(QWidget):
    def __init__(self, default_val=44100, parent=None):
        super().__init__(parent)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self._title = QLabel("CHUNK SIZE")
        self._title.setStyleSheet(_title_lbl_ss())
        self._title.setFixedWidth(90)
        row.addWidget(self._title)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(44000)
        self._slider.setMaximum(930000)
        self._slider.setValue(default_val)
        self._slider.setSingleStep(44100)
        self._slider.setPageStep(220500)
        self._slider.setStyleSheet(_slider_ss())
        self._slider.setFixedHeight(28)
        row.addWidget(self._slider, 1)

        self._spin = QSpinBox()
        self._spin.setRange(1, 9999999)
        self._spin.setValue(default_val)
        self._spin.setSingleStep(44100)
        self._spin.setFixedWidth(100)
        self._spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._spin.setStyleSheet(
            f"QSpinBox{{"
            f"font-family:'Courier New',monospace;font-size:13px;font-weight:bold;"
            f"color:{theme_manager.accent};background:{theme_manager.theme.input_bg};"
            f"border:1px solid {theme_manager.theme.border_dim};border-radius:4px;"
            f"padding:2px 4px;}}"
            f"QSpinBox::up-button, QSpinBox::down-button{{"
            f"width:0px;border:none;}}"
        )
        row.addWidget(self._spin)

        vl.addLayout(row)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.editingFinished.connect(self._on_spin_changed)

    def _on_slider_changed(self, val):
        self._spin.setValue(val)

    def _on_spin_changed(self):
        self._slider.setValue(self._spin.value())

    def set_value(self, val):
        val = max(self._slider.minimum(), min(self._slider.maximum(), val))
        self._slider.setValue(val)
        self._spin.setValue(val)

    def get_value(self):
        return self._spin.value()

    def reapply_theme(self):
        self._slider.setStyleSheet(_slider_ss())
        self._spin.setStyleSheet(
            f"QSpinBox{{"
            f"font-family:'Courier New',monospace;font-size:13px;font-weight:bold;"
            f"color:{theme_manager.accent};background:{theme_manager.theme.input_bg};"
            f"border:1px solid {theme_manager.theme.border_dim};border-radius:4px;"
            f"padding:2px 4px;}}"
            f"QSpinBox::up-button, QSpinBox::down-button{{"
            f"width:0px;border:none;}}"
        )


class CkptSettingsDialog(QDialog):
    settings_saved = Signal(str, dict)

    def __init__(self, ckpt_name, yaml_path, arch, existing_settings=None, parent=None):
        super().__init__(parent)
        self._ckpt_name = ckpt_name
        self._yaml_path = yaml_path
        self._arch = arch
        self._existing = existing_settings or {}

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(440, 340)
        self.setModal(True)

        self._build_ui()
        self._load_existing()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Custom title bar
        self._title_bar = _TitleBar("CHECKPOINT SETTINGS", self)
        root.addWidget(self._title_bar)

        # Content area
        self._content = QWidget()
        self._content.setStyleSheet("background:" + theme_manager.theme.bg + ";")
        cv = QVBoxLayout(self._content)
        cv.setContentsMargins(28, 16, 28, 20)
        cv.setSpacing(12)

        # Subtitle — regular (non-italic) dim text with the checkpoint name
        # picked out in the app's accent blue, in both themes.
        sub = QLabel(
            "Configuration overrides for "
            f"<span style=\"color:{theme_manager.accent};font-weight:bold;\">"
            + self._ckpt_name + "</span>")
        self._sub = sub
        self._sub.setTextFormat(Qt.RichText)
        self._sub.setStyleSheet(
            "font-size:9px;color:" + _dim_label_color() + ";"
            "background:transparent;font-family:'Montserrat';")
        cv.addWidget(self._sub)

        # Sliders container
        self._sliders_frame = QFrame()
        self._sliders_frame.setStyleSheet(
            "QFrame{background:" + theme_manager.theme.card + ";border:1px solid " + theme_manager.theme.border + ";"
            "border-radius:6px;}"
        )
        sliders_layout = QVBoxLayout(self._sliders_frame)
        sliders_layout.setContentsMargins(16, 14, 16, 14)
        sliders_layout.setSpacing(14)

        # Chunk size slider — custom continuous value
        chunk_default = self._existing.get("chunk_size", 44100)
        self._chunk_slider = _ChunkSlider(chunk_default)
        sliders_layout.addWidget(self._chunk_slider)

        # Overlap slider
        overlap_default = self._existing.get("overlap", 8)
        self._overlap_slider = _SliderRow("Overlap", 1, 64, overlap_default)
        sliders_layout.addWidget(self._overlap_slider)

        # Batch size slider
        batch_default = self._existing.get("batch_size", 4)
        self._batch_slider = _SliderRow("Batch Size", 1, 16, batch_default)
        sliders_layout.addWidget(self._batch_slider)

        cv.addWidget(self._sliders_frame, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 4, 0, 0)

        _err = QColor(theme_manager.theme.error)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedSize(120, 36)
        self._btn_cancel.setStyleSheet(
            "QPushButton{"
            "background:" + theme_manager.theme.surface + ";color:" + theme_manager.theme.text_muted + ";"
            "border:1px solid " + theme_manager.theme.border_dim + ";border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:10px;}"
            "QPushButton:hover{color:" + theme_manager.theme.error + ";border:1px solid rgba(" + str(_err.red()) + "," + str(_err.green()) + "," + str(_err.blue()) + ",0.40);}"
        )
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)

        btn_row.addStretch()

        self._btn_save = QPushButton("Save Settings")
        self._btn_save.setFixedSize(140, 36)
        self._btn_save.setStyleSheet(
            "QPushButton{"
            "background:" + theme_manager.accent + ";color:" + theme_manager._accent_text + ";border:none;border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:10px;}"
            "QPushButton:hover{background:" + theme_manager._accent_hover + ";}"
            "QPushButton:pressed{background:" + theme_manager.accent + ";}"
        )
        self._btn_save.clicked.connect(self._save)
        btn_row.addWidget(self._btn_save)

        cv.addLayout(btn_row)
        root.addWidget(self._content, 1)

    def _load_existing(self):
        values = {}

        if self._yaml_path and os.path.isfile(self._yaml_path):
            try:
                import yaml
                with open(self._yaml_path, "r", encoding="utf-8") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)

                inference = config.get("inference", {})
                audio = config.get("audio", {})

                cs = inference.get("chunk_size") or audio.get("chunk_size")
                if cs is None:
                    dim_t = audio.get("dim_t")
                    hop = audio.get("hop_length")
                    if dim_t is not None and hop is not None:
                        cs = (dim_t - 1) * hop
                if cs is not None:
                    values["chunk_size"] = int(cs)

                ol = inference.get("num_overlap")
                if ol is not None:
                    values["overlap"] = int(ol)

                bs = inference.get("batch_size")
                if bs is not None:
                    values["batch_size"] = int(bs)
            except Exception as e:
                print("[ckpt_settings] Failed to read YAML:", e)

        saved = self._existing
        if saved.get("chunk_size") is not None:
            values["chunk_size"] = saved["chunk_size"]
        if saved.get("overlap") is not None:
            values["overlap"] = saved["overlap"]
        if saved.get("batch_size") is not None:
            values["batch_size"] = saved["batch_size"]

        print("[ckpt_settings] YAML path:", self._yaml_path)
        print("[ckpt_settings] YAML exists:", os.path.isfile(self._yaml_path) if self._yaml_path else False)
        print("[ckpt_settings] Extracted values:", values)

        if "chunk_size" in values:
            self._chunk_slider.set_value(values["chunk_size"])
        if "overlap" in values:
            self._overlap_slider.set_value(values["overlap"])
        if "batch_size" in values:
            self._batch_slider.set_value(values["batch_size"])

    def _save(self):
        settings = {
            "chunk_size": self._chunk_slider.get_value(),
            "overlap": self._overlap_slider.get_value(),
            "batch_size": self._batch_slider.get_value(),
        }
        self.settings_saved.emit(self._ckpt_name, settings)
        self.accept()

    def reapply_theme(self):
        self._title_bar.reapply_theme()
        self._content.setStyleSheet("background:" + theme_manager.theme.bg + ";")
        self._sub.setStyleSheet(
            "font-size:9px;color:" + _dim_label_color() + ";"
            "background:transparent;font-family:'Montserrat';")
        self._sub.setText(
            "Configuration overrides for "
            f"<span style=\"color:{theme_manager.accent};font-weight:bold;\">"
            + self._ckpt_name + "</span>")
        self._sliders_frame.setStyleSheet(
            "QFrame{background:" + theme_manager.theme.card + ";border:1px solid " + theme_manager.theme.border + ";"
            "border-radius:6px;}"
        )
        self._chunk_slider.reapply_theme()
        self._overlap_slider.reapply_theme()
        self._batch_slider.reapply_theme()
        self._btn_save.setStyleSheet(
            "QPushButton{"
            "background:" + theme_manager.accent + ";color:" + theme_manager._accent_text + ";border:none;border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;"
            "font-size:10px;}"
            "QPushButton:hover{background:" + theme_manager._accent_hover + ";}"
            "QPushButton:pressed{background:" + theme_manager.accent + ";}"
        )
