"""
ui/pages/settings_page.py
Model registration panel — CKPT + YAML + architecture type → registered model list.
Files are copied to project folders during registration.
Supports local files and HuggingFace URL downloads.
"""
import os
import shutil
from datetime import datetime, timezone, timedelta
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QLineEdit, QFileDialog,
    QScrollArea, QSizePolicy, QMessageBox, QProgressBar,
    QDialog,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QPoint, QEvent
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath
from ui.theme import theme_manager, UIConstants
from backend import update_checker as uc
from ui.widgets.common import (
    PageHeader, outline_button_ss, ChevronCombo, EllipsisButton, add_button_hover,
    GlyphButton, _outline_icon_color,
)

from backend.downloader import HuggingFaceDownloader
from backend.yaml_analyzer import classify_model_type
from backend.model_manager import (
    fetch_model_index, fetch_repo_meta, fetch_tree_info, fetch_folder_tree,
    is_installed, ModelInfo, MODEL_TYPE_TO_ARCH,
)
from ui.pages.model_manager_dialog import ModelInstallDialog
from ui.pages.inference_page import _ExpandArrow, _SearchBar


class _ClickableFrame(QFrame):
    """QFrame that emits a clicked signal on a left-click anywhere on it."""
    clicked = Signal()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class _ElidedLabel(QLabel):
    """QLabel that truncates its text with an ellipsis instead of forcing the
    layout wider — keeps ultra-long model/file names inside their card."""

    def __init__(self, text="", elide=Qt.ElideRight, parent=None):
        super().__init__(parent)
        self._full = text
        self._elide = elide
        # Don't let the text's natural width push the card past the viewport.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        self.setWordWrap(False)
        self._apply()

    def setText(self, text):
        self._full = text
        self._apply()

    def text(self):
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    def _apply(self):
        w = self.width()
        if w > 0:
            super().setText(self.fontMetrics().elidedText(self._full, self._elide, w))
        else:
            super().setText(self._full)

def _rgba_str(hex_str, alpha):
    from PySide6.QtGui import QColor
    c = QColor(hex_str)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def _relative_time(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    diff = datetime.now(timezone.utc) - dt
    secs = diff.total_seconds()
    if secs < 60:
        return "Updated just now"
    if secs < 3600:
        return f"Updated {int(secs // 60)} min ago"
    if secs < 86400:
        return f"Updated {int(secs // 3600)} hr ago"
    days = int(secs // 86400)
    if days == 1:
        return "Updated 1 day ago"
    if days < 7:
        return f"Updated {days} days ago"
    if secs < 2592000:
        return f"Updated {int(secs // 604800)} wk ago"
    return f"Updated {int(secs // 2592000)} mo ago"


ARCH_TYPES = [
    "Apollo Architecture", "Bandit Architecture", "BS Roformer Architecture",
    "Demucs Architecture", "MDX23c Architecture", "MDX-Net Architecture",
    "Medley Vox Architecture", "Melband Roformer Architecture",
    "SCNet Architecture", "VR Architecture",
]

MODEL_TYPES = [
    "vocals",
    "instrumental",
    "dereverb / deecho",
    "denoise",
    "phantom centre",
    "karaoke",
    "dual target (instrumental & vocals)",
    "multi stems",
    "super resolution",
    "drums",
    "bass",
    "piano",
    "guitar",
    "wind",
    "strings",
    "percussion",
    "keys",
]

ARCH_TO_MODEL_FOLDER = {
    "MDX Architecture": "models/mdx",  # legacy label from before the MDX split
    "MDX23c Architecture": "models/mdx",
    "MDX-Net Architecture": "models/mdxnet",
    "VR Architecture": "models/vr",
    "Demucs Architecture": "models/demucs",
    "BS Roformer Architecture": "models/bs_roformer",
    "Melband Roformer Architecture": "models/melband_roformer",
    "Medley Vox Architecture": "models/medley_vox",
    "SCNet Architecture": "models/scnet",
    "Apollo Architecture": "models/apollo",
    "Bandit Architecture": "models/bandit",
}

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
from backend.paths import APP_DIR as _DATA_ROOT  # writable root for downloads




class _RadioCheck(QFrame):
    clicked = Signal(str)

    def __init__(self, text, checked=False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self._hovered = False
        self.setCursor(Qt.PointingHandCursor)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(10)

        self._circle = QLabel()
        self._circle.setFixedSize(16, 16)
        hl.addWidget(self._circle)

        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{theme_manager.theme.text_dim};font-family:'Montserrat',sans-serif;"
            f"font-size:10px;font-weight:700;letter-spacing:1px;"
            f"background:transparent;"
        )
        hl.addWidget(lbl)

        self._render()

    def _render(self):
        from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
        from PySide6.QtCore import QPointF
        pm = QPixmap(16, 16)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = 8.0, 8.0

        if self._checked:
            ba = 210 if self._hovered else 170
            ac = QColor(theme_manager.accent)
            ac.setAlpha(ba)
            p.setPen(QPen(ac, 1.5))
            ac2 = QColor(theme_manager.accent)
            ac2.setAlpha(10)
            p.setBrush(ac2)
            p.drawEllipse(QPointF(cx, cy), 7.0, 7.0)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme_manager.accent))
            p.drawEllipse(QPointF(cx, cy), 4.0, 4.0)
        else:
            ba = 70 if self._hovered else 50
            tc = QColor(theme_manager.theme.text)
            tc.setAlpha(ba)
            p.setPen(QPen(tc, 1.5))
            fa = 14 if self._hovered else 8
            tc2 = QColor(theme_manager.theme.text)
            tc2.setAlpha(fa)
            p.setBrush(tc2)
            p.drawEllipse(QPointF(cx, cy), 7.0, 7.0)

        p.end()
        self._circle.setPixmap(pm)

    def is_checked(self):
        return self._checked

    def set_checked(self, v):
        if self._checked == v:
            return
        self._checked = v
        self._render()

    def mousePressEvent(self, e):
        if not self._checked:
            self._checked = True
            self._render()
            self.clicked.emit("")
        super().mousePressEvent(e)

    def enterEvent(self, e):
        self._hovered = True
        self._render()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self._render()
        super().leaveEvent(e)


def _section_hdr(text):
    # matches the INFERENCE page's section header (bar-to-text gap, tracking)
    l = QLabel(text.upper())
    l.setStyleSheet(
        f"font-family:'Montserrat',sans-serif;font-size:{UIConstants.SEC_HDR_FONT_SIZE + 1}px;font-weight:bold;"
        f"color:{theme_manager.theme.text};background:transparent;padding-left:8px;"
        f"border-left:3px solid {theme_manager.accent};letter-spacing:1.5px;"
    )
    l.setFixedHeight(18)
    return l


class _ComboBox(QComboBox):
    popupOpened = Signal()
    popupClosed = Signal()

    def showPopup(self):
        super().showPopup()
        self.popupOpened.emit()

    def hidePopup(self):
        super().hidePopup()
        self.popupClosed.emit()


class _InputField(QFrame):
    def __init__(self, label, placeholder="", browse=False, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.input_bg};border:1px solid {theme_manager.theme.border_visible};border-radius:6px;}}"
            f"QFrame:hover{{border:1px solid {theme_manager.theme.border_dim};}}"
            f"QFrame:focus-within{{border:1px solid {theme_manager.accent};}}"
        )
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.setSpacing(10)

        self.edit = QLineEdit()
        self.edit.setReadOnly(browse)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setStyleSheet(
            f"QLineEdit{{background:transparent;border:none;color:{theme_manager.theme.text_dim};"
            f"font-family:'Montserrat';font-size:12px;}}"
            f"QLineEdit:focus{{border:none;}}"
        )
        hl.addWidget(self.edit, 1)

        if browse:
            self.btn = EllipsisButton()
            hl.addWidget(self.btn)
        else:
            self.btn = None
            # Invisible spacer the same size as the EllipsisButton keeps this
            # field's sizeHint/minimumSizeHint identical to the browse
            # variant. Otherwise the two modes' fields get different layout
            # shares when the column is squeezed, shifting every field 1px.
            self._dummy = QWidget()
            self._dummy.setStyleSheet("background:transparent;")
            self._dummy.setFixedSize(26, 26)
            hl.addWidget(self._dummy)

    def value(self):
        return self.edit.text().strip()

    def set_value(self, v):
        self.edit.setText(v)

    def clear(self):
        self.edit.clear()


class _DialogCombo(ChevronCombo):
    """The app's standard chevron combo, used in the Edit Model Type dialog
    (the native arrow is hidden by the dialog's stylesheet)."""


class _EditTypeDialog(QDialog):
    """Roomier, theme-aware replacement for QInputDialog.getItem — styled
    for light and dark mode with Montserrat and the app's blue accent."""

    def __init__(self, name, current, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Model Type")
        self.setModal(True)
        self.setMinimumWidth(400)
        t = theme_manager.theme
        self.setStyleSheet(f"QDialog{{background:{t.card};}}")

        root = QVBoxLayout(self)
        root.setContentsMargins(30, 26, 30, 26)
        root.setSpacing(18)

        lbl = QLabel(f"Select type for {name}:")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:13px;color:{t.text};"
            "background:transparent;border:none;"
        )
        root.addWidget(lbl)

        self._combo = _DialogCombo()
        self._combo.addItems(items)
        idx = items.index(current) if current in items else 0
        self._combo.setCurrentIndex(idx)
        self._combo.setMinimumHeight(42)
        self._combo.setStyleSheet(
            f"QComboBox{{background:{t.surface_alt};color:{t.text};"
            f"border:1px solid {t.border_visible};border-radius:8px;"
            "font-family:'Montserrat';font-size:13px;padding:0 30px 0 12px;}"
            f"QComboBox:hover{{border:1px solid {theme_manager.accent};}}"
            f"QComboBox:focus{{border:1px solid {theme_manager.accent};}}"
            f"QComboBox::drop-down{{width:0;border:none;}}"
            f"QComboBox::down-arrow{{width:0;height:0;border:none;}}"
            f"QComboBox QAbstractItemView{{"
            f"background:{t.card};border:1px solid {t.border_visible};"
            f"color:{t.text};selection-background-color:{theme_manager.accent};"
            f"selection-color:{theme_manager._accent_text};outline:none;}}"
            f"QComboBox QAbstractItemView::item{{padding:8px 14px;min-height:28px;}}"
            f"QComboBox QAbstractItemView::item:hover{{background:{theme_manager._accent_soft};}}"
        )
        root.addWidget(self._combo)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(38)
        cancel_btn.setStyleSheet(
            f"QPushButton{{background:{t.surface_alt};color:{t.text};"
            f"border:1px solid {t.border_dim};border-radius:8px;"
            "font-family:'Montserrat';font-weight:600;font-size:12px;padding:0 22px;}"
            f"QPushButton:hover{{background:{t.border_visible};}}"
        )
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("OK")
        ok_btn.setMinimumHeight(38)
        ok_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};"
            "border:none;border-radius:8px;"
            "font-family:'Montserrat';font-weight:600;font-size:12px;padding:0 26px;}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
        )
        ok_btn.clicked.connect(self.accept)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        root.addLayout(btn_row)

    def selected(self):
        return self._combo.currentText()


class _ModelCard(QFrame):
    remove_requested = Signal(str)
    type_changed = Signal(str, str)

    def __init__(self, name, arch, model_type, ckpt, yaml, added=None, parent=None,
                 backend_module="", custom_backend_enabled=False):
        super().__init__(parent)
        self._name = name
        self._ckpt = ckpt
        self._yaml = yaml
        self._type = model_type
        self._added = added or datetime.now().strftime("%b %d, %Y  %H:%M")
        self._backend_module = backend_module
        self._custom = custom_backend_enabled

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setStyleSheet(
            f"QFrame{{background:{theme_manager.theme.card};border:1px solid {theme_manager.theme.border_visible};border-radius:8px;}}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(3)

        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(8)

        n = QLabel(name)
        n.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:15px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;border:none;"
        )
        name_row.addWidget(n)

        if custom_backend_enabled and backend_module:
            badge = QLabel("CUSTOM")
            badge.setStyleSheet(
                f"font-family:'Montserrat';font-size:8px;font-weight:700;"
                f"color:{theme_manager.accent};background:{theme_manager._accent_soft};"
                f"padding:2px 7px;border-radius:3px;letter-spacing:0.5px;"
                f"border:1px solid {theme_manager._accent_glow};"
            )
            badge.setFixedHeight(18)
            name_row.addWidget(badge)

        name_row.addStretch()
        left.addLayout(name_row)

        a = QLabel(arch)
        a.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;color:{theme_manager.accent};background:transparent;border:none;"
        )
        left.addWidget(a)

        type_row = QHBoxLayout()
        type_row.setContentsMargins(0, 0, 0, 0)
        type_row.setSpacing(6)

        self._type_label = QLabel(model_type.capitalize())
        self._type_label.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_dim};background:transparent;border:none;"
        )
        type_row.addWidget(self._type_label)

        edit_btn = QPushButton("Edit")
        edit_btn.setMinimumHeight(18)
        edit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        edit_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px solid {theme_manager.theme.border_visible};"
            f"color:{theme_manager.theme.text_dim};font-family:'Montserrat';font-size:8px;font-weight:600;border-radius:3px;"
            f"padding:0 6px;}}"
            f"QPushButton:hover{{background:{theme_manager.accent};border-color:{theme_manager.accent};color:{theme_manager._accent_text};}}"
        )
        edit_btn.clicked.connect(self._edit_type)
        type_row.addWidget(edit_btn)
        type_row.addStretch()

        left.addLayout(type_row)

        top.addLayout(left, 1)
        top.addStretch()

        rm = QPushButton("✕")
        rm.setMinimumSize(30, 30)
        rm.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.border_visible};color:{theme_manager.theme.text_dim};border:none;"
            f"border-radius:6px;font-size:12px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.error};color:{theme_manager.theme.text};}}"
        )
        rm.clicked.connect(lambda: self.remove_requested.emit(self._name))
        top.addWidget(rm)

        root.addLayout(top)
        root.addSpacing(14)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{theme_manager.theme.border_visible};border:none;")
        root.addWidget(sep)
        root.addSpacing(14)

        meta = QHBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(56)

        def _m(lbl_txt, val_txt):
            c = QVBoxLayout()
            c.setContentsMargins(0, 0, 0, 0)
            c.setSpacing(3)
            l = QLabel(lbl_txt)
            l.setStyleSheet(
                f"font-family:'Montserrat';font-size:9px;font-weight:700;color:{theme_manager.theme.text_dim};"
                f"background:transparent;border:none;letter-spacing:1px;"
            )
            c.addWidget(l)
            v = QLabel(val_txt)
            v.setStyleSheet(
                f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_dim};"
                f"background:transparent;border:none;"
            )
            c.addWidget(v)
            meta.addLayout(c)

        _m("BACKEND", "CUSTOM" if self._custom else "OFFICIAL")
        _m("SIZE", self._get_size())
        _m("ADDED", self._added)
        meta.addStretch()
        root.addLayout(meta)

    def _edit_type(self):
        dlg = _EditTypeDialog(self._name, self._type, MODEL_TYPES, self)
        if dlg.exec() == QDialog.Accepted:
            type_val = dlg.selected()
            if type_val:
                self._type = type_val
                self._type_label.setText(type_val.capitalize())
                self.type_changed.emit(self._name, type_val)

    def _get_size(self):
        try:
            if os.path.exists(self._ckpt):
                size = os.path.getsize(self._ckpt)
                if size >= 1024 * 1024:
                    return f"{size / (1024 * 1024):.0f} MB"
                return f"{size / 1024:.0f} KB"
        except OSError:
            pass
        return "—"


class _DownloadWorker(QThread):
    progress = Signal(str, int, int)
    speed = Signal(float)
    status = Signal(str)
    finished = Signal(bool, str, dict)
    error = Signal(str)

    def __init__(self, ckpt_url, yaml_url, ckpt_dest, yaml_dest,
                 backend_url="", backend_dest=""):
        super().__init__()
        self._ckpt_url = ckpt_url
        self._yaml_url = yaml_url
        self._ckpt_dest = ckpt_dest
        self._yaml_dest = yaml_dest
        self._backend_url = backend_url
        self._backend_dest = backend_dest
        self._cancelled = False
        from backend.download_utils import _make_session
        self._session = _make_session()

    def _cleanup(self):
        for p in (self._ckpt_dest, self._yaml_dest, self._backend_dest):
            if p and os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

    def run(self):
        import threading

        def _resolve(url):
            u = url.strip() if url else ""
            return u.replace("/blob/", "/resolve/", 1) if "/blob/" in u else u

        ckpt_url = _resolve(self._ckpt_url)
        yaml_url = _resolve(self._yaml_url)
        backend_url = _resolve(self._backend_url)

        self.status.emit("Starting download...")

        from backend.download_utils import parallel_download, stream_download

        results = {}
        lock = threading.Lock()
        file_progress: dict[str, list[int]] = {}

        def _download(label, url, dest, use_parallel=False):
            with lock:
                file_progress[label] = [0, 0]

            self.status.emit(f"Downloading {label}...")

            def _progress(n, c, t):
                with lock:
                    file_progress[label][0] = c
                    if t > 0:
                        file_progress[label][1] = t
                    total_dl = sum(v[0] for v in file_progress.values())
                    total_all = sum(v[1] for v in file_progress.values())
                self.progress.emit(n, total_dl, total_all)

            if use_parallel:
                ok, msg = parallel_download(
                    url, dest,
                    progress_callback=_progress,
                    speed_callback=lambda mbps: self.speed.emit(mbps),
                    should_cancel=lambda: self._cancelled,
                    session=self._session,
                )
            else:
                ok, msg = stream_download(
                    url, dest,
                    progress_callback=_progress,
                    should_cancel=lambda: self._cancelled,
                    chunk_size=1048576,
                    timeout=(60, 60),
                    session=self._session,
                )
            with lock:
                results[label] = (ok, msg)

        # Parallel for checkpoint (large), sequential for config/backend (small)
        threads = []
        if ckpt_url and self._ckpt_dest:
            t = threading.Thread(target=_download,
                                 args=("checkpoint", ckpt_url, self._ckpt_dest, True))
            t.start()
            threads.append(t)
        if yaml_url and self._yaml_dest:
            t = threading.Thread(target=_download,
                                 args=("config", yaml_url, self._yaml_dest, False))
            t.start()
            threads.append(t)
        if backend_url and self._backend_dest:
            t = threading.Thread(target=_download,
                                 args=("backend", backend_url, self._backend_dest, False))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        self._session.close()

        if self._cancelled:
            self._cleanup()
            self.finished.emit(False, "Cancelled", {})
            return

        for label, (ok, msg) in results.items():
            if not ok:
                self._cleanup()
                self.finished.emit(False, f"{label}: {msg}", {})
                return

        file_info = {
            "ckpt_name": os.path.basename(self._ckpt_dest),
            "yaml_name": os.path.basename(self._yaml_dest),
            "ckpt_path": os.path.normpath(self._ckpt_dest),
            "yaml_path": os.path.normpath(self._yaml_dest),
        }
        self.finished.emit(True, "Download complete", file_info)

    def cancel(self):
        self._cancelled = True


class _DownloadProgressDialog(QDialog):
    def __init__(self, worker, model_name, parent=None):
        super().__init__(parent)
        self._worker = worker
        self._completed = False
        self._drag_pos = None

        self.setFixedSize(440, 210)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        # ensure the QDialog{background;...} rule actually paints
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setModal(True)

        self.setStyleSheet(f"""
            QDialog{{background:{theme_manager.theme.bg};border:1px solid {theme_manager.theme.border_dim};border-radius:{UIConstants.CARD_RADIUS_PAINT}px;}}
            QLabel{{background:transparent;border:none;}}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(0)

        self._name_lbl = QLabel(model_name)
        self._name_lbl.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:15px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;"
        )
        root.addWidget(self._name_lbl)
        root.addSpacing(8)

        self._status_lbl = QLabel("Connecting to HuggingFace...")
        self._status_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_dim};background:transparent;"
        )
        root.addWidget(self._status_lbl)
        root.addSpacing(16)

        self._bar = QProgressBar()
        self._bar.setFixedHeight(4)
        self._bar.setStyleSheet(f"""
            QProgressBar{{background:{theme_manager.theme.border};border:none;border-radius:2px;}}
            QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 {theme_manager.accent},stop:1 {theme_manager.accent});border-radius:2px;}}
        """)
        self._bar.setValue(0)
        root.addWidget(self._bar)
        root.addSpacing(8)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(0)

        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setStyleSheet(
            f"font-family:'Courier New',monospace;font-size:11px;font-weight:bold;"
            f"color:{theme_manager.accent};background:transparent;"
        )
        progress_row.addWidget(self._pct_lbl)
        progress_row.addStretch()
        self._speed_lbl = QLabel("")
        self._speed_lbl.setStyleSheet(
            f"font-family:'Courier New',monospace;font-size:11px;font-weight:bold;"
            f"color:{theme_manager.theme.text_muted};background:transparent;"
        )
        progress_row.addWidget(self._speed_lbl)
        progress_row.addSpacing(12)
        self._size_lbl = QLabel("")
        self._size_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_muted};background:transparent;"
        )
        progress_row.addWidget(self._size_lbl)
        root.addLayout(progress_row)

        root.addStretch()

        self._action_btn = QPushButton("Cancel")
        self._action_btn.setMinimumHeight(40)
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface_alt};color:{theme_manager.theme.text_dim};border:1px solid {theme_manager.theme.border_dim};"
            f"font-family:'Montserrat',sans-serif;font-weight:600;font-size:11px;"
            f"border-radius:8px;padding:0 24px;}}"
            f"QPushButton:hover{{background:{theme_manager.theme.border_visible};color:{theme_manager.theme.text};}}"
            f"QPushButton:disabled{{background:{theme_manager.theme.surface};color:{theme_manager.theme.disabled_text};border-color:transparent;}}"
        )
        self._action_btn.clicked.connect(self._on_action)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addStretch()
        btn_row.addWidget(self._action_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        worker.progress.connect(self._on_progress)
        worker.speed.connect(self._on_speed)
        worker.status.connect(self._on_status)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)

    def _on_progress(self, filename, downloaded, total):
        pct = 0
        if total > 0:
            pct = int(downloaded / total * 100)
            self._bar.setValue(pct)
            mb_dl = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self._pct_lbl.setText(f"{pct}%")
            self._size_lbl.setText(f"{mb_dl:.1f} MB / {mb_total:.1f} MB")
        else:
            mb_dl = downloaded / (1024 * 1024)
            self._pct_lbl.setText(f"{mb_dl:.1f} MB")
            self._size_lbl.setText("")

    def _on_status(self, msg):
        lower = msg.lower()
        if "connecting" in lower:
            self._status_lbl.setText("Connecting to HuggingFace...")
        elif "downloading checkpoint" in lower:
            self._status_lbl.setText("Downloading checkpoint...")
        elif "downloading config" in lower or "downloading yaml" in lower:
            self._status_lbl.setText("Downloading configuration...")
        elif "downloading backend" in lower:
            self._status_lbl.setText("Downloading backend script...")
        elif "backend script downloaded" in lower:
            pass
        elif "download complete" in lower:
            self._status_lbl.setText("Verifying download...")
        else:
            self._status_lbl.setText(msg)

    def _on_speed(self, mbps: float):
        mbs = mbps / 8  # megabits -> megabytes
        if mbs < 1:
            self._speed_lbl.setText(f"{mbs * 1024:.0f} KB/s")
        else:
            self._speed_lbl.setText(f"{mbs:.2f} MB/s")

    def _on_finished(self, success, msg, file_info):
        if success:
            self._status_lbl.setText("Verifying files...")
        else:
            self._show_final(False, msg)

    def _on_error(self, msg):
        self._show_final(False, msg)

    def _show_final(self, ok, msg):
        self._completed = True
        if ok:
            self._size_lbl.setText("Model registered successfully")
            self._bar.setValue(100)
        self._action_btn.setEnabled(True)
        self._action_btn.setText("Close")

    def on_registered(self, ok=True):
        self._status_lbl.setText("Registering model..." if ok else "Registration Failed")
        if ok:
            QTimer.singleShot(600, lambda: self._show_final(True, ""))
        else:
            self._show_final(False, "")

    def _on_action(self):
        if self._completed:
            self.accept()
        else:
            self._status_lbl.setText("Cancelling...")
            self._action_btn.setEnabled(False)
            self._worker.cancel()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()


class _MgrFetchThread(QThread):
    done = Signal(object, str, object)
    error = Signal(str)

    def run(self):
        try:
            models = fetch_model_index()
            try:
                last_modified = fetch_repo_meta()
            except Exception:
                last_modified = ""
            try:
                tree_info = fetch_tree_info()
            except Exception:
                tree_info = {}
            self.done.emit(models, last_modified, tree_info)
        except Exception as e:
            self.error.emit(str(e))


# Fetch threads outliving their page (e.g. when a theme switch rebuilds the
# settings page mid-fetch). They must not be garbage-collected or destroyed
# while running, so they are parked here until they finish.
_PENDING_FETCH_THREADS = set()


def _thread_alive(thread):
    """False once the thread's C++ object has been deleted (its retire hook
    may have fired between our checks), True otherwise."""
    try:
        thread.isRunning()
        return True
    except RuntimeError:
        return False


def _retire_fetch_thread(thread):
    _PENDING_FETCH_THREADS.discard(thread)
    thread.deleteLater()


def orphan_fetch_threads(manager):
    """Detach an about-to-be-destroyed _FolderManagerWidget from its
    in-flight fetch threads.

    QThreads must never be destroyed while running, and requests cannot be
    interrupted mid-transfer — so instead of blocking on them, running
    threads are un-parented and parked in a module-level set (their results
    simply go nowhere); the manager's own model-index thread additionally
    gets retired once it finishes. Returns the running model-index thread
    so the caller can defer the manager's deletion until it is done."""
    if manager is None:
        return None
    for th in list(getattr(manager, "_folder_fetch_threads", []) or []):
        if _thread_alive(th) and th.isRunning():
            th.setParent(None)
            _PENDING_FETCH_THREADS.add(th)
            th.finished.connect(lambda th=th: _retire_fetch_thread(th))
    fetch = getattr(manager, "_fetch_thread", None)
    if fetch is not None and _thread_alive(fetch) and fetch.isRunning():
        for sig, slot in ((fetch.done, manager._on_loaded),
                          (fetch.error, manager._on_error)):
            try:
                sig.disconnect(slot)
            except Exception:
                pass
        _PENDING_FETCH_THREADS.add(fetch)
        fetch.finished.connect(
            lambda f=fetch: _retire_fetch_thread(f))
        return fetch
    return None


class _FolderFetchThread(QThread):
    """Fetches one folder's file dates/sizes off the UI thread."""
    done = Signal(str, object, object)
    failed = Signal(str)

    def __init__(self, folder_key, parent=None):
        super().__init__(parent)
        self._key = folder_key

    def run(self):
        try:
            dates, sizes = fetch_folder_tree(self._key)
            self.done.emit(self._key, dates, sizes)
        except Exception:
            self.failed.emit(self._key)


def _folder_arch_color(folder_key):
    """Accent color for a model-manager folder icon, using the same
    per-architecture dot colors as the rest of the GUI (arch_dot_* tokens).
    Returns None for folders that match no known architecture, which the
    icon renders with the neutral theme tint."""
    t = theme_manager.theme
    key = (folder_key or "").lower().strip()
    direct = getattr(t, f"arch_dot_{key}", None)
    if direct:
        return direct
    arch = MODEL_TYPE_TO_ARCH.get(key)
    if arch:
        c = getattr(t, f"arch_dot_{arch.lower().split()[0]}", None)
        if c:
            return c
    # Substring fallbacks for folder names that map to no known model_type.
    for pat, dot in (
        ("mel_band", "melband"), ("melband", "melband"),
        ("htdemucs", "demucs"), ("demucs", "demucs"),
        ("bs_roformer", "bs"), ("roformer", "bs"),
        ("scnet", "scnet"), ("apollo", "apollo"),
        ("bandit", "bandit"), ("mdx", "mdx"),
    ):
        if pat in key:
            return getattr(t, f"arch_dot_{dot}", None)
    return None


class _ColorDot(QWidget):
    """Small solid dot in front of a model-manager folder, color-coded by
    architecture (same arch_dot_* palette as the rest of the GUI)."""

    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(8, 8)

    def paintEvent(self, event):
        if not self._color:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._color))
        p.drawEllipse(self.rect())
        p.end()


class _FolderIcon(QWidget):
    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(18, 18)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._color:
            c = QColor(self._color)
            c.setAlpha(170)
        else:
            c = QColor(theme_manager.theme.text)
            c.setAlpha(70)
        pen = QPen(c, 1.3)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        w = self.width()
        h = self.height()
        path = QPainterPath()
        path.moveTo(w * 0.1, h * 0.3)
        path.lineTo(w * 0.1, h * 0.9)
        path.lineTo(w * 0.9, h * 0.9)
        path.lineTo(w * 0.9, h * 0.3)
        path.lineTo(w * 0.55, h * 0.3)
        path.lineTo(w * 0.45, h * 0.15)
        path.lineTo(w * 0.1, h * 0.15)
        path.closeSubpath()
        p.drawPath(path)
        p.end()


class _FolderManagerWidget(QWidget):
    model_installed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: list[ModelInfo] = []
        self._model_type_map: dict[str, list[ModelInfo]] = {}
        self._folder_order: list[str] = []
        self._loading_folders: set = set()
        self._folder_fetch_threads: list = []
        self._dot_phase = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.setInterval(400)
        self._dot_timer.timeout.connect(self._tick_dots)

        self._expanded: set = set()
        self._folder_meta: dict = {}
        self._folder_tree_dates: dict[str, str] = {}
        self._folder_file_dates: dict[str, str] = {}
        self._folder_file_sizes: dict[str, int] = {}
        self._show_new_badge: dict[str, bool] = {}

        self.setStyleSheet("background:transparent;")
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(10)

        # Loading
        self._load_lbl = QLabel("Loading models from HuggingFace...")
        self._load_lbl.setAlignment(Qt.AlignCenter)
        self._load_lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:12px;color:{theme_manager.theme.text_dim};background:transparent;border:none;"
        )
        lo.addWidget(self._load_lbl)

        # Scrollable list
        self._scroll_widget = QWidget()
        self._scroll_widget.setStyleSheet("background:transparent;")
        self._list_layout = QVBoxLayout(self._scroll_widget)
        self._list_layout.setContentsMargins(0, 0, 8, 26)  # air under last row
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea{{background:transparent;border:none;}}
            QScrollBar:vertical{{width:4px;background:transparent;margin:0;}}
            QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};
            border-radius:2px;min-height:30px;}}
            QScrollBar::handle:vertical:hover{{background:{theme_manager.theme.border_dim};}}
            QScrollBar::add-line:vertical{{height:0;}}
            QScrollBar::sub-line:vertical{{height:0;}}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical{{background:transparent;}}
        """)
        scroll.setWidget(self._scroll_widget)
        lo.addWidget(scroll, 1)

        # Fetch
        self._fetch_thread = _MgrFetchThread()
        self._fetch_thread.done.connect(self._on_loaded)
        self._fetch_thread.error.connect(self._on_error)
        self._fetch_thread.start()

    # ──── Data ────

    def _on_loaded(self, models, last_modified, tree_info):
        self._fetch_thread = None  # the thread retires itself when done
        self._load_lbl.setVisible(False)
        self._models = models
        self._folder_tree_dates = tree_info

        # ── NEW badge tracking ──
        self._show_new_badge.clear()
        from backend import settings as settings_store
        data = settings_store.load()
        tracker = data.setdefault("_model_tracker", {})
        snapshot = data.get("_model_snapshot_keys", [])
        snapshot_date = data.get("_model_snapshot_date", "")

        current_keys = {m.key for m in models}

        refresh_snapshot = False
        if snapshot_date:
            try:
                sd = datetime.fromisoformat(snapshot_date.replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - sd).total_seconds() >= 14 * 86400:
                    refresh_snapshot = True
            except Exception:
                refresh_snapshot = True
        else:
            refresh_snapshot = True

        if refresh_snapshot or not snapshot:
            data["_model_snapshot_keys"] = list(current_keys)
            data["_model_snapshot_date"] = last_modified or datetime.now(timezone.utc).isoformat()
            # First run or expired: use lastModified as first_seen for all models
            ref_date = last_modified or datetime.now(timezone.utc).isoformat()
            for key in current_keys:
                tracker[key] = ref_date
            tracker = {k: v for k, v in tracker.items() if k in current_keys}
            data["_model_tracker"] = tracker
            settings_store.save(data)
        else:
            snapshot_set = set(snapshot)
            new_keys = current_keys - snapshot_set
            now_iso = datetime.now(timezone.utc).isoformat()
            for key in new_keys:
                tracker[key] = now_iso
            # Isi first_seen untuk model yg belum punya tracker (migrasi data lama)
            ref_date = last_modified or now_iso
            for key in current_keys:
                if key not in tracker:
                    tracker[key] = ref_date
            tracker = {k: v for k, v in tracker.items() if k in current_keys}
            data["_model_tracker"] = tracker
            data["_model_snapshot_keys"] = list(current_keys)
            settings_store.save(data)

        # Compute NEW badges for all tracked models
        now = datetime.now(timezone.utc)
        for key, first_seen_str in tracker.items():
            try:
                fs = datetime.fromisoformat(first_seen_str)
                self._show_new_badge[key] = (now - fs).total_seconds() < 14 * 86400
            except Exception:
                self._show_new_badge[key] = False

        # ── populate type map ──
        self._model_type_map.clear()
        self._folder_order.clear()
        for m in models:
            key = m.model_type
            if key not in self._model_type_map:
                self._model_type_map[key] = []
                self._folder_order.append(key)
            self._model_type_map[key].append(m)
        self._folder_order.sort()  # alphabetical, like the MODEL LIBRARY
        self._render()

    def _on_error(self, msg):
        self._fetch_thread = None  # the thread retires itself when done
        self._load_lbl.setText(f"Failed: {msg}")

    # ──── Expand / collapse ────

    def _toggle_folder(self, folder_key):
        if folder_key in self._expanded:
            self._expanded.discard(folder_key)
            self._render()
            return
        self._expanded.add(folder_key)
        if folder_key in self._folder_meta:
            self._render()
            return
        # First expand: show the loading overlay and fetch in the background
        self._loading_folders.add(folder_key)
        self._dot_timer.start()
        self._render()
        thread = _FolderFetchThread(folder_key, self)
        thread.done.connect(self._on_folder_fetched)
        thread.failed.connect(self._on_folder_fetch_error)
        self._folder_fetch_threads.append(thread)
        thread.start()

    def _on_folder_fetched(self, folder_key, dates, sizes):
        self._finish_folder_fetch(folder_key, dates, sizes)

    def _on_folder_fetch_error(self, folder_key):
        self._finish_folder_fetch(folder_key, {}, {})

    def _finish_folder_fetch(self, folder_key, dates, sizes):
        self._folder_meta[folder_key] = (dates, sizes)
        self._folder_file_dates.update(dates)
        self._folder_file_sizes.update(sizes)
        self._loading_folders.discard(folder_key)
        if not self._loading_folders:
            self._dot_timer.stop()
            self._dot_phase = 0
        self._render()

    def _tick_dots(self):
        self._dot_phase = (self._dot_phase + 1) % 4
        text = "Loading" + "." * self._dot_phase
        for lbl in self.findChildren(QLabel):
            if lbl.objectName() == "loadDots":
                lbl.setText(text)

    def eventFilter(self, obj, event):
        # keep the loading overlay sized to its folder card
        if event.type() == QEvent.Resize:
            for child in obj.findChildren(QFrame):
                if child.objectName() == "loadOverlay":
                    child.setGeometry(0, 0, obj.width(), obj.height())
        return super().eventFilter(obj, event)

    def _clear_list(self):
        for i in range(self._list_layout.count()):
            item = self._list_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self._list_layout.invalidate()

    def _show_loading(self):
        lbl = QLabel("Loading models...")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_muted};background:transparent;border:none;"
        )
        self._list_layout.addWidget(lbl)

    # ──── Search ────

    def set_search_text(self, text):
        """Filter folders by name (driven by the page-level search field,
        which sits on the same row as the REGISTER MODEL header)."""
        self._render(text.lower().strip())

    # ──── Render ────

    def _clear(self):
        for i in reversed(range(self._list_layout.count())):
            item = self._list_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

    def _render(self, search_term=""):
        self._clear()
        self._render_root(search_term)

    def _render_root(self, search_term=""):
        folders = self._folder_order[:]
        if search_term:
            folders = [f for f in folders if search_term in f.lower()]

        if not folders:
            lbl = QLabel("No folders found")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_muted};background:transparent;border:none;"
            )
            self._list_layout.addWidget(lbl)
            return

        for fk in folders:
            entry_count = len(self._model_type_map[fk])
            installed_count = sum(1 for m in self._model_type_map[fk] if is_installed(m))

            card = _ClickableFrame()
            card.setStyleSheet(
                f"QFrame{{background:{theme_manager.theme.input_bg};"
                f"border:1px solid {theme_manager.theme.border_visible};border-radius:8px;}}"
                f"QFrame:hover{{border:1px solid {theme_manager.accent};}}"
            )
            card.setCursor(Qt.PointingHandCursor)
            card.setFixedHeight(60)
            card.clicked.connect(lambda x=fk: self._toggle_folder(x))
            clo = QHBoxLayout(card)
            clo.setContentsMargins(18, 0, 14, 0)
            clo.setSpacing(12)

            clo.addWidget(_ColorDot(_folder_arch_color(fk)))
            fi = _FolderIcon()  # neutral gray; the dot carries the arch color
            clo.addWidget(fi)

            label = _ElidedLabel(f"{fk}/")
            label.setStyleSheet(
                f"font-family:'Montserrat';font-size:13px;font-weight:700;color:{theme_manager.theme.text};background:transparent;border:none;"
            )
            clo.addWidget(label, 1)

            info_parts = []
            folder_date = self._folder_tree_dates.get(fk, "")
            if folder_date:
                info_parts.append(_relative_time(folder_date))
            info_parts.append(f"{installed_count}/{entry_count} installed")
            info_lbl = QLabel(" • ".join(info_parts))
            info_lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:10px;color:{theme_manager.theme.text_muted};background:transparent;border:none;"
            )
            clo.addWidget(info_lbl)

            arrow = _ExpandArrow()
            if fk in self._expanded:
                arrow.set_angle(90.0)
            arrow.clicked.connect(lambda x=fk: self._toggle_folder(x))
            clo.addWidget(arrow, 0, Qt.AlignVCenter)

            self._list_layout.addWidget(card)

            if fk in self._loading_folders:
                overlay = QFrame(card)
                overlay.setObjectName("loadOverlay")
                overlay.setAttribute(Qt.WA_StyledBackground, True)
                overlay.setStyleSheet(
                    f"QFrame#loadOverlay{{"
                    f"background:{_rgba_str(theme_manager.theme.surface, 215)};"
                    f"border-radius:8px;}}"
                )
                ol = QHBoxLayout(overlay)
                ol.setContentsMargins(0, 0, 0, 0)
                dots = QLabel("Loading" + "." * self._dot_phase)
                dots.setObjectName("loadDots")
                dots.setAlignment(Qt.AlignCenter)
                dots.setStyleSheet(
                    "font-family:'Montserrat',sans-serif;font-size:11px;font-weight:700;"
                    f"color:{theme_manager.theme.text_dim};"
                    "background:transparent;border:none;letter-spacing:1px;"
                )
                ol.addWidget(dots)
                overlay.setGeometry(0, 0, card.width(), card.height())
                overlay.show()
                overlay.raise_()
                card.installEventFilter(self)

            if fk in self._expanded:
                self._list_layout.addWidget(self._render_inside(fk, search_term))

    def _model_new_and_date(self, info, folder_key=""):
        model_date = ""
        for url in (info.checkpoint_url, info.config_url, info.backend_script_url):
            if not url:
                continue
            fname = url.split("/")[-1].split("?")[0]
            dstr = self._folder_file_dates.get(fname)
            if dstr and dstr > model_date:
                model_date = dstr
        if not model_date:
            model_date = self._folder_tree_dates.get(folder_key, "")
        is_new = False
        if model_date:
            try:
                fd = datetime.fromisoformat(model_date.replace("Z", "+00:00"))
                is_new = fd >= datetime.now(timezone.utc) - timedelta(days=14)
            except Exception:
                pass
        return (not is_new, model_date)

    def _model_sort_ts(self, info):
        """Newest-first sort key: the latest last-commit timestamp among the
        model's files. Returns 0.0 when no date is known, which keeps undated
        models at the bottom in manifest order."""
        ts = 0.0
        for url in (info.checkpoint_url, info.config_url, info.backend_script_url):
            if not url:
                continue
            fname = url.split("/")[-1].split("?")[0]
            dstr = self._folder_file_dates.get(fname)
            if not dstr:
                continue
            try:
                ts = max(ts, datetime.fromisoformat(
                    dstr.replace("Z", "+00:00")).timestamp())
            except Exception:
                pass
        return ts

    def _render_inside(self, folder_key, search_term=""):
        models = self._model_type_map.get(folder_key, [])
        if search_term:
            models = [m for m in models
                      if search_term in m.full_name.lower()
                      or search_term in m.key.lower()]
        models = sorted(models, key=lambda m: self._model_sort_ts(m), reverse=True)

        container = QWidget()
        container.setStyleSheet("background:transparent;")
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(24, 0, 0, 0)
        hbox.setSpacing(0)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        if not models:
            lbl = QLabel("No models found")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:11px;color:{theme_manager.theme.text_muted};background:transparent;border:none;"
            )
            col.addWidget(lbl)
            hbox.addLayout(col, 1)
            return container

        for info in models:
            installed = is_installed(info)
            card = QFrame()
            card.setStyleSheet(
                f"QFrame{{background:{theme_manager.theme.input_bg};"
                f"border:1px solid {theme_manager.theme.border_visible};border-radius:8px;}}"
            )
            clo = QVBoxLayout(card)
            clo.setContentsMargins(16, 12, 16, 12)
            clo.setSpacing(6)

            # Main row: left (info) + right (install centered)
            main_row = QHBoxLayout()
            main_row.setContentsMargins(0, 0, 0, 0)
            main_row.setSpacing(8)

            left_col = QVBoxLayout()
            left_col.setContentsMargins(0, 0, 0, 0)
            left_col.setSpacing(6)

            name_lbl = _ElidedLabel(info.full_name)
            name_lbl.setStyleSheet(
                f"font-family:'Montserrat';font-size:12px;font-weight:700;"
                f"color:{theme_manager.theme.text};background:transparent;border:none;"
            )
            left_col.addWidget(name_lbl)

            # Files
            ckpt_name = info.checkpoint_url.split("/")[-1].split("?")[0]
            yaml_name = info.config_url.split("/")[-1].split("?")[0]

            ckpt_row = _ElidedLabel(f"┣━  {ckpt_name}", elide=Qt.ElideMiddle)
            ckpt_row.setStyleSheet(
                f"font-family:'Courier New',monospace;font-size:11px;"
                f"color:{theme_manager.theme.text_dim};background:transparent;border:none;"
            )
            left_col.addWidget(ckpt_row)

            yaml_row = _ElidedLabel(f"┣━  {yaml_name}", elide=Qt.ElideMiddle)
            yaml_row.setStyleSheet(
                f"font-family:'Courier New',monospace;font-size:11px;"
                f"color:{theme_manager.theme.text_dim};background:transparent;border:none;"
            )
            left_col.addWidget(yaml_row)

            if info.backend_script_url:
                py_name = info.backend_script_url.split("/")[-1].split("?")[0]
                py_row = _ElidedLabel(f"┗━  {py_name}  (custom backend)", elide=Qt.ElideMiddle)
                py_row.setStyleSheet(
                    f"font-family:'Courier New',monospace;font-size:11px;"
                    f"color:{theme_manager.theme.text_muted};background:transparent;border:none;"
                )
                left_col.addWidget(py_row)

            # Size
            ckpt_name = info.checkpoint_url.split("/")[-1].split("?")[0]
            file_size = self._folder_file_sizes.get(ckpt_name, 0)
            if not file_size:
                file_size = info.file_size
            size_lbl = None
            if file_size:
                sz = file_size
                size_text = f"{sz / 1024 / 1024:.1f} MB" if sz >= 1024 * 1024 else f"{sz / 1024:.0f} KB"
                size_lbl = QLabel(size_text)

            if size_lbl:
                size_lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:10px;"
                    f"color:{theme_manager.theme.text_muted};background:transparent;border:none;"
                )

            not_new, model_date = self._model_new_and_date(info, folder_key)
            updated_row = QHBoxLayout()
            updated_row.setContentsMargins(0, 0, 0, 0)
            updated_row.setSpacing(6)

            if model_date:
                updated_lbl = QLabel(_relative_time(model_date))
                updated_lbl.setStyleSheet(
                    f"font-family:'Montserrat';font-size:10px;"
                    f"color:{theme_manager.theme.text_muted};background:transparent;border:none;"
                )
                updated_row.addWidget(updated_lbl)

            if size_lbl:
                updated_row.addWidget(size_lbl)

            if not not_new:
                new_badge = QLabel("NEW")
                new_badge.setFixedHeight(18)
                # Complementary (inverted) of the accent blue so the badge
                # pops against the blue UI; text chosen by luminance so it
                # stays readable on whatever the inverted color is.
                _ac = QColor(theme_manager.accent)
                _inv = QColor(255 - _ac.red(), 255 - _ac.green(), 255 - _ac.blue())
                _txt = "#14161A" if _inv.lightnessF() > 0.5 else "#E8EDF3"
                new_badge.setStyleSheet(
                    f"background:{_inv.name()};"
                    f"color:{_txt};"
                    f"font-weight:bold;font-size:9px;border-radius:3px;"
                    f"padding:0 6px;border:1px solid {_inv.name()};"
                )
                updated_row.addWidget(new_badge)

            if updated_row.count():
                updated_row.addStretch()
                left_col.addLayout(updated_row)

            main_row.addLayout(left_col, 1)

            if installed:
                inst_btn = QPushButton("✓ Installed")
                inst_btn.setFixedHeight(30)
                inst_btn.setEnabled(False)
                inst_btn.setStyleSheet(
                    f"QPushButton{{background:{theme_manager.theme.disabled_bg};"
                    f"color:{theme_manager.theme.text_muted};border:none;"
                    f"font-weight:600;font-size:9px;border-radius:5px;padding:0 14px;}}"
                )
            else:
                inst_btn = QPushButton("Install")
                inst_btn.setFixedHeight(30)
                inst_btn.setStyleSheet(
                    f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};border:none;"
                    f"font-family:'Montserrat',sans-serif;font-weight:600;font-size:9px;"
                    f"border-radius:5px;padding:0 14px;}}"
                    f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
                )
                inst_btn.clicked.connect(lambda _, x=info: self._install(x))
            main_row.addWidget(inst_btn, 0, Qt.AlignVCenter)

            clo.addLayout(main_row)

            col.addWidget(card)

        hbox.addLayout(col, 1)
        return container

    def _install(self, info):
        ckpt_name = info.checkpoint_url.split("/")[-1].split("?")[0]
        file_size = self._folder_file_sizes.get(ckpt_name, 0)
        if file_size:
            info.file_size = file_size
        dialog = ModelInstallDialog(info, self)
        if dialog.exec() == ModelInstallDialog.Accepted:
            self._render()
            self.model_installed.emit()


class SettingsPage(QWidget):
    model_registered = Signal(dict)
    model_removed = Signal(str)
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsPage")
        # Object-name scoped so the background doesn't cascade into child
        # dialogs (QMessageBox etc.) and overwrite their button styles.
        self.setStyleSheet(f"#settingsPage{{background:{theme_manager.theme.bg};}}")
        self._registered = []
        self._download_worker = None
        self._download_mode = "manager"
        self._pending_backend_module = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 68)
        # 32px gap under the page header (the columns then carry the usual
        # +1px, so the content lands at 168px — the exact same y as the
        # INFERENCE page's MODEL LIBRARY row).
        root.setSpacing(32)

        # Updates — docked at the right of the page header, on the headliner
        # line (same pattern as the LOG button in CONSOLE).
        self._update_btn = QPushButton("Check For Updates")
        self._update_btn.setCursor(Qt.PointingHandCursor)
        self._update_btn.setFixedHeight(30)
        self._update_btn.setStyleSheet(
            f"QPushButton{{background:{theme_manager.theme.surface};"
            f"color:{theme_manager.theme.text_dim};"
            f"border:1px solid {theme_manager.theme.border_dim};border-radius:4px;"
            "font-family:'Montserrat',sans-serif;font-weight:600;font-size:9px;"
            "padding:0 14px;}"
            + add_button_hover()
            + f"QPushButton:disabled{{color:{theme_manager.theme.disabled_text};}}")
        self._update_btn.clicked.connect(self._check_updates)
        self._update_status = QLabel(
            f"Current version {uc.app_version()}")
        self._update_status.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:10px;"
            f"color:{theme_manager.theme.text_muted};background:transparent;")
        update_w = QWidget()
        update_w.setStyleSheet("background:transparent;")
        uw = QHBoxLayout(update_w)
        # no top margin: a margin here would make the header taller than the
        # other pages' headers (dead space under the subtitle), pushing the
        # section below further down. Centered like the LOG button in CONSOLE.
        uw.setContentsMargins(0, 0, 0, 0)
        uw.setSpacing(12)
        uw.addWidget(self._update_btn)
        uw.addWidget(self._update_status)

        header = PageHeader(
            "SETTINGS",
            "ADD, REGISTER, AND CONFIGURE MODELS",
            highlight="MODELS",
        )
        header.add_extra(update_w)
        root.addWidget(header)

        main = QHBoxLayout()
        main.setSpacing(48)
        main.setContentsMargins(0, 0, 0, 0)

        left = QWidget()
        left.setStyleSheet("background:transparent;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        # 20px instead of 24: the column content is ~33px taller than the
        # available page height, which made Qt squeeze and round the group
        # heights (1px mode differences). With fixed 65px groups this fits.
        ll.setSpacing(20)

        # Search folders sits on the same row as the REGISTER MODEL header,
        # right-aligned and styled like the MODEL LIBRARY search bar.
        reg_hdr = QHBoxLayout()
        # right margin matches the folder cards' 8px margin below, so the
        # search box right edge aligns with the cards' right edge
        reg_hdr.setContentsMargins(0, 0, 8, 0)
        reg_hdr.setSpacing(8)
        # centered against the search box, exactly like the INFERENCE page's
        # MODEL LIBRARY header row
        reg_hdr.addWidget(_section_hdr("Register Model"))
        reg_hdr.addStretch()
        self._folder_search = _SearchBar("Search folders\u2026")
        self._folder_search.setMaximumWidth(155)
        self._folder_search.setVisible(False)  # shown in MODEL MANAGER mode
        reg_hdr.addWidget(self._folder_search)

        # Invisible placeholder that keeps the row's height when the search
        # field is hidden (URL / LOCAL FILES modes), so everything below
        # stays on the same height across modes.
        self._folder_search_ph = QWidget()
        self._folder_search_ph.setFixedSize(155, 32)
        self._folder_search_ph.setVisible(False)
        reg_hdr.addWidget(self._folder_search_ph)
        ll.addLayout(reg_hdr)
        ll.addSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(24)

        self._mode_manager = _RadioCheck("MODEL MANAGER", checked=True)
        self._mode_url = _RadioCheck("DOWNLOAD FROM URL")
        self._mode_local = _RadioCheck("LOCAL FILES")

        self._mode_manager.clicked.connect(lambda: self._set_mode("manager"))
        self._mode_url.clicked.connect(lambda: self._set_mode("url"))
        self._mode_local.clicked.connect(lambda: self._set_mode("local"))

        mode_row.addWidget(self._mode_manager)
        mode_row.addWidget(self._mode_url)
        mode_row.addWidget(self._mode_local)
        mode_row.addStretch()
        ll.addLayout(mode_row)

        def _field_group(label, field):
            # Fixed-height container: the column squeezes/rounds flexible
            # items when the content is taller than the page, which shifted
            # the LOCAL FILES vs DOWNLOAD FROM URL fields 1px apart. A fixed
            # group can't be redistributed, so every mode lines up exactly.
            w = QWidget()
            w.setStyleSheet("background:transparent;")
            w.setFixedHeight(65)
            g = QVBoxLayout(w)
            g.setContentsMargins(0, 0, 0, 0)
            g.setSpacing(6)
            g.addWidget(label)
            field.setFixedHeight(48)
            g.addWidget(field)
            return w

        self._ckpt_label = QLabel("CHECKPOINT")
        self._ckpt_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )

        self._ckpt = _InputField("Checkpoint", "Select checkpoint file (.ckpt)", browse=True)
        self._ckpt.btn.clicked.connect(lambda: self._browse(
            self._ckpt, "Checkpoint (*.ckpt *.bin *.th *.chpt);;All (*.*)"
        ))
        self._grp_ckpt = _field_group(self._ckpt_label, self._ckpt)
        ll.addWidget(self._grp_ckpt)

        self._yaml_label = QLabel("CONFIG YAML")
        self._yaml_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )

        self._yaml = _InputField("Config YAML", "Select config yaml/json file", browse=True)
        self._yaml.btn.clicked.connect(lambda: self._browse(
            self._yaml, "Config (*.yaml *.yml *.json);;All (*.*)"
        ))
        self._grp_yaml = _field_group(self._yaml_label, self._yaml)
        ll.addWidget(self._grp_yaml)

        self._ckpt_url_label = QLabel("CHECKPOINT")
        self._ckpt_url_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )

        self._ckpt_url = _InputField("Checkpoint", "Paste HuggingFace checkpoint URL...")
        self._grp_ckpt_url = _field_group(self._ckpt_url_label, self._ckpt_url)
        ll.addWidget(self._grp_ckpt_url)

        self._yaml_url_label = QLabel("CONFIG YAML")
        self._yaml_url_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )

        self._yaml_url = _InputField("Config File", "Paste HuggingFace config URL (.yaml .json)...")
        self._grp_yaml_url = _field_group(self._yaml_url_label, self._yaml_url)
        ll.addWidget(self._grp_yaml_url)

        self._arch_label = QLabel("ARCHITECTURE")
        self._arch_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )

        self._arch_combo = _ComboBox()
        self._arch_combo.addItems(ARCH_TYPES)
        self._arch_combo.setMinimumHeight(48)
        self._arch_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._arch_combo.setStyleSheet(
            f"QComboBox{{background:transparent;border:none;"
            f"color:{theme_manager.theme.text_dim};font-family:'Montserrat';font-size:12px;padding:0 14px;}}"
            f"QComboBox::drop-down{{width:0;border:none;}}"
            f"QComboBox::down-arrow{{width:0;height:0;border:none;}}"
            f"QComboBox QAbstractItemView{{background:{theme_manager.theme.input_bg};border:1px solid {theme_manager.theme.border_visible};"
            f"color:{theme_manager.theme.text};selection-background-color:{theme_manager.accent};selection-color:{theme_manager._accent_text};}}"
        )
        self._arch_arrow = _ExpandArrow()
        self._arch_combo.popupOpened.connect(lambda: self._arch_arrow.set_down(True))
        self._arch_combo.popupClosed.connect(lambda: self._arch_arrow.set_down(False))
        self._arch_w = QWidget()
        self._arch_w.setStyleSheet(
            f"background:{theme_manager.theme.input_bg};"
            f"border:1px solid {theme_manager.theme.border_visible};"
            f"border-radius:6px;"
        )
        self._arch_w.setMinimumHeight(48)
        arch_h = QHBoxLayout(self._arch_w)
        arch_h.setContentsMargins(0, 0, 14, 0)
        arch_h.setSpacing(10)
        arch_h.addWidget(self._arch_combo, 1)
        arch_h.addWidget(self._arch_arrow)
        self._grp_arch = _field_group(self._arch_label, self._arch_w)
        ll.addWidget(self._grp_arch)

        self._type_label = QLabel("TYPE")
        self._type_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )

        self._type_combo = _ComboBox()
        self._type_combo.addItems(MODEL_TYPES)
        self._type_combo.setMinimumHeight(48)
        self._type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._type_combo.setStyleSheet(
            f"QComboBox{{background:transparent;border:none;"
            f"color:{theme_manager.theme.text_dim};font-family:'Montserrat';font-size:12px;padding:0 14px;}}"
            f"QComboBox::drop-down{{width:0;border:none;}}"
            f"QComboBox::down-arrow{{width:0;height:0;border:none;}}"
            f"QComboBox QAbstractItemView{{background:{theme_manager.theme.input_bg};border:1px solid {theme_manager.theme.border_visible};"
            f"color:{theme_manager.theme.text};selection-background-color:{theme_manager.accent};selection-color:{theme_manager._accent_text};}}"
        )
        self._type_arrow = _ExpandArrow()
        self._type_combo.popupOpened.connect(lambda: self._type_arrow.set_down(True))
        self._type_combo.popupClosed.connect(lambda: self._type_arrow.set_down(False))
        self._type_w = QWidget()
        self._type_w.setStyleSheet(
            f"background:{theme_manager.theme.input_bg};"
            f"border:1px solid {theme_manager.theme.border_visible};"
            f"border-radius:6px;"
        )
        self._type_w.setMinimumHeight(48)
        type_h = QHBoxLayout(self._type_w)
        type_h.setContentsMargins(0, 0, 14, 0)
        type_h.setSpacing(10)
        type_h.addWidget(self._type_combo, 1)
        type_h.addWidget(self._type_arrow)
        self._grp_type = _field_group(self._type_label, self._type_w)
        ll.addWidget(self._grp_type)

        # Local mode: Backend Script file picker
        self._backend_script_label = QLabel("BACKEND SCRIPT (.PY)")
        self._backend_script_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )

        self._backend_script = _InputField("Backend Script",
            "Optional: select custom bs_roformer.py file", browse=True)
        self._backend_script.btn.clicked.connect(lambda: self._browse(
            self._backend_script, "Python (*.py);;All (*.*)"
        ))
        self._grp_backend_script = _field_group(self._backend_script_label, self._backend_script)
        ll.addWidget(self._grp_backend_script)

        # URL mode: Backend Script URL
        self._backend_url_label = QLabel("BACKEND SCRIPT (.PY URL)")
        self._backend_url_label.setStyleSheet(
            f"font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
            f"color:{theme_manager.theme.text_label};background:transparent;letter-spacing:1px;"
        )
        self._backend_url = _InputField("Backend Script URL",
            "Paste HuggingFace bs_roformer.py URL...")
        self._grp_backend_url = _field_group(self._backend_url_label, self._backend_url)
        ll.addWidget(self._grp_backend_url)

        ll.addSpacing(8)

        self._reg_btn = GlyphButton("Register Model", "+", _outline_icon_color,
                                    glyph_size=18, text_size=12)
        self._reg_btn.setFixedHeight(44)
        self._reg_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._reg_btn.setStyleSheet(outline_button_ss())
        self._reg_btn.clicked.connect(self._register)
        ll.addWidget(self._reg_btn)

        self._download_section = QWidget()
        self._download_section.setVisible(False)
        self._download_section.setStyleSheet("background:transparent;")
        dl = QVBoxLayout(self._download_section)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(12)

        self._download_btn = GlyphButton("Download Model", "+", _outline_icon_color,
                                         glyph_size=18, text_size=12)
        self._download_btn.setFixedHeight(44)
        self._download_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._download_btn.setStyleSheet(outline_button_ss())
        self._download_btn.clicked.connect(self._start_download)
        dl.addWidget(self._download_btn)

        ll.addWidget(self._download_section)

        self._model_mgr = _FolderManagerWidget()
        self._model_mgr.model_installed.connect(self._refresh_registered)
        self._model_mgr.setVisible(False)
        self._folder_search.textChanged.connect(self._model_mgr.set_search_text)
        ll.addWidget(self._model_mgr, 2)

        # No trailing stretch: the manager panel stretches to the bottom so
        # both scroll panels end on the same line (with the page's bottom
        # margin as breathing room).
        main.addWidget(left, 1)

        right = QWidget()
        right.setStyleSheet("background:transparent;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(24)

        # 7px push so REGISTERED MODELS sits on the same line as the
        # REGISTER MODEL / MODEL LIBRARY / CONFIGURATION headers (y 175)
        rl.addSpacing(7)
        rl.addWidget(_section_hdr("Registered Models"))
        rl.addSpacing(4)

        self._list_container = QWidget()
        self._list_container.setStyleSheet("background:transparent;")
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 10, 26)  # air under last row
        self._list_layout.setSpacing(12)
        self._list_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea{{background:transparent;border:none;}}
            QScrollBar:vertical{{width:4px;background:transparent;margin:0;}}
            QScrollBar::handle:vertical{{background:{theme_manager.theme.scrollbar_handle};
            border-radius:2px;min-height:30px;}}
            QScrollBar::handle:vertical:hover{{background:{theme_manager.theme.border_dim};}}
            QScrollBar::add-line:vertical{{height:0;}}
            QScrollBar::sub-line:vertical{{height:0;}}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical{{background:transparent;}}
        """)
        scroll.setWidget(self._list_container)
        rl.addWidget(scroll, 1)

        main.addWidget(right, 1)
        root.addLayout(main)

        # Model Manager is the default source; apply its visibility state now
        # that every section widget exists.
        self._set_mode("manager")

    def _check_updates(self):
        from ui.widgets.update_dialog import check_now
        self._update_btn.setEnabled(False)
        self._update_status.setText("Checking…")

        def _status(text):
            self._update_status.setText(text)
            self._update_btn.setEnabled(True)

        check_now(self, _status)

    def _set_mode(self, mode):
        self._download_mode = mode
        is_manager = mode == "manager"
        is_url = mode == "url"
        is_local = mode == "local"

        self._mode_local.set_checked(is_local)
        self._mode_url.set_checked(is_url)
        self._mode_manager.set_checked(is_manager)

        visible_local = is_local
        visible_url = is_url
        # toggle the fixed-height group containers (they hide their labels
        # and fields with them)
        self._grp_ckpt.setVisible(visible_local)
        self._grp_yaml.setVisible(visible_local)
        self._grp_backend_script.setVisible(visible_local)
        self._reg_btn.setVisible(visible_local)

        self._grp_ckpt_url.setVisible(visible_url)
        self._grp_yaml_url.setVisible(visible_url)
        self._grp_backend_url.setVisible(visible_url)
        self._download_section.setVisible(visible_url)

        self._model_mgr.setVisible(is_manager)
        self._folder_search.setVisible(is_manager)
        self._folder_search_ph.setVisible(not is_manager)

        show_arch_type = not is_manager
        self._grp_arch.setVisible(show_arch_type)
        self._grp_type.setVisible(show_arch_type)

        if is_local:
            self._ckpt.edit.setFocus()
        elif is_url:
            self._ckpt_url.edit.setFocus()
        self.update()
        self.repaint()

    def _refresh_registered(self):
        from backend import settings as settings_store
        path = getattr(self, '_export_settings_timer', None)
        if path:
            path.stop()
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.setParent(None)
                w.deleteLater()
        self._list_layout.addStretch()
        self._registered.clear()
        data = settings_store.load()
        self.load_settings(data.get("registered_models", []))

    def reapply_theme(self):
        self.setStyleSheet(f"#settingsPage{{background:{theme_manager.theme.bg};}}")

    def _browse(self, row, filt):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path:
            row.set_value(path)
            if row == self._yaml:
                self._auto_detect_type(path)

    def _ensure_custom_init(self, dir_path):
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.isfile(init_file):
            try:
                with open(init_file, "w") as f:
                    f.write("")
            except OSError:
                pass

    def _auto_detect_type(self, yaml_path):
        if not yaml_path or not os.path.isfile(yaml_path):
            return
        if yaml_path.lower().endswith('.json'):
            model_type = "dual target (instrumental & vocals)"
            idx = self._type_combo.findText(model_type)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
            return
        detected = classify_model_type(yaml_path)
        if detected and detected in MODEL_TYPES:
            idx = self._type_combo.findText(detected)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)

    def _copy_file(self, src, dest_dir, dest_name):
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, dest_name)
        if os.path.exists(dest):
            reply = QMessageBox.question(
                self, "File Exists",
                f"{dest_name} already exists in destination.\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return None
        shutil.copy2(src, dest)
        return os.path.normpath(dest)

    def _register(self):
        ckpt_src = self._ckpt.value()
        yaml_src = self._yaml.value()
        arch = self._arch_combo.currentText()
        if not ckpt_src:
            QMessageBox.warning(self, "Missing", "Please select a checkpoint file.")
            return
        if not yaml_src:
            QMessageBox.warning(self, "Missing", "Please select a config file.")
            return
        if not os.path.isfile(ckpt_src):
            QMessageBox.warning(self, "Error", f"Checkpoint file not found:\n{ckpt_src}")
            return
        if not os.path.isfile(yaml_src):
            QMessageBox.warning(self, "Error", f"Config file not found:\n{yaml_src}")
            return

        ckpt_name = os.path.basename(ckpt_src)
        yaml_name = os.path.basename(yaml_src)
        ext = os.path.splitext(yaml_name)[1].lower()
        if yaml_name.lower() == "config.yaml" or yaml_name.lower() == "config.json":
            base = os.path.splitext(ckpt_name)[0]
            yaml_name = base + (ext if ext else ".yaml")

        backend_script_path = self._backend_script.value().strip()
        custom_backend = bool(backend_script_path) and os.path.isfile(backend_script_path)
        if custom_backend:
            backend_module = os.path.splitext(ckpt_name)[0]
            dest_dir = os.path.abspath(
                os.path.join(_REPO_ROOT, "models", "custom", backend_module))
            ckpt_dest_dir = dest_dir
            yaml_dest_dir = dest_dir
        else:
            backend_module = ""
            model_folder = ARCH_TO_MODEL_FOLDER.get(arch, "models")
            ckpt_dest_dir = os.path.abspath(os.path.join(_DATA_ROOT, model_folder))
            yaml_dest_dir = os.path.abspath(os.path.join(_DATA_ROOT, "configs"))

        ckpt_dest = self._copy_file(ckpt_src, ckpt_dest_dir, ckpt_name)
        if ckpt_dest is None:
            return

        yaml_dest = self._copy_file(yaml_src, yaml_dest_dir, yaml_name)
        if yaml_dest is None:
            try:
                os.remove(ckpt_dest)
            except OSError:
                pass
            return

        if custom_backend:
            self._copy_file(backend_script_path, ckpt_dest_dir, "bs_roformer.py")
            self._ensure_custom_init(ckpt_dest_dir)

        if yaml_name.lower().endswith('.json'):
            model_type = "dual target (instrumental & vocals)"
            idx = self._type_combo.findText(model_type)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
        else:
            detected = classify_model_type(yaml_dest)
            if detected and detected in MODEL_TYPES:
                idx = self._type_combo.findText(detected)
                if idx >= 0:
                    self._type_combo.setCurrentIndex(idx)
        model_type = self._type_combo.currentText()
        self._finalize_registration(ckpt_name, yaml_name, ckpt_dest, yaml_dest, arch, model_type, backend_module)

    def _start_download(self):
        ckpt_url = self._ckpt_url.value()
        yaml_url = self._yaml_url.value()
        arch = self._arch_combo.currentText()

        if not ckpt_url:
            QMessageBox.warning(self, "Missing", "Please enter a checkpoint URL.")
            return
        if not yaml_url:
            QMessageBox.warning(self, "Missing", "Please enter a config YAML URL.")
            return

        if not HuggingFaceDownloader.is_hf_url(ckpt_url):
            QMessageBox.warning(self, "Invalid URL",
                "Checkpoint URL must be a HuggingFace resolve/main or blob/main link.\n"
                "Example: https://huggingface.co/user/model/resolve/main/model.ckpt")
            return
        if not HuggingFaceDownloader.is_hf_url(yaml_url):
            QMessageBox.warning(self, "Invalid URL",
                "Config URL must be a HuggingFace resolve/main or blob/main link.\n"
                "Example: https://huggingface.co/user/model/resolve/main/model.yaml")
            return

        ckpt_name = HuggingFaceDownloader.extract_filename(ckpt_url)
        yaml_name = HuggingFaceDownloader.extract_filename(yaml_url)
        ext = os.path.splitext(yaml_name)[1].lower()
        if yaml_name.lower() == "config.yaml" or yaml_name.lower() == "config.json":
            base = os.path.splitext(ckpt_name)[0]
            yaml_name = base + ext if ext else ".yaml"

        backend_url_value = self._backend_url.value().strip()
        custom_backend = bool(backend_url_value)
        if custom_backend:
            backend_module = os.path.splitext(ckpt_name)[0]
            dest_dir = os.path.abspath(
                os.path.join(_REPO_ROOT, "models", "custom", backend_module))
            ckpt_dest_dir = dest_dir
            yaml_dest_dir = dest_dir
        else:
            backend_module = ""
            model_folder = ARCH_TO_MODEL_FOLDER.get(arch, "models")
            ckpt_dest_dir = os.path.abspath(os.path.join(_DATA_ROOT, model_folder))
            yaml_dest_dir = os.path.abspath(os.path.join(_DATA_ROOT, "configs"))
            dest_dir = ""
        ckpt_dest = os.path.join(ckpt_dest_dir, ckpt_name)
        yaml_dest = os.path.join(yaml_dest_dir, yaml_name)

        if os.path.exists(ckpt_dest) or os.path.exists(yaml_dest):
            existing = []
            if os.path.exists(ckpt_dest):
                existing.append(ckpt_name)
            if os.path.exists(yaml_dest):
                existing.append(yaml_name)
            reply = QMessageBox.question(
                self, "Files Exist",
                f"{' and '.join(existing)} already exist.\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self._pending_backend_module = backend_module

        self._download_btn.setEnabled(False)

        backend_url = backend_url_value if custom_backend else ""
        backend_dest = os.path.join(dest_dir, "bs_roformer.py") if custom_backend else ""
        self._download_worker = _DownloadWorker(
            ckpt_url, yaml_url, ckpt_dest, yaml_dest,
            backend_url, backend_dest)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.finished.connect(self._download_worker.deleteLater)
        self._download_worker.start()

        self._progress_dialog = _DownloadProgressDialog(
            self._download_worker, ckpt_name, self)
        self._progress_dialog.finished.connect(self._on_dialog_closed)
        self._progress_dialog.show()

    def _on_download_finished(self, success, msg, file_info):
        self._download_btn.setEnabled(True)
        if not success:
            return

        ckpt_name = file_info.get("ckpt_name", "")
        yaml_name = file_info.get("yaml_name", "")
        ckpt_dest = file_info.get("ckpt_path", "")
        yaml_dest = file_info.get("yaml_path", "")

        if not ckpt_name or not ckpt_dest or not os.path.isfile(ckpt_dest):
            self._progress_dialog.on_registered(False)
            return

        if yaml_name.lower() == "config.yaml" or yaml_name.lower() == "config.json":
            ext = os.path.splitext(yaml_name)[1].lower() or ".yaml"
            base = os.path.splitext(ckpt_name)[0]
            yaml_name = base + ext
        arch = self._arch_combo.currentText()
        detected = classify_model_type(yaml_dest)
        if detected and detected in MODEL_TYPES:
            idx = self._type_combo.findText(detected)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
        model_type = self._type_combo.currentText()
        backend_module = getattr(self, '_pending_backend_module', '')
        if backend_module:
            self._ensure_custom_init(
                os.path.abspath(os.path.join(_REPO_ROOT, "models", "custom", backend_module)))
        self._pending_reg = {
            "ckpt_name": ckpt_name,
            "yaml_name": yaml_name,
            "ckpt_dest": ckpt_dest,
            "yaml_dest": yaml_dest,
            "arch": arch,
            "model_type": model_type,
            "backend_module": backend_module,
        }
        self._ckpt_url.clear()
        self._yaml_url.clear()

        self._progress_dialog.on_registered(True)

    def _on_dialog_closed(self, result):
        reg = getattr(self, '_pending_reg', None)
        if reg:
            self._finalize_registration(
                reg["ckpt_name"], reg["yaml_name"],
                reg["ckpt_dest"], reg["yaml_dest"],
                reg["arch"], reg["model_type"],
                reg["backend_module"],
            )
            self._pending_reg = None
        self._progress_dialog = None

    def _finalize_registration(self, ckpt_name, yaml_name, ckpt_dest, yaml_dest, arch, model_type,
                               backend_module=""):
        name = ckpt_name
        model = {
            "name": name, "ckpt": ckpt_dest, "yaml": yaml_dest,
            "arch": arch, "type": model_type,
            "backend_module": backend_module,
            "custom_backend_enabled": bool(backend_module),
        }

        if any(m["name"] == name for m in self._registered):
            QMessageBox.information(self, "Already registered",
                                    f"{name} is already registered.")
            return

        self._registered.append(model)
        self._add_item_widget(model)
        self.model_registered.emit(model)
        self.settings_changed.emit()

    def _add_item_widget(self, model):
        item = _ModelCard(
            model["name"], model["arch"],
            model.get("type", "unknown"),
            model["ckpt"], model["yaml"],
            backend_module=model.get("backend_module", ""),
            custom_backend_enabled=model.get("custom_backend_enabled", False),
        )
        item.remove_requested.connect(self._remove_model)
        item.type_changed.connect(self._on_type_changed)
        self._list_layout.insertWidget(self._list_layout.count() - 1, item)
        self._list_layout.activate()
        self._list_container.updateGeometry()

    def _on_type_changed(self, name, new_type):
        for m in self._registered:
            if m["name"] == name:
                m["type"] = new_type
                break
        self.settings_changed.emit()

    def _delete_model_files(self, model):
        import os as _os
        deleted = []
        failed = []

        ckpt = model.get("ckpt")
        yaml = model.get("yaml")

        for label, path in [("CKPT", ckpt), ("YAML", yaml)]:
            if not path:
                continue
            try:
                if _os.path.isfile(path):
                    _os.remove(path)
                    if _os.path.exists(path):
                        failed.append(f"{label}: {path} — still exists after removal")
                    else:
                        deleted.append(f"{label}: {path}")
                else:
                    failed.append(f"{label}: {path} — not found on disk")
            except OSError as e:
                failed.append(f"{label}: {path} — {e}")

        backend_module = model.get("backend_module", "")
        if backend_module:
            module_dir = _os.path.abspath(
                _os.path.join(_REPO_ROOT, "models", "custom", backend_module))
            if _os.path.isdir(module_dir):
                for fname in _os.listdir(module_dir):
                    if fname.endswith(".py"):
                        fpath = _os.path.join(module_dir, fname)
                        try:
                            _os.remove(fpath)
                            if _os.path.exists(fpath):
                                failed.append(f"BACKEND: {fpath} — still exists after removal")
                            else:
                                deleted.append(f"BACKEND: {fpath}")
                        except OSError as e:
                            failed.append(f"BACKEND: {fpath} — {e}")
                try:
                    remaining = _os.listdir(module_dir)
                    if not remaining:
                        _os.rmdir(module_dir)
                        deleted.append(f"DIR: {module_dir}")
                except OSError:
                    pass

        return deleted, failed

    def _remove_model(self, name):
        model = next((m for m in self._registered if m["name"] == name), None)

        deleted = []
        failed = []
        if model:
            deleted, failed = self._delete_model_files(model)

        self._registered = [m for m in self._registered if m["name"] != name]
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for m in self._registered:
            self._add_item_widget(m)

        if failed:
            QMessageBox.warning(
                self, "Deletion Incomplete",
                "Some model files could not be deleted:\n\n" +
                "\n".join(failed)
            )

        self.model_removed.emit(name)
        self.settings_changed.emit()

    def save_settings(self):
        return self._registered

    def load_settings(self, models: list):
        for m in models:
            if "backend_module" not in m:
                m["backend_module"] = ""
                m["custom_backend_enabled"] = False
            if not any(r["name"] == m["name"] for r in self._registered):
                self._registered.append(m)
                self._add_item_widget(m)
                self.model_registered.emit(m)
