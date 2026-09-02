"""ui/widgets/pretrained_models_dialog.py
Pre-trained checkpoint picker for the TRAINING tab.

Opened from the CONFIGURATION column; fetches the ZFTurbo pre-trained catalog
at runtime (backend.pretrained_catalog), lists it grouped by section, and lets
the user download a checkpoint + its config. On install the row offers
"Use for fine-tuning", which fills the TRAINING page's Config path and Resume
checkpoint rows with the freshly downloaded files.

Follows the app's download pattern: a plain QThread for the fetch, the shared
parallel downloader for the weights, and themed rows/cards.
"""
from __future__ import annotations

import re
import threading
from collections import Counter
from typing import Optional

from PySide6.QtCore import Qt, Signal, QThread, QTimer, QRect, QSize, QEvent, QUrl
from PySide6.QtGui import QFont, QFontMetrics, QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QScrollArea, QSizePolicy, QWidget, QProgressBar, QMessageBox, QLayout,
)

from ui.theme import theme_manager
from ui.pages.inference_page import _LinkBadge
from backend import pretrained_catalog as catalog
from backend import settings as settings_store

# Upstream doc page per catalog source — the ⓘ-style link badge on each model
# row opens the exact table the entry came from.
_GITHUB_LINKS = {
    "pretrained":
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md",
    "mel_roformer":
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/mel_roformer_experiments.md",
}


def _card_ss():
    t = theme_manager.theme
    return (
        "QFrame#ptCard{"
        f"background:{t.surface};border:1px solid {t.border_visible};"
        "border-radius:10px;}"
    )


def _installed_btn_ss():
    """Grey, disabled 'Installed' chip — mirrors the settings page's installed
    affordance so an installed start is clearly no longer actionable."""
    t = theme_manager.theme
    return (
        f"QPushButton{{background:{t.disabled_bg};color:{t.text_muted};border:none;"
        "font-family:'Montserrat',sans-serif;font-weight:600;font-size:10px;"
        "border-radius:6px;padding:0 14px;}"
    )


def _card_title_ss():
    t = theme_manager.theme
    return (
        "font-family:'Montserrat',sans-serif;font-size:9px;font-weight:700;"
        f"color:{t.text};background:transparent;letter-spacing:1.5px;"
    )


def _pill_btn_ss(accent: bool = False):
    t = theme_manager.theme
    if accent:
        return (
            f"QPushButton{{background:{theme_manager.accent};color:{theme_manager._accent_text};"
            "border:none;border-radius:6px;font-family:'Montserrat',sans-serif;"
            "font-weight:600;font-size:10px;padding:0 14px;}"
            f"QPushButton:hover{{background:{theme_manager._accent_hover};}}"
            f"QPushButton:disabled{{background:{t.disabled_bg};color:{t.text_muted};}}"
        )
    return (
        f"QPushButton{{background:{t.surface_alt};color:{t.text};"
        f"border:1px solid {t.border};border-radius:6px;font-family:'Montserrat',sans-serif;"
        "font-weight:600;font-size:10px;padding:0 14px;}"
        f"QPushButton:hover{{background:{t.border_visible};}}"
        f"QPushButton:disabled{{color:{t.text_muted};}}"
    )


# ── Fetch thread ─────────────────────────────────────────────────────────────

class _FetchThread(QThread):
    loaded = Signal(list)
    failed = Signal(str)

    def run(self):
        try:
            self.loaded.emit(catalog.fetch_catalog())
        except Exception as exc:
            self.failed.emit(str(exc))


# ── Install thread ───────────────────────────────────────────────────────────

class _InstallThread(QThread):
    progress = Signal(str, int, int)   # filename, done, total
    status = Signal(str)
    speed = Signal(float)
    finished_signal = Signal(bool, str)

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._cancelled = False

    def run(self):
        try:
            ok, msg = catalog.install_model(
                self._model,
                progress_callback=lambda n, c, t: self.progress.emit(n, c, t),
                status_callback=self.status.emit,
                speed_callback=self.speed.emit,
                cancel_callback=lambda: self._cancelled,
            )
            self.finished_signal.emit(ok, msg)
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))


# ── Row ──────────────────────────────────────────────────────────────────────

class _ModelRow(QFrame):
    """One pre-trained model with its Install / Use-for-fine-tuning actions."""
    install_clicked = Signal(object)          # PretrainedModel
    use_clicked = Signal(object)              # PretrainedModel (must be installed)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setObjectName("ptCard")
        self._model = model
        self._installed = model.is_installed
        self.setStyleSheet(_card_ss())

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 12, 20, 12)
        root.setSpacing(8)

        # ── Top: name + GitHub link + INSTALLED + action buttons ──
        # The two buttons sit on the same line as the name, so a card is
        # three rows tall (title, arch, metrics) instead of four.
        top = QHBoxLayout()
        top.setSpacing(8)
        self._name_lbl = QLabel(model.name)
        self._name_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:13px;font-weight:bold;"
            f"color:{theme_manager.theme.text};background:transparent;border:none;"
        )
        top.addWidget(self._name_lbl)

        # Link badge → the upstream doc table this entry was parsed from.
        doc = _GITHUB_LINKS.get(getattr(model, "source", "pretrained"),
                                _GITHUB_LINKS["pretrained"])
        doc_name = "mel_roformer_experiments.md" if "mel_roformer" in doc else "pretrained_models.md"
        self._link_badge = _LinkBadge(f"Open the upstream table ({doc_name})")
        self._link_badge.clicked.connect(lambda u=doc: QDesktopServices.openUrl(QUrl(u)))
        top.addWidget(self._link_badge, 0, Qt.AlignVCenter)

        top.addStretch()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            "font-family:'Montserrat';font-size:10px;color:"
            f"{theme_manager.accent if self._installed else theme_manager.theme.text_muted};"
            "background:transparent;border:none;"
        )
        top.addWidget(self._status_lbl)

        self._install_btn = QPushButton("Install")
        self._install_btn.setFixedHeight(32)
        self._install_btn.setMinimumWidth(96)
        self._install_btn.setCursor(Qt.PointingHandCursor)
        self._install_btn.setStyleSheet(_pill_btn_ss(accent=True))
        self._install_btn.clicked.connect(lambda: self.install_clicked.emit(self._model))
        top.addWidget(self._install_btn)

        self._use_btn = QPushButton("Use for fine-tuning")
        self._use_btn.setFixedHeight(32)
        self._use_btn.setMinimumWidth(170)
        self._use_btn.setCursor(Qt.PointingHandCursor)
        self._use_btn.setStyleSheet(_pill_btn_ss(accent=False))
        self._use_btn.clicked.connect(lambda: self.use_clicked.emit(self._model))
        top.addWidget(self._use_btn)
        root.addLayout(top)

        # ── Meta line: arch • instruments • metrics ──
        bits = []
        if model.arch_hint:
            hint = model.arch_hint.replace(" Architecture", "").replace(" Model", "")
            bits.append(hint)
        if model.instruments:
            bits.append(model.instruments)
        meta = "   ·   ".join(bits)
        if meta:
            self._meta_lbl = QLabel(meta)
            self._meta_lbl.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;"
                f"color:{theme_manager.theme.text_sec};background:transparent;border:none;"
            )
            root.addWidget(self._meta_lbl)

        if model.metrics:
            self._metrics_lbl = QLabel(model.metrics)
            self._metrics_lbl.setStyleSheet(
                "font-family:'Montserrat';font-size:10px;"
                f"color:{theme_manager.theme.text_dim};background:transparent;border:none;"
            )
            self._metrics_lbl.setWordWrap(True)
            root.addWidget(self._metrics_lbl)

        # ── Install status line (download progress / message) ── Hidden until
        # a download starts, so an idle card stays only three rows tall.
        self._status_row = QWidget()
        self._status_row.setStyleSheet("background:transparent;")
        prog = QHBoxLayout(self._status_row)
        prog.setContentsMargins(0, 0, 0, 0)
        prog.setSpacing(10)
        self._status_msg = QLabel("")
        self._status_msg.setStyleSheet(
            "font-family:'Montserrat';font-size:10px;"
            f"color:{theme_manager.theme.text_dim};background:transparent;border:none;"
        )
        prog.addWidget(self._status_msg, 1)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            "QProgressBar{background:" + theme_manager.theme.border + ";border:none;border-radius:2px;}"
            "QProgressBar::chunk{background:" + theme_manager.accent + ";border-radius:2px;}"
        )
        self._progress.setValue(0)
        prog.addWidget(self._progress, 1)
        self._status_row.setVisible(False)
        root.addWidget(self._status_row)

        # Buttons exist now — render the installed/available state.
        self._refresh_status()

    def _refresh_status(self):
        if self._installed:
            # Reuse the settings-page installed affordance: a grey disabled
            # "Installed" chip in place of the blue Install button.
            self._status_lbl.setText("")
            self._install_btn.setText("Installed")
            self._install_btn.setEnabled(False)
            self._install_btn.setCursor(Qt.ArrowCursor)
            self._install_btn.setStyleSheet(_installed_btn_ss())
        else:
            self._status_lbl.setText("")
            self._install_btn.setText("Install")
            self._install_btn.setCursor(Qt.PointingHandCursor)
            self._install_btn.setStyleSheet(_pill_btn_ss(accent=True))
        self._use_btn.setEnabled(self._installed)

    def set_installing(self):
        self._install_btn.setEnabled(False)
        self._status_row.setVisible(True)
        self._progress.setValue(0)
        self._status_msg.setText("")

    def set_progress(self, cur, total):
        if total:
            self._progress.setValue(int(cur * 100 / total))

    def set_status(self, text, error=False):
        self._status_row.setVisible(True)
        self._status_msg.setText(text)
        self._status_msg.setStyleSheet(
            "font-family:'Montserrat';font-size:10px;color:"
            f"{theme_manager.theme.error if error else theme_manager.theme.text_dim};"
            "background:transparent;border:none;"
        )

    def finish_install(self, ok, msg):
        self._install_btn.setEnabled(True)
        if ok:
            self._installed = True
            self._refresh_status()
            self.set_status("Installed — ready to fine-tune.")
        else:
            self._progress.setValue(0)
            self.set_status(msg, error=True)

    @property
    def model(self):
        return self._model


class _SectionHeader(QFrame):
    def __init__(self, title, count, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;border:none;")
        hl = QHBoxLayout(self)
        hl.setContentsMargins(4, 4, 0, 2)
        hl.setSpacing(10)
        lbl = QLabel(title.upper())
        lbl.setStyleSheet(_card_title_ss())
        hl.addWidget(lbl)
        cnt = QLabel(f"{count}")
        cnt.setStyleSheet(
            "font-family:'Montserrat';font-size:9px;color:"
            f"{theme_manager.theme.text_label};background:transparent;border:none;"
            "border:1px solid " + theme_manager.theme.border + ";border-radius:8px;"
            "padding:1px 6px;"
        )
        hl.addWidget(cnt)
        hl.addStretch()


# ── Filter chips ─────────────────────────────────────────────────────────────

def _chip_ss(active: bool) -> str:
    """Filter chip: quiet pill when idle, accent-bordered when selected
    (same accent vocabulary as the training page's LATEST chip). The width
    is set per-button from the text metrics; padding here only centers the
    label inside that width."""
    t = theme_manager.theme
    if active:
        return (
            f"QPushButton{{background:{theme_manager._accent_soft};color:{theme_manager.accent};"
            f"border:1px solid {theme_manager.accent};border-radius:9px;"
            "font-family:'Montserrat',sans-serif;font-weight:700;font-size:9px;"
            "padding:0;}"
        )
    return (
        f"QPushButton{{background:{t.surface_alt};color:{t.text_dim};"
        f"border:1px solid {t.border};border-radius:9px;"
        "font-family:'Montserrat',sans-serif;font-weight:600;font-size:9px;"
        "padding:0;}"
        f"QPushButton:hover{{color:{t.text};border:1px solid {t.border_visible};}}"
    )


class _FlowLayout(QLayout):
    """Wrapping row of widgets (Qt's classic FlowLayout, trimmed to what the
    chip bars need) — the architecture families don't fit one line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._vspace = 6
        self.setSpacing(8)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._do_layout(QRect(0, 0, w, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def invalidate(self):
        super().invalidate()

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        row_x = rect.x() + m.left()
        x, y = row_x, rect.y() + m.top()
        right = rect.right() - m.right()
        line_h = 0
        for item in self._items:
            w, h = item.sizeHint().width(), item.sizeHint().height()
            if x + w > right and x > row_x:      # wrap to the next line
                x = row_x
                y += line_h + self._vspace
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(x, y, w, h))
            x += w + self.spacing()
            line_h = max(line_h, h)
        return y + line_h + m.bottom() - rect.y()


class _ChipBar(QFrame):
    """One filter row: a small-caps caption on the left, chip buttons that
    wrap onto as many lines as needed on the right. Single-select ('All' +
    the groups). Emits `changed` when the selection moves; the dialog
    re-populates the list from the active filters."""
    changed = Signal()

    CAPTION_W = 108

    def __init__(self, caption, choices, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;border:none;")
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        cap = QLabel(caption.upper())
        cap.setFixedWidth(self.CAPTION_W)
        cap.setStyleSheet(_card_title_ss())
        hl.addWidget(cap, 0, Qt.AlignVCenter)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        self._holder = holder
        self._flow = _FlowLayout(holder)
        hl.addWidget(holder, 1)
        self._btns = []
        f = QFont("Montserrat")
        f.setPixelSize(9)
        f.setBold(True)
        fm = QFontMetrics(f)
        for key, label in choices:
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(24)
            # QPushButton's own sizeHint carries ~60px of style chrome per
            # button, which balloons a wrapping chip row; the width we want
            # is just the rendered text plus the stylesheet's 10px side pads.
            b.setFixedWidth(min(fm.horizontalAdvance(label) + 26, 280))
            b.clicked.connect(lambda _=False, k=key: self.select(k))
            self._flow.addWidget(b)
            self._btns.append((key, b))
        self._holder.installEventFilter(self)
        self.select(choices[0][0])

    def eventFilter(self, obj, ev):
        # The holder's Resize is the reliable trigger: it fires whenever the
        # layout assigns the bar its real width, including that first layout
        # pass (the bar's own resizeEvent fires before the holder has its
        # final width, which would lock in a stale height).
        if obj is self._holder and ev.type() == QEvent.Resize:
            QTimer.singleShot(0, self._sync_height)
        return super().eventFilter(obj, ev)

    def _sync_height(self):
        """Set the bar height from the holder's *current* width. One pass can
        be stale — the box layout resizes in phases, so the width the filter
        saw may grow again on the next pass — hence re-check after applying
        (converges immediately; the width doesn't depend on the height)."""
        w = self._holder.width()
        if w <= 0:
            return
        want = max(self._flow.heightForWidth(w), 22) + 2
        if self.height() != want:
            self.setFixedHeight(want)
            QTimer.singleShot(0, self._sync_height)

    def select(self, key):
        changed = key != self.current()
        for k, b in self._btns:
            b.setStyleSheet(_chip_ss(k == key))
        self._current = key
        if changed:
            self.changed.emit()

    def current(self):
        return getattr(self, "_current", None)


def _target_key(m):
    """Which target group a model belongs to — the stem layout it separates.
    Falls back to the doc section, so MUSDB18HQ multi-stem rows without an
    instruments cell still land in Multi-stem."""
    sec = (m.section or "").lower()
    inst = (m.instruments or "").lower()
    stems = [s.strip() for s in re.split(r"[/,+]", inst) if s.strip()]
    if "vocal" in sec or stems == ["vocals"]:      # vocals (+ other) only
        return "vocals"
    if "single" in sec or len(stems) == 1:
        return "single"
    if "multi" in sec or len(stems) > 1:
        return "multi"
    return "other"


def _arch_key(m):
    """Architecture family token for one model ("" when the hint is empty)."""
    h = (m.arch_hint or "").replace(" Architecture", "").replace(" Model", "")
    h = h.strip().lower()
    h = re.sub(r"[\s_\-]+", "_", h)
    if h in ("melband_roformer", "mel_band_roformer"):
        return "mel_band_roformer"
    if h in ("bs_roformer",):
        return "bs_roformer"
    return h


_ARCH_LABELS = {
    "bs_roformer": "BS RoFormer",
    "mel_band_roformer": "Mel-Band RoFormer",
    "scnet": "SCNet",
    "ht_demucs": "HT Demucs",
    "demucs": "Demucs",
    "mdx23c": "MDX23C",
    "bsmamba2": "BS Mamba2",
    "dttnet": "DTTNet",
    "vitlarge23": "VitLarge23",
    "swin_upernet": "Swin Upernet",
    "apollo": "Apollo",
    "bandit": "Bandit",
    "conformer": "Conformer",
}


def _arch_label(key):
    return _ARCH_LABELS.get(key, key.replace("_", " ").title())


class PretrainedModelsDialog(QDialog):
    """Browse + install pre-trained checkpoints, then wire one into training."""
    use_for_training = Signal(str, str)   # config_path, checkpoint_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: list = []
        self._rows: dict = {}       # id(model) -> _ModelRow
        self._fetch_thread: Optional[QThread] = None
        self._install_threads: list = []
        self._arch_keys = None      # arch chips currently shown (rebuild guard)
        self.setWindowTitle("Pre-trained Models")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.resize(960, 760)
        self.setMinimumSize(760, 620)
        # Restore the last-used filters. The target chips always exist, so it
        # can be applied immediately; the architecture set only exists once
        # the catalog is loaded, so it is deferred (self._pending_arch) until
        # _on_loaded, and only applied if still a valid choice there.
        saved = settings_store.load_pretrained_filters()
        self._saved_target = saved.get("target")
        self._pending_arch = saved.get("arch")
        self._build_ui()
        valid_targets = {k for k, _ in self._target_bar._btns}
        if self._saved_target in valid_targets:
            self._target_bar.select(self._saved_target)
        self._start_fetch()

    def _build_ui(self):
        t = theme_manager.theme
        self.setStyleSheet("QDialog{background:" + t.bg + ";}")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 26)
        root.setSpacing(0)

        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_lbl = QLabel("PRE-TRAINED MODELS")
        title_lbl.setStyleSheet(
            "font-family:'Montserrat',sans-serif;font-size:14px;font-weight:bold;"
            f"color:{t.text};background:transparent;border:none;letter-spacing:1px;"
        )
        title_row.addWidget(title_lbl)
        title_row.addStretch()

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedHeight(32)
        self._refresh_btn.setMinimumWidth(96)
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.setStyleSheet(_pill_btn_ss(accent=False))
        self._refresh_btn.clicked.connect(self._refresh)
        title_row.addWidget(self._refresh_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.setFixedHeight(32)
        self._close_btn.setMinimumWidth(96)
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.setStyleSheet(_pill_btn_ss(accent=True))
        self._close_btn.clicked.connect(self.accept)
        title_row.addWidget(self._close_btn)
        root.addLayout(title_row)
        root.addSpacing(10)

        sub = QLabel(
            "Download a ZFTurbo pre-trained checkpoint to fine-tune from. "
            "Each entry installs the weights plus its matching config."
        )
        sub.setStyleSheet(
            "font-family:'Montserrat';font-size:10px;color:" + t.text_sec
            + ";background:transparent;border:none;"
        )
        sub.setWordWrap(True)
        root.addWidget(sub)
        root.addSpacing(22)

        # ── Filter chips: target (stems) + architecture family ──
        self._target_bar = _ChipBar("Target", [
            ("all", "All"),
            ("vocals", "Vocals"),
            ("single", "Single-stem"),
            ("multi", "Multi-stem"),
        ])
        self._target_bar.changed.connect(self._populate)
        self._target_bar.changed.connect(self._persist_filters)
        root.addWidget(self._target_bar)
        root.addSpacing(8)

        self._arch_bar = _ChipBar("Architecture", [("all", "All")])
        self._arch_bar.changed.connect(self._populate)
        self._arch_bar.changed.connect(self._persist_filters)
        root.addWidget(self._arch_bar)
        root.addSpacing(20)
        self._root_lay = root

        # Loading / error
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            "font-family:'Montserrat';font-size:11px;color:" + t.text_sec
            + ";background:transparent;border:none;"
        )
        root.addWidget(self._status_lbl)
        root.addSpacing(14)

        # Scroll area of sections
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:4px;background:transparent;margin:0;}"
            f"QScrollBar::handle:vertical{{background:{t.scrollbar_handle};"
            "border-radius:2px;min-height:30px;}"
            "QScrollBar::add-line:vertical{height:0;}"
            "QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
        )
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._content = QWidget()
        self._content.setStyleSheet("background:transparent;")
        self._list_layout = QVBoxLayout(self._content)
        self._list_layout.setContentsMargins(0, 6, 10, 10)
        self._list_layout.setSpacing(12)
        # The trailing stretch is (re)added by _populate.
        self._scroll.setWidget(self._content)
        root.addWidget(self._scroll, 1)

    def _start_fetch(self):
        self._status_lbl.setText("Loading pre-trained catalog…")
        self._models = []
        self._populate([])
        thread = _FetchThread(self)
        self._fetch_thread = thread
        thread.loaded.connect(self._on_loaded)
        thread.failed.connect(self._on_fetch_failed)
        # Do NOT deleteLater the thread: the Python attribute still points at it
        # after `finished`, so a later _refresh() would dereference a dead C++
        # object. Instead clear the ref on finish; the dialog owns the thread
        # and reaps it on close.
        thread.finished.connect(lambda: self._on_fetch_finished(thread))
        thread.start()

    def _on_fetch_finished(self, thread):
        if self._fetch_thread is thread:
            self._fetch_thread = None

    def _refresh(self):
        if self._fetch_thread and self._fetch_thread.isRunning():
            return
        self._start_fetch()

    def _on_loaded(self, models):
        self._models = models
        self._populate()
        # Apply a cross-session architecture restore now that the catalog has
        # produced the arch chips. If the saved arch is no longer a valid
        # choice under the restored target (e.g. a doc change dropped the
        # family), it falls back to All — the rebuild already did that.
        arch = self._pending_arch
        self._pending_arch = None
        if arch is not None and arch in self._arch_keys:
            self._arch_bar.select(arch)

    def _persist_filters(self, *_):
        """Save the current filter selection for the next dialog session."""
        settings_store.save_pretrained_filters({
            "target": self._target_bar.current() or "all",
            "arch": self._arch_bar.current() or "all",
        })

    def _on_fetch_failed(self, msg):
        self._status_lbl.setText(
            f"Could not load the catalog. Check your connection. ({msg})"
        )

    def _rebuild_arch_bar(self, pool):
        """Rebuild the architecture chips from what the current Target filter
        can reach, with per-choice counts. Kept only when the choice set is
        unchanged, so switching chips never resets the user's selection."""
        counts = Counter(k for m in pool if (k := _arch_key(m)))
        choices = [("all", "All")] + [
            # A count only earns its space on the big families; "Apollo  1"
            # is pure noise.
            (k, _arch_label(k) if n == 1 else f"{_arch_label(k)}  {n}")
            for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        if [k for k, _ in choices] == self._arch_keys:
            return
        self._arch_keys = [k for k, _ in choices]
        keep = self._arch_bar.current()
        new_bar = _ChipBar("Architecture", choices)
        new_bar.changed.connect(self._populate)
        new_bar.changed.connect(self._persist_filters)
        self._root_lay.replaceWidget(self._arch_bar, new_bar)
        self._arch_bar.hide()
        self._arch_bar.deleteLater()
        self._arch_bar = new_bar
        # Keep the user's architecture choice when it still exists in the new
        # pool (e.g. staying on Mel-Band while switching targets); otherwise
        # fall back to All. Blocked: _populate is already running.
        new_bar.blockSignals(True)
        new_bar.select(keep if keep in self._arch_keys else "all")
        new_bar.blockSignals(False)

    def _populate(self, *_):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows = {}

        target = self._target_bar.current() or "all"
        by_target = [m for m in self._models
                     if target == "all" or _target_key(m) == target]
        self._rebuild_arch_bar(by_target)

        arch = self._arch_bar.current() or "all"
        visible = [m for m in by_target
                   if arch == "all" or _arch_key(m) == arch]

        # group by section, preserving order
        sections = []
        section_map = {}
        for m in visible:
            if m.section not in section_map:
                section_map[m.section] = []
                sections.append(m.section)
            section_map[m.section].append(m)

        for sec in sections:
            group = section_map[sec]
            self._list_layout.addWidget(_SectionHeader(sec, len(group)))
            for m in group:
                row = _ModelRow(m)
                row.install_clicked.connect(self._on_install)
                row.use_clicked.connect(self._on_use)
                self._rows[id(m)] = row
                self._list_layout.addWidget(row)
            self._list_layout.addSpacing(6)
        self._list_layout.addStretch()

        if self._models:
            total, shown = len(self._models), len(visible)
            self._status_lbl.setText(
                f"Showing {shown} of {total} pre-trained checkpoints."
                if shown != total else
                f"Loaded {total} pre-trained checkpoints."
            )

    def _on_install(self, model):
        row = self._rows.get(id(model))
        if row is None:
            return
        row.set_installing()
        thread = _InstallThread(model)
        thread.progress.connect(lambda n, c, t, r=row: r.set_progress(c, t))
        thread.status.connect(lambda s, r=row: r.set_status(s))
        thread.finished_signal.connect(lambda ok, msg, r=row, m=model: self._on_install_done(m, r, ok, msg))
        thread.finished.connect(thread.deleteLater)
        # keep a reference so it isn't garbage-collected mid-download
        self._install_threads.append(thread)
        thread.start()

    def _on_install_done(self, model, row, ok, msg):
        row.finish_install(ok, msg)

    def _on_use(self, model):
        if not model.is_installed:
            QMessageBox.information(self, "Not installed",
                                    "Install this pre-trained model first.")
            return
        if not model.config_path or not model.checkpoint_path:
            QMessageBox.information(self, "Missing files",
                                    "The config or checkpoint is missing. Re-install it.")
            return
        self.use_for_training.emit(model.config_path, model.checkpoint_path)
        self.accept()
