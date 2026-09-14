"""Regression: expanding a Model Manager folder must not force horizontal
scroll.

mdx23c cards with a 6-stem l1-freq score grid used to report a min-width
wider than the left pane, so the folder list grew sideways (Install, type
badge, and extra stems sat past the viewport). This test pins the layout
in a narrow offscreen pane: no h-scroll, inner width fits, badge + Install
stay inside the viewport.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
)

import ui.pages.settings_page as sp  # noqa: E402
import backend.settings as bs  # noqa: E402
from backend.model_manager import ModelInfo  # noqa: E402
from ui.pages.inference_page import _MetricColumns  # noqa: E402

FAILURES = []

PANE_W = 420

DRUM_SCORES = {
    "stems": ["kick", "snare", "toms", "hh", "cymbals", "other"],
    "metrics": {
        stem: {"l1_freq": val}
        for stem, val in (
            ("kick", 61.80), ("snare", 42.79), ("toms", 59.65),
            ("hh", 50.74), ("cymbals", 50.84), ("other", 44.51),
        )
    },
}

TWO_STEM_SCORES = {
    "stems": ["instrum", "vocals"],
    "metrics": {
        "instrum": {"l1_freq": 36.61},
        "vocals": {"l1_freq": 35.86},
    },
}

SCORES_BY_CKPT = {
    "mdx23c_drumsep_5stem_aufr33_jarredou.ckpt": DRUM_SCORES,
    "mdx23c_mid_side_v2e_gilliaan.ckpt": {
        "stems": ["center", "wide"],
        "metrics": {
            "center": {"l1_freq": 50.36},
            "wide": {"l1_freq": 50.32},
        },
    },
}


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def _canned_models():
    return [
        ModelInfo(
            key="mdx23c_drumsep_5stem_aufr33_jarredou.ckpt",
            full_name="MDX23C DrumSep 5 stems by Aufr33 & Jarredou",
            arch="MDX23c Architecture", stem_type="drums",
            category="drums", model_type="mdx23c",
            stems=["kick", "snare", "toms", "hh", "cymbals", "other"],
            target_instrument="drums",
            checkpoint_url=("https://huggingface.co/demo/resolve/main/"
                            "mdx23c_drumsep_5stem_aufr33_jarredou.ckpt"),
            config_url=("https://huggingface.co/demo/resolve/main/"
                        "mdx23c_drumsep_5stem_aufr33_jarredou_config.yaml"),
        ),
        ModelInfo(
            key="mdx23c_mid_side_v2e_gilliaan.ckpt",
            full_name="MDX23C Mid-Side v2e by Gilliaan",
            arch="MDX23c Architecture", stem_type="phantom centre",
            category="phantom centre", model_type="mdx23c",
            stems=["center", "wide"], target_instrument=None,
            checkpoint_url=("https://huggingface.co/demo/resolve/main/"
                            "mdx23c_mid_side_v2e_gilliaan.ckpt"),
            config_url=("https://huggingface.co/demo/resolve/main/"
                        "mdx23c_mid_side_v2e_gilliaan_config.yaml"),
        ),
        ModelInfo(
            key="htdemucs_demo.ckpt",
            full_name="HTDemucs Demo",
            arch="HTDemucs Architecture", stem_type="vocals",
            category="vocals", model_type="htdemucs",
            stems=["vocals", "other"], target_instrument="vocals",
            checkpoint_url=("https://huggingface.co/demo/resolve/main/"
                            "htdemucs_demo.ckpt"),
            config_url=("https://huggingface.co/demo/resolve/main/"
                        "htdemucs_demo.yaml"),
        ),
    ]


def _in_viewport_x(child, scroll):
    vp = scroll.viewport()
    left = child.mapTo(vp, QPoint(0, 0)).x()
    right = child.mapTo(vp, QPoint(child.width(), 0)).x()
    return left >= 0 and right <= vp.width()


def _adjacent_stem_gap(metric):
    a, b = metric._labels[0], metric._labels[1]
    return b.x() - (a.x() + a.width())


def main():
    app = QApplication.instance() or QApplication([])

    two = _MetricColumns(pixel=10, weight=600, left=0, right=0)
    two.set_scores(TWO_STEM_SCORES, "l1_freq")
    wide = QWidget()
    wide.resize(640, 80)
    hl = QHBoxLayout(wide)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.addWidget(two)
    hl.addStretch()
    wide.show()
    QTest.qWait(50)
    app.processEvents()
    gap = _adjacent_stem_gap(two)
    check("2-stem INSTRUM/VOCALS keep the 28px drum-row gap",
          abs(gap - _MetricColumns._STEM_GAP) <= 4)
    wide.close()

    sp.fetch_model_index = lambda: _canned_models()
    sp.fetch_repo_meta = lambda: ""
    sp.fetch_tree_info = lambda: {}
    bs.load = lambda: {"registered_models": [], "_model_tracker": {}}
    bs.save = lambda data: None

    mgr = sp._FolderManagerWidget()
    for _ in range(60):
        if "mdx23c" in mgr._folder_order:
            break
        QTest.qWait(50)
    QTest.qWait(50)

    check("index loaded mdx23c folder", "mdx23c" in mgr._folder_order)

    mgr._scores_store.get = (
        lambda fn, _s=SCORES_BY_CKPT: _s.get((fn or "").lower()))
    mgr._scores_store.request = lambda fn: None
    mgr._folder_meta["mdx23c"] = ({}, {})

    mgr.set_sort_metric("l1_freq")

    host = QWidget()
    host.setFixedWidth(PANE_W)
    host.resize(PANE_W, 700)
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(mgr)
    host.show()
    QTest.qWait(50)

    mgr._toggle_folder("mdx23c")
    QTest.qWait(80)
    app.processEvents()

    scroll = mgr._scroll
    check("horizontal scrollbar policy AlwaysOff",
          scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff)
    check("horizontal scrollbar maximum is 0",
          scroll.horizontalScrollBar().maximum() == 0)
    check("inner list width fits the viewport",
          mgr._scroll_widget.width() <= scroll.viewport().width() + 2)

    install = None
    for btn in mgr.findChildren(QPushButton):
        if btn.text() == "Install" and btn.isVisible():
            install = btn
            break
    check("Install button visible after expand", install is not None)
    if install is not None:
        check("Install stays inside the viewport horizontally",
              _in_viewport_x(install, scroll))

    tags = [w for w in mgr.findChildren(sp.QLabel)
            if w.objectName() == "mgrTypeTag" and w.isVisible()]
    check("type badge visible after expand", len(tags) >= 1)
    if tags:
        check("type badge stays inside the viewport horizontally",
              _in_viewport_x(tags[0], scroll))

    metrics = [m for m in mgr.findChildren(_MetricColumns) if m.has_content()]
    check("l1-freq metric columns rendered", len(metrics) >= 1)
    six = next((m for m in metrics if len(m._labels) == 6), None)
    two_mgr = next((m for m in metrics if len(m._labels) == 2), None)
    check("6-stem drum metrics present", six is not None)
    if six:
        stems = {lbl.text() for lbl in six._labels}
        check("all 6 drum stems still shown (wrapped, not clipped away)",
              stems == {"KICK", "SNARE", "TOMS", "HH", "CYMBALS", "OTHER"})
        check("metric block fits the viewport horizontally",
              _in_viewport_x(six, scroll))
    check("2-stem metrics present", two_mgr is not None)
    if two_mgr:
        check("2-stem card keeps the same compact stem gap",
              abs(_adjacent_stem_gap(two_mgr) - _MetricColumns._STEM_GAP) <= 4)

    host.close()
    mgr.deleteLater()
    app.processEvents()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
