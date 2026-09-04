"""Regression test: NOT-RUNNABLE badge on model library cards.

A registered model whose effective engine type has no dispatch branch (a zoo
model newer than the build, a drifted label, a hand-registered config) shows
a red 'NOT RUNNABLE' badge next to the model name so the user sees the
problem before clicking Run — the pre-run validation still blocks the run as
a backstop, but the card must make the state visible up front.

Drives the real _ArchCard.add_model path (the same code on_model_registered
feeds) offscreen with the shared engine-branch verdict; no torch, no network.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from ui.theme import theme_manager  # noqa: E402
from ui.pages.inference_page import _ArchCard, _ModelItem  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def main():
    app = QApplication([])
    theme_manager.init_app(app)
    card = _ArchCard("MDX23c Architecture")

    tmp = tempfile.mkdtemp(prefix="msst_badge_test_")
    yaml_path = os.path.join(tmp, "m.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("training:\n  instruments: [vocals, other]\n")

    def add(name, engine_type):
        card.add_model(name, os.path.join(tmp, name + ".ckpt"), yaml_path,
                       "MDX23c Architecture", "vocals", engine_type)

    add("good_mdx", "mdx23c")
    add("future_zoo", "hyperace_v9000")   # no engine branch

    items = {i._name: i for i in card.findChildren(_ModelItem)}
    check("both models added", set(items) == {"good_mdx", "future_zoo"})

    runnable = {n: i._runnable for n, i in items.items()}
    check("supported model flagged runnable", runnable["good_mdx"] is True)
    check("unsupported model flagged not-runnable",
          runnable["future_zoo"] is False)

    def badges(item):
        return [l for l in item.findChildren(QLabel)
                if l.text() == "NOT RUNNABLE"]

    check("no badge on supported model", badges(items["good_mdx"]) == [])
    b = badges(items["future_zoo"])
    check("badge on unsupported model", len(b) == 1)
    if b:
        tt = b[0].toolTip()
        check("badge tooltip names the model type",
              "hyperace_v9000" in tt and "no branch" in tt)
        check("badge tooltip hints at a fix", "update the app" in tt)

    # scnet_tran (a *supported* variant) must never be flagged even though
    # its coarse arch label maps to plain scnet.
    card2 = _ArchCard("SCNet Architecture")
    card2.add_model("tran_ok", os.path.join(tmp, "t.ckpt"), yaml_path,
                    "SCNet Architecture", "vocals", "scnet_tran")
    t_items = {i._name: i for i in card2.findChildren(_ModelItem)}
    check("scnet_tran stays runnable",
          "tran_ok" in t_items and t_items["tran_ok"]._runnable is True)

    # Manual/URL DTTNet registration (no precise engine_type recorded): the
    # arch map must resolve to the engine's 'dttnet' branch — NOT the stale
    # 'dtt_net' spelling — so the card is not flagged as NOT RUNNABLE.
    dcard = _ArchCard("DTTNet Architecture")
    dcard.add_model("dttnet_vocalsg32_ep4082_fix.ckpt",
                    os.path.join(tmp, "d.ckpt"), yaml_path,
                    "DTTNet Architecture", "vocals", "")
    d_items = {i._name: i for i in dcard.findChildren(_ModelItem)}
    d_item = d_items.get("dttnet_vocalsg32_ep4082_fix.ckpt")
    check("dttnet arch entry runnable",
          d_item is not None and d_item._runnable is True)
    check("dttnet entry has no badge",
          d_item is not None and badges(d_item) == [])

    # The '···' per-model menu must stay on-screen even when a row packs a
    # long ckpt filename + CUSTOM + type badges (the "3-dot missing in arch
    # mode" regression): the label is shrinkable/elided so the fixed
    # right-side cluster is never pushed past the card's visible edge.
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from PySide6.QtCore import Qt
    host = QWidget()
    host.resize(340, 420)
    v = QVBoxLayout(host)
    wcard = _ArchCard("DTTNet Architecture")
    wcard.add_model("dttnet_vocalsg32_ep4082_fix.ckpt",
                    os.path.join(tmp, "d.ckpt"), yaml_path,
                    "DTTNet Architecture", "vocals", "dttnet",
                    backend_module="bs_roformer", custom_backend_enabled=True)
    v.addWidget(wcard)
    host.show()
    app.processEvents()
    wcard._toggle_expand(animated=False)
    app.processEvents()
    wit = wcard.findChildren(_ModelItem)[0]
    dot_right = wit._dots.mapTo(wcard, wit._dots.rect().topRight()).x()
    check("3-dot menu inside card edge in narrow pane",
          dot_right <= wcard.width() - 1)
    check("long label elides when space is tight",
          wit._lbl.text() != wit._display and wit._lbl.text().endswith("…"))
    check("full name still searchable via _display",
          wit._display == "dttnet_vocalsg32_ep4082_fix.ckpt")
    host.close()

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    app.processEvents()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
