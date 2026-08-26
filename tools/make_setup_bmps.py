"""
tools/make_setup_bmps.py
------------------------
Generates the Inno Setup wizard images from the app's own branding:

  build/setup/wizard.png        164x314  — mvsep wordmark, transparent
  build/setup/wizardsmall.png   55x58    — app icon, transparent

Called automatically by tools/build_all.bat before ISCC.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "build", "setup")


def main():
    from PySide6.QtGui import QImage, QPainter, Qt
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QRectF

    app = QApplication([])

    icon = QImage(os.path.join(ROOT, "resources", "app_icon.png"))
    wordmark = QImage(os.path.join(ROOT, "resources", "mvsep.png"))
    if icon.isNull() or wordmark.isNull():
        print("branding images missing", file=sys.stderr)
        return 1

    def fit(img, max_w, max_h):
        return img.scaled(max_w, max_h, Qt.KeepAspectRatio,
                          Qt.SmoothTransformation)

    def render(path, w, h, parts):
        canvas = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        canvas.fill(Qt.transparent)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        for img, cx, cy in parts:
            p.drawImage(QRectF(cx - img.width() / 2, cy - img.height() / 2,
                               img.width(), img.height()), img)
        p.end()
        ok = canvas.save(path, "PNG")
        print(("wrote " if ok else "FAILED ") + path)
        return 0 if ok else 1

    rc = 0
    # Large side panel: mvsep wordmark, transparent background.
    os.makedirs(OUT, exist_ok=True)
    word_l = fit(wordmark, 124, 44)
    rc |= render(os.path.join(OUT, "wizard.png"), 164, 314,
                 [(word_l, 82, 157)])
    # Small corner logo: app icon only, transparent background.
    icon_s = fit(icon, 47, 46)
    rc |= render(os.path.join(OUT, "wizardsmall.png"), 55, 58,
                 [(icon_s, 27, 29)])
    return rc


if __name__ == "__main__":
    sys.exit(main())
