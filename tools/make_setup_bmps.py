"""
tools/make_setup_bmps.py
------------------------
Generates the Inno Setup wizard images from the app's own branding:

  build/setup/wizard.bmp        164x314  — app_icon.png + mvsep wordmark
  build/setup/wizardsmall.bmp   55x58    — mvsep wordmark

Called automatically by tools/build_all.bat before ISCC.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "build", "setup")

BG = "#14161A"  # brand dark (theme bg)


def main():
    from PySide6.QtGui import QImage, QPainter, QColor, Qt
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

    def render(path, w, h, parts, transparent=False):
        canvas = QImage(w, h, QImage.Format_ARGB32_Premultiplied
                        if transparent else QImage.Format_RGB888)
        if not transparent:
            canvas.fill(QColor(BG))
        p = QPainter(canvas)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        for img, cx, cy in parts:
            p.drawImage(QRectF(cx - img.width() / 2, cy - img.height() / 2,
                               img.width(), img.height()), img)
        p.end()
        ok = canvas.save(path, "BMP")
        print(("wrote " if ok else "FAILED ") + path)
        return 0 if ok else 1

    rc = 0
    # Large side panel: mvsep wordmark on the brand dark background.
    os.makedirs(OUT, exist_ok=True)
    word_l = fit(wordmark, 124, 44)
    rc |= render(os.path.join(OUT, "wizard.bmp"), 164, 314,
                 [(word_l, 82, 157)])
    # Small corner logo: app icon only, transparent PNG (Inno Setup 6.3+).
    icon_s = fit(icon, 47, 46)
    png_path = os.path.join(OUT, "wizardsmall.png")
    ok = icon_s.save(png_path, "PNG")
    print(("wrote " if ok else "FAILED ") + png_path)
    rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
