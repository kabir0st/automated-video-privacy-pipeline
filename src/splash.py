"""Startup splash shown while the heavy modules load.

Must import nothing beyond PyQt6 — the point is to get pixels on screen
before cv2/torch/insightface/onnxruntime start importing. The pixmap is
painted at runtime so the frozen build needs no bundled image asset.
"""

import sys

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

# Keep in sync with the Nordic design tokens in ui.py.
_BG       = "#232831"
_BORDER   = "#3E4654"
_TEXT     = "#ECEFF4"
_TEXT_DIM = "#7A8494"
_ACCENT   = "#88C0D0"

_W, _H = 420, 240
_MARGIN = 32


def _paint_pixmap(ratio: float) -> QPixmap:
    pix = QPixmap(int(_W * ratio), int(_H * ratio))
    pix.setDevicePixelRatio(ratio)
    pix.fill(QColor(_BG))

    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    p.setPen(QColor(_BORDER))
    p.drawRect(QRectF(0.5, 0.5, _W - 1, _H - 1))

    p.fillRect(QRectF(_MARGIN, 92, 44, 3), QColor(_ACCENT))

    title_font = QFont()
    title_font.setPointSize(13)
    title_font.setWeight(QFont.Weight.DemiBold)
    p.setFont(title_font)
    p.setPen(QColor(_TEXT))
    p.drawText(
        QRectF(_MARGIN, 104, _W - 2 * _MARGIN, 32),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        "Automated Video Privacy Pipeline",
    )
    p.end()
    return pix


class _Splash(QSplashScreen):
    """QSplashScreen draws messages flush against the pixmap edge; pad them."""

    def drawContents(self, painter: QPainter) -> None:  # type: ignore[override]
        painter.setPen(QColor(_TEXT_DIM))
        painter.drawText(
            QRectF(_MARGIN, _H - 56, _W - 2 * _MARGIN, 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.message(),
        )


# If we create the QApplication it must outlive this call — PyQt destroys
# the C++ application (and every widget) once the wrapper is collected.
_owned_app: QApplication | None = None


def show_splash() -> QSplashScreen:
    global _owned_app
    app = QApplication.instance()
    if app is None:
        app = _owned_app = QApplication(sys.argv)
    screen = app.primaryScreen()
    ratio = screen.devicePixelRatio() if screen else 1.0
    splash = _Splash(_paint_pixmap(ratio))
    # ui.py applies an app-wide stylesheet whose first rule is
    # `QWidget { background-color: transparent; }` while this splash is still
    # alive — that turned the splash transparent (blank/garbled) during the slow
    # window build. An explicit, more-specific per-widget rule (type+id beats the
    # bare `QWidget` selector) keeps it opaque no matter what the app sets later.
    splash.setObjectName("FBISplash")
    splash.setStyleSheet(f"QSplashScreen#FBISplash {{ background-color: {_BG}; }}")
    splash.showMessage("Starting up…")
    splash.show()
    # Paint now — the heavy imports that follow block the event loop.
    app.processEvents()
    # The PyInstaller bootloader's native splash (--splash) covered the onefile
    # extraction gap before any Python ran; hand off to this Qt splash now that
    # it is on screen so the two never overlap. No-op outside a --splash build.
    try:
        import pyi_splash  # type: ignore[import-not-found]
        pyi_splash.close()
    except Exception:
        pass
    return splash


def update(splash: QSplashScreen | None, msg: str) -> None:
    """Set the splash message and repaint it immediately.

    Startup work (imports, model downloads) blocks the event loop, so a
    showMessage alone never paints — processEvents() is required. No-op when
    there is no splash (e.g. ``ui.py`` run directly)."""
    if splash is None:
        return
    splash.showMessage(msg)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
