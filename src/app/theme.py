"""Design tokens and the application stylesheet (dark, high contrast)."""
from __future__ import annotations

BG = "#15181e"
PANEL = "#1d2129"
PANEL_ALT = "#242a34"
BORDER = "#333b48"
TEXT = "#e6eaf0"
TEXT_DIM = "#8a94a6"
ACCENT = "#7fc4de"
ACCENT_2 = "#c8a4ff"
OK = "#79c37e"
WARN = "#e2b04a"
BAD = "#e2645a"
MANUAL = "#f2a35c"
SELECT = "#ffffff"

# Track lane colours by state (RGB tuples for QColor / BGR for cv2 overlays).
LANE_KEPT = (121, 195, 126)
LANE_SUSPECT = (226, 176, 74)
LANE_OFF = (110, 118, 132)
LANE_MANUAL = (242, 163, 92)


def lane_rgb(kept: bool, suspicion: float, manual: bool = False) -> tuple[int, int, int]:
    if manual:
        return LANE_MANUAL
    if not kept:
        return LANE_OFF
    if suspicion >= 0.5:
        return LANE_SUSPECT
    return LANE_KEPT


STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QToolBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; spacing: 6px; padding: 4px; }}
QToolButton {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 10px; }}
QToolButton:hover {{ border-color: {ACCENT}; }}
QToolButton:checked {{ background: {ACCENT}; color: {BG}; }}
QPushButton {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; }}
QPushButton#primary {{ background: {ACCENT}; color: {BG}; font-weight: 600; }}
QPushButton#danger {{ border-color: {BAD}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 8px; }}
QComboBox QAbstractItemView {{ background: {PANEL_ALT}; selection-background-color: {ACCENT}; selection-color: {BG}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {TEXT_DIM}; font-size: 11px; letter-spacing: 1px; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#h1 {{ font-size: 15px; font-weight: 600; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {TEXT_DIM}; }}
QProgressBar {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QSplitter::handle {{ background: {BORDER}; }}
QScrollBar:vertical {{ background: {PANEL}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {PANEL}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 24px; }}
QListWidget {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QListWidget::item:selected {{ background: {PANEL_ALT}; color: {TEXT}; }}
QCheckBox {{ spacing: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; background: {ACCENT}; border-radius: 7px; }}
QMenu {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT}; color: {BG}; }}
QToolTip {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""
