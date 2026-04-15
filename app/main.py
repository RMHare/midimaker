"""
app/main.py
============
Entry point for the MidiMaker application.

Responsibilities:
- Validate Python version (≥ 3.10)
- Initialise loguru logging
- Build a QApplication with a dark Fusion palette
- Show a splash screen while loading
- Detect CUDA availability
- Launch MainWindow
- Catch unhandled exceptions and display user-friendly dialogs
"""

from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Python version check (before any other imports)
# ---------------------------------------------------------------------------

if sys.version_info < (3, 10):
    print(
        f"MidiMaker requires Python 3.10 or later. "
        f"You are running Python {sys.version}.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so "from core.xxx" works
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports (after path is set)
# ---------------------------------------------------------------------------

from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QSplashScreen,
    QWidget,
)


# ---------------------------------------------------------------------------
# Dark Fusion palette
# ---------------------------------------------------------------------------

def _apply_dark_palette(app: QApplication) -> None:
    """Apply a custom dark colour palette to the QApplication."""
    app.setStyle("Fusion")

    palette = QPalette()
    dark = QColor(22, 22, 30)
    dark_mid = QColor(30, 30, 40)
    dark_light = QColor(45, 45, 58)
    mid = QColor(60, 60, 76)
    text = QColor(220, 220, 228)
    dim_text = QColor(140, 140, 155)
    highlight = QColor(85, 136, 204)
    highlight_text = QColor(255, 255, 255)
    link = QColor(100, 170, 255)
    btn = QColor(38, 38, 52)
    btn_text = QColor(210, 210, 220)

    palette.setColor(QPalette.ColorRole.Window,          dark)
    palette.setColor(QPalette.ColorRole.WindowText,      text)
    palette.setColor(QPalette.ColorRole.Base,            dark_mid)
    palette.setColor(QPalette.ColorRole.AlternateBase,   dark_light)
    palette.setColor(QPalette.ColorRole.ToolTipBase,     dark_mid)
    palette.setColor(QPalette.ColorRole.ToolTipText,     text)
    palette.setColor(QPalette.ColorRole.Text,            text)
    palette.setColor(QPalette.ColorRole.Button,          btn)
    palette.setColor(QPalette.ColorRole.ButtonText,      btn_text)
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link,            link)
    palette.setColor(QPalette.ColorRole.Highlight,       highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, highlight_text)
    palette.setColor(QPalette.ColorRole.Dark,            dark)
    palette.setColor(QPalette.ColorRole.Mid,             mid)
    palette.setColor(QPalette.ColorRole.Shadow,          QColor(5, 5, 10))
    palette.setColor(QPalette.ColorRole.Light,           dark_light)
    palette.setColor(QPalette.ColorRole.Midlight,        dark_mid)
    palette.setColor(QPalette.ColorRole.PlaceholderText, dim_text)

    # Disabled state
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, dim_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dim_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dim_text)

    app.setPalette(palette)

    # Global stylesheet additions
    app.setStyleSheet(
        "QToolTip { color: #e0e0e0; background: #252535; border: 1px solid #555; padding: 4px; }"
        "QScrollBar:vertical { background: #1e1e28; width: 12px; margin: 0; }"
        "QScrollBar::handle:vertical { background: #3a3a50; border-radius: 5px; min-height: 20px; }"
        "QScrollBar::handle:vertical:hover { background: #4a4a60; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        "QScrollBar:horizontal { background: #1e1e28; height: 12px; margin: 0; }"
        "QScrollBar::handle:horizontal { background: #3a3a50; border-radius: 5px; min-width: 20px; }"
        "QScrollBar::handle:horizontal:hover { background: #4a4a60; }"
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"
        "QPushButton { "
        "  background: #2d3a4a; color: #ccc; border-radius: 4px; "
        "  padding: 5px 12px; border: 1px solid #3a4a5a; "
        "}"
        "QPushButton:hover { background: #3a4a5a; color: white; }"
        "QPushButton:pressed { background: #22303e; }"
        "QPushButton:disabled { background: #1e2530; color: #444; border-color: #2a3040; }"
        "QLineEdit, QTextEdit, QPlainTextEdit { "
        "  background: #1e1e28; color: #ddd; border: 1px solid #3a3a50; "
        "  border-radius: 3px; padding: 4px; "
        "}"
        "QLineEdit:focus, QTextEdit:focus { border-color: #5588cc; }"
        "QComboBox { "
        "  background: #1e1e28; color: #ddd; border: 1px solid #3a3a50; "
        "  border-radius: 3px; padding: 4px 8px; "
        "}"
        "QComboBox::drop-down { border: none; width: 20px; }"
        "QComboBox QAbstractItemView { background: #252535; color: #ddd; "
        "  selection-background-color: #2d3a52; }"
        "QSpinBox, QDoubleSpinBox { "
        "  background: #1e1e28; color: #ddd; border: 1px solid #3a3a50; "
        "  border-radius: 3px; padding: 3px 6px; "
        "}"
        "QSlider::groove:horizontal { background: #2a2a38; height: 4px; border-radius: 2px; }"
        "QSlider::handle:horizontal { background: #5588cc; width: 14px; height: 14px; "
        "  border-radius: 7px; margin: -5px 0; }"
        "QSlider::sub-page:horizontal { background: #4477aa; border-radius: 2px; }"
        "QCheckBox { color: #ccc; spacing: 6px; }"
        "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; "
        "  border-radius: 2px; background: #1e1e28; }"
        "QCheckBox::indicator:checked { background: #5588cc; border-color: #5588cc; }"
        "QTabBar::tab { font-family: 'Segoe UI', Arial, sans-serif; }"
    )


# ---------------------------------------------------------------------------
# Splash screen
# ---------------------------------------------------------------------------

def _build_splash() -> QSplashScreen:
    """Build a simple dark splash screen."""
    from PySide6.QtGui import QPixmap, QPainter
    from PySide6.QtCore import QRect

    pm = QPixmap(520, 280)
    pm.fill(QColor(18, 18, 26))

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Background gradient strip
    from PySide6.QtGui import QLinearGradient
    grad = QLinearGradient(0, 0, 520, 0)
    grad.setColorAt(0, QColor(30, 50, 90))
    grad.setColorAt(1, QColor(18, 18, 26))
    p.fillRect(0, 0, 520, 6, grad)

    # Title
    p.setPen(QColor(220, 230, 255))
    font = QFont("Arial", 28, QFont.Weight.Bold)
    p.setFont(font)
    p.drawText(QRect(0, 60, 520, 60), Qt.AlignmentFlag.AlignCenter, "🎹 MidiMaker")

    # Subtitle
    p.setPen(QColor(140, 160, 200))
    sub_font = QFont("Arial", 13)
    p.setFont(sub_font)
    p.drawText(
        QRect(0, 130, 520, 36),
        Qt.AlignmentFlag.AlignCenter,
        "Symbolic Music Generation Workstation",
    )

    # Version
    p.setPen(QColor(80, 100, 140))
    ver_font = QFont("Arial", 10)
    p.setFont(ver_font)
    p.drawText(
        QRect(0, 180, 520, 28),
        Qt.AlignmentFlag.AlignCenter,
        "v0.1.0  |  Python + PySide6 + PyTorch",
    )

    p.setPen(QColor(60, 90, 140))
    p.drawText(
        QRect(0, 240, 520, 28),
        Qt.AlignmentFlag.AlignCenter,
        "Loading…",
    )
    p.end()

    splash = QSplashScreen(pm, Qt.WindowType.WindowStaysOnTopHint)
    splash.setFont(QFont("Arial", 10))
    return splash


# ---------------------------------------------------------------------------
# CUDA info
# ---------------------------------------------------------------------------

def _check_cuda() -> str:
    """Return a short string describing CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"CUDA available: {name}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS available"
        # Distinguish CPU-only build from driver/GPU absence
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if not cuda_version:
            return (
                "No GPU — PyTorch was installed without CUDA support. "
                "Re-run setup.bat to install the CUDA-enabled build."
            )
        return f"No GPU detected (PyTorch CUDA {cuda_version} present — check drivers)"
    except ImportError:
        return "PyTorch not installed"


# ---------------------------------------------------------------------------
# Global exception hook
# ---------------------------------------------------------------------------

def _install_exception_hook(app: QApplication) -> None:
    """Install a global exception handler that shows a dialog instead of crashing."""

    def _hook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical(f"Unhandled exception:\n{msg}")
        # Only show dialog if a QApplication exists and event loop is running
        try:
            dlg = QMessageBox()
            dlg.setWindowTitle("Unexpected Error")
            dlg.setIcon(QMessageBox.Icon.Critical)
            dlg.setText(
                "<b>An unexpected error occurred.</b><br><br>"
                "MidiMaker has encountered an error and may not be able to continue. "
                "Please save your work if possible."
            )
            dlg.setDetailedText(msg)
            dlg.exec()
        except Exception:
            pass
        # Call the original hook
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """Configure loguru for the application."""
    from core.utils import setup_logging
    log_dir = Path.home() / ".midimaker" / "logs"
    setup_logging(log_dir=log_dir, level="DEBUG")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Start the MidiMaker application. Returns the process exit code."""

    # Enable Hi-DPI scaling
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("MidiMaker")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("MidiMaker")

    _apply_dark_palette(app)
    _install_exception_hook(app)
    _setup_logging()

    logger.info(f"MidiMaker starting — Python {sys.version.split()[0]}")
    cuda_info = _check_cuda()
    logger.info(f"Compute: {cuda_info}")

    # Splash screen
    splash = _build_splash()
    splash.show()
    splash.showMessage(
        f"  {cuda_info}",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        QColor(140, 160, 200),
    )
    app.processEvents()

    # Import and construct main window (may take a moment if models need init)
    try:
        from app.main_window import MainWindow
        window = MainWindow()
    except Exception as exc:
        splash.finish(None)
        logger.critical(f"Failed to launch MainWindow: {exc}")
        QMessageBox.critical(
            None,
            "Launch Failed",
            f"MidiMaker failed to start:\n\n{exc}\n\n"
            "Check that all dependencies are installed (pip install -r requirements.txt).",
        )
        return 1

    # Close splash and show window
    splash.finish(window)
    window.show()

    logger.info("MainWindow shown. Entering event loop.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
