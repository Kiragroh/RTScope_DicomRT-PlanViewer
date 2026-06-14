from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from planeval_viewer.gui.app_utils import application_icon, configure_windows_taskbar_icon
from planeval_viewer.gui.main_window import MainWindow


def _excepthook(exc_type, exc_value, exc_tb):
    message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Unexpected Error")
        box.setText("An unexpected error occurred.")
        box.setDetailedText(message)
        box.exec()
    except Exception:
        pass


def main() -> int:
    sys.excepthook = _excepthook
    configure_windows_taskbar_icon()
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("PlanEval Viewer")
    icon = application_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
