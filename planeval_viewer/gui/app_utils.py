from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from PySide6.QtGui import QIcon


APP_ORG = "RTScope"
APP_NAME = "PlanEvalViewer"
APP_USER_MODEL_ID = "RTScope.PlanEvalViewer"


def logo_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logo.png"


def application_icon() -> QIcon:
    path = logo_path()
    return QIcon(str(path)) if path.exists() else QIcon()


def configure_windows_taskbar_icon() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        return


def detach_console_window() -> bool:
    if sys.platform != "win32":
        return False
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
        return bool(ctypes.windll.kernel32.FreeConsole() or hwnd)
    except Exception:
        return False
