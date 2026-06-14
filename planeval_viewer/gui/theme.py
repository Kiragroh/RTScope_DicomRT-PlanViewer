from __future__ import annotations


APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #171b24;
    color: #dce4ef;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QToolBar {
    background: #252a3a;
    border-bottom: 1px solid #4b5368;
    spacing: 6px;
    padding: 4px;
}
QPushButton, QToolButton {
    background: #30364a;
    color: #edf3fb;
    border: 1px solid #59627a;
    border-radius: 4px;
    padding: 5px 9px;
}
QPushButton:hover, QToolButton:hover {
    background: #3c4560;
}
QPushButton:pressed, QToolButton:pressed {
    background: #53617f;
}
QTabWidget::pane {
    background: #0b0f16;
    border: 1px solid #475064;
    border-radius: 4px;
}
QTabBar::tab {
    background: #30364a;
    color: #dce4ef;
    border: 1px solid #59627a;
    border-bottom: 0;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    padding: 7px 12px;
}
QTabBar::tab:selected {
    background: #64b5ff;
    color: #061017;
    border: 2px solid #d9edff;
    font-weight: 800;
}
QTabBar::tab:hover {
    background: #3c4560;
}
#DvhResultsTabs QTabBar::tab {
    background: #334155;
    color: #e2e8f0;
    border: 1px solid #64748b;
    padding: 8px 14px;
    font-weight: 700;
}
#DvhResultsTabs QTabBar::tab:selected {
    background: #facc15;
    color: #111827;
    border: 2px solid #fde68a;
}
#DvhResultsTabs QTabBar::tab:hover {
    background: #475569;
}
#VisibilityShowAllButton, #VisibilityHideAllButton, #VisibilityMatchedButton {
    color: #061017;
    font-weight: 800;
    border-radius: 6px;
    padding: 8px 10px;
}
#VisibilityShowAllButton {
    background: #31c46b;
    border: 2px solid #b8ffd0;
}
#VisibilityShowAllButton:hover {
    background: #4ee885;
}
#VisibilityHideAllButton {
    background: #ff6574;
    border: 2px solid #ffd1d6;
}
#VisibilityHideAllButton:hover {
    background: #ff8390;
}
#VisibilityMatchedButton {
    background: #64b5ff;
    border: 2px solid #d9edff;
}
#VisibilityMatchedButton:hover {
    background: #86c6ff;
}
QLabel {
    color: #dce4ef;
}
QLineEdit, QTextEdit, QTableWidget, QPlainTextEdit {
    background: #0b0f16;
    color: #dce4ef;
    border: 1px solid #475064;
    border-radius: 4px;
    selection-background-color: #0d6efd;
}
QLineEdit {
    padding: 5px 7px;
}
QHeaderView::section {
    background: #2f3547;
    color: #edf3fb;
    border: 0;
    border-right: 1px solid #4b5368;
    border-bottom: 1px solid #4b5368;
    padding: 5px;
    font-weight: 600;
}
QTableWidget {
    gridline-color: #313849;
    alternate-background-color: #111722;
}
QTableWidget::item:selected {
    background: #0d6efd;
    color: #ffffff;
}
QSplitter::handle {
    background: #3b4356;
}
QScrollBar:vertical {
    background: #0b0f16;
    border: 1px solid #475064;
    border-radius: 4px;
    width: 18px;
}
QScrollBar::handle:vertical {
    background: #6b7894;
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QSlider::groove:horizontal {
    background: #111722;
    border: 1px solid #475064;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #8ab4f8;
    border: 1px solid #a8c7fa;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
#ViewerShell, #ThreeDShell, #DvhShell, #QaShell {
    background: #05070b;
    border: 1px solid #4b5368;
    border-radius: 4px;
}
#PaneTitle {
    background: #05070b;
    color: #edf3fb;
    font-weight: 700;
    padding: 2px 4px;
}
#ThreeDTargetsCheckbox, #ThreeDOarsCheckbox {
    background: #111827;
    color: #e5edf8;
    border: 1px solid #53617f;
    border-radius: 4px;
    padding: 3px 7px;
}
#ThreeDTargetsCheckbox::indicator, #ThreeDOarsCheckbox::indicator {
    width: 13px;
    height: 13px;
}
#ThreeDTargetsCheckbox::indicator:checked, #ThreeDOarsCheckbox::indicator:checked {
    background: #64b5ff;
    border: 1px solid #d9edff;
}
#ThreeDTargetsCheckbox::indicator:unchecked, #ThreeDOarsCheckbox::indicator:unchecked {
    background: #05070b;
    border: 1px solid #6b7894;
}
#DoseScalePanel {
    background: #05070b;
    border-left: 1px solid #2f3547;
}
#DoseColorScale {
    background: qlineargradient(
        x1:0, y1:1, x2:0, y2:0,
        stop:0 #1d4ed8,
        stop:0.35 #22c55e,
        stop:0.65 #facc15,
        stop:1 #ef4444
    );
    border: 1px solid #475064;
    border-radius: 3px;
}
#DoseScaleValue, #CursorDoseLabel {
    color: #b8c4d6;
    font-size: 9pt;
}
#ViewerTitle {
    background: #05070b;
    color: #edf3fb;
    font-size: 12pt;
    font-weight: 600;
    padding: 6px 8px;
}
#StatusLog {
    background: #0b0f16;
    color: #b8c4d6;
    border-top: 1px solid #4b5368;
    padding: 6px 10px;
}
QStatusBar {
    background: #0b0f16;
    color: #dce4ef;
    border-top: 1px solid #4b5368;
}
#BusyIndicator {
    background: #0b0f16;
    border: 0;
    border-top: 1px solid #4b5368;
    padding: 4px;
}
#BusyIndicator::chunk {
    background: #64b5ff;
}
#LoadingOverlay {
    background: rgba(5, 7, 11, 220);
}
#LoadingMessage {
    color: #edf3fb;
    font-size: 14pt;
    font-weight: 700;
    padding: 10px;
}
#LoadingOverlayProgress {
    background: #0b0f16;
    border: 1px solid #53617f;
    border-radius: 4px;
    min-height: 10px;
}
#LoadingOverlayProgress::chunk {
    background: #64b5ff;
}
"""
