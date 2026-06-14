from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ManualMappingPanel(QWidget):
    mapping_requested = Signal(str, str)
    export_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.reference_combo = QComboBox()
        self.reference_combo.setEditable(True)
        self.reference_combo.setPlaceholderText("Hub/RefDB Name")
        self.local_roi_combo = QComboBox()
        self.apply_button = QPushButton("Zuordnung speichern")
        self.export_button = QPushButton("JSON exportieren")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Hub / Constraint ROI", "Lokale ROI"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Offen"))
        controls.addWidget(self.reference_combo, 2)
        controls.addWidget(QLabel("Lokale ROI"))
        controls.addWidget(self.local_roi_combo, 2)
        controls.addWidget(self.apply_button)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)
        layout.addWidget(self.table)

        self.apply_button.clicked.connect(self._emit_mapping)
        self.export_button.clicked.connect(self.export_requested)

    def set_mappings(self, mappings: dict[str, str]) -> None:
        items = sorted(mappings.items())
        self.table.setRowCount(len(items))
        for row, (reference_name, local_roi) in enumerate(items):
            self.table.setItem(row, 0, QTableWidgetItem(reference_name))
            self.table.setItem(row, 1, QTableWidgetItem(local_roi))

    def set_options(self, reference_names: list[str], local_roi_names: list[str]) -> None:
        _replace_combo_items(self.reference_combo, sorted(set(filter(None, reference_names))))
        _replace_combo_items(self.local_roi_combo, local_roi_names)

    def _emit_mapping(self) -> None:
        reference_name = self.reference_combo.currentText().strip()
        local_roi = self.local_roi_combo.currentText().strip()
        if reference_name and local_roi:
            self.mapping_requested.emit(reference_name, local_roi)


def _replace_combo_items(combo: QComboBox, values: list[str]) -> None:
    current = combo.currentText()
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(values)
        if current:
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)
            elif values:
                combo.setCurrentIndex(0)
        elif values:
            combo.setCurrentIndex(0)
    finally:
        combo.blockSignals(False)
