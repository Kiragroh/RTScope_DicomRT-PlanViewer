from __future__ import annotations

from typing import Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from planeval_viewer.dicom_io.models import PlanDataset
from planeval_viewer.plan_targets import is_target_roi
from planeval_viewer.refdb.matching import RoiLookup


class RoiPanel(QWidget):
    selection_changed = Signal(str)
    visibility_changed = Signal(str, bool)
    mapping_requested = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.plan: PlanDataset | None = None
        self.lookups: dict[str, RoiLookup] = {}
        self._updating = False

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("ROI suchen")
        self.show_all_button = QPushButton("Alle an")
        self.hide_all_button = QPushButton("Alle aus")
        self.show_matched_button = QPushButton("Nur Matches")
        self.mapping_reference_combo = QComboBox()
        self.mapping_reference_combo.setEditable(True)
        self.mapping_reference_combo.setPlaceholderText("Hub/RefDB Name")
        self.map_selected_button = QPushButton("Zuordnen")
        self.map_selected_button.setToolTip("Ausgewaehlte lokale ROI dem Hub/RefDB-Namen zuordnen")
        self.show_all_button.setObjectName("VisibilityShowAllButton")
        self.hide_all_button.setObjectName("VisibilityHideAllButton")
        self.show_matched_button.setObjectName("VisibilityMatchedButton")
        self.show_all_button.setToolTip("Alle ROI-Konturen und DVH-Linien anzeigen")
        self.hide_all_button.setToolTip("Alle ROI-Konturen und DVH-Linien ausblenden")
        self.show_matched_button.setToolTip("Nur ROIs mit RefDB-Match anzeigen")
        for button in (
            self.show_all_button,
            self.hide_all_button,
            self.show_matched_button,
        ):
            button.setMinimumHeight(40)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "ROI", "Match", "Status", "Farbe"])
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        title = QLabel("Layer / ROIs")
        title.setStyleSheet("font-weight: 700; font-size: 12pt;")
        visibility_tools = QHBoxLayout()
        visibility_tools.setContentsMargins(0, 0, 0, 0)
        visibility_tools.setSpacing(4)
        visibility_tools.addWidget(self.show_all_button)
        visibility_tools.addWidget(self.hide_all_button)
        visibility_tools.addWidget(self.show_matched_button)
        mapping_tools = QHBoxLayout()
        mapping_tools.setContentsMargins(0, 0, 0, 0)
        mapping_tools.setSpacing(4)
        mapping_tools.addWidget(QLabel("Mapping"))
        mapping_tools.addWidget(self.mapping_reference_combo, 1)
        mapping_tools.addWidget(self.map_selected_button)
        layout.addWidget(title)
        layout.addWidget(self.filter_edit)
        layout.addLayout(visibility_tools)
        layout.addLayout(mapping_tools)
        layout.addWidget(self.table)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.table.itemSelectionChanged.connect(self._emit_selection)
        self.table.itemChanged.connect(self._emit_visibility)
        self.show_all_button.clicked.connect(self.show_all)
        self.hide_all_button.clicked.connect(self.hide_all)
        self.show_matched_button.clicked.connect(self.show_matched_only)
        self.map_selected_button.clicked.connect(self._emit_mapping_request)

    def set_plan(self, plan: PlanDataset | None) -> None:
        self.plan = plan
        self._rebuild()

    def set_lookups(self, lookups: Mapping[str, RoiLookup]) -> None:
        self.lookups = dict(lookups)
        self._rebuild()

    def set_mapping_references(self, reference_names: list[str]) -> None:
        current = self.mapping_reference_combo.currentText()
        self.mapping_reference_combo.blockSignals(True)
        try:
            self.mapping_reference_combo.clear()
            values = sorted(set(filter(None, reference_names)))
            self.mapping_reference_combo.addItems(values)
            if current:
                index = self.mapping_reference_combo.findText(current)
                if index >= 0:
                    self.mapping_reference_combo.setCurrentIndex(index)
                else:
                    self.mapping_reference_combo.setEditText(current)
            elif values:
                self.mapping_reference_combo.setCurrentIndex(0)
        finally:
            self.mapping_reference_combo.blockSignals(False)

    def selected_roi_name(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 1)
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def show_all(self) -> None:
        self._set_visibility(lambda _roi_name: True)

    def hide_all(self) -> None:
        self._set_visibility(lambda _roi_name: False)

    def show_matched_only(self) -> None:
        self._set_visibility(
            lambda roi_name: self._is_matched_or_target(roi_name)
        )

    def _is_matched_or_target(self, roi_name: str) -> bool:
        lookup = self.lookups.get(roi_name)
        if lookup is not None and lookup.matched_name:
            return True
        roi = self.plan.roi_by_name(roi_name) if self.plan is not None else None
        return bool(roi is not None and is_target_roi(roi))

    def _rebuild(self) -> None:
        self._updating = True
        try:
            rois = self.plan.rois if self.plan else []
            self.table.setRowCount(len(rois))
            for row, roi in enumerate(rois):
                lookup = self.lookups.get(roi.name)

                visible = QTableWidgetItem()
                visible.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                visible.setCheckState(Qt.CheckState.Checked if roi.visible else Qt.CheckState.Unchecked)
                visible.setData(Qt.ItemDataRole.UserRole, roi.name)
                self.table.setItem(row, 0, visible)

                name = QTableWidgetItem(roi.name)
                name.setData(Qt.ItemDataRole.UserRole, roi.name)
                self.table.setItem(row, 1, name)

                match_name = lookup.matched_name if lookup else ""
                self.table.setItem(row, 2, QTableWidgetItem(match_name))

                status = lookup.status if lookup else "pending"
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor(_status_color(status)))
                self.table.setItem(row, 3, status_item)

                color = (lookup.color if lookup and lookup.color else roi.color) or "#6EA8FE"
                color_item = QTableWidgetItem("  ")
                color_item.setBackground(QColor(color))
                color_item.setToolTip(color)
                self.table.setItem(row, 4, color_item)
        finally:
            self._updating = False
        self._apply_filter()

    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        for row in range(self.table.rowCount()):
            roi = self.table.item(row, 1).text().lower()
            match = self.table.item(row, 2).text().lower()
            self.table.setRowHidden(row, bool(needle and needle not in roi and needle not in match))

    def _emit_selection(self) -> None:
        name = self.selected_roi_name()
        if name:
            lookup = self.lookups.get(name)
            if lookup and lookup.matched_name:
                self.mapping_reference_combo.setEditText(lookup.matched_name)
            self.selection_changed.emit(name)

    def _emit_mapping_request(self) -> None:
        reference_name = self.mapping_reference_combo.currentText().strip()
        roi_name = self.selected_roi_name()
        if reference_name and roi_name:
            self.mapping_requested.emit(reference_name, roi_name)

    def _emit_visibility(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 0:
            return
        roi_name = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        if self.plan is not None:
            roi = self.plan.roi_by_name(roi_name)
            if roi is not None:
                roi.visible = visible
        self.visibility_changed.emit(roi_name, visible)

    def _set_visibility(self, predicate) -> None:
        if self.plan is None:
            return
        changed: list[tuple[str, bool]] = []
        self._updating = True
        try:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item is None:
                    continue
                roi_name = item.data(Qt.ItemDataRole.UserRole)
                visible = bool(predicate(roi_name))
                item.setCheckState(
                    Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
                )
                roi = self.plan.roi_by_name(roi_name)
                if roi is not None and roi.visible != visible:
                    roi.visible = visible
                    changed.append((roi_name, visible))
                elif roi is not None:
                    roi.visible = visible
                    changed.append((roi_name, visible))
        finally:
            self._updating = False
        for roi_name, visible in changed:
            self.visibility_changed.emit(roi_name, visible)


def _status_color(status: str) -> str:
    if status == "matched":
        return "#1f7a3f"
    if status == "not found":
        return "#b7791f"
    if status == "error":
        return "#b42318"
    return "#6b7280"
