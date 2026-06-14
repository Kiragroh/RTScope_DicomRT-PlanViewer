from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from planeval_viewer.computations.bev_geometry import plan_aperture_modulation_score
from planeval_viewer.computations.complexity import summarize_beam_complexity
from planeval_viewer.computations.constraints import (
    compute_constraint_metric,
    constraint_passes,
)
from planeval_viewer.computations.dvh import DvhCurve, compute_all_dvhs
from planeval_viewer.computations.target_metrics import compute_paddick_metrics_for_roi
from planeval_viewer.dicom_io.models import PlanDataset, RoiGeometry
from planeval_viewer.gui.bev_panel import BevPanel
from planeval_viewer.plan_targets import is_target_roi, select_default_target_name
from planeval_viewer.refdb.matching import RoiLookup
from planeval_viewer.refdb.models import ConstraintRow, ConstraintTable


class QAPanel(QWidget):
    status_message = Signal(str)
    target_changed = Signal(str)
    manual_mapping_requested = Signal(str, str)
    missing_constraints_changed = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.plan: PlanDataset | None = None
        self.lookups: dict[str, RoiLookup] = {}
        self.dvhs: dict[str, DvhCurve] = {}
        self._dvh_plan: PlanDataset | None = None
        self.visible_rois: set[str] = set()
        self.dvh_plot_items: dict[str, pg.PlotDataItem] = {}
        self.constraint_tables: dict[str, ConstraintTable] = {}
        self.manual_mappings: dict[str, str] = {}
        self.missing_constraints: list[str] = []
        self.dvh_legend = None

        self.compute_button = QPushButton("Compute QA")
        self.target_combo = QComboBox()
        self.constraint_table_combo = QComboBox()
        self.missing_constraint_combo = QComboBox()
        self.manual_roi_combo = QComboBox()
        self.apply_mapping_button = QPushButton("Zuordnen")
        self.missing_constraints_label = QLabel("")
        self.missing_constraints_label.setWordWrap(True)
        self.dvh_plot = pg.PlotWidget()
        self.dvh_plot.setBackground("#05070b")
        self.dvh_plot.showGrid(x=True, y=True, alpha=0.18)
        self.dvh_plot.setLabel("bottom", "Dose", units="Gy")
        self.dvh_plot.setLabel("left", "Volume", units="%")

        self.dvh_table = _table(["ROI", "Volume cc", "Dmin", "Dmean", "Dmax", "D95"])
        self.evaluation_table = _table(
            [
                "Constraint ROI",
                "Local ROI",
                "Metric",
                "Value",
                "Unit",
                "Optimal",
                "Maximal",
                "Status",
                "Table",
            ]
        )
        self.plan_info_table = _table(["Metric", "Value"])
        self.metrics_table = _table(["Metric", "Value"])
        self.complexity_table = _table(["Beam", "CP", "MU", "Mean area", "Leaf travel", "PAM"])
        self.bev_panel = BevPanel()

        self.header_widget = QWidget()
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.compute_button)
        header.addWidget(QLabel("Target"))
        header.addWidget(self.target_combo, 1)

        self.dvh_page = QWidget()
        dvh_layout = QVBoxLayout(self.dvh_page)
        dvh_layout.setContentsMargins(8, 8, 8, 8)
        dvh_layout.addWidget(self.dvh_plot, 2)
        self.dvh_results_tabs = self._build_dvh_results_tabs()
        dvh_layout.addWidget(self.dvh_results_tabs, 1)

        self.plan_page = QWidget()
        plan_layout = QVBoxLayout(self.plan_page)
        plan_layout.setContentsMargins(8, 8, 8, 8)
        plan_layout.addWidget(QLabel("Plan info"))
        plan_layout.addWidget(self.plan_info_table, 1)
        plan_layout.addWidget(QLabel("Paddick / target metrics"))
        plan_layout.addWidget(self.metrics_table, 1)
        plan_layout.addWidget(QLabel("Plan complexity"))
        plan_layout.addWidget(self.complexity_table, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.dvh_page, "DVH")
        self.tabs.addTab(self.plan_page, "Plan")
        self.tabs.addTab(self.bev_panel, "MLC / BEV")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.header_widget)
        layout.addWidget(self.tabs, 1)

        self.compute_button.clicked.connect(self.compute_qa)
        self.target_combo.currentTextChanged.connect(lambda _text: self._target_changed())
        self.constraint_table_combo.currentIndexChanged.connect(
            lambda _index: self._populate_evaluation_table()
        )
        self.apply_mapping_button.clicked.connect(self._apply_manual_mapping)
        self._populate_constraint_table_combo()

    def set_plan(self, plan: PlanDataset | None) -> None:
        self.plan = plan
        self.dvhs = {}
        self._dvh_plan = None
        self.dvh_plot_items = {}
        self.visible_rois = {roi.name for roi in plan.rois if roi.visible} if plan else set()
        self._populate_target_combo()
        self._populate_manual_roi_combo()
        self._clear_tables()
        self.dvh_plot.clear()
        self.bev_panel.set_plan(plan)
        self.bev_panel.set_target_roi(self._selected_target_roi())
        self._populate_constraint_table_combo()
        self._populate_plan_info_table()
        self._populate_complexity_table()
        self._set_missing_constraints([])

    def set_lookups(self, lookups: Mapping[str, RoiLookup]) -> None:
        self.lookups = dict(lookups)
        self._populate_constraint_table_combo()
        if self.dvhs:
            self._populate_evaluation_table()

    def set_manual_mappings(self, mappings: Mapping[str, str]) -> None:
        self.manual_mappings = dict(mappings)
        if self.dvhs:
            self._populate_evaluation_table()

    def set_roi_visible(self, roi_name: str, visible: bool) -> None:
        if visible:
            self.visible_rois.add(roi_name)
        else:
            self.visible_rois.discard(roi_name)
        item = self.dvh_plot_items.get(roi_name)
        if item is not None:
            item.setVisible(visible)
        if self.dvhs:
            self._populate_dvh_table()
            self._populate_evaluation_table()

    def compute_qa(self) -> None:
        if self.plan is None:
            self.status_message.emit("No plan loaded for QA.")
            return
        if self.plan.dose is None:
            self.status_message.emit("No RTDOSE available for QA.")
            return

        if self.dvhs and self._dvh_plan is self.plan:
            self.status_message.emit("Refreshing QA tables...")
        else:
            self.status_message.emit("Computing DVH and plan QA...")
            self.dvhs = compute_all_dvhs(self.plan, bin_width_gy=0.5)
            self._dvh_plan = self.plan
        self._populate_dvh_plot()
        self._populate_dvh_table()
        self._populate_evaluation_table()
        self._populate_plan_info_table()
        self._populate_metrics_table()
        self._populate_complexity_table()
        self.status_message.emit(f"QA computed for {len(self.dvhs)} ROIs.")

    def _build_dvh_results_tabs(self) -> QTabWidget:
        results_tabs = QTabWidget()
        results_tabs.setObjectName("DvhResultsTabs")
        results_tabs.addTab(self.dvh_table, "DVH table")

        constraint_page = QWidget()
        layout = QVBoxLayout(constraint_page)
        layout.setContentsMargins(0, 0, 0, 0)
        selector = QHBoxLayout()
        selector.setContentsMargins(0, 0, 0, 0)
        selector.addWidget(QLabel("Hub table"))
        selector.addWidget(self.constraint_table_combo, 1)
        layout.addLayout(selector)
        layout.addWidget(self.evaluation_table)
        layout.addWidget(self.missing_constraints_label)
        manual_row = QHBoxLayout()
        manual_row.setContentsMargins(0, 0, 0, 0)
        manual_row.addWidget(QLabel("Fehlend"))
        manual_row.addWidget(self.missing_constraint_combo, 1)
        manual_row.addWidget(QLabel("Lokale ROI"))
        manual_row.addWidget(self.manual_roi_combo, 1)
        manual_row.addWidget(self.apply_mapping_button)
        layout.addLayout(manual_row)
        results_tabs.addTab(constraint_page, "Constraint check")
        return results_tabs

    def _target_changed(self) -> None:
        self.bev_panel.set_target_roi(self._selected_target_roi())
        self._populate_metrics_table()
        self._populate_complexity_table()
        self.target_changed.emit(self.selected_target_name())

    def _populate_target_combo(self) -> None:
        current = self.target_combo.currentText()
        self.target_combo.blockSignals(True)
        try:
            self.target_combo.clear()
            rois = self.plan.rois if self.plan else []
            targets = [roi.name for roi in rois if is_target_roi(roi)]
            target_set = set(targets)
            names = targets + [roi.name for roi in rois if roi.name not in target_set]
            self.target_combo.addItems(names)
            desired = current or select_default_target_name(self.plan)
            if desired:
                index = self.target_combo.findText(desired)
                if index >= 0:
                    self.target_combo.setCurrentIndex(index)
        finally:
            self.target_combo.blockSignals(False)

    def _populate_manual_roi_combo(self) -> None:
        current = self.manual_roi_combo.currentText()
        self.manual_roi_combo.blockSignals(True)
        try:
            self.manual_roi_combo.clear()
            self.manual_roi_combo.addItems([roi.name for roi in self.plan.rois] if self.plan else [])
            if current:
                index = self.manual_roi_combo.findText(current)
                if index >= 0:
                    self.manual_roi_combo.setCurrentIndex(index)
        finally:
            self.manual_roi_combo.blockSignals(False)

    def _populate_constraint_table_combo(self) -> None:
        current = self.constraint_table_combo.currentData()
        self.constraint_table_combo.blockSignals(True)
        try:
            self.constraint_table_combo.clear()
            self.constraint_tables = {}
            for roi_name, lookup in sorted(self.lookups.items()):
                if lookup.result is None:
                    continue
                for table in lookup.result.constraint_tables:
                    if not table.constraints:
                        continue
                    key = _table_key(table)
                    if key in self.constraint_tables:
                        self.constraint_tables[key] = _merge_constraint_table(
                            self.constraint_tables[key],
                            table,
                        )
                        continue
                    self.constraint_tables[key] = table
                    label = table.name or str(table.id or "Table")
                    self.constraint_table_combo.addItem(label, key)
            if current is not None:
                for index in range(self.constraint_table_combo.count()):
                    if self.constraint_table_combo.itemData(index) == current:
                        self.constraint_table_combo.setCurrentIndex(index)
                        break
        finally:
            self.constraint_table_combo.blockSignals(False)

    def _populate_dvh_plot(self) -> None:
        self.dvh_plot.clear()
        self.dvh_plot_items = {}
        if self.plan is None:
            return
        for roi in self.plan.rois:
            dvh = self.dvhs.get(roi.name)
            if dvh is None or dvh.dose_bins_gy.size == 0:
                continue
            color = _roi_color(roi, self.lookups.get(roi.name))
            item = self.dvh_plot.plot(
                dvh.dose_bins_gy,
                dvh.volume_pct,
                pen=pg.mkPen(color, width=2),
            )
            item.setVisible(roi.name in self.visible_rois)
            self.dvh_plot_items[roi.name] = item

    def _populate_dvh_table(self) -> None:
        rows = [dvh for dvh in self._visible_dvhs() if dvh.dose_bins_gy.size > 0]
        self.dvh_table.setRowCount(len(rows))
        for row, dvh in enumerate(rows):
            values = [
                dvh.roi_name,
                _fmt(dvh.volume_cc),
                _fmt(dvh.dmin),
                _fmt(dvh.dmean),
                _fmt(dvh.dmax),
                _fmt(dvh.dose_at_volume_pct(95)),
            ]
            _set_row(self.dvh_table, row, values)

    def _populate_evaluation_table(self) -> None:
        selected_key = self.constraint_table_combo.currentData()
        rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        missing: list[str] = []
        if selected_key is None:
            self.evaluation_table.setRowCount(0)
            self.missing_constraints_label.clear()
            self._set_missing_constraints([])
            return
        table = self.constraint_tables.get(str(selected_key))
        if table is None:
            self.evaluation_table.setRowCount(0)
            self.missing_constraints_label.clear()
            self._set_missing_constraints([])
            return
        for constraint in table.constraints:
            constraint_name = constraint.oar_raw or constraint.source
            local_roi = self._local_roi_for_constraint(constraint_name)
            if not local_roi or local_roi not in self.dvhs:
                missing.append(constraint_name or constraint.metric)
                rows.append(
                    (
                        constraint_name,
                        "",
                        constraint.metric,
                        "",
                        constraint.unit,
                        _fmt(constraint.limit_optimal),
                        _fmt(constraint.limit_maximal),
                        "missing",
                        table.name or constraint.source,
                    )
                )
                continue
            value = compute_constraint_metric(self.dvhs[local_roi], constraint)
            rows.append(
                (
                    constraint_name,
                    local_roi,
                    constraint.metric,
                    _fmt(value),
                    constraint.unit,
                    _fmt(constraint.limit_optimal),
                    _fmt(constraint.limit_maximal),
                    constraint_passes(value, constraint),
                    table.name or constraint.source,
                )
            )
        self.evaluation_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            _set_row(self.evaluation_table, row, values)
            item = self.evaluation_table.item(row, 7)
            if item is not None:
                item.setForeground(QColor(_status_color(item.text())))
        if missing:
            self.missing_constraints_label.setText(
                "Fehlende Zuordnung: " + ", ".join(sorted(set(filter(None, missing))))
            )
        else:
            self.missing_constraints_label.clear()
        self._set_missing_constraints(missing)

    def _local_roi_for_constraint(self, constraint_name: str) -> str:
        normalized = _normalize_name(constraint_name)
        if not normalized:
            return ""
        manual = self.manual_mappings.get(constraint_name)
        if manual:
            return manual
        for reference_name, local_roi in self.manual_mappings.items():
            if _names_match(reference_name, constraint_name):
                return local_roi
        if self.plan is not None:
            for roi in self.plan.rois:
                if _normalize_name(roi.name) == normalized:
                    return roi.name
        for roi_name, lookup in self.lookups.items():
            candidates = [
                lookup.matched_name,
                lookup.reference_name,
                lookup.source_name,
                *lookup.aliases,
            ]
            if any(_names_match(candidate, constraint_name) for candidate in candidates):
                return roi_name
        if self.plan is not None:
            for roi in self.plan.rois:
                if _names_match(roi.name, constraint_name):
                    return roi.name
        return ""

    def _populate_missing_constraint_combo(self, missing: list[str]) -> None:
        current = self.missing_constraint_combo.currentText()
        self.missing_constraint_combo.blockSignals(True)
        try:
            self.missing_constraint_combo.clear()
            self.missing_constraint_combo.addItems(sorted(set(filter(None, missing))))
            if current:
                index = self.missing_constraint_combo.findText(current)
                if index >= 0:
                    self.missing_constraint_combo.setCurrentIndex(index)
        finally:
            self.missing_constraint_combo.blockSignals(False)

    def _set_missing_constraints(self, missing: list[str]) -> None:
        names = sorted(set(filter(None, missing)))
        self.missing_constraints = names
        self._populate_missing_constraint_combo(names)
        self.missing_constraints_changed.emit(names)

    def missing_constraint_names(self) -> list[str]:
        return list(self.missing_constraints)

    def _apply_manual_mapping(self) -> None:
        reference_name = self.missing_constraint_combo.currentText().strip()
        local_roi = self.manual_roi_combo.currentText().strip()
        if reference_name and local_roi:
            self.manual_mapping_requested.emit(reference_name, local_roi)

    def _populate_plan_info_table(self) -> None:
        if self.plan is None:
            self.plan_info_table.setRowCount(0)
            return
        info = self.plan.plan_info
        total_mu = sum(beam.meterset or 0.0 for beam in self.plan.beams)
        control_points = sum(len(beam.control_points) for beam in self.plan.beams)
        rows = [
            ("Plan", str(info.get("plan_label") or info.get("plan_name") or "")),
            ("Fractions", str(info.get("number_of_fractions") or "")),
            ("Prescription Gy", _fmt(_safe_float(info.get("prescription_dose_gy")))),
            ("Dose/Fx Gy", _fmt(_safe_float(info.get("dose_per_fraction_gy")))),
            ("Total MU", _fmt(total_mu)),
            ("Beams", str(len(self.plan.beams))),
            ("Control points", str(control_points)),
        ]
        self.plan_info_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            _set_row(self.plan_info_table, row, values)

    def _populate_metrics_table(self) -> None:
        self.metrics_table.setRowCount(0)
        if self.plan is None:
            return
        prescription = _safe_float(self.plan.plan_info.get("prescription_dose_gy"))
        target = self._selected_target_roi()
        if prescription is None or target is None:
            return
        metrics = compute_paddick_metrics_for_roi(self.plan, target, prescription)
        if metrics is None:
            return
        rows = [
            ("Target", target.name),
            ("Prescription Gy", _fmt(prescription)),
            ("Target volume cc", _fmt(metrics.target_volume_cc)),
            ("Prescription isodose volume cc", _fmt(metrics.prescription_isodose_volume_cc)),
            ("Target covered volume cc", _fmt(metrics.target_covered_volume_cc)),
            ("Coverage", _fmt(metrics.coverage)),
            ("Selectivity", _fmt(metrics.selectivity)),
            ("Paddick CI", _fmt(metrics.paddick_ci)),
            ("Gradient index", _fmt(metrics.gradient_index)),
            ("Homogeneity index", _fmt(metrics.homogeneity_index)),
            ("D2 Gy", _fmt(metrics.d2_gy)),
            ("D98 Gy", _fmt(metrics.d98_gy)),
        ]
        self.metrics_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            _set_row(self.metrics_table, row, values)

    def _populate_complexity_table(self) -> None:
        beams = self.plan.beams if self.plan else []
        target = self._selected_target_roi()
        self.complexity_table.setRowCount(len(beams))
        for row, beam in enumerate(beams):
            summary = summarize_beam_complexity(beam)
            pam = plan_aperture_modulation_score(beam, target)
            _set_row(
                self.complexity_table,
                row,
                [
                    summary.beam_name,
                    str(summary.control_points),
                    _fmt(beam.meterset),
                    _fmt(summary.mean_area),
                    _fmt(summary.total_leaf_travel),
                    _fmt(pam),
                ],
            )

    def _selected_target_roi(self) -> RoiGeometry | None:
        if self.plan is None:
            return None
        name = self.selected_target_name()
        return self.plan.roi_by_name(name) if name else None

    def selected_target_name(self) -> str:
        return self.target_combo.currentText().strip()

    def _visible_dvhs(self) -> list[DvhCurve]:
        if self.plan is None:
            return []
        rows: list[DvhCurve] = []
        for roi in self.plan.rois:
            if roi.name in self.visible_rois and roi.name in self.dvhs:
                rows.append(self.dvhs[roi.name])
        return rows

    def _clear_tables(self) -> None:
        for table in (
            self.dvh_table,
            self.evaluation_table,
            self.plan_info_table,
            self.metrics_table,
            self.complexity_table,
        ):
            table.setRowCount(0)


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setStretchLastSection(True)
    for index in range(len(headers)):
        mode = QHeaderView.ResizeMode.ResizeToContents if index else QHeaderView.ResizeMode.Stretch
        table.horizontalHeader().setSectionResizeMode(index, mode)
    return table


def _set_row(table: QTableWidget, row: int, values: list[str] | tuple[str, ...]) -> None:
    for column, value in enumerate(values):
        table.setItem(row, column, QTableWidgetItem(str(value)))


def _table_key(table: ConstraintTable) -> str:
    identifier = table.id if table.id is not None else table.name
    return str(identifier or table.name or id(table))


def _merge_constraint_table(left: ConstraintTable, right: ConstraintTable) -> ConstraintTable:
    constraints: list[ConstraintRow] = []
    seen: set[tuple[str, str, str, str, float | None, float | None, str]] = set()
    for row in (*left.constraints, *right.constraints):
        key = (
            _normalize_name(row.oar_raw),
            row.metric.strip().lower(),
            row.unit.strip().lower(),
            row.comparator.strip(),
            row.limit_optimal,
            row.limit_maximal,
            row.priority.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        constraints.append(row)
    return replace(
        left,
        constraints=tuple(constraints),
        is_bilateral=left.is_bilateral or right.is_bilateral,
    )


def _roi_color(roi: RoiGeometry, lookup: RoiLookup | None) -> str:
    return (lookup.color if lookup and lookup.color else roi.color) or "#8ab4f8"


def _names_match(left: str, right: str) -> bool:
    return bool(_normalized_name_variants(left) & _normalized_name_variants(right))


def _normalized_name_variants(value: str) -> set[str]:
    text = str(value).strip()
    if not text:
        return set()
    variants = {text}
    suffixes = (
        "_L/R",
        "_R/L",
        "_R+L",
        "_L+R",
        "-L/R",
        "-R/L",
        "-R+L",
        "-L+R",
        " L/R",
        " R/L",
        " R+L",
        " L+R",
        "_left/right",
        "_right/left",
    )
    unilateral_suffixes = (
        "_L",
        "_R",
        "-L",
        "-R",
        " L",
        " R",
        "_left",
        "_right",
        " left",
        " right",
        " links",
        " rechts",
        " li",
        " re",
    )
    lower_text = text.lower()
    for suffix in suffixes + unilateral_suffixes:
        if lower_text.endswith(suffix.lower()):
            variants.add(text[: -len(suffix)].strip(" _-"))
    normalized = {_normalize_name(variant) for variant in variants}
    return {variant for variant in normalized if variant}


def _normalize_name(value: str) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3g}"


def _status_color(status: str) -> str:
    if status == "optimal":
        return "#22c55e"
    if status == "acceptable":
        return "#f59e0b"
    if status == "fail":
        return "#ef4444"
    if status == "missing":
        return "#f97316"
    return "#94a3b8"
