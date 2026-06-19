from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSettings, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QLinearGradient, QPainter, QPen, QPolygon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollBar,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from planeval_viewer.dicom_io.loader import load_plan_variants
from planeval_viewer.dicom_io.models import PlanDataset
from planeval_viewer.gui.app_utils import (
    APP_NAME,
    APP_ORG,
    application_icon,
    detach_console_window,
)
from planeval_viewer.gui.details_panel import DetailsPanel
from planeval_viewer.gui.dicom_tag_panel import DicomTagPanel
from planeval_viewer.gui.manual_mapping_panel import ManualMappingPanel
from planeval_viewer.gui.qa_panel import QAPanel
from planeval_viewer.gui.roi_panel import RoiPanel
from planeval_viewer.gui.theme import APP_STYLESHEET
from planeval_viewer.gui.viewer import AxialPlanViewer, ISODOSE_PERCENT_COLORS
from planeval_viewer.paths import default_manual_mappings_path, default_refdb_cache_path
from planeval_viewer.refdb.cache import RefDbCache
from planeval_viewer.refdb.client import RefDbClient
from planeval_viewer.refdb.manual_mappings import ManualMappingStore, apply_manual_mappings
from planeval_viewer.refdb.matching import RoiLookup, map_rois_to_results
from planeval_viewer.refdb.models import ConstraintRow, ConstraintTable, RefDbLookupResult
from planeval_viewer.refdb.offline import OfflineRefDb


class DoseRangeBar(QWidget):
    range_preview_changed = Signal(int, int)
    range_changed = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.minimum_value = 0
        self.maximum_value = 100
        self.lower_value = 0
        self.upper_value = 100
        self.reference_value = 100
        self._active_handle: str | None = None
        self.setObjectName("DoseRangeBar")
        self.setMinimumSize(58, 150)
        self.setMouseTracking(True)
        self.setToolTip(
            "Dosisfenster: beide Marken direkt auf der Farbskala ziehen. "
            "Isodosen sind Prozent der Verschreibungsdosis."
        )

    def set_range(self, minimum: int, maximum: int) -> None:
        self.minimum_value = int(minimum)
        self.maximum_value = max(self.minimum_value + 1, int(maximum))
        self.set_range_values(self.lower_value, self.upper_value, emit=False)

    def set_reference_value(self, value: int) -> None:
        self.reference_value = max(1, int(value))
        self.update()

    def set_range_values(self, lower: int, upper: int, emit: bool = True) -> None:
        lower, upper = self._clamped_range(lower, upper)
        changed = lower != self.lower_value or upper != self.upper_value
        self.lower_value = lower
        self.upper_value = upper
        self.update()
        if emit and changed:
            self.range_changed.emit(self.lower_value, self.upper_value)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bar = self._bar_rect()
        painter.setOpacity(1.0 if self.isEnabled() else 0.38)
        painter.fillRect(self.rect(), QColor("#05070b"))

        gradient = QLinearGradient(bar.left(), bar.bottom(), bar.left(), bar.top())
        stops = self._gradient_stops()
        if stops:
            for position, color in stops:
                gradient.setColorAt(position, color)
        else:
            gradient.setColorAt(0.0, QColor("#1d4ed8"))
            gradient.setColorAt(0.5, QColor("#22c55e"))
            gradient.setColorAt(1.0, QColor("#ef4444"))
        painter.fillRect(bar, gradient)
        painter.setPen(QPen(QColor("#8b96aa"), 1))
        painter.drawRoundedRect(bar, 3, 3)

        lower_y = self._value_to_y(self.lower_value)
        upper_y = self._value_to_y(self.upper_value)
        selected = QRect(bar.left(), upper_y, bar.width(), max(1, lower_y - upper_y))
        painter.fillRect(selected, QColor(255, 255, 255, 38))

        for percent, rgba in ISODOSE_PERCENT_COLORS:
            value = self.reference_value * percent / 100.0
            if value < self.minimum_value or value > self.maximum_value:
                continue
            y = self._value_to_y(value)
            color = _qcolor_from_rgba(rgba, 235)
            painter.setPen(QPen(color, 2))
            painter.drawLine(bar.left() - 4, y, bar.right() + 4, y)

        self._draw_handle(painter, lower_y, QColor("#e8eef8"))
        self._draw_handle(painter, upper_y, QColor("#ffffff"))

    def mousePressEvent(self, event) -> None:
        if not self.isEnabled():
            return
        y = int(event.position().y()) if hasattr(event, "position") else event.y()
        lower_distance = abs(y - self._value_to_y(self.lower_value))
        upper_distance = abs(y - self._value_to_y(self.upper_value))
        self._active_handle = "lower" if lower_distance <= upper_distance else "upper"
        self._move_active_handle(y)

    def mouseMoveEvent(self, event) -> None:
        if not self.isEnabled() or self._active_handle is None:
            return
        y = int(event.position().y()) if hasattr(event, "position") else event.y()
        self._move_active_handle(y)

    def mouseReleaseEvent(self, _event) -> None:
        if self._active_handle is not None:
            self.range_changed.emit(self.lower_value, self.upper_value)
        self._active_handle = None

    def _move_active_handle(self, y: int) -> None:
        value = self._y_to_value(y)
        old_range = (self.lower_value, self.upper_value)
        if self._active_handle == "lower":
            self.set_range_values(value, self.upper_value, emit=False)
        elif self._active_handle == "upper":
            self.set_range_values(self.lower_value, value, emit=False)
        if old_range != (self.lower_value, self.upper_value):
            self.range_preview_changed.emit(self.lower_value, self.upper_value)

    def _clamped_range(self, lower: int, upper: int) -> tuple[int, int]:
        lower = max(self.minimum_value, min(int(lower), self.maximum_value - 1))
        upper = max(lower + 1, min(int(upper), self.maximum_value))
        return lower, upper

    def _bar_rect(self) -> QRect:
        return QRect(20, 8, max(20, self.width() - 36), max(20, self.height() - 16))

    def _value_to_y(self, value: float) -> int:
        bar = self._bar_rect()
        span = max(1, self.maximum_value - self.minimum_value)
        fraction = (float(value) - self.minimum_value) / span
        return int(round(bar.bottom() - np.clip(fraction, 0.0, 1.0) * bar.height()))

    def _y_to_value(self, y: int) -> int:
        bar = self._bar_rect()
        span = max(1, self.maximum_value - self.minimum_value)
        fraction = (bar.bottom() - y) / max(1, bar.height())
        return int(round(self.minimum_value + np.clip(fraction, 0.0, 1.0) * span))

    def _gradient_stops(self) -> list[tuple[float, QColor]]:
        span = max(1, self.maximum_value - self.minimum_value)
        stops: list[tuple[float, QColor]] = []
        for percent, rgba in reversed(ISODOSE_PERCENT_COLORS):
            value = self.reference_value * percent / 100.0
            position = (value - self.minimum_value) / span
            if 0.0 <= position <= 1.0:
                stops.append((float(position), _qcolor_from_rgba(rgba, 255)))
        if stops and stops[0][0] > 0.0:
            stops.insert(0, (0.0, stops[0][1]))
        if stops and stops[-1][0] < 1.0:
            stops.append((1.0, stops[-1][1]))
        return stops

    def _draw_handle(self, painter: QPainter, y: int, color: QColor) -> None:
        bar = self._bar_rect()
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#0b0f16"), 1))
        left = QPolygon(
            [
                QPoint(bar.left() - 12, y),
                QPoint(bar.left() - 3, y - 6),
                QPoint(bar.left() - 3, y + 6),
            ]
        )
        right = QPolygon(
            [
                QPoint(bar.right() + 12, y),
                QPoint(bar.right() + 3, y - 6),
                QPoint(bar.right() + 3, y + 6),
            ]
        )
        painter.drawPolygon(left)
        painter.drawPolygon(right)
        painter.setPen(QPen(color, 2))
        painter.drawLine(bar.left() - 1, y, bar.right() + 1, y)


def _qcolor_from_rgba(
    color: tuple[float, float, float, float],
    alpha: int,
) -> QColor:
    return QColor(
        int(np.clip(color[0], 0.0, 1.0) * 255),
        int(np.clip(color[1], 0.0, 1.0) * 255),
        int(np.clip(color[2], 0.0, 1.0) * 255),
        alpha,
    )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PlanEval Viewer")
        self.resize(1500, 920)
        self.setStyleSheet(APP_STYLESHEET)
        icon = application_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        self.plan: PlanDataset | None = None
        self.plan_variants: list[PlanDataset] = []
        self._loading_plan_combo = False
        self.settings = QSettings(APP_ORG, APP_NAME)
        self.pre_render_on_open = _settings_bool(self.settings, "preRenderOnOpen", True)
        self.hide_console_window = _settings_bool(self.settings, "hideConsoleWindow", False)
        self.ct_privacy_blur = _settings_bool(self.settings, "ctPrivacyBlur", False)
        if self.hide_console_window:
            detach_console_window()
        self.roi_lookups: dict[str, RoiLookup] = {}
        self.refdb_client = RefDbClient()
        self.refdb_cache = RefDbCache(default_refdb_cache_path())
        self.offline_refdb = OfflineRefDb()
        self.manual_mapping_store = ManualMappingStore(default_manual_mappings_path())
        self.manual_mappings = self.manual_mapping_store.load()

        self.viewer = AxialPlanViewer()
        self.viewer.setObjectName("TwoDPlanViewer")
        self.three_d_viewer = AxialPlanViewer()
        self.three_d_viewer.setObjectName("PersistentThreeDViewer")
        self.viewer.set_ct_privacy_blur(self.ct_privacy_blur)
        self.three_d_viewer.set_ct_privacy_blur(self.ct_privacy_blur)
        self.roi_panel = RoiPanel()
        self.details_panel = DetailsPanel()
        self.mapping_panel = ManualMappingPanel()
        self.dicom_tag_panel = DicomTagPanel()
        self.qa_panel = QAPanel()
        self.view_mode_actions: dict[str, QAction] = {}
        self.view_mode_group = QActionGroup(self)
        self.view_mode_group.setExclusive(True)
        self.privacy_blur_action: QAction | None = None
        self.roi_panel.setMinimumWidth(320)
        self.details_panel.setMinimumWidth(360)
        self.qa_panel.setMinimumWidth(520)
        self.left_tabs = QTabWidget()
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.dashboard_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_view_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_eval_splitter = QSplitter(Qt.Orientation.Vertical)
        self.status_log = QLabel("Ready")
        self.status_log.setObjectName("StatusLog")
        self.status_log.setWordWrap(True)
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setObjectName("BusyIndicator")
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setFixedWidth(160)
        self.busy_indicator.setVisible(False)
        self._busy_depth = 0
        self.loading_overlay: QWidget | None = None
        self.loading_message: QLabel | None = None
        self.loading_overlay_progress: QProgressBar | None = None
        self._pending_dicom_folder: Path | None = None
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.slice_scrollbar.setMinimum(0)
        self.slice_scrollbar.setMaximum(0)
        self.plan_combo = QComboBox()
        self.plan_combo.setVisible(False)
        self.plan_combo.setMinimumWidth(180)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(35)
        self.dose_range_bar = DoseRangeBar()
        self.dose_range_bar.setEnabled(False)
        self.dose_min_spin = QDoubleSpinBox()
        self.dose_min_spin.setDecimals(1)
        self.dose_min_spin.setSingleStep(0.1)
        self.dose_min_spin.setSuffix(" Gy")
        self.dose_min_spin.setEnabled(False)
        self.dose_max_spin = QDoubleSpinBox()
        self.dose_max_spin.setDecimals(1)
        self.dose_max_spin.setSingleStep(0.1)
        self.dose_max_spin.setSuffix(" Gy")
        self.dose_max_spin.setEnabled(False)
        self.dose_display_mode_combo = QComboBox()
        self.dose_display_mode_combo.addItem("Overlay", "overlay")
        self.dose_display_mode_combo.addItem("Isodosen", "isodose")
        self.dose_scale_value_label = QLabel("Dose n/a")
        self.dose_scale_value_label.setObjectName("DoseScaleValue")
        self._updating_dose_controls = False
        self.ct3d_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.ct3d_opacity_slider.setMinimum(0)
        self.ct3d_opacity_slider.setMaximum(300)
        self.ct3d_opacity_slider.setValue(180)
        self.ct3d_opacity_slider.setToolTip("Transparenz/Staerke der 3D-CT-Darstellung")
        self.ct3d_mode_combo = QComboBox()
        self.ct3d_mode_combo.addItem("Body/Bone", "surface")
        self.ct3d_mode_combo.addItem("Volume", "volume")
        self.three_d_targets_checkbox = QCheckBox("PTVs")
        self.three_d_targets_checkbox.setObjectName("ThreeDTargetsCheckbox")
        self.three_d_targets_checkbox.setToolTip("Zielvolumina in der 3D-Ansicht anzeigen")
        self.three_d_targets_checkbox.setChecked(True)
        self.three_d_oars_checkbox = QCheckBox("OARs")
        self.three_d_oars_checkbox.setObjectName("ThreeDOarsCheckbox")
        self.three_d_oars_checkbox.setToolTip("OAR-/Kontextstrukturen in der 3D-Ansicht anzeigen")
        self.three_d_oars_checkbox.setChecked(True)
        saved_ct_mode = str(self.settings.value("ct3dRenderMode", "surface") or "surface")
        saved_ct_index = self.ct3d_mode_combo.findData(saved_ct_mode)
        if saved_ct_index >= 0:
            self.ct3d_mode_combo.setCurrentIndex(saved_ct_index)
            self.three_d_viewer.set_3d_ct_render_mode(saved_ct_mode)
        self.collapse_3d_action: QAction | None = None
        self.select_plan_action: QAction | None = None

        self._build_toolbar()
        self._build_layout()
        self._connect_signals()

    def open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "DICOM-Planordner öffnen")
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder: Path) -> None:
        self._set_busy(True, "Loading DICOM plan...")
        try:
            plans = load_plan_variants(folder)
        except Exception as exc:
            QMessageBox.critical(self, "DICOM load failed", str(exc))
            self._set_status("DICOM load failed.")
            self._set_busy(False)
            return

        if not plans:
            QMessageBox.warning(self, "No DICOM found", "No readable DICOM objects found.")
            self._set_busy(False, "No readable DICOM objects found.")
            return

        self.plan_variants = plans
        self._populate_plan_combo(plans)
        selected_index = 0
        if len(plans) > 1:
            self._set_busy(False, f"{len(plans)} plan variants found.")
            selected = self._select_plan_variant_index(plans, current_index=0, initial=True)
            if selected is None:
                self._set_status("Plan selection cancelled. Use Plan waehlen to load a variant.")
                return
            selected_index = selected
            self._set_busy(True, "Loading selected plan...")
        try:
            self._load_plan_variant(selected_index, folder, run_lookup=True, run_preload=True)
        finally:
            self._set_busy(False)

    def _load_plan_variant(
        self,
        index: int,
        folder: Path | None,
        run_lookup: bool,
        run_preload: bool,
    ) -> None:
        if index < 0 or index >= len(self.plan_variants):
            return
        plan = self.plan_variants[index]
        self.plan = plan
        self._set_plan_combo_index(index)
        self.roi_lookups = {}
        self.viewer.set_plan(plan)
        self.three_d_viewer.set_plan(plan)
        self._three_d_structure_visibility_changed()
        if plan.ct is not None:
            self.three_d_viewer.set_view_mode("3d")
        self._sync_view_mode_action()
        self.roi_panel.set_plan(plan)
        self.details_panel.set_plan(plan)
        self.qa_panel.set_plan(plan)
        if folder is not None:
            self._pending_dicom_folder = folder
            self.dicom_tag_panel.set_pending_folder(folder)
        self.qa_panel.set_manual_mappings(self.manual_mappings)
        self.mapping_panel.set_mappings(self.manual_mappings)
        self._update_mapping_options()
        self._apply_lookup_state()
        self._target_changed(self.qa_panel.selected_target_name())
        self._configure_slice_slider(plan)
        self._configure_dose_scale(plan)
        self._set_status(
            f"Loaded plan with {len(plan.rois)} ROIs"
            + (f" and {len(plan.warnings)} warnings." if plan.warnings else ".")
        )
        if self.left_tabs.currentWidget() is self.dicom_tag_panel:
            self._load_pending_dicom_tags()
        if run_lookup and plan.rois:
            self.lookup_refdb()
        if run_preload and self.pre_render_on_open:
            self._pre_render_loaded_case()

    def _populate_plan_combo(self, plans: list[PlanDataset]) -> None:
        self._loading_plan_combo = True
        try:
            self.plan_combo.clear()
            for index, plan in enumerate(plans):
                self.plan_combo.addItem(_plan_label(plan, index), index)
            self.plan_combo.setVisible(len(plans) > 1)
            self.plan_combo.setCurrentIndex(0)
            if self.select_plan_action is not None:
                self.select_plan_action.setEnabled(len(plans) > 1)
        finally:
            self._loading_plan_combo = False

    def _set_plan_combo_index(self, index: int) -> None:
        self._loading_plan_combo = True
        try:
            if 0 <= index < self.plan_combo.count():
                self.plan_combo.setCurrentIndex(index)
        finally:
            self._loading_plan_combo = False

    def lookup_refdb(self) -> None:
        if self.plan is None or not self.plan.rois:
            self._set_status("No ROIs available for RefDB lookup.")
            return

        roi_names = self.plan.roi_names
        fx = _safe_int(self.plan.plan_info.get("number_of_fractions"))
        self._set_busy(True, "Querying RefDB...")
        try:
            cached = self.refdb_cache.load_many(roi_names)
            results = self.refdb_client.lookup_batch(
                roi_names,
                fx=fx,
                use_server_fraction_filter=False,
            )
            offline = self.offline_refdb.lookup_many(roi_names, fx=fx)
            merged = _merge_lookup_sources(roi_names, results, cached, offline)
            self.refdb_cache.store_many(merged)

            lookups = map_rois_to_results(roi_names, merged)
            self.roi_lookups = {lookup.source_name: lookup for lookup in lookups}
            self._apply_lookup_state()
            self._update_selected_roi_details()

            unresolved = sum(1 for lookup in lookups if lookup.error)
            matched = sum(1 for lookup in lookups if lookup.matched_name)
            self._set_busy(False, f"RefDB lookup: {matched} matched, {unresolved} not found.")
        except Exception as exc:
            self._set_busy(False, f"RefDB lookup failed: {exc}")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Plan tools")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_folder_dialog)
        toolbar.addAction(open_action)

        reload_action = QAction("Reload RefDB", self)
        reload_action.triggered.connect(self.lookup_refdb)
        toolbar.addAction(reload_action)

        settings_action = QAction("Einstellungen", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        toolbar.addAction(settings_action)

        qa_action = QAction("Compute QA", self)
        qa_action.triggered.connect(self.compute_qa)
        toolbar.addAction(qa_action)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Plan"))
        toolbar.addWidget(self.plan_combo)
        self.select_plan_action = QAction("Plan waehlen", self)
        self.select_plan_action.setEnabled(False)
        self.select_plan_action.setToolTip("Auswahlfenster fuer weitere RTPLAN-Varianten oeffnen")
        self.select_plan_action.triggered.connect(self._open_plan_selection_dialog)
        toolbar.addAction(self.select_plan_action)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("View"))
        for mode, label in (
            ("axial", "Axial"),
            ("sagittal", "Sagittal"),
            ("coronal", "Coronal"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setActionGroup(self.view_mode_group)
            action.triggered.connect(
                lambda _checked=False, view_mode=mode: self._set_view_mode(view_mode)
            )
            toolbar.addAction(action)
            self.view_mode_actions[mode] = action
        self.view_mode_actions["axial"].setChecked(True)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Slice"))
        self.slice_slider.setFixedWidth(260)
        toolbar.addWidget(self.slice_slider)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Dose"))
        self.opacity_slider.setFixedWidth(140)
        toolbar.addWidget(self.opacity_slider)

        toolbar.addWidget(QLabel("3D CT"))
        self.ct3d_mode_combo.setFixedWidth(110)
        toolbar.addWidget(self.ct3d_mode_combo)
        self.ct3d_opacity_slider.setFixedWidth(140)
        toolbar.addWidget(self.ct3d_opacity_slider)
        self.collapse_3d_action = QAction("3D einklappen", self)
        self.collapse_3d_action.setCheckable(True)
        self.collapse_3d_action.toggled.connect(self.set_3d_panel_collapsed)
        toolbar.addAction(self.collapse_3d_action)

        toolbar.addSeparator()
        lung_action = QAction("Lung WL", self)
        lung_action.triggered.connect(lambda: self.viewer.set_window(-600, 1500))
        toolbar.addAction(lung_action)

        soft_action = QAction("Soft WL", self)
        soft_action.triggered.connect(lambda: self.viewer.set_window(40, 400))
        toolbar.addAction(soft_action)

        bone_action = QAction("Bone WL", self)
        bone_action.triggered.connect(lambda: self.viewer.set_window(500, 1800))
        toolbar.addAction(bone_action)

        toolbar.addSeparator()
        self.privacy_blur_action = QAction("CT Blur", self)
        self.privacy_blur_action.setCheckable(True)
        self.privacy_blur_action.setChecked(self.ct_privacy_blur)
        self.privacy_blur_action.setToolTip(
            "Privacy-Modus fuer Screenshots: CT-Anatomie weichzeichnen, "
            "Dosis, Isodosen und Konturen scharf lassen."
        )
        self.privacy_blur_action.toggled.connect(self._ct_privacy_blur_changed)
        toolbar.addAction(self.privacy_blur_action)

    def _build_layout(self) -> None:
        self.left_tabs.addTab(self.roi_panel, "ROIs")
        self.left_tabs.addTab(self.details_panel, "Details")
        self.left_tabs.addTab(self.mapping_panel, "Mappings")
        self.left_tabs.addTab(self.dicom_tag_panel, "DICOM Tags")
        self.left_tabs.setMinimumWidth(320)

        self._build_dashboard()
        self.main_splitter.addWidget(self.left_tabs)
        self.main_splitter.addWidget(self.dashboard_splitter)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 7)
        self.main_splitter.setSizes([320, 1180])

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.main_splitter, 1)
        status_row = QWidget()
        status_layout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(0)
        status_layout.addWidget(self.busy_indicator)
        status_layout.addWidget(self.status_log, 1)
        layout.addWidget(status_row)
        self.setCentralWidget(root)
        self._build_loading_overlay(root)

    def _build_loading_overlay(self, parent: QWidget) -> None:
        self.loading_overlay = QWidget(parent)
        self.loading_overlay.setObjectName("LoadingOverlay")
        self.loading_overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        overlay_layout = QVBoxLayout(self.loading_overlay)
        overlay_layout.setContentsMargins(20, 20, 20, 20)
        overlay_layout.addStretch(1)
        self.loading_message = QLabel("Loading...")
        self.loading_message.setObjectName("LoadingMessage")
        self.loading_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_overlay_progress = QProgressBar()
        self.loading_overlay_progress.setObjectName("LoadingOverlayProgress")
        self.loading_overlay_progress.setRange(0, 0)
        self.loading_overlay_progress.setFixedWidth(260)
        overlay_layout.addWidget(self.loading_message)
        overlay_layout.addWidget(
            self.loading_overlay_progress,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        overlay_layout.addStretch(1)
        self.loading_overlay.hide()
        self._position_loading_overlay()

    def _position_loading_overlay(self) -> None:
        if self.loading_overlay is None:
            return
        parent = self.loading_overlay.parentWidget()
        if parent is not None:
            self.loading_overlay.setGeometry(parent.rect())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._position_loading_overlay()

    def _connect_signals(self) -> None:
        self.slice_slider.valueChanged.connect(self.viewer.set_slice_index)
        self.slice_scrollbar.valueChanged.connect(self.viewer.set_slice_index)
        self.viewer.slice_changed.connect(self._sync_slice_controls)
        self.opacity_slider.valueChanged.connect(
            self._dose_opacity_changed
        )
        self.dose_range_bar.range_preview_changed.connect(self._dose_range_bar_preview_changed)
        self.dose_range_bar.range_changed.connect(self._dose_range_bar_changed)
        self.dose_min_spin.valueChanged.connect(self._dose_spin_changed)
        self.dose_max_spin.valueChanged.connect(self._dose_spin_changed)
        self.dose_display_mode_combo.currentIndexChanged.connect(self._dose_display_mode_changed)
        self.ct3d_opacity_slider.valueChanged.connect(
            lambda value: self.three_d_viewer.set_3d_ct_opacity(value / 100.0)
        )
        self.ct3d_mode_combo.currentIndexChanged.connect(self._ct3d_mode_changed)
        self.three_d_targets_checkbox.toggled.connect(self._three_d_structure_visibility_changed)
        self.three_d_oars_checkbox.toggled.connect(self._three_d_structure_visibility_changed)
        self.plan_combo.currentIndexChanged.connect(self._plan_combo_changed)
        self.roi_panel.visibility_changed.connect(self._set_roi_visible)
        self.roi_panel.selection_changed.connect(self._roi_selection_changed)
        self.roi_panel.mapping_requested.connect(self._manual_mapping_requested)
        self.qa_panel.status_message.connect(self._set_status)
        self.qa_panel.target_changed.connect(self._target_changed)
        self.qa_panel.manual_mapping_requested.connect(self._manual_mapping_requested)
        self.qa_panel.missing_constraints_changed.connect(self._update_mapping_options)
        self.mapping_panel.mapping_requested.connect(self._manual_mapping_requested)
        self.mapping_panel.export_requested.connect(self._export_manual_mappings)
        self.left_tabs.currentChanged.connect(self._left_tab_changed)
        try:
            self.qa_panel.compute_button.clicked.disconnect(self.qa_panel.compute_qa)
        except (RuntimeError, TypeError):
            pass
        self.qa_panel.compute_button.clicked.connect(self.compute_qa)

    def _ct3d_mode_changed(self, _index: int) -> None:
        mode = self.ct3d_mode_combo.currentData() or "surface"
        self.three_d_viewer.set_3d_ct_render_mode(str(mode))
        self.settings.setValue("ct3dRenderMode", str(mode))

    def _three_d_structure_visibility_changed(self, _checked: bool = True) -> None:
        self.three_d_viewer.set_3d_structure_group_visibility(
            show_targets=self.three_d_targets_checkbox.isChecked(),
            show_oars=self.three_d_oars_checkbox.isChecked(),
        )

    def _dose_opacity_changed(self, value: int) -> None:
        opacity = value / 100.0
        self.viewer.set_dose_opacity(opacity)
        self.three_d_viewer.set_dose_opacity(opacity)

    def _dose_range_bar_preview_changed(self, lower: int, upper: int) -> None:
        if self._updating_dose_controls:
            return
        self._sync_dose_range_controls(lower, upper)

    def _dose_range_bar_changed(self, lower: int, upper: int) -> None:
        if self._updating_dose_controls:
            return
        self._set_dose_range_tenths(lower, upper)

    def _dose_spin_changed(self, _value: float) -> None:
        if self._updating_dose_controls:
            return
        lower = int(round(self.dose_min_spin.value() * 10.0))
        upper = int(round(self.dose_max_spin.value() * 10.0))
        if lower >= upper:
            sender = self.sender()
            if sender is self.dose_min_spin:
                upper = min(self.dose_range_bar.maximum_value, lower + 1)
            else:
                lower = max(0, upper - 1)
        self._set_dose_range_tenths(lower, upper)

    def _sync_dose_range_controls(self, lower: int, upper: int) -> tuple[float, float]:
        lower = max(0, min(int(lower), self.dose_range_bar.maximum_value - 1))
        upper = max(lower + 1, min(int(upper), self.dose_range_bar.maximum_value))
        min_gy = lower / 10.0
        max_gy = upper / 10.0
        self._updating_dose_controls = True
        try:
            self.dose_range_bar.set_range_values(lower, upper, emit=False)
            self.dose_min_spin.setValue(min_gy)
            self.dose_max_spin.setValue(max_gy)
        finally:
            self._updating_dose_controls = False
        self.dose_scale_value_label.setText(f"{min_gy:.1f}-{max_gy:.1f} Gy")
        return min_gy, max_gy

    def _set_dose_range_tenths(self, lower: int, upper: int) -> None:
        min_gy, max_gy = self._sync_dose_range_controls(lower, upper)
        self.viewer.set_dose_display_range_gy(min_gy, max_gy)
        self.three_d_viewer.set_dose_display_range_gy(min_gy, max_gy)

    def _dose_display_mode_changed(self, _index: int) -> None:
        mode = str(self.dose_display_mode_combo.currentData() or "overlay")
        self.viewer.set_dose_display_mode(mode)
        self.three_d_viewer.set_dose_display_mode(mode)

    def _ct_privacy_blur_changed(self, enabled: bool) -> None:
        self.ct_privacy_blur = bool(enabled)
        self.settings.setValue("ctPrivacyBlur", self.ct_privacy_blur)
        self.viewer.set_ct_privacy_blur(self.ct_privacy_blur)
        self.three_d_viewer.set_ct_privacy_blur(self.ct_privacy_blur)

    def set_3d_panel_collapsed(self, collapsed: bool) -> None:
        self.three_d_shell.setVisible(not collapsed)
        if not collapsed and self.plan is not None and self.plan.ct is not None:
            self.three_d_viewer.set_view_mode("3d")
        if self.collapse_3d_action is not None:
            self.collapse_3d_action.blockSignals(True)
            try:
                self.collapse_3d_action.setChecked(collapsed)
                self.collapse_3d_action.setText(
                    "3D ausklappen" if collapsed else "3D einklappen"
                )
            finally:
                self.collapse_3d_action.blockSignals(False)

    def _plan_combo_changed(self, index: int) -> None:
        if self._loading_plan_combo:
            return
        if index < 0:
            return
        if index < len(self.plan_variants) and self.plan is self.plan_variants[index]:
            return
        self._set_busy(True, "Loading selected plan...")
        try:
            self._load_plan_variant(index, None, run_lookup=True, run_preload=True)
        finally:
            self._set_busy(False)

    def _open_plan_selection_dialog(self) -> None:
        if not self.plan_variants:
            self._set_status("No plan variants loaded.")
            return
        current_index = self.plan_combo.currentIndex()
        if current_index < 0:
            current_index = 0
        selected = self._select_plan_variant_index(
            self.plan_variants,
            current_index=current_index,
            initial=False,
        )
        if selected is None:
            return
        if selected == current_index and self.plan is self.plan_variants[selected]:
            return
        self._set_busy(True, "Loading selected plan...")
        try:
            self._load_plan_variant(selected, None, run_lookup=True, run_preload=True)
        finally:
            self._set_busy(False)

    def _select_plan_variant_index(
        self,
        plans: list[PlanDataset],
        current_index: int = 0,
        initial: bool = False,
    ) -> int | None:
        if len(plans) <= 1:
            return 0 if plans else None
        dialog = PlanSelectionDialog(
            plans=plans,
            current_index=current_index,
            initial=initial,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_index()

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(
            pre_render_on_open=self.pre_render_on_open,
            hide_console_window=self.hide_console_window,
            ct_render_mode=self.three_d_viewer.ct_3d_render_mode,
            help_text=self.dicom_layout_help_text(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.pre_render_on_open = dialog.pre_render_checkbox.isChecked()
        self.hide_console_window = dialog.hide_console_checkbox.isChecked()
        self.settings.setValue("preRenderOnOpen", self.pre_render_on_open)
        self.settings.setValue("hideConsoleWindow", self.hide_console_window)
        mode = dialog.ct_mode_combo.currentData() or "surface"
        index = self.ct3d_mode_combo.findData(mode)
        if index >= 0:
            self.ct3d_mode_combo.setCurrentIndex(index)
        self.three_d_viewer.set_3d_ct_render_mode(str(mode))
        self.settings.setValue("ct3dRenderMode", str(mode))
        if self.hide_console_window:
            detach_console_window()

    def dicom_layout_help_text(self) -> str:
        return (
            "DICOM layout: Waehle einen Ordner, der CT, RTSTRUCT, RTDOSE und optional "
            "RTPLAN enthaelt. Unterordner sind erlaubt. RTPLAN optional: CT/RTSTRUCT/"
            "RTDOSE laden auch ohne Plan. CT-Slices sollten zu einer SeriesInstanceUID "
            "gehoeren und gleiche Orientierung/PixelSpacing haben. RTSTRUCT und RTDOSE "
            "sollten zur gleichen FrameOfReference/Plan-Referenz gehoeren; wenn mehrere "
            "RTPLAN-Dateien vorhanden sind, fragt RTScope beim Laden nach der Planvariante. "
            "Weitere Varianten koennen danach ueber Plan waehlen geoeffnet werden."
        )

    def compute_qa(self) -> None:
        self._set_busy(True, "Computing plan QA...")
        try:
            self.qa_panel.compute_qa()
        finally:
            self._set_busy(False)

    def _pre_render_loaded_case(self) -> None:
        if self.plan is None:
            return
        self._set_busy(True, "Pre-rendering views and QA...")
        current_mode = self.viewer.view_mode
        try:
            if self.plan.ct is not None:
                for mode in ("axial", "sagittal", "coronal"):
                    self.viewer.set_view_mode(mode)
                    self._configure_slice_slider(self.plan)
                self.three_d_viewer.set_view_mode("3d")
            if self.plan.dose is not None:
                self.qa_panel.compute_qa()
        finally:
            if self.plan is not None and self.plan.ct is not None:
                self.viewer.set_view_mode(current_mode)
                self.three_d_viewer.set_view_mode("3d")
                self._configure_slice_slider(self.plan)
                self._sync_view_mode_action()
            self._set_busy(False, "Pre-render complete.")

    def _left_tab_changed(self, _index: int) -> None:
        if self.left_tabs.currentWidget() is self.dicom_tag_panel:
            self._load_pending_dicom_tags()

    def _load_pending_dicom_tags(self) -> None:
        folder = self._pending_dicom_folder
        if folder is None:
            return
        self._pending_dicom_folder = None
        self._set_busy(True, "Loading DICOM tag browser...")
        try:
            self.dicom_tag_panel.set_folder(folder)
            self._set_busy(False, "DICOM tag browser loaded.")
        except Exception as exc:
            self._set_busy(False, f"DICOM tag browser failed: {exc}")

    def _update_mapping_options(self, reference_names: list[str] | None = None) -> None:
        references = set(reference_names or self.qa_panel.missing_constraint_names())
        references.update(self.manual_mappings.keys())
        for lookup in self.roi_lookups.values():
            references.update(
                item
                for item in (lookup.matched_name, lookup.reference_name, *lookup.aliases)
                if item
            )
            if lookup.result is None:
                continue
            for table in lookup.result.constraint_tables:
                for constraint in table.constraints:
                    if constraint.oar_raw:
                        references.add(constraint.oar_raw)
        local_rois = self.plan.roi_names if self.plan else []
        sorted_references = sorted(references)
        self.mapping_panel.set_options(sorted_references, local_rois)
        self.roi_panel.set_mapping_references(sorted_references)

    def _export_manual_mappings(self) -> None:
        filename, _selected = QFileDialog.getSaveFileName(
            self,
            "Mapping JSON exportieren",
            "manual_mappings.json",
            "JSON (*.json);;All files (*)",
        )
        if not filename:
            return
        ManualMappingStore(Path(filename)).store(self.manual_mappings)
        self._set_status(f"Manual mappings exported: {filename}")

    def _configure_slice_slider(self, plan: PlanDataset) -> None:
        self.slice_slider.blockSignals(True)
        self.slice_scrollbar.blockSignals(True)
        try:
            if plan.ct is None:
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(0)
                self.slice_slider.setValue(0)
                self.slice_scrollbar.setMinimum(0)
                self.slice_scrollbar.setMaximum(0)
                self.slice_scrollbar.setValue(0)
            else:
                max_slice = max(0, self.viewer.slice_count() - 1)
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(max_slice)
                self.slice_slider.setValue(self.viewer.slice_index)
                self.slice_scrollbar.setMinimum(0)
                self.slice_scrollbar.setMaximum(max_slice)
                self.slice_scrollbar.setValue(self.viewer.slice_index)
        finally:
            self.slice_slider.blockSignals(False)
            self.slice_scrollbar.blockSignals(False)

    def _configure_dose_scale(self, plan: PlanDataset) -> None:
        max_dose = 0.0
        if plan.dose is not None and plan.dose.values_gy.size:
            max_dose = float(np.nanmax(plan.dose.values_gy))
        self._updating_dose_controls = True
        try:
            if max_dose <= 0:
                self.dose_range_bar.setEnabled(False)
                self.dose_min_spin.setEnabled(False)
                self.dose_max_spin.setEnabled(False)
                self.dose_range_bar.set_range(0, 100)
                self.dose_range_bar.set_reference_value(100)
                self.dose_range_bar.set_range_values(0, 100, emit=False)
                self.dose_min_spin.setRange(0.0, 10.0)
                self.dose_max_spin.setRange(0.1, 10.0)
                self.dose_min_spin.setValue(0.0)
                self.dose_max_spin.setValue(10.0)
                self.dose_scale_value_label.setText("Dose n/a")
                self.viewer.set_dose_display_range_gy(0.0, None)
                self.three_d_viewer.set_dose_display_range_gy(0.0, None)
                return
            max_tenths = max(1, int(np.ceil(max_dose * 10.0)))
            max_display_gy = max_tenths / 10.0
            prescription = _safe_positive_float(plan.plan_info.get("prescription_dose_gy"))
            reference_tenths = (
                int(round(prescription * 10.0)) if prescription is not None else max_tenths
            )
            self.dose_range_bar.setEnabled(True)
            self.dose_min_spin.setEnabled(True)
            self.dose_max_spin.setEnabled(True)
            self.dose_range_bar.set_range(0, max_tenths)
            self.dose_range_bar.set_reference_value(reference_tenths)
            self.dose_range_bar.set_range_values(0, max_tenths, emit=False)
            self.dose_min_spin.setRange(0.0, max_display_gy)
            self.dose_max_spin.setRange(0.1, max_display_gy)
            self.dose_min_spin.setValue(0.0)
            self.dose_max_spin.setValue(max_display_gy)
        finally:
            self._updating_dose_controls = False
        self._set_dose_range_tenths(self.dose_range_bar.lower_value, self.dose_range_bar.upper_value)
        self._dose_display_mode_changed(self.dose_display_mode_combo.currentIndex())

    def _build_dashboard(self) -> None:
        self.left_view_splitter.addWidget(self._build_3d_shell())
        self.left_view_splitter.addWidget(self._build_viewer_shell())
        self.left_view_splitter.setStretchFactor(0, 1)
        self.left_view_splitter.setStretchFactor(1, 1)
        self.left_view_splitter.setSizes([460, 460])

        self.right_eval_splitter.addWidget(self._build_dvh_shell())
        self.right_eval_splitter.addWidget(self._build_qa_shell())
        self.right_eval_splitter.setStretchFactor(0, 1)
        self.right_eval_splitter.setStretchFactor(1, 1)
        self.right_eval_splitter.setSizes([460, 460])

        self.dashboard_splitter.setObjectName("FourViewDashboard")
        self.dashboard_splitter.addWidget(self.left_view_splitter)
        self.dashboard_splitter.addWidget(self.right_eval_splitter)
        self.dashboard_splitter.setStretchFactor(0, 1)
        self.dashboard_splitter.setStretchFactor(1, 1)
        self.dashboard_splitter.setSizes([700, 700])

    def _build_3d_shell(self) -> QWidget:
        self.three_d_shell = QWidget()
        self.three_d_shell.setObjectName("ThreeDShell")
        layout = QVBoxLayout(self.three_d_shell)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("3D")
        title.setObjectName("PaneTitle")
        collapse_button = QPushButton("Einklappen")
        collapse_button.setToolTip("3D-Ansicht einklappen, damit die 2D-Ansicht den Platz nutzt")
        collapse_button.clicked.connect(
            lambda: self.collapse_3d_action.setChecked(True)
            if self.collapse_3d_action is not None
            else self.set_3d_panel_collapsed(True)
        )
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.three_d_targets_checkbox)
        header.addWidget(self.three_d_oars_checkbox)
        header.addWidget(collapse_button)
        layout.addLayout(header)
        layout.addWidget(self.three_d_viewer, 1)
        return self.three_d_shell

    def _build_viewer_shell(self) -> QWidget:
        self.viewer_shell = QWidget()
        self.viewer_shell.setObjectName("ViewerShell")
        layout = QHBoxLayout(self.viewer_shell)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(self._build_dose_scale_panel())
        self.slice_scrollbar.setFixedWidth(18)
        layout.addWidget(self.slice_scrollbar)
        return self.viewer_shell

    def _build_dose_scale_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("DoseScalePanel")
        panel.setFixedWidth(112)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)
        title = QLabel("Dose")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dose_scale_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dose_min_spin.setFixedWidth(92)
        self.dose_max_spin.setFixedWidth(92)
        self.dose_display_mode_combo.setFixedWidth(104)
        layout.addWidget(title)
        layout.addWidget(self.dose_scale_value_label)
        layout.addWidget(self.dose_max_spin)
        layout.addWidget(self.dose_range_bar, 1, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.dose_min_spin)
        layout.addWidget(self.dose_display_mode_combo)
        return panel

    def _build_dvh_shell(self) -> QWidget:
        self.dvh_shell = QWidget()
        self.dvh_shell.setObjectName("DvhShell")
        layout = QVBoxLayout(self.dvh_shell)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        title = QLabel("DVH")
        title.setObjectName("PaneTitle")
        layout.addWidget(title)
        _remove_widget_from_parent_layout(self.qa_panel.dvh_plot)
        layout.addWidget(self.qa_panel.dvh_plot, 1)
        return self.dvh_shell

    def _build_qa_shell(self) -> QWidget:
        self.qa_shell = QWidget()
        self.qa_shell.setObjectName("QaShell")
        layout = QVBoxLayout(self.qa_shell)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        _remove_widget_from_parent_layout(self.qa_panel.header_widget)
        layout.addWidget(self.qa_panel.header_widget)

        self.qa_dashboard_tabs = QTabWidget()
        self.qa_dashboard_tabs.setObjectName("QaDashboardTabs")
        _remove_tab(self.qa_panel.tabs, self.qa_panel.dvh_page)
        _remove_tab(self.qa_panel.tabs, self.qa_panel.plan_page)
        _remove_tab(self.qa_panel.tabs, self.qa_panel.bev_panel)
        _remove_widget_from_parent_layout(self.qa_panel.dvh_results_tabs)
        self.qa_dashboard_tabs.addTab(self.qa_panel.dvh_results_tabs, "DVH / Constraints")
        self.qa_dashboard_tabs.addTab(self.qa_panel.plan_page, "Plan")
        self.qa_dashboard_tabs.addTab(self.qa_panel.bev_panel, "MLC / BEV")
        layout.addWidget(self.qa_dashboard_tabs, 1)
        return self.qa_shell

    def _sync_slice_controls(self, index: int) -> None:
        self.slice_slider.blockSignals(True)
        self.slice_scrollbar.blockSignals(True)
        try:
            self.slice_slider.setValue(index)
            self.slice_scrollbar.setValue(index)
        finally:
            self.slice_slider.blockSignals(False)
            self.slice_scrollbar.blockSignals(False)

    def _set_view_mode(self, mode: str) -> None:
        if mode == "3d":
            self.set_3d_panel_collapsed(False)
            return
        self.viewer.set_view_mode(mode)
        if self.plan is not None:
            self._configure_slice_slider(self.plan)
        self._sync_view_mode_action()

    def _sync_view_mode_action(self) -> None:
        action = self.view_mode_actions.get(self.viewer.view_mode)
        if action is not None:
            action.setChecked(True)

    def _set_roi_visible(self, roi_name: str, visible: bool) -> None:
        if self.plan is not None:
            roi = self.plan.roi_by_name(roi_name)
            if roi is not None:
                roi.visible = visible
        self.viewer.set_roi_visible(roi_name, visible)
        self.three_d_viewer.set_roi_visible(roi_name, visible)
        self.qa_panel.set_roi_visible(roi_name, visible)

    def _update_selected_roi_details(self) -> None:
        name = self.roi_panel.selected_roi_name()
        self.details_panel.set_roi(name, self.roi_lookups.get(name))

    def _roi_selection_changed(self, _name: str) -> None:
        self._update_selected_roi_details()

    def _target_changed(self, name: str) -> None:
        self.viewer.set_focus_roi(name or None)
        self.three_d_viewer.set_focus_roi(name or None)
        self.details_panel.set_target_name(name)

    def _manual_mapping_requested(self, reference_name: str, local_roi: str) -> None:
        self.manual_mappings = self.manual_mapping_store.upsert(reference_name, local_roi)
        self._apply_lookup_state()
        if self.qa_panel.dvhs:
            self.qa_panel._populate_evaluation_table()
        self._update_mapping_options()
        self._set_status(f"Manual mapping saved: {reference_name} -> {local_roi}")

    def _apply_lookup_state(self) -> None:
        roi_names = self.plan.roi_names if self.plan else []
        self.roi_lookups = apply_manual_mappings(
            self.roi_lookups,
            roi_names=roi_names,
            mappings=self.manual_mappings,
        )
        self.roi_panel.set_lookups(self.roi_lookups)
        self.viewer.set_lookups(self.roi_lookups)
        self.three_d_viewer.set_lookups(self.roi_lookups)
        self.qa_panel.set_lookups(self.roi_lookups)
        self.qa_panel.set_manual_mappings(self.manual_mappings)
        self.mapping_panel.set_mappings(self.manual_mappings)
        self._update_mapping_options()

    def _set_status(self, message: str) -> None:
        self.status_log.setText(message)
        if self._busy_depth > 0 and self.loading_message is not None:
            self.loading_message.setText(message)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        if busy:
            self._busy_depth += 1
        else:
            self._busy_depth = max(0, self._busy_depth - 1)
        self.busy_indicator.setVisible(self._busy_depth > 0)
        if message:
            self._set_status(message)
        if self.loading_overlay is not None:
            if self._busy_depth > 0:
                if self.loading_message is not None and message:
                    self.loading_message.setText(message)
                self._position_loading_overlay()
                self.loading_overlay.show()
                self.loading_overlay.raise_()
            else:
                self.loading_overlay.hide()
        QApplication.processEvents()


def _merge_lookup_sources(
    roi_names: list[str],
    results: list[RefDbLookupResult],
    cached: dict[str, RefDbLookupResult],
    offline: dict[str, RefDbLookupResult] | None = None,
) -> list[RefDbLookupResult]:
    merged: list[RefDbLookupResult] = []
    for index, name in enumerate(roi_names):
        result = results[index] if index < len(results) else None
        cached_result = cached.get(name)
        offline_result = (offline or {}).get(name)
        combined = _merge_lookup_result_candidates(
            index=index,
            query=name,
            candidates=(result, cached_result, offline_result),
        )
        if combined is not None:
            merged.append(combined)
            continue
        if result is not None:
            merged.append(result)
        else:
            merged.append(
                RefDbLookupResult(
                    query_index=index,
                    query=name,
                    error="No RefDB result returned for this query",
                )
            )
    return merged


def _merge_lookup_result_candidates(
    index: int,
    query: str,
    candidates: tuple[RefDbLookupResult | None, ...],
) -> RefDbLookupResult | None:
    ok_candidates = [
        candidate for candidate in candidates if candidate is not None and candidate.ok
    ]
    if not ok_candidates:
        return None
    base = ok_candidates[0]
    tables = _merge_constraint_tables_for_lookup(ok_candidates)
    raw = _lookup_result_to_dict(
        replace(base, query_index=index, query=query, constraint_tables=tables)
    )
    return RefDbLookupResult.from_dict(raw)


def _merge_constraint_tables_for_lookup(
    candidates: list[RefDbLookupResult],
) -> tuple[ConstraintTable, ...]:
    by_key: dict[str, ConstraintTable] = {}
    for candidate in candidates:
        for table in candidate.constraint_tables:
            key = str(table.id if table.id is not None else table.name)
            existing = by_key.get(key)
            by_key[key] = (
                table if existing is None else _merge_constraint_table(existing, table)
            )
    return tuple(by_key.values())


def _merge_constraint_table(left: ConstraintTable, right: ConstraintTable) -> ConstraintTable:
    constraints: list[ConstraintRow] = []
    seen: set[tuple[str, str, str, str, float | None, float | None, str]] = set()
    for row in (*left.constraints, *right.constraints):
        key = (
            _normalize_lookup_name(row.oar_raw),
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


def _lookup_result_to_dict(result: RefDbLookupResult) -> dict[str, object]:
    return {
        "query_index": result.query_index,
        "query": result.query,
        "matched_name": result.matched_name,
        "reference_name": result.reference_name,
        "side": result.side,
        "color": result.color,
        "aliases": list(result.aliases),
        "bilateral_included": result.bilateral_included,
        "bilateral_name": result.bilateral_name,
        "constraint_tables": [_constraint_table_to_dict(table) for table in result.constraint_tables],
    }


def _constraint_table_to_dict(table: ConstraintTable) -> dict[str, object]:
    return {
        "id": table.id,
        "name": table.name,
        "site": table.site,
        "regime": table.regime,
        "indication_detail": table.indication_detail,
        "dpf_min": table.dpf_min,
        "dpf_max": table.dpf_max,
        "fx_min": table.fx_min,
        "fx_max": table.fx_max,
        "td_min": table.td_min,
        "td_max": table.td_max,
        "prescriptions": list(table.prescriptions),
        "is_bilateral": table.is_bilateral,
        "constraints": [
            {
                "oar_raw": row.oar_raw,
                "metric": row.metric,
                "unit": row.unit,
                "comparator": row.comparator,
                "limit_optimal": row.limit_optimal,
                "limit_maximal": row.limit_maximal,
                "priority": row.priority,
                "source": row.source,
                "comment": row.comment,
            }
            for row in table.constraints
        ],
    }


def _normalize_lookup_name(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _remove_tab(tabs: QTabWidget, widget: QWidget) -> None:
    index = tabs.indexOf(widget)
    if index >= 0:
        tabs.removeTab(index)


def _remove_widget_from_parent_layout(widget: QWidget) -> None:
    parent = widget.parentWidget()
    if parent is not None and parent.layout() is not None:
        parent.layout().removeWidget(widget)


class PlanSelectionDialog(QDialog):
    def __init__(
        self,
        plans: list[PlanDataset],
        current_index: int,
        initial: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plan waehlen")
        self.plan_list = QListWidget()
        self.plan_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.plan_list.setMinimumSize(520, 260)
        for index, plan in enumerate(plans):
            item = QListWidgetItem(_plan_selection_text(plan, index))
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.plan_list.addItem(item)
        if plans:
            self.plan_list.setCurrentRow(max(0, min(current_index, len(plans) - 1)))
        self.plan_list.itemDoubleClicked.connect(lambda _item: self.accept())

        intro_text = (
            "Mehrere RTPLAN-Dateien gefunden. Waehle die Planvariante, die jetzt "
            "geladen werden soll."
            if initial
            else "Waehle eine geladene Planvariante."
        )
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Laden")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.plan_list, 1)
        layout.addWidget(buttons)

    def selected_index(self) -> int | None:
        item = self.plan_list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        try:
            return int(data)
        except (TypeError, ValueError):
            return None


class SettingsDialog(QDialog):
    def __init__(
        self,
        pre_render_on_open: bool,
        hide_console_window: bool,
        ct_render_mode: str,
        help_text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.pre_render_checkbox = QCheckBox("Ansichten und QA beim Falloeffnen vorbereiten")
        self.pre_render_checkbox.setChecked(pre_render_on_open)
        self.hide_console_checkbox = QCheckBox("Konsolenfenster ausblenden")
        self.hide_console_checkbox.setChecked(hide_console_window)
        self.ct_mode_combo = QComboBox()
        self.ct_mode_combo.addItem("Body/Bone surface", "surface")
        self.ct_mode_combo.addItem("Volume rendering", "volume")
        index = self.ct_mode_combo.findData(ct_render_mode)
        if index >= 0:
            self.ct_mode_combo.setCurrentIndex(index)
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.pre_render_checkbox)
        layout.addWidget(self.hide_console_checkbox)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("3D CT"))
        mode_row.addWidget(self.ct_mode_combo, 1)
        layout.addLayout(mode_row)
        layout.addWidget(QLabel("DICOM Laden"))
        layout.addWidget(help_label)
        layout.addWidget(buttons)


def _plan_label(plan: PlanDataset, index: int) -> str:
    info = plan.plan_info
    label = str(info.get("plan_label") or info.get("plan_name") or "").strip()
    if not label:
        label = f"Plan {index + 1}" if plan.beams else "Image / structure set"
    return label


def _plan_selection_text(plan: PlanDataset, index: int) -> str:
    beam_count = len(plan.beams)
    roi_count = len(plan.rois)
    dose_text = "Dose vorhanden" if plan.dose is not None else "keine Dose"
    fraction_count = _safe_int(plan.plan_info.get("number_of_fractions"))
    fraction_text = f", {fraction_count} Fx" if fraction_count else ""
    warning_text = f", {len(plan.warnings)} Warnungen" if plan.warnings else ""
    return (
        f"{index + 1}. {_plan_label(plan, index)}\n"
        f"{beam_count} Beams, {roi_count} ROIs, {dose_text}{fraction_text}{warning_text}"
    )


def _settings_bool(settings: QSettings, key: str, default: bool) -> bool:
    value = settings.value(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _safe_int(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _safe_positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
