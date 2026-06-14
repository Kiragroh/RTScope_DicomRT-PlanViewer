import json
import os

import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox, QStatusBar, QTabWidget

import planeval_viewer.gui.qa_panel as qa_panel_module
from planeval_viewer.dicom_io.models import (
    BeamGeometry,
    ControlPointGeometry,
    CtVolume,
    DoseVolume,
    PlanDataset,
    RoiGeometry,
)
from planeval_viewer.gui.bev_panel import BevPanel
from planeval_viewer.gui.manual_mapping_panel import ManualMappingPanel
from planeval_viewer.gui.main_window import MainWindow
from planeval_viewer.gui.qa_panel import QAPanel
from planeval_viewer.gui.theme import APP_STYLESHEET
from planeval_viewer.refdb.manual_mappings import ManualMappingStore
from planeval_viewer.refdb.matching import RoiLookup
from planeval_viewer.refdb.cache import RefDbCache
from planeval_viewer.refdb.models import ConstraintRow, ConstraintTable, RefDbLookupResult


def test_qa_panel_computes_dvh_constraints_paddick_and_complexity():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan()
    lookup = _lookup_with_constraint("PTV")

    panel.set_plan(plan)
    panel.set_lookups({"PTV": lookup})
    panel.compute_qa()

    assert "PTV" in panel.dvhs
    assert panel.dvh_table.rowCount() == 1
    assert panel.evaluation_table.rowCount() == 1
    assert panel.metrics_table.rowCount() >= 4
    assert panel.complexity_table.rowCount() == 1
    assert panel.plan_info_table.rowCount() >= 2
    assert len(panel.dvh_plot.listDataItems()) == 1
    assert panel.dvh_legend is None
    assert panel.dvh_results_tabs.objectName() == "DvhResultsTabs"
    assert panel.dvh_results_tabs.tabText(0) == "DVH table"
    assert panel.dvh_results_tabs.tabText(1) == "Constraint check"
    assert "#DvhResultsTabs QTabBar::tab:selected" in APP_STYLESHEET

    panel.close()
    app.processEvents()


def test_dvh_lines_follow_roi_visibility_without_recomputing():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan_two_rois()

    panel.set_plan(plan)
    panel.compute_qa()

    assert set(panel.dvh_plot_items) == {"OAR", "PTV"}
    panel.set_roi_visible("OAR", False)

    assert panel.dvh_plot_items["PTV"].isVisible()
    assert not panel.dvh_plot_items["OAR"].isVisible()

    panel.set_roi_visible("OAR", True)
    assert panel.dvh_plot_items["OAR"].isVisible()

    panel.close()
    app.processEvents()


def test_roi_panel_batch_visibility_buttons_drive_viewer_and_dvh(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan_two_rois()
    lookup = _lookup_with_constraint("PTV")

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    window.load_folder(tmp_path)
    window.roi_lookups = {"PTV": lookup}
    window.roi_panel.set_lookups(window.roi_lookups)
    window.qa_panel.set_lookups(window.roi_lookups)
    window.qa_panel.compute_qa()

    window.roi_panel.hide_all_button.click()
    assert all(not roi.visible for roi in plan.rois)
    assert all(not item.isVisible() for item in window.qa_panel.dvh_plot_items.values())

    window.roi_panel.show_matched_button.click()
    assert plan.roi_by_name("PTV").visible
    assert not plan.roi_by_name("OAR").visible
    assert window.qa_panel.dvh_plot_items["PTV"].isVisible()
    assert not window.qa_panel.dvh_plot_items["OAR"].isVisible()

    window.roi_panel.show_all_button.click()
    assert all(roi.visible for roi in plan.rois)

    window.close()
    app.processEvents()


def test_show_matched_keeps_target_visible_even_without_refdb_match(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan_two_rois()
    lookup = _lookup_with_constraint("OAR")

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)

    window.load_folder(tmp_path)
    window.roi_lookups = {"OAR": lookup}
    window.roi_panel.set_lookups(window.roi_lookups)
    window.show()
    app.processEvents()

    window.roi_panel.show_matched_button.click()

    assert plan.roi_by_name("PTV").visible
    assert plan.roi_by_name("OAR").visible

    window.close()
    app.processEvents()


def test_roi_visibility_buttons_have_clear_labels_tooltips_and_style_hooks():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.roi_panel.show_all_button.text() == "Alle an"
    assert window.roi_panel.hide_all_button.text() == "Alle aus"
    assert window.roi_panel.show_matched_button.text() == "Nur Matches"
    assert window.roi_panel.show_all_button.objectName() == "VisibilityShowAllButton"
    assert window.roi_panel.hide_all_button.objectName() == "VisibilityHideAllButton"
    assert window.roi_panel.show_matched_button.objectName() == "VisibilityMatchedButton"
    assert window.roi_panel.show_all_button.minimumHeight() >= 38
    assert window.roi_panel.hide_all_button.minimumHeight() >= 38
    assert window.roi_panel.show_matched_button.minimumHeight() >= 38
    assert "anzeigen" in window.roi_panel.show_all_button.toolTip()
    assert "ausblenden" in window.roi_panel.hide_all_button.toolTip()
    assert "RefDB" in window.roi_panel.show_matched_button.toolTip()
    assert "#31c46b" in APP_STYLESHEET
    assert "#ff6574" in APP_STYLESHEET
    assert "#64b5ff" in APP_STYLESHEET

    window.close()
    app.processEvents()


def test_constraint_check_can_filter_to_selected_hub_table():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan()
    lookup = _lookup_with_two_tables("PTV")

    panel.set_plan(plan)
    panel.set_lookups({"PTV": lookup})
    panel.compute_qa()

    assert panel.constraint_table_combo.count() == 2
    assert panel.constraint_table_combo.itemText(0) == "Primary"
    panel.constraint_table_combo.setCurrentIndex(1)

    assert panel.evaluation_table.rowCount() == 1
    assert panel.evaluation_table.item(0, 2).text() == "Dmean"

    panel.close()
    app.processEvents()


def test_constraint_check_merges_same_hub_table_across_structure_lookups():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan_two_rois()
    table_for_ptv = ConstraintTable(
        id=7,
        name="Whole Hub Protocol",
        constraints=(ConstraintRow(oar_raw="PTV_REF", metric="Dmax", unit="Gy"),),
    )
    table_for_oar = ConstraintTable(
        id=7,
        name="Whole Hub Protocol",
        constraints=(ConstraintRow(oar_raw="OAR_REF", metric="Dmean", unit="Gy"),),
    )
    ptv_result = RefDbLookupResult(
        query_index=0,
        query="PTV",
        matched_name="PTV_REF",
        reference_name="PTV_REF",
        constraint_tables=(table_for_ptv,),
    )
    oar_result = RefDbLookupResult(
        query_index=1,
        query="OAR",
        matched_name="OAR_REF",
        reference_name="OAR_REF",
        constraint_tables=(table_for_oar,),
    )

    panel.set_plan(plan)
    panel.set_lookups(
        {
            "PTV": RoiLookup(source_name="PTV", matched_name="PTV_REF", result=ptv_result),
            "OAR": RoiLookup(source_name="OAR", matched_name="OAR_REF", result=oar_result),
        }
    )
    panel.compute_qa()

    assert panel.constraint_table_combo.count() == 1
    assert panel.constraint_table_combo.itemText(0) == "Whole Hub Protocol"
    assert panel.evaluation_table.rowCount() == 2
    assert {panel.evaluation_table.item(row, 1).text() for row in range(2)} == {"PTV", "OAR"}

    panel.close()
    app.processEvents()


def test_constraint_check_evaluates_whole_table_and_reports_missing_matches():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan_two_rois()
    table = ConstraintTable(
        id=99,
        name="Whole protocol",
        constraints=(
            ConstraintRow(oar_raw="OAR_REF", metric="Dmax", unit="Gy"),
            ConstraintRow(oar_raw="Missing_REF", metric="Dmean", unit="Gy"),
        ),
    )
    result = RefDbLookupResult(
        query_index=0,
        query="OAR",
        matched_name="OAR_REF",
        reference_name="OAR_REF",
        constraint_tables=(table,),
    )

    panel.set_plan(plan)
    panel.set_lookups({"OAR": RoiLookup(source_name="OAR", matched_name="OAR_REF", result=result)})
    panel.compute_qa()

    assert panel.constraint_table_combo.count() == 1
    assert panel.constraint_table_combo.itemText(0) == "Whole protocol"
    assert panel.evaluation_table.rowCount() == 2
    assert panel.evaluation_table.item(0, 0).text() == "OAR_REF"
    assert panel.evaluation_table.item(0, 1).text() == "OAR"
    assert panel.evaluation_table.item(1, 0).text() == "Missing_REF"
    assert panel.evaluation_table.item(1, 7).text() == "missing"
    assert "Missing_REF" in panel.missing_constraints_label.text()

    panel.close()
    app.processEvents()


def test_constraint_missing_mapping_button_emits_reference_and_local_roi():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan_two_rois()
    emitted = []

    panel.set_plan(plan)
    panel.manual_mapping_requested.connect(lambda ref, roi: emitted.append((ref, roi)))
    panel._populate_missing_constraint_combo(["Missing_REF"])
    panel.manual_roi_combo.setCurrentText("OAR")
    panel.apply_mapping_button.click()

    assert emitted == [("Missing_REF", "OAR")]

    panel.close()
    app.processEvents()


def test_constraint_missing_constraints_are_emitted_for_mapping_panel():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan_two_rois()
    table = ConstraintTable(
        id=99,
        name="Whole protocol",
        constraints=(ConstraintRow(oar_raw="Missing_REF", metric="Dmean", unit="Gy"),),
    )
    result = RefDbLookupResult(
        query_index=0,
        query="OAR",
        matched_name="OAR_REF",
        reference_name="OAR_REF",
        constraint_tables=(table,),
    )
    emitted = []

    panel.missing_constraints_changed.connect(lambda names: emitted.append(list(names)))
    panel.set_plan(plan)
    panel.set_lookups({"OAR": RoiLookup(source_name="OAR", matched_name="OAR_REF", result=result)})
    panel.compute_qa()

    assert emitted[-1] == ["Missing_REF"]

    panel.close()
    app.processEvents()


def test_manual_mapping_panel_dropdown_emits_mapping_and_export():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = ManualMappingPanel()
    mappings = []
    exports = []

    panel.mapping_requested.connect(lambda ref, roi: mappings.append((ref, roi)))
    panel.export_requested.connect(lambda: exports.append(True))
    panel.set_options(["Missing_REF"], ["OAR"])
    panel.apply_button.click()
    panel.export_button.click()

    assert mappings == [("Missing_REF", "OAR")]
    assert exports == [True]

    panel.close()
    app.processEvents()


def test_qa_reuses_existing_dvh_cache_on_second_compute(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    plan = _qa_plan()
    calls = []

    original_compute_all_dvhs = qa_panel_module.compute_all_dvhs

    def record_compute(*args, **kwargs):
        calls.append(1)
        return original_compute_all_dvhs(*args, **kwargs)

    monkeypatch.setattr(qa_panel_module, "compute_all_dvhs", record_compute)
    panel.set_plan(plan)

    panel.compute_qa()
    panel.compute_qa()

    assert len(calls) == 1
    assert "PTV" in panel.dvhs

    panel.close()
    app.processEvents()


def test_plan_metrics_show_mu_pam_and_bev_target_context():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()

    panel.set_plan(_qa_plan())
    panel.compute_qa()
    panel.tabs.setCurrentWidget(panel.bev_panel)
    panel.bev_panel.set_control_point_index(1)

    plan_info = _table_values(panel.plan_info_table)
    assert "Total MU" in plan_info
    assert "Prescription Gy" in plan_info
    assert panel.complexity_table.horizontalHeaderItem(5).text() == "PAM"
    assert "Gantry" in panel.bev_panel.control_point_label.text()
    assert panel.bev_panel.target_roi is not None
    assert len(panel.bev_panel.target_items) > 0

    panel.close()
    app.processEvents()


def test_bev_panel_draws_target_as_closed_outline():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = BevPanel()

    panel.set_plan(_qa_plan())
    panel.set_control_point_index(1)

    assert len(panel.target_items) == 1
    item = panel.target_items[0]
    assert np.isclose(item.xData[0], item.xData[-1])
    assert np.isclose(item.yData[0], item.yData[-1])

    panel.close()
    app.processEvents()


def test_bev_panel_exposes_beams_and_control_points():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = BevPanel()

    panel.set_plan(_qa_plan())
    assert panel.beam_combo.count() == 1
    assert panel.control_point_slider.maximum() == 1

    panel.set_control_point_index(1)

    assert "CP 2/2" in panel.control_point_label.text()
    assert len(panel.bev_plot.listDataItems()) > 0

    panel.close()
    app.processEvents()


def test_bev_panel_draws_leaf_banks_as_rectangular_material():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = BevPanel()

    panel.set_plan(_qa_plan())
    panel.set_control_point_index(1)

    assert any(type(item).__name__ == "BarGraphItem" for item in panel.bev_plot.items())

    panel.close()
    app.processEvents()


def test_bev_panel_draws_double_stack_layers_and_closed_leaves():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = BevPanel()
    plan = _qa_plan_double_stack_closed_leaf()

    panel.set_plan(plan)
    panel.set_control_point_index(0)

    assert len(panel.mlc_layer_items) >= 2
    assert len(panel.closed_leaf_items) == 1

    panel.close()
    app.processEvents()


def test_bev_panel_defaults_to_mlc_beam_with_most_control_points_and_animates():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = BevPanel()

    panel.set_plan(_qa_plan_static_plus_arc())

    assert panel.beam_combo.currentIndex() == 1
    assert panel.control_point_slider.maximum() == 3
    assert "Gantry=10.0" in panel.control_point_label.text()
    assert len(panel.linac_items) > 0

    panel._advance_control_point()

    assert panel.control_point_slider.value() == 1
    assert "Gantry=20.0" in panel.control_point_label.text()
    assert len(panel.target_items) > 0

    panel.close()
    app.processEvents()


def test_bev_video_preserves_user_zoom_between_control_points():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = BevPanel()

    panel.set_plan(_qa_plan_static_plus_arc())
    view_box = panel.bev_plot.getViewBox()
    view_box.setXRange(-8.0, 8.0, padding=0)
    view_box.setYRange(-6.0, 6.0, padding=0)
    before = [axis[:] for axis in view_box.viewRange()]

    panel._advance_control_point()

    after = view_box.viewRange()
    assert np.allclose(after[0], before[0])
    assert np.allclose(after[1], before[1])

    panel.close()
    app.processEvents()


def test_main_window_updates_qa_panel_when_plan_loads(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan()

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)

    window.load_folder(tmp_path)

    assert window.qa_panel.plan is plan
    assert window.qa_panel.bev_panel.beam_combo.count() == 1
    assert window.main_splitter.widget(1) is window.dashboard_splitter
    assert window.right_eval_splitter.widget(0) is window.dvh_shell
    assert window.right_eval_splitter.widget(1) is window.qa_shell
    assert isinstance(window.left_tabs, QTabWidget)
    assert window.left_tabs.tabText(0) == "ROIs"
    assert window.left_tabs.tabText(1) == "Details"
    assert window.left_tabs.tabText(2) == "Mappings"
    assert window.left_tabs.tabText(3) == "DICOM Tags"
    assert window.pre_render_on_open is True
    assert "RTPLAN optional" in window.dicom_layout_help_text()

    window.close()
    app.processEvents()


def test_main_window_preloads_views_and_qa_by_default(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan()
    calls = []

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: calls.append("preload"))

    window.load_folder(tmp_path)

    assert calls == ["preload"]

    window.close()
    app.processEvents()


def test_main_window_switches_between_multiple_plans(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    first = _qa_plan()
    first.plan_info["plan_label"] = "Plan A"
    second = _qa_plan_two_rois()
    second.plan_info["plan_label"] = "Plan B"

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [first, second])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)

    window.load_folder(tmp_path)

    assert window.plan_combo.count() == 2
    assert not window.plan_combo.isHidden()
    assert isinstance(window.plan_combo, QComboBox)

    window.plan_combo.setCurrentIndex(1)

    assert window.plan is second
    assert window.qa_panel.plan is second
    assert window.roi_panel.plan is second

    window.close()
    app.processEvents()


def test_main_window_defers_dicom_tag_scan_until_tab_is_opened(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan()
    loaded_folders = []

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)
    monkeypatch.setattr(window.dicom_tag_panel, "set_folder", lambda folder: loaded_folders.append(folder))

    window.load_folder(tmp_path)

    assert loaded_folders == []
    window.left_tabs.setCurrentWidget(window.dicom_tag_panel)

    assert loaded_folders == [tmp_path]

    window.close()
    app.processEvents()


def test_main_window_3d_ct_opacity_slider_updates_viewer():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    values = []

    window.three_d_viewer.set_3d_ct_opacity = lambda opacity: values.append(opacity)
    window.ct3d_opacity_slider.setValue(220)

    assert window.ct3d_opacity_slider.value() == 220
    assert values[-1] == 2.2

    window.close()
    app.processEvents()


def test_main_window_vertical_dose_scale_updates_2d_and_3d_viewers(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan()
    plan.ct = CtVolume(
        voxels=np.zeros((1, 10, 10), dtype=np.float32),
        z_positions=[0.0],
        pixel_spacing=(1.0, 1.0),
        origin_xy=(0.0, 0.0),
    )

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)

    window.load_folder(tmp_path)
    window.dose_min_spin.setValue(1.2)
    window.dose_max_spin.setValue(4.2)

    assert window.dose_range_bar.lower_value == 12
    assert window.dose_range_bar.upper_value == 42
    assert window.viewer.dose_display_min_gy == 1.2
    assert window.viewer.dose_display_max_gy == 4.2
    assert window.three_d_viewer.dose_display_min_gy == 1.2
    assert window.three_d_viewer.dose_display_max_gy == 4.2
    assert "1.2-4.2 Gy" in window.dose_scale_value_label.text()

    window.dose_range_bar.set_range_values(20, 42)

    assert window.dose_min_spin.value() == 2.0
    assert window.viewer.dose_display_min_gy == 2.0

    window.dose_display_mode_combo.setCurrentText("Isodosen")

    assert window.viewer.dose_display_mode == "isodose"
    assert window.three_d_viewer.dose_display_mode == "isodose"

    window.close()
    app.processEvents()


def test_dose_range_bar_drag_previews_without_rerender_until_release(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan()
    plan.ct = CtVolume(
        voxels=np.zeros((1, 10, 10), dtype=np.float32),
        z_positions=[0.0],
        pixel_spacing=(1.0, 1.0),
        origin_xy=(0.0, 0.0),
    )

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)
    window.load_folder(tmp_path)

    updates: list[tuple[str, float, float | None]] = []
    window.viewer.set_dose_display_range_gy = lambda low, high: updates.append(("2d", low, high))
    window.three_d_viewer.set_dose_display_range_gy = lambda low, high: updates.append(
        ("3d", low, high)
    )

    window.dose_range_bar._active_handle = "lower"
    window.dose_range_bar._move_active_handle(window.dose_range_bar._value_to_y(20))

    assert window.dose_min_spin.value() == 2.0
    assert window.dose_scale_value_label.text().startswith("2.0-")
    assert updates == []

    window.dose_range_bar.mouseReleaseEvent(None)

    assert updates == [("2d", 2.0, 6.0), ("3d", 2.0, 6.0)]

    window.close()
    app.processEvents()


def test_main_window_has_single_status_footer_and_loading_overlay():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.findChildren(QStatusBar) == []
    assert window.loading_overlay.isHidden()

    window._set_busy(True, "Loading DICOM plan...")

    assert not window.loading_overlay.isHidden()
    assert window.loading_message.text() == "Loading DICOM plan..."
    assert not window.busy_indicator.isHidden()

    window._set_busy(False, "Ready")

    assert window.loading_overlay.isHidden()
    assert window.status_log.text() == "Ready"

    window.close()
    app.processEvents()


def test_load_folder_keeps_loading_overlay_visible_while_plan_variant_renders(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan()
    overlay_visible_during_variant: list[bool] = []

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])

    def record_variant_load(*_args, **_kwargs):
        overlay_visible_during_variant.append(not window.loading_overlay.isHidden())

    monkeypatch.setattr(window, "_load_plan_variant", record_variant_load)

    window.load_folder(tmp_path)

    assert overlay_visible_during_variant == [True]
    assert window.loading_overlay.isHidden()

    window.close()
    app.processEvents()


def test_3d_header_checkboxes_control_target_and_oar_visibility(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan_two_rois()
    plan.ct = CtVolume(
        voxels=np.zeros((1, 10, 10), dtype=np.float32),
        z_positions=[0.0],
        pixel_spacing=(1.0, 1.0),
        origin_xy=(0.0, 0.0),
    )

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)
    window.load_folder(tmp_path)

    updates: list[tuple[bool, bool]] = []
    window.three_d_viewer.set_3d_structure_group_visibility = (
        lambda show_targets, show_oars: updates.append((show_targets, show_oars))
    )

    assert window.three_d_targets_checkbox.isChecked()
    assert window.three_d_oars_checkbox.isChecked()

    window.three_d_oars_checkbox.setChecked(False)
    assert updates[-1] == (True, False)

    window.three_d_targets_checkbox.setChecked(False)
    assert updates[-1] == (False, False)

    window.close()
    app.processEvents()


def test_refdb_lookup_uses_bundled_offline_constraints_when_hub_is_unavailable(
    monkeypatch, tmp_path
):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan_two_rois()
    plan.plan_info["number_of_fractions"] = 3
    plan.rois[1].name = "SpinalCord"
    window.refdb_cache = RefDbCache(tmp_path / "empty_cache.json")
    window.refdb_client.lookup_batch = lambda names, **_kwargs: [
        RefDbLookupResult(query_index=index, query=name, error="hub unavailable")
        for index, name in enumerate(names)
    ]

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)
    window.load_folder(tmp_path)

    lookup = window.roi_lookups["SpinalCord"]
    assert lookup.matched_name == "SpinalCord"
    assert window.qa_panel.constraint_table_combo.count() >= 1
    assert "STX3:Stereotaxie_3Fx" in [
        window.qa_panel.constraint_table_combo.itemText(index)
        for index in range(window.qa_panel.constraint_table_combo.count())
    ]

    window.close()
    app.processEvents()


def test_main_window_persists_manual_mapping_and_updates_left_panel(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan_two_rois()
    window.manual_mapping_store = ManualMappingStore(tmp_path / "manual.json")
    window.manual_mappings = {}

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)
    window.load_folder(tmp_path)

    window._manual_mapping_requested("OAR_REF", "OAR")

    assert window.manual_mapping_store.load() == {"OAR_REF": "OAR"}
    assert window.roi_lookups["OAR"].matched_name == "OAR_REF"
    assert window.mapping_panel.table.item(0, 0).text() == "OAR_REF"
    assert window.mapping_panel.table.item(0, 1).text() == "OAR"

    window.close()
    app.processEvents()


def test_left_roi_panel_can_save_manual_mapping_without_hub(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = _qa_plan_two_rois()
    window.manual_mapping_store = ManualMappingStore(tmp_path / "manual.json")
    window.manual_mappings = {}

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)
    window.load_folder(tmp_path)

    for row in range(window.roi_panel.table.rowCount()):
        if window.roi_panel.table.item(row, 1).text() == "OAR":
            window.roi_panel.table.selectRow(row)
            break
    window.roi_panel.mapping_reference_combo.setEditText("SpinalCord")
    window.roi_panel.map_selected_button.click()

    assert window.manual_mapping_store.load() == {"SpinalCord": "OAR"}
    assert window.roi_lookups["OAR"].matched_name == "SpinalCord"
    assert window.roi_panel.table.item(window.roi_panel.table.currentRow(), 2).text() == "SpinalCord"

    window.close()
    app.processEvents()


def test_main_window_exports_manual_mapping_json(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    export_path = tmp_path / "exported_mappings.json"
    window.manual_mappings = {"Missing_REF": "OAR"}

    monkeypatch.setattr(
        "planeval_viewer.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "JSON (*.json)"),
    )

    window._export_manual_mappings()

    assert json.loads(export_path.read_text(encoding="utf-8")) == {
        "mappings": {"Missing_REF": "OAR"}
    }

    window.close()
    app.processEvents()


def _qa_plan() -> PlanDataset:
    dose = np.zeros((1, 10, 10), dtype=np.float32)
    dose[:, 2:8, 2:8] = 6.0
    roi = RoiGeometry(
        number=1,
        name="PTV",
        color="#ffcc00",
        contours_by_z={
            0.0: [
                np.array(
                    [
                        [2.0, 2.0, 0.0],
                        [7.0, 2.0, 0.0],
                        [7.0, 7.0, 0.0],
                        [2.0, 7.0, 0.0],
                    ]
                )
            ]
        },
    )
    beam = BeamGeometry(
        number=1,
        name="Arc 1",
        leaf_boundaries=(-10.0, 0.0, 10.0),
        control_points=[
            ControlPointGeometry(
                index=0,
                meterset_weight=0.0,
                gantry_angle=0.0,
                jaws_x=(-20.0, 20.0),
                jaws_y=(-10.0, 10.0),
                isocenter_xyz=(4.5, 4.5, 0.0),
                mlc_x1=(-5.0, -4.0),
                mlc_x2=(5.0, 4.0),
            ),
            ControlPointGeometry(
                index=1,
                meterset_weight=1.0,
                gantry_angle=20.0,
                jaws_x=(-20.0, 20.0),
                jaws_y=(-10.0, 10.0),
                isocenter_xyz=(4.5, 4.5, 0.0),
                mlc_x1=(-6.0, -3.0),
                mlc_x2=(6.0, 3.0),
            ),
        ],
        meterset=123.0,
    )
    return PlanDataset(
        ct=None,
        rois=[roi],
        dose=DoseVolume(
            values_gy=dose,
            z_positions=[0.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        plan_info={"prescription_dose_gy": 5.0},
        beams=[beam],
    )


def _qa_plan_two_rois() -> PlanDataset:
    plan = _qa_plan()
    plan.rois.append(
        RoiGeometry(
            number=2,
            name="OAR",
            color="#00aaff",
            contours_by_z={
                0.0: [
                    np.array(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ]
                    )
                ]
            },
        )
    )
    return plan


def _qa_plan_static_plus_arc() -> PlanDataset:
    plan = _qa_plan()
    static = BeamGeometry(
        number=1,
        name="Setup",
        control_points=[
            ControlPointGeometry(index=0, gantry_angle=0.0),
            ControlPointGeometry(index=1, gantry_angle=0.0),
        ],
    )
    arc = BeamGeometry(
        number=2,
        name="Arc",
        leaf_boundaries=(-10.0, 0.0, 10.0),
        meterset=50.0,
        control_points=[
            ControlPointGeometry(
                index=i,
                meterset_weight=i / 3,
                gantry_angle=10.0 * (i + 1),
                jaws_x=(-20.0, 20.0),
                jaws_y=(-10.0, 10.0),
                isocenter_xyz=(4.5, 4.5, 0.0),
                mlc_x1=(-5.0 - i, -4.0),
                mlc_x2=(5.0 + i, 4.0),
            )
            for i in range(4)
        ],
    )
    plan.beams = [static, arc]
    return plan


def _qa_plan_double_stack_closed_leaf() -> PlanDataset:
    from planeval_viewer.dicom_io.models import MlcLayerGeometry

    plan = _qa_plan()
    cp = ControlPointGeometry(
        index=0,
        gantry_angle=0.0,
        jaws_x=(-50.0, 50.0),
        jaws_y=(-30.0, 30.0),
        mlc_x1=(-5.0, 0.0),
        mlc_x2=(5.0, 0.0),
        mlc_layers=(
            MlcLayerGeometry(
                device_type="MLCX1",
                leaf_boundaries=(-30.0, 0.0, 30.0),
                mlc_x1=(-10.0, 0.0),
                mlc_x2=(10.0, 0.0),
            ),
            MlcLayerGeometry(
                device_type="MLCX2",
                leaf_boundaries=(-20.0, 20.0),
                mlc_x1=(-5.0,),
                mlc_x2=(5.0,),
            ),
        ),
    )
    plan.beams = [
        BeamGeometry(
            number=1,
            name="Halcyon",
            leaf_boundaries=(-30.0, 0.0, 30.0),
            control_points=[cp],
        )
    ]
    return plan


def _lookup_with_constraint(roi_name: str) -> RoiLookup:
    result = RefDbLookupResult(
        query_index=0,
        query=roi_name,
        matched_name=roi_name,
        reference_name=roi_name,
        color="#ffcc00",
        constraint_tables=(
            ConstraintTable(
                name="Synthetic",
                constraints=(
                    ConstraintRow(
                        oar_raw=roi_name,
                        metric="Dmax",
                        unit="Gy",
                        comparator="<=",
                        limit_optimal=5.0,
                        limit_maximal=7.0,
                    ),
                ),
            ),
        ),
    )
    return RoiLookup(
        source_name=roi_name,
        matched_name=roi_name,
        reference_name=roi_name,
        color="#ffcc00",
        result=result,
    )


def _lookup_with_two_tables(roi_name: str) -> RoiLookup:
    result = RefDbLookupResult(
        query_index=0,
        query=roi_name,
        matched_name=roi_name,
        reference_name=roi_name,
        color="#ffcc00",
        constraint_tables=(
            ConstraintTable(
                id=1,
                name="Primary",
                constraints=(ConstraintRow(oar_raw=roi_name, metric="Dmax", unit="Gy"),),
            ),
            ConstraintTable(
                id=2,
                name="Secondary",
                constraints=(ConstraintRow(oar_raw=roi_name, metric="Dmean", unit="Gy"),),
            ),
        ),
    )
    return RoiLookup(
        source_name=roi_name,
        matched_name=roi_name,
        reference_name=roi_name,
        color="#ffcc00",
        result=result,
    )


def _table_values(table):
    values = {}
    for row in range(table.rowCount()):
        key = table.item(row, 0)
        value = table.item(row, 1)
        if key is not None and value is not None:
            values[key.text()] = value.text()
    return values
