import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import planeval_viewer.gui.viewer as viewer_module
from planeval_viewer.dicom_io.models import (
    BeamGeometry,
    ControlPointGeometry,
    CtVolume,
    DoseVolume,
    PlanDataset,
    RoiGeometry,
)
from planeval_viewer.gui.main_window import MainWindow
from planeval_viewer.gui.viewer import AxialPlanViewer, CT_VOLUME_ALPHA_MAX


def test_viewer_steps_slices_and_clamps_to_volume_bounds():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 2, 2), dtype=np.float32),
            z_positions=[0, 1, 2, 3, 4],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    assert viewer.slice_index == 2

    viewer.step_slices(1)
    assert viewer.slice_index == 3

    viewer.step_slices(99)
    assert viewer.slice_index == 4

    viewer.step_slices(-99)
    assert viewer.slice_index == 0

    viewer.close()
    app.processEvents()


def test_ctrl_wheel_zooms_without_changing_slice(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 2, 2), dtype=np.float32),
            z_positions=[0, 1, 2, 3, 4],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )
    viewer.set_plan(plan)
    scaled = []

    def record_scale(scale):
        scaled.append(scale)

    monkeypatch.setattr(viewer.view, "scaleBy", record_scale)

    event = _FakeWheelEvent(120, Qt.KeyboardModifier.ControlModifier)
    handled = viewer._handle_wheel_event(event)

    assert handled
    assert event.accepted
    assert viewer.slice_index == 2
    assert scaled == [(0.85, 0.85)]

    viewer.close()
    app.processEvents()


def test_plain_wheel_scrolls_slices_without_zoom(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 2, 2), dtype=np.float32),
            z_positions=[0, 1, 2, 3, 4],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )
    viewer.set_plan(plan)
    scaled = []
    monkeypatch.setattr(viewer.view, "scaleBy", lambda scale: scaled.append(scale))

    event = _FakeWheelEvent(-120, Qt.KeyboardModifier.NoModifier)
    handled = viewer._handle_wheel_event(event)

    assert handled
    assert event.accepted
    assert viewer.slice_index == 3
    assert scaled == []

    viewer.close()
    app.processEvents()


def test_viewer_switches_axial_sagittal_coronal_and_3d_modes():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 4, 3), dtype=np.float32),
            z_positions=[0, 1, 2, 3, 4],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    assert viewer.view_mode == "axial"
    assert viewer.slice_count() == 5

    viewer.set_view_mode("sagittal")
    assert viewer.view_mode == "sagittal"
    assert viewer.slice_count() == 3
    assert viewer.slice_index == 1
    assert "Sagittal" in viewer.title.text()

    viewer.set_view_mode("coronal")
    assert viewer.view_mode == "coronal"
    assert viewer.slice_count() == 4
    assert viewer.slice_index == 2
    assert "Coronal" in viewer.title.text()

    viewer.set_view_mode("3d")
    assert viewer.view_mode == "3d"
    assert viewer.slice_count() == 1
    assert viewer.slice_index == 0
    assert "3D" in viewer.title.text()

    viewer.close()
    app.processEvents()


def test_main_window_view_mode_actions_update_viewer_and_slice_controls(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 4, 3), dtype=np.float32),
            z_positions=[0, 1, 2, 3, 4],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)

    window.load_folder(tmp_path)
    assert window.view_mode_actions["axial"].isChecked()
    assert "3d" not in window.view_mode_actions
    assert window.three_d_viewer.plan is plan
    assert window.three_d_viewer.view_mode == "3d"

    window.view_mode_actions["sagittal"].trigger()
    assert window.viewer.view_mode == "sagittal"
    assert window.three_d_viewer.view_mode == "3d"
    assert window.slice_slider.maximum() == 2
    assert window.slice_scrollbar.maximum() == 2

    window.view_mode_actions["coronal"].trigger()
    assert window.viewer.view_mode == "coronal"
    assert window.three_d_viewer.view_mode == "3d"
    assert window.slice_slider.maximum() == 3
    assert window.slice_scrollbar.maximum() == 3

    window.close()
    app.processEvents()


def test_main_window_four_view_dashboard_keeps_3d_visible_and_collapsible(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 4, 3), dtype=np.float32),
            z_positions=[0, 1, 2, 3, 4],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )

    monkeypatch.setattr("planeval_viewer.gui.main_window.load_plan_variants", lambda _folder: [plan])
    monkeypatch.setattr(window, "lookup_refdb", lambda: None)
    monkeypatch.setattr(window, "_pre_render_loaded_case", lambda: None)

    window.load_folder(tmp_path)

    assert window.dashboard_splitter.orientation() == Qt.Orientation.Horizontal
    assert window.left_view_splitter.orientation() == Qt.Orientation.Vertical
    assert window.right_eval_splitter.orientation() == Qt.Orientation.Vertical
    assert window.left_view_splitter.widget(0) is window.three_d_shell
    assert window.left_view_splitter.widget(1) is window.viewer_shell
    assert window.right_eval_splitter.widget(0) is window.dvh_shell
    assert window.right_eval_splitter.widget(1) is window.qa_shell
    assert window.left_view_splitter.widget(0).sizePolicy().verticalStretch() == 1
    assert window.left_view_splitter.widget(1).sizePolicy().verticalStretch() == 1
    assert window.dashboard_splitter.widget(0).sizePolicy().horizontalStretch() == 1
    assert window.dashboard_splitter.widget(1).sizePolicy().horizontalStretch() == 1
    assert window.three_d_viewer.view_mode == "3d"
    assert not window.three_d_shell.isHidden()

    window.set_3d_panel_collapsed(True)

    assert window.three_d_shell.isHidden()
    assert not window.viewer_shell.isHidden()

    window.set_3d_panel_collapsed(False)

    assert not window.three_d_shell.isHidden()
    assert window.three_d_viewer.view_mode == "3d"

    window.close()
    app.processEvents()


def test_orthogonal_views_use_physical_spacing_and_superior_slice_at_top():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    voxels = np.arange(5 * 4 * 3, dtype=np.float32).reshape((5, 4, 3))
    plan = PlanDataset(
        ct=CtVolume(
            voxels=voxels,
            z_positions=[0.0, 5.0, 10.0, 15.0, 20.0],
            pixel_spacing=(2.0, 3.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_view_mode("sagittal")
    sagittal = viewer._current_ct_image()
    sagittal_rect = viewer.ct_item.mapRectToParent(viewer.ct_item.boundingRect())

    assert sagittal[0, 0] == voxels[-1, 0, viewer.slice_index]
    assert sagittal_rect.width() == 4 * 2.0
    assert sagittal_rect.height() == 5 * 5.0

    viewer.set_view_mode("coronal")
    coronal = viewer._current_ct_image()
    coronal_rect = viewer.ct_item.mapRectToParent(viewer.ct_item.boundingRect())

    assert coronal[0, 0] == voxels[-1, viewer.slice_index, 0]
    assert coronal_rect.width() == 3 * 3.0
    assert coronal_rect.height() == 5 * 5.0

    viewer.close()
    app.processEvents()


def test_3d_mode_uses_interactive_opengl_view_with_structure_contours():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    roi = RoiGeometry(
        number=1,
        name="PTV",
        color="#ffcc00",
        contours_by_z={
            0.0: [
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [0.0, 2.0, 0.0],
                    ],
                    dtype=float,
                )
            ]
        },
    )
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((3, 3, 3), dtype=np.float32),
            z_positions=[0.0, 2.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[roi],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_view_mode("3d")

    assert viewer.gl_view is not None
    assert hasattr(viewer.gl_view, "orbit")
    assert viewer.view_stack.currentWidget() is viewer.gl_view
    assert len(viewer._gl_items) > 0
    volume_items = [item for item in viewer._gl_items if type(item).__name__ == "GLVolumeItem"]
    assert not volume_items
    assert not any(type(item).__name__ == "GLScatterPlotItem" for item in viewer._gl_items)
    assert not any(type(item).__name__ == "GLImageItem" for item in viewer._gl_items)
    assert any(type(item).__name__ == "GLMeshItem" for item in viewer._gl_items)
    assert any(type(item).__name__ == "GLLinePlotItem" for item in viewer._gl_items)

    viewer.close()
    app.processEvents()


def test_3d_ct_volume_and_surface_are_translucent():
    assert CT_VOLUME_ALPHA_MAX <= 12
    assert 0.0 < viewer_module.CT_SURFACE_ALPHA < viewer_module.CT_BONE_ALPHA < 1.0


def test_3d_ct_volume_alpha_is_masked_to_body():
    ct = CtVolume(
        voxels=np.array(
            [
                [[60.0, 60.0], [700.0, 60.0]],
                [[60.0, 60.0], [60.0, 60.0]],
            ],
            dtype=np.float32,
        ),
        z_positions=[10.0, 12.0],
        pixel_spacing=(2.0, 3.0),
        origin_xy=(100.0, 200.0),
    )
    body_mask = np.array(
        [
            [[False, True], [True, True]],
            [[True, True], [True, False]],
        ],
        dtype=bool,
    )

    rgba, spacing = viewer_module._ct_masked_volume_rgba(ct, body_mask)

    assert rgba[..., 3].max() <= CT_VOLUME_ALPHA_MAX
    assert rgba[0, 0, 0, 3] == 0
    assert rgba[1, 1, 1, 3] == 0
    assert np.count_nonzero(rgba[..., 3]) > 0
    assert spacing == (3.0, 2.0, 2.0)


def test_3d_ct_render_mode_can_switch_between_volume_and_surface(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((4, 4, 4), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )
    calls = []

    viewer.set_plan(plan)
    monkeypatch.setattr(viewer, "_add_masked_ct_volume", lambda *_args: calls.append("volume"))
    monkeypatch.setattr(viewer, "_add_ct_surface_mesh", lambda *_args: calls.append("surface"))

    viewer.set_3d_ct_render_mode("volume")
    viewer.set_view_mode("3d")
    assert "volume" in calls
    assert "surface" not in calls

    calls.clear()
    viewer.set_3d_ct_render_mode("surface")
    assert "surface" in calls
    assert "volume" not in calls

    viewer.close()
    app.processEvents()


def test_3d_ct_opacity_scales_masked_alpha_without_mutating_base():
    rgba = np.zeros((2, 2, 2, 4), dtype=np.ubyte)
    rgba[..., 3] = np.array(
        [
            [[0, 4], [8, 12]],
            [[1, 2], [3, 5]],
        ],
        dtype=np.ubyte,
    )

    scaled = viewer_module._scaled_volume_alpha(rgba, 2.0)

    assert scaled[..., 3].max() == 24
    assert scaled[0, 0, 0, 3] == 0
    assert rgba[..., 3].max() == 12


def test_3d_dose_overlay_adds_isodose_surface():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    dose_values = np.zeros((5, 6, 6), dtype=np.float32)
    dose_values[1:4, 2:5, 2:5] = 10.0
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 6, 6), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=DoseVolume(
            values_gy=dose_values,
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        plan_info={"prescription_dose_gy": 8.0},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_view_mode("3d")

    assert any(
        type(item).__name__ == "GLMeshItem" and getattr(item, "_planeval_role", "") == "dose"
        for item in viewer._gl_items
    )

    viewer.close()
    app.processEvents()


def test_3d_camera_orientation_survives_overlay_redraw():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    dose_values = np.zeros((5, 6, 6), dtype=np.float32)
    dose_values[1:4, 2:5, 2:5] = 10.0
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 6, 6), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=DoseVolume(
            values_gy=dose_values,
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_view_mode("3d")
    viewer.gl_view.opts["azimuth"] = 31
    viewer.gl_view.opts["elevation"] = 42
    viewer.gl_view.opts["distance"] = 321

    viewer.set_dose_opacity(0.65)

    assert viewer.gl_view.opts["azimuth"] == 31
    assert viewer.gl_view.opts["elevation"] == 42
    assert viewer.gl_view.opts["distance"] == 321

    viewer.close()
    app.processEvents()


def test_dose_display_scale_controls_2d_levels_and_3d_isodose_threshold(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    dose_values = np.linspace(0.0, 12.0, num=5 * 6 * 6, dtype=np.float32).reshape((5, 6, 6))
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 6, 6), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=DoseVolume(
            values_gy=dose_values,
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        plan_info={"prescription_dose_gy": 10.0},
        warnings=[],
    )
    thresholds = []

    viewer.set_plan(plan)
    viewer.set_dose_display_range_gy(0.0, 12.0)
    assert np.allclose(viewer.dose_item.levels, [0.0, 12.0])

    def record_threshold(_ct, _volume, threshold):
        thresholds.append(threshold)
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint32)

    monkeypatch.setattr(viewer, "_dose_surface_vertices_faces", record_threshold)
    viewer.set_view_mode("3d")

    assert [round(value, 1) for value in thresholds] == [
        11.0,
        10.5,
        10.0,
        9.5,
        9.0,
        8.0,
        7.0,
        5.0,
        3.0,
        1.0,
    ]

    viewer.close()
    app.processEvents()


def test_isodose_mode_draws_colored_lines_instead_of_dose_overlay():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    dose_values = np.tile(np.linspace(0.0, 12.0, num=24, dtype=np.float32), (24, 1))
    dose_values = dose_values.reshape((1, 24, 24))
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((1, 24, 24), dtype=np.float32),
            z_positions=[0.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=DoseVolume(
            values_gy=dose_values,
            z_positions=[0.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        plan_info={"prescription_dose_gy": 10.0},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_dose_display_range_gy(0.0, 12.0)
    viewer.set_dose_display_mode("isodose")

    assert not viewer.dose_item.isVisible()
    isodose_percents = {
        round(getattr(item, "_planeval_dose_percent", 0.0), 1)
        for item in viewer._contour_items
        if getattr(item, "_planeval_role", "") == "isodose"
    }
    assert isodose_percents >= {110.0, 105.0, 100.0, 95.0, 90.0, 80.0, 70.0, 50.0, 30.0, 10.0}

    viewer.set_dose_display_mode("overlay")

    assert viewer.dose_item.isVisible()

    viewer.close()
    app.processEvents()


def test_ct_privacy_blur_obscures_ct_texture_without_hiding_isodose_lines():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    ct = np.zeros((1, 32, 32), dtype=np.float32)
    ct[:, 8:24, 8:24] = 900.0
    dose_values = np.tile(np.linspace(0.0, 12.0, num=32, dtype=np.float32), (32, 1))
    plan = PlanDataset(
        ct=CtVolume(
            voxels=ct,
            z_positions=[0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=DoseVolume(
            values_gy=dose_values.reshape((1, 32, 32)),
            z_positions=[0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        plan_info={"prescription_dose_gy": 10.0},
        warnings=[],
    )

    viewer.set_plan(plan)
    original_display = np.array(viewer.ct_item.image)
    viewer.set_dose_display_mode("isodose")
    viewer.set_ct_privacy_blur(True)

    blurred_display = np.array(viewer.ct_item.image)
    assert blurred_display.var() < original_display.var()
    assert not np.array_equal(blurred_display, original_display)
    assert any(
        getattr(item, "_planeval_role", "") == "isodose"
        for item in viewer._contour_items
    )

    viewer.close()
    app.processEvents()


def test_isocenter_is_rendered_in_2d_and_3d_views():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 8, 8), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        beams=[
            BeamGeometry(
                number=1,
                name="Arc",
                control_points=[
                    ControlPointGeometry(index=0, isocenter_xyz=(3.0, 4.0, 2.0))
                ],
            )
        ],
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_slice_index(2)

    assert any(getattr(item, "_planeval_role", "") == "isocenter" for item in viewer._contour_items)

    viewer.set_view_mode("3d")

    assert any(getattr(item, "_planeval_role", "") == "isocenter" for item in viewer._gl_items)

    viewer.close()
    app.processEvents()


def test_hover_sampling_reports_ct_and_dose_at_display_pixel():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    voxels = np.arange(1 * 5 * 5, dtype=np.float32).reshape((1, 5, 5))
    dose = np.zeros((1, 5, 5), dtype=np.float32)
    dose[0, 3, 2] = 7.5
    plan = PlanDataset(
        ct=CtVolume(
            voxels=voxels,
            z_positions=[0.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=DoseVolume(
            values_gy=dose,
            z_positions=[0.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    sample = viewer.sample_at_display_position(2.1, 3.2)

    assert sample is not None
    assert sample["ct_hu"] == voxels[0, 3, 2]
    assert sample["dose_gy"] == 7.5

    viewer.close()
    app.processEvents()


def test_3d_structure_selection_follows_left_visibility_controls():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((6, 10, 10), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[
            _box_roi("External", "#888888", 0.5),
            _box_roi("PTV", "#ff0000", 1.0),
            _box_roi("OAR", "#00aaff", 2.0),
            _box_roi("Kidney", "#00ff99", 3.0),
            _box_roi("1AA_ROI_Isocenter", "#ffffff", 4.0),
        ],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_roi_visible("OAR", False)

    names = [roi.name for roi in viewer._rois_for_3d_meshes()]
    assert names == ["PTV", "Kidney"]

    viewer.set_roi_visible("OAR", True)
    names = [roi.name for roi in viewer._rois_for_3d_meshes()]
    assert names == ["PTV", "OAR", "Kidney"]

    viewer.close()
    app.processEvents()


def test_3d_structure_group_visibility_can_hide_targets_and_oars():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((6, 10, 10), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[
            _box_roi("External", "#888888", 0.5),
            _box_roi("PTV", "#ff0000", 1.0),
            _box_roi("CTV", "#ff8800", 2.0),
            _box_roi("SpinalCord", "#00aaff", 3.0),
            _box_roi("Kidney", "#00ff99", 4.0),
        ],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)

    viewer.set_3d_structure_group_visibility(show_targets=False, show_oars=True)
    assert [roi.name for roi in viewer._rois_for_3d_meshes()] == ["SpinalCord", "Kidney"]

    viewer.set_3d_structure_group_visibility(show_targets=True, show_oars=False)
    assert [roi.name for roi in viewer._rois_for_3d_meshes()] == ["PTV", "CTV"]

    viewer.set_3d_structure_group_visibility(show_targets=False, show_oars=False)
    assert [roi.name for roi in viewer._rois_for_3d_meshes()] == []

    viewer.close()
    app.processEvents()


def test_window_presets_do_not_rebuild_3d_volume(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((4, 4, 4), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )
    viewer.set_plan(plan)
    viewer.set_view_mode("3d")
    redraws = []

    monkeypatch.setattr(viewer, "_draw_3d_view", lambda: redraws.append("redraw"))

    viewer.set_window(-600, 1500)

    assert viewer.window_center == -600
    assert viewer.window_width == 1500
    assert redraws == []

    viewer.close()
    app.processEvents()


def test_3d_roi_mesh_uses_cropped_mask_instead_of_full_grid(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    roi = _box_roi("PTV", "#ff0000", 8.0)
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((24, 128, 128), dtype=np.float32),
            z_positions=[float(index) for index in range(24)],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[roi],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    monkeypatch.setattr(
        viewer_module,
        "_roi_mask_on_ct_grid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full grid used")),
    )

    vertices, faces = viewer._roi_mesh_vertices_faces(roi, plan.ct)

    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3

    viewer.close()
    app.processEvents()


def test_3d_mode_renders_focused_structure_mesh_with_context():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    rois = [
        _box_roi("PTV", "#ffcc00", 1.0),
        _box_roi("OAR", "#00aaff", 2.0),
    ]
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 5, 5), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=rois,
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_focus_roi("OAR")
    viewer.set_view_mode("3d")

    assert "Target=OAR" in viewer.title.text()
    assert sum(1 for item in viewer._gl_items if type(item).__name__ == "GLMeshItem") >= 1
    assert any(type(item).__name__ == "GLLinePlotItem" for item in viewer._gl_items)
    assert not any(type(item).__name__ == "GLImageItem" for item in viewer._gl_items)

    viewer.close()
    app.processEvents()


def test_3d_view_is_embedded_and_precreated_when_ct_plan_loads():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((3, 3, 3), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)

    assert viewer.gl_view is not None
    assert viewer.gl_view.parent() is viewer
    assert not viewer.gl_view.isWindow()
    assert viewer.view_stack.currentWidget() is viewer.canvas

    viewer.close()
    app.processEvents()


def test_3d_ct_bone_surface_mesh_uses_high_hu_threshold():
    voxels = np.full((7, 8, 9), -1024.0, dtype=np.float32)
    voxels[1:6, 1:7, 1:8] = 40.0
    voxels[3:5, 3:5, 3:6] = 700.0
    ct = CtVolume(
        voxels=voxels,
        z_positions=[10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
        pixel_spacing=(2.0, 3.0),
        origin_xy=(100.0, 200.0),
    )

    vertices, faces = viewer_module._ct_bone_surface_vertices_faces(ct)

    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert vertices[:, 0].min() >= 100.0
    assert vertices[:, 1].min() >= 200.0
    assert vertices[:, 2].min() >= 10.0


def test_3d_ct_surface_mesh_uses_body_threshold_and_patient_coordinates():
    voxels = np.full((5, 7, 8), -1024.0, dtype=np.float32)
    voxels[1:4, 2:6, 2:7] = 40.0
    ct = CtVolume(
        voxels=voxels,
        z_positions=[10.0, 12.0, 14.0, 16.0, 18.0],
        pixel_spacing=(2.0, 3.0),
        origin_xy=(100.0, 200.0),
    )

    vertices, faces = viewer_module._ct_body_surface_vertices_faces(ct)

    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert vertices[:, 0].min() >= 100.0
    assert vertices[:, 1].min() >= 200.0
    assert vertices[:, 2].min() >= 10.0


def test_3d_context_selection_skips_isocenter_helper_rois():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((5, 8, 8), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[
            _box_roi("PTV", "#ffcc00", 1.0),
            _box_roi("1AA_ROI_Isocenter", "#ffffff", 2.0),
            _box_roi("SpinalCord", "#00aaff", 3.0),
            _box_roi("Kidney", "#00aaff", 4.0),
            _box_roi("Lung", "#00aaff", 5.0),
        ],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)

    assert "1AA_ROI_Isocenter" not in [roi.name for roi in viewer._rois_for_3d_meshes()]

    viewer.close()
    app.processEvents()


def test_3d_mode_prefers_ptv_target_over_ctv():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((4, 6, 6), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0, 3.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[
            _box_roi("CTV", "#00aaff", 1.0),
            _box_roi("PTV", "#ffcc00", 2.0),
        ],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_view_mode("3d")

    assert viewer.focus_roi_name == "PTV"
    assert "Target=PTV" in viewer.title.text()

    viewer.close()
    app.processEvents()


def test_orthogonal_structure_rendering_uses_closed_outlines_not_slice_stripes():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    viewer = AxialPlanViewer()
    roi = RoiGeometry(
        number=1,
        name="PTV",
        color="#ffcc00",
        contours_by_z={
            z: [
                np.array(
                    [
                        [1.0, 1.0, z],
                        [2.0, 1.0, z],
                        [2.0, 2.0, z],
                        [1.0, 2.0, z],
                    ],
                    dtype=float,
                )
            ]
            for z in (0.0, 1.0, 2.0)
        },
    )
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((3, 4, 4), dtype=np.float32),
            z_positions=[0.0, 1.0, 2.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[roi],
        dose=None,
        plan_info={},
        warnings=[],
    )

    viewer.set_plan(plan)
    viewer.set_view_mode("sagittal")
    viewer.set_slice_index(1)

    assert len(viewer._contour_items) <= 2
    assert any(len(np.unique(item.yData)) > 2 for item in viewer._contour_items)

    viewer.close()
    app.processEvents()


class _FakeWheelEvent:
    def __init__(self, delta_y, modifiers):
        self._delta_y = delta_y
        self._modifiers = modifiers
        self.accepted = False
        self.ignored = False

    def angleDelta(self):
        return _FakeDelta(self._delta_y)

    def modifiers(self):
        return self._modifiers

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class _FakeDelta:
    def __init__(self, delta_y):
        self._delta_y = delta_y

    def y(self):
        return self._delta_y


def _box_roi(name: str, color: str, offset: float) -> RoiGeometry:
    return RoiGeometry(
        number=1,
        name=name,
        color=color,
        contours_by_z={
            z: [
                np.array(
                    [
                        [offset, offset, z],
                        [offset + 1.5, offset, z],
                        [offset + 1.5, offset + 1.5, z],
                        [offset, offset + 1.5, z],
                    ],
                    dtype=float,
                )
            ]
            for z in (1.0, 2.0, 3.0)
        },
    )
