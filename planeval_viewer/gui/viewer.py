from __future__ import annotations

from typing import Mapping

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
from skimage.draw import polygon as draw_polygon

from planeval_viewer.computations.dose_grid import resample_dose_to_ct_grid
from planeval_viewer.dicom_io.models import CtVolume, PlanDataset
from planeval_viewer.plan_targets import select_default_target_name
from planeval_viewer.refdb.matching import RoiLookup


pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

CT_VOLUME_ALPHA_MAX = 12
CT_SURFACE_ALPHA = 0.10
CT_BONE_ALPHA = 0.62
CONTEXT_ROI_ALPHA = 0.88
TARGET_ROI_ALPHA = 0.86
ISODOSE_PERCENT_COLORS: tuple[tuple[float, tuple[float, float, float, float]], ...] = (
    (110.0, (0.94, 0.18, 0.95, 0.94)),
    (105.0, (0.95, 0.05, 0.14, 0.94)),
    (100.0, (0.28, 0.84, 0.32, 0.92)),
    (95.0, (0.10, 0.72, 0.22, 0.90)),
    (90.0, (0.05, 0.78, 0.92, 0.88)),
    (80.0, (0.16, 0.38, 1.0, 0.86)),
    (70.0, (1.0, 0.56, 0.08, 0.82)),
    (50.0, (0.48, 0.42, 1.0, 0.76)),
    (30.0, (0.62, 0.22, 0.18, 0.68)),
    (10.0, (0.62, 0.68, 0.12, 0.62)),
)


class AxialPlanViewer(QWidget):
    slice_changed = Signal(int)
    VALID_VIEW_MODES = ("axial", "sagittal", "coronal", "3d")

    def __init__(self) -> None:
        super().__init__()
        self.plan: PlanDataset | None = None
        self.slice_index = 0
        self.view_mode = "axial"
        self.dose_opacity = 0.35
        self.dose_display_min_gy: float = 0.0
        self.dose_display_max_gy: float | None = None
        self.dose_display_mode = "overlay"
        self.ct_3d_opacity = 1.8
        self.ct_3d_render_mode = "surface"
        self.ct_privacy_blur = False
        self.show_3d_targets = True
        self.show_3d_oars = True
        self.window_center = 40.0
        self.window_width = 400.0
        self.visible_rois: set[str] = set()
        self.focus_roi_name: str | None = None
        self.lookups: dict[str, RoiLookup] = {}
        self._dose_slice_cache: dict[tuple[int, int], np.ndarray] = {}
        self._dose_volume_cache: np.ndarray | None = None
        self._roi_mask_cache: dict[str, np.ndarray] = {}
        self._roi_mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._ct_surface_mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._dose_surface_mesh_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._ct_blur_cache: dict[tuple[str, int], np.ndarray] = {}
        self._roi_bounds_cache: dict[str, tuple[float, float, float, float, float, float]] = {}
        self._body_mask_cache: np.ndarray | None = None
        self._masked_ct_rgba_cache: tuple[np.ndarray, tuple[float, float, float]] | None = None
        self._contour_items: list[pg.PlotDataItem] = []
        self.gl_view = None
        self._gl_module = None
        self._gl_items: list[object] = []
        self._auto_range_pending = True
        self._reset_3d_camera_pending = True

        self.title = QLabel("No plan loaded")
        self.title.setObjectName("ViewerTitle")
        self.cursor_label = QLabel("")
        self.cursor_label.setObjectName("CursorDoseLabel")
        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground("#05070b")
        self.canvas.viewport().installEventFilter(self)
        self.canvas.scene().sigMouseMoved.connect(self._mouse_moved)
        self.view = self.canvas.addViewBox(lockAspect=True, enableMouse=True)
        self.view.setMenuEnabled(False)
        self.view.invertY(True)
        self.ct_item = pg.ImageItem()
        self.dose_item = pg.ImageItem()
        self.dose_item.setOpacity(self.dose_opacity)
        self.dose_item.setLookupTable(_dose_lut())
        self.view.addItem(self.ct_item)
        self.view.addItem(self.dose_item)
        self.view_stack = QStackedLayout()
        self.view_stack.setContentsMargins(0, 0, 0, 0)
        self.view_stack.addWidget(self.canvas)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addWidget(self.cursor_label)
        layout.addLayout(self.view_stack, 1)
        self._show_empty("No plan loaded")

    def set_plan(self, plan: PlanDataset | None) -> None:
        self.plan = plan
        self.view_mode = "axial"
        if plan and plan.ct is not None:
            self.slice_index = max(0, self.slice_count() // 2)
            self.visible_rois = {roi.name for roi in plan.rois}
            self.focus_roi_name = _first_target_name(plan)
        else:
            self.slice_index = 0
            self.visible_rois = set()
            self.focus_roi_name = None
        self._dose_slice_cache.clear()
        self._dose_volume_cache = None
        self._roi_mask_cache.clear()
        self._roi_mesh_cache.clear()
        self._ct_surface_mesh_cache.clear()
        self._dose_surface_mesh_cache.clear()
        self._ct_blur_cache.clear()
        self._roi_bounds_cache.clear()
        self._body_mask_cache = None
        self._masked_ct_rgba_cache = None
        self._auto_range_pending = True
        self._reset_3d_camera_pending = True
        self.draw_slice()
        if plan and plan.ct is not None:
            self._ensure_3d_view()
            self._show_2d_view()
        self.slice_changed.emit(self.slice_index)

    def set_view_mode(self, mode: str) -> None:
        normalized = mode.lower()
        if normalized not in self.VALID_VIEW_MODES:
            raise ValueError(f"Unsupported view mode: {mode}")
        if normalized == self.view_mode:
            return

        self.view_mode = normalized
        count = self.slice_count()
        self.slice_index = 0 if normalized == "3d" else max(0, count // 2)
        self._auto_range_pending = True
        self.draw_slice()
        self.slice_changed.emit(self.slice_index)

    def slice_count(self) -> int:
        if self.plan is None or self.plan.ct is None:
            return 0
        depth, rows, columns = self.plan.ct.voxels.shape
        if self.view_mode == "sagittal":
            return int(columns)
        if self.view_mode == "coronal":
            return int(rows)
        if self.view_mode == "3d":
            return 1
        return int(depth)

    def set_lookups(self, lookups: Mapping[str, RoiLookup]) -> None:
        self.lookups = dict(lookups)
        self.draw_slice()

    def set_slice_index(self, index: int) -> None:
        if self.plan is None or self.plan.ct is None:
            return
        max_index = max(0, self.slice_count() - 1)
        new_index = max(0, min(index, max_index))
        if new_index == self.slice_index:
            return
        self.slice_index = new_index
        self.draw_slice()
        self.slice_changed.emit(self.slice_index)

    def step_slices(self, delta: int) -> None:
        self.set_slice_index(self.slice_index + int(delta))

    def set_dose_opacity(self, opacity: float) -> None:
        self.dose_opacity = max(0.0, min(opacity, 1.0))
        self.dose_item.setOpacity(self.dose_opacity)
        if self.view_mode == "3d":
            self.draw_slice()
        else:
            self.draw_slice()

    def set_dose_display_max_gy(self, max_gy: float | None) -> None:
        self.set_dose_display_range_gy(self.dose_display_min_gy, max_gy)

    def set_dose_display_range_gy(
        self,
        min_gy: float | None,
        max_gy: float | None,
    ) -> None:
        lower = 0.0 if min_gy is None else max(0.0, float(min_gy))
        upper = None if max_gy is None else max(0.1, float(max_gy))
        if upper is not None and lower >= upper:
            lower = max(0.0, upper - 0.1)
        self.dose_display_min_gy = lower
        self.dose_display_max_gy = upper
        if self.view_mode == "3d":
            self.draw_slice()
        else:
            self.draw_slice()

    def set_dose_display_mode(self, mode: str) -> None:
        normalized = mode.lower()
        if normalized not in {"overlay", "isodose"}:
            normalized = "overlay"
        self.dose_display_mode = normalized
        if self.view_mode == "3d":
            self.draw_slice()
        else:
            self.draw_slice()

    def set_3d_ct_opacity(self, opacity: float) -> None:
        self.ct_3d_opacity = max(0.0, min(float(opacity), 3.0))
        if self.view_mode == "3d":
            self.draw_slice()

    def set_3d_ct_render_mode(self, mode: str) -> None:
        normalized = mode.lower()
        if normalized not in {"surface", "volume"}:
            normalized = "surface"
        self.ct_3d_render_mode = normalized
        if self.view_mode == "3d":
            self.draw_slice()

    def set_ct_privacy_blur(self, enabled: bool) -> None:
        self.ct_privacy_blur = bool(enabled)
        if self.view_mode != "3d":
            self.draw_slice()

    def set_3d_structure_group_visibility(
        self,
        show_targets: bool,
        show_oars: bool,
    ) -> None:
        self.show_3d_targets = bool(show_targets)
        self.show_3d_oars = bool(show_oars)
        if self.view_mode == "3d":
            self.draw_slice()

    def set_roi_visible(self, roi_name: str, visible: bool) -> None:
        if visible:
            self.visible_rois.add(roi_name)
        else:
            self.visible_rois.discard(roi_name)
        self.draw_slice()

    def set_focus_roi(self, roi_name: str | None) -> None:
        self.focus_roi_name = roi_name
        if self.view_mode == "3d":
            self.draw_slice()

    def set_window(self, center: float, width: float) -> None:
        self.window_center = center
        self.window_width = max(1.0, width)
        if self.view_mode == "3d":
            return
        self.draw_slice()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._handle_wheel_event(event):
            event.ignore()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is self.canvas.viewport() and event.type() == QEvent.Type.Wheel:
            return self._handle_wheel_event(event)
        return super().eventFilter(watched, event)

    def _handle_wheel_event(self, event) -> bool:
        if self.view_mode == "3d":
            event.ignore()
            return False

        angle_delta = event.angleDelta().y()
        if angle_delta == 0:
            event.ignore()
            return False

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 0.85 if angle_delta > 0 else 1.15
            self.view.scaleBy((factor, factor))
            event.accept()
            return True

        self.step_slices(-1 if angle_delta > 0 else 1)
        event.accept()
        return True

    def draw_slice(self) -> None:
        self._clear_contours()
        if self.plan is None:
            self._show_empty("No plan loaded")
            return
        if self.plan.ct is None:
            self._show_empty("No CT available")
            return
        if self.view_mode == "3d":
            self._draw_3d_view()
            return

        self._show_2d_view()
        ct = self.plan.ct
        image = self._current_ct_image()
        image_rect = self._current_image_rect()
        vmin = self.window_center - self.window_width / 2
        vmax = self.window_center + self.window_width / 2
        self.ct_item.setImage(image, autoLevels=False, levels=(vmin, vmax))
        self.ct_item.setRect(image_rect)

        dose = self._dose_image_for_current_view()
        if dose is not None and self.dose_opacity > 0:
            max_dose = self._dose_display_max_for(dose)
            if self.dose_display_mode == "isodose":
                self.dose_item.clear()
                self.dose_item.setVisible(False)
                self._draw_isodose_lines_2d(dose)
            else:
                self.dose_item.setImage(
                    np.ma.filled(np.ma.masked_less_equal(dose, self.dose_display_min_gy), 0.0),
                    autoLevels=False,
                    levels=(self.dose_display_min_gy, max(self.dose_display_min_gy + 0.1, max_dose)),
                )
                self.dose_item.setRect(image_rect)
                self.dose_item.setVisible(True)
        else:
            self.dose_item.clear()
            self.dose_item.setVisible(False)

        self._draw_contours()
        self._draw_isocenter_2d()
        self.title.setText(self._slice_title())
        if self._auto_range_pending:
            self.view.autoRange(padding=0.02)
            self._auto_range_pending = False

    def _dose_display_max_for(self, dose: np.ndarray) -> float:
        if self.dose_display_max_gy is not None:
            return self.dose_display_max_gy
        return float(np.nanmax(dose)) if np.size(dose) else 1.0

    def sample_at_display_position(self, x: float, y: float) -> dict[str, float] | None:
        if self.plan is None or self.plan.ct is None or self.view_mode == "3d":
            return None
        indices = self._voxel_indices_from_display_position(float(x), float(y))
        if indices is None:
            return None
        z_index, row_index, col_index = indices
        ct = self.plan.ct
        sample: dict[str, float] = {
            "ct_hu": float(ct.voxels[z_index, row_index, col_index]),
            "x_mm": float(ct.origin_xy[0] + col_index * ct.pixel_spacing[1]),
            "y_mm": float(ct.origin_xy[1] + row_index * ct.pixel_spacing[0]),
            "z_mm": float(ct.z_positions[z_index]),
        }
        dose_volume = self._dose_volume_for_ct_grid()
        if dose_volume is not None:
            sample["dose_gy"] = float(dose_volume[z_index, row_index, col_index])
        return sample

    def _voxel_indices_from_display_position(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int, int] | None:
        if self.plan is None or self.plan.ct is None:
            return None
        ct = self.plan.ct
        depth, rows, columns = ct.voxels.shape
        row_spacing, col_spacing = ct.pixel_spacing
        if self.view_mode == "sagittal":
            col_index = self.slice_index
            row_index = int(np.floor(x / row_spacing))
            z_index = _z_index_from_display_y(ct.z_positions, y)
        elif self.view_mode == "coronal":
            col_index = int(np.floor(x / col_spacing))
            row_index = self.slice_index
            z_index = _z_index_from_display_y(ct.z_positions, y)
        else:
            col_index = int(np.floor(x / col_spacing))
            row_index = int(np.floor(y / row_spacing))
            z_index = self.slice_index
        if not (0 <= z_index < depth and 0 <= row_index < rows and 0 <= col_index < columns):
            return None
        return z_index, row_index, col_index

    def _mouse_moved(self, scene_position) -> None:
        if self.view_mode == "3d" or self.plan is None or self.plan.ct is None:
            self.cursor_label.setText("")
            return
        try:
            if not self.view.sceneBoundingRect().contains(scene_position):
                self.cursor_label.setText("")
                return
            point = self.view.mapSceneToView(scene_position)
        except Exception:
            return
        sample = self.sample_at_display_position(point.x(), point.y())
        if sample is None:
            self.cursor_label.setText("")
            return
        dose = sample.get("dose_gy")
        dose_text = f"  Dose={dose:.2f} Gy" if dose is not None else "  Dose=n/a"
        self.cursor_label.setText(
            f"x={sample['x_mm']:.1f} y={sample['y_mm']:.1f} z={sample['z_mm']:.1f} mm"
            f"  HU={sample['ct_hu']:.0f}{dose_text}"
        )

    def _current_ct_image(self) -> np.ndarray:
        if self.plan is None or self.plan.ct is None:
            return np.zeros((1, 1), dtype=np.float32)
        voxels = self.plan.ct.voxels
        if self.view_mode == "sagittal":
            image = _superior_first(voxels[:, :, self.slice_index], self.plan.ct.z_positions)
            return self._ct_image_for_display(image)
        if self.view_mode == "coronal":
            image = _superior_first(voxels[:, self.slice_index, :], self.plan.ct.z_positions)
            return self._ct_image_for_display(image)
        if self.view_mode == "3d":
            return np.max(voxels, axis=0)
        return self._ct_image_for_display(voxels[self.slice_index])

    def _ct_image_for_display(self, image: np.ndarray) -> np.ndarray:
        if not self.ct_privacy_blur:
            return image
        cache_key = (self.view_mode, int(self.slice_index))
        cached = self._ct_blur_cache.get(cache_key)
        if cached is not None:
            return cached
        blurred = _privacy_blurred_ct_image(image)
        self._ct_blur_cache[cache_key] = blurred
        return blurred

    def _current_image_rect(self) -> QRectF:
        if self.plan is None or self.plan.ct is None:
            return QRectF(0.0, 0.0, 1.0, 1.0)
        ct = self.plan.ct
        _depth, rows, columns = ct.voxels.shape
        row_spacing, col_spacing = ct.pixel_spacing
        if self.view_mode == "sagittal":
            return QRectF(0.0, 0.0, rows * row_spacing, _z_extent(ct.z_positions))
        if self.view_mode == "coronal":
            return QRectF(0.0, 0.0, columns * col_spacing, _z_extent(ct.z_positions))
        return QRectF(0.0, 0.0, columns * col_spacing, rows * row_spacing)

    def _slice_title(self) -> str:
        if self.plan is None or self.plan.ct is None:
            return "No CT available"
        ct = self.plan.ct
        count = max(1, self.slice_count())
        if self.view_mode == "sagittal":
            x_position = ct.origin_xy[0] + self.slice_index * ct.pixel_spacing[1]
            return f"Sagittal slice {self.slice_index + 1}/{count}  x={x_position:.2f} mm"
        if self.view_mode == "coronal":
            y_position = ct.origin_xy[1] + self.slice_index * ct.pixel_spacing[0]
            return f"Coronal slice {self.slice_index + 1}/{count}  y={y_position:.2f} mm"
        if self.view_mode == "3d":
            return "3D MIP overview"
        z_position = ct.z_positions[self.slice_index]
        return f"Axial slice {self.slice_index + 1}/{count}  z={z_position:.2f} mm"

    def _dose_image_for_current_view(self) -> np.ndarray | None:
        if self.view_mode == "axial":
            return self._dose_slice_for_current_ct()
        volume = self._dose_volume_for_ct_grid()
        if volume is None:
            return None
        if self.view_mode == "sagittal":
            return _superior_first(volume[:, :, self.slice_index], self.plan.ct.z_positions)
        if self.view_mode == "coronal":
            return _superior_first(volume[:, self.slice_index, :], self.plan.ct.z_positions)
        if self.view_mode == "3d":
            return np.max(volume, axis=0)
        return None

    def _dose_volume_for_ct_grid(self) -> np.ndarray | None:
        if self.plan is None or self.plan.ct is None or self.plan.dose is None:
            return None
        ct = self.plan.ct
        dose = self.plan.dose
        if dose.values_gy.shape == ct.voxels.shape:
            return dose.values_gy
        if self._dose_volume_cache is None:
            self._dose_volume_cache = resample_dose_to_ct_grid(ct, dose)
        return self._dose_volume_cache

    def _dose_slice_for_current_ct(self) -> np.ndarray | None:
        if self.plan is None or self.plan.ct is None or self.plan.dose is None:
            return None
        ct = self.plan.ct
        dose = self.plan.dose
        dose_index = dose.nearest_slice_index(ct.z_positions[self.slice_index])
        dose_slice = dose.values_gy[dose_index]
        if dose_slice.shape != ct.voxels[self.slice_index].shape:
            return self._resample_dose_slice_to_ct(dose_slice, dose_index)
        return dose_slice

    def _resample_dose_slice_to_ct(
        self, dose_slice: np.ndarray, dose_index: int
    ) -> np.ndarray | None:
        if self.plan is None or self.plan.ct is None or self.plan.dose is None:
            return None
        ct = self.plan.ct
        dose = self.plan.dose
        if not ct.is_axial_aligned:
            return None
        if not np.allclose(dose.orientation, ct.orientation, rtol=0, atol=1e-3):
            return None

        cache_key = (self.slice_index, dose_index)
        cached = self._dose_slice_cache.get(cache_key)
        if cached is not None:
            return cached

        dose_rows, dose_cols = dose_slice.shape
        dose_row_spacing, dose_col_spacing = dose.pixel_spacing
        dose_origin_x, dose_origin_y = dose.origin_xy
        dose_y = dose_origin_y + np.arange(dose_rows, dtype=float) * dose_row_spacing
        dose_x = dose_origin_x + np.arange(dose_cols, dtype=float) * dose_col_spacing

        ct_rows, ct_cols = ct.voxels[self.slice_index].shape
        ct_row_spacing, ct_col_spacing = ct.pixel_spacing
        ct_origin_x, ct_origin_y = ct.origin_xy
        ct_y = ct_origin_y + np.arange(ct_rows, dtype=float) * ct_row_spacing
        ct_x = ct_origin_x + np.arange(ct_cols, dtype=float) * ct_col_spacing
        grid_y, grid_x = np.meshgrid(ct_y, ct_x, indexing="ij")
        points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

        interpolator = RegularGridInterpolator(
            (dose_y, dose_x),
            dose_slice,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        resampled = interpolator(points).reshape((ct_rows, ct_cols))
        self._dose_slice_cache[cache_key] = resampled
        return resampled

    def _draw_contours(self) -> None:
        if self.plan is None or self.plan.ct is None:
            return
        ct = self.plan.ct
        if not ct.is_axial_aligned:
            return

        if self.view_mode == "sagittal":
            self._draw_orthogonal_contours(axis="x")
            return
        if self.view_mode == "coronal":
            self._draw_orthogonal_contours(axis="y")
            return
        if self.view_mode == "3d":
            self._draw_projected_contours()
            return

        current_z = ct.z_positions[self.slice_index]
        z_tolerance = _slice_tolerance(ct.z_positions)
        for roi in self.plan.rois:
            if roi.name not in self.visible_rois:
                continue
            contours = _contours_on_slice(roi.contours_by_z, current_z, z_tolerance)
            if not contours:
                continue
            color = self.lookups.get(roi.name, RoiLookup(roi.name)).color or roi.color
            for contour in contours:
                x, y = _axial_display_xy(ct, contour)
                item = pg.PlotDataItem(
                    x=x,
                    y=y,
                    pen=pg.mkPen(color=color, width=2),
                    connect="all",
                )
                self.view.addItem(item)
                self._contour_items.append(item)

    def _draw_orthogonal_contours(self, axis: str) -> None:
        if self.plan is None or self.plan.ct is None:
            return
        ct = self.plan.ct
        for roi in self.plan.rois:
            if roi.name not in self.visible_rois:
                continue
            color = self.lookups.get(roi.name, RoiLookup(roi.name)).color or roi.color
            for x_values, y_values in self._orthogonal_roi_outlines(roi, axis):
                item = pg.PlotDataItem(
                    x=x_values,
                    y=y_values,
                    pen=pg.mkPen(color=color, width=2),
                )
                self.view.addItem(item)
                self._contour_items.append(item)

    def _orthogonal_roi_outlines(self, roi, axis: str) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.plan is None or self.plan.ct is None:
            return []
        if not self._roi_intersects_orthogonal_slice(roi, axis):
            return []
        section = _orthogonal_section_mask(roi, self.plan.ct, axis, self.slice_index)
        if not np.any(section):
            return []
        row_spacing, col_spacing = self.plan.ct.pixel_spacing
        x_spacing = row_spacing if axis == "x" else col_spacing
        z_spacing = _z_spacing(self.plan.ct.z_positions)
        outlines: list[tuple[np.ndarray, np.ndarray]] = []
        for contour in measure.find_contours(section.astype(np.float32), 0.5):
            if contour.shape[0] < 3:
                continue
            outlines.append((contour[:, 1] * x_spacing, contour[:, 0] * z_spacing))
        return outlines

    def _roi_mask_for_ct(self, roi) -> np.ndarray:
        if self.plan is None or self.plan.ct is None:
            return np.zeros((0, 0, 0), dtype=bool)
        cached = self._roi_mask_cache.get(roi.name)
        if cached is not None:
            return cached
        mask = _roi_mask_on_ct_grid(roi, self.plan.ct)
        self._roi_mask_cache[roi.name] = mask
        return mask

    def _roi_intersects_orthogonal_slice(self, roi, axis: str) -> bool:
        if self.plan is None or self.plan.ct is None:
            return False
        ct = self.plan.ct
        bounds = self._roi_bounds_cache.get(roi.name)
        if bounds is None:
            bounds = _roi_bounds(roi)
            self._roi_bounds_cache[roi.name] = bounds
        x_min, x_max, y_min, y_max, _z_min, _z_max = bounds
        row_spacing, col_spacing = ct.pixel_spacing
        if axis == "x":
            position = ct.origin_xy[0] + self.slice_index * col_spacing
            tolerance = col_spacing
            return x_min - tolerance <= position <= x_max + tolerance
        position = ct.origin_xy[1] + self.slice_index * row_spacing
        tolerance = row_spacing
        return y_min - tolerance <= position <= y_max + tolerance

    def _draw_projected_contours(self) -> None:
        if self.plan is None or self.plan.ct is None:
            return
        ct = self.plan.ct
        for roi in self.plan.rois:
            if roi.name not in self.visible_rois:
                continue
            color = self.lookups.get(roi.name, RoiLookup(roi.name)).color or roi.color
            for contours in roi.contours_by_z.values():
                for contour in contours:
                    x, y = _axial_display_xy(ct, contour)
                    item = pg.PlotDataItem(
                        x=x,
                        y=y,
                        pen=pg.mkPen(color=color, width=1),
                        connect="all",
                    )
                    self.view.addItem(item)
                    self._contour_items.append(item)

    def _draw_isodose_lines_2d(self, dose: np.ndarray) -> None:
        if dose.size == 0:
            return
        rect = self._current_image_rect()
        rows, columns = dose.shape
        x_scale = rect.width() / max(1, columns)
        y_scale = rect.height() / max(1, rows)
        for level, color, percent in _dose_display_levels(
            dose,
            self.dose_display_min_gy,
            self.dose_display_max_gy,
            _dose_reference_gy(self.plan, dose),
        ):
            try:
                contours = measure.find_contours(np.asarray(dose, dtype=np.float32), level)
            except ValueError:
                continue
            pen = pg.mkPen(color=_rgba_to_255(color, alpha=0.95), width=2)
            for contour in contours:
                if contour.shape[0] < 2:
                    continue
                item = pg.PlotDataItem(
                    x=rect.x() + contour[:, 1] * x_scale,
                    y=rect.y() + contour[:, 0] * y_scale,
                    pen=pen,
                )
                setattr(item, "_planeval_role", "isodose")
                setattr(item, "_planeval_dose_gy", float(level))
                setattr(item, "_planeval_dose_percent", float(percent))
                self.view.addItem(item)
                self._contour_items.append(item)

    def _draw_isocenter_2d(self) -> None:
        position = self._isocenter_display_position()
        if position is None:
            return
        x, y = position
        length = 6.0
        pen = pg.mkPen("#00ffff", width=2)
        horizontal = pg.PlotDataItem(
            x=[x - length, x + length],
            y=[y, y],
            pen=pen,
        )
        vertical = pg.PlotDataItem(
            x=[x, x],
            y=[y - length, y + length],
            pen=pen,
        )
        for item in (horizontal, vertical):
            setattr(item, "_planeval_role", "isocenter")
            self.view.addItem(item)
            self._contour_items.append(item)

    def _isocenter_display_position(self) -> tuple[float, float] | None:
        if self.plan is None or self.plan.ct is None:
            return None
        isocenter = _plan_isocenter(self.plan)
        if isocenter is None:
            return None
        ct = self.plan.ct
        iso_x, iso_y, iso_z = isocenter
        row_spacing, col_spacing = ct.pixel_spacing
        if self.view_mode == "sagittal":
            slice_x = ct.origin_xy[0] + self.slice_index * col_spacing
            if abs(iso_x - slice_x) > col_spacing:
                return None
            return iso_y - ct.origin_xy[1], _z_display_position(ct.z_positions, iso_z)
        if self.view_mode == "coronal":
            slice_y = ct.origin_xy[1] + self.slice_index * row_spacing
            if abs(iso_y - slice_y) > row_spacing:
                return None
            return iso_x - ct.origin_xy[0], _z_display_position(ct.z_positions, iso_z)
        if self.view_mode == "axial":
            current_z = ct.z_positions[self.slice_index]
            if abs(iso_z - current_z) > _slice_tolerance(ct.z_positions):
                return None
            return iso_x - ct.origin_xy[0], iso_y - ct.origin_xy[1]
        return None

    def _draw_3d_view(self) -> None:
        if self.plan is None or self.plan.ct is None:
            self._show_empty("No CT available")
            return

        gl_view = self._ensure_3d_view()
        self._clear_gl_items()

        ct = self.plan.ct
        center = _volume_center(ct)
        span = _volume_span(ct)
        if self._reset_3d_camera_pending:
            gl_view.opts["distance"] = max(span * 2.2, 100.0)
            gl_view.opts["azimuth"] = -45
            gl_view.opts["elevation"] = 20
            self._reset_3d_camera_pending = False

        if self.ct_3d_render_mode == "volume":
            self._add_masked_ct_volume(ct, center)
        else:
            self._add_ct_surface_mesh(ct, center)
        self._add_dose_surface_3d(ct, center)
        self._add_gl_box(ct, center)
        self._add_gl_axes(ct, center)
        self._add_isocenter_3d(ct, center)
        target_name = self._target_name_for_3d()
        rois = self._rois_for_3d_meshes()
        for roi in [item for item in rois if item.name != target_name]:
            added = self._add_roi_mesh(
                roi,
                ct,
                center,
                (0.0, 0.16, 1.0, CONTEXT_ROI_ALPHA),
                draw_edges=True,
                edge_color=(0.0, 0.32, 1.0, 1.0),
            )
            if not added:
                self._add_roi_contour_lines_3d(
                    roi,
                    center,
                    color=(0.0, 0.32, 1.0, 0.95),
                    width=2.0,
                )
            else:
                self._add_roi_contour_lines_3d(
                    roi,
                    center,
                    color=(0.08, 0.38, 1.0, 0.98),
                    width=1.7,
                )
        for roi in [item for item in rois if item.name == target_name]:
            added = self._add_roi_mesh(
                roi,
                ct,
                center,
                (1.0, 0.04, 0.02, TARGET_ROI_ALPHA),
                draw_edges=True,
                edge_color=(1.0, 0.18, 0.1, 1.0),
            )
            if not added:
                self._add_roi_contour_lines_3d(
                    roi,
                    center,
                    color=(1.0, 0.08, 0.02, 0.95),
                    width=2.5,
                )
            else:
                self._add_roi_contour_lines_3d(
                    roi,
                    center,
                    color=(1.0, 0.08, 0.02, 1.0),
                    width=3.2,
                )

        target = f"  Target={target_name}" if target_name else ""
        self.title.setText(f"3D CT volume and structures{target}")
        self.ct_item.clear()
        self.dose_item.clear()
        self.dose_item.setVisible(False)
        self.view_stack.setCurrentWidget(gl_view)
        gl_view.update()

    def _rois_for_3d_meshes(self):
        if self.plan is None:
            return []
        visible = [roi for roi in self.plan.rois if roi.name in self.visible_rois]
        rois = [
            roi
            for roi in visible
            if not _looks_like_body(roi.name)
            and not _looks_like_helper_roi(roi.name)
        ]
        filtered = []
        for roi in rois:
            is_target = _looks_like_target(roi.name)
            if is_target and not self.show_3d_targets:
                continue
            if not is_target and not self.show_3d_oars:
                continue
            filtered.append(roi)
        return filtered

    def _target_name_for_3d(self) -> str | None:
        if self.plan is None:
            return None
        if self.focus_roi_name and self.plan.roi_by_name(self.focus_roi_name) is not None:
            return self.focus_roi_name
        return _first_target_name(self.plan)

    def _ensure_3d_view(self):
        if self.gl_view is not None:
            return self.gl_view
        import pyqtgraph.opengl as gl

        self._gl_module = gl
        self.gl_view = gl.GLViewWidget(parent=self)
        self.gl_view.setBackgroundColor("#05070b")
        self.gl_view.setCameraPosition(distance=400, elevation=20, azimuth=-45)
        self.view_stack.addWidget(self.gl_view)
        return self.gl_view

    def _add_ct_volume(self, ct: CtVolume, center: np.ndarray) -> None:
        if self._gl_module is None or self.gl_view is None:
            return
        rgba, spacing = _ct_volume_rgba(ct)
        if rgba.size == 0:
            return
        item = self._gl_module.GLVolumeItem(
            rgba,
            sliceDensity=1,
            smooth=True,
            glOptions="translucent",
        )
        col_spacing, row_spacing, z_spacing = spacing
        item.scale(col_spacing, row_spacing, z_spacing)
        item.translate(*_ct_volume_translation(ct, center))
        self.gl_view.addItem(item)
        self._gl_items.append(item)

    def _add_masked_ct_volume(self, ct: CtVolume, center: np.ndarray) -> None:
        if self._gl_module is None or self.gl_view is None:
            return
        rgba, spacing = self._masked_ct_rgba_for_ct(ct)
        rgba = _scaled_volume_alpha(rgba, self.ct_3d_opacity)
        if rgba.size == 0:
            return
        item = self._gl_module.GLVolumeItem(
            rgba,
            sliceDensity=1,
            smooth=True,
            glOptions="translucent",
        )
        col_spacing, row_spacing, z_spacing = spacing
        item.scale(col_spacing, row_spacing, z_spacing)
        item.translate(*_ct_volume_translation(ct, center))
        self.gl_view.addItem(item)
        self._gl_items.append(item)

    def _masked_ct_rgba_for_ct(self, ct: CtVolume) -> tuple[np.ndarray, tuple[float, float, float]]:
        if self._masked_ct_rgba_cache is None:
            if self._body_mask_cache is None:
                self._body_mask_cache = _body_mask_for_ct(ct, self.plan.rois if self.plan else [])
            self._masked_ct_rgba_cache = _ct_masked_volume_rgba(ct, self._body_mask_cache)
        return self._masked_ct_rgba_cache

    def _add_ct_surface_mesh(self, ct: CtVolume, center: np.ndarray) -> None:
        if self._gl_module is None or self.gl_view is None:
            return
        body_vertices, body_faces = self._ct_surface_vertices_faces(ct, "body")
        bone_vertices, bone_faces = self._ct_surface_vertices_faces(ct, "bone")
        self._add_ct_mesh_item(
            body_vertices,
            body_faces,
            center=center,
            color=_scaled_alpha_color((0.62, 0.66, 0.70, CT_SURFACE_ALPHA), self.ct_3d_opacity),
            smooth=True,
            role="ct-body",
        )
        self._add_ct_mesh_item(
            bone_vertices,
            bone_faces,
            center=center,
            color=_scaled_alpha_color((0.92, 0.90, 0.82, CT_BONE_ALPHA), self.ct_3d_opacity),
            smooth=False,
            role="ct-bone",
        )

    def _ct_surface_vertices_faces(
        self,
        ct: CtVolume,
        surface: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._ct_surface_mesh_cache.get(surface)
        if cached is not None:
            return cached
        if surface == "bone":
            cached = _ct_bone_surface_vertices_faces(ct)
        else:
            cached = _ct_body_surface_vertices_faces(ct)
        self._ct_surface_mesh_cache[surface] = cached
        return cached

    def _add_dose_surface_3d(self, ct: CtVolume, center: np.ndarray) -> None:
        if self._gl_module is None or self.gl_view is None or self.dose_opacity <= 0:
            return
        dose_volume = self._dose_volume_for_ct_grid()
        if dose_volume is None or dose_volume.size == 0 or not np.any(dose_volume > 0):
            return
        for threshold, color, percent in _dose_display_levels(
            dose_volume,
            self.dose_display_min_gy,
            self.dose_display_max_gy,
            _dose_reference_gy(self.plan, dose_volume),
        ):
            vertices, faces = self._dose_surface_vertices_faces(ct, dose_volume, threshold)
            self._add_ct_mesh_item(
                vertices,
                faces,
                center=center,
                color=(
                    color[0],
                    color[1],
                    color[2],
                    min(0.78, max(0.18, self.dose_opacity * color[3])),
                ),
                smooth=True,
                role="dose",
                extra_attrs={
                    "_planeval_dose_gy": float(threshold),
                    "_planeval_dose_percent": float(percent),
                },
            )

    def _dose_surface_vertices_faces(
        self,
        ct: CtVolume,
        dose_volume: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._dose_surface_mesh_cache.get(float(threshold))
        if cached is not None:
            return cached
        cached = _dose_surface_vertices_faces(dose_volume, ct, threshold)
        self._dose_surface_mesh_cache[float(threshold)] = cached
        return cached

    def _add_ct_mesh_item(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        center: np.ndarray,
        color: tuple[float, float, float, float],
        smooth: bool,
        role: str = "",
        extra_attrs: Mapping[str, object] | None = None,
    ) -> None:
        if self._gl_module is None or self.gl_view is None:
            return
        if vertices.size == 0 or faces.size == 0:
            return
        item = self._gl_module.GLMeshItem(
            vertexes=vertices - center,
            faces=faces,
            faceColor=color,
            smooth=smooth,
            drawEdges=False,
            shader="shaded",
            glOptions="translucent",
        )
        if role:
            setattr(item, "_planeval_role", role)
        for name, value in (extra_attrs or {}).items():
            setattr(item, name, value)
        self.gl_view.addItem(item)
        self._gl_items.append(item)

    def _add_roi_mesh(
        self,
        roi,
        ct: CtVolume,
        center: np.ndarray,
        color: tuple[float, float, float, float],
        draw_edges: bool = False,
        edge_color: tuple[float, float, float, float] | None = None,
    ) -> bool:
        if self._gl_module is None or self.gl_view is None:
            return False
        vertices_xyz, faces = self._roi_mesh_vertices_faces(roi, ct)
        if vertices_xyz.size == 0 or faces.size == 0:
            return False
        item = self._gl_module.GLMeshItem(
            vertexes=vertices_xyz - center,
            faces=faces,
            faceColor=color,
            smooth=False,
            drawEdges=draw_edges,
            edgeColor=edge_color or color,
            shader="shaded",
            glOptions="translucent",
        )
        self.gl_view.addItem(item)
        self._gl_items.append(item)
        return True

    def _roi_mesh_vertices_faces(self, roi, ct: CtVolume) -> tuple[np.ndarray, np.ndarray]:
        cached = self._roi_mesh_cache.get(roi.name)
        if cached is not None:
            return cached
        empty = (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=np.uint32),
        )
        mask, z_start, row_start, col_start = _cropped_roi_mask_on_ct_grid(roi, ct)
        if mask.size == 0 or not np.any(mask) or np.all(mask):
            self._roi_mesh_cache[roi.name] = empty
            return empty
        z_spacing = _z_spacing(ct.z_positions)
        row_spacing, col_spacing = ct.pixel_spacing
        try:
            vertices, faces, _normals, _values = measure.marching_cubes(
                mask.astype(np.float32),
                level=0.5,
                spacing=(z_spacing, row_spacing, col_spacing),
            )
        except ValueError:
            self._roi_mesh_cache[roi.name] = empty
            return empty
        z_positions = ct.z_positions[z_start : z_start + mask.shape[0]]
        z_min = min(float(position) for position in z_positions) if z_positions else 0.0
        origin_x, origin_y = ct.origin_xy
        row_spacing, col_spacing = ct.pixel_spacing
        vertices_xyz = np.column_stack(
            (
                origin_x + col_start * col_spacing + vertices[:, 2],
                origin_y + row_start * row_spacing + vertices[:, 1],
                z_min + vertices[:, 0],
            )
        )
        cached = (vertices_xyz, faces)
        self._roi_mesh_cache[roi.name] = cached
        return cached

    def _add_roi_contour_lines_3d(
        self,
        roi,
        center: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
    ) -> None:
        for contours in roi.contours_by_z.values():
            for contour in contours:
                if contour.shape[0] < 2:
                    continue
                self._add_gl_line(_closed_contour(contour) - center, color=color, width=width)

    def _add_gl_box(self, ct, center: np.ndarray) -> None:
        x_min, x_max, y_min, y_max, z_min, z_max = _volume_bounds(ct)
        corners = {
            "lbf": np.array([x_min, y_min, z_min], dtype=float),
            "rbf": np.array([x_max, y_min, z_min], dtype=float),
            "ltf": np.array([x_min, y_max, z_min], dtype=float),
            "rtf": np.array([x_max, y_max, z_min], dtype=float),
            "lbb": np.array([x_min, y_min, z_max], dtype=float),
            "rbb": np.array([x_max, y_min, z_max], dtype=float),
            "ltb": np.array([x_min, y_max, z_max], dtype=float),
            "rtb": np.array([x_max, y_max, z_max], dtype=float),
        }
        edges = (
            ("lbf", "rbf"),
            ("rbf", "rtf"),
            ("rtf", "ltf"),
            ("ltf", "lbf"),
            ("lbb", "rbb"),
            ("rbb", "rtb"),
            ("rtb", "ltb"),
            ("ltb", "lbb"),
            ("lbf", "lbb"),
            ("rbf", "rbb"),
            ("rtf", "rtb"),
            ("ltf", "ltb"),
        )
        for start_key, end_key in edges:
            self._add_gl_line(
                np.vstack((corners[start_key], corners[end_key])) - center,
                color=(0.75, 0.82, 0.92, 0.55),
                width=1.0,
            )

    def _add_gl_axes(self, ct, center: np.ndarray) -> None:
        x_min, x_max, y_min, y_max, z_min, z_max = _volume_bounds(ct)
        origin = np.array([x_min, y_min, z_min], dtype=float)
        self._add_gl_line(
            np.vstack((origin, np.array([x_max, y_min, z_min], dtype=float))) - center,
            color=(1.0, 0.25, 0.25, 0.9),
            width=2.0,
        )
        self._add_gl_line(
            np.vstack((origin, np.array([x_min, y_max, z_min], dtype=float))) - center,
            color=(0.2, 0.9, 0.35, 0.9),
            width=2.0,
        )
        self._add_gl_line(
            np.vstack((origin, np.array([x_min, y_min, z_max], dtype=float))) - center,
            color=(0.35, 0.6, 1.0, 0.9),
            width=2.0,
        )

    def _add_isocenter_3d(self, ct: CtVolume, center: np.ndarray) -> None:
        isocenter = _plan_isocenter(self.plan)
        if isocenter is None:
            return
        iso = np.array(isocenter, dtype=float)
        length = max(4.0, _volume_span(ct) * 0.035)
        axes = (
            np.array([[iso[0] - length, iso[1], iso[2]], [iso[0] + length, iso[1], iso[2]]]),
            np.array([[iso[0], iso[1] - length, iso[2]], [iso[0], iso[1] + length, iso[2]]]),
            np.array([[iso[0], iso[1], iso[2] - length], [iso[0], iso[1], iso[2] + length]]),
        )
        for points in axes:
            self._add_gl_line(
                points - center,
                color=(0.0, 1.0, 1.0, 1.0),
                width=2.5,
                role="isocenter",
            )

    def _add_gl_line(
        self,
        points: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
        role: str = "",
    ) -> None:
        if self._gl_module is None or self.gl_view is None or points.size == 0:
            return
        item = self._gl_module.GLLinePlotItem(
            pos=np.asarray(points, dtype=float),
            color=color,
            width=width,
            antialias=True,
            mode="line_strip",
        )
        if role:
            setattr(item, "_planeval_role", role)
        self.gl_view.addItem(item)
        self._gl_items.append(item)

    def _clear_gl_items(self) -> None:
        if self.gl_view is not None:
            for item in self._gl_items:
                self.gl_view.removeItem(item)
        self._gl_items.clear()

    def _show_2d_view(self) -> None:
        self.view_stack.setCurrentWidget(self.canvas)

    def _clear_contours(self) -> None:
        for item in self._contour_items:
            self.view.removeItem(item)
        self._contour_items.clear()

    def _show_empty(self, message: str) -> None:
        self._show_2d_view()
        self._clear_gl_items()
        self.ct_item.clear()
        self.dose_item.clear()
        self.dose_item.setVisible(False)
        self.title.setText(message)


def _dose_lut() -> np.ndarray:
    try:
        return pg.colormap.get("turbo").getLookupTable(nPts=256)
    except Exception:
        return pg.colormap.get("inferno").getLookupTable(nPts=256)


def _slice_tolerance(z_positions: list[float]) -> float:
    if len(z_positions) < 2:
        return 0.75
    diffs = np.abs(np.diff(sorted(z_positions)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.75
    return max(0.75, float(np.median(diffs)) / 2.0)


def _contours_on_slice(
    contours_by_z: dict[float, list[np.ndarray]], z_position: float, tolerance: float
) -> list[np.ndarray]:
    contours: list[np.ndarray] = []
    for z_key, items in contours_by_z.items():
        if abs(float(z_key) - z_position) <= tolerance:
            contours.extend(items)
    return contours


def _contour_line_intersections(
    contour: np.ndarray,
    fixed_axis: int,
    fixed_position: float,
) -> list[float]:
    if contour.shape[0] < 2:
        return []
    other_axis = 1 if fixed_axis == 0 else 0
    values: list[float] = []
    closed = np.vstack([contour, contour[0]])
    eps = 1e-6
    for start, end in zip(closed[:-1], closed[1:]):
        start_fixed = float(start[fixed_axis])
        end_fixed = float(end[fixed_axis])
        start_delta = start_fixed - fixed_position
        end_delta = end_fixed - fixed_position

        if abs(start_delta) <= eps and abs(end_delta) <= eps:
            values.extend([float(start[other_axis]), float(end[other_axis])])
            continue
        if start_delta * end_delta > 0:
            continue
        if abs(end_fixed - start_fixed) <= eps:
            continue
        fraction = (fixed_position - start_fixed) / (end_fixed - start_fixed)
        if -eps <= fraction <= 1.0 + eps:
            other_value = float(
                start[other_axis] + fraction * (end[other_axis] - start[other_axis])
            )
            values.append(other_value)
    return sorted(_unique_float_values(values))


def _unique_float_values(values: list[float], tolerance: float = 1e-5) -> list[float]:
    unique: list[float] = []
    for value in sorted(values):
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return unique


def _paired_intervals(values: list[float]) -> list[tuple[float, float]]:
    if len(values) < 2:
        return []
    intervals: list[tuple[float, float]] = []
    for index in range(0, len(values) - 1, 2):
        start = values[index]
        end = values[index + 1]
        if abs(end - start) > 1e-5:
            intervals.append((start, end))
    return intervals


def _superior_first(image: np.ndarray, z_positions: list[float]) -> np.ndarray:
    if len(z_positions) >= 2 and float(z_positions[-1]) > float(z_positions[0]):
        return np.flip(image, axis=0)
    return image


def _privacy_blurred_ct_image(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    if data.size == 0:
        return data.copy()
    finite = np.isfinite(data)
    if not np.all(finite):
        fill = float(np.nanmedian(data[finite])) if np.any(finite) else 0.0
        data = np.where(finite, data, fill).astype(np.float32, copy=False)
    sigma_y = max(2.5, data.shape[0] / 24.0)
    sigma_x = max(2.5, data.shape[1] / 24.0)
    return gaussian_filter(data, sigma=(sigma_y, sigma_x), mode="nearest")


def _z_spacing(z_positions: list[float]) -> float:
    if len(z_positions) < 2:
        return 1.0
    diffs = np.abs(np.diff(np.array(z_positions, dtype=float)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _z_extent(z_positions: list[float]) -> float:
    return max(1.0, len(z_positions) * _z_spacing(z_positions))


def _z_display_position(z_positions: list[float], z_position: float) -> float:
    z_max = max(float(position) for position in z_positions) if z_positions else 0.0
    return z_max + _z_spacing(z_positions) / 2.0 - float(z_position)


def _z_index_from_display_y(z_positions: list[float], display_y: float) -> int:
    if not z_positions:
        return 0
    z_max = max(float(position) for position in z_positions)
    z_position = z_max + _z_spacing(z_positions) / 2.0 - float(display_y)
    distances = np.abs(np.array(z_positions, dtype=float) - z_position)
    return int(np.argmin(distances))


def _axial_display_xy(ct: CtVolume, contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin_x, origin_y = ct.origin_xy
    return contour[:, 0] - origin_x, contour[:, 1] - origin_y


def _volume_bounds(ct: CtVolume) -> tuple[float, float, float, float, float, float]:
    _depth, rows, columns = ct.voxels.shape
    row_spacing, col_spacing = ct.pixel_spacing
    origin_x, origin_y = ct.origin_xy
    z_spacing = _z_spacing(ct.z_positions)
    z_min = min(float(position) for position in ct.z_positions) if ct.z_positions else 0.0
    z_max = max(float(position) for position in ct.z_positions) if ct.z_positions else z_spacing
    return (
        origin_x,
        origin_x + max(1, columns - 1) * col_spacing,
        origin_y,
        origin_y + max(1, rows - 1) * row_spacing,
        z_min,
        z_max,
    )


def _volume_center(ct: CtVolume) -> np.ndarray:
    x_min, x_max, y_min, y_max, z_min, z_max = _volume_bounds(ct)
    return np.array(
        [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0],
        dtype=float,
    )


def _volume_span(ct: CtVolume) -> float:
    x_min, x_max, y_min, y_max, z_min, z_max = _volume_bounds(ct)
    return max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)


def _ct_volume_translation(ct: CtVolume, center: np.ndarray) -> tuple[float, float, float]:
    x_min, _x_max, y_min, _y_max, z_min, _z_max = _volume_bounds(ct)
    return (
        float(x_min - center[0]),
        float(y_min - center[1]),
        float(z_min - center[2]),
    )


def _ct_volume_rgba(
    ct: CtVolume,
    max_axis: int = 96,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    voxels = np.asarray(ct.voxels, dtype=np.float32)
    if voxels.size == 0:
        return np.zeros((0, 0, 0, 4), dtype=np.ubyte), (1.0, 1.0, 1.0)
    steps = tuple(max(1, int(np.ceil(size / max_axis))) for size in voxels.shape)
    sampled = voxels[:: steps[0], :: steps[1], :: steps[2]]
    clipped = np.clip(sampled, -700.0, 900.0)
    gray = ((clipped + 700.0) / 1600.0 * 165.0).astype(np.ubyte)
    alpha_source = np.clip((sampled + 250.0) / 1150.0, 0.0, 1.0)
    alpha = (alpha_source**2.4 * CT_VOLUME_ALPHA_MAX).astype(np.ubyte)
    rgba = np.zeros((sampled.shape[2], sampled.shape[1], sampled.shape[0], 4), dtype=np.ubyte)
    transposed_gray = np.transpose(gray, (2, 1, 0))
    rgba[..., 0] = transposed_gray
    rgba[..., 1] = transposed_gray
    rgba[..., 2] = transposed_gray
    rgba[..., 3] = np.transpose(alpha, (2, 1, 0))
    row_spacing, col_spacing = ct.pixel_spacing
    return (
        rgba,
        (
            col_spacing * steps[2],
            row_spacing * steps[1],
            _z_spacing(ct.z_positions) * steps[0],
        ),
    )


def _body_mask_for_ct(ct: CtVolume, rois) -> np.ndarray:
    for roi in rois:
        if not _looks_like_body(roi.name):
            continue
        mask = _roi_mask_on_ct_grid(roi, ct)
        if np.any(mask):
            return mask
    voxels = np.asarray(ct.voxels, dtype=np.float32)
    if voxels.size == 0:
        return np.zeros_like(voxels, dtype=bool)
    mask = np.isfinite(voxels) & (voxels > -650.0)
    if not np.any(mask):
        return np.ones_like(voxels, dtype=bool)
    return _largest_component(mask)


def _ct_masked_volume_rgba(
    ct: CtVolume,
    body_mask: np.ndarray,
    max_axis: int = 96,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    voxels = np.asarray(ct.voxels, dtype=np.float32)
    if voxels.size == 0:
        return np.zeros((0, 0, 0, 4), dtype=np.ubyte), (1.0, 1.0, 1.0)

    steps = tuple(max(1, int(np.ceil(size / max_axis))) for size in voxels.shape)
    sampled = voxels[:: steps[0], :: steps[1], :: steps[2]]
    mask = np.asarray(body_mask, dtype=bool)
    if mask.shape != voxels.shape:
        mask = np.ones_like(voxels, dtype=bool)
    sampled_mask = mask[:: steps[0], :: steps[1], :: steps[2]]

    clipped = np.clip(sampled, -700.0, 900.0)
    gray = ((clipped + 700.0) / 1600.0 * 180.0).astype(np.ubyte)
    alpha_source = np.clip((sampled + 450.0) / 1700.0, 0.0, 1.0)
    alpha = (alpha_source**2.0 * CT_VOLUME_ALPHA_MAX).astype(np.ubyte)
    alpha = np.where(sampled_mask, alpha, 0).astype(np.ubyte)

    rgba = np.zeros((sampled.shape[2], sampled.shape[1], sampled.shape[0], 4), dtype=np.ubyte)
    transposed_gray = np.transpose(gray, (2, 1, 0))
    rgba[..., 0] = transposed_gray
    rgba[..., 1] = transposed_gray
    rgba[..., 2] = transposed_gray
    rgba[..., 3] = np.transpose(alpha, (2, 1, 0))
    row_spacing, col_spacing = ct.pixel_spacing
    return (
        rgba,
        (
            col_spacing * steps[2],
            row_spacing * steps[1],
            _z_spacing(ct.z_positions) * steps[0],
        ),
    )


def _scaled_volume_alpha(rgba: np.ndarray, opacity: float) -> np.ndarray:
    scaled = np.array(rgba, copy=True)
    if scaled.size == 0:
        return scaled
    alpha = scaled[..., 3].astype(np.float32) * max(0.0, float(opacity))
    scaled[..., 3] = np.clip(alpha, 0, 255).astype(np.ubyte)
    return scaled


def _scaled_alpha_color(
    color: tuple[float, float, float, float],
    opacity: float,
) -> tuple[float, float, float, float]:
    return (color[0], color[1], color[2], min(1.0, max(0.0, color[3] * max(0.0, opacity))))


def _dose_reference_gy(plan: PlanDataset | None, dose_volume: np.ndarray) -> float | None:
    if plan is not None:
        try:
            prescription = float(plan.plan_info.get("prescription_dose_gy") or 0.0)
        except (TypeError, ValueError):
            prescription = 0.0
        if prescription > 0:
            return prescription
    if dose_volume.size == 0:
        return None
    try:
        max_dose = float(np.nanmax(dose_volume))
    except ValueError:
        return None
    return max_dose if max_dose > 0 else None


def _dose_display_levels(
    dose_volume: np.ndarray,
    display_min_gy: float,
    display_max_gy: float | None,
    reference_gy: float | None = None,
) -> list[tuple[float, tuple[float, float, float, float], float]]:
    max_dose = float(np.nanmax(dose_volume)) if dose_volume.size else 0.0
    min_dose = float(np.nanmin(dose_volume)) if dose_volume.size else 0.0
    if max_dose <= 0:
        return []
    try:
        reference = float(reference_gy or 0.0)
    except (TypeError, ValueError):
        reference = 0.0
    if reference <= 0:
        reference = _dose_reference_gy(None, dose_volume) or max_dose
    scale_max = max_dose if display_max_gy is None else float(display_max_gy)
    scale_min = max(0.0, float(display_min_gy))
    if scale_max <= scale_min:
        return []
    levels: list[tuple[float, tuple[float, float, float, float], float]] = []
    seen: set[float] = set()
    for percent, color in ISODOSE_PERCENT_COLORS:
        threshold = reference * percent / 100.0
        if threshold < scale_min or threshold > scale_max:
            continue
        if threshold <= min_dose or threshold >= max_dose:
            continue
        rounded = round(threshold, 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        levels.append((threshold, color, percent))
    return levels


def _rgba_to_255(
    color: tuple[float, float, float, float],
    alpha: float | None = None,
) -> tuple[int, int, int, int]:
    use_alpha = color[3] if alpha is None else alpha
    return (
        int(np.clip(color[0], 0.0, 1.0) * 255),
        int(np.clip(color[1], 0.0, 1.0) * 255),
        int(np.clip(color[2], 0.0, 1.0) * 255),
        int(np.clip(use_alpha, 0.0, 1.0) * 255),
    )


def _dose_surface_vertices_faces(
    dose_volume: np.ndarray,
    ct: CtVolume,
    threshold: float,
    max_axis: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    volume = np.asarray(dose_volume, dtype=np.float32)
    if volume.size == 0 or float(np.nanmax(volume)) <= threshold or float(np.nanmin(volume)) >= threshold:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint32)
    steps = tuple(max(1, int(np.ceil(size / max_axis))) for size in volume.shape)
    sampled = volume[:: steps[0], :: steps[1], :: steps[2]]
    if sampled.size == 0 or float(np.nanmax(sampled)) <= threshold:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint32)
    z_spacing = _z_spacing(ct.z_positions) * steps[0]
    row_spacing, col_spacing = ct.pixel_spacing
    try:
        vertices, faces, _normals, _values = measure.marching_cubes(
            sampled,
            level=threshold,
            spacing=(z_spacing, row_spacing * steps[1], col_spacing * steps[2]),
        )
    except ValueError:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint32)
    z_min = min(float(position) for position in ct.z_positions) if ct.z_positions else 0.0
    origin_x, origin_y = ct.origin_xy
    vertices_xyz = np.column_stack(
        (
            origin_x + vertices[:, 2],
            origin_y + vertices[:, 1],
            z_min + vertices[:, 0],
        )
    )
    return vertices_xyz, faces


def _ct_body_surface_vertices_faces(
    ct: CtVolume,
    max_axis: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    return _ct_threshold_surface_vertices_faces(
        ct,
        threshold_hu=-500.0,
        largest_component=True,
        max_axis=max_axis,
    )


def _ct_bone_surface_vertices_faces(
    ct: CtVolume,
    max_axis: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    return _ct_threshold_surface_vertices_faces(
        ct,
        threshold_hu=180.0,
        largest_component=False,
        max_axis=max_axis,
    )


def _ct_threshold_surface_vertices_faces(
    ct: CtVolume,
    threshold_hu: float,
    largest_component: bool,
    max_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    voxels = np.asarray(ct.voxels, dtype=np.float32)
    if voxels.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint32)

    steps = tuple(max(1, int(np.ceil(size / max_axis))) for size in voxels.shape)
    sampled = voxels[:: steps[0], :: steps[1], :: steps[2]]
    mask = np.isfinite(sampled) & (sampled > threshold_hu)
    if not np.any(mask):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint32)

    if largest_component:
        mask = _largest_component(mask)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    row_spacing, col_spacing = ct.pixel_spacing
    spacing = np.array(
        (
            _z_spacing(ct.z_positions) * steps[0],
            row_spacing * steps[1],
            col_spacing * steps[2],
        ),
        dtype=float,
    )
    try:
        vertices, faces, _normals, _values = measure.marching_cubes(
            padded.astype(np.float32),
            level=0.5,
            spacing=tuple(spacing),
        )
    except ValueError:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint32)

    vertices = vertices - spacing
    z_min = min(float(position) for position in ct.z_positions) if ct.z_positions else 0.0
    origin_x, origin_y = ct.origin_xy
    vertices_xyz = np.column_stack(
        (
            origin_x + vertices[:, 2],
            origin_y + vertices[:, 1],
            z_min + vertices[:, 0],
        )
    )
    return vertices_xyz.astype(float), faces.astype(np.uint32, copy=False)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask, connectivity=1)
    if labels.max() <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _window_to_uint8(values: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2.0
    upper = center + width / 2.0
    normalized = np.clip((values - lower) / max(width, 1.0), 0.0, 1.0)
    return (normalized * 255.0).astype(np.ubyte)


def _orthogonal_section_mask(
    roi,
    ct: CtVolume,
    axis: str,
    slice_index: int,
) -> np.ndarray:
    depth, rows, columns = ct.voxels.shape
    width = rows if axis == "x" else columns
    section = np.zeros((depth, width), dtype=bool)
    row_spacing, col_spacing = ct.pixel_spacing
    origin_x, origin_y = ct.origin_xy
    if axis == "x":
        fixed_position = origin_x + slice_index * col_spacing
        fixed_axis = 0
        display_origin = origin_y
        display_spacing = row_spacing
    else:
        fixed_position = origin_y + slice_index * row_spacing
        fixed_axis = 1
        display_origin = origin_x
        display_spacing = col_spacing

    z_ascending = len(ct.z_positions) < 2 or float(ct.z_positions[-1]) > float(ct.z_positions[0])
    for z_position, contours in roi.contours_by_z.items():
        z_index = ct.nearest_slice_index(float(z_position))
        display_z_index = depth - 1 - z_index if z_ascending else z_index
        for contour in contours:
            intersections = _contour_line_intersections(
                contour,
                fixed_axis=fixed_axis,
                fixed_position=fixed_position,
            )
            for start, end in _paired_intervals(intersections):
                start_px = int(np.floor((min(start, end) - display_origin) / display_spacing))
                end_px = int(np.ceil((max(start, end) - display_origin) / display_spacing))
                start_px = max(0, min(width - 1, start_px))
                end_px = max(0, min(width - 1, end_px))
                if end_px >= start_px:
                    section[display_z_index, start_px : end_px + 1] = True
    return section


def _closed_contour(contour: np.ndarray) -> np.ndarray:
    points = np.asarray(contour[:, :3], dtype=float)
    if points.shape[0] == 0 or np.allclose(points[0], points[-1], rtol=0, atol=1e-6):
        return points
    return np.vstack((points, points[0]))


def _roi_bounds(roi) -> tuple[float, float, float, float, float, float]:
    arrays: list[np.ndarray] = []
    for contours in roi.contours_by_z.values():
        arrays.extend(np.asarray(contour[:, :3], dtype=float) for contour in contours if contour.size)
    if not arrays:
        return (np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf)
    points = np.vstack(arrays)
    return (
        float(np.nanmin(points[:, 0])),
        float(np.nanmax(points[:, 0])),
        float(np.nanmin(points[:, 1])),
        float(np.nanmax(points[:, 1])),
        float(np.nanmin(points[:, 2])),
        float(np.nanmax(points[:, 2])),
    )


def _first_target_name(plan: PlanDataset) -> str | None:
    return select_default_target_name(plan) or None


def _plan_isocenter(plan: PlanDataset | None) -> tuple[float, float, float] | None:
    if plan is None:
        return None
    for beam in plan.beams:
        for cp in beam.control_points:
            if cp.isocenter_xyz is not None:
                return cp.isocenter_xyz
    value = plan.plan_info.get("isocenter_xyz")
    if isinstance(value, (tuple, list)) and len(value) == 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None
    return None


def _looks_like_target(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in ("PTV", "CTV", "GTV", "ITV"))


def _looks_like_body(name: str) -> bool:
    upper = name.upper()
    markers = ("BODY", "EXTERNAL", "OUTER", "SKIN", "PATIENT", "KONTUR", "AUSSEN")
    return any(marker in upper for marker in markers)


def _looks_like_helper_roi(name: str) -> bool:
    upper = name.upper()
    markers = (
        "ISO",
        "ISOCENTER",
        "MARKER",
        "SETUP",
        "HELP",
        "CONTROL",
        "COUCH",
        "TABLE",
    )
    return any(marker in upper for marker in markers)


def _context_roi_priority(name: str, lookup: RoiLookup | None) -> tuple[int, str]:
    upper = name.upper()
    if any(marker in upper for marker in ("CORD", "SPINAL", "MYELON", "BRAINSTEM", "KANAL")):
        return (0, name.lower())
    if lookup is not None and lookup.status == "matched":
        return (1, name.lower())
    return (2, name.lower())


def _rgba_color(color: str | None, alpha: float = 0.95) -> tuple[float, float, float, float]:
    qcolor = QColor(color or "#8ab4f8")
    if not qcolor.isValid():
        qcolor = QColor("#8ab4f8")
    return (qcolor.redF(), qcolor.greenF(), qcolor.blueF(), alpha)


def _roi_mask_on_ct_grid(roi, ct: CtVolume) -> np.ndarray:
    depth, rows, columns = ct.voxels.shape
    mask = np.zeros((depth, rows, columns), dtype=bool)
    for z_position, contours in roi.contours_by_z.items():
        z_index = ct.nearest_slice_index(float(z_position))
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            x_pixel, y_pixel = ct.patient_xy_to_pixel(contour)
            rr, cc = draw_polygon(y_pixel, x_pixel, shape=(rows, columns))
            mask[z_index, rr, cc] = True
    return mask


def _cropped_roi_mask_on_ct_grid(
    roi,
    ct: CtVolume,
    padding_voxels: int = 2,
) -> tuple[np.ndarray, int, int, int]:
    depth, rows, columns = ct.voxels.shape
    x_min, x_max, y_min, y_max, z_min, z_max = _roi_bounds(roi)
    if not np.isfinite([x_min, x_max, y_min, y_max, z_min, z_max]).all():
        return np.zeros((0, 0, 0), dtype=bool), 0, 0, 0

    row_spacing, col_spacing = ct.pixel_spacing
    origin_x, origin_y = ct.origin_xy
    pad = max(0, int(padding_voxels))
    col_start = max(0, int(np.floor((x_min - origin_x) / col_spacing)) - pad)
    col_end = min(columns - 1, int(np.ceil((x_max - origin_x) / col_spacing)) + pad)
    row_start = max(0, int(np.floor((y_min - origin_y) / row_spacing)) - pad)
    row_end = min(rows - 1, int(np.ceil((y_max - origin_y) / row_spacing)) + pad)
    z_first = ct.nearest_slice_index(z_min)
    z_last = ct.nearest_slice_index(z_max)
    z_start = max(0, min(z_first, z_last) - pad)
    z_end = min(depth - 1, max(z_first, z_last) + pad)

    if col_end < col_start or row_end < row_start or z_end < z_start:
        return np.zeros((0, 0, 0), dtype=bool), z_start, row_start, col_start

    crop_rows = row_end - row_start + 1
    crop_columns = col_end - col_start + 1
    mask = np.zeros((z_end - z_start + 1, crop_rows, crop_columns), dtype=bool)
    for z_position, contours in roi.contours_by_z.items():
        z_index = ct.nearest_slice_index(float(z_position))
        if z_index < z_start or z_index > z_end:
            continue
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            x_pixel, y_pixel = ct.patient_xy_to_pixel(contour)
            rr, cc = draw_polygon(
                y_pixel - row_start,
                x_pixel - col_start,
                shape=(crop_rows, crop_columns),
            )
            mask[z_index - z_start, rr, cc] = True
    return mask, z_start, row_start, col_start
