from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from planeval_viewer.dicom_io.models import CtVolume, DoseVolume, RoiGeometry


def roi_mask_on_ct_grid(ct: CtVolume, roi: RoiGeometry) -> np.ndarray:
    return _roi_mask_on_grid(
        shape=ct.voxels.shape,
        z_positions=ct.z_positions,
        is_axial_aligned=ct.is_axial_aligned,
        xy_to_pixel=ct.patient_xy_to_pixel,
        roi=roi,
    )


def roi_mask_on_dose_grid(dose: DoseVolume, roi: RoiGeometry) -> np.ndarray:
    return _roi_mask_on_grid(
        shape=dose.values_gy.shape,
        z_positions=dose.z_positions,
        is_axial_aligned=dose.is_axial_aligned,
        xy_to_pixel=dose.patient_xy_to_pixel,
        roi=roi,
    )


def _roi_mask_on_grid(
    shape: tuple[int, int, int],
    z_positions: list[float],
    is_axial_aligned: bool,
    xy_to_pixel,
    roi: RoiGeometry,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not is_axial_aligned:
        return mask

    tolerance = _slice_tolerance(z_positions)
    for z_key, contours in roi.contours_by_z.items():
        slice_index = int(np.argmin(np.abs(np.array(z_positions, dtype=float) - z_key)))
        if abs(z_positions[slice_index] - z_key) > tolerance:
            continue
        for contour in contours:
            x, y = xy_to_pixel(contour)
            _fill_polygon(mask[slice_index], x, y)
    return mask


def _fill_polygon(slice_mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    if x.size < 3 or y.size < 3:
        return
    rows, cols = slice_mask.shape
    xmin = max(0, int(np.floor(np.min(x))) - 1)
    xmax = min(cols - 1, int(np.ceil(np.max(x))) + 1)
    ymin = max(0, int(np.floor(np.min(y))) - 1)
    ymax = min(rows - 1, int(np.ceil(np.max(y))) + 1)
    if xmax < xmin or ymax < ymin:
        return

    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    polygon = MplPath(np.column_stack((x, y)))
    inside = polygon.contains_points(points, radius=0.5).reshape(yy.shape)
    slice_mask[ymin : ymax + 1, xmin : xmax + 1] |= inside


def _slice_tolerance(z_positions: list[float]) -> float:
    if len(z_positions) < 2:
        return 0.75
    diffs = np.abs(np.diff(sorted(z_positions)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.75
    return max(0.75, float(np.median(diffs)) / 2.0)
