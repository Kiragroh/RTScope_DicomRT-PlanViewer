from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from planeval_viewer.dicom_io.models import CtVolume, DoseVolume


def resample_dose_to_ct_grid(ct: CtVolume, dose: DoseVolume) -> np.ndarray | None:
    if not ct.is_axial_aligned:
        return None
    if not np.allclose(dose.orientation, ct.orientation, rtol=0, atol=1e-3):
        return None

    output = np.zeros_like(ct.voxels, dtype=np.float32)
    for ct_index, z_position in enumerate(ct.z_positions):
        dose_index = dose.nearest_slice_index(z_position)
        dose_slice = dose.values_gy[dose_index]
        if dose_slice.shape == ct.voxels[ct_index].shape:
            output[ct_index] = dose_slice
        else:
            output[ct_index] = _resample_dose_slice(ct, dose, dose_slice, ct_index)
    return output


def _resample_dose_slice(
    ct: CtVolume,
    dose: DoseVolume,
    dose_slice: np.ndarray,
    ct_index: int,
) -> np.ndarray:
    dose_rows, dose_cols = dose_slice.shape
    dose_row_spacing, dose_col_spacing = dose.pixel_spacing
    dose_origin_x, dose_origin_y = dose.origin_xy
    dose_y = dose_origin_y + np.arange(dose_rows, dtype=float) * dose_row_spacing
    dose_x = dose_origin_x + np.arange(dose_cols, dtype=float) * dose_col_spacing

    ct_rows, ct_cols = ct.voxels[ct_index].shape
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
    return interpolator(points).reshape((ct_rows, ct_cols)).astype(np.float32)
