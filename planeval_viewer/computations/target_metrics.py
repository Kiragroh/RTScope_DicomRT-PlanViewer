from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planeval_viewer.computations.masks import roi_mask_on_dose_grid
from planeval_viewer.dicom_io.models import PlanDataset, RoiGeometry


@dataclass(frozen=True)
class PaddickMetrics:
    target_volume_cc: float
    prescription_isodose_volume_cc: float
    target_covered_volume_cc: float
    half_prescription_isodose_volume_cc: float
    coverage: float
    selectivity: float
    paddick_ci: float
    gradient_index: float
    d2_gy: float
    d98_gy: float
    homogeneity_index: float


def compute_paddick_metrics_from_masks(
    target_mask: np.ndarray,
    dose_gy: np.ndarray,
    prescription_gy: float,
    voxel_volume_cc: float,
) -> PaddickMetrics:
    target = np.asarray(target_mask, dtype=bool)
    dose = np.asarray(dose_gy, dtype=float)
    tv = float(np.sum(target) * voxel_volume_cc)
    piv_mask = dose >= prescription_gy
    half_mask = dose >= prescription_gy * 0.5
    piv = float(np.sum(piv_mask) * voxel_volume_cc)
    tvpiv = float(np.sum(target & piv_mask) * voxel_volume_cc)
    half = float(np.sum(half_mask) * voxel_volume_cc)

    coverage = tvpiv / tv if tv > 0 else 0.0
    selectivity = tvpiv / piv if piv > 0 else 0.0
    paddick_ci = (tvpiv * tvpiv) / (tv * piv) if tv > 0 and piv > 0 else 0.0
    gradient_index = half / piv if piv > 0 else 0.0
    target_doses = dose[target]
    if target_doses.size and prescription_gy > 0:
        d2_gy = float(np.percentile(target_doses, 98))
        d98_gy = float(np.percentile(target_doses, 2))
        homogeneity_index = float(round(1.0 - ((d2_gy - d98_gy) / prescription_gy), 3))
    else:
        d2_gy = 0.0
        d98_gy = 0.0
        homogeneity_index = 0.0
    return PaddickMetrics(
        target_volume_cc=tv,
        prescription_isodose_volume_cc=piv,
        target_covered_volume_cc=tvpiv,
        half_prescription_isodose_volume_cc=half,
        coverage=coverage,
        selectivity=selectivity,
        paddick_ci=paddick_ci,
        gradient_index=gradient_index,
        d2_gy=d2_gy,
        d98_gy=d98_gy,
        homogeneity_index=homogeneity_index,
    )


def compute_paddick_metrics_for_roi(
    plan: PlanDataset,
    roi: RoiGeometry,
    prescription_gy: float,
) -> PaddickMetrics | None:
    if plan.dose is None or not plan.dose.is_axial_aligned:
        return None
    mask = roi_mask_on_dose_grid(plan.dose, roi)
    if not np.any(mask):
        return None
    row_spacing, col_spacing = plan.dose.pixel_spacing
    if len(plan.dose.z_positions) > 1:
        slice_spacing = float(np.median(np.abs(np.diff(sorted(plan.dose.z_positions)))))
    else:
        slice_spacing = 1.0
    voxel_volume_cc = row_spacing * col_spacing * slice_spacing / 1000.0
    return compute_paddick_metrics_from_masks(
        target_mask=mask,
        dose_gy=plan.dose.values_gy,
        prescription_gy=prescription_gy,
        voxel_volume_cc=voxel_volume_cc,
    )
