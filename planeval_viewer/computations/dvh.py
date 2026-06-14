from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from planeval_viewer.computations.masks import roi_mask_on_dose_grid
from planeval_viewer.dicom_io.models import PlanDataset, RoiGeometry


@dataclass(frozen=True)
class DvhCurve:
    roi_name: str
    dose_bins_gy: np.ndarray
    volume_pct: np.ndarray
    volume_cc_curve: np.ndarray
    volume_cc: float
    dmin: float | None
    dmax: float | None
    dmean: float | None

    @classmethod
    def empty(cls, roi_name: str) -> "DvhCurve":
        return cls(
            roi_name=roi_name,
            dose_bins_gy=np.array([], dtype=float),
            volume_pct=np.array([], dtype=float),
            volume_cc_curve=np.array([], dtype=float),
            volume_cc=0.0,
            dmin=None,
            dmax=None,
            dmean=None,
        )

    def dose_at_volume_pct(self, volume_pct: float) -> float | None:
        if self.volume_pct.size == 0:
            return None
        idx = np.searchsorted(-self.volume_pct, -float(volume_pct), side="left")
        if idx >= self.dose_bins_gy.size:
            return None
        return float(self.dose_bins_gy[idx])

    def dose_at_volume_cc(self, volume_cc: float) -> float | None:
        if self.volume_cc <= 0:
            return None
        return self.dose_at_volume_pct(float(volume_cc) / self.volume_cc * 100.0)

    def volume_pct_at_dose(self, dose_gy: float) -> float:
        if self.dose_bins_gy.size == 0:
            return 0.0
        idx = np.searchsorted(self.dose_bins_gy, float(dose_gy), side="left")
        if idx >= self.volume_pct.size:
            return 0.0
        return float(self.volume_pct[idx])

    def volume_cc_at_dose(self, dose_gy: float) -> float:
        return self.volume_pct_at_dose(dose_gy) / 100.0 * self.volume_cc

    @property
    def stats(self) -> dict[str, float | None]:
        return {
            "Volume cc": self.volume_cc,
            "Dmin": self.dmin,
            "Dmax": self.dmax,
            "Dmean": self.dmean,
            "D2": self.dose_at_volume_pct(2),
            "D50": self.dose_at_volume_pct(50),
            "D95": self.dose_at_volume_pct(95),
            "D98": self.dose_at_volume_pct(98),
        }


def compute_dvh_from_samples(
    roi_name: str,
    dose_values_gy: Iterable[float] | np.ndarray,
    voxel_volume_cc: float,
    bin_width_gy: float = 0.1,
) -> DvhCurve:
    values = np.asarray(list(dose_values_gy), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return DvhCurve.empty(roi_name)

    values.sort()
    dmax = float(values[-1])
    dmin = float(values[0])
    dmean = float(np.mean(values))
    volume_cc = float(values.size * voxel_volume_cc)

    dose_bins, counts = np.unique(values, return_counts=True)
    cumulative = np.cumsum(counts[::-1])[::-1]
    volume_pct = cumulative / values.size * 100.0
    volume_cc_curve = volume_pct / 100.0 * volume_cc

    return DvhCurve(
        roi_name=roi_name,
        dose_bins_gy=dose_bins,
        volume_pct=volume_pct,
        volume_cc_curve=volume_cc_curve,
        volume_cc=volume_cc,
        dmin=dmin,
        dmax=dmax,
        dmean=dmean,
    )


def compute_dvh_for_roi(
    plan: PlanDataset,
    roi: RoiGeometry,
    bin_width_gy: float = 0.1,
) -> DvhCurve:
    if plan.ct is None or plan.dose is None:
        return DvhCurve.empty(roi.name)
    if not plan.dose.is_axial_aligned:
        return DvhCurve.empty(roi.name)
    return _compute_dvh_for_roi_on_dose_grid(plan, roi, bin_width_gy)


def _compute_dvh_for_roi_on_dose_grid(
    plan: PlanDataset,
    roi: RoiGeometry,
    bin_width_gy: float,
) -> DvhCurve:
    if plan.dose is None:
        return DvhCurve.empty(roi.name)
    mask = roi_mask_on_dose_grid(plan.dose, roi)
    if not np.any(mask):
        return DvhCurve.empty(roi.name)
    row_spacing, col_spacing = plan.dose.pixel_spacing
    if len(plan.dose.z_positions) > 1:
        slice_spacing = float(np.median(np.abs(np.diff(sorted(plan.dose.z_positions)))))
    else:
        slice_spacing = 1.0
    voxel_volume_cc = row_spacing * col_spacing * slice_spacing / 1000.0
    return compute_dvh_from_samples(roi.name, plan.dose.values_gy[mask], voxel_volume_cc, bin_width_gy)


def compute_all_dvhs(plan: PlanDataset, bin_width_gy: float = 0.1) -> dict[str, DvhCurve]:
    if plan.dose is None or not plan.dose.is_axial_aligned:
        return {roi.name: DvhCurve.empty(roi.name) for roi in plan.rois}
    return {
        roi.name: _compute_dvh_for_roi_on_dose_grid(plan, roi, bin_width_gy)
        for roi in plan.rois
    }
