from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planeval_viewer.dicom_io.models import BeamGeometry, ControlPointGeometry


def aperture_area(cp: ControlPointGeometry) -> float:
    if not cp.has_mlc:
        return 0.0
    n = cp.leaf_count
    jaw_height = 1.0
    if cp.jaws_y is not None:
        jaw_height = abs(cp.jaws_y[1] - cp.jaws_y[0]) / max(n, 1)
    widths = np.maximum(0.0, np.array(cp.mlc_x2[:n]) - np.array(cp.mlc_x1[:n]))
    return float(np.sum(widths) * jaw_height)


def aperture_perimeter(cp: ControlPointGeometry) -> float:
    if not cp.has_mlc:
        return 0.0
    n = cp.leaf_count
    jaw_height = 1.0
    if cp.jaws_y is not None:
        jaw_height = abs(cp.jaws_y[1] - cp.jaws_y[0]) / max(n, 1)
    widths = np.maximum(0.0, np.array(cp.mlc_x2[:n]) - np.array(cp.mlc_x1[:n]))
    open_count = int(np.sum(widths > 0))
    return float(2.0 * np.sum(widths) + 2.0 * open_count * jaw_height)


def leaf_travel_mm(first: ControlPointGeometry, second: ControlPointGeometry) -> float:
    if not first.has_mlc or not second.has_mlc:
        return 0.0
    n = min(first.leaf_count, second.leaf_count)
    x1 = np.abs(np.array(second.mlc_x1[:n]) - np.array(first.mlc_x1[:n]))
    x2 = np.abs(np.array(second.mlc_x2[:n]) - np.array(first.mlc_x2[:n]))
    return float(np.sum(x1) + np.sum(x2))


@dataclass(frozen=True)
class BeamComplexity:
    beam_name: str
    control_points: int
    mean_area: float
    mean_perimeter: float
    total_leaf_travel: float


def summarize_beam_complexity(beam: BeamGeometry) -> BeamComplexity:
    cps = beam.control_points
    areas = [aperture_area(cp) for cp in cps]
    perimeters = [aperture_perimeter(cp) for cp in cps]
    travel = sum(leaf_travel_mm(a, b) for a, b in zip(cps, cps[1:]))
    return BeamComplexity(
        beam_name=beam.name,
        control_points=len(cps),
        mean_area=float(np.mean(areas)) if areas else 0.0,
        mean_perimeter=float(np.mean(perimeters)) if perimeters else 0.0,
        total_leaf_travel=float(travel),
    )
