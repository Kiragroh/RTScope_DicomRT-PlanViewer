from __future__ import annotations

import math

import numpy as np

from planeval_viewer.dicom_io.models import BeamGeometry, ControlPointGeometry, RoiGeometry


def project_roi_to_bev(
    roi: RoiGeometry,
    cp: ControlPointGeometry,
) -> list[tuple[np.ndarray, np.ndarray]]:
    projected: list[tuple[np.ndarray, np.ndarray]] = []
    for contours in roi.contours_by_z.values():
        for contour in contours:
            if contour.shape[0] < 2:
                continue
            projected.append(project_points_to_bev(contour, cp))
    return projected


def project_roi_outline_to_bev(
    roi: RoiGeometry,
    cp: ControlPointGeometry,
) -> tuple[np.ndarray, np.ndarray] | None:
    points = _projected_target_points(roi, cp)
    if points.shape[0] < 3:
        return None
    hull = _convex_hull_2d(points)
    if hull.shape[0] < 3:
        return None
    closed = np.vstack((hull, hull[0]))
    return closed[:, 0], closed[:, 1]


def project_points_to_bev(
    points_xyz: np.ndarray,
    cp: ControlPointGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=float)
    isocenter = np.asarray(cp.isocenter_xyz or (0.0, 0.0, 0.0), dtype=float)
    relative = points - isocenter

    gantry = math.radians(cp.gantry_angle or 0.0)
    collimator = math.radians(cp.collimator_angle or 0.0)

    x_bev = math.cos(gantry) * relative[:, 0] + math.sin(gantry) * relative[:, 1]
    y_bev = relative[:, 2]

    x_coll = math.cos(collimator) * x_bev - math.sin(collimator) * y_bev
    y_coll = math.sin(collimator) * x_bev + math.cos(collimator) * y_bev
    return x_coll, y_coll


def plan_aperture_modulation_score(
    beam: BeamGeometry,
    target_roi: RoiGeometry | None,
) -> float | None:
    if target_roi is None:
        return None
    cps = [cp for cp in beam.control_points if cp.has_mlc]
    if not cps:
        return None

    weights = _control_point_weights(cps)
    weighted_outside = 0.0
    weighted_total = 0.0
    for cp, weight in zip(cps, weights):
        points = _projected_target_points(target_roi, cp)
        if points.size == 0:
            continue
        inside = _points_inside_aperture(points, beam, cp)
        outside_fraction = 1.0 - float(np.mean(inside))
        weighted_outside += outside_fraction * weight
        weighted_total += weight
    if weighted_total <= 0:
        return None
    return weighted_outside / weighted_total


def _projected_target_points(roi: RoiGeometry, cp: ControlPointGeometry) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for x, y in project_roi_to_bev(roi, cp):
        arrays.append(np.column_stack((x, y)))
    if not arrays:
        return np.empty((0, 2), dtype=float)
    return np.vstack(arrays)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(np.asarray(points, dtype=float), decimals=6), axis=0)
    if unique.shape[0] < 3:
        return np.empty((0, 2), dtype=float)
    ordered = sorted((float(point[0]), float(point[1])) for point in unique)

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return np.empty((0, 2), dtype=float)
    return np.array(hull, dtype=float)


def _points_inside_aperture(
    points: np.ndarray,
    beam: BeamGeometry,
    cp: ControlPointGeometry,
) -> np.ndarray:
    inside = np.zeros(points.shape[0], dtype=bool)
    if not cp.has_mlc:
        return inside
    leaf_edges = _leaf_edges(beam, cp)
    for index, (x, y) in enumerate(points):
        leaf = int(np.searchsorted(leaf_edges, y, side="right") - 1)
        if leaf < 0 or leaf >= cp.leaf_count:
            continue
        inside[index] = cp.mlc_x1[leaf] <= x <= cp.mlc_x2[leaf]
    return inside


def _leaf_edges(beam: BeamGeometry, cp: ControlPointGeometry) -> np.ndarray:
    if len(beam.leaf_boundaries) >= cp.leaf_count + 1:
        return np.array(beam.leaf_boundaries[: cp.leaf_count + 1], dtype=float)
    if cp.jaws_y is not None:
        return np.linspace(cp.jaws_y[0], cp.jaws_y[1], cp.leaf_count + 1)
    return np.arange(cp.leaf_count + 1, dtype=float)


def _control_point_weights(cps: list[ControlPointGeometry]) -> list[float]:
    weights: list[float] = []
    previous = cps[0].meterset_weight
    for cp in cps:
        delta = max(0.0, cp.meterset_weight - previous)
        weights.append(delta)
        previous = cp.meterset_weight
    if sum(weights) <= 0:
        return [1.0 / len(cps)] * len(cps)
    return weights
