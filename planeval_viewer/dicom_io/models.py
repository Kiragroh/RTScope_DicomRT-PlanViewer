from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CtVolume:
    voxels: np.ndarray
    z_positions: list[float]
    pixel_spacing: tuple[float, float]
    origin_xy: tuple[float, float]
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @property
    def is_axial_aligned(self) -> bool:
        return bool(
            np.allclose(
                np.array(self.orientation, dtype=float),
                np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=float),
                rtol=0,
                atol=1e-3,
            )
        )

    def nearest_slice_index(self, z_position: float) -> int:
        distances = np.abs(np.array(self.z_positions, dtype=float) - float(z_position))
        return int(np.argmin(distances))

    def patient_xy_to_pixel(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row_spacing, col_spacing = self.pixel_spacing
        origin_x, origin_y = self.origin_xy
        x = (points_xyz[:, 0] - origin_x) / col_spacing
        y = (points_xyz[:, 1] - origin_y) / row_spacing
        return x, y


@dataclass
class DoseVolume:
    values_gy: np.ndarray
    z_positions: list[float]
    pixel_spacing: tuple[float, float]
    origin_xy: tuple[float, float]
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @property
    def is_axial_aligned(self) -> bool:
        return bool(
            np.allclose(
                np.array(self.orientation, dtype=float),
                np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=float),
                rtol=0,
                atol=1e-3,
            )
        )

    def nearest_slice_index(self, z_position: float) -> int:
        distances = np.abs(np.array(self.z_positions, dtype=float) - float(z_position))
        return int(np.argmin(distances))

    def patient_xy_to_pixel(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row_spacing, col_spacing = self.pixel_spacing
        origin_x, origin_y = self.origin_xy
        x = (points_xyz[:, 0] - origin_x) / col_spacing
        y = (points_xyz[:, 1] - origin_y) / row_spacing
        return x, y


@dataclass
class RoiGeometry:
    number: int
    name: str
    color: str
    contours_by_z: dict[float, list[np.ndarray]] = field(default_factory=dict)
    interpreted_type: str = ""
    visible: bool = True


@dataclass(frozen=True)
class MlcLayerGeometry:
    device_type: str
    leaf_boundaries: tuple[float, ...] = ()
    mlc_x1: tuple[float, ...] = ()
    mlc_x2: tuple[float, ...] = ()

    @property
    def leaf_count(self) -> int:
        return min(len(self.mlc_x1), len(self.mlc_x2))


@dataclass
class ControlPointGeometry:
    index: int
    meterset_weight: float = 0.0
    gantry_angle: float | None = None
    collimator_angle: float | None = None
    couch_angle: float | None = None
    jaws_x: tuple[float, float] | None = None
    jaws_y: tuple[float, float] | None = None
    isocenter_xyz: tuple[float, float, float] | None = None
    mlc_x1: tuple[float, ...] = ()
    mlc_x2: tuple[float, ...] = ()
    mlc_layers: tuple[MlcLayerGeometry, ...] = ()

    @property
    def has_mlc(self) -> bool:
        return self.leaf_count > 0 or any(layer.leaf_count > 0 for layer in self.mlc_layers)

    @property
    def leaf_count(self) -> int:
        return min(len(self.mlc_x1), len(self.mlc_x2))


@dataclass
class BeamGeometry:
    number: int
    name: str
    treatment_machine: str = ""
    radiation_type: str = ""
    meterset: float | None = None
    leaf_boundaries: tuple[float, ...] = ()
    mlc_leaf_boundaries: dict[str, tuple[float, ...]] = field(default_factory=dict)
    control_points: list[ControlPointGeometry] = field(default_factory=list)

    @property
    def control_point_count(self) -> int:
        return len(self.control_points)


@dataclass
class PlanDataset:
    ct: CtVolume | None
    rois: list[RoiGeometry]
    dose: DoseVolume | None
    plan_info: dict[str, Any]
    beams: list[BeamGeometry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def roi_names(self) -> list[str]:
        return [roi.name for roi in self.rois]

    def roi_by_name(self, name: str) -> RoiGeometry | None:
        for roi in self.rois:
            if roi.name == name:
                return roi
        return None
