from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from planeval_viewer.dicom_io.models import (
    BeamGeometry,
    ControlPointGeometry,
    CtVolume,
    DoseVolume,
    MlcLayerGeometry,
    PlanDataset,
    RoiGeometry,
)


def load_plan_folder(folder: str | Path) -> PlanDataset:
    plans = load_plan_variants(folder)
    if plans:
        return plans[0]
    return PlanDataset(
        ct=None,
        rois=[],
        dose=None,
        plan_info={"plan_label": "No DICOM plan"},
        warnings=["No readable DICOM objects found."],
    )


def load_plan_variants(folder: str | Path) -> list[PlanDataset]:
    folder_path = Path(folder)
    datasets = _read_dicom_headers(folder_path)
    by_modality = _group_by_modality(datasets)
    datasets_by_path = {path: dataset for path, dataset in datasets}

    common_warnings: list[str] = []
    ct_paths = by_modality.get("CT", [])
    ct = _load_ct(ct_paths, common_warnings)
    plan_paths = by_modality.get("RTPLAN", [])
    if not plan_paths:
        return [
            _build_plan_dataset(
                ct=ct,
                dose_paths=by_modality.get("RTDOSE", []),
                rtstruct_paths=by_modality.get("RTSTRUCT", []),
                plan_paths=[],
                common_warnings=common_warnings,
                plan_label="Image / structure set",
            )
        ]

    plans: list[PlanDataset] = []
    for plan_path in sorted(plan_paths, key=lambda item: item.name.lower()):
        plan_dataset = datasets_by_path.get(plan_path)
        plan_uid = str(getattr(plan_dataset, "SOPInstanceUID", "") or "")
        dose_paths = _paths_referencing_sop(
            by_modality.get("RTDOSE", []),
            datasets_by_path,
            plan_uid,
        )
        rtstruct_paths = _paths_referencing_sop(
            by_modality.get("RTSTRUCT", []),
            datasets_by_path,
            plan_uid,
        )
        plan = _build_plan_dataset(
            ct=ct,
            dose_paths=dose_paths,
            rtstruct_paths=rtstruct_paths,
            plan_paths=[plan_path],
            common_warnings=common_warnings,
            plan_label=plan_path.stem,
        )
        plan.plan_info.setdefault("source_plan_path", str(plan_path))
        plans.append(plan)
    return plans


def _build_plan_dataset(
    ct: CtVolume | None,
    dose_paths: list[Path],
    rtstruct_paths: list[Path],
    plan_paths: list[Path],
    common_warnings: list[str],
    plan_label: str,
) -> PlanDataset:
    warnings = list(common_warnings)
    dose = _load_dose(dose_paths, warnings)
    rois = _load_rtstruct(rtstruct_paths, warnings)
    plan_info = _load_plan_info(plan_paths, warnings)
    beams = _load_beams(plan_paths, warnings)
    plan_info.setdefault("plan_label", plan_label)
    if ct is None:
        warnings.append("No CT series found. Image display will be limited.")
    elif not ct.is_axial_aligned:
        warnings.append("CT orientation is not simple axial HFS. Contour overlay may be disabled.")
    if dose is None:
        warnings.append("No RTDOSE found. Dose overlay is unavailable.")
    if not rois:
        warnings.append("No RTSTRUCT contours found.")

    return PlanDataset(
        ct=ct,
        rois=rois,
        dose=dose,
        plan_info=plan_info,
        beams=beams,
        warnings=warnings,
    )


def _paths_referencing_sop(
    paths: list[Path],
    datasets_by_path: dict[Path, Dataset],
    sop_uid: str,
) -> list[Path]:
    if not sop_uid:
        return paths
    matched = [
        path
        for path in paths
        if sop_uid in _referenced_sop_uids(datasets_by_path.get(path))
    ]
    return matched or paths


def _referenced_sop_uids(dataset: Dataset | None) -> set[str]:
    if dataset is None:
        return set()
    uids: set[str] = set()
    value = getattr(dataset, "ReferencedSOPInstanceUID", None)
    if value:
        uids.add(str(value))
    for element in dataset:
        if element.VR != "SQ":
            continue
        for child in element.value:
            if isinstance(child, Dataset):
                uids.update(_referenced_sop_uids(child))
    return uids


def _read_dicom_headers(folder: Path) -> list[tuple[Path, Dataset]]:
    items: list[tuple[Path, Dataset]] = []
    for path in _iter_files(folder):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(dataset, "SOPClassUID", None) or getattr(dataset, "Modality", None):
            items.append((path, dataset))
    return items


def _iter_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file():
            yield path


def _group_by_modality(items: list[tuple[Path, Dataset]]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path, dataset in items:
        modality = str(getattr(dataset, "Modality", "")).upper()
        if modality:
            grouped.setdefault(modality, []).append(path)
    return grouped


def _load_ct(paths: list[Path], warnings: list[str]) -> CtVolume | None:
    if not paths:
        return None

    slices: list[Dataset] = []
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception as exc:
            warnings.append(f"Skipped unreadable CT slice {path.name}: {exc}")

    if not slices:
        return None

    slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    voxels = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        voxels.append(arr * slope + intercept)

    first = slices[0]
    pixel_spacing = (
        float(first.PixelSpacing[0]),
        float(first.PixelSpacing[1]),
    )
    origin_xy = (
        float(first.ImagePositionPatient[0]),
        float(first.ImagePositionPatient[1]),
    )
    orientation = tuple(float(x) for x in getattr(first, "ImageOrientationPatient", (1, 0, 0, 0, 1, 0)))
    z_positions = [float(ds.ImagePositionPatient[2]) for ds in slices]
    return CtVolume(
        voxels=np.stack(voxels, axis=0),
        z_positions=z_positions,
        pixel_spacing=pixel_spacing,
        origin_xy=origin_xy,
        orientation=orientation,
    )


def _load_dose(paths: list[Path], warnings: list[str]) -> DoseVolume | None:
    if not paths:
        return None

    try:
        ds = pydicom.dcmread(str(paths[0]), force=True)
        values = ds.pixel_array.astype(np.float32) * float(getattr(ds, "DoseGridScaling", 1.0))
        if values.ndim == 2:
            values = values[np.newaxis, :, :]
    except Exception as exc:
        warnings.append(f"RTDOSE could not be read: {exc}")
        return None

    position = getattr(ds, "ImagePositionPatient", (0.0, 0.0, 0.0))
    offsets = list(getattr(ds, "GridFrameOffsetVector", []))
    if offsets:
        if float(offsets[0]) == 0.0:
            z_positions = [float(position[2]) + float(offset) for offset in offsets]
        else:
            z_positions = [float(offset) for offset in offsets]
    else:
        z_positions = [float(position[2]) + i for i in range(values.shape[0])]

    pixel_spacing = (
        float(ds.PixelSpacing[0]),
        float(ds.PixelSpacing[1]),
    )
    origin_xy = (float(position[0]), float(position[1]))
    orientation = tuple(float(x) for x in getattr(ds, "ImageOrientationPatient", (1, 0, 0, 0, 1, 0)))
    return DoseVolume(
        values_gy=values,
        z_positions=z_positions,
        pixel_spacing=pixel_spacing,
        origin_xy=origin_xy,
        orientation=orientation,
    )


def _load_rtstruct(paths: list[Path], warnings: list[str]) -> list[RoiGeometry]:
    if not paths:
        return []

    try:
        ds = pydicom.dcmread(str(paths[0]), force=True)
    except Exception as exc:
        warnings.append(f"RTSTRUCT could not be read: {exc}")
        return []

    names: dict[int, tuple[str, str]] = {}
    for item in getattr(ds, "StructureSetROISequence", []) or []:
        number = int(getattr(item, "ROINumber", -1))
        names[number] = (
            str(getattr(item, "ROIName", f"ROI {number}")),
            str(getattr(item, "RTROIInterpretedType", "")),
        )

    rois: list[RoiGeometry] = []
    for contour_item in getattr(ds, "ROIContourSequence", []) or []:
        number = int(getattr(contour_item, "ReferencedROINumber", -1))
        name, interpreted_type = names.get(number, (f"ROI {number}", ""))
        color = _dicom_color_to_hex(getattr(contour_item, "ROIDisplayColor", None))
        contours_by_z: dict[float, list[np.ndarray]] = {}

        for contour in getattr(contour_item, "ContourSequence", []) or []:
            raw = getattr(contour, "ContourData", None)
            if not raw:
                continue
            points = np.array(raw, dtype=np.float32).reshape(-1, 3)
            z_key = round(float(points[0, 2]), 2)
            contours_by_z.setdefault(z_key, []).append(points)

        rois.append(
            RoiGeometry(
                number=number,
                name=name,
                color=color,
                contours_by_z=contours_by_z,
                interpreted_type=interpreted_type,
            )
        )

    rois.sort(key=lambda item: item.name.lower())
    return rois


def _load_plan_info(paths: list[Path], warnings: list[str]) -> dict[str, Any]:
    if not paths:
        return {}

    try:
        ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
    except Exception as exc:
        warnings.append(f"RTPLAN could not be read: {exc}")
        return {}

    info: dict[str, Any] = {
        "patient_name": str(getattr(ds, "PatientName", "")),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "plan_label": str(getattr(ds, "RTPlanLabel", "")),
        "plan_name": str(getattr(ds, "RTPlanName", "")),
    }

    fraction_groups = getattr(ds, "FractionGroupSequence", []) or []
    if fraction_groups:
        info["number_of_fractions"] = int(
            getattr(fraction_groups[0], "NumberOfFractionsPlanned", 0) or 0
        )

    prescription = _first_prescription_dose(ds)
    if prescription is not None:
        info["prescription_dose_gy"] = prescription
        fractions = info.get("number_of_fractions")
        if fractions:
            info["dose_per_fraction_gy"] = prescription / fractions

    return info


def _load_beams(paths: list[Path], warnings: list[str]) -> list[BeamGeometry]:
    if not paths:
        return []
    try:
        ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
    except Exception as exc:
        warnings.append(f"RTPLAN beams could not be read: {exc}")
        return []

    metersets = _beam_metersets(ds)
    beams: list[BeamGeometry] = []
    for beam_ds in getattr(ds, "BeamSequence", []) or []:
        number = int(getattr(beam_ds, "BeamNumber", len(beams) + 1))
        mlc_leaf_boundaries = _mlc_leaf_boundaries(beam_ds)
        leaf_boundaries = _primary_leaf_boundaries(mlc_leaf_boundaries)
        control_points = _control_points_from_beam(beam_ds, mlc_leaf_boundaries)
        beams.append(
            BeamGeometry(
                number=number,
                name=str(getattr(beam_ds, "BeamName", f"Beam {number}")),
                treatment_machine=str(getattr(beam_ds, "TreatmentMachineName", "")),
                radiation_type=str(getattr(beam_ds, "RadiationType", "")),
                meterset=metersets.get(number),
                leaf_boundaries=leaf_boundaries,
                mlc_leaf_boundaries=mlc_leaf_boundaries,
                control_points=control_points,
            )
        )
    return beams


def _beam_metersets(ds: Dataset) -> dict[int, float]:
    metersets: dict[int, float] = {}
    for fg in getattr(ds, "FractionGroupSequence", []) or []:
        for ref in getattr(fg, "ReferencedBeamSequence", []) or []:
            number = int(getattr(ref, "ReferencedBeamNumber", -1))
            meterset = getattr(ref, "BeamMeterset", None)
            if number >= 0 and meterset is not None:
                try:
                    metersets[number] = float(meterset)
                except (TypeError, ValueError):
                    pass
    return metersets


def _mlc_leaf_boundaries(beam_ds: Dataset) -> dict[str, tuple[float, ...]]:
    boundaries_by_device: dict[str, tuple[float, ...]] = {}
    for device in getattr(beam_ds, "BeamLimitingDeviceSequence", []) or []:
        device_type = str(getattr(device, "RTBeamLimitingDeviceType", "")).upper()
        if device_type.startswith("MLC"):
            boundaries = getattr(device, "LeafPositionBoundaries", None)
            if boundaries is not None:
                boundaries_by_device[device_type] = tuple(float(item) for item in boundaries)
    return boundaries_by_device


def _primary_leaf_boundaries(
    boundaries_by_device: dict[str, tuple[float, ...]]
) -> tuple[float, ...]:
    if not boundaries_by_device:
        return ()
    preferred_key = sorted(boundaries_by_device)[0]
    return boundaries_by_device[preferred_key]


def _control_point_from_dataset(
    cp_ds: Dataset,
    mlc_leaf_boundaries: dict[str, tuple[float, ...]] | None = None,
) -> ControlPointGeometry:
    jaws_x: tuple[float, float] | None = None
    jaws_y: tuple[float, float] | None = None
    mlc_x1: tuple[float, ...] = ()
    mlc_x2: tuple[float, ...] = ()
    mlc_layers: list[MlcLayerGeometry] = []
    mlc_leaf_boundaries = mlc_leaf_boundaries or {}

    for device in getattr(cp_ds, "BeamLimitingDevicePositionSequence", []) or []:
        device_type = str(getattr(device, "RTBeamLimitingDeviceType", "")).upper()
        positions = tuple(float(item) for item in getattr(device, "LeafJawPositions", []) or [])
        if device_type in {"X", "ASYMX"} and len(positions) >= 2:
            jaws_x = (positions[0], positions[1])
        elif device_type in {"Y", "ASYMY"} and len(positions) >= 2:
            jaws_y = (positions[0], positions[1])
        elif device_type.startswith("MLC"):
            layer_x1, layer_x2 = _split_mlc_leaf_banks(positions)
            layer = MlcLayerGeometry(
                device_type=device_type,
                leaf_boundaries=mlc_leaf_boundaries.get(device_type, ()),
                mlc_x1=layer_x1,
                mlc_x2=layer_x2,
            )
            mlc_layers.append(layer)
            if not mlc_x1 and not mlc_x2:
                mlc_x1, mlc_x2 = layer_x1, layer_x2

    return ControlPointGeometry(
        index=int(getattr(cp_ds, "ControlPointIndex", 0) or 0),
        meterset_weight=float(getattr(cp_ds, "CumulativeMetersetWeight", 0.0) or 0.0),
        gantry_angle=_optional_float(getattr(cp_ds, "GantryAngle", None)),
        collimator_angle=_optional_float(getattr(cp_ds, "BeamLimitingDeviceAngle", None)),
        couch_angle=_optional_float(getattr(cp_ds, "PatientSupportAngle", None)),
        jaws_x=jaws_x,
        jaws_y=jaws_y,
        isocenter_xyz=_optional_triplet(getattr(cp_ds, "IsocenterPosition", None)),
        mlc_x1=mlc_x1,
        mlc_x2=mlc_x2,
        mlc_layers=tuple(mlc_layers),
    )


def _control_points_from_beam(
    beam_ds: Dataset,
    mlc_leaf_boundaries: dict[str, tuple[float, ...]] | None = None,
) -> list[ControlPointGeometry]:
    control_points: list[ControlPointGeometry] = []
    previous: ControlPointGeometry | None = None
    for cp_ds in getattr(beam_ds, "ControlPointSequence", []) or []:
        cp = _control_point_from_dataset(cp_ds, mlc_leaf_boundaries)
        if previous is not None:
            cp = ControlPointGeometry(
                index=cp.index,
                meterset_weight=cp.meterset_weight,
                gantry_angle=cp.gantry_angle if cp.gantry_angle is not None else previous.gantry_angle,
                collimator_angle=cp.collimator_angle if cp.collimator_angle is not None else previous.collimator_angle,
                couch_angle=cp.couch_angle if cp.couch_angle is not None else previous.couch_angle,
                jaws_x=cp.jaws_x if cp.jaws_x is not None else previous.jaws_x,
                jaws_y=cp.jaws_y if cp.jaws_y is not None else previous.jaws_y,
                isocenter_xyz=cp.isocenter_xyz if cp.isocenter_xyz is not None else previous.isocenter_xyz,
                mlc_x1=cp.mlc_x1 if cp.mlc_x1 else previous.mlc_x1,
                mlc_x2=cp.mlc_x2 if cp.mlc_x2 else previous.mlc_x2,
                mlc_layers=cp.mlc_layers if cp.mlc_layers else previous.mlc_layers,
            )
        control_points.append(cp)
        previous = cp
    return control_points


def _split_mlc_leaf_banks(positions: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(positions) < 2:
        return (), ()
    split = len(positions) // 2
    return tuple(positions[:split]), tuple(positions[split:])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_triplet(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        items = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError, IndexError):
        return None
    return items if len(items) == 3 else None


def _first_prescription_dose(ds: Dataset) -> float | None:
    for item in getattr(ds, "DoseReferenceSequence", []) or []:
        value = getattr(item, "TargetPrescriptionDose", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _dicom_color_to_hex(value: Any) -> str:
    if value is None:
        return "#6EA8FE"
    try:
        r, g, b = [max(0, min(255, int(v))) for v in value[:3]]
    except Exception:
        return "#6EA8FE"
    return f"#{r:02X}{g:02X}{b:02X}"
