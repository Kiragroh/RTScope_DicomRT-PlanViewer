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
    ct_cache: dict[tuple[Path, ...], CtVolume | None] = {}
    plan_paths = by_modality.get("RTPLAN", [])
    if not plan_paths:
        warnings = list(common_warnings)
        rtstruct_paths = by_modality.get("RTSTRUCT", [])
        dose_paths = by_modality.get("RTDOSE", [])
        selected_ct_paths = _select_ct_paths(
            ct_paths,
            datasets_by_path,
            rtstruct_paths=rtstruct_paths,
            dose_paths=dose_paths,
            plan_paths=[],
            anchor_path=None,
            warnings=warnings,
        )
        ct = _load_ct_cached(selected_ct_paths, warnings, ct_cache)
        return [
            _build_plan_dataset(
                ct=ct,
                dose_paths=dose_paths,
                rtstruct_paths=rtstruct_paths,
                plan_paths=[],
                common_warnings=warnings,
                plan_label="Image / structure set",
            )
        ]

    plans: list[PlanDataset] = []
    for plan_path in sorted(plan_paths, key=lambda item: item.name.lower()):
        plan_dataset = datasets_by_path.get(plan_path)
        plan_uid = str(getattr(plan_dataset, "SOPInstanceUID", "") or "")
        dose_paths = _paths_associated_with_plan(
            by_modality.get("RTDOSE", []),
            datasets_by_path,
            plan_uid,
            plan_path,
        )
        rtstruct_paths = _paths_associated_with_plan(
            by_modality.get("RTSTRUCT", []),
            datasets_by_path,
            plan_uid,
            plan_path,
        )
        warnings = list(common_warnings)
        selected_ct_paths = _select_ct_paths(
            ct_paths,
            datasets_by_path,
            rtstruct_paths=rtstruct_paths,
            dose_paths=dose_paths,
            plan_paths=[plan_path],
            anchor_path=plan_path,
            warnings=warnings,
        )
        ct = _load_ct_cached(selected_ct_paths, warnings, ct_cache)
        plan = _build_plan_dataset(
            ct=ct,
            dose_paths=dose_paths,
            rtstruct_paths=rtstruct_paths,
            plan_paths=[plan_path],
            common_warnings=warnings,
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


def _paths_associated_with_plan(
    paths: list[Path],
    datasets_by_path: dict[Path, Dataset],
    sop_uid: str,
    plan_path: Path,
) -> list[Path]:
    if not paths:
        return []
    if sop_uid:
        matched = [
            path
            for path in paths
            if sop_uid in _referenced_sop_uids(datasets_by_path.get(path))
        ]
        if matched:
            return matched
    nearby = _paths_near_anchor(paths, plan_path)
    return nearby or paths


def _paths_near_anchor(paths: list[Path], anchor_path: Path) -> list[Path]:
    anchor_dir = anchor_path.parent
    nearby = [
        path
        for path in paths
        if _is_relative_to(path.parent, anchor_dir) or _is_relative_to(anchor_dir, path.parent)
    ]
    return sorted(nearby, key=lambda item: str(item).lower())


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _select_ct_paths(
    ct_paths: list[Path],
    datasets_by_path: dict[Path, Dataset],
    rtstruct_paths: list[Path],
    dose_paths: list[Path],
    plan_paths: list[Path],
    anchor_path: Path | None,
    warnings: list[str],
) -> list[Path]:
    if not ct_paths:
        return []

    reference_uids: set[str] = set()
    for path in [*rtstruct_paths, *dose_paths, *plan_paths]:
        reference_uids.update(_referenced_sop_uids(datasets_by_path.get(path)))
    if reference_uids:
        matched = [
            path
            for path in ct_paths
            if _dataset_uid(datasets_by_path.get(path), "SOPInstanceUID") in reference_uids
        ]
        if matched:
            return _single_ct_series_from_matches(
                matched,
                ct_paths,
                datasets_by_path,
                warnings,
                "RT object references",
            )

    reference_frames: set[str] = set()
    for path in [*rtstruct_paths, *dose_paths, *plan_paths]:
        reference_frames.update(_frame_of_reference_uids(datasets_by_path.get(path)))
    if reference_frames:
        matched = [
            path
            for path in ct_paths
            if _dataset_uid(datasets_by_path.get(path), "FrameOfReferenceUID") in reference_frames
        ]
        if matched:
            return _single_ct_series_from_matches(
                matched,
                ct_paths,
                datasets_by_path,
                warnings,
                "FrameOfReferenceUID",
            )

    if anchor_path is not None:
        nearby = _paths_near_anchor(ct_paths, anchor_path)
        if nearby:
            return _largest_ct_series(nearby, datasets_by_path, warnings, warn_if_multiple=True)

    return _largest_ct_series(ct_paths, datasets_by_path, warnings, warn_if_multiple=True)


def _single_ct_series_from_matches(
    matched_paths: list[Path],
    all_ct_paths: list[Path],
    datasets_by_path: dict[Path, Dataset],
    warnings: list[str],
    source: str,
) -> list[Path]:
    groups = _ct_series_groups(all_ct_paths, datasets_by_path)
    matched_groups = _ct_series_groups(matched_paths, datasets_by_path)
    if not matched_groups:
        return []
    series_key = max(
        matched_groups,
        key=lambda key: (len(matched_groups[key]), len(groups.get(key, [])), key),
    )
    if len(matched_groups) > 1:
        warnings.append(
            f"Multiple CT series matched {source}; using the series with most referenced slices."
        )
    return sorted(groups.get(series_key, matched_groups[series_key]), key=lambda item: str(item).lower())


def _largest_ct_series(
    ct_paths: list[Path],
    datasets_by_path: dict[Path, Dataset],
    warnings: list[str],
    warn_if_multiple: bool,
) -> list[Path]:
    groups = _ct_series_groups(ct_paths, datasets_by_path)
    if not groups:
        return []
    series_key = max(groups, key=lambda key: (len(groups[key]), key))
    if warn_if_multiple and len(groups) > 1:
        warnings.append("Multiple CT series found; using the largest single CT series.")
    return sorted(groups[series_key], key=lambda item: str(item).lower())


def _ct_series_groups(
    ct_paths: list[Path],
    datasets_by_path: dict[Path, Dataset],
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in ct_paths:
        dataset = datasets_by_path.get(path)
        key = _dataset_uid(dataset, "SeriesInstanceUID") or f"path:{path.parent}"
        groups.setdefault(key, []).append(path)
    return groups


def _dataset_uid(dataset: Dataset | None, keyword: str) -> str:
    if dataset is None:
        return ""
    value = getattr(dataset, keyword, None)
    return str(value) if value else ""


def _frame_of_reference_uids(dataset: Dataset | None) -> set[str]:
    if dataset is None:
        return set()
    uids: set[str] = set()
    value = getattr(dataset, "FrameOfReferenceUID", None)
    if value:
        uids.add(str(value))
    for element in dataset:
        if element.VR != "SQ":
            continue
        for child in element.value:
            if isinstance(child, Dataset):
                uids.update(_frame_of_reference_uids(child))
    return uids


def _load_ct_cached(
    paths: list[Path],
    warnings: list[str],
    cache: dict[tuple[Path, ...], CtVolume | None],
) -> CtVolume | None:
    key = tuple(sorted(paths, key=lambda item: str(item).lower()))
    if key not in cache:
        cache[key] = _load_ct(list(key), warnings)
    return cache[key]


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

    dose_items: list[tuple[str, DoseVolume]] = []
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            dose_items.append((_dose_summation_type(ds), _dose_volume_from_dataset(ds)))
        except Exception as exc:
            warnings.append(f"RTDOSE could not be read: {exc}")

    if not dose_items:
        return None

    complete_doses = [
        volume for summation_type, volume in dose_items if summation_type in {"PLAN", "MULTI_PLAN"}
    ]
    if complete_doses:
        if len(complete_doses) > 1:
            warnings.append("Multiple RTDOSE PLAN files found; using the first plan dose.")
        if len(dose_items) > len(complete_doses):
            warnings.append("Using RTDOSE PLAN; additional beam dose files were not summed.")
        return complete_doses[0]

    beam_doses = [
        volume for summation_type, volume in dose_items if summation_type in {"BEAM", "CONTROL_POINT"}
    ]
    volumes_to_sum = beam_doses or [volume for _summation_type, volume in dose_items]
    if len(volumes_to_sum) == 1:
        return volumes_to_sum[0]

    label = "BEAM" if beam_doses else "compatible"
    return _sum_dose_volumes(volumes_to_sum, label, warnings)


def _dose_volume_from_dataset(ds: Dataset) -> DoseVolume:
    values = ds.pixel_array.astype(np.float32) * float(getattr(ds, "DoseGridScaling", 1.0))
    if values.ndim == 2:
        values = values[np.newaxis, :, :]

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


def _dose_summation_type(ds: Dataset) -> str:
    return str(getattr(ds, "DoseSummationType", "") or "").strip().upper()


def _sum_dose_volumes(
    volumes: list[DoseVolume],
    label: str,
    warnings: list[str],
) -> DoseVolume:
    base = volumes[0]
    summed_values = np.array(base.values_gy, dtype=np.float32, copy=True)
    summed_count = 1
    skipped_count = 0
    for volume in volumes[1:]:
        if not _dose_grids_compatible(base, volume):
            skipped_count += 1
            continue
        summed_values += volume.values_gy.astype(np.float32, copy=False)
        summed_count += 1

    if skipped_count:
        warnings.append(f"Skipped {skipped_count} RTDOSE file(s) with incompatible dose grid.")
    if summed_count > 1:
        warnings.append(f"Summed {summed_count} RTDOSE {label} files on identical dose grid.")
    return DoseVolume(
        values_gy=summed_values,
        z_positions=list(base.z_positions),
        pixel_spacing=base.pixel_spacing,
        origin_xy=base.origin_xy,
        orientation=base.orientation,
    )


def _dose_grids_compatible(first: DoseVolume, second: DoseVolume) -> bool:
    return bool(
        first.values_gy.shape == second.values_gy.shape
        and np.allclose(first.pixel_spacing, second.pixel_spacing, rtol=0, atol=1e-4)
        and np.allclose(first.origin_xy, second.origin_xy, rtol=0, atol=1e-3)
        and np.allclose(first.orientation, second.orientation, rtol=0, atol=1e-4)
        and len(first.z_positions) == len(second.z_positions)
        and np.allclose(first.z_positions, second.z_positions, rtol=0, atol=1e-3)
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
