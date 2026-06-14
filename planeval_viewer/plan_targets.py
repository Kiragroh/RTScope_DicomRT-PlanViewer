from __future__ import annotations

import re
from typing import Any

from planeval_viewer.dicom_io.models import PlanDataset, RoiGeometry


TARGET_PRIORITY = ("PTV", "GTV", "ITV", "CTV")


def select_default_target_name(plan: PlanDataset | None) -> str:
    if plan is None or not plan.rois:
        return ""

    explicit = _target_from_plan_info(plan)
    if explicit:
        return explicit

    for marker in TARGET_PRIORITY:
        for roi in plan.rois:
            if _roi_is_target_type(roi, marker):
                return roi.name
        for roi in plan.rois:
            if marker in _normalized(roi.name).upper():
                return roi.name

    return plan.rois[0].name


def is_target_roi(roi: RoiGeometry) -> bool:
    normalized_type = _normalized(roi.interpreted_type).upper()
    normalized_name = _normalized(roi.name).upper()
    return normalized_type in TARGET_PRIORITY or any(
        marker in normalized_name for marker in TARGET_PRIORITY
    )


def _target_from_plan_info(plan: PlanDataset) -> str:
    by_name = {_normalized(roi.name): roi.name for roi in plan.rois}
    by_number = {roi.number: roi.name for roi in plan.rois}

    direct_name = str(plan.plan_info.get("target_roi_name") or "")
    if _normalized(direct_name) in by_name:
        return by_name[_normalized(direct_name)]

    direct_number = _safe_int(plan.plan_info.get("target_roi_number"))
    if direct_number in by_number:
        return by_number[direct_number]

    for item in plan.plan_info.get("target_prescriptions") or ():
        if not isinstance(item, dict):
            continue
        number = _safe_int(item.get("referenced_roi_number"))
        if number in by_number:
            return by_number[number]
        for key in ("roi_name", "description"):
            candidate = str(item.get(key) or "")
            if _normalized(candidate) in by_name:
                return by_name[_normalized(candidate)]
    return ""


def _roi_is_target_type(roi: RoiGeometry, marker: str) -> bool:
    interpreted = _normalized(roi.interpreted_type).upper()
    return interpreted == marker


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
