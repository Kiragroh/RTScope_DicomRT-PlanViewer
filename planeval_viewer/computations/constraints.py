from __future__ import annotations

import re
from dataclasses import dataclass

from planeval_viewer.computations.dvh import DvhCurve
from planeval_viewer.refdb.models import ConstraintRow


@dataclass(frozen=True)
class ConstraintEvaluation:
    roi_name: str
    metric: str
    unit: str
    value: float | None
    limit_optimal: float | None
    limit_maximal: float | None
    status: str
    source: str = ""
    priority: str = ""


def compute_constraint_metric(dvh: DvhCurve, row: ConstraintRow) -> float | None:
    metric = row.metric.strip()
    unit = row.unit.strip().lower()
    lower = metric.lower()
    if lower == "dmax":
        return dvh.dmax
    if lower == "dmin":
        return dvh.dmin
    if lower == "dmean":
        return dvh.dmean

    dcc = re.match(r"^d(\d+(?:\.\d+)?)cc$", lower)
    if dcc:
        return dvh.dose_at_volume_cc(float(dcc.group(1)))

    dpct = re.match(r"^d(\d+(?:\.\d+)?)%?$", lower)
    if dpct:
        return dvh.dose_at_volume_pct(float(dpct.group(1)))

    vgy = re.match(r"^v(\d+(?:\.\d+)?)gy$", lower)
    if vgy:
        dose = float(vgy.group(1))
        if unit in {"cc", "cm3", "ml"}:
            return dvh.volume_cc_at_dose(dose)
        return dvh.volume_pct_at_dose(dose)

    return None


def constraint_passes(value: float | None, row: ConstraintRow) -> str:
    if value is None:
        return "not evaluable"
    optimal = row.limit_optimal
    maximal = row.limit_maximal
    if optimal is not None and _compare(value, row.comparator, optimal):
        return "optimal"
    if maximal is not None and _compare(value, row.comparator, maximal):
        return "acceptable"
    if optimal is not None and maximal is None:
        return "optimal" if _compare(value, row.comparator, optimal) else "fail"
    return "fail"


def evaluate_constraints_for_dvh(
    roi_name: str,
    dvh: DvhCurve,
    rows: list[ConstraintRow],
) -> list[ConstraintEvaluation]:
    evaluations: list[ConstraintEvaluation] = []
    for row in rows:
        value = compute_constraint_metric(dvh, row)
        evaluations.append(
            ConstraintEvaluation(
                roi_name=roi_name,
                metric=row.metric,
                unit=row.unit,
                value=value,
                limit_optimal=row.limit_optimal,
                limit_maximal=row.limit_maximal,
                status=constraint_passes(value, row),
                source=row.source,
                priority=row.priority,
            )
        )
    return evaluations


def _compare(value: float, comparator: str, limit: float) -> bool:
    if comparator == ">=":
        return value >= limit
    if comparator == ">":
        return value > limit
    if comparator in {"=", "=="}:
        return abs(value - limit) < 1e-6
    if comparator == "<":
        return value < limit
    return value <= limit
