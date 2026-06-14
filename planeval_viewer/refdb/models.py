from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def normalize_optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ConstraintRow:
    oar_raw: str = ""
    metric: str = ""
    unit: str = ""
    comparator: str = "<="
    limit_optimal: float | None = None
    limit_maximal: float | None = None
    priority: str = ""
    source: str = ""
    comment: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstraintRow":
        return cls(
            oar_raw=str(data.get("oar_raw") or ""),
            metric=str(data.get("metric") or ""),
            unit=str(data.get("unit") or ""),
            comparator=str(data.get("comparator") or "<="),
            limit_optimal=normalize_optional_number(data.get("limit_optimal")),
            limit_maximal=normalize_optional_number(data.get("limit_maximal")),
            priority=str(data.get("priority") or ""),
            source=str(data.get("source") or ""),
            comment=str(data.get("comment") or ""),
        )


@dataclass(frozen=True)
class ConstraintTable:
    id: int | None = None
    name: str = ""
    site: str = ""
    regime: str = ""
    indication_detail: str = ""
    dpf_min: float | None = None
    dpf_max: float | None = None
    fx_min: float | None = None
    fx_max: float | None = None
    td_min: float | None = None
    td_max: float | None = None
    prescriptions: tuple[str, ...] = ()
    constraints: tuple[ConstraintRow, ...] = ()
    is_bilateral: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstraintTable":
        table_id = normalize_optional_number(data.get("id"))
        return cls(
            id=int(table_id) if table_id is not None else None,
            name=str(data.get("name") or ""),
            site=str(data.get("site") or ""),
            regime=str(data.get("regime") or ""),
            indication_detail=str(data.get("indication_detail") or ""),
            dpf_min=normalize_optional_number(data.get("dpf_min")),
            dpf_max=normalize_optional_number(data.get("dpf_max")),
            fx_min=normalize_optional_number(data.get("fx_min")),
            fx_max=normalize_optional_number(data.get("fx_max")),
            td_min=normalize_optional_number(data.get("td_min")),
            td_max=normalize_optional_number(data.get("td_max")),
            prescriptions=tuple(str(item) for item in data.get("prescriptions") or ()),
            constraints=tuple(
                ConstraintRow.from_dict(item) for item in data.get("constraints") or ()
            ),
            is_bilateral=bool(data.get("is_bilateral", False)),
        )

    def matches_fraction(self, fx: int | float | None) -> bool:
        if fx is None:
            return True
        if self.fx_min is not None and fx < self.fx_min:
            return False
        if self.fx_max is not None and fx > self.fx_max:
            return False
        return True


@dataclass(frozen=True)
class RefDbLookupResult:
    query_index: int = -1
    query: str = ""
    matched_name: str = ""
    reference_name: str = ""
    side: str = ""
    color: str = ""
    aliases: tuple[str, ...] = ()
    bilateral_included: bool = False
    bilateral_name: str = ""
    constraint_tables: tuple[ConstraintTable, ...] = ()
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefDbLookupResult":
        return cls(
            query_index=int(data.get("query_index", -1)),
            query=str(data.get("query") or data.get("q") or ""),
            matched_name=str(data.get("matched_name") or ""),
            reference_name=str(data.get("reference_name") or ""),
            side=str(data.get("side") or ""),
            color=str(data.get("color") or ""),
            aliases=tuple(str(item) for item in data.get("aliases") or ()),
            bilateral_included=bool(data.get("bilateral_included", False)),
            bilateral_name=str(data.get("bilateral_name") or ""),
            constraint_tables=tuple(
                ConstraintTable.from_dict(item)
                for item in data.get("constraint_tables") or ()
            ),
            error=str(data.get("error") or ""),
            raw=dict(data),
        )

    def filtered_for_fraction(self, fx: int | float | None) -> "RefDbLookupResult":
        if fx is None or not self.constraint_tables:
            return self
        return RefDbLookupResult(
            query_index=self.query_index,
            query=self.query,
            matched_name=self.matched_name,
            reference_name=self.reference_name,
            side=self.side,
            color=self.color,
            aliases=self.aliases,
            bilateral_included=self.bilateral_included,
            bilateral_name=self.bilateral_name,
            constraint_tables=tuple(
                table for table in self.constraint_tables if table.matches_fraction(fx)
            ),
            error=self.error,
            raw=self.raw,
        )
