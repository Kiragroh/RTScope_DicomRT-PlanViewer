from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from planeval_viewer.refdb.models import RefDbLookupResult


@dataclass(frozen=True)
class RoiLookup:
    source_name: str
    matched_name: str = ""
    reference_name: str = ""
    color: str = ""
    side: str = ""
    error: str = ""
    aliases: tuple[str, ...] = ()
    result: RefDbLookupResult | None = field(default=None, compare=False)

    @property
    def status(self) -> str:
        if self.error:
            return "not found"
        if self.matched_name:
            return "matched"
        return "unmatched"


def map_rois_to_results(
    roi_names: Sequence[str], results: Sequence[dict[str, Any] | RefDbLookupResult]
) -> list[RoiLookup]:
    parsed = [
        item if isinstance(item, RefDbLookupResult) else RefDbLookupResult.from_dict(item)
        for item in results
    ]
    by_index = {item.query_index: item for item in parsed}

    lookups: list[RoiLookup] = []
    for index, roi_name in enumerate(roi_names):
        result = by_index.get(index)
        if result is None:
            lookups.append(
                RoiLookup(
                    source_name=roi_name,
                    error="No RefDB result returned for this query",
                )
            )
            continue

        lookups.append(
            RoiLookup(
                source_name=roi_name,
                matched_name=result.matched_name,
                reference_name=result.reference_name,
                color=result.color,
                side=result.side,
                error=result.error,
                aliases=result.aliases,
                result=result,
            )
        )
    return lookups
