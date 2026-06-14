from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from planeval_viewer.refdb.models import ConstraintTable, RefDbLookupResult


DEFAULT_OFFLINE_CATALOG_PATH = Path(__file__).with_name("offline_examples") / (
    "stereotaxie_tables.json"
)


class OfflineRefDb:
    def __init__(self, path: Path = DEFAULT_OFFLINE_CATALOG_PATH) -> None:
        self.path = path
        self._catalog: dict[str, Any] | None = None

    def lookup_many(self, queries: Iterable[str], fx: int | None = None) -> dict[str, RefDbLookupResult]:
        catalog = self._load_catalog()
        tables_by_id = {
            str(table.get("id") or table.get("name") or index): ConstraintTable.from_dict(table)
            for index, table in enumerate(catalog.get("constraint_tables") or ())
            if isinstance(table, dict)
        }
        structures = [
            item for item in catalog.get("structures") or () if isinstance(item, dict)
        ]
        results: dict[str, RefDbLookupResult] = {}
        for query_index, query in enumerate(queries):
            match = _match_structure(str(query), structures)
            if match is None:
                continue
            table_ids = [str(item) for item in match.get("table_ids") or ()]
            tables = tuple(
                table
                for table_id, table in tables_by_id.items()
                if table_id in table_ids and table.matches_fraction(fx)
            )
            result_data = {
                "query_index": query_index,
                "query": str(query),
                "matched_name": match.get("canonical_name") or match.get("matched_name") or str(query),
                "reference_name": match.get("reference_name") or match.get("canonical_name") or str(query),
                "color": match.get("color") or "",
                "aliases": match.get("aliases") or (),
                "constraint_tables": [_constraint_table_to_dict(table) for table in tables],
            }
            results[str(query)] = RefDbLookupResult.from_dict(result_data)
        return results

    def _load_catalog(self) -> dict[str, Any]:
        if self._catalog is not None:
            return self._catalog
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        self._catalog = data if isinstance(data, dict) else {}
        return self._catalog


def _match_structure(query: str, structures: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized_query = _normalize_name(query)
    if not normalized_query:
        return None
    for structure in structures:
        candidates = [
            structure.get("canonical_name") or "",
            structure.get("reference_name") or "",
            *(structure.get("aliases") or ()),
        ]
        if normalized_query in {_normalize_name(candidate) for candidate in candidates}:
            return structure
    return None


def _constraint_table_to_dict(table: ConstraintTable) -> dict[str, Any]:
    return {
        "id": table.id,
        "name": table.name,
        "site": table.site,
        "regime": table.regime,
        "indication_detail": table.indication_detail,
        "dpf_min": table.dpf_min,
        "dpf_max": table.dpf_max,
        "fx_min": table.fx_min,
        "fx_max": table.fx_max,
        "td_min": table.td_min,
        "td_max": table.td_max,
        "prescriptions": list(table.prescriptions),
        "is_bilateral": table.is_bilateral,
        "constraints": [
            {
                "oar_raw": row.oar_raw,
                "metric": row.metric,
                "unit": row.unit,
                "comparator": row.comparator,
                "limit_optimal": row.limit_optimal,
                "limit_maximal": row.limit_maximal,
                "priority": row.priority,
                "source": row.source,
                "comment": row.comment,
            }
            for row in table.constraints
        ],
    }


def _normalize_name(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())

