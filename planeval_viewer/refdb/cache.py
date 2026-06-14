from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from planeval_viewer.refdb.models import RefDbLookupResult


class RefDbCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load_many(self, queries: Iterable[str]) -> dict[str, RefDbLookupResult]:
        raw = self._load_raw()
        results: dict[str, RefDbLookupResult] = {}
        for query in queries:
            item = raw.get(query)
            if isinstance(item, dict):
                results[query] = RefDbLookupResult.from_dict(item)
        return results

    def store_many(self, results: Iterable[RefDbLookupResult]) -> None:
        raw = self._load_raw()
        for result in results:
            if result.ok and result.query:
                raw[result.query] = result.raw or _result_to_cache_dict(result)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            return

    def _load_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}


def _result_to_cache_dict(result: RefDbLookupResult) -> dict[str, Any]:
    return {
        "query_index": result.query_index,
        "query": result.query,
        "matched_name": result.matched_name,
        "reference_name": result.reference_name,
        "side": result.side,
        "color": result.color,
        "aliases": list(result.aliases),
        "bilateral_included": result.bilateral_included,
        "bilateral_name": result.bilateral_name,
        "constraint_tables": [],
    }
