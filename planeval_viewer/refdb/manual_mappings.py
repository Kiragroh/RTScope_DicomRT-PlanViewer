from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from planeval_viewer.refdb.matching import RoiLookup


class ManualMappingStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        raw = data.get("mappings") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return {}
        return {str(key): str(value) for key, value in raw.items() if key and value}

    def store(self, mappings: Mapping[str, str]) -> None:
        data = {"mappings": dict(sorted(mappings.items()))}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def upsert(self, reference_name: str, local_roi_name: str) -> dict[str, str]:
        mappings = self.load()
        if reference_name and local_roi_name:
            mappings[reference_name] = local_roi_name
            self.store(mappings)
        return mappings


def apply_manual_mappings(
    lookups: Mapping[str, RoiLookup],
    roi_names: list[str],
    mappings: Mapping[str, str],
) -> dict[str, RoiLookup]:
    updated = dict(lookups)
    for reference_name, local_roi in mappings.items():
        if local_roi not in roi_names:
            continue
        current = updated.get(local_roi)
        if current is None:
            updated[local_roi] = RoiLookup(
                source_name=local_roi,
                matched_name=reference_name,
                reference_name=reference_name,
            )
        else:
            updated[local_roi] = replace(
                current,
                matched_name=reference_name,
                reference_name=current.reference_name or reference_name,
                error="",
            )
    return updated
