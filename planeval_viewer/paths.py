from __future__ import annotations

import os
from pathlib import Path


APP_DIR_NAME = "PlanEvalViewer"


def default_refdb_cache_path() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        base = Path(local_appdata)
    else:
        base = Path.home() / ".planeval_viewer"
    return base / APP_DIR_NAME / "refdb_lookup_cache.json"


def default_manual_mappings_path() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        base = Path(local_appdata)
    else:
        base = Path.home() / ".planeval_viewer"
    return base / APP_DIR_NAME / "manual_roi_mappings.json"
