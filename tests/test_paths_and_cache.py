from pathlib import Path

from planeval_viewer.refdb.cache import RefDbCache
from planeval_viewer.refdb.models import RefDbLookupResult


def test_default_cache_path_uses_local_appdata_not_current_working_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(Path("C:/Windows/System32"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    from planeval_viewer.paths import default_refdb_cache_path

    cache_path = default_refdb_cache_path()

    assert cache_path == tmp_path / "PlanEvalViewer" / "refdb_lookup_cache.json"


def test_cache_store_ignores_permission_errors(monkeypatch, tmp_path):
    cache = RefDbCache(tmp_path / "cache.json")

    def deny_mkdir(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)

    cache.store_many(
        [
            RefDbLookupResult(
                query_index=0,
                query="Auge links",
                matched_name="Eye_L",
            )
        ]
    )
