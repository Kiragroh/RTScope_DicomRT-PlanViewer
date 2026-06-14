from planeval_viewer.refdb.manual_mappings import ManualMappingStore, apply_manual_mappings
from planeval_viewer.refdb.matching import RoiLookup


def test_manual_mapping_store_persists_reference_to_local_roi(tmp_path):
    store = ManualMappingStore(tmp_path / "mappings.json")

    mappings = store.upsert("SpinalCord", "RM")

    assert mappings == {"SpinalCord": "RM"}
    assert ManualMappingStore(tmp_path / "mappings.json").load() == {"SpinalCord": "RM"}


def test_manual_mappings_update_roi_lookup_matches():
    lookups = {"RM": RoiLookup(source_name="RM", error="unmatched")}

    updated = apply_manual_mappings(
        lookups,
        roi_names=["RM", "PTV"],
        mappings={"SpinalCord": "RM", "Missing": "NotInPlan"},
    )

    assert updated["RM"].matched_name == "SpinalCord"
    assert updated["RM"].status == "matched"
    assert "NotInPlan" not in updated
