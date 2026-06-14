from pathlib import Path

from planeval_viewer.refdb.models import ConstraintTable, RefDbLookupResult
from planeval_viewer.refdb.offline import OfflineRefDb
from planeval_viewer.gui.main_window import _merge_lookup_sources


def test_offline_refdb_matches_alias_and_filters_stereotaxy_tables(tmp_path):
    catalog = tmp_path / "offline.json"
    catalog.write_text(
        """
{
  "structures": [
    {
      "canonical_name": "SpinalCord",
      "reference_name": "SpinalCord",
      "color": "#ffff00",
      "aliases": ["SpinalCord", "Myelon", "Cord"],
      "table_ids": [71, 72]
    }
  ],
  "constraint_tables": [
    {
      "id": 71,
      "name": "STX1:Stereotaxie_1Fx",
      "fx_min": 1,
      "fx_max": 1,
      "constraints": [
        {
          "oar_raw": "SpinalCord",
          "metric": "D0.03cc",
          "unit": "Gy",
          "limit_optimal": 10,
          "limit_maximal": 12
        }
      ]
    },
    {
      "id": 72,
      "name": "STX3:Stereotaxie_3Fx",
      "fx_min": 3,
      "fx_max": 3,
      "constraints": [
        {
          "oar_raw": "SpinalCord",
          "metric": "D0.03cc",
          "unit": "Gy",
          "limit_optimal": 18,
          "limit_maximal": 21
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    results = OfflineRefDb(catalog).lookup_many(["Myelon"], fx=3)

    assert results["Myelon"].matched_name == "SpinalCord"
    assert results["Myelon"].color == "#ffff00"
    assert [table.name for table in results["Myelon"].constraint_tables] == [
        "STX3:Stereotaxie_3Fx"
    ]
    assert results["Myelon"].constraint_tables[0].constraints[0].limit_optimal == 18


def test_merge_lookup_sources_adds_offline_constraints_to_empty_live_table():
    live = RefDbLookupResult(
        query_index=0,
        query="SpinalCord",
        matched_name="SpinalCord",
        constraint_tables=(ConstraintTable(id=71, name="STX1:Stereotaxie_1Fx"),),
    )
    offline = RefDbLookupResult.from_dict(
        {
            "query_index": 0,
            "query": "SpinalCord",
            "matched_name": "SpinalCord",
            "constraint_tables": [
                {
                    "id": 71,
                    "name": "STX1:Stereotaxie_1Fx",
                    "constraints": [
                        {
                            "oar_raw": "SpinalCord",
                            "metric": "D0.03cc",
                            "unit": "Gy",
                            "limit_optimal": 10,
                        }
                    ],
                }
            ],
        }
    )

    merged = _merge_lookup_sources(
        ["SpinalCord"],
        results=[live],
        cached={},
        offline={"SpinalCord": offline},
    )

    assert merged[0].matched_name == "SpinalCord"
    assert len(merged[0].constraint_tables) == 1
    assert merged[0].constraint_tables[0].constraints[0].oar_raw == "SpinalCord"
