from planeval_viewer.refdb.models import (
    ConstraintTable,
    RefDbLookupResult,
    normalize_optional_number,
)


def test_normalize_optional_number_handles_empty_values():
    assert normalize_optional_number(None) is None
    assert normalize_optional_number("") is None
    assert normalize_optional_number("  ") is None
    assert normalize_optional_number("12") == 12.0


def test_constraint_table_fraction_match_ignores_missing_bounds():
    table = ConstraintTable.from_dict({"name": "Any", "fx_min": "", "fx_max": None})
    assert table.matches_fraction(12)


def test_constraint_table_fraction_match_checks_numeric_bounds():
    table = ConstraintTable.from_dict({"name": "12Fx", "fx_min": "10", "fx_max": 12})
    assert table.matches_fraction(12)
    assert not table.matches_fraction(15)


def test_lookup_result_parses_success_and_errors():
    result = RefDbLookupResult.from_dict(
        {"query_index": 0, "query": "Auge links", "matched_name": "Eye_L"}
    )
    error = RefDbLookupResult.from_dict(
        {"query_index": 1, "q": "bad", "error": "not found"}
    )

    assert result.ok
    assert result.query == "Auge links"
    assert error.ok is False
    assert error.error == "not found"
