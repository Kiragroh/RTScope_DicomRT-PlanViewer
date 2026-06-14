from planeval_viewer.refdb.client import RefDbClient, base_urls_from_environment
from planeval_viewer.refdb.matching import RoiLookup, map_rois_to_results


class FakeTransport:
    def __init__(self):
        self.calls = []

    def post_json(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return {
            "results": [
                {
                    "query_index": 0,
                    "query": "Auge links",
                    "matched_name": "Eye_L",
                    "color": "#800080",
                },
                {
                    "query_index": 1,
                    "q": "unknown",
                    "error": "Keine Struktur gefunden",
                },
            ]
        }


class BatchFailGetTransport:
    def __init__(self):
        self.post_calls = []
        self.get_calls = []

    def post_json(self, url, payload, timeout):
        self.post_calls.append((url, payload, timeout))
        raise OSError("batch unavailable")

    def get_json(self, url, timeout):
        self.get_calls.append((url, timeout))
        return {
            "query": "Auge links",
            "matched_name": "Eye_L",
            "reference_name": "Eye_L/R",
            "aliases": ["Auge"],
        }


def test_refdb_urls_are_read_from_environment_style_list():
    assert base_urls_from_environment("http://hub1:5001; http://hub2:5001\nhttp://hub3:5001") == (
        "http://hub1:5001",
        "http://hub2:5001",
        "http://hub3:5001",
    )


def test_batch_lookup_posts_queries_without_fx_when_disabled():
    transport = FakeTransport()
    client = RefDbClient(base_urls=["http://hub"], transport=transport)

    results = client.lookup_batch(
        ["Auge links", "unknown"], fx=12, use_server_fraction_filter=False
    )

    assert transport.calls[0][0] == "http://hub/api/refdb/lookup/batch"
    assert transport.calls[0][1] == {
        "queries": [{"q": "Auge links"}, {"q": "unknown"}]
    }
    assert results[0].matched_name == "Eye_L"
    assert results[1].error == "Keine Struktur gefunden"


def test_batch_lookup_can_include_server_fraction_filter():
    transport = FakeTransport()
    client = RefDbClient(base_urls=["http://hub"], transport=transport)

    client.lookup_batch(["Auge links"], fx=12, use_server_fraction_filter=True)

    assert transport.calls[0][1] == {"queries": [{"q": "Auge links", "fx": 12}]}


def test_batch_lookup_falls_back_to_documented_get_endpoint():
    transport = BatchFailGetTransport()
    client = RefDbClient(base_urls=["http://hub"], transport=transport)

    results = client.lookup_batch(["Auge links"], fx=12, use_server_fraction_filter=True)

    assert transport.post_calls
    assert transport.get_calls
    assert transport.get_calls[0][0].startswith("http://hub/api/refdb/lookup?")
    assert "q=Auge+links" in transport.get_calls[0][0]
    assert "fx=12" in transport.get_calls[0][0]
    assert results[0].matched_name == "Eye_L"
    assert results[0].query_index == 0


def test_roi_mapping_preserves_unmatched_errors():
    lookups = map_rois_to_results(
        ["Auge links", "unknown"],
        [
            {
                "query_index": 0,
                "query": "Auge links",
                "matched_name": "Eye_L",
                "color": "#800080",
            },
            {
                "query_index": 1,
                "q": "unknown",
                "error": "Keine Struktur gefunden",
            },
        ],
    )

    assert lookups[0] == RoiLookup(
        source_name="Auge links",
        matched_name="Eye_L",
        color="#800080",
        error="",
    )
    assert lookups[1].source_name == "unknown"
    assert lookups[1].matched_name == ""
    assert lookups[1].error == "Keine Struktur gefunden"


def test_lookup_error_is_displayed_as_not_found_status():
    lookup = RoiLookup(source_name="unknown", error="Keine Struktur gefunden")

    assert lookup.status == "not found"
