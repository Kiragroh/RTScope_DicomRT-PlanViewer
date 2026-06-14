from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen

from planeval_viewer.refdb.models import RefDbLookupResult


ENV_BASE_URLS = "PLANEVAL_REFDB_URLS"
DEFAULT_BASE_URLS: tuple[str, ...] = ()


class JsonTransport(Protocol):
    def post_json(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        ...

    def get_json(self, url: str, timeout: float) -> dict[str, Any]:
        ...


@dataclass
class UrllibJsonTransport:
    def post_json(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)

    def get_json(self, url: str, timeout: float) -> dict[str, Any]:
        with urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)


class RefDbClient:
    def __init__(
        self,
        base_urls: Sequence[str] | None = None,
        transport: JsonTransport | None = None,
        timeout: float = 5.0,
    ) -> None:
        configured_urls = DEFAULT_BASE_URLS if base_urls is None else tuple(base_urls)
        if base_urls is None:
            configured_urls = base_urls_from_environment()
        self.base_urls = tuple(url.rstrip("/") for url in configured_urls if url.strip())
        self.transport = transport or UrllibJsonTransport()
        self.timeout = timeout

    def lookup_batch(
        self,
        roi_names: Sequence[str],
        fx: int | None = None,
        use_server_fraction_filter: bool = False,
    ) -> list[RefDbLookupResult]:
        clean_names = [name.strip() for name in roi_names if name.strip()]
        if not clean_names:
            return []

        queries: list[dict[str, Any]] = []
        for name in clean_names:
            query: dict[str, Any] = {"q": name}
            if fx is not None and use_server_fraction_filter:
                query["fx"] = int(fx)
            queries.append(query)

        payload = {"queries": queries}
        last_error = ""
        for base_url in self.base_urls:
            url = f"{base_url}/api/refdb/lookup/batch"
            try:
                response = self.transport.post_json(url, payload, self.timeout)
            except (OSError, TimeoutError, URLError, ValueError) as exc:
                last_error = str(exc)
                continue

            results = [
                RefDbLookupResult.from_dict(item)
                for item in response.get("results", [])
            ]
            if fx is not None and not use_server_fraction_filter:
                results = [item.filtered_for_fraction(fx) for item in results]
            return _ensure_result_for_each_query(clean_names, results)

        get_results = self._lookup_batch_via_get(
            clean_names,
            fx=fx,
            use_server_fraction_filter=use_server_fraction_filter,
        )
        if get_results is not None:
            return get_results

        return [
            RefDbLookupResult(
                query_index=index,
                query=name,
                error=f"RefDB unavailable: {last_error or 'no base URL configured'}",
            )
            for index, name in enumerate(clean_names)
        ]

    def _lookup_batch_via_get(
        self,
        roi_names: Sequence[str],
        fx: int | None,
        use_server_fraction_filter: bool,
    ) -> list[RefDbLookupResult] | None:
        getter = getattr(self.transport, "get_json", None)
        if getter is None:
            return None
        for base_url in self.base_urls:
            results: list[RefDbLookupResult] = []
            for index, name in enumerate(roi_names):
                params: dict[str, Any] = {"q": name}
                if fx is not None and use_server_fraction_filter:
                    params["fx"] = int(fx)
                url = f"{base_url}/api/refdb/lookup?{urlencode(params)}"
                try:
                    data = getter(url, self.timeout)
                except (OSError, TimeoutError, URLError, ValueError) as exc:
                    results.append(
                        RefDbLookupResult(
                            query_index=index,
                            query=name,
                            error=str(exc),
                        )
                    )
                    continue
                item = RefDbLookupResult.from_dict(
                    {
                        **data,
                        "query_index": index,
                        "query": data.get("query") or name,
                    }
                )
                if fx is not None and not use_server_fraction_filter:
                    item = item.filtered_for_fraction(fx)
                results.append(item)
            if results:
                return results
        return None


def _ensure_result_for_each_query(
    roi_names: Sequence[str], results: Sequence[RefDbLookupResult]
) -> list[RefDbLookupResult]:
    by_index = {result.query_index: result for result in results}
    complete: list[RefDbLookupResult] = []
    for index, name in enumerate(roi_names):
        complete.append(
            by_index.get(
                index,
                RefDbLookupResult(
                    query_index=index,
                    query=name,
                    error="No RefDB result returned for this query",
                ),
            )
        )
    return complete


def base_urls_from_environment(value: str | None = None) -> tuple[str, ...]:
    raw = os.environ.get(ENV_BASE_URLS, "") if value is None else value
    separators = (";", ",", "\n")
    for separator in separators[1:]:
        raw = raw.replace(separator, separators[0])
    return tuple(part.strip().rstrip("/") for part in raw.split(separators[0]) if part.strip())
