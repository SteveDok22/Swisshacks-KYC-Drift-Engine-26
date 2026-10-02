"""
Tests for the two caches that keep live entities from re-hitting rate-limited
external APIs on every request:

- GleifAdapter persists successful ``ultimate-parent`` / ``direct-children``
  lookups, and never persists errors or partial pagination walks.
- DriftEngine._cached_analysis memoises live entities only, with a TTL;
  synthetic customers are always recomputed.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.drift import service as drift_service
from app.sources.gleif import GleifAdapter

_LEI = "529900T8BM49AURSDO55"
_PARENT = "PARENTLEI00000000001X"
_CHILD = "CHILDLEIA0000000001X"


class _MemoryCache:
    """Stand-in for DiskCache (which is disabled under pytest)."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self.store.get(key)

    def set(self, key: str, data: Any) -> None:
        self.store[key] = data


def _adapter(handler) -> tuple[GleifAdapter, _MemoryCache, list[str]]:
    calls: list[str] = []

    def recording(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return handler(request)

    client = httpx.AsyncClient(
        base_url="https://api.gleif.org/api/v1",
        transport=httpx.MockTransport(recording),
    )
    adapter = GleifAdapter(http_client=client)
    cache = _MemoryCache()
    adapter._cache = cache  # type: ignore[assignment]
    return adapter, cache, calls


# ---------------------------------------------------------------------------
# GLEIF ultimate-parent
# ---------------------------------------------------------------------------


async def test_parent_lookup_is_cached_after_success():
    adapter, _, calls = _adapter(
        lambda r: httpx.Response(200, json={"data": {"id": _PARENT}})
    )
    assert await adapter._get_parent_lei(_LEI) == _PARENT
    assert await adapter._get_parent_lei(_LEI) == _PARENT
    assert len(calls) == 1


async def test_parent_lookup_caches_successful_no_parent():
    adapter, _, calls = _adapter(lambda r: httpx.Response(200, json={"data": None}))
    assert await adapter._get_parent_lei(_LEI) is None
    assert await adapter._get_parent_lei(_LEI) is None
    assert len(calls) == 1


@pytest.mark.parametrize("status", [404, 429, 500])
async def test_parent_lookup_error_is_not_cached(status):
    adapter, cache, calls = _adapter(lambda r: httpx.Response(status))
    assert await adapter._get_parent_lei(_LEI) is None
    assert await adapter._get_parent_lei(_LEI) is None
    assert len(calls) == 2
    assert cache.store == {}


# ---------------------------------------------------------------------------
# GLEIF direct-children
# ---------------------------------------------------------------------------


async def test_children_lookup_is_cached_after_complete_walk():
    adapter, _, calls = _adapter(
        lambda r: httpx.Response(200, json={"data": [{"id": _CHILD}], "links": {}})
    )
    assert await adapter._get_children_leis(_LEI) == [_CHILD]
    assert await adapter._get_children_leis(_LEI) == [_CHILD]
    assert len(calls) == 1


async def test_children_lookup_partial_walk_is_not_cached():
    # Page 1 succeeds and points at page 2, which is rate-limited.
    def handler(request: httpx.Request) -> httpx.Response:
        if "page" in str(request.url) and "number" in str(request.url):
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={
                "data": [{"id": _CHILD}],
                "links": {"next": f"https://api.gleif.org/api/v1/lei-records/{_LEI}/direct-children?page[number]=2"},
            },
        )

    adapter, cache, _ = _adapter(handler)
    # Partial result is still returned (graceful degradation is unchanged)...
    assert await adapter._get_children_leis(_LEI) == [_CHILD]
    # ...but never persisted.
    assert cache.store == {}


async def test_children_lookup_error_is_not_cached():
    adapter, cache, calls = _adapter(lambda r: httpx.Response(429))
    assert await adapter._get_children_leis(_LEI) == []
    assert await adapter._get_children_leis(_LEI) == []
    assert len(calls) == 2
    assert cache.store == {}


# ---------------------------------------------------------------------------
# DriftEngine._cached_analysis
# ---------------------------------------------------------------------------


def _counting_engine(monkeypatch) -> tuple[drift_service.DriftEngine, dict[str, int]]:
    engine = drift_service.DriftEngine()
    counts: dict[str, int] = {}

    def fake_analyze(cust):
        counts[cust.drift_id] = counts.get(cust.drift_id, 0) + 1
        return {"drift_id": cust.drift_id}

    monkeypatch.setattr(engine, "_analyze_customer", fake_analyze)
    return engine, counts


def test_live_entity_analysis_is_reused(monkeypatch):
    engine, counts = _counting_engine(monkeypatch)
    cust = engine._book[0]
    monkeypatch.setattr(cust, "mode", "live")

    first = engine._cached_analysis(cust)
    second = engine._cached_analysis(cust)
    assert first is second
    assert counts[cust.drift_id] == 1


def test_live_entity_analysis_expires_after_ttl(monkeypatch):
    engine, counts = _counting_engine(monkeypatch)
    cust = engine._book[0]
    monkeypatch.setattr(cust, "mode", "live")

    engine._cached_analysis(cust)
    stamp, analysis = engine._analysis_cache[cust.drift_id]
    engine._analysis_cache[cust.drift_id] = (
        stamp - engine._ANALYSIS_CACHE_TTL - 1.0,
        analysis,
    )
    engine._cached_analysis(cust)
    assert counts[cust.drift_id] == 2


def test_synthetic_customer_is_always_recomputed(monkeypatch):
    engine, counts = _counting_engine(monkeypatch)
    cust = next(c for c in engine._book if c.mode == "synthetic")

    engine._cached_analysis(cust)
    engine._cached_analysis(cust)
    assert counts[cust.drift_id] == 2
    assert cust.drift_id not in engine._analysis_cache
