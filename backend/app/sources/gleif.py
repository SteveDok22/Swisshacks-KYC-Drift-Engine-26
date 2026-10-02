"""
GLEIF — Global Legal Entity Identifier Foundation.

WHAT IT PROVIDES
    The global LEI reference data set (CC0 open data): per-LEI legal name,
    registration status (ISSUED / LAPSED / RETIRED / ANNULLED), legal
    jurisdiction, headquarters/legal address, and — crucially — the
    relationship graph: ``ultimate-parent`` and ``direct-children`` LEIs.
    The closest thing to a free, global, machine-readable ownership tree.

WHY IT MATTERS HERE  (Use cases 3, 4, 5, 8, 10)
    Diffing resolves: a legal-name change confirmed across borders (Case 8), a
    jurisdiction change (Case 4), an entity going ANNULLED (strong adverse), and
    — via the parent/children endpoints — a changed ultimate parent or a new
    subsidiary, i.e. real UBO/ownership drift (Cases 3, 5) feeding the contagion
    layer instead of a synthetic graph.

COST / ACCESS  →  FREE, no API key  (status: PLANNED — adapter fully implemented)
    Fully open JSON:API. Max 200 records/request; fair-use rate limiting (429).

    Base URL:  https://api.gleif.org/api/v1/
    Endpoints: GET /lei-records/{lei}
               GET /lei-records/{lei}/ultimate-parent
               GET /lei-records/{lei}/direct-children
               GET /lei-records?filter[entity.legalName]=...   (reverse lookup)

SIGNAL FLOW
    This is a *registry diff* adapter. ``fetch()`` returns the current
    ``EntitySnapshot``; ownership drift is then surfaced by diffing it against the
    stored KYC baseline. ``fetch_signals()`` performs that diff via
    ``ownership_change_signals(baseline, current)`` when the caller supplies BOTH
    snapshots, emitting ``ownership_change`` ``PublicSignal``s (use case 3). It
    does NO network I/O and returns an empty list when either snapshot is absent —
    in particular the live public-intel aggregator, which cannot carry per-source
    KYC baselines, gets ``[]``; the drift engine supplies the snapshot pair and
    layers the resulting signals in directly (see ``drift/service.py``).

OWNERSHIP REPRESENTATION
    beneficial_owners — list containing the ultimate-parent LEI (entity-level
                        UBO proxy; individual-person resolution requires a paid
                        source such as OpenCorporates)
    officers          — list of direct-child LEIs (subsidiary nodes for the
                        ownership-contagion PageRank layer; stored here because
                        no ``subsidiaries`` field exists on EntitySnapshot)

PAGINATION NOTE
    ``_get_children_leis`` follows the GLEIF JSON:API ``links.next`` cursor to
    paginate direct children (requesting the max page[size]=200), bounded by an
    overall ``_MAX_DIRECT_CHILDREN`` cap and a ``_MAX_CHILDREN_PAGES`` loop guard.
    It degrades gracefully — a transport error or non-2xx page stops the walk and
    returns the children gathered so far rather than raising or discarding them.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.logging import get_logger
from app.sources.base import EntitySnapshot, PublicSignal, RegistryAdapter, diff_snapshots
from app.sources.cost import AdapterStatus, CostMixin, SourceCost

logger = get_logger(__name__)

_BASE_URL = "https://api.gleif.org/api/v1"

# Human-readable provenance stamped on every GLEIF-derived PublicSignal.
_SIGNAL_SOURCE = "GLEIF (Global LEI)"

# --- direct-children pagination tuning ---------------------------------------
# GLEIF's JSON:API paginates ``direct-children`` (default page[size]=10, max
# 200) and exposes a ``links.next`` cursor. Fetching only page 1 truncates the
# ownership graph for large holding companies — the deferred follow-up from
# PR #45 (UC3) and PR #43 (UC5). ``_get_children_leis`` now follows the cursor.
#
# _CHILDREN_PAGE_SIZE — request the largest allowed page to minimise round-trips.
# _MAX_DIRECT_CHILDREN — overall cap on collected child LEIs; bounds memory and
#   request volume (the contagion layer only needs a representative subgraph, not
#   the entire tree). Pagination stops once this many children are gathered.
# _MAX_CHILDREN_PAGES — defensive ceiling on pages followed, guarding against a
#   malformed self-referential ``links.next`` looping forever.
_CHILDREN_PAGE_SIZE = 200
_MAX_DIRECT_CHILDREN = 1000
_MAX_CHILDREN_PAGES = 50

# Map the ownership-related routing keys produced by ``diff_snapshots`` to the
# noun-form ``PublicSignal.signal_type`` vocabulary. Use case 3 is scoped to
# ownership drift, so only the UBO (ultimate-parent LEI) and officer (direct-child
# LEI) changes are forwarded here; name / jurisdiction / status diffs belong to
# use cases 4 / 8 and are intentionally left unmapped in this adapter.
_OWNERSHIP_DIFF_TO_SIGNAL: dict[str, str] = {
    "ubo_added": "ownership_change",
    "ubo_removed": "ownership_change",
    "officer_added": "ownership_change",
    "officer_removed": "ownership_change",
}

_OWNERSHIP_HEADLINES: dict[str, str] = {
    "ubo_added": "Ultimate parent (UBO) added",
    "ubo_removed": "Ultimate parent (UBO) removed",
    "officer_added": "Direct subsidiary added",
    "officer_removed": "Direct subsidiary removed",
}

# GLEIF registration status → internal dissolution_status vocabulary.
# "LAPSED" means the LEI registration expired but the entity's legal existence
# is unconfirmed — NOT the same as "active". We store it as "lapsed" so the
# diff layer can detect ISSUED→LAPSED transitions (a potential KYC signal)
# rather than silently treating both as identical "active" states.
# "PENDING_*" registrations are also in-flight; status unknown → None.
_REG_STATUS_MAP: dict[str, str | None] = {
    "ISSUED": "active",
    "LAPSED": "lapsed",             # renewal expired — entity status unconfirmed
    "PENDING_VALIDATION": None,     # in-flight — true status unknown
    "PENDING_TRANSFER": None,
    "PENDING_ARCHIVAL": "dissolved",
    "RETIRED": "dissolved",
    "ANNULLED": "dissolved",
    "DUPLICATE": "dissolved",
    "TRANSFERRED": "dissolved",
    "MERGED": "dissolved",
}


class GleifAdapter(CostMixin, RegistryAdapter):
    """Global LEI connector — calls api.gleif.org/api/v1 (no key required)."""

    source_name = "gleif"
    display_name = "GLEIF (Global LEI)"
    base_url = _BASE_URL
    docs_url = "https://www.gleif.org/en/lei-data/gleif-api"
    cost = SourceCost.FREE
    status = AdapterStatus.PLANNED
    requires_api_key = False
    use_cases = (3, 4, 5, 8, 10)
    signal_types = (
        "name_change",
        "jurisdiction_change",
        "status_change",
        "ownership_change",
    )

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        # Accept an injected client so tests can pass a mock without any
        # network activity. Track ownership so aclose() only closes clients
        # we created — never ones the caller manages externally.
        if http_client is not None:
            self._http = http_client
            self._owns_client = False
        else:
            self._http = httpx.AsyncClient(
                base_url=_BASE_URL,
                headers={"Accept": "application/vnd.api+json"},
                timeout=10.0,
            )
            self._owns_client = True
        from app.core.api_cache import DiskCache  # local import avoids circular at module load
        self._cache = DiskCache("gleif")

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this adapter owns it."""
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> GleifAdapter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def record_url(self, entity_id: str) -> str | None:
        # entity_id is the 20-char LEI code.
        return f"https://search.gleif.org/#/record/{entity_id}"

    async def fetch(
        self, drift_id: str, name: str, **kwargs: Any
    ) -> EntitySnapshot | None:
        """
        Fetch the current GLEIF record for the entity.

        Parameters
        ----------
        drift_id:   internal drift engine ID for the customer.
        name:       legal name used for reverse lookup when ``lei`` is absent.
        lei:        (kwarg) the 20-char LEI code — preferred over name lookup.

        Returns None if the entity is not found in GLEIF, the API is
        unreachable, or rate-limited (429 — caller decides on retry).
        """
        lei: str | None = kwargs.get("lei")
        if not lei:
            lei = await self._lookup_lei_by_name(name)
        if not lei:
            return None

        raw = await self._get_lei_record(lei)
        if raw is None:
            return None

        # Fetch parent and children in parallel — both are optional; failures
        # are swallowed inside the helper methods.
        parent_lei, children_leis = await asyncio.gather(
            self._get_parent_lei(lei),
            self._get_children_leis(lei),
        )

        return self._normalize(drift_id, lei, raw, parent_lei, children_leis)

    async def fetch_signals(
        self,
        drift_id: str,
        name: str,
        since_month: int = 0,
        *,
        baseline: EntitySnapshot | None = None,
        current: EntitySnapshot | None = None,
        **kwargs: Any,
    ) -> list[PublicSignal]:
        """
        Emit ``ownership_change`` signals from the GLEIF ownership-chain diff.

        GLEIF is a registry diff adapter: it generates signals by comparing a
        baseline ``EntitySnapshot`` against the current one, not from a live news
        feed. The caller (the drift engine) supplies both via ``baseline`` and
        ``current``; this method does NO network I/O. Returns an empty list when
        either snapshot is missing — including the live aggregator path, which
        passes neither — satisfying the abstract contract without spurious calls.
        """
        if baseline is None or current is None:
            return []
        return ownership_change_signals(baseline, current, month=since_month)

    # ------------------------------------------------------------------
    # HTTP helpers — all use the ``is_success`` guard so that any non-2xx
    # response (404, 429, 500, 503, …) is swallowed and returns None/[].
    # A transient GLEIF outage must not propagate as an unhandled exception
    # through the drift service.
    # ------------------------------------------------------------------

    async def _get_lei_record(self, lei: str) -> dict[str, Any] | None:
        cached = self._cache.get(f"record:{lei}")
        if cached is not None:
            return cached
        try:
            resp = await self._http.get(f"/lei-records/{lei}")
        except httpx.TransportError:
            return None
        if not resp.is_success:
            return None
        data = resp.json()
        self._cache.set(f"record:{lei}", data)
        return data  # type: ignore[no-any-return]

    async def _lookup_lei_by_name(self, name: str) -> str | None:
        cached = self._cache.get(f"name:{name}")
        if cached is not None:
            return cached
        try:
            resp = await self._http.get(
                "/lei-records",
                params={"filter[entity.legalName]": name, "page[size]": "1"},
            )
        except httpx.TransportError:
            return None
        if not resp.is_success:
            return None
        items = resp.json().get("data", [])
        if not items:
            return None
        lei = items[0].get("id") or items[0].get("attributes", {}).get("lei")
        if lei:
            self._cache.set(f"name:{name}", lei)
        return lei  # type: ignore[no-any-return]

    async def _get_parent_lei(self, lei: str) -> str | None:
        # Wrapped as {"lei": ...} so a successful "no parent" answer is cacheable
        # too (a bare None would read back as a cache miss). Only 2xx responses
        # are cached; errors and 429s are retried on the next call.
        cached = self._cache.get(f"parent:{lei}")
        if cached is not None:
            return cached.get("lei")  # type: ignore[no-any-return]
        try:
            resp = await self._http.get(f"/lei-records/{lei}/ultimate-parent")
        except httpx.TransportError:
            return None
        if not resp.is_success:
            return None
        data = resp.json().get("data")
        parent = (data.get("id") or data.get("attributes", {}).get("lei")) if data else None
        self._cache.set(f"parent:{lei}", {"lei": parent})
        return parent  # type: ignore[no-any-return]

    async def _get_children_leis(self, lei: str) -> list[str]:
        """
        Fetch direct-child LEIs, following GLEIF JSON:API pagination.

        Walks the ``links.next`` cursor until it is exhausted, the overall
        ``_MAX_DIRECT_CHILDREN`` cap is reached, or ``_MAX_CHILDREN_PAGES`` is
        hit. Degrades gracefully: any transport error or non-2xx page stops the
        walk and returns whatever was collected so far, so a mid-pagination
        GLEIF hiccup yields a partial graph rather than an exception or [].
        Single-page responses (absent/null ``links.next``) return after one
        request, matching the previous behaviour. Only a walk that finished
        without errors is cached; a partial result is never persisted.
        """
        cached = self._cache.get(f"children:{lei}")
        if cached is not None:
            return list(cached.get("leis", []))

        leis: list[str] = []
        # First request: explicitly ask for the largest page; subsequent pages
        # follow the absolute ``links.next`` URL, which already carries page[size].
        url: str | None = f"/lei-records/{lei}/direct-children"
        params: dict[str, str] | None = {"page[size]": str(_CHILDREN_PAGE_SIZE)}

        for _ in range(_MAX_CHILDREN_PAGES):
            if url is None:
                break
            try:
                resp = await self._http.get(url, params=params)
            except httpx.TransportError:
                return leis
            if not resp.is_success:
                return leis

            body = resp.json()
            for item in body.get("data", []):
                child = item.get("id") or item.get("attributes", {}).get("lei")
                if child:
                    leis.append(child)
                    if len(leis) >= _MAX_DIRECT_CHILDREN:
                        self._cache.set(f"children:{lei}", {"leis": leis})
                        return leis

            # Follow the JSON:API cursor. ``links.next`` is an absolute URL with
            # its own encoded query string, so clear ``params`` to avoid resending.
            links = body.get("links") or {}
            url = links.get("next")
            params = None

        self._cache.set(f"children:{lei}", {"leis": leis})
        return leis

    # ------------------------------------------------------------------
    # Normalisation — pure, no I/O
    # ------------------------------------------------------------------

    def _normalize(
        self,
        drift_id: str,
        lei: str,
        raw: dict[str, Any],
        parent_lei: str | None,
        children_leis: list[str],
    ) -> EntitySnapshot:
        data = raw.get("data", {})
        attrs = data.get("attributes", {})
        entity = attrs.get("entity", {})
        registration = attrs.get("registration", {})

        return EntitySnapshot(
            drift_id=drift_id,
            name=_extract_name(entity),
            source=self.source_name,
            legal_form=_extract_legal_form(entity),
            jurisdiction=entity.get("jurisdiction") or None,
            registered_address=_extract_address(entity),
            dissolution_status=_extract_status(entity, registration),
            # Ownership nodes: parent LEI as entity-level UBO proxy; direct
            # children stored in ``officers`` (no ``subsidiaries`` field exists
            # on EntitySnapshot) for the ownership-contagion PageRank layer.
            beneficial_owners=[parent_lei] if parent_lei else [],
            officers=children_leis,
            raw_data={
                "lei": lei,
                "registration_status": registration.get("status"),
                "entity_status": entity.get("status"),
                # GLEIF records a documented former legal name (real name change,
                # e.g. "Weight Watchers International, Inc." -> "WW International,
                # Inc."). Surfaced as a real name_change signal — see
                # name_change_signals() — without needing a baseline diff.
                "former_legal_name": _extract_former_legal_name(entity),
            },
        )


# ---------------------------------------------------------------------------
# Pure normalisation helpers — module-level so tests can call them directly.
# ---------------------------------------------------------------------------

def _extract_name(entity: dict[str, Any]) -> str:
    legal_name = entity.get("legalName", {})
    if isinstance(legal_name, dict):
        return legal_name.get("name", "") or ""
    # Older records may return a plain string.
    return str(legal_name) if legal_name else ""


def _extract_legal_form(entity: dict[str, Any]) -> str | None:
    lf = entity.get("legalForm")
    if not lf or not isinstance(lf, dict):
        return None
    return lf.get("id") or lf.get("other") or None


def _extract_former_legal_name(entity: dict[str, Any]) -> str | None:
    """Return the documented previous legal name from a GLEIF record, if any.

    GLEIF stores former names in ``entity.otherNames`` with
    ``type == "PREVIOUS_LEGAL_NAME"``. Returns the first such name (most recent
    rename), or ``None`` when the entity has never been renamed.
    """
    for other in entity.get("otherNames") or []:
        if not isinstance(other, dict):
            continue
        if (other.get("type") or "").upper() == "PREVIOUS_LEGAL_NAME":
            name = (other.get("name") or "").strip()
            if name:
                return name
    return None


def name_change_signals(
    current: EntitySnapshot | None, *, month: int = 0
) -> list[PublicSignal]:
    """Emit a real ``name_change`` signal from GLEIF's documented former name.

    Unlike the ownership diff, this needs no baseline: GLEIF itself records the
    previous legal name on the live record (``PREVIOUS_LEGAL_NAME``), so a single
    fetch is enough to prove a rename. Drives the UC8 re-KYC score floor.
    """
    if current is None:
        return []
    raw = current.raw_data or {}
    former = (raw.get("former_legal_name") or "").strip()
    now = (current.name or "").strip()
    if not former or not now or former.casefold() == now.casefold():
        return []
    lei = raw.get("lei")
    source_url = f"https://search.gleif.org/#/record/{lei}" if lei else None
    return [
        PublicSignal(
            month=month,
            signal_type="name_change",
            headline=(
                f"GLEIF records a legal-name change: '{former}' -> '{now}' "
                f"(PREVIOUS_LEGAL_NAME on the LEI record) — re-KYC required"
            ),
            severity=0.85,
            source=_SIGNAL_SOURCE,
            source_url=source_url,
        )
    ]


def _extract_address(entity: dict[str, Any]) -> str | None:
    addr = entity.get("legalAddress")
    if not addr or not isinstance(addr, dict):
        return None
    parts: list[str] = []
    lines = addr.get("addressLines")
    if lines:
        parts.append(lines[0])
    if city := addr.get("city"):
        parts.append(city)
    if country := addr.get("country"):
        parts.append(country)
    return ", ".join(parts) if parts else None


def _extract_status(
    entity: dict[str, Any], registration: dict[str, Any]
) -> str | None:
    # Entity-level INACTIVE is definitive — overrides registration status.
    if entity.get("status") == "INACTIVE":
        return "dissolved"
    return _REG_STATUS_MAP.get(registration.get("status", ""))


# ---------------------------------------------------------------------------
# Ownership-chain diff → PublicSignal (use case 3) and bulk fetch helper.
# Module-level so the drift engine can call them without a live aggregator.
# ---------------------------------------------------------------------------

def ownership_change_signals(
    baseline: EntitySnapshot,
    current: EntitySnapshot,
    *,
    month: int = 0,
) -> list[PublicSignal]:
    """
    Diff a baseline GLEIF ownership chain against the current one and emit
    ``ownership_change`` ``PublicSignal``s (use case 3).

    Pure compute — no network. ``beneficial_owners`` carries the ultimate-parent
    LEI and ``officers`` the direct-child LEIs, so a UBO/officer add or removal is
    a real parent/subsidiary change in the GLEIF relationship graph.

    ``baseline`` and ``current`` MUST originate from the same source (both GLEIF):
    the parent/child identifiers are LEIs, only comparable within one registry
    (see the same-source contract in ``sources/base.py``). A cross-source pair is
    rejected to avoid spurious add/remove churn.
    """
    if baseline.source != current.source:
        logger.warning(
            "gleif_baseline_source_mismatch",
            drift_id=current.drift_id,
            baseline_source=baseline.source,
            current_source=current.source,
        )
        return []

    lei = current.raw_data.get("lei")
    source_url = f"https://search.gleif.org/#/record/{lei}" if lei else None

    signals: list[PublicSignal] = []
    for change in diff_snapshots(baseline, current):
        signal_type = _OWNERSHIP_DIFF_TO_SIGNAL.get(change.drift_signal_type)
        if signal_type is None:
            continue  # name / jurisdiction / status diffs belong to UC 4 / UC 8
        label = _OWNERSHIP_HEADLINES.get(change.drift_signal_type, change.drift_signal_type)
        # Removals carry the old value, additions the new — surface whichever exists.
        value = change.new_value if change.new_value is not None else change.old_value
        signals.append(
            PublicSignal(
                month=month,
                signal_type=signal_type,
                headline=f"{label}: {value!r}",
                severity=change.severity,
                source=_SIGNAL_SOURCE,
                source_url=source_url,
            )
        )
    return signals


async def gather_ownership_snapshots(
    entities: list[tuple[str, str]],
    *,
    adapter: GleifAdapter | None = None,
) -> dict[str, EntitySnapshot]:
    """
    Fetch current GLEIF ``EntitySnapshot``s for many customers in parallel.

    ``entities`` is a list of ``(drift_id, name)`` pairs. Returns a mapping of
    ``drift_id -> EntitySnapshot`` containing only the entities GLEIF resolved —
    unmatched names and per-entity errors are dropped so one bad lookup never
    sinks the batch. Used by the drift engine to build a real ownership-contagion
    graph (use case 3) instead of the synthetic demo graph. A caller may inject a
    shared ``adapter``; otherwise one is created and closed here.
    """
    if not entities:
        return {}

    own_adapter = adapter is None
    client = adapter or GleifAdapter()
    try:
        results = await asyncio.gather(
            *(client.fetch(drift_id, name) for drift_id, name in entities),
            return_exceptions=True,
        )
    finally:
        if own_adapter:
            await client.aclose()

    out: dict[str, EntitySnapshot] = {}
    for (drift_id, _name), result in zip(entities, results, strict=True):
        if isinstance(result, EntitySnapshot):
            out[drift_id] = result
        elif isinstance(result, Exception):
            logger.warning(
                "gleif_ownership_fetch_entity_error",
                drift_id=drift_id,
                error=str(result),
            )
    return out
