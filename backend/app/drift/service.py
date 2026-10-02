"""
DriftEngine — orchestrator for KYC drift detection.

Parallels the existing RiskEngine pattern. Combines the passive layers
(BOCPD behavioral drift, drift velocity, ownership contagion, deterministic
checks) and routes customers through the cost-aware cascade.

For the hackathon MVP, the customer book is the synthetic suite from
simulator.py (deterministic, ground-truth-labeled). In production this would
read from the bank's transaction store and registry feeds.

The engine retains the generated customer book so IDs and injected scenarios
remain stable within one process. This mutable demo state is process-local.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
import zlib
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from app.core.config import (
    DRIFT_CONFIRMATION_LIFT_RANGE,
    DRIFT_CONFIRMATION_MAX_AMPLIFICATION,
    DRIFT_INTERNAL_ACCUMULATED_WEIGHT,
    DRIFT_INTERNAL_CONTAGION_WEIGHT,
    DRIFT_INTERNAL_VELOCITY_WEIGHT,
    DRIFT_PUBLIC_RISK_WEIGHT,
    settings,
)
from app.core.logging import get_logger
from app.db.kyc_baseline import (
    EntitySnapshotDB,
    load_latest_snapshot,
    store_snapshot,
)
from app.db.session import session_scope
from app.drift.bocpd import BOCPD, standardize
from app.drift.business_model import (
    BusinessModelComparison,
    CachedEmbedding,
    compare_business_model,
    text_fingerprint,
)
from app.drift.cascade import CascadeRouter, CustomerSignal, Tier
from app.drift.causal import causal_assessment
from app.drift.contagion import (
    OwnershipGraph,
    build_demo_graph,
    build_graph_from_snapshots,
)
from app.drift.dormancy import assess_dormancy
from app.drift.public_intel import (
    PublicSignal,
    assess_public_risk,
    confirmation_lift,
    detect_news_spike_month,
    gather_public_signals_sync,
    generate_signals_for_customer,
)
from app.drift.simulator import SyntheticCustomer, generate_book, generate_customer
from app.drift.stability import assess_stability, cohort_volatility
from app.drift.timetravel import replay_trajectory
from app.drift.velocity import compute_drift_series, velocity_band
from app.ml.base import score_to_level
from app.ml.extractors import DriftFeatureExtractor
from app.schemas.drift import (
    AsOfPointOut,
    CascadeCostReport,
    CausalVerdictOut,
    ContagionGraph,
    DormancyOut,
    DriftSubjectDetail,
    DriftSubjectSummary,
    DriftTimelinePoint,
    LayerContribution,
    PublicSignalOut,
    ReplayResult,
    StabilityOut,
    UboScreeningOut,
)
from app.schemas.enums import DecisionAction
from app.services.anthropic_client import get_anthropic_client
from app.sources.base import EntitySnapshot
from app.sources.firecrawl import FirecrawlAdapter
from app.sources.gleif import (
    gather_ownership_snapshots,
    name_change_signals,
    ownership_change_signals,
)
from app.sources.wayback import WaybackAdapter

logger = get_logger(__name__)

T2_LLM_SYSTEM_MESSAGE = (
    "You are a careful AML/KYC compliance analyst. Return valid JSON only. "
    "Do not invent facts. Use only the provided evidence. Do not recommend "
    "automatic account blocking. Recommend human compliance actions only."
)

LLM_PARSE_FALLBACK = {
    "verdict": "ambiguous",
    "confidence": 0.0,
    "rationale": "LLM response could not be parsed as valid JSON.",
    "key_evidence": [],
    "recommended_action": "Request information",
}

# Sanctioned seed entity for the contagion demo — must match the drift_id of
# Castor Trade Finance AG so the contagion graph node is reachable by drift_id.
SANCTIONED_SEED = "drift-011"
# Display name for the sanctioned seed node (kept in sync with build_demo_graph
# so the live-LEI and synthetic graphs label the flagged entity identically).
# drift-011 (Castor Trade Finance AG) is the flagship combo entity that anchors
# the contagion graph; it is the sanctioned node from which risk propagates.
SANCTIONED_SEED_NAME = "Castor Trade Finance AG"
# Wall-clock cap for the startup GLEIF ownership fetch (live mode only). On any
# timeout or failure the engine falls back to the synthetic demo graph.
_GLEIF_FETCH_TIMEOUT_S = 30.0
# Customers wired into the ownership graph as contagion-affected (1 hop from Castor).
# drift-003 = Alpine Logistics Group AG (combined), drift-008 = Bernina Wealth Partners AG.
CONTAGION_AFFECTED = {"drift-003", "drift-008"}
DRIFT_ANALYSIS_VERSION = "drift-v1"

# Wall-clock cap for the live business-model website fetch (Wayback + Firecrawl)
# in live mode. The cosine comparison must never block the engine, so any
# overrun (a slow archive.org capture, a hung scrape) is abandoned and the
# customer simply gets no business-model signal. Sized above the Wayback polite
# delay + two HTTP round-trips, well under the cascade's per-customer budget.
# Generous bound: a cold UC9 fetch chains Wayback (CDX, up to ~30s) + Firecrawl +
# embedding. Only the first uncached load pays it; cached loads are instant.
_WEBSITE_FETCH_TIMEOUT_S = 60.0
# raw_data key under which the comparator's embeddings are persisted, keyed by
# the SHA-256 text fingerprint the comparator returns. A re-scan reads this back
# as a read-through cache to skip re-embedding when the website text is unchanged.
_BUSINESS_MODEL_EMBEDDINGS_KEY = "business_model_embeddings"

# UC 4 — structural-change signal types that mandate a re-KYC review. A
# confirmed jurisdiction or legal-form change is a hard regulatory trigger, so
# its presence floors the drift score regardless of behavioral signal strength.
# These noun-form types are emitted by the ZEFIX/GLEIF diff path
# (sources/base.diff_snapshots -> adapter.fetch_signals); the offline synthetic
# feed never produces them, so this floor only fires on live registry data.
RE_KYC_FLOOR_SIGNAL_TYPES = frozenset({"jurisdiction_change", "legal_form_change"})
# Mandatory re-KYC drift-score floor (0..100) for a confirmed structural change.
RE_KYC_SCORE_FLOOR = 50.0


def requires_re_kyc_floor(public_signals: Iterable[PublicSignal]) -> bool:
    """Return True if any public signal is a confirmed structural change that
    mandates re-KYC (UC 4 — jurisdiction or legal-form change).

    Pure predicate over the signal list so the floor policy can be unit-tested
    without constructing the engine — the offline synthetic feed never emits
    these registry-sourced types.
    """
    return any(s.signal_type in RE_KYC_FLOOR_SIGNAL_TYPES for s in public_signals)


# A confirmed (definitive) OFAC/EU sanctions match — on the entity itself or on a
# newly-added UBO — is the strongest possible signal: mandatory escalation to
# critical regardless of how benign the transaction behaviour looks (UC 5 / UC 8).
# Floor the drift score here. Fires on the live OpenSanctions definitive hits and
# on the synthetic ownership_shift / combined sanctioned-UBO scenarios, which
# mirror that path. Probable (sub-threshold) matches are excluded — they carry a
# lower severity and still surface via the public layer without forcing critical.
# 90 lands in the CRITICAL band (>=86 in score_to_level) — a confirmed sanctions
# match is critical in any AML workflow.
SANCTIONS_SCORE_FLOOR = 90.0

# News-derived "narrative" signal types. These come from news feeds (Event
# Registry / GDELT) and the website comparator — sources that are unreliable in
# live mode (rate limits; the real adverse event often predates the query
# window). Registry/screening types (ownership_change, sanctions,
# jurisdiction_change, name_change, domain_change) are NOT here: those are
# authoritative live and must never be overwritten by a modeled fallback.
_NARRATIVE_SIGNAL_TYPES = frozenset(
    {
        "news",
        "adverse_media",
        "funding_event",
        "business_model_change",
        "corridor_alert",
    }
)


def is_definitively_sanctioned(public_signals: Iterable[PublicSignal]) -> bool:
    """Return True if any public signal is a definitive sanctions match.

    Covers a direct ``sanctions`` hit at the critical severity band and a
    UBO-screening ``ownership_change`` flagged ``definitive`` in its ``meta``.
    Pure predicate so the floor policy is unit-testable without the engine.
    """
    for s in public_signals:
        if s.signal_type == "sanctions" and s.severity >= 0.90:
            return True
        if (
            s.signal_type == "ownership_change"
            and s.meta
            and s.meta.get("kind") == "ubo_screening"
            and s.meta.get("definitive")
        ):
            return True
    return False


def recommend_drift_action(
    score: float,
    causal_label: str,
    is_suspicious: bool,
) -> DecisionAction:
    """Single authoritative mapping used by API responses and decisions."""
    if is_suspicious:
        return DecisionAction.ESCALATE
    if causal_label == "benign":
        return DecisionAction.ALLOW
    if score >= 70 or causal_label == "risk":
        return DecisionAction.ESCALATE
    if score >= 40 or causal_label == "ambiguous":
        return DecisionAction.STEP_UP_VERIFICATION
    return DecisionAction.ALLOW


def confirmation_amplification(lift: float) -> float:
    """Map a confirmation lift (>= 1) to a multiplicative score amplification.

    The lift's excess over 1 is mapped, over a window of
    ``DRIFT_CONFIRMATION_LIFT_RANGE``, into up to
    ``DRIFT_CONFIRMATION_MAX_AMPLIFICATION`` of additional weight, e.g. a lift of
    ``1 + DRIFT_CONFIRMATION_LIFT_RANGE`` (or higher) saturates at the maximum.
    """
    return 1.0 + min(
        (lift - 1.0) / DRIFT_CONFIRMATION_LIFT_RANGE,
        1.0,
    ) * DRIFT_CONFIRMATION_MAX_AMPLIFICATION


def compute_drift_analysis(
    cust: SyntheticCustomer,
    *,
    cohort_cv: float,
    propagated_risk: float = 0.0,
    public_signals: list[PublicSignal] | None = None,
) -> dict:
    """Passive-layer drift analysis shared by DriftEngine (live) and the
    offline training pipeline.

    Produces every raw signal plus the heuristic ``drift_score`` (after the
    causal modulation) but BEFORE the XGBoost blend and the suspicious-stability
    / dormancy-break floors — those are scoring *policy* the caller applies.

    Offline callers (model training) pass ``propagated_risk=0.0`` and
    ``public_signals=None``: there is no contagion graph and no live adapters,
    so those layers are neutral. Crucially the ``internal_risk`` formula and
    every other layer are computed here identically for both paths, so the
    training feature distribution can never silently diverge from inference.

    NOTE: with no live signals, ``propagated_risk``, ``public_risk`` and
    ``confirmation_lift`` are fixed at their neutral defaults (0.0, 0.0, 1.0).
    The model therefore learns nothing from those three of the twenty features
    during offline training — they only carry signal at inference time.
    """
    signals = public_signals or []

    ds = compute_drift_series(cust.metric_windows())
    latest_velocity = ds.velocity[-1] if ds.velocity else 0.0
    max_velocity = max(ds.velocity) if ds.velocity else 0.0
    final_drift = ds.drift_bits[-1] if ds.drift_bits else 0.0

    # --- INTERNAL: BOCPD on daily volume ---
    daily = standardize(cust.daily_volume_series())
    bres = BOCPD(hazard=1 / 500).run(daily)
    cp_day = bres.detected_changepoints[0] if bres.detected_changepoints else None
    cp_month = cust.day_to_month(cp_day) if cp_day is not None else None
    if cp_month is not None:
        internal_peak_month = cp_month
    elif ds.velocity:
        internal_peak_month = ds.windows[int(np.argmax(ds.velocity))]
    else:
        internal_peak_month = None

    # Internal risk 0..1: velocity (leading) + accumulated drift + contagion
    vel_norm = min(max_velocity / 3.0, 1.0)
    drift_norm = min(final_drift / 20.0, 1.0)
    internal_risk = min(
        DRIFT_INTERNAL_VELOCITY_WEIGHT * vel_norm
        + DRIFT_INTERNAL_ACCUMULATED_WEIGHT * drift_norm
        + DRIFT_INTERNAL_CONTAGION_WEIGHT * propagated_risk,
        1.0,
    )

    # --- PUBLIC + Confirmation Lift ---
    pi = assess_public_risk(signals, months=cust.months)
    # News-volume regime change (UC 1): BOCPD over the weekly event-count series.
    # When a sustained news spike is found, its onset month is the public anchor
    # for the confirmation-lift temporal window — a better "when did the external
    # story break" marker than the single peak-severity signal. Falls back to the
    # peak-severity month when no spike is detected (or there are no signals).
    news_spike_month = detect_news_spike_month(signals, cust.months)
    public_peak_month = (
        news_spike_month if news_spike_month is not None else pi.peak_signal_month
    )
    lift = confirmation_lift(
        pi.public_risk, internal_risk,
        public_peak_month, internal_peak_month,
    )

    # --- Fused score 0..100 ---
    base = max(internal_risk, pi.public_risk * DRIFT_PUBLIC_RISK_WEIGHT)
    amplification = confirmation_amplification(lift)
    score = min(base * amplification * 100.0, 100.0)

    # --- CAUSAL / SUSPICIOUS STABILITY / DORMANCY ---
    # UC6: surface public funding_event months so the causal layer can corroborate
    # a >= 5x scale jump as acquisition/funding-driven (scale risk) rather than
    # an unexplained volume jump.
    funding_event_months = [s.month for s in signals if s.signal_type == "funding_event"]
    causal = causal_assessment(
        cust.causal_windows(), funding_event_months=funding_event_months
    )
    stability = assess_stability(
        cust.monthly_volume,
        cohort_cv,
        counterparty_monthly=cust.counterparty_risk,
        corridor_monthly=cust.corridor_risk,
        public_risk=pi.public_risk,
    )
    dormancy = assess_dormancy(cust.monthly_volume)

    # Causal modulation — demote benign (life-shaped) drift, confirm risk-shaped.
    causal_factor = 0.45 + 0.55 * causal.p_risk
    score = min(score * causal_factor, 100.0)

    return {
        "drift_series": ds,
        "latest_velocity": latest_velocity,
        "max_velocity": max_velocity,
        "final_drift": final_drift,
        "bocpd_changepoint_day": cp_day,
        "bocpd_changepoint_month": cp_month,
        "propagated_risk": propagated_risk,
        "internal_risk": internal_risk,
        "internal_peak_month": internal_peak_month,
        "public_signals": signals,
        "public_risk": pi.public_risk,
        "public_peak_month": pi.peak_signal_month,
        "news_spike_month": news_spike_month,
        "confirmation_lift": lift,
        "causal": causal,
        "stability": stability,
        "dormancy": dormancy,
        "drift_score": score,
    }


class DriftEngine:
    """Orchestrates drift detection over the customer book."""

    _LIST_CACHE_TTL: float = 30.0  # seconds — controls list_subjects() hot-path cache
    # Per-customer analysis reuse for LIVE entities across list/detail/scan
    # requests. They hit external APIs (rate-limited, up to ~30 s each), so
    # re-analysing them on every request made the workspace take minutes to
    # load. Synthetic customers are cheap and deterministic and stay uncached.
    _ANALYSIS_CACHE_TTL: float = 600.0  # seconds

    def __init__(self) -> None:
        self._book: list[SyntheticCustomer] = generate_book()
        self._router = CascadeRouter()
        # GLEIF ownership data (live mode only). ``_gleif_snapshots`` holds each
        # customer's current ownership chain (parent + child LEIs); the engine
        # diffs it against ``_gleif_baselines`` (persisted GLEIF KYC baselines) to
        # emit ownership_change signals. Both stay empty in offline/mock mode.
        self._gleif_snapshots: dict[str, EntitySnapshot] = {}
        self._gleif_baselines: dict[str, EntitySnapshot] = {}
        self._graph: OwnershipGraph = self._build_ownership_graph()
        # Load persisted GLEIF onboarding baselines so the live ownership diff
        # has a same-source anchor to fire against (PR #45 follow-up). Live mode
        # only — offline keeps the empty mapping so the diff stays inert and the
        # offline scores are unchanged.
        if settings.external_apis_enabled or any(
            getattr(c, "mode", "synthetic") == "live" for c in self._book
        ):
            self._gleif_baselines = self._load_gleif_baselines()
        # Contagion is computed once (sanctions already hit in demo state)
        self._contagion = self._graph.propagate(seeds=[SANCTIONED_SEED])
        # Cohort volatility reference for Suspicious Stability (computed once
        # over the whole book — the norm against which smoothness is judged).
        self._cohort_cv = cohort_volatility([c.monthly_volume for c in self._book])
        self._list_cache: list[DriftSubjectSummary] | None = None
        self._list_cache_at: float = 0.0
        self._analysis_cache: dict[str, tuple[float, dict]] = {}
        # Optional XGBoost drift model — loaded lazily; absent = heuristic-only.
        self._drift_extractor = DriftFeatureExtractor()
        self._drift_model = self._load_drift_model()
        # Running count of ML blend failures; escalates to ERROR after threshold.
        self._ml_blend_failure_count: int = 0

    def _load_drift_model(self):
        """Load the drift XGBoost model if available; return None otherwise."""
        try:
            from app.ml.registry import get_registry
            from app.schemas.enums import CaseType

            registry = get_registry()
            return registry.get(CaseType.KYC_DRIFT)
        except Exception:
            # Heuristic-only is a valid mode, but make the fallback visible —
            # the startup registry logs `model_file_missing`; this parallel
            # path must not swallow a genuine load error in silence.
            logger.warning("drift_model_load_failed", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Ownership-contagion graph (Layer 3) — real GLEIF vs synthetic demo
    # ------------------------------------------------------------------ #
    def _build_ownership_graph(self) -> OwnershipGraph:
        """Build the ownership-contagion graph from real LEI data when possible.

        Live mode (``external_apis_enabled``): fetch each customer's current GLEIF
        ownership chain (ultimate-parent + direct-child LEIs) and build a real
        graph via :func:`build_graph_from_snapshots`. Offline/mock mode, or any
        case where GLEIF resolves no ownership links (e.g. unmatched synthetic
        names, a rate-limited API, or a transport error), degrades to the
        synthetic :func:`build_demo_graph` so the contagion demo always has a
        topology to propagate over.
        """
        drift_id_names = {c.drift_id: c.name for c in self._book}
        if not settings.external_apis_enabled:
            return build_demo_graph(drift_id_names)

        self._gleif_snapshots = self._fetch_gleif_snapshots(
            [(c.drift_id, c.name) for c in self._book]
        )
        # Seed the same flagged entity used by the synthetic demo graph into the
        # real-LEI graph so contagion propagates over live LEIs (PR #45 follow-up).
        graph = build_graph_from_snapshots(
            self._gleif_snapshots,
            sanctioned_seed=SANCTIONED_SEED,
            sanctioned_seed_name=SANCTIONED_SEED_NAME,
            contagion_affected=CONTAGION_AFFECTED,
        )
        if graph is None:
            logger.info(
                "ownership_graph_fallback_demo",
                reason="gleif_returned_no_ownership_links",
            )
            return build_demo_graph(drift_id_names)
        logger.info(
            "ownership_graph_from_gleif", customers=len(self._gleif_snapshots)
        )
        return graph

    @staticmethod
    def _fetch_gleif_snapshots(
        entities: list[tuple[str, str]],
    ) -> dict[str, EntitySnapshot]:
        """Synchronous bridge to the async GLEIF ownership fetch.

        Mirrors the public-intel sync bridge: runs the async fetch in a dedicated
        thread with its own event loop, so it is safe to call from the synchronous
        engine constructor even under a running FastAPI loop. Any failure (timeout
        included) degrades to an empty mapping; the caller then falls back to the
        demo graph. ``shutdown(wait=False)`` so a hung worker never re-blocks us.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(asyncio.run, gather_ownership_snapshots(entities))
        try:
            return future.result(timeout=_GLEIF_FETCH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — honour the graceful-degradation contract
            logger.warning("gleif_ownership_fetch_failed", exc_info=True)
            return {}
        finally:
            pool.shutdown(wait=False)

    @staticmethod
    def _load_gleif_baselines() -> dict[str, EntitySnapshot]:
        """Synchronous bridge to load persisted GLEIF onboarding baselines.

        Mirrors ``_fetch_gleif_snapshots``: runs the async DB read in a dedicated
        thread with its own event loop. It also creates its OWN short-lived async
        engine inside that thread (the way ``gather_ownership_snapshots`` owns its
        httpx client) so it never touches the app's engine, which is bound to a
        different loop — calling that engine from this thread would raise. Any
        failure (timeout included) degrades to an empty mapping, leaving the
        ownership diff inert exactly as in offline mode.
        """
        async def _load() -> dict[str, EntitySnapshot]:
            from sqlalchemy.ext.asyncio import (
                AsyncSession,
                async_sessionmaker,
                create_async_engine,
            )

            from app.db.kyc_baseline import load_gleif_baselines

            engine = create_async_engine(
                settings.database_url,
                connect_args={"check_same_thread": False},
            )
            try:
                factory = async_sessionmaker(
                    engine, class_=AsyncSession, expire_on_commit=False
                )
                async with factory() as session:
                    return await load_gleif_baselines(session)
            finally:
                await engine.dispose()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(asyncio.run, _load())
        try:
            return future.result(timeout=_GLEIF_FETCH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — honour the graceful-degradation contract
            logger.warning("gleif_baseline_load_failed", exc_info=True)
            return {}
        finally:
            pool.shutdown(wait=False)

    def _gleif_ownership_signals(self, cust: SyntheticCustomer) -> list[PublicSignal]:
        """Emit ``ownership_change`` signals from the GLEIF ownership-chain diff.

        Diffs the customer's current GLEIF snapshot against its persisted GLEIF
        KYC baseline (same source) via :func:`ownership_change_signals`. The seed
        persists a dedicated ``source="gleif"`` onboarding baseline per customer
        (``_load_gleif_baselines`` loads it in live mode), satisfying the
        same-source contract that excludes the ``internal`` baseline. Returns
        ``[]`` when either side is absent — offline mode, an unmatched customer,
        or before a GLEIF baseline has been persisted.
        """
        baseline = self._gleif_baselines.get(cust.drift_id)
        current = self._gleif_snapshots.get(cust.drift_id)
        if baseline is None or current is None:
            return []
        return ownership_change_signals(baseline, current)

    def _gleif_name_change_signals(self, cust: SyntheticCustomer) -> list[PublicSignal]:
        """Emit a real ``name_change`` signal from GLEIF's documented former name.

        Needs only the live GLEIF record (no baseline): GLEIF stores the previous
        legal name on the record itself. Uses the prebuilt ownership-graph
        snapshot when present (live-graph mode); otherwise fetches the single
        record directly (per-entity live mode, the default). Returns ``[]`` when
        the entity has never been renamed or GLEIF is unreachable.
        """
        current = self._gleif_snapshots.get(cust.drift_id)
        if current is None:
            current = self._fetch_gleif_record(cust.name)
        return name_change_signals(current)

    @staticmethod
    def _fetch_gleif_record(name: str) -> EntitySnapshot | None:
        """Fetch a single live GLEIF record (thread-isolated, cached, best-effort)."""
        async def _fetch() -> EntitySnapshot | None:
            from app.sources.gleif import GleifAdapter

            adapter = GleifAdapter()
            try:
                return await adapter.fetch("", name)
            finally:
                await adapter.aclose()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(asyncio.run, _fetch())
        try:
            return future.result(timeout=_GLEIF_FETCH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — best-effort; degrade to no signal
            logger.warning("gleif_name_change_fetch_failed", exc_info=True)
            return None
        finally:
            pool.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # Public-intelligence acquisition (live vs offline)
    # ------------------------------------------------------------------ #
    def _public_signals(self, cust: SyntheticCustomer) -> list[PublicSignal]:
        """Acquire public-intel signals for one customer.

        Single seam controlling external-API usage:
          - ``external_apis_enabled`` True  -> run the real source adapters in
            parallel via the public-intel aggregator (live HTTP).
          - ``external_apis_enabled`` False -> bypass all adapters and emit
            deterministic, scenario-aligned synthetic signals. No network I/O.

        The synthetic path reuses ``generate_signals_for_customer`` (also used by
        the time-travel replay), seeded by ``drift_id`` so the same customer
        always yields the same signals across requests.
        """
        if settings.external_apis_enabled or getattr(cust, "mode", "synthetic") == "live":
            signals = gather_public_signals_sync(cust.drift_id, cust.name)
            # The aggregator cannot carry per-source KYC baselines, so the GLEIF
            # ownership-chain diff (use case 3) is layered in here alongside the
            # other adapters' signals, then re-sorted by month.
            signals.extend(self._gleif_ownership_signals(cust))
            # Real GLEIF name-change (UC4/UC8) from the record's former legal name.
            signals.extend(self._gleif_name_change_signals(cust))
            # Hybrid fallback: registry/screening sources (GLEIF, OpenSanctions,
            # WHOIS) are reliable live, but the live NEWS feeds frequently have no
            # recent coverage of a given entity — rate limits, or the real adverse
            # event predates the query window (e.g. a 2023 short-seller report seen
            # from a 2026 "today"). When the live feed yields no news-derived
            # narrative signal, fill the gap from the deterministic scenario so the
            # public layer is never empty. These are tagged ("(modeled)" source +
            # meta provenance) so they are never passed off as live data.
            if not any(
                s.signal_type in _NARRATIVE_SIGNAL_TYPES for s in signals
            ):
                # Prefer REAL recent articles (real title + real article URL) so
                # the live entity's signal cards link straight to the source.
                # Only when the live news feed is genuinely empty do we fall back
                # to the deterministic, clearly-labelled modeled narrative.
                real_news = self._live_news_signals(cust)
                if real_news:
                    signals.extend(real_news)
                else:
                    signals.extend(self._modeled_narrative_signals(cust))
            return sorted(signals, key=lambda s: s.month)
        return generate_signals_for_customer(
            cust.drift_id,
            cust.name,
            cust.scenario,
            months=cust.months,
            drift_start_month=cust.drift_start_month,
            # zlib.crc32 (not builtin hash) so the seed is stable across processes
            # and restarts — builtin str hashing is salted per-process by
            # PYTHONHASHSEED, which would make "deterministic" signals vary.
            seed=zlib.crc32(cust.drift_id.encode()) % 10000,
        )

    @staticmethod
    def _live_news_signals(cust: SyntheticCustomer) -> list[PublicSignal]:
        """Real recent-article news signals for a live entity (real title + URL).

        Synchronous bridge (same thread-isolated ``asyncio.run`` pattern as the
        GLEIF loaders) so it is safe under the running FastAPI loop. Best-effort:
        any failure or timeout returns ``[]`` and the caller falls back to the
        modeled narrative. Results are disk-cached by the adapter, so the demo
        replays them offline.
        """
        async def _fetch() -> list[PublicSignal]:
            from app.sources.event_registry import EventRegistryAdapter

            adapter = EventRegistryAdapter()
            return await adapter.fetch_recent_news(cust.drift_id, cust.name)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(asyncio.run, _fetch())
        try:
            return future.result(timeout=_GLEIF_FETCH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — best-effort; fall back to modeled
            logger.warning("live_news_fetch_failed", exc_info=True)
            return []
        finally:
            pool.shutdown(wait=False)

    def _modeled_narrative_signals(
        self, cust: SyntheticCustomer
    ) -> list[PublicSignal]:
        """News-derived narrative signals from the scenario, tagged as modeled.

        Used only as the live-mode hybrid fallback (see ``_public_signals``) when
        the live news feed returns no narrative coverage. Each signal's ``source``
        gets a ``" (modeled)"`` suffix and ``meta["provenance"] = "scenario"`` so
        the UI and any downstream consumer can tell it apart from authoritative
        live data. Registry/screening types are excluded — those only ever come
        from real adapters in live mode.
        """
        modeled = generate_signals_for_customer(
            cust.drift_id,
            cust.name,
            cust.scenario,
            months=cust.months,
            drift_start_month=cust.drift_start_month,
            seed=zlib.crc32(cust.drift_id.encode()) % 10000,
        )
        tagged: list[PublicSignal] = []
        for s in modeled:
            if s.signal_type not in _NARRATIVE_SIGNAL_TYPES:
                continue
            tagged.append(
                replace(
                    s,
                    source=f"{s.source} (modeled)",
                    meta={**(s.meta or {}), "provenance": "scenario"},
                )
            )
        return tagged

    # ------------------------------------------------------------------ #
    # Business-model drift (UC 9)
    # ------------------------------------------------------------------ #
    def _business_model_comparison(
        self, cust: SyntheticCustomer
    ) -> BusinessModelComparison | None:
        """Compare onboarding vs current website text for a silent pivot (UC 9).

        Two acquisition paths, selected by the ``external_apis_enabled`` master
        switch — exactly the same seam as ``_public_signals``:

        - OFFLINE (default): the synthetic demo path. The texts ride on the
          customer (only the ``domain_pivot`` scenario carries them); every other
          scenario returns ``None`` (no signal). No network, no DB.
        - LIVE: source the onboarding "before" text from Wayback and the current
          "after" text from Firecrawl via a bounded sync bridge, with the
          comparator's embeddings persisted to (and re-read from)
          ``EntitySnapshotDB.raw_data`` as a read-through cache. Missing domain,
          adapter error, or an absent embeddings backend all degrade to ``None``.

        Either way the pure comparator never raises and never requires a model
        download at request time — an unavailable backend yields a skipped result.
        """
        if settings.external_apis_enabled or getattr(cust, "mode", "synthetic") == "live":
            return self._live_business_model_comparison(cust)

        onboarding = cust.onboarding_website_text
        current = cust.current_website_text
        if not onboarding or not current:
            return None
        return compare_business_model(
            cust.drift_id,
            cust.name,
            onboarding,
            current,
            # Align the emitted signal with the pivot onset (the WHOIS registrant
            # change) so Confirmation Lift can time-match it to internal drift.
            month=cust.drift_start_month or 0,
        )

    def _live_business_model_comparison(
        self, cust: SyntheticCustomer
    ) -> BusinessModelComparison | None:
        """Synchronous bridge to the async live website-text comparison.

        Mirrors the public-intel / GLEIF sync bridges: runs the async fetch +
        compare in a dedicated thread with its own event loop, so it is safe to
        call from the synchronous engine even under a running FastAPI loop. Bounded
        by ``_WEBSITE_FETCH_TIMEOUT_S``; any failure (timeout included) degrades to
        ``None`` (no signal). ``shutdown(wait=False)`` so a hung worker never
        re-blocks the engine — it unwinds on its own once the inner work is
        abandoned.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(
            asyncio.run,
            self._gather_business_model_async(
                cust.drift_id, cust.name, month=cust.drift_start_month or 0
            ),
        )
        try:
            return future.result(timeout=_WEBSITE_FETCH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — honour the graceful-degradation contract
            logger.warning(
                "business_model_live_fetch_failed", drift_id=cust.drift_id, exc_info=True
            )
            return None
        finally:
            pool.shutdown(wait=False)

    async def _gather_business_model_async(
        self, drift_id: str, name: str, *, month: int
    ) -> BusinessModelComparison | None:
        """Source live website texts, run the comparator, and persist embeddings.

        Pipeline:
          1. Read the customer's domain + any cached embeddings from the latest
             persisted snapshot's ``raw_data`` (the read-through cache). No domain
             → no live wiring (returns ``None``), so this never derives a domain
             from the name slug or hits the network for an unknown customer.
          2. Fetch the onboarding text (Wayback) and current text (Firecrawl).
          3. Build per-side caches from the persisted embeddings (matched on the
             SHA-256 fingerprint of the stripped text) and run the comparator,
             which reuses a cached vector only while its fingerprint still matches.
          4. Persist the embeddings the comparator actually used, but only when
             they changed — a re-scan with unchanged text is a pure cache hit and
             writes nothing.
        """
        domain, cached_map, onboarding_date = await self._load_website_baseline(drift_id)
        if not domain:
            logger.info("business_model_live_no_domain", drift_id=drift_id)
            return None

        # Fetch the archived (Wayback) and current (Firecrawl) pages concurrently
        # — they are independent, so this keeps a cold run well inside the
        # _WEBSITE_FETCH_TIMEOUT_S budget instead of summing both fetches.
        (wayback_text, wayback_url), (current_text, current_url) = await asyncio.gather(
            self._fetch_wayback_text(drift_id, name, domain, onboarding_date),
            self._fetch_firecrawl_text(drift_id, name, domain),
        )

        # Fingerprint the STRIPPED text the comparator embeds, so a persisted
        # vector is reused only when it covers exactly that normalised text.
        wb_fp = text_fingerprint((wayback_text or "").strip())
        cur_fp = text_fingerprint((current_text or "").strip())
        wb_cache = (
            CachedEmbedding(wb_fp, cached_map[wb_fp]) if wb_fp in cached_map else None
        )
        cur_cache = (
            CachedEmbedding(cur_fp, cached_map[cur_fp]) if cur_fp in cached_map else None
        )

        result = compare_business_model(
            drift_id,
            name,
            wayback_text,
            current_text,
            month=month,
            wayback_cache=wb_cache,
            current_cache=cur_cache,
            source_url=current_url,
        )

        # Enrich the result with the archived/live URLs and two really-short
        # before/after LLM summaries so the UC9 panel reads at a glance. Use the
        # human-facing Wayback URL (strip the ``id_`` raw-bytes modifier, which
        # serves unstyled content that does not render as a page in a browser).
        if result.is_change and not result.skipped:
            summary = self._summarize_website_change(
                name, wayback_text or "", current_text or ""
            )
            display_wayback_url = (
                wayback_url.replace("id_/", "/", 1) if wayback_url else None
            )
            result = replace(
                result,
                wayback_url=display_wayback_url,
                current_url=current_url,
                summary=summary,
            )

        # Persist the embeddings actually used (never the degenerate ones a skip
        # withholds). Keep only the two current fingerprints so the cache stays
        # bounded and self-prunes; skip the write when nothing changed.
        if (
            not result.skipped
            and result.wayback_embedding is not None
            and result.current_embedding is not None
        ):
            new_map = {
                result.wayback_embedding.fingerprint: result.wayback_embedding.vector,
                result.current_embedding.fingerprint: result.current_embedding.vector,
            }
            if new_map != cached_map:
                await self._persist_website_embeddings(
                    drift_id=drift_id,
                    name=name,
                    domain=domain,
                    current_text=current_text or "",
                    current_url=current_url,
                    onboarding_date=onboarding_date,
                    embeddings=new_map,
                )
        return result

    async def _load_website_baseline(
        self, drift_id: str
    ) -> tuple[str | None, dict[str, list[float]], str | None]:
        """Read the persisted domain + cached embeddings for one customer.

        Returns ``(domain, embeddings_by_fingerprint, onboarding_date)`` from the
        latest snapshot's ``raw_data``. Degrades to ``(None, {}, None)`` on any DB
        error or when no snapshot (or no domain) exists — the live website path is
        then inert for that customer.
        """
        try:
            async with session_scope() as session:
                latest = await load_latest_snapshot(session, drift_id)
        except Exception:  # noqa: BLE001 — DB unavailable must degrade, not crash
            logger.warning(
                "business_model_baseline_load_failed", drift_id=drift_id, exc_info=True
            )
            return None, {}, None
        if latest is None:
            return None, {}, None
        raw = latest.raw_data or {}
        embeddings = raw.get(_BUSINESS_MODEL_EMBEDDINGS_KEY) or {}
        return raw.get("domain"), embeddings, raw.get("onboarding_date")

    async def _persist_website_embeddings(
        self,
        *,
        drift_id: str,
        name: str,
        domain: str,
        current_text: str,
        current_url: str | None,
        onboarding_date: str | None,
        embeddings: dict[str, list[float]],
    ) -> None:
        """Append a Firecrawl snapshot carrying the comparator's embeddings.

        Append-only, matching the snapshot store's contract: a fresh capture is a
        new row, and ``load_latest_snapshot`` returns it on the next scan so the
        read-through cache hits. Best-effort — a persistence failure must not sink
        the comparison the caller already computed.
        """
        try:
            async with session_scope() as session:
                await store_snapshot(
                    session,
                    EntitySnapshotDB(
                        drift_id=drift_id,
                        snapshot_date=date.today(),
                        snapshot_type="triggered",
                        source="firecrawl",
                        name=name,
                        raw_data={
                            "domain": domain,
                            "url": current_url,
                            "website_text": current_text,
                            "onboarding_date": onboarding_date,
                            "scraped_at": datetime.now(UTC).isoformat(),
                            _BUSINESS_MODEL_EMBEDDINGS_KEY: embeddings,
                        },
                    ),
                )
        except Exception:  # noqa: BLE001 — persistence is best-effort
            logger.warning(
                "business_model_embeddings_persist_failed",
                drift_id=drift_id,
                exc_info=True,
            )

    async def _fetch_wayback_text(
        self, drift_id: str, name: str, domain: str, onboarding_date: str | None
    ) -> tuple[str | None, str | None]:
        """Fetch onboarding website text + snapshot URL via Wayback. (None, None) on error."""
        adapter = WaybackAdapter()
        try:
            snap = await adapter.fetch(
                drift_id, name, domain=domain, onboarding_date=onboarding_date
            )
        except Exception:  # noqa: BLE001 — a Wayback outage degrades to no signal
            logger.warning(
                "business_model_wayback_failed", drift_id=drift_id, exc_info=True
            )
            return None, None
        finally:
            await self._safe_aclose(adapter)
        if snap is None:
            return None, None
        return snap.raw_data.get("website_text"), snap.raw_data.get("snapshot_url")

    async def _fetch_firecrawl_text(
        self, drift_id: str, name: str, domain: str
    ) -> tuple[str | None, str | None]:
        """Fetch the current website text + URL via Firecrawl. (None, None) on error."""
        adapter = FirecrawlAdapter()
        try:
            snap = await adapter.fetch(drift_id, name, domain=domain)
        except Exception:  # noqa: BLE001 — a scrape failure degrades to no signal
            logger.warning(
                "business_model_firecrawl_failed", drift_id=drift_id, exc_info=True
            )
            return None, None
        finally:
            await self._safe_aclose(adapter)
        if snap is None:
            return None, None
        return snap.raw_data.get("website_text"), snap.raw_data.get("url")

    @staticmethod
    def _summarize_website_change(name: str, before: str, after: str) -> str | None:
        """One really-short LLM summary of how the website changed (the diff).

        A single short sentence (~max 16 words). One LLM call, disk-cached, so it
        costs tokens once per entity and replays offline. ``None`` on any failure.
        """
        try:
            llm = get_anthropic_client()
            prompt = (
                f"Company: {name}\n\n"
                f"WEBSITE AT ONBOARDING (excerpt):\n{before[:1500]}\n\n"
                f"WEBSITE NOW (excerpt):\n{after[:1500]}\n\n"
                "In ONE short sentence (max 16 words), say what changed between "
                "the onboarding website and the current one. If nothing "
                "substantive changed, answer exactly: 'Minor wording changes only.'"
            )
            text, _, _ = llm.complete(prompt, max_tokens=70)
            return text.strip()[:200] or None
        except Exception:  # noqa: BLE001 — summary is best-effort
            logger.warning("business_model_summary_failed", exc_info=True)
            return None

    @staticmethod
    async def _safe_aclose(adapter: Any) -> None:
        """Close an adapter's HTTP client if it owns one; never raise."""
        aclose = getattr(adapter, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except Exception:  # noqa: BLE001
            logger.debug("business_model_adapter_aclose_failed", exc_info=True)

    # ------------------------------------------------------------------ #
    # Core per-customer analysis
    # ------------------------------------------------------------------ #
    def _analyze_customer(self, cust: SyntheticCustomer) -> dict:
        """Run all passive layers for one customer. Returns raw signals.

        Two explicit layers, matching the AMINA Challenge 4 architecture:
          - PUBLIC INTELLIGENCE: external signals (news, sanctions, adverse
            media, ownership changes, funding events) -> public_risk
          - INTERNAL BANK DATA: BOCPD drift, velocity, ownership contagion
            -> internal_risk
        The two are fused, then amplified by Confirmation Lift when an
        external signal co-occurs in time with internal drift.

        The passive-layer computation is shared with the offline training
        pipeline via the module-level ``compute_drift_analysis``. This method
        layers the live inputs (contagion graph, public adapters) plus the two
        scoring policies that training does not need — the XGBoost blend and the
        regulatory floors — on top of that shared analysis.
        """
        # PUBLIC: real adapter signals via aggregator (live), or the deterministic
        # synthetic generator (offline/mock). See _public_signals.
        public_signals = self._public_signals(cust)

        # Business-model drift (UC 9): fold a website/domain pivot into the public
        # layer so it lifts public_risk and time-aligns for Confirmation Lift,
        # exactly like any other external signal. Absent texts / embedder → no
        # signal, no effect.
        bm = self._business_model_comparison(cust)
        if bm is not None and bm.signal is not None:
            public_signals = [*public_signals, bm.signal]
        # Surface the real live website texts (Wayback onboarding vs Firecrawl
        # current) so the UC9 side-by-side diff panel renders them. For the
        # synthetic domain_pivot entity these are already set on the customer.
        if bm is not None and bm.wayback_text and bm.current_text:
            cust.onboarding_website_text = bm.wayback_text
            cust.current_website_text = bm.current_text

        # --- Shared passive-layer analysis (identical formula to training) ---
        analysis = compute_drift_analysis(
            cust,
            cohort_cv=self._cohort_cv,
            propagated_risk=self._contagion.propagated_risk.get(cust.drift_id, 0.0),
            public_signals=public_signals,
        )
        analysis["is_business_model_change"] = bm.is_change if bm is not None else False
        analysis["business_model_distance"] = (
            round(bm.distance, 4) if bm is not None else 0.0
        )
        analysis["onboarding_website_url"] = bm.wayback_url if bm is not None else None
        analysis["current_website_url"] = bm.current_url if bm is not None else None
        analysis["business_model_summary"] = bm.summary if bm is not None else None
        score = analysis["drift_score"]

        # XGBoost ML blend — applied BEFORE the regulatory floors below.
        # The floors enforce "cannot hide below the radar" invariants for
        # suspicious-stability and dormancy-break customers; blending after
        # the floors would silently lower the floored score when the ML model
        # disagrees, violating those invariants.
        ml_score: float | None = None
        if self._drift_model is not None:
            try:
                features = self._drift_extractor.extract(analysis)
                feat_df = self._drift_extractor.to_dataframe(features)
                ml_prob = float(self._drift_model.model.predict_proba(feat_df)[0][1])
                ml_score = ml_prob * 100.0
                score = 0.60 * score + 0.40 * ml_score
            except Exception:
                # ML blend is best-effort — fall through to the heuristic score,
                # but make the degradation visible: a silent failure would leave
                # operators on heuristic-only scores with no signal that the ML
                # layer stopped contributing.
                self._ml_blend_failure_count += 1
                log_fn = (
                    logger.error
                    if self._ml_blend_failure_count >= 3
                    else logger.warning
                )
                log_fn(
                    "drift_ml_blend_failed",
                    drift_id=cust.drift_id,
                    consecutive_failures=self._ml_blend_failure_count,
                    exc_info=True,
                )

        # Suspicious-stability ELEVATION — the slow-walker keeps drift low ON
        # PURPOSE, so it would otherwise slip through with a near-zero score.
        # When suspicion is high we floor the score upward: a flagged
        # slow-walker cannot hide below the radar.
        stability = analysis["stability"]
        if stability.is_suspicious:
            score = max(score, 50.0 + stability.suspicion * 40.0)

        # Dormancy-break ELEVATION — a reactivated sleeper starts from a quiet
        # baseline, so drift/velocity under-react. When a genuine dormant->active
        # burst is detected, floor the score upward so it surfaces for review.
        # NOTE: this floor is applied AFTER the causal demotion above and will
        # override it on purpose — a reactivated shell must surface even if the
        # causal layer reads the new activity as (so far) benign-shaped. This is
        # the same "cannot hide below the radar" policy as the stability floor.
        dormancy = analysis["dormancy"]
        if dormancy.is_dormancy_break:
            score = max(score, 55.0 + dormancy.dormancy_break * 35.0)

        # Structural re-KYC ELEVATION (UC 4) — a confirmed jurisdiction or
        # legal-form change is a mandatory re-KYC trigger under structural-risk
        # rules, so floor the score at RE_KYC_SCORE_FLOOR however weak the
        # behavioral signal is. Same "cannot hide below the radar" policy as the
        # stability/dormancy floors above, applied to registry-sourced structural
        # change rather than transaction behaviour. Applied AFTER the ML blend so
        # the model can never lower a regulatory floor (see the note above it).
        if requires_re_kyc_floor(analysis["public_signals"]):
            score = max(score, RE_KYC_SCORE_FLOOR)

        # Sanctions ELEVATION (UC 5 / UC 8) — a confirmed sanctions match on the
        # entity or a newly-added UBO is a mandatory escalation. Floor at
        # SANCTIONS_SCORE_FLOOR so a sanctioned entity with a deliberately clean
        # transaction profile (the ownership_shift scenario) still surfaces as
        # critical. Same "cannot hide below the radar" policy as the floors above.
        if is_definitively_sanctioned(analysis["public_signals"]):
            score = max(score, SANCTIONS_SCORE_FLOOR)

        # Name-change ELEVATION (UC8) — a confirmed legal-entity name change is a
        # mandatory re-KYC trigger. The ZEFIX/GLEIF/WHOIS diffs (live adapters)
        # or the name_cycling scenario (offline) surface a `name_change` public
        # signal. Floor the score at 60 regardless of other signals: an identity
        # reset must surface for review even when the transactions look clean —
        # the Mossack Fonseca shelf-cycling pattern of renaming shell companies
        # to reset the KYC review clock. Same "cannot hide below the radar"
        # policy as the stability and dormancy floors above, and applied last so
        # it cannot be undercut by the causal demotion or a low ML blend.
        name_changed = any(
            s.signal_type == "name_change" for s in analysis["public_signals"]
        )
        if name_changed:
            score = max(score, 60.0)

        analysis["name_changed"] = name_changed
        analysis["drift_score"] = score
        analysis["ml_score"] = ml_score
        return analysis

    def _cached_analysis(self, cust: SyntheticCustomer) -> dict:
        """``_analyze_customer``, memoised per drift_id for live entities only.

        Used by the request paths only; ``_analyze_customer`` itself stays
        uncached so callers that change inputs between calls see fresh results.
        Callers must treat the returned dict as read-only.
        """
        if getattr(cust, "mode", "synthetic") != "live":
            return self._analyze_customer(cust)
        now = time.monotonic()
        hit = self._analysis_cache.get(cust.drift_id)
        if hit is not None and now - hit[0] < self._ANALYSIS_CACHE_TTL:
            return hit[1]
        analysis = self._analyze_customer(cust)
        self._analysis_cache[cust.drift_id] = (time.monotonic(), analysis)
        return analysis

    def _build_layers(self, cust: SyntheticCustomer, analysis: dict) -> list[LayerContribution]:
        """Construct explainable per-layer contributions."""
        prop = analysis["propagated_risk"]
        max_vel = analysis["max_velocity"]
        final_drift = analysis["final_drift"]
        cp = analysis["bocpd_changepoint_day"]

        layers = [
            LayerContribution(
                layer=1, name="Deterministic (sanctions/PEP)",
                llr=0.0, status="ok",
                detail="No direct watchlist match",
            ),
            LayerContribution(
                layer=2, name="Public intelligence",
                llr=round(analysis["public_risk"] * 5, 2),
                status="deviation" if analysis["public_risk"] > 0.4 else (
                    "notable" if analysis["public_risk"] > 0.2 else "ok"
                ),
                detail=(
                    f"{len(analysis['public_signals'])} external signal(s), "
                    f"public risk {analysis['public_risk']:.2f}"
                    + (
                        f"; confirms internal drift (lift {analysis['confirmation_lift']:.1f}x)"
                        if analysis["confirmation_lift"] > 1.5 else ""
                    )
                    if analysis["public_signals"]
                    else "No external signals"
                ),
            ),
            LayerContribution(
                layer=3, name="Ownership contagion",
                llr=round(prop * 5, 2),
                status="deviation" if prop > 0.1 else "ok",
                detail=(
                    f"Propagated risk {prop:.2f} from sanctioned entity "
                    f"({self._contagion.hops_from_seed.get(cust.drift_id, '-')} hops)"
                    if prop > 0.01 else "No ownership path to flagged entities"
                ),
            ),
            LayerContribution(
                layer=4, name="Behavioral drift (BOCPD)",
                llr=round(min(max_vel, 5.0), 2),
                status="deviation" if cp is not None else "ok",
                detail=(
                    f"Regime change detected at day {cp}"
                    if cp is not None else "No regime change in transaction stream"
                ),
            ),
            LayerContribution(
                layer=5, name="Declared consistency / velocity",
                llr=round(min(final_drift / 5, 5.0), 2),
                status=velocity_band(max_vel) if max_vel > 0.3 else "ok",
                detail=f"Drift velocity peaked at {max_vel:.2f} bits/month",
            ),
        ]
        return layers

    def _build_t2_adjudication_prompt(
        self,
        cust: SyntheticCustomer,
        analysis: dict,
    ) -> str:
        """Build a strict JSON-only adjudication prompt for T2 cases."""
        causal = analysis["causal"]
        stability = analysis["stability"]
        dormancy = analysis["dormancy"]
        signature = causal.signature
        context: dict[str, Any] = {
            "customer": {
                "id": cust.drift_id,
                "name": cust.name,
                "scenario": cust.scenario,
            },
            "risk_scores": {
                "drift_score": round(analysis["drift_score"], 3),
                "internal_risk": round(analysis["internal_risk"], 3),
                "public_risk": round(analysis["public_risk"], 3),
                "confirmation_lift": round(analysis["confirmation_lift"], 3),
                "propagated_risk": round(analysis["propagated_risk"], 3),
            },
            "causal_assessment": {
                "label": causal.label,
                "p_risk": round(causal.p_risk, 3),
                "causal_likelihood_ratio": round(causal.causal_llr, 3),
                "contributions": {
                    k: round(v, 3) for k, v in causal.contributions.items()
                },
            },
            "drift_signature": {
                "volume_change": round(signature.volume_change, 3),
                "margin_change": round(signature.margin_change, 3),
                "counterparty_risk_change": round(signature.counterparty_change, 3),
                "corridor_risk_change": round(signature.corridor_change, 3),
            },
            "suspicious_stability": {
                "is_suspicious": stability.is_suspicious,
                "score": round(stability.suspicion, 3),
                "detail": stability.detail,
            },
            "dormancy_break": {
                "is_dormancy_break": dormancy.is_dormancy_break,
                "score": round(dormancy.dormancy_break, 3),
                "baseline_volume": round(dormancy.baseline_volume, 1),
                "active_volume": round(dormancy.active_volume, 1),
                "detail": dormancy.detail,
            },
            "public_signals": [
                {
                    "type": signal.signal_type,
                    "headline": signal.headline,
                    "severity": round(signal.severity, 3),
                    "source": signal.source,
                    "month": signal.month,
                }
                for signal in analysis["public_signals"]
            ],
        }

        return (
            "Adjudicate this T2 KYC drift case by comparing three hypotheses:\n"
            "1. Risk-shaped or causal drift hypothesis.\n"
            "2. Benign business-change hypothesis.\n"
            "3. Ambiguous or insufficient-evidence hypothesis.\n\n"
            "Use only the structured evidence below. Do not invent facts. "
            "Do not recommend automatic blocking; recommend human compliance "
            "actions such as enhanced due diligence, request for information, "
            "monitoring, or no immediate action.\n\n"
            "Structured evidence:\n"
            f"{json.dumps(context, indent=2, sort_keys=True)}\n\n"
            "Return JSON only with this exact shape:\n"
            "{\n"
            '  "verdict": "risk" | "benign" | "ambiguous",\n'
            '  "confidence": number,\n'
            '  "rationale": string,\n'
            '  "key_evidence": string[],\n'
            '  "recommended_action": string\n'
            "}"
        )

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
        """Parse and normalize a T2 adjudication JSON response defensively."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return dict(LLM_PARSE_FALLBACK)
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return dict(LLM_PARSE_FALLBACK)

        if not isinstance(payload, dict):
            return dict(LLM_PARSE_FALLBACK)

        verdict = payload.get("verdict")
        if verdict not in {"risk", "benign", "ambiguous"}:
            verdict = "ambiguous"

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        key_evidence = payload.get("key_evidence", [])
        if not isinstance(key_evidence, list):
            key_evidence = []
        key_evidence = [str(item) for item in key_evidence]

        rationale = payload.get("rationale", "")
        recommended_action = payload.get("recommended_action", "Request information")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "rationale": str(rationale),
            "key_evidence": key_evidence,
            "recommended_action": str(recommended_action),
        }

    def _run_t2_llm_adjudication(
        self,
        cust: SyntheticCustomer,
        analysis: dict,
    ) -> dict[str, Any]:
        """Execute the real or mock Anthropic T2 adjudication path."""
        llm = get_anthropic_client()
        llm_mode = "mock" if llm.is_mock else "real"
        text, was_cached, tokens_used = llm.complete(
            self._build_t2_adjudication_prompt(cust, analysis),
            system=T2_LLM_SYSTEM_MESSAGE,
            max_tokens=700,
        )

        return {
            "drift_id": cust.drift_id,
            "drift_name": cust.name,
            "llm_mode": llm_mode,
            "was_cached": was_cached,
            "tokens_used": tokens_used,
            "response": self._parse_llm_json(text),
        }

    # ------------------------------------------------------------------ #
    # Public API methods
    # ------------------------------------------------------------------ #
    def list_subjects(self) -> list[DriftSubjectSummary]:
        now = time.monotonic()
        if self._list_cache is not None and now - self._list_cache_at < self._LIST_CACHE_TTL:
            return list(self._list_cache)

        out: list[DriftSubjectSummary] = []
        for cust in self._book:
            a = self._cached_analysis(cust)
            signal = CustomerSignal(
                drift_id=cust.drift_id,
                drift_score=a["drift_score"],
                propagated_risk=a["propagated_risk"],
            )
            decision = self._router.route_one(signal)
            out.append(
                DriftSubjectSummary(
                    drift_id=cust.drift_id,
                    name=cust.name,
                    drift_score=round(a["drift_score"], 1),
                    drift_velocity=round(a["max_velocity"], 3),
                    velocity_band=velocity_band(a["max_velocity"]),
                    reached_tier=decision.reached_tier.name,
                    sanctions_hit=False,
                    propagated_risk=round(a["propagated_risk"], 3),
                    public_risk=round(a["public_risk"], 3),
                    confirmation_lift=round(a["confirmation_lift"], 2),
                    causal_label=a["causal"].label,
                    causal_p_risk=round(a["causal"].p_risk, 3),
                    suspicion=round(a["stability"].suspicion, 3),
                    is_suspicious=a["stability"].is_suspicious,
                    dormancy_break=round(a["dormancy"].dormancy_break, 3),
                    is_dormancy_break=a["dormancy"].is_dormancy_break,
                    is_name_changed=a["name_changed"],
                    risk_level=score_to_level(a["drift_score"]).value,
                    scenario=cust.scenario,
                    mode=getattr(cust, "mode", "synthetic"),
                )
            )
        out.sort(key=lambda c: c.drift_score, reverse=True)
        self._list_cache = out
        self._list_cache_at = time.monotonic()
        return list(out)

    def get_subject(self, drift_id: str) -> DriftSubjectDetail | None:
        cust = next((c for c in self._book if c.drift_id == drift_id), None)
        if cust is None:
            return None
        a = self._cached_analysis(cust)
        ds = a["drift_series"]

        signal = CustomerSignal(
            drift_id=cust.drift_id,
            drift_score=a["drift_score"],
            propagated_risk=a["propagated_risk"],
        )
        decision = self._router.route_one(signal)
        recommended_action = recommend_drift_action(
            a["drift_score"],
            a["causal"].label,
            a["stability"].is_suspicious,
        )

        # Mark the timeline point at the BOCPD changepoint month (mapped from a
        # day index in _analyze_customer). A changepoint that lands in the
        # baseline window — before the first timeline point — matches no point
        # and is correctly left unmarked, as is the no-changepoint (None) case.
        cp_month = a["bocpd_changepoint_month"]

        timeline = [
            DriftTimelinePoint(
                month=ds.windows[i],
                drift_bits=round(ds.drift_bits[i], 3),
                velocity=round(ds.velocity[i], 3),
                acceleration=round(ds.acceleration[i], 3),
                bocpd_changepoint=ds.windows[i] == cp_month,
            )
            for i in range(len(ds.windows))
        ]

        # Case 5: UBO / ownership-chain sanctions hits. The OpenSanctions adapter
        # tags each screened-UBO signal with structured ``meta`` (screened name,
        # matched watchlist entity, score) so we surface them without re-parsing
        # the headline. Synthetic ``ownership_change`` signals carry no such meta
        # and are correctly excluded.
        ubo_screening = [
            UboScreeningOut(
                screened_ubo=s.meta["ubo_name"],
                matched_entity=s.meta["matched_entity"],
                score=s.meta["score"],
                severity=round(s.severity, 3),
                month=s.month,
                definitive=bool(s.meta.get("definitive", False)),
                source_url=s.source_url,
            )
            for s in a["public_signals"]
            if s.meta and s.meta.get("kind") == "ubo_screening"
        ]

        return DriftSubjectDetail(
            drift_id=cust.drift_id,
            name=cust.name,
            drift_score=round(a["drift_score"], 1),
            drift_velocity=round(a["max_velocity"], 3),
            velocity_band=velocity_band(a["max_velocity"]),
            reached_tier=decision.reached_tier.name,
            recommended_action=recommended_action,
            risk_level=score_to_level(a["drift_score"]),
            escalation_reasons=decision.escalation_reasons,
            layers=self._build_layers(cust, a),
            timeline=timeline,
            scenario=cust.scenario,
            mode=getattr(cust, "mode", "synthetic"),
            drift_start_month=cust.drift_start_month,
            sanctions_month=cust.sanctions_month,
            bocpd_changepoint_day=a["bocpd_changepoint_day"],
            news_spike_month=a["news_spike_month"],
            public_risk=round(a["public_risk"], 3),
            internal_risk=round(a["internal_risk"], 3),
            confirmation_lift=round(a["confirmation_lift"], 2),
            public_signals=[
                PublicSignalOut(**s.to_dict()) for s in a["public_signals"]
            ],
            ubo_screening=ubo_screening,
            is_name_changed=a["name_changed"],
            is_business_model_change=a.get("is_business_model_change", False),
            business_model_distance=a.get("business_model_distance", 0.0),
            # The raw crawled website text is intentionally NOT returned — it is
            # long/noisy and the UI shows only the LLM summary + the two links.
            onboarding_website_text=None,
            current_website_text=None,
            onboarding_website_url=a.get("onboarding_website_url"),
            current_website_url=a.get("current_website_url"),
            business_model_summary=a.get("business_model_summary"),
            causal=CausalVerdictOut(
                causal_llr=round(a["causal"].causal_llr, 2),
                p_risk=round(a["causal"].p_risk, 3),
                label=a["causal"].label,
                volume_change=round(a["causal"].signature.volume_change, 3),
                margin_change=round(a["causal"].signature.margin_change, 3),
                counterparty_change=round(a["causal"].signature.counterparty_change, 3),
                corridor_change=round(a["causal"].signature.corridor_change, 3),
                contributions={k: round(v, 2) for k, v in a["causal"].contributions.items()},
            ),
            stability=StabilityOut(
                suspicion=round(a["stability"].suspicion, 3),
                stability_anomaly=round(a["stability"].stability_anomaly, 3),
                environmental_movement=round(a["stability"].environmental_movement, 3),
                own_volatility=round(a["stability"].own_volatility, 4),
                cohort_volatility=round(a["stability"].cohort_volatility, 4),
                is_suspicious=a["stability"].is_suspicious,
                detail=a["stability"].detail,
            ),
            dormancy=DormancyOut(
                dormancy_break=round(a["dormancy"].dormancy_break, 3),
                dormancy_depth=round(a["dormancy"].dormancy_depth, 3),
                activation_strength=round(a["dormancy"].activation_strength, 3),
                baseline_volume=round(a["dormancy"].baseline_volume, 1),
                active_volume=round(a["dormancy"].active_volume, 1),
                is_dormancy_break=a["dormancy"].is_dormancy_break,
                detail=a["dormancy"].detail,
            ),
        )

    def scan(self) -> CascadeCostReport:
        """Run full cascade over the book, return cost report."""
        signals = []
        analyses: dict[str, tuple[SyntheticCustomer, dict]] = {}
        for cust in self._book:
            a = self._cached_analysis(cust)
            analyses[cust.drift_id] = (cust, a)
            signals.append(
                CustomerSignal(
                    drift_id=cust.drift_id,
                    drift_score=a["drift_score"],
                    propagated_risk=a["propagated_risk"],
                )
            )
        report = self._router.route_book(signals)
        llm_adjudications = []
        for decision in report.decisions:
            if decision.reached_tier != Tier.T2_LLM:
                continue
            cust, analysis = analyses[decision.drift_id]
            llm_adjudications.append(
                self._run_t2_llm_adjudication(cust, analysis)
            )

        actual_t2_llm_calls = len(llm_adjudications)
        real_t2_llm_calls = sum(
            1 for item in llm_adjudications
            if item["llm_mode"] == "real" and not item["was_cached"]
        )
        mock_t2_llm_calls = sum(
            1 for item in llm_adjudications if item["llm_mode"] == "mock"
        )
        total_tokens_used = sum(item["tokens_used"] for item in llm_adjudications)
        # Model reflects current config; set only when at least one uncached real call was made.
        adjudication_model = settings.anthropic_model_main if real_t2_llm_calls > 0 else None
        llm_all = len(signals) * 0.05
        savings = 100.0 * (1 - report.total_cost / llm_all) if llm_all > 0 else 0.0
        summary = (
            f"{report.summary_line()}. Actual T2 LLM adjudications: "
            f"{actual_t2_llm_calls} total, {real_t2_llm_calls} real, "
            f"{mock_t2_llm_calls} mock."
        )
        return CascadeCostReport(
            total_customers=report.total_customers,
            tier_counts=report.tier_counts,
            tier_costs={k: round(v, 4) for k, v in report.tier_costs.items()},
            total_cost=round(report.total_cost, 2),
            summary=summary,
            llm_on_everything_cost=round(llm_all, 2),
            savings_pct=round(savings, 1),
            actual_t2_llm_calls=actual_t2_llm_calls,
            real_t2_llm_calls=real_t2_llm_calls,
            mock_t2_llm_calls=mock_t2_llm_calls,
            tokens_used=total_tokens_used,
            model=adjudication_model,
            llm_adjudications=llm_adjudications,
        )

    def contagion_graph(self) -> ContagionGraph:
        data = self._graph.to_cytoscape(self._contagion)
        return ContagionGraph(
            nodes=data["nodes"],
            edges=data["edges"],
            seeds=self._contagion.seeds,
        )

    def replay(self, drift_id: str) -> ReplayResult | None:
        """Time-Travel Audit: as-of replay proving no look-ahead bias."""
        cust = next((c for c in self._book if c.drift_id == drift_id), None)
        if cust is None:
            return None

        prop = self._contagion.propagated_risk.get(drift_id, 0.0)
        # The seed entity is sanctioned at the customer's sanctions_month;
        # before that, contagion risk does not exist (no look-ahead).
        listing_month = cust.sanctions_month

        traj = replay_trajectory(
            cust,
            propagated_risk_final=prop,
            contagion_listing_month=listing_month,
        )

        return ReplayResult(
            drift_id=cust.drift_id,
            name=cust.name,
            points=[
                AsOfPointOut(
                    month=p.month,
                    as_of_score=p.as_of_score,
                    velocity=p.velocity,
                    public_risk=p.public_risk,
                    contagion_active=p.contagion_active,
                    causal_p_risk=p.causal_p_risk,
                )
                for p in traj["points"]
            ],
            alert_month=traj["alert_month"],
            sanctions_month=traj["sanctions_month"],
            lead_time_months=traj["lead_time_months"],
            alert_threshold=traj["alert_threshold"],
        )

    def inject_scenario(self, scenario: str, name: str) -> DriftSubjectDetail:
        """Red-team: add a synthetic customer with a chosen drift scenario."""
        new_id = f"injected-{len(self._book) + 1:03d}"
        cust = generate_customer(
            drift_id=new_id, name=name, scenario=scenario,
            # zlib.crc32, not builtin hash — same reasoning as _public_signals:
            # PYTHONHASHSEED randomises str hashing per-process, so injected
            # scenarios would produce different customers across restarts.
            seed=zlib.crc32(new_id.encode()) % 10000,
        )
        self._book.append(cust)
        detail = self.get_subject(new_id)
        assert detail is not None
        return detail


# Process-local singleton: mutable demo state (injected scenarios) is not shared
# across worker processes. Deploy this MVP with exactly one API worker; replace
# the in-memory state with a shared store before scaling out.
_engine: DriftEngine | None = None


def get_drift_engine() -> DriftEngine:
    global _engine
    if _engine is None:
        logger.warning(
            "drift_engine_single_worker_required",
            reason=(
                "DriftEngine uses process-local mutable state; configure exactly "
                "one API worker until state is moved to a shared store."
            ),
        )
        _engine = DriftEngine()
    return _engine
