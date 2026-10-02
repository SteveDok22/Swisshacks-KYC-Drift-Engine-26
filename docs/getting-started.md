# Getting Started

Run the demo, configure modes, and know what to look at first.

## Prerequisites

Just **Docker** (with Compose). Nothing else is needed for the offline demo — no
Python, Node, or API keys.

## Run

```bash
docker compose up --build          # backend :8001 · frontend :3000
```

- App: **http://localhost:3000**
- API docs (Swagger): **http://localhost:8001/docs**

> Need a **public URL** to show the demo from your laptop (no deploy)? See
> [public-demo.md](public-demo.md) — a one-command tunnel.

The default is the **offline synthetic demo** — 15 deterministic Swiss entities
plus 5 live entities that replay from committed response caches. No keys required.

## Compose services

| Service | Port | Purpose |
|---|---|---|
| `backend` | 8001 (8000 inside Docker) | FastAPI app (uvicorn, hot reload) |
| `frontend` | 3000 | Next.js dev server, proxies `/api/backend/*` → backend |
| `backend-tests` | — | pytest suite (profile `test`; not started by `up`) |

Backend and frontend source are **bind-mounted**, so edits hot-reload. SQLite is
**disposable**: the schema is dropped, recreated, and reseeded on every backend
start (KYC baselines, 10 clients + 19 cases, and a ~97-entry audit trail).

## Tests

```bash
docker compose run --rm backend-tests
```

Runs the full suite in an in-memory SQLite DB. The disk API cache is auto-disabled
under pytest so adapter tests see their mocks.

## Modes & configuration

The platform has **two modes from one codebase**:

- **Offline synthetic (default)** — deterministic mock signals, zero outbound
  calls, no keys. This is what you get out of the box.
- **Live** — real external-API calls. Toggled two ways:
  - `EXTERNAL_APIS_ENABLED` (default `false`) — global switch for *all* subjects.
  - **Per-entity `mode="live"`** — the 5 live entities fire real adapters
    regardless of the global switch (their responses are cached, see below).

Keys are read from `backend/.env.shared` (team hackathon keys, loaded by
`docker-compose`), overridable by a personal, gitignored `backend/.env`:

| Variable | Used by | Needed? |
|---|---|---|
| `ANTHROPIC_API_KEY` | T2 LLM adjudication, UC9 website-diff summary | for live LLM |
| `EVENT_REGISTRY_API_KEY` | news (Event Registry) | optional (GDELT is the free fallback) |
| `OPENSANCTIONS_API_KEY` | sanctions screening | optional (free tier works) |
| `FIRECRAWL_API_KEY` | current website scrape (UC9) | optional (plain-HTTP fallback) |

GLEIF, GDELT, Wayback, and WHOIS/RDAP need **no key**. ZEFIX needs a free
registered account and **degrades gracefully** without one. Set
`ANTHROPIC_FORCE_MOCK=1` to pin the LLM to deterministic mock responses.

## Disk API cache (offline live mode)

Real adapter responses **and** LLM completions are written to
`backend/data/api_cache/{service}/` and **committed to the repo**, so the live
entities replay fully offline and for free. Write-through: a cache miss makes one
live call, saves it, and the next run is instant.

To (re)populate after changing a live entity:

```bash
# open each live entity's detail in the UI, or hit the endpoint twice:
for id in drift-live-001 drift-live-002 drift-live-003 drift-live-004 drift-live-005; do
  curl -s "http://localhost:8001/api/v1/drift/subjects/$id" -o /dev/null
done
git add backend/data/api_cache/        # commit the populated caches
```

Two passes warm both the disk cache and the engine's in-memory analysis cache.
Details: [live-entities.md](live-entities.md).

## What to look at first

1. **Drift workspace** (`/`) — a radar of all **20 subjects** by score × velocity.
   The upper-right quadrant is the priority queue.
2. **Castor Trade Finance AG** (`drift-011`) — the flagship synthetic case:
   structuring + a new sanctioned UBO + adverse media, routed to T2 Claude.
3. **Live entities** —
   - **Temenos AG** (`drift-live-001`) for **UC9 website-drift**: a real Wayback
     (onboarding) ↔ Firecrawl (current) diff with a one-line AI summary and links
     to both versions.
   - **Rosneft Trading S.A.** (`drift-live-002`) for a **real OpenSanctions** hit
     with a clickable `opensanctions.org` entity link.
4. **Case Queue** (`/cases`) — 10 clients, 19 compliance cases.
5. **Audit log** (`/audit`) — ~97 seeded, backdated compliance events (named
   officers, decisions, scans, RFIs); filter by actor, event type, risk, or date.

See [use-cases.md](use-cases.md) for the full entity ↔ use-case map and
[architecture.md](architecture.md) for how the pieces fit together.
