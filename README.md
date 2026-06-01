# Autonomous Web Research Agent

A local research assistant built with FastAPI, LangGraph, and Ollama. Search the
web via Tavily + DuckDuckGo, deep-read pages with BeautifulSoup, verify each
draft answer against the retrieved evidence, and present results with source
tiers and trust signals — all running on your own machine.

## Features

- **Streaming UI** (`/`) — sidebar with chat history & search, model selector,
  light/dark theme, live trace of every tool call, and a stop button that
  cancels the active search.
- **Three local models** via Ollama, switchable per-prompt:
  - Llama 3.1 (8B) — default
  - Qwen 2.5 (7B)
  - Llama 3.2 (3B) — faster, lower quality
- **Source reliability scoring** — sources are classified as `high` / `medium` /
  `low` / `prediction`, then assigned a 0-1 reliability score using curated
  domains, official-domain heuristics, HTTPS, sparse metadata, self-publishing,
  speculative wording, and betting/prediction signals. Prediction-tier sources
  are explicitly excluded from factual answers.
- **Reliability-aware ranking** — search results are ordered by topical
  relevance first, then source reliability, with recent dated sources boosted
  for current/latest questions.
- **Source-directed lookup retries** — when generic search does not surface a
  strong source, the tools retry with authoritative source profiles for market
  data, weather, sports, official statistics, software docs, health, and
  science questions.
- **Structured market data lookup** — finance questions can resolve tickers
  and retrieve dated daily close data from Yahoo Finance before generic snippets.
- **Evidence extraction pass** — after the draft answer is generated, a
  deterministic Python pass maps claims to retrieved sources and flags weak or
  unsupported evidence.
- **Extra inference passes** — before retrieval, a query-understanding LLM pass
  classifies intent and answer shape without generating search queries; after
  extraction, a verifier/critic LLM pass revises unsupported, over-cautious, or
  conflicting answers against the retrieved evidence.
- **Rank-aware leaderboard answers** — ordinal questions such as "3rd top
  scorer" are parsed as exact-rank requests, and tool output surfaces
  rank-specific direct hints instead of only the overall leader. The API also
  retries the exact original rank question before verification, so the verifier
  sees the requested row even when the model's first search drifted.
- **Continuous trust signals** — `source_quality_score`, `source_reliability_score`,
  `citation_strength`, `citation_coverage`, `multi_source_claim_ratio`, plus a
  tier breakdown chart.
- **Conversational memory** — last 5 turns of a chat are passed back into the
  agent so follow-up questions resolve correctly.
- **24-hour result cache** — repeated questions return instantly (skipped when
  prior conversation context is present).

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- A free [Tavily API key](https://tavily.com/) (for the high-quality search tool)
- The three models pulled into Ollama:

```bash
ollama pull llama3.1
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
```

You can run with fewer models by editing `MODEL_REGISTRY` in `config.py`.

## Setup

1. Clone the project and `cd` into it.
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # PowerShell
   .\venv\Scripts\Activate.ps1
   # CMD
   .\venv\Scripts\activate.bat
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` (copy `.env.example`) and add your Tavily key:

   ```env
   TAVILY_API_KEY=tvly-...
   ```

## Run

In one terminal, start Ollama:

```bash
ollama serve
```

In another terminal, start the API:

```bash
uvicorn main:app --reload
```

Then open the UI at **http://127.0.0.1:8000/**.

Swagger / OpenAPI docs are at **http://127.0.0.1:8000/docs** if you want to
test the raw endpoints.

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | Uvicorn entrypoint (just re-exports `app`) |
| `api.py` | FastAPI routes, SSE streaming, exact-rank retry, agent orchestration |
| `agent.py` | LangGraph ReAct agent builder (one per model) |
| `tools.py` | Tavily + DuckDuckGo + deep page reader; defensive arg coercion for Llama 3.1's tool-calling quirks |
| `llm_passes.py` | Extra query-understanding and verifier/critic LLM calls |
| `rank_utils.py` | Shared ordinal/rank parsing helpers |
| `fact_extraction.py` | Claim extraction, source enrichment, trust signal computation |
| `source_quality.py` | Domain reputation and reliability scoring |
| `cache_store.py` | JSON-file result cache with 24h TTL and lazy expired-entry pruning |
| `chat_store.py` | JSON-file chat history |
| `config.py` | Models, prompts, constants, `.env` loading |
| `static/` | Frontend SPA (`index.html`, `app.js`, `styles.css`) |

## Endpoints

- `GET /` — the UI
- `GET /ask_agent_stream?question=...&model=...&chat_id=...` — Server-Sent
  Events stream used by the UI. Emits `start`, `understanding`, `tool_call`,
  `tool_result`, `answer`, `extracting`, `verifying`, `final`, `done`, `error`
  events.
- `GET /ask_agent?question=...&model=...` — Legacy non-streaming endpoint.
  Returns the full result as JSON. Useful for Swagger / command-line testing.
- `GET /api/chats` and `GET /api/chats/{id}` — chat list and detail
- `PATCH /api/chats/{id}` — rename chat
- `DELETE /api/chats/{id}` — delete chat
- `POST /api/cache/clear` — wipe the result cache
- `GET /api/models` — list available model keys
- `GET /api/health` — health check

## Data files

The server creates two JSON files in the project root at runtime:

- `chats.json` — all conversation history
- `cache.json` — cached results (auto-pruned of expired entries on read)

Both are excluded from git via `.gitignore`.

## Notes

- The first run takes a few seconds because all three agents are built at
  startup. Subsequent requests reuse the cached LLM clients.
- Explicit ordinal/rank questions use an exact-original-question search retry.
  This keeps answers like "3rd top scorer" focused on the requested row rather
  than the overall leader or a club-by-club table.
- If `tavily_search_results_json` fails for Llama 3.1 with a malformed-args
  error, the tool returns a teaching error message that prompts the model
  to retry with a clean string query. See the defensive coercion logic in
  `tools.py` for details.
