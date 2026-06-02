"""Extra LLM inference passes for understanding and answer verification.

Most calls do not perform retrieval. High accuracy mode has one deliberately
slower planning pass that proposes supplemental source-directed queries, then
audits the enlarged evidence set before the final answer is composed.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import MODEL_TEMPERATURE
from fact_extraction import coerce_message_content, sanitize_answer
from rank_utils import ordinal_label, requested_rank
from source_quality import source_reliability

logger = logging.getLogger("deep_research")

_LLM_CACHE: dict[str, ChatOllama] = {}


def _get_llm(model_id: str) -> ChatOllama:
    llm = _LLM_CACHE.get(model_id)
    if llm is None:
        llm = ChatOllama(model=model_id, temperature=MODEL_TEMPERATURE)
        _LLM_CACHE[model_id] = llm
    return llm


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            cleaned.append(s[:120])
    return cleaned[:limit]


def _fallback_understanding(question: str) -> dict[str, Any]:
    q = (question or "").strip()
    q_lower = q.lower()
    question_type = "general"
    answer_type = ""
    if re.search(r"\b(stock|share|ticker|quote|closing price|market cap)\b", q_lower):
        question_type = "finance"
        answer_type = "market value + date"
    elif re.search(r"\b(top scorer|top scorers|standings|score|match|league|lig|s[üu]per)\b", q_lower):
        question_type = "sports"
        answer_type = "leader(s) + statistic"
        rank = requested_rank(q)
        if rank:
            answer_type = f"{ordinal_label(rank)} ranked entry + statistic"
    elif re.search(r"\b(weather|temperature|forecast|rain|snow)\b", q_lower):
        question_type = "weather"
        answer_type = "weather value + place + time"
    elif re.search(r"\b(version|release|changelog|api|sdk|package)\b", q_lower):
        question_type = "software"
        answer_type = "version/release fact"

    season_match = re.search(r"\b(?:20\d{2}[/-](?:20)?\d{2}|20\d{2}\s*-\s*(?:20)?\d{2})\b", q)
    year_match = re.search(r"\b20\d{2}\b", q)
    time_context = season_match.group(0) if season_match else (year_match.group(0) if year_match else "")

    entities: list[str] = []
    if re.search(r"turkish|s[üu]per\s+lig|super\s+lig", q, re.IGNORECASE):
        entities.append("Turkish Süper Lig")
    ticker_match = re.search(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,3})?\b", q)
    if ticker_match:
        entities.append(ticker_match.group(0))

    return {
        "question_type": question_type,
        "entities": entities,
        "time_context": time_context,
        "answer_type": answer_type or "direct factual answer",
        "requested_rank": requested_rank(q),
        "ambiguities": [],
        "confidence": "medium",
    }


def understand_query(question: str, model_id: str) -> dict[str, Any]:
    """Classify intent and entities before retrieval.

    The pass must not invent answers or generate search queries. Its result is
    inserted into the agent context as interpretation metadata only.
    """
    system = SystemMessage(content=(
        "You classify user questions for a web research assistant. "
        "Return only valid JSON. Do not answer the question. Do not generate "
        "search queries. Do not invent candidate answers, names, numbers, or "
        "facts not present in the user's words."
    ))
    human = HumanMessage(content=f"""
Question:
{question}

Return JSON exactly with these fields:
{{
  "question_type": "finance|sports|weather|current_news|software|official_stats|health_science|historical_fact|general",
  "entities": ["entities explicitly mentioned by the user"],
  "time_context": "date/season/relative time explicitly requested, or empty string",
  "requested_rank": "integer rank if the user explicitly asks for one, otherwise null",
  "answer_type": "the shape of the requested answer, e.g. player(s)+goals, price+date, name, yes/no",
  "ambiguities": ["real ambiguities that may affect the answer"],
  "confidence": "low|medium|high"
}}
""".strip())

    try:
        raw = _get_llm(model_id).invoke([system, human])
        data = _extract_json_object(coerce_message_content(getattr(raw, "content", raw)))
    except Exception as exc:
        logger.info("Query understanding pass failed: %s", exc)
        return _fallback_understanding(question)

    if not data:
        return _fallback_understanding(question)
    fallback = _fallback_understanding(question)
    question_type = str(data.get("question_type") or "general").strip().lower()
    allowed_types = {
        "finance", "sports", "weather", "current_news", "software",
        "official_stats", "health_science", "historical_fact", "general",
    }
    if question_type not in allowed_types:
        question_type = "general"
    confidence = str(data.get("confidence") or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    requested = (
        int(data["requested_rank"])
        if isinstance(data.get("requested_rank"), int)
        else requested_rank(question)
    )
    return {
        "question_type": question_type,
        "entities": _string_list(data.get("entities")) or fallback["entities"],
        "time_context": str(data.get("time_context") or fallback["time_context"] or "").strip()[:160],
        "requested_rank": requested,
        "answer_type": str(data.get("answer_type") or fallback["answer_type"] or "").strip()[:160],
        "ambiguities": _string_list(data.get("ambiguities"), limit=4),
        "confidence": confidence,
    }


def understanding_context_message(understanding: dict[str, Any]) -> SystemMessage | None:
    if not understanding:
        return None
    return SystemMessage(content=(
        "Question understanding metadata. This is not evidence and must not be "
        "cited as a source. Use it only to interpret what the user is asking. "
        "If requested_rank is not null, preserve that exact ordinal/rank in "
        "tool queries and final answers; do not answer the overall leader "
        "unless that exact rank is requested.\n"
        f"{json.dumps(understanding, ensure_ascii=False)}"
    ))


def refine_search_queries(
    *,
    question: str,
    model_id: str,
    understanding: dict[str, Any] | None = None,
) -> list[str]:
    """Rewrite the user's question into focused web search queries before retrieval.

    This runs before the agent searches so retrieval surfaces stronger sources
    up front. It restructures wording only — it must not invent candidate
    answers, and it must preserve the user's exact entities, dates, seasons,
    ranks, metrics, and constraints. Returns [] when refinement adds nothing.
    """
    system = SystemMessage(content=(
        "You rewrite a user's question into focused web search queries for a "
        "research assistant. Return only valid JSON. Produce 2 to 3 short "
        "queries that improve retrieval precision and source quality. Preserve "
        "the user's exact entity, date, season, rank, metric, and wording "
        "constraints. Do not invent or guess answers, names, numbers, prices, "
        "dates, or rankings, and never put a guessed answer into a query."
    ))
    human = HumanMessage(content=(
        "Question:\n"
        f"{question}\n\n"
        "Question understanding metadata:\n"
        f"{json.dumps(understanding or {}, ensure_ascii=False)}\n\n"
        "Return JSON exactly like:\n"
        "{\n"
        '  "queries": ["focused query 1", "focused query 2"]\n'
        "}"
    ))
    try:
        raw = _get_llm(model_id).invoke([system, human])
        data = _extract_json_object(coerce_message_content(getattr(raw, "content", raw)))
    except Exception as exc:
        logger.info("Search query refinement failed: %s", exc)
        return []

    queries = data.get("queries") if isinstance(data, dict) else []
    if not isinstance(queries, list):
        return []
    return _dedupe_queries(question, [str(q) for q in queries], limit=3)


def search_plan_context_message(queries: list[str]) -> SystemMessage | None:
    cleaned = [str(q).strip() for q in (queries or []) if str(q).strip()][:3]
    if not cleaned:
        return None
    lines = "\n".join(f"- {q}" for q in cleaned)
    return SystemMessage(content=(
        "Suggested starting web search queries, already focused for this "
        "question. Use them with your search tools first, then refine if the "
        "results are weak. They are guidance, not evidence, and must not be "
        "cited as a source.\n"
        f"{lines}"
    ))


def _compact_sources(sources: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    sorted_sources = sorted(
        sources or [],
        key=lambda s: float(s.get("reliability_score") or 0),
        reverse=True,
    )
    for source in sorted_sources[:limit]:
        compact.append({
            "url": source.get("url", ""),
            "title": source.get("title", ""),
            "domain": source.get("domain", ""),
            "tier": source.get("tier", ""),
            "reliability_score": source.get("reliability_score", 0),
            "snippet": str(source.get("snippet") or "")[:900],
        })
    return compact


def _direct_hints(tool_messages: list[dict[str, str]], *, limit: int = 8) -> list[str]:
    hints: list[str] = []
    for message in tool_messages or []:
        content = message.get("content") or ""
        for line in content.splitlines():
            if line.startswith("DIRECT_DATA_HINT") or line.startswith("DATA_HINT:"):
                hints.append(line[:500])
                if len(hints) >= limit:
                    return hints
    return hints


def _hint_source_url(hint: str) -> str:
    match = re.search(r"\s+-\s+source:\s*(https?://\S+)", hint or "", re.IGNORECASE)
    return match.group(1).rstrip(").,;") if match else ""


def _hint_source_score(hint: str) -> float:
    url = _hint_source_url(hint)
    if not url:
        return 0.45
    reliability = source_reliability(url)
    return float(reliability.get("score") or 0.35)


def _norm_candidate_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _choose_supported_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose among conflicting direct hints by support, not arrival order."""
    if not candidates:
        return None

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            _norm_candidate_text(str(candidate.get("answer") or "")),
            _norm_candidate_text(str(candidate.get("value") or "")),
            _norm_candidate_text(str(candidate.get("date") or "")),
        )
        source_score = float(candidate.get("source_score") or 0.0)
        existing = grouped.get(key)
        if not existing:
            grouped[key] = {
                **candidate,
                "best_source_score": source_score,
                "source_count": 1,
                "sources": [candidate.get("source_url")] if candidate.get("source_url") else [],
            }
            continue
        existing["best_source_score"] = max(float(existing.get("best_source_score") or 0.0), source_score)
        existing["source_count"] = int(existing.get("source_count") or 1) + 1
        if candidate.get("source_url") and candidate.get("source_url") not in existing["sources"]:
            existing["sources"].append(candidate.get("source_url"))
        if source_score > float(existing.get("source_score") or 0.0):
            for field in ("answer", "value", "date", "final_answer", "issue", "source_url", "source_score"):
                existing[field] = candidate.get(field)

    ranked = sorted(
        grouped.values(),
        key=lambda c: (
            float(c.get("best_source_score") or 0.0) + min(int(c.get("source_count") or 1) - 1, 3) * 0.04,
            int(c.get("source_count") or 1),
        ),
        reverse=True,
    )
    best = ranked[0]
    if len(ranked) == 1:
        return best

    second = ranked[1]
    best_score = float(best.get("best_source_score") or 0.0)
    second_score = float(second.get("best_source_score") or 0.0)
    if best_score >= 0.55 and best_score >= second_score + 0.10:
        best["conflict_note"] = "Resolved conflicting direct hints by source reliability"
        return best
    if int(best.get("source_count") or 1) > int(second.get("source_count") or 1) and best_score >= second_score:
        best["conflict_note"] = "Resolved conflicting direct hints by corroboration"
        return best
    return None


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip())
    except (ValueError, TypeError):
        return None


def _choose_dated_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose among dated structured hints (market, FX) by recency then reliability.

    "Most recent" questions are answered by the newest dated value, so a stale
    source must not win on arrival order. Ties on date fall back to source
    reliability, which is how arrival-order first-match silently went wrong.
    """
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (
            _parse_iso_date(c.get("date")) or date.min,
            float(c.get("source_score") or 0.0),
        ),
    )


def _season_start_year(text: str) -> int:
    """Start year of a football season string like '2024/2025', '2024/25', '2024-25'."""
    match = re.search(r"20\d{2}", text or "")
    return int(match.group(0)) if match else 0


def _prefer_latest_season(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop stale-season candidates when sources disagree on the season.

    A higher-reliability source reporting last season must not outrank a fresher
    source reporting the current season, so resolve season conflicts by recency
    before resolving the remaining tie by source reliability.
    """
    seasons = {str(c.get("date") or "").strip() for c in candidates if str(c.get("date") or "").strip()}
    if len(seasons) <= 1:
        return candidates
    latest = max(_season_start_year(s) for s in seasons)
    filtered = [c for c in candidates if _season_start_year(str(c.get("date") or "")) == latest]
    return filtered or candidates


def _fallback_high_accuracy_queries(question: str, understanding: dict[str, Any] | None) -> list[str]:
    q = (question or "").strip()
    q_type = str((understanding or {}).get("question_type") or "").lower()
    answer_type = str((understanding or {}).get("answer_type") or "").lower()
    season = _current_european_football_season()
    queries: list[str] = []

    if q_type == "finance" or re.search(r"\b(stock|share|ticker|closing price|market cap)\b", q, re.IGNORECASE):
        queries.extend([
            f"{q} Yahoo Finance historical data daily close",
            f"{q} Nasdaq MarketWatch previous close historical price",
            f"{q} official market data closing price",
        ])
    elif q_type == "sports" or re.search(r"\b(scorer|standings|league|lig|rank|goals?)\b", q, re.IGNORECASE):
        queries.extend([
            f"{q} {season} site:footystats.org/clubs OR site:espn.com/soccer/team/stats",
            f"{q} official statistics table",
            f"{q} StatBunker Transfermarkt WorldFootball top scorers",
        ])
    elif q_type in {"official_stats", "health_science", "software"}:
        queries.extend([
            f"{q} official source",
            f"{q} primary source data",
            f"{q} latest reliable source",
        ])
    elif "date" in answer_type or re.search(r"\b(today|yesterday|latest|current|recent)\b", q, re.IGNORECASE):
        queries.extend([
            f"{q} latest reliable source",
            f"{q} official source",
            f"{q} dated source",
        ])
    else:
        queries.extend([
            f"{q} official source",
            f"{q} reliable source",
            f"{q} corroborating source",
        ])

    return _dedupe_queries(q, queries, limit=3)


def _current_european_football_season() -> str:
    today = date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}/{start_year + 1}"


def _dedupe_queries(question: str, queries: list[str], *, limit: int = 4) -> list[str]:
    original_key = re.sub(r"\s+", " ", (question or "").strip().lower())
    cleaned: list[str] = []
    seen = {original_key}
    for query in queries or []:
        q = re.sub(r"\s+", " ", str(query or "").strip())
        if not q:
            continue
        q = q[:240]
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(q)
        if len(cleaned) >= limit:
            break
    return cleaned


def plan_high_accuracy_queries(
    *,
    question: str,
    model_id: str,
    understanding: dict[str, Any] | None = None,
) -> list[str]:
    """Generate supplemental queries for high accuracy mode.

    This is intentionally separate from the first query-understanding pass. It
    may propose search queries, but it must not propose candidate answers.
    """
    fallback = _fallback_high_accuracy_queries(question, understanding)
    system = SystemMessage(content=(
        "You plan supplemental web searches for a high-accuracy research mode. "
        "Return only valid JSON. Generate 2 to 4 search queries that improve "
        "source quality or freshness. Preserve the user's exact entity, date, "
        "season, rank, metric, and wording constraints. Do not invent candidate "
        "answers, names, scores, prices, dates, values, or rankings. Do not add "
        "a guessed answer to any query. Prefer official, primary, or table/data "
        "sources when suitable."
    ))
    human = HumanMessage(content=(
        "Question:\n"
        f"{question}\n\n"
        "Question understanding metadata:\n"
        f"{json.dumps(understanding or {}, ensure_ascii=False)}\n\n"
        "Return JSON exactly like:\n"
        "{\n"
        '  "queries": ["supplemental query 1", "supplemental query 2"]\n'
        "}"
    ))
    try:
        raw = _get_llm(model_id).invoke([system, human])
        data = _extract_json_object(coerce_message_content(getattr(raw, "content", raw)))
    except Exception as exc:
        logger.info("High-accuracy query planning failed: %s", exc)
        return fallback

    queries = data.get("queries") if isinstance(data, dict) else []
    if not isinstance(queries, list):
        return fallback
    planned = _dedupe_queries(question, [str(q) for q in queries] + fallback, limit=4)
    return planned or fallback


def _compact_tool_messages(
    tool_messages: list[dict[str, str]],
    *,
    max_total_chars: int = 32000,
    max_message_chars: int = 6500,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    remaining = max_total_chars
    for index, message in enumerate(tool_messages or [], start=1):
        if remaining <= 0:
            break
        name = str(message.get("name") or "tool")
        content = str(message.get("content") or "")
        if not content:
            continue
        budget = min(max_message_chars, remaining)
        selected = _select_high_value_lines(content, budget=budget)
        if not selected:
            selected = content[:budget]
        compact.append({
            "index": index,
            "tool": name,
            "content": selected[:budget],
        })
        remaining -= len(compact[-1]["content"])
    return compact


def _select_high_value_lines(content: str, *, budget: int) -> str:
    lines = (content or "").splitlines()
    if len(content) <= budget:
        return content

    kept: list[str] = []
    current_result_lines = 0
    for line in lines:
        high_value = (
            line.startswith("DIRECT DATA HINTS")
            or line.startswith("DIRECT_DATA_HINT")
            or line.startswith("DATA_HINT:")
            or line.startswith("TAVILY SUMMARY")
            or line.startswith("RESULT ")
            or line.startswith("TITLE:")
            or line.startswith("URL:")
            or line.startswith("RELIABILITY_SIGNALS:")
            or line.startswith("CONTENT:")
        )
        if line.startswith("RESULT "):
            current_result_lines = 0
        if line.startswith("CONTENT:"):
            line = line[:1100]
        if high_value or (kept and current_result_lines < 2 and line.strip()):
            kept.append(line)
            current_result_lines += 1
        if sum(len(x) + 1 for x in kept) >= budget:
            break
    return "\n".join(kept)[:budget]


def _evidence_candidates_from_hints(
    question: str,
    tool_messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    wanted_rank = requested_rank(question)
    for hint in _direct_hints(tool_messages, limit=20):
        url_match = re.search(r"\s+-\s+source:\s*(https?://\S+)", hint, re.IGNORECASE)
        source_urls = [url_match.group(1)] if url_match else []
        ranked = re.search(
            r"Leaderboard rank\s+(\d{1,2})\s+candidate:\s*(.+?)\s+was listed\s+([0-9a-z]+)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if ranked and (wanted_rank is None or int(ranked.group(1)) == wanted_rank):
            rank = int(ranked.group(1))
            candidates.append({
                "answer": ranked.group(2).strip(),
                "value": f"{ranked.group(4)} goals",
                "date": "",
                "source_urls": source_urls,
                "support_level": "direct",
                "reliability_notes": f"Direct parsed leaderboard row for {ordinal_label(rank)} place.",
                "conflicts": [],
            })
            continue

        leader = re.search(
            r"Leaderboard candidate:\s*(.+?)\s+(?:were joint leaders|was the leader|leaders?)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if leader and wanted_rank is None:
            candidates.append({
                "answer": leader.group(1).strip(),
                "value": f"{leader.group(2)} goals",
                "date": "",
                "source_urls": source_urls,
                "support_level": "direct",
                "reliability_notes": "Direct parsed leaderboard leader hint.",
                "conflicts": [],
            })
            continue

        club_scorer = re.search(
            r"Club top scorer:\s*(.+?)'s top scorer for the\s+(.+?)\s+season\s+is\s+(.+?)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if club_scorer:
            candidates.append({
                "answer": club_scorer.group(3).strip(),
                "value": f"{club_scorer.group(4)} goals",
                "date": club_scorer.group(2).strip(),
                "source_urls": source_urls,
                "support_level": "direct",
                "reliability_notes": f"Direct parsed club top-scorer evidence for {club_scorer.group(1).strip()}.",
                "conflicts": [],
            })
            continue

        espn_club_scorer = re.search(
            r"Club top scorer:\s*ESPN lists the club's top scorer for the\s+(.+?)\s+season\s+as\s+(.+?)\s+with\s+(\d{1,3})\s+goals(?:\s+in\s+(\d{1,3})\s+appearances)?",
            hint,
            re.IGNORECASE,
        )
        if espn_club_scorer:
            value = f"{espn_club_scorer.group(3)} goals"
            if espn_club_scorer.group(4):
                value += f" in {espn_club_scorer.group(4)} appearances"
            candidates.append({
                "answer": espn_club_scorer.group(2).strip(),
                "value": value,
                "date": espn_club_scorer.group(1).strip(),
                "source_urls": source_urls,
                "support_level": "direct",
                "reliability_notes": "Direct parsed ESPN club top-scorer row.",
                "conflicts": [],
            })
            continue

        market = re.search(
            r"Structured market data:\s*([A-Z0-9.\-]+)\s+most recent daily close was\s+(\$?[0-9][0-9,]*(?:\.\d+)?)\s+on\s+(\d{4}-\d{2}-\d{2})",
            hint,
            re.IGNORECASE,
        )
        if market:
            candidates.append({
                "answer": market.group(1),
                "value": market.group(2),
                "date": market.group(3),
                "source_urls": source_urls,
                "support_level": "direct",
                "reliability_notes": "Structured market data daily close.",
                "conflicts": [],
            })
    return candidates[:8]


def extract_evidence_table(
    *,
    question: str,
    model_id: str,
    tool_messages: list[dict[str, str]],
    understanding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured evidence table from raw retrieved tool output."""
    deterministic_candidates = _evidence_candidates_from_hints(question, tool_messages)
    compact_messages = _compact_tool_messages(tool_messages)
    system = SystemMessage(content=(
        "You extract a structured evidence table for a high-accuracy web "
        "research mode. Return only valid JSON. Use only the provided tool "
        "outputs. Do not answer from memory. Do not invent missing names, "
        "numbers, dates, ranks, prices, or URLs. Preserve exact values. If "
        "the user asks for an ordinal or rank, candidates must answer that "
        "exact rank rather than the overall leader. Mark stale or date-mismatched "
        "evidence as weak. A tied answer is valid when the sources show a tie."
    ))
    human = HumanMessage(content=(
        "Question:\n"
        f"{question}\n\n"
        "Question understanding metadata:\n"
        f"{json.dumps(understanding or {}, ensure_ascii=False)}\n\n"
        "Deterministic candidates already parsed from direct hints:\n"
        f"{json.dumps(deterministic_candidates, ensure_ascii=False)}\n\n"
        "Raw retrieved tool evidence, compacted:\n"
        f"{json.dumps(compact_messages, ensure_ascii=False)}\n\n"
        "Return JSON exactly with these fields:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "answer": "candidate answer text",\n'
        '      "value": "number/price/statistic/unit if any",\n'
        '      "date": "date/season if any",\n'
        '      "source_urls": ["https://..."],\n'
        '      "support_level": "direct|indirect|weak",\n'
        '      "reliability_notes": "short note",\n'
        '      "conflicts": ["brief conflict notes"]\n'
        "    }\n"
        "  ],\n"
        '  "conflicts": ["cross-source conflicts or stale-data issues"],\n'
        '  "missing": ["information still missing, or empty array"]\n'
        "}"
    ))

    try:
        raw = _get_llm(model_id).invoke([system, human])
        data = _extract_json_object(coerce_message_content(getattr(raw, "content", raw)))
    except Exception as exc:
        logger.info("Evidence table extraction failed: %s", exc)
        return {
            "candidates": deterministic_candidates,
            "conflicts": [],
            "missing": [str(exc)[:120]],
        }

    if not data:
        return {
            "candidates": deterministic_candidates,
            "conflicts": [],
            "missing": ["Evidence table extraction returned no JSON"],
        }

    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    normalized_candidates: list[dict[str, Any]] = []
    for candidate in deterministic_candidates + candidates:
        if not isinstance(candidate, dict):
            continue
        support = str(candidate.get("support_level") or "weak").strip().lower()
        if support not in {"direct", "indirect", "weak"}:
            support = "weak"
        urls = [
            str(url).strip()
            for url in (candidate.get("source_urls") or [])
            if str(url).strip().startswith("http")
        ][:5]
        answer = str(candidate.get("answer") or "").strip()
        value = str(candidate.get("value") or "").strip()
        if not answer and not value:
            continue
        normalized_candidates.append({
            "answer": answer[:240],
            "value": value[:120],
            "date": str(candidate.get("date") or "").strip()[:80],
            "source_urls": urls,
            "support_level": support,
            "reliability_notes": str(candidate.get("reliability_notes") or "").strip()[:220],
            "conflicts": _string_list(candidate.get("conflicts"), limit=4),
        })
        if len(normalized_candidates) >= 10:
            break

    return {
        "candidates": normalized_candidates,
        "conflicts": _string_list(data.get("conflicts"), limit=6),
        "missing": _string_list(data.get("missing"), limit=6),
    }


def _is_club_scoped_leaderboard_block(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(
        re.search(r"\b(top scorers at each club|club top scorers|per club|by club)\b", lowered)
        or re.search(r"\|\s*pos\s*\|\s*team\s*\|\s*top scorer\s*\|\s*goals\s*\|", lowered)
    )


def _result_block_around(content: str, position: int) -> str:
    block_start = content.rfind("\nRESULT ", 0, position)
    if block_start == -1:
        block_start = max(0, position - 1200)
    block_end = content.find("\nRESULT ", position + 1)
    if block_end == -1:
        block_end = min(len(content), position + 1200)
    return content[block_start:block_end]


def _ranked_leaderboard_answer_from_tool_content(
    *,
    question: str,
    tool_messages: list[dict[str, str]],
) -> dict[str, Any] | None:
    wanted_rank = requested_rank(question)
    if wanted_rank is None:
        return None

    best: tuple[str, int] | None = None
    for message in tool_messages or []:
        content = message.get("content") or ""
        for row_match in re.finditer(
            r"\|\s*(\d{1,2})\s*\|\s*([^|\n]+?)\s*\|[^|\n]*\|\s*\d{1,3}\s*\|\s*(\d{1,3})\s*\|",
            content,
        ):
            try:
                position = int(row_match.group(1))
                goals = int(row_match.group(3))
            except ValueError:
                continue
            if position != wanted_rank:
                continue
            if _is_club_scoped_leaderboard_block(_result_block_around(content, row_match.start())):
                continue
            name = re.sub(r"\s+", " ", row_match.group(2)).strip(" 🏆")
            if not name:
                continue
            if best is None or goals > best[1]:
                best = (name, goals)

    if not best:
        return None
    name, goals = best
    return {
        "status": "revise",
        "final_answer": f"{name} was listed {ordinal_label(wanted_rank)} with {goals} goals.",
        "issues": ["Answered the requested rank by parsing retrieved leaderboard rows"],
        "confidence": "high",
    }


def _fallback_verification(
    *,
    question: str,
    answer: str,
    tool_messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Deterministic safety net when the verifier LLM fails or emits bad JSON."""
    answer = _resolve_scoreline_contradiction(answer)
    hints = _direct_hints(tool_messages)
    if not hints:
        return {"status": "skipped", "final_answer": answer, "issues": [], "confidence": "unknown"}
    direct = _direct_hint_verification(question=question, tool_messages=tool_messages)
    if direct:
        return direct

    answer_lower = (answer or "").lower()
    hedged = bool(re.search(
        r"\b(could not find|couldn't find|not definitive|no definitive|not confirmed|does not provide|do not provide)\b",
        answer_lower,
    ))

    for hint in hints:
        wanted_rank = requested_rank(question)
        ranked_leaderboard = re.search(
            r"Leaderboard rank\s+(\d{1,2})\s+candidate:\s*(.+?)\s+was listed\s+([0-9a-z]+)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if ranked_leaderboard and (wanted_rank is None or int(ranked_leaderboard.group(1)) == wanted_rank):
            rank = int(ranked_leaderboard.group(1))
            name = ranked_leaderboard.group(2).strip()
            goals = ranked_leaderboard.group(4)
            return {
                "status": "revise",
                "final_answer": f"{name} was listed {ordinal_label(rank)} with {goals} goals.",
                "issues": ["Answered the requested rank from direct leaderboard evidence"],
                "confidence": "medium",
            }

        espn_club_scorer = re.search(
            r"Club top scorer:\s*ESPN lists the club's top scorer for the\s+(.+?)\s+season\s+as\s+(.+?)\s+with\s+(\d{1,3})\s+goals(?:\s+in\s+(\d{1,3})\s+appearances)?",
            hint,
            re.IGNORECASE,
        )
        if espn_club_scorer and ("scorer" in question.lower() or "goals" in question.lower()):
            season = espn_club_scorer.group(1).strip()
            name = espn_club_scorer.group(2).strip()
            goals = espn_club_scorer.group(3)
            appearances = espn_club_scorer.group(4)
            app_text = f" in {appearances} appearances" if appearances else ""
            return {
                "status": "revise",
                "final_answer": f"{name} is listed as the club's top goal scorer for the {season} season with {goals} goals{app_text}.",
                "issues": ["Answered from ESPN club scorer evidence"],
                "confidence": "medium",
            }

        leaderboard = re.search(
            r"Leaderboard candidate:\s*(.+?)\s+(?:were joint leaders|was the leader|leaders?)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if wanted_rank is None and leaderboard and (hedged or "top scorer" in question.lower() or "scorer" in question.lower()):
            names = leaderboard.group(1).strip()
            goals = leaderboard.group(2)
            return {
                "status": "revise",
                "final_answer": f"{names} were joint top scorers with {goals} goals.",
                "issues": ["Replaced hedged answer with direct leaderboard evidence"],
                "confidence": "medium",
            }

        club_scorer = re.search(
            r"Club top scorer:\s*(.+?)'s top scorer for the\s+(.+?)\s+season\s+is\s+(.+?)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if club_scorer and ("scorer" in question.lower() or "goals" in question.lower()):
            club = club_scorer.group(1).strip()
            season = club_scorer.group(2).strip()
            name = club_scorer.group(3).strip()
            goals = club_scorer.group(4)
            return {
                "status": "revise",
                "final_answer": f"{name} is {club}'s top goal scorer for the {season} season with {goals} goals.",
                "issues": ["Answered from direct club scorer evidence"],
                "confidence": "medium",
            }

        market = re.search(
            r"Structured market data:\s*([A-Z0-9.\-]+)\s+most recent daily close was\s+(\$?[0-9][0-9,]*(?:\.\d+)?)\s+on\s+(\d{4}-\d{2}-\d{2})",
            hint,
            re.IGNORECASE,
        )
        if market and (hedged or "close" in question.lower() or "stock" in question.lower()):
            symbol, price, day = market.group(1), market.group(2), market.group(3)
            return {
                "status": "revise",
                "final_answer": f"{symbol}'s most recent daily close was {price} on {day}.",
                "issues": ["Replaced hedged answer with structured market data"],
                "confidence": "medium",
            }

    return {"status": "skipped", "final_answer": answer, "issues": [], "confidence": "unknown"}


def _direct_hint_verification(
    *,
    question: str,
    tool_messages: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Authoritative deterministic repair from exact direct data hints.

    These hints are produced by parsers over retrieved evidence, so for exact
    rank/market questions they should outrank any later LLM interpretation.
    """
    wanted_rank = requested_rank(question)
    rank_candidates: list[dict[str, Any]] = []
    for hint in _direct_hints(tool_messages):
        ranked_leaderboard = re.search(
            r"Leaderboard rank\s+(\d{1,2})\s+candidate:\s*(.+?)\s+was listed\s+([0-9a-z]+)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if ranked_leaderboard and wanted_rank is not None and int(ranked_leaderboard.group(1)) == wanted_rank:
            name = ranked_leaderboard.group(2).strip()
            goals = int(ranked_leaderboard.group(4))
            rank_candidates.append({
                "answer": name,
                "value": str(goals),
                "date": "",
                "goals": goals,
                "source_url": _hint_source_url(hint),
                "source_score": _hint_source_score(hint),
            })

    if rank_candidates and wanted_rank is not None:
        # When sources agree on the goal count, joint holders of the rank are a
        # real tie and should both be listed. When they disagree, resolve by
        # source reliability/corroboration rather than arrival order, falling
        # back to the highest goal count only if the conflict is unresolvable.
        chosen_rank = _choose_supported_candidate(rank_candidates)
        if chosen_rank:
            chosen_goals = int(chosen_rank.get("goals") or chosen_rank.get("value") or 0)
        else:
            chosen_goals = max(int(c["goals"]) for c in rank_candidates)
        names: list[str] = []
        for candidate in rank_candidates:
            if int(candidate["goals"]) == chosen_goals and candidate["answer"] not in names:
                names.append(candidate["answer"])
        joined = " and ".join(names[:3])
        verb = "were" if len(names) > 1 else "was"
        issues = ["Answered the requested rank from direct leaderboard evidence"]
        if chosen_rank and chosen_rank.get("conflict_note"):
            issues.append(str(chosen_rank["conflict_note"]))
        return {
            "status": "revise",
            "final_answer": f"{joined} {verb} listed {ordinal_label(wanted_rank)} with {chosen_goals} goals.",
            "issues": issues,
            "confidence": "high",
        }

    club_candidates: list[dict[str, Any]] = []
    for hint in _direct_hints(tool_messages, limit=30):
        club_scorer = re.search(
            r"Club top scorer:\s*(.+?)'s top scorer for the\s+(.+?)\s+season\s+is\s+(.+?)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if club_scorer and ("scorer" in question.lower() or "goals" in question.lower()):
            club = club_scorer.group(1).strip()
            season = club_scorer.group(2).strip()
            name = club_scorer.group(3).strip()
            goals = club_scorer.group(4)
            club_candidates.append({
                "answer": name,
                "value": f"{goals} goals",
                "date": season,
                "source_url": _hint_source_url(hint),
                "source_score": _hint_source_score(hint),
                "final_answer": f"{name} is {club}'s top goal scorer for the {season} season with {goals} goals.",
                "issue": "Answered from direct club scorer evidence",
            })

        espn_club_scorer = re.search(
            r"Club top scorer:\s*ESPN lists the club's top scorer for the\s+(.+?)\s+season\s+as\s+(.+?)\s+with\s+(\d{1,3})\s+goals(?:\s+in\s+(\d{1,3})\s+appearances)?",
            hint,
            re.IGNORECASE,
        )
        if espn_club_scorer and ("scorer" in question.lower() or "goals" in question.lower()):
            season = espn_club_scorer.group(1).strip()
            name = espn_club_scorer.group(2).strip()
            goals = espn_club_scorer.group(3)
            appearances = espn_club_scorer.group(4)
            app_text = f" in {appearances} appearances" if appearances else ""
            club_candidates.append({
                "answer": name,
                "value": f"{goals} goals{app_text}",
                "date": season,
                "source_url": _hint_source_url(hint),
                "source_score": _hint_source_score(hint),
                "final_answer": f"{name} is listed as the club's top goal scorer for the {season} season with {goals} goals{app_text}.",
                "issue": "Answered from ESPN club scorer evidence",
            })

    chosen_club = _choose_supported_candidate(_prefer_latest_season(club_candidates))
    if chosen_club:
        issues = [str(chosen_club.get("issue") or "Answered from direct club scorer evidence")]
        if chosen_club.get("conflict_note"):
            issues.append(str(chosen_club["conflict_note"]))
        return {
            "status": "revise",
            "final_answer": str(chosen_club.get("final_answer") or ""),
            "issues": issues,
            "confidence": "high",
        }

    if "close" in question.lower() or "stock" in question.lower():
        market_candidates: list[dict[str, Any]] = []
        for hint in _direct_hints(tool_messages):
            market = re.search(
                r"Structured market data:\s*([A-Z0-9.\-]+)\s+most recent daily close was\s+(\$?[0-9][0-9,]*(?:\.\d+)?)\s+on\s+(\d{4}-\d{2}-\d{2})",
                hint,
                re.IGNORECASE,
            )
            if market:
                symbol, price, day = market.group(1), market.group(2), market.group(3)
                market_candidates.append({
                    "final_answer": f"{symbol}'s most recent daily close was {price} on {day}.",
                    "date": day,
                    "source_score": _hint_source_score(hint),
                })
        chosen_market = _choose_dated_candidate(market_candidates)
        if chosen_market:
            return {
                "status": "revise",
                "final_answer": chosen_market["final_answer"],
                "issues": ["Answered from structured market data"],
                "confidence": "high",
            }

    fx_candidates: list[dict[str, Any]] = []
    for hint in _direct_hints(tool_messages):
        fx = re.search(
            r"Structured FX data:\s*([A-Za-z]{3})/([A-Za-z]{3})\s+was\s+([0-9][0-9,]*(?:\.\d+)?)\s+on\s+(\d{4}-\d{2}-\d{2})",
            hint,
            re.IGNORECASE,
        )
        if fx:
            base, quote, rate, day = fx.group(1).upper(), fx.group(2).upper(), fx.group(3), fx.group(4)
            fx_candidates.append({
                "final_answer": f"1 {base} = {rate} {quote} (as of {day}).",
                "date": day,
                "source_score": _hint_source_score(hint),
            })
    chosen_fx = _choose_dated_candidate(fx_candidates)
    if chosen_fx:
        return {
            "status": "revise",
            "final_answer": chosen_fx["final_answer"],
            "issues": ["Answered from structured exchange-rate data"],
            "confidence": "high",
        }
    return _ranked_leaderboard_answer_from_tool_content(question=question, tool_messages=tool_messages)


def rank_answer_from_tool_messages(
    question: str,
    tool_messages: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Return a deterministic answer when retrieved evidence contains the requested rank."""
    if requested_rank(question) is None:
        return None
    return _direct_hint_verification(question=question, tool_messages=tool_messages)


def direct_answer_from_tool_messages(
    question: str,
    tool_messages: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Return a deterministic answer from any direct structured data hint."""
    return _direct_hint_verification(question=question, tool_messages=tool_messages)


def synthesize_answer_from_evidence(
    *,
    question: str,
    draft_answer: str,
    model_id: str,
    tool_messages: list[dict[str, str]],
    extraction: dict[str, Any],
    understanding: dict[str, Any] | None = None,
    evidence_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """High-accuracy answer pass that writes only from normalized evidence."""
    direct_hint_result = _direct_hint_verification(question=question, tool_messages=tool_messages)
    if direct_hint_result:
        return {
            "status": "deterministic",
            "final_answer": direct_hint_result.get("final_answer", draft_answer),
            "issues": direct_hint_result.get("issues", []),
            "confidence": direct_hint_result.get("confidence", "high"),
        }

    evidence_packet = {
        "question": question,
        "query_understanding": understanding or {},
        "draft_answer": draft_answer,
        "evidence_table": evidence_table or {},
        "direct_data_hints": _direct_hints(tool_messages, limit=12),
        "sources": _compact_sources(extraction.get("sources", []), limit=10),
        "facts": (extraction.get("facts") or [])[:12],
        "trust_signals": extraction.get("trust_signals", {}),
        "raw_tool_evidence": _compact_tool_messages(tool_messages, max_total_chars=18000, max_message_chars=4000),
    }
    system = SystemMessage(content=(
        "You are an evidence-first answer composer for a high-accuracy web "
        "research mode. Return only valid JSON. Use only the provided evidence "
        "packet. Do not use outside knowledge. Start with the answer when the "
        "evidence supports it. Preserve exact names, numbers, dates, ranks, "
        "currencies, and units from the evidence. If the user asks for a rank "
        "or ordinal, answer that exact rank, not the overall leader. Prefer "
        "DIRECT_DATA_HINT values when present. Prefer direct candidates from "
        "the evidence_table over the draft answer. Prefer newer and higher "
        "reliability sources when evidence conflicts. A tied result is a valid "
        "complete answer when the evidence shows a tie. Do not mention the "
        "draft, the evidence packet, tools, or verification process in the "
        "final answer."
    ))
    human = HumanMessage(content=(
        "Write the best concise final answer from this evidence packet.\n\n"
        "Return JSON exactly with these fields:\n"
        "{\n"
        '  "status": "answered|insufficient|conflict",\n'
        '  "final_answer": "concise user-facing answer",\n'
        '  "issues": ["brief evidence issues, or an empty array"],\n'
        '  "confidence": "low|medium|high"\n'
        "}\n\n"
        f"Evidence packet:\n{json.dumps(evidence_packet, ensure_ascii=False)}"
    ))

    try:
        raw = _get_llm(model_id).invoke([system, human])
        data = _extract_json_object(coerce_message_content(getattr(raw, "content", raw)))
    except Exception as exc:
        logger.info("Evidence synthesis pass failed: %s", exc)
        return {
            "status": "failed",
            "final_answer": draft_answer,
            "issues": [str(exc)[:120]],
            "confidence": "unknown",
        }

    if not data:
        return {
            "status": "failed",
            "final_answer": draft_answer,
            "issues": ["Evidence synthesis returned no JSON"],
            "confidence": "unknown",
        }

    status = str(data.get("status") or "answered").strip().lower()
    if status not in {"answered", "insufficient", "conflict"}:
        status = "answered"
    final_answer = _clean_verifier_answer(str(data.get("final_answer") or draft_answer).strip())
    if not final_answer:
        final_answer = draft_answer
    confidence = str(data.get("confidence") or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "status": status,
        "final_answer": final_answer,
        "issues": _string_list(data.get("issues"), limit=6),
        "confidence": confidence,
    }


def accept_synthesized_answer(synthesis: dict[str, Any], draft_answer: str) -> bool:
    """Decide whether the evidence synthesis should replace the draft answer.

    The synthesis pass runs on a local model and can occasionally produce a
    weaker, hedged answer than the agent's own draft. Adopt it only when it is
    a deterministic structured result, or when the model composed a genuinely
    different evidence-based answer that it did not itself flag as a non-answer.
    When the model signals it could not answer (failed/insufficient at low or
    unknown confidence), keep the original draft — unless that draft was empty.
    """
    synthesized = str(synthesis.get("final_answer") or "").strip()
    draft = (draft_answer or "").strip()
    if not synthesized or synthesized == draft:
        return False
    if synthesis.get("status") == "deterministic":
        return True
    if not draft:
        return True
    status = str(synthesis.get("status") or "").lower()
    confidence = str(synthesis.get("confidence") or "").lower()
    if status in {"failed", "insufficient"} and confidence in {"low", "unknown"}:
        return False
    return True


_HEDGE_RE = re.compile(
    r"(?:could\s+not|couldn'?t|can\s*not|cannot|can'?t|was\s+unable|unable\s+to|not\s+able\s+to|"
    r"do(?:es)?\s+not\s+(?:have|appear|know)|don'?t\s+(?:have|know)|"
    r"not\s+enough\s+(?:information|data)|insufficient\s+(?:information|data)|"
    r"no\s+(?:reliable\s+)?(?:information|data|results?|answer|definitive))",
    re.IGNORECASE,
)

_VALUE_ANSWER_HINTS = ("price", "number", "goals", "statistic", "value", "date", "percent", "rank", "score", "amount", "rate")
_VALUE_QUESTION_TYPES = {"finance", "sports", "weather", "official_stats"}

# Escalation thresholds. Deliberately conservative — they are the boundary
# between the free normal path and the costly retrieval/synthesis path, and are
# meant to be calibrated against eval/run_eval.py's pass-vs-fail signal gaps.
_ESCALATION_TRUST_OK = 0.6
_ESCALATION_COVERAGE_OK = 0.5
_ESCALATION_MIN_SOURCES = 2


def _answer_shape_gap(question: str, draft: str, understanding: dict[str, Any] | None) -> bool:
    """True when the question type expects a value but the draft has no digit.

    A finance/sports/stats/ranked question that comes back with no number is
    almost always a miss, so it is a strong, cheap escalation trigger. Kept
    conservative: only fires when a numeric/dated answer is genuinely expected.
    """
    u = understanding or {}
    answer_type = str(u.get("answer_type") or "").lower()
    expects_value = (
        u.get("requested_rank") is not None
        or str(u.get("question_type") or "").lower() in _VALUE_QUESTION_TYPES
        or any(hint in answer_type for hint in _VALUE_ANSWER_HINTS)
    )
    if not expects_value:
        return False
    return not re.search(r"\d", draft or "")


def assess_escalation(
    *,
    question: str,
    draft_answer: str,
    trust_signals: dict[str, Any],
    understanding: dict[str, Any] | None,
    tool_messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Decide how hard to work on an answer, from cheap signals only (no LLM).

    Level 0: accept the draft (well-grounded, or already resolved by a
        deterministic structured hint).
    Level 1: weakly grounded — try forced exact + supplemental retrieval, and
        only compose from evidence if that retrieval does not strengthen it.
    Level 2: the draft is a hedge/non-answer or is missing the value the
        question demands — retrieve and then compose from evidence.
    """
    if direct_answer_from_tool_messages(question, tool_messages):
        return {"level": 0, "reasons": ["deterministic structured hint already resolves the question"]}

    draft = (draft_answer or "").strip()
    signals = trust_signals or {}
    trust = float(signals.get("answer_trust_score") or 0.0)
    coverage = float(signals.get("citation_coverage") or 0.0)
    total_sources = int(signals.get("total_sources") or 0)

    hedged = not draft or bool(_HEDGE_RE.search(draft))
    shape_gap = _answer_shape_gap(question, draft, understanding)
    weak = (
        trust < _ESCALATION_TRUST_OK
        or coverage < _ESCALATION_COVERAGE_OK
        or total_sources < _ESCALATION_MIN_SOURCES
    )

    reasons: list[str] = []
    if hedged:
        reasons.append("draft is a hedge or non-answer")
    if shape_gap:
        reasons.append("draft lacks the number/value the question type expects")
    if weak:
        reasons.append(
            f"weak grounding (trust={trust:.2f}, coverage={coverage:.2f}, sources={total_sources})"
        )

    if hedged or shape_gap:
        return {"level": 2, "reasons": reasons}
    if weak:
        return {"level": 1, "reasons": reasons}
    return {"level": 0, "reasons": ["draft is well-grounded"]}


def accept_verified_answer(verification: dict[str, Any], draft_answer: str) -> bool:
    """Decide whether the verifier's revision should replace the draft answer.

    The verifier runs on a local model in every non-strict request and is the
    last gate before the user. It reliably fixes unsupported facts and strips
    hedging, but on a weak model it can also overwrite a usable draft with a
    low-confidence "needs more search" hedge. Adopt its revision unless it is
    signalling that it could not verify (needs_more_search at low or unknown
    confidence) — then keep the draft, unless the draft was empty.
    """
    revised = str(verification.get("final_answer") or "").strip()
    draft = (draft_answer or "").strip()
    if not revised or revised == draft:
        return False
    if not draft:
        return True
    status = str(verification.get("status") or "").lower()
    confidence = str(verification.get("confidence") or "").lower()
    if status == "needs_more_search" and confidence in {"low", "unknown"}:
        return False
    return True


_SHOOTOUT_RE = re.compile(r"\b(?:penalt(?:y|ies)|shoot[\s-]?out|on penalties)\b", re.IGNORECASE)
_DRAW_RE = re.compile(
    r"\b(?:tied|drew|draw|level|all\s+square|deadlock)\b|\b(\d{1,2})\s*[-–]\s*\1\b",
    re.IGNORECASE,
)
_LEADING_SCORELINE_RE = re.compile(
    r"^\s*[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,3}\s+(\d{1,2})\s*[-–]\s*(\d{1,2})\s+"
    r"[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,3}\s*[,.;:–-]\s*"
)


def _resolve_scoreline_contradiction(answer: str) -> str:
    """Drop a leading decisive scoreline that contradicts a stated shootout.

    A penalty shootout means the match was level after extra time, so an answer
    like "PSG 2-1 Arsenal, ... tied 1-1 ... PSG won 4-3 on penalties" is
    self-contradictory. Only fires when the text mentions BOTH a shootout and a
    level result, and only strips a leading "TeamA N-M TeamB" headline where
    N != M. It never rewrites or invents anything.
    """
    text = (answer or "").strip()
    if not text or not _SHOOTOUT_RE.search(text) or not _DRAW_RE.search(text):
        return answer
    match = _LEADING_SCORELINE_RE.match(text)
    if not match or int(match.group(1)) == int(match.group(2)):
        return answer
    remainder = text[match.end():].lstrip(" ,.;:–-")
    if not remainder:
        return answer
    return remainder[0].upper() + remainder[1:]


def _clean_verifier_answer(text: str) -> str:
    cleaned = sanitize_answer(text)
    cleaned = re.sub(r"\s+-\s+source:\s*https?://\S+\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = _resolve_scoreline_contradiction(cleaned)
    return cleaned


def verify_answer(
    *,
    question: str,
    answer: str,
    model_id: str,
    tool_messages: list[dict[str, str]],
    extraction: dict[str, Any],
    understanding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the draft answer against retrieved evidence and optionally revise."""
    direct_hint_result = _direct_hint_verification(question=question, tool_messages=tool_messages)
    if direct_hint_result:
        return direct_hint_result

    evidence_packet = {
        "question": question,
        "query_understanding": understanding or {},
        "draft_answer": answer,
        "direct_data_hints": _direct_hints(tool_messages),
        "sources": _compact_sources(extraction.get("sources", [])),
        "facts": (extraction.get("facts") or [])[:8],
        "trust_signals": extraction.get("trust_signals", {}),
    }

    system = SystemMessage(content=(
        "You are a factual verifier for a web research assistant. Return only "
        "valid JSON. Use only the provided evidence. Do not use outside "
        "knowledge. Your job is to improve correctness and confidence, not to "
        "make the answer timid. If evidence directly answers the question, "
        "start the final answer with that answer. A tie is a valid definitive "
        "answer: list all tied leaders and the shared value. Remove hedging "
        "like 'could not find a definitive answer' when the evidence contains "
        "the requested fact. If the draft contains a number/date/name not in "
        "evidence, replace it with the supported value. If the user asked for "
        "a specific rank or ordinal, the final_answer must answer that exact "
        "rank, not the overall winner or top entry. If a rank-specific "
        "DIRECT_DATA_HINT is present, prefer it. If sources have dates, prefer "
        "the newest relevant source for current/recent leaderboards. The final_answer must "
        "answer the user directly and concisely. Do not mention rejected or "
        "incorrect candidates unless the user asked about them or the sources "
        "show a real unresolved conflict. Do not copy literal DIRECT_DATA_HINT "
        "syntax or '- source: URL' text into the final answer. If a match was "
        "decided by a penalty shootout, it was level after extra time: report "
        "the level score and the shootout result, and never also state a "
        "decisive scoreline as the regular result. Never describe the draft, "
        "the evidence, or the verification process in the final_answer (for "
        "example 'the draft answer is not supported' or 'cannot be verified "
        "with the provided information'): either answer the question directly "
        "or state plainly that you could not find the information."
    ))
    human = HumanMessage(content=(
        "Audit this draft answer against the evidence packet below.\n\n"
        "Return JSON exactly with these fields:\n"
        "{\n"
        '  "status": "ok|revise|conflict|needs_more_search",\n'
        '  "final_answer": "concise user-facing answer",\n'
        '  "issues": ["actual issues found, or an empty array"],\n'
        '  "confidence": "low|medium|high"\n'
        "}\n\n"
        f"Evidence packet:\n{json.dumps(evidence_packet, ensure_ascii=False)}"
    ))

    try:
        raw = _get_llm(model_id).invoke([system, human])
        data = _extract_json_object(coerce_message_content(getattr(raw, "content", raw)))
    except Exception as exc:
        logger.info("Verifier pass failed: %s", exc)
        return _fallback_verification(question=question, answer=answer, tool_messages=tool_messages)

    if not data:
        return _fallback_verification(question=question, answer=answer, tool_messages=tool_messages)

    status = str(data.get("status") or "ok").strip().lower()
    if status not in {"ok", "revise", "conflict", "needs_more_search"}:
        status = "ok"
    final_answer = _clean_verifier_answer(str(data.get("final_answer") or answer).strip())
    if not final_answer:
        final_answer = answer
    confidence = str(data.get("confidence") or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "status": status,
        "final_answer": final_answer,
        "issues": _string_list(data.get("issues"), limit=6),
        "confidence": confidence,
    }
