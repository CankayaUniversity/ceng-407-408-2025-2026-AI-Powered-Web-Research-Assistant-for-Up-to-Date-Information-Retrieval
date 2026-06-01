"""Extra LLM inference passes for understanding and answer verification.

These calls do not perform retrieval and do not generate search queries. They
only help the main agent interpret the user's intent and then audit the draft
answer against evidence already retrieved by tools.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import MODEL_TEMPERATURE
from fact_extraction import coerce_message_content, sanitize_answer
from rank_utils import ordinal_label, requested_rank

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
    rank_candidates: list[tuple[int, str]] = []
    for hint in _direct_hints(tool_messages):
        ranked_leaderboard = re.search(
            r"Leaderboard rank\s+(\d{1,2})\s+candidate:\s*(.+?)\s+was listed\s+([0-9a-z]+)\s+with\s+(\d{1,3})\s+goals?",
            hint,
            re.IGNORECASE,
        )
        if ranked_leaderboard and wanted_rank is not None and int(ranked_leaderboard.group(1)) == wanted_rank:
            name = ranked_leaderboard.group(2).strip()
            goals = int(ranked_leaderboard.group(4))
            rank_candidates.append((goals, name))

    if rank_candidates and wanted_rank is not None:
        max_goals = max(goals for goals, _name in rank_candidates)
        deduped_names: list[str] = []
        for _goals, name in rank_candidates:
            if _goals == max_goals and name not in deduped_names:
                deduped_names.append(name)
        joined = " and ".join(deduped_names[:3])
        verb = "were" if len(deduped_names) > 1 else "was"
        return {
            "status": "revise",
            "final_answer": f"{joined} {verb} listed {ordinal_label(wanted_rank)} with {max_goals} goals.",
            "issues": ["Answered the requested rank from direct leaderboard evidence"],
            "confidence": "high",
        }

    for hint in _direct_hints(tool_messages):
        market = re.search(
            r"Structured market data:\s*([A-Z0-9.\-]+)\s+most recent daily close was\s+(\$?[0-9][0-9,]*(?:\.\d+)?)\s+on\s+(\d{4}-\d{2}-\d{2})",
            hint,
            re.IGNORECASE,
        )
        if market and ("close" in question.lower() or "stock" in question.lower()):
            symbol, price, day = market.group(1), market.group(2), market.group(3)
            return {
                "status": "revise",
                "final_answer": f"{symbol}'s most recent daily close was {price} on {day}.",
                "issues": ["Answered from structured market data"],
                "confidence": "high",
            }

    for hint in _direct_hints(tool_messages):
        fx = re.search(
            r"Structured FX data:\s*([A-Za-z]{3})/([A-Za-z]{3})\s+was\s+([0-9][0-9,]*(?:\.\d+)?)\s+on\s+(\d{4}-\d{2}-\d{2})",
            hint,
            re.IGNORECASE,
        )
        if fx:
            base, quote, rate, day = fx.group(1).upper(), fx.group(2).upper(), fx.group(3), fx.group(4)
            return {
                "status": "revise",
                "final_answer": f"1 {base} = {rate} {quote} (as of {day}).",
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
