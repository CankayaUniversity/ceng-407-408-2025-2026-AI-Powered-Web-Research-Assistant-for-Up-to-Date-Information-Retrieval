"""Heuristic claim extraction and trust-signal computation.

Pure-Python — no LLM call. Splits the answer into sentences, scores each
sentence against retrieved sources via token overlap, date matches, and
source reliability, then assembles continuous trust signals.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from source_quality import source_reliability
from source_relevance import MIN_RELEVANCE_SCORE, relevance_score


@dataclass
class SourceEntry:
    url: str
    title: str
    domain: str
    source_tools: list[str]
    relevance_score: float | None = None
    reliability_score: float = 0.35
    reliability_label: str = "Low"
    reliability_reasons: list[str] | None = None
    snippet: str = ""
    tier: str = "low"


@dataclass
class FactItem:
    claim_text: str
    evidence_urls: list[str]
    source_tools: list[str]
    fact_quality_flags: dict[str, Any]


def _safe_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _claim_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]{4,}", text or "")
        if token.lower() not in {"with", "from", "that", "this", "when", "what", "where", "which"}
    }


def _claim_source_overlap(claim_text: str, source: "SourceEntry") -> float:
    claim_tokens = _claim_tokens(claim_text)
    if not claim_tokens:
        return 0.0
    source_text = f"{source.title} {source.snippet}".lower()
    overlap_count = sum(1 for token in claim_tokens if token in source_text)
    return overlap_count / max(len(claim_tokens), 1)


def _claim_source_date_match(claim_text: str, source: "SourceEntry") -> bool:
    claim_dates = set(re.findall(r"\b(?:19|20)\d{2}\b", claim_text))
    if not claim_dates:
        return False
    source_text = f"{source.title} {source.snippet}".lower()
    source_dates = set(re.findall(r"\b(?:19|20)\d{2}\b", source_text))
    return bool(claim_dates.intersection(source_dates))


def _score_source_for_claim(claim_text: str, source: "SourceEntry") -> float:
    claim_tokens = _claim_tokens(claim_text)
    if not claim_tokens:
        return 0.0

    token_overlap_score = _claim_source_overlap(claim_text, source)
    date_bonus = 0.35 if _claim_source_date_match(claim_text, source) else 0.0

    tavily_bonus = float(source.relevance_score) * 0.25 if source.relevance_score is not None else 0.0
    reliability_bonus = max(0.0, min(1.0, source.reliability_score)) * 0.45

    return token_overlap_score + date_bonus + tavily_bonus + reliability_bonus


def _parse_search_results_text(tool_content: str) -> list[dict[str, Any]]:
    """Parse the unified RESULT N — TIER: X format emitted by tools.py."""
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    if not tool_content:
        return items
    for line in tool_content.split("\n"):
        if line.startswith("RESULT "):
            if current and current.get("url"):
                items.append(current)
            current = {
                "url": "",
                "title": "",
                "content": "",
                "score": None,
                "_source_tier": "low",
                "_source_reliability_score": None,
                "_source_reliability_reasons": [],
            }
            tier_match = re.search(r"TIER:\s*(\w+)", line, re.IGNORECASE)
            if tier_match:
                current["_source_tier"] = tier_match.group(1).lower()
            rel_match = re.search(r"RELEVANCE:\s*([0-9.]+)", line, re.IGNORECASE)
            if rel_match:
                try:
                    current["score"] = float(rel_match.group(1))
                except ValueError:
                    pass
            reliability_match = re.search(r"RELIABILITY:\s*([0-9.]+)", line, re.IGNORECASE)
            if reliability_match:
                try:
                    current["_source_reliability_score"] = float(reliability_match.group(1))
                except ValueError:
                    pass
        elif current is not None:
            if line.startswith("TITLE: "):
                current["title"] = line[7:].strip()
            elif line.startswith("URL: "):
                current["url"] = line[5:].strip()
            elif line.startswith("RELIABILITY_SIGNALS: "):
                current["_source_reliability_reasons"] = [
                    part.strip()
                    for part in line[21:].split(";")
                    if part.strip()
                ]
            elif line.startswith("CONTENT: "):
                body = line[9:].strip()
                if body.startswith("(empty"):
                    body = ""
                current["content"] = body
    if current and current.get("url"):
        items.append(current)
    return items


def normalize_sources(tool_messages: list[dict[str, str]]) -> list["SourceEntry"]:
    by_url: dict[str, SourceEntry] = {}

    for message in tool_messages:
        tool_name = message.get("name", "")
        content = message.get("content", "")

        if tool_name in ("tavily_search_results_json", "duckduckgo_results_json"):
            parsed_items = _parse_search_results_text(content)
        elif tool_name == "deep_site_reader":
            parsed_items = [{"title": "Deep page read", "url": "", "content": content, "score": None, "_source_tier": "low"}]
        else:
            parsed_items = []

        for item in parsed_items:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            existing = by_url.get(url)
            if existing:
                if tool_name and tool_name not in existing.source_tools:
                    existing.source_tools.append(tool_name)
                if existing.relevance_score is None and isinstance(item.get("score"), (float, int)):
                    existing.relevance_score = float(item["score"])
                if not existing.snippet and item.get("content"):
                    existing.snippet = str(item["content"])[:400]
                if not existing.reliability_reasons and item.get("_source_reliability_reasons"):
                    existing.reliability_reasons = list(item["_source_reliability_reasons"])
                continue

            snippet = str(item.get("content", ""))[:400]
            title = str(item.get("title", "")).strip()
            reliability = source_reliability(url, title, snippet)
            tier = item.get("_source_tier") or reliability["tier"]
            reliability_score = (
                float(item["_source_reliability_score"])
                if isinstance(item.get("_source_reliability_score"), (float, int))
                else float(reliability["score"])
            )
            reliability_reasons = (
                list(item.get("_source_reliability_reasons") or [])
                or list(reliability["reasons"])
            )

            by_url[url] = SourceEntry(
                url=url,
                title=title,
                domain=_safe_domain(url),
                source_tools=[tool_name] if tool_name else [],
                relevance_score=float(item["score"]) if isinstance(item.get("score"), (float, int)) else None,
                reliability_score=max(0.0, min(1.0, reliability_score)),
                reliability_label=str(reliability.get("label") or ""),
                reliability_reasons=reliability_reasons,
                snippet=snippet,
                tier=tier,
            )

    return list(by_url.values())


_SOURCES_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?"
    r"(?:\*\*)?"
    r"(?:Sources?|References?|Citations?|See\s+also|Read\s+more|Further\s+reading)"
    r"(?:\*\*)?"
    r"\s*:?\s*\n[\s\S]*$",
    re.IGNORECASE,
)

_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_BARE_URL_RE = re.compile(r"https?://\S+")
_LIST_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_NON_CLAIM_PREFIX_RE = re.compile(
    r"^\s*(?:Sources?|References?|Citations?|Source|Citation|See\s+also|Read\s+more)\s*:",
    re.IGNORECASE,
)
# Generic advice / CTAs / tool meta — not verifiable factual claims.
_NON_FACTUAL_CLAIM_RES = (
    re.compile(
        r"^\s*(?:please\s+)?(?:contact|reach out to|check with|verify with|confirm with|consult)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfor more (?:details|information|info)\b", re.IGNORECASE),
    re.compile(r"\b(?:it is|it's) recommended to\b", re.IGNORECASE),
    re.compile(r"^\s*(?:\*\*)?(?:note|disclaimer)(?:\*\*)?\s*:", re.IGNORECASE),
    re.compile(
        r"\b(?:double[- ]check|cross[- ]check).*\b(?:official|primary)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:tavily|duckduckgo)\b", re.IGNORECASE),
    re.compile(r"\bbased on (?:the )?search results\b", re.IGNORECASE),
    re.compile(r"\bsubject to change as new data becomes available\b", re.IGNORECASE),
    re.compile(r"\b(?:verify|confirm) (?:with|using) (?:official|primary) sources\b", re.IGNORECASE),
)

_META_NOTE_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:note|disclaimer)(?:\*\*)?\s*:[\s\S]*$",
    re.IGNORECASE,
)


def _strip_sources_section(text: str) -> str:
    """Remove a trailing 'Sources:' / 'References:' block — it's metadata,
    not a factual claim, and confuses the sentence splitter."""
    return _SOURCES_SECTION_RE.sub("", text or "").strip()


def filter_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop meta/advisory sentences that should not appear as extracted facts."""
    return [
        fact
        for fact in facts
        if not _is_non_claim(str(fact.get("claim_text", "")))
    ]


STRICT_NO_CLAIMS_MESSAGE = (
    "No statements from the research pass could be matched to retrieved sources. "
    "Try rephrasing the question, or turn off strict mode for a full model-written summary."
)


def cited_facts_only(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Facts that have at least one supporting source URL attached."""
    cited: list[dict[str, Any]] = []
    for fact in facts or []:
        flags = fact.get("fact_quality_flags") or {}
        urls = fact.get("evidence_urls") or []
        if not flags.get("has_source") or not urls:
            continue
        text = str(fact.get("claim_text", "")).strip()
        if not text or _is_non_claim(text):
            continue
        cited.append(fact)
    return cited


def compose_strict_answer(facts: list[dict[str, Any]]) -> str:
    """Build the user-visible answer by joining only source-backed extracted claims."""
    seen: set[str] = set()
    parts: list[str] = []
    for fact in cited_facts_only(facts):
        text = str(fact.get("claim_text", "")).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        if text and text[-1] not in ".!?":
            text += "."
        parts.append(text)
    if not parts:
        return STRICT_NO_CLAIMS_MESSAGE
    return " ".join(parts)


def coerce_message_content(content: object) -> str:
    """Normalize LangChain message content (str or block list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def sanitize_answer(text: str) -> str:
    """Remove trailing Note/Disclaimer blocks and tool-meta sentences from the answer."""
    if not isinstance(text, str):
        text = coerce_message_content(text)
    cleaned = _strip_sources_section(text or "")
    cleaned = _META_NOTE_BLOCK_RE.sub("", cleaned).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[\.\!\?])\s+|\n+", cleaned)
        if sentence.strip()
    ]
    kept = [sentence for sentence in sentences if not _is_non_claim(sentence)]
    result = " ".join(kept).strip()
    # If every sentence was classified as meta/noise but prose remained after
    # stripping Sources/Note blocks, keep that prose rather than an empty UI.
    if not result and cleaned.strip():
        return cleaned.strip()
    return result


def _is_non_claim(sentence: str) -> bool:
    """True when a sentence is markdown noise (a sources header, a bare
    link line, a bullet whose body is just a markdown link, or generic advice)."""
    s = (sentence or "").strip()
    if not s:
        return True
    if _NON_CLAIM_PREFIX_RE.match(s):
        return True
    if any(pattern.search(s) for pattern in _NON_FACTUAL_CLAIM_RES):
        return True
    # Strip away link markup and list bullets; what remains should be prose.
    residual = _MARKDOWN_LINK_RE.sub("", s)
    residual = _BARE_URL_RE.sub("", residual)
    residual = _LIST_BULLET_RE.sub("", residual)
    residual = re.sub(r"[\s,\-:*•]+", "", residual)
    # Fewer than ~8 non-link characters → almost certainly not a claim.
    if len(residual) < 8:
        return True
    return False


def _extract_claims_from_answer(answer_text: str, sources: list["SourceEntry"]) -> list[dict[str, Any]]:
    """Split the answer into sentences and map each to its best-scoring sources."""
    cleaned_answer = sanitize_answer(answer_text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[\.\!\?])\s+|\n+", cleaned_answer)
        if sentence.strip()
    ]
    claims = []
    for sentence in sentences:
        if _is_non_claim(sentence):
            continue
        scored = [
            (source.url, _score_source_for_claim(sentence, source))
            for source in sources
            if source.url
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        evidence_urls = [url for url, score in scored[:3] if score >= 0.08]
        claims.append({"claim_text": sentence, "evidence_urls": evidence_urls})
    return claims


def _claim_citation_strength(evidence_urls: list[str], url_to_source: dict[str, "SourceEntry"]) -> float:
    """Max source reliability score of this claim's supporting sources; 0.0 if uncited."""
    if not evidence_urls:
        return 0.0
    weights = [url_to_source[url].reliability_score for url in evidence_urls if url in url_to_source]
    return max(weights) if weights else 0.0


def extract_claims(
    answer_text: str,
    tool_messages: list[dict[str, str]],
    question: str | None = None,
) -> dict[str, Any]:
    sources = normalize_sources(tool_messages)
    if question:
        sources = [
            s
            for s in sources
            if (
                s.relevance_score
                if isinstance(s.relevance_score, (int, float))
                else relevance_score(
                    question,
                    s.title,
                    s.snippet,
                    s.url,
                )
            )
            >= MIN_RELEVANCE_SCORE
        ]
    raw_claims = _extract_claims_from_answer(answer_text, sources)

    url_to_source = {source.url: source for source in sources}
    facts: list[FactItem] = []

    for claim in raw_claims:
        claim_text = str(claim.get("claim_text", "")).strip()
        if not claim_text:
            continue
        evidence_urls = [url for url in claim.get("evidence_urls", []) if url in url_to_source]
        tools = sorted({
            tool
            for url in evidence_urls
            if url in url_to_source
            for tool in url_to_source[url].source_tools
        })
        has_source = len(evidence_urls) > 0
        multi_source = len({_safe_domain(url) for url in evidence_urls if _safe_domain(url)}) >= 2
        weak_evidence = not has_source or len(evidence_urls) == 1
        needs_review = not has_source or weak_evidence

        facts.append(
            FactItem(
                claim_text=claim_text,
                evidence_urls=evidence_urls,
                source_tools=tools,
                fact_quality_flags={
                    "has_source": has_source,
                    "multi_source": multi_source,
                    "weak_evidence": weak_evidence,
                    "needs_review": needs_review,
                },
            )
        )

    total_claims = len(facts)
    cited_claims = sum(1 for fact in facts if fact.fact_quality_flags["has_source"])
    multi_source_claims = sum(1 for fact in facts if fact.fact_quality_flags["multi_source"])
    source_count_sum = sum(len(fact.evidence_urls) for fact in facts)
    distinct_domains = sorted({source.domain for source in sources if source.domain})

    source_scores = [s.reliability_score for s in sources]
    source_quality_score = (sum(source_scores) / len(source_scores)) if source_scores else 0.0

    high_count = sum(1 for s in sources if s.tier == "high")
    medium_count = sum(1 for s in sources if s.tier == "medium")
    low_count = sum(1 for s in sources if s.tier == "low")
    prediction_count = sum(1 for s in sources if s.tier == "prediction")
    total_sources_count = len(sources)
    high_tier_ratio = (high_count / total_sources_count) if total_sources_count else 0.0

    per_claim_strengths = [_claim_citation_strength(fact.evidence_urls, url_to_source) for fact in facts]
    citation_strength = (sum(per_claim_strengths) / len(per_claim_strengths)) if per_claim_strengths else 0.0

    citation_coverage = (cited_claims / total_claims) if total_claims else 0.0
    multi_source_claim_ratio = (multi_source_claims / total_claims) if total_claims else 0.0
    source_quality_score_r = round(source_quality_score, 3)
    citation_strength_r = round(citation_strength, 3)

    # Composite 0–1 score for UI summary (weights emphasize sources + citations).
    answer_trust_score = round(
        0.30 * source_quality_score_r
        + 0.30 * citation_strength_r
        + 0.25 * citation_coverage
        + 0.15 * multi_source_claim_ratio,
        3,
    )

    trust_signals = {
        "answer_trust_score": answer_trust_score,
        "citation_coverage": citation_coverage,
        "unsupported_claims": total_claims - cited_claims,
        "avg_sources_per_claim": (source_count_sum / total_claims) if total_claims else 0.0,
        "multi_source_claim_ratio": multi_source_claim_ratio,
        "distinct_domain_count": len(distinct_domains),
        "source_quality_score": source_quality_score_r,
        "source_reliability_score": source_quality_score_r,
        "high_tier_ratio": round(high_tier_ratio, 3),
        "citation_strength": citation_strength_r,
        "tier_breakdown": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "prediction": prediction_count,
        },
        "total_sources": total_sources_count,
    }

    return {
        "facts": [asdict(fact) for fact in facts],
        "sources": [asdict(source) for source in sources],
        "trust_signals": trust_signals,
    }
