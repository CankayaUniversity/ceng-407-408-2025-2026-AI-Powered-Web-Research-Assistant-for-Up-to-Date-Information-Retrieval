"""Heuristic claim extraction and trust-signal computation.

Pure-Python — no LLM call. Splits the answer into sentences, scores each
sentence against retrieved sources via token overlap, date matches, and
source-tier weight, then assembles continuous trust signals.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from source_quality import classify_source, tier_weight


@dataclass
class SourceEntry:
    url: str
    title: str
    domain: str
    source_tools: list[str]
    relevance_score: float | None = None
    snippet: str = ""
    tier: str = "low"


@dataclass
class FactItem:
    claim_text: str
    evidence_urls: list[str]
    source_tools: list[str]
    fact_quality_flags: dict[str, bool]


def _safe_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _claim_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]{4,}", text or "")
        if token.lower() not in {"with", "from", "that", "this", "when", "what", "where", "which"}
    }


_TIER_SCORE_BONUS = {"high": 0.45, "medium": 0.20, "low": 0.0, "prediction": -0.60}


def _score_source_for_claim(claim_text: str, source: "SourceEntry") -> float:
    claim_tokens = _claim_tokens(claim_text)
    if not claim_tokens:
        return 0.0

    source_text = f"{source.title} {source.snippet}".lower()
    overlap_count = sum(1 for token in claim_tokens if token in source_text)
    token_overlap_score = overlap_count / max(len(claim_tokens), 1)

    claim_dates = set(re.findall(r"\b(?:19|20)\d{2}\b", claim_text))
    source_dates = set(re.findall(r"\b(?:19|20)\d{2}\b", source_text))
    date_bonus = 0.35 if claim_dates and claim_dates.intersection(source_dates) else 0.0

    tavily_bonus = float(source.relevance_score) * 0.25 if source.relevance_score is not None else 0.0
    tier_bonus = _TIER_SCORE_BONUS.get(source.tier, 0.0)

    return token_overlap_score + date_bonus + tavily_bonus + tier_bonus


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
            current = {"url": "", "title": "", "content": "", "score": None, "_source_tier": "low"}
            tier_match = re.search(r"TIER:\s*(\w+)", line, re.IGNORECASE)
            if tier_match:
                current["_source_tier"] = tier_match.group(1).lower()
        elif current is not None:
            if line.startswith("TITLE: "):
                current["title"] = line[7:].strip()
            elif line.startswith("URL: "):
                current["url"] = line[5:].strip()
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
                continue

            snippet = str(item.get("content", ""))[:400]
            title = str(item.get("title", "")).strip()
            tier = item.get("_source_tier") or classify_source(url, title, snippet)

            by_url[url] = SourceEntry(
                url=url,
                title=title,
                domain=_safe_domain(url),
                source_tools=[tool_name] if tool_name else [],
                relevance_score=float(item["score"]) if isinstance(item.get("score"), (float, int)) else None,
                snippet=snippet,
                tier=tier,
            )

    return list(by_url.values())


def _extract_claims_from_answer(answer_text: str, sources: list["SourceEntry"]) -> list[dict[str, Any]]:
    """Split the answer into sentences and map each to its best-scoring sources."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[\.\!\?])\s+", answer_text or "")
        if sentence.strip()
    ]
    claims = []
    for sentence in sentences:
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
    """Max tier weight of this claim's supporting sources; 0.0 if uncited."""
    if not evidence_urls:
        return 0.0
    weights = [tier_weight(url_to_source[url].tier) for url in evidence_urls if url in url_to_source]
    return max(weights) if weights else 0.0


def extract_claims(answer_text: str, tool_messages: list[dict[str, str]]) -> dict[str, Any]:
    sources = normalize_sources(tool_messages)
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

    source_tier_weights = [tier_weight(s.tier) for s in sources]
    source_quality_score = (sum(source_tier_weights) / len(source_tier_weights)) if source_tier_weights else 0.0

    high_count = sum(1 for s in sources if s.tier == "high")
    medium_count = sum(1 for s in sources if s.tier == "medium")
    low_count = sum(1 for s in sources if s.tier == "low")
    prediction_count = sum(1 for s in sources if s.tier == "prediction")
    total_sources_count = len(sources)
    high_tier_ratio = (high_count / total_sources_count) if total_sources_count else 0.0

    per_claim_strengths = [_claim_citation_strength(fact.evidence_urls, url_to_source) for fact in facts]
    citation_strength = (sum(per_claim_strengths) / len(per_claim_strengths)) if per_claim_strengths else 0.0

    trust_signals = {
        "citation_coverage": (cited_claims / total_claims) if total_claims else 0.0,
        "avg_sources_per_claim": (source_count_sum / total_claims) if total_claims else 0.0,
        "multi_source_claim_ratio": (multi_source_claims / total_claims) if total_claims else 0.0,
        "distinct_domain_count": len(distinct_domains),
        "source_quality_score": round(source_quality_score, 3),
        "high_tier_ratio": round(high_tier_ratio, 3),
        "citation_strength": round(citation_strength, 3),
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
