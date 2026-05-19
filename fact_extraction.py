import json
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_ollama import ChatOllama

from config import MODEL_NAME
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


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s,\]\)]+", text or "")


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

    # Boost date matches for fact-like claims.
    claim_dates = set(re.findall(r"\b(?:19|20)\d{2}\b", claim_text))
    source_dates = set(re.findall(r"\b(?:19|20)\d{2}\b", source_text))
    date_bonus = 0.35 if claim_dates and claim_dates.intersection(source_dates) else 0.0

    tavily_bonus = float(source.relevance_score) * 0.25 if source.relevance_score is not None else 0.0

    # Use pre-computed tier (avoids redundant classify_source call)
    tier_bonus = _TIER_SCORE_BONUS.get(source.tier, 0.0)

    return token_overlap_score + date_bonus + tavily_bonus + tier_bonus


def _enrich_evidence_urls(claim_text: str, initial_urls: list[str], sources: list["SourceEntry"]) -> list[str]:
    """Keep LLM-predicted URLs first, then add highest-scoring sources from distinct domains."""
    ordered_urls: list[str] = []
    seen: set[str] = set()
    for url in initial_urls:
        if url and url not in seen:
            ordered_urls.append(url)
            seen.add(url)

    scored_sources = [
        (source, _score_source_for_claim(claim_text, source))
        for source in sources
        if source.url and source.url not in seen
    ]
    scored_sources.sort(key=lambda item: item[1], reverse=True)

    used_domains = {_safe_domain(url) for url in ordered_urls if _safe_domain(url)}
    for source, score in scored_sources:
        if score < 0.10:  # Lowered from 0.20 — catches medium-relevance sources
            break
        domain = source.domain
        # Enforce domain diversity once we already have 2+ URLs
        if domain and domain in used_domains and len(ordered_urls) >= 2:
            continue
        ordered_urls.append(source.url)
        if domain:
            used_domains.add(domain)
        if len(ordered_urls) >= 3:
            break

    return ordered_urls


def _parse_tavily_results(tool_content: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(tool_content)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict) and item.get("url")]
    except Exception:
        return []
    return []


def _parse_duckduckgo_results(tool_content: str) -> list[dict[str, Any]]:
    items = []
    if not tool_content:
        return items

    segments = [segment.strip() for segment in tool_content.split("snippet: ") if segment.strip()]
    for segment in segments:
        title_match = re.search(r"title:\s*(.*?),\s*link:", segment, flags=re.IGNORECASE | re.DOTALL)
        link_match = re.search(r"link:\s*(https?://[^\s,]+)", segment, flags=re.IGNORECASE)
        snippet = segment.split(", title:", 1)[0].strip()
        if not link_match:
            continue
        items.append(
            {
                "title": title_match.group(1).strip() if title_match else "",
                "url": link_match.group(1).strip(),
                "content": snippet,
                "score": None,
            }
        )
    return items


def normalize_sources(tool_messages: list[dict[str, str]]) -> list["SourceEntry"]:
    by_url: dict[str, SourceEntry] = {}

    for message in tool_messages:
        tool_name = message.get("name", "")
        content = message.get("content", "")

        if tool_name == "tavily_search_results_json":
            parsed_items = _parse_tavily_results(content)
        elif tool_name == "duckduckgo_results_json":
            parsed_items = _parse_duckduckgo_results(content)
        elif tool_name == "deep_site_reader":
            parsed_items = [{"title": "Deep page read", "url": "", "content": content, "score": None}]
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
            tier = classify_source(url, title, snippet)

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


def _json_from_text(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def _llm_extract_claims(answer_text: str, sources: list["SourceEntry"], model_id: str = MODEL_NAME) -> list[dict[str, Any]]:
    model = ChatOllama(model=model_id, temperature=0)
    source_lines = [f"- {s.url} | {s.title}" for s in sources[:15]]
    prompt = (
        "Extract factual claims from the answer and map each claim to supporting source URLs.\n"
        "Return ONLY valid JSON as a list of objects in this exact format:\n"
        '[{"claim_text":"...","evidence_urls":["https://..."]}]\n'
        "If unsure, return an empty evidence_urls list for that claim.\n\n"
        f"Answer:\n{answer_text}\n\n"
        f"Available sources:\n{chr(10).join(source_lines)}"
    )
    response = model.invoke(prompt)
    parsed = _json_from_text(str(response.content))
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict) and item.get("claim_text")]
    return []


def _fallback_extract_claims(answer_text: str, sources: list["SourceEntry"]) -> list[dict[str, Any]]:
    """Heuristic fallback: split into sentences, score each source per sentence.

    Unlike the old approach (source_urls[:1] for all), this assigns each claim
    the sources that best match it based on token overlap, dates, and tier — so
    different claims get different supporting sources, producing continuous metrics.
    """
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
        # Keep top-3 sources above a low threshold; claims that don't overlap stay uncited
        evidence_urls = [url for url, score in scored[:3] if score >= 0.08]
        claims.append({"claim_text": sentence, "evidence_urls": evidence_urls})
    return claims


def _claim_citation_strength(evidence_urls: list[str], url_to_source: dict[str, "SourceEntry"]) -> float:
    """Max tier weight of this claim's supporting sources; 0.0 if uncited."""
    if not evidence_urls:
        return 0.0
    weights = [tier_weight(url_to_source[url].tier) for url in evidence_urls if url in url_to_source]
    return max(weights) if weights else 0.0


def extract_claims(answer_text: str, tool_messages: list[dict[str, str]], model_id: str = MODEL_NAME) -> dict[str, Any]:
    sources = normalize_sources(tool_messages)

    try:
        raw_claims = _llm_extract_claims(answer_text, sources, model_id)
    except Exception:
        raw_claims = []
    if not raw_claims:
        raw_claims = _fallback_extract_claims(answer_text, sources)

    url_to_source = {source.url: source for source in sources}
    facts: list[FactItem] = []

    for claim in raw_claims:
        claim_text = str(claim.get("claim_text", "")).strip()
        if not claim_text:
            continue
        llm_urls = [url for url in claim.get("evidence_urls", []) if url in url_to_source]
        evidence_urls = _enrich_evidence_urls(claim_text, llm_urls, sources)
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

    # ── Continuous source-quality metrics ────────────────────────────────────
    source_tier_weights = [tier_weight(s.tier) for s in sources]
    source_quality_score = (sum(source_tier_weights) / len(source_tier_weights)) if source_tier_weights else 0.0

    high_count = sum(1 for s in sources if s.tier == "high")
    medium_count = sum(1 for s in sources if s.tier == "medium")
    low_count = sum(1 for s in sources if s.tier == "low")
    prediction_count = sum(1 for s in sources if s.tier == "prediction")
    total_sources_count = len(sources)
    high_tier_ratio = (high_count / total_sources_count) if total_sources_count else 0.0

    # Per-claim citation strength: max tier weight of supporting sources, averaged
    per_claim_strengths = [_claim_citation_strength(fact.evidence_urls, url_to_source) for fact in facts]
    citation_strength = (sum(per_claim_strengths) / len(per_claim_strengths)) if per_claim_strengths else 0.0

    trust_signals = {
        # Coverage metrics (now continuous because fallback no longer forces a single source)
        "citation_coverage": (cited_claims / total_claims) if total_claims else 0.0,
        "avg_sources_per_claim": (source_count_sum / total_claims) if total_claims else 0.0,
        "multi_source_claim_ratio": (multi_source_claims / total_claims) if total_claims else 0.0,
        "distinct_domain_count": len(distinct_domains),
        # Quality metrics: reflect the reputation tier of retrieved sources
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
