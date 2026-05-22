"""Topic relevance scoring for search results (independent of domain tier).

Domain tier (high/medium/low) measures *reputation*, not whether a result
answers the user's question. premierleague.com can be HIGH while completely
off-topic for a Fenerbahçe query.
"""

from __future__ import annotations

import re
import unicodedata

# Minimum score to keep a result in agent context and UI sources (0–1).
MIN_RELEVANCE_SCORE = 0.22

_QUERY_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where",
    "which", "how", "who", "why", "about", "latest", "current", "recent",
    "bir", "ile", "için", "icin", "olan", "gibi", "daha", "son", "olan",
    "the", "are", "was", "were", "has", "have", "had", "does", "did",
})

_FOOTBALL_QUERY_RE = re.compile(
    r"\b("
    r"fenerbahçe|fenerbahce|galatasaray|beşiktaş|besiktas|trabzonspor|"
    r"football|futbol|maç|mac|match|skor|score|lig|league|uefa|şampiyon|"
    r"goal|gol|premier\s+league|süper\s+lig|super\s+lig"
    r")\b",
    re.IGNORECASE,
)

_BASKETBALL_RESULT_RE = re.compile(
    r"\b(beko|basketball|basket\s*bol|nba|euroligue|euroleague)\b",
    re.IGNORECASE,
)

_FPL_RESULT_RE = re.compile(
    r"\b(fantasy\s+premier|fantasy\s+football|\bfpl\b|gw\d{1,2}\b)\b",
    re.IGNORECASE,
)

_DIFFERENT_CLUB_RE = re.compile(
    r"\b(manchester\s+city|liverpool|arsenal|chelsea|barcelona|real\s+madrid)\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    # Fold Turkish chars for matching fenerbahce ↔ fenerbahçe
    return (
        lowered.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )


def query_tokens(query: str) -> set[str]:
    normalized = _normalize_text(query)
    words = re.findall(r"[a-z0-9ğüşıöç]{3,}", normalized, re.IGNORECASE)
    tokens: set[str] = set()
    for w in words:
        w = _normalize_text(w)
        if len(w) < 3 or w in _QUERY_STOPWORDS:
            continue
        tokens.add(w)
    return tokens


def relevance_score(
    query: str,
    title: str = "",
    snippet: str = "",
    url: str = "",
) -> float:
    """
    How well this result matches the search query / user question (0.0–1.0).
    """
    tokens = query_tokens(query)
    if not tokens:
        return 1.0

    hay = _normalize_text(f"{title} {snippet} {url}")
    if not hay.strip():
        return 0.0

    hits = sum(1 for t in tokens if t in hay)
    score = hits / max(len(tokens), 1)

    # Strong token match — boost
    if hits >= 2 and hits >= len(tokens) * 0.5:
        score = min(1.0, score + 0.15)

    q_norm = _normalize_text(query)
    title_snip = _normalize_text(f"{title} {snippet}")

    # Football question + basketball team page
    if _FOOTBALL_QUERY_RE.search(q_norm) and _BASKETBALL_RESULT_RE.search(title_snip):
        score *= 0.08

    # Fantasy Premier League noise when user did not ask for FPL
    if _FPL_RESULT_RE.search(title_snip) and not _FPL_RESULT_RE.search(q_norm):
        score *= 0.12

    # Mention of unrelated big clubs when query is about a Turkish club
    if _FOOTBALL_QUERY_RE.search(q_norm) and _DIFFERENT_CLUB_RE.search(title_snip):
        # Only penalize if query club name not in result
        club_in_query = any(
            c in q_norm
            for c in ("fenerbahce", "galatasaray", "besiktas", "trabzonspor")
        )
        club_in_result = any(
            c in title_snip
            for c in ("fenerbahce", "galatasaray", "besiktas", "trabzonspor")
        )
        if club_in_query and not club_in_result:
            score *= 0.2

    return round(min(1.0, max(0.0, score)), 3)


def filter_by_relevance(
    query: str,
    items: list[dict],
    *,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[dict]:
    """Drop off-topic results; attach _relevance_score to survivors."""
    kept: list[dict] = []
    for item in items:
        title = str(item.get("title") or "")
        content = str(item.get("content") or item.get("snippet") or "")
        url = str(item.get("url") or "")
        rel = relevance_score(query, title, content, url)
        if rel >= min_score:
            kept.append({**item, "_relevance_score": rel})
    return kept


def sort_by_relevance_then_tier(
    query: str,
    items: list[dict],
    tier_weight_fn,
    classify_fn,
) -> list[dict]:
    """Primary: topical relevance. Secondary: domain tier + search score."""

    def sort_key(item: dict) -> float:
        title = item.get("title", "") or ""
        content = item.get("content") or item.get("snippet") or ""
        url = item.get("url", "") or ""
        rel = item.get("_relevance_score")
        if rel is None:
            rel = relevance_score(query, title, content, url)
        tier = item.get("_source_tier") or classify_fn(url, title, content)
        tw = tier_weight_fn(tier)
        original = item.get("score")
        orig = float(original) if isinstance(original, (int, float)) else 0.0
        # Relevance dominates; tier breaks ties
        return rel * 100.0 + tw * 10.0 + orig

    return sorted(items, key=sort_key, reverse=True)
