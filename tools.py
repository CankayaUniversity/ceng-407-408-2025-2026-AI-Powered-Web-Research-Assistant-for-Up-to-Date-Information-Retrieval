import json
import re

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from config import (
    DEEP_READER_MAX_CHARS,
    DEEP_READER_TIMEOUT_SECONDS,
    DUCKDUCKGO_MAX_RESULTS,
    TAVILY_MAX_RESULTS,
)
from source_quality import classify_source, sort_results


_tavily_underlying = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
_duckduckgo_underlying = DuckDuckGoSearchResults(num_results=DUCKDUCKGO_MAX_RESULTS)


def _annotate_tavily_items(items: list[dict]) -> list[dict]:
    sorted_items = sort_results(items)
    annotated = []
    for item in sorted_items:
        url = item.get("url", "") or ""
        title = item.get("title", "") or ""
        content = item.get("content", "") or ""
        tier = classify_source(url, title, content)
        annotated.append({**item, "_source_tier": tier})
    return annotated


def _parse_ddg_segments(text: str) -> list[dict]:
    """Parse DuckDuckGo's textual `snippet: ..., title: ..., link: ...` format."""
    if not text:
        return []
    segments = re.split(r"(?=snippet:\s)", text)
    items = []
    for seg in segments:
        seg = seg.strip().lstrip(",").strip()
        if not seg:
            continue
        link_match = re.search(r"link:\s*(https?://[^\s,]+)", seg, flags=re.IGNORECASE)
        if not link_match:
            continue
        title_match = re.search(r"title:\s*(.*?)(?:,\s*link:|$)", seg, flags=re.IGNORECASE | re.DOTALL)
        snippet_match = re.search(r"snippet:\s*(.*?)(?:,\s*title:|$)", seg, flags=re.IGNORECASE | re.DOTALL)
        items.append(
            {
                "url": link_match.group(1).strip(),
                "title": title_match.group(1).strip() if title_match else "",
                "snippet": snippet_match.group(1).strip() if snippet_match else seg,
            }
        )
    return items


def _format_ddg_with_tiers(items: list[dict]) -> str:
    parts = []
    for item in items:
        tier = classify_source(item.get("url", ""), item.get("title", ""), item.get("snippet", ""))
        parts.append(
            f"[TIER: {tier.upper()}] snippet: {item.get('snippet', '')}, "
            f"title: {item.get('title', '')}, link: {item.get('url', '')}"
        )
    return ", ".join(parts)


@tool
def tavily_search_results_json(query: str) -> str:
    """Search the web via Tavily for current, authoritative information. Results are
    returned as JSON, sorted so the highest-quality sources appear first. Each item
    has a `_source_tier` field with one of these values:
      - "high"       — Wikipedia, official org sites (uefa.com, fifa.com, nasa.gov),
                       major newswires (Reuters, AP), major newspapers (BBC, NYT).
                       Use these for factual claims.
      - "medium"     — Mainstream news (CNN, Bloomberg), established sports news
                       (ESPN), trusted tech publications. Acceptable for facts.
      - "low"        — Unknown or unfamiliar sites. Use only if no higher-tier
                       source is available and corroborate with another source.
      - "prediction" — Betting/odds sites, "who will win" articles, previews of
                       upcoming events. NEVER cite these as facts. They describe
                       events that have NOT yet happened.
    """
    try:
        raw = _tavily_underlying.invoke({"query": query})
    except Exception as exc:
        return f"Tavily search error: {exc}"
    if isinstance(raw, str):
        try:
            items = json.loads(raw)
        except Exception:
            return raw
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    if not isinstance(items, list):
        return json.dumps(items, ensure_ascii=False)
    annotated = _annotate_tavily_items(items)
    return json.dumps(annotated, ensure_ascii=False)


@tool
def duckduckgo_results_json(query: str) -> str:
    """Search the web via DuckDuckGo. Returns results sorted with the highest-quality
    sources first. Each result is prefixed with [TIER: HIGH|MEDIUM|LOW|PREDICTION].
    Same tier semantics as the Tavily tool — never treat PREDICTION-tier results as
    facts; they describe events that have not yet happened.
    """
    try:
        raw = _duckduckgo_underlying.invoke({"query": query})
    except Exception as exc:
        return f"DuckDuckGo search error: {exc}"
    raw_str = str(raw) if raw is not None else ""
    items = _parse_ddg_segments(raw_str)
    if not items:
        return raw_str
    items = sort_results(items)
    return _format_ddg_with_tiers(items)


@tool
def deep_site_reader(url: str) -> str:
    """Reads and extracts visible text from a specific URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=DEEP_READER_TIMEOUT_SECONDS, verify=False)
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()

        text_content = soup.get_text(separator=" ", strip=True)
        return text_content[:DEEP_READER_MAX_CHARS]
    except Exception as e:
        return f"This site could not be accessed (Error: {str(e)}). Please use Tavily summary results."


def build_tools():
    return [tavily_search_results_json, duckduckgo_results_json, deep_site_reader]
