import json
import logging
import re
import threading
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("deep_research")

from config import (
    DEEP_READER_MAX_CHARS,
    DEEP_READER_TIMEOUT_SECONDS,
    DUCKDUCKGO_MAX_RESULTS,
    TAVILY_MAX_RESULTS,
)
from source_quality import classify_source, sort_results


_tavily_underlying = None
_duckduckgo_underlying = None
_client_lock = threading.Lock()


# --------------------------------------------------------------------------
# Defensive input coercion
# --------------------------------------------------------------------------
# Llama 3.1's tool calling sometimes hallucinates the JSON schema definition
# into the argument value — passing {"type": "string"} instead of the actual
# query — and invents extra parameters like {"object": "<nil>", "url": "<nil>"}.
# Qwen 2.5 and Llama 3.2 don't do this. To keep all three models working from
# the same tool definitions, we coerce malformed inputs into clean strings
# before they reach the search clients.

_SCHEMA_LEAK_VALUES = {"string", "query", "<nil>", "nil", "null", "none"}


def _sanitize_string_arg(value: Any) -> str:
    """Extract a usable string from whatever shape the LLM hands us."""
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in _SCHEMA_LEAK_VALUES:
            return cleaned
        return ""

    if isinstance(value, dict):
        # Common schema-leak pattern from Llama 3.1: {"type": "string"} or
        # {"type": "string", "description": "..."} as the VALUE of `query`.
        if "type" in value and value.get("type") in {"string", "str"}:
            return ""
        # Sometimes the actual query is nested under another key
        for nested_key in ("query", "q", "search", "input", "text", "value", "url"):
            nested = value.get(nested_key)
            if isinstance(nested, str):
                nested_clean = nested.strip()
                if nested_clean and nested_clean.lower() not in _SCHEMA_LEAK_VALUES:
                    return nested_clean
        return ""

    if isinstance(value, list) and value:
        return _sanitize_string_arg(value[0])

    return ""


class SearchToolInput(BaseModel):
    """Input schema for web-search tools.

    Uses a pre-validator to swallow Llama 3.1's schema-leak hallucinations
    (e.g. passing {"type": "string"} as the query value).
    """

    query: str = Field(..., description="The search query as a plain string.")

    @field_validator("query", mode="before")
    @classmethod
    def coerce_query(cls, v):
        return _sanitize_string_arg(v)


class DeepReaderInput(BaseModel):
    """Input schema for deep_site_reader with the same defensive coercion."""

    url: str = Field(..., description="The URL to read as a plain string.")

    @field_validator("url", mode="before")
    @classmethod
    def coerce_url(cls, v):
        return _sanitize_string_arg(v)


# --------------------------------------------------------------------------
# Deterministic relative-date substitution
# --------------------------------------------------------------------------
# The system prompt tells the agent to translate "yesterday", "today", etc.
# into specific dates before searching. Small models ignore this. Tavily and
# DDG cannot interpret the literal word "yesterday" — they pattern-match text,
# so an article from any past day matches. To stop the agent from passing
# week-old results off as "yesterday's", we substitute deterministically here
# right before the query hits the search engine.

_REL_DATE_TOKEN_RE = re.compile(r"\b(yesterday|today|tonight|tomorrow)\b", re.IGNORECASE)


def _substitute_relative_dates(query: str) -> str:
    """Replace relative time words with the absolute calendar date.

    Example: "premier league results yesterday" -> "premier league results May 19, 2026".
    The natural-language format ("May 19, 2026") matches how news articles
    typically date their content, giving better search recall than ISO format.
    """
    if not query or not _REL_DATE_TOKEN_RE.search(query):
        return query
    today = date.today()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    mapping = {
        "yesterday": yesterday.strftime("%B %d, %Y"),
        "today":     today.strftime("%B %d, %Y"),
        "tonight":   today.strftime("%B %d, %Y"),
        "tomorrow":  tomorrow.strftime("%B %d, %Y"),
    }
    def _sub(match: re.Match) -> str:
        return mapping[match.group(1).lower()]
    return _REL_DATE_TOKEN_RE.sub(_sub, query)


def _get_tavily():
    global _tavily_underlying
    if _tavily_underlying is None:
        with _client_lock:
            if _tavily_underlying is None:  # double-checked locking
                _tavily_underlying = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
    return _tavily_underlying


def _get_duckduckgo():
    global _duckduckgo_underlying
    if _duckduckgo_underlying is None:
        with _client_lock:
            if _duckduckgo_underlying is None:
                _duckduckgo_underlying = DuckDuckGoSearchResults(num_results=DUCKDUCKGO_MAX_RESULTS)
    return _duckduckgo_underlying


# --------------------------------------------------------------------------
# Content cleanup
# --------------------------------------------------------------------------
# Tavily and DuckDuckGo extract page text by stripping HTML, which leaves
# behind alt-text placeholders ("imgalt"), image labels ("Image 1"), nav
# scraps ("Show more games", "Advertisement"), and form strings like "WWWLL"
# that look meaningless to an LLM. Removing this noise BEFORE the agent reads
# the content reduces hallucination and makes source snippets readable in
# the UI.

_JUNK_PATTERNS = [
    re.compile(r"\bimgalt\b", re.IGNORECASE),
    re.compile(r"\bImage\s*\d+(?:\s*:[^\n,]{0,60})?", re.IGNORECASE),
    re.compile(r"\bAdvertisement\b", re.IGNORECASE),
    re.compile(r"\bShow more (?:games|matches|results)\b", re.IGNORECASE),
    re.compile(r"\bNo data\b"),
    re.compile(r"\bWatch Live Stream(?: For Free)?\b", re.IGNORECASE),
    re.compile(r"\bSubscribe to .{0,40}\b", re.IGNORECASE),
    # JS rendering failures (e.g. "undefined undefined")
    re.compile(r"\bundefined\b", re.IGNORECASE),
    # Stripped JS closures (e.g. `")` ` "),` artifacts)
    re.compile(r'"\)\s*[,.]?'),
    # Form strings like "WWWLL", "?WWTWW" that bleed in from fixture tables
    re.compile(r"\?[WLTD]{3,}\b"),
    re.compile(r"\b[WLTD]{4,}\b"),
    # Doubled latin words from CSS / data-attribute leaks
    # ("norsknorsk", "svenskasvenska", "ItalianoItaliano", "PolskiPolski")
    re.compile(r"\b([A-Za-zÀ-ÿ]{4,30})\1+\b"),
    # Doubled two-word phrases ("Bahasa MelayuBahasa Melayu", "中文(台灣)中文(台灣)")
    re.compile(r"\b(\w{4,15}\s\w{4,15})\1+", re.UNICODE),
    # Doubled non-ASCII glyph runs (CJK, Thai, Lao, Arabic, etc.)
    re.compile(r"([^\x00-\x7f]{2,15})\1+"),
]

_NEWLINE_RUN_RE = re.compile(r"(\n\s*){2,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_IMAGE_REF_RE = re.compile(r"\bImage\s*\d+", re.IGNORECASE)


def _is_mostly_junk(text: str) -> bool:
    """True if the (already-cleaned) text looks like a language menu / nav
    soup rather than article content. When this fires, we drop the content
    so the agent relies on the title alone."""
    if not text or len(text) < 60:
        return False
    # Many "Image N" references that survived = navigation menu
    if len(_IMAGE_REF_RE.findall(text)) >= 4:
        return True
    # Low letter ratio = mostly punctuation / glyph soup
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.45:
        return True
    return False


def _clean_content(text: str, max_len: int = 800) -> str:
    """Strip common scraping artifacts and collapse whitespace.

    If after cleanup the result is short, mostly punctuation, or still looks
    like navigation noise, returns empty string — the title alone will carry
    the signal.
    """
    if not text:
        return ""
    cleaned = text
    for pat in _JUNK_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = _NEWLINE_RUN_RE.sub("\n", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()
    # Cleanup yielded too few real letters — call it empty so the formatter
    # emits the "TITLE above IS the answer" hint instead of garbage.
    alpha_chars = sum(1 for c in cleaned if c.isalpha())
    if alpha_chars < 5:
        return ""
    if _is_mostly_junk(cleaned):
        return ""
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip() + "…"
    return cleaned


def _annotate_tavily_items(items: list[dict]) -> list[dict]:
    sorted_items = sort_results(items)
    annotated = []
    for item in sorted_items:
        url = item.get("url", "") or ""
        title = item.get("title", "") or ""
        raw_content = item.get("content", "") or ""
        cleaned_content = _clean_content(raw_content)
        tier = classify_source(url, title, cleaned_content)
        annotated.append({**item, "content": cleaned_content, "_source_tier": tier})
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


def _domain_of(url: str) -> str:
    try:
        return urlparse(url or "").netloc.replace("www.", "")
    except Exception:
        return url or ""


def _format_results_for_llm(items: list[dict]) -> str:
    """Render search results in a clean, line-based text format.

    - Drops prediction-tier results entirely (they were already weighted
      near-zero and just add noise to the agent's context).
    - Each result becomes a labelled block: TIER, TITLE, URL, CONTENT.
    - When CONTENT is empty (junk-detected), explicitly tells the agent
      the TITLE is the data.

    This format is easy for the LLM to read AND easy to parse with regex
    for downstream fact extraction.
    """
    useful = [it for it in items if (it.get("_source_tier") or "low") != "prediction"]
    if not useful:
        return ("No usable results — only prediction / betting / preview sites came back. "
                "Try a different search query (e.g. add 'final score', 'result', or the date).")

    lines = []
    for idx, item in enumerate(useful, start=1):
        tier = (item.get("_source_tier") or "low").upper()
        url = (item.get("url") or "").strip()
        domain = _domain_of(url)
        title = (item.get("title") or "").strip() or "(no title)"
        content = (item.get("content") or "").strip()

        lines.append(f"RESULT {idx} — TIER: {tier} — {domain}")
        lines.append(f"TITLE: {title}")
        lines.append(f"URL: {url}")
        if content:
            lines.append(f"CONTENT: {content}")
        else:
            lines.append("CONTENT: (empty — the TITLE above IS the answer; use it directly)")
        lines.append("")

    return "\n".join(lines).rstrip()


_EMPTY_QUERY_RETRY_MSG = (
    "Error: the 'query' argument was missing or malformed. "
    "Call this tool again with a single string parameter named 'query', "
    "containing the user's question as plain text. "
    'Example: {"query": "weather in Ankara today"}. '
    "Do NOT pass a JSON schema, a dict, or extra parameters."
)


_SEARCH_TOOL_DOC = """Search the web. Returns up to ~5 results in this format:

    RESULT 1 — TIER: HIGH — wikipedia.org
    TITLE: Headline / article title (often contains the answer directly)
    URL: https://...
    CONTENT: Body snippet, or "(empty — the TITLE above IS the answer)"

    RESULT 2 — TIER: MEDIUM — ...

Source tiers:
  HIGH    — Wikipedia, official org sites, major newswires (Reuters, AP, BBC)
  MEDIUM  — mainstream news, established publications (CNN, ESPN, Bloomberg)
  LOW     — unknown sites; corroborate before relying on it
  Prediction-tier results (betting / odds / preview sites) are DROPPED before
  you see them — never use those as facts.

IMPORTANT — READ TITLES:
When CONTENT is "(empty — the TITLE above IS the answer)", the TITLE itself
contains the data you need. Titles like "Bournemouth 1-0 Manchester City (05/19)"
or "Apple Q4 2024: $94.9B revenue" directly state the answer. Do NOT respond
with "the search results did not confirm this" when the answer is in the title.

Call exactly like: {"query": "your search question as a plain string"}.
"""


@tool(args_schema=SearchToolInput)
def tavily_search_results_json(query: str) -> str:
    if not query or not query.strip():
        return _EMPTY_QUERY_RETRY_MSG
    effective_query = _substitute_relative_dates(query.strip())
    if effective_query != query.strip():
        logger.info("Tavily: relative-date substitution %r -> %r", query, effective_query)
    try:
        raw = _get_tavily().invoke({"query": effective_query})
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
        return str(items)
    annotated = _annotate_tavily_items(items)
    return _format_results_for_llm(annotated)


tavily_search_results_json.__doc__ = _SEARCH_TOOL_DOC


@tool(args_schema=SearchToolInput)
def duckduckgo_results_json(query: str) -> str:
    if not query or not query.strip():
        return _EMPTY_QUERY_RETRY_MSG
    effective_query = _substitute_relative_dates(query.strip())
    if effective_query != query.strip():
        logger.info("DuckDuckGo: relative-date substitution %r -> %r", query, effective_query)
    try:
        raw = _get_duckduckgo().invoke({"query": effective_query})
    except Exception as exc:
        return f"DuckDuckGo search error: {exc}"
    raw_str = str(raw) if raw is not None else ""
    items = _parse_ddg_segments(raw_str)
    if not items:
        return raw_str[:2000]
    items = sort_results(items)
    # Normalize DDG items to the same shape as Tavily, then use the same formatter.
    normalized = []
    for it in items:
        cleaned_snippet = _clean_content(it.get("snippet", ""), max_len=500)
        normalized.append({
            "url": it.get("url", ""),
            "title": it.get("title", ""),
            "content": cleaned_snippet,
            "_source_tier": classify_source(it.get("url", ""), it.get("title", ""), cleaned_snippet),
        })
    return _format_results_for_llm(normalized)


duckduckgo_results_json.__doc__ = _SEARCH_TOOL_DOC


# --------------------------------------------------------------------------
# Wikipedia tool
# --------------------------------------------------------------------------
# Wikipedia is HIGH-tier and authoritative. For entities (people, orgs,
# places), historical events, and especially sports-season articles
# (e.g. "2025-26 Premier League season" has every matchday's complete
# results), it's the single best source. Tavily/DDG give a scattering of
# snippets across multiple sites; Wikipedia gives one comprehensive article.
# That's what kills the "matchup fabrication" failure mode — the agent sees
# all pairings in one place instead of crossing entities from different
# snippets.

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {
    "User-Agent": "DeepResearchAgent/1.0 (local research assistant; +https://example.local)"
}
WIKIPEDIA_MAX_CHARS = 8000
WIKIPEDIA_FOCUS_WINDOW = 4000

_MONTH_NAMES_TUPLE = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_DATE_IN_QUERY_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES_TUPLE) + r")\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)


def _focus_around_date(text: str, query: str) -> str | None:
    """When the query contains a "Month DD, YYYY" date and that date appears
    in the article text, return a window centered on the first occurrence.

    Returns None if no date or no match — caller falls back to full text.
    This stops the agent from drowning in a 100K-character season article
    when it only needs a specific matchday.
    """
    match = _DATE_IN_QUERY_RE.search(query)
    if not match:
        return None
    needle = f"{match.group(1)} {match.group(2)}"  # e.g. "May 19"
    idx = text.lower().find(needle.lower())
    if idx == -1:
        return None
    half = WIKIPEDIA_FOCUS_WINDOW // 2
    start = max(0, idx - half)
    end = min(len(text), idx + half)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


@tool(args_schema=SearchToolInput)
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for the most relevant article and return its content.

    Wikipedia is HIGH-tier — when the article matches the question, prefer
    it over Tavily/DDG snippets. Use this for:
      - Entities: people, organizations, places, products, concepts
      - Sports seasons: "2025-26 Premier League season" lists every match
        result, top scorers, standings — the single most reliable source
        for "what happened on date X" sports queries
      - Historical events with established facts
      - Biographies, definitions, technical concepts

    Less useful for: breaking news from the last 24-48 hours (Wikipedia
    may not yet have it — fall back to Tavily/DuckDuckGo for those).

    If the query contains a specific date (e.g. "May 19, 2026"), the tool
    focuses the returned excerpt around that date in the article so the
    agent sees the relevant matchday/event directly.

    Call exactly like: {"query": "2025-26 Premier League season"} or
    {"query": "Bournemouth A.F.C."} or {"query": "LangGraph"}.
    """
    if not query or not query.strip():
        return _EMPTY_QUERY_RETRY_MSG
    effective_query = _substitute_relative_dates(query.strip())
    if effective_query != query.strip():
        logger.info("Wikipedia: relative-date substitution %r -> %r", query, effective_query)

    # Step 1: full-text search to find the best matching article title.
    # We use action=query&list=search (full-text relevance ranking) instead of
    # opensearch (title-prefix matching). opensearch was too literal — it
    # returned the 2022-23 PL article when the query was "2025-26 Premier
    # League season" because of partial title overlap. Full-text search
    # correctly ranks the actual 2025-26 article first.
    try:
        search_resp = requests.get(
            WIKIPEDIA_API_BASE,
            params={
                "action": "query",
                "list": "search",
                "srsearch": effective_query,
                "srlimit": 3,
                "format": "json",
            },
            headers=WIKIPEDIA_HEADERS,
            timeout=10,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except Exception as exc:
        return f"Wikipedia search error: {exc}"

    hits = (search_data.get("query") or {}).get("search") or []
    if not hits:
        return f"No Wikipedia article found for '{query}'."

    top_title = hits[0].get("title") or ""
    if not top_title:
        return f"No Wikipedia article found for '{query}'."
    top_url = f"https://en.wikipedia.org/wiki/{top_title.replace(' ', '_')}"

    # Step 2: fetch plain-text extract of the article
    try:
        content_resp = requests.get(
            WIKIPEDIA_API_BASE,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": top_title,
                "format": "json",
                "redirects": 1,
            },
            headers=WIKIPEDIA_HEADERS,
            timeout=15,
        )
        content_resp.raise_for_status()
        content_data = content_resp.json()
    except Exception as exc:
        return f"Wikipedia content fetch error: {exc}"

    pages = (content_data.get("query") or {}).get("pages") or {}
    if not pages:
        return f"Wikipedia article not found for '{top_title}'."
    page = next(iter(pages.values()))
    extract = (page.get("extract") or "").strip()
    if not extract:
        return f"Wikipedia article '{top_title}' has no content."

    # If the query carries a date AND the article mentions it, focus the
    # excerpt around that date. Otherwise return the article's lead section.
    focused = _focus_around_date(extract, effective_query)
    content_block = focused if focused else extract
    if len(content_block) > WIKIPEDIA_MAX_CHARS:
        content_block = content_block[:WIKIPEDIA_MAX_CHARS].rstrip() + "…[truncated]"

    return (
        f"RESULT 1 — TIER: HIGH — en.wikipedia.org\n"
        f"TITLE: {top_title}\n"
        f"URL: {top_url}\n"
        f"CONTENT: {content_block}"
    )


@tool(args_schema=DeepReaderInput)
def deep_site_reader(url: str) -> str:
    """Reads and extracts visible text from a specific URL.

    Call exactly like: {"url": "https://example.com/page"}.
    """
    if not url or not url.strip():
        return (
            "Error: the 'url' argument was missing or malformed. "
            "Call this tool again with {\"url\": \"https://...\"} — a single string."
        )
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url.strip(), headers=headers, timeout=DEEP_READER_TIMEOUT_SECONDS, verify=False)
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()

        text_content = soup.get_text(separator=" ", strip=True)
        return text_content[:DEEP_READER_MAX_CHARS]
    except Exception as e:
        return f"This site could not be accessed (Error: {str(e)}). Please use Tavily summary results."


def build_tools():
    return [
        wikipedia_search,
        tavily_search_results_json,
        duckduckgo_results_json,
        deep_site_reader,
    ]
