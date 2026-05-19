import json
import re
import threading
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

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
    re.compile(r"\bImage\s+\d+(?::[^\n,]{0,40})?", re.IGNORECASE),
    re.compile(r"\bAdvertisement\b", re.IGNORECASE),
    re.compile(r"\bShow more (?:games|matches|results)\b", re.IGNORECASE),
    re.compile(r"\bNo data\b"),
    re.compile(r"\bWatch Live Stream(?: For Free)?\b", re.IGNORECASE),
    re.compile(r"\bSubscribe to .{0,40}\b", re.IGNORECASE),
    # Stripped JS closures (e.g. `")` ` "),` artifacts)
    re.compile(r'"\)\s*[,.]?'),
    # Form strings like "WWWLL", "?WWTWW" that bleed in from fixture tables
    re.compile(r"\?[WLTD]{3,}\b"),
    re.compile(r"\b[WLTD]{4,}\b"),
]

_NEWLINE_RUN_RE = re.compile(r"(\n\s*){2,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _clean_content(text: str, max_len: int = 800) -> str:
    """Strip common scraping artifacts and collapse whitespace."""
    if not text:
        return ""
    cleaned = text
    for pat in _JUNK_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = _NEWLINE_RUN_RE.sub("\n", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()
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


def _format_ddg_with_tiers(items: list[dict]) -> str:
    parts = []
    for item in items:
        cleaned_snippet = _clean_content(item.get("snippet", ""), max_len=500)
        tier = classify_source(item.get("url", ""), item.get("title", ""), cleaned_snippet)
        parts.append(
            f"[TIER: {tier.upper()}] snippet: {cleaned_snippet}, "
            f"title: {item.get('title', '')}, link: {item.get('url', '')}"
        )
    return ", ".join(parts)


_EMPTY_QUERY_RETRY_MSG = (
    "Error: the 'query' argument was missing or malformed. "
    "Call this tool again with a single string parameter named 'query', "
    "containing the user's question as plain text. "
    'Example: {"query": "weather in Ankara today"}. '
    "Do NOT pass a JSON schema, a dict, or extra parameters."
)


@tool(args_schema=SearchToolInput)
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

    Call exactly like: {"query": "your search question as a plain string"}.
    """
    if not query or not query.strip():
        return _EMPTY_QUERY_RETRY_MSG
    try:
        raw = _get_tavily().invoke({"query": query.strip()})
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


@tool(args_schema=SearchToolInput)
def duckduckgo_results_json(query: str) -> str:
    """Search the web via DuckDuckGo. Returns results sorted with the highest-quality
    sources first. Each result is prefixed with [TIER: HIGH|MEDIUM|LOW|PREDICTION].
    Same tier semantics as the Tavily tool — never treat PREDICTION-tier results as
    facts; they describe events that have not yet happened.

    Call exactly like: {"query": "your search question as a plain string"}.
    """
    if not query or not query.strip():
        return _EMPTY_QUERY_RETRY_MSG
    try:
        raw = _get_duckduckgo().invoke({"query": query.strip()})
    except Exception as exc:
        return f"DuckDuckGo search error: {exc}"
    raw_str = str(raw) if raw is not None else ""
    items = _parse_ddg_segments(raw_str)
    if not items:
        return raw_str
    items = sort_results(items)
    return _format_ddg_with_tiers(items)


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
    return [tavily_search_results_json, duckduckgo_results_json, deep_site_reader]
