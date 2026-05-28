import json
import logging
import re
import threading
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from tavily import TavilyClient

logger = logging.getLogger("deep_research")

from config import (
    DEEP_READER_MAX_CHARS,
    DEEP_READER_TIMEOUT_SECONDS,
    DUCKDUCKGO_MAX_RESULTS,
    TAVILY_MAX_RESULTS,
)
from source_quality import classify_source, sort_results, tier_weight
from source_relevance import (
    filter_by_relevance,
    relevance_score,
    sort_by_relevance_then_tier,
)


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
# Recency-aware Tavily configuration
# --------------------------------------------------------------------------
# Tavily exposes a "news" topic + `days` window + `search_depth=advanced`
# + `include_answer=True` which together produce dramatically better results
# for time-sensitive queries. We detect time-sensitive queries via a regex
# over the user's question (handed off as the tool argument) and switch into
# news mode automatically. No LLM call needed for the detection.

_TIME_SENSITIVE_RE = re.compile(
    r"\b("
    r"today|yesterday|tonight|tomorrow|"
    r"this\s+(?:week|month|year|morning|afternoon|evening)|"
    r"last\s+(?:week|month|night|weekend|year)|"
    r"next\s+(?:week|month|weekend)|"
    r"latest|recent(?:ly)?|current(?:ly)?|now|"
    r"breaking\s+news|news\b|"
    r"202[5-9]|20[3-9]\d"  # explicit recent years (2025+)
    r")\b",
    re.IGNORECASE,
)

NEWS_MODE_DAYS = 14  # window for news-mode queries


def _is_time_sensitive(query: str) -> bool:
    return bool(_TIME_SENSITIVE_RE.search(query or ""))


def _get_tavily():
    global _tavily_underlying
    if _tavily_underlying is None:
        with _client_lock:
            if _tavily_underlying is None:  # double-checked locking
                import os
                api_key = os.getenv("TAVILY_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "TAVILY_API_KEY is not set in the environment. "
                        "Add it to your .env file (see .env.example)."
                    )
                _tavily_underlying = TavilyClient(api_key=api_key)
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


def _prepare_search_items(items: list[dict], query: str) -> list[dict]:
    """Filter off-topic hits, then rank by relevance before domain tier."""
    filtered = filter_by_relevance(query, items)
    if not filtered:
        # Avoid empty context — fall back to best raw hits
        filtered = [dict(it) for it in items[: max(TAVILY_MAX_RESULTS, 3)]]
        for it in filtered:
            it["_relevance_score"] = relevance_score(
                query,
                str(it.get("title") or ""),
                str(it.get("content") or it.get("snippet") or ""),
                str(it.get("url") or ""),
            )

    ranked = sort_by_relevance_then_tier(
        query, filtered, tier_weight, classify_source
    )
    return ranked


def _annotate_tavily_items(items: list[dict], query: str) -> list[dict]:
    sorted_items = _prepare_search_items(items, query)
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


def _format_published_date(raw: Any) -> str:
    """Normalise a published_date to a short YYYY-MM-DD label.

    Tavily emits two formats depending on topic:
      - Default topic: ISO ("2026-05-19" or "2026-05-19T14:23:00Z")
      - News topic:    RFC 2822 ("Fri, 08 May 2026 14:23:00 GMT")
    Returns empty string when nothing parseable is present.
    """
    if not raw:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Try RFC 2822 first — its parser is strict so it won't false-positive
    # on ISO strings.
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            return dt.strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        pass
    # ISO-like fallback. Take everything before the first 'T' or space.
    for sep in ("T", " "):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    # Sanity-check it looks like YYYY-MM-DD.
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return ""


def _format_results_for_llm(items: list[dict], tavily_answer: str = "") -> str:
    """Render search results in a clean, line-based text format.

    - Drops prediction-tier results entirely (they were already weighted
      near-zero and just add noise to the agent's context).
    - Each result becomes a labelled block: TIER, TITLE, URL, DATE, CONTENT.
    - When CONTENT is empty (junk-detected), explicitly tells the agent
      the TITLE is the data.
    - When Tavily returned its own synthesised answer (include_answer=True),
      surface it as a leading TAVILY SUMMARY block so the model sees it
      as additional evidence.

    This format is easy for the LLM to read AND easy to parse with regex
    for downstream fact extraction.
    """
    useful = [it for it in items if (it.get("_source_tier") or "low") != "prediction"]
    if not useful and not tavily_answer:
        return ("No usable results — only prediction / betting / preview sites came back. "
                "Try a different search query (e.g. add 'final score', 'result', or the date).")

    lines = []
    if tavily_answer and tavily_answer.strip():
        lines.append("TAVILY SUMMARY (server-side synthesis from the top results; "
                     "use as supporting evidence, but verify against the individual "
                     "RESULTs below):")
        lines.append(tavily_answer.strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    for idx, item in enumerate(useful, start=1):
        tier = (item.get("_source_tier") or "low").upper()
        url = (item.get("url") or "").strip()
        domain = _domain_of(url)
        title = (item.get("title") or "").strip() or "(no title)"
        content = (item.get("content") or "").strip()
        pub_date = _format_published_date(item.get("published_date"))
        date_suffix = f" — {pub_date}" if pub_date else ""

        rel = item.get("_relevance_score")
        rel_suffix = f" — RELEVANCE: {rel:.2f}" if isinstance(rel, (int, float)) else ""
        lines.append(f"RESULT {idx} — TIER: {tier}{rel_suffix} — {domain}{date_suffix}")
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

    q = query.strip()
    time_sensitive = _is_time_sensitive(q)
    # Build kwargs. When the query is time-sensitive we switch into Tavily's
    # news topic with a recency window AND advanced search depth (longer
    # content per result). Otherwise default general search.
    search_kwargs: dict[str, Any] = {
        "query": q,
        "max_results": TAVILY_MAX_RESULTS,
        "include_answer": True,
    }
    if time_sensitive:
        search_kwargs["topic"] = "news"
        search_kwargs["days"] = NEWS_MODE_DAYS
        search_kwargs["search_depth"] = "advanced"

    try:
        response = _get_tavily().search(**search_kwargs)
    except Exception as exc:
        return f"Tavily search error: {exc}"

    if not isinstance(response, dict):
        return str(response)

    items = response.get("results") or []
    if not isinstance(items, list):
        items = []
    tavily_answer = response.get("answer") or ""
    if time_sensitive:
        logger.info("Tavily news-mode query (days=%d): %r — got %d results",
                    NEWS_MODE_DAYS, q, len(items))

    annotated = _annotate_tavily_items(items, q)
    return _format_results_for_llm(annotated, tavily_answer=tavily_answer)


tavily_search_results_json.__doc__ = _SEARCH_TOOL_DOC


@tool(args_schema=SearchToolInput)
def duckduckgo_results_json(query: str) -> str:
    if not query or not query.strip():
        return _EMPTY_QUERY_RETRY_MSG
    try:
        raw = _get_duckduckgo().invoke({"query": query.strip()})
    except Exception as exc:
        return f"DuckDuckGo search error: {exc}"
    raw_str = str(raw) if raw is not None else ""
    items = _parse_ddg_segments(raw_str)
    if not items:
        return raw_str[:2000]
    # Normalize DDG items to the same shape as Tavily, then relevance-filter + rank.
    normalized = []
    for it in items:
        cleaned_snippet = _clean_content(it.get("snippet", ""), max_len=500)
        normalized.append({
            "url": it.get("url", ""),
            "title": it.get("title", ""),
            "content": cleaned_snippet,
            "snippet": cleaned_snippet,
        })
    prepared = _prepare_search_items(normalized, query.strip())
    for it in prepared:
        it["_source_tier"] = classify_source(
            it.get("url", ""), it.get("title", ""), it.get("content", "")
        )
    return _format_results_for_llm(prepared)


duckduckgo_results_json.__doc__ = _SEARCH_TOOL_DOC


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
