import logging
import re
import threading
from datetime import date as local_date
from datetime import datetime, timezone
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
from rank_utils import ordinal_label, requested_rank
from source_quality import classify_source, source_reliability, tier_weight
from source_relevance import (
    filter_by_relevance,
    is_text_relevant_to_query,
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
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}

_FINANCE_QUERY_RE = re.compile(
    r"\b(stock|share|shares|ticker|quote|closing price|close price|market cap|nasdaq|nyse|"
    r"earnings|revenue|dividend|sec filing|10-k|10-q|tesla|apple|microsoft|nvidia)\b",
    re.IGNORECASE,
)

_WEATHER_QUERY_RE = re.compile(
    r"\b(weather|temperature|forecast|rain|snow|humidity|wind|climate)\b",
    re.IGNORECASE,
)

_SPORTS_QUERY_RE = re.compile(
    r"\b(score|scorer|scorers|top scorer|top scorers|goals?|goal scorers?|"
    r"match|game|fixture|standings|leaderboard|rankings?|won|lost|league|lig|"
    r"s[üu]per lig|nba|nfl|mlb|nhl|uefa|premier league|champions league|fifa|olympics)\b",
    re.IGNORECASE,
)

_LEADERBOARD_QUERY_RE = re.compile(
    r"\b(top scorer|top scorers|goal scorers?|scorers|leaderboard|rankings?|standings)\b",
    re.IGNORECASE,
)

_CLUB_TOP_SCORER_QUERY_RE = re.compile(
    r"\b(top\s+(?:goal\s*)?scorer|goal\s+scorer|goalscorer|most\s+goals|scorers?)\b",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_OFFICIAL_STATS_QUERY_RE = re.compile(
    r"\b(inflation|unemployment|gdp|cpi|population|interest rate|exchange rate|"
    r"official statistics|census|exports|imports|deficit)\b",
    re.IGNORECASE,
)

_SOFTWARE_QUERY_RE = re.compile(
    r"\b(version|release|changelog|documentation|api|sdk|package|library|"
    r"python|javascript|node|npm|pypi|github|langgraph|ollama)\b",
    re.IGNORECASE,
)

_HEALTH_SCIENCE_QUERY_RE = re.compile(
    r"\b(health|disease|vaccine|medicine|drug|clinical trial|study|research|"
    r"paper|journal|pubmed|who|cdc|nih|fda)\b",
    re.IGNORECASE,
)


def _is_time_sensitive(query: str) -> bool:
    return bool(_TIME_SENSITIVE_RE.search(query or ""))


def _authoritative_query_variants(query: str) -> list[str]:
    """Source-directed retries for factual lookup questions."""
    q = query.strip()
    variants: list[str] = []
    football_season = _current_european_football_season()

    if _FINANCE_QUERY_RE.search(q):
        variants.extend([
            f"{q} market data official close price",
            f"{q} site:nasdaq.com OR site:finance.yahoo.com OR site:marketwatch.com",
        ])
    if _WEATHER_QUERY_RE.search(q):
        variants.append(f"{q} site:weather.gov OR site:weather.com OR site:timeanddate.com")
    if _SPORTS_QUERY_RE.search(q):
        variants.append(f"{q} official result ESPN league site:espn.com OR site:nba.com OR site:uefa.com")
        if _LEADERBOARD_QUERY_RE.search(q):
            variants.append(f"{q} top scorers goals table site:worldfootball.net OR site:transfermarkt.com OR site:topscorersfootball.com OR site:statbunker.com")
            variants.append(f"{q} {football_season} site:espn.com/soccer/team/stats OR site:espn.co.uk/football/team/stats")
    if _OFFICIAL_STATS_QUERY_RE.search(q):
        variants.append(f"{q} official data statistics site:bls.gov OR site:bea.gov OR site:census.gov OR site:tuik.gov.tr OR site:eurostat.europa.eu")
    if _SOFTWARE_QUERY_RE.search(q):
        variants.append(f"{q} official documentation release notes site:github.com OR site:docs.python.org OR site:npmjs.com OR site:pypi.org")
    if _HEALTH_SCIENCE_QUERY_RE.search(q):
        variants.append(f"{q} official source study site:who.int OR site:cdc.gov OR site:nih.gov OR site:fda.gov OR site:pubmed.ncbi.nlm.nih.gov")

    if not variants:
        variants.append(f"{q} official source")

    deduped: list[str] = []
    seen = {q.lower()}
    for variant in variants:
        normalized = variant.lower()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(variant)
    return deduped[:3]


def _current_european_football_season() -> str:
    today = local_date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}/{start_year + 1}"


def _has_strong_result(items: list[dict]) -> bool:
    return any(
        (item.get("_source_reliability_score") or 0) >= 0.70
        for item in items
        if (item.get("_source_tier") or "low") != "prediction"
    )


def _merge_search_items(primary: list[dict], supplemental: list[dict]) -> list[dict]:
    by_url: dict[str, dict] = {}
    for item in primary + supplemental:
        url = (item.get("url") or "").strip()
        key = url or f"{item.get('title', '')}|{item.get('content', '')}"
        existing = by_url.get(key)
        if not existing:
            by_url[key] = item
            continue
        existing_score = existing.get("_source_reliability_score") or 0
        item_score = item.get("_source_reliability_score") or 0
        if item_score > existing_score:
            by_url[key] = item
    return list(by_url.values())


def _money(value: Any, currency: str = "USD") -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return ""
    if currency.upper() == "USD":
        return f"${amount:,.2f}"
    return f"{amount:,.2f} {currency.upper()}"


_FINANCE_SEARCH_NOISE_RE = re.compile(
    r"\b(what|was|were|is|are|the|a|an|of|for|on|in|at|to|from|"
    r"closing|close|price|stock|share|shares|ticker|market|cap|quote|"
    r"yesterday|today|latest|current|recent|last|previous)\b",
    re.IGNORECASE,
)


def _finance_symbol_search_terms(query: str) -> list[str]:
    candidates: list[str] = []

    ticker_match = re.search(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", query)
    if ticker_match:
        candidates.append(ticker_match.group(0))

    for pattern in (
        r"\b(?:of|for)\s+([A-Za-z][A-Za-z0-9 .,&'-]{1,50}?)\s+(?:stock|share|shares|ticker)\b",
        r"\b([A-Za-z][A-Za-z0-9 .,&'-]{1,50}?)\s+(?:stock|share|shares|ticker)\b",
    ):
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip(" ?.,"))

    simplified = _FINANCE_SEARCH_NOISE_RE.sub(" ", query)
    simplified = re.sub(r"[^A-Za-z0-9 .,&'-]+", " ", simplified)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if simplified:
        candidates.append(simplified)

    candidates.append(query)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        key = candidate.lower()
        if candidate and key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped[:4]


def _resolve_yahoo_symbol(query: str) -> dict[str, Any] | None:
    if not _FINANCE_QUERY_RE.search(query or ""):
        return None
    for search_term in _finance_symbol_search_terms(query):
        try:
            response = requests.get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                params={"q": search_term, "quotes_count": 5, "news_count": 0},
                headers=_HTTP_HEADERS,
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.info("Yahoo Finance symbol lookup failed for %r: %s", search_term, exc)
            continue

        for quote in data.get("quotes") or []:
            quote_type = str(quote.get("quoteType") or "").upper()
            symbol = str(quote.get("symbol") or "").strip()
            if symbol and quote_type in {"EQUITY", "ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY"}:
                return quote
    return None


def _yahoo_chart_result(symbol: str, rng: str = "10d") -> dict[str, Any] | None:
    """Fetch one daily chart result object from Yahoo Finance, or None on error."""
    try:
        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": rng, "interval": "1d"},
            headers=_HTTP_HEADERS,
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.info("Yahoo Finance chart lookup failed for %r: %s", symbol, exc)
        return None
    result = ((data.get("chart") or {}).get("result") or [None])[0]
    return result if isinstance(result, dict) else None


def _chart_dated_closes(result: dict) -> list[tuple[str, float]]:
    """Extract [(YYYY-MM-DD, close), ...] from a Yahoo chart result object."""
    timestamps = result.get("timestamp") or []
    quote_data = (((result.get("indicators") or {}).get("quote") or [{}])[0] or {})
    closes = quote_data.get("close") or []
    dated: list[tuple[str, float]] = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        try:
            day = datetime.fromtimestamp(int(ts), timezone.utc).date().isoformat()
            dated.append((day, float(close)))
        except (TypeError, ValueError, OSError):
            continue
    return dated


def _finance_structured_market_items(query: str) -> list[dict]:
    """Fetch structured market data before generic finance snippets."""
    quote = _resolve_yahoo_symbol(query)
    if not quote:
        return []

    symbol = str(quote.get("symbol") or "").strip()
    if not symbol:
        return []

    result = _yahoo_chart_result(symbol)
    if result is None:
        return []

    meta = result.get("meta") or {}
    currency = str(meta.get("currency") or "USD")
    dated_closes = _chart_dated_closes(result)
    if not dated_closes and meta.get("regularMarketPrice") is None:
        return []

    latest_close = dated_closes[-1] if dated_closes else None
    previous_close = dated_closes[-2] if len(dated_closes) >= 2 else None
    regular_price = _money(meta.get("regularMarketPrice"), currency)
    latest_price = _money(latest_close[1], currency) if latest_close else ""
    previous_price = _money(previous_close[1], currency) if previous_close else ""

    name = str(meta.get("longName") or meta.get("shortName") or quote.get("shortname") or symbol)
    exchange = str(meta.get("fullExchangeName") or meta.get("exchangeName") or quote.get("exchDisp") or "")
    source_url = f"https://finance.yahoo.com/quote/{symbol}"

    facts = [f"Yahoo Finance structured market data for {symbol} ({name})"]
    if exchange:
        facts.append(f"Exchange: {exchange}")
    if latest_close and latest_price:
        facts.append(f"Most recent daily close: {latest_price} on {latest_close[0]}")
    if previous_close and previous_price:
        facts.append(f"Previous daily close before that: {previous_price} on {previous_close[0]}")
    if regular_price:
        facts.append(f"Regular market price: {regular_price}")
    facts.append(
        "For a non-trading calendar day, answer with the most recent available trading close and state the exact date."
    )

    hint_parts = []
    if latest_close and latest_price:
        hint_parts.append(f"{symbol} most recent daily close was {latest_price} on {latest_close[0]}")
    if regular_price:
        hint_parts.append(f"regular market price was {regular_price}")

    return [{
        "title": f"Structured market data: {name} ({symbol})",
        "url": source_url,
        "content": ". ".join(facts),
        "score": 1.0,
        "published_date": latest_close[0] if latest_close else None,
        "_source_tier": "high",
        "_source_reliability_score": 0.92,
        "_source_reliability_label": "High",
        "_source_reliability_reasons": [
            "Structured market data source",
            "Dated daily close series",
            "Uses HTTPS",
        ],
        "_relevance_score": 1.0,
        "_direct_data_hint": "Structured market data: " + "; ".join(hint_parts) if hint_parts else "",
    }]


# --------------------------------------------------------------------------
# Structured exchange-rate (FX) lookup
# --------------------------------------------------------------------------
# Currency questions ("EUR/TRY", "euro to lira exchange rate") never reached
# the Yahoo structured path: _FINANCE_QUERY_RE only matches equities and
# _resolve_yahoo_symbol rejects CURRENCY quote types. We resolve the pair
# directly to Yahoo's "<BASE><QUOTE>=X" chart symbol and emit a clean
# DIRECT_DATA_HINT, so the verifier answers deterministically the same way
# stock closes already do — independent of which model drafted.

_FX_INTENT_RE = re.compile(
    r"\b(exchange rate|exchange rates|forex|fx rate|currency conversion|"
    r"currency rate|conversion rate|convert|how much is|worth in|rate of)\b",
    re.IGNORECASE,
)
_CURRENCY_PAIR_RE = re.compile(r"\b[A-Za-z]{3}\s*/\s*[A-Za-z]{3}\b")

_CURRENCY_ALIASES = {
    "usd": "USD", "us dollar": "USD", "american dollar": "USD",
    "dollar": "USD", "dollars": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "british pound": "GBP", "pound sterling": "GBP",
    "sterling": "GBP", "pound": "GBP", "pounds": "GBP",
    "try": "TRY", "turkish lira": "TRY", "lira": "TRY", "tl": "TRY",
    "jpy": "JPY", "japanese yen": "JPY", "yen": "JPY",
    "chf": "CHF", "swiss franc": "CHF",
    "cad": "CAD", "canadian dollar": "CAD",
    "aud": "AUD", "australian dollar": "AUD",
    "cny": "CNY", "chinese yuan": "CNY", "yuan": "CNY", "renminbi": "CNY", "rmb": "CNY",
    "inr": "INR", "indian rupee": "INR", "rupee": "INR", "rupees": "INR",
    "rub": "RUB", "russian ruble": "RUB", "ruble": "RUB", "rouble": "RUB",
    "krw": "KRW", "korean won": "KRW",
    "brl": "BRL", "brazilian real": "BRL",
    "mxn": "MXN", "mexican peso": "MXN",
    "zar": "ZAR", "south african rand": "ZAR",
    "aed": "AED", "uae dirham": "AED", "dirham": "AED",
    "sar": "SAR", "saudi riyal": "SAR",
    "sek": "SEK", "nok": "NOK", "dkk": "DKK", "pln": "PLN",
}
_CURRENCY_ALT = "|".join(re.escape(alias) for alias in sorted(_CURRENCY_ALIASES, key=len, reverse=True))
_CURRENCY_ALIAS_RE = re.compile(r"\b(?:" + _CURRENCY_ALT + r")\b")

# "<currency> to/in/vs <currency>" without an explicit exchange-rate word, e.g.
# "1 euro to try", "usd in eur", "pound vs dollar". Both sides must be currency
# tokens, which already rules out non-FX text like "did they try to win".
_FX_CONNECTOR_RE = re.compile(
    r"\b(" + _CURRENCY_ALT + r")\b\s*"
    r"(?:\bto\b|\binto\b|\bin\b|\bper\b|\bvs\b|\bversus\b|\bagainst\b|/|-)\s*"
    r"(?:\d[\d.,]*\s+)?(?:the\s+)?"
    r"\b(" + _CURRENCY_ALT + r")\b",
    re.IGNORECASE,
)
# Aliases that are also common English / proper words. A connector match made
# up of ONLY these (e.g. "try to pound") is too risky to treat as FX.
_AMBIGUOUS_CURRENCY_WORDS = {"try", "pound", "pounds", "sterling"}


def _extract_currencies(query: str) -> list[str]:
    """Currencies mentioned in the query as ISO codes, in order of appearance."""
    found: list[str] = []
    for match in _CURRENCY_ALIAS_RE.finditer((query or "").lower()):
        code = _CURRENCY_ALIASES.get(match.group(0))
        if code and code not in found:
            found.append(code)
    return found


def _has_connector_fx_intent(query: str) -> bool:
    """True for "<currency> to/in/vs <currency>" with at least one unambiguous side."""
    match = _FX_CONNECTOR_RE.search(query or "")
    if not match:
        return False
    left, right = match.group(1).lower(), match.group(2).lower()
    return left not in _AMBIGUOUS_CURRENCY_WORDS or right not in _AMBIGUOUS_CURRENCY_WORDS


def _resolve_currency_pair(query: str) -> tuple[str, str] | None:
    """Return (base, quote) ISO codes for an FX query, else None.

    Gated on FX intent — an exchange-rate phrase, a slash pair, or a
    currency-to-currency connector ("1 euro to try") — so a stray currency word
    in a non-FX question (e.g. "did they try to win") never triggers a lookup.
    """
    q = query or ""
    currencies = _extract_currencies(q)
    if not currencies:
        return None
    if not (
        _FX_INTENT_RE.search(q)
        or _CURRENCY_PAIR_RE.search(q)
        or _has_connector_fx_intent(q)
    ):
        return None
    base = currencies[0]
    quote = currencies[1] if len(currencies) > 1 else "USD"
    if base == quote:
        return None
    return base, quote


def looks_like_fx_query(query: str) -> bool:
    """True when the query is an exchange-rate / currency-pair lookup."""
    return _resolve_currency_pair(query) is not None


def _fx_structured_items(query: str) -> list[dict]:
    """Fetch a dated exchange rate from Yahoo Finance for currency-pair queries."""
    pair = _resolve_currency_pair(query)
    if not pair:
        return []
    base, quote = pair
    symbol = f"{base}{quote}=X"

    result = _yahoo_chart_result(symbol)
    if result is None:
        return []

    meta = result.get("meta") or {}
    dated = _chart_dated_closes(result)
    if dated:
        day, rate = dated[-1]
    elif meta.get("regularMarketPrice") is not None:
        try:
            rate = float(meta["regularMarketPrice"])
        except (TypeError, ValueError):
            return []
        day = local_date.today().isoformat()
    else:
        return []
    previous = dated[-2] if len(dated) >= 2 else None

    facts = [
        f"Yahoo Finance structured exchange rate for {base}/{quote}",
        f"Most recent rate: 1 {base} = {rate:.4f} {quote} on {day}",
    ]
    if previous:
        facts.append(f"Previous daily rate: 1 {base} = {previous[1]:.4f} {quote} on {previous[0]}")
    facts.append(
        "For a non-trading calendar day, this is the most recent available rate; state the exact date."
    )

    hint = (
        f"Structured FX data: {base}/{quote} was {rate:.4f} on {day} "
        f"(1 {base} = {rate:.4f} {quote})"
    )

    return [{
        "title": f"Structured exchange rate: {base}/{quote}",
        "url": f"https://finance.yahoo.com/quote/{base}{quote}=X",
        "content": ". ".join(facts),
        "score": 1.0,
        "published_date": day,
        "_source_tier": "high",
        "_source_reliability_score": 0.92,
        "_source_reliability_label": "High",
        "_source_reliability_reasons": [
            "Structured market data source",
            "Dated daily rate series",
            "Uses HTTPS",
        ],
        "_relevance_score": 1.0,
        "_direct_data_hint": hint,
    }]


def _fold_for_match(text: str) -> str:
    return (
        (text or "").lower()
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("?", "")
    )


_KNOWN_TOPSCORERSFOOTBALL_CLUBS = {
    "besiktas": ("Beşiktaş", "https://www.topscorersfootball.com/team/besiktas"),
    "besiktas jk": ("Beşiktaş", "https://www.topscorersfootball.com/team/besiktas"),
    "beşiktaş": ("Beşiktaş", "https://www.topscorersfootball.com/team/besiktas"),
    "beşiktaş jk": ("Beşiktaş", "https://www.topscorersfootball.com/team/besiktas"),
}


def _known_topscorersfootball_club(query: str) -> tuple[str, str] | None:
    folded = _fold_for_match(query)
    if re.search(r"\bbe.?ikta.?\b", folded):
        return ("Beşiktaş", "https://www.topscorersfootball.com/team/besiktas")
    for alias, club in _KNOWN_TOPSCORERSFOOTBALL_CLUBS.items():
        if _fold_for_match(alias) in folded:
            return club
    return None


def _parse_topscorersfootball_team_page(html: str, expected_club: str = "") -> dict[str, str] | None:
    if not html:
        return None
    season = _current_european_football_season()
    season_year = season.split("/", 1)[0]
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find(id=f"teamYear{season_year}")
    if container is None:
        header = soup.find(string=re.compile(rf"top scorers season\s+{re.escape(season)}", re.IGNORECASE))
        container = header.find_parent("table") if header else None
    if container is None:
        return None

    text = container.get_text(" ", strip=True)
    if expected_club and _fold_for_match(expected_club) not in _fold_for_match(text):
        return None

    current_league = ""
    for row in container.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if not cells:
            continue
        if len(cells) == 2 and ("Süper Lig" in cells[0] or "Super Lig" in cells[0]):
            current_league = cells[0]
            continue
        if len(cells) >= 4 and current_league:
            player = cells[0].strip()
            goals = re.search(r"\b(\d{1,3})\b", cells[3])
            if player and goals:
                return {
                    "club": expected_club or "Club",
                    "season": season,
                    "league": current_league,
                    "name": player,
                    "goals": goals.group(1),
                }

    return None


def _football_club_scorer_structured_items(query: str) -> list[dict]:
    """Direct club-season scorer lookup from a parseable team top-scorer table."""
    q = query or ""
    if not _SPORTS_QUERY_RE.search(q) or not _CLUB_TOP_SCORER_QUERY_RE.search(q):
        return []

    known = _known_topscorersfootball_club(q)
    if not known:
        return []

    expected_club, url = known
    try:
        response = requests.get(url, headers=_HTTP_HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        logger.info("TopScorersFootball club scorer lookup failed for %r: %s", expected_club, exc)
        return []

    parsed = _parse_topscorersfootball_team_page(response.text, expected_club)
    if not parsed:
        return []

    season = parsed["season"]
    name = parsed["name"]
    goals = parsed["goals"]
    club = parsed["club"]
    league = parsed["league"]
    content = (
        f"{club} top scorers season {season}: {name} leads the {league} section with {goals} goals. "
        f"For a 'this season' club top-scorer question, answer with the current season leader and goals."
    )
    hint = (
        f"Club top scorer: {club}'s top scorer for the {season} {league} season "
        f"is {name} with {goals} goals"
    )
    return [{
        "title": f"Structured club scorer data: {club} {season}",
        "url": url,
        "content": content,
        "score": 1.0,
        "published_date": local_date.today().isoformat(),
        "_source_tier": "medium",
        "_source_reliability_score": 0.74,
        "_source_reliability_label": "Medium",
        "_source_reliability_reasons": [
            "Established football statistics source",
            "Parseable club-season scorer table",
            "Uses HTTPS",
        ],
        "_relevance_score": 1.0,
        "_direct_data_hint": hint,
    }]


def _structured_data_items(query: str) -> list[dict]:
    """High-reliability structured data for a query."""
    items = _finance_structured_market_items(query)
    if items:
        return items
    items = _fx_structured_items(query)
    if items:
        return items
    return _football_club_scorer_structured_items(query)



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
        return []

    ranked = sort_by_relevance_then_tier(
        query, filtered, tier_weight, classify_source
    )
    return ranked


def _sort_by_relevance_then_reliability(items: list[dict], query: str) -> list[dict]:
    """Final ranking: topic relevance first, then reliability and recency."""
    def key(item: dict) -> tuple[float, float, float, float]:
        rel = item.get("_relevance_score")
        if not isinstance(rel, (int, float)):
            rel = relevance_score(
                query,
                str(item.get("title") or ""),
                str(item.get("content") or item.get("snippet") or ""),
                str(item.get("url") or ""),
            )
        reliability = item.get("_source_reliability_score")
        if not isinstance(reliability, (int, float)):
            assessment = source_reliability(
                str(item.get("url") or ""),
                str(item.get("title") or ""),
                str(item.get("content") or item.get("snippet") or ""),
            )
            reliability = assessment["score"]
        original = item.get("score")
        original_component = float(original) if isinstance(original, (int, float)) else 0.0
        return (
            float(rel),
            float(reliability),
            _published_date_boost(item, query),
            original_component,
        )

    return sorted(items, key=key, reverse=True)


def _annotate_tavily_items(items: list[dict], query: str) -> list[dict]:
    sorted_items = _prepare_search_items(items, query)
    annotated = []
    for item in sorted_items:
        url = item.get("url", "") or ""
        title = item.get("title", "") or ""
        raw_content = item.get("content", "") or ""
        cleaned_content = _clean_content(raw_content)
        reliability = source_reliability(url, title, cleaned_content)
        published_date = item.get("published_date") or _infer_published_date_from_text(f"{title} {cleaned_content}")
        annotated.append({
            **item,
            "content": cleaned_content,
            "published_date": published_date,
            "_source_tier": reliability["tier"],
            "_source_reliability_score": reliability["score"],
            "_source_reliability_label": reliability["label"],
            "_source_reliability_reasons": reliability["reasons"],
        })
    return _sort_by_relevance_then_reliability(annotated, query)


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


def _infer_published_date_from_text(text: str) -> str:
    """Extract simple dates from snippets such as 'Feb 4, 2026 · ...'."""
    if not text:
        return ""
    iso = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if iso:
        return iso.group(0)
    month = re.search(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
        r")\s+(\d{1,2}),\s+(20\d{2})\b",
        text,
        re.IGNORECASE,
    )
    if month:
        month_num = _MONTHS.get(month.group(1).lower()[:3])
        day = int(month.group(2))
        year = int(month.group(3))
        if month_num:
            return f"{year:04d}-{month_num:02d}-{day:02d}"
    return ""


def _published_date_boost(item: dict, query: str) -> float:
    formatted = _format_published_date(item.get("published_date"))
    time_sensitive = _is_time_sensitive(query)
    if not formatted:
        return -2.0 if time_sensitive else 0.0
    if not time_sensitive:
        return 0.5
    try:
        published = local_date.fromisoformat(formatted)
    except ValueError:
        return -2.0
    age_days = (local_date.today() - published).days
    if age_days < 0:
        return 2.0
    if age_days <= 3:
        return 8.0
    if age_days <= 14:
        return 6.0
    if age_days <= 30:
        return 4.0
    if age_days <= 365:
        return 1.5
    return -1.0


def _extract_direct_data_hint(query: str, item: dict) -> str:
    structured_hint = str(item.get("_direct_data_hint") or "").strip()
    if structured_hint:
        return structured_hint

    text = " ".join(
        part.strip()
        for part in (
            str(item.get("title") or ""),
            str(item.get("content") or ""),
        )
        if part and part.strip()
    )
    if not text:
        return ""

    club_scoped_leaderboard = bool(
        re.search(r"\b(top scorers at each club|club top scorers|per club|by club)\b", text, re.IGNORECASE)
        or re.search(r"\|\s*pos\s*\|\s*team\s*\|\s*top scorer\s*\|\s*goals\s*\|", text, re.IGNORECASE)
    )

    if _FINANCE_QUERY_RE.search(query or ""):
        date_match = re.search(
            r"\b(?:on\s+)?((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s+\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b",
            text,
            re.IGNORECASE,
        )
        patterns = [
            r"\bmost recent daily close\b.{0,80}?\$?([0-9]{1,4}(?:,[0-9]{3})*(?:\.\d{1,4})?)",
            r"\bclosing price\b.{0,80}?\bwas\s+\$?([0-9]{1,4}(?:,[0-9]{3})*(?:\.\d{1,4})?)",
            r"\bclosed\s+(?:at|around)\s+\$?([0-9]{1,4}(?:,[0-9]{3})*(?:\.\d{1,4})?)",
            r"\bregular market price\b.{0,80}?\$?([0-9]{1,4}(?:,[0-9]{3})*(?:\.\d{1,4})?)",
            r"\bcurrent price\b.{0,80}?\bis\s+\$?([0-9]{1,4}(?:,[0-9]{3})*(?:\.\d{1,4})?)\s*(?:USD)?",
            r"\bprevious close\b.{0,80}?\$?([0-9]{1,4}(?:,[0-9]{3})*(?:\.\d{1,4})?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            value = match.group(1)
            if not value.startswith("$"):
                value = f"${value}"
            hint = f"Market data candidate: {value}"
            if date_match:
                hint += f" on {date_match.group(1)}"
            return hint

    if _LEADERBOARD_QUERY_RE.search(query or ""):
        if club_scoped_leaderboard and not re.search(r"\b(each club|per club|by club|club)\b", query or "", re.IGNORECASE):
            return ""
        wanted_rank = requested_rank(query)
        domain = _domain_of(str(item.get("url") or ""))
        if "espn" in domain and re.search(r"\btop scorers\b", text, re.IGNORECASE):
            scorer_section = re.split(
                r"\b(?:Top Assists|Most Assists|Assists|Yellow Cards|Discipline|Goalkeepers)\b",
                text[text.lower().find("top scorers"):],
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            row_match = re.search(
                r"\b1\s*\|\s*([A-Za-zÀ-ÿğüşöçıİĞÜŞÖÇ' .-]{2,80}?)\s*\|\s*(\d{1,3})\s*\|\s*(\d{1,3})\b",
                scorer_section,
                re.IGNORECASE,
            )
            if row_match and (wanted_rank is None or wanted_rank == 1):
                name = re.sub(r"\s+", " ", row_match.group(1)).strip()
                played = row_match.group(2)
                goals = row_match.group(3)
                season_match = re.search(r"\b(20\d{2})[-/](\d{2})\b", text)
                if season_match:
                    season = f"{season_match.group(1)}-{season_match.group(2)}"
                else:
                    season = _current_european_football_season()
                return (
                    f"Club top scorer: ESPN lists the club's top scorer for the "
                    f"{season} season as {name} with {goals} goals in {played} appearances"
                )

        club_current = re.search(
            r"([A-Za-zÀ-ÿğüşöçıİĞÜŞÖÇ' .-]{2,80})'s current top 3 scorers "
            r"for the (\d{4}/\d{4}) season are "
            r"([^.,;]{2,80}?) with (\d{1,3}) goals?",
            text,
            re.IGNORECASE,
        )
        if club_current and (wanted_rank is None or wanted_rank == 1):
            club = club_current.group(1).strip()
            season = club_current.group(2).strip()
            name = club_current.group(3).strip()
            goals = club_current.group(4).strip()
            return (
                f"Club top scorer: {club}'s top scorer for the {season} season "
                f"is {name} with {goals} goals"
            )

        exact_match = re.search(
            r"\bMost goals (?:was|were) scored by\s+(.{3,120}?),\s+scoring\s+(\d{1,3})\s+goals?\b",
            text,
            re.IGNORECASE,
        )
        if exact_match and (wanted_rank is None or wanted_rank == 1):
            names = re.sub(r"\s+and\s+", " and ", exact_match.group(1).strip())
            goals = exact_match.group(2)
            return f"Leaderboard candidate: {names} were joint leaders with {goals} goals"

        rows: list[tuple[int, str, int]] = []
        for row_match in re.finditer(
            r"\|\s*(\d{1,2})\s*\|\s*([^|\n]+?)\s*\|[^|\n]*\|\s*\d{1,2}\s*\|\s*(\d{1,3})\s*\|",
            text,
        ):
            try:
                position = int(row_match.group(1))
                goals = int(row_match.group(3))
            except ValueError:
                continue
            if position > 10:
                continue
            player = re.sub(r"\s+", " ", row_match.group(2)).strip(" 🏆")
            if player:
                rows.append((position, player, goals))
        if rows:
            if wanted_rank is not None:
                exact_rows = [(player, goals) for position, player, goals in rows if position == wanted_rank]
                if exact_rows:
                    joined = " and ".join(player for player, _goals in exact_rows[:3])
                    goals = exact_rows[0][1]
                    return f"Leaderboard rank {wanted_rank} candidate: {joined} was listed {ordinal_label(wanted_rank)} with {goals} goals"

            max_goals = max(goal_count for _position, _player, goal_count in rows)
            leaders = [player for _position, player, goal_count in rows if goal_count == max_goals]
            if leaders:
                joined = " and ".join(leaders[:3])
                verb = "were joint leaders" if len(leaders) > 1 else "was the leader"
                return f"Leaderboard candidate: {joined} {verb} with {max_goals} goals"

    return ""


def _direct_hint_is_allowed(query: str, item: dict, hint: str) -> bool:
    """Only promote high-confidence evidence to DATA_HINT/DIRECT_DATA_HINT.

    Low-trust snippets can still appear as normal RESULT content, but they must
    not become deterministic facts that later model passes obey blindly.
    """
    if not hint:
        return False
    if str(item.get("_direct_data_hint") or "").strip() == hint:
        return True
    reliability = item.get("_source_reliability_score")
    if not isinstance(reliability, (int, float)):
        reliability = source_reliability(
            str(item.get("url") or ""),
            str(item.get("title") or ""),
            str(item.get("content") or item.get("snippet") or ""),
        )["score"]
    if float(reliability) >= 0.70:
        return True
    if _FINANCE_QUERY_RE.search(query or "") and float(reliability) >= 0.62:
        return True
    return False


_UNHELPFUL_SUMMARY_RE = re.compile(
    r"\b(not mentioned|not provided|do not provide|does not provide|could not find|"
    r"couldn't find|no definitive|not confirmed|unavailable)\b",
    re.IGNORECASE,
)


def _summary_is_unhelpful(summary: str, items: list[dict]) -> bool:
    if not summary or not _UNHELPFUL_SUMMARY_RE.search(summary):
        return False
    # If individual results have directly relevant content, avoid letting a
    # negative server-side synthesis override the evidence the model can read.
    return any(
        (item.get("_relevance_score") or 0) >= 0.70 and str(item.get("content") or "").strip()
        for item in items
        if (item.get("_source_tier") or "low") != "prediction"
    )


def _format_results_for_llm(items: list[dict], tavily_answer: str = "", query: str = "") -> str:
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
        return ("No relevant results for this query. "
                "Do not answer from unrelated topics. "
                "Say you could not find information that directly matches what was asked.")

    lines = []
    direct_hints: list[tuple[int | None, int | None, float, str]] = []
    for idx, item in enumerate(useful, start=1):
        hint = _extract_direct_data_hint(query, item)
        if hint and _direct_hint_is_allowed(query, item, hint):
            goal_match = re.search(r"\bwith\s+(\d{1,3})\s+goals?\b", hint, re.IGNORECASE)
            goal_count = int(goal_match.group(1)) if goal_match else None
            rank_match = re.search(r"\bLeaderboard rank\s+(\d{1,2})\s+candidate\b", hint, re.IGNORECASE)
            rank = int(rank_match.group(1)) if rank_match else None
            reliability = item.get("_source_reliability_score")
            rel_score = float(reliability) if isinstance(reliability, (int, float)) else 0.0
            direct_hints.append((rank, goal_count, rel_score, f"DIRECT_DATA_HINT {idx}: {hint} - source: {(item.get('url') or '').strip()}"))
    if direct_hints:
        ranked_hints = sorted(direct_hints, key=lambda h: h[2], reverse=True)
        hint_lines = [line for _rank, _goal_count, _rel_score, line in ranked_hints]
        if _LEADERBOARD_QUERY_RE.search(query or ""):
            wanted_rank = requested_rank(query)
            if wanted_rank is not None:
                rank_entries = [(goal_count, rel_score, line) for rank, goal_count, rel_score, line in ranked_hints if rank == wanted_rank]
                rank_lines = [line for _goal_count, _rel_score, line in rank_entries]
                rank_goal_counts = [goal_count for goal_count, _rel_score, _line in rank_entries if goal_count is not None]
                if rank_goal_counts:
                    max_rank_goals = max(rank_goal_counts)
                    rank_lines = [line for goal_count, _rel_score, line in rank_entries if goal_count == max_rank_goals]
                if rank_lines:
                    hint_lines = rank_lines
                else:
                    hint_lines = []
        if hint_lines:
            lines.append("DIRECT DATA HINTS (use these exact values when they directly answer the user; if hints conflict, prefer higher-reliability and corroborated sources over larger numeric values):")
            lines.extend(hint_lines[:3])
            lines.append("")
            lines.append("---")
            lines.append("")

    if tavily_answer and tavily_answer.strip() and not _summary_is_unhelpful(tavily_answer, useful):
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
        reliability_score = item.get("_source_reliability_score")
        reliability_suffix = (
            f" — RELIABILITY: {reliability_score:.2f}"
            if isinstance(reliability_score, (int, float))
            else ""
        )
        lines.append(f"RESULT {idx} — TIER: {tier}{rel_suffix}{reliability_suffix} — {domain}{date_suffix}")
        lines.append(f"TITLE: {title}")
        lines.append(f"URL: {url}")
        reasons = item.get("_source_reliability_reasons") or []
        if reasons:
            lines.append(f"RELIABILITY_SIGNALS: {'; '.join(str(r) for r in reasons[:4])}")
        data_hint = _extract_direct_data_hint(query, item)
        if data_hint and _direct_hint_is_allowed(query, item, data_hint):
            lines.append(f"DATA_HINT: {data_hint}")
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

    RESULT 1 — TIER: HIGH — RELEVANCE: 0.92 — RELIABILITY: 0.90 — wikipedia.org
    TITLE: Headline / article title (often contains the answer directly)
    URL: https://...
    RELIABILITY_SIGNALS: Curated reference source; Uses HTTPS
    CONTENT: Body snippet, or "(empty — the TITLE above IS the answer)"

    RESULT 2 — TIER: MEDIUM — ...

Source tiers:
  HIGH    — Wikipedia, official org sites, major newswires (Reuters, AP, BBC)
  MEDIUM  — mainstream news, established publications (CNN, ESPN, Bloomberg)
  LOW     — unknown sites; corroborate before relying on it
  Prediction-tier results (betting / odds / preview sites) are DROPPED before
  you see them — never use those as facts.

Reliability:
  Prefer higher RELIABILITY scores and corroboration from multiple independent
  sources. Do not use prediction-tier results as factual evidence.
  For market prices, if the requested calendar day was not a trading day, use
  the most recent available trading close and state the exact date.
  For standings, awards, top scorers, leaderboards, or rankings, a tie for
  first is a complete answer: list every tied leader and the shared value
  confidently instead of saying there is no single definitive answer.
  If the user asks for a specific ordinal/rank ("3rd top scorer", "second
  place", "5th largest"), answer that exact displayed rank, not the overall
  leader. Prefer DIRECT_DATA_HINT lines for rank-specific questions.

IMPORTANT — READ TITLES:
When CONTENT is "(empty — the TITLE above IS the answer)", the TITLE itself
contains the data you need. Titles like "Bournemouth 1-0 Manchester City (05/19)"
or "Apple Q4 2024: $94.9B revenue" directly state the answer. Do NOT respond
with "the search results did not confirm this" when the answer is in the title.

RELEVANCE — CRITICAL:
Each result may show RELEVANCE: 0.XX. Ignore any result that does not mention
the user's main subject (person, team, product, org). Same city or country
alone is NOT enough. Do not use off-topic results (e.g. NATO summit when the
user asked about a DJ). If no relevant results remain, say nothing was found.

Call exactly like: {"query": "your search question as a plain string"}.
"""


@tool(args_schema=SearchToolInput)
def tavily_search_results_json(query: str) -> str:
    if not query or not query.strip():
        return _EMPTY_QUERY_RETRY_MSG

    q = query.strip()
    # Market quote pages are usually evergreen data pages rather than news
    # articles. News-mode search often misses them, so finance lookups stay on
    # general search even when the user says "yesterday" or "latest".
    time_sensitive = _is_time_sensitive(q) and not _FINANCE_QUERY_RE.search(q)
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
    if tavily_answer and not is_text_relevant_to_query(q, tavily_answer):
        tavily_answer = ""
    if time_sensitive:
        logger.info("Tavily news-mode query (days=%d): %r — got %d results",
                    NEWS_MODE_DAYS, q, len(items))

    annotated = _annotate_tavily_items(items, q)
    structured_items = _structured_data_items(q)
    if structured_items:
        tavily_answer = ""
        annotated = _sort_by_relevance_then_reliability(
            _merge_search_items(structured_items, annotated),
            q,
        )
    if not _has_strong_result(annotated):
        supplemental: list[dict] = []
        for variant in _authoritative_query_variants(q):
            try:
                fallback_response = _get_tavily().search(
                    query=variant,
                    max_results=max(2, TAVILY_MAX_RESULTS - 1),
                    include_answer=False,
                    search_depth="advanced",
                )
            except Exception as exc:
                logger.info("Tavily source-directed fallback failed for %r: %s", variant, exc)
                continue
            if isinstance(fallback_response, dict) and isinstance(fallback_response.get("results"), list):
                supplemental.extend(_annotate_tavily_items(fallback_response["results"], q))
        if supplemental:
            annotated = _sort_by_relevance_then_reliability(
                _merge_search_items(annotated, supplemental),
                q,
            )
    return _format_results_for_llm(annotated, tavily_answer=tavily_answer, query=q)


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
            "published_date": _infer_published_date_from_text(f"{it.get('title', '')} {cleaned_snippet}"),
        })
    prepared = _prepare_search_items(normalized, query.strip())
    for it in prepared:
        reliability = source_reliability(
            it.get("url", ""), it.get("title", ""), it.get("content", "")
        )
        it["_source_tier"] = reliability["tier"]
        it["_source_reliability_score"] = reliability["score"]
        it["_source_reliability_label"] = reliability["label"]
        it["_source_reliability_reasons"] = reliability["reasons"]
    structured_items = _structured_data_items(query.strip())
    if structured_items:
        prepared = _merge_search_items(structured_items, prepared)
    if not _has_strong_result(prepared):
        supplemental = []
        for variant in _authoritative_query_variants(query.strip()):
            try:
                fallback_raw = _get_duckduckgo().invoke({"query": variant})
            except Exception as exc:
                logger.info("DuckDuckGo source-directed fallback failed for %r: %s", variant, exc)
                continue
            for it in _parse_ddg_segments(str(fallback_raw) if fallback_raw is not None else ""):
                cleaned_snippet = _clean_content(it.get("snippet", ""), max_len=500)
                supplemental.append({
                    "url": it.get("url", ""),
                    "title": it.get("title", ""),
                    "content": cleaned_snippet,
                    "snippet": cleaned_snippet,
                    "published_date": _infer_published_date_from_text(f"{it.get('title', '')} {cleaned_snippet}"),
                })
        if supplemental:
            supplemental = _prepare_search_items(supplemental, query.strip())
            for it in supplemental:
                reliability = source_reliability(
                    it.get("url", ""), it.get("title", ""), it.get("content", "")
                )
                it["_source_tier"] = reliability["tier"]
                it["_source_reliability_score"] = reliability["score"]
                it["_source_reliability_label"] = reliability["label"]
                it["_source_reliability_reasons"] = reliability["reasons"]
            prepared = _merge_search_items(prepared, supplemental)
    prepared = _sort_by_relevance_then_reliability(prepared, query.strip())
    return _format_results_for_llm(prepared, query=query.strip())


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
