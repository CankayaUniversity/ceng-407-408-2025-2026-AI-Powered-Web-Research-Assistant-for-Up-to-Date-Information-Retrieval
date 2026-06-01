"""Source reputation and reliability scoring for web search results.

Helps the agent prefer authoritative sources (Wikipedia, official organisation
sites, major newswires) over prediction/betting/opinion sites that flood
search results for current-events queries - especially temporal queries like
"last UCL winner" where prediction articles for the upcoming edition often
out-rank actual past-results coverage.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


HIGH_TIER_DOMAINS = {
    # Encyclopedias
    "wikipedia.org", "wikidata.org", "britannica.com",
    # Sports - official governing bodies and leagues
    "uefa.com", "fifa.com", "olympics.com", "olympic.org",
    "premierleague.com", "laliga.com", "bundesliga.com",
    "nba.com", "nfl.com", "mlb.com", "nhl.com",
    "fiba.basketball", "iaaf.org", "worldathletics.org",
    "icc-cricket.com", "atptour.com", "wtatennis.com",
    # Official governments / intergovernmentals
    "un.org", "who.int", "imf.org", "worldbank.org",
    "europa.eu", "ec.europa.eu", "eurostat.europa.eu", "ecb.europa.eu",
    "nasa.gov", "noaa.gov", "nih.gov", "cdc.gov", "fda.gov",
    "nist.gov", "energy.gov", "census.gov", "bls.gov", "sec.gov", "bea.gov",
    "data.gov",
    # Turkiye - resmi istatistik / ekonomi / devlet
    "tuik.gov.tr", "tcmb.gov.tr", "hmb.gov.tr", "turkiye.gov.tr",
    "saglik.gov.tr", "meb.gov.tr", "tubitak.gov.tr", "resmigazete.gov.tr",
    # Major newswires
    "reuters.com", "apnews.com", "afp.com",
    # Major newspapers / public broadcasters
    "bbc.com", "bbc.co.uk", "nytimes.com", "theguardian.com",
    "wsj.com", "washingtonpost.com", "ft.com", "economist.com",
    "npr.org", "pbs.org",
    # Tech docs and reference
    "developer.mozilla.org", "docs.python.org", "docs.microsoft.com",
    "learn.microsoft.com", "cloud.google.com", "docs.aws.amazon.com",
    "developer.apple.com",
    # Market data / exchanges / securities filings
    "nasdaq.com", "nyse.com", "sec.gov", "finance.yahoo.com",
    "markets.businessinsider.com", "tradingview.com",
    "investor.apple.com", "ir.tesla.com", "ir.aboutamazon.com",
    "abc.xyz", "investor.microsoft.com", "investor.nvidia.com",
    "investor.meta.com",
    # Weather / climate
    "weather.gov", "metoffice.gov.uk",
    # Peer-reviewed / scientific
    "nature.com", "science.org", "sciencemag.org",
    "thelancet.com", "nejm.org", "pubmed.ncbi.nlm.nih.gov",
}

MEDIUM_TIER_DOMAINS = {
    # Sports news
    "espn.com", "espn.co.uk", "skysports.com", "goal.com", "marca.com",
    "as.com", "lequipe.fr", "sport.es", "tuttosport.com",
    "sportingnews.com", "theathletic.com", "footballitalia.net",
    "fourfourtwo.com", "talksport.com", "worldfootball.net",
    "statbunker.com", "fbref.com", "soccerway.com", "topscorersfootball.com",
    # Turkish sports
    "trtspor.com.tr", "fotomac.com.tr", "mackolik.com", "ajansspor.com",
    "sporx.com", "fanatik.com.tr", "ntvspor.net",
    # Live scores
    "livescore.com", "sofascore.com", "flashscore.com", "transfermarkt.com",
    # Mainstream news
    "cnn.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com",
    "bloomberg.com", "axios.com", "politico.com",
    "aljazeera.com", "dw.com", "france24.com",
    # Turkiye - ajans / ekonomi haber
    "aa.com.tr", "anadoluajansi.com.tr", "bloomberght.com",
    "dunya.com", "ekonomim.com", "trthaber.com",
    # Tech publications
    "techcrunch.com", "wired.com", "theverge.com", "arstechnica.com",
    "engadget.com", "venturebeat.com", "zdnet.com",
    # Business / finance
    "forbes.com", "businessinsider.com", "marketwatch.com", "cnbc.com",
    "stockanalysis.com", "morningstar.com", "investing.com",
    "barchart.com", "companiesmarketcap.com", "macrotrends.net",
    # Weather / reference data
    "weather.com", "accuweather.com", "timeanddate.com",
    # Official-ish package registries and software references
    "pypi.org", "npmjs.com", "crates.io", "packagist.org",
    "mvnrepository.com",
    # Tech Q&A
    "stackoverflow.com", "stackexchange.com",
    # Code hosting / docs
    "github.com", "gitlab.com",
}

PREDICTION_DOMAIN_PATTERNS = (
    "odds", "betting", "betfair", "draftkings", "bet365", "fanduel",
    "tipster", "punter", "bookmaker", "pickwatch", "oddschecker",
    "wager", "parlay", "bettingexpert",
)

PREDICTION_TITLE_PATTERNS = (
    r"\bpredictions?\b",
    r"\bodds\b",
    r"\bbetting\b",
    r"\bpreview(?!s? of last)\b",
    r"\bwho will win\b",
    r"\bcould win\b",
    r"\bmight win\b",
    r"\bfavou?rite[s]? to win\b",
    r"\bcontender[s]?\b",
    r"\btips?\s+(?:for|to)\b",
    r"\bpicks?\b",
    r"\bfantasy\s+(?:football|premier|nba)\b",
    r"\bupcoming\b",
    r"\bahead of\b",
    r"\bset to (?:win|face|play|meet)\b",
    r"\bexpected to win\b",
    r"\blikely to win\b",
    r"\bwill be held\b",
    r"\bwill take place\b",
)
PREDICTION_TITLE_RE = re.compile("|".join(PREDICTION_TITLE_PATTERNS), re.IGNORECASE)

HIGH_TIER_SUFFIXES = (
    ".parliament.uk", ".police.uk", ".nhs.uk", ".ac.uk", ".gov.uk",
    ".edu.au", ".gov.au",
    ".gov.tr", ".edu.tr", ".bel.tr", ".k12.tr", ".pol.tr", ".mil.tr",
    ".gouv.fr", ".gov.de", ".gov.nl", ".gov.se", ".gov.pl", ".gov.it",
    ".gc.ca", ".gov.ca",
    ".go.jp", ".go.kr", ".go.id",
    ".gov.in", ".gov.sg", ".gov.nz", ".gov.br", ".gov.tw", ".gov.hk",
    ".int",
    ".edu", ".gov", ".mil",
)

SELF_PUBLISHING_HINTS = (
    "medium.com", "substack.com", "wordpress.com", "blogspot.com",
    "tumblr.com", "patreon.com",
)
UGC_HINTS = ("reddit.com", "quora.com", "answers.com")
OFFICIAL_WORDS = ("official", "government", "ministry", "department", "statistics", "exchange")
MARKET_WORDS = ("stock", "quote", "market data", "close", "closing price", "previous close", "nasdaq", "nyse")
SPARSE_TITLE_RE = re.compile(r"^\s*(?:home|index|untitled|404|loading)\s*$", re.IGNORECASE)
SPECULATIVE_TEXT_RE = re.compile(
    r"\b(rumou?r|unconfirmed|reportedly|might|could|expected|forecast|prediction|projected)\b",
    re.IGNORECASE,
)


def _has_high_tier_suffix(domain: str) -> bool:
    return any(domain.endswith(suffix) for suffix in HIGH_TIER_SUFFIXES)


def _normalize_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _root_domain(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) < 2:
        return domain
    if len(parts) >= 3:
        if parts[-1] == "tr" and parts[-2] in {"gov", "edu", "bel", "k12", "pol", "mil", "org", "com", "net"}:
            return ".".join(parts[-3:])
        if parts[-1] in {"uk", "au", "nz", "jp", "kr", "tw", "hk", "br", "in", "sg"} and parts[-2] in {
            "co", "gov", "ac", "com", "org", "go", "edu", "net",
        }:
            return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_prediction_source(url: str, title: str = "", snippet: str = "") -> bool:
    domain = _normalize_domain(url)
    for pattern in PREDICTION_DOMAIN_PATTERNS:
        if pattern in domain:
            return True
    haystack = f"{title} {snippet}"[:500]
    return bool(PREDICTION_TITLE_RE.search(haystack))


def _base_tier(domain: str, root: str, title: str, snippet: str) -> str:
    if domain in HIGH_TIER_DOMAINS or root in HIGH_TIER_DOMAINS:
        return "high"
    if domain in MEDIUM_TIER_DOMAINS or root in MEDIUM_TIER_DOMAINS:
        return "medium"
    if _has_high_tier_suffix(domain):
        return "high"
    if domain.endswith(".org") and any(
        token in domain for token in ("foundation", "institute", "council", "museum", "archive")
    ):
        return "medium"
    text = f"{title} {snippet}".lower()
    if domain.endswith(".com") and any(word in text for word in MARKET_WORDS):
        return "medium"
    return "low"


def source_reliability(url: str, title: str = "", snippet: str = "") -> dict:
    """Return tier plus continuous 0-1 reliability score and reasons."""
    if not url:
        return {"tier": "low", "score": 0.15, "label": "Low", "reasons": ["Missing source URL"]}

    domain = _normalize_domain(url)
    root = _root_domain(domain)
    text = f"{title} {snippet}".strip()
    reasons: list[str] = []

    if is_prediction_source(url, title, snippet):
        return {
            "tier": "prediction",
            "score": 0.02,
            "label": "Prediction",
            "reasons": ["Prediction, betting, odds, or preview-style source"],
        }

    tier = _base_tier(domain, root, title, snippet)
    score = {"high": 0.82, "medium": 0.64, "low": 0.34}.get(tier, 0.34)

    if tier == "high":
        if domain in HIGH_TIER_DOMAINS or root in HIGH_TIER_DOMAINS:
            reasons.append("Curated high-reliability source")
        elif _has_high_tier_suffix(domain):
            reasons.append("Official or academic domain suffix")
    elif tier == "medium":
        reasons.append("Curated established publication or platform")
    else:
        reasons.append("Domain is not in the curated reliability registry")

    if url.lower().startswith("https://"):
        score += 0.03
        reasons.append("Uses HTTPS")
    else:
        score -= 0.06
        reasons.append("Does not use HTTPS")

    if any(domain.endswith(hint) or root == hint for hint in SELF_PUBLISHING_HINTS):
        score -= 0.18
        reasons.append("Self-publishing platform")
    if any(domain.endswith(hint) or root == hint for hint in UGC_HINTS):
        score -= 0.14
        reasons.append("User-generated content")
    if SPARSE_TITLE_RE.search(title or ""):
        score -= 0.08
        reasons.append("Sparse or generic page title")
    if len(text) < 40:
        score -= 0.06
        reasons.append("Sparse metadata/snippet")
    if SPECULATIVE_TEXT_RE.search(text):
        score -= 0.08
        reasons.append("Speculative wording detected")
    if any(word in text.lower() for word in OFFICIAL_WORDS) and tier != "low":
        score += 0.04
        reasons.append("Official/data-oriented wording")
    if any(word in text.lower() for word in MARKET_WORDS) and root in {
        "finance.yahoo.com", "nasdaq.com", "marketwatch.com", "businessinsider.com", "tradingview.com",
    }:
        score += 0.06
        reasons.append("Established market data source")

    score = max(0.0, min(1.0, score))
    if score >= 0.78:
        label = "High"
        tier = "high"
    elif score >= 0.55:
        label = "Medium"
        if tier == "low":
            tier = "medium"
    else:
        label = "Low"
        if tier != "prediction":
            tier = "low"

    return {"tier": tier, "score": round(score, 2), "label": label, "reasons": reasons[:5]}


def classify_source(url: str, title: str = "", snippet: str = "") -> str:
    """Return reputation tier: high, medium, low, or prediction."""
    return source_reliability(url, title, snippet)["tier"]


def tier_weight(tier: str) -> float:
    """Numeric weight retained for older callers/UI formulas."""
    return {"high": 1.0, "medium": 0.65, "low": 0.35, "prediction": 0.02}.get(tier, 0.35)
