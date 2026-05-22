"""Source reputation classification for web search results.

Helps the agent prefer authoritative sources (Wikipedia, official organisation
sites, major newswires) over prediction/betting/opinion sites that flood
search results for current-events queries — especially temporal queries like
"last UCL winner" where prediction articles for the *upcoming* edition often
out-rank actual past-results coverage.
"""

import re
from urllib.parse import urlparse


HIGH_TIER_DOMAINS = {
    # Encyclopedias
    "wikipedia.org", "wikidata.org", "britannica.com",
    # Sports — official governing bodies and leagues
    "uefa.com", "fifa.com", "olympics.com", "olympic.org",
    "premierleague.com", "laliga.com", "bundesliga.com",
    "nba.com", "nfl.com", "mlb.com", "nhl.com",
    "fiba.basketball", "iaaf.org", "worldathletics.org",
    "icc-cricket.com", "atptour.com", "wtatennis.com",
    # Official governments / intergovernmentals
    "un.org", "who.int", "imf.org", "worldbank.org",
    "europa.eu", "ecb.europa.eu",
    "nasa.gov", "noaa.gov", "nih.gov", "cdc.gov", "fda.gov",
    "nist.gov", "energy.gov",
    # Major newswires
    "reuters.com", "apnews.com", "afp.com",
    # Major newspapers / public broadcasters
    "bbc.com", "bbc.co.uk", "nytimes.com", "theguardian.com",
    "wsj.com", "washingtonpost.com", "ft.com", "economist.com",
    "npr.org", "pbs.org",
    # Tech docs and reference
    "developer.mozilla.org", "docs.python.org",
    # Peer-reviewed / scientific
    "nature.com", "science.org", "sciencemag.org",
    "thelancet.com", "nejm.org",
}

MEDIUM_TIER_DOMAINS = {
    # Sports news (reasonably reliable, but not official)
    "espn.com", "skysports.com", "goal.com", "marca.com",
    "as.com", "lequipe.fr", "sport.es", "tuttosport.com",
    "sportingnews.com", "theathletic.com", "footballitalia.net",
    "fourfourtwo.com", "talksport.com",
    # Turkish sports
    "trtspor.com.tr", "fotomac.com.tr", "mackolik.com", "ajansspor.com",
    "sporx.com", "fanatik.com.tr", "ntvspor.net",
    # Live scores (topic-relevant even if not "official")
    "livescore.com", "sofascore.com", "flashscore.com", "transfermarkt.com",
    # Mainstream news
    "cnn.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com",
    "bloomberg.com", "axios.com", "politico.com",
    "aljazeera.com", "dw.com", "france24.com",
    # Tech publications
    "techcrunch.com", "wired.com", "theverge.com", "arstechnica.com",
    "engadget.com", "venturebeat.com", "zdnet.com",
    # Business / finance
    "forbes.com", "businessinsider.com", "marketwatch.com", "cnbc.com",
    # Tech Q&A
    "stackoverflow.com", "stackexchange.com",
    # Code hosting / docs
    "github.com", "gitlab.com",
}

# Domain substring patterns that strongly indicate prediction/betting content
PREDICTION_DOMAIN_PATTERNS = (
    "odds", "betting", "betfair", "draftkings", "bet365", "fanduel",
    "tipster", "punter", "bookmaker", "pickwatch", "oddschecker",
    "wager", "parlay", "bettingexpert",
)

# Title/snippet patterns suggesting speculative or preview content
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
    # Handle common multi-part TLDs (.co.uk, .gov.uk, .com.au, .ac.uk)
    last_two = ".".join(parts[-2:])
    if parts[-1] in {"uk", "au", "nz", "jp"} and parts[-2] in {"co", "gov", "ac", "com", "org"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last_two


def is_prediction_source(url: str, title: str = "", snippet: str = "") -> bool:
    domain = _normalize_domain(url)
    for pattern in PREDICTION_DOMAIN_PATTERNS:
        if pattern in domain:
            return True
    haystack = f"{title} {snippet}"[:400]
    if PREDICTION_TITLE_RE.search(haystack):
        return True
    return False


def classify_source(url: str, title: str = "", snippet: str = "") -> str:
    """Return reputation tier: 'high', 'medium', 'low', or 'prediction'."""
    if not url:
        return "low"
    if is_prediction_source(url, title, snippet):
        return "prediction"
    domain = _normalize_domain(url)
    root = _root_domain(domain)
    if domain in HIGH_TIER_DOMAINS or root in HIGH_TIER_DOMAINS:
        return "high"
    if domain in MEDIUM_TIER_DOMAINS or root in MEDIUM_TIER_DOMAINS:
        return "medium"
    # Government / academic top-level domains
    if domain.endswith((".gov", ".edu", ".mil", ".gov.uk", ".gov.au", ".ac.uk", ".edu.au")):
        return "high"
    if domain.endswith(".org") and any(token in domain for token in ("foundation", "institute", "council")):
        return "medium"
    return "low"


def tier_weight(tier: str) -> float:
    """Numeric weight for ranking (higher = better)."""
    return {"high": 1.0, "medium": 0.65, "low": 0.35, "prediction": 0.02}.get(tier, 0.35)


def sort_results(results: list[dict]) -> list[dict]:
    """Sort search results so higher-tier sources come first.

    Original Tavily relevance is preserved as a secondary signal.
    """
    def key(item):
        url = item.get("url", "") or ""
        title = item.get("title", "") or ""
        content = item.get("content") or item.get("snippet") or ""
        tier = classify_source(url, title, content)
        weight = tier_weight(tier)
        original_score = item.get("score")
        original_component = float(original_score) if isinstance(original_score, (int, float)) else 0.0
        return -(weight * 10.0 + original_component)
    return sorted(results, key=key)
