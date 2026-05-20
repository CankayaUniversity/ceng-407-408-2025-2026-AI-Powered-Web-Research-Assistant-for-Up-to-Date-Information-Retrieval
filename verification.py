"""Post-agent verification pass.

Re-checks the agent's draft answer against the tool results (search snippets,
deep-read pages) and rewrites unsupported claims. This catches the common
small-model failure where the agent calls its tools but then synthesizes the
final answer from stale training memory instead of the retrieved content.
"""

import json
from urllib.parse import urlparse

from langchain_ollama import ChatOllama


# ChatOllama clients are cached per model_id to avoid repeating handshake work
# on every verification pass.
_LLM_CACHE: dict[str, ChatOllama] = {}


def _get_llm(model_id: str) -> ChatOllama:
    llm = _LLM_CACHE.get(model_id)
    if llm is None:
        llm = ChatOllama(model=model_id, temperature=0)
        _LLM_CACHE[model_id] = llm
    return llm


VERIFICATION_PROMPT = """You are a strict fact-checker. Your only job is to verify a draft answer against live web search results and correct any factual errors.

QUESTION ASKED:
{question}

DRAFT ANSWER (may contain stale or hallucinated information from outdated training data):
{answer}

LIVE WEB SEARCH RESULTS (current, authoritative — TRUST THESE OVER ANY PRIOR KNOWLEDGE):
{sources}

TODAY'S DATE: {today}

SOURCE TIERS:
Each source is annotated with a tier in its content:
- HIGH tier (Wikipedia, official org sites, major newswires) — strongest evidence.
- MEDIUM tier (mainstream news, established publications) — solid evidence.
- LOW tier (unknown blogs, small sites) — weak evidence; corroborate before relying on it.
- PREDICTION tier (betting/odds sites, "who will win" articles, previews of upcoming events) — NEVER use these as evidence for past facts. They are speculation about events that have NOT yet happened.

YOUR TASK:
For every specific factual claim in the draft — names of people in any office, current champions/winners, dates, prices, statistics, current events, software versions, recent developments — check whether the HIGH or MEDIUM tier search results confirm it.

RULES:
1. If a claim CONTRADICTS the high/medium-tier search results, REPLACE it with what those results say. Quote specific names, dates, or numbers.
2. If a claim is ONLY supported by PREDICTION-tier sources, REMOVE it — prediction sites describe future events, not past facts. Replace with "the search results did not confirm this".
3. If a claim is NOT mentioned anywhere in the search results, replace it with "the search results did not confirm this".
4. For temporal queries ("last winner of X", "most recent Y", "who won Z"): the answer must come from a source describing a COMPLETED, PAST event from a HIGH or MEDIUM tier source. Predictions or previews of upcoming events are not valid answers.

5. NUMBERS (scores, prices, dates, statistics) — symmetric rule:
   (a) If the draft cites a value AND a HIGH/MEDIUM-tier source contains that exact value attached to a real past event (e.g. "Match ends, X 2 Y 1", "Final Score: 2-1", a score paired with a date) → KEEP the value. Make sure the date and source domain are stated in the draft. This is a CONFIRMED fact and must not be removed.
   (b) If the draft REFUSES to report a value but a HIGH/MEDIUM-tier source DOES contain a confirmed past result for the question → REWRITE the draft to include that confirmed result. An over-cautious "I couldn't find a score" is just as wrong as inventing one.
   (c) If the draft cites a value that is NOT in any HIGH/MEDIUM-tier source → REPLACE with "the search results did not confirm this value".
   (d) Confirmed past results are valid answers to questions that don't specify "current" or "latest". Do NOT refuse a confirmed past score just because a more recent match might exist.

6. Do NOT add information that is not in the high/medium-tier search results.
7. Do NOT use your own knowledge — your training is outdated.
8. Preserve the original answer's bullet/paragraph structure and language style.
9. If the draft is fully consistent with the high/medium-tier search results, return it unchanged.

CRITICAL EXAMPLES:
- Draft says "the president of X is Joe Biden" but high-tier results mention "President Donald Trump" → REPLACE with "the president of X is Donald Trump".
- Draft says "Manchester City won the last Champions League" based on prediction-tier sources about an upcoming final, but high-tier Wikipedia/UEFA sources say "Paris Saint-Germain won the 2024-25 UEFA Champions League" → REPLACE with the high-tier confirmed answer.
- Draft cites "Bournemouth beat Manchester City 2-1 (Nov 2, 2024) per ESPN" and ESPN content includes "Match ends, Bournemouth 2, Manchester City 1" → KEEP IT UNCHANGED. This is a confirmed past result from a medium-tier source. Do not remove it just because it isn't the "latest" match.
- Draft says "I could not find a match score" but ESPN's content clearly contains "Bournemouth 2, Manchester City 1" → REWRITE to: "Per ESPN, Bournemouth beat Manchester City 2-1 on November 2, 2024." The score IS in the sources — don't hide it.
- Draft uses old prices, old officials, old dates with no source backing → REPLACE with high/medium-tier values.

OUTPUT FORMAT:
Output ONLY the corrected final answer text. No preamble. No "Here is the corrected answer:". No explanation. No apology. Just the answer."""


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url or ""


def _format_tavily_for_verifier(content: str, max_chars_per_item: int = 800) -> str:
    """Parse Tavily's JSON output and re-render each item with the tier label
    front-and-center, so the verifier can't misread it. Each item becomes:

        [MEDIUM TIER · espn.com] "Bournemouth 2-1 Man City (Nov 2, 2024)"
          Match ends, Bournemouth 2, Manchester City 1. …
    """
    try:
        items = json.loads(content)
    except Exception:
        return content[:3000]
    if not isinstance(items, list):
        return content[:3000]

    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tier = (item.get("_source_tier") or "low").upper()
        domain = _domain_of(item.get("url") or "")
        title = (item.get("title") or "").strip()
        body = (item.get("content") or "").strip()
        if body and len(body) > max_chars_per_item:
            body = body[:max_chars_per_item].rstrip() + "…"
        lines.append(f"[{tier} TIER · {domain}] {title}")
        if body:
            lines.append(f"  {body}")
        else:
            lines.append("  (no usable body text — rely on the title above)")
    return "\n".join(lines)


def build_sources_text(
    tool_messages: list[dict],
    max_sources: int = 6,
    max_chars_each: int = 3000,
) -> str:
    if not tool_messages:
        return ""
    parts = []
    for index, message in enumerate(tool_messages[:max_sources], start=1):
        raw_content = (message.get("content") or "").strip()
        if not raw_content:
            continue
        name = message.get("name") or "search_tool"
        if name == "tavily_search_results_json":
            formatted = _format_tavily_for_verifier(raw_content)
        else:
            # DuckDuckGo already prefixes each result with [TIER: X];
            # deep_site_reader is plain text — just trim.
            formatted = raw_content[:max_chars_each]
        parts.append(f"--- TOOL CALL {index} (from {name}) ---\n{formatted}")
    return "\n\n".join(parts)


def verify_answer(
    question: str,
    answer: str,
    tool_messages: list[dict],
    model_id: str,
    today_iso: str,
) -> tuple[str, bool]:
    """Verify and correct the answer against tool results.

    Returns (final_answer, was_changed).
    """
    if not answer or not answer.strip():
        return answer, False

    sources_text = build_sources_text(tool_messages)
    if not sources_text:
        return answer, False

    prompt = VERIFICATION_PROMPT.format(
        question=question,
        answer=answer,
        sources=sources_text,
        today=today_iso,
    )

    try:
        llm = _get_llm(model_id)
        response = llm.invoke(prompt)
        corrected = (response.content or "").strip()
        # Strip the common "Here is the corrected answer:" preamble if the model added one.
        for prefix in (
            "Here is the corrected answer:",
            "Corrected answer:",
            "Here's the corrected answer:",
            "Final answer:",
        ):
            if corrected.lower().startswith(prefix.lower()):
                corrected = corrected[len(prefix):].lstrip(" :\n").strip()
                break
        if not corrected or len(corrected) < 12:
            return answer, False
        return corrected, corrected.strip() != answer.strip()
    except Exception:
        return answer, False
