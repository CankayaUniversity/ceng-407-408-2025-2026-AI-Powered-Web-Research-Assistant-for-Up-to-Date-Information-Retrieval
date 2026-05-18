"""Post-agent verification pass.

Re-checks the agent's draft answer against the tool results (search snippets,
deep-read pages) and rewrites unsupported claims. This catches the common
small-model failure where the agent calls its tools but then synthesizes the
final answer from stale training memory instead of the retrieved content.
"""

from langchain_ollama import ChatOllama


VERIFICATION_PROMPT = """You are a strict fact-checker. Your only job is to verify a draft answer against live web search results and correct any factual errors.

QUESTION ASKED:
{question}

DRAFT ANSWER (may contain stale or hallucinated information from outdated training data):
{answer}

LIVE WEB SEARCH RESULTS (current, authoritative — TRUST THESE OVER ANY PRIOR KNOWLEDGE):
{sources}

TODAY'S DATE: {today}

YOUR TASK:
For every specific factual claim in the draft — names of people in any office (president, prime minister, CEO, rector, minister), dates, prices, statistics, current events, software versions, recent developments — check whether the live search results confirm it.

RULES:
1. If a claim CONTRADICTS the search results, REPLACE it with what the search results actually say. Quote specific names, dates, or numbers from the sources.
2. If a claim is NOT mentioned anywhere in the search results, replace it with "the search results did not confirm this".
3. Do NOT add information that is not in the search results.
4. Do NOT use your own knowledge — your training is outdated and unreliable for current facts.
5. Preserve the original answer's bullet/paragraph structure and language style.
6. If the draft is fully consistent with the search results, return it unchanged.

CRITICAL EXAMPLE OF WHAT TO FIX:
- Draft says "the president of X is Joe Biden" but search results mention "President Donald Trump" → REPLACE with "the president of X is Donald Trump" (or whatever name appears in the live results).
- Draft uses old prices, old officials, old dates from training data → REPLACE with values from the search results.

OUTPUT FORMAT:
Output ONLY the corrected final answer text that should be shown to the user. No preamble. No "Here is the corrected answer:". No explanation. No apology. Just the answer."""


def build_sources_text(
    tool_messages: list[dict],
    max_sources: int = 6,
    max_chars_each: int = 3000,
) -> str:
    if not tool_messages:
        return ""
    parts = []
    for index, message in enumerate(tool_messages[:max_sources], start=1):
        content = (message.get("content") or "")[:max_chars_each].strip()
        if not content:
            continue
        name = message.get("name") or "search_tool"
        parts.append(f"--- SOURCE {index} (from {name}) ---\n{content}")
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
        llm = ChatOllama(model=model_id, temperature=0)
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
