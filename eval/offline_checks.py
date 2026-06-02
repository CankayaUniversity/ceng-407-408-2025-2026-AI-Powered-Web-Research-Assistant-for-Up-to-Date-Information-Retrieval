"""Offline accuracy-logic checks — no Ollama, no network, no server.

These pin the deterministic primitives that decide correctness: the two
answer-acceptance gates and the source/recency-weighted conflict resolvers.
They run anywhere and are fast, so they are the regression net for the parts
of the pipeline that are pure logic.

Run:  python eval/offline_checks.py
Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_passes import (  # noqa: E402
    _choose_dated_candidate,
    _choose_supported_candidate,
    _prefer_latest_season,
    _season_start_year,
    accept_synthesized_answer,
    accept_verified_answer,
    assess_escalation,
)
from rank_utils import ordinal_label, requested_rank  # noqa: E402

_failures: list[str] = []


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        _failures.append(name)


# ── accept_synthesized_answer ────────────────────────────────────────────────
check(
    "synthesis: deterministic result is always adopted",
    accept_synthesized_answer({"status": "deterministic", "final_answer": "X"}, "Y") is True,
)
check(
    "synthesis: failed+low confidence keeps the draft",
    accept_synthesized_answer({"status": "failed", "confidence": "low", "final_answer": "Z"}, "Y") is False,
)
check(
    "synthesis: insufficient+unknown keeps the draft",
    accept_synthesized_answer({"status": "insufficient", "confidence": "unknown", "final_answer": "Z"}, "Y") is False,
)
check(
    "synthesis: answered+medium is adopted",
    accept_synthesized_answer({"status": "answered", "confidence": "medium", "final_answer": "Z"}, "Y") is True,
)
check(
    "synthesis: empty revision is rejected",
    accept_synthesized_answer({"status": "answered", "final_answer": ""}, "Y") is False,
)
check(
    "synthesis: revision equal to draft is rejected",
    accept_synthesized_answer({"status": "answered", "final_answer": "Y"}, "Y") is False,
)
check(
    "synthesis: empty draft accepts even a weak revision",
    accept_synthesized_answer({"status": "insufficient", "confidence": "low", "final_answer": "Z"}, "") is True,
)

# ── accept_verified_answer ───────────────────────────────────────────────────
check(
    "verifier: needs_more_search+low keeps the draft",
    accept_verified_answer({"status": "needs_more_search", "confidence": "low", "final_answer": "Z"}, "Y") is False,
)
check(
    "verifier: revise+high is adopted",
    accept_verified_answer({"status": "revise", "confidence": "high", "final_answer": "Z"}, "Y") is True,
)
check(
    "verifier: ok+medium is adopted",
    accept_verified_answer({"status": "ok", "confidence": "medium", "final_answer": "Z"}, "Y") is True,
)
check(
    "verifier: revision equal to draft is rejected",
    accept_verified_answer({"status": "revise", "final_answer": "Y"}, "Y") is False,
)
check(
    "verifier: empty draft accepts even needs_more_search",
    accept_verified_answer({"status": "needs_more_search", "confidence": "low", "final_answer": "Z"}, "") is True,
)

# ── _choose_dated_candidate (market / FX recency) ────────────────────────────
_newer_low_score = {"final_answer": "fresh", "date": "2026-05-20", "source_score": 0.3}
_older_high_score = {"final_answer": "stale", "date": "2026-05-01", "source_score": 0.9}
check(
    "dated: newest date wins over a higher-reliability stale source",
    _choose_dated_candidate([_older_high_score, _newer_low_score])["final_answer"] == "fresh",
)
check(
    "dated: same date falls back to source reliability",
    _choose_dated_candidate([
        {"final_answer": "low", "date": "2026-05-20", "source_score": 0.3},
        {"final_answer": "high", "date": "2026-05-20", "source_score": 0.9},
    ])["final_answer"] == "high",
)
check("dated: empty input returns None", _choose_dated_candidate([]) is None)

# ── _choose_supported_candidate (conflict resolution) ────────────────────────
check(
    "supported: single candidate is returned as-is",
    (_choose_supported_candidate([
        {"answer": "A", "value": "10", "date": "", "source_url": "u", "source_score": 0.5},
    ]) or {}).get("answer") == "A",
)
check(
    "supported: clearly more reliable source wins a conflict",
    (_choose_supported_candidate([
        {"answer": "A", "value": "10", "date": "", "source_url": "ua", "source_score": 0.7},
        {"answer": "B", "value": "12", "date": "", "source_url": "ub", "source_score": 0.45},
    ]) or {}).get("answer") == "A",
)
check(
    "supported: corroboration breaks a tie of equal reliability",
    (_choose_supported_candidate([
        {"answer": "A", "value": "10", "date": "", "source_url": "u1", "source_score": 0.5},
        {"answer": "A", "value": "10", "date": "", "source_url": "u2", "source_score": 0.5},
        {"answer": "B", "value": "12", "date": "", "source_url": "u3", "source_score": 0.5},
    ]) or {}).get("answer") == "A",
)
check(
    "supported: an unresolvable near-tie returns None",
    _choose_supported_candidate([
        {"answer": "A", "value": "10", "date": "", "source_url": "ua", "source_score": 0.50},
        {"answer": "B", "value": "12", "date": "", "source_url": "ub", "source_score": 0.45},
    ]) is None,
)

# ── season recency ───────────────────────────────────────────────────────────
check("season: parses 2024/2025", _season_start_year("2024/2025") == 2024)
check("season: parses 2024/25", _season_start_year("2024/25") == 2024)
check("season: empty -> 0", _season_start_year("") == 0)
_latest = _prefer_latest_season([
    {"answer": "Old", "date": "2023/2024"},
    {"answer": "New", "date": "2024/2025"},
])
check(
    "season: stale-season candidate is dropped when seasons conflict",
    len(_latest) == 1 and _latest[0]["answer"] == "New",
)
check(
    "season: single season is left untouched",
    len(_prefer_latest_season([{"answer": "X", "date": "2024/2025"}])) == 1,
)

# ── rank parsing ─────────────────────────────────────────────────────────────
check("rank: '3rd top scorer' -> 3", requested_rank("who is the 3rd top scorer") == 3)
check("rank: no ordinal -> None", requested_rank("who is the top scorer") is None)
check("rank: ordinal_label(3) -> '3rd'", ordinal_label(3) == "3rd")
check("rank: ordinal_label(1) -> '1st'", ordinal_label(1) == "1st")


# ── assess_escalation (adaptive escalation trigger) ──────────────────────────
_STRONG = {"answer_trust_score": 0.85, "citation_coverage": 0.9, "total_sources": 4}
_WEAK = {"answer_trust_score": 0.2, "citation_coverage": 0.2, "total_sources": 1}


def _level(draft, signals, understanding):
    return assess_escalation(
        question="Tell me about the topic",
        draft_answer=draft,
        trust_signals=signals,
        understanding=understanding,
        tool_messages=[],
    )["level"]


check(
    "escalation: well-grounded, well-shaped draft stays Level 0",
    _level("There are 7 recognised types, all documented.", _STRONG, {}) == 0,
)
check(
    "escalation: weak grounding (no hedge/shape gap) is Level 1",
    _level("There are 7 recognised types, all documented.", _WEAK, {}) == 1,
)
check(
    "escalation: a hedge/non-answer is Level 2",
    _level("I could not find a definitive answer.", _STRONG, {}) == 2,
)
check(
    "escalation: value question with no number is Level 2",
    _level("Apple stock performed notably.", _STRONG, {"question_type": "finance"}) == 2,
)
check(
    "escalation: empty draft is Level 2",
    _level("", _STRONG, {}) == 2,
)


if _failures:
    print(f"\n{len(_failures)} check(s) failed:")
    for name in _failures:
        print(f"  - {name}")
    sys.exit(1)
print("\nAll offline checks passed.")
